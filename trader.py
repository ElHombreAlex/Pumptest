"""
Trade executor for Pump.fun via PumpPortal.

Supports two modes:
  - dry_run=True  → logs the intended trade without sending a transaction
  - dry_run=False → signs and broadcasts a real Solana transaction

The flow for a real trade:
  1. POST to pumpportal.fun/api/trade-local → receive unsigned serialised tx
  2. Deserialise with solders
  3. Sign with local wallet keypair using VersionedTransaction(message, [keypair])
  4. Send via Solana JSON-RPC; fall back to secondary endpoints on failure
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
from solders.keypair import Keypair  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from solana.rpc.async_api import AsyncClient  # type: ignore
from solana.rpc.types import TxOpts  # type: ignore

from config import cfg
from models import AnalysisResult, Position, PositionStatus, TokenEvent

log = logging.getLogger(__name__)


class TradeExecutor:
    """Builds, signs, and broadcasts Pump.fun transactions."""

    def __init__(self) -> None:
        self._dry_run = cfg.DRY_RUN
        self._keypair: Optional[Keypair] = None
        self._rpc_clients: list[AsyncClient] = []

        if not self._dry_run:
            self._keypair = _load_keypair(cfg.WALLET_PRIVATE_KEY)
            # Primary RPC first, then any configured fallbacks.
            # timeout=30 replaces the solana-py default of 10s — public mainnet
            # RPC nodes regularly take 15-25s under load, causing TimeoutError
            # crashes with the default value.
            urls = [cfg.SOLANA_RPC_URL] + cfg.SOLANA_RPC_FALLBACK_URLS
            self._rpc_clients = [AsyncClient(url, timeout=30) for url in urls if url]
            log.info("Loaded %d RPC endpoint(s)", len(self._rpc_clients))

    # ── Buy ───────────────────────────────────────────────────────────────────

    async def buy(
        self,
        token: TokenEvent,
        analysis: AnalysisResult,
        entry_market_cap: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Execute a buy order.
        Returns an open Position on success, None on failure.

        ``entry_market_cap`` should be the most recent observed market cap at
        the time the buy decision is made (from the live trade stream), so that
        PnL is calculated against the actual entry price rather than the
        token-creation market cap (which can be 20+ seconds stale).
        """
        buy_sol = min(analysis.suggested_buy_sol, cfg.MAX_BUY_SOL)
        if buy_sol <= 0:
            buy_sol = cfg.MAX_BUY_SOL

        log.info(
            "[%s] BUY %.4f SOL | score=%d | %s",
            token.symbol,
            buy_sol,
            analysis.confidence_score,
            token.mint[:12],
        )

        if self._dry_run:
            log.info("[DRY-RUN] Would buy %s with %.4f SOL", token.symbol, buy_sol)
            return _make_position(token, analysis, buy_sol, entry_market_cap, simulated=True)

        tx_bytes = await self._build_tx(
            action="buy",
            mint=token.mint,
            amount=buy_sol,
            denominated_in_sol=True,
        )
        if not tx_bytes:
            return None

        sig = await self._sign_and_send(tx_bytes)
        if not sig:
            return None

        log.info("[%s] Buy confirmed: %s", token.symbol, sig)
        return _make_position(token, analysis, buy_sol, entry_market_cap, simulated=False)

    # ── Sell ──────────────────────────────────────────────────────────────────

    async def sell(self, position: Position, pct: float = 100.0) -> bool:
        """
        Sell ``pct`` % of a position (default 100%).
        Returns True on success.
        """
        log.info(
            "[%s] SELL %.0f%% | mcap=%.2f SOL | mint=%s",
            position.symbol,
            pct,
            position.current_market_cap,
            position.mint[:12],
        )

        if self._dry_run:
            log.info("[DRY-RUN] Would sell %.0f%% of %s", pct, position.symbol)
            return True

        # Use real on-chain balance — our stored estimate can differ from reality.
        # If the wallet holds 0 tokens the buy tx failed on-chain; skip the sell.
        real_balance = await self._get_token_balance(position.mint)
        if real_balance is not None:
            if real_balance == 0:
                log.warning(
                    "[%s] Wallet holds 0 tokens — already sold or buy failed "
                    "on-chain.  Marking position closed.",
                    position.symbol,
                )
                return True  # nothing to sell; close the position cleanly
            token_amount = int(real_balance * (pct / 100.0))
            log.info(
                "[%s] On-chain balance: %d tokens → selling %d",
                position.symbol, real_balance, token_amount,
            )
        else:
            # RPC query failed — fall back to stored estimate
            token_amount = int(position.token_amount * (pct / 100.0))
            log.warning(
                "[%s] Could not query on-chain balance; using stored estimate %d",
                position.symbol, token_amount,
            )

        if token_amount <= 0:
            log.error("[%s] Sell amount is 0 — skipping", position.symbol)
            return False

        tx_bytes = await self._build_tx(
            action="sell",
            mint=position.mint,
            amount=token_amount,
            denominated_in_sol=False,
            pool="auto",  # auto detects bonding curve vs Raydium
        )
        if not tx_bytes:
            return False

        sig = await self._sign_and_send(tx_bytes)
        if not sig:
            return False

        log.info("[%s] Sell confirmed: %s", position.symbol, sig)
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_token_balance(self, mint: str) -> Optional[int]:
        """
        Return the wallet's raw token balance (base units, not UI amount).
        Returns None if the query fails so callers can fall back gracefully.
        """
        try:
            from solders.pubkey import Pubkey  # type: ignore
            from solana.rpc.types import TokenAccountOpts  # type: ignore
            wallet_pk = self._keypair.pubkey()
            mint_pk = Pubkey.from_string(mint)
            opts = TokenAccountOpts(mint=mint_pk)
            for rpc in self._rpc_clients:
                try:
                    resp = await rpc.get_token_accounts_by_owner_json_parsed(
                        wallet_pk, opts
                    )
                    if resp.value:
                        ta = resp.value[0].account.data.parsed["info"]["tokenAmount"]
                        # `amount` is raw base units; divide by 10^decimals to get
                        # the display (UI) token count that PumpPortal expects.
                        decimals = int(ta.get("decimals", 6))
                        return int(ta["amount"]) // (10 ** decimals)
                    # No account found → wallet never received these tokens
                    return 0
                except Exception as exc:
                    log.debug("Balance RPC query failed: %s", exc)
        except Exception as exc:
            log.warning("_get_token_balance unavailable: %s", exc)
        return None

    async def _build_tx(
        self,
        action: str,
        mint: str,
        amount,
        denominated_in_sol: bool,
        pool: str = "auto",
    ) -> Optional[bytes]:
        """
        Ask PumpPortal to build a serialised transaction and return raw bytes.
        """
        payload = {
            "publicKey": cfg.WALLET_PUBLIC_KEY,
            "action": action,
            "mint": mint,
            "amount": amount,
            "denominatedInSol": str(denominated_in_sol).lower(),
            "slippage": cfg.SLIPPAGE,
            "priorityFee": cfg.PRIORITY_FEE,
            "pool": pool,
        }

        headers = {}
        if cfg.PUMPPORTAL_API_KEY:
            headers["Authorization"] = f"Bearer {cfg.PUMPPORTAL_API_KEY}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    cfg.PUMPPORTAL_TRADE_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.error(
                            "PumpPortal API error %d: %s | payload=%s",
                            resp.status,
                            body[:500],
                            payload,
                        )
                        return None
                    data = await resp.read()
                    return data
        except asyncio.TimeoutError:
            log.error("PumpPortal API timed out after 15s building %s tx", action)
            return None
        except aiohttp.ClientError as exc:
            log.error("HTTP error building tx: %s", exc)
            return None

    async def _sign_and_send(self, tx_bytes: bytes) -> Optional[str]:
        """
        Deserialise, sign, and broadcast the transaction.

        Signing: VersionedTransaction(message, [keypair]) is the correct solders
        API — it embeds the signature in the proper Solana transaction format.
        The old sign_message() call was wrong: it produces a raw Ed25519 sig
        without the transaction context, which the network always rejects.

        RPC failover: tries each configured endpoint in order; moves on if one
        fails so a single bad/rate-limited RPC doesn't block the trade.
        """
        assert self._keypair is not None

        # Deserialise the unsigned tx returned by PumpPortal, then re-sign it.
        try:
            unsigned_tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(unsigned_tx.message, [self._keypair])
            serialised = bytes(signed_tx)
        except Exception as exc:
            log.exception("Failed to deserialise/sign transaction: %s", exc)
            return None

        # Try each RPC in order; return the first successful signature.
        # asyncio.TimeoutError is caught explicitly — it can originate inside
        # aiohttp's connector layer before the awaited call returns, bypassing
        # a plain `except Exception` in some Python/aiohttp version combinations.
        for i, rpc in enumerate(self._rpc_clients):
            try:
                result = await rpc.send_raw_transaction(
                    serialised,
                    opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"),
                )
                if result.value:
                    if i > 0:
                        log.info("Transaction landed via fallback RPC #%d", i)
                    return str(result.value)
                log.warning("RPC #%d returned no signature: %s", i, result)
            except asyncio.TimeoutError:
                log.warning("RPC #%d timed out (>30s) — trying next endpoint", i)
            except Exception as exc:
                log.warning("RPC #%d failed: %s", i, exc)

        log.error("All %d RPC endpoint(s) failed for this transaction", len(self._rpc_clients))
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_keypair(private_key_b58: str) -> Keypair:
    import base58  # type: ignore
    secret = base58.b58decode(private_key_b58)
    return Keypair.from_bytes(secret)


def _make_position(
    token: TokenEvent,
    analysis: AnalysisResult,
    buy_sol: float,
    entry_market_cap: Optional[float],
    simulated: bool,
) -> Position:
    # Use the live market cap at buy time when available; fall back to the
    # creation-event market cap (which may be 20+ seconds stale by now).
    mcap = (entry_market_cap if entry_market_cap and entry_market_cap > 0
            else token.initial_market_cap) or 1.0
    # Rough estimate: tokens received = (SOL * total_supply / mcap)
    # We track value in market-cap terms rather than exact token amount
    token_amount = (buy_sol / mcap) * 1_000_000_000  # normalised units

    position = Position(
        mint=token.mint,
        symbol=token.symbol,
        entry_sol=buy_sol,
        entry_market_cap=mcap,
        token_amount=token_amount,
        entry_price_per_token=mcap / 1_000_000_000,
        take_profit_market_cap=mcap * cfg.TAKE_PROFIT_MULTIPLIER,
        stop_loss_market_cap=mcap * cfg.STOP_LOSS_MULTIPLIER,
        current_market_cap=mcap,
        status=PositionStatus.OPEN,
        confidence_score=analysis.confidence_score,
    )

    label = "[SIM]" if simulated else "[LIVE]"
    log.info(
        "%s Position opened | %s | entry_mcap=%.2f | TP=%.2f | SL=%.2f",
        label,
        token.symbol,
        mcap,
        position.take_profit_market_cap,
        position.stop_loss_market_cap,
    )
    return position
