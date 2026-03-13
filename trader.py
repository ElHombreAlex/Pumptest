"""
Trade executor for Pump.fun via PumpPortal.

Supports two modes:
  - dry_run=True  → logs the intended trade without sending a transaction
  - dry_run=False → signs and broadcasts a real Solana transaction

The flow for a real trade:
  1. POST to pumpportal.fun/api/trade-local → receive unsigned serialised tx
  2. Deserialise with solders
  3. Sign with local wallet keypair
  4. Send via Solana JSON-RPC
"""
from __future__ import annotations

import base64
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
        self._rpc: Optional[AsyncClient] = None

        if not self._dry_run:
            self._keypair = _load_keypair(cfg.WALLET_PRIVATE_KEY)
            self._rpc = AsyncClient(cfg.SOLANA_RPC_URL)

    # ── Buy ───────────────────────────────────────────────────────────────────

    async def buy(self, token: TokenEvent, analysis: AnalysisResult) -> Optional[Position]:
        """
        Execute a buy order.
        Returns an open Position on success, None on failure.
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
            return _make_position(token, analysis, buy_sol, simulated=True)

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
        return _make_position(token, analysis, buy_sol, simulated=False)

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

        amount_str = f"{pct}%"
        tx_bytes = await self._build_tx(
            action="sell",
            mint=position.mint,
            amount=amount_str,
            denominated_in_sol=False,
        )
        if not tx_bytes:
            return False

        sig = await self._sign_and_send(tx_bytes)
        if not sig:
            return False

        log.info("[%s] Sell confirmed: %s", position.symbol, sig)
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _build_tx(
        self,
        action: str,
        mint: str,
        amount,
        denominated_in_sol: bool,
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
            "pool": "pump",
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
                            "PumpPortal API error %d: %s", resp.status, body[:200]
                        )
                        return None
                    data = await resp.read()
                    return data
        except aiohttp.ClientError as exc:
            log.error("HTTP error building tx: %s", exc)
            return None

    async def _sign_and_send(self, tx_bytes: bytes) -> Optional[str]:
        """Deserialise, sign, and broadcast the transaction."""
        assert self._keypair is not None
        assert self._rpc is not None

        try:
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = self._keypair.sign_message(
                bytes(tx.message)
            )

            # Re-attach signature
            tx.signatures[0] = signed_tx
            serialised = bytes(tx)

            result = await self._rpc.send_raw_transaction(
                serialised,
                opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"),
            )
            if result.value:
                return str(result.value)
            log.error("send_raw_transaction returned no signature: %s", result)
            return None
        except Exception as exc:
            log.exception("Error signing/sending transaction: %s", exc)
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
    simulated: bool,
) -> Position:
    mcap = token.initial_market_cap or 1.0
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
