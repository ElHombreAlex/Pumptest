"""
Main trading agent orchestrator.

Wires together:
  PumpPortalClient  →  TokenAnalyzer  →  TradeExecutor  →  RiskManager

Flow:
  1. PumpPortalClient streams new token creation events.
  2. For each token that passes pre-filters, we subscribe to its trade stream
     and collect initial trade data.
  3. After WARM_UP_TRADES trades (or WARM_UP_SECS seconds), we ask Claude
     to score the token.
  4. If the score meets MIN_CONFIDENCE_SCORE and we have capacity, we buy.
  5. RiskManager watches the trade stream and exits positions at TP/SL.
  6. A periodic summary is printed to the console.
  7. Stale token metadata is evicted from memory every TOKEN_CACHE_TTL seconds
     to keep long-running sessions lean.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time

from config import cfg
from models import TokenEvent, TradeEvent
from pumpportal_client import PumpPortalClient
from analyzer import TokenAnalyzer
from trader import TradeExecutor
from risk_manager import RiskManager
from persistence import PositionStore

log = logging.getLogger(__name__)

SUMMARY_INTERVAL = 60    # print portfolio summary every N seconds
STALE_CHECK_INTERVAL = 60  # check for stale positions every N seconds

# Evict decided/untracked tokens from the in-memory cache after this long
TOKEN_CACHE_TTL = 1800   # 30 minutes
EVICTION_INTERVAL = 300  # run eviction sweep every 5 minutes


class TradingAgent:
    """
    The top-level agent that ties all components together.
    """

    def __init__(self) -> None:
        self._client = PumpPortalClient()
        self._analyzer = TokenAnalyzer()
        self._executor = TradeExecutor()

        # Initialise persistence store and restore any positions that were open
        # before the last shutdown
        self._store = PositionStore()
        self._risk = RiskManager(sell_fn=self._executor.sell, store=self._store)

        restored = self._store.load_open()
        for pos in restored:
            self._risk.restore_position(pos)
        if restored:
            log.info("Restored %d open position(s) from database", len(restored))

        # mint → ring buffer of recent trades (size capped by deque maxlen)
        self._trade_buffer: dict[str, collections.deque[TradeEvent]] = (
            collections.defaultdict(lambda: collections.deque(maxlen=20))
        )
        # mint → unix timestamp when we first saw this token
        self._token_seen_at: dict[str, float] = {}
        # mints currently being analysed (prevent duplicate concurrent analysis)
        self._analysing: set[str] = set()
        # tokens we've already made a decision on (buy or skip)
        self._decided: set[str] = set()
        # token metadata cache: mint → TokenEvent
        self._token_cache: dict[str, TokenEvent] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the agent. This blocks indefinitely."""
        self._client.on_new_token(self._on_new_token)
        self._client.on_new_trade(self._on_trade)

        # Re-subscribe to mints we still have open positions for
        for mint in self._risk.open_mints():
            await self._client.subscribe_token(mint)

        await asyncio.gather(
            self._client.run(),
            self._summary_loop(),
            self._eviction_loop(),
            self._stale_position_loop(),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def _on_new_token(self, token: TokenEvent) -> None:
        """Fired when PumpPortal reports a new token creation."""
        self._token_cache[token.mint] = token

        log.info(
            "NEW TOKEN %-10s %-8s | creator=%s | mcap=%.2f SOL",
            (token.name or "")[:10],
            token.symbol or "???",
            (token.creator or "")[:8],
            token.initial_market_cap,
        )

        # Quick pre-filter: skip tokens with no name/symbol
        if not token.name or not token.symbol:
            log.debug("Skipping unnamed token %s", token.mint[:12])
            return

        # Subscribe to trade stream for this token
        await self._client.subscribe_token(token.mint)
        self._token_seen_at[token.mint] = time.time()

        # Schedule warm-up analysis; will fire after WARM_UP_SECONDS even if
        # fewer than WARM_UP_TRADES trades have arrived.
        asyncio.create_task(
            self._safe_warm_up_and_analyse(token),
            name=f"warmup-{token.mint[:8]}",
        )

    async def _on_trade(self, event: TradeEvent) -> None:
        """Fired for every trade on a tracked token."""
        # Buffer trades for analysis warm-up
        self._trade_buffer[event.mint].append(event)

        # Update risk manager for open positions
        await self._risk.on_trade(event)

        # Trigger analysis early if we already have enough warm-up trades
        buf = self._trade_buffer[event.mint]
        if (
            event.mint not in self._decided
            and event.mint not in self._analysing
            and len(buf) >= cfg.WARM_UP_TRADES
        ):
            token = self._token_cache.get(event.mint)
            if token:
                asyncio.create_task(
                    self._safe_analyse_and_trade(token),
                    name=f"analyse-{token.mint[:8]}",
                )

    # ── Core logic ────────────────────────────────────────────────────────────

    async def _safe_warm_up_and_analyse(self, token: TokenEvent) -> None:
        """Error-bounded wrapper for the warm-up timer task."""
        try:
            await asyncio.sleep(cfg.WARM_UP_SECONDS)
            if token.mint not in self._decided and token.mint not in self._analysing:
                await self._analyse_and_trade(token)
        except Exception:
            log.exception(
                "Unhandled error in warm-up task for %s (%s)",
                token.symbol,
                token.mint[:8],
            )
            # Prevent this token from being retried endlessly
            self._decided.add(token.mint)
            self._analysing.discard(token.mint)

    async def _safe_analyse_and_trade(self, token: TokenEvent) -> None:
        """Error-bounded wrapper for the trade-triggered analysis task."""
        try:
            await self._analyse_and_trade(token)
        except Exception:
            log.exception(
                "Unhandled error analysing %s (%s)",
                token.symbol,
                token.mint[:8],
            )
            self._decided.add(token.mint)
            self._analysing.discard(token.mint)

    async def _analyse_and_trade(self, token: TokenEvent) -> None:
        if token.mint in self._decided or token.mint in self._analysing:
            return

        self._analysing.add(token.mint)
        try:
            trades = list(self._trade_buffer[token.mint])
            analysis = await self._analyzer.analyse(token, trades)

            log.info(
                "[%s] Score=%d | %s | flags=%s",
                token.symbol,
                analysis.confidence_score,
                analysis.recommendation,
                analysis.risk_flags or "none",
            )
            log.info("[%s] Reasoning: %s", token.symbol, analysis.reasoning)

            self._decided.add(token.mint)

            if (
                analysis.recommendation == "BUY"
                and analysis.confidence_score >= cfg.MIN_CONFIDENCE_SCORE
                and self._risk.can_open_position()
                and not self._risk.is_tracking(token.mint)
            ):
                # Use the most recent observed market cap as the entry price so
                # PnL reflects our actual cost basis, not the token-creation
                # market cap which is 20+ seconds stale by the time we buy.
                recent_trades = list(self._trade_buffer[token.mint])
                current_mcap = next(
                    (t.new_market_cap for t in reversed(recent_trades) if t.new_market_cap),
                    None,
                )
                position = await self._executor.buy(token, analysis, entry_market_cap=current_mcap)
                if position:
                    self._risk.add_position(position)
            else:
                log.info(
                    "[%s] Not buying — score=%d, rec=%s, positions=%d/%d",
                    token.symbol,
                    analysis.confidence_score,
                    analysis.recommendation,
                    self._risk.position_count(),
                    cfg.MAX_POSITIONS,
                )
                # Unsubscribe to keep the WebSocket connection lean
                await self._client.unsubscribe_token(token.mint)

        finally:
            self._analysing.discard(token.mint)

    # ── Background loops ──────────────────────────────────────────────────────

    async def _summary_loop(self) -> None:
        while True:
            await asyncio.sleep(SUMMARY_INTERVAL)
            log.info("\n%s", self._risk.summary())

    async def _eviction_loop(self) -> None:
        """
        Periodically purge stale token metadata from memory.

        Only evicts tokens that:
          - We have already made a decision on (buy or skip), AND
          - Are not currently tracked by the risk manager (no open position), AND
          - Were first seen more than TOKEN_CACHE_TTL seconds ago
        """
        while True:
            await asyncio.sleep(EVICTION_INTERVAL)
            now = time.time()
            stale = [
                mint
                for mint, seen_at in self._token_seen_at.items()
                if (
                    now - seen_at > TOKEN_CACHE_TTL
                    and mint in self._decided
                    and not self._risk.is_tracking(mint)
                )
            ]
            for mint in stale:
                self._token_cache.pop(mint, None)
                self._token_seen_at.pop(mint, None)
                self._decided.discard(mint)
                self._trade_buffer.pop(mint, None)
            if stale:
                log.debug("Evicted %d stale token entries from memory", len(stale))

    async def _stale_position_loop(self) -> None:
        """
        Periodically check for open positions that haven't received a trade
        update in cfg.STALE_POSITION_MINUTES minutes.

        A stale position means the token has gone dark (no buyers/sellers) —
        either it dumped to zero or it's so illiquid that TP/SL will never
        fire on its own.  We log a warning and attempt to sell immediately.
        """
        threshold = cfg.STALE_POSITION_MINUTES * 60
        while True:
            await asyncio.sleep(STALE_CHECK_INTERVAL)
            for pos in self._risk.stale_positions(threshold):
                log.warning(
                    "[%s] Position stale — no trade update in %d min | "
                    "mcap=%.2f SOL | pnl=%+.4f SOL — triggering sell",
                    pos.symbol,
                    cfg.STALE_POSITION_MINUTES,
                    pos.current_market_cap,
                    pos.pnl_sol,
                )
                await self._risk.trigger_close(pos.mint, reason="stale")
