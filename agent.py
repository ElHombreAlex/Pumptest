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

log = logging.getLogger(__name__)

# How long / how many trades to wait before analysing a new token
WARM_UP_TRADES = 3       # collect at least N trades before asking Claude
WARM_UP_SECS = 10        # … or wait this many seconds, whichever comes first
SUMMARY_INTERVAL = 60    # print portfolio summary every N seconds


class TradingAgent:
    """
    The top-level agent that ties all components together.
    """

    def __init__(self) -> None:
        self._client = PumpPortalClient()
        self._analyzer = TokenAnalyzer()
        self._executor = TradeExecutor()
        self._risk = RiskManager(sell_fn=self._executor.sell)

        # mint → list of recent trades (ring buffer)
        self._trade_buffer: dict[str, collections.deque[TradeEvent]] = (
            collections.defaultdict(lambda: collections.deque(maxlen=20))
        )
        # mint → timestamp when we first saw the token
        self._token_seen_at: dict[str, float] = {}
        # mints currently being analysed (avoid duplicate analysis)
        self._analysing: set[str] = set()
        # tokens we've already decided on (buy or skip)
        self._decided: set[str] = set()
        # token metadata cache: mint → TokenEvent
        self._token_cache: dict[str, TokenEvent] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the agent. This blocks indefinitely."""
        self._client.on_new_token(self._on_new_token)
        self._client.on_new_trade(self._on_trade)

        await asyncio.gather(
            self._client.run(),
            self._summary_loop(),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def _on_new_token(self, token: TokenEvent) -> None:
        """Fired when PumpPortal reports a new token creation."""
        # Cache metadata for later trade-event lookups
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

        # Schedule warm-up analysis; will fire after WARM_UP_SECS even if
        # fewer than WARM_UP_TRADES trades have arrived.
        asyncio.create_task(self._warm_up_and_analyse(token))

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
            and len(buf) >= WARM_UP_TRADES
        ):
            token = self._token_cache.get(event.mint)
            if token:
                asyncio.create_task(self._analyse_and_trade(token))

    # ── Core logic ────────────────────────────────────────────────────────────

    async def _warm_up_and_analyse(self, token: TokenEvent) -> None:
        """Wait for the warm-up period, then trigger analysis if not yet decided."""
        await asyncio.sleep(WARM_UP_SECS)
        if token.mint not in self._decided and token.mint not in self._analysing:
            await self._analyse_and_trade(token)

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
                position = await self._executor.buy(token, analysis)
                if position:
                    self._risk.add_position(position)
            else:
                log.info(
                    "[%s] Not buying — score=%d, rec=%s, positions=%d/%d",
                    token.symbol,
                    analysis.confidence_score,
                    analysis.recommendation,
                    len(self._risk._positions),
                    cfg.MAX_POSITIONS,
                )
                # Unsubscribe to keep the WebSocket connection lean
                await self._client.unsubscribe_token(token.mint)

        finally:
            self._analysing.discard(token.mint)

    # ── Portfolio summary loop ────────────────────────────────────────────────

    async def _summary_loop(self) -> None:
        while True:
            await asyncio.sleep(SUMMARY_INTERVAL)
            log.info("\n%s", self._risk.summary())
