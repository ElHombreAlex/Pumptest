"""
Risk manager and position tracker.

Responsibilities:
- Track all open positions (in memory + persisted to SQLite via PositionStore)
- Update positions with new market-cap data from trade events
- Trigger take-profit / stop-loss exits with exponential-backoff retry
- Enforce max-open-positions limit
- Compute and log P&L
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable, Optional

from config import cfg
from models import Position, PositionStatus, TradeEvent
from persistence import PositionStore

log = logging.getLogger(__name__)

SellCallback = Callable[[Position], Awaitable[bool]]

# Retry configuration for failed sell orders
MAX_SELL_RETRIES = 3
SELL_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt (2 → 4 → 8)


class RiskManager:
    """Monitors open positions and fires exits when TP/SL levels are hit."""

    def __init__(self, sell_fn: SellCallback, store: Optional[PositionStore] = None) -> None:
        self._sell_fn = sell_fn
        self._store = store
        self._positions: dict[str, Position] = {}   # mint → Position
        self._closed: list[Position] = []

    # ── Position management ───────────────────────────────────────────────────

    def add_position(self, position: Position) -> None:
        self._positions[position.mint] = position
        if self._store:
            self._store.save(position)
        log.info(
            "Position added: %s | entry_mcap=%.2f | TP=%.2f | SL=%.2f",
            position.symbol,
            position.entry_market_cap,
            position.take_profit_market_cap,
            position.stop_loss_market_cap,
        )

    def restore_position(self, position: Position) -> None:
        """Re-load a position from persistent storage without saving it again."""
        self._positions[position.mint] = position
        log.info(
            "Restored position: %s | entry_mcap=%.2f | current_mcap=%.2f",
            position.symbol,
            position.entry_market_cap,
            position.current_market_cap,
        )

    def can_open_position(self) -> bool:
        return len(self._positions) < cfg.MAX_POSITIONS

    def is_tracking(self, mint: str) -> bool:
        return mint in self._positions

    def position_count(self) -> int:
        return len(self._positions)

    def open_mints(self) -> list[str]:
        """Return the mint addresses of all currently open positions."""
        return list(self._positions.keys())

    # ── Real-time price update ────────────────────────────────────────────────

    async def on_trade(self, event: TradeEvent) -> None:
        """
        Called for every trade event. Updates the relevant position and checks
        if TP/SL targets have been hit.
        """
        pos = self._positions.get(event.mint)
        if not pos:
            return

        pos.current_market_cap = event.new_market_cap
        pos.pnl_sol = (
            pos.entry_sol * (event.new_market_cap / pos.entry_market_cap) - pos.entry_sol
        )

        log.debug(
            "[%s] mcap=%.2f | pnl=%.4f SOL | TP=%.2f | SL=%.2f",
            pos.symbol,
            pos.current_market_cap,
            pos.pnl_sol,
            pos.take_profit_market_cap,
            pos.stop_loss_market_cap,
        )

        # ── Take-profit ───────────────────────────────────────────────────────
        if event.new_market_cap >= pos.take_profit_market_cap:
            log.info(
                "[%s] TAKE PROFIT triggered | mcap=%.2f >= %.2f | pnl=+%.4f SOL",
                pos.symbol,
                event.new_market_cap,
                pos.take_profit_market_cap,
                pos.pnl_sol,
            )
            await self._close_position(pos, reason="take_profit")
            return

        # ── Stop-loss ─────────────────────────────────────────────────────────
        if event.new_market_cap <= pos.stop_loss_market_cap:
            log.info(
                "[%s] STOP LOSS triggered | mcap=%.2f <= %.2f | pnl=%.4f SOL",
                pos.symbol,
                event.new_market_cap,
                pos.stop_loss_market_cap,
                pos.pnl_sol,
            )
            await self._close_position(pos, reason="stop_loss")

    # ── Portfolio summary ─────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["── Open Positions ──────────────────────────────"]
        if not self._positions:
            lines.append("  (none)")
        for pos in self._positions.values():
            pnl_pct = (
                (pos.current_market_cap / pos.entry_market_cap - 1) * 100
                if pos.entry_market_cap
                else 0
            )
            lines.append(
                f"  {pos.symbol:10s} | "
                f"entry={pos.entry_market_cap:8.2f} SOL | "
                f"now={pos.current_market_cap:8.2f} SOL | "
                f"pnl={pos.pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)"
            )

        total_pnl = sum(p.pnl_sol for p in self._closed)
        lines.append(f"── Closed this session: {len(self._closed)} | Session P&L: {total_pnl:+.4f} SOL")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _close_position(self, pos: Position, reason: str) -> None:
        """
        Attempt to sell a position, retrying with exponential backoff on failure.

        Retry schedule: 2s → 4s → 8s (MAX_SELL_RETRIES = 3 attempts total).
        If all attempts fail, the position remains open and will be retried on
        the next TP/SL trigger from an incoming trade event.
        """
        for attempt in range(1, MAX_SELL_RETRIES + 1):
            try:
                success = await self._sell_fn(pos)
            except Exception as exc:
                log.warning(
                    "[%s] Sell attempt %d/%d raised an exception: %s",
                    pos.symbol, attempt, MAX_SELL_RETRIES, exc,
                )
                success = False

            if success:
                pos.status = PositionStatus.CLOSED
                pos.closed_at = time.time()
                del self._positions[pos.mint]
                self._closed.append(pos)
                if self._store:
                    self._store.save(pos)
                log.info(
                    "[%s] Position closed (%s) | final pnl=%+.4f SOL",
                    pos.symbol,
                    reason,
                    pos.pnl_sol,
                )
                return

            # Not the last attempt — wait before retrying
            if attempt < MAX_SELL_RETRIES:
                delay = SELL_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "[%s] Sell failed (attempt %d/%d) — retrying in %.0fs",
                    pos.symbol, attempt, MAX_SELL_RETRIES, delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted; position stays open and will be retried on the
        # next trade event that re-triggers the same TP/SL condition
        log.error(
            "[%s] Sell FAILED after %d attempts (%s) — position remains open",
            pos.symbol, MAX_SELL_RETRIES, reason,
        )
