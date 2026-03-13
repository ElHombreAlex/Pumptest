"""
Risk manager and position tracker.

Responsibilities:
- Track all open positions
- Update positions with new market-cap data from trade events
- Trigger take-profit / stop-loss exits
- Enforce max-open-positions limit
- Compute and log P&L
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Awaitable

from config import cfg
from models import Position, PositionStatus, TradeEvent

log = logging.getLogger(__name__)

SellCallback = Callable[[Position], Awaitable[bool]]


class RiskManager:
    """Monitors open positions and fires exits when TP/SL levels are hit."""

    def __init__(self, sell_fn: SellCallback) -> None:
        self._sell_fn = sell_fn
        self._positions: dict[str, Position] = {}   # mint → Position
        self._closed: list[Position] = []

    # ── Position management ───────────────────────────────────────────────────

    def add_position(self, position: Position) -> None:
        self._positions[position.mint] = position
        log.info(
            "Position added: %s | entry_mcap=%.2f | TP=%.2f | SL=%.2f",
            position.symbol,
            position.entry_market_cap,
            position.take_profit_market_cap,
            position.stop_loss_market_cap,
        )

    def can_open_position(self) -> bool:
        return len(self._positions) < cfg.MAX_POSITIONS

    def is_tracking(self, mint: str) -> bool:
        return mint in self._positions

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
        lines.append(f"── Closed: {len(self._closed)} | Total P&L: {total_pnl:+.4f} SOL")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _close_position(self, pos: Position, reason: str) -> None:
        success = await self._sell_fn(pos)
        if success:
            pos.status = PositionStatus.CLOSED
            pos.closed_at = time.time()
            del self._positions[pos.mint]
            self._closed.append(pos)
            log.info(
                "[%s] Position closed (%s) | final pnl=%+.4f SOL",
                pos.symbol,
                reason,
                pos.pnl_sol,
            )
        else:
            log.error("[%s] Sell order FAILED during %s — retrying next tick", pos.symbol, reason)
