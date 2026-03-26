"""
In-session trade memory store.

Records closed-position outcomes so the analyzer can include recent
performance context in Claude prompts, improving decision quality over time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    """A record of a completed trade."""
    symbol: str
    recommendation: str      # BUY / WATCH / SKIP
    confidence_score: int
    pnl_sol: float
    close_reason: str        # take_profit / stop_loss / stale / trailing_stop
    entry_market_cap: float
    peak_market_cap: float
    closed_at: float = field(default_factory=time.time)


class MemoryStore:
    """Accumulates closed-trade outcomes for the current session."""

    def __init__(self, max_records: int = 200) -> None:
        self._records: list[TradeRecord] = []
        self._max = max_records

    def record(self, rec: TradeRecord) -> None:
        self._records.append(rec)
        if len(self._records) > self._max:
            self._records.pop(0)

    def get_summary(self) -> str:
        """Return a short text summary suitable for injecting into Claude prompts."""
        if not self._records:
            return "No closed trades this session yet."

        total = len(self._records)
        wins = sum(1 for r in self._records if r.pnl_sol > 0)
        total_pnl = sum(r.pnl_sol for r in self._records)
        avg_score = sum(r.confidence_score for r in self._records) / total

        lines = [
            f"Session: {total} trades | Win rate: {wins}/{total} "
            f"({wins * 100 // total}%) | PnL: {total_pnl:+.4f} SOL | "
            f"Avg confidence: {avg_score:.0f}",
            "Recent closes (last 5):",
        ]
        for r in self._records[-5:]:
            lines.append(
                f"  {r.symbol:10s} score={r.confidence_score:3d} "
                f"pnl={r.pnl_sol:+.4f} SOL  [{r.close_reason}]"
            )
        return "\n".join(lines)

    def all_records(self) -> list[TradeRecord]:
        return list(self._records)
