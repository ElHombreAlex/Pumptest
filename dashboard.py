"""
Optional Rich live dashboard for the trading agent.

Usage (started automatically when DASHBOARD_ENABLED=true):
    dashboard = Dashboard(agent, refresh_secs=cfg.DASHBOARD_REFRESH_SECS)
    dashboard.start()          # spawns a background asyncio task
    ...
    dashboard.stop()           # cancels the task on shutdown
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from agent import TradingAgent

log = logging.getLogger(__name__)
_console = Console()


class Dashboard:
    """Renders a periodically-refreshed Rich table of open positions + session stats."""

    def __init__(self, agent: "TradingAgent", refresh_secs: float = 5.0) -> None:
        self._agent = agent
        self._refresh_secs = refresh_secs
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background refresh loop as an asyncio task."""
        self._task = asyncio.create_task(self._run(), name="dashboard")
        log.info("Dashboard started (refresh every %.0fs)", self._refresh_secs)

    def stop(self) -> None:
        """Cancel the background task."""
        if self._task and not self._task.done():
            self._task.cancel()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            with Live(
                self._render(),
                console=_console,
                refresh_per_second=1,
                screen=False,
                transient=False,
            ) as live:
                while True:
                    await asyncio.sleep(self._refresh_secs)
                    live.update(self._render())
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Dashboard error — dashboard stopped")

    def _render(self) -> Table:
        risk = self._agent._risk
        memory = getattr(self._agent, "_memory", None)

        table = Table(
            title="[bold cyan]Pump.fun AI Trading Agent[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            expand=False,
            show_lines=False,
        )
        table.add_column("Symbol", style="cyan", min_width=10)
        table.add_column("Entry mcap", justify="right")
        table.add_column("Now mcap", justify="right")
        table.add_column("Peak mcap", justify="right")
        table.add_column("PnL (SOL)", justify="right", min_width=12)
        table.add_column("TP / SL", justify="right")

        for pos in risk._positions.values():
            pnl_color = "green" if pos.pnl_sol >= 0 else "red"
            pnl_text = Text(f"{pos.pnl_sol:+.4f}", style=pnl_color)
            table.add_row(
                pos.symbol,
                f"{pos.entry_market_cap:.2f}",
                f"{pos.current_market_cap:.2f}",
                f"{pos.peak_mcap:.2f}",
                pnl_text,
                f"{pos.take_profit_market_cap:.1f} / {pos.stop_loss_market_cap:.1f}",
            )

        if not risk._positions:
            table.add_row("[dim]–[/dim]", "", "", "", "", "")

        closed = risk._closed
        total_pnl = sum(p.pnl_sol for p in closed)
        pnl_color = "green" if total_pnl >= 0 else "red"
        table.add_section()
        table.add_row(
            f"[bold]Closed: {len(closed)}[/bold]",
            "", "", "",
            Text(f"{total_pnl:+.4f}", style=f"bold {pnl_color}"),
            "",
        )

        if memory:
            summary = memory.get_summary()
            table.add_section()
            table.add_row(f"[dim]{summary}[/dim]", "", "", "", "", "")

        return table
