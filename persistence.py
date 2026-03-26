"""
SQLite-backed position persistence.

Positions are written to disk whenever they are opened or closed, so the agent
can resume tracking them after a restart without losing state.

Usage:
    store = PositionStore()                   # opens/creates positions.db
    store.save(position)                      # insert or update a row
    open_positions = store.load_open()        # called once at startup
    store.close()                             # optional explicit close
"""
from __future__ import annotations

import logging
import sqlite3
from typing import List

from models import Position, PositionStatus

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "positions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    mint                    TEXT    PRIMARY KEY,
    symbol                  TEXT    NOT NULL,
    entry_sol               REAL    NOT NULL,
    entry_market_cap        REAL    NOT NULL,
    token_amount            REAL    NOT NULL,
    entry_price_per_token   REAL    NOT NULL,
    take_profit_market_cap  REAL    NOT NULL,
    stop_loss_market_cap    REAL    NOT NULL,
    status                  TEXT    NOT NULL DEFAULT 'open',
    current_market_cap      REAL    NOT NULL DEFAULT 0,
    pnl_sol                 REAL    NOT NULL DEFAULT 0,
    confidence_score        INTEGER NOT NULL DEFAULT 0,
    opened_at               REAL    NOT NULL,
    closed_at               REAL
);
"""

_MIGRATIONS = [
    "ALTER TABLE positions ADD COLUMN confidence_score INTEGER NOT NULL DEFAULT 0",
]


class PositionStore:
    """Thin SQLite wrapper for persisting Position objects."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        # check_same_thread=False is safe here because asyncio runs all
        # callbacks on a single OS thread
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._apply_migrations()
        log.debug("Position store initialised at %s", db_path)

    def _apply_migrations(self) -> None:
        """Idempotently add any columns that didn't exist in older DB files."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    def save(self, pos: Position) -> None:
        """Insert or update a position row (upsert by mint)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions (
                mint, symbol, entry_sol, entry_market_cap, token_amount,
                entry_price_per_token, take_profit_market_cap, stop_loss_market_cap,
                status, current_market_cap, pnl_sol, confidence_score, opened_at, closed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pos.mint,
                pos.symbol,
                pos.entry_sol,
                pos.entry_market_cap,
                pos.token_amount,
                pos.entry_price_per_token,
                pos.take_profit_market_cap,
                pos.stop_loss_market_cap,
                pos.status.value,
                pos.current_market_cap,
                pos.pnl_sol,
                pos.confidence_score,
                pos.opened_at,
                pos.closed_at,
            ),
        )
        self._conn.commit()

    def load_open(self) -> List[Position]:
        """Return all positions whose status is still 'open'."""
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status = 'open'"
        ).fetchall()

        positions: List[Position] = []
        for row in rows:
            try:
                pos = Position(
                    mint=row["mint"],
                    symbol=row["symbol"],
                    entry_sol=row["entry_sol"],
                    entry_market_cap=row["entry_market_cap"],
                    token_amount=row["token_amount"],
                    entry_price_per_token=row["entry_price_per_token"],
                    take_profit_market_cap=row["take_profit_market_cap"],
                    stop_loss_market_cap=row["stop_loss_market_cap"],
                    status=PositionStatus(row["status"]),
                    current_market_cap=row["current_market_cap"],
                    pnl_sol=row["pnl_sol"],
                    confidence_score=row["confidence_score"] or 0,
                    opened_at=row["opened_at"],
                    closed_at=row["closed_at"],
                )
                positions.append(pos)
            except Exception as exc:
                log.warning("Skipping corrupt position row %s: %s", row["mint"], exc)

        return positions

    def close(self) -> None:
        self._conn.close()
