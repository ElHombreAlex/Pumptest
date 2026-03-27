#!/usr/bin/env python3
"""
liquidate_all.py — manually close all open positions in the DB.

Usage:
    python3 liquidate_all.py           # interactive confirmation
    python3 liquidate_all.py --force   # skip confirmation (for scripting)
    python3 liquidate_all.py --dry-run # show what would happen, no trades sent
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ── Config from .env ──────────────────────────────────────────────────────────

WALLET_PRIVATE_KEY: str = os.getenv("WALLET_PRIVATE_KEY", "")
WALLET_PUBLIC_KEY: str = os.getenv("WALLET_PUBLIC_KEY", "")
SOLANA_RPC_URL: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_RPC_FALLBACK_URLS: list[str] = [
    u.strip()
    for u in os.getenv("SOLANA_RPC_FALLBACK_URLS", "").split(",")
    if u.strip()
]
PUMPPORTAL_TRADE_URL: str = "https://pumpportal.fun/api/trade-local"
PUMPPORTAL_API_KEY: str = os.getenv("PUMPPORTAL_API_KEY", "")
SLIPPAGE: int = int(os.getenv("SLIPPAGE", "10"))
PRIORITY_FEE: float = float(os.getenv("PRIORITY_FEE", "0.00001"))
DB_PATH: str = os.getenv("DB_PATH", "positions.db")


# ── Keypair loading ───────────────────────────────────────────────────────────

def _load_keypair(private_key_b58: str):
    import base58  # type: ignore
    from solders.keypair import Keypair  # type: ignore
    secret = base58.b58decode(private_key_b58)
    return Keypair.from_bytes(secret)


# ── PumpPortal + RPC ──────────────────────────────────────────────────────────

async def _build_sell_tx(mint: str) -> Optional[bytes]:
    """Ask PumpPortal to build a serialised sell-100% transaction."""
    payload = {
        "publicKey": WALLET_PUBLIC_KEY,
        "action": "sell",
        "mint": mint,
        "amount": "100%",
        "denominatedInSol": "false",
        "slippage": SLIPPAGE,
        "priorityFee": PRIORITY_FEE,
        "pool": "auto",
    }
    headers = {}
    if PUMPPORTAL_API_KEY:
        headers["Authorization"] = f"Bearer {PUMPPORTAL_API_KEY}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                PUMPPORTAL_TRADE_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  PumpPortal API error {resp.status}: {body[:200]}")
                    return None
                return await resp.read()
    except asyncio.TimeoutError:
        print("  PumpPortal API timed out after 15s")
        return None
    except aiohttp.ClientError as exc:
        print(f"  HTTP error: {exc}")
        return None


async def _sign_and_send(tx_bytes: bytes, keypair) -> Optional[str]:
    """Deserialise, sign, and broadcast via RPC with fallover."""
    from solders.transaction import VersionedTransaction  # type: ignore
    from solana.rpc.async_api import AsyncClient  # type: ignore
    from solana.rpc.types import TxOpts  # type: ignore

    try:
        unsigned_tx = VersionedTransaction.from_bytes(tx_bytes)
        signed_tx = VersionedTransaction(unsigned_tx.message, [keypair])
        serialised = bytes(signed_tx)
    except Exception as exc:
        print(f"  Failed to deserialise/sign transaction: {exc}")
        return None

    urls = [SOLANA_RPC_URL] + SOLANA_RPC_FALLBACK_URLS
    rpc_clients = [AsyncClient(url, timeout=30) for url in urls if url]

    for i, rpc in enumerate(rpc_clients):
        try:
            result = await rpc.send_raw_transaction(
                serialised,
                opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"),
            )
            if result.value:
                if i > 0:
                    print(f"  (landed via fallback RPC #{i})")
                return str(result.value)
            print(f"  RPC #{i} returned no signature: {result}")
        except asyncio.TimeoutError:
            print(f"  RPC #{i} timed out (>30s), trying next...")
        except Exception as exc:
            print(f"  RPC #{i} failed: {exc}")

    print(f"  All {len(rpc_clients)} RPC endpoint(s) failed")
    return None


async def sell_position(mint: str, keypair) -> bool:
    """Attempt to sell 100% of a position. Returns True on success."""
    tx_bytes = await _build_sell_tx(mint)
    if not tx_bytes:
        return False
    sig = await _sign_and_send(tx_bytes, keypair)
    if not sig:
        return False
    print(f"  Sell confirmed: {sig}")
    return True


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_open_positions(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM positions WHERE status = 'open'").fetchall()
    return [dict(row) for row in rows]


def close_position(conn: sqlite3.Connection, mint: str, note: str) -> None:
    conn.execute(
        "UPDATE positions SET status = ?, closed_at = ? WHERE mint = ?",
        (note, time.time(), mint),
    )
    conn.commit()


# ── Display ───────────────────────────────────────────────────────────────────

def fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_position(pos: dict) -> None:
    pnl = pos.get("pnl_sol", 0.0) or 0.0
    pnl_sign = "+" if pnl >= 0 else ""
    print(
        f"  mint:        {pos['mint']}\n"
        f"  symbol:      {pos['symbol']}\n"
        f"  entry_mcap:  {pos['entry_market_cap']:.4f} SOL\n"
        f"  current_mcap:{pos['current_market_cap']:.4f} SOL\n"
        f"  pnl:         {pnl_sign}{pnl:.6f} SOL\n"
        f"  opened_at:   {fmt_time(pos.get('opened_at'))}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Liquidate all open positions")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show positions without sending trades")
    args = parser.parse_args()

    dry_run: bool = args.dry_run

    # Validate config for live mode
    if not dry_run:
        if not WALLET_PRIVATE_KEY:
            print("ERROR: WALLET_PRIVATE_KEY not set in .env")
            return
        if not WALLET_PUBLIC_KEY:
            print("ERROR: WALLET_PUBLIC_KEY not set in .env")
            return

    # Open DB
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    positions = load_open_positions(conn)

    if not positions:
        print("No open positions found.")
        conn.close()
        return

    print(f"\nFound {len(positions)} open position(s):\n")
    for i, pos in enumerate(positions, 1):
        print(f"[{i}/{len(positions)}]")
        print_position(pos)
        print()

    if dry_run:
        print("[DRY-RUN] No trades will be sent.")
        conn.close()
        return

    # Confirmation
    if not args.force:
        answer = input(f"Found {len(positions)} open position(s). Liquidate all? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            conn.close()
            return

    # Load keypair once
    try:
        keypair = _load_keypair(WALLET_PRIVATE_KEY)
    except Exception as exc:
        print(f"ERROR: Failed to load keypair: {exc}")
        conn.close()
        return

    # Liquidate
    print()
    succeeded = 0
    failed = 0

    for i, pos in enumerate(positions, 1):
        symbol = pos["symbol"]
        mint = pos["mint"]
        print(f"[{i}/{len(positions)}] Selling {symbol} ({mint[:12]}...)...")

        ok = await sell_position(mint, keypair)

        if ok:
            close_position(conn, mint, "manually_liquidated")
            print(f"  => closed as 'manually_liquidated'")
            succeeded += 1
        else:
            close_position(conn, mint, "liquidation_failed")
            print(f"  => sell failed; closed as 'liquidation_failed'")
            failed += 1

        print()

    conn.close()

    # Summary
    print("=" * 50)
    print(f"Liquidation complete.")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(positions)}")


if __name__ == "__main__":
    asyncio.run(main())
