#!/usr/bin/env python3
"""
Pump.fun AI Trading Agent — entry point.

Usage:
    python main.py

Required environment variables (see .env.example):
    ANTHROPIC_API_KEY   — Claude AI key
    WALLET_PUBLIC_KEY   — Solana wallet public key
    WALLET_PRIVATE_KEY  — Solana wallet private key (base58), only if DRY_RUN=false
    DRY_RUN             — Set to "false" to execute real trades (default: true)

Recommended (for reliable Solana tx submission):
    SOLANA_RPC_URL      — Helius or QuickNode RPC endpoint
"""
import asyncio
import logging
import sys

from rich.logging import RichHandler

from config import cfg


LOG_FILE = "trading.log"


def _setup_logging() -> None:
    # Console handler — rich formatting for interactive use
    console_handler = RichHandler(rich_tracebacks=True, markup=False)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    # File handler — DEBUG level so raw WS frames and other diagnostic
    # messages are captured; console stays at INFO to avoid terminal noise.
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
    # Console stays INFO — only the file gets DEBUG output
    console_handler.setLevel(logging.INFO)

    # Quiet noisy libraries
    for lib in ("websockets", "asyncio", "aiohttp"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _validate_config() -> None:
    errors = []
    if not cfg.ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is not set")
    if not cfg.DRY_RUN:
        if not cfg.WALLET_PRIVATE_KEY:
            errors.append("WALLET_PRIVATE_KEY is required when DRY_RUN=false")
        if not cfg.WALLET_PUBLIC_KEY:
            errors.append("WALLET_PUBLIC_KEY is required when DRY_RUN=false")
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(1)


async def main() -> None:
    _setup_logging()
    _validate_config()

    log = logging.getLogger("main")

    mode = "DRY-RUN" if cfg.DRY_RUN else "LIVE TRADING"
    log.info("Starting Pump.fun AI Trading Agent — %s", mode)
    log.info(
        "Settings: max_buy=%.4f SOL | min_score=%d | max_positions=%d | TP=×%.1f | SL=×%.1f",
        cfg.MAX_BUY_SOL,
        cfg.MIN_CONFIDENCE_SCORE,
        cfg.MAX_POSITIONS,
        cfg.TAKE_PROFIT_MULTIPLIER,
        cfg.STOP_LOSS_MULTIPLIER,
    )

    # Lazy import so that missing deps produce a cleaner error
    from agent import TradingAgent
    from dashboard import Dashboard

    agent = TradingAgent()

    dash: Dashboard | None = None
    if cfg.DASHBOARD_ENABLED:
        dash = Dashboard(agent, refresh_secs=cfg.DASHBOARD_REFRESH_SECS)

    try:
        if dash:
            # Dashboard must be started inside the running event loop, after
            # the agent is constructed but before agent.run() blocks.
            asyncio.get_event_loop().call_soon(dash.start)
        await agent.run()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    finally:
        if dash:
            dash.stop()


if __name__ == "__main__":
    asyncio.run(main())
