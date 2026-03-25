"""
Configuration for the Pump.fun Trading Agent.
Copy .env.example to .env and fill in your values.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Anthropic / Claude ────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # ── Solana wallet ─────────────────────────────────────────────────────────
    WALLET_PRIVATE_KEY: str = os.getenv("WALLET_PRIVATE_KEY", "")  # base58
    WALLET_PUBLIC_KEY: str = os.getenv("WALLET_PUBLIC_KEY", "")

    # ── RPC endpoint (Helius / QuickNode recommended for production) ──────────
    SOLANA_RPC_URL: str = os.getenv(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )
    # Optional comma-separated list of fallback RPC endpoints tried in order
    # when the primary endpoint fails or rate-limits a transaction
    SOLANA_RPC_FALLBACK_URLS: list[str] = [
        u.strip()
        for u in os.getenv("SOLANA_RPC_FALLBACK_URLS", "").split(",")
        if u.strip()
    ]

    # ── PumpPortal API ────────────────────────────────────────────────────────
    PUMPPORTAL_WS_URL: str = "wss://pumpportal.fun/api/data"
    PUMPPORTAL_TRADE_URL: str = "https://pumpportal.fun/api/trade-local"
    PUMPPORTAL_API_KEY: str = os.getenv("PUMPPORTAL_API_KEY", "")

    # ── Risk management ───────────────────────────────────────────────────────
    # Max SOL to spend per trade
    MAX_BUY_SOL: float = float(os.getenv("MAX_BUY_SOL", "0.01"))
    # Minimum AI confidence score (0-100) required to buy
    MIN_CONFIDENCE_SCORE: int = int(os.getenv("MIN_CONFIDENCE_SCORE", "70"))
    # Take-profit multiplier (1.5 = sell when price is 50% above entry)
    TAKE_PROFIT_MULTIPLIER: float = float(os.getenv("TAKE_PROFIT_MULTIPLIER", "1.5"))
    # Stop-loss multiplier (0.7 = sell when price drops 30% below entry)
    STOP_LOSS_MULTIPLIER: float = float(os.getenv("STOP_LOSS_MULTIPLIER", "0.7"))
    # Max open positions at once
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))
    # Slippage tolerance %
    SLIPPAGE: int = int(os.getenv("SLIPPAGE", "10"))
    # Priority fee in SOL
    PRIORITY_FEE: float = float(os.getenv("PRIORITY_FEE", "0.00001"))

    # ── Token filtering ───────────────────────────────────────────────────────
    # Skip tokens where name/symbol looks like a rug or scam
    ENABLE_AI_FILTER: bool = os.getenv("ENABLE_AI_FILTER", "true").lower() == "true"
    # Only analyze if initial buy activity meets threshold (SOL volume in first trades)
    MIN_INITIAL_SOL_VOLUME: float = float(os.getenv("MIN_INITIAL_SOL_VOLUME", "0.5"))

    # ── Claude rate limiting ───────────────────────────────────────────────────
    # Max Claude API calls per minute. Excess calls queue and wait rather than
    # failing, so analysis is delayed rather than dropped.
    MAX_CLAUDE_RPM: int = int(os.getenv("MAX_CLAUDE_RPM", "20"))

    # ── Warm-up window before Claude analysis ─────────────────────────────────
    # Collect at least this many trades before asking Claude to score a token
    WARM_UP_TRADES: int = int(os.getenv("WARM_UP_TRADES", "10"))
    # …or wait this many seconds after token creation, whichever comes first
    WARM_UP_SECONDS: int = int(os.getenv("WARM_UP_SECONDS", "20"))

    # ── Stale position eviction ───────────────────────────────────────────────
    # If an open position hasn't received a trade update in this many minutes,
    # it is considered stale (token is dead/illiquid) and a sell is triggered.
    STALE_POSITION_MINUTES: int = int(os.getenv("STALE_POSITION_MINUTES", "30"))

    # ── Dry-run mode: log decisions but don't send transactions ───────────────
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"


cfg = Config()
