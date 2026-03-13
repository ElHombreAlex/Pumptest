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
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # ── Solana wallet ─────────────────────────────────────────────────────────
    WALLET_PRIVATE_KEY: str = os.getenv("WALLET_PRIVATE_KEY", "")  # base58
    WALLET_PUBLIC_KEY: str = os.getenv("WALLET_PUBLIC_KEY", "")

    # ── RPC endpoint (Helius / QuickNode recommended for production) ──────────
    SOLANA_RPC_URL: str = os.getenv(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )

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

    # ── Dry-run mode: log decisions but don't send transactions ───────────────
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"


cfg = Config()
