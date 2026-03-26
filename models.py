"""
Shared data models used across the agent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class TokenEvent:
    """Represents a new token creation event from PumpPortal."""
    mint: str
    name: str
    symbol: str
    description: str
    image_uri: str
    metadata_uri: str
    twitter: str
    telegram: str
    website: str
    creator: str
    created_timestamp: float
    # Populated as trades accumulate
    initial_buy_sol: float = 0.0
    initial_market_cap: float = 0.0
    trade_count: int = 0
    # Bonding curve virtual reserves at creation time (from vSolInBondingCurve /
    # vTokensInBondingCurve fields in the PumpPortal event)
    v_sol_reserves: float = 0.0
    v_token_reserves: float = 0.0


@dataclass
class TradeEvent:
    """Represents a real-time trade on a token."""
    mint: str
    trader: str
    action: TradeAction
    sol_amount: float
    token_amount: float
    new_market_cap: float
    # Latest virtual reserves from PumpPortal (vSolInBondingCurve / vTokensInBondingCurve)
    v_sol_reserves: float = 0.0
    v_token_reserves: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnalysisResult:
    """AI analysis result for a token."""
    mint: str
    confidence_score: int          # 0-100
    recommendation: str            # BUY / SKIP / WATCH
    reasoning: str
    risk_flags: list[str]
    suggested_buy_sol: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    """An open trading position."""
    mint: str
    symbol: str
    entry_sol: float               # SOL spent on buy
    entry_market_cap: float
    token_amount: float
    entry_price_per_token: float
    take_profit_market_cap: float
    stop_loss_market_cap: float
    status: PositionStatus = PositionStatus.OPEN
    current_market_cap: float = 0.0
    peak_mcap: float = 0.0          # highest market cap seen since entry (for trailing stop)
    pnl_sol: float = 0.0
    opened_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
