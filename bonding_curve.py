"""
Pump.fun bonding curve calculator.

Implements the constant-product AMM (x * y = k) used by the
Pump.fun on-chain program (ID: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P).

Reference: https://github.com/pump-fun/pump-public-docs
DeepWiki: https://deepwiki.com/pump-fun/pump-public-docs/3.1-pump-bonding-curve-mechanism
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Constants ─────────────────────────────────────────────────────────────────

PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Default initial virtual reserves when a token is created
INITIAL_VIRTUAL_TOKEN_RESERVES = 1_073_000_000_000_000   # 1.073 × 10^15 (6 dp tokens)
INITIAL_VIRTUAL_SOL_RESERVES   =        30_000_000_000   # 30 SOL in lamports
INITIAL_REAL_TOKEN_RESERVES    =   793_100_000_000_000   # tradeable supply

# Graduation threshold: ~85.5 SOL raised → token migrates to PumpSwap
GRADUATION_SOL_THRESHOLD = 85.0


@dataclass
class BondingCurveState:
    """
    Snapshot of a bonding curve account's reserve state.
    All values as returned by PumpPortal (SOL in decimal, tokens in base units).
    """
    virtual_token_reserves: float
    virtual_sol_reserves: float    # SOL (not lamports)
    real_token_reserves: float
    real_sol_reserves: float
    complete: bool = False

    @classmethod
    def from_initial(cls) -> "BondingCurveState":
        """Return a fresh bonding curve at token genesis."""
        return cls(
            virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
            virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES / 1e9,
            real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
            real_sol_reserves=0.0,
        )

    # ── Price helpers ─────────────────────────────────────────────────────────

    @property
    def price_per_token_sol(self) -> float:
        """Spot price of 1 token in SOL (using virtual reserves)."""
        if not self.virtual_token_reserves:
            return 0.0
        return self.virtual_sol_reserves / (self.virtual_token_reserves / 1e6)

    @property
    def market_cap_sol(self) -> float:
        """
        Total market cap in SOL.
        Formula: virtual_sol_reserves * 1_000_000_000 / virtual_token_reserves
        (matches Pump.fun's own display — uses lamport-denominated formula)
        """
        if not self.virtual_token_reserves:
            return 0.0
        return (self.virtual_sol_reserves * 1e9) / self.virtual_token_reserves

    @property
    def graduation_progress_pct(self) -> float:
        """How far along the bonding curve (0–100%)."""
        if not INITIAL_REAL_TOKEN_RESERVES:
            return 0.0
        return 100.0 - (
            (self.real_token_reserves * 100.0) / INITIAL_REAL_TOKEN_RESERVES
        )


class BondingCurveCalculator:
    """
    Stateless helper that computes buy/sell costs using the constant-product AMM.

    All SOL values are in decimal SOL (not lamports).
    Token amounts are in base units (Pump.fun tokens have 6 decimal places,
    so 1 visible token = 1_000_000 base units, but bonding curve uses
    its own large-integer repr).
    """

    @staticmethod
    def buy_cost(
        state: BondingCurveState,
        token_amount: float,
    ) -> float:
        """
        How much SOL is required to buy `token_amount` base units.
        """
        k = state.virtual_token_reserves * state.virtual_sol_reserves
        new_virtual_tokens = state.virtual_token_reserves - token_amount
        if new_virtual_tokens <= 0:
            return float("inf")
        new_virtual_sol = k / new_virtual_tokens
        return new_virtual_sol - state.virtual_sol_reserves

    @staticmethod
    def tokens_for_sol(
        state: BondingCurveState,
        sol_amount: float,
    ) -> float:
        """
        How many base-unit tokens you receive for `sol_amount` SOL.
        """
        k = state.virtual_token_reserves * state.virtual_sol_reserves
        new_virtual_sol = state.virtual_sol_reserves + sol_amount
        new_virtual_tokens = k / new_virtual_sol
        return state.virtual_token_reserves - new_virtual_tokens

    @staticmethod
    def sell_output(
        state: BondingCurveState,
        token_amount: float,
    ) -> float:
        """
        SOL received for selling `token_amount` base units.
        """
        k = state.virtual_token_reserves * state.virtual_sol_reserves
        new_virtual_tokens = state.virtual_token_reserves + token_amount
        new_virtual_sol = k / new_virtual_tokens
        return state.virtual_sol_reserves - new_virtual_sol

    @staticmethod
    def price_impact_pct(
        state: BondingCurveState,
        sol_amount: float,
    ) -> float:
        """
        Estimated price impact % for buying `sol_amount` SOL worth of tokens.
        """
        spot = state.price_per_token_sol
        if spot == 0:
            return 100.0
        tokens = BondingCurveCalculator.tokens_for_sol(state, sol_amount)
        if tokens == 0:
            return 100.0
        effective_price = sol_amount / (tokens / 1e6)
        return abs(effective_price - spot) / spot * 100.0
