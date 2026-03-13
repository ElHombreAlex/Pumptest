"""
Token analyzer powered by Claude AI.

Given a new token's metadata and its first few trades, Claude scores the token
on a 0-100 confidence scale and recommends whether to BUY, WATCH, or SKIP.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from config import cfg
from models import AnalysisResult, TokenEvent, TradeEvent
from bonding_curve import BondingCurveCalculator, BondingCurveState

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Solana memecoin analyst specialising in Pump.fun token launches.
Your job is to evaluate newly launched tokens and assess their short-term trading potential.

You must output ONLY valid JSON — no markdown fences, no extra text.

Output format:
{
  "confidence_score": <integer 0-100>,
  "recommendation": "<BUY|WATCH|SKIP>",
  "reasoning": "<2-4 sentence explanation>",
  "risk_flags": ["<flag1>", "<flag2>"],
  "suggested_buy_sol": <float>
}

Scoring guidelines:
- 80-100 → Strong buy signals, low red flags
- 60-79  → Moderate interest, some risk present
- 40-59  → Uncertain, recommend WATCH
- 0-39   → High risk, recommend SKIP

Key signals to evaluate:
POSITIVE: Engaging name/description, active social links, reasonable initial buy, growing trade momentum
NEGATIVE: Generic/scam name, copy-paste description, no socials, huge dev buy (>50% supply), rapid dump pattern

Risk flags to check:
- "rug_risk"           — dev bought >30% of supply initially
- "no_socials"         — no Twitter/Telegram/website
- "pump_and_dump"      — price rose sharply and is now falling
- "copy_token"         — name/symbol matches a known token
- "low_liquidity"      — market cap below 2 SOL
- "inactive_creator"   — creator has no on-chain history
- "suspicious_name"    — all caps, excessive symbols, misleading
"""

_USER_TEMPLATE = """\
New Pump.fun token launched. Analyse it:

## Token Metadata
- Name:        {name}
- Symbol:      {symbol}
- Description: {description}
- Twitter:     {twitter}
- Telegram:    {telegram}
- Website:     {website}
- Creator:     {creator}
- Created:     {created_timestamp}

## Launch Metrics
- Initial dev buy est. (SOL):  {initial_buy_sol:.4f}
- Initial market cap (SOL):    {initial_market_cap:.2f}
- Graduation progress:         {graduation_pct:.1f}%  (100% = migrates to PumpSwap at ~85 SOL)
- Price impact if I buy {max_buy_sol} SOL: {price_impact:.2f}%

## First {trade_count} Trade(s)
{trade_summary}

## Agent Context
- Max SOL I can spend: {max_buy_sol}
- My confidence threshold: {min_score}/100
"""


class TokenAnalyzer:
    """Uses Claude to score and filter new tokens."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self._cache: dict[str, AnalysisResult] = {}

    async def analyse(
        self,
        token: TokenEvent,
        recent_trades: list[TradeEvent],
    ) -> AnalysisResult:
        """
        Ask Claude to evaluate the token and return a structured AnalysisResult.
        Results are cached by mint address.
        """
        if token.mint in self._cache:
            return self._cache[token.mint]

        trade_summary = _format_trades(recent_trades)

        # Compute bonding curve metrics if we have reserve data
        bc_state = BondingCurveState(
            virtual_token_reserves=1_073_000_000_000_000,  # approximation if no live data
            virtual_sol_reserves=max(token.v_sol_reserves, 30.0),
            real_token_reserves=793_100_000_000_000,
            real_sol_reserves=0.0,
        )
        graduation_pct = bc_state.graduation_progress_pct
        price_impact = BondingCurveCalculator.price_impact_pct(bc_state, cfg.MAX_BUY_SOL)

        user_msg = _USER_TEMPLATE.format(
            name=token.name or "Unknown",
            symbol=token.symbol or "???",
            description=(token.description or "No description")[:400],
            twitter=token.twitter or "none",
            telegram=token.telegram or "none",
            website=token.website or "none",
            creator=token.creator or "unknown",
            created_timestamp=token.created_timestamp,
            initial_buy_sol=token.initial_buy_sol,
            initial_market_cap=token.initial_market_cap,
            graduation_pct=graduation_pct,
            price_impact=price_impact,
            trade_count=len(recent_trades),
            trade_summary=trade_summary or "No trades yet.",
            max_buy_sol=cfg.MAX_BUY_SOL,
            min_score=cfg.MIN_CONFIDENCE_SCORE,
        )

        log.info("Analysing token %s (%s) with Claude …", token.symbol, token.mint[:8])

        try:
            response = self._client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            payload: dict[str, Any] = json.loads(raw)

            result = AnalysisResult(
                mint=token.mint,
                confidence_score=int(payload.get("confidence_score", 0)),
                recommendation=payload.get("recommendation", "SKIP"),
                reasoning=payload.get("reasoning", ""),
                risk_flags=payload.get("risk_flags", []),
                suggested_buy_sol=float(payload.get("suggested_buy_sol", 0.0)),
            )

        except json.JSONDecodeError as exc:
            log.error("Claude returned non-JSON for %s: %s", token.mint[:8], exc)
            result = _skip_result(token.mint, "Claude response was not valid JSON")
        except anthropic.APIError as exc:
            log.error("Anthropic API error: %s", exc)
            result = _skip_result(token.mint, f"API error: {exc}")

        self._cache[token.mint] = result
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_trades(trades: list[TradeEvent]) -> str:
    if not trades:
        return ""
    lines = []
    for t in trades[:10]:  # show at most 10 trades to Claude
        lines.append(
            f"  [{t.action.value.upper()}] {t.sol_amount:.4f} SOL | "
            f"mcap={t.new_market_cap:.2f} SOL | "
            f"trader={t.trader[:8]}…"
        )
    return "\n".join(lines)


def _skip_result(mint: str, reason: str) -> AnalysisResult:
    return AnalysisResult(
        mint=mint,
        confidence_score=0,
        recommendation="SKIP",
        reasoning=reason,
        risk_flags=["analysis_failed"],
        suggested_buy_sol=0.0,
    )
