"""
Token analyzer powered by Claude AI.

Given a new token's metadata and its first few trades, Claude scores the token
on a 0-100 confidence scale and recommends whether to BUY, WATCH, or SKIP.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from config import cfg
from models import AnalysisResult, TokenEvent, TradeEvent
from bonding_curve import (
    BondingCurveCalculator,
    BondingCurveState,
    INITIAL_VIRTUAL_TOKEN_RESERVES,
    INITIAL_VIRTUAL_SOL_RESERVES,
    INITIAL_REAL_TOKEN_RESERVES,
)

log = logging.getLogger(__name__)

# Difference between virtual and real token reserves at genesis.
# Used to derive real_token_reserves from the live virtual value.
_VIRTUAL_TOKEN_OFFSET = INITIAL_VIRTUAL_TOKEN_RESERVES - INITIAL_REAL_TOKEN_RESERVES

# Initial virtual SOL in decimal SOL (30 SOL), used to derive real_sol_reserves.
_INITIAL_VIRTUAL_SOL = INITIAL_VIRTUAL_SOL_RESERVES / 1e9  # lamports → SOL

_SYSTEM_PROMPT = """\
Respond with ONLY a raw JSON object, no markdown, no code blocks, no explanation.

You are a Pump.fun short-term trading analyst. Pump.fun tokens are inherently
speculative and ephemeral — they are NOT traditional crypto projects. Most have
no socials, no website, and no roadmap by design. Your job is NOT to find
high-quality projects; it is to identify tokens with strong early momentum that
are likely to pump further in the next few minutes so a small position can be
exited at profit.

Output format — a single JSON object, nothing else:
{
  "confidence_score": <integer 0-100>,
  "recommendation": "<BUY|WATCH|SKIP>",
  "reasoning": "<2-3 sentence explanation focused on momentum signals>",
  "risk_flags": ["<flag1>", "<flag2>"],
  "suggested_buy_sol": <float>
}

── SCORING FRAMEWORK ────────────────────────────────────────────────────────

Score 60–100 (BUY) — strong early momentum, likely to continue:
  • Multiple buys in early trades with consistent or growing trade sizes
  • Buy/sell ratio skewed heavily toward buys (>70% buys)
  • Market cap rising across the first trades (upward price trend)
  • Bonus +10–20 pts if Twitter, Telegram, or website is present
  • Bonus +5 pts for an engaging, original name/description

Score 38–59 (WATCH) — some positive signals but mixed:
  • Mix of buys and sells, no clear momentum direction
  • Early trades present but trade sizes are very small or erratic
  • Decent name/concept but no social presence
  • Market cap flat or slightly rising

Score 0–37 (SKIP) — clear dump signals or no activity:
  • Majority of early trades are SELLS (dump already underway)
  • Single large whale buy then immediate sells from other wallets
  • Zero trades after warm-up window (no interest)
  • Market cap falling across trades (price declining from open)

── IMPORTANT CALIBRATION NOTES ──────────────────────────────────────────────

- NO SOCIALS is normal on Pump.fun. Do NOT score below 38 purely because
  there are no social links. Treat it as neutral, not a red flag by itself.
- A token with 8 buys and 2 sells and a rising market cap should score 55–70
  even with no name recognition and no socials.
- Only flag "rug_risk" if the DEV buy was >50% of supply AND early sells
  are already visible. A large dev buy alone is not enough.
- "pump_and_dump" should only be flagged if price is ALREADY falling in the
  trade data — not speculatively.

Risk flags (only include flags that are clearly evidenced in the trade data):
- "early_dump"      — majority of early trades are sells, price falling
- "whale_and_dump"  — one huge buy immediately followed by many sells
- "zero_activity"   — fewer than 3 trades in the warm-up window
- "rug_risk"        — dev bought >50% of supply AND early dump visible
- "no_socials"      — only include if combined with other negative signals

Respond with ONLY a raw JSON object, no markdown, no code blocks, no explanation.\
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
        bc_state = _build_bonding_curve_state(token, recent_trades)
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
                model=cfg.ANTHROPIC_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            # Validate response structure before accessing fields
            if not response.content or not hasattr(response.content[0], "text"):
                raise ValueError("Unexpected Claude response structure")

            raw = response.content[0].text.strip()
            payload: dict[str, Any] = _parse_json_response(raw)

            # Clamp numeric fields to valid ranges — Claude can occasionally
            # return out-of-bounds values that would corrupt position sizing.
            raw_score = payload.get("confidence_score", 0)
            confidence_score = max(0, min(100, int(raw_score)))

            raw_buy = payload.get("suggested_buy_sol", 0.0)
            suggested_buy_sol = max(0.0, float(raw_buy))

            result = AnalysisResult(
                mint=token.mint,
                confidence_score=confidence_score,
                recommendation=payload.get("recommendation", "SKIP"),
                reasoning=payload.get("reasoning", ""),
                risk_flags=payload.get("risk_flags", []),
                suggested_buy_sol=suggested_buy_sol,
            )

        except json.JSONDecodeError as exc:
            log.warning(
                "Claude returned non-JSON for %s: %s | raw: %.200s",
                token.mint[:8], exc, raw if "raw" in dir() else "<no response>",
            )
            result = _skip_result(token.mint, "Claude response was not valid JSON")
        except anthropic.APIError as exc:
            log.error("Anthropic API error: %s", exc)
            result = _skip_result(token.mint, f"API error: {exc}")
        except Exception as exc:
            log.error("Unexpected error analysing %s: %s", token.mint[:8], exc)
            result = _skip_result(token.mint, f"Unexpected error: {exc}")

        self._cache[token.mint] = result
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from Claude's response.

    Attempt order:
      1. Direct parse of the stripped response (ideal path)
      2. Strip markdown code fences (```json ... ``` or ``` ... ```) then parse
      3. Substring extraction — find the first '{' and last '}' and parse that

    Raises json.JSONDecodeError if all three attempts fail, which the caller
    catches and logs with the raw response for diagnosis.
    """
    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences and retry
    cleaned = raw
    if "```" in cleaned:
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        # Remove closing fence
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 3. Substring extraction — grab everything between the first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    # All attempts failed — re-raise so the caller can log the raw text
    raise json.JSONDecodeError("No valid JSON object found in response", raw, 0)


def _build_bonding_curve_state(
    token: TokenEvent,
    recent_trades: list[TradeEvent],
) -> BondingCurveState:
    """
    Build a BondingCurveState from the freshest available reserve data.

    Priority order:
      1. Most recent trade event (most up-to-date snapshot)
      2. Token creation event (close to genesis state)
      3. Hard-coded initial constants (fallback if API omits reserve fields)

    Real reserves are derived from virtual reserves by subtracting the fixed
    virtual offsets baked into pump.fun's initial curve state:
      real_sol    = v_sol    - 30 SOL      (initial virtual SOL offset)
      real_tokens = v_tokens - 279.9T base units  (virtual − real at genesis)
    """
    v_sol: float = 0.0
    v_tokens: float = 0.0

    # Try the most recent trade first
    for trade in reversed(recent_trades):
        if trade.v_sol_reserves > 0 and trade.v_token_reserves > 0:
            v_sol = trade.v_sol_reserves
            v_tokens = trade.v_token_reserves
            break

    # Fall back to token creation event
    if not v_sol and token.v_sol_reserves > 0:
        v_sol = token.v_sol_reserves
    if not v_tokens and token.v_token_reserves > 0:
        v_tokens = token.v_token_reserves

    # Final fallback: initial genesis constants
    if not v_sol:
        v_sol = _INITIAL_VIRTUAL_SOL
    if not v_tokens:
        v_tokens = float(INITIAL_VIRTUAL_TOKEN_RESERVES)

    # PumpPortal returns vTokensInBondingCurve in decimal token units (~1e9),
    # but the bonding curve constants (_VIRTUAL_TOKEN_OFFSET) are in base units
    # (6 decimal places, ~1e15). Convert before subtracting.
    v_tokens_base = v_tokens * 1_000_000
    real_tokens = max(0.0, v_tokens_base - _VIRTUAL_TOKEN_OFFSET)
    real_sol = max(0.0, v_sol - _INITIAL_VIRTUAL_SOL)

    return BondingCurveState(
        virtual_token_reserves=v_tokens,
        virtual_sol_reserves=v_sol,
        real_token_reserves=real_tokens,
        real_sol_reserves=real_sol,
    )


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
