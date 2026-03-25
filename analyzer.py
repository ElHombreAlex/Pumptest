"""
Token analyzer powered by Claude AI.

Given a new token's metadata and its first few trades, Claude scores the token
on a 0-100 confidence scale and recommends whether to BUY, WATCH, or SKIP.
"""
from __future__ import annotations

import collections
import json
import logging
import re
import time
from typing import Any

import asyncio

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
Respond ONLY with a raw JSON object — no markdown, no explanation.

You are a Pump.fun momentum analyst. Score tokens on early trade signals only.
No socials/website is normal and NEUTRAL — never penalise for this alone.

Output (single JSON object, nothing else):
{"confidence_score":<0-100>,"recommendation":"<BUY|WATCH|SKIP>","reasoning":"<2 sentences>","risk_flags":[...],"suggested_buy_sol":<float>}

SCORE BANDS:
60-100 BUY  — >70% buys, rising mcap, consistent sizes. +10 if socials present.
38-59  WATCH — mixed signals, flat/slow mcap, erratic sizes.
0-37   SKIP  — majority sells, whale-then-dump, zero trades, falling mcap.

RISK FLAGS (only when clearly evidenced):
early_dump, whale_and_dump, zero_activity,
rug_risk (dev >50% supply AND sells already visible — not speculation)

CALIBRATION:
- 8 buys + 2 sells + rising mcap → score 55-70 even with no socials.
- pump_and_dump only if price is ALREADY falling in the data.

Respond ONLY with a raw JSON object.\
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


class _SlidingWindowRateLimiter:
    """
    Sliding-window rate limiter for the Claude API.

    Tracks the timestamps of recent calls in a deque.  Before each call,
    entries older than 60 seconds are dropped.  If the window is full
    (len >= max_rpm), the caller awaits until the oldest entry expires.
    """

    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max_rpm
        self._calls: collections.deque[float] = collections.deque()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            # Evict calls that are outside the 60-second window
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) < self._max_rpm:
                self._calls.append(now)
                return
            # Window is full — wait until the oldest call rolls off
            wait = 60.0 - (now - self._calls[0])
            log.debug(
                "Claude RPM cap reached (%d/%d) — waiting %.1fs",
                len(self._calls),
                self._max_rpm,
                wait,
            )
            await asyncio.sleep(max(0.05, wait))


class TokenAnalyzer:
    """Uses Claude to score and filter new tokens."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self._cache: dict[str, AnalysisResult] = {}
        self._rate_limiter = _SlidingWindowRateLimiter(cfg.MAX_CLAUDE_RPM)

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

        await self._rate_limiter.acquire()

        try:
            response = self._client.messages.create(
                model=cfg.ANTHROPIC_MODEL,
                max_tokens=512,
                # system is a list so we can attach cache_control, which tells
                # Anthropic to cache this (long, static) prompt between calls.
                # Cache hits cost ~10× less than input tokens, dramatically
                # reducing cost when many tokens are analysed per session.
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
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
