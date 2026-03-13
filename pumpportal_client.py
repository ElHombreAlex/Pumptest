"""
PumpPortal WebSocket client.

Maintains a single persistent connection to wss://pumpportal.fun/api/data
and dispatches token-creation and trade events to registered callbacks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

from models import TokenEvent, TradeEvent, TradeAction

log = logging.getLogger(__name__)

# Type aliases
TokenCallback = Callable[[TokenEvent], Awaitable[None]]
TradeCallback = Callable[[TradeEvent], Awaitable[None]]

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
RECONNECT_DELAY = 5  # seconds between reconnect attempts


class PumpPortalClient:
    """
    Async WebSocket client for PumpPortal real-time data.

    Usage:
        client = PumpPortalClient()
        client.on_new_token(my_token_handler)
        client.on_new_trade(my_trade_handler)
        await client.run()          # blocks; reconnects automatically
    """

    def __init__(self) -> None:
        self._token_callbacks: list[TokenCallback] = []
        self._trade_callbacks: list[TradeCallback] = []
        self._subscribed_mints: set[str] = set()
        self._ws = None
        self._running = False

    # ── Public subscription helpers ───────────────────────────────────────────

    def on_new_token(self, cb: TokenCallback) -> None:
        self._token_callbacks.append(cb)

    def on_new_trade(self, cb: TradeCallback) -> None:
        self._trade_callbacks.append(cb)

    async def subscribe_token(self, mint: str) -> None:
        """Subscribe to real-time trades for a specific token mint."""
        if mint in self._subscribed_mints:
            return
        self._subscribed_mints.add(mint)
        if self._ws:
            try:
                await self._ws.send(
                    json.dumps({"method": "subscribeTokenTrade", "keys": [mint]})
                )
                log.debug("Subscribed to trades for %s", mint)
            except Exception as exc:
                log.warning("Could not subscribe to %s: %s", mint, exc)

    async def unsubscribe_token(self, mint: str) -> None:
        """Stop receiving trade events for a token."""
        self._subscribed_mints.discard(mint)
        if self._ws:
            try:
                await self._ws.send(
                    json.dumps({"method": "unsubscribeTokenTrade", "keys": [mint]})
                )
            except Exception:
                pass

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as exc:
                log.error("WebSocket error: %s — reconnecting in %ds", exc, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    def stop(self) -> None:
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_and_listen(self) -> None:
        log.info("Connecting to PumpPortal WebSocket …")
        async with websockets.connect(
            PUMPPORTAL_WS,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            log.info("Connected. Subscribing to new tokens …")

            # Subscribe to new token creations
            await ws.send(json.dumps({"method": "subscribeNewToken"}))

            # Re-subscribe to any tokens we were tracking before reconnect
            if self._subscribed_mints:
                await ws.send(
                    json.dumps(
                        {
                            "method": "subscribeTokenTrade",
                            "keys": list(self._subscribed_mints),
                        }
                    )
                )

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._dispatch(data)
                except json.JSONDecodeError:
                    log.warning("Received non-JSON message: %s", raw[:120])
                except Exception as exc:
                    log.exception("Error dispatching message: %s", exc)

        self._ws = None

    async def _dispatch(self, data: dict) -> None:
        """Route an incoming message to the right callbacks."""
        # New token creation
        if "mint" in data and "traderPublicKey" not in data:
            event = _parse_token_event(data)
            if event:
                for cb in self._token_callbacks:
                    await cb(event)
            return

        # Trade event
        if "traderPublicKey" in data and "mint" in data:
            event = _parse_trade_event(data)
            if event:
                for cb in self._trade_callbacks:
                    await cb(event)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_token_event(data: dict) -> TokenEvent | None:
    try:
        return TokenEvent(
            mint=data.get("mint", ""),
            name=data.get("name", ""),
            symbol=data.get("symbol", ""),
            description=data.get("description", ""),
            image_uri=data.get("image", ""),
            metadata_uri=data.get("metadataUri", ""),
            twitter=data.get("twitter", ""),
            telegram=data.get("telegram", ""),
            website=data.get("website", ""),
            creator=data.get("traderPublicKey", ""),
            created_timestamp=data.get("timestamp", time.time()),
            initial_buy_sol=_lamports_to_sol(data.get("initialBuy", 0)),
            initial_market_cap=data.get("marketCapSol", 0.0),
        )
    except Exception as exc:
        log.debug("Could not parse token event: %s | %s", exc, data)
        return None


def _parse_trade_event(data: dict) -> TradeEvent | None:
    try:
        return TradeEvent(
            mint=data.get("mint", ""),
            trader=data.get("traderPublicKey", ""),
            action=TradeAction.BUY if data.get("txType") == "buy" else TradeAction.SELL,
            sol_amount=_lamports_to_sol(data.get("solAmount", 0)),
            token_amount=data.get("tokenAmount", 0),
            new_market_cap=data.get("marketCapSol", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )
    except Exception as exc:
        log.debug("Could not parse trade event: %s | %s", exc, data)
        return None


def _lamports_to_sol(lamports: int | float) -> float:
    return lamports / 1_000_000_000 if lamports else 0.0
