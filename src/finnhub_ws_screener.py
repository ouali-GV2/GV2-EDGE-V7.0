"""
FINNHUB WEBSOCKET SCREENER V8
==============================

C1 FIX (P1 + P6 CRITIQUES):
- P1: Finnhub REST 60 req/min = 6% couverture/cycle → WebSocket = 100%
- P6: Tout etait en polling → WebSocket = streaming temps reel

Ce module remplace le polling REST pour les donnees de marche Finnhub
par un flux WebSocket temps reel. Les trades arrivent en push, plus
besoin de faire des requetes HTTP individuelles.

Architecture:
- WebSocket connection to wss://ws.finnhub.io
- Subscribe to trade updates for universe tickers
- Callbacks on volume spikes, price moves, new trades
- Auto-reconnect with exponential backoff
- Feeds TickerStateBuffer (V8) and Hot Ticker Queue

Usage:
    screener = FinnhubWSScreener(api_key="your_key")
    screener.on_volume_spike(callback)
    screener.on_price_move(callback)
    await screener.start(tickers=["AAPL", "TSLA", ...])

Note: Finnhub free tier allows 1 WebSocket connection with unlimited symbols.
This replaces ~3000 REST calls/cycle with 1 persistent connection.
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field

from utils.logger import get_logger
from config import FINNHUB_API_KEY

logger = get_logger("FINNHUB_WS")


# ============================================================================
# Configuration
# ============================================================================

FINNHUB_WS_URL = "wss://ws.finnhub.io"

# Volume spike detection
VOLUME_SPIKE_THRESHOLD = 3.0      # 3x average = spike
VOLUME_WINDOW_SECONDS = 300       # 5 min rolling window

# Price move detection
PRICE_MOVE_THRESHOLD_PCT = 0.03   # 3% move = significant
PRICE_WINDOW_SECONDS = 300        # 5 min rolling window

# Reconnection
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_BASE_DELAY = 2          # seconds
RECONNECT_MAX_DELAY = 60          # seconds

# Subscription batching
SUBSCRIBE_BATCH_SIZE = 50         # Subscribe 50 at a time
SUBSCRIBE_BATCH_DELAY = 0.1       # 100ms between batches


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TickerTradeState:
    """Rolling trade state for a ticker."""
    ticker: str
    trades: List[Dict] = field(default_factory=list)
    volume_1min: float = 0.0
    volume_5min: float = 0.0
    volume_baseline: float = 0.0     # Average volume per 5 min (from history)
    last_price: float = 0.0
    price_5min_ago: float = 0.0
    last_update: Optional[datetime] = None

    def add_trade(self, price: float, volume: float, timestamp: float):
        """Add a trade and update rolling state."""
        now = datetime.utcnow()
        trade = {"price": price, "volume": volume, "ts": timestamp}
        self.trades.append(trade)

        self.last_price = price
        self.last_update = now

        # Cleanup old trades (keep last 10 min)
        cutoff = time.time() - 600
        self.trades = [t for t in self.trades if t["ts"] > cutoff]

        # Recalculate rolling volumes
        now_ts = time.time()
        self.volume_1min = sum(
            t["volume"] for t in self.trades
            if t["ts"] > now_ts - 60
        )
        self.volume_5min = sum(
            t["volume"] for t in self.trades
            if t["ts"] > now_ts - 300
        )

        # Track price from 5 min ago
        trades_5min = [t for t in self.trades if t["ts"] < now_ts - 240]
        if trades_5min:
            self.price_5min_ago = trades_5min[-1]["price"]
        elif self.price_5min_ago == 0:
            self.price_5min_ago = price

    @property
    def volume_ratio(self) -> float:
        """Current volume vs baseline."""
        if self.volume_baseline <= 0:
            return 0.0
        return self.volume_5min / self.volume_baseline

    @property
    def price_change_pct(self) -> float:
        """Price change over last 5 min."""
        if self.price_5min_ago <= 0:
            return 0.0
        return (self.last_price - self.price_5min_ago) / self.price_5min_ago


@dataclass
class WSScreenerEvent:
    """Event emitted by the screener."""
    ticker: str
    event_type: str          # VOLUME_SPIKE, PRICE_MOVE, TRADE_BURST
    timestamp: datetime
    price: float
    volume_ratio: float
    price_change_pct: float
    details: Dict = field(default_factory=dict)


# ============================================================================
# Finnhub WebSocket Screener
# ============================================================================

class FinnhubWSScreener:
    """
    Real-time Finnhub WebSocket screener.

    Replaces REST polling with streaming trades for all universe tickers.
    One persistent connection handles unlimited symbols (Finnhub free tier).
    """

    def __init__(self, api_key: str = None):
        self._api_key = api_key or FINNHUB_API_KEY
        self._ws = None
        self._running = False
        self._connected = False

        # Subscribed tickers
        self._subscribed: Set[str] = set()

        # Trade state per ticker
        self._states: Dict[str, TickerTradeState] = {}

        # Callbacks
        self._volume_spike_callbacks: List[Callable] = []
        self._price_move_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []

        # Stats
        self._trades_received = 0
        self._events_emitted = 0
        self._reconnections = 0
        self._start_time: Optional[datetime] = None

        # Volume baselines (loaded from history)
        self._baselines: Dict[str, float] = {}

    # ========================================================================
    # Public API
    # ========================================================================

    def on_volume_spike(self, callback: Callable) -> None:
        """Register callback for volume spike events."""
        self._volume_spike_callbacks.append(callback)

    def on_price_move(self, callback: Callable) -> None:
        """Register callback for price move events."""
        self._price_move_callbacks.append(callback)

    def on_trade(self, callback: Callable) -> None:
        """Register callback for all trades (high frequency)."""
        self._trade_callbacks.append(callback)

    def set_volume_baselines(self, baselines: Dict[str, float]) -> None:
        """
        Set average volume baselines per ticker (from historical data).

        baselines: {ticker: avg_volume_per_5min}
        """
        self._baselines = baselines
        # Update existing states
        for ticker, baseline in baselines.items():
            if ticker in self._states:
                self._states[ticker].volume_baseline = baseline

    async def start(self, tickers: List[str]) -> None:
        """
        Start the WebSocket screener.

        Connects to Finnhub WS and subscribes to all tickers.
        Runs until stop() is called.
        """
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )
            return

        self._running = True
        self._start_time = datetime.utcnow()
        attempt = 0

        logger.info(f"Starting Finnhub WS Screener for {len(tickers)} tickers")

        while self._running:
            try:
                url = f"{FINNHUB_WS_URL}?token={self._api_key}"

                async with websockets.connect(url, ping_interval=30) as ws:
                    self._ws = ws
                    self._connected = True
                    attempt = 0

                    logger.info("Connected to Finnhub WebSocket")

                    # Subscribe to all tickers in batches
                    await self._subscribe_batch(tickers)

                    # Listen for messages
                    async for message in ws:
                        if not self._running:
                            break
                        self._handle_message(message)

            except Exception as e:
                self._connected = False
                attempt += 1
                self._reconnections += 1

                if attempt > MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        f"Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) "
                        f"reached. Stopping WS screener."
                    )
                    break

                delay = min(
                    RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                    RECONNECT_MAX_DELAY
                )
                logger.warning(
                    f"WS disconnected: {e}. "
                    f"Reconnecting in {delay}s (attempt {attempt})"
                )
                await asyncio.sleep(delay)

        self._running = False
        self._connected = False
        logger.info("Finnhub WS Screener stopped")

    def stop(self) -> None:
        """Stop the WebSocket screener."""
        self._running = False
        logger.info("Stopping Finnhub WS Screener...")

    def get_ticker_state(self, ticker: str) -> Optional[TickerTradeState]:
        """Get current trade state for a ticker."""
        return self._states.get(ticker.upper())

    def get_active_spikes(self) -> List[WSScreenerEvent]:
        """Get all tickers currently showing volume spikes."""
        events = []
        now = datetime.utcnow()

        for ticker, state in self._states.items():
            if state.volume_ratio >= VOLUME_SPIKE_THRESHOLD:
                events.append(WSScreenerEvent(
                    ticker=ticker,
                    event_type="VOLUME_SPIKE",
                    timestamp=now,
                    price=state.last_price,
                    volume_ratio=state.volume_ratio,
                    price_change_pct=state.price_change_pct,
                ))

        events.sort(key=lambda e: e.volume_ratio, reverse=True)
        return events

    def get_stats(self) -> Dict:
        """Get screener statistics."""
        uptime = 0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "connected": self._connected,
            "subscribed_tickers": len(self._subscribed),
            "active_states": len(self._states),
            "trades_received": self._trades_received,
            "events_emitted": self._events_emitted,
            "reconnections": self._reconnections,
            "uptime_seconds": round(uptime),
            "trades_per_second": round(
                self._trades_received / max(1, uptime), 1
            ),
        }

    # ========================================================================
    # Internal methods
    # ========================================================================

    async def _subscribe_batch(self, tickers: List[str]) -> None:
        """Subscribe to tickers in batches to avoid overwhelming the WS."""
        for i in range(0, len(tickers), SUBSCRIBE_BATCH_SIZE):
            batch = tickers[i:i + SUBSCRIBE_BATCH_SIZE]

            for ticker in batch:
                ticker = ticker.upper()
                if ticker not in self._subscribed:
                    msg = json.dumps({"type": "subscribe", "symbol": ticker})
                    await self._ws.send(msg)
                    self._subscribed.add(ticker)

                    # Initialize state
                    if ticker not in self._states:
                        self._states[ticker] = TickerTradeState(
                            ticker=ticker,
                            volume_baseline=self._baselines.get(ticker, 0)
                        )

            if i + SUBSCRIBE_BATCH_SIZE < len(tickers):
                await asyncio.sleep(SUBSCRIBE_BATCH_DELAY)

        logger.info(f"Subscribed to {len(self._subscribed)} tickers")

    def _handle_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            msg = json.loads(raw_message)

            if msg.get("type") == "ping":
                return

            if msg.get("type") != "trade":
                return

            trades = msg.get("data", [])

            for trade in trades:
                symbol = trade.get("s", "")
                price = trade.get("p", 0)
                volume = trade.get("v", 0)
                timestamp = trade.get("t", 0) / 1000  # ms to seconds

                if not symbol or price <= 0:
                    continue

                self._trades_received += 1

                # Update state
                state = self._states.get(symbol)
                if state:
                    state.add_trade(price, volume, timestamp)

                    # Check for events
                    self._check_events(state)

                    # Notify trade callbacks (if any)
                    for cb in self._trade_callbacks:
                        try:
                            cb(symbol, price, volume)
                        except Exception:
                            pass

        except json.JSONDecodeError:
            logger.debug(f"Invalid WS message: {raw_message[:100]}")
        except Exception as e:
            logger.debug(f"WS message error: {e}")

    def _check_events(self, state: TickerTradeState) -> None:
        """Check if a ticker state triggers any events."""
        now = datetime.utcnow()

        # Volume spike
        if state.volume_ratio >= VOLUME_SPIKE_THRESHOLD:
            event = WSScreenerEvent(
                ticker=state.ticker,
                event_type="VOLUME_SPIKE",
                timestamp=now,
                price=state.last_price,
                volume_ratio=state.volume_ratio,
                price_change_pct=state.price_change_pct,
                details={
                    "volume_5min": state.volume_5min,
                    "volume_baseline": state.volume_baseline,
                }
            )
            self._events_emitted += 1

            for cb in self._volume_spike_callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.debug(f"Volume spike callback error: {e}")

        # Price move
        if abs(state.price_change_pct) >= PRICE_MOVE_THRESHOLD_PCT:
            event = WSScreenerEvent(
                ticker=state.ticker,
                event_type="PRICE_MOVE",
                timestamp=now,
                price=state.last_price,
                volume_ratio=state.volume_ratio,
                price_change_pct=state.price_change_pct,
                details={
                    "price_5min_ago": state.price_5min_ago,
                    "direction": "UP" if state.price_change_pct > 0 else "DOWN",
                }
            )
            self._events_emitted += 1

            for cb in self._price_move_callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.debug(f"Price move callback error: {e}")


# ============================================================================
# Singleton
# ============================================================================

_screener_instance: Optional[FinnhubWSScreener] = None


def get_finnhub_ws_screener() -> FinnhubWSScreener:
    """Get singleton FinnhubWSScreener instance."""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = FinnhubWSScreener()
    return _screener_instance


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "FinnhubWSScreener",
    "TickerTradeState",
    "WSScreenerEvent",
    "get_finnhub_ws_screener",
]
