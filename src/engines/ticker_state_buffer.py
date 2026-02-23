"""
TICKER STATE BUFFER V8 - Ring Buffer for Acceleration Tracking
==============================================================

Core V8 module: Provides the data foundation for anticipatory detection.

Why this exists (P5 CRITICAL from REVIEW):
- V7 uses instantaneous values: momentum = abs(price_change), volume = ratio
- V7 detects stocks that ALREADY MOVED, not those ABOUT TO MOVE
- V8 needs derivatives (velocity, acceleration) which require TIME SERIES

This module:
1. Maintains a circular buffer of TickerSnapshots (price, volume, spread, etc.)
2. Computes rolling velocity (1st derivative) and acceleration (2nd derivative)
3. Computes z-scores vs 20-day baseline for anomaly detection
4. Identifies the ACCUMULATING state (volume accelerating, price stable)
   → This is the EARLIEST detectable signal before a breakout

Buffer capacity: 120 entries per ticker (2 hours at 1-min intervals)
Memory: ~50 bytes/snapshot × 120 × 200 tickers = ~1.2MB (negligible)
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class TickerSnapshot:
    """Single point-in-time snapshot of a ticker's state."""
    timestamp: datetime
    price: float
    volume: int              # Period volume (not cumulative)
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0  # (ask - bid) / mid
    vwap: float = 0.0


@dataclass
class DerivativeState:
    """Computed derivatives for a ticker at current time."""
    # Price derivatives
    price_velocity: float = 0.0         # $/min (1st derivative)
    price_acceleration: float = 0.0     # $/min² (2nd derivative)
    price_velocity_pct: float = 0.0     # %/min

    # Volume derivatives
    volume_velocity: float = 0.0        # shares/min change
    volume_acceleration: float = 0.0    # shares/min² change
    volume_ratio: float = 0.0           # Current vs baseline

    # Spread derivatives
    spread_velocity: float = 0.0        # Spread change rate
    spread_tightening: bool = False     # Spread getting smaller (bullish)

    # Z-scores vs 20-day baseline
    price_zscore: float = 0.0           # Price move anomaly
    volume_zscore: float = 0.0          # Volume anomaly
    spread_zscore: float = 0.0          # Spread anomaly

    # Composite state
    accumulation_score: float = 0.0     # 0-1: volume up + price flat = accumulation
    breakout_readiness: float = 0.0     # 0-1: how close to breakout

    # Classification
    state: str = "DORMANT"              # DORMANT/ACCUMULATING/LAUNCHING/BREAKOUT/EXHAUSTED

    # Metadata
    samples: int = 0                    # Number of samples used
    confidence: float = 0.0             # Data quality (0-1)

    def is_accumulating(self) -> bool:
        return self.state == "ACCUMULATING"

    def is_launching(self) -> bool:
        return self.state in ("LAUNCHING", "BREAKOUT")


@dataclass
class BaselineStats:
    """20-day baseline statistics for z-score computation."""
    price_mean: float = 0.0
    price_std: float = 0.01   # Avoid division by zero
    volume_mean: float = 1.0
    volume_std: float = 1.0
    spread_mean: float = 0.01
    spread_std: float = 0.005
    last_updated: Optional[datetime] = None


# ============================================================================
# Ticker State Buffer
# ============================================================================

class TickerStateBuffer:
    """
    Ring buffer that stores time-series snapshots for each ticker.

    Features:
    - O(1) append (deque)
    - O(n) derivative computation (but n ≤ 120, so fast)
    - Memory-efficient: auto-evicts old data
    - Thread-safe for single-writer/multi-reader (deque is GIL-protected)
    """

    def __init__(self, max_snapshots: int = 120, derivative_window: int = 5):
        """
        Args:
            max_snapshots: Max entries per ticker (default 120 = 2h at 1-min)
            derivative_window: Number of recent samples for derivative calc
        """
        self._buffers: Dict[str, deque] = {}
        self._baselines: Dict[str, BaselineStats] = {}
        self._max_snapshots = max_snapshots
        self._derivative_window = derivative_window

    def push(self, ticker: str, snapshot: TickerSnapshot) -> None:
        """Add a new snapshot for a ticker."""
        ticker = ticker.upper()
        if ticker not in self._buffers:
            self._buffers[ticker] = deque(maxlen=self._max_snapshots)
        self._buffers[ticker].append(snapshot)

    def push_raw(
        self,
        ticker: str,
        price: float,
        volume: int,
        bid: float = 0.0,
        ask: float = 0.0,
        vwap: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Convenience method to push raw values."""
        ts = timestamp or datetime.utcnow()
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else price
        spread_pct = (ask - bid) / mid if mid > 0 and bid > 0 else 0.0

        self.push(ticker, TickerSnapshot(
            timestamp=ts,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            vwap=vwap
        ))

    def set_baseline(self, ticker: str, baseline: BaselineStats) -> None:
        """Set 20-day baseline statistics for a ticker."""
        self._baselines[ticker.upper()] = baseline

    def set_baseline_raw(
        self,
        ticker: str,
        avg_price: float,
        price_std: float,
        avg_volume: float,
        volume_std: float,
        avg_spread: float = 0.01,
        spread_std: float = 0.005
    ) -> None:
        """Convenience method to set baseline from raw values."""
        self._baselines[ticker.upper()] = BaselineStats(
            price_mean=avg_price,
            price_std=max(price_std, 0.01),
            volume_mean=max(avg_volume, 1.0),
            volume_std=max(volume_std, 1.0),
            spread_mean=max(avg_spread, 0.001),
            spread_std=max(spread_std, 0.001),
            last_updated=datetime.utcnow()
        )

    def get_derivative_state(self, ticker: str) -> DerivativeState:
        """
        Compute current derivative state for a ticker.

        Returns DerivativeState with velocity, acceleration, z-scores, and state.
        """
        ticker = ticker.upper()
        buf = self._buffers.get(ticker)

        if not buf or len(buf) < 3:
            return DerivativeState(
                state="DORMANT",
                samples=len(buf) if buf else 0,
                confidence=0.0
            )

        snapshots = list(buf)
        n = len(snapshots)
        w = min(self._derivative_window, n)
        baseline = self._baselines.get(ticker, BaselineStats())

        # === Price derivatives ===
        recent_prices = [s.price for s in snapshots[-w:]]
        price_velocities = self._compute_velocities(recent_prices)
        price_vel = price_velocities[-1] if price_velocities else 0.0
        price_acc = self._compute_acceleration(price_velocities)

        base_price = snapshots[0].price if snapshots[0].price > 0 else 1.0
        price_vel_pct = price_vel / base_price

        # === Volume derivatives ===
        recent_volumes = [s.volume for s in snapshots[-w:]]
        vol_velocities = self._compute_velocities(recent_volumes)
        vol_vel = vol_velocities[-1] if vol_velocities else 0.0
        vol_acc = self._compute_acceleration(vol_velocities)

        current_vol = snapshots[-1].volume
        vol_ratio = current_vol / baseline.volume_mean if baseline.volume_mean > 0 else 1.0

        # === Spread derivatives ===
        recent_spreads = [s.spread_pct for s in snapshots[-w:]]
        spread_velocities = self._compute_velocities(recent_spreads)
        spread_vel = spread_velocities[-1] if spread_velocities else 0.0
        spread_tightening = spread_vel < 0  # Negative = tightening (bullish)

        # === Z-scores ===
        price_change = (snapshots[-1].price - snapshots[0].price) / base_price
        price_z = price_change / baseline.price_std if baseline.price_std > 0 else 0.0
        volume_z = (current_vol - baseline.volume_mean) / baseline.volume_std if baseline.volume_std > 0 else 0.0
        spread_z = (snapshots[-1].spread_pct - baseline.spread_mean) / baseline.spread_std if baseline.spread_std > 0 else 0.0

        # === Accumulation detection ===
        # Key insight: volume accelerating + price STABLE = institutional accumulation
        # This is the EARLIEST detectable signal before a breakout
        abs_price_change = abs(price_change)
        is_price_stable = abs_price_change < 0.02  # <2% price change
        is_volume_rising = vol_vel > 0 and vol_acc > 0
        is_volume_anomaly = volume_z > 1.5

        accumulation_score = 0.0
        if is_volume_rising and is_price_stable:
            accumulation_score = min(1.0, volume_z / 3.0)  # Normalize z-score to 0-1
            if is_volume_anomaly:
                accumulation_score = min(1.0, accumulation_score * 1.3)
            if spread_tightening:
                accumulation_score = min(1.0, accumulation_score * 1.2)

        # === Breakout readiness ===
        breakout_readiness = 0.0
        if accumulation_score > 0.3:
            breakout_readiness = accumulation_score * 0.5
        if price_vel_pct > 0.001 and vol_acc > 0:  # Price starting to move with volume
            breakout_readiness = min(1.0, breakout_readiness + 0.3)
        if volume_z > 2.0 and abs(price_z) > 1.0:  # Both price and volume anomalous
            breakout_readiness = min(1.0, breakout_readiness + 0.2)

        # === State classification ===
        state = self._classify_state(
            price_vel_pct, price_acc, vol_vel, vol_acc,
            volume_z, price_z, accumulation_score, breakout_readiness
        )

        # === Confidence ===
        confidence = min(1.0, n / 20)  # Full confidence at 20+ samples

        return DerivativeState(
            price_velocity=price_vel,
            price_acceleration=price_acc,
            price_velocity_pct=price_vel_pct,
            volume_velocity=vol_vel,
            volume_acceleration=vol_acc,
            volume_ratio=vol_ratio,
            spread_velocity=spread_vel,
            spread_tightening=spread_tightening,
            price_zscore=price_z,
            volume_zscore=volume_z,
            spread_zscore=spread_z,
            accumulation_score=accumulation_score,
            breakout_readiness=breakout_readiness,
            state=state,
            samples=n,
            confidence=confidence
        )

    def get_snapshots(self, ticker: str, last_n: int = 0) -> List[TickerSnapshot]:
        """Get raw snapshots for a ticker."""
        ticker = ticker.upper()
        buf = self._buffers.get(ticker)
        if not buf:
            return []
        snapshots = list(buf)
        return snapshots[-last_n:] if last_n > 0 else snapshots

    def get_tracked_tickers(self) -> List[str]:
        """Get list of tickers currently tracked."""
        return list(self._buffers.keys())

    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "tracked_tickers": len(self._buffers),
            "baselines_set": len(self._baselines),
            "total_snapshots": sum(len(b) for b in self._buffers.values()),
            "avg_snapshots_per_ticker": (
                sum(len(b) for b in self._buffers.values()) / max(1, len(self._buffers))
            )
        }

    def clear(self, ticker: Optional[str] = None) -> None:
        """Clear buffer for a ticker or all tickers."""
        if ticker:
            self._buffers.pop(ticker.upper(), None)
        else:
            self._buffers.clear()

    # ========================================================================
    # Internal computation methods
    # ========================================================================

    @staticmethod
    def _compute_velocities(values: List[float]) -> List[float]:
        """Compute 1st derivative (velocity) from a list of values."""
        if len(values) < 2:
            return []
        return [values[i] - values[i - 1] for i in range(1, len(values))]

    @staticmethod
    def _compute_acceleration(velocities: List[float]) -> float:
        """Compute 2nd derivative (acceleration) from velocities."""
        if len(velocities) < 2:
            return 0.0
        # Use last two velocities for current acceleration
        return velocities[-1] - velocities[-2]

    @staticmethod
    def _classify_state(
        price_vel_pct: float,
        price_acc: float,
        vol_vel: float,
        vol_acc: float,
        volume_z: float,
        price_z: float,
        accumulation_score: float,
        breakout_readiness: float
    ) -> str:
        """
        Classify ticker state based on derivatives.

        States (in order of progression):
        - DORMANT: No significant activity
        - ACCUMULATING: Volume rising but price stable (EARLIEST signal)
        - LAUNCHING: Price starting to move with volume confirmation
        - BREAKOUT: Clear breakout in progress (price + volume both strong)
        - EXHAUSTED: Move decelerating, momentum fading
        """
        # EXHAUSTED: Price was moving but now decelerating
        if price_vel_pct > 0.005 and price_acc < -0.001 and vol_acc < 0:
            return "EXHAUSTED"

        # BREAKOUT: Strong price + volume move
        if price_z > 2.0 and volume_z > 2.0 and price_vel_pct > 0.003:
            return "BREAKOUT"

        # LAUNCHING: Price starting to move with volume
        if breakout_readiness > 0.5 and price_vel_pct > 0.001:
            return "LAUNCHING"

        # ACCUMULATING: Volume rising, price stable (key V8 state)
        if accumulation_score > 0.3:
            return "ACCUMULATING"

        return "DORMANT"


# ============================================================================
# Singleton
# ============================================================================

_buffer_instance: Optional[TickerStateBuffer] = None


def get_ticker_state_buffer() -> TickerStateBuffer:
    """Get singleton TickerStateBuffer instance."""
    global _buffer_instance
    if _buffer_instance is None:
        _buffer_instance = TickerStateBuffer()
    return _buffer_instance
