"""
SMALLCAP RADAR V8 - Anticipatory Intraday Top Gainer Detection
================================================================

The crown jewel of V8: an integrated radar that detects small-cap top gainers
BEFORE they become visible on public screeners.

Detection Layers (earliest → latest):
1. ACCUMULATING (5-15 min before move): Volume z-score rising, price flat
2. PRE_LAUNCH (1-5 min before move): Price starting to tick up with volume
3. LAUNCHING (at move start): Confirmed breakout beginning
4. CONFIRMED (after +5%): Standard V7 detection (too late for early entry)

Filters specific to small-cap top gainers:
- Market cap < $2B (small cap focus)
- Price $0.50-$20 (penny to small cap range)
- Average volume > 500K (tradeable liquidity)
- Float < 100M shares (tighter float = bigger moves)

Integration:
- Feeds AccelerationScore into MonsterScore V4
- Produces RadarSignal for SignalProducer V8
- Respects V8 Risk Guard (no longer blocked by S-3 shelfs)

Performance:
- Scan cycle: <500ms for 200 tickers (buffer reads only, no API calls)
- Memory: ~2MB for 200 tickers with full buffers
- Can run at 1-second intervals without bottleneck
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Callable
import logging

from src.engines.ticker_state_buffer import (
    TickerStateBuffer,
    DerivativeState,
    get_ticker_state_buffer,
)
from src.engines.acceleration_engine import (
    AccelerationEngine,
    AccelerationScore,
    AccelerationAlert,
    get_acceleration_engine,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Small-cap filter criteria
SMALLCAP_MAX_MARKET_CAP = 2_000_000_000     # $2B
SMALLCAP_MIN_PRICE = 0.50                    # $0.50
SMALLCAP_MAX_PRICE = 20.00                   # $20
# C2 FIX: Removed SMALLCAP_MIN_AVG_VOLUME = 500_000
# This filter was redundant (universe_loader already filters) and blocked
# low-float tickers that are prime top gainer candidates.
# Volume quality is assessed dynamically via z-scores instead.
SMALLCAP_MIN_AVG_VOLUME = 50_000             # 50K daily (minimal liquidity only)
SMALLCAP_MAX_FLOAT = 100_000_000             # 100M shares

# Radar sensitivity levels
RADAR_SENSITIVITY = {
    "ULTRA": {    # Catch everything (more false positives)
        "min_accumulation": 0.20,
        "min_volume_zscore": 1.0,
        "min_confidence": 0.20,
    },
    "HIGH": {     # Good balance (recommended)
        "min_accumulation": 0.30,
        "min_volume_zscore": 1.5,
        "min_confidence": 0.30,
    },
    "STANDARD": { # Conservative (fewer false positives)
        "min_accumulation": 0.45,
        "min_volume_zscore": 2.0,
        "min_confidence": 0.40,
    },
}

# Scan interval
RADAR_SCAN_INTERVAL_SECONDS = 5  # How often to scan (fast: buffer reads only)


# ============================================================================
# Data Classes
# ============================================================================

class RadarPriority:
    """Priority levels for radar signals."""
    CRITICAL = "CRITICAL"     # Breakout in progress, immediate attention
    HIGH = "HIGH"             # Pre-launch detected, prepare order
    MEDIUM = "MEDIUM"         # Accumulating, add to hot watchlist
    LOW = "LOW"               # Minor activity, monitor
    NONE = "NONE"


@dataclass
class RadarBlip:
    """
    A single detection on the radar.

    Represents a ticker showing unusual anticipatory activity
    before a potential move.
    """
    ticker: str
    timestamp: datetime

    # Detection
    priority: str = RadarPriority.NONE
    detection_phase: str = "DORMANT"     # ACCUMULATING/PRE_LAUNCH/LAUNCHING/BREAKOUT

    # Scores
    radar_score: float = 0.0              # Composite 0-1 score
    acceleration_score: float = 0.0       # From AccelerationEngine
    accumulation_score: float = 0.0       # Volume-based accumulation
    breakout_readiness: float = 0.0       # How close to breakout

    # Derivatives
    volume_zscore: float = 0.0
    price_zscore: float = 0.0
    price_velocity_pct: float = 0.0       # %/min price change rate

    # Context
    gap_quality: float = 0.0              # From PM Scanner V8 (if premarket)
    catalyst_active: bool = False          # Has active catalyst
    is_repeat_gainer: bool = False         # Previous top gainer history

    # Timing
    estimated_time_to_move: str = ""       # "1-5 min", "5-15 min", etc.
    confidence: float = 0.0

    # Monster Score integration
    monster_score_boost: float = 0.0       # Additive boost for MonsterScore

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "detection_phase": self.detection_phase,
            "radar_score": round(self.radar_score, 3),
            "acceleration_score": round(self.acceleration_score, 3),
            "accumulation_score": round(self.accumulation_score, 3),
            "breakout_readiness": round(self.breakout_readiness, 3),
            "volume_zscore": round(self.volume_zscore, 2),
            "price_zscore": round(self.price_zscore, 2),
            "price_velocity_pct": round(self.price_velocity_pct, 5),
            "estimated_time_to_move": self.estimated_time_to_move,
            "confidence": round(self.confidence, 3),
            "monster_score_boost": round(self.monster_score_boost, 4),
        }


@dataclass
class RadarScanResult:
    """Result of a full radar scan."""
    timestamp: datetime
    scan_duration_ms: float = 0.0
    tickers_scanned: int = 0

    # Blips by priority
    critical: List[RadarBlip] = field(default_factory=list)
    high: List[RadarBlip] = field(default_factory=list)
    medium: List[RadarBlip] = field(default_factory=list)
    low: List[RadarBlip] = field(default_factory=list)

    @property
    def all_blips(self) -> List[RadarBlip]:
        return self.critical + self.high + self.medium + self.low

    @property
    def actionable_count(self) -> int:
        return len(self.critical) + len(self.high)

    def get_top_n(self, n: int = 5) -> List[RadarBlip]:
        """Get top N blips by radar score."""
        all_sorted = sorted(self.all_blips, key=lambda b: b.radar_score, reverse=True)
        return all_sorted[:n]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "scan_duration_ms": round(self.scan_duration_ms, 1),
            "tickers_scanned": self.tickers_scanned,
            "counts": {
                "critical": len(self.critical),
                "high": len(self.high),
                "medium": len(self.medium),
                "low": len(self.low),
            },
            "top_blips": [b.to_dict() for b in self.get_top_n(5)],
        }


# ============================================================================
# SmallCap Radar
# ============================================================================

class SmallCapRadar:
    """
    V8 Anticipatory Small-Cap Top Gainer Radar.

    Usage:
        radar = SmallCapRadar()

        # Single scan
        result = radar.scan()
        for blip in result.critical:
            print(f"🚨 {blip.ticker}: {blip.detection_phase} score={blip.radar_score:.2f}")

        # Get score for MonsterScore integration
        boost = radar.get_monster_boost(ticker)

        # Continuous monitoring
        radar.on_blip(lambda blip: handle_detection(blip))
        await radar.run_continuous()
    """

    def __init__(
        self,
        sensitivity: str = "HIGH",
        buffer: Optional[TickerStateBuffer] = None,
        engine: Optional[AccelerationEngine] = None,
    ):
        self._buffer = buffer or get_ticker_state_buffer()
        self._engine = engine or get_acceleration_engine()
        self._sensitivity = RADAR_SENSITIVITY.get(sensitivity, RADAR_SENSITIVITY["HIGH"])

        # Ticker context (enriched from external sources)
        self._ticker_context: Dict[str, Dict] = {}

        # Callbacks
        self._blip_callbacks: List[Callable] = []

        # Stats
        self._total_scans = 0
        self._total_blips = 0
        self._running = False

    def set_ticker_context(self, ticker: str, context: Dict) -> None:
        """
        Set context for a ticker (from external sources).

        Context dict can include:
        - market_cap: float
        - float_shares: int
        - avg_volume: int
        - has_catalyst: bool
        - is_repeat_gainer: bool
        - gap_quality: float (from PM Scanner V8)
        """
        self._ticker_context[ticker.upper()] = context

    def on_blip(self, callback: Callable) -> None:
        """Register callback for radar blips."""
        self._blip_callbacks.append(callback)

    def scan(self) -> RadarScanResult:
        """
        Perform a full radar scan of all tracked tickers.

        Fast operation: reads buffer state only, no API calls.
        Typical execution: <500ms for 200 tickers.
        """
        import time
        start = time.monotonic()
        now = datetime.utcnow()

        result = RadarScanResult(timestamp=now)
        tickers = self._buffer.get_tracked_tickers()
        result.tickers_scanned = len(tickers)

        for ticker in tickers:
            blip = self._evaluate_ticker(ticker, now)
            if blip is None:
                continue

            # Categorize by priority
            if blip.priority == RadarPriority.CRITICAL:
                result.critical.append(blip)
            elif blip.priority == RadarPriority.HIGH:
                result.high.append(blip)
            elif blip.priority == RadarPriority.MEDIUM:
                result.medium.append(blip)
            elif blip.priority == RadarPriority.LOW:
                result.low.append(blip)

            self._total_blips += 1

            # Notify callbacks for actionable blips
            if blip.priority in (RadarPriority.CRITICAL, RadarPriority.HIGH):
                for cb in self._blip_callbacks:
                    try:
                        cb(blip)
                    except Exception as e:
                        logger.error(f"Blip callback error: {e}")

        elapsed = (time.monotonic() - start) * 1000
        result.scan_duration_ms = elapsed
        self._total_scans += 1

        # Log summary
        if result.actionable_count > 0:
            logger.info(
                f"RADAR SCAN: {result.tickers_scanned} tickers in {elapsed:.0f}ms → "
                f"🚨 {len(result.critical)} critical, "
                f"🔶 {len(result.high)} high, "
                f"🔹 {len(result.medium)} medium"
            )

        return result

    def get_monster_boost(self, ticker: str) -> float:
        """
        Get MonsterScore boost for a ticker based on radar state.

        Returns additive boost (0.0 to 0.18) for MonsterScore integration.
        """
        score = self._engine.score(ticker)
        return score.get_monster_score_boost()

    def get_acceleration_score(self, ticker: str) -> AccelerationScore:
        """Get full acceleration score for a ticker."""
        return self._engine.score(ticker)

    async def run_continuous(self, interval_seconds: float = RADAR_SCAN_INTERVAL_SECONDS):
        """
        Run radar continuously (async).

        Call this in the main event loop for real-time monitoring.
        """
        self._running = True
        logger.info(f"SmallCap Radar V8 started (interval={interval_seconds}s)")

        while self._running:
            try:
                result = self.scan()

                # Log critical detections
                for blip in result.critical:
                    logger.warning(
                        f"🚨 RADAR CRITICAL: {blip.ticker} "
                        f"phase={blip.detection_phase} "
                        f"score={blip.radar_score:.2f} "
                        f"vol_z={blip.volume_zscore:.1f} "
                        f"{blip.estimated_time_to_move}"
                    )

            except Exception as e:
                logger.error(f"Radar scan error: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop continuous radar."""
        self._running = False

    def get_stats(self) -> Dict:
        """Get radar statistics."""
        return {
            "total_scans": self._total_scans,
            "total_blips": self._total_blips,
            "sensitivity": self._sensitivity,
            "tracked_tickers": len(self._buffer.get_tracked_tickers()),
            "engine_stats": self._engine.get_stats(),
        }

    # ========================================================================
    # Internal methods
    # ========================================================================

    def _evaluate_ticker(self, ticker: str, now: datetime) -> Optional[RadarBlip]:
        """Evaluate a single ticker for radar detection."""
        accel_score = self._engine.score(ticker)

        # Skip dormant or low-data tickers
        if accel_score.state == "DORMANT" and accel_score.acceleration_score < 0.1:
            return None

        if accel_score.confidence < self._sensitivity["min_confidence"]:
            return None

        # Get derivative state for detailed info
        ds = self._buffer.get_derivative_state(ticker)

        # Get ticker context
        ctx = self._ticker_context.get(ticker, {})

        # Compute radar score (composite)
        radar_score = self._compute_radar_score(accel_score, ctx)

        # Determine detection phase and priority
        phase, priority = self._classify_detection(accel_score, ds, radar_score)

        if priority == RadarPriority.NONE:
            return None

        # Estimate time to move
        etm = self._estimate_time_to_move(accel_score, ds)

        # Compute monster score boost
        boost = accel_score.get_monster_score_boost()

        return RadarBlip(
            ticker=ticker,
            timestamp=now,
            priority=priority,
            detection_phase=phase,
            radar_score=radar_score,
            acceleration_score=accel_score.acceleration_score,
            accumulation_score=accel_score.accumulation_score,
            breakout_readiness=accel_score.breakout_readiness,
            volume_zscore=accel_score.volume_zscore,
            price_zscore=accel_score.price_zscore,
            price_velocity_pct=ds.price_velocity_pct,
            gap_quality=ctx.get("gap_quality", 0.0),
            catalyst_active=ctx.get("has_catalyst", False),
            is_repeat_gainer=ctx.get("is_repeat_gainer", False),
            estimated_time_to_move=etm,
            confidence=accel_score.confidence,
            monster_score_boost=boost,
        )

    def _compute_radar_score(self, accel: AccelerationScore, ctx: Dict) -> float:
        """
        Compute composite radar score (0-1).

        Factors:
        - Acceleration score: 50% (core V8 metric)
        - Volume z-score: 20% (anomaly strength)
        - Context boost: 15% (catalyst, repeat gainer)
        - Gap quality: 15% (PM gap sweet spot)
        """
        # Core acceleration
        core = accel.acceleration_score * 0.50

        # Volume anomaly (normalize z-score to 0-1)
        vol_norm = min(1.0, max(0.0, accel.volume_zscore / 4.0))
        vol_component = vol_norm * 0.20

        # Context boost
        ctx_score = 0.0
        if ctx.get("has_catalyst"):
            ctx_score += 0.5
        if ctx.get("is_repeat_gainer"):
            ctx_score += 0.3
        # Small float boost (tighter float = bigger potential move)
        float_shares = ctx.get("float_shares", 0)
        if 0 < float_shares < 20_000_000:
            ctx_score += 0.2  # Ultra low float
        elif 0 < float_shares < 50_000_000:
            ctx_score += 0.1  # Low float
        ctx_component = min(1.0, ctx_score) * 0.15

        # Gap quality (from PM Scanner V8)
        gap_component = ctx.get("gap_quality", 0.0) * 0.15

        return min(1.0, core + vol_component + ctx_component + gap_component)

    def _classify_detection(
        self,
        accel: AccelerationScore,
        ds: DerivativeState,
        radar_score: float
    ) -> Tuple[str, str]:
        """Classify detection phase and priority."""

        # BREAKOUT
        if accel.state == "BREAKOUT":
            return "BREAKOUT", RadarPriority.CRITICAL

        # LAUNCHING
        if accel.state == "LAUNCHING":
            return "LAUNCHING", RadarPriority.CRITICAL

        # PRE_LAUNCH: High readiness
        if (
            accel.breakout_readiness >= 0.50
            and accel.volume_zscore >= self._sensitivity["min_volume_zscore"]
        ):
            return "PRE_LAUNCH", RadarPriority.HIGH

        # ACCUMULATING
        if (
            accel.state == "ACCUMULATING"
            and accel.accumulation_score >= self._sensitivity["min_accumulation"]
        ):
            return "ACCUMULATING", RadarPriority.MEDIUM

        # LOW: Some activity but not enough
        if radar_score > 0.15:
            return "WATCHING", RadarPriority.LOW

        return "DORMANT", RadarPriority.NONE

    @staticmethod
    def _estimate_time_to_move(accel: AccelerationScore, ds: DerivativeState) -> str:
        """Estimate time before significant price move."""
        if accel.state == "BREAKOUT":
            return "NOW"
        if accel.state == "LAUNCHING":
            return "0-2 min"
        if accel.breakout_readiness > 0.6:
            return "1-5 min"
        if accel.state == "ACCUMULATING":
            if accel.accumulation_score > 0.6:
                return "3-10 min"
            return "5-15 min"
        return "15+ min"


# ============================================================================
# Singleton
# ============================================================================

_radar_instance: Optional[SmallCapRadar] = None


def get_smallcap_radar(sensitivity: str = "HIGH") -> SmallCapRadar:
    """Get singleton SmallCapRadar instance."""
    global _radar_instance
    if _radar_instance is None:
        _radar_instance = SmallCapRadar(sensitivity=sensitivity)
    return _radar_instance
