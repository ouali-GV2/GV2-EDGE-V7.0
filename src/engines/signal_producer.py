"""
SIGNAL PRODUCER V8.0
====================

Moteur de détection ILLIMITÉ - Enhanced with Anticipatory Detection.

V8 ENHANCEMENTS:
- NEW: AccelerationState integration (ACCUMULATING/LAUNCHING/BREAKOUT)
- NEW: Acceleration boost from V8 derivatives (velocity + z-scores)
- CHANGED: Pre-Spike boost now considers acceleration state
- CHANGED: EARLY_SIGNAL now triggered by ACCUMULATING state (earliest detection)

Principe fondamental:
- Produit des signaux 24/7
- AUCUNE limite de trades n'existe à ce niveau
- AUCUN blocage basé sur les contraintes d'exécution
- Le moteur "pense" toujours, l'exécution "décide" ensuite

Ce module est la COUCHE 1 de l'architecture V8:
1. SIGNAL PRODUCER (ici) → Détection pure (with V8 acceleration)
2. ORDER COMPUTER → Calcul d'ordres
3. EXECUTION GATE → Limites et autorisations

Sources de signaux:
- Monster Score V4 (with acceleration component)
- AccelerationEngine V8 (derivatives + z-scores)
- SmallCapRadar V8 (anticipatory detection)
- Pre-Spike Radar
- Event Hub
- Social Buzz
- NLP Classification
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
import uuid

from utils.logger import get_logger

# Import signal types
from src.models.signal_types import (
    SignalType,
    PreSpikeState,
    PreHaltState,
    UnifiedSignal,
)

logger = get_logger("SIGNAL_PRODUCER")


# ============================================================================
# Configuration
# ============================================================================

# Monster Score thresholds for signal generation
THRESHOLDS = {
    "BUY_STRONG": 0.80,      # Monster Score >= 0.80 = BUY_STRONG
    "BUY": 0.65,             # Monster Score >= 0.65 = BUY
    "WATCH": 0.50,           # Monster Score >= 0.50 = WATCH
    "EARLY_SIGNAL": 0.40,    # Monster Score >= 0.40 = EARLY_SIGNAL
}

# Pre-Spike state boosts
PRE_SPIKE_BOOSTS = {
    PreSpikeState.LAUNCHING: 0.10,    # +10% to monster score
    PreSpikeState.READY: 0.05,        # +5%
    PreSpikeState.CHARGING: 0.02,     # +2%
    PreSpikeState.DORMANT: 0.0,
    PreSpikeState.EXHAUSTED: -0.05,   # -5%
}

# Catalyst type confidence boosts
CATALYST_BOOSTS = {
    "FDA_APPROVAL": 0.15,
    "FDA_TRIAL_RESULT": 0.12,
    "EARNINGS_BEAT": 0.10,
    "GUIDANCE_RAISE": 0.10,
    "CONTRACT_WIN": 0.10,
    "MERGER_ACQUISITION": 0.12,
    "BUYOUT": 0.15,
    "PARTNERSHIP": 0.08,
    "ANALYST_UPGRADE": 0.05,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DetectionInput:
    """Input data for signal detection"""
    ticker: str
    current_price: float

    # Scores from other modules
    monster_score: float = 0.0
    catalyst_score: float = 0.0

    # Pre-Spike Radar
    pre_spike_state: PreSpikeState = PreSpikeState.DORMANT
    pre_spike_score: float = 0.0

    # Catalyst info
    catalyst_type: Optional[str] = None
    catalyst_confidence: float = 0.0
    catalyst_age_hours: float = 0.0

    # Social buzz
    social_buzz_score: float = 0.0
    social_acceleration: float = 0.0

    # Repeat gainer
    is_repeat_gainer: bool = False
    repeat_gainer_score: float = 0.0

    # Market context
    volume_ratio: float = 1.0        # Current vs average
    price_change_pct: float = 0.0
    market_session: str = "RTH"

    # V8: Acceleration Engine data
    acceleration_state: str = "DORMANT"     # DORMANT/ACCUMULATING/LAUNCHING/BREAKOUT/EXHAUSTED
    acceleration_score: float = 0.0         # 0-1 composite acceleration score
    volume_zscore: float = 0.0              # Volume z-score vs 20-day baseline
    accumulation_score: float = 0.0         # 0-1 accumulation detection
    breakout_readiness: float = 0.0         # 0-1 how close to breakout


@dataclass
class DetectionResult:
    """Result of signal detection"""
    ticker: str
    signal_type: SignalType
    raw_score: float                   # Before adjustments
    adjusted_score: float              # After boosts
    confidence: float
    reasons: List[str]
    timestamp: datetime


# ============================================================================
# Signal Producer
# ============================================================================

class SignalProducer:
    """
    Moteur de détection illimité

    Usage:
        producer = SignalProducer()

        # Process single ticker
        signal = await producer.detect(input_data)

        # Process batch
        signals = await producer.detect_batch(inputs)

        # Register callback for real-time signals
        producer.on_signal(lambda s: print(s))

    IMPORTANT:
    - Ce module ne vérifie JAMAIS les limites de trades
    - Il produit des signaux 24/7
    - Les contraintes d'exécution sont gérées par ExecutionGate
    """

    def __init__(self):
        # Configuration
        self.thresholds = THRESHOLDS.copy()

        # Callbacks
        self._signal_callbacks: List[Callable] = []

        # Stats
        self._signals_produced = 0
        self._signals_by_type: Dict[str, int] = {}

        # Cache for deduplication
        self._recent_signals: Dict[str, datetime] = {}
        self._signal_cooldown_seconds = 300  # 5 min between same ticker signals

    def configure(
        self,
        thresholds: Dict[str, float] = None,
        signal_cooldown: int = 300
    ):
        """Configure producer settings"""
        if thresholds:
            self.thresholds.update(thresholds)
        self._signal_cooldown_seconds = signal_cooldown

    def on_signal(self, callback: Callable):
        """Register callback for new signals"""
        self._signal_callbacks.append(callback)

    async def detect(self, input_data: DetectionInput) -> UnifiedSignal:
        """
        Detect signal for a single ticker

        IMPORTANT: This method NEVER checks execution limits.
        It always produces a signal based on market data.
        """
        ticker = input_data.ticker.upper()

        logger.debug(f"Detecting signal for {ticker}")

        # Step 1: Calculate adjusted score
        raw_score = input_data.monster_score
        adjusted_score, reasons = self._calculate_adjusted_score(input_data)

        # Step 2: Determine signal type
        signal_type = self._determine_signal_type(adjusted_score, input_data)

        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(input_data, adjusted_score)

        # Step 4: Create detection result
        result = DetectionResult(
            ticker=ticker,
            signal_type=signal_type,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            confidence=confidence,
            reasons=reasons,
            timestamp=datetime.utcnow()
        )

        # Step 5: Build unified signal
        signal = self._build_unified_signal(input_data, result)

        # Step 6: Update stats
        self._update_stats(signal)

        # Step 7: Notify callbacks (if actionable)
        if signal.is_actionable():
            await self._notify_callbacks(signal)

        return signal

    async def detect_batch(
        self,
        inputs: List[DetectionInput]
    ) -> List[UnifiedSignal]:
        """
        Detect signals for multiple tickers

        Processes in parallel for efficiency.
        """
        tasks = [self.detect(inp) for inp in inputs]
        signals = await asyncio.gather(*tasks)
        return list(signals)

    def _calculate_adjusted_score(
        self,
        input_data: DetectionInput
    ) -> tuple[float, List[str]]:
        """
        Calculate adjusted monster score with boosts

        Boosts are ADDITIVE, not multiplicative.
        """
        score = input_data.monster_score
        reasons = []

        # Pre-Spike boost
        spike_boost = PRE_SPIKE_BOOSTS.get(input_data.pre_spike_state, 0)
        if spike_boost != 0:
            score += spike_boost
            reasons.append(f"Pre-Spike {input_data.pre_spike_state.value}: {spike_boost:+.2f}")

        # Catalyst boost
        if input_data.catalyst_type:
            cat_boost = CATALYST_BOOSTS.get(input_data.catalyst_type, 0.05)
            # Scale by confidence
            cat_boost *= input_data.catalyst_confidence
            score += cat_boost
            reasons.append(f"Catalyst {input_data.catalyst_type}: {cat_boost:+.2f}")

        # Repeat gainer boost
        if input_data.is_repeat_gainer:
            rg_boost = min(0.10, input_data.repeat_gainer_score * 0.15)
            score += rg_boost
            reasons.append(f"Repeat Gainer: {rg_boost:+.2f}")

        # Social buzz boost (if acceleration detected)
        if input_data.social_acceleration > 2.0:
            buzz_boost = min(0.08, (input_data.social_acceleration - 2) * 0.02)
            score += buzz_boost
            reasons.append(f"Social Buzz x{input_data.social_acceleration:.1f}: {buzz_boost:+.2f}")

        # Volume confirmation
        if input_data.volume_ratio > 3.0:
            vol_boost = min(0.05, (input_data.volume_ratio - 3) * 0.01)
            score += vol_boost
            reasons.append(f"Volume {input_data.volume_ratio:.1f}x: {vol_boost:+.2f}")

        # V8: Acceleration state boost (derivative-based anticipation)
        if input_data.acceleration_state == "ACCUMULATING":
            accel_boost = min(0.08, 0.03 + input_data.accumulation_score * 0.05)
            score += accel_boost
            reasons.append(f"V8 Accumulating (score={input_data.accumulation_score:.2f}): {accel_boost:+.2f}")
        elif input_data.acceleration_state == "LAUNCHING":
            accel_boost = min(0.12, 0.06 + input_data.breakout_readiness * 0.06)
            score += accel_boost
            reasons.append(f"V8 Launching (readiness={input_data.breakout_readiness:.2f}): {accel_boost:+.2f}")
        elif input_data.acceleration_state == "BREAKOUT":
            accel_boost = min(0.15, 0.08 + input_data.acceleration_score * 0.07)
            score += accel_boost
            reasons.append(f"V8 Breakout (accel={input_data.acceleration_score:.2f}): {accel_boost:+.2f}")
        elif input_data.acceleration_state == "EXHAUSTED":
            score -= 0.05
            reasons.append("V8 Exhausted: -0.05")

        # V8: Volume z-score bonus (anomaly detection replaces absolute thresholds)
        if input_data.volume_zscore > 2.5:
            z_boost = min(0.06, (input_data.volume_zscore - 2.0) * 0.03)
            score += z_boost
            reasons.append(f"V8 Volume z={input_data.volume_zscore:.1f}: {z_boost:+.2f}")

        # Cap score at 1.0
        score = min(1.0, max(0.0, score))

        return score, reasons

    def _determine_signal_type(
        self,
        adjusted_score: float,
        input_data: DetectionInput
    ) -> SignalType:
        """
        Determine signal type based on adjusted score.

        V8: ACCUMULATING/LAUNCHING states can elevate signal type for
        earlier detection of potential top gainers.
        """

        # V8: BREAKOUT state = immediate BUY_STRONG if score supports it
        if input_data.acceleration_state == "BREAKOUT":
            if adjusted_score >= self.thresholds["BUY"] * 0.85:
                return SignalType.BUY_STRONG

        # Special case: LAUNCHING pre-spike can elevate signal
        if (
            input_data.pre_spike_state == PreSpikeState.LAUNCHING
            or input_data.acceleration_state == "LAUNCHING"
        ):
            if adjusted_score >= self.thresholds["BUY"] * 0.9:
                return SignalType.BUY_STRONG

        # Standard thresholds
        if adjusted_score >= self.thresholds["BUY_STRONG"]:
            return SignalType.BUY_STRONG

        if adjusted_score >= self.thresholds["BUY"]:
            return SignalType.BUY

        if adjusted_score >= self.thresholds["WATCH"]:
            return SignalType.WATCH

        if adjusted_score >= self.thresholds["EARLY_SIGNAL"]:
            if input_data.catalyst_type or input_data.pre_spike_state != PreSpikeState.DORMANT:
                return SignalType.EARLY_SIGNAL

        # V8: ACCUMULATING state can trigger EARLY_SIGNAL even below threshold
        # This is the KEY V8 innovation: detecting stocks BEFORE the move
        if (
            input_data.acceleration_state == "ACCUMULATING"
            and input_data.accumulation_score >= 0.4
            and adjusted_score >= self.thresholds["EARLY_SIGNAL"] * 0.85
        ):
            return SignalType.EARLY_SIGNAL

        # C3 FIX: Adaptive score floor for high volume z-score
        # Problem P3: tickers with volume z > 2.5 but no catalyst get NO_SIGNAL
        # even though abnormal volume alone is a strong precursor signal.
        # If volume z-score > 2.5, allow EARLY_SIGNAL even without catalyst.
        if (
            input_data.volume_zscore > 2.5
            and adjusted_score >= self.thresholds["EARLY_SIGNAL"] * 0.80
        ):
            return SignalType.EARLY_SIGNAL

        return SignalType.NO_SIGNAL

    def _calculate_confidence(
        self,
        input_data: DetectionInput,
        adjusted_score: float
    ) -> float:
        """
        Calculate signal confidence

        Confidence is based on:
        - Data quality
        - Source agreement
        - Historical accuracy for similar setups
        """
        confidence = 0.5  # Base confidence

        # Score strength contributes
        confidence += adjusted_score * 0.2

        # Catalyst confidence
        if input_data.catalyst_type:
            confidence += input_data.catalyst_confidence * 0.15

        # Pre-Spike confirmation
        if input_data.pre_spike_state in [PreSpikeState.READY, PreSpikeState.LAUNCHING]:
            confidence += 0.10

        # Volume confirmation
        if input_data.volume_ratio > 2.0:
            confidence += 0.05

        # Repeat gainer history
        if input_data.is_repeat_gainer:
            confidence += 0.10

        # V8: Acceleration state confirmation
        if input_data.acceleration_state in ("LAUNCHING", "BREAKOUT"):
            confidence += 0.10
        elif input_data.acceleration_state == "ACCUMULATING":
            confidence += 0.05

        # V8: Volume z-score anomaly confirmation
        if input_data.volume_zscore > 2.0:
            confidence += 0.05

        return min(1.0, max(0.0, confidence))

    def _build_unified_signal(
        self,
        input_data: DetectionInput,
        result: DetectionResult
    ) -> UnifiedSignal:
        """Build unified signal from detection result"""

        signal_id = f"sig_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{input_data.ticker}"

        # Build badges
        badges = []
        if input_data.is_repeat_gainer:
            badges.append("🔄 Repeat Gainer")
        if input_data.pre_spike_state == PreSpikeState.LAUNCHING:
            badges.append("🚀 Launching")
        if input_data.social_acceleration > 3.0:
            badges.append("📈 Social Buzz")
        if input_data.catalyst_type:
            badges.append(f"⚡ {input_data.catalyst_type}")
        # V8: Acceleration state badges
        if input_data.acceleration_state == "ACCUMULATING":
            badges.append("🔍 Accumulating")
        elif input_data.acceleration_state == "LAUNCHING":
            badges.append("🚀 V8 Launch")
        elif input_data.acceleration_state == "BREAKOUT":
            badges.append("💥 Breakout")
        if input_data.volume_zscore > 2.5:
            badges.append(f"📊 Vol z={input_data.volume_zscore:.1f}")

        return UnifiedSignal(
            id=signal_id,
            ticker=input_data.ticker,
            timestamp=result.timestamp,

            # Detection
            signal_type=result.signal_type,
            monster_score=result.adjusted_score,
            catalyst_type=input_data.catalyst_type,
            catalyst_confidence=input_data.catalyst_confidence,

            # States
            pre_spike_state=input_data.pre_spike_state,
            pre_halt_state=PreHaltState.LOW,  # Will be set by Pre-Halt Engine

            # Order and Execution will be set by other layers
            proposed_order=None,
            execution=None,

            # Metadata
            badges=badges,
            market_session=input_data.market_session,
            signal_preserved=True
        )

    def _update_stats(self, signal: UnifiedSignal):
        """Update production statistics"""
        self._signals_produced += 1

        signal_type = signal.signal_type.value
        self._signals_by_type[signal_type] = self._signals_by_type.get(signal_type, 0) + 1

        # Update recent signals cache
        self._recent_signals[signal.ticker] = signal.timestamp

    async def _notify_callbacks(self, signal: UnifiedSignal):
        """Notify registered callbacks"""
        for callback in self._signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _should_signal(self, ticker: str) -> bool:
        """Check if we should produce a signal (cooldown check)"""
        if ticker not in self._recent_signals:
            return True

        last_signal = self._recent_signals[ticker]
        elapsed = (datetime.utcnow() - last_signal).total_seconds()

        return elapsed >= self._signal_cooldown_seconds

    def get_stats(self) -> Dict[str, Any]:
        """Get production statistics"""
        return {
            "total_signals": self._signals_produced,
            "by_type": self._signals_by_type.copy(),
            "recent_tickers": len(self._recent_signals)
        }

    def reset_stats(self):
        """Reset statistics"""
        self._signals_produced = 0
        self._signals_by_type.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

_producer_instance = None


def get_signal_producer() -> SignalProducer:
    """Get singleton producer instance"""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = SignalProducer()
    return _producer_instance


async def quick_detect(
    ticker: str,
    monster_score: float,
    current_price: float,
    catalyst_type: str = None,
    pre_spike_state: PreSpikeState = PreSpikeState.DORMANT
) -> UnifiedSignal:
    """Quick signal detection for single ticker"""
    producer = get_signal_producer()

    input_data = DetectionInput(
        ticker=ticker,
        current_price=current_price,
        monster_score=monster_score,
        catalyst_type=catalyst_type,
        catalyst_confidence=0.8 if catalyst_type else 0.0,
        pre_spike_state=pre_spike_state
    )

    return await producer.detect(input_data)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "SignalProducer",
    "DetectionInput",
    "DetectionResult",
    "get_signal_producer",
    "quick_detect",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    async def test():
        producer = SignalProducer()

        print("=" * 60)
        print("SIGNAL PRODUCER TEST")
        print("=" * 60)

        # Test cases
        test_cases = [
            DetectionInput(
                ticker="BIOX",
                current_price=4.50,
                monster_score=0.85,
                catalyst_type="FDA_APPROVAL",
                catalyst_confidence=0.9,
                pre_spike_state=PreSpikeState.LAUNCHING
            ),
            DetectionInput(
                ticker="ACME",
                current_price=2.30,
                monster_score=0.70,
                catalyst_type="EARNINGS_BEAT",
                catalyst_confidence=0.7,
                pre_spike_state=PreSpikeState.READY,
                is_repeat_gainer=True,
                repeat_gainer_score=0.6
            ),
            DetectionInput(
                ticker="XYZZ",
                current_price=1.20,
                monster_score=0.45,
                pre_spike_state=PreSpikeState.CHARGING,
                volume_ratio=4.5
            ),
        ]

        for input_data in test_cases:
            signal = await producer.detect(input_data)
            print(f"\n{input_data.ticker}:")
            print(f"  Raw score: {input_data.monster_score:.2f}")
            print(f"  Adjusted: {signal.monster_score:.2f}")
            print(f"  Signal: {signal.signal_type.value}")
            print(f"  Badges: {signal.badges}")
            print(f"  Actionable: {signal.is_actionable()}")

        print(f"\nStats: {producer.get_stats()}")

    asyncio.run(test())
