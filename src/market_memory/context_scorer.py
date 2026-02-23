"""
Context Scorer - MRP and EP Scoring System

Provides contextual scoring based on market memory:
- MRP (Missed Recovery Potential): How often similar misses turned into wins
- EP (Edge Probability): Probability of edge based on historical patterns
- CS (Context Score): Combined contextual adjustment

Uses data from:
- MissedTracker: Historical miss outcomes
- PatternLearner: Learned patterns and profiles
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import statistics

from .missed_tracker import MissedTracker, MissReason, MissOutcome, get_missed_tracker
from .pattern_learner import PatternLearner, TickerProfile, get_pattern_learner

logger = logging.getLogger(__name__)


@dataclass
class MRPScore:
    """Missed Recovery Potential score."""
    score: float = 0.0          # 0-100
    confidence: float = 0.0     # 0-100

    # Components
    historical_win_rate: float = 0.0  # % of similar misses that were wins
    recent_win_rate: float = 0.0      # Recent trend
    sample_size: int = 0

    # Interpretation
    interpretation: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "historical_win_rate": self.historical_win_rate,
            "sample_size": self.sample_size,
            "interpretation": self.interpretation,
        }


@dataclass
class EPScore:
    """Edge Probability score."""
    score: float = 0.0          # 0-100
    confidence: float = 0.0     # 0-100

    # Components
    base_probability: float = 0.0     # Historical win rate
    pattern_bonus: float = 0.0        # From pattern matches
    ticker_bonus: float = 0.0         # From ticker profile
    time_bonus: float = 0.0           # From time patterns
    context_bonus: float = 0.0        # From market context

    # Factors
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "base_probability": self.base_probability,
            "bonuses": {
                "pattern": self.pattern_bonus,
                "ticker": self.ticker_bonus,
                "time": self.time_bonus,
                "context": self.context_bonus,
            },
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
        }


@dataclass
class ContextScore:
    """Combined context score with full breakdown."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Composite score
    final_score: float = 0.0        # 0-100
    confidence: float = 0.0          # 0-100

    # Component scores
    mrp: MRPScore = field(default_factory=MRPScore)
    ep: EPScore = field(default_factory=EPScore)

    # Signal adjustment
    signal_adjustment: float = 0.0   # -30 to +30
    size_multiplier: float = 1.0     # 0.25 to 1.5

    # Action recommendation
    action: str = ""  # "EXECUTE", "REDUCE", "AVOID", "WATCH"
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "final_score": self.final_score,
            "confidence": self.confidence,
            "mrp": self.mrp.to_dict(),
            "ep": self.ep.to_dict(),
            "signal_adjustment": self.signal_adjustment,
            "size_multiplier": self.size_multiplier,
            "action": self.action,
            "reasoning": self.reasoning,
        }


@dataclass
class ScorerConfig:
    """Configuration for context scorer."""
    # Weights for final score
    mrp_weight: float = 0.3
    ep_weight: float = 0.7

    # Minimum samples for confidence
    min_samples_mrp: int = 5
    min_samples_ep: int = 10

    # Score thresholds
    high_score_threshold: float = 70.0
    low_score_threshold: float = 40.0

    # Adjustment limits
    max_positive_adjustment: float = 30.0
    max_negative_adjustment: float = -30.0

    # Size multiplier range
    min_size_multiplier: float = 0.25
    max_size_multiplier: float = 1.5

    # Recent window for trend analysis
    recent_days: int = 30


class ContextScorer:
    """
    Calculates contextual scores based on market memory.

    Usage:
        scorer = ContextScorer()

        # Get full context score
        score = scorer.score(
            ticker="AAPL",
            signal_type="BUY_STRONG",
            signal_score=75.0,
            signal_price=150.0
        )

        # Apply to signal
        adjusted_signal_score = base_score + score.signal_adjustment
        position_size = base_size * score.size_multiplier

        # Check action
        if score.action == "AVOID":
            skip_trade()
    """

    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        missed_tracker: Optional[MissedTracker] = None,
        pattern_learner: Optional[PatternLearner] = None
    ):
        self.config = config or ScorerConfig()

        # Dependencies
        self._tracker = missed_tracker or get_missed_tracker()
        self._learner = pattern_learner or get_pattern_learner()

        # Cache
        self._score_cache: Dict[str, Tuple[ContextScore, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def score(
        self,
        ticker: str,
        signal_type: str,
        signal_score: float,
        signal_price: float,
        signal_time: Optional[datetime] = None,
        miss_reason: Optional[MissReason] = None,
        sector: Optional[str] = None,
        market_context: Optional[Dict] = None
    ) -> ContextScore:
        """
        Calculate full context score for a signal.

        Args:
            ticker: Stock ticker
            signal_type: Type of signal (BUY_STRONG, BUY, etc.)
            signal_score: Base signal score
            signal_price: Current price
            signal_time: Time of signal (default: now)
            miss_reason: If evaluating a potential miss
            sector: Sector for sector-based patterns
            market_context: Additional market context

        Returns:
            Complete ContextScore
        """
        ticker = ticker.upper()
        signal_time = signal_time or datetime.now()
        market_context = market_context or {}

        # Check cache
        cache_key = f"{ticker}_{signal_type}_{signal_score}"
        if cache_key in self._score_cache:
            cached, cached_time = self._score_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached

        # Build context score
        ctx = ContextScore(ticker=ticker, timestamp=signal_time)

        # Calculate MRP
        ctx.mrp = self._calculate_mrp(ticker, signal_type, miss_reason)

        # Calculate EP
        ctx.ep = self._calculate_ep(
            ticker, signal_type, signal_score, signal_time, sector, market_context
        )

        # Calculate final score (weighted average)
        ctx.final_score = (
            ctx.mrp.score * self.config.mrp_weight +
            ctx.ep.score * self.config.ep_weight
        )

        ctx.confidence = (
            ctx.mrp.confidence * self.config.mrp_weight +
            ctx.ep.confidence * self.config.ep_weight
        )

        # Calculate signal adjustment
        ctx.signal_adjustment = self._calculate_adjustment(ctx)

        # Calculate size multiplier
        ctx.size_multiplier = self._calculate_size_multiplier(ctx)

        # Determine action
        ctx.action, ctx.reasoning = self._determine_action(ctx, signal_type)

        # Cache result
        self._score_cache[cache_key] = (ctx, datetime.now())

        return ctx

    def _calculate_mrp(
        self,
        ticker: str,
        signal_type: str,
        miss_reason: Optional[MissReason] = None
    ) -> MRPScore:
        """Calculate Missed Recovery Potential score."""
        mrp = MRPScore()

        # Get historical misses for this ticker
        misses = self._tracker.get_misses_for_ticker(ticker)

        # Filter by signal type if relevant
        similar_misses = [
            m for m in misses
            if m.signal_type == signal_type and m.outcome != MissOutcome.UNKNOWN
        ]

        if not similar_misses:
            mrp.interpretation = "No historical data for this ticker/signal combination"
            return mrp

        mrp.sample_size = len(similar_misses)

        # Calculate win rates
        wins = [
            m for m in similar_misses
            if m.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN, MissOutcome.SMALL_WIN]
        ]
        mrp.historical_win_rate = (len(wins) / len(similar_misses)) * 100

        # Recent trend (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=self.config.recent_days)
        recent_misses = [m for m in similar_misses if m.signal_time > recent_cutoff]
        if recent_misses:
            recent_wins = [
                m for m in recent_misses
                if m.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN, MissOutcome.SMALL_WIN]
            ]
            mrp.recent_win_rate = (len(recent_wins) / len(recent_misses)) * 100
        else:
            mrp.recent_win_rate = mrp.historical_win_rate

        # If evaluating a specific miss reason, factor that in
        if miss_reason:
            reason_misses = [m for m in similar_misses if m.miss_reason == miss_reason]
            if reason_misses:
                reason_wins = [
                    m for m in reason_misses
                    if m.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN]
                ]
                reason_win_rate = (len(reason_wins) / len(reason_misses)) * 100
                # Blend with overall rate
                mrp.historical_win_rate = (mrp.historical_win_rate + reason_win_rate) / 2

        # Calculate MRP score
        # High MRP = similar misses often turned into wins = should override blocks
        mrp.score = mrp.historical_win_rate

        # Adjust for trend
        if mrp.recent_win_rate > mrp.historical_win_rate + 10:
            mrp.score += 10  # Recent trend is improving
        elif mrp.recent_win_rate < mrp.historical_win_rate - 10:
            mrp.score -= 10  # Recent trend is declining

        mrp.score = max(0, min(100, mrp.score))

        # Confidence based on sample size
        mrp.confidence = min(100, (mrp.sample_size / self.config.min_samples_mrp) * 50)

        # Interpretation
        if mrp.score >= 70:
            mrp.interpretation = "High MRP: Similar misses often became winners"
            mrp.recommendation = "Consider overriding block if conditions allow"
        elif mrp.score >= 50:
            mrp.interpretation = "Moderate MRP: Mixed historical outcomes"
            mrp.recommendation = "Proceed with normal caution"
        else:
            mrp.interpretation = "Low MRP: Similar misses often resulted in losses"
            mrp.recommendation = "Block decision appears justified"

        return mrp

    def _calculate_ep(
        self,
        ticker: str,
        signal_type: str,
        signal_score: float,
        signal_time: datetime,
        sector: Optional[str],
        market_context: Dict
    ) -> EPScore:
        """Calculate Edge Probability score."""
        ep = EPScore()

        # Get ticker profile from pattern learner
        profile = self._learner.get_ticker_profile(ticker)

        # Base probability from historical win rate
        if profile and profile.is_reliable:
            ep.base_probability = profile.win_rate
            ep.confidence = min(100, profile.total_trades * 5)
        else:
            ep.base_probability = 50.0  # No history, assume neutral
            ep.confidence = 20.0

        # Get score adjustment from pattern learner
        pattern_adj = self._learner.get_score_adjustment(
            ticker, signal_time, signal_score, sector
        )

        # Break down pattern adjustment into components
        if profile:
            if profile.is_favorable:
                ep.ticker_bonus = 5.0
                ep.positive_factors.append(f"Ticker has {profile.win_rate:.0f}% historical win rate")
            elif profile.is_dangerous:
                ep.ticker_bonus = -10.0
                ep.negative_factors.append(f"Ticker has dangerous loss potential ({profile.worst_trade_pct:.0f}%)")

            # Time bonus
            current_period = self._get_time_period(signal_time)
            if current_period and profile.best_time_of_day == current_period:
                ep.time_bonus = 5.0
                ep.positive_factors.append(f"Optimal trading time ({current_period})")
            elif current_period:
                # Check if this is a bad time
                time_stats = self._learner._time_stats.get(current_period, {})
                if time_stats.get("win_rate", 50) < 40:
                    ep.time_bonus = -5.0
                    ep.negative_factors.append(f"Historically weak time period ({current_period})")

        # Pattern bonus from overall pattern matching
        ep.pattern_bonus = pattern_adj - ep.ticker_bonus - ep.time_bonus

        # Context bonus from market conditions
        if market_context:
            if market_context.get("market_trend") == "BULLISH":
                ep.context_bonus = 5.0
                ep.positive_factors.append("Bullish market environment")
            elif market_context.get("market_trend") == "BEARISH":
                ep.context_bonus = -5.0
                ep.negative_factors.append("Bearish market environment")

            if market_context.get("sector_strong") == sector:
                ep.context_bonus += 3.0
                ep.positive_factors.append("Strong sector rotation")

        # Calculate final EP score
        ep.score = (
            ep.base_probability +
            ep.pattern_bonus +
            ep.ticker_bonus +
            ep.time_bonus +
            ep.context_bonus
        )

        ep.score = max(0, min(100, ep.score))

        return ep

    def _get_time_period(self, dt: datetime) -> Optional[str]:
        """Get time period string."""
        hour = dt.hour
        if 4 <= hour < 9:
            return "PREMARKET"
        elif 9 <= hour < 11:
            return "MORNING"
        elif 11 <= hour < 14:
            return "MIDDAY"
        elif 14 <= hour < 16:
            return "AFTERNOON"
        elif 16 <= hour < 20:
            return "AFTERHOURS"
        return None

    def _calculate_adjustment(self, ctx: ContextScore) -> float:
        """Calculate signal score adjustment."""
        # Base adjustment from EP score deviation from 50
        adjustment = (ctx.ep.score - 50) * 0.4  # Scale to reasonable range

        # MRP influence
        if ctx.mrp.confidence > 50:
            mrp_influence = (ctx.mrp.score - 50) * 0.2
            adjustment += mrp_influence

        # Cap adjustment
        adjustment = max(
            self.config.max_negative_adjustment,
            min(self.config.max_positive_adjustment, adjustment)
        )

        return round(adjustment, 1)

    def _calculate_size_multiplier(self, ctx: ContextScore) -> float:
        """Calculate position size multiplier."""
        # Start at 1.0
        multiplier = 1.0

        # Adjust based on final score
        if ctx.final_score >= 80:
            multiplier = 1.25
        elif ctx.final_score >= 70:
            multiplier = 1.1
        elif ctx.final_score <= 30:
            multiplier = 0.5
        elif ctx.final_score <= 40:
            multiplier = 0.75

        # Adjust for confidence
        if ctx.confidence < 30:
            multiplier *= 0.8  # Low confidence = reduce size

        # Adjust for negative factors
        if len(ctx.ep.negative_factors) >= 2:
            multiplier *= 0.75

        # Cap multiplier
        multiplier = max(
            self.config.min_size_multiplier,
            min(self.config.max_size_multiplier, multiplier)
        )

        return round(multiplier, 2)

    def _determine_action(
        self,
        ctx: ContextScore,
        signal_type: str
    ) -> Tuple[str, List[str]]:
        """Determine recommended action."""
        reasoning = []

        # Strong signals with good context
        if ctx.final_score >= self.config.high_score_threshold:
            if ctx.confidence >= 50:
                reasoning.append(f"High context score ({ctx.final_score:.0f}) with good confidence")
                reasoning.extend(ctx.ep.positive_factors[:2])
                return "EXECUTE", reasoning
            else:
                reasoning.append(f"High context score but low confidence ({ctx.confidence:.0f}%)")
                return "WATCH", reasoning

        # Weak signals
        if ctx.final_score <= self.config.low_score_threshold:
            reasoning.append(f"Low context score ({ctx.final_score:.0f})")
            reasoning.extend(ctx.ep.negative_factors[:2])
            if ctx.final_score <= 25:
                return "AVOID", reasoning
            else:
                return "REDUCE", reasoning

        # Middle ground
        if signal_type in ["BUY_STRONG", "STRONG_BUY"]:
            reasoning.append("Strong signal with moderate context")
            return "EXECUTE", reasoning
        else:
            reasoning.append("Moderate signal with moderate context")
            return "WATCH", reasoning

    def quick_score(
        self,
        ticker: str,
        signal_type: str = "BUY"
    ) -> Tuple[float, str]:
        """
        Quick context score without full calculation.

        Returns:
            Tuple of (score, action)
        """
        ctx = self.score(ticker, signal_type, 50.0, 0.0)
        return ctx.final_score, ctx.action

    def get_ticker_summary(self, ticker: str) -> Dict:
        """Get summary of context data for a ticker."""
        profile = self._learner.get_ticker_profile(ticker)
        misses = self._tracker.get_misses_for_ticker(ticker)

        return {
            "ticker": ticker,
            "has_profile": profile is not None,
            "profile": profile.to_dict() if profile else None,
            "total_misses": len(misses),
            "recent_misses": len([
                m for m in misses
                if m.signal_time > datetime.now() - timedelta(days=30)
            ]),
        }

    def clear_cache(self) -> None:
        """Clear score cache."""
        self._score_cache.clear()


# Singleton instance
_scorer: Optional[ContextScorer] = None


def get_context_scorer(config: Optional[ScorerConfig] = None) -> ContextScorer:
    """Get singleton ContextScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ContextScorer(config)
    return _scorer


# ============================================================================
# MARKET MEMORY STABILITY CHECK & SIGNAL ENRICHMENT
# ============================================================================

# Minimum requirements for Market Memory to be considered "stable"
MIN_TOTAL_MISSES = 50           # Minimum missed signals tracked
MIN_TRADES_RECORDED = 30        # Minimum trades recorded
MIN_PATTERNS_LEARNED = 10       # Minimum patterns identified
MIN_TICKER_PROFILES = 20        # Minimum ticker profiles built


def is_market_memory_stable() -> Tuple[bool, Dict]:
    """
    Check if Market Memory has enough data to activate MRP/EP.

    Returns:
        Tuple of (is_stable, stats_dict)
    """
    scorer = get_context_scorer()
    tracker = scorer._tracker
    learner = scorer._learner

    stats = {
        "total_misses": tracker.get_stats().total_recorded if hasattr(tracker, 'get_stats') else 0,
        "trades_recorded": len(learner._trades) if hasattr(learner, '_trades') else 0,
        "patterns_learned": len(learner._patterns) if hasattr(learner, '_patterns') else 0,
        "ticker_profiles": len(learner._profiles) if hasattr(learner, '_profiles') else 0,
        "requirements": {
            "min_misses": MIN_TOTAL_MISSES,
            "min_trades": MIN_TRADES_RECORDED,
            "min_patterns": MIN_PATTERNS_LEARNED,
            "min_profiles": MIN_TICKER_PROFILES,
        }
    }

    is_stable = (
        stats["total_misses"] >= MIN_TOTAL_MISSES and
        stats["trades_recorded"] >= MIN_TRADES_RECORDED and
        stats["patterns_learned"] >= MIN_PATTERNS_LEARNED and
        stats["ticker_profiles"] >= MIN_TICKER_PROFILES
    )

    stats["is_stable"] = is_stable
    stats["missing"] = []

    if stats["total_misses"] < MIN_TOTAL_MISSES:
        stats["missing"].append(f"misses: {stats['total_misses']}/{MIN_TOTAL_MISSES}")
    if stats["trades_recorded"] < MIN_TRADES_RECORDED:
        stats["missing"].append(f"trades: {stats['trades_recorded']}/{MIN_TRADES_RECORDED}")
    if stats["patterns_learned"] < MIN_PATTERNS_LEARNED:
        stats["missing"].append(f"patterns: {stats['patterns_learned']}/{MIN_PATTERNS_LEARNED}")
    if stats["ticker_profiles"] < MIN_TICKER_PROFILES:
        stats["missing"].append(f"profiles: {stats['ticker_profiles']}/{MIN_TICKER_PROFILES}")

    return is_stable, stats


def enrich_signal_with_context(
    signal,  # UnifiedSignal
    force_enable: bool = False
) -> None:
    """
    Enrich a UnifiedSignal with MRP/EP context scores.

    IMPORTANT:
    - Only activates when Market Memory is stable (or force_enable=True)
    - MRP/EP are INFORMATIONAL ONLY, they don't block or modify execution
    - This is a O(1) lookup operation, no latency impact

    Args:
        signal: UnifiedSignal to enrich (modified in place)
        force_enable: Force enable even if memory not stable (for testing)
    """
    # Check if Market Memory is stable
    is_stable, stats = is_market_memory_stable()

    if not is_stable and not force_enable:
        # Market Memory not ready - leave context fields empty
        signal.context_active = False
        return

    # Only enrich actionable signals
    if not signal.is_actionable():
        signal.context_active = False
        return

    try:
        scorer = get_context_scorer()

        # Calculate context score (O(1) lookup from cache or quick calculation)
        ctx = scorer.score(
            ticker=signal.ticker,
            signal_type=signal.signal_type.value,
            signal_score=signal.monster_score * 100,
            signal_price=signal.proposed_order.entry_price if signal.proposed_order else 0.0
        )

        # Populate context fields
        signal.context_mrp = round(ctx.mrp.score, 1)
        signal.context_ep = round(ctx.ep.score, 1)
        signal.context_confidence = round(ctx.confidence, 1)
        signal.context_active = True

        logger.debug(
            f"Context enrichment for {signal.ticker}: "
            f"MRP={signal.context_mrp}, EP={signal.context_ep}, "
            f"confidence={signal.context_confidence}"
        )

    except Exception as e:
        logger.warning(f"Context enrichment failed for {signal.ticker}: {e}")
        signal.context_active = False


def get_memory_status() -> Dict:
    """Get Market Memory status for monitoring."""
    is_stable, stats = is_market_memory_stable()
    return {
        "stable": is_stable,
        "stats": stats,
        "mrp_ep_active": is_stable,
    }


# ============================================================================
# MARKET MEMORY V2 — Segmented by Catalyst Type (A14)
# ============================================================================

import threading as _threading

CATALYST_SEGMENTS = [
    "EARNINGS", "FDA", "SQUEEZE", "MERGER", "PARTNERSHIP",
    "GUIDANCE", "INSIDER", "SOCIAL_MOMENTUM", "TECHNICAL", "OTHER"
]


class ContextScorerV2(ContextScorer):
    """
    Market Memory V2 — MRP/EP segmente par catalyst type.

    V1 (actuel): Un seul MRP/EP par ticker
        mrp = get_mrp("AAPL")  # 0.65

    V2: MRP/EP segmente par catalyst type
        mrp_earnings = get_mrp("AAPL", catalyst="EARNINGS")  # 0.80
        mrp_fda = get_mrp("AAPL", catalyst="FDA")            # 0.45

    Un ticker peut avoir un excellent historique sur les earnings
    mais mauvais sur FDA. Le systeme apprend cette nuance.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._segmented_scores: Dict[str, Dict[str, Dict]] = {}
        self._segment_lock = _threading.Lock()
        logger.info("ContextScorerV2 initialized — segmented by catalyst type")

    def score_segmented(self, ticker: str, signal_type: str, signal_score: float,
                        signal_price: float, catalyst_type: str = "OTHER",
                        **kwargs) -> ContextScore:
        """
        Score avec segmentation par catalyst type.

        Blending: 60% segmented + 40% general quand assez de samples.
        """
        base_score = self.score(ticker, signal_type, signal_score, signal_price, **kwargs)

        segment = self._get_segment(ticker, catalyst_type)

        if segment and segment.get("samples", 0) >= 5:
            seg_mrp = segment.get("mrp", base_score.mrp.score)
            seg_ep = segment.get("ep", base_score.ep.score)

            blend_mrp = seg_mrp * 0.6 + base_score.mrp.score * 0.4
            blend_ep = seg_ep * 0.6 + base_score.ep.score * 0.4

            base_score.mrp.score = blend_mrp
            base_score.ep.score = blend_ep
            base_score.confidence = min(100, base_score.confidence + 10)

            config = self.config if hasattr(self, 'config') else None
            mrp_w = config.mrp_weight if config else 0.3
            ep_w = config.ep_weight if config else 0.7
            base_score.final_score = blend_mrp * mrp_w + blend_ep * ep_w

        return base_score

    def record_outcome(self, ticker: str, catalyst_type: str,
                       was_profitable: bool, move_pct: float,
                       signal_type: str = "BUY") -> None:
        """
        Enregistre le resultat d'un trade pour apprentissage segmente.
        """
        catalyst_type = catalyst_type.upper()
        if catalyst_type not in CATALYST_SEGMENTS:
            catalyst_type = "OTHER"

        with self._segment_lock:
            if ticker not in self._segmented_scores:
                self._segmented_scores[ticker] = {}

            if catalyst_type not in self._segmented_scores[ticker]:
                self._segmented_scores[ticker][catalyst_type] = {
                    "mrp": 50.0, "ep": 50.0, "confidence": 0,
                    "samples": 0, "wins": 0, "total_move": 0.0,
                }

            seg = self._segmented_scores[ticker][catalyst_type]
            seg["samples"] += 1
            seg["total_move"] += move_pct

            if was_profitable:
                seg["wins"] += 1

            win_rate = seg["wins"] / max(1, seg["samples"])
            avg_move = seg["total_move"] / max(1, seg["samples"])

            seg["mrp"] = min(100, win_rate * 70 + min(30, abs(avg_move)))

            confidence_factor = min(1.0, seg["samples"] / 20)
            seg["ep"] = win_rate * 100 * confidence_factor + 50 * (1 - confidence_factor)
            seg["confidence"] = min(100, seg["samples"] * 5)

    def _get_segment(self, ticker: str, catalyst_type: str) -> Optional[Dict]:
        """Recupere les donnees segmentees."""
        with self._segment_lock:
            return self._segmented_scores.get(ticker, {}).get(catalyst_type.upper())

    def get_all_segments(self, ticker: str) -> Dict[str, Dict]:
        """Retourne tous les segments pour un ticker."""
        with self._segment_lock:
            return dict(self._segmented_scores.get(ticker, {}))

    def get_best_catalyst(self, ticker: str) -> Optional[str]:
        """Retourne le catalyst type avec le meilleur historique."""
        segments = self.get_all_segments(ticker)
        if not segments:
            return None
        best = max(segments.items(), key=lambda x: x[1].get("ep", 0))
        if best[1].get("samples", 0) >= 3:
            return best[0]
        return None

    def get_segment_summary(self, ticker: str) -> Dict:
        """Resume des segments pour un ticker."""
        segments = self.get_all_segments(ticker)
        summary = {}
        for cat, data in segments.items():
            if data.get("samples", 0) > 0:
                summary[cat] = {
                    "mrp": round(data["mrp"], 1),
                    "ep": round(data["ep"], 1),
                    "win_rate": round(data["wins"] / max(1, data["samples"]) * 100, 1),
                    "samples": data["samples"],
                    "avg_move": round(data["total_move"] / max(1, data["samples"]), 2),
                }
        return summary

    def get_v2_status(self) -> Dict:
        """Status du scorer V2."""
        with self._segment_lock:
            total_tickers = len(self._segmented_scores)
            total_segments = sum(len(v) for v in self._segmented_scores.values())
            total_samples = sum(
                seg.get("samples", 0)
                for ticker_data in self._segmented_scores.values()
                for seg in ticker_data.values()
            )
        return {
            "version": "V2",
            "tickers_with_segments": total_tickers,
            "total_segments": total_segments,
            "total_samples": total_samples,
            "catalyst_types": CATALYST_SEGMENTS,
        }


_scorer_v2: Optional[ContextScorerV2] = None
_scorer_v2_lock = _threading.Lock()


def get_context_scorer_v2() -> ContextScorerV2:
    """Get singleton ContextScorerV2 instance."""
    global _scorer_v2
    with _scorer_v2_lock:
        if _scorer_v2 is None:
            _scorer_v2 = ContextScorerV2()
    return _scorer_v2
