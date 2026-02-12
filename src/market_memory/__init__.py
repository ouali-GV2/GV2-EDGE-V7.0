"""
Market Memory - Learning System for GV2-EDGE

Components:
- MissedTracker: Track and analyze missed opportunities
- PatternLearner: Learn from historical trading patterns
- ContextScorer: Calculate MRP/EP contextual scores
- MemoryStore: Persistence layer for all memory data

Scoring System:
- MRP (Missed Recovery Potential): How often similar misses became winners
- EP (Edge Probability): Probability of edge based on historical patterns
- CS (Context Score): Combined contextual adjustment for signals

Usage:
    from src.market_memory import (
        get_missed_tracker,
        get_pattern_learner,
        get_context_scorer,
        MissReason
    )

    # Track a miss
    tracker = get_missed_tracker()
    miss_id = tracker.record_miss(
        ticker="AAPL",
        signal_type="BUY_STRONG",
        signal_price=150.0,
        miss_reason=MissReason.DAILY_TRADE_LIMIT
    )

    # Get context score for a signal
    scorer = get_context_scorer()
    score = scorer.score(
        ticker="AAPL",
        signal_type="BUY_STRONG",
        signal_score=75.0,
        signal_price=150.0
    )

    # Apply adjustments
    adjusted_score = base_score + score.signal_adjustment
    position_size = base_size * score.size_multiplier
"""

# Missed Tracker
from .missed_tracker import (
    MissedTracker,
    TrackerConfig,
    MissedSignal,
    MissStats,
    MissReason,
    MissOutcome,
    get_missed_tracker,
)

# Pattern Learner
from .pattern_learner import (
    PatternLearner,
    LearnerConfig,
    Pattern,
    PatternType,
    TickerProfile,
    TradeRecord,
    Outcome,
    get_pattern_learner,
)

# Context Scorer
from .context_scorer import (
    ContextScorer,
    ScorerConfig,
    ContextScore,
    MRPScore,
    EPScore,
    get_context_scorer,
    # Activation control
    is_market_memory_stable,
    enrich_signal_with_context,
    get_memory_status,
    # Thresholds
    MIN_TOTAL_MISSES,
    MIN_TRADES_RECORDED,
    MIN_PATTERNS_LEARNED,
    MIN_TICKER_PROFILES,
)

# Memory Store
from .memory_store import (
    MemoryStore,
    StoreConfig,
    StorageBackend,
    get_memory_store,
)

__all__ = [
    # Missed Tracker
    "MissedTracker",
    "TrackerConfig",
    "MissedSignal",
    "MissStats",
    "MissReason",
    "MissOutcome",
    "get_missed_tracker",
    # Pattern Learner
    "PatternLearner",
    "LearnerConfig",
    "Pattern",
    "PatternType",
    "TickerProfile",
    "TradeRecord",
    "Outcome",
    "get_pattern_learner",
    # Context Scorer
    "ContextScorer",
    "ScorerConfig",
    "ContextScore",
    "MRPScore",
    "EPScore",
    "get_context_scorer",
    # Activation control
    "is_market_memory_stable",
    "enrich_signal_with_context",
    "get_memory_status",
    "MIN_TOTAL_MISSES",
    "MIN_TRADES_RECORDED",
    "MIN_PATTERNS_LEARNED",
    "MIN_TICKER_PROFILES",
    # Memory Store
    "MemoryStore",
    "StoreConfig",
    "StorageBackend",
    "get_memory_store",
]
