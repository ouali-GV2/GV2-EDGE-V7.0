"""
Missed Tracker - Track and Analyze Missed Opportunities

Captures signals that were:
- Blocked by execution limits
- Filtered by risk guard
- Passed on due to capital constraints
- Not acted on for any reason

Analyzes what would have happened to learn from misses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MissReason(Enum):
    """Reasons why a signal was missed."""
    # Execution limits
    DAILY_TRADE_LIMIT = "DAILY_TRADE_LIMIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    CAPITAL_LIMIT = "CAPITAL_LIMIT"

    # Risk blocks
    RISK_DILUTION = "RISK_DILUTION"
    RISK_COMPLIANCE = "RISK_COMPLIANCE"
    RISK_HALT = "RISK_HALT"
    RISK_VOLATILITY = "RISK_VOLATILITY"

    # Market conditions
    MARKET_HOURS = "MARKET_HOURS"
    HALTED = "HALTED"
    NO_LIQUIDITY = "NO_LIQUIDITY"

    # System
    SYSTEM_ERROR = "SYSTEM_ERROR"
    TIMEOUT = "TIMEOUT"

    # Manual
    MANUAL_SKIP = "MANUAL_SKIP"
    WATCHLIST_ONLY = "WATCHLIST_ONLY"

    # Other
    UNKNOWN = "UNKNOWN"


class MissOutcome(Enum):
    """Outcome of a missed opportunity."""
    BIG_WIN = "BIG_WIN"         # Would have been >20% gain
    WIN = "WIN"                 # Would have been 5-20% gain
    SMALL_WIN = "SMALL_WIN"     # Would have been 0-5% gain
    BREAKEVEN = "BREAKEVEN"     # Would have been -2% to +2%
    SMALL_LOSS = "SMALL_LOSS"   # Would have been -5% to -2%
    LOSS = "LOSS"               # Would have been -5% to -20%
    BIG_LOSS = "BIG_LOSS"       # Would have been >-20% loss
    UNKNOWN = "UNKNOWN"         # Not enough data


@dataclass
class MissedSignal:
    """A signal that was not acted upon."""
    id: str
    ticker: str
    signal_time: datetime

    # Signal details
    signal_type: str  # BUY_STRONG, BUY, etc.
    signal_score: float
    signal_price: float

    # Why it was missed
    miss_reason: MissReason
    miss_details: str = ""

    # What was proposed
    proposed_shares: int = 0
    proposed_value: float = 0.0
    proposed_stop: Optional[float] = None
    proposed_target: Optional[float] = None

    # Outcome tracking
    outcome: MissOutcome = MissOutcome.UNKNOWN
    outcome_price: Optional[float] = None
    outcome_time: Optional[datetime] = None
    outcome_pnl_pct: Optional[float] = None
    outcome_pnl_value: Optional[float] = None

    # Price tracking
    high_after: Optional[float] = None
    low_after: Optional[float] = None
    close_1d: Optional[float] = None
    close_5d: Optional[float] = None

    # Context
    market_context: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)

    # Analysis
    was_correct_miss: Optional[bool] = None  # True = dodged a bullet
    lessons: List[str] = field(default_factory=list)

    def calculate_outcome(self) -> None:
        """Calculate outcome based on price movement."""
        if not self.outcome_price or not self.signal_price:
            return

        pct_change = ((self.outcome_price - self.signal_price) / self.signal_price) * 100
        self.outcome_pnl_pct = pct_change

        if self.proposed_value:
            self.outcome_pnl_value = self.proposed_value * (pct_change / 100)

        # Determine outcome category
        if pct_change > 20:
            self.outcome = MissOutcome.BIG_WIN
        elif pct_change > 5:
            self.outcome = MissOutcome.WIN
        elif pct_change > 0:
            self.outcome = MissOutcome.SMALL_WIN
        elif pct_change > -2:
            self.outcome = MissOutcome.BREAKEVEN
        elif pct_change > -5:
            self.outcome = MissOutcome.SMALL_LOSS
        elif pct_change > -20:
            self.outcome = MissOutcome.LOSS
        else:
            self.outcome = MissOutcome.BIG_LOSS

        # Was the miss correct?
        if self.outcome in [MissOutcome.LOSS, MissOutcome.BIG_LOSS]:
            self.was_correct_miss = True
        elif self.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN]:
            self.was_correct_miss = False
        else:
            self.was_correct_miss = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "signal_time": self.signal_time.isoformat(),
            "signal_type": self.signal_type,
            "signal_score": self.signal_score,
            "signal_price": self.signal_price,
            "miss_reason": self.miss_reason.value,
            "outcome": self.outcome.value,
            "outcome_pnl_pct": self.outcome_pnl_pct,
            "was_correct_miss": self.was_correct_miss,
        }


@dataclass
class MissStats:
    """Statistics about missed opportunities."""
    total_misses: int = 0
    analyzed_misses: int = 0

    # By outcome
    big_wins_missed: int = 0
    wins_missed: int = 0
    losses_avoided: int = 0
    big_losses_avoided: int = 0

    # Value
    total_potential_gain: float = 0.0
    total_potential_loss: float = 0.0
    net_missed_pnl: float = 0.0

    # By reason
    by_reason: Dict[str, int] = field(default_factory=dict)

    # Accuracy
    correct_miss_rate: float = 0.0  # % of times miss was right decision

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_misses": self.total_misses,
            "big_wins_missed": self.big_wins_missed,
            "wins_missed": self.wins_missed,
            "losses_avoided": self.losses_avoided,
            "big_losses_avoided": self.big_losses_avoided,
            "net_missed_pnl": self.net_missed_pnl,
            "correct_miss_rate": self.correct_miss_rate,
        }


@dataclass
class TrackerConfig:
    """Configuration for missed tracker."""
    # Tracking
    track_duration_days: int = 5  # How long to track after miss
    outcome_check_hours: List[int] = field(default_factory=lambda: [1, 4, 24, 120])

    # Thresholds
    significant_move_pct: float = 5.0
    big_move_pct: float = 20.0

    # Storage
    max_history_days: int = 90
    max_misses_stored: int = 10000


class MissedTracker:
    """
    Tracks and analyzes missed trading opportunities.

    Usage:
        tracker = MissedTracker()

        # Record a miss
        tracker.record_miss(
            ticker="AAPL",
            signal_type="BUY_STRONG",
            signal_price=150.0,
            miss_reason=MissReason.DAILY_TRADE_LIMIT,
            proposed_value=5000.0
        )

        # Update with outcome
        tracker.update_outcome("miss_123", outcome_price=165.0)

        # Get analysis
        stats = tracker.get_stats()
        lessons = tracker.get_lessons()
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

        # Storage
        self._misses: Dict[str, MissedSignal] = {}
        self._by_ticker: Dict[str, List[str]] = {}
        self._by_reason: Dict[MissReason, List[str]] = {r: [] for r in MissReason}

        # Pending outcome checks
        self._pending_checks: List[Tuple[str, datetime]] = []

        # Statistics
        self._stats = MissStats()

        # Counter for IDs
        self._counter = 0

    def record_miss(
        self,
        ticker: str,
        signal_type: str,
        signal_price: float,
        miss_reason: MissReason,
        signal_score: float = 0.0,
        proposed_shares: int = 0,
        proposed_value: float = 0.0,
        proposed_stop: Optional[float] = None,
        proposed_target: Optional[float] = None,
        miss_details: str = "",
        risk_flags: Optional[List[str]] = None,
        market_context: Optional[Dict] = None
    ) -> str:
        """
        Record a missed opportunity.

        Returns:
            Miss ID
        """
        self._counter += 1
        miss_id = f"miss_{datetime.now().strftime('%Y%m%d')}_{self._counter:04d}"

        miss = MissedSignal(
            id=miss_id,
            ticker=ticker.upper(),
            signal_time=datetime.now(),
            signal_type=signal_type,
            signal_score=signal_score,
            signal_price=signal_price,
            miss_reason=miss_reason,
            miss_details=miss_details,
            proposed_shares=proposed_shares,
            proposed_value=proposed_value,
            proposed_stop=proposed_stop,
            proposed_target=proposed_target,
            risk_flags=risk_flags or [],
            market_context=market_context or {}
        )

        # Store
        self._misses[miss_id] = miss

        # Index by ticker
        if ticker not in self._by_ticker:
            self._by_ticker[ticker] = []
        self._by_ticker[ticker].append(miss_id)

        # Index by reason
        self._by_reason[miss_reason].append(miss_id)

        # Schedule outcome checks
        for hours in self.config.outcome_check_hours:
            check_time = datetime.now() + timedelta(hours=hours)
            self._pending_checks.append((miss_id, check_time))

        # Update stats
        self._stats.total_misses += 1
        reason_key = miss_reason.value
        self._stats.by_reason[reason_key] = self._stats.by_reason.get(reason_key, 0) + 1

        logger.info(
            f"Recorded miss: {miss_id} - {ticker} {signal_type} "
            f"@ ${signal_price:.2f} ({miss_reason.value})"
        )

        return miss_id

    def update_outcome(
        self,
        miss_id: str,
        outcome_price: float,
        high_after: Optional[float] = None,
        low_after: Optional[float] = None
    ) -> Optional[MissedSignal]:
        """Update a miss with outcome data."""
        miss = self._misses.get(miss_id)
        if not miss:
            return None

        miss.outcome_price = outcome_price
        miss.outcome_time = datetime.now()
        miss.high_after = high_after
        miss.low_after = low_after

        # Calculate outcome
        miss.calculate_outcome()

        # Update stats
        self._stats.analyzed_misses += 1

        if miss.outcome == MissOutcome.BIG_WIN:
            self._stats.big_wins_missed += 1
            if miss.outcome_pnl_value:
                self._stats.total_potential_gain += miss.outcome_pnl_value
        elif miss.outcome == MissOutcome.WIN:
            self._stats.wins_missed += 1
            if miss.outcome_pnl_value:
                self._stats.total_potential_gain += miss.outcome_pnl_value
        elif miss.outcome == MissOutcome.BIG_LOSS:
            self._stats.big_losses_avoided += 1
            if miss.outcome_pnl_value:
                self._stats.total_potential_loss += abs(miss.outcome_pnl_value)
        elif miss.outcome == MissOutcome.LOSS:
            self._stats.losses_avoided += 1
            if miss.outcome_pnl_value:
                self._stats.total_potential_loss += abs(miss.outcome_pnl_value)

        if miss.outcome_pnl_value:
            self._stats.net_missed_pnl += miss.outcome_pnl_value

        # Update correct miss rate
        analyzed = [m for m in self._misses.values() if m.was_correct_miss is not None]
        if analyzed:
            correct = sum(1 for m in analyzed if m.was_correct_miss)
            self._stats.correct_miss_rate = (correct / len(analyzed)) * 100

        # Generate lessons
        self._generate_lessons(miss)

        logger.info(
            f"Updated miss outcome: {miss_id} - {miss.outcome.value} "
            f"({miss.outcome_pnl_pct:.1f}%)"
        )

        return miss

    def _generate_lessons(self, miss: MissedSignal) -> None:
        """Generate lessons learned from a miss."""
        lessons = []

        # Big win missed due to limits
        if miss.outcome in [MissOutcome.BIG_WIN, MissOutcome.WIN]:
            if miss.miss_reason == MissReason.DAILY_TRADE_LIMIT:
                lessons.append(
                    f"LESSON: Missed {miss.outcome_pnl_pct:.1f}% gain on {miss.ticker} "
                    f"due to trade limit. Consider prioritizing higher-score signals."
                )
            elif miss.miss_reason in [MissReason.RISK_DILUTION, MissReason.RISK_COMPLIANCE]:
                lessons.append(
                    f"REVIEW: Risk guard blocked {miss.ticker} which gained {miss.outcome_pnl_pct:.1f}%. "
                    f"Risk flags: {', '.join(miss.risk_flags)}"
                )

        # Correctly avoided loss
        elif miss.outcome in [MissOutcome.BIG_LOSS, MissOutcome.LOSS]:
            if miss.miss_reason in [MissReason.RISK_DILUTION, MissReason.RISK_COMPLIANCE]:
                lessons.append(
                    f"VALIDATED: Risk guard correctly blocked {miss.ticker} "
                    f"(would have lost {abs(miss.outcome_pnl_pct):.1f}%)"
                )

        miss.lessons = lessons

    def get_pending_checks(self) -> List[Tuple[str, datetime]]:
        """Get outcome checks due now."""
        now = datetime.now()
        due = [(mid, t) for mid, t in self._pending_checks if t <= now]
        self._pending_checks = [(mid, t) for mid, t in self._pending_checks if t > now]
        return due

    def get_miss(self, miss_id: str) -> Optional[MissedSignal]:
        """Get a specific miss."""
        return self._misses.get(miss_id)

    def get_misses_for_ticker(self, ticker: str) -> List[MissedSignal]:
        """Get all misses for a ticker."""
        miss_ids = self._by_ticker.get(ticker.upper(), [])
        return [self._misses[mid] for mid in miss_ids if mid in self._misses]

    def get_misses_by_reason(self, reason: MissReason) -> List[MissedSignal]:
        """Get all misses for a reason."""
        miss_ids = self._by_reason.get(reason, [])
        return [self._misses[mid] for mid in miss_ids if mid in self._misses]

    def get_recent_misses(
        self,
        hours: int = 24,
        outcome_filter: Optional[List[MissOutcome]] = None
    ) -> List[MissedSignal]:
        """Get recent misses."""
        cutoff = datetime.now() - timedelta(hours=hours)
        misses = [
            m for m in self._misses.values()
            if m.signal_time > cutoff
        ]

        if outcome_filter:
            misses = [m for m in misses if m.outcome in outcome_filter]

        return sorted(misses, key=lambda x: x.signal_time, reverse=True)

    def get_big_misses(self, min_pct: float = 10.0) -> List[MissedSignal]:
        """Get misses with significant outcomes."""
        return [
            m for m in self._misses.values()
            if m.outcome_pnl_pct and abs(m.outcome_pnl_pct) >= min_pct
        ]

    def get_stats(self) -> MissStats:
        """Get miss statistics."""
        return self._stats

    def get_lessons(self, limit: int = 10) -> List[str]:
        """Get recent lessons learned."""
        all_lessons = []
        for miss in sorted(self._misses.values(), key=lambda x: x.signal_time, reverse=True):
            all_lessons.extend(miss.lessons)
            if len(all_lessons) >= limit:
                break
        return all_lessons[:limit]

    def get_reason_analysis(self) -> Dict[str, Dict]:
        """Analyze outcomes by miss reason."""
        analysis = {}

        for reason in MissReason:
            misses = self.get_misses_by_reason(reason)
            if not misses:
                continue

            analyzed = [m for m in misses if m.outcome != MissOutcome.UNKNOWN]
            if not analyzed:
                continue

            wins = sum(1 for m in analyzed if m.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN])
            losses = sum(1 for m in analyzed if m.outcome in [MissOutcome.LOSS, MissOutcome.BIG_LOSS])

            avg_pnl = sum(m.outcome_pnl_pct or 0 for m in analyzed) / len(analyzed)

            analysis[reason.value] = {
                "total": len(misses),
                "analyzed": len(analyzed),
                "wins_missed": wins,
                "losses_avoided": losses,
                "avg_outcome_pct": avg_pnl,
                "correct_miss_rate": (losses / len(analyzed) * 100) if analyzed else 0,
            }

        return analysis

    def should_override_block(
        self,
        ticker: str,
        miss_reason: MissReason
    ) -> Tuple[bool, str]:
        """
        Suggest whether to override a block based on history.

        Returns:
            Tuple of (should_override, reason)
        """
        # Get historical misses for this ticker with same reason
        ticker_misses = self.get_misses_for_ticker(ticker)
        same_reason = [m for m in ticker_misses if m.miss_reason == miss_reason]

        if not same_reason:
            return False, "No historical data"

        # Analyze outcomes
        analyzed = [m for m in same_reason if m.outcome != MissOutcome.UNKNOWN]
        if len(analyzed) < 3:
            return False, "Insufficient history"

        wins = sum(1 for m in analyzed if m.outcome in [MissOutcome.WIN, MissOutcome.BIG_WIN])
        win_rate = wins / len(analyzed)

        if win_rate > 0.7:
            return True, f"Historical win rate {win_rate:.0%} suggests override"
        elif win_rate < 0.3:
            return False, f"Historical win rate {win_rate:.0%} supports block"
        else:
            return False, f"Historical win rate {win_rate:.0%} is inconclusive"

    def cleanup_old(self, days: int = 90) -> int:
        """Remove misses older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = [
            mid for mid, miss in self._misses.items()
            if miss.signal_time < cutoff
        ]

        for mid in to_remove:
            miss = self._misses.pop(mid)
            if miss.ticker in self._by_ticker:
                if mid in self._by_ticker[miss.ticker]:
                    self._by_ticker[miss.ticker].remove(mid)
            if mid in self._by_reason[miss.miss_reason]:
                self._by_reason[miss.miss_reason].remove(mid)

        return len(to_remove)


# Singleton instance
_tracker: Optional[MissedTracker] = None


def get_missed_tracker(config: Optional[TrackerConfig] = None) -> MissedTracker:
    """Get singleton MissedTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MissedTracker(config)
    return _tracker
