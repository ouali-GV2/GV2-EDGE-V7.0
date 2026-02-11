"""
Pattern Learner - Learn from Historical Trading Patterns

Analyzes historical trades and signals to identify:
- Winning patterns (what works)
- Losing patterns (what to avoid)
- Ticker-specific behaviors
- Time-based patterns
- Sector correlations
- Signal quality patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import logging
import statistics

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns to learn."""
    TICKER_BEHAVIOR = "TICKER_BEHAVIOR"     # How specific tickers behave
    TIME_OF_DAY = "TIME_OF_DAY"             # Time-based patterns
    DAY_OF_WEEK = "DAY_OF_WEEK"             # Day-based patterns
    SIGNAL_QUALITY = "SIGNAL_QUALITY"       # Signal score vs outcome
    SECTOR_MOMENTUM = "SECTOR_MOMENTUM"     # Sector correlations
    VOLATILITY = "VOLATILITY"              # Volatility patterns
    CATALYST = "CATALYST"                  # Catalyst-driven patterns
    STREAK = "STREAK"                      # Win/loss streaks


class Outcome(Enum):
    """Trade outcomes."""
    BIG_WIN = "BIG_WIN"       # >15%
    WIN = "WIN"               # 5-15%
    SMALL_WIN = "SMALL_WIN"   # 0-5%
    BREAKEVEN = "BREAKEVEN"   # -2% to +2%
    SMALL_LOSS = "SMALL_LOSS" # -5% to -2%
    LOSS = "LOSS"             # -15% to -5%
    BIG_LOSS = "BIG_LOSS"     # <-15%


@dataclass
class TradeRecord:
    """Record of a historical trade."""
    id: str
    ticker: str
    entry_time: datetime
    exit_time: Optional[datetime] = None

    # Prices
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None

    # Position
    shares: int = 0
    position_value: float = 0.0

    # Outcome
    pnl_dollars: float = 0.0
    pnl_percent: float = 0.0
    outcome: Outcome = Outcome.BREAKEVEN

    # Signal info
    signal_type: str = ""
    signal_score: float = 0.0

    # Context
    sector: str = ""
    market_cap: Optional[float] = None
    volume_ratio: float = 1.0  # vs average
    volatility: float = 0.0

    # Tags
    catalysts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def calculate_outcome(self) -> None:
        """Calculate outcome category."""
        if self.pnl_percent > 15:
            self.outcome = Outcome.BIG_WIN
        elif self.pnl_percent > 5:
            self.outcome = Outcome.WIN
        elif self.pnl_percent > 0:
            self.outcome = Outcome.SMALL_WIN
        elif self.pnl_percent > -2:
            self.outcome = Outcome.BREAKEVEN
        elif self.pnl_percent > -5:
            self.outcome = Outcome.SMALL_LOSS
        elif self.pnl_percent > -15:
            self.outcome = Outcome.LOSS
        else:
            self.outcome = Outcome.BIG_LOSS


@dataclass
class Pattern:
    """A learned pattern."""
    id: str
    pattern_type: PatternType
    name: str
    description: str

    # Statistics
    sample_size: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Confidence
    confidence: float = 0.0  # 0-100
    min_samples: int = 10

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Examples
    examples: List[str] = field(default_factory=list)  # Trade IDs

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_reliable(self) -> bool:
        """Check if pattern has enough data to be reliable."""
        return self.sample_size >= self.min_samples and self.confidence >= 60

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "sample_size": self.sample_size,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "confidence": self.confidence,
            "conditions": self.conditions,
        }


@dataclass
class TickerProfile:
    """Learned profile for a specific ticker."""
    ticker: str
    total_trades: int = 0

    # Performance
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    # Patterns
    best_time_of_day: Optional[str] = None  # "MORNING", "MIDDAY", "AFTERNOON"
    best_day_of_week: Optional[str] = None
    avg_hold_time_hours: float = 0.0

    # Characteristics
    avg_volatility: float = 0.0
    typical_spread_pct: float = 0.0
    halts_experienced: int = 0

    # Flags
    is_reliable: bool = False  # Enough history
    is_favorable: bool = False  # Good win rate
    is_dangerous: bool = False  # High loss potential

    # Recent performance
    recent_win_rate: float = 0.0  # Last 10 trades
    trend: str = ""  # "IMPROVING", "DECLINING", "STABLE"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "is_favorable": self.is_favorable,
            "is_dangerous": self.is_dangerous,
            "best_time": self.best_time_of_day,
        }


@dataclass
class LearnerConfig:
    """Configuration for pattern learner."""
    # Minimums for reliability
    min_trades_for_pattern: int = 10
    min_trades_for_ticker_profile: int = 5

    # Time periods
    time_periods = {
        "PREMARKET": (time(4, 0), time(9, 30)),
        "MORNING": (time(9, 30), time(11, 30)),
        "MIDDAY": (time(11, 30), time(14, 0)),
        "AFTERNOON": (time(14, 0), time(16, 0)),
        "AFTERHOURS": (time(16, 0), time(20, 0)),
    }

    # Thresholds
    favorable_win_rate: float = 55.0
    dangerous_loss_threshold: float = -10.0

    # Pattern detection
    detect_time_patterns: bool = True
    detect_ticker_patterns: bool = True
    detect_sector_patterns: bool = True
    detect_signal_patterns: bool = True


class PatternLearner:
    """
    Learns patterns from historical trading data.

    Usage:
        learner = PatternLearner()

        # Add historical trades
        learner.add_trade(trade_record)

        # Learn patterns
        learner.learn_all()

        # Get insights
        profile = learner.get_ticker_profile("AAPL")
        patterns = learner.get_patterns(PatternType.TIME_OF_DAY)

        # Get recommendations
        score_adj = learner.get_score_adjustment("AAPL", signal_time)
    """

    def __init__(self, config: Optional[LearnerConfig] = None):
        self.config = config or LearnerConfig()

        # Trade history
        self._trades: Dict[str, TradeRecord] = {}
        self._by_ticker: Dict[str, List[str]] = defaultdict(list)
        self._by_sector: Dict[str, List[str]] = defaultdict(list)

        # Learned data
        self._patterns: Dict[str, Pattern] = {}
        self._ticker_profiles: Dict[str, TickerProfile] = {}

        # Aggregated stats
        self._time_stats: Dict[str, Dict] = {}
        self._day_stats: Dict[str, Dict] = {}
        self._signal_stats: Dict[str, Dict] = {}

        # Counter
        self._pattern_counter = 0

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade to history."""
        self._trades[trade.id] = trade
        self._by_ticker[trade.ticker].append(trade.id)
        if trade.sector:
            self._by_sector[trade.sector].append(trade.id)

    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def learn_all(self) -> Dict[str, int]:
        """
        Run all pattern learning.

        Returns:
            Dict with counts of patterns learned by type
        """
        results = {}

        if self.config.detect_ticker_patterns:
            results["ticker_profiles"] = self._learn_ticker_patterns()

        if self.config.detect_time_patterns:
            results["time_patterns"] = self._learn_time_patterns()

        if self.config.detect_signal_patterns:
            results["signal_patterns"] = self._learn_signal_patterns()

        if self.config.detect_sector_patterns:
            results["sector_patterns"] = self._learn_sector_patterns()

        logger.info(f"Pattern learning complete: {results}")

        return results

    def _learn_ticker_patterns(self) -> int:
        """Learn ticker-specific patterns."""
        count = 0

        for ticker, trade_ids in self._by_ticker.items():
            if len(trade_ids) < self.config.min_trades_for_ticker_profile:
                continue

            trades = [self._trades[tid] for tid in trade_ids]
            profile = self._build_ticker_profile(ticker, trades)
            self._ticker_profiles[ticker] = profile
            count += 1

        return count

    def _build_ticker_profile(
        self,
        ticker: str,
        trades: List[TradeRecord]
    ) -> TickerProfile:
        """Build profile for a ticker."""
        profile = TickerProfile(ticker=ticker, total_trades=len(trades))

        # Basic stats
        pnls = [t.pnl_percent for t in trades]
        wins = [t for t in trades if t.pnl_percent > 0]

        profile.win_rate = (len(wins) / len(trades)) * 100
        profile.avg_pnl_pct = statistics.mean(pnls) if pnls else 0
        profile.best_trade_pct = max(pnls) if pnls else 0
        profile.worst_trade_pct = min(pnls) if pnls else 0

        # Volatility
        if any(t.volatility for t in trades):
            vols = [t.volatility for t in trades if t.volatility]
            profile.avg_volatility = statistics.mean(vols)

        # Best time of day
        time_performance = defaultdict(list)
        for trade in trades:
            period = self._get_time_period(trade.entry_time.time())
            if period:
                time_performance[period].append(trade.pnl_percent)

        if time_performance:
            best_time = max(
                time_performance.keys(),
                key=lambda k: statistics.mean(time_performance[k])
            )
            profile.best_time_of_day = best_time

        # Best day of week
        day_performance = defaultdict(list)
        for trade in trades:
            day = trade.entry_time.strftime("%A")
            day_performance[day].append(trade.pnl_percent)

        if day_performance:
            best_day = max(
                day_performance.keys(),
                key=lambda k: statistics.mean(day_performance[k])
            )
            profile.best_day_of_week = best_day

        # Hold time
        hold_times = []
        for trade in trades:
            if trade.exit_time:
                hold = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                hold_times.append(hold)
        if hold_times:
            profile.avg_hold_time_hours = statistics.mean(hold_times)

        # Flags
        profile.is_reliable = len(trades) >= self.config.min_trades_for_ticker_profile
        profile.is_favorable = profile.win_rate >= self.config.favorable_win_rate
        profile.is_dangerous = profile.worst_trade_pct <= self.config.dangerous_loss_threshold

        # Recent trend
        recent = trades[-10:] if len(trades) >= 10 else trades
        recent_wins = [t for t in recent if t.pnl_percent > 0]
        profile.recent_win_rate = (len(recent_wins) / len(recent)) * 100

        if profile.recent_win_rate > profile.win_rate + 10:
            profile.trend = "IMPROVING"
        elif profile.recent_win_rate < profile.win_rate - 10:
            profile.trend = "DECLINING"
        else:
            profile.trend = "STABLE"

        return profile

    def _learn_time_patterns(self) -> int:
        """Learn time-of-day patterns."""
        count = 0

        # Aggregate by time period
        for period, (start, end) in self.config.time_periods.items():
            trades_in_period = [
                t for t in self._trades.values()
                if self._is_in_time_range(t.entry_time.time(), start, end)
            ]

            if len(trades_in_period) < self.config.min_trades_for_pattern:
                continue

            pnls = [t.pnl_percent for t in trades_in_period]
            wins = [t for t in trades_in_period if t.pnl_percent > 0]

            self._time_stats[period] = {
                "sample_size": len(trades_in_period),
                "win_rate": (len(wins) / len(trades_in_period)) * 100,
                "avg_pnl": statistics.mean(pnls),
                "std_pnl": statistics.stdev(pnls) if len(pnls) > 1 else 0,
            }

            # Create pattern if significant
            stats = self._time_stats[period]
            if stats["win_rate"] > 60 or stats["win_rate"] < 40:
                self._pattern_counter += 1
                pattern = Pattern(
                    id=f"time_{self._pattern_counter}",
                    pattern_type=PatternType.TIME_OF_DAY,
                    name=f"{period}_PATTERN",
                    description=f"Trading during {period} has {stats['win_rate']:.1f}% win rate",
                    sample_size=stats["sample_size"],
                    win_rate=stats["win_rate"],
                    avg_pnl_pct=stats["avg_pnl"],
                    confidence=min(100, stats["sample_size"] / 2),
                    conditions={"time_period": period}
                )
                self._patterns[pattern.id] = pattern
                count += 1

        return count

    def _learn_signal_patterns(self) -> int:
        """Learn signal score patterns."""
        count = 0

        # Bin by signal score ranges
        score_bins = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

        for low, high in score_bins:
            trades = [
                t for t in self._trades.values()
                if low <= t.signal_score < high
            ]

            if len(trades) < self.config.min_trades_for_pattern:
                continue

            pnls = [t.pnl_percent for t in trades]
            wins = [t for t in trades if t.pnl_percent > 0]

            bin_key = f"{low}-{high}"
            self._signal_stats[bin_key] = {
                "sample_size": len(trades),
                "win_rate": (len(wins) / len(trades)) * 100,
                "avg_pnl": statistics.mean(pnls),
            }

            # Create pattern
            stats = self._signal_stats[bin_key]
            self._pattern_counter += 1
            pattern = Pattern(
                id=f"signal_{self._pattern_counter}",
                pattern_type=PatternType.SIGNAL_QUALITY,
                name=f"SIGNAL_SCORE_{bin_key}",
                description=f"Signals with score {bin_key} have {stats['win_rate']:.1f}% win rate",
                sample_size=stats["sample_size"],
                win_rate=stats["win_rate"],
                avg_pnl_pct=stats["avg_pnl"],
                confidence=min(100, stats["sample_size"] / 2),
                conditions={"score_range": (low, high)}
            )
            self._patterns[pattern.id] = pattern
            count += 1

        return count

    def _learn_sector_patterns(self) -> int:
        """Learn sector-specific patterns."""
        count = 0

        for sector, trade_ids in self._by_sector.items():
            if len(trade_ids) < self.config.min_trades_for_pattern:
                continue

            trades = [self._trades[tid] for tid in trade_ids]
            pnls = [t.pnl_percent for t in trades]
            wins = [t for t in trades if t.pnl_percent > 0]

            win_rate = (len(wins) / len(trades)) * 100
            avg_pnl = statistics.mean(pnls)

            # Create pattern if noteworthy
            if win_rate > 60 or win_rate < 40 or abs(avg_pnl) > 5:
                self._pattern_counter += 1
                pattern = Pattern(
                    id=f"sector_{self._pattern_counter}",
                    pattern_type=PatternType.SECTOR_MOMENTUM,
                    name=f"SECTOR_{sector}",
                    description=f"Trading {sector} sector has {win_rate:.1f}% win rate",
                    sample_size=len(trades),
                    win_rate=win_rate,
                    avg_pnl_pct=avg_pnl,
                    confidence=min(100, len(trades) / 2),
                    conditions={"sector": sector}
                )
                self._patterns[pattern.id] = pattern
                count += 1

        return count

    def _get_time_period(self, t: time) -> Optional[str]:
        """Get time period for a time."""
        for period, (start, end) in self.config.time_periods.items():
            if self._is_in_time_range(t, start, end):
                return period
        return None

    def _is_in_time_range(self, t: time, start: time, end: time) -> bool:
        """Check if time is in range."""
        return start <= t < end

    # Query methods

    def get_ticker_profile(self, ticker: str) -> Optional[TickerProfile]:
        """Get profile for a ticker."""
        return self._ticker_profiles.get(ticker.upper())

    def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        reliable_only: bool = True
    ) -> List[Pattern]:
        """Get learned patterns."""
        patterns = list(self._patterns.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if reliable_only:
            patterns = [p for p in patterns if p.is_reliable()]

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def get_score_adjustment(
        self,
        ticker: str,
        signal_time: datetime,
        signal_score: float,
        sector: Optional[str] = None
    ) -> float:
        """
        Get score adjustment based on learned patterns.

        Returns:
            Adjustment value (-20 to +20)
        """
        adjustment = 0.0

        # Ticker-based adjustment
        profile = self._ticker_profiles.get(ticker.upper())
        if profile and profile.is_reliable:
            if profile.is_favorable:
                adjustment += 5.0
            elif profile.is_dangerous:
                adjustment -= 10.0

            # Time-of-day bonus
            current_period = self._get_time_period(signal_time.time())
            if current_period and current_period == profile.best_time_of_day:
                adjustment += 3.0

        # Time period adjustment
        current_period = self._get_time_period(signal_time.time())
        if current_period and current_period in self._time_stats:
            stats = self._time_stats[current_period]
            if stats["win_rate"] > 60:
                adjustment += 3.0
            elif stats["win_rate"] < 40:
                adjustment -= 5.0

        # Signal score patterns
        for bin_key, stats in self._signal_stats.items():
            low, high = map(int, bin_key.split("-"))
            if low <= signal_score < high:
                if stats["win_rate"] > 60:
                    adjustment += 2.0
                elif stats["win_rate"] < 40:
                    adjustment -= 3.0
                break

        # Cap adjustment
        return max(-20.0, min(20.0, adjustment))

    def get_trading_recommendation(
        self,
        ticker: str,
        signal_time: datetime
    ) -> Dict[str, Any]:
        """Get trading recommendation based on patterns."""
        rec = {
            "ticker": ticker,
            "should_trade": True,
            "confidence": 50.0,
            "warnings": [],
            "advantages": [],
            "suggested_adjustments": {},
        }

        profile = self._ticker_profiles.get(ticker.upper())
        if profile:
            if profile.is_dangerous:
                rec["warnings"].append(f"Ticker has {profile.worst_trade_pct:.1f}% worst loss")
                rec["suggested_adjustments"]["reduce_size"] = 0.5

            if profile.is_favorable:
                rec["advantages"].append(f"Ticker has {profile.win_rate:.1f}% win rate")
                rec["confidence"] += 10

            if profile.trend == "DECLINING":
                rec["warnings"].append("Recent performance declining")
                rec["confidence"] -= 10
            elif profile.trend == "IMPROVING":
                rec["advantages"].append("Recent performance improving")
                rec["confidence"] += 5

            # Time recommendation
            current_period = self._get_time_period(signal_time.time())
            if profile.best_time_of_day and current_period != profile.best_time_of_day:
                rec["warnings"].append(
                    f"Best time for this ticker is {profile.best_time_of_day}"
                )

        rec["confidence"] = max(0, min(100, rec["confidence"]))
        rec["should_trade"] = rec["confidence"] >= 40 and not any(
            "worst loss" in w for w in rec["warnings"]
        )

        return rec

    def get_stats(self) -> Dict:
        """Get learner statistics."""
        return {
            "total_trades": len(self._trades),
            "tickers_profiled": len(self._ticker_profiles),
            "patterns_learned": len(self._patterns),
            "favorable_tickers": sum(
                1 for p in self._ticker_profiles.values() if p.is_favorable
            ),
            "dangerous_tickers": sum(
                1 for p in self._ticker_profiles.values() if p.is_dangerous
            ),
        }


# Singleton instance
_learner: Optional[PatternLearner] = None


def get_pattern_learner(config: Optional[LearnerConfig] = None) -> PatternLearner:
    """Get singleton PatternLearner instance."""
    global _learner
    if _learner is None:
        _learner = PatternLearner(config)
    return _learner
