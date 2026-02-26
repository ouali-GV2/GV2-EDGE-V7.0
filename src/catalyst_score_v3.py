# ============================
# CATALYST SCORE V3
# Enhanced Event-Based Scoring with Weighted Catalysts
# ============================
#
# V3 Enhancements over V2:
# 1. Weighted catalyst types (FDA > Earnings > Contract > etc.)
# 2. Temporal decay (fresh events > old events)
# 3. Quality assessment (source reliability, confirmation)
# 4. Catalyst confluence (multiple catalysts = higher score)
# 5. Historical performance tracking (learn from past)
#
# Architecture: ADDITIVE (V6 design principle)
# - Works alongside existing event_hub.py
# - Does NOT replace, only enhances
# - Provides boost multiplier for Monster Score

import math
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("CATALYST_V3")


# ============================
# CATALYST TYPE DEFINITIONS
# ============================

class CatalystType(Enum):
    """
    Catalyst types ordered by typical impact on small-cap stocks.
    UNIFIED TAXONOMY V6 - Aligned with nlp_event_parser and news_flow_screener.
    """
    # Tier 1: Critical Impact (0.90-1.00)
    FDA_APPROVAL = "fda_approval"
    PDUFA_DECISION = "pdufa_decision"
    BUYOUT_CONFIRMED = "buyout_confirmed"

    # Tier 2: High Impact (0.75-0.89)
    FDA_TRIAL_POSITIVE = "fda_trial_positive"
    BREAKTHROUGH_DESIGNATION = "breakthrough_designation"
    FDA_FAST_TRACK = "fda_fast_track"
    MERGER_ACQUISITION = "merger_acquisition"
    EARNINGS_BEAT_BIG = "earnings_beat_big"
    MAJOR_CONTRACT = "major_contract"
    MAJOR_PARTNERSHIP = "major_partnership"

    # Tier 3: Medium-High Impact (0.60-0.74)
    GUIDANCE_RAISE = "guidance_raise"
    EARNINGS_BEAT = "earnings_beat"
    PARTNERSHIP = "partnership"
    PRICE_TARGET_RAISE = "price_target_raise"
    NEW_PRODUCT = "new_product"
    PATENT_GRANTED = "patent_granted"
    INSIDER_BUYING = "insider_buying"

    # Tier 4: Medium Impact (0.45-0.59)
    ANALYST_UPGRADE = "analyst_upgrade"
    SHORT_SQUEEZE_SIGNAL = "short_squeeze_signal"
    UNUSUAL_VOLUME_NEWS = "unusual_volume_news"
    CONFERENCE_PRESENTATION = "conference_presentation"
    STOCK_BUYBACK = "stock_buyback"

    # Tier 5: Speculative (0.30-0.44)
    BUYOUT_RUMOR = "buyout_rumor"
    SOCIAL_MEDIA_SURGE = "social_media_surge"
    BREAKING_POSITIVE = "breaking_positive"
    FDA_SPECULATION = "fda_speculation"

    # Legacy/Other
    DIVIDEND_INCREASE = "dividend_increase"
    MANAGEMENT_CHANGE = "management_change"
    SOCIAL_MOMENTUM = "social_momentum"  # Legacy alias for SOCIAL_MEDIA_SURGE
    MERGER_ANNOUNCEMENT = "merger_announcement"  # Legacy alias for MERGER_ACQUISITION
    UNKNOWN = "unknown"


# Catalyst type weights (impact multiplier)
# UNIFIED TAXONOMY V6 - Consistent with all scoring modules
CATALYST_TYPE_WEIGHTS: Dict[CatalystType, float] = {
    # Tier 1: Critical Impact (0.90-1.00)
    CatalystType.FDA_APPROVAL: 1.0,
    CatalystType.PDUFA_DECISION: 0.95,
    CatalystType.BUYOUT_CONFIRMED: 0.92,

    # Tier 2: High Impact (0.75-0.89)
    CatalystType.FDA_TRIAL_POSITIVE: 0.88,
    CatalystType.BREAKTHROUGH_DESIGNATION: 0.85,
    CatalystType.FDA_FAST_TRACK: 0.82,
    CatalystType.MERGER_ACQUISITION: 0.88,
    CatalystType.EARNINGS_BEAT_BIG: 0.85,
    CatalystType.MAJOR_CONTRACT: 0.80,
    CatalystType.MAJOR_PARTNERSHIP: 0.78,

    # Tier 3: Medium-High Impact (0.60-0.74)
    CatalystType.GUIDANCE_RAISE: 0.72,
    CatalystType.EARNINGS_BEAT: 0.68,
    CatalystType.PARTNERSHIP: 0.65,
    CatalystType.PRICE_TARGET_RAISE: 0.62,
    CatalystType.NEW_PRODUCT: 0.60,
    CatalystType.PATENT_GRANTED: 0.58,
    CatalystType.INSIDER_BUYING: 0.55,

    # Tier 4: Medium Impact (0.45-0.59)
    CatalystType.ANALYST_UPGRADE: 0.55,
    CatalystType.SHORT_SQUEEZE_SIGNAL: 0.52,
    CatalystType.UNUSUAL_VOLUME_NEWS: 0.48,
    CatalystType.CONFERENCE_PRESENTATION: 0.45,
    CatalystType.STOCK_BUYBACK: 0.42,

    # Tier 5: Speculative (0.30-0.44)
    CatalystType.BUYOUT_RUMOR: 0.42,
    CatalystType.SOCIAL_MEDIA_SURGE: 0.38,
    CatalystType.BREAKING_POSITIVE: 0.35,
    CatalystType.FDA_SPECULATION: 0.32,

    # Legacy/Other
    CatalystType.DIVIDEND_INCREASE: 0.40,
    CatalystType.MANAGEMENT_CHANGE: 0.35,
    CatalystType.SOCIAL_MOMENTUM: 0.38,  # Alias
    CatalystType.MERGER_ANNOUNCEMENT: 0.88,  # Alias
    CatalystType.UNKNOWN: 0.30,
}


# ============================
# SOURCE RELIABILITY
# ============================

class SourceTier(Enum):
    """Source reliability tiers"""
    OFFICIAL = "official"      # SEC filings, company PR
    MAJOR_NEWS = "major_news"  # Reuters, Bloomberg, WSJ
    FINANCIAL = "financial"    # Finnhub, Yahoo Finance
    SOCIAL = "social"          # Twitter, Reddit, StockTwits
    UNKNOWN = "unknown"


SOURCE_RELIABILITY: Dict[SourceTier, float] = {
    SourceTier.OFFICIAL: 1.0,
    SourceTier.MAJOR_NEWS: 0.90,
    SourceTier.FINANCIAL: 0.75,
    SourceTier.SOCIAL: 0.50,
    SourceTier.UNKNOWN: 0.40,
}


# ============================
# DATA CLASSES
# ============================

@dataclass
class Catalyst:
    """Single catalyst event"""
    ticker: str
    catalyst_type: CatalystType
    headline: str
    source: SourceTier
    timestamp: datetime
    raw_impact_score: float = 0.5  # From NLP (0-1)
    confirmed: bool = False  # Multiple sources confirm
    details: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.catalyst_type, str):
            self.catalyst_type = self._parse_catalyst_type(self.catalyst_type)
        if isinstance(self.source, str):
            self.source = self._parse_source(self.source)

    def _parse_catalyst_type(self, type_str: str) -> CatalystType:
        """Parse string to CatalystType enum - UNIFIED TAXONOMY V6"""
        type_map = {
            # Tier 1: Critical
            "fda_approval": CatalystType.FDA_APPROVAL,
            "pdufa_decision": CatalystType.PDUFA_DECISION,
            "buyout_confirmed": CatalystType.BUYOUT_CONFIRMED,

            # Tier 2: High
            "fda_trial_positive": CatalystType.FDA_TRIAL_POSITIVE,
            "fda_trial_result": CatalystType.FDA_TRIAL_POSITIVE,  # Legacy alias
            "breakthrough_designation": CatalystType.BREAKTHROUGH_DESIGNATION,
            "fda_fast_track": CatalystType.FDA_FAST_TRACK,
            "merger_acquisition": CatalystType.MERGER_ACQUISITION,
            "merger_announcement": CatalystType.MERGER_ACQUISITION,  # Legacy alias
            "earnings_beat_big": CatalystType.EARNINGS_BEAT_BIG,
            "major_contract": CatalystType.MAJOR_CONTRACT,
            "major_partnership": CatalystType.MAJOR_PARTNERSHIP,

            # Tier 3: Medium-High
            "guidance_raise": CatalystType.GUIDANCE_RAISE,
            "earnings_beat": CatalystType.EARNINGS_BEAT,
            "partnership": CatalystType.PARTNERSHIP,
            "price_target_raise": CatalystType.PRICE_TARGET_RAISE,
            "new_product": CatalystType.NEW_PRODUCT,
            "patent_granted": CatalystType.PATENT_GRANTED,
            "insider_buying": CatalystType.INSIDER_BUYING,

            # Tier 4: Medium
            "analyst_upgrade": CatalystType.ANALYST_UPGRADE,
            "short_squeeze_signal": CatalystType.SHORT_SQUEEZE_SIGNAL,
            "short_squeeze": CatalystType.SHORT_SQUEEZE_SIGNAL,  # Legacy alias
            "unusual_volume_news": CatalystType.UNUSUAL_VOLUME_NEWS,
            "conference_presentation": CatalystType.CONFERENCE_PRESENTATION,
            "stock_buyback": CatalystType.STOCK_BUYBACK,

            # Tier 5: Speculative
            "buyout_rumor": CatalystType.BUYOUT_RUMOR,
            "social_media_surge": CatalystType.SOCIAL_MEDIA_SURGE,
            "social_momentum": CatalystType.SOCIAL_MEDIA_SURGE,  # Legacy alias
            "breaking_positive": CatalystType.BREAKING_POSITIVE,
            "fda_speculation": CatalystType.FDA_SPECULATION,

            # Legacy/Other
            "dividend_increase": CatalystType.DIVIDEND_INCREASE,
            "management_change": CatalystType.MANAGEMENT_CHANGE,
        }
        return type_map.get(type_str.lower(), CatalystType.UNKNOWN)

    def _parse_source(self, source_str: str) -> SourceTier:
        """Parse string to SourceTier enum"""
        source_map = {
            "sec": SourceTier.OFFICIAL,
            "official": SourceTier.OFFICIAL,
            "company_pr": SourceTier.OFFICIAL,
            "reuters": SourceTier.MAJOR_NEWS,
            "bloomberg": SourceTier.MAJOR_NEWS,
            "wsj": SourceTier.MAJOR_NEWS,
            "major_news": SourceTier.MAJOR_NEWS,
            "finnhub": SourceTier.FINANCIAL,
            "yahoo": SourceTier.FINANCIAL,
            "financial": SourceTier.FINANCIAL,
            "twitter": SourceTier.SOCIAL,
            "reddit": SourceTier.SOCIAL,
            "stocktwits": SourceTier.SOCIAL,
            "social": SourceTier.SOCIAL,
        }
        return source_map.get(source_str.lower(), SourceTier.UNKNOWN)


@dataclass
class CatalystScore:
    """Calculated catalyst score with all components"""
    ticker: str
    catalysts: List[Catalyst]

    # Component scores
    type_score: float = 0.0       # From catalyst type weight
    recency_score: float = 0.0   # Temporal decay applied
    quality_score: float = 0.0   # Source reliability + confirmation
    confluence_score: float = 0.0  # Multiple catalyst bonus

    # Final scores
    raw_score: float = 0.0       # Combined before normalization
    final_score: float = 0.0     # Normalized 0-1
    boost_multiplier: float = 1.0  # For Monster Score

    # Metadata
    catalyst_count: int = 0
    primary_catalyst: Optional[CatalystType] = None
    alert_level: str = "NONE"    # NONE, LOW, MEDIUM, HIGH, CRITICAL

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "ticker": self.ticker,
            "catalyst_count": self.catalyst_count,
            "primary_catalyst": self.primary_catalyst.value if self.primary_catalyst else None,
            "type_score": round(self.type_score, 3),
            "recency_score": round(self.recency_score, 3),
            "quality_score": round(self.quality_score, 3),
            "confluence_score": round(self.confluence_score, 3),
            "raw_score": round(self.raw_score, 3),
            "final_score": round(self.final_score, 3),
            "boost_multiplier": round(self.boost_multiplier, 3),
            "alert_level": self.alert_level,
        }


# ============================
# TEMPORAL DECAY
# ============================

def calculate_temporal_decay(
    event_time: datetime,
    now: Optional[datetime] = None,
    half_life_hours: float = 24.0
) -> float:
    """
    Calculate temporal decay for catalyst freshness.

    Uses exponential decay with configurable half-life.
    - Fresh (0-6h): ~0.85-1.0
    - Recent (6-24h): ~0.5-0.85
    - Old (24-72h): ~0.125-0.5
    - Stale (>72h): <0.125

    Args:
        event_time: When the catalyst occurred
        now: Current time (default: now)
        half_life_hours: Hours for score to decay by 50%

    Returns:
        Decay multiplier (0-1)
    """
    if now is None:
        now = datetime.now()

    hours_since = (now - event_time).total_seconds() / 3600

    if hours_since <= 0:
        return 1.0

    # Exponential decay: score = e^(-λt) where λ = ln(2)/half_life
    decay_constant = 0.693 / half_life_hours
    decay = math.exp(-decay_constant * hours_since)

    return max(0.05, decay)  # Floor at 5%


def calculate_proximity_boost(
    event_time: datetime,
    now: Optional[datetime] = None
) -> float:
    """
    Calculate proximity boost for upcoming events.

    For FUTURE events (earnings, FDA dates), closer = higher boost.

    Returns:
        Proximity multiplier (1.0-1.5)
    """
    if now is None:
        now = datetime.now()

    hours_until = (event_time - now).total_seconds() / 3600

    if hours_until <= 0:
        # Event already happened, no proximity boost
        return 1.0

    if hours_until <= 24:
        return 1.5  # Today
    elif hours_until <= 48:
        return 1.3  # Tomorrow
    elif hours_until <= 168:
        return 1.1  # This week
    else:
        return 1.0  # No boost


# ============================
# QUALITY ASSESSMENT
# ============================

def calculate_quality_score(
    catalyst: Catalyst,
    confirmation_count: int = 1
) -> float:
    """
    Calculate quality score based on source reliability and confirmation.

    Args:
        catalyst: The catalyst to score
        confirmation_count: Number of sources confirming

    Returns:
        Quality score (0-1)
    """
    # Base reliability from source tier
    base_reliability = SOURCE_RELIABILITY.get(catalyst.source, 0.4)

    # Confirmation bonus (multiple sources = higher confidence)
    confirmation_multiplier = 1.0
    if confirmation_count >= 3:
        confirmation_multiplier = 1.3
    elif confirmation_count >= 2:
        confirmation_multiplier = 1.15

    # Official confirmation flag
    if catalyst.confirmed:
        confirmation_multiplier = max(confirmation_multiplier, 1.2)

    quality = base_reliability * confirmation_multiplier

    return min(1.0, quality)


# ============================
# CONFLUENCE SCORING
# ============================

def calculate_confluence_bonus(
    catalysts: List[Catalyst],
    max_bonus: float = 0.4
) -> float:
    """
    Calculate bonus for multiple catalysts.

    Multiple catalysts on the same ticker = higher conviction.
    Different types of catalysts = even higher.

    Args:
        catalysts: List of catalysts for the ticker
        max_bonus: Maximum confluence bonus

    Returns:
        Confluence bonus (0-max_bonus)
    """
    if len(catalysts) <= 1:
        return 0.0

    # Count unique catalyst types
    unique_types = set(c.catalyst_type for c in catalysts)

    # Base bonus for multiple catalysts
    count_bonus = min(0.15, (len(catalysts) - 1) * 0.05)

    # Additional bonus for diversity of catalyst types
    diversity_bonus = min(0.25, (len(unique_types) - 1) * 0.08)

    total_bonus = count_bonus + diversity_bonus

    return min(max_bonus, total_bonus)


# ============================
# MAIN SCORING ENGINE
# ============================

class CatalystScorerV3:
    """
    Enhanced Catalyst Scoring Engine V3

    Combines:
    - Type weighting
    - Temporal decay
    - Quality assessment
    - Confluence scoring
    - Historical performance tracking
    """

    def __init__(
        self,
        db_path: str = "data/catalyst_history.db",
        decay_half_life_hours: float = 24.0,
        min_score_threshold: float = 0.1,
        max_boost: float = 1.6
    ):
        """
        Initialize the scorer.

        Args:
            db_path: Path to SQLite database for history
            decay_half_life_hours: Half-life for temporal decay
            min_score_threshold: Minimum score to consider
            max_boost: Maximum boost multiplier
        """
        self.db_path = Path(db_path)
        self.decay_half_life = decay_half_life_hours
        self.min_threshold = min_score_threshold
        self.max_boost = max_boost

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"CatalystScorerV3 initialized (half_life={decay_half_life_hours}h)")

    def _init_db(self):
        """Initialize SQLite database for history tracking"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()

            # Catalyst events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS catalyst_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    catalyst_type TEXT NOT NULL,
                    headline TEXT,
                    source TEXT,
                    timestamp DATETIME NOT NULL,
                    raw_impact_score REAL,
                    final_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS catalyst_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    catalyst_type TEXT NOT NULL,
                    catalyst_score REAL,
                    price_before REAL,
                    price_after_1h REAL,
                    price_after_1d REAL,
                    price_after_1w REAL,
                    actual_return_1d REAL,
                    actual_return_1w REAL,
                    event_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_catalyst_ticker
                ON catalyst_events(ticker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_catalyst_type
                ON catalyst_events(catalyst_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_catalyst_timestamp
                ON catalyst_events(timestamp)
            """)

            conn.commit()

    def score_catalysts(
        self,
        ticker: str,
        catalysts: List[Catalyst],
        now: Optional[datetime] = None
    ) -> CatalystScore:
        """
        Calculate comprehensive catalyst score for a ticker.

        Args:
            ticker: Stock ticker
            catalysts: List of catalyst events
            now: Current time (default: now)

        Returns:
            CatalystScore with all components
        """
        if now is None:
            now = datetime.now()

        if not catalysts:
            return CatalystScore(
                ticker=ticker,
                catalysts=[],
                catalyst_count=0,
                alert_level="NONE"
            )

        # Sort by timestamp (newest first)
        sorted_catalysts = sorted(
            catalysts,
            key=lambda c: c.timestamp,
            reverse=True
        )

        # Calculate component scores
        type_scores = []
        recency_scores = []
        quality_scores = []

        for catalyst in sorted_catalysts:
            # Type score
            type_weight = CATALYST_TYPE_WEIGHTS.get(
                catalyst.catalyst_type,
                0.3
            )
            type_scores.append(type_weight * catalyst.raw_impact_score)

            # Recency score (apply decay)
            if catalyst.timestamp <= now:
                # Past event: apply decay
                decay = calculate_temporal_decay(
                    catalyst.timestamp,
                    now,
                    self.decay_half_life
                )
            else:
                # Future event: apply proximity boost
                decay = calculate_proximity_boost(catalyst.timestamp, now)
            recency_scores.append(decay)

            # Quality score
            quality = calculate_quality_score(catalyst)
            quality_scores.append(quality)

        # Aggregate scores (weighted by recency)
        total_weight = sum(recency_scores)
        if total_weight > 0:
            type_score = sum(
                t * r for t, r in zip(type_scores, recency_scores)
            ) / total_weight
            quality_score = sum(
                q * r for q, r in zip(quality_scores, recency_scores)
            ) / total_weight
            recency_score = max(recency_scores)  # Best recency
        else:
            type_score = sum(type_scores) / len(type_scores)
            quality_score = sum(quality_scores) / len(quality_scores)
            recency_score = 0.5

        # Confluence bonus
        confluence_score = calculate_confluence_bonus(sorted_catalysts)

        # Combined raw score
        # S5-3 FIX: confluence_score was counted twice (in weighted sum AND as additive).
        # Removed the additive line — confluence already weighted at 15%.
        raw_score = (
            type_score * 0.40 +        # Type is most important
            recency_score * 0.25 +     # Freshness matters
            quality_score * 0.20 +     # Source reliability
            confluence_score * 0.15    # Multiple catalysts bonus
        )

        # Normalize to 0-1
        final_score = min(1.0, raw_score)

        # Calculate boost multiplier for Monster Score
        # Linear interpolation: score 0.3 = 1.0x, score 1.0 = max_boost
        if final_score < self.min_threshold:
            boost = 1.0
        else:
            boost_range = self.max_boost - 1.0
            score_range = 1.0 - self.min_threshold
            boost = 1.0 + boost_range * (
                (final_score - self.min_threshold) / score_range
            )

        # Determine alert level
        alert_level = self._determine_alert_level(final_score, sorted_catalysts)

        # Primary catalyst (highest impact)
        primary = max(
            sorted_catalysts,
            key=lambda c: CATALYST_TYPE_WEIGHTS.get(c.catalyst_type, 0.3)
        ).catalyst_type

        result = CatalystScore(
            ticker=ticker,
            catalysts=sorted_catalysts,
            type_score=type_score,
            recency_score=recency_score,
            quality_score=quality_score,
            confluence_score=confluence_score,
            raw_score=raw_score,
            final_score=final_score,
            boost_multiplier=min(self.max_boost, boost),
            catalyst_count=len(sorted_catalysts),
            primary_catalyst=primary,
            alert_level=alert_level
        )

        logger.debug(
            f"{ticker}: CatalystScore V3 = {final_score:.3f} "
            f"(type={type_score:.2f}, recency={recency_score:.2f}, "
            f"quality={quality_score:.2f}, confluence={confluence_score:.2f}) "
            f"boost={boost:.2f}x"
        )

        return result

    def _determine_alert_level(
        self,
        score: float,
        catalysts: List[Catalyst]
    ) -> str:
        """Determine alert level based on score and catalyst types"""

        # Check for high-impact catalyst types
        # S5-3 FIX: Added PDUFA_DECISION and BREAKTHROUGH_DESIGNATION to high_impact_types
        high_impact_types = {
            CatalystType.FDA_APPROVAL,
            CatalystType.PDUFA_DECISION,
            CatalystType.BREAKTHROUGH_DESIGNATION,
            CatalystType.BUYOUT_CONFIRMED,
            CatalystType.MAJOR_PARTNERSHIP,
            CatalystType.FDA_TRIAL_POSITIVE,
            CatalystType.EARNINGS_BEAT_BIG,
        }

        has_high_impact = any(
            c.catalyst_type in high_impact_types for c in catalysts
        )

        if score >= 0.8 or (score >= 0.6 and has_high_impact):
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "NONE"

    def record_catalyst(self, catalyst: Catalyst, final_score: float):
        """Record a catalyst event to history"""
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO catalyst_events
                    (ticker, catalyst_type, headline, source, timestamp,
                     raw_impact_score, final_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    catalyst.ticker,
                    catalyst.catalyst_type.value,
                    catalyst.headline,
                    catalyst.source.value,
                    catalyst.timestamp.isoformat(),
                    catalyst.raw_impact_score,
                    final_score
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record catalyst: {e}")

    def record_performance(
        self,
        ticker: str,
        catalyst_type: CatalystType,
        catalyst_score: float,
        price_before: float,
        price_after_1d: float,
        price_after_1w: Optional[float] = None,
        event_date: Optional[datetime] = None
    ):
        """Record catalyst performance for learning"""
        try:
            if event_date is None:
                event_date = datetime.now()

            return_1d = (price_after_1d - price_before) / price_before
            return_1w = None
            if price_after_1w:
                return_1w = (price_after_1w - price_before) / price_before

            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO catalyst_performance
                    (ticker, catalyst_type, catalyst_score, price_before,
                     price_after_1d, price_after_1w, actual_return_1d,
                     actual_return_1w, event_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    catalyst_type.value,
                    catalyst_score,
                    price_before,
                    price_after_1d,
                    price_after_1w,
                    return_1d,
                    return_1w,
                    event_date.date().isoformat()
                ))
                conn.commit()

            logger.info(
                f"Recorded performance: {ticker} {catalyst_type.value} "
                f"return_1d={return_1d:.2%}"
            )
        except Exception as e:
            logger.error(f"Failed to record performance: {e}")

    def get_type_performance_stats(
        self,
        lookback_days: int = 90
    ) -> Dict[str, Dict]:
        """
        Get historical performance stats by catalyst type.

        Useful for adaptive weight adjustment.
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).date()

        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        catalyst_type,
                        COUNT(*) as count,
                        AVG(actual_return_1d) as avg_return_1d,
                        AVG(actual_return_1w) as avg_return_1w,
                        SUM(CASE WHEN actual_return_1d > 0 THEN 1 ELSE 0 END) as wins,
                        AVG(catalyst_score) as avg_score
                    FROM catalyst_performance
                    WHERE event_date >= ?
                    GROUP BY catalyst_type
                    ORDER BY avg_return_1d DESC
                """, (cutoff.isoformat(),))

                stats = {}
                for row in cursor.fetchall():
                    catalyst_type = row[0]
                    count = row[1]
                    win_rate = row[4] / count if count > 0 else 0

                    stats[catalyst_type] = {
                        "count": count,
                        "avg_return_1d": row[2],
                        "avg_return_1w": row[3],
                        "win_rate": win_rate,
                        "avg_score": row[5]
                    }

                return stats
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}

    def get_recent_catalysts(
        self,
        ticker: str,
        hours: int = 72
    ) -> List[Dict]:
        """Get recent catalysts for a ticker"""
        cutoff = datetime.now() - timedelta(hours=hours)

        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT catalyst_type, headline, source, timestamp,
                           raw_impact_score, final_score
                    FROM catalyst_events
                    WHERE ticker = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (ticker, cutoff.isoformat()))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "catalyst_type": row[0],
                        "headline": row[1],
                        "source": row[2],
                        "timestamp": row[3],
                        "raw_impact_score": row[4],
                        "final_score": row[5]
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get recent catalysts: {e}")
            return []


# ============================
# HELPER FUNCTIONS
# ============================

def get_catalyst_boost(
    ticker: str,
    catalysts: List[Catalyst],
    scorer: Optional[CatalystScorerV3] = None
) -> Tuple[float, CatalystScore]:
    """
    Convenience function to get catalyst boost for Monster Score.

    Args:
        ticker: Stock ticker
        catalysts: List of catalyst events
        scorer: Optional scorer instance (creates one if None)

    Returns:
        Tuple of (boost_multiplier, full_score)
    """
    if scorer is None:
        scorer = CatalystScorerV3()

    score = scorer.score_catalysts(ticker, catalysts)
    return score.boost_multiplier, score


def create_catalyst_from_event(
    ticker: str,
    event_type: str,
    headline: str,
    source: str = "financial",
    timestamp: Optional[datetime] = None,
    impact_score: float = 0.5
) -> Catalyst:
    """
    Create a Catalyst object from event data.

    Convenience function for integration with event_hub.py
    """
    if timestamp is None:
        timestamp = datetime.now()

    return Catalyst(
        ticker=ticker,
        catalyst_type=event_type,
        headline=headline,
        source=source,
        timestamp=timestamp,
        raw_impact_score=impact_score
    )


# ============================
# INTEGRATION WITH EVENT HUB
# ============================

def enhance_event_with_catalyst_score(
    event: Dict,
    scorer: Optional[CatalystScorerV3] = None
) -> Dict:
    """
    Enhance an event dict from event_hub with V3 catalyst score.

    Args:
        event: Event dict from event_hub.py
        scorer: Optional scorer instance

    Returns:
        Enhanced event dict with catalyst_v3 data
    """
    if scorer is None:
        scorer = CatalystScorerV3()

    # Extract catalyst from event
    catalyst = create_catalyst_from_event(
        ticker=event.get("ticker", ""),
        event_type=event.get("event_type", "unknown"),
        headline=event.get("headline", ""),
        source=event.get("source", "financial"),
        timestamp=datetime.fromisoformat(event["timestamp"])
            if "timestamp" in event else datetime.now(),
        impact_score=event.get("impact_score", 0.5)
    )

    # Score
    score = scorer.score_catalysts(catalyst.ticker, [catalyst])

    # Add to event
    event["catalyst_v3"] = score.to_dict()
    event["catalyst_boost"] = score.boost_multiplier

    return event


# ============================
# SINGLETON (S5-3 FIX)
# ============================

_catalyst_scorer_lock = threading.Lock()
_catalyst_scorer_instance: Optional[CatalystScorerV3] = None


def get_catalyst_scorer() -> CatalystScorerV3:
    """Thread-safe singleton for CatalystScorerV3."""
    global _catalyst_scorer_instance
    if _catalyst_scorer_instance is None:
        with _catalyst_scorer_lock:
            if _catalyst_scorer_instance is None:
                _catalyst_scorer_instance = CatalystScorerV3()
    return _catalyst_scorer_instance


# ============================
# TESTING
# ============================

if __name__ == "__main__":
    # Test the scorer
    scorer = CatalystScorerV3(db_path="data/test_catalyst.db")

    # Create test catalysts
    test_catalysts = [
        Catalyst(
            ticker="ABCD",
            catalyst_type=CatalystType.FDA_APPROVAL,
            headline="ABCD receives FDA approval for drug XYZ",
            source=SourceTier.OFFICIAL,
            timestamp=datetime.now() - timedelta(hours=2),
            raw_impact_score=0.9,
            confirmed=True
        ),
        Catalyst(
            ticker="ABCD",
            catalyst_type=CatalystType.ANALYST_UPGRADE,
            headline="Goldman upgrades ABCD to Buy",
            source=SourceTier.MAJOR_NEWS,
            timestamp=datetime.now() - timedelta(hours=6),
            raw_impact_score=0.7
        ),
    ]

    # Score
    result = scorer.score_catalysts("ABCD", test_catalysts)

    print("\n" + "="*50)
    print("CATALYST SCORE V3 TEST")
    print("="*50)
    print(f"Ticker: {result.ticker}")
    print(f"Catalyst Count: {result.catalyst_count}")
    print(f"Primary Catalyst: {result.primary_catalyst.value}")
    print(f"\nComponent Scores:")
    print(f"  Type Score:       {result.type_score:.3f}")
    print(f"  Recency Score:    {result.recency_score:.3f}")
    print(f"  Quality Score:    {result.quality_score:.3f}")
    print(f"  Confluence Score: {result.confluence_score:.3f}")
    print(f"\nFinal Score: {result.final_score:.3f}")
    print(f"Boost Multiplier: {result.boost_multiplier:.2f}x")
    print(f"Alert Level: {result.alert_level}")
    print("="*50)
