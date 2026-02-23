"""
Dilution Detector V8 - SEC Filings and Offering Analysis

V8 FIX (P4 CRITICAL): Distinction between potential and active dilution
- Old: S-3 shelf = CRITICAL risk = blocked 90 days (wrong)
- New: 4-tier classification system:
  1. ACTIVE_OFFERING → 0.20x (confirmed offering, pricing imminent)
  2. SHELF_RECENT <30 days → 0.60x (recently filed, monitor closely)
  3. SHELF_DORMANT >30 days → 0.85x (on file but unused)
  4. CAPACITY_ONLY → 1.00x (no action, just paperwork)

Detects dilution risks from:
- S-3 shelf registrations (CAPACITY only, not active dilution)
- Prospectus supplements (imminent offering - ACTIVE)
- ATM (At-The-Market) programs
- Direct offerings / registered direct
- PIPE deals
- Warrant exercises
- Convertible note conversions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class DilutionType(Enum):
    """Types of dilution events."""
    S3_SHELF = "S3_SHELF"                    # Shelf registration
    PROSPECTUS_SUPPLEMENT = "PROSPECTUS"     # 424B filing
    ATM_PROGRAM = "ATM"                      # At-the-market offering
    DIRECT_OFFERING = "DIRECT"               # Registered direct
    PIPE_DEAL = "PIPE"                       # Private investment in public equity
    WARRANT_EXERCISE = "WARRANT"             # Warrant exercise/reset
    CONVERTIBLE_NOTE = "CONVERTIBLE"         # Convert to shares
    SECONDARY_OFFERING = "SECONDARY"         # Follow-on offering
    MIXED_SHELF = "MIXED_SHELF"              # Debt + equity shelf


class DilutionRisk(Enum):
    """Risk severity levels."""
    CRITICAL = "CRITICAL"   # Imminent dilution, avoid entirely
    HIGH = "HIGH"           # Active program, reduce position
    MEDIUM = "MEDIUM"       # Potential risk, monitor closely
    LOW = "LOW"             # Minimal current risk
    NONE = "NONE"           # No dilution detected


class DilutionTier(Enum):
    """
    V8: 4-tier dilution classification to distinguish potential from active risk.

    A shelf registration (S-3) is NOT the same as an active offering.
    Previously all were treated as CRITICAL → block 90 days (WRONG).
    """
    ACTIVE_OFFERING = "ACTIVE_OFFERING"     # Confirmed offering, pricing imminent → 0.20x
    SHELF_RECENT = "SHELF_RECENT"           # S-3 filed <30 days ago → 0.60x (monitor)
    SHELF_DORMANT = "SHELF_DORMANT"         # S-3 filed >30 days ago, unused → 0.85x
    CAPACITY_ONLY = "CAPACITY_ONLY"         # Old shelf / no activity → 1.00x (no impact)
    NONE = "NONE"                           # No dilution filings


# V8: Position multipliers by dilution tier
DILUTION_TIER_MULTIPLIERS = {
    DilutionTier.ACTIVE_OFFERING: 0.20,
    DilutionTier.SHELF_RECENT: 0.60,
    DilutionTier.SHELF_DORMANT: 0.85,
    DilutionTier.CAPACITY_ONLY: 1.00,
    DilutionTier.NONE: 1.00,
}


@dataclass
class DilutionEvent:
    """Represents a dilution event or filing."""
    ticker: str
    event_type: DilutionType
    filing_date: datetime
    effective_date: Optional[datetime] = None

    # Offering details
    shares_registered: Optional[int] = None
    dollar_amount: Optional[float] = None
    price_per_share: Optional[float] = None

    # Shelf details
    shelf_capacity_remaining: Optional[float] = None
    shelf_expiry: Optional[datetime] = None

    # Status
    is_active: bool = True
    is_completed: bool = False

    # SEC filing reference
    accession_number: Optional[str] = None
    filing_url: Optional[str] = None

    # Computed
    dilution_percent: Optional[float] = None  # vs current shares outstanding

    def days_since_filing(self) -> int:
        """Days since the filing date."""
        return (datetime.now() - self.filing_date).days

    def is_stale(self, days: int = 365) -> bool:
        """Check if filing is old/stale."""
        return self.days_since_filing() > days


@dataclass
class DilutionProfile:
    """Complete dilution profile for a ticker."""
    ticker: str
    risk_level: DilutionRisk = DilutionRisk.NONE
    risk_score: float = 0.0  # 0-100

    # V8: Dilution tier classification (replaces binary logic)
    dilution_tier: DilutionTier = DilutionTier.NONE

    # Active events
    events: List[DilutionEvent] = field(default_factory=list)

    # Aggregated info
    total_shelf_capacity: float = 0.0
    active_atm_capacity: float = 0.0
    warrants_outstanding: int = 0
    convertible_debt: float = 0.0

    # Shares info
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    last_filing_date: Optional[datetime] = None

    # Flags
    has_active_offering: bool = False
    has_active_atm: bool = False
    has_recent_s3: bool = False
    has_toxic_financing: bool = False  # Death spiral converts, etc.

    def get_block_reason(self) -> Optional[str]:
        """Get reason string if trade should be blocked."""
        # V8: Only block on truly active/toxic dilution
        if self.has_toxic_financing:
            return "TOXIC_FINANCING"
        if self.has_active_offering:
            return "ACTIVE_OFFERING"
        # V8: S-3 shelf alone is NOT a block reason anymore
        return None

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on dilution tier (V8).

        V8 CHANGE: Uses tier-based multipliers instead of binary risk levels.
        An S-3 shelf >30 days old → 0.85x (not 0.0x as before).
        """
        # V8: Prefer tier-based multiplier if available
        if self.dilution_tier != DilutionTier.NONE:
            return DILUTION_TIER_MULTIPLIERS.get(self.dilution_tier, 1.0)

        # Fallback to risk-level based (legacy)
        multipliers = {
            DilutionRisk.CRITICAL: 0.20,  # V8: Raised from 0.0 (still tradeable micro-size)
            DilutionRisk.HIGH: 0.25,
            DilutionRisk.MEDIUM: 0.50,
            DilutionRisk.LOW: 0.75,
            DilutionRisk.NONE: 1.0
        }
        return multipliers.get(self.risk_level, 1.0)


# SEC filing type patterns
SEC_FILING_PATTERNS = {
    DilutionType.S3_SHELF: [
        r"S-3",
        r"S-3/A",
        r"S-3ASR",  # Automatic shelf registration
    ],
    DilutionType.PROSPECTUS_SUPPLEMENT: [
        r"424B[1-5]",
        r"FWP",      # Free writing prospectus
        r"PROSUP",
    ],
    DilutionType.ATM_PROGRAM: [
        r"at.the.market",
        r"ATM\s+(?:program|offering|agreement)",
        r"equity\s+distribution\s+agreement",
        r"sales\s+agreement",
    ],
    DilutionType.DIRECT_OFFERING: [
        r"registered\s+direct",
        r"direct\s+offering",
        r"RDO",
    ],
    DilutionType.PIPE_DEAL: [
        r"private\s+placement",
        r"PIPE",
        r"private\s+investment",
    ],
    DilutionType.WARRANT_EXERCISE: [
        r"warrant\s+(?:exercise|inducement|amendment)",
        r"exercise\s+price\s+(?:reduction|reset)",
    ],
    DilutionType.CONVERTIBLE_NOTE: [
        r"convertible\s+(?:note|debenture|bond)",
        r"conversion\s+(?:price|shares)",
    ],
}

# Toxic financing patterns (death spiral, etc.)
TOXIC_PATTERNS = [
    r"variable\s+(?:rate|price)\s+convert",
    r"floating\s+conversion",
    r"reset\s+(?:provision|price)",
    r"equity\s+line",
    r"ELOC",
    r"death\s+spiral",
]


class DilutionDetector:
    """
    Detects and tracks dilution risks for tickers.

    Usage:
        detector = DilutionDetector()

        # Check single ticker
        profile = await detector.analyze(ticker, sec_filings)
        if profile.risk_level == DilutionRisk.CRITICAL:
            block_trade()

        # Batch check
        profiles = await detector.analyze_batch(tickers)
    """

    def __init__(self):
        # Cache of dilution profiles
        self._profiles: Dict[str, DilutionProfile] = {}

        # Known toxic tickers (manually flagged)
        self._toxic_tickers: Set[str] = set()

        # Cache TTL
        self._cache_ttl = timedelta(hours=1)

        # Compile regex patterns
        self._filing_patterns = self._compile_patterns(SEC_FILING_PATTERNS)
        self._toxic_patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def _compile_patterns(
        self,
        patterns: Dict[DilutionType, List[str]]
    ) -> Dict[DilutionType, List[re.Pattern]]:
        """Compile regex patterns."""
        compiled = {}
        for dtype, pattern_list in patterns.items():
            compiled[dtype] = [
                re.compile(p, re.IGNORECASE)
                for p in pattern_list
            ]
        return compiled

    async def analyze(
        self,
        ticker: str,
        sec_filings: Optional[List[Dict]] = None,
        shares_outstanding: Optional[int] = None,
        float_shares: Optional[int] = None,
        force_refresh: bool = False
    ) -> DilutionProfile:
        """
        Analyze dilution risk for a ticker.

        Args:
            ticker: Stock ticker
            sec_filings: List of SEC filing dicts with keys:
                - form_type: "S-3", "424B5", etc.
                - filed_date: datetime or str
                - description: Filing description
                - accession_number: SEC accession number
                - url: Link to filing
            shares_outstanding: Current shares outstanding
            float_shares: Current float
            force_refresh: Force cache refresh

        Returns:
            DilutionProfile with risk assessment
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._profiles:
            cached = self._profiles[ticker]
            if datetime.now() - cached.last_updated < self._cache_ttl:
                return cached

        # Create new profile
        profile = DilutionProfile(
            ticker=ticker,
            shares_outstanding=shares_outstanding,
            float_shares=float_shares
        )

        # Check if manually flagged as toxic
        if ticker in self._toxic_tickers:
            profile.has_toxic_financing = True
            profile.risk_level = DilutionRisk.CRITICAL
            profile.risk_score = 100.0
            self._profiles[ticker] = profile
            return profile

        # Analyze SEC filings if provided
        if sec_filings:
            await self._analyze_filings(profile, sec_filings)

        # Calculate overall risk
        self._calculate_risk(profile)

        # Cache result
        self._profiles[ticker] = profile

        return profile

    async def _analyze_filings(
        self,
        profile: DilutionProfile,
        filings: List[Dict]
    ) -> None:
        """
        Analyze SEC filings for dilution events.

        V8 FIX (P4): Properly classify S-3 shelfs vs active offerings.
        """
        for filing in filings:
            event = self._parse_filing(profile.ticker, filing)
            if event:
                profile.events.append(event)

                # Update flags
                if event.event_type == DilutionType.ATM_PROGRAM and event.is_active:
                    profile.has_active_atm = True
                    if event.shelf_capacity_remaining:
                        profile.active_atm_capacity += event.shelf_capacity_remaining

                if event.event_type == DilutionType.S3_SHELF:
                    # V8 FIX: Classify S-3 by age instead of binary recent/old
                    days = event.days_since_filing()
                    if days < 30:
                        profile.has_recent_s3 = True
                    # Note: S-3 shelf alone does NOT set has_active_offering
                    if event.shelf_capacity_remaining:
                        profile.total_shelf_capacity += event.shelf_capacity_remaining

                # V8: Only these types indicate ACTIVE dilution
                if event.event_type in [
                    DilutionType.DIRECT_OFFERING,
                    DilutionType.PROSPECTUS_SUPPLEMENT,
                    DilutionType.SECONDARY_OFFERING,
                ]:
                    if event.days_since_filing() < 7:
                        profile.has_active_offering = True

                if event.filing_date:
                    if not profile.last_filing_date or event.filing_date > profile.last_filing_date:
                        profile.last_filing_date = event.filing_date

        # Check for toxic patterns in filing descriptions
        for filing in filings:
            desc = filing.get("description", "")
            for pattern in self._toxic_patterns:
                if pattern.search(desc):
                    profile.has_toxic_financing = True
                    break

        # V8: Determine dilution tier
        profile.dilution_tier = self._classify_dilution_tier(profile)

    def _parse_filing(
        self,
        ticker: str,
        filing: Dict
    ) -> Optional[DilutionEvent]:
        """Parse a single SEC filing into a DilutionEvent."""
        form_type = filing.get("form_type", "")
        description = filing.get("description", "")

        # Determine dilution type
        event_type = None
        for dtype, patterns in self._filing_patterns.items():
            for pattern in patterns:
                if pattern.search(form_type) or pattern.search(description):
                    event_type = dtype
                    break
            if event_type:
                break

        if not event_type:
            return None

        # Parse filing date
        filed_date = filing.get("filed_date")
        if isinstance(filed_date, str):
            try:
                filed_date = datetime.fromisoformat(filed_date.replace("Z", "+00:00"))
            except ValueError:
                filed_date = datetime.now()
        elif not isinstance(filed_date, datetime):
            filed_date = datetime.now()

        # Extract dollar amounts if present
        dollar_amount = self._extract_dollar_amount(description)
        shares = self._extract_share_count(description)

        event = DilutionEvent(
            ticker=ticker,
            event_type=event_type,
            filing_date=filed_date,
            dollar_amount=dollar_amount,
            shares_registered=shares,
            accession_number=filing.get("accession_number"),
            filing_url=filing.get("url")
        )

        # Set shelf capacity for S-3 filings
        if event_type == DilutionType.S3_SHELF and dollar_amount:
            event.shelf_capacity_remaining = dollar_amount
            # Shelves typically valid for 3 years
            event.shelf_expiry = filed_date + timedelta(days=365 * 3)

        return event

    def _extract_dollar_amount(self, text: str) -> Optional[float]:
        """Extract dollar amount from text."""
        patterns = [
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)",
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:billion|B)",
            r"([\d,]+(?:\.\d+)?)\s*(?:million|M)\s*(?:dollars|\$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(",", ""))
                if "billion" in text.lower() or "B" in match.group(0):
                    amount *= 1_000_000_000
                else:
                    amount *= 1_000_000
                return amount

        return None

    def _extract_share_count(self, text: str) -> Optional[int]:
        """Extract share count from text."""
        patterns = [
            r"([\d,]+)\s*shares",
            r"([\d,]+(?:\.\d+)?)\s*(?:million|M)\s*shares",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count = float(match.group(1).replace(",", ""))
                if "million" in text.lower() or "M" in match.group(0):
                    count *= 1_000_000
                return int(count)

        return None

    def _classify_dilution_tier(self, profile: DilutionProfile) -> DilutionTier:
        """
        V8: Classify dilution into 4-tier system.

        Tier 1 - ACTIVE_OFFERING: Confirmed offering announced, pricing within days
        Tier 2 - SHELF_RECENT: S-3 filed <30 days ago, intent unclear
        Tier 3 - SHELF_DORMANT: S-3 filed >30 days ago, no follow-up activity
        Tier 4 - CAPACITY_ONLY: Old shelf (>1yr) or no meaningful capacity
        """
        # Toxic financing = always treat as active
        if profile.has_toxic_financing:
            return DilutionTier.ACTIVE_OFFERING

        # Active offering (prospectus supplement, direct offering within 7 days)
        if profile.has_active_offering:
            return DilutionTier.ACTIVE_OFFERING

        # Active ATM program
        if profile.has_active_atm and profile.active_atm_capacity > 5_000_000:
            return DilutionTier.ACTIVE_OFFERING

        # Recent S-3 shelf (<30 days)
        if profile.has_recent_s3:
            return DilutionTier.SHELF_RECENT

        # Check for dormant shelfs (S-3 filed 30-365 days ago)
        has_dormant_shelf = False
        for event in profile.events:
            if event.event_type == DilutionType.S3_SHELF:
                days = event.days_since_filing()
                if 30 <= days <= 365:
                    has_dormant_shelf = True
                    break

        if has_dormant_shelf:
            return DilutionTier.SHELF_DORMANT

        # Old shelf (>1 year) or minimal capacity
        if profile.total_shelf_capacity > 0:
            return DilutionTier.CAPACITY_ONLY

        return DilutionTier.NONE

    def _calculate_risk(self, profile: DilutionProfile) -> None:
        """
        Calculate overall risk level and score.

        V8 FIX (P4): Risk calculation respects dilution tier.
        S-3 shelfs contribute less to risk score than active offerings.
        """
        score = 0.0

        # Critical flags (truly dangerous)
        if profile.has_toxic_financing:
            score += 50
        if profile.has_active_offering:
            score += 40

        # High risk factors
        if profile.has_active_atm:
            score += 25
            if profile.active_atm_capacity > 0:
                score += min(15, profile.active_atm_capacity / 10_000_000)

        # V8 FIX: S-3 shelf score depends on tier, not binary
        if profile.dilution_tier == DilutionTier.SHELF_RECENT:
            score += 15  # V8: Reduced from 20 (recent but not active)
        elif profile.dilution_tier == DilutionTier.SHELF_DORMANT:
            score += 5   # V8: Dormant shelf = minimal risk
        elif profile.dilution_tier == DilutionTier.CAPACITY_ONLY:
            score += 2   # V8: Old shelf = negligible

        # Medium risk factors
        if profile.total_shelf_capacity > 50_000_000:
            score += 10  # V8: Reduced from 15
        elif profile.total_shelf_capacity > 0:
            score += 5   # V8: Reduced from 10

        # Event-based scoring
        for event in profile.events:
            if event.is_stale(365):
                continue

            days = event.days_since_filing()
            recency_multiplier = max(0.1, 1.0 - (days / 180))

            # V8: Different base scores for active vs shelf events
            event_scores = {
                DilutionType.PROSPECTUS_SUPPLEMENT: 30,  # Active = high score
                DilutionType.DIRECT_OFFERING: 25,
                DilutionType.PIPE_DEAL: 20,
                DilutionType.ATM_PROGRAM: 15,
                DilutionType.S3_SHELF: 5,               # V8: Reduced from 10 (shelf ≠ active)
                DilutionType.WARRANT_EXERCISE: 15,
                DilutionType.CONVERTIBLE_NOTE: 20,
            }

            base_score = event_scores.get(event.event_type, 5)
            score += base_score * recency_multiplier

        # Cap at 100
        profile.risk_score = min(100.0, score)

        # Determine risk level
        # V8: CRITICAL only for truly dangerous situations
        if profile.has_toxic_financing or profile.has_active_offering:
            profile.risk_level = DilutionRisk.CRITICAL
        elif profile.risk_score >= 70:
            profile.risk_level = DilutionRisk.CRITICAL
        elif profile.risk_score >= 45:
            profile.risk_level = DilutionRisk.HIGH
        elif profile.risk_score >= 25:
            profile.risk_level = DilutionRisk.MEDIUM
        elif profile.risk_score >= 10:
            profile.risk_level = DilutionRisk.LOW
        else:
            profile.risk_level = DilutionRisk.NONE

        logger.debug(
            f"{profile.ticker} dilution: tier={profile.dilution_tier.value}, "
            f"risk={profile.risk_level.value}, score={profile.risk_score:.0f}, "
            f"multiplier={profile.get_position_multiplier():.2f}"
        )

    async def analyze_batch(
        self,
        tickers: List[str],
        filings_by_ticker: Optional[Dict[str, List[Dict]]] = None
    ) -> Dict[str, DilutionProfile]:
        """Analyze multiple tickers concurrently."""
        filings_by_ticker = filings_by_ticker or {}

        tasks = [
            self.analyze(ticker, filings_by_ticker.get(ticker))
            for ticker in tickers
        ]

        profiles = await asyncio.gather(*tasks)

        return {
            ticker: profile
            for ticker, profile in zip(tickers, profiles)
        }

    def flag_toxic(self, ticker: str) -> None:
        """Manually flag a ticker as having toxic financing."""
        self._toxic_tickers.add(ticker.upper())
        # Invalidate cache
        if ticker.upper() in self._profiles:
            del self._profiles[ticker.upper()]

    def unflag_toxic(self, ticker: str) -> None:
        """Remove toxic flag from a ticker."""
        self._toxic_tickers.discard(ticker.upper())
        if ticker.upper() in self._profiles:
            del self._profiles[ticker.upper()]

    def get_cached_profile(self, ticker: str) -> Optional[DilutionProfile]:
        """Get cached profile without refresh."""
        return self._profiles.get(ticker.upper())

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a ticker or all tickers."""
        if ticker:
            self._profiles.pop(ticker.upper(), None)
        else:
            self._profiles.clear()

    def get_high_risk_tickers(self) -> List[str]:
        """Get list of tickers with HIGH or CRITICAL risk."""
        return [
            ticker for ticker, profile in self._profiles.items()
            if profile.risk_level in [DilutionRisk.HIGH, DilutionRisk.CRITICAL]
        ]


# Singleton instance
_detector: Optional[DilutionDetector] = None


def get_dilution_detector() -> DilutionDetector:
    """Get singleton DilutionDetector instance."""
    global _detector
    if _detector is None:
        _detector = DilutionDetector()
    return _detector
