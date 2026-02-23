"""
Unified Risk Guard V8 - Central Risk Orchestrator

V8 FIXES (from REVIEW):
- P3 CRITICAL: Fixed over-blocking of top gainers small-caps
  - Added momentum_override: if catalyst + price >5% + volume z-score >2.0, reduce penalty 50%
  - Minimum position floor: 0.10 for penny, 0.25 for standard (never 0.0 unless truly toxic)
  - Block ONLY on: confirmed delisting, active halt, toxic financing, manual block
- P16: Replaced catastrophic multiplicative stacking with MIN-based approach
  - Old: 0.5 × 0.5 × 0.25 = 0.0625 (94% reduction from 3 medium risks)
  - New: min(0.5, 0.5, 0.25) = 0.25 (worst single factor, not product)

Combines all risk detection modules into a single interface:
- Dilution Detector: SEC filings, offerings, toxic financing
- Compliance Checker: Exchange deficiencies, delisting risk
- Halt Monitor: Trading halts, LULD tracking, halt prediction
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging

from .dilution_detector import (
    DilutionDetector,
    DilutionProfile,
    DilutionRisk,
    get_dilution_detector
)
from .compliance_checker import (
    ComplianceChecker,
    ComplianceProfile,
    ComplianceRisk,
    get_compliance_checker
)
from .halt_monitor import (
    HaltMonitor,
    HaltProfile,
    HaltPrediction,
    HaltRisk,
    HaltCode,
    get_halt_monitor
)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Unified risk levels."""
    CRITICAL = "CRITICAL"     # Block all trades
    HIGH = "HIGH"             # Reduce position significantly
    ELEVATED = "ELEVATED"     # Reduce position moderately
    MODERATE = "MODERATE"     # Minor adjustment
    LOW = "LOW"               # Normal operations


class RiskCategory(Enum):
    """Categories of risk."""
    DILUTION = "DILUTION"
    COMPLIANCE = "COMPLIANCE"
    HALT = "HALT"
    COMBINED = "COMBINED"


class TradeAction(Enum):
    """Recommended trade actions."""
    ALLOW = "ALLOW"               # Proceed normally
    REDUCE = "REDUCE"             # Reduce position size
    REDUCE_SIGNIFICANT = "REDUCE_SIGNIFICANT"  # Major reduction
    DELAY = "DELAY"               # Wait before trading
    BLOCK = "BLOCK"               # Do not trade
    ALERT_ONLY = "ALERT_ONLY"     # Signal ok, don't execute


@dataclass
class RiskFlag:
    """Individual risk flag."""
    category: RiskCategory
    level: RiskLevel
    code: str
    message: str
    blocking: bool = False
    position_multiplier: float = 1.0

    def __str__(self) -> str:
        return f"[{self.level.value}] {self.category.value}: {self.message}"


@dataclass
class RiskAssessment:
    """Complete risk assessment for a ticker."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Overall assessment
    overall_level: RiskLevel = RiskLevel.LOW
    overall_score: float = 0.0  # 0-100

    # Recommended action
    action: TradeAction = TradeAction.ALLOW
    position_multiplier: float = 1.0  # 0.0 to 1.0

    # Individual risk flags
    flags: List[RiskFlag] = field(default_factory=list)

    # Component profiles
    dilution_profile: Optional[DilutionProfile] = None
    compliance_profile: Optional[ComplianceProfile] = None
    halt_profile: Optional[HaltProfile] = None

    # Block info
    is_blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)

    # Summary
    summary: str = ""

    def get_blocking_flags(self) -> List[RiskFlag]:
        """Get flags that block trading."""
        return [f for f in self.flags if f.blocking]

    def get_critical_flags(self) -> List[RiskFlag]:
        """Get critical level flags."""
        return [f for f in self.flags if f.level == RiskLevel.CRITICAL]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "overall_level": self.overall_level.value,
            "overall_score": self.overall_score,
            "action": self.action.value,
            "position_multiplier": self.position_multiplier,
            "is_blocked": self.is_blocked,
            "block_reasons": self.block_reasons,
            "flags": [
                {
                    "category": f.category.value,
                    "level": f.level.value,
                    "code": f.code,
                    "message": f.message,
                    "blocking": f.blocking
                }
                for f in self.flags
            ],
            "summary": self.summary
        }


@dataclass
class MomentumContext:
    """V8: Market context for momentum override decisions."""
    price_change_pct: float = 0.0    # Current session price change %
    volume_zscore: float = 0.0       # Volume z-score vs 20-day baseline
    has_catalyst: bool = False        # Active catalyst detected
    monster_score: float = 0.0        # Current monster score


@dataclass
class GuardConfig:
    """Configuration for UnifiedGuard."""
    # Enable/disable components
    enable_dilution: bool = True
    enable_compliance: bool = True
    enable_halt: bool = True

    # Thresholds for blocking
    # V8: Only block on truly dangerous conditions (not potential risks)
    block_on_critical: bool = True
    block_on_active_offering: bool = True
    block_on_delisting_risk: bool = True
    block_on_halt: bool = True
    block_on_halt_imminent: bool = True

    # Position sizing
    # V8 FIX (P3): Increased minimum floors to prevent total blocking of small-caps
    min_position_multiplier: float = 0.10       # Minimum 10% (penny stocks)
    min_position_multiplier_standard: float = 0.25  # V8: Minimum 25% (standard stocks >$1)

    # V8 FIX (P16): Use MIN instead of MULTIPLY for combining risk multipliers
    # Old: apply_combined_multipliers = True → 0.5 × 0.5 × 0.25 = 0.0625
    # New: apply_combined_multipliers = False → min(0.5, 0.5, 0.25) = 0.25
    apply_combined_multipliers: bool = False  # V8: DEFAULT CHANGED to False (MIN mode)

    # V8: Momentum override - reduce penalty when stock shows strong momentum
    enable_momentum_override: bool = True
    momentum_override_min_price_change: float = 0.05   # +5% price change
    momentum_override_min_volume_zscore: float = 2.0    # Volume 2+ std devs above mean
    momentum_override_penalty_reduction: float = 0.50   # Reduce penalty by 50%

    # Cache
    cache_ttl_seconds: int = 300  # 5 minutes

    # Monitoring
    alert_on_high_risk: bool = True
    log_assessments: bool = True


class UnifiedGuard:
    """
    Central risk guard that combines all risk modules.

    Usage:
        guard = UnifiedGuard()

        # Full assessment
        assessment = await guard.assess(ticker)
        if assessment.is_blocked:
            reject_trade(assessment.block_reasons)
        else:
            adjusted_size = base_size * assessment.position_multiplier

        # Quick check
        if guard.is_blocked(ticker):
            skip_ticker()

        # Monitor halt
        guard.on_halt(ticker, HaltCode.LUDP)
    """

    def __init__(self, config: Optional[GuardConfig] = None):
        self.config = config or GuardConfig()

        # Component modules
        self._dilution: DilutionDetector = get_dilution_detector()
        self._compliance: ComplianceChecker = get_compliance_checker()
        self._halt: HaltMonitor = get_halt_monitor()

        # Assessment cache
        self._cache: Dict[str, RiskAssessment] = {}
        self._cache_times: Dict[str, datetime] = {}

        # Watchlist (tickers requiring extra scrutiny)
        self._watchlist: Set[str] = set()

        # Block list (manual blocks)
        self._blocklist: Set[str] = set()

        # Listeners for risk events
        self._listeners: List[callable] = []

    async def assess(
        self,
        ticker: str,
        current_price: Optional[float] = None,
        volatility: Optional[float] = None,
        normal_volatility: Optional[float] = None,
        sec_filings: Optional[List[Dict]] = None,
        force_refresh: bool = False,
        momentum_context: Optional[MomentumContext] = None
    ) -> RiskAssessment:
        """
        Perform complete risk assessment for a ticker.

        Args:
            ticker: Stock ticker
            current_price: Current price for halt prediction
            volatility: Current volatility
            normal_volatility: Baseline volatility
            sec_filings: SEC filings for dilution/compliance analysis
            force_refresh: Bypass cache
            momentum_context: V8 - Market momentum context for override decisions

        Returns:
            Complete RiskAssessment
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._cache:
            cache_time = self._cache_times.get(ticker)
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < self.config.cache_ttl_seconds:
                    return self._cache[ticker]

        # Create assessment
        assessment = RiskAssessment(ticker=ticker)

        # Check manual block list
        if ticker in self._blocklist:
            assessment.is_blocked = True
            assessment.block_reasons.append("MANUAL_BLOCK")
            assessment.action = TradeAction.BLOCK
            assessment.position_multiplier = 0.0
            assessment.overall_level = RiskLevel.CRITICAL
            assessment.flags.append(RiskFlag(
                category=RiskCategory.COMBINED,
                level=RiskLevel.CRITICAL,
                code="MANUAL_BLOCK",
                message="Ticker is manually blocked",
                blocking=True,
                position_multiplier=0.0
            ))
            self._cache_assessment(assessment)
            return assessment

        # Run component assessments concurrently
        tasks = []

        if self.config.enable_dilution:
            tasks.append(self._assess_dilution(ticker, sec_filings))
        else:
            tasks.append(asyncio.coroutine(lambda: None)())

        if self.config.enable_compliance:
            tasks.append(self._assess_compliance(ticker, sec_filings))
        else:
            tasks.append(asyncio.coroutine(lambda: None)())

        if self.config.enable_halt:
            tasks.append(self._assess_halt(
                ticker, current_price, volatility, normal_volatility
            ))
        else:
            tasks.append(asyncio.coroutine(lambda: None)())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process dilution results
        if self.config.enable_dilution and not isinstance(results[0], Exception):
            dilution_profile, dilution_flags = results[0] or (None, [])
            if dilution_profile:
                assessment.dilution_profile = dilution_profile
                assessment.flags.extend(dilution_flags)

        # Process compliance results
        if self.config.enable_compliance and not isinstance(results[1], Exception):
            compliance_profile, compliance_flags = results[1] or (None, [])
            if compliance_profile:
                assessment.compliance_profile = compliance_profile
                assessment.flags.extend(compliance_flags)

        # Process halt results
        if self.config.enable_halt and not isinstance(results[2], Exception):
            halt_profile, halt_flags = results[2] or (None, [])
            if halt_profile:
                assessment.halt_profile = halt_profile
                assessment.flags.extend(halt_flags)

        # Calculate overall assessment (V8: with momentum context)
        self._calculate_overall(assessment, current_price, momentum_context)

        # Cache result
        self._cache_assessment(assessment)

        # Notify listeners if high risk
        if self.config.alert_on_high_risk:
            if assessment.overall_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                self._notify_risk(assessment)

        # Log if enabled
        if self.config.log_assessments:
            logger.info(
                f"Risk assessment for {ticker}: {assessment.overall_level.value} "
                f"(score={assessment.overall_score:.1f}, action={assessment.action.value})"
            )

        return assessment

    async def _assess_dilution(
        self,
        ticker: str,
        sec_filings: Optional[List[Dict]]
    ) -> Tuple[Optional[DilutionProfile], List[RiskFlag]]:
        """Assess dilution risk."""
        flags = []

        try:
            profile = await self._dilution.analyze(ticker, sec_filings)

            if profile.risk_level == DilutionRisk.CRITICAL:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.CRITICAL,
                    code="DILUTION_CRITICAL",
                    message=f"Critical dilution risk: {profile.get_block_reason() or 'high score'}",
                    blocking=self.config.block_on_active_offering,
                    position_multiplier=0.0
                ))

            elif profile.risk_level == DilutionRisk.HIGH:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.HIGH,
                    code="DILUTION_HIGH",
                    message=f"High dilution risk (score={profile.risk_score:.0f})",
                    blocking=False,
                    position_multiplier=0.25
                ))

            elif profile.risk_level == DilutionRisk.MEDIUM:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.ELEVATED,
                    code="DILUTION_MEDIUM",
                    message=f"Elevated dilution risk (score={profile.risk_score:.0f})",
                    blocking=False,
                    position_multiplier=0.50
                ))

            # Specific flags
            if profile.has_active_offering:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.CRITICAL,
                    code="ACTIVE_OFFERING",
                    message="Active stock offering in progress",
                    blocking=self.config.block_on_active_offering,
                    position_multiplier=0.0
                ))

            if profile.has_toxic_financing:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.CRITICAL,
                    code="TOXIC_FINANCING",
                    message="Toxic financing detected (variable rate converts)",
                    blocking=True,
                    position_multiplier=0.0
                ))

            if profile.has_active_atm:
                flags.append(RiskFlag(
                    category=RiskCategory.DILUTION,
                    level=RiskLevel.HIGH,
                    code="ACTIVE_ATM",
                    message=f"Active ATM program (${profile.active_atm_capacity/1e6:.1f}M capacity)",
                    blocking=False,
                    position_multiplier=0.25
                ))

            return profile, flags

        except Exception as e:
            logger.error(f"Error assessing dilution for {ticker}: {e}")
            return None, []

    async def _assess_compliance(
        self,
        ticker: str,
        sec_filings: Optional[List[Dict]]
    ) -> Tuple[Optional[ComplianceProfile], List[RiskFlag]]:
        """Assess compliance risk."""
        flags = []

        try:
            profile = await self._compliance.analyze(
                ticker,
                sec_filings=sec_filings
            )

            if profile.risk_level == ComplianceRisk.CRITICAL:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.CRITICAL,
                    code="COMPLIANCE_CRITICAL",
                    message=f"Critical compliance risk: {profile.get_block_reason() or 'delisting risk'}",
                    blocking=self.config.block_on_delisting_risk,
                    position_multiplier=0.0
                ))

            elif profile.risk_level == ComplianceRisk.HIGH:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.HIGH,
                    code="COMPLIANCE_HIGH",
                    message=f"High compliance risk (score={profile.risk_score:.0f})",
                    blocking=False,
                    position_multiplier=0.25
                ))

            elif profile.risk_level == ComplianceRisk.MEDIUM:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.ELEVATED,
                    code="COMPLIANCE_MEDIUM",
                    message=f"Elevated compliance risk (score={profile.risk_score:.0f})",
                    blocking=False,
                    position_multiplier=0.50
                ))

            # Specific flags
            if profile.has_delisting_risk:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.CRITICAL,
                    code="DELISTING_RISK",
                    message="Delisting determination pending",
                    blocking=self.config.block_on_delisting_risk,
                    position_multiplier=0.0
                ))

            if profile.has_pending_reverse_split:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.HIGH,
                    code="REVERSE_SPLIT_PENDING",
                    message="Reverse stock split pending/announced",
                    blocking=False,
                    position_multiplier=0.25
                ))

            if profile.days_below_dollar >= 30:
                flags.append(RiskFlag(
                    category=RiskCategory.COMPLIANCE,
                    level=RiskLevel.ELEVATED,
                    code="BID_PRICE_DEFICIENCY",
                    message=f"Below $1 for {profile.days_below_dollar} consecutive days",
                    blocking=False,
                    position_multiplier=0.50
                ))

            return profile, flags

        except Exception as e:
            logger.error(f"Error assessing compliance for {ticker}: {e}")
            return None, []

    async def _assess_halt(
        self,
        ticker: str,
        current_price: Optional[float],
        volatility: Optional[float],
        normal_volatility: Optional[float]
    ) -> Tuple[Optional[HaltProfile], List[RiskFlag]]:
        """Assess halt risk."""
        flags = []

        try:
            profile = self._halt.get_profile(ticker)

            # Check if currently halted
            if profile.is_halted:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.CRITICAL,
                    code="CURRENTLY_HALTED",
                    message=f"Trading halted: {profile.current_halt.halt_code.value if profile.current_halt else 'unknown'}",
                    blocking=self.config.block_on_halt,
                    position_multiplier=0.0
                ))
                return profile, flags

            # Get halt prediction
            prediction = self._halt.predict_halt(
                ticker,
                current_price=current_price,
                volatility=volatility,
                normal_volatility=normal_volatility
            )
            profile.prediction = prediction

            if prediction.risk_level == HaltRisk.IMMINENT:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.CRITICAL,
                    code="HALT_IMMINENT",
                    message=f"Halt imminent ({prediction.probability:.0f}% probability): {', '.join(prediction.factors[:2])}",
                    blocking=self.config.block_on_halt_imminent,
                    position_multiplier=0.0
                ))

            elif prediction.risk_level == HaltRisk.HIGH:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.HIGH,
                    code="HALT_HIGH_RISK",
                    message=f"High halt risk ({prediction.probability:.0f}%): {', '.join(prediction.factors[:2])}",
                    blocking=False,
                    position_multiplier=0.25
                ))

            elif prediction.risk_level == HaltRisk.ELEVATED:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.ELEVATED,
                    code="HALT_ELEVATED_RISK",
                    message=f"Elevated halt risk ({prediction.probability:.0f}%)",
                    blocking=False,
                    position_multiplier=0.50
                ))

            # LULD specific flags
            if prediction.near_luld_limit:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.HIGH,
                    code="NEAR_LULD_LIMIT",
                    message="Price near LULD band limit",
                    blocking=False,
                    position_multiplier=0.25
                ))

            # Frequent halter flag
            if profile.frequent_halter:
                flags.append(RiskFlag(
                    category=RiskCategory.HALT,
                    level=RiskLevel.MODERATE,
                    code="FREQUENT_HALTER",
                    message=f"Frequent halter ({profile.halts_this_week} halts this week)",
                    blocking=False,
                    position_multiplier=0.75
                ))

            return profile, flags

        except Exception as e:
            logger.error(f"Error assessing halt risk for {ticker}: {e}")
            return None, []

    def _calculate_overall(
        self,
        assessment: RiskAssessment,
        current_price: Optional[float] = None,
        momentum_context: Optional[MomentumContext] = None
    ) -> None:
        """
        Calculate overall risk level and action.

        V8 CHANGES:
        - P16: Default to MIN mode (worst single factor) instead of multiplicative
        - P3: Momentum override reduces penalty for stocks with strong breakout signals
        - P3: Higher minimum floor for standard stocks ($1+)
        - P3: Only truly dangerous conditions cause total block (0.0)
        """
        if not assessment.flags:
            assessment.overall_level = RiskLevel.LOW
            assessment.action = TradeAction.ALLOW
            assessment.position_multiplier = 1.0
            assessment.summary = "No risk flags detected"
            return

        # Find highest risk level
        level_priority = {
            RiskLevel.CRITICAL: 5,
            RiskLevel.HIGH: 4,
            RiskLevel.ELEVATED: 3,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 1,
        }

        max_level = RiskLevel.LOW
        for flag in assessment.flags:
            if level_priority[flag.level] > level_priority[max_level]:
                max_level = flag.level

        assessment.overall_level = max_level

        # Calculate overall score (weighted by level)
        score = 0.0
        for flag in assessment.flags:
            level_scores = {
                RiskLevel.CRITICAL: 40,
                RiskLevel.HIGH: 25,
                RiskLevel.ELEVATED: 15,
                RiskLevel.MODERATE: 8,
                RiskLevel.LOW: 3,
            }
            score += level_scores.get(flag.level, 5)

        assessment.overall_score = min(100.0, score)

        # V8 FIX (P3): Only block on truly dangerous conditions
        # Filter blocking flags to only include confirmed dangers
        HARD_BLOCK_CODES = {
            "CURRENTLY_HALTED", "TOXIC_FINANCING", "DELISTING_RISK",
            "MANUAL_BLOCK", "ACTIVE_OFFERING"
        }
        blocking_flags = [
            f for f in assessment.get_blocking_flags()
            if f.code in HARD_BLOCK_CODES
        ]

        if blocking_flags:
            assessment.is_blocked = True
            assessment.block_reasons = [f.code for f in blocking_flags]
            assessment.action = TradeAction.BLOCK
            assessment.position_multiplier = 0.0
        else:
            # V8 FIX (P16): Use MIN mode by default instead of multiplicative
            if self.config.apply_combined_multipliers:
                # Legacy mode: Multiply all multipliers (can be catastrophic)
                multiplier = 1.0
                for flag in assessment.flags:
                    multiplier *= flag.position_multiplier
            else:
                # V8 DEFAULT: Use worst single factor (MIN mode)
                # This prevents 0.5 × 0.5 × 0.25 = 0.0625 situations
                multipliers = [f.position_multiplier for f in assessment.flags if f.position_multiplier < 1.0]
                multiplier = min(multipliers) if multipliers else 1.0

            # V8 FIX (P3): Momentum override for stocks in active breakout
            if (
                self.config.enable_momentum_override
                and momentum_context is not None
                and multiplier < 1.0
            ):
                has_strong_momentum = (
                    momentum_context.price_change_pct >= self.config.momentum_override_min_price_change
                    and momentum_context.volume_zscore >= self.config.momentum_override_min_volume_zscore
                )
                has_catalyst_momentum = (
                    momentum_context.has_catalyst
                    and momentum_context.price_change_pct >= 0.03  # 3% with catalyst
                )

                if has_strong_momentum or has_catalyst_momentum:
                    # Reduce the penalty by configured amount (default 50%)
                    penalty = 1.0 - multiplier
                    reduced_penalty = penalty * (1.0 - self.config.momentum_override_penalty_reduction)
                    old_multiplier = multiplier
                    multiplier = 1.0 - reduced_penalty

                    logger.info(
                        f"{assessment.ticker} MOMENTUM OVERRIDE: "
                        f"multiplier {old_multiplier:.2f} → {multiplier:.2f} "
                        f"(price {momentum_context.price_change_pct:+.1%}, "
                        f"vol z={momentum_context.volume_zscore:.1f}, "
                        f"catalyst={momentum_context.has_catalyst})"
                    )

            # V8 FIX (P3): Apply appropriate minimum floor based on price
            is_penny = current_price is not None and current_price < 1.0
            min_floor = (
                self.config.min_position_multiplier if is_penny
                else self.config.min_position_multiplier_standard
            )
            multiplier = max(multiplier, min_floor)
            assessment.position_multiplier = multiplier

            # Determine action
            if multiplier <= 0.0:
                assessment.action = TradeAction.BLOCK
                assessment.is_blocked = True
            elif multiplier <= 0.25:
                assessment.action = TradeAction.REDUCE_SIGNIFICANT
            elif multiplier <= 0.75:
                assessment.action = TradeAction.REDUCE
            else:
                assessment.action = TradeAction.ALLOW

        # Generate summary
        categories = set(f.category.value for f in assessment.flags)
        momentum_note = ""
        if momentum_context and momentum_context.price_change_pct >= 0.03:
            momentum_note = f" [MOMENTUM: {momentum_context.price_change_pct:+.1%}]"

        assessment.summary = (
            f"{assessment.overall_level.value} risk ({assessment.overall_score:.0f} score) - "
            f"Categories: {', '.join(categories)} - "
            f"Action: {assessment.action.value} "
            f"(position x{assessment.position_multiplier:.2f}){momentum_note}"
        )

    def _cache_assessment(self, assessment: RiskAssessment) -> None:
        """Cache an assessment."""
        self._cache[assessment.ticker] = assessment
        self._cache_times[assessment.ticker] = datetime.now()

    def is_blocked(self, ticker: str) -> bool:
        """Quick check if ticker is blocked."""
        ticker = ticker.upper()

        # Check manual block list
        if ticker in self._blocklist:
            return True

        # Check cached assessment
        if ticker in self._cache:
            return self._cache[ticker].is_blocked

        # Check if halted
        if self._halt.is_halted(ticker):
            return True

        return False

    def quick_check(self, ticker: str) -> Tuple[bool, Optional[str]]:
        """
        Quick check if ticker can be traded.

        Returns:
            Tuple of (can_trade, block_reason)
        """
        ticker = ticker.upper()

        # Check manual block
        if ticker in self._blocklist:
            return False, "MANUAL_BLOCK"

        # Check halt
        if self._halt.is_halted(ticker):
            return False, "HALTED"

        # Check cached critical assessment
        if ticker in self._cache:
            assessment = self._cache[ticker]
            if assessment.is_blocked:
                return False, assessment.block_reasons[0] if assessment.block_reasons else "BLOCKED"

        return True, None

    def block_ticker(self, ticker: str, reason: str = "") -> None:
        """Manually block a ticker."""
        ticker = ticker.upper()
        self._blocklist.add(ticker)
        # Invalidate cache
        self._cache.pop(ticker, None)
        logger.info(f"Manually blocked ticker: {ticker} - {reason}")

    def unblock_ticker(self, ticker: str) -> None:
        """Remove manual block from ticker."""
        ticker = ticker.upper()
        self._blocklist.discard(ticker)
        self._cache.pop(ticker, None)
        logger.info(f"Unblocked ticker: {ticker}")

    def add_to_watchlist(self, ticker: str) -> None:
        """Add ticker to watchlist for extra scrutiny."""
        self._watchlist.add(ticker.upper())

    def remove_from_watchlist(self, ticker: str) -> None:
        """Remove ticker from watchlist."""
        self._watchlist.discard(ticker.upper())

    def on_halt(
        self,
        ticker: str,
        halt_code: HaltCode,
        halt_time: Optional[datetime] = None,
        halt_price: Optional[float] = None
    ) -> None:
        """Record a halt event."""
        self._halt.record_halt(ticker, halt_code, halt_time, halt_price)
        # Invalidate cache
        self._cache.pop(ticker.upper(), None)

    def on_resume(
        self,
        ticker: str,
        resume_time: Optional[datetime] = None,
        resume_price: Optional[float] = None
    ) -> None:
        """Record a halt resume."""
        self._halt.record_resume(ticker, resume_time, resume_price)
        # Invalidate cache
        self._cache.pop(ticker.upper(), None)

    def flag_toxic(self, ticker: str) -> None:
        """Flag ticker as having toxic financing."""
        self._dilution.flag_toxic(ticker)
        self._cache.pop(ticker.upper(), None)

    def get_blocked_tickers(self) -> Set[str]:
        """Get all blocked tickers."""
        blocked = self._blocklist.copy()
        blocked.update(self._halt.get_halted_tickers())
        return blocked

    def get_high_risk_tickers(self) -> List[str]:
        """Get tickers with HIGH or CRITICAL risk from cache."""
        result = []
        for ticker, assessment in self._cache.items():
            if assessment.overall_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                result.append(ticker)
        return result

    def add_risk_listener(self, callback: callable) -> None:
        """Add callback for high-risk events."""
        self._listeners.append(callback)

    def _notify_risk(self, assessment: RiskAssessment) -> None:
        """Notify listeners of high-risk assessment."""
        for callback in self._listeners:
            try:
                callback(assessment)
            except Exception as e:
                logger.error(f"Error in risk listener: {e}")

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear assessment cache."""
        if ticker:
            self._cache.pop(ticker.upper(), None)
            self._cache_times.pop(ticker.upper(), None)
        else:
            self._cache.clear()
            self._cache_times.clear()

    async def assess_batch(
        self,
        tickers: List[str],
        prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, RiskAssessment]:
        """Assess multiple tickers concurrently."""
        prices = prices or {}

        tasks = [
            self.assess(ticker, current_price=prices.get(ticker))
            for ticker in tickers
        ]

        assessments = await asyncio.gather(*tasks, return_exceptions=True)

        result = {}
        for ticker, assessment in zip(tickers, assessments):
            if isinstance(assessment, Exception):
                logger.error(f"Error assessing {ticker}: {assessment}")
                result[ticker] = RiskAssessment(
                    ticker=ticker,
                    overall_level=RiskLevel.MODERATE,
                    summary=f"Assessment error: {assessment}"
                )
            else:
                result[ticker] = assessment

        return result

    def get_stats(self) -> Dict:
        """Get guard statistics."""
        return {
            "cached_assessments": len(self._cache),
            "blocked_tickers": len(self._blocklist),
            "halted_tickers": len(self._halt.get_halted_tickers()),
            "watchlist_size": len(self._watchlist),
            "high_risk_count": len(self.get_high_risk_tickers()),
        }


# Singleton instance
_guard: Optional[UnifiedGuard] = None


def get_unified_guard(config: Optional[GuardConfig] = None) -> UnifiedGuard:
    """Get singleton UnifiedGuard instance."""
    global _guard
    if _guard is None:
        _guard = UnifiedGuard(config)
    return _guard
