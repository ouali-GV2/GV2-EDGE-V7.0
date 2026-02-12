"""
SIGNAL TYPES V7.0
=================

Types unifiés pour la séparation Détection / Exécution.

Architecture V7:
1. DETECTION (illimitée) → SignalType
2. ORDER COMPUTATION (toujours calculé) → ProposedOrder
3. EXECUTION GATE (limites) → ExecutionStatus

Principe fondamental:
- La détection ne s'arrête JAMAIS
- Les limites n'affectent QUE l'exécution
- Le trader voit TOUS les signaux
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


# ============================================================================
# LAYER 1: DETECTION (Signal Types)
# ============================================================================

class SignalType(Enum):
    """
    Signal de détection (JAMAIS bloqué par les limites)

    Ces signaux sont produits par le moteur de détection
    indépendamment de toute contrainte d'exécution.
    """
    # Primary signals
    BUY_STRONG = "BUY_STRONG"      # High confidence, strong catalyst
    BUY = "BUY"                    # Standard buy signal

    # Secondary signals
    WATCH = "WATCH"                # Worth monitoring, not actionable yet
    EARLY_SIGNAL = "EARLY_SIGNAL"  # Pre-spike detection, too early to act

    # Neutral
    NO_SIGNAL = "NO_SIGNAL"        # No actionable signal detected

    def is_actionable(self) -> bool:
        """Check if signal suggests action"""
        return self in [SignalType.BUY_STRONG, SignalType.BUY]

    def get_priority(self) -> int:
        """Get signal priority (higher = more important)"""
        priorities = {
            SignalType.BUY_STRONG: 100,
            SignalType.BUY: 80,
            SignalType.EARLY_SIGNAL: 60,
            SignalType.WATCH: 40,
            SignalType.NO_SIGNAL: 0,
        }
        return priorities.get(self, 0)


class PreSpikeState(Enum):
    """Pre-Spike Radar state"""
    DORMANT = "DORMANT"        # No activity
    CHARGING = "CHARGING"      # Building momentum
    READY = "READY"            # Ready to move
    LAUNCHING = "LAUNCHING"    # Move starting
    EXHAUSTED = "EXHAUSTED"    # Move completed


class PreHaltState(Enum):
    """Pre-Halt Engine state"""
    LOW = "LOW"              # Normal conditions
    MEDIUM = "MEDIUM"        # Elevated risk
    HIGH = "HIGH"            # High halt probability


# ============================================================================
# LAYER 2: ORDER COMPUTATION (Always Calculated)
# ============================================================================

class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class TimingStrategy(Enum):
    """When to execute"""
    IMMEDIATE = "IMMEDIATE"           # Execute now
    ON_BREAKOUT = "ON_BREAKOUT"       # Wait for breakout confirmation
    ON_PULLBACK = "ON_PULLBACK"       # Wait for pullback entry
    ON_OPEN = "ON_OPEN"               # Execute at market open
    VWAP = "VWAP"                     # VWAP execution


@dataclass
class ProposedOrder:
    """
    Ordre théorique (TOUJOURS calculé, même si non exécutable)

    Cet ordre est généré pour CHAQUE signal BUY/BUY_STRONG,
    indépendamment des limites d'exécution.
    """
    # Core order details
    ticker: str
    side: OrderSide
    order_type: OrderType

    # Pricing
    price_target: float              # Target entry price
    price_limit: Optional[float]     # Limit price if applicable

    # Sizing
    size_shares: int                 # Number of shares
    size_usd: float                  # Dollar amount
    size_pct_portfolio: float        # % of portfolio

    # Risk management
    stop_loss: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_targets: List[float] = field(default_factory=list)

    # Metrics
    risk_reward_ratio: float = 0.0
    confidence: float = 0.0          # 0.0 - 1.0

    # Timing
    timing_strategy: TimingStrategy = TimingStrategy.IMMEDIATE
    valid_until: Optional[datetime] = None

    # Rationale
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "price_target": self.price_target,
            "price_limit": self.price_limit,
            "size_shares": self.size_shares,
            "size_usd": self.size_usd,
            "size_pct_portfolio": self.size_pct_portfolio,
            "stop_loss": self.stop_loss,
            "take_profit_targets": self.take_profit_targets,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confidence": self.confidence,
            "timing_strategy": self.timing_strategy.value,
            "rationale": self.rationale
        }


# ============================================================================
# LAYER 3: EXECUTION GATE (Limits Applied Here)
# ============================================================================

class ExecutionStatus(Enum):
    """
    Statut d'exécution (SEULE couche avec limites)

    Le signal de détection reste VISIBLE même si l'exécution est bloquée.
    """
    # Allowed
    EXECUTE_ALLOWED = "EXECUTE_ALLOWED"       # Full execution permitted
    EXECUTE_REDUCED = "EXECUTE_REDUCED"       # Reduced size permitted

    # Blocked
    EXECUTE_BLOCKED = "EXECUTE_BLOCKED"       # Blocked but signal visible

    # Deferred
    EXECUTE_DELAYED = "EXECUTE_DELAYED"       # Queued for later
    EXECUTE_PENDING = "EXECUTE_PENDING"       # Waiting for confirmation

    # Alert only
    ALERT_ONLY = "ALERT_ONLY"                 # Notification only, no execution

    def is_executable(self) -> bool:
        """Check if execution is permitted"""
        return self in [
            ExecutionStatus.EXECUTE_ALLOWED,
            ExecutionStatus.EXECUTE_REDUCED
        ]


class BlockReason(Enum):
    """Reason for execution block"""
    # Trade limits
    DAILY_TRADE_LIMIT = "DAILY_TRADE_LIMIT"           # Max trades/day reached
    CAPITAL_INSUFFICIENT = "CAPITAL_INSUFFICIENT"     # Not enough buying power
    POSITION_LIMIT = "POSITION_LIMIT"                 # Max position size reached

    # Risk guards
    DILUTION_HIGH = "DILUTION_HIGH"
    COMPLIANCE_HIGH = "COMPLIANCE_HIGH"
    DELISTING_HIGH = "DELISTING_HIGH"
    PRE_HALT_HIGH = "PRE_HALT_HIGH"
    PENNY_STOCK_RISK = "PENNY_STOCK_RISK"

    # Market conditions
    MARKET_CLOSED = "MARKET_CLOSED"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"

    # Technical
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    API_ERROR = "API_ERROR"

    # User-defined
    MANUAL_BLOCK = "MANUAL_BLOCK"


@dataclass
class ExecutionDecision:
    """
    Décision d'exécution finale

    Combine le signal de détection avec les contraintes d'exécution.
    Le signal original est TOUJOURS préservé.
    """
    # Execution outcome
    status: ExecutionStatus

    # Size adjustment
    size_multiplier: float = 1.0     # 1.0 = full, 0.5 = half, 0.0 = blocked

    # Block details (if blocked)
    blocked_by: List[BlockReason] = field(default_factory=list)
    block_message: str = ""

    # Metadata
    would_have_executed: bool = False   # True if signal was actionable
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "size_multiplier": self.size_multiplier,
            "blocked_by": [r.value for r in self.blocked_by],
            "block_message": self.block_message,
            "would_have_executed": self.would_have_executed,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# UNIFIED SIGNAL OUTPUT
# ============================================================================

@dataclass
class UnifiedSignal:
    """
    Signal unifié V7 - Combine détection, ordre et exécution

    Principe: L'information ne disparaît JAMAIS

    Même si:
    - Limite de trades atteinte
    - Risque élevé
    - Capital insuffisant

    Le trader voit TOUJOURS:
    - Le signal de détection original
    - L'ordre proposé
    - La raison du blocage (si applicable)
    """
    # Identification
    id: str
    ticker: str
    timestamp: datetime

    # === LAYER 1: DETECTION (always populated) ===
    signal_type: SignalType
    monster_score: float                    # 0.0 - 1.0
    catalyst_type: Optional[str] = None
    catalyst_confidence: float = 0.0

    # Pre-Spike state
    pre_spike_state: PreSpikeState = PreSpikeState.DORMANT

    # Pre-Halt state
    pre_halt_state: PreHaltState = PreHaltState.LOW

    # === CONTEXT SCORES (MRP/EP - informational only, non-blocking) ===
    # Activated only when Market Memory is stable (min_samples reached)
    context_mrp: Optional[float] = None         # 0-100, Missed Recovery Potential
    context_ep: Optional[float] = None          # 0-100, Edge Probability
    context_confidence: Optional[float] = None  # 0-100, data confidence
    context_active: bool = False                # True if MRP/EP are populated

    # === LAYER 2: PROPOSED ORDER (always computed for BUY signals) ===
    proposed_order: Optional[ProposedOrder] = None

    # === LAYER 3: EXECUTION DECISION ===
    execution: Optional[ExecutionDecision] = None

    # === METADATA ===
    # Risk badges (visible to trader)
    badges: List[str] = field(default_factory=list)

    # Context
    market_session: str = "RTH"             # PRE, RTH, POST, CLOSED
    account_mode: str = "LIVE"              # LIVE, PAPER, BACKTEST

    # Original signal preserved
    signal_preserved: bool = True

    def is_actionable(self) -> bool:
        """Check if signal is actionable"""
        return self.signal_type.is_actionable()

    def is_executable(self) -> bool:
        """Check if execution is permitted"""
        if self.execution is None:
            return False
        return self.execution.status.is_executable()

    def get_final_signal(self) -> str:
        """
        Get display signal for trader

        Shows detection signal + execution status
        """
        base = self.signal_type.value

        if self.execution:
            if self.execution.status == ExecutionStatus.EXECUTE_BLOCKED:
                return f"{base} (BLOCKED)"
            elif self.execution.status == ExecutionStatus.EXECUTE_REDUCED:
                pct = int(self.execution.size_multiplier * 100)
                return f"{base} (REDUCED {pct}%)"
            elif self.execution.status == ExecutionStatus.ALERT_ONLY:
                return f"{base} (ALERT)"

        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),

            "detection": {
                "signal_type": self.signal_type.value,
                "monster_score": self.monster_score,
                "catalyst_type": self.catalyst_type,
                "catalyst_confidence": self.catalyst_confidence,
                "pre_spike_state": self.pre_spike_state.value,
                "pre_halt_state": self.pre_halt_state.value
            },

            "proposed_order": self.proposed_order.to_dict() if self.proposed_order else None,

            "execution": self.execution.to_dict() if self.execution else None,

            "badges": self.badges,
            "market_session": self.market_session,
            "account_mode": self.account_mode,
            "signal_preserved": self.signal_preserved,

            # Context scores (MRP/EP) - only populated when Market Memory stable
            "context": {
                "active": self.context_active,
                "mrp": self.context_mrp,
                "ep": self.context_ep,
                "confidence": self.context_confidence
            } if self.context_active else None,

            "display": {
                "final_signal": self.get_final_signal(),
                "is_actionable": self.is_actionable(),
                "is_executable": self.is_executable()
            }
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Layer 1: Detection
    "SignalType",
    "PreSpikeState",
    "PreHaltState",

    # Layer 2: Order
    "OrderSide",
    "OrderType",
    "TimingStrategy",
    "ProposedOrder",

    # Layer 3: Execution
    "ExecutionStatus",
    "BlockReason",
    "ExecutionDecision",

    # Unified
    "UnifiedSignal",
]
