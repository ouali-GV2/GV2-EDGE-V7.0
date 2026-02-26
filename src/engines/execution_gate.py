"""
EXECUTION GATE V7.0
===================

SEULE couche avec limites d'exécution.

Principe fondamental:
- Les limites n'affectent QUE l'exécution, jamais la détection
- Le signal original est TOUJOURS préservé et visible
- L'ordre théorique est TOUJOURS calculé

Ce module est la COUCHE 3 de l'architecture V7:
1. SIGNAL PRODUCER → Détection pure (illimitée)
2. ORDER COMPUTER → Calcul d'ordres (toujours)
3. EXECUTION GATE (ici) → Limites et autorisations

Limites gérées:
- Daily trade limit (ex: 5 trades/jour)
- Capital disponible
- Position concentration
- Pre-Halt risk
- Risk Guard flags (dilution, compliance, delisting)
- Market hours
- Broker connection status
"""

import threading
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger

# Import signal types
from src.models.signal_types import (
    SignalType,
    PreHaltState,
    ExecutionStatus,
    BlockReason,
    ExecutionDecision,
    UnifiedSignal,
)

logger = get_logger("EXECUTION_GATE")


# ============================================================================
# Configuration
# ============================================================================

# Default limits
DEFAULT_DAILY_TRADE_LIMIT = 5
DEFAULT_MAX_POSITION_PCT = 0.10         # 10% max in single position
DEFAULT_MAX_TOTAL_EXPOSURE = 0.80       # 80% max total exposure
DEFAULT_MIN_ORDER_USD = 100             # Minimum order size

# Penny stock threshold
PENNY_STOCK_THRESHOLD = 1.0


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AccountState:
    """Current account state"""
    # Capital
    total_capital: float = 100000.0
    available_cash: float = 100000.0
    total_exposure: float = 0.0

    # Positions
    current_positions: Dict[str, float] = field(default_factory=dict)

    # Daily stats
    trades_today: int = 0
    daily_pnl: float = 0.0
    daily_trade_limit: int = DEFAULT_DAILY_TRADE_LIMIT

    # Risk limits
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE
    max_daily_loss: float = -5000.0     # Stop trading if exceeded

    # Connection status
    broker_connected: bool = True


@dataclass
class RiskFlags:
    """Risk guard flags for a ticker"""
    ticker: str

    # Risk levels
    dilution_risk: str = "LOW"          # LOW, MEDIUM, HIGH
    compliance_risk: str = "LOW"
    delisting_risk: str = "LOW"

    # Price info
    current_price: float = 0.0
    is_penny_stock: bool = False

    # Badges
    badges: List[str] = field(default_factory=list)


@dataclass
class MarketState:
    """Current market state"""
    is_market_open: bool = True
    session: str = "RTH"                # PRE, RTH, POST, CLOSED
    is_holiday: bool = False
    circuit_breaker_active: bool = False


# ============================================================================
# Execution Gate
# ============================================================================

class ExecutionGate:
    """
    Gate d'exécution - applique les limites

    Usage:
        gate = ExecutionGate()
        gate.set_account(account_state)

        # Check execution for signal
        signal = gate.evaluate(signal, risk_flags, market_state)

        # Signal.execution now contains the decision
        if signal.is_executable():
            broker.submit(signal.proposed_order)
        else:
            log_blocked_signal(signal)

    IMPORTANT:
    - Le signal original est TOUJOURS préservé (signal_preserved=True)
    - L'ordre théorique est TOUJOURS visible
    - Seul l'autorisation d'exécution change
    """

    def __init__(self):
        self.account = AccountState()
        self.market = MarketState()

        # Tracking
        self._today = date.today()
        self._daily_signals: List[UnifiedSignal] = []
        self._blocked_signals: List[UnifiedSignal] = []

    def set_account(self, account: AccountState):
        """Update account state"""
        self.account = account
        self._check_day_rollover()

    def set_market(self, market: MarketState):
        """Update market state"""
        self.market = market

    def evaluate(
        self,
        signal: UnifiedSignal,
        risk_flags: Optional[RiskFlags] = None,
        market_state: Optional[MarketState] = None
    ) -> UnifiedSignal:
        """
        Evaluate execution permission for a signal

        IMPORTANT:
        - Signal detection is NEVER blocked
        - Only execution authorization changes
        - Original signal is ALWAYS preserved
        """
        self._check_day_rollover()

        if market_state:
            self.market = market_state

        # Only evaluate actionable signals
        if not signal.is_actionable():
            signal.execution = ExecutionDecision(
                status=ExecutionStatus.ALERT_ONLY,
                size_multiplier=0.0,
                would_have_executed=False
            )
            return signal

        logger.debug(f"Evaluating execution for {signal.ticker}")

        # Run all checks
        blocks, size_multiplier = self._run_all_checks(signal, risk_flags)

        # Build execution decision
        if blocks:
            # Determine status based on blocks
            if any(b in [BlockReason.PRE_HALT_HIGH, BlockReason.DILUTION_HIGH,
                        BlockReason.DELISTING_HIGH, BlockReason.PENNY_STOCK_RISK]
                   for b in blocks):
                status = ExecutionStatus.EXECUTE_BLOCKED
            elif BlockReason.DAILY_TRADE_LIMIT in blocks:
                status = ExecutionStatus.EXECUTE_BLOCKED
            elif BlockReason.MARKET_CLOSED in blocks:
                status = ExecutionStatus.EXECUTE_DELAYED
            elif size_multiplier > 0:
                status = ExecutionStatus.EXECUTE_REDUCED
            else:
                status = ExecutionStatus.EXECUTE_BLOCKED

            execution = ExecutionDecision(
                status=status,
                size_multiplier=size_multiplier,
                blocked_by=blocks,
                block_message=self._build_block_message(blocks),
                would_have_executed=True
            )

            # Track blocked signal
            self._blocked_signals.append(signal)

        else:
            # All checks passed
            execution = ExecutionDecision(
                status=ExecutionStatus.EXECUTE_ALLOWED,
                size_multiplier=size_multiplier,
                would_have_executed=True
            )

        signal.execution = execution

        # Add risk badges to signal
        if risk_flags and risk_flags.badges:
            signal.badges.extend(risk_flags.badges)

        # Track signal
        self._daily_signals.append(signal)

        return signal

    def _run_all_checks(
        self,
        signal: UnifiedSignal,
        risk_flags: Optional[RiskFlags]
    ) -> tuple[List[BlockReason], float]:
        """Run all execution checks"""

        blocks = []
        size_multiplier = 1.0

        # Check 1: Daily trade limit
        if self.account.trades_today >= self.account.daily_trade_limit:
            blocks.append(BlockReason.DAILY_TRADE_LIMIT)
            size_multiplier = 0.0

        # Check 2: Capital availability
        if signal.proposed_order:
            required = signal.proposed_order.size_usd
            if required > self.account.available_cash:
                blocks.append(BlockReason.CAPITAL_INSUFFICIENT)
                # Allow reduced size if some capital available
                if self.account.available_cash > DEFAULT_MIN_ORDER_USD:
                    size_multiplier = min(size_multiplier,
                                         self.account.available_cash / required)
                else:
                    size_multiplier = 0.0

        # Check 3: Position concentration
        if signal.proposed_order and self.account.total_capital > 0:
            position_pct = signal.proposed_order.size_usd / self.account.total_capital
            if position_pct > self.account.max_position_pct:
                blocks.append(BlockReason.POSITION_LIMIT)
                size_multiplier = min(size_multiplier,
                                     self.account.max_position_pct / position_pct)

        # Check 4: Pre-Halt risk
        if signal.pre_halt_state == PreHaltState.HIGH:
            blocks.append(BlockReason.PRE_HALT_HIGH)
            size_multiplier = 0.0

        elif signal.pre_halt_state == PreHaltState.MEDIUM:
            # Reduce size for medium halt risk
            size_multiplier = min(size_multiplier, 0.5)

        # Check 5: Risk Guard flags
        if risk_flags:
            risk_blocks, risk_multiplier = self._check_risk_flags(risk_flags)
            blocks.extend(risk_blocks)
            size_multiplier = min(size_multiplier, risk_multiplier)

        # Check 6: Market hours
        if not self.market.is_market_open:
            blocks.append(BlockReason.MARKET_CLOSED)
            # Don't set size to 0 - order can be queued

        # Check 7: Circuit breaker
        if self.market.circuit_breaker_active:
            blocks.append(BlockReason.CIRCUIT_BREAKER)
            size_multiplier = 0.0

        # Check 8: Broker connection
        if not self.account.broker_connected:
            blocks.append(BlockReason.BROKER_DISCONNECTED)
            size_multiplier = 0.0

        # Check 9: Daily P&L limit
        if self.account.daily_pnl <= self.account.max_daily_loss:
            # Don't add to blocks, just reduce size
            size_multiplier = min(size_multiplier, 0.5)

        return blocks, max(0.0, size_multiplier)

    def _check_risk_flags(
        self,
        risk_flags: RiskFlags
    ) -> tuple[List[BlockReason], float]:
        """Check risk guard flags"""

        blocks = []
        multiplier = 1.0

        is_penny = risk_flags.is_penny_stock or risk_flags.current_price < PENNY_STOCK_THRESHOLD

        # Penny stock rules (strict)
        if is_penny:
            if risk_flags.dilution_risk == "HIGH":
                blocks.append(BlockReason.DILUTION_HIGH)
                blocks.append(BlockReason.PENNY_STOCK_RISK)
                multiplier = 0.0

            elif risk_flags.compliance_risk == "HIGH":
                blocks.append(BlockReason.COMPLIANCE_HIGH)
                blocks.append(BlockReason.PENNY_STOCK_RISK)
                multiplier = 0.0

            elif risk_flags.delisting_risk == "HIGH":
                blocks.append(BlockReason.DELISTING_HIGH)
                blocks.append(BlockReason.PENNY_STOCK_RISK)
                multiplier = 0.0

            # 2 MEDIUM = block for penny
            medium_count = sum(1 for r in [risk_flags.dilution_risk,
                                           risk_flags.compliance_risk,
                                           risk_flags.delisting_risk]
                              if r == "MEDIUM")
            if medium_count >= 2:
                blocks.append(BlockReason.PENNY_STOCK_RISK)
                multiplier = 0.0

            # 1 MEDIUM = reduced size
            elif medium_count == 1:
                multiplier = min(multiplier, 0.5)

            # Even all LOW = reduced size for penny
            else:
                multiplier = min(multiplier, 0.75)

        # Standard stock rules (> $1)
        else:
            # Count HIGH risks
            high_count = sum(1 for r in [risk_flags.dilution_risk,
                                         risk_flags.compliance_risk,
                                         risk_flags.delisting_risk]
                            if r == "HIGH")

            if high_count >= 3:
                blocks.append(BlockReason.DILUTION_HIGH)
                multiplier = 0.0

            elif high_count >= 2:
                multiplier = 0.0  # Block but don't add specific reason

            elif high_count == 1:
                # Single HIGH = reduced size
                multiplier = min(multiplier, 0.5)
                if risk_flags.dilution_risk == "HIGH":
                    blocks.append(BlockReason.DILUTION_HIGH)
                elif risk_flags.compliance_risk == "HIGH":
                    blocks.append(BlockReason.COMPLIANCE_HIGH)
                elif risk_flags.delisting_risk == "HIGH":
                    blocks.append(BlockReason.DELISTING_HIGH)

            # Count MEDIUM risks
            medium_count = sum(1 for r in [risk_flags.dilution_risk,
                                           risk_flags.compliance_risk,
                                           risk_flags.delisting_risk]
                              if r == "MEDIUM")

            if medium_count >= 3:
                multiplier = min(multiplier, 0.6)
            elif medium_count >= 2:
                multiplier = min(multiplier, 0.7)
            elif medium_count == 1:
                multiplier = min(multiplier, 0.85)

        return blocks, multiplier

    def _build_block_message(self, blocks: List[BlockReason]) -> str:
        """Build human-readable block message"""

        messages = {
            BlockReason.DAILY_TRADE_LIMIT: f"Daily trade limit ({self.account.daily_trade_limit}) reached",
            BlockReason.CAPITAL_INSUFFICIENT: "Insufficient capital",
            BlockReason.POSITION_LIMIT: "Position size limit exceeded",
            BlockReason.PRE_HALT_HIGH: "High halt risk detected",
            BlockReason.DILUTION_HIGH: "High dilution risk",
            BlockReason.COMPLIANCE_HIGH: "Compliance warning active",
            BlockReason.DELISTING_HIGH: "Delisting risk detected",
            BlockReason.PENNY_STOCK_RISK: "Penny stock risk protection",
            BlockReason.MARKET_CLOSED: "Market closed",
            BlockReason.CIRCUIT_BREAKER: "Circuit breaker active",
            BlockReason.BROKER_DISCONNECTED: "Broker disconnected",
        }

        parts = [messages.get(b, b.value) for b in blocks]
        return " | ".join(parts)

    def _check_day_rollover(self):
        """Check if day has changed and reset counters"""
        today = date.today()
        if today != self._today:
            self._today = today
            self._daily_signals.clear()
            self._blocked_signals.clear()
            # Note: account.trades_today should be reset externally

    def record_trade_executed(self):
        """Record that a trade was executed"""
        self.account.trades_today += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics"""
        total = len(self._daily_signals)
        blocked = len(self._blocked_signals)
        executed = total - blocked

        return {
            "date": self._today.isoformat(),
            "trades_today": self.account.trades_today,
            "trade_limit": self.account.daily_trade_limit,
            "signals_evaluated": total,
            "signals_allowed": executed,
            "signals_blocked": blocked,
            "block_rate": blocked / total if total > 0 else 0,
            "blocked_by_reason": self._get_block_breakdown()
        }

    def _get_block_breakdown(self) -> Dict[str, int]:
        """Get breakdown of blocks by reason"""
        breakdown = {}
        for signal in self._blocked_signals:
            if signal.execution:
                for reason in signal.execution.blocked_by:
                    breakdown[reason.value] = breakdown.get(reason.value, 0) + 1
        return breakdown

    def get_blocked_signals(self) -> List[UnifiedSignal]:
        """Get list of blocked signals today"""
        return self._blocked_signals.copy()


# ============================================================================
# Convenience Functions
# ============================================================================

_gate_instance = None
_gate_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_execution_gate() -> ExecutionGate:
    """Get singleton gate instance"""
    global _gate_instance
    with _gate_lock:
        if _gate_instance is None:
            _gate_instance = ExecutionGate()
    return _gate_instance


def quick_evaluate(
    signal: UnifiedSignal,
    trades_today: int = 0,
    available_cash: float = 100000,
    current_price: float = None,
    dilution_risk: str = "LOW",
    pre_halt_state: PreHaltState = PreHaltState.LOW
) -> UnifiedSignal:
    """Quick execution evaluation"""

    gate = get_execution_gate()

    gate.set_account(AccountState(
        available_cash=available_cash,
        trades_today=trades_today
    ))

    risk_flags = None
    if current_price:
        risk_flags = RiskFlags(
            ticker=signal.ticker,
            current_price=current_price,
            is_penny_stock=current_price < PENNY_STOCK_THRESHOLD,
            dilution_risk=dilution_risk
        )

    # Update signal's pre-halt state
    signal.pre_halt_state = pre_halt_state

    return gate.evaluate(signal, risk_flags)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "ExecutionGate",
    "AccountState",
    "RiskFlags",
    "MarketState",
    "get_execution_gate",
    "quick_evaluate",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from src.engines.signal_producer import quick_detect
    from src.engines.order_computer import compute_order_for_signal
    from src.models.signal_types import PreSpikeState
    import asyncio

    async def test():
        print("=" * 60)
        print("EXECUTION GATE TEST")
        print("=" * 60)

        gate = ExecutionGate()

        # Test 1: Normal execution
        print("\n--- Test 1: Normal execution ---")
        signal = await quick_detect(
            ticker="BIOX",
            monster_score=0.85,
            current_price=4.50,
            catalyst_type="FDA_APPROVAL",
            pre_spike_state=PreSpikeState.LAUNCHING
        )
        signal = compute_order_for_signal(signal, current_price=4.50)

        gate.set_account(AccountState(
            trades_today=2,
            daily_trade_limit=5,
            available_cash=50000
        ))

        signal = gate.evaluate(signal)
        print(f"  Signal: {signal.signal_type.value}")
        print(f"  Execution: {signal.execution.status.value}")
        print(f"  Size multiplier: {signal.execution.size_multiplier}")

        # Test 2: Trade limit reached
        print("\n--- Test 2: Trade limit reached ---")
        signal2 = await quick_detect(
            ticker="ACME",
            monster_score=0.80,
            current_price=2.50,
            catalyst_type="EARNINGS_BEAT"
        )
        signal2 = compute_order_for_signal(signal2, current_price=2.50)

        gate.set_account(AccountState(
            trades_today=5,
            daily_trade_limit=5,
            available_cash=50000
        ))

        signal2 = gate.evaluate(signal2)
        print(f"  Signal: {signal2.signal_type.value}")
        print(f"  Execution: {signal2.execution.status.value}")
        print(f"  Blocked by: {[b.value for b in signal2.execution.blocked_by]}")
        print(f"  Message: {signal2.execution.block_message}")
        print(f"  Would have executed: {signal2.execution.would_have_executed}")

        # Test 3: Penny stock with HIGH risk
        print("\n--- Test 3: Penny stock with HIGH dilution risk ---")
        signal3 = await quick_detect(
            ticker="XYZZ",
            monster_score=0.90,
            current_price=0.75,
            catalyst_type="CONTRACT_WIN"
        )
        signal3 = compute_order_for_signal(signal3, current_price=0.75)

        gate.set_account(AccountState(trades_today=0, available_cash=50000))

        risk_flags = RiskFlags(
            ticker="XYZZ",
            current_price=0.75,
            is_penny_stock=True,
            dilution_risk="HIGH",
            badges=["⚠️ DILUTION HIGH"]
        )

        signal3 = gate.evaluate(signal3, risk_flags)
        print(f"  Signal: {signal3.signal_type.value}")
        print(f"  Execution: {signal3.execution.status.value}")
        print(f"  Blocked by: {[b.value for b in signal3.execution.blocked_by]}")
        print(f"  Badges: {signal3.badges}")

        # Test 4: Pre-Halt HIGH
        print("\n--- Test 4: Pre-Halt HIGH ---")
        signal4 = await quick_detect(
            ticker="HALT",
            monster_score=0.95,
            current_price=5.00,
            catalyst_type="BUYOUT",
            pre_spike_state=PreSpikeState.LAUNCHING
        )
        signal4.pre_halt_state = PreHaltState.HIGH
        signal4 = compute_order_for_signal(signal4, current_price=5.00)

        gate.set_account(AccountState(trades_today=0, available_cash=50000))
        signal4 = gate.evaluate(signal4)

        print(f"  Signal: {signal4.signal_type.value}")
        print(f"  Pre-Halt: {signal4.pre_halt_state.value}")
        print(f"  Execution: {signal4.execution.status.value}")
        print(f"  Blocked by: {[b.value for b in signal4.execution.blocked_by]}")

        print(f"\n--- Gate Stats ---")
        print(gate.get_stats())

    asyncio.run(test())
