"""
ORDER COMPUTER V7.0
===================

Calcul d'ordres théoriques TOUJOURS exécuté.

Principe fondamental:
- Pour CHAQUE signal BUY/BUY_STRONG, un ordre est calculé
- L'ordre est proposé MÊME s'il n'est pas exécutable
- Les limites n'empêchent pas le calcul, seulement l'exécution

Ce module est la COUCHE 2 de l'architecture V7:
1. SIGNAL PRODUCER → Détection pure
2. ORDER COMPUTER (ici) → Calcul d'ordres
3. EXECUTION GATE → Limites et autorisations

Responsabilités:
- Position sizing (risk-based)
- Stop loss calculation
- Take profit targets
- Order type selection
- Risk/Reward calculation
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from utils.logger import get_logger

# Import signal types
from src.models.signal_types import (
    SignalType,
    PreSpikeState,
    PreHaltState,
    OrderSide,
    OrderType,
    TimingStrategy,
    ProposedOrder,
    UnifiedSignal,
)

logger = get_logger("ORDER_COMPUTER")


# ============================================================================
# Configuration
# ============================================================================

# Default risk parameters
DEFAULT_RISK_PER_TRADE_PCT = 0.02      # 2% of capital per trade
DEFAULT_MAX_POSITION_PCT = 0.10        # 10% max in single position
DEFAULT_STOP_LOSS_PCT = 0.08           # 8% default stop loss

# Signal-based adjustments
SIGNAL_SIZING = {
    SignalType.BUY_STRONG: {
        "size_multiplier": 1.0,
        "stop_loss_pct": 0.07,
        "order_type": OrderType.MARKET,
    },
    SignalType.BUY: {
        "size_multiplier": 0.75,
        "stop_loss_pct": 0.08,
        "order_type": OrderType.LIMIT,
    },
}

# Pre-Spike adjustments
PRE_SPIKE_ADJUSTMENTS = {
    PreSpikeState.LAUNCHING: {
        "size_multiplier": 1.0,
        "timing": TimingStrategy.IMMEDIATE,
    },
    PreSpikeState.READY: {
        "size_multiplier": 0.9,
        "timing": TimingStrategy.ON_BREAKOUT,
    },
    PreSpikeState.CHARGING: {
        "size_multiplier": 0.7,
        "timing": TimingStrategy.ON_PULLBACK,
    },
}

# Pre-Halt risk adjustments
PRE_HALT_ADJUSTMENTS = {
    PreHaltState.LOW: {"size_multiplier": 1.0},
    PreHaltState.MEDIUM: {"size_multiplier": 0.5},
    PreHaltState.HIGH: {"size_multiplier": 0.0},  # Don't size for high halt risk
}

# Take profit levels (multipliers of risk)
TAKE_PROFIT_LEVELS = [1.5, 2.5, 4.0]  # 1.5R, 2.5R, 4R


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PortfolioContext:
    """Current portfolio state"""
    total_capital: float = 100000.0
    available_cash: float = 100000.0
    current_positions: int = 0
    max_positions: int = 10

    # Risk settings (overridable)
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT


@dataclass
class MarketContext:
    """Current market conditions"""
    # Prices
    current_price: float
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0

    # Volume
    current_volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 1.0

    # Volatility
    atr: float = 0.0                   # Average True Range
    atr_pct: float = 0.0               # ATR as % of price
    daily_range_pct: float = 0.0

    # Support/Resistance (optional)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None


# ============================================================================
# Order Computer
# ============================================================================

class OrderComputer:
    """
    Calcule les ordres théoriques

    Usage:
        computer = OrderComputer()

        # Set portfolio context
        computer.set_portfolio(PortfolioContext(
            total_capital=50000,
            available_cash=45000
        ))

        # Compute order for signal
        signal = await producer.detect(...)
        signal = computer.compute_order(signal, market_context)

    IMPORTANT:
    - L'ordre est TOUJOURS calculé pour les signaux actionables
    - L'ordre est théorique - l'exécution est décidée par ExecutionGate
    - Même si le capital est insuffisant, l'ordre est calculé (pour logging)
    """

    def __init__(self):
        self.portfolio = PortfolioContext()
        self._orders_computed = 0

    def set_portfolio(self, portfolio: PortfolioContext):
        """Update portfolio context"""
        self.portfolio = portfolio

    def compute_order(
        self,
        signal: UnifiedSignal,
        market: MarketContext
    ) -> UnifiedSignal:
        """
        Compute order for a signal

        IMPORTANT: This method ALWAYS computes an order for actionable signals.
        The order is theoretical - execution decision is made separately.
        """
        # Only compute for actionable signals
        if not signal.is_actionable():
            logger.debug(f"{signal.ticker}: Not actionable, skipping order computation")
            return signal

        logger.debug(f"Computing order for {signal.ticker}")

        # Step 1: Calculate position size
        size_shares, size_usd = self._calculate_position_size(
            signal,
            market
        )

        # Step 2: Determine order type and price
        order_type, price_target, price_limit = self._determine_order_params(
            signal,
            market
        )

        # Step 3: Calculate stop loss
        stop_loss, stop_loss_pct = self._calculate_stop_loss(
            signal,
            market
        )

        # Step 4: Calculate take profit targets
        take_profits = self._calculate_take_profits(
            market.current_price,
            stop_loss
        )

        # Step 5: Calculate risk/reward
        risk_reward = self._calculate_risk_reward(
            market.current_price,
            stop_loss,
            take_profits
        )

        # Step 6: Determine timing strategy
        timing = self._determine_timing(signal, market)

        # Step 7: Build rationale
        rationale = self._build_rationale(signal, market, size_shares, stop_loss_pct)

        # Step 8: Create ProposedOrder
        order = ProposedOrder(
            ticker=signal.ticker,
            side=OrderSide.BUY,
            order_type=order_type,
            price_target=price_target,
            price_limit=price_limit,
            size_shares=size_shares,
            size_usd=size_usd,
            size_pct_portfolio=size_usd / self.portfolio.total_capital if self.portfolio.total_capital > 0 else 0,
            stop_loss=stop_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit_targets=take_profits,
            risk_reward_ratio=risk_reward,
            confidence=signal.monster_score,
            timing_strategy=timing,
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
            rationale=rationale
        )

        # Attach to signal
        signal.proposed_order = order

        self._orders_computed += 1

        return signal

    def _calculate_position_size(
        self,
        signal: UnifiedSignal,
        market: MarketContext
    ) -> tuple[int, float]:
        """
        Calculate position size based on risk management rules

        Uses: Fixed fractional position sizing with signal-based adjustments
        """
        # Base size from risk per trade
        risk_amount = self.portfolio.total_capital * self.portfolio.risk_per_trade_pct
        max_position = self.portfolio.total_capital * self.portfolio.max_position_pct

        # Get signal-based multiplier
        signal_config = SIGNAL_SIZING.get(signal.signal_type, {})
        signal_multiplier = signal_config.get("size_multiplier", 1.0)

        # Get pre-spike adjustment
        spike_config = PRE_SPIKE_ADJUSTMENTS.get(signal.pre_spike_state, {})
        spike_multiplier = spike_config.get("size_multiplier", 1.0)

        # Get pre-halt adjustment
        halt_config = PRE_HALT_ADJUSTMENTS.get(signal.pre_halt_state, {})
        halt_multiplier = halt_config.get("size_multiplier", 1.0)

        # Combined multiplier
        total_multiplier = signal_multiplier * spike_multiplier * halt_multiplier

        # Calculate size in USD
        # Use ATR-based sizing if available
        if market.atr > 0:
            # Risk = ATR * 2 (2 ATR stop)
            risk_per_share = market.atr * 2
            shares_from_risk = int(risk_amount / risk_per_share)
            size_usd = shares_from_risk * market.current_price
        else:
            # Use percentage-based sizing
            stop_pct = signal_config.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
            risk_per_share = market.current_price * stop_pct
            shares_from_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            size_usd = shares_from_risk * market.current_price

        # Apply multiplier
        size_usd *= total_multiplier

        # Cap at max position
        size_usd = min(size_usd, max_position)

        # Cap at available cash
        size_usd = min(size_usd, self.portfolio.available_cash)

        # Calculate shares
        if market.current_price > 0:
            size_shares = int(size_usd / market.current_price)
        else:
            size_shares = 0

        # Recalculate USD based on shares
        size_usd = size_shares * market.current_price

        return size_shares, size_usd

    def _determine_order_params(
        self,
        signal: UnifiedSignal,
        market: MarketContext
    ) -> tuple[OrderType, float, Optional[float]]:
        """Determine order type and prices"""

        signal_config = SIGNAL_SIZING.get(signal.signal_type, {})
        order_type = signal_config.get("order_type", OrderType.LIMIT)

        price_target = market.current_price
        price_limit = None

        if order_type == OrderType.MARKET:
            # Market order - use ask if available
            price_target = market.ask if market.ask > 0 else market.current_price

        elif order_type == OrderType.LIMIT:
            # Limit order - bid + small premium
            if market.bid > 0 and market.ask > 0:
                # Place limit between bid and mid
                mid = (market.bid + market.ask) / 2
                price_limit = round(market.bid + (mid - market.bid) * 0.3, 2)
            else:
                # Slight discount from current
                price_limit = round(market.current_price * 0.998, 2)

            price_target = price_limit

        return order_type, price_target, price_limit

    def _calculate_stop_loss(
        self,
        signal: UnifiedSignal,
        market: MarketContext
    ) -> tuple[float, float]:
        """Calculate stop loss level"""

        signal_config = SIGNAL_SIZING.get(signal.signal_type, {})
        base_stop_pct = signal_config.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)

        # Use ATR-based stop if available
        if market.atr > 0:
            # 2 ATR stop
            atr_stop = market.current_price - (market.atr * 2)
            atr_stop_pct = (market.current_price - atr_stop) / market.current_price
            # Use ATR stop if tighter than percentage stop
            if atr_stop_pct < base_stop_pct:
                stop_loss = round(atr_stop, 2)
                stop_loss_pct = atr_stop_pct
            else:
                stop_loss = round(market.current_price * (1 - base_stop_pct), 2)
                stop_loss_pct = base_stop_pct
        else:
            stop_loss = round(market.current_price * (1 - base_stop_pct), 2)
            stop_loss_pct = base_stop_pct

        # Consider support level if available
        if market.nearest_support and market.nearest_support < market.current_price:
            support_stop = market.nearest_support * 0.98  # 2% below support
            # Use support-based stop if tighter
            if support_stop > stop_loss:
                stop_loss = round(support_stop, 2)
                stop_loss_pct = (market.current_price - stop_loss) / market.current_price

        return stop_loss, stop_loss_pct

    def _calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float
    ) -> List[float]:
        """Calculate take profit levels based on risk multiples"""

        risk = entry_price - stop_loss
        if risk <= 0:
            return []

        take_profits = []
        for multiple in TAKE_PROFIT_LEVELS:
            tp = round(entry_price + (risk * multiple), 2)
            take_profits.append(tp)

        return take_profits

    def _calculate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profits: List[float]
    ) -> float:
        """Calculate risk/reward ratio"""

        risk = entry_price - stop_loss
        if risk <= 0 or not take_profits:
            return 0.0

        # Use middle take profit for R/R
        mid_tp_idx = len(take_profits) // 2
        reward = take_profits[mid_tp_idx] - entry_price

        return round(reward / risk, 2) if risk > 0 else 0.0

    def _determine_timing(
        self,
        signal: UnifiedSignal,
        market: MarketContext
    ) -> TimingStrategy:
        """Determine optimal entry timing"""

        # Check pre-spike state
        spike_config = PRE_SPIKE_ADJUSTMENTS.get(signal.pre_spike_state, {})
        timing = spike_config.get("timing", TimingStrategy.IMMEDIATE)

        # Adjust for spread
        if market.spread_pct > 0.02:  # > 2% spread
            # Wide spread - wait for better entry
            if timing == TimingStrategy.IMMEDIATE:
                timing = TimingStrategy.ON_PULLBACK

        return timing

    def _build_rationale(
        self,
        signal: UnifiedSignal,
        market: MarketContext,
        size_shares: int,
        stop_loss_pct: float
    ) -> str:
        """Build human-readable rationale"""

        parts = []

        # Signal strength
        parts.append(f"{signal.signal_type.value} signal (score: {signal.monster_score:.2f})")

        # Catalyst
        if signal.catalyst_type:
            parts.append(f"Catalyst: {signal.catalyst_type}")

        # Pre-spike state
        if signal.pre_spike_state != PreSpikeState.DORMANT:
            parts.append(f"Pre-Spike: {signal.pre_spike_state.value}")

        # Size rationale
        size_pct = (size_shares * market.current_price / self.portfolio.total_capital * 100)
        parts.append(f"Size: {size_shares} shares ({size_pct:.1f}% of portfolio)")

        # Risk
        parts.append(f"Stop: {stop_loss_pct*100:.1f}%")

        return " | ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics"""
        return {
            "orders_computed": self._orders_computed,
            "portfolio_capital": self.portfolio.total_capital,
            "available_cash": self.portfolio.available_cash
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_computer_instance = None
_computer_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_order_computer() -> OrderComputer:
    """Get singleton computer instance"""
    global _computer_instance
    with _computer_lock:
        if _computer_instance is None:
            _computer_instance = OrderComputer()
    return _computer_instance


def compute_order_for_signal(
    signal: UnifiedSignal,
    current_price: float,
    portfolio_capital: float = 100000,
    available_cash: float = 100000
) -> UnifiedSignal:
    """Quick order computation"""
    computer = get_order_computer()

    computer.set_portfolio(PortfolioContext(
        total_capital=portfolio_capital,
        available_cash=available_cash
    ))

    market = MarketContext(current_price=current_price)

    return computer.compute_order(signal, market)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "OrderComputer",
    "PortfolioContext",
    "MarketContext",
    "get_order_computer",
    "compute_order_for_signal",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from src.engines.signal_producer import quick_detect
    import asyncio

    async def test():
        print("=" * 60)
        print("ORDER COMPUTER TEST")
        print("=" * 60)

        computer = OrderComputer()
        computer.set_portfolio(PortfolioContext(
            total_capital=50000,
            available_cash=45000,
            risk_per_trade_pct=0.02,
            max_position_pct=0.10
        ))

        # Test signal
        signal = await quick_detect(
            ticker="BIOX",
            monster_score=0.85,
            current_price=4.50,
            catalyst_type="FDA_APPROVAL",
            pre_spike_state=PreSpikeState.LAUNCHING
        )

        print(f"\nBefore order computation:")
        print(f"  Signal: {signal.signal_type.value}")
        print(f"  Proposed order: {signal.proposed_order}")

        # Compute order
        market = MarketContext(
            current_price=4.50,
            bid=4.48,
            ask=4.52,
            spread_pct=0.009,
            atr=0.35,
            atr_pct=0.078
        )

        signal = computer.compute_order(signal, market)

        print(f"\nAfter order computation:")
        if signal.proposed_order:
            order = signal.proposed_order
            print(f"  Side: {order.side.value}")
            print(f"  Type: {order.order_type.value}")
            print(f"  Size: {order.size_shares} shares (${order.size_usd:,.2f})")
            print(f"  Price target: ${order.price_target:.2f}")
            print(f"  Stop loss: ${order.stop_loss:.2f} ({order.stop_loss_pct*100:.1f}%)")
            print(f"  Take profits: {order.take_profit_targets}")
            print(f"  Risk/Reward: {order.risk_reward_ratio}")
            print(f"  Timing: {order.timing_strategy.value}")
            print(f"  Rationale: {order.rationale}")

        print(f"\nStats: {computer.get_stats()}")

    asyncio.run(test())
