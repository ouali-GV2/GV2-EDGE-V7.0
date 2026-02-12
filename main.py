# ============================
# GV2-EDGE MAIN ENTRY POINT
# V7.0 Architecture Integration
# ============================

# Load environment variables FIRST (before any other imports)
from dotenv import load_dotenv
load_dotenv()

import time
import asyncio
import threading
import datetime
from typing import Dict, List, Optional

from utils.logger import get_logger
from utils.time_utils import is_premarket, is_market_open, is_market_closed, is_after_hours
from src.universe_loader import load_universe
from src.signal_logger import log_signal
from src.watch_list import get_watch_list, get_watch_upgrades

from alerts.telegram_alerts import send_signal_alert
from monitoring.system_guardian import run_guardian
from weekly_deep_audit import run_weekly_audit_v2
from src.afterhours_scanner import run_afterhours_scanner
from daily_audit import run_daily_audit

# Config
from config import (
    USE_V7_ARCHITECTURE,
    DAILY_TRADE_LIMIT,
    MAX_POSITION_PCT,
    MANUAL_CAPITAL,
    ENABLE_PRE_HALT_ENGINE,
    ENABLE_IBKR_NEWS_TRIGGER,
    ENABLE_RISK_GUARD,
    ENABLE_MARKET_MEMORY,
)

logger = get_logger("MAIN")

# ============================
# V7.0 ARCHITECTURE IMPORTS
# ============================

# Signal Producer (Layer 1 - Detection)
from src.engines.signal_producer import (
    SignalProducer,
    DetectionInput,
    get_signal_producer,
)
from src.models.signal_types import (
    SignalType,
    PreSpikeState,
    PreHaltState,
    UnifiedSignal,
)

# Order Computer (Layer 2 - Order Calculation)
from src.engines.order_computer import (
    OrderComputer,
    PortfolioContext,
    MarketContext,
    get_order_computer,
)

# Execution Gate (Layer 3 - Limits)
from src.engines.execution_gate import (
    ExecutionGate,
    AccountState,
    RiskFlags,
    MarketState,
    get_execution_gate,
)

# Risk Guard
from src.risk_guard import (
    get_unified_guard,
    RiskLevel,
    TradeAction,
)

# Market Memory (MRP/EP)
from src.market_memory import (
    enrich_signal_with_context,
    is_market_memory_stable,
    get_missed_tracker,
    MissReason,
)

# Pre-Halt Engine
from src.pre_halt_engine import (
    PreHaltEngine,
    get_pre_halt_engine,
)

# IBKR News Trigger
from src.ibkr_news_trigger import (
    IBKRNewsTrigger,
    get_news_trigger,
)

# Monster Score (for detection input)
from src.scoring.monster_score import compute_monster_score

# Feature Engine (for market context)
from src.feature_engine import compute_features

# IBKR connector for prices
from src.ibkr_connector import get_ibkr

# Legacy imports (fallback)
from src.signal_engine import generate_signal
from src.portfolio_engine import process_signal

# Anticipation Engine (V5)
from src.anticipation_engine import (
    get_anticipation_engine,
    run_anticipation_scan,
    get_watch_early_signals,
    get_buy_signals,
    check_signal_upgrades,
    clear_expired_signals,
    get_engine_status,
    SignalLevel
)

# News Flow Screener (V5)
from src.news_flow_screener import (
    run_news_flow_screener,
    get_events_by_type,
    get_calendar_view
)

# Options Flow via IBKR (V5)
from src.options_flow_ibkr import (
    scan_options_flow,
    get_options_flow_score
)

# Extended Hours Quotes (V5)
from src.extended_hours_quotes import (
    get_extended_quote,
    scan_afterhours_gaps,
    scan_premarket_gaps,
    get_extended_hours_boost
)


# ============================
# V7.0 GLOBAL STATE
# ============================

class V7State:
    """V7.0 Architecture State Manager"""

    def __init__(self):
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.date.today()

        # Singleton instances
        self._producer: Optional[SignalProducer] = None
        self._computer: Optional[OrderComputer] = None
        self._gate: Optional[ExecutionGate] = None
        self._guard = None
        self._pre_halt: Optional[PreHaltEngine] = None
        self._news_trigger: Optional[IBKRNewsTrigger] = None
        self._missed_tracker = None

    def check_day_rollover(self):
        """Reset daily counters if new day"""
        today = datetime.date.today()
        if today != self.last_reset_date:
            logger.info("New trading day - resetting counters")
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def record_trade(self, pnl: float = 0.0):
        """Record executed trade"""
        self.trades_today += 1
        self.daily_pnl += pnl

    @property
    def producer(self) -> SignalProducer:
        if self._producer is None:
            self._producer = get_signal_producer()
        return self._producer

    @property
    def computer(self) -> OrderComputer:
        if self._computer is None:
            self._computer = get_order_computer()
            self._computer.set_portfolio(PortfolioContext(
                total_capital=MANUAL_CAPITAL,
                available_cash=MANUAL_CAPITAL,
                risk_per_trade_pct=0.02,
                max_position_pct=MAX_POSITION_PCT
            ))
        return self._computer

    @property
    def gate(self) -> ExecutionGate:
        if self._gate is None:
            self._gate = get_execution_gate()
        return self._gate

    @property
    def guard(self):
        if self._guard is None and ENABLE_RISK_GUARD:
            self._guard = get_unified_guard()
        return self._guard

    @property
    def pre_halt(self) -> Optional[PreHaltEngine]:
        if self._pre_halt is None and ENABLE_PRE_HALT_ENGINE:
            self._pre_halt = get_pre_halt_engine()
        return self._pre_halt

    @property
    def news_trigger(self) -> Optional[IBKRNewsTrigger]:
        if self._news_trigger is None and ENABLE_IBKR_NEWS_TRIGGER:
            self._news_trigger = get_news_trigger()
        return self._news_trigger

    @property
    def missed_tracker(self):
        if self._missed_tracker is None and ENABLE_MARKET_MEMORY:
            self._missed_tracker = get_missed_tracker()
        return self._missed_tracker


# Global V7 state
_v7_state: Optional[V7State] = None


def get_v7_state() -> V7State:
    """Get V7.0 state singleton"""
    global _v7_state
    if _v7_state is None:
        _v7_state = V7State()
    return _v7_state


# ============================
# V7.0 SIGNAL PIPELINE
# ============================

async def process_ticker_v7(ticker: str, state: V7State) -> Optional[UnifiedSignal]:
    """
    V7.0 Signal Pipeline for single ticker

    Flow:
    1. SignalProducer -> Detection (never blocked)
    2. OrderComputer -> Order calculation (always computed)
    3. ExecutionGate -> Execution decision (limits applied here)

    Returns: UnifiedSignal with all layers populated
    """
    try:
        # === STEP 1: Get monster score and features ===
        score_data = compute_monster_score(ticker)

        if not score_data:
            return None

        monster_score = score_data.get("monster_score", 0)

        # Get features for market context
        features = compute_features(ticker)

        # Get current price
        ibkr = get_ibkr()
        quote = None
        if ibkr and ibkr.connected:
            quote = ibkr.get_quote(ticker, use_cache=True)

        current_price = quote.get("last", 0) if quote else score_data.get("price", 0)

        if not current_price or current_price <= 0:
            return None

        # === STEP 2: Build detection input ===
        detection_input = DetectionInput(
            ticker=ticker,
            current_price=current_price,
            monster_score=monster_score,
            catalyst_score=score_data.get("components", {}).get("event", 0),
            pre_spike_state=PreSpikeState.DORMANT,
            catalyst_type=score_data.get("catalyst_type"),
            catalyst_confidence=score_data.get("catalyst_confidence", 0.5),
            volume_ratio=features.get("volume_ratio", 1.0) if features else 1.0,
            price_change_pct=features.get("price_change", 0) if features else 0,
            market_session="RTH" if is_market_open() else ("PRE" if is_premarket() else "POST")
        )

        # === STEP 3: SIGNAL PRODUCER (Layer 1 - Detection) ===
        signal = await state.producer.detect(detection_input)

        if not signal.is_actionable():
            logger.debug(f"{ticker}: {signal.signal_type.value} (score: {monster_score:.2f})")
            return signal

        # === STEP 4: PRE-HALT ENGINE (sets pre_halt_state) ===
        if state.pre_halt and ENABLE_PRE_HALT_ENGINE:
            try:
                halt_assessment = state.pre_halt.assess(
                    ticker=ticker,
                    current_price=current_price,
                    volatility=features.get("volatility", 0.05) if features else 0.05
                )
                signal.pre_halt_state = halt_assessment.pre_halt_state
            except Exception as e:
                logger.debug(f"Pre-halt assessment error {ticker}: {e}")

        # === STEP 5: MRP/EP ENRICHMENT (Market Memory) ===
        if ENABLE_MARKET_MEMORY:
            try:
                enrich_signal_with_context(signal)
            except Exception as e:
                logger.debug(f"MRP/EP enrichment error {ticker}: {e}")

        # === STEP 6: ORDER COMPUTER (Layer 2 - Always Computed) ===
        market_context = MarketContext(
            current_price=current_price,
            bid=quote.get("bid", current_price * 0.998) if quote else current_price * 0.998,
            ask=quote.get("ask", current_price * 1.002) if quote else current_price * 1.002,
            spread_pct=0.004,
            current_volume=int(quote.get("volume", 0)) if quote else 0,
            atr=features.get("atr", current_price * 0.05) if features else current_price * 0.05
        )

        signal = state.computer.compute_order(signal, market_context)

        # === STEP 7: RISK GUARD (get risk assessment) ===
        risk_flags = None
        if state.guard and ENABLE_RISK_GUARD:
            try:
                assessment = await state.guard.assess(
                    ticker=ticker,
                    current_price=current_price,
                    volatility=features.get("volatility") if features else None
                )

                risk_flags = RiskFlags(
                    ticker=ticker,
                    dilution_risk=assessment.dilution_profile.risk_level.value if assessment.dilution_profile else "LOW",
                    compliance_risk=assessment.compliance_profile.risk_level.value if assessment.compliance_profile else "LOW",
                    delisting_risk="HIGH" if assessment.compliance_profile and assessment.compliance_profile.has_delisting_risk else "LOW",
                    current_price=current_price,
                    is_penny_stock=current_price < 1.0,
                    badges=[str(f) for f in assessment.flags[:3]]
                )
            except Exception as e:
                logger.debug(f"Risk guard error {ticker}: {e}")

        # === STEP 8: EXECUTION GATE (Layer 3 - Limits) ===
        state.check_day_rollover()

        state.gate.set_account(AccountState(
            total_capital=MANUAL_CAPITAL,
            available_cash=MANUAL_CAPITAL,
            trades_today=state.trades_today,
            daily_trade_limit=DAILY_TRADE_LIMIT,
            daily_pnl=state.daily_pnl,
            broker_connected=ibkr.connected if ibkr else False
        ))

        state.gate.set_market(MarketState(
            is_market_open=is_market_open(),
            session=detection_input.market_session,
            is_holiday=False
        ))

        signal = state.gate.evaluate(signal, risk_flags)

        return signal

    except Exception as e:
        logger.error(f"V7 pipeline error {ticker}: {e}", exc_info=True)
        return None


def handle_signal_result(signal: UnifiedSignal, state: V7State):
    """
    Handle signal result - log, alert, track misses

    The signal is ALWAYS visible to the trader, even if blocked.
    """
    if not signal or not signal.is_actionable():
        return

    ticker = signal.ticker
    final_signal = signal.get_final_signal()

    logger.info(
        f"{final_signal}: {ticker} "
        f"(monster: {signal.monster_score:.2f}, pre-halt: {signal.pre_halt_state.value})"
    )

    # Check execution status
    if signal.is_executable():
        if signal.execution and signal.execution.size_multiplier < 1.0:
            logger.info(f"  Size reduced to {signal.execution.size_multiplier*100:.0f}%")

        # Build trade plan for logging and alerts
        if signal.proposed_order:
            order = signal.proposed_order
            trade_plan = {
                "ticker": ticker,
                "signal": signal.signal_type.value,
                "shares": order.size_shares,
                "entry": order.price_target,
                "stop": order.stop_loss,
                "monster_score": signal.monster_score,
                "confidence": order.confidence,
                "notes": f"V7.0 | {signal.pre_halt_state.value}"
            }

            log_signal(trade_plan)
            send_signal_alert(trade_plan)

            logger.info(
                f"  TRADE PLAN: {order.size_shares} shares @ ${order.price_target:.2f} "
                f"(stop: ${order.stop_loss:.2f})"
            )
    else:
        # Signal blocked - track the miss if enabled
        if signal.execution:
            block_reasons = [r.value for r in signal.execution.blocked_by]
            logger.info(f"  BLOCKED: {', '.join(block_reasons)}")

            # Track miss for Market Memory
            if state.missed_tracker and ENABLE_MARKET_MEMORY:
                try:
                    miss_reason = MissReason.OTHER
                    if "DAILY_TRADE_LIMIT" in block_reasons:
                        miss_reason = MissReason.DAILY_TRADE_LIMIT
                    elif "CAPITAL_INSUFFICIENT" in block_reasons:
                        miss_reason = MissReason.CAPITAL_INSUFFICIENT
                    elif "POSITION_LIMIT" in block_reasons:
                        miss_reason = MissReason.POSITION_LIMIT
                    elif "PRE_HALT_HIGH" in block_reasons:
                        miss_reason = MissReason.PRE_HALT_HIGH
                    elif any(r in block_reasons for r in ["DILUTION_HIGH", "COMPLIANCE_HIGH", "PENNY_STOCK_RISK"]):
                        miss_reason = MissReason.RISK_GUARD_BLOCK

                    state.missed_tracker.record_miss(
                        ticker=ticker,
                        signal_type=signal.signal_type.value,
                        signal_price=signal.proposed_order.price_target if signal.proposed_order else 0,
                        monster_score=signal.monster_score,
                        miss_reason=miss_reason
                    )
                except Exception as e:
                    logger.debug(f"Failed to track miss: {e}")

            send_signal_alert({
                "ticker": ticker,
                "signal": f"{signal.signal_type.value} (BLOCKED)",
                "monster_score": signal.monster_score,
                "notes": f"BLOCKED: {signal.execution.block_message}"
            })

    # Show MRP/EP context if active
    if signal.context_active:
        logger.info(
            f"  Context: MRP={signal.context_mrp:.0f}, EP={signal.context_ep:.0f} "
            f"(confidence: {signal.context_confidence:.0f}%)"
        )


# ============================
# EDGE CORE CYCLE V7.0
# ============================

def edge_cycle_v7():
    """
    V7.0 Edge Cycle - Uses new unified signal architecture

    Detection -> Order -> Execution (with full visibility)
    """
    state = get_v7_state()
    state.check_day_rollover()

    universe = load_universe()

    if universe is None or universe.empty:
        logger.warning("Universe empty - skipping cycle")
        return

    logger.info(f"V7 CYCLE: Scanning {len(universe)} tickers")

    # Process tickers
    async def process_all():
        for _, row in universe.iterrows():
            ticker = row["ticker"]

            try:
                signal = await process_ticker_v7(ticker, state)

                if signal:
                    handle_signal_result(signal, state)

            except Exception as e:
                logger.error(f"V7 error on {ticker}: {e}", exc_info=True)

    # Run async processing
    asyncio.run(process_all())

    # Log cycle summary
    producer_stats = state.producer.get_stats()
    gate_stats = state.gate.get_stats()

    logger.info(
        f"Cycle complete: {producer_stats['total_signals']} signals produced, "
        f"{gate_stats['signals_allowed']} allowed, {gate_stats['signals_blocked']} blocked"
    )


# ============================
# LEGACY EDGE CYCLE (Fallback)
# ============================

def edge_cycle():
    """
    Legacy cycle - Used when USE_V7_ARCHITECTURE = False
    """
    universe = load_universe()

    if universe is None or universe.empty:
        logger.warning("Universe empty - skipping cycle")
        return

    logger.info(f"Scanning {len(universe)} tickers")

    for _, row in universe.iterrows():

        ticker = row["ticker"]

        try:
            signal = generate_signal(ticker)

            if not signal or signal["signal"] == "HOLD":
                continue

            from src.ensemble_engine import apply_confluence
            signal = apply_confluence(signal)

            trade_plan = process_signal(signal)

            if not trade_plan:
                logger.warning(f"Could not create trade plan for {ticker}")
                continue

            log_signal(trade_plan)

            logger.info(
                f"TRADE PLAN: {trade_plan['signal']} {trade_plan['shares']} shares of {ticker} "
                f"@ ${trade_plan['entry']} (stop: ${trade_plan['stop']})"
            )

            send_signal_alert(trade_plan)

        except Exception as e:
            logger.error(f"EDGE error on {ticker}: {e}", exc_info=True)


# ============================
# IBKR NEWS TRIGGER HANDLER
# ============================

def process_ibkr_news(headlines: List[Dict]):
    """
    Process incoming IBKR news headlines for early alerts
    """
    if not ENABLE_IBKR_NEWS_TRIGGER:
        return

    state = get_v7_state()

    if not state.news_trigger:
        return

    for item in headlines:
        ticker = item.get("ticker")
        headline = item.get("headline", "")

        if not ticker or not headline:
            continue

        try:
            trigger = state.news_trigger.process_headline(ticker, headline)

            if trigger and trigger.should_alert:
                logger.info(
                    f"NEWS TRIGGER: {ticker} - {trigger.trigger_type.value} "
                    f"(level: {trigger.trigger_level.value})"
                )

                send_signal_alert({
                    "ticker": ticker,
                    "signal": f"NEWS_{trigger.trigger_level.value}",
                    "notes": f"NEWS: {headline[:100]}\n"
                            f"Type: {trigger.trigger_type.value}\n"
                            f"Actions: {trigger.recommended_actions}"
                })

        except Exception as e:
            logger.debug(f"News trigger error: {e}")


# ============================
# WEEKLY AUDIT SCHEDULER
# ============================

last_audit_day = None

def should_run_weekly_audit():
    now = datetime.datetime.utcnow()
    return now.weekday() == 4 and now.hour == 22  # Friday 22h UTC


# ============================
# DAILY AUDIT SCHEDULER
# ============================

last_daily_audit_day = None

def should_run_daily_audit():
    """Run daily audit at 20:30 UTC (after US market close)"""
    now = datetime.datetime.utcnow()
    return now.hour == 20 and now.minute >= 30


# ============================
# WATCH LIST SCHEDULER
# ============================

last_watch_list_day = None

def should_generate_watch_list():
    """Generate watch list once per day at 3 AM UTC (before PM)"""
    now = datetime.datetime.utcnow()
    return now.hour == 3  # 3 AM UTC


def generate_and_send_watch_list():
    """
    Generate watch list and send summary via Telegram
    """
    global last_watch_list_day

    now_date = datetime.datetime.utcnow().date()

    if now_date == last_watch_list_day:
        return

    logger.info("Generating daily WATCH list...")

    universe = load_universe()
    if universe is None or universe.empty:
        return

    tickers = universe["ticker"].tolist()
    watch_list = get_watch_list(universe_tickers=tickers)

    if not watch_list:
        logger.info("No WATCH signals today")
        return

    upgrades = get_watch_upgrades(watch_list)

    summary = f"DAILY WATCH LIST ({len(watch_list)} signals)\n\n"

    for watch in watch_list[:5]:
        summary += f"TARGET: {watch['ticker']} - {watch['event_type']}\n"
        summary += f"   {watch['days_to_event']} days | Impact: {watch['impact']:.2f}\n"
        summary += f"   {watch['reason']}\n\n"

    if upgrades:
        summary += f"\nUPGRADES TO BUY ({len(upgrades)}):\n"
        for upgrade in upgrades:
            summary += f"UPGRADED: {upgrade['ticker']} - {upgrade['reason']}\n"

    send_signal_alert({
        "ticker": "WATCH_LIST",
        "signal": "WATCH",
        "notes": summary
    })

    last_watch_list_day = now_date
    logger.info(f"Sent watch list: {len(watch_list)} signals, {len(upgrades)} upgrades")


# ============================
# MAIN LOOP
# ============================

def run_edge():
    global last_audit_day
    global last_daily_audit_day

    arch_mode = "V7.0" if USE_V7_ARCHITECTURE else "LEGACY"
    logger.info(f"GV2-EDGE LIVE ENGINE STARTED ({arch_mode} ARCHITECTURE)")

    if USE_V7_ARCHITECTURE:
        state = get_v7_state()
        logger.info(f"  V7 State initialized")
        logger.info(f"  Daily trade limit: {DAILY_TRADE_LIMIT}")
        logger.info(f"  Pre-Halt Engine: {'ENABLED' if ENABLE_PRE_HALT_ENGINE else 'DISABLED'}")
        logger.info(f"  Risk Guard: {'ENABLED' if ENABLE_RISK_GUARD else 'DISABLED'}")
        logger.info(f"  Market Memory: {'ENABLED' if ENABLE_MARKET_MEMORY else 'DISABLED'}")
        logger.info(f"  News Trigger: {'ENABLED' if ENABLE_IBKR_NEWS_TRIGGER else 'DISABLED'}")

    while True:
        try:
            now = datetime.datetime.utcnow().date()

            # ---- Daily WATCH list (3 AM UTC) ----
            if should_generate_watch_list():
                logger.info("Generating daily WATCH list")
                generate_and_send_watch_list()

            # ---- Daily Audit (20:30 UTC - after market close) ----
            if should_run_daily_audit() and now != last_daily_audit_day:
                logger.info("Running Daily Audit")
                try:
                    run_daily_audit(send_telegram=True)
                    last_daily_audit_day = now
                except Exception as e:
                    logger.error(f"Daily Audit failed: {e}", exc_info=True)

            # ---- Weekly audit V2 ----
            if should_run_weekly_audit() and now != last_audit_day:
                logger.info("Running Weekly Deep Audit V2")
                run_weekly_audit_v2(days_back=7)
                last_audit_day = now

            # ---- After-hours catalyst scan ----
            if is_after_hours():
                logger.info("AFTER-HOURS session - ANTICIPATION MODE")

                universe = load_universe()
                tickers = universe["ticker"].tolist() if universe is not None else []

                # News Flow Screener
                try:
                    logger.info("Step 1: News Flow Screener...")
                    ticker_events = run_news_flow_screener(tickers, hours_back=6)

                    events_by_type = get_events_by_type(ticker_events)
                    for event_type, event_list in events_by_type.items():
                        if event_list:
                            tickers_with_event = [e['ticker'] for e in event_list[:5]]
                            logger.info(f"  {event_type}: {tickers_with_event}")

                except Exception as e:
                    logger.error(f"News Flow Screener failed: {e}", exc_info=True)
                    ticker_events = {}

                # Extended Hours Gaps
                try:
                    logger.info("Step 2: Extended Hours Gap Scan...")
                    ah_gaps = scan_afterhours_gaps(tickers[:100], min_gap=0.03)

                    for gap in ah_gaps[:10]:
                        logger.info(f"  GAP {gap.ticker}: gap={gap.gap_pct*100:+.1f}%, vol={gap.volume:,}")

                except Exception as e:
                    logger.error(f"Extended Hours scan failed: {e}", exc_info=True)
                    ah_gaps = []

                # Options Flow
                try:
                    high_priority = list(set(
                        list(ticker_events.keys())[:20] +
                        [g.ticker for g in ah_gaps[:10]]
                    ))

                    if high_priority:
                        logger.info(f"Step 3: Options Flow on {len(high_priority)} high-priority...")
                        options_signals = scan_options_flow(high_priority)

                        for ticker, signals in options_signals.items():
                            for sig in signals:
                                if sig.score >= 0.5:
                                    logger.info(f"  OPTIONS {ticker}: {sig.signal_type} (score: {sig.score:.2f})")

                except Exception as e:
                    logger.error(f"Options Flow scan failed: {e}", exc_info=True)

                # Anticipation Engine
                try:
                    logger.info("Step 4: Anticipation Engine...")
                    results = run_anticipation_scan(tickers, mode="afterhours")

                    for signal_dict in results.get("new_signals", []):
                        if signal_dict.get("signal_level") in ["BUY", "BUY_STRONG"]:
                            boost, boost_details = get_extended_hours_boost(signal_dict["ticker"])

                            send_signal_alert({
                                "ticker": signal_dict["ticker"],
                                "signal": signal_dict["signal_level"],
                                "monster_score": signal_dict.get("combined_score", 0),
                                "notes": f"ANTICIPATION ({signal_dict.get('urgency', 'N/A')})\n"
                                        f"Catalyst: {signal_dict.get('catalyst_type', 'N/A')}\n"
                                        f"AH Boost: +{boost:.2f}\n"
                                        f"{signal_dict.get('catalyst_summary', '')[:100]}"
                            })
                            logger.info(f"ANTICIPATION ALERT: {signal_dict['ticker']} - {signal_dict['signal_level']}")

                        elif signal_dict.get("signal_level") == "WATCH_EARLY":
                            logger.info(f"WATCH_EARLY: {signal_dict['ticker']} (score: {signal_dict.get('combined_score', 0):.2f})")

                    status = get_engine_status()
                    logger.info(f"Engine: {status['suspects_count']} suspects, {status['watch_signals_count']} signals")

                except Exception as e:
                    logger.error(f"Anticipation scan failed: {e}", exc_info=True)

                run_afterhours_scanner(tickers=tickers[:50])

                time.sleep(600)  # 10 min in after-hours

            # ---- Pre-market session ----
            elif is_premarket():
                logger.info("PRE-MARKET session - CONFIRMATION MODE")

                universe = load_universe()
                tickers = universe["ticker"].tolist() if universe is not None else []

                try:
                    results = run_anticipation_scan(tickers, mode="premarket")
                    upgrades = results.get("upgrades", [])

                    for upgrade in upgrades:
                        send_signal_alert({
                            "ticker": upgrade["ticker"],
                            "signal": upgrade.get("signal_level", "BUY"),
                            "monster_score": upgrade.get("combined_score", 0),
                            "notes": f"UPGRADED from WATCH_EARLY\n"
                                    f"Catalyst: {upgrade.get('catalyst_type', 'N/A')}\n"
                                    f"{upgrade.get('catalyst_summary', '')[:100]}"
                        })
                        logger.info(f"UPGRADE ALERT: {upgrade['ticker']} -> {upgrade.get('signal_level')}")

                    status = get_engine_status()
                    logger.info(f"PM Engine: {status['watch_signals_count']} signals active")

                except Exception as e:
                    logger.error(f"PM anticipation failed: {e}", exc_info=True)

                # Run V7 or legacy cycle
                if USE_V7_ARCHITECTURE:
                    edge_cycle_v7()
                else:
                    edge_cycle()

                time.sleep(300)  # every 5 min

            # ---- Regular market hours ----
            elif is_market_open():
                logger.info("REGULAR MARKET session")

                clear_expired_signals(max_age_hours=24)

                # Run V7 or legacy cycle
                if USE_V7_ARCHITECTURE:
                    edge_cycle_v7()
                else:
                    edge_cycle()

                time.sleep(180)  # every 3 min

            # ---- Market closed ----
            elif is_market_closed():
                logger.info("Market closed - idle")
                time.sleep(900)  # 15 min sleep

            else:
                time.sleep(300)

        except Exception as e:
            logger.error(f"Main loop crash: {e}", exc_info=True)
            time.sleep(60)


# ============================
# SYSTEM GUARDIAN THREAD
# ============================

def start_guardian():
    run_guardian()


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":

    logger.info("Booting GV2-EDGE")

    guardian_thread = threading.Thread(
        target=start_guardian,
        daemon=True
    )
    guardian_thread.start()

    run_edge()
