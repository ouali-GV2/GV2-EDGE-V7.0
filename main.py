import time
import threading
import datetime

from utils.logger import get_logger
from utils.time_utils import is_premarket, is_market_open, is_market_closed, is_after_hours
from utils.api_guard import api_safe_call

from src.universe_loader import load_universe
from src.signal_engine import generate_signal
from src.portfolio_engine import process_signal
from src.signal_logger import log_signal
from src.watch_list import get_watch_list, get_watch_upgrades

from alerts.telegram_alerts import send_signal_alert
from monitoring.system_guardian import run_system_guardian
from weekly_deep_audit import run_weekly_audit_v2
from src.afterhours_scanner import run_afterhours_scanner
from daily_audit import run_daily_audit

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

logger = get_logger("MAIN")

# ============================
# EDGE CORE CYCLE
# ============================

def edge_cycle():
    """
    Cycle principal du moteur EDGE - VERSION SIMPLIFIÉE FONCTIONNELLE
    """
    universe = load_universe()

    if universe is None or universe.empty:
        logger.warning("Universe empty - skipping cycle")
        return

    logger.info(f"Scanning {len(universe)} tickers")

    for _, row in universe.iterrows():

        ticker = row["ticker"]

        try:
            # ------------------------
            # SIGNAL GENERATION
            # ------------------------
            # generate_signal() encapsule toute la logique:
            # - Récupère les events via get_events_by_ticker()
            # - Calcule les features
            # - Calcule le monster_score
            # - Détermine le signal (BUY/BUY_STRONG/HOLD)
            
            signal = generate_signal(ticker)

            if not signal or signal["signal"] == "HOLD":
                continue

            # ------------------------
            # ENSEMBLE (CONFLUENCE)
            # ------------------------
            # Applique le boost de confluence sur le signal
            from src.ensemble_engine import apply_confluence
            signal = apply_confluence(signal)

            # ------------------------
            # PORTFOLIO ENGINE
            # ------------------------
            # Calcule la taille de position et les stops
            trade_plan = process_signal(signal)
            
            if not trade_plan:
                logger.warning(f"Could not create trade plan for {ticker}")
                continue
            
            # ------------------------
            # LOG SIGNAL (PERSISTENCE)
            # ------------------------
            # Store signal in database for audit
            log_signal(trade_plan)
            
            logger.info(
                f"TRADE PLAN: {trade_plan['signal']} {trade_plan['shares']} shares of {ticker} "
                f"@ ${trade_plan['entry']} (stop: ${trade_plan['stop']})"
            )

            # ------------------------
            # ALERT
            # ------------------------
            send_signal_alert(trade_plan)

        except Exception as e:
            logger.error(f"EDGE error on {ticker}: {e}", exc_info=True)


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
    
    Daily summary of upcoming high-impact events
    """
    global last_watch_list_day
    
    now_date = datetime.datetime.utcnow().date()
    
    if now_date == last_watch_list_day:
        return  # Already sent today
    
    logger.info("Generating daily WATCH list...")
    
    universe = load_universe()
    if universe is None or universe.empty:
        return
    
    tickers = universe["ticker"].tolist()
    
    # Generate watch list
    watch_list = get_watch_list(universe_tickers=tickers)
    
    if not watch_list:
        logger.info("No WATCH signals today")
        return
    
    # Check for upgrades (WATCH → BUY)
    upgrades = get_watch_upgrades(watch_list)
    
    # Send summary
    summary = f"📅 DAILY WATCH LIST ({len(watch_list)} signals)\n\n"
    
    # Top 5 high-priority watches
    for watch in watch_list[:5]:
        summary += f"🎯 {watch['ticker']} - {watch['event_type']}\n"
        summary += f"   {watch['days_to_event']} days | Impact: {watch['impact']:.2f}\n"
        summary += f"   {watch['reason']}\n\n"
    
    if upgrades:
        summary += f"\n⬆️ UPGRADES TO BUY ({len(upgrades)}):\n"
        for upgrade in upgrades:
            summary += f"✅ {upgrade['ticker']} - {upgrade['reason']}\n"
    
    # Send via Telegram
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

    logger.info("GV2-EDGE LIVE ENGINE STARTED (V4 WITH DAILY AUDIT)")

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

            # ---- After-hours catalyst scan (ANTICIPATION MODE V5) ----
            if is_after_hours():
                logger.info("🌙 AFTER-HOURS session - ANTICIPATION MODE V5 ACTIVE")
                
                universe = load_universe()
                tickers = universe["ticker"].tolist() if universe is not None else []
                
                # === STEP 1: News Flow Screener (Global News → Tickers) ===
                try:
                    logger.info("📰 Step 1: News Flow Screener...")
                    ticker_events = run_news_flow_screener(tickers, hours_back=6)
                    
                    # Log events by type
                    events_by_type = get_events_by_type(ticker_events)
                    for event_type, event_list in events_by_type.items():
                        if event_list:
                            tickers_with_event = [e['ticker'] for e in event_list[:5]]
                            logger.info(f"  📅 {event_type}: {tickers_with_event}")
                    
                except Exception as e:
                    logger.error(f"News Flow Screener failed: {e}", exc_info=True)
                    ticker_events = {}
                
                # === STEP 2: Extended Hours Gaps ===
                try:
                    logger.info("📈 Step 2: Extended Hours Gap Scan...")
                    ah_gaps = scan_afterhours_gaps(tickers[:100], min_gap=0.03)
                    
                    for gap in ah_gaps[:10]:
                        logger.info(f"  🔥 {gap.ticker}: gap={gap.gap_pct*100:+.1f}%, vol={gap.volume:,}")
                    
                except Exception as e:
                    logger.error(f"Extended Hours scan failed: {e}", exc_info=True)
                    ah_gaps = []
                
                # === STEP 3: Options Flow (on high-priority tickers) ===
                try:
                    # Combine tickers from news + gaps
                    high_priority = list(set(
                        list(ticker_events.keys())[:20] + 
                        [g.ticker for g in ah_gaps[:10]]
                    ))
                    
                    if high_priority:
                        logger.info(f"📊 Step 3: Options Flow on {len(high_priority)} high-priority...")
                        options_signals = scan_options_flow(high_priority)
                        
                        for ticker, signals in options_signals.items():
                            for sig in signals:
                                if sig.score >= 0.5:
                                    logger.info(f"  📉 {ticker}: {sig.signal_type} (score: {sig.score:.2f})")
                    
                except Exception as e:
                    logger.error(f"Options Flow scan failed: {e}", exc_info=True)
                
                # === STEP 4: Anticipation Engine (IBKR radar + Grok+Polygon) ===
                try:
                    logger.info("🎯 Step 4: Anticipation Engine...")
                    results = run_anticipation_scan(tickers, mode="afterhours")
                    
                    # Send alerts for new BUY signals
                    for signal_dict in results.get("new_signals", []):
                        if signal_dict.get("signal_level") in ["BUY", "BUY_STRONG"]:
                            # Add extended hours boost to notes
                            boost, boost_details = get_extended_hours_boost(signal_dict["ticker"])
                            
                            send_signal_alert({
                                "ticker": signal_dict["ticker"],
                                "signal": signal_dict["signal_level"],
                                "monster_score": signal_dict.get("combined_score", 0),
                                "notes": f"🎯 ANTICIPATION ({signal_dict.get('urgency', 'N/A')})\n"
                                        f"Catalyst: {signal_dict.get('catalyst_type', 'N/A')}\n"
                                        f"AH Boost: +{boost:.2f}\n"
                                        f"{signal_dict.get('catalyst_summary', '')[:100]}"
                            })
                            logger.info(f"🚨 ANTICIPATION ALERT: {signal_dict['ticker']} - {signal_dict['signal_level']}")
                        
                        elif signal_dict.get("signal_level") == "WATCH_EARLY":
                            logger.info(f"👀 WATCH_EARLY: {signal_dict['ticker']} (score: {signal_dict.get('combined_score', 0):.2f})")
                    
                    # Log engine status
                    status = get_engine_status()
                    logger.info(f"📊 Engine: {status['suspects_count']} suspects, {status['watch_signals_count']} signals, {status['grok_remaining']} Grok calls left")
                    
                except Exception as e:
                    logger.error(f"Anticipation scan failed: {e}", exc_info=True)
                
                # Also run legacy after-hours scanner (for backward compatibility)
                run_afterhours_scanner(tickers=tickers[:50])
                
                time.sleep(600)  # 10 min in after-hours (more frequent for anticipation)

            # ---- Trading sessions ----
            elif is_premarket():
                logger.info("🌅 PRE-MARKET session - CONFIRMATION MODE")
                
                # Continue anticipation scanning
                universe = load_universe()
                tickers = universe["ticker"].tolist() if universe is not None else []
                
                try:
                    # Run anticipation scan in premarket mode
                    results = run_anticipation_scan(tickers, mode="premarket")
                    
                    # Check for signal upgrades (WATCH_EARLY -> BUY)
                    upgrades = results.get("upgrades", [])
                    
                    for upgrade in upgrades:
                        send_signal_alert({
                            "ticker": upgrade["ticker"],
                            "signal": upgrade.get("signal_level", "BUY"),
                            "monster_score": upgrade.get("combined_score", 0),
                            "notes": f"⬆️ UPGRADED from WATCH_EARLY\n"
                                    f"Catalyst: {upgrade.get('catalyst_type', 'N/A')}\n"
                                    f"{upgrade.get('catalyst_summary', '')[:100]}"
                        })
                        logger.info(f"🚀 UPGRADE ALERT: {upgrade['ticker']} → {upgrade.get('signal_level')}")
                    
                    # Log status
                    status = get_engine_status()
                    logger.info(f"📊 PM Engine: {status['watch_signals_count']} signals active")
                    
                except Exception as e:
                    logger.error(f"PM anticipation failed: {e}", exc_info=True)
                
                # Run regular edge cycle
                edge_cycle()
                time.sleep(300)  # every 5 min

            elif is_market_open():
                logger.info("📈 REGULAR MARKET session")
                
                # Clear old anticipation signals
                clear_expired_signals(max_age_hours=24)
                
                # Run regular edge cycle
                edge_cycle()
                time.sleep(180)  # every 3 min

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
    run_system_guardian()


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
