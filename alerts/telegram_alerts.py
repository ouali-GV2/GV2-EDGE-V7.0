"""
GV2-EDGE V7.0 — Telegram Alerts System
=======================================

Alertes enrichies avec:
- V7.0 Architecture (SignalProducer -> OrderComputer -> ExecutionGate)
- UnifiedSignal support (detection always visible, execution gated)
- MRP/EP Context (Market Memory) badges
- Pre-Halt Engine alerts
- IBKR News Trigger integration
- Risk Guard status (dilution, compliance, halt)
- Blocked signals notification (full transparency)
- EVENT_TYPE emojis (18 types, 5 tiers)

Architecture V7 Detection/Execution Separation
"""

import requests
from typing import Dict, Any, Optional, List

from utils.logger import get_logger
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("TELEGRAM_ALERTS")

TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# ============================
# V6 EVENT TYPE TAXONOMY
# ============================

EVENT_TYPE_EMOJI = {
    # TIER 1 - CRITICAL IMPACT (0.90-1.00)
    "FDA_APPROVAL": "\U0001F3C6",       # Trophy
    "PDUFA_DECISION": "\U0001F4C5",     # Calendar
    "BUYOUT_CONFIRMED": "\U0001F4B0",   # Money bag

    # TIER 2 - HIGH IMPACT (0.75-0.89)
    "FDA_TRIAL_POSITIVE": "\U00002705", # Green check
    "BREAKTHROUGH_DESIGNATION": "\U0001F31F", # Star
    "FDA_FAST_TRACK": "\U0001F680",     # Rocket
    "MERGER_ACQUISITION": "\U0001F91D", # Handshake
    "EARNINGS_BEAT_BIG": "\U0001F4C8",  # Chart up
    "MAJOR_CONTRACT": "\U0001F4DD",     # Contract

    # TIER 3 - MODERATE IMPACT (0.60-0.74)
    "GUIDANCE_RAISE": "\U0001F4CA",     # Bar chart
    "EARNINGS_BEAT": "\U0001F4B5",      # Dollar
    "PARTNERSHIP": "\U0001F517",        # Link
    "PRICE_TARGET_RAISE": "\U0001F3AF", # Target

    # TIER 4 - LOW-MODERATE IMPACT (0.45-0.59)
    "ANALYST_UPGRADE": "\U0001F4DD",    # Note
    "SHORT_SQUEEZE_SIGNAL": "\U0001F4A5", # Collision
    "UNUSUAL_VOLUME_NEWS": "\U0001F4CA", # Chart

    # TIER 5 - SPECULATIVE (0.30-0.44)
    "BUYOUT_RUMOR": "\U0001F914",       # Thinking
    "SOCIAL_MEDIA_SURGE": "\U0001F4F1", # Phone
    "BREAKING_POSITIVE": "\U0001F4F0",  # Newspaper

    # Legacy mappings
    "earnings": "\U0001F4B5",
    "fda": "\U0001F3E5",
    "default": "\U0001F4E2"             # Megaphone
}

TIER_LABELS = {
    1: "\U0001F534 TIER 1 - CRITICAL",    # Red circle
    2: "\U0001F7E0 TIER 2 - HIGH",        # Orange circle
    3: "\U0001F7E1 TIER 3 - MODERATE",    # Yellow circle
    4: "\U0001F7E2 TIER 4 - LOW-MOD",     # Green circle
    5: "\U0001F535 TIER 5 - SPECULATIVE"  # Blue circle
}

# Impact to tier mapping
IMPACT_TO_TIER = {
    (0.90, 1.00): 1,
    (0.75, 0.89): 2,
    (0.60, 0.74): 3,
    (0.45, 0.59): 4,
    (0.30, 0.44): 5
}


# ============================
# Helper Functions
# ============================

def get_event_emoji(event_type: str) -> str:
    """Get emoji for event type"""
    return EVENT_TYPE_EMOJI.get(event_type, EVENT_TYPE_EMOJI["default"])


def get_tier_from_impact(impact: float) -> int:
    """Get tier number from impact score"""
    if impact >= 0.90:
        return 1
    elif impact >= 0.75:
        return 2
    elif impact >= 0.60:
        return 3
    elif impact >= 0.45:
        return 4
    else:
        return 5


def get_tier_label(tier: int) -> str:
    """Get tier label with emoji"""
    return TIER_LABELS.get(tier, TIER_LABELS[5])


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for NLP sentiment"""
    sentiment_map = {
        "VERY_BULLISH": "\U0001F680\U0001F680",    # Double rocket
        "BULLISH": "\U0001F680",                   # Rocket
        "SLIGHTLY_BULLISH": "\U0001F4C8",          # Chart up
        "NEUTRAL": "\U00002796",                   # Minus
        "SLIGHTLY_BEARISH": "\U0001F4C9",          # Chart down
        "BEARISH": "\U0001F4A9",                   # Poop
        "VERY_BEARISH": "\U0001F4A9\U0001F4A9"     # Double poop
    }
    return sentiment_map.get(sentiment, "\U00002753")  # Question mark


def get_signal_emoji(signal_type: str) -> str:
    """Get emoji for signal type"""
    signal_map = {
        "BUY_STRONG": "\U0001F525\U0001F680",     # Fire + Rocket
        "BUY": "\U00002705",                       # Green check
        "WATCH_EARLY": "\U0001F440",              # Eyes
        "WATCH": "\U0001F50D",                     # Magnifying glass
        "AVOID": "\U0001F6AB"                      # No entry
    }
    return signal_map.get(signal_type, "\U0001F4E2")


# ============================
# Send message
# ============================

def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to Telegram

    Args:
        text: Message text
        parse_mode: "Markdown" or "HTML"

    Returns:
        Success status
    """
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode
        }

        r = requests.post(TELEGRAM_URL, json=payload, timeout=5)

        if r.status_code != 200:
            logger.warning(f"Telegram error: {r.text}")
            return False

        return True

    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ============================
# V7 Signal Alert (Enhanced)
# ============================

def send_signal_alert(
    signal: Dict[str, Any],
    position: Optional[Dict[str, Any]] = None,
    v7_data: Optional[Dict[str, Any]] = None
):
    """
    Send enhanced V7.0 signal alert

    Supports both legacy dict format and V7 UnifiedSignal fields.
    V7 shows ALL signals (detection never blocked) with execution status.

    Args:
        signal: Signal data (ticker, signal, monster_score, confidence, notes)
        position: Optional position data (entry, stop, shares, risk_amount)
        v7_data: Optional V7 enrichment data:
            - pre_halt_state: Pre-halt risk state
            - context_mrp: MRP score (0-100)
            - context_ep: EP score (0-100)
            - context_active: Boolean if MRP/EP active
            - execution_blocked: Boolean if execution blocked
            - block_reasons: List of block reasons
            - risk_flags: Dict of risk assessments
    """
    ticker = signal.get('ticker', 'UNKNOWN')
    signal_type = signal.get('signal', signal.get('signal_type', 'BUY'))
    monster_score = signal.get('monster_score', 0)
    confidence = signal.get('confidence', 0)
    notes = signal.get('notes', '')

    # Check if signal is blocked (V7)
    is_blocked = "(BLOCKED)" in signal_type or signal.get('blocked', False)
    clean_signal = signal_type.replace(" (BLOCKED)", "")

    # Signal emoji
    signal_emoji = get_signal_emoji(clean_signal)
    if is_blocked:
        signal_emoji = "\U0001F6AB"  # No entry

    # Build header
    msg = f"""
{signal_emoji} *GV2-EDGE V7.0 SIGNAL*

\U0001F4CA Ticker: `{ticker}`
\U000026A1 Signal: *{signal_type}*
\U0001F3AF Monster Score: `{monster_score:.2f}`
"""

    if confidence > 0:
        msg += f"\U0001F4C8 Confidence: `{confidence:.2f}`\n"

    # V7 enrichments
    if v7_data:
        pre_halt = v7_data.get('pre_halt_state', 'NORMAL')
        context_mrp = v7_data.get('context_mrp')
        context_ep = v7_data.get('context_ep')
        context_active = v7_data.get('context_active', False)
        block_reasons = v7_data.get('block_reasons', [])
        risk_flags = v7_data.get('risk_flags', {})

        msg += "\n*--- V7 Intelligence ---*\n"

        # Pre-Halt State
        if pre_halt != 'NORMAL':
            halt_emoji = "\U0001F534" if pre_halt == 'HIGH' else "\U0001F7E0"
            msg += f"{halt_emoji} Pre-Halt: `{pre_halt}`\n"

        # MRP/EP Context (Market Memory)
        if context_active and context_mrp is not None:
            msg += f"\U0001F4CA MRP: `{context_mrp:.0f}` | EP: `{context_ep:.0f}`\n"

        # Risk Flags
        if risk_flags:
            dilution = risk_flags.get('dilution_risk', 'LOW')
            compliance = risk_flags.get('compliance_risk', 'LOW')
            if dilution != 'LOW' or compliance != 'LOW':
                msg += f"\U000026A0 Risk: Dilution={dilution}, Compliance={compliance}\n"

        # Block reasons
        if block_reasons:
            msg += f"\U0001F6AB Blocked: `{', '.join(block_reasons)}`\n"

    # Legacy V6 data support
    elif 'event_type' in signal or 'catalyst_score' in signal:
        event_type = signal.get('event_type')
        event_impact = signal.get('event_impact', 0)

        if event_type:
            event_emoji = get_event_emoji(event_type)
            tier = get_tier_from_impact(event_impact)
            tier_label = get_tier_label(tier)
            msg += f"\n{event_emoji} Event: `{event_type}`\n"
            msg += f"\U0001F3C5 {tier_label}\n"

    # Notes (from main.py)
    if notes:
        msg += f"\n\U0001F4DD `{notes}`\n"

    # Position info
    if position:
        msg += f"""
*--- Position ---*
\U0001F4B0 Entry: `{position.get('entry', 'N/A')}`
\U0001F6D1 Stop: `{position.get('stop', 'N/A')}`
\U0001F4E6 Shares: `{position.get('shares', 'N/A')}`
\U00002696 Risk: `${position.get('risk_amount', 'N/A')}`
"""

    send_message(msg)


# ============================
# Pre-Spike Radar Alert
# ============================

def send_pre_spike_alert(
    ticker: str,
    signals_count: int,
    signals_detail: Dict[str, bool],
    acceleration_score: float,
    monster_score: float
):
    """
    Send Pre-Spike Radar alert when multiple signals detected

    Args:
        ticker: Stock ticker
        signals_count: Number of signals (1-4)
        signals_detail: Dict of signal name -> active status
        acceleration_score: Overall acceleration score
        monster_score: Current monster score
    """
    # Alert level based on signals
    if signals_count >= 4:
        alert_level = "\U0001F534\U0001F534\U0001F534 IMMINENT"  # Red x3
    elif signals_count >= 3:
        alert_level = "\U0001F534\U0001F534 HIGH"              # Red x2
    elif signals_count >= 2:
        alert_level = "\U0001F7E0 MODERATE"                    # Orange
    else:
        alert_level = "\U0001F7E1 WATCH"                       # Yellow

    msg = f"""
\U0001F4E1 *PRE-SPIKE RADAR ALERT*

\U0001F4CA Ticker: `{ticker}`
{alert_level} ({signals_count}/4 signals)

*Active Signals:*
"""

    # Signal details with checkmarks
    signal_names = {
        'volume_acceleration': '\U0001F4C8 Volume Acceleration',
        'bid_ask_tightening': '\U0001F4B1 Bid-Ask Tightening',
        'price_compression': '\U0001F5DC Price Compression',
        'dark_pool_activity': '\U0001F5A5 Dark Pool Activity'
    }

    for signal_key, display_name in signal_names.items():
        is_active = signals_detail.get(signal_key, False)
        status = "\U00002705" if is_active else "\U0000274C"
        msg += f"{status} {display_name}\n"

    msg += f"""
\U0001F680 Acceleration Score: `{acceleration_score:.2f}`
\U0001F3AF Monster Score: `{monster_score:.2f}`

\U000023F0 *ACTION: Monitor closely for entry*
"""

    send_message(msg)


# ============================
# Catalyst Alert
# ============================

def send_catalyst_alert(
    ticker: str,
    catalyst_type: str,
    catalyst_score: float,
    source: str,
    headline: str,
    tier: int,
    confluence_count: int = 1
):
    """
    Send Catalyst Score V3 alert for significant catalysts

    Args:
        ticker: Stock ticker
        catalyst_type: Type from unified taxonomy
        catalyst_score: Calculated catalyst score
        source: News source
        headline: News headline
        tier: Catalyst tier (1-5)
        confluence_count: Number of concurrent catalysts
    """
    event_emoji = get_event_emoji(catalyst_type)
    tier_label = get_tier_label(tier)

    # Urgency based on tier
    urgency_map = {
        1: "\U0001F6A8 CRITICAL CATALYST",
        2: "\U0001F525 HOT CATALYST",
        3: "\U0001F4E2 CATALYST DETECTED",
        4: "\U0001F4E1 MINOR CATALYST",
        5: "\U0001F50D SPECULATIVE CATALYST"
    }
    urgency = urgency_map.get(tier, urgency_map[5])

    msg = f"""
{event_emoji} *{urgency}*

\U0001F4CA Ticker: `{ticker}`
\U0001F3F7 Type: `{catalyst_type}`
{tier_label}

\U0001F9EA Catalyst Score: `{catalyst_score:.2f}`
"""

    if confluence_count > 1:
        msg += f"\U0001F4A5 CONFLUENCE: `{confluence_count}` concurrent catalysts!\n"

    msg += f"""
\U0001F4F0 Source: `{source}`
\U0001F4DD _{headline[:100]}..._
"""

    send_message(msg)


# ============================
# NLP Sentiment Alert
# ============================

def send_nlp_sentiment_alert(
    ticker: str,
    sentiment_direction: str,
    sentiment_score: float,
    news_count: int,
    dominant_category: str,
    urgency_level: str
):
    """
    Send NLP Enrichi sentiment summary alert

    Args:
        ticker: Stock ticker
        sentiment_direction: Sentiment direction from NLP Enrichi
        sentiment_score: Aggregated sentiment score
        news_count: Number of news analyzed
        dominant_category: Most common news category
        urgency_level: NEWS urgency level
    """
    sentiment_emoji = get_sentiment_emoji(sentiment_direction)

    # Urgency emoji
    urgency_emojis = {
        "BREAKING": "\U0001F6A8",
        "HIGH": "\U0001F534",
        "MODERATE": "\U0001F7E0",
        "NORMAL": "\U0001F7E2",
        "LOW": "\U0001F535"
    }
    urgency_emoji = urgency_emojis.get(urgency_level, "\U0001F7E2")

    msg = f"""
{sentiment_emoji} *NLP SENTIMENT ALERT*

\U0001F4CA Ticker: `{ticker}`
\U0001F9E0 Sentiment: *{sentiment_direction}*
\U0001F4CA Score: `{sentiment_score:.2f}`

{urgency_emoji} Urgency: `{urgency_level}`
\U0001F4F0 News Analyzed: `{news_count}`
\U0001F3F7 Category: `{dominant_category}`
"""

    send_message(msg)


# ============================
# Repeat Gainer Alert
# ============================

def send_repeat_gainer_alert(
    ticker: str,
    past_spikes: int,
    avg_spike_pct: float,
    last_spike_date: str,
    volatility_score: float,
    monster_score: float
):
    """
    Send Repeat Gainer Memory alert

    Args:
        ticker: Stock ticker
        past_spikes: Number of historical spikes
        avg_spike_pct: Average spike percentage
        last_spike_date: Date of last spike
        volatility_score: Current volatility score from memory
        monster_score: Current monster score
    """
    # Badge level based on past spikes
    if past_spikes >= 5:
        badge = "\U0001F451 SERIAL RUNNER"      # Crown
    elif past_spikes >= 3:
        badge = "\U0001F525 HOT REPEAT"         # Fire
    else:
        badge = "\U0001F501 KNOWN MOVER"        # Repeat

    msg = f"""
\U0001F501 *REPEAT GAINER DETECTED*

\U0001F4CA Ticker: `{ticker}`
{badge}

\U0001F4C8 Historical Spikes: `{past_spikes}`
\U0001F4B9 Avg Spike: `+{avg_spike_pct:.1f}%`
\U0001F4C5 Last Spike: `{last_spike_date}`
\U0001F30B Volatility Score: `{volatility_score:.2f}`

\U0001F3AF Current Monster Score: `{monster_score:.2f}`

\U000026A0 *Known for explosive moves - size appropriately*
"""

    send_message(msg)


# ============================
# Daily Audit Summary Alert
# ============================

def send_daily_audit_alert(report: Dict[str, Any]):
    """
    Send daily audit summary via Telegram

    Args:
        report: Audit report dict with summary, hit_analysis, miss_analysis, etc.
    """
    summary = report.get("summary", {})
    grade = report.get("performance_grade", "?")
    audit_date = report.get("audit_date", "N/A")

    # Grade emojis
    grade_emoji = {
        "A": "\U0001F3C6",  # Trophy
        "B": "\U00002705",  # Check
        "C": "\U000026A0", # Warning
        "D": "\U0001F7E0",  # Orange
        "F": "\U0000274C"   # X
    }

    hit_rate = summary.get('hit_rate', 0)
    early_catch = summary.get('early_catch_rate', 0)
    miss_rate = summary.get('miss_rate', 0)
    fp_count = summary.get('fp_count', 0)
    avg_lead = summary.get('avg_lead_time_hours', 0)

    msg = f"""
\U0001F4CA *GV2-EDGE V7.0 DAILY AUDIT*
\U0001F4C5 Date: `{audit_date}`

{grade_emoji.get(grade, '\U00002753')} *Performance Grade: {grade}*

*--- Core Metrics ---*
\U0001F4C8 Hit Rate: `{hit_rate*100:.1f}%`
\U000023F1 Early Catches: `{early_catch*100:.1f}%`
\U000023F0 Avg Lead Time: `{avg_lead:.1f}h`
\U0000274C Miss Rate: `{miss_rate*100:.1f}%`
\U0001F3AF False Positives: `{fp_count}`
"""

    # V7 Module Stats
    v7_stats = report.get("v7_stats", report.get("v6_stats", {}))
    if v7_stats:
        msg += "\n*--- V7 Module Performance ---*\n"

        if 'signals_produced' in v7_stats:
            msg += f"\U0001F4E1 Signals Produced: `{v7_stats['signals_produced']}`\n"

        if 'signals_blocked' in v7_stats:
            msg += f"\U0001F6AB Signals Blocked: `{v7_stats['signals_blocked']}`\n"

        if 'mrp_ep_active' in v7_stats:
            msg += f"\U0001F9E0 MRP/EP Active: `{v7_stats['mrp_ep_active']}`\n"

        if 'pre_halt_triggers' in v7_stats:
            msg += f"\U000026A0 Pre-Halt Triggers: `{v7_stats['pre_halt_triggers']}`\n"

        if 'risk_guard_blocks' in v7_stats:
            msg += f"\U0001F6E1 Risk Guard Blocks: `{v7_stats['risk_guard_blocks']}`\n"

    # Top hits
    hit_analysis = report.get("hit_analysis", {})
    hits = hit_analysis.get("hits", [])[:3]
    if hits:
        msg += "\n*\U0001F3AF TOP HITS:*\n"
        for hit in hits:
            msg += f"  \U00002022 `{hit['ticker']}`: +{hit.get('gainer_change_pct', 0):.0f}% (lead: {hit.get('lead_time_hours', 0):.1f}h)\n"

    # Top misses
    miss_analysis = report.get("miss_analysis", {})
    misses = miss_analysis.get("missed_tickers", [])[:3]
    if misses:
        msg += f"\n*\U0000274C TOP MISSES:* {', '.join(misses)}"

    send_message(msg)


# ============================
# Weekly Audit Summary Alert
# ============================

def send_weekly_audit_alert(report: Dict[str, Any]):
    """
    Send weekly audit summary via Telegram

    Args:
        report: Weekly audit report dict
    """
    period = report.get("period", {})
    metrics = report.get("metrics", {})
    recommendations = report.get("recommendations", [])

    trend = metrics.get("trend", "stable")
    trend_emoji = "\U0001F4C8" if trend == "improving" else "\U0001F4C9" if trend == "declining" else "\U00002796"

    msg = f"""
\U0001F4CA *GV2-EDGE V7.0 WEEKLY AUDIT*
\U0001F4C5 Period: `{period.get('start', 'N/A')}` to `{period.get('end', 'N/A')}`
\U0001F4C6 Days with data: `{period.get('days_with_data', 0)}`

*--- Weekly Averages ---*
\U0001F4C8 Avg Hit Rate: `{metrics.get('avg_hit_rate', 0)*100:.1f}%`
\U000023F1 Avg Early Catch: `{metrics.get('avg_early_catch_rate', 0)*100:.1f}%`
\U000023F0 Avg Lead Time: `{metrics.get('avg_lead_time_hours', 0):.1f}h`
\U0000274C Avg Miss Rate: `{metrics.get('avg_miss_rate', 0)*100:.1f}%`
\U0001F3AF Total FPs: `{metrics.get('total_false_positives', 0)}`

{trend_emoji} *Trend: {trend.upper()}*
"""

    # Daily grades
    daily_grades = report.get("daily_grades", [])
    if daily_grades:
        grades_str = " ".join(daily_grades)
        msg += f"\n\U0001F4CA Daily Grades: `{grades_str}`\n"

    # Recommendations
    if recommendations:
        msg += "\n*\U0001F4A1 RECOMMENDATIONS:*\n"
        for rec in recommendations[:3]:
            action = rec.get('action', '').upper()
            component = rec.get('component', '')
            reason = rec.get('reason', '')
            msg += f"  \U00002022 {action} `{component}`: _{reason}_\n"

    send_message(msg)


# ============================
# System Alert
# ============================

def send_system_alert(text: str, level: str = "info"):
    """
    Send system alert

    Args:
        text: Alert message
        level: "info", "warning", "error", "critical"
    """
    level_config = {
        "info": ("\U0001F535", "INFO"),
        "warning": ("\U000026A0", "WARNING"),
        "error": ("\U0001F534", "ERROR"),
        "critical": ("\U0001F6A8", "CRITICAL")
    }

    emoji, label = level_config.get(level, level_config["info"])

    msg = f"{emoji} *GV2-EDGE V7.0 {label}*\n\n{text}"
    send_message(msg)


# ============================
# Market Session Alert
# ============================

def send_session_alert(session: str, active_signals: int = 0):
    """
    Send market session transition alert

    Args:
        session: Session name (PREMARKET, RTH, AFTER_HOURS, CLOSED)
        active_signals: Number of active signals
    """
    session_config = {
        "PREMARKET": ("\U0001F305", "Pre-Market Open", "Scanning for overnight catalysts..."),
        "RTH": ("\U0001F4C8", "Market Open", "Regular trading hours active"),
        "AFTER_HOURS": ("\U0001F319", "After-Hours", "Extended hours monitoring"),
        "CLOSED": ("\U0001F4A4", "Market Closed", "System in standby mode")
    }

    emoji, label, desc = session_config.get(session, ("\U00002753", session, ""))

    msg = f"""
{emoji} *{label}*

{desc}
\U0001F4CA Active Signals: `{active_signals}`
"""

    send_message(msg)


# ============================
# Pre-Halt Alert (V7)
# ============================

def send_pre_halt_alert(
    ticker: str,
    pre_halt_state: str,
    risk_level: str,
    recommendation: str,
    size_multiplier: float
):
    """
    Send Pre-Halt Engine alert when halt risk detected

    Args:
        ticker: Stock ticker
        pre_halt_state: NORMAL, ELEVATED, HIGH
        risk_level: Overall risk assessment
        recommendation: EXECUTE, WAIT, REDUCE, BLOCKED
        size_multiplier: Position size adjustment (0.0-1.0)
    """
    state_emoji = {
        "NORMAL": "\U0001F7E2",
        "ELEVATED": "\U0001F7E0",
        "HIGH": "\U0001F534"
    }

    rec_emoji = {
        "EXECUTE": "\U00002705",
        "WAIT": "\U000023F0",
        "REDUCE": "\U000026A0",
        "BLOCKED": "\U0001F6AB"
    }

    msg = f"""
{state_emoji.get(pre_halt_state, '\U00002753')} *PRE-HALT ENGINE ALERT*

\U0001F4CA Ticker: `{ticker}`
\U000026A0 Halt Risk: *{pre_halt_state}*

{rec_emoji.get(recommendation, '\U00002753')} Recommendation: `{recommendation}`
\U0001F4CA Size Multiplier: `{size_multiplier*100:.0f}%`

*Monitor closely - halt risk elevated*
"""

    send_message(msg)


# ============================
# News Trigger Alert (V7)
# ============================

def send_news_trigger_alert(
    ticker: str,
    headline: str,
    trigger_type: str,
    trigger_level: str,
    actions: List[str]
):
    """
    Send IBKR News Trigger alert for early news detection

    Args:
        ticker: Stock ticker
        headline: News headline
        trigger_type: HALT_RISK, CATALYST_MAJOR, SPIKE_FORMING, RISK_ALERT
        trigger_level: CRITICAL, HIGH, MEDIUM, LOW
        actions: Recommended actions list
    """
    level_emoji = {
        "CRITICAL": "\U0001F6A8",
        "HIGH": "\U0001F534",
        "MEDIUM": "\U0001F7E0",
        "LOW": "\U0001F7E1"
    }

    type_emoji = {
        "HALT_RISK": "\U000026D4",
        "CATALYST_MAJOR": "\U0001F4A5",
        "SPIKE_FORMING": "\U0001F4C8",
        "RISK_ALERT": "\U000026A0"
    }

    msg = f"""
{level_emoji.get(trigger_level, '\U0001F4F0')} *IBKR NEWS TRIGGER*

\U0001F4CA Ticker: `{ticker}`
{type_emoji.get(trigger_type, '\U0001F4F0')} Type: `{trigger_type}`
Level: *{trigger_level}*

\U0001F4F0 _{headline[:150]}_

*Actions:*
"""

    for action in actions[:3]:
        msg += f"\U00002022 {action}\n"

    send_message(msg)


# ============================
# IBKR Connection Alert (V7.1)
# ============================

def send_ibkr_connection_alert(status: str, details: dict = None):
    """
    Send IBKR connection status alert.

    Args:
        status: "disconnected", "reconnected", "failed"
        details: Optional dict with downtime_seconds, reconnections, etc.
    """
    details = details or {}

    if status == "disconnected":
        msg = (
            "\U0001F50C *IBKR DISCONNECTED*\n\n"
            "Connection lost. Attempting automatic reconnection...\n"
            "\U000023F3 Backoff: 0s, 2s, 5s, 15s, 30s (5 attempts max)"
        )

    elif status == "reconnected":
        downtime = details.get("downtime_seconds", 0)
        total_reconnections = details.get("reconnections", 0)

        if downtime < 60:
            downtime_str = f"{downtime:.0f}s"
        else:
            downtime_str = f"{downtime / 60:.1f}min"

        msg = (
            f"\U00002705 *IBKR RECONNECTED*\n\n"
            f"\U000023F1 Downtime: `{downtime_str}`\n"
            f"\U0001F504 Total reconnections: `{total_reconnections}`\n\n"
            f"Data feed restored."
        )

    elif status == "failed":
        total_disconnections = details.get("disconnections", 0)
        msg = (
            f"\U0001F6A8 *IBKR CONNECTION FAILED*\n\n"
            f"5 reconnection attempts exhausted.\n"
            f"\U0001F504 Total disconnections: `{total_disconnections}`\n\n"
            f"\U000026A0 *Fallback: Finnhub data active*\n"
            f"Manual intervention may be required."
        )

    else:
        msg = f"\U0001F50C *IBKR {status.upper()}*"

    send_message(msg)


# ============================
# V9 ENRICHED SIGNAL ALERT (A17)
# ============================

def send_enriched_signal_alert(
    ticker: str,
    signal_type: str,
    monster_score: float,
    entry: float = 0,
    stop: float = 0,
    target: float = 0,
    catalyst_type: str = "",
    catalyst_detail: str = "",
    float_info: Dict = None,
    radar_state: str = "",
    radar_details: str = "",
    patterns: List[str] = None,
    room_to_resistance: float = 0,
    resistance_price: float = 0,
    mrp: float = 0,
    ep: float = 0,
    catalyst_segment: str = "",
    earnings_info: Dict = None,
    conference_info: Dict = None,
    squeeze_info: Dict = None,
    v7_data: Dict = None,
):
    """
    Alerte enrichie V9 avec contexte complet.

    Format:
      BUY_STRONG AAPL | Score: 0.85
      Entry: $150.00 | Stop: $147.50 | Target: $157.50
      Catalyst: EARNINGS BMO (beat 85% historique)
      Float: 15.2M | SI: 18.5% | DTC: 3.2
      Radar: LAUNCHING (vol z=3.2, price vel +0.5%/min)
      Pattern: VWAP Reclaim + HOD Break
      Room to resistance: 8.5% ($162.75)
      MRP: 0.78 | EP: 0.82 (earnings-specific)
    """
    patterns = patterns or []
    float_info = float_info or {}
    earnings_info = earnings_info or {}
    conference_info = conference_info or {}
    squeeze_info = squeeze_info or {}
    v7_data = v7_data or {}

    # Signal header
    emoji = get_signal_emoji(signal_type)
    msg = f"{emoji} *{signal_type}* `{ticker}` | Score: `{monster_score:.2f}`\n"

    # Entry / Stop / Target
    if entry > 0:
        parts = [f"Entry: `${entry:.2f}`"]
        if stop > 0:
            parts.append(f"Stop: `${stop:.2f}`")
        if target > 0:
            parts.append(f"Target: `${target:.2f}`")
        msg += "\U0001F3AF " + " | ".join(parts) + "\n"

    # Catalyst
    if catalyst_type:
        cat_emoji = get_event_emoji(catalyst_type)
        cat_line = f"{cat_emoji} Catalyst: `{catalyst_type}`"
        if catalyst_detail:
            cat_line += f" ({catalyst_detail})"
        msg += cat_line + "\n"

    # Earnings info
    if earnings_info:
        timing = earnings_info.get("timing", "")
        beat_rate = earnings_info.get("beat_rate", 0)
        days = earnings_info.get("days_until", "")
        msg += f"\U0001F4C8 Earnings: `{timing}` J-{days} (beat {beat_rate:.0%})\n"

    # Conference info
    if conference_info:
        conf_name = conference_info.get("conference", "")
        conf_status = conference_info.get("status", "")
        if conf_name:
            msg += f"\U0001F3E2 Conference: `{conf_name}` ({conf_status})\n"

    # Float / SI / Squeeze
    if float_info:
        float_m = float_info.get("float_shares", 0) / 1_000_000 if float_info.get("float_shares") else 0
        si_pct = float_info.get("short_pct_float", 0)
        dtc = float_info.get("days_to_cover", 0)
        borrow = float_info.get("borrow_status", "")

        parts = []
        if float_m > 0:
            parts.append(f"Float: `{float_m:.1f}M`")
        if si_pct > 0:
            parts.append(f"SI: `{si_pct:.1f}%`")
        if dtc > 0:
            parts.append(f"DTC: `{dtc:.1f}`")
        if borrow and borrow != "UNKNOWN":
            parts.append(f"CTB: `{borrow}`")

        if parts:
            msg += "\U0001F4CA " + " | ".join(parts) + "\n"

    # Squeeze info
    if squeeze_info and squeeze_info.get("squeeze_score", 0) > 0.3:
        sq_score = squeeze_info.get("squeeze_score", 0)
        sq_signals = squeeze_info.get("signals", [])
        msg += f"\U0001F4A5 Squeeze: `{sq_score:.2f}` ({', '.join(sq_signals[:3])})\n"

    # Radar state
    if radar_state and radar_state != "DORMANT":
        msg += f"\U0001F6F0 Radar: `{radar_state}`"
        if radar_details:
            msg += f" ({radar_details})"
        msg += "\n"

    # Patterns
    if patterns:
        pattern_str = " + ".join(patterns[:4])
        msg += f"\U0001F4D0 Pattern: `{pattern_str}`\n"

    # Room to resistance
    if room_to_resistance > 0:
        msg += f"\U0001F4CF Room: `{room_to_resistance:.1f}%`"
        if resistance_price > 0:
            msg += f" (`${resistance_price:.2f}`)"
        msg += "\n"

    # MRP / EP
    if mrp > 0 or ep > 0:
        mrp_ep = f"\U0001F9E0 MRP: `{mrp:.0f}` | EP: `{ep:.0f}`"
        if catalyst_segment:
            mrp_ep += f" ({catalyst_segment})"
        msg += mrp_ep + "\n"

    # Risk badges from v7_data
    risk_flags = v7_data.get("risk_flags", {})
    if risk_flags:
        badges = []
        if risk_flags.get("dilution_risk", "LOW") not in ("LOW",):
            badges.append(f"DIL:{risk_flags['dilution_risk']}")
        if risk_flags.get("compliance_risk", "LOW") not in ("LOW",):
            badges.append(f"CMP:{risk_flags['compliance_risk']}")
        if badges:
            msg += "\U000026A0 Risk: " + " | ".join(badges) + "\n"

    # Block info
    block_reasons = v7_data.get("block_reasons", [])
    if block_reasons:
        msg += f"\U0001F6AB Blocked: `{', '.join(block_reasons[:3])}`\n"

    send_message(msg)


def send_multi_radar_alert(
    ticker: str,
    signal_type: str,
    final_score: float,
    agreement: str,
    lead_radar: str,
    radar_scores: Dict[str, float] = None,
    active_radars: List[str] = None,
):
    """
    Alerte Multi-Radar V9 — resultat de la confluence des 4 radars.
    """
    radar_scores = radar_scores or {}
    active_radars = active_radars or []

    emoji = get_signal_emoji(signal_type)

    msg = f"{emoji} *MULTI-RADAR* `{ticker}` | `{signal_type}`\n"
    msg += f"\U0001F3AF Score: `{final_score:.2f}` | Agreement: `{agreement}`\n"
    msg += f"\U0001F451 Lead: `{lead_radar}` | Active: `{len(active_radars)}/4`\n\n"

    # Radar breakdown
    radar_emojis = {
        "flow": "\U0001F30A",      # Wave
        "catalyst": "\U0001F4A1",   # Light bulb
        "smart_money": "\U0001F4B0", # Money bag
        "sentiment": "\U0001F4AC",  # Speech
    }

    for name in ["flow", "catalyst", "smart_money", "sentiment"]:
        r_emoji = radar_emojis.get(name, "\U00002B50")
        score = radar_scores.get(name, 0)
        active = "\U00002705" if name in active_radars else "\U0000274C"
        msg += f"  {active} {r_emoji} `{name}`: `{score:.2f}`\n"

    send_message(msg)


# ============================
# Test Connection
# ============================

if __name__ == "__main__":
    # Test basic connection
    send_message("\U00002705 GV2-EDGE V7.0 Telegram connected")

    # Test V7 signal alert
    test_signal = {
        "ticker": "TEST",
        "signal": "BUY_STRONG",
        "monster_score": 0.85,
        "confidence": 0.92,
        "notes": "V7.0 | NORMAL"
    }

    test_v7_data = {
        "pre_halt_state": "NORMAL",
        "context_mrp": 72,
        "context_ep": 68,
        "context_active": True,
        "block_reasons": [],
        "risk_flags": {"dilution_risk": "LOW", "compliance_risk": "LOW"}
    }

    send_signal_alert(test_signal, v7_data=test_v7_data)

    # Test blocked signal
    test_blocked = {
        "ticker": "BLOCKED_TEST",
        "signal": "BUY (BLOCKED)",
        "monster_score": 0.72,
        "notes": "BLOCKED: DAILY_TRADE_LIMIT"
    }

    send_signal_alert(test_blocked)

    print("Test alerts sent!")
