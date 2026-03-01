"""
WATCH LIST - Calendar-Based Prediction
=======================================

Génère des signaux WATCH pour events imminents (J-7 à J-1)
permettant d'anticiper les movers AVANT leur explosion.

Concept:
- Scanner calendar events (earnings, FDA, etc.)
- Identifier high-probability setups
- Alert J-7 à J-1 pour préparation
- Upgrade to BUY/BUY_STRONG when technical confirms

Timeline:
J-7: WATCH signal (early warning)
J-3: WATCH upgraded (proximity boost)
J-1: May upgrade to BUY if setup OK
J-Day PM: BUY_STRONG (execution)
"""

from datetime import datetime, timedelta, timezone
import pandas as pd

from utils.logger import get_logger
from src.event_engine.event_hub import get_events
from src.feature_engine import compute_features
from src.pm_scanner import compute_pm_metrics

logger = get_logger("WATCH_LIST")


# ============================
# Calendar-based prediction
# ============================

def generate_watch_list(universe_tickers=None, days_forward=7, min_impact=0.7):
    """
    Generate WATCH list from upcoming calendar events
    
    Args:
        universe_tickers: List of tickers in universe
        days_forward: How many days ahead to scan (default 7)
        min_impact: Minimum event impact to include (default 0.7)
    
    Returns:
        List of watch signals with predictions
    """
    logger.info(f"Generating watch list for next {days_forward} days...")
    
    # Get all events
    events = get_events(universe_tickers=universe_tickers)
    
    if not events:
        logger.warning("No events found for watch list")
        return []
    
    today = datetime.now(timezone.utc).date()
    cutoff_date = today + timedelta(days=days_forward)
    
    watch_list = []
    
    for event in events:
        try:
            # Parse event date
            event_date_str = event.get("date", "")
            if not event_date_str:
                continue
            
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
            
            # Calculate days to event
            days_to_event = (event_date - today).days
            
            # Filter: within next X days (but not today/past)
            if days_to_event < 1 or days_to_event > days_forward:
                continue
            
            # Filter: high impact only
            impact = event.get("boosted_impact", event.get("impact", 0))
            if impact < min_impact:
                continue
            
            ticker = event.get("ticker", "")
            if not ticker:
                continue
            
            # Calculate probability score
            probability = calculate_probability(event, days_to_event)
            
            # Build watch signal
            watch_signal = {
                "ticker": ticker,
                "signal": "WATCH",
                "event_type": event.get("type", "unknown"),
                "event_date": event_date_str,
                "days_to_event": days_to_event,
                "impact": impact,
                "probability": probability,
                "reason": build_watch_reason(event, days_to_event, probability),
                "action": "Monitor for PM activity day-of"
            }
            
            # Add technical readiness if close to event
            if days_to_event <= 3:
                watch_signal["technical_check"] = check_technical_setup(ticker)
            
            watch_list.append(watch_signal)
        
        except Exception as e:
            logger.warning(f"Error processing event for watch list: {e}")
            continue
    
    # Sort by days_to_event (closest first) and impact
    watch_list.sort(key=lambda x: (x["days_to_event"], -x["impact"]))
    
    logger.info(f"Generated {len(watch_list)} WATCH signals")
    
    return watch_list


# ============================
# Probability calculation
# ============================

def calculate_probability(event, days_to_event):
    """
    Calculate probability of significant move based on event
    
    Factors:
    - Event type (earnings = high, partnership = medium)
    - Event impact score
    - Proximity (closer = higher certainty)
    - Historical context (if available)
    
    Returns:
        Probability score 0-1
    """
    base_probability = 0.5
    
    # Factor 1: Event type (aligned with unified taxonomy V6)
    event_type = event.get("type", "")
    type_multipliers = {
        # TIER 1 - CRITICAL
        "FDA_APPROVAL": 0.95,
        "PDUFA_DECISION": 0.92,
        "BUYOUT_CONFIRMED": 0.90,

        # TIER 2 - HIGH
        "FDA_TRIAL_POSITIVE": 0.85,
        "BREAKTHROUGH_DESIGNATION": 0.82,
        "FDA_FAST_TRACK": 0.80,
        "MERGER_ACQUISITION": 0.85,
        "EARNINGS_BEAT_BIG": 0.82,
        "MAJOR_CONTRACT": 0.78,

        # TIER 3 - MEDIUM-HIGH
        "GUIDANCE_RAISE": 0.70,
        "EARNINGS_BEAT": 0.65,
        "PARTNERSHIP": 0.62,
        "PRICE_TARGET_RAISE": 0.60,

        # TIER 4 - MEDIUM
        "ANALYST_UPGRADE": 0.52,
        "SHORT_SQUEEZE_SIGNAL": 0.55,
        "UNUSUAL_VOLUME_NEWS": 0.48,

        # TIER 5 - SPECULATIVE
        "BUYOUT_RUMOR": 0.42,
        "SOCIAL_MEDIA_SURGE": 0.38,
        "BREAKING_POSITIVE": 0.35,

        # Legacy mappings (backwards compatibility)
        "earnings": 0.65,
        "FDA_TRIAL_RESULT": 0.85,  # Map to FDA_TRIAL_POSITIVE
    }

    type_mult = type_multipliers.get(event_type, 0.5)
    
    # Factor 2: Impact score
    impact = event.get("boosted_impact", event.get("impact", 0.5))
    
    # Factor 3: Proximity boost (closer = more certain)
    if days_to_event <= 1:
        proximity_mult = 1.2
    elif days_to_event <= 3:
        proximity_mult = 1.1
    else:
        proximity_mult = 1.0
    
    # Calculate final probability
    probability = base_probability * type_mult * impact * proximity_mult
    
    # Clamp 0-1
    probability = max(0, min(1, probability))
    
    return probability


# ============================
# Technical setup check
# ============================

def check_technical_setup(ticker):
    """
    Check if technical setup is favorable
    
    For stocks 1-3 days before event:
    - Check for consolidation
    - Check for higher lows
    - Check volume pattern
    
    Returns:
        dict with technical readiness score
    """
    try:
        # Get features
        feats = compute_features(ticker, include_advanced=False)
        
        if not feats:
            return {"ready": False, "score": 0, "reason": "No data"}
        
        # Check key indicators
        consolidation = feats.get("tight_consolidation", 0)
        higher_lows = feats.get("higher_lows", 0)
        volume_ok = feats.get("volume_spike", 0) < 3  # Not already spiking
        
        # Calculate readiness
        readiness_score = (consolidation + higher_lows) / 2
        
        if readiness_score >= 0.6 and volume_ok:
            return {
                "ready": True,
                "score": readiness_score,
                "reason": "Good setup - tight range, higher lows"
            }
        elif readiness_score >= 0.4:
            return {
                "ready": "maybe",
                "score": readiness_score,
                "reason": "Decent setup - monitor closely"
            }
        else:
            return {
                "ready": False,
                "score": readiness_score,
                "reason": "Weak setup - choppy price action"
            }
    
    except Exception as e:
        logger.warning(f"Technical check failed for {ticker}: {e}")
        return {"ready": "unknown", "score": 0, "reason": "Error"}


# ============================
# Watch reason builder
# ============================

def build_watch_reason(event, days_to_event, probability):
    """Build human-readable reason for WATCH signal"""
    event_type = event.get("type", "event")
    impact = event.get("boosted_impact", event.get("impact", 0))
    
    # Format event type nicely
    type_display = event_type.replace("_", " ").title()
    
    # Build reason
    if days_to_event == 1:
        time_str = "TOMORROW"
    elif days_to_event <= 3:
        time_str = f"in {days_to_event} days"
    else:
        time_str = f"in {days_to_event} days"
    
    reason = f"{type_display} {time_str} (impact: {impact:.2f}, prob: {probability*100:.0f}%)"
    
    return reason


# ============================
# Upgrade WATCH to BUY
# ============================

def check_watch_upgrade(watch_signal):
    """
    Check if WATCH signal should upgrade to BUY
    
    Criteria:
    - 1 day before event
    - Technical setup ready
    - High probability (>0.7)
    
    Returns:
        Upgraded signal or None
    """
    ticker = watch_signal["ticker"]
    days_to_event = watch_signal["days_to_event"]
    probability = watch_signal["probability"]
    
    # Only upgrade if very close to event
    if days_to_event > 1:
        return None
    
    # Check technical setup
    technical = watch_signal.get("technical_check")
    
    if not technical:
        technical = check_technical_setup(ticker)
    
    # Upgrade criteria
    if technical.get("ready") == True and probability >= 0.7:
        return {
            "ticker": ticker,
            "signal": "BUY",
            "confidence": probability,
            "reason": f"WATCH upgraded - {watch_signal['reason']} + technical ready",
            "original_watch": watch_signal
        }
    
    return None


# ============================
# Public API
# ============================

def get_watch_list(universe_tickers=None):
    """Get current watch list"""
    return generate_watch_list(universe_tickers=universe_tickers)


def get_watch_upgrades(watch_list):
    """Check for watch signals ready to upgrade to BUY"""
    upgrades = []
    
    for watch in watch_list:
        upgrade = check_watch_upgrade(watch)
        if upgrade:
            upgrades.append(upgrade)
    
    return upgrades


if __name__ == "__main__":
    # Test
    from src.universe_loader import load_universe
    
    universe = load_universe()
    tickers = universe["ticker"].tolist()[:100] if universe is not None else []
    
    watch_list = get_watch_list(universe_tickers=tickers)
    
    print(f"\n📅 WATCH LIST ({len(watch_list)} signals)")
    print("=" * 60)
    
    for watch in watch_list[:10]:
        print(f"\n🎯 {watch['ticker']} - {watch['signal']}")
        print(f"   Event: {watch['event_type']}")
        print(f"   Date: {watch['event_date']} ({watch['days_to_event']} days)")
        print(f"   Impact: {watch['impact']:.2f} | Probability: {watch['probability']*100:.0f}%")
        print(f"   Reason: {watch['reason']}")
        
        if watch.get("technical_check"):
            tech = watch["technical_check"]
            print(f"   Technical: {tech['ready']} (score: {tech['score']:.2f})")
