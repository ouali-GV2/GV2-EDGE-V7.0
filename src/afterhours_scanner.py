"""
AFTER-HOURS SCANNER - Post-Market Catalyst Detection
====================================================

Scanne les catalysts qui surviennent après la clôture du marché (16h-20h ET):
- Earnings releases post-market
- Breaking news after-hours
- FDA approvals
- M&A announcements

Objectif: Détecter les movers du lendemain AVANT le gap d'ouverture.
"""

from datetime import datetime, time
import pytz
from utils.logger import get_logger
from utils.api_guard import safe_get
from utils.time_utils import is_after_hours

from src.event_engine.event_hub import (
    fetch_earnings_events,
    fetch_company_news,
    fetch_breaking_news
)
from src.event_engine.nlp_event_parser import parse_many_texts

from alerts.telegram_alerts import send_signal_alert

from config import FINNHUB_API_KEY

logger = get_logger("AFTERHOURS_SCANNER")


# ============================
# After-hours earnings detection
# ============================

def scan_afterhours_earnings():
    """
    Detect earnings releases after market close
    
    Focus: BMO (Before Market Open) next day = reported after close today
    """
    logger.info("Scanning after-hours earnings releases...")
    
    earnings = fetch_earnings_events(days_forward=1)
    
    afterhours_earnings = []
    
    for event in earnings:
        # Check if earnings are scheduled for tomorrow morning
        # (meaning they report tonight after hours)
        
        ticker = event.get("ticker", "")
        eps_actual = event.get("eps_actual")
        eps_estimate = event.get("eps_estimate", 0)
        
        # If eps_actual exists = already reported
        if eps_actual is not None and eps_estimate:
            # Calculate beat/miss
            beat_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) if eps_estimate != 0 else 0
            
            if abs(beat_pct) >= 0.10:  # 10%+ beat or miss
                impact = "positive" if beat_pct > 0 else "negative"
                
                afterhours_earnings.append({
                    "ticker": ticker,
                    "type": "earnings_afterhours",
                    "eps_actual": eps_actual,
                    "eps_estimate": eps_estimate,
                    "beat_pct": beat_pct,
                    "impact": impact,
                    "impact_score": min(1.0, abs(beat_pct))
                })
    
    logger.info(f"Found {len(afterhours_earnings)} after-hours earnings events")
    
    return afterhours_earnings


# ============================
# After-hours breaking news
# ============================

def scan_afterhours_news(tickers=None):
    """
    Scan for major news releases after market close
    
    Args:
        tickers: list of tickers to monitor
    """
    logger.info("Scanning after-hours news...")
    
    news_events = []
    
    # Breaking general news
    breaking = fetch_breaking_news(category="general")
    
    # Company-specific news if tickers provided
    if tickers:
        for ticker in tickers[:30]:  # Limit to avoid rate limits
            company_news = fetch_company_news(ticker, days_back=1)
            
            for news in company_news:
                # Check if news is recent (within last 4 hours)
                timestamp = news.get("timestamp", 0)
                
                if timestamp:
                    news_time = datetime.fromtimestamp(timestamp)
                    age_hours = (datetime.utcnow() - news_time).total_seconds() / 3600
                    
                    if age_hours <= 4:  # Recent news
                        news_events.append(news)
    
    # Parse all news with NLP
    news_texts = [n.get("text", "") for n in news_events]
    news_texts.extend(breaking)
    
    parsed = parse_many_texts(news_texts)
    
    # Filter high-impact events
    major_events = [
        e for e in parsed 
        if e.get("impact", 0) >= 0.7
    ]
    
    logger.info(f"Found {len(major_events)} high-impact after-hours news events")
    
    return major_events


# ============================
# After-hours catalyst aggregation
# ============================

def scan_all_afterhours_catalysts(tickers=None):
    """
    Aggregate all after-hours catalysts
    
    Returns:
        List of high-impact events to alert on
    """
    all_catalysts = []
    
    # Earnings
    earnings = scan_afterhours_earnings()
    all_catalysts.extend(earnings)
    
    # News
    news = scan_afterhours_news(tickers=tickers)
    all_catalysts.extend(news)
    
    # Sort by impact score
    all_catalysts.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
    
    return all_catalysts


# ============================
# Alert on major catalysts
# ============================

def alert_afterhours_setups(catalysts, threshold=0.7):
    """
    Send alerts for major after-hours catalysts
    
    Args:
        catalysts: list of catalyst events
        threshold: minimum impact score to alert on
    """
    high_impact = [c for c in catalysts if c.get("impact_score", 0) >= threshold]
    
    for catalyst in high_impact:
        ticker = catalyst.get("ticker", "UNKNOWN")
        cat_type = catalyst.get("type", "event")
        impact = catalyst.get("impact_score", 0)
        
        # Build alert message
        if cat_type == "earnings_afterhours":
            beat_pct = catalyst.get("beat_pct", 0)
            direction = "BEAT" if beat_pct > 0 else "MISS"
            
            alert_data = {
                "ticker": ticker,
                "signal": "AH_EARNINGS_SETUP",
                "monster_score": impact,
                "confidence": 0.85,
                "notes": f"Earnings {direction} by {abs(beat_pct)*100:.1f}% after-hours. Gap likely tomorrow."
            }
        else:
            alert_data = {
                "ticker": ticker,
                "signal": "AH_NEWS_SETUP",
                "monster_score": impact,
                "confidence": 0.75,
                "notes": f"Major news after-hours: {catalyst.get('category', 'Breaking')}"
            }
        
        # Send alert
        send_signal_alert(alert_data)
        
        logger.info(f"Alerted on after-hours setup: {ticker} ({cat_type})")


# ============================
# After-hours scanner loop
# ============================

def run_afterhours_scanner(tickers=None):
    """
    Main after-hours scanning function
    
    To be called during after-hours (16:00-20:00 ET)
    """
    if not is_after_hours():
        logger.info("Not in after-hours window, skipping scan")
        return []
    
    logger.info("=" * 60)
    logger.info("AFTER-HOURS CATALYST SCANNER")
    logger.info("=" * 60)
    
    catalysts = scan_all_afterhours_catalysts(tickers=tickers)
    
    logger.info(f"Total after-hours catalysts found: {len(catalysts)}")
    
    # Alert on high-impact setups
    alert_afterhours_setups(catalysts, threshold=0.7)
    
    logger.info("After-hours scan complete")
    logger.info("=" * 60)
    
    return catalysts


if __name__ == "__main__":
    # Test
    from src.universe_loader import load_universe
    
    universe = load_universe()
    tickers = universe["ticker"].tolist()[:50] if universe is not None else []
    
    catalysts = run_afterhours_scanner(tickers=tickers)
    
    print(f"\nFound {len(catalysts)} after-hours catalysts")
    for cat in catalysts[:5]:
        print(f"  - {cat.get('ticker')}: {cat.get('type')} (impact: {cat.get('impact_score', 0):.2f})")
