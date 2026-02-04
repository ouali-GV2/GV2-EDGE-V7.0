"""
HISTORICAL BEAT RATE ANALYZER
==============================

Analyse l'historique des earnings pour calculer:
- Taux de beat (% earnings supérieurs aux estimates)
- Trend récent (amélioration/détérioration)
- Amplitude moyenne des beats/misses
- Prédiction probabilité du prochain beat

Sources:
- Finnhub earnings history
- IBKR fundamental data (si disponible)
- Cache pour éviter sur-sollicitation API
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get

from config import FINNHUB_API_KEY, USE_IBKR_DATA

logger = get_logger("BEAT_RATE_ANALYZER")

cache = Cache(ttl=86400)  # 24h cache (earnings history stable)

FINNHUB_EARNINGS_HISTORY = "https://finnhub.io/api/v1/stock/earnings"


# ============================
# Fetch earnings history
# ============================

def fetch_earnings_history(ticker, quarters=8):
    """
    Fetch historical earnings for a ticker
    
    Args:
        ticker: Stock symbol
        quarters: Number of past quarters to fetch (default 8 = 2 years)
    
    Returns:
        List of earnings reports with actual vs estimate
    """
    cache_key = f"earnings_history_{ticker}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    try:
        params = {
            "symbol": ticker,
            "token": FINNHUB_API_KEY
        }
        
        r = safe_get(FINNHUB_EARNINGS_HISTORY, params=params, timeout=10)
        data = r.json()
        
        if not data:
            return []
        
        # Extract relevant data
        earnings_history = []
        
        for report in data[:quarters]:
            actual = report.get("actual")
            estimate = report.get("estimate")
            period = report.get("period")
            
            if actual is not None and estimate is not None and estimate != 0:
                earnings_history.append({
                    "period": period,
                    "actual": actual,
                    "estimate": estimate,
                    "beat": actual > estimate,
                    "miss": actual < estimate,
                    "beat_amount": actual - estimate,
                    "beat_pct": (actual - estimate) / abs(estimate) if estimate != 0 else 0
                })
        
        cache.set(cache_key, earnings_history)
        
        logger.info(f"Fetched {len(earnings_history)} earnings reports for {ticker}")
        
        return earnings_history
    
    except Exception as e:
        logger.error(f"Failed to fetch earnings history for {ticker}: {e}")
        return []


# ============================
# Calculate beat rate
# ============================

def calculate_beat_rate(ticker, quarters=4):
    """
    Calculate historical beat rate
    
    Args:
        ticker: Stock symbol
        quarters: Number of quarters to analyze (default 4 = 1 year)
    
    Returns:
        dict with beat_rate, consistency, avg_beat_pct, trend
    """
    earnings = fetch_earnings_history(ticker, quarters=quarters)
    
    if not earnings:
        return {
            "beat_rate": 0.5,  # Neutral default
            "consistency": 0,
            "avg_beat_pct": 0,
            "trend": "unknown",
            "confidence": 0,
            "sample_size": 0
        }
    
    # Calculate metrics
    total_reports = len(earnings)
    beats = sum(1 for e in earnings if e["beat"])
    misses = sum(1 for e in earnings if e["miss"])
    
    beat_rate = beats / total_reports if total_reports > 0 else 0.5
    
    # Avg beat percentage
    beat_pcts = [e["beat_pct"] for e in earnings]
    avg_beat_pct = np.mean(beat_pcts) if beat_pcts else 0
    
    # Consistency score (less variance = more consistent)
    consistency = 1 - min(1, np.std(beat_pcts)) if len(beat_pcts) > 1 else 0
    
    # Trend analysis (recent vs older)
    if total_reports >= 4:
        recent_beats = sum(1 for e in earnings[:2] if e["beat"])
        older_beats = sum(1 for e in earnings[2:4] if e["beat"])
        
        if recent_beats > older_beats:
            trend = "improving"
        elif recent_beats < older_beats:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Confidence based on sample size
    confidence = min(1.0, total_reports / quarters)
    
    return {
        "beat_rate": beat_rate,
        "consistency": consistency,
        "avg_beat_pct": avg_beat_pct,
        "trend": trend,
        "confidence": confidence,
        "sample_size": total_reports,
        "recent_beats": beats,
        "recent_misses": misses
    }


# ============================
# Predict next earnings
# ============================

def predict_next_earnings(ticker, quarters=4):
    """
    Predict probability of next earnings beat
    
    Factors:
    - Historical beat rate
    - Trend (improving/declining)
    - Consistency
    - Recent performance
    
    Returns:
        Probability score 0-1
    """
    analysis = calculate_beat_rate(ticker, quarters=quarters)
    
    beat_rate = analysis["beat_rate"]
    trend = analysis["trend"]
    consistency = analysis["consistency"]
    confidence = analysis["confidence"]
    
    # Base probability = historical beat rate
    probability = beat_rate
    
    # Adjust for trend
    if trend == "improving":
        probability *= 1.2  # +20% if improving
    elif trend == "declining":
        probability *= 0.8  # -20% if declining
    
    # Adjust for consistency (consistent = more reliable)
    probability *= (0.8 + 0.2 * consistency)
    
    # Weight by confidence (more data = more reliable)
    probability = probability * confidence + 0.5 * (1 - confidence)
    
    # Clamp 0-1
    probability = max(0, min(1, probability))
    
    return {
        "probability": probability,
        "beat_rate": beat_rate,
        "trend": trend,
        "consistency": consistency,
        "confidence": confidence,
        "reasoning": build_prediction_reasoning(analysis)
    }


# ============================
# Build reasoning
# ============================

def build_prediction_reasoning(analysis):
    """Build human-readable reasoning for prediction"""
    beat_rate = analysis["beat_rate"]
    trend = analysis["trend"]
    sample_size = analysis["sample_size"]
    
    reasoning = f"Beat {analysis['recent_beats']}/{sample_size} last quarters"
    
    if beat_rate >= 0.75:
        reasoning += " (strong history)"
    elif beat_rate >= 0.5:
        reasoning += " (decent history)"
    else:
        reasoning += " (weak history)"
    
    if trend == "improving":
        reasoning += ", trend improving"
    elif trend == "declining":
        reasoning += ", trend declining"
    
    return reasoning


# ============================
# Batch analysis
# ============================

def analyze_multiple_tickers(tickers, quarters=4):
    """
    Analyze beat rate for multiple tickers
    
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for ticker in tickers:
        prediction = predict_next_earnings(ticker, quarters=quarters)
        
        results.append({
            "ticker": ticker,
            "probability": prediction["probability"],
            "beat_rate": prediction["beat_rate"],
            "trend": prediction["trend"],
            "confidence": prediction["confidence"],
            "reasoning": prediction["reasoning"]
        })
    
    return pd.DataFrame(results)


# ============================
# Integration with watch list
# ============================

def enhance_watch_with_beat_rate(watch_signal):
    """
    Enhance WATCH signal with beat rate analysis
    
    Args:
        watch_signal: Watch signal dict
    
    Returns:
        Enhanced watch signal with beat rate probability
    """
    ticker = watch_signal.get("ticker")
    event_type = watch_signal.get("event_type", "")
    
    # Only enhance earnings-related events
    if "earning" not in event_type.lower():
        return watch_signal
    
    # Get prediction
    prediction = predict_next_earnings(ticker, quarters=4)
    
    # Enhance watch signal
    watch_signal["beat_probability"] = prediction["probability"]
    watch_signal["beat_reasoning"] = prediction["reasoning"]
    watch_signal["beat_confidence"] = prediction["confidence"]
    
    # Adjust overall probability
    original_prob = watch_signal.get("probability", 0.5)
    beat_prob = prediction["probability"]
    
    # Weighted average (70% beat rate, 30% other factors)
    enhanced_prob = 0.7 * beat_prob + 0.3 * original_prob
    
    watch_signal["probability"] = enhanced_prob
    watch_signal["reason"] += f" | {prediction['reasoning']}"
    
    return watch_signal


# ============================
# Public API (for monster_score)
# ============================

def get_earnings_probability(ticker):
    """
    Get probability of earnings beat for a ticker
    
    Wrapper function for use in monster_score.py
    
    Args:
        ticker: Stock symbol
    
    Returns:
        float: Probability 0-1, or None if no data
    """
    try:
        prediction = predict_next_earnings(ticker, quarters=4)
        return prediction.get("probability")
    except Exception as e:
        logger.warning(f"get_earnings_probability failed for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # Test
    test_ticker = "AAPL"
    
    print(f"\n📊 EARNINGS HISTORY ANALYSIS: {test_ticker}")
    print("=" * 60)
    
    # Fetch history
    history = fetch_earnings_history(test_ticker, quarters=8)
    
    print(f"\nLast {len(history)} earnings reports:")
    for report in history[:4]:
        beat_str = "✅ BEAT" if report["beat"] else "❌ MISS"
        print(f"  {report['period']}: ${report['actual']:.2f} vs ${report['estimate']:.2f} {beat_str} ({report['beat_pct']*100:+.1f}%)")
    
    # Calculate beat rate
    analysis = calculate_beat_rate(test_ticker, quarters=4)
    
    print(f"\n📈 BEAT RATE ANALYSIS:")
    print(f"  Beat Rate: {analysis['beat_rate']*100:.0f}% ({analysis['recent_beats']}/{analysis['sample_size']})")
    print(f"  Avg Beat: {analysis['avg_beat_pct']*100:+.1f}%")
    print(f"  Trend: {analysis['trend']}")
    print(f"  Consistency: {analysis['consistency']:.2f}")
    
    # Prediction
    prediction = predict_next_earnings(test_ticker, quarters=4)
    
    print(f"\n🎯 NEXT EARNINGS PREDICTION:")
    print(f"  Probability of Beat: {prediction['probability']*100:.0f}%")
    print(f"  Confidence: {prediction['confidence']:.2f}")
    print(f"  Reasoning: {prediction['reasoning']}")
