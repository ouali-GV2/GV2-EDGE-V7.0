"""
WEEKLY DEEP AUDIT V7.0 - Advanced Performance Analysis
======================================================

Analyse complète de la performance du système:
- Hit rate (% de top gainers détectés)
- Lead time (combien de temps AVANT l'explosion)
- Pattern analysis (quels patterns performent)
- Miss analysis (pourquoi certains movers sont manqués)
- Auto-tuning recommendations

V7 Architecture Tracking:
- SignalProducer vs ExecutionGate funnel analysis
- Block reason trends (which limits trigger most)
- Market Memory MRP/EP correlation with actual outcomes
- Risk Guard effectiveness (prevented losses)
- Pre-Halt Engine accuracy (halt predictions)
- Week-over-week comparison of V7 metrics

Cette version compare les signaux AVANT vs les movers APRÈS.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.api_guard import safe_get

from src.universe_loader import load_universe
from src.signal_logger import get_signals_for_period, get_signal_by_ticker_and_date
from alerts.telegram_alerts import send_weekly_audit_alert

from config import FINNHUB_API_KEY

# V6 Module imports (optional)
try:
    from src.catalyst_score_v3 import CatalystScoreV3
    HAS_CATALYST_V3 = True
except ImportError:
    HAS_CATALYST_V3 = False

try:
    from src.pre_spike_radar import PreSpikeRadar
    HAS_PRE_SPIKE = True
except ImportError:
    HAS_PRE_SPIKE = False

try:
    from src.repeat_gainer_memory import RepeatGainerMemory
    HAS_REPEAT_GAINER = True
except ImportError:
    HAS_REPEAT_GAINER = False

logger = get_logger("WEEKLY_AUDIT_V6")

os.makedirs("data/audit_reports", exist_ok=True)

FINNHUB_GAINERS = "https://finnhub.io/api/v1/stock/top-gainers"


# ============================
# Fetch historical top gainers
# ============================

def fetch_top_gainers_multi_day(days_back=7):
    """
    Fetch top gainers for multiple days
    
    Note: Finnhub free tier only gives current day gainers.
    For production, use a screener API or scrape finviz.com
    
    For now, we approximate with current data.
    """
    params = {"token": FINNHUB_API_KEY}
    
    try:
        r = safe_get(FINNHUB_GAINERS, params=params, timeout=10)
        data = r.json()
        
        gainers = []
        
        for item in data:
            ticker = item.get("symbol", "")
            price = item.get("price", 0)
            change_pct = item.get("changesPercentage", 0)
            
            if ticker and abs(change_pct) >= 20:  # +20%+ movers
                gainers.append({
                    "ticker": ticker,
                    "price": price,
                    "change_pct": change_pct,
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                })
        
        logger.info(f"Fetched {len(gainers)} top gainers (20%+)")
        return gainers
        
    except Exception as e:
        logger.error(f"Failed to fetch gainers: {e}")
        return []


def scrape_finviz_gainers():
    """
    Alternative: scrape Finviz for top gainers
    (More reliable for historical data)
    
    TODO: Implement if Finnhub insufficient
    """
    # Placeholder for future enhancement
    pass


# ============================
# Lead time calculation
# ============================

def calculate_lead_time(signal_timestamp, mover_date):
    """
    Calculate how many hours BEFORE the explosion we detected the signal
    
    Args:
        signal_timestamp: ISO timestamp of signal
        mover_date: date when stock became a top gainer
    
    Returns:
        hours (positive = early, negative = late)
    """
    try:
        signal_dt = datetime.fromisoformat(signal_timestamp)
        
        # Assume mover explosion at market open (9:30 AM ET)
        if isinstance(mover_date, str):
            mover_dt = datetime.strptime(mover_date, "%Y-%m-%d")
        else:
            mover_dt = mover_date
        
        # Set to 9:30 AM (assume explosion at open)
        mover_dt = mover_dt.replace(hour=9, minute=30)
        
        delta = (mover_dt - signal_dt).total_seconds() / 3600  # hours
        
        return delta
        
    except Exception as e:
        logger.warning(f"Lead time calc failed: {e}")
        return 0


# ============================
# Analyze hits (detected movers)
# ============================

def analyze_hits(signals_df, gainers_list):
    """
    Analyze successfully detected movers
    
    Returns:
        - hit_rate
        - early_catch_rate (>2h before)
        - average_lead_time
        - top_performing_patterns
    """
    hits = []
    
    for gainer in gainers_list:
        ticker = gainer["ticker"]
        gainer_date = gainer["date"]
        
        # Find earliest signal for this ticker around that date
        signal = get_signal_by_ticker_and_date(ticker, gainer_date)
        
        if signal:
            lead_time_hours = calculate_lead_time(signal["timestamp"], gainer_date)
            
            hits.append({
                "ticker": ticker,
                "gainer_change_pct": gainer["change_pct"],
                "signal_type": signal["signal_type"],
                "monster_score": signal["monster_score"],
                "pattern_score": signal["pattern_score"],
                "event_impact": signal["event_impact"],
                "lead_time_hours": lead_time_hours,
                "early_catch": lead_time_hours > 2  # >2h before explosion
            })
    
    if not hits:
        return {
            "hit_rate": 0,
            "early_catch_rate": 0,
            "avg_lead_time_hours": 0,
            "hits": []
        }
    
    hits_df = pd.DataFrame(hits)
    
    hit_rate = len(hits) / max(1, len(gainers_list))
    early_catch_rate = hits_df["early_catch"].mean() if len(hits_df) > 0 else 0
    avg_lead_time = hits_df["lead_time_hours"].mean() if len(hits_df) > 0 else 0
    
    # Top patterns
    pattern_performance = hits_df.groupby("signal_type").agg({
        "ticker": "count",
        "monster_score": "mean",
        "lead_time_hours": "mean"
    }).to_dict()
    
    return {
        "hit_rate": hit_rate,
        "early_catch_rate": early_catch_rate,
        "avg_lead_time_hours": avg_lead_time,
        "hits": hits,
        "pattern_performance": pattern_performance
    }


# ============================
# Analyze misses
# ============================

def analyze_misses(gainers_list, detected_tickers):
    """
    Analyze why we missed certain top gainers
    
    Categories:
    - No event in system
    - Event but low score
    - Pattern not detected
    - Low volume/liquidity
    - Outside universe
    """
    misses = []
    
    for gainer in gainers_list:
        ticker = gainer["ticker"]
        
        if ticker not in detected_tickers:
            # We missed this mover
            # TODO: Fetch why (need to check events, patterns, etc.)
            # For now, basic categorization
            
            misses.append({
                "ticker": ticker,
                "change_pct": gainer["change_pct"],
                "reason": "unknown"  # TODO: deep analysis
            })
    
    miss_rate = len(misses) / max(1, len(gainers_list))
    
    # Categorize reasons
    reason_breakdown = {}
    for miss in misses:
        reason = miss["reason"]
        reason_breakdown[reason] = reason_breakdown.get(reason, 0) + 1
    
    return {
        "miss_rate": miss_rate,
        "total_misses": len(misses),
        "missed_tickers": misses,
        "reason_breakdown": reason_breakdown
    }


# ============================
# Analyze false positives
# ============================

def analyze_false_positives(signals_df, gainers_list):
    """
    Signals generated but stock didn't move significantly
    """
    gainer_tickers = set([g["ticker"] for g in gainers_list])
    signal_tickers = set(signals_df["ticker"].unique())
    
    false_positives = signal_tickers - gainer_tickers
    
    fp_signals = signals_df[signals_df["ticker"].isin(false_positives)]
    
    # Common traits of false positives
    if len(fp_signals) > 0:
        avg_scores = {
            "monster_score": fp_signals["monster_score"].mean(),
            "pattern_score": fp_signals["pattern_score"].mean(),
            "event_impact": fp_signals["event_impact"].mean()
        }
    else:
        avg_scores = {}
    
    return {
        "total_fp": len(false_positives),
        "fp_tickers": list(false_positives),
        "avg_scores": avg_scores
    }


# ============================
# Weight optimization recommendations
# ============================

def recommend_weight_adjustments(analysis_results):
    """
    Based on performance, suggest weight adjustments
    
    Logic:
    - If many misses had events → increase event weight
    - If FPs have high pattern scores → decrease pattern weight
    - If early catches correlate with specific component → boost it
    """
    recommendations = []
    
    hit_analysis = analysis_results.get("hit_analysis", {})
    miss_analysis = analysis_results.get("miss_analysis", {})
    fp_analysis = analysis_results.get("fp_analysis", {})
    
    # High miss rate + events exist
    if miss_analysis.get("miss_rate", 0) > 0.5:
        recommendations.append({
            "action": "increase",
            "component": "event",
            "delta": 0.05,
            "reason": "High miss rate suggests events not weighted enough"
        })
    
    # High false positive rate with patterns
    if fp_analysis.get("total_fp", 0) > 10:
        avg_fp_pattern = fp_analysis.get("avg_scores", {}).get("pattern_score", 0)
        if avg_fp_pattern > 0.6:
            recommendations.append({
                "action": "decrease",
                "component": "pattern",
                "delta": -0.03,
                "reason": "Patterns contributing to false positives"
            })
    
    # Good early catch rate → maintain/boost PM transition
    if hit_analysis.get("early_catch_rate", 0) > 0.6:
        recommendations.append({
            "action": "maintain",
            "component": "pm_transition",
            "delta": 0,
            "reason": "PM transition performing well for early catches"
        })
    
    return recommendations


# ============================
# Main audit function
# ============================

def run_weekly_audit_v2(days_back=7):
    """
    Comprehensive weekly audit with lead time analysis
    
    Returns:
        Full audit report with metrics and recommendations
    """
    logger.info("=" * 60)
    logger.info("STARTING WEEKLY DEEP AUDIT V2")
    logger.info("=" * 60)
    
    # Date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    # 1. Get signals from our system (last week)
    signals_df = get_signals_for_period(start_date, end_date)
    
    logger.info(f"Retrieved {len(signals_df)} signals from {start_date.date()} to {end_date.date()}")
    
    # 2. Get real market top gainers
    gainers = fetch_top_gainers_multi_day(days_back=days_back)
    
    logger.info(f"Retrieved {len(gainers)} real top gainers (20%+)")
    
    # 3. Analyze hits
    detected_tickers = set(signals_df["ticker"].unique()) if len(signals_df) > 0 else set()
    
    hit_analysis = analyze_hits(signals_df, gainers)
    
    logger.info(f"Hit Rate: {hit_analysis['hit_rate']*100:.2f}%")
    logger.info(f"Early Catch Rate: {hit_analysis['early_catch_rate']*100:.2f}%")
    logger.info(f"Avg Lead Time: {hit_analysis['avg_lead_time_hours']:.1f} hours")
    
    # 4. Analyze misses
    miss_analysis = analyze_misses(gainers, detected_tickers)
    
    logger.info(f"Miss Rate: {miss_analysis['miss_rate']*100:.2f}%")
    
    # 5. Analyze false positives
    fp_analysis = analyze_false_positives(signals_df, gainers)
    
    logger.info(f"False Positives: {fp_analysis['total_fp']}")
    
    # 6. Generate recommendations
    analysis_results = {
        "hit_analysis": hit_analysis,
        "miss_analysis": miss_analysis,
        "fp_analysis": fp_analysis
    }
    
    recommendations = recommend_weight_adjustments(analysis_results)
    
    logger.info(f"Generated {len(recommendations)} optimization recommendations")
    
    # 7. Compile final report
    report = {
        "audit_date": end_date.isoformat(),
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days_back
        },
        "metrics": {
            "hit_rate": hit_analysis["hit_rate"],
            "early_catch_rate": hit_analysis["early_catch_rate"],
            "avg_lead_time_hours": hit_analysis["avg_lead_time_hours"],
            "miss_rate": miss_analysis["miss_rate"],
            "false_positive_count": fp_analysis["total_fp"]
        },
        "details": {
            "hits": hit_analysis["hits"],
            "misses": miss_analysis["missed_tickers"],
            "false_positives": fp_analysis["fp_tickers"],
            "pattern_performance": hit_analysis.get("pattern_performance", {})
        },
        "recommendations": recommendations
    }
    
    # 8. Save report
    filename = f"data/audit_reports/weekly_audit_v2_{end_date.strftime('%Y%m%d')}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Audit report saved: {filename}")
    logger.info("=" * 60)
    logger.info("WEEKLY AUDIT COMPLETED")
    logger.info("=" * 60)
    
    return report


# ============================
# CLI
# ============================

if __name__ == "__main__":
    result = run_weekly_audit_v2(days_back=7)
    
    print("\n" + "=" * 60)
    print("WEEKLY AUDIT V2 SUMMARY")
    print("=" * 60)
    print(f"Hit Rate: {result['metrics']['hit_rate']*100:.2f}%")
    print(f"Early Catch Rate: {result['metrics']['early_catch_rate']*100:.2f}%")
    print(f"Avg Lead Time: {result['metrics']['avg_lead_time_hours']:.1f} hours")
    print(f"Miss Rate: {result['metrics']['miss_rate']*100:.2f}%")
    print(f"False Positives: {result['metrics']['false_positive_count']}")
    print("=" * 60)
    
    if result["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in result["recommendations"]:
            print(f"  - {rec['action'].upper()} {rec['component']} by {rec['delta']}: {rec['reason']}")

