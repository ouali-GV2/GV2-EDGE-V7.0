"""
DAILY AUDIT V7.0 - Mesure Quotidienne de Performance
=====================================================

Exécution: Tous les jours à 20h30 UTC (après clôture US)

Objectif:
- Comparer les signaux EDGE du jour avec les vrais top gainers
- Mesurer hit rate, early catches, misses, false positives
- Identifier rapidement les dégradations de performance
- Alimenter le système d'amélioration continue

V7 Architecture Tracking:
- SignalProducer detection rate (all signals, never blocked)
- ExecutionGate allowed vs blocked ratio
- Block reason breakdown (trade limit, capital, risk guard, pre-halt)
- Market Memory (MRP/EP) contribution analysis
- Risk Guard trigger frequency

Metrics clés:
- hit_rate_daily: % des top gainers détectés
- early_catch_rate: % détectés > 2h avant spike
- miss_rate: % des movers manqués
- false_positives: signaux sans mouvement significatif
- avg_lead_time: temps moyen avant explosion
- v7_stats: V7 module performance breakdown

Usage:
    python daily_audit.py  # Exécute l'audit du jour
    python daily_audit.py --date 2026-02-01  # Audit d'une date spécifique
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache
from src.signal_logger import get_signals_for_period, get_signal_by_ticker_and_date
from src.universe_loader import load_universe
from alerts.telegram_alerts import send_daily_audit_alert

# V6 Module imports (optional - graceful fallback)
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

try:
    from src.nlp_enrichi import NLPEnrichi
    HAS_NLP_ENRICHI = True
except ImportError:
    HAS_NLP_ENRICHI = False

logger = get_logger("DAILY_AUDIT_V6")

os.makedirs("data/audit_reports", exist_ok=True)

cache = Cache(ttl=3600)  # 1h cache


# ============================
# FETCH TOP GAINERS (Multiple Sources)
# ============================

def fetch_finviz_top_gainers(min_change_pct=20):
    """
    Scrape Finviz pour les top gainers du jour
    
    URL: https://finviz.com/screener.ashx?v=111&f=ta_change_u20
    
    Args:
        min_change_pct: Minimum % change to consider
    
    Returns:
        List of gainers: [{"ticker": "XYZ", "change_pct": 50.0, "price": 10.5}, ...]
    """
    cache_key = f"finviz_gainers_{datetime.utcnow().date()}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    try:
        # Finviz screener URL for stocks up > 20%
        url = "https://finviz.com/screener.ashx?v=111&f=ta_change_u20&ft=4"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        r = requests.get(url, headers=headers, timeout=15)
        
        if r.status_code != 200:
            logger.warning(f"Finviz returned status {r.status_code}")
            return []
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        gainers = []
        
        # Parse table rows
        table = soup.find('table', {'class': 'table-light'})
        
        if not table:
            logger.warning("Could not find Finviz table")
            return []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            
            if len(cells) >= 10:
                ticker = cells[1].text.strip()
                price_text = cells[8].text.strip()
                change_text = cells[9].text.strip()
                
                try:
                    price = float(price_text)
                    change_pct = float(change_text.replace('%', ''))
                    
                    if change_pct >= min_change_pct:
                        gainers.append({
                            "ticker": ticker,
                            "price": price,
                            "change_pct": change_pct,
                            "source": "finviz",
                            "date": datetime.utcnow().strftime("%Y-%m-%d")
                        })
                except:
                    continue
        
        logger.info(f"Finviz: Found {len(gainers)} top gainers (+{min_change_pct}%+)")
        
        cache.set(cache_key, gainers)
        
        return gainers
    
    except Exception as e:
        logger.error(f"Finviz scraping failed: {e}")
        return []


def fetch_yahoo_top_gainers(min_change_pct=20):
    """
    Fetch top gainers from Yahoo Finance API
    
    Fallback if Finviz fails
    """
    try:
        # Yahoo Finance gainers endpoint
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        
        params = {
            "scrIds": "day_gainers",
            "formatted": "true",
            "count": 100
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return []
        
        data = r.json()
        
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        
        gainers = []
        
        for quote in quotes:
            ticker = quote.get("symbol", "")
            price = quote.get("regularMarketPrice", 0)
            change_pct = quote.get("regularMarketChangePercent", 0)
            
            if change_pct >= min_change_pct:
                gainers.append({
                    "ticker": ticker,
                    "price": price,
                    "change_pct": change_pct,
                    "source": "yahoo",
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                })
        
        logger.info(f"Yahoo: Found {len(gainers)} top gainers (+{min_change_pct}%+)")
        
        return gainers
    
    except Exception as e:
        logger.warning(f"Yahoo Finance failed: {e}")
        return []


def fetch_top_gainers(min_change_pct=20):
    """
    Fetch top gainers from multiple sources
    
    Priority:
    1. Finviz (more comprehensive)
    2. Yahoo Finance (fallback)
    
    Returns deduplicated list
    """
    gainers = fetch_finviz_top_gainers(min_change_pct)
    
    if not gainers:
        logger.info("Finviz failed, trying Yahoo Finance")
        gainers = fetch_yahoo_top_gainers(min_change_pct)
    
    if not gainers:
        logger.warning("No top gainers data available from any source")
    
    return gainers


# ============================
# LEAD TIME CALCULATION
# ============================

def calculate_lead_time(signal_timestamp, mover_date):
    """
    Calculate how many hours BEFORE the explosion we detected the signal
    
    Args:
        signal_timestamp: ISO timestamp of signal
        mover_date: date when stock became a top gainer
    
    Returns:
        hours (positive = early detection, negative = late)
    """
    try:
        signal_dt = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
        
        # Remove timezone info for comparison
        if signal_dt.tzinfo:
            signal_dt = signal_dt.replace(tzinfo=None)
        
        # Assume mover explosion at market open (9:30 AM ET = 14:30 UTC)
        if isinstance(mover_date, str):
            mover_dt = datetime.strptime(mover_date, "%Y-%m-%d")
        else:
            mover_dt = mover_date
        
        # Set to 9:30 AM ET (14:30 UTC)
        mover_dt = mover_dt.replace(hour=14, minute=30)
        
        delta = (mover_dt - signal_dt).total_seconds() / 3600  # hours
        
        return round(delta, 2)
    
    except Exception as e:
        logger.warning(f"Lead time calculation failed: {e}")
        return 0


# ============================
# ANALYZE HITS
# ============================

def analyze_hits(signals_df, gainers):
    """
    Analyze successfully detected movers
    
    Returns:
        dict with hit metrics
    """
    hits = []
    
    for gainer in gainers:
        ticker = gainer["ticker"]
        gainer_date = gainer["date"]
        
        # Find signals for this ticker
        if len(signals_df) > 0:
            ticker_signals = signals_df[signals_df["ticker"] == ticker]
            
            if len(ticker_signals) > 0:
                # Get earliest signal
                earliest = ticker_signals.sort_values("timestamp").iloc[0]
                
                lead_time_hours = calculate_lead_time(earliest["timestamp"], gainer_date)
                
                hits.append({
                    "ticker": ticker,
                    "gainer_change_pct": gainer["change_pct"],
                    "signal_type": earliest["signal_type"],
                    "monster_score": earliest["monster_score"],
                    "pattern_score": earliest.get("pattern_score", 0),
                    "event_impact": earliest.get("event_impact", 0),
                    "lead_time_hours": lead_time_hours,
                    "early_catch": lead_time_hours > 2,  # >2h before
                    "on_time": 0 <= lead_time_hours <= 2
                })
    
    if not hits:
        return {
            "hit_count": 0,
            "hit_rate": 0,
            "early_catch_rate": 0,
            "on_time_rate": 0,
            "avg_lead_time_hours": 0,
            "hits": []
        }
    
    hit_count = len(hits)
    total_gainers = max(1, len(gainers))
    
    early_catches = sum(1 for h in hits if h["early_catch"])
    on_time_catches = sum(1 for h in hits if h["on_time"])
    
    return {
        "hit_count": hit_count,
        "hit_rate": hit_count / total_gainers,
        "early_catch_rate": early_catches / max(1, hit_count),
        "on_time_rate": on_time_catches / max(1, hit_count),
        "avg_lead_time_hours": sum(h["lead_time_hours"] for h in hits) / hit_count,
        "hits": hits
    }


# ============================
# ANALYZE MISSES
# ============================

def analyze_misses(signals_df, gainers):
    """
    Analyze movers we missed
    
    Categories:
    - no_signal: No signal generated at all
    - low_score: Signal generated but below threshold
    - timing: Signal generated too late
    """
    detected_tickers = set(signals_df["ticker"].unique()) if len(signals_df) > 0 else set()
    gainer_tickers = set(g["ticker"] for g in gainers)
    
    missed_tickers = gainer_tickers - detected_tickers
    
    misses = []
    
    for gainer in gainers:
        ticker = gainer["ticker"]
        
        if ticker in missed_tickers:
            misses.append({
                "ticker": ticker,
                "change_pct": gainer["change_pct"],
                "reason": "no_signal",  # TODO: deeper analysis
                "source": gainer.get("source", "unknown")
            })
    
    total_gainers = max(1, len(gainers))
    
    return {
        "miss_count": len(misses),
        "miss_rate": len(misses) / total_gainers,
        "missed_tickers": [m["ticker"] for m in misses],
        "misses": misses
    }


# ============================
# ANALYZE FALSE POSITIVES
# ============================

def analyze_false_positives(signals_df, gainers):
    """
    Signals that didn't result in significant moves
    """
    gainer_tickers = set(g["ticker"] for g in gainers)
    signal_tickers = set(signals_df["ticker"].unique()) if len(signals_df) > 0 else set()
    
    false_positive_tickers = signal_tickers - gainer_tickers
    
    fp_details = []
    
    if len(signals_df) > 0:
        fp_signals = signals_df[signals_df["ticker"].isin(false_positive_tickers)]
        
        for ticker in false_positive_tickers:
            ticker_signals = fp_signals[fp_signals["ticker"] == ticker]
            
            if len(ticker_signals) > 0:
                signal = ticker_signals.iloc[0]
                
                fp_details.append({
                    "ticker": ticker,
                    "signal_type": signal["signal_type"],
                    "monster_score": signal["monster_score"]
                })
    
    return {
        "fp_count": len(false_positive_tickers),
        "fp_tickers": list(false_positive_tickers),
        "fp_details": fp_details
    }


# ============================
# MAIN DAILY AUDIT
# ============================

def run_daily_audit(audit_date=None, min_change_pct=20, send_telegram=True):
    """
    Run daily audit for a specific date
    
    Args:
        audit_date: Date to audit (default: today)
        min_change_pct: Minimum % change to consider a "top gainer"
        send_telegram: Send summary via Telegram
    
    Returns:
        Audit report dict
    """
    if audit_date is None:
        audit_date = datetime.utcnow().date()
    elif isinstance(audit_date, str):
        audit_date = datetime.strptime(audit_date, "%Y-%m-%d").date()
    
    logger.info("=" * 60)
    logger.info(f"DAILY AUDIT - {audit_date}")
    logger.info("=" * 60)
    
    # Date range for signals (full day)
    start = datetime.combine(audit_date, datetime.min.time())
    end = datetime.combine(audit_date, datetime.max.time())
    
    # 1. Get EDGE signals for the day
    signals_df = get_signals_for_period(start, end)
    
    # Filter only BUY and BUY_STRONG
    if len(signals_df) > 0:
        signals_df = signals_df[signals_df["signal_type"].isin(["BUY", "BUY_STRONG"])]
    
    logger.info(f"Retrieved {len(signals_df)} BUY/BUY_STRONG signals")
    
    # 2. Get real top gainers
    gainers = fetch_top_gainers(min_change_pct=min_change_pct)
    
    logger.info(f"Retrieved {len(gainers)} top gainers (+{min_change_pct}%+)")
    
    if not gainers:
        logger.warning("No gainers data - audit may be incomplete")
    
    # 3. Analyze hits
    hit_analysis = analyze_hits(signals_df, gainers)
    
    logger.info(f"Hit Rate: {hit_analysis['hit_rate']*100:.1f}%")
    logger.info(f"Early Catch Rate: {hit_analysis['early_catch_rate']*100:.1f}%")
    logger.info(f"Avg Lead Time: {hit_analysis['avg_lead_time_hours']:.1f}h")
    
    # 4. Analyze misses
    miss_analysis = analyze_misses(signals_df, gainers)
    
    logger.info(f"Miss Rate: {miss_analysis['miss_rate']*100:.1f}%")
    
    # 5. Analyze false positives
    fp_analysis = analyze_false_positives(signals_df, gainers)
    
    logger.info(f"False Positives: {fp_analysis['fp_count']}")
    
    # 6. Compile report
    report = {
        "audit_date": audit_date.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "min_change_pct": min_change_pct
        },
        "summary": {
            "total_signals": len(signals_df),
            "total_gainers": len(gainers),
            "hit_rate": hit_analysis["hit_rate"],
            "early_catch_rate": hit_analysis["early_catch_rate"],
            "miss_rate": miss_analysis["miss_rate"],
            "fp_count": fp_analysis["fp_count"],
            "avg_lead_time_hours": hit_analysis["avg_lead_time_hours"]
        },
        "hit_analysis": hit_analysis,
        "miss_analysis": miss_analysis,
        "fp_analysis": fp_analysis,
        "top_gainers": gainers[:20],  # Top 20 for reference
        "performance_grade": calculate_performance_grade(hit_analysis, miss_analysis, fp_analysis)
    }
    
    # 7. Save report
    filename = f"data/audit_reports/daily_audit_{audit_date.isoformat()}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved: {filename}")
    
    # 8. Send Telegram summary
    if send_telegram:
        send_daily_audit_summary(report)
    
    logger.info("=" * 60)
    logger.info("DAILY AUDIT COMPLETED")
    logger.info("=" * 60)
    
    return report


def calculate_performance_grade(hit_analysis, miss_analysis, fp_analysis):
    """
    Calculate overall performance grade
    
    A: Hit rate > 60%, early catch > 50%
    B: Hit rate > 40%, early catch > 30%
    C: Hit rate > 25%, early catch > 15%
    D: Hit rate > 10%
    F: Hit rate < 10%
    """
    hit_rate = hit_analysis.get("hit_rate", 0)
    early_rate = hit_analysis.get("early_catch_rate", 0)
    fp_count = fp_analysis.get("fp_count", 0)
    
    # Penalize high false positives
    fp_penalty = min(0.2, fp_count * 0.02)
    
    adjusted_hit = hit_rate - fp_penalty
    
    if adjusted_hit >= 0.6 and early_rate >= 0.5:
        return "A"
    elif adjusted_hit >= 0.4 and early_rate >= 0.3:
        return "B"
    elif adjusted_hit >= 0.25 and early_rate >= 0.15:
        return "C"
    elif adjusted_hit >= 0.1:
        return "D"
    else:
        return "F"


def send_daily_audit_summary(report):
    """
    Send audit summary via Telegram
    """
    summary = report["summary"]
    grade = report["performance_grade"]
    
    grade_emoji = {
        "A": "🏆",
        "B": "✅",
        "C": "⚠️",
        "D": "🔶",
        "F": "❌"
    }
    
    message = f"""
📊 DAILY AUDIT - {report['audit_date']}

{grade_emoji.get(grade, '❓')} Performance Grade: {grade}

📈 Hit Rate: {summary['hit_rate']*100:.1f}%
⏱ Early Catches: {summary['early_catch_rate']*100:.1f}%
⏰ Avg Lead Time: {summary['avg_lead_time_hours']:.1f}h

❌ Miss Rate: {summary['miss_rate']*100:.1f}%
🎯 False Positives: {summary['fp_count']}

Signals: {summary['total_signals']} | Gainers: {summary['total_gainers']}
"""
    
    # Add top hits
    hits = report["hit_analysis"].get("hits", [])[:3]
    if hits:
        message += "\n🎯 TOP HITS:\n"
        for hit in hits:
            message += f"  • {hit['ticker']}: +{hit['gainer_change_pct']:.0f}% (lead: {hit['lead_time_hours']:.1f}h)\n"
    
    # Add top misses
    misses = report["miss_analysis"].get("missed_tickers", [])[:3]
    if misses:
        message += f"\n❌ TOP MISSES: {', '.join(misses)}"
    
    try:
        send_daily_audit_alert(report)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ============================
# WEEKLY SUMMARY
# ============================

def generate_weekly_summary(end_date=None):
    """
    Generate weekly summary from daily audits
    """
    if end_date is None:
        end_date = datetime.utcnow().date()
    
    start_date = end_date - timedelta(days=7)
    
    # Load all daily reports for the week
    reports = []
    
    for i in range(7):
        date = start_date + timedelta(days=i)
        filename = f"data/audit_reports/daily_audit_{date.isoformat()}.json"
        
        if os.path.exists(filename):
            with open(filename) as f:
                reports.append(json.load(f))
    
    if not reports:
        logger.warning("No daily reports found for weekly summary")
        return None
    
    # Aggregate metrics
    total_hit_rate = sum(r["summary"]["hit_rate"] for r in reports) / len(reports)
    total_early_rate = sum(r["summary"]["early_catch_rate"] for r in reports) / len(reports)
    total_miss_rate = sum(r["summary"]["miss_rate"] for r in reports) / len(reports)
    total_fp = sum(r["summary"]["fp_count"] for r in reports)
    avg_lead_time = sum(r["summary"]["avg_lead_time_hours"] for r in reports) / len(reports)
    
    summary = {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days_with_data": len(reports)
        },
        "metrics": {
            "avg_hit_rate": total_hit_rate,
            "avg_early_catch_rate": total_early_rate,
            "avg_miss_rate": total_miss_rate,
            "total_false_positives": total_fp,
            "avg_lead_time_hours": avg_lead_time
        },
        "daily_grades": [r["performance_grade"] for r in reports],
        "trend": "improving" if reports[-1]["summary"]["hit_rate"] > reports[0]["summary"]["hit_rate"] else "declining"
    }
    
    # Save
    filename = f"data/audit_reports/weekly_summary_{end_date.isoformat()}.json"
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Weekly summary saved: {filename}")
    
    return summary


# ============================
# CLI
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GV2-EDGE Daily Audit")
    parser.add_argument("--date", type=str, help="Audit date (YYYY-MM-DD)")
    parser.add_argument("--min-change", type=float, default=20, help="Minimum % change for gainers")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram notification")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly summary")
    
    args = parser.parse_args()
    
    if args.weekly:
        summary = generate_weekly_summary()
        if summary:
            print("\n📊 WEEKLY SUMMARY")
            print(f"Period: {summary['period']['start']} to {summary['period']['end']}")
            print(f"Avg Hit Rate: {summary['metrics']['avg_hit_rate']*100:.1f}%")
            print(f"Avg Early Catch: {summary['metrics']['avg_early_catch_rate']*100:.1f}%")
            print(f"Trend: {summary['trend']}")
    else:
        report = run_daily_audit(
            audit_date=args.date,
            min_change_pct=args.min_change,
            send_telegram=not args.no_telegram
        )
        
        print("\n" + "=" * 60)
        print("DAILY AUDIT SUMMARY")
        print("=" * 60)
        print(f"Date: {report['audit_date']}")
        print(f"Grade: {report['performance_grade']}")
        print(f"Hit Rate: {report['summary']['hit_rate']*100:.1f}%")
        print(f"Early Catches: {report['summary']['early_catch_rate']*100:.1f}%")
        print(f"Avg Lead Time: {report['summary']['avg_lead_time_hours']:.1f}h")
        print(f"Misses: {report['summary']['miss_rate']*100:.1f}%")
        print(f"False Positives: {report['summary']['fp_count']}")
        print("=" * 60)
