"""
DAILY AUDIT V8.0 - Mesure Quotidienne de Performance
=====================================================

Execution: Tous les jours a 20h30 UTC (apres cloture US)

Objectif:
- Comparer les signaux EDGE du jour avec les vrais top gainers
- Mesurer hit rate, early catches, misses, false positives
- Identifier rapidement les degradations de performance
- Alimenter le systeme d'amelioration continue

V8 Architecture Tracking:
- SignalProducer detection rate (all signals, never blocked)
- ExecutionGate allowed vs blocked ratio (funnel analysis)
- Block reason breakdown (trade limit, capital, risk guard, pre-halt)
- AccelerationEngine state distribution (DORMANT/ACCUMULATING/LAUNCHING/BREAKOUT/EXHAUSTED)
- SmallCapRadar blip accuracy (predicted vs actual movers)
- Risk Guard V8 MIN-mode effectiveness
- V8 lead time improvement vs baseline

Metrics cles:
- hit_rate_daily: % des top gainers detectes
- early_catch_rate: % detectes > 2h avant spike
- miss_rate: % des movers manques (with categorized reasons)
- false_positives: signaux sans mouvement significatif (graduated scoring)
- avg_lead_time: temps moyen avant explosion
- v8_stats: V8 module performance breakdown
- funnel_stats: SignalProducer -> ExecutionGate conversion

Usage:
    python daily_audit.py                        # Audit du jour
    python daily_audit.py --date 2026-02-01      # Audit date specifique
    python daily_audit.py --weekly                # Resume hebdomadaire
"""

import os
import json
import argparse
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get
from src.signal_logger import get_signals_for_period, get_signal_by_ticker_and_date
from src.universe_loader import load_universe
from alerts.telegram_alerts import send_daily_audit_alert
from config import FINNHUB_API_KEY

# V8 Module imports (graceful fallback)
try:
    from src.engines.acceleration_engine import get_acceleration_engine
    HAS_ACCELERATION = True
except ImportError:
    HAS_ACCELERATION = False

try:
    from src.engines.smallcap_radar import get_smallcap_radar
    HAS_RADAR = True
except ImportError:
    HAS_RADAR = False

try:
    from src.engines.execution_gate import get_execution_gate
    HAS_GATE = True
except ImportError:
    HAS_GATE = False

logger = get_logger("DAILY_AUDIT_V8")

os.makedirs("data/audit_reports", exist_ok=True)

cache = Cache(ttl=3600)  # 1h cache

# Lead time reference points (UTC) for small caps
# Pre-market movers often spike 4:00-9:30 AM ET = 9:00-14:30 UTC
# RTH movers spike 9:30-11:00 AM ET = 14:30-16:00 UTC
SPIKE_REFERENCE_PREMARKET_UTC = (9, 0)   # 4:00 AM ET
SPIKE_REFERENCE_OPEN_UTC = (14, 30)      # 9:30 AM ET
SPIKE_REFERENCE_MIDDAY_UTC = (16, 0)     # 11:00 AM ET


# ============================
# FETCH TOP GAINERS (Multiple Sources)
# ============================

def fetch_finviz_top_gainers(min_change_pct=20):
    """
    Scrape Finviz for today's top gainers.

    Uses two parsing strategies: first try data-table attribute,
    then fall back to class-based lookup, so that a minor Finviz
    layout change doesn't silently return zero results.
    """
    cache_key = f"finviz_gainers_{datetime.utcnow().date()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("bs4 not installed - Finviz scraping unavailable")
        return []

    try:
        url = "https://finviz.com/screener.ashx?v=111&f=ta_change_u20&ft=4"
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }

        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            logger.warning(f"Finviz returned status {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, 'html.parser')

        # Strategy 1: data-table attribute (modern Finviz)
        table = soup.find('table', attrs={'data-table': 'screener'})
        # Strategy 2: class-based fallback
        if not table:
            table = soup.find('table', {'class': 'table-light'})
        if not table:
            # Strategy 3: find largest table with >5 rows
            for t in soup.find_all('table'):
                rows = t.find_all('tr')
                if len(rows) > 5:
                    table = t
                    break

        if not table:
            logger.warning("Could not find Finviz screener table (layout changed?)")
            return []

        rows = table.find_all('tr')
        if len(rows) < 2:
            logger.warning("Finviz table has no data rows")
            return []

        # Parse header to find column indices dynamically
        header_row = rows[0]
        header_cells = header_row.find_all(['th', 'td'])
        col_map = {}
        for idx, cell in enumerate(header_cells):
            text = cell.get_text(strip=True).lower()
            if text in ('ticker', 'symbol'):
                col_map['ticker'] = idx
            elif text == 'price':
                col_map['price'] = idx
            elif text == 'change':
                col_map['change'] = idx

        # Fallback to hardcoded indices if header parsing fails
        if 'ticker' not in col_map:
            col_map = {'ticker': 1, 'price': 8, 'change': 9}

        gainers = []
        parse_errors = 0

        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) <= max(col_map.values()):
                continue

            try:
                ticker = cells[col_map['ticker']].get_text(strip=True)
                price_text = cells[col_map['price']].get_text(strip=True)
                change_text = cells[col_map['change']].get_text(strip=True)

                if not ticker or not price_text or not change_text:
                    parse_errors += 1
                    continue

                price = float(price_text.replace(',', ''))
                change_pct = float(change_text.replace('%', '').replace('+', ''))

                if change_pct >= min_change_pct:
                    gainers.append({
                        "ticker": ticker,
                        "price": price,
                        "change_pct": change_pct,
                        "source": "finviz",
                        "date": datetime.utcnow().strftime("%Y-%m-%d")
                    })
            except (ValueError, IndexError):
                parse_errors += 1
                continue

        if parse_errors > 0:
            logger.info(f"Finviz: {parse_errors} rows skipped (parse errors)")

        logger.info(f"Finviz: Found {len(gainers)} top gainers (+{min_change_pct}%+)")
        cache.set(cache_key, gainers)
        return gainers

    except Exception as e:
        logger.error(f"Finviz scraping failed: {e}")
        return []


def fetch_finnhub_top_gainers(min_change_pct=20):
    """
    Fetch top gainers from Finnhub API (replaces dead Yahoo endpoint).
    """
    try:
        url = "https://finnhub.io/api/v1/stock/top-gainers"
        params = {"token": FINNHUB_API_KEY}

        r = safe_get(url, params=params, timeout=10)
        data = r.json()

        gainers = []
        for item in data:
            ticker = item.get("symbol", "")
            price = item.get("price", 0)
            change_pct = item.get("changesPercentage", 0)

            if ticker and change_pct >= min_change_pct:
                gainers.append({
                    "ticker": ticker,
                    "price": price,
                    "change_pct": change_pct,
                    "source": "finnhub",
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                })

        logger.info(f"Finnhub: Found {len(gainers)} top gainers (+{min_change_pct}%+)")
        return gainers

    except Exception as e:
        logger.warning(f"Finnhub gainers fetch failed: {e}")
        return []


def fetch_top_gainers(min_change_pct=20):
    """
    Fetch top gainers from multiple sources with deduplication.

    Priority:
    1. Finviz (most comprehensive for small caps)
    2. Finnhub (reliable API fallback)
    3. Merge & deduplicate if both succeed
    """
    finviz_gainers = fetch_finviz_top_gainers(min_change_pct)
    finnhub_gainers = fetch_finnhub_top_gainers(min_change_pct)

    # Merge: Finviz primary, Finnhub fills gaps
    seen_tickers = set()
    merged = []

    for g in finviz_gainers:
        seen_tickers.add(g["ticker"])
        merged.append(g)

    for g in finnhub_gainers:
        if g["ticker"] not in seen_tickers:
            seen_tickers.add(g["ticker"])
            merged.append(g)

    if not merged:
        logger.warning("No top gainers data available from any source")

    logger.info(f"Total gainers (merged): {len(merged)} "
                f"(finviz={len(finviz_gainers)}, finnhub={len(finnhub_gainers)})")
    return merged


# ============================
# LEAD TIME CALCULATION (V8 - multi-reference)
# ============================

def calculate_lead_time(signal_timestamp, mover_date, spike_hour_utc=None):
    """
    Calculate how many hours BEFORE the explosion we detected the signal.

    V8 fix: Instead of assuming 9:30 AM ET for all movers, we use
    the signal's own session context to pick a reference point:
    - Pre-market signals -> reference = 9:00 UTC (4:00 AM ET PM start)
    - RTH signals -> reference = 14:30 UTC (9:30 AM ET open)
    - If spike_hour_utc provided, use it directly (most accurate)

    Returns:
        hours (positive = early detection, negative = late)
    """
    try:
        if isinstance(signal_timestamp, str):
            signal_dt = datetime.fromisoformat(
                signal_timestamp.replace('Z', '+00:00')
            )
        else:
            signal_dt = signal_timestamp

        # Strip timezone for consistent comparison
        if hasattr(signal_dt, 'tzinfo') and signal_dt.tzinfo:
            signal_dt = signal_dt.replace(tzinfo=None)

        if isinstance(mover_date, str):
            mover_dt = datetime.strptime(mover_date, "%Y-%m-%d")
        else:
            mover_dt = datetime.combine(mover_date, datetime.min.time())

        # Pick reference point
        if spike_hour_utc is not None:
            ref_h, ref_m = spike_hour_utc, 0
        elif signal_dt.hour < 14:
            # Pre-market signal -> spike reference = market open
            ref_h, ref_m = SPIKE_REFERENCE_OPEN_UTC
        else:
            # RTH signal -> spike reference = mid-morning
            ref_h, ref_m = SPIKE_REFERENCE_MIDDAY_UTC

        mover_dt = mover_dt.replace(hour=ref_h, minute=ref_m)
        delta_hours = (mover_dt - signal_dt).total_seconds() / 3600

        return round(delta_hours, 2)

    except Exception as e:
        logger.warning(f"Lead time calculation failed: {e}")
        return 0


# ============================
# V8 ACCELERATION STATE TRACKING
# ============================

def collect_v8_stats():
    """
    Collect V8 engine statistics for the current session.

    Returns dict with acceleration engine and radar stats, or empty
    sections if V8 modules are unavailable.
    """
    v8 = {
        "acceleration_engine": {"available": HAS_ACCELERATION},
        "smallcap_radar": {"available": HAS_RADAR},
        "execution_gate": {"available": HAS_GATE},
    }

    if HAS_ACCELERATION:
        try:
            engine = get_acceleration_engine()
            accumulating = engine.get_accumulating()
            top_movers = engine.get_top_movers(limit=20)

            state_counts = {"DORMANT": 0, "ACCUMULATING": 0,
                            "LAUNCHING": 0, "BREAKOUT": 0, "EXHAUSTED": 0}
            for ticker, score in top_movers:
                state = score.state if hasattr(score, 'state') else "UNKNOWN"
                state_counts[state] = state_counts.get(state, 0) + 1

            v8["acceleration_engine"].update({
                "accumulating_count": len(accumulating),
                "top_movers_count": len(top_movers),
                "state_distribution": state_counts,
                "accumulating_tickers": [t for t, _ in accumulating[:10]],
            })
        except Exception as e:
            v8["acceleration_engine"]["error"] = str(e)

    if HAS_RADAR:
        try:
            radar = get_smallcap_radar()
            scan = radar.scan()

            v8["smallcap_radar"].update({
                "critical_blips": len(scan.critical),
                "high_blips": len(scan.high),
                "medium_blips": len(scan.medium),
                "low_blips": len(scan.low),
                "actionable_count": scan.actionable_count,
                "scan_duration_ms": scan.scan_duration_ms,
                "critical_tickers": [b.ticker for b in scan.critical],
                "high_tickers": [b.ticker for b in scan.high[:10]],
            })
        except Exception as e:
            v8["smallcap_radar"]["error"] = str(e)

    if HAS_GATE:
        try:
            gate = get_execution_gate()
            gate_stats = gate.get_stats()

            v8["execution_gate"].update({
                "signals_evaluated": gate_stats.get("signals_evaluated", 0),
                "signals_allowed": gate_stats.get("signals_allowed", 0),
                "signals_blocked": gate_stats.get("signals_blocked", 0),
                "block_rate": gate_stats.get("block_rate", 0),
                "block_breakdown": gate_stats.get("blocked_by_reason", {}),
                "trades_today": gate_stats.get("trades_today", 0),
                "trade_limit": gate_stats.get("trade_limit", 0),
            })
        except Exception as e:
            v8["execution_gate"]["error"] = str(e)

    return v8


# ============================
# FUNNEL ANALYSIS (V8)
# ============================

def analyze_funnel(all_signals_df):
    """
    Analyze the full pipeline funnel:
      SignalProducer (all signals) -> Actionable (BUY/BUY_STRONG)
      -> ExecutionGate (allowed / blocked / reduced)

    Returns dict with funnel metrics. Uses signal metadata when
    available, falls back to ExecutionGate singleton stats.
    """
    if len(all_signals_df) == 0:
        return {
            "total_signals": 0,
            "actionable": 0,
            "non_actionable": 0,
            "funnel_rate": 0,
            "gate_stats": {},
        }

    total = len(all_signals_df)

    # Count signal types
    type_counts = {}
    if "signal_type" in all_signals_df.columns:
        type_counts = all_signals_df["signal_type"].value_counts().to_dict()

    actionable_types = {"BUY", "BUY_STRONG"}
    actionable = sum(v for k, v in type_counts.items() if k in actionable_types)
    non_actionable = total - actionable

    # V8 types
    early_signals = type_counts.get("EARLY_SIGNAL", 0)
    watch_signals = type_counts.get("WATCH", 0)

    # Gate stats from singleton
    gate_stats = {}
    if HAS_GATE:
        try:
            gate = get_execution_gate()
            gate_stats = gate.get_stats()
        except Exception:
            pass

    return {
        "total_signals": total,
        "actionable": actionable,
        "non_actionable": non_actionable,
        "early_signals": early_signals,
        "watch_signals": watch_signals,
        "signal_type_breakdown": type_counts,
        "funnel_rate": actionable / total if total > 0 else 0,
        "gate_stats": gate_stats,
    }


# ============================
# ANALYZE HITS (V8)
# ============================

def analyze_hits(signals_df, gainers):
    """
    Analyze successfully detected movers.

    V8 improvements:
    - Tracks which acceleration state the signal was in
    - Records V8 metadata (acceleration_score, volume_zscore)
    - Better lead time with session-aware reference points
    """
    hits = []

    for gainer in gainers:
        ticker = gainer["ticker"]
        gainer_date = gainer["date"]

        if len(signals_df) == 0:
            continue

        ticker_signals = signals_df[signals_df["ticker"] == ticker]
        if len(ticker_signals) == 0:
            continue

        # Get earliest signal
        earliest = ticker_signals.sort_values("timestamp").iloc[0]
        lead_time_hours = calculate_lead_time(
            earliest["timestamp"], gainer_date
        )

        # Extract V8 metadata from signal if stored
        metadata = {}
        raw_meta = earliest.get("metadata", "{}")
        if isinstance(raw_meta, str):
            try:
                metadata = json.loads(raw_meta) if raw_meta else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        hit = {
            "ticker": ticker,
            "gainer_change_pct": gainer["change_pct"],
            "signal_type": earliest.get("signal_type", "UNKNOWN"),
            "monster_score": earliest.get("monster_score", 0),
            "pattern_score": earliest.get("pattern_score", 0),
            "event_impact": earliest.get("event_impact", 0),
            "lead_time_hours": lead_time_hours,
            "early_catch": lead_time_hours > 2,
            "on_time": 0 <= lead_time_hours <= 2,
            "signal_count": len(ticker_signals),
            "signal_timestamp": str(earliest.get("timestamp", "")),
            # V8 fields
            "acceleration_state": metadata.get("acceleration_state", "N/A"),
            "acceleration_score": metadata.get("acceleration_score", 0),
            "volume_zscore": metadata.get("volume_zscore", 0),
        }
        hits.append(hit)

    if not hits:
        return {
            "hit_count": 0,
            "hit_rate": 0,
            "early_catch_rate": 0,
            "on_time_rate": 0,
            "avg_lead_time_hours": 0,
            "v8_state_distribution": {},
            "hits": []
        }

    hit_count = len(hits)
    total_gainers = max(1, len(gainers))
    early_catches = sum(1 for h in hits if h["early_catch"])
    on_time_catches = sum(1 for h in hits if h["on_time"])

    # V8: which acceleration states produced hits?
    state_dist = {}
    for h in hits:
        st = h.get("acceleration_state", "N/A")
        state_dist[st] = state_dist.get(st, 0) + 1

    return {
        "hit_count": hit_count,
        "hit_rate": hit_count / total_gainers,
        "early_catch_rate": early_catches / max(1, hit_count),
        "on_time_rate": on_time_catches / max(1, hit_count),
        "avg_lead_time_hours": sum(h["lead_time_hours"] for h in hits) / hit_count,
        "v8_state_distribution": state_dist,
        "hits": hits
    }


# ============================
# ANALYZE MISSES (V8 - categorized)
# ============================

def analyze_misses(all_signals_df, buy_signals_df, gainers):
    """
    Analyze movers we missed with CATEGORIZED reasons.

    V8 categories:
    - outside_universe: ticker not in our tradable universe
    - no_signal: in universe but no signal generated at all
    - weak_signal: signal generated but below BUY threshold (WATCH/EARLY_SIGNAL)
    - blocked_by_gate: BUY signal generated but ExecutionGate blocked it
    - low_score: BUY signal but monster_score < 0.50
    """
    # Build ticker sets
    all_signal_tickers = set()
    buy_signal_tickers = set()

    if len(all_signals_df) > 0 and "ticker" in all_signals_df.columns:
        all_signal_tickers = set(all_signals_df["ticker"].unique())
    if len(buy_signals_df) > 0 and "ticker" in buy_signals_df.columns:
        buy_signal_tickers = set(buy_signals_df["ticker"].unique())

    gainer_tickers = set(g["ticker"] for g in gainers)

    # Load current universe for outside_universe check
    try:
        universe_df = load_universe()
        universe_tickers = set(universe_df["ticker"].values) if len(universe_df) > 0 else set()
    except Exception:
        universe_tickers = set()

    misses = []
    reason_counts = {
        "outside_universe": 0,
        "no_signal": 0,
        "weak_signal": 0,
        "blocked_by_gate": 0,
        "low_score": 0,
    }

    for gainer in gainers:
        ticker = gainer["ticker"]

        if ticker in buy_signal_tickers:
            continue  # Not a miss - we had a BUY signal

        # Determine reason
        if universe_tickers and ticker not in universe_tickers:
            reason = "outside_universe"
        elif ticker not in all_signal_tickers:
            reason = "no_signal"
        elif ticker in all_signal_tickers and ticker not in buy_signal_tickers:
            # We had some signal but not BUY/BUY_STRONG
            # Check what type of signal we had
            ticker_sigs = all_signals_df[all_signals_df["ticker"] == ticker]
            signal_types = set(ticker_sigs["signal_type"].values) if len(ticker_sigs) > 0 else set()

            if signal_types & {"WATCH", "EARLY_SIGNAL", "NO_SIGNAL"}:
                reason = "weak_signal"
            else:
                reason = "blocked_by_gate"
        else:
            reason = "no_signal"

        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Get additional context for the miss
        miss_detail = {
            "ticker": ticker,
            "change_pct": gainer["change_pct"],
            "reason": reason,
            "source": gainer.get("source", "unknown"),
            "in_universe": ticker in universe_tickers if universe_tickers else "unknown",
        }

        # If we had any signal, attach its details
        if ticker in all_signal_tickers and len(all_signals_df) > 0:
            ticker_sigs = all_signals_df[all_signals_df["ticker"] == ticker]
            if len(ticker_sigs) > 0:
                best = ticker_sigs.sort_values("monster_score", ascending=False).iloc[0]
                miss_detail["best_signal_type"] = best.get("signal_type", "")
                miss_detail["best_monster_score"] = float(best.get("monster_score", 0))

        misses.append(miss_detail)

    total_gainers = max(1, len(gainers))

    return {
        "miss_count": len(misses),
        "miss_rate": len(misses) / total_gainers,
        "reason_breakdown": reason_counts,
        "missed_tickers": [m["ticker"] for m in misses],
        "misses": misses
    }


# ============================
# ANALYZE FALSE POSITIVES (V8 - graduated)
# ============================

def analyze_false_positives(signals_df, gainers, soft_threshold_pct=10):
    """
    Analyze signals that didn't result in significant moves.

    V8 fix: graduated classification instead of binary.
    - hard_fp: signal ticker gained < 5% (clearly wrong)
    - soft_fp: signal ticker gained 5%-threshold (close but not a gainer)
    - near_miss: signal ticker gained threshold-20% (almost caught it)

    A ticker that gains +15% but our threshold is +20% is a near_miss,
    not a false positive.
    """
    gainer_tickers = set(g["ticker"] for g in gainers)
    gainer_changes = {g["ticker"]: g["change_pct"] for g in gainers}

    signal_tickers = set()
    if len(signals_df) > 0 and "ticker" in signals_df.columns:
        signal_tickers = set(signals_df["ticker"].unique())

    fp_tickers = signal_tickers - gainer_tickers

    hard_fp = []
    soft_fp = []
    near_miss = []

    for ticker in fp_tickers:
        if len(signals_df) == 0:
            continue

        ticker_signals = signals_df[signals_df["ticker"] == ticker]
        if len(ticker_signals) == 0:
            continue

        signal = ticker_signals.sort_values("monster_score", ascending=False).iloc[0]

        detail = {
            "ticker": ticker,
            "signal_type": signal.get("signal_type", ""),
            "monster_score": float(signal.get("monster_score", 0)),
            "actual_change_pct": gainer_changes.get(ticker, 0),
        }

        actual = gainer_changes.get(ticker, 0)
        if actual >= soft_threshold_pct:
            near_miss.append(detail)
        elif actual >= 5:
            soft_fp.append(detail)
        else:
            hard_fp.append(detail)

    return {
        "fp_count": len(fp_tickers),
        "hard_fp_count": len(hard_fp),
        "soft_fp_count": len(soft_fp),
        "near_miss_count": len(near_miss),
        "hard_fp": hard_fp,
        "soft_fp": soft_fp,
        "near_miss": near_miss,
        "fp_tickers": list(fp_tickers),
    }


# ============================
# TREND TRACKING (V8)
# ============================

def load_previous_audit(audit_date):
    """
    Load the previous trading day's audit for trend comparison.
    Skips weekends automatically.
    """
    for days_back in range(1, 5):
        prev_date = audit_date - timedelta(days=days_back)
        # Skip weekends
        if prev_date.weekday() >= 5:
            continue
        filename = f"data/audit_reports/daily_audit_{prev_date.isoformat()}.json"
        if os.path.exists(filename):
            try:
                with open(filename) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


def compute_trend(current_report, previous_report):
    """
    Compare current audit against previous day to detect trends.
    """
    if not previous_report:
        return {"comparison_available": False}

    curr = current_report.get("summary", {})
    prev = previous_report.get("summary", {})

    def delta(key):
        return curr.get(key, 0) - prev.get(key, 0)

    return {
        "comparison_available": True,
        "previous_date": previous_report.get("audit_date", ""),
        "hit_rate_delta": round(delta("hit_rate"), 4),
        "early_catch_delta": round(delta("early_catch_rate"), 4),
        "miss_rate_delta": round(delta("miss_rate"), 4),
        "fp_delta": delta("fp_count"),
        "lead_time_delta": round(delta("avg_lead_time_hours"), 2),
        "grade_previous": previous_report.get("performance_grade", "?"),
        "improving": delta("hit_rate") > 0 and delta("miss_rate") <= 0,
    }


# ============================
# PERFORMANCE GRADE (V8)
# ============================

def calculate_performance_grade(hit_analysis, miss_analysis, fp_analysis):
    """
    Calculate overall performance grade.

    V8: Uses hard_fp_count (not total fp including near-misses)
    for a fairer penalty.

    A+: Hit rate > 70%, early catch > 60%
    A:  Hit rate > 60%, early catch > 50%
    B:  Hit rate > 40%, early catch > 30%
    C:  Hit rate > 25%, early catch > 15%
    D:  Hit rate > 10%
    F:  Hit rate < 10%
    """
    hit_rate = hit_analysis.get("hit_rate", 0)
    early_rate = hit_analysis.get("early_catch_rate", 0)

    # V8: penalize only hard false positives
    hard_fp = fp_analysis.get("hard_fp_count", fp_analysis.get("fp_count", 0))
    fp_penalty = min(0.2, hard_fp * 0.02)

    adjusted_hit = hit_rate - fp_penalty

    if adjusted_hit >= 0.7 and early_rate >= 0.6:
        return "A+"
    elif adjusted_hit >= 0.6 and early_rate >= 0.5:
        return "A"
    elif adjusted_hit >= 0.4 and early_rate >= 0.3:
        return "B"
    elif adjusted_hit >= 0.25 and early_rate >= 0.15:
        return "C"
    elif adjusted_hit >= 0.1:
        return "D"
    else:
        return "F"


# ============================
# MAIN DAILY AUDIT
# ============================

def run_daily_audit(audit_date=None, min_change_pct=20, send_telegram=True):
    """
    Run daily audit for a specific date.

    V8 improvements:
    - Full funnel analysis (Producer -> Gate)
    - V8 acceleration state tracking
    - Categorized miss analysis
    - Graduated false positive scoring
    - Trend comparison with previous day
    """
    if audit_date is None:
        audit_date = datetime.utcnow().date()
    elif isinstance(audit_date, str):
        audit_date = datetime.strptime(audit_date, "%Y-%m-%d").date()

    logger.info("=" * 60)
    logger.info(f"DAILY AUDIT V8 - {audit_date}")
    logger.info("=" * 60)

    # Date range for signals (full day)
    start = datetime.combine(audit_date, datetime.min.time())
    end = datetime.combine(audit_date, datetime.max.time())

    # 1. Get ALL signals (not just BUY) for funnel analysis
    all_signals_df = get_signals_for_period(start, end)
    logger.info(f"Retrieved {len(all_signals_df)} total signals")

    # Filter BUY/BUY_STRONG for hit/miss analysis
    buy_signals_df = pd.DataFrame()
    if len(all_signals_df) > 0 and "signal_type" in all_signals_df.columns:
        buy_signals_df = all_signals_df[
            all_signals_df["signal_type"].isin(["BUY", "BUY_STRONG"])
        ]

    logger.info(f"  -> {len(buy_signals_df)} BUY/BUY_STRONG signals")

    # 2. Funnel analysis
    funnel = analyze_funnel(all_signals_df)
    logger.info(f"Funnel: {funnel['total_signals']} total -> "
                f"{funnel['actionable']} actionable "
                f"({funnel['funnel_rate']*100:.0f}%)")

    # 3. Get real top gainers
    gainers = fetch_top_gainers(min_change_pct=min_change_pct)
    logger.info(f"Retrieved {len(gainers)} top gainers (+{min_change_pct}%+)")

    if not gainers:
        logger.warning("No gainers data - audit may be incomplete")

    # 4. Analyze hits
    hit_analysis = analyze_hits(buy_signals_df, gainers)
    logger.info(f"Hit Rate: {hit_analysis['hit_rate']*100:.1f}%")
    logger.info(f"Early Catch Rate: {hit_analysis['early_catch_rate']*100:.1f}%")
    logger.info(f"Avg Lead Time: {hit_analysis['avg_lead_time_hours']:.1f}h")
    if hit_analysis.get("v8_state_distribution"):
        logger.info(f"V8 States in Hits: {hit_analysis['v8_state_distribution']}")

    # 5. Analyze misses (V8: categorized)
    miss_analysis = analyze_misses(all_signals_df, buy_signals_df, gainers)
    logger.info(f"Miss Rate: {miss_analysis['miss_rate']*100:.1f}%")
    logger.info(f"Miss Reasons: {miss_analysis['reason_breakdown']}")

    # 6. Analyze false positives (V8: graduated)
    fp_analysis = analyze_false_positives(
        buy_signals_df, gainers, soft_threshold_pct=min_change_pct // 2
    )
    logger.info(f"False Positives: {fp_analysis['fp_count']} "
                f"(hard={fp_analysis['hard_fp_count']}, "
                f"soft={fp_analysis['soft_fp_count']}, "
                f"near_miss={fp_analysis['near_miss_count']})")

    # 7. V8 engine stats
    v8_stats = collect_v8_stats()

    # 8. Performance grade
    grade = calculate_performance_grade(hit_analysis, miss_analysis, fp_analysis)

    # 9. Compile report
    report = {
        "audit_version": "V8.0",
        "audit_date": audit_date.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "min_change_pct": min_change_pct
        },
        "summary": {
            "total_signals": len(all_signals_df),
            "buy_signals": len(buy_signals_df),
            "total_gainers": len(gainers),
            "hit_rate": hit_analysis["hit_rate"],
            "early_catch_rate": hit_analysis["early_catch_rate"],
            "miss_rate": miss_analysis["miss_rate"],
            "fp_count": fp_analysis["fp_count"],
            "hard_fp_count": fp_analysis["hard_fp_count"],
            "avg_lead_time_hours": hit_analysis["avg_lead_time_hours"],
        },
        "performance_grade": grade,
        "funnel": funnel,
        "hit_analysis": hit_analysis,
        "miss_analysis": miss_analysis,
        "fp_analysis": fp_analysis,
        "v8_stats": v8_stats,
        "top_gainers": gainers[:20],
    }

    # 10. Trend comparison
    prev_report = load_previous_audit(audit_date)
    report["trend"] = compute_trend(report, prev_report)
    if report["trend"]["comparison_available"]:
        logger.info(f"Trend vs {report['trend']['previous_date']}: "
                    f"hit_rate {'+'if report['trend']['hit_rate_delta']>=0 else ''}"
                    f"{report['trend']['hit_rate_delta']*100:.1f}pp")

    # 11. Save report
    filename = f"data/audit_reports/daily_audit_{audit_date.isoformat()}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {filename}")

    # 12. Send Telegram summary
    if send_telegram:
        send_daily_audit_summary(report)

    logger.info("=" * 60)
    logger.info("DAILY AUDIT V8 COMPLETED")
    logger.info("=" * 60)

    return report


def send_daily_audit_summary(report):
    """Send audit summary via Telegram."""
    summary = report["summary"]
    grade = report["performance_grade"]
    trend = report.get("trend", {})
    v8 = report.get("v8_stats", {})

    grade_emoji = {
        "A+": "🏆", "A": "🏆", "B": "✅",
        "C": "⚠️", "D": "🔶", "F": "❌"
    }

    # Trend arrows
    def arrow(val):
        if val > 0:
            return "↑"
        elif val < 0:
            return "↓"
        return "→"

    trend_str = ""
    if trend.get("comparison_available"):
        trend_str = (
            f"\n📊 vs {trend['previous_date']}: "
            f"HR {arrow(trend['hit_rate_delta'])} "
            f"EC {arrow(trend['early_catch_delta'])} "
            f"MR {arrow(trend['miss_rate_delta'])}"
        )

    # V8 stats snippet
    v8_str = ""
    accel = v8.get("acceleration_engine", {})
    if accel.get("available") and "state_distribution" in accel:
        dist = accel["state_distribution"]
        v8_str = (
            f"\n🔬 V8 States: ACC={dist.get('ACCUMULATING',0)} "
            f"LCH={dist.get('LAUNCHING',0)} BRK={dist.get('BREAKOUT',0)}"
        )

    # Miss reasons
    miss_reasons = report.get("miss_analysis", {}).get("reason_breakdown", {})
    miss_str = ""
    if miss_reasons:
        parts = [f"{k}={v}" for k, v in miss_reasons.items() if v > 0]
        if parts:
            miss_str = f"\n📋 Miss reasons: {', '.join(parts)}"

    message = (
        f"📊 DAILY AUDIT V8 - {report['audit_date']}\n\n"
        f"{grade_emoji.get(grade, '❓')} Grade: {grade}\n\n"
        f"📈 Hit Rate: {summary['hit_rate']*100:.1f}%\n"
        f"⏱ Early Catches: {summary['early_catch_rate']*100:.1f}%\n"
        f"⏰ Avg Lead Time: {summary['avg_lead_time_hours']:.1f}h\n\n"
        f"❌ Miss Rate: {summary['miss_rate']*100:.1f}%\n"
        f"🎯 FP: {summary['hard_fp_count']} hard / "
        f"{summary['fp_count']} total\n\n"
        f"Signals: {summary['total_signals']} total | "
        f"{summary['buy_signals']} BUY | "
        f"Gainers: {summary['total_gainers']}"
        f"{trend_str}{v8_str}{miss_str}"
    )

    # Top hits
    hits = report["hit_analysis"].get("hits", [])[:3]
    if hits:
        message += "\n\n🎯 TOP HITS:\n"
        for hit in hits:
            state_badge = ""
            if hit.get("acceleration_state") not in ("N/A", None):
                state_badge = f" [{hit['acceleration_state']}]"
            message += (
                f"  {hit['ticker']}: +{hit['gainer_change_pct']:.0f}% "
                f"(lead: {hit['lead_time_hours']:.1f}h){state_badge}\n"
            )

    # Top misses
    misses = report["miss_analysis"].get("misses", [])[:3]
    if misses:
        message += "\n❌ TOP MISSES:\n"
        for m in misses:
            message += f"  {m['ticker']}: +{m['change_pct']:.0f}% ({m['reason']})\n"

    try:
        send_daily_audit_alert(report)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ============================
# WEEKLY SUMMARY (from daily reports)
# ============================

def generate_weekly_summary(end_date=None):
    """
    Generate weekly summary by aggregating daily audit reports.
    """
    if end_date is None:
        end_date = datetime.utcnow().date()

    start_date = end_date - timedelta(days=7)

    # Load all daily reports for the week
    reports = []
    for i in range(7):
        d = start_date + timedelta(days=i)
        filename = f"data/audit_reports/daily_audit_{d.isoformat()}.json"
        if os.path.exists(filename):
            try:
                with open(filename) as f:
                    reports.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue

    if not reports:
        logger.warning("No daily reports found for weekly summary")
        return None

    n = len(reports)

    def avg_metric(key):
        vals = [r["summary"].get(key, 0) for r in reports]
        return sum(vals) / n if n > 0 else 0

    # Aggregate miss reason breakdown
    miss_reasons_total = {}
    for r in reports:
        reasons = r.get("miss_analysis", {}).get("reason_breakdown", {})
        for k, v in reasons.items():
            miss_reasons_total[k] = miss_reasons_total.get(k, 0) + v

    # V8: aggregate acceleration state hits
    v8_state_hits = {}
    for r in reports:
        states = r.get("hit_analysis", {}).get("v8_state_distribution", {})
        for k, v in states.items():
            v8_state_hits[k] = v8_state_hits.get(k, 0) + v

    summary = {
        "audit_version": "V8.0",
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days_with_data": n
        },
        "metrics": {
            "avg_hit_rate": avg_metric("hit_rate"),
            "avg_early_catch_rate": avg_metric("early_catch_rate"),
            "avg_miss_rate": avg_metric("miss_rate"),
            "total_false_positives": sum(
                r["summary"].get("fp_count", 0) for r in reports
            ),
            "total_hard_fp": sum(
                r["summary"].get("hard_fp_count", 0) for r in reports
            ),
            "avg_lead_time_hours": avg_metric("avg_lead_time_hours"),
        },
        "daily_grades": [r.get("performance_grade", "?") for r in reports],
        "miss_reason_totals": miss_reasons_total,
        "v8_state_hits_total": v8_state_hits,
        "trend": (
            "improving"
            if reports[-1]["summary"].get("hit_rate", 0)
               > reports[0]["summary"].get("hit_rate", 0)
            else "declining"
        ),
    }

    filename = f"data/audit_reports/weekly_summary_{end_date.isoformat()}.json"
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Weekly summary saved: {filename}")
    return summary


# ============================
# CLI
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GV2-EDGE Daily Audit V8")
    parser.add_argument("--date", type=str, help="Audit date (YYYY-MM-DD)")
    parser.add_argument("--min-change", type=float, default=20,
                        help="Minimum %% change for gainers")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Skip Telegram notification")
    parser.add_argument("--weekly", action="store_true",
                        help="Generate weekly summary")

    args = parser.parse_args()

    if args.weekly:
        result = generate_weekly_summary()
        if result:
            print("\n📊 WEEKLY SUMMARY V8")
            print(f"Period: {result['period']['start']} to {result['period']['end']}")
            print(f"Days with data: {result['period']['days_with_data']}")
            print(f"Avg Hit Rate: {result['metrics']['avg_hit_rate']*100:.1f}%")
            print(f"Avg Early Catch: {result['metrics']['avg_early_catch_rate']*100:.1f}%")
            print(f"Avg Lead Time: {result['metrics']['avg_lead_time_hours']:.1f}h")
            print(f"Grades: {result['daily_grades']}")
            print(f"Miss Reasons: {result['miss_reason_totals']}")
            print(f"V8 State Hits: {result['v8_state_hits_total']}")
            print(f"Trend: {result['trend']}")
    else:
        report = run_daily_audit(
            audit_date=args.date,
            min_change_pct=args.min_change,
            send_telegram=not args.no_telegram
        )

        print("\n" + "=" * 60)
        print("DAILY AUDIT V8 SUMMARY")
        print("=" * 60)
        print(f"Date: {report['audit_date']}")
        print(f"Grade: {report['performance_grade']}")
        print(f"Hit Rate: {report['summary']['hit_rate']*100:.1f}%")
        print(f"Early Catches: {report['summary']['early_catch_rate']*100:.1f}%")
        print(f"Avg Lead Time: {report['summary']['avg_lead_time_hours']:.1f}h")
        print(f"Misses: {report['summary']['miss_rate']*100:.1f}%")
        print(f"  Reasons: {report['miss_analysis']['reason_breakdown']}")
        print(f"FP: {report['summary']['hard_fp_count']} hard / "
              f"{report['summary']['fp_count']} total")
        print(f"Funnel: {report['funnel']['total_signals']} signals -> "
              f"{report['funnel']['actionable']} actionable")
        if report["trend"].get("comparison_available"):
            t = report["trend"]
            print(f"Trend vs {t['previous_date']}: "
                  f"HR {t['hit_rate_delta']*100:+.1f}pp, "
                  f"EC {t['early_catch_delta']*100:+.1f}pp")
        print("=" * 60)
