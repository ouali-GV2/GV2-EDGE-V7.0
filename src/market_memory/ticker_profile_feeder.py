"""
Ticker Profile Feeder — GV2-EDGE V9.0
======================================
Fetch multi-source async pour alimenter le TickerProfileStore.

Sources par priorité :
  - Float / shares    : FloatAnalyzer (Yahoo + IBKR fallback)
  - Market cap        : Finnhub /stock/profile2
  - Insider / Inst %  : Yahoo /quoteSummary defaultKeyStatistics
  - Short / DTC       : squeeze_boost (Finnhub short-interest)
  - Borrow rate       : FloatAnalyzer → IBKR tick 236
  - Dilution flags    : DilutionDetector (SEC EDGAR)
  - Reverse splits    : Finnhub /stock/split
  - History           : repeat_gainer_memory + signal_logger
  - ATR / Avg volume  : feature_engine (buffer first)

Appelé par :
  - weekend_scanner._scan_ticker_profiles() (batch hebdomadaire)
  - Usage direct : asyncio.run(update_ticker_profile("AAPL"))
"""

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

from utils.logger import get_logger
from utils.api_guard import pool_safe_get
from config import FINNHUB_API_KEY

from src.market_memory.ticker_profile_store import get_ticker_profile_store

logger = get_logger("TICKER_PROFILE_FEEDER")

FINNHUB_PROFILE2 = "https://finnhub.io/api/v1/stock/profile2"
FINNHUB_SPLIT    = "https://finnhub.io/api/v1/stock/split"
YAHOO_SUMMARY    = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tp-feeder")


# ============================================================================
# Individual data fetchers (sync, run in executor)
# ============================================================================

def _fetch_float_data(ticker: str) -> Dict:
    """Float, shares_outstanding, short_pct, DTC, borrow_rate via FloatAnalyzer."""
    result = {}
    try:
        from src.float_analysis import get_float_analyzer
        fa = get_float_analyzer()
        analysis = fa.analyze(ticker)
        if analysis:
            if analysis.float_shares > 0:
                result["float_shares"] = float(analysis.float_shares)
            if analysis.shares_outstanding > 0:
                result["shares_outstanding"] = float(analysis.shares_outstanding)
            if analysis.short_pct_float > 0:
                result["short_interest_pct"] = round(analysis.short_pct_float, 2)
            if analysis.days_to_cover > 0:
                result["days_to_cover"] = round(analysis.days_to_cover, 2)
            if analysis.cost_to_borrow_pct > 0:
                result["borrow_rate"] = round(analysis.cost_to_borrow_pct, 2)
    except Exception as e:
        logger.debug(f"FloatAnalyzer failed for {ticker}: {e}")
    return result


def _fetch_finnhub_profile(ticker: str) -> Dict:
    """Market cap, shares outstanding (fallback) from Finnhub profile2."""
    result = {}
    try:
        r = pool_safe_get(
            FINNHUB_PROFILE2,
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            provider="finnhub", task_type="PROFILE",
        )
        if r and r.status_code == 200:
            data = r.json()
            mc = data.get("marketCapitalization")
            if mc and mc > 0:
                result["market_cap"] = float(mc) * 1_000_000  # millions → USD
            # Fallback shares if FloatAnalyzer returned nothing
            so = data.get("shareOutstanding")
            if so and so > 0:
                result["_shares_outstanding_fb"] = float(so) * 1_000_000
    except Exception as e:
        logger.debug(f"Finnhub profile2 failed for {ticker}: {e}")
    return result


def _fetch_yahoo_ownership(ticker: str) -> Dict:
    """Insider %, institutional % from Yahoo Finance quoteSummary."""
    result = {}
    try:
        url = YAHOO_SUMMARY.format(ticker=ticker)
        r = pool_safe_get(
            url,
            params={"modules": "defaultKeyStatistics"},
            provider="finnhub",  # routed through pool fallback
            task_type="PROFILE",
            timeout=8,
        )
        if r and r.status_code == 200:
            body = r.json()
            stats = (
                body.get("quoteSummary", {})
                    .get("result", [{}])[0]
                    .get("defaultKeyStatistics", {})
            )
            ins = stats.get("heldPercentInsiders", {}).get("raw")
            if ins is not None:
                result["insider_pct"] = round(ins * 100, 2)
            inst = stats.get("heldPercentInstitutions", {}).get("raw")
            if inst is not None:
                result["institutional_pct"] = round(inst * 100, 2)
            # Float fallback
            fs = stats.get("floatShares", {}).get("raw")
            if fs and fs > 0:
                result["_float_shares_fb"] = float(fs)
    except Exception as e:
        logger.debug(f"Yahoo ownership failed for {ticker}: {e}")
    return result


def _fetch_reverse_splits(ticker: str) -> Dict:
    """Count of reverse splits from Finnhub /stock/split (last 10 years)."""
    result = {"reverse_split_count": 0, "last_reverse_split": None}
    try:
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=3650)).strftime("%Y-%m-%d")
        date_to = now.strftime("%Y-%m-%d")
        r = pool_safe_get(
            FINNHUB_SPLIT,
            params={
                "symbol": ticker, "from": date_from,
                "to": date_to, "token": FINNHUB_API_KEY,
            },
            provider="finnhub", task_type="PROFILE",
        )
        if r and r.status_code == 200:
            splits = r.json()
            if isinstance(splits, list):
                # Reverse split: fromFactor < toFactor (e.g., 1→10 = 1:10 reverse)
                rev = [s for s in splits if s.get("fromFactor", 1) < s.get("toFactor", 1)]
                result["reverse_split_count"] = len(rev)
                if rev:
                    last = max(rev, key=lambda s: s.get("date", ""))
                    result["last_reverse_split"] = last.get("date")
    except Exception as e:
        logger.debug(f"Finnhub split failed for {ticker}: {e}")
    return result


def _fetch_dilution_flags(ticker: str) -> Dict:
    """Shelf/ATM/warrants/dilution_tier from DilutionDetector."""
    result = {}
    try:
        from src.risk_guard.dilution_detector import DilutionDetector
        detector = DilutionDetector()
        # Run async dilution check in a fresh event loop (called from executor)
        loop = asyncio.new_event_loop()
        try:
            profile = loop.run_until_complete(detector.analyze(ticker))
        finally:
            loop.close()
        if profile:
            result["shelf_active"] = 1 if profile.has_recent_s3 else 0
            result["atm_active"]   = 1 if profile.has_active_atm else 0
            result["warrants_outstanding"] = 1 if (profile.warrants_outstanding or 0) > 0 else 0
            result["dilution_tier"] = profile.dilution_tier.value if profile.dilution_tier else "NONE"
    except Exception as e:
        logger.debug(f"DilutionDetector failed for {ticker}: {e}")
    return result


def _fetch_gainer_history(ticker: str) -> Dict:
    """top_gainer_count, avg_move_pct, best_session, catalyst_affinity from repeat_gainer_memory."""
    result = {}
    try:
        from src.repeat_gainer_memory import get_ticker_history
        history = get_ticker_history(ticker, days_back=365)
        if history:
            result["top_gainer_count"] = len(history)
            moves = [h.gain_pct for h in history if h.gain_pct]
            if moves:
                result["avg_move_pct"] = round(sum(moves) / len(moves), 1)
            # Catalyst affinity from most common catalyst
            catalysts = [h.catalyst for h in history if h.catalyst]
            if catalysts:
                result["catalyst_affinity"] = max(set(catalysts), key=catalysts.count)
    except Exception as e:
        logger.debug(f"repeat_gainer_memory failed for {ticker}: {e}")
    return result


def _fetch_signal_history(ticker: str) -> Dict:
    """best_session, halt_count from signal_logger."""
    result = {}
    try:
        from src.signal_logger import get_signal_history
        df = get_signal_history(days_back=90)
        if df is not None and not df.empty:
            t_df = df[df["ticker"] == ticker] if "ticker" in df.columns else df.iloc[0:0]
            if not t_df.empty:
                # Best session = most common session in BUY/BUY_STRONG signals
                buy_df = t_df[t_df.get("signal_type", "").isin(["BUY", "BUY_STRONG"])] \
                    if "signal_type" in t_df.columns else t_df.iloc[0:0]
                if not buy_df.empty and "session" in buy_df.columns:
                    sessions = buy_df["session"].dropna().tolist()
                    if sessions:
                        result["best_session"] = max(set(sessions), key=sessions.count)
                # Halt count from execution status
                if "execution_status" in t_df.columns:
                    halts = t_df[t_df["execution_status"].str.contains("HALT|BLOCKED", na=False)]
                    result["halt_count"] = len(halts)
    except Exception as e:
        logger.debug(f"signal_logger failed for {ticker}: {e}")
    return result


def _fetch_technical_baseline(ticker: str) -> Dict:
    """ATR_14, avg_daily_volume from feature_engine (buffer first)."""
    result = {}
    try:
        from src.feature_engine import fetch_candles
        df = fetch_candles(ticker, resolution="D", lookback=1440)  # ~20 trading days
        if df is not None and len(df) >= 5:
            # ATR 14 (True Range average)
            tr = (df["high"] - df["low"]).abs()
            result["atr_14"] = round(tr.tail(14).mean(), 4)
            result["avg_daily_volume"] = round(df["volume"].mean(), 0)
    except Exception as e:
        logger.debug(f"feature_engine failed for {ticker}: {e}")
    return result


# ============================================================================
# Main feeder
# ============================================================================

async def update_ticker_profile(ticker: str) -> Optional[Dict]:
    """
    Fetch all data sources for one ticker and upsert into TickerProfileStore.

    Returns the profile dict, or None on critical failure.
    """
    ticker = ticker.upper().strip()
    store = get_ticker_profile_store()

    loop = asyncio.get_event_loop()
    source_flags: Dict[str, bool] = {}

    # Run all sync fetchers concurrently in executor threads
    tasks = {
        "float":    loop.run_in_executor(_executor, _fetch_float_data, ticker),
        "profile":  loop.run_in_executor(_executor, _fetch_finnhub_profile, ticker),
        "yahoo":    loop.run_in_executor(_executor, _fetch_yahoo_ownership, ticker),
        "splits":   loop.run_in_executor(_executor, _fetch_reverse_splits, ticker),
        "dilution": loop.run_in_executor(_executor, _fetch_dilution_flags, ticker),
        "history":  loop.run_in_executor(_executor, _fetch_gainer_history, ticker),
        "signals":  loop.run_in_executor(_executor, _fetch_signal_history, ticker),
        "technical":loop.run_in_executor(_executor, _fetch_technical_baseline, ticker),
    }

    results = {}
    for name, coro in tasks.items():
        try:
            results[name] = await asyncio.wait_for(coro, timeout=20.0)
            source_flags[name] = bool(results[name])
        except Exception as e:
            logger.debug(f"Source {name} timed out/failed for {ticker}: {e}")
            results[name] = {}
            source_flags[name] = False

    # Merge all results into one profile dict
    profile: Dict = {"ticker": ticker}

    # Float data (primary)
    float_data = results.get("float", {})
    profile.update({k: v for k, v in float_data.items() if not k.startswith("_")})

    # Finnhub profile2 (market_cap + fallback shares)
    finnhub_data = results.get("profile", {})
    if finnhub_data.get("market_cap"):
        profile["market_cap"] = finnhub_data["market_cap"]
    if not profile.get("shares_outstanding") and finnhub_data.get("_shares_outstanding_fb"):
        profile["shares_outstanding"] = finnhub_data["_shares_outstanding_fb"]

    # Yahoo (insider/institutional + float fallback)
    yahoo_data = results.get("yahoo", {})
    for field in ("insider_pct", "institutional_pct"):
        if yahoo_data.get(field) is not None:
            profile[field] = yahoo_data[field]
    if not profile.get("float_shares") and yahoo_data.get("_float_shares_fb"):
        profile["float_shares"] = yahoo_data["_float_shares_fb"]

    # Reverse splits
    splits_data = results.get("splits", {})
    profile["reverse_split_count"] = splits_data.get("reverse_split_count", 0)
    if splits_data.get("last_reverse_split"):
        profile["last_reverse_split"] = splits_data["last_reverse_split"]

    # Dilution flags
    profile.update(results.get("dilution", {}))

    # Historical behavior
    history_data = results.get("history", {})
    for field in ("top_gainer_count", "avg_move_pct", "catalyst_affinity"):
        if history_data.get(field) is not None:
            profile[field] = history_data[field]

    # Signal history
    signal_data = results.get("signals", {})
    for field in ("best_session", "halt_count"):
        if signal_data.get(field) is not None:
            profile[field] = signal_data[field]

    # Technical baseline
    profile.update(results.get("technical", {}))

    # Source flags
    profile["source_flags"] = source_flags

    # Upsert
    ok = store.upsert(profile)
    if ok:
        quality = profile.get("data_quality", 0)
        logger.info(f"Profile updated: {ticker} quality={quality:.0%} sources={sum(source_flags.values())}/{len(source_flags)}")
    else:
        logger.warning(f"Profile upsert failed for {ticker}")
        return None

    return profile


async def batch_update(
    tickers: List[str],
    max_concurrent: int = 3,
    max_age_days: int = 7,
    force: bool = False,
) -> Dict:
    """
    Update profiles for a list of tickers respecting max_concurrent limit.

    Args:
        tickers: list of ticker symbols
        max_concurrent: max parallel fetches (default 3 — API friendly)
        max_age_days: skip tickers updated within N days (unless force=True)
        force: update all tickers regardless of age

    Returns:
        {"updated": N, "skipped": N, "failed": N}
    """
    store = get_ticker_profile_store()
    sem = asyncio.Semaphore(max_concurrent)
    stats = {"updated": 0, "skipped": 0, "failed": 0}

    async def _update_one(ticker: str):
        if not force and not store.needs_update(ticker, max_age_days):
            stats["skipped"] += 1
            return
        async with sem:
            result = await update_ticker_profile(ticker)
            if result:
                stats["updated"] += 1
            else:
                stats["failed"] += 1

    await asyncio.gather(*[_update_one(t) for t in tickers], return_exceptions=True)

    logger.info(
        f"batch_update done: {stats['updated']} updated, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats
