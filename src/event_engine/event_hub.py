"""
Event Hub V9 - Central Catalyst Event Hub
==========================================

Aggregates, caches, and enriches catalyst events from all sources:
- SEC 8-K filings (via EventHub pipeline)
- Finnhub company news
- FDA calendar (PDUFA, trials, conferences)
- NLP classification via Grok (18 event types, 5 tiers)

Cache TTL: 15 min. Singleton: get_event_hub()
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

from utils.cache import Cache
from utils.logger import get_logger
from utils.api_guard import safe_get, pool_safe_get
from utils.data_validator import validate_event

from src.event_engine.nlp_event_parser import parse_many_texts

from config import FINNHUB_API_KEY, EVENT_PROXIMITY_DAYS

logger = get_logger("EVENT_HUB")

cache = Cache(ttl=15 * 60)  # 15 min

# FDA Calendar integration (PDUFA + trials)
try:
    from src.fda_calendar import get_all_fda_events
    FDA_CALENDAR_AVAILABLE = True
    logger.info("✅ FDA Calendar integration enabled")
except Exception as e:
    FDA_CALENDAR_AVAILABLE = False
    logger.warning(f"⚠️ FDA Calendar unavailable: {e}")


# ============================
# API ENDPOINTS (OPTIMIZED)
# ============================

FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"
FINNHUB_EARNINGS = "https://finnhub.io/api/v1/calendar/earnings"
FINNHUB_PRESS_RELEASES = "https://finnhub.io/api/v1/press-releases"
FINNHUB_GENERAL_NEWS = "https://finnhub.io/api/v1/news"


# ============================
# Fetch company-specific news
# ============================

def fetch_company_news(ticker, days_back=3):
    """
    Fetch ticker-specific news (MUCH better than general news)
    """
    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    try:
        r = pool_safe_get(FINNHUB_COMPANY_NEWS, params=params, timeout=5, provider="finnhub", task_type="COMPANY_NEWS")
        data = r.json()

        events = []
        for item in data:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            source = item.get("source", "")
            timestamp = item.get("datetime", 0)

            if headline:
                events.append({
                    "text": f"{headline}. {summary}",
                    "ticker": ticker,
                    "source": source,
                    "timestamp": timestamp,
                    "type": "news"
                })

        return events

    except Exception as e:
        logger.warning(f"Company news fetch failed for {ticker}: {e}")
        return []


# ============================
# Fetch earnings calendar
# ============================

def fetch_earnings_events(days_forward=7):
    """
    Fetch upcoming earnings (major catalyst)
    """
    today = datetime.now(timezone.utc)
    from_date = today.strftime("%Y-%m-%d")
    to_date = (today + timedelta(days=days_forward)).strftime("%Y-%m-%d")

    params = {
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    try:
        r = pool_safe_get(FINNHUB_EARNINGS, params=params, timeout=10, provider="finnhub", task_type="EARNINGS_CALENDAR")
        data = r.json()

        events = []
        earnings_list = data.get("earningsCalendar", [])

        for item in earnings_list:
            ticker = item.get("symbol", "")
            date = item.get("date", "")
            eps_estimate = item.get("epsEstimate", 0)
            eps_actual = item.get("epsActual", None)

            if ticker and date:
                # Earnings event
                text = f"Earnings report scheduled for {ticker} on {date}"
                if eps_actual is not None:
                    # Already reported
                    text = f"{ticker} reported earnings: EPS ${eps_actual} vs ${eps_estimate} est"

                events.append({
                    "text": text,
                    "ticker": ticker,
                    "date": date,
                    "type": "earnings",
                    "eps_estimate": eps_estimate,
                    "eps_actual": eps_actual
                })

        logger.info(f"Fetched {len(events)} earnings events")
        return events

    except Exception as e:
        logger.warning(f"Earnings fetch failed: {e}")
        return []


# ============================
# Fetch breaking news (fallback)
# ============================

def fetch_breaking_news(category="general"):
    """
    General market news — returns structured event dicts.
    Extracts ticker from headline text (TickerExtractor) when 'related' is empty.
    """
    params = {
        "category": category,
        "token": FINNHUB_API_KEY
    }

    # Lazy-load ticker extractor (no universe needed — extract_all validates syntax only)
    try:
        from src.processors.ticker_extractor import TickerExtractor
        _extractor = TickerExtractor()
    except Exception:
        _extractor = None

    try:
        r = pool_safe_get(FINNHUB_GENERAL_NEWS, params=params, timeout=10, provider="finnhub", task_type="GENERAL_NEWS")
        data = r.json()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        events = []
        for item in data:
            headline = item.get("headline", "")
            summary  = item.get("summary", "")
            if not headline:
                continue

            # Prefer 'related' field; fallback to NLP extraction from headline
            ticker = (item.get("related", "") or "").strip().upper()
            if not ticker and _extractor:
                extracted = _extractor.extract_all(f"{headline}. {summary}")
                ticker = extracted[0] if extracted else ""

            ts = item.get("datetime", 0)
            try:
                date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else today
            except Exception:
                date = today

            events.append({
                "text":     f"{headline}. {summary}",
                "headline": headline,
                "ticker":   ticker,
                "type":     "news",
                "date":     date,
                "source":   item.get("source", ""),
                "impact":   0.5,
            })

        logger.info(f"Fetched {len(events)} breaking news")
        return events

    except Exception as e:
        logger.warning(f"Breaking news fetch failed: {e}")
        return []


# ============================
# Event proximity boost
# ============================

def proximity_boost(event_date):
    try:
        d = datetime.strptime(event_date, "%Y-%m-%d")
        delta = abs((d - datetime.now(timezone.utc)).days)

        if delta <= 1:
            return 1.5  # ↑ Increased (today/tomorrow = critical)
        elif delta <= 3:
            return 1.3  # ↑ Increased
        elif delta <= EVENT_PROXIMITY_DAYS:
            return 1.1
        else:
            return 1.0

    except:
        return 1.0


# ============================
# Build full event list (OPTIMIZED)
# ============================

def build_events(tickers=None, force_refresh=False):
    """
    OPTIMIZED: Multi-source event aggregation
    
    Sources:
    1. Company-specific news (ticker-focused)
    2. Earnings calendar
    3. Breaking news (fallback)
    
    Args:
        tickers: list of tickers to fetch company news for
        force_refresh: bypass cache
    """
    # FIX: Créer universe_set pour filtrage rapide des FDA events
    universe_set = set(t.upper() for t in tickers) if tickers else set()
    
    cached = cache.get("events_v2")

    if cached and not force_refresh:
        return cached

    # Charge le cache fichier si récent (< 15 min) pour éviter les appels Grok inutiles au redémarrage
    if not force_refresh:
        file_cached = load_cache_if_fresh(max_age_seconds=900)
        if file_cached:
            cache.set("events_v2", file_cached)
            logger.info(f"Event hub: loaded {len(file_cached)} events from fresh file cache (skipping NLP)")
            return file_cached

    try:
        all_events = []

        # SOURCE 1: Company-specific news (if tickers provided)
        if tickers:
            for ticker in tickers[:50]:  # Limit to avoid rate limits
                company_events = fetch_company_news(ticker, days_back=3)
                all_events.extend(company_events)

        # SOURCE 2: Earnings calendar (critical catalyst)
        earnings_events = fetch_earnings_events(days_forward=7)
        all_events.extend(earnings_events)

        # SOURCE 3: Breaking news (structured dicts, ticker from 'related' field)
        breaking_news_events = fetch_breaking_news(category="general")
        all_events.extend(breaking_news_events)

        # SOURCE 4: FDA Calendar (PDUFA + trials + conferences) 🧬
        if FDA_CALENDAR_AVAILABLE:
            try:
                fda_events = get_all_fda_events()

                # Filter by universe if provided
                if universe_set:
                    fda_events = [e for e in fda_events if e.get("ticker", "").upper() in universe_set]

                logger.info(f"FDA Calendar: {len(fda_events)} events (PDUFA/trials/conferences)")
                all_events.extend(fda_events)
            except Exception as e:
                logger.warning(f"FDA calendar fetch failed: {e}")

        # NLP enrichment (best-effort — failures do NOT drop events)
        text_events = [e["text"] for e in all_events if "text" in e]
        try:
            nlp_enriched = parse_many_texts(text_events)
        except Exception as nlp_err:
            logger.warning(f"NLP enrichment skipped: {nlp_err}")
            nlp_enriched = []

        # Direct structured events — ALL types (earnings, news, FDA, …)
        _IMPACT = {
            "earnings": 0.8,
            "PDUFA": 0.85,
            "TRIAL_RESULT": 0.75,
            "CONFERENCE": 0.6,
            "news": 0.5,
        }
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        parsed_events = []
        for event in all_events:
            if "ticker" not in event:
                continue
            etype = event.get("type", "news")
            parsed_events.append({
                "ticker":   event["ticker"],
                "type":     etype,
                "date":     event.get("date", today_str),
                "impact":   _IMPACT.get(etype, 0.6),
                "category": etype,
                "metadata": {k: v for k, v in event.items() if k != "text"},
            })

        # Append NLP-enriched events (may add extra ticker-specific signals)
        parsed_events.extend(nlp_enriched)

        # Apply proximity boost
        boosted_events = []

        for e in parsed_events:
            if not validate_event(e):
                continue

            boost = proximity_boost(e.get("date", ""))

            raw_impact = e.get("impact", 0.5)
            if raw_impact < 0:
                # S5-6: bearish event — apply boost to magnitude, preserve sign, clamp to [-1, 0]
                e["boosted_impact"] = max(-1.0, raw_impact * boost)
                e.setdefault("is_bearish", True)
            else:
                e["boosted_impact"] = min(1.0, raw_impact * boost)
                e.setdefault("is_bearish", False)

            boosted_events.append(e)

        # Deduplicate by ticker + date
        unique_events = deduplicate_events(boosted_events)

        cache.set("events_v2", unique_events)
        save_cache(unique_events)

        logger.info(f"Built {len(unique_events)} unique events from {len(all_events)} raw sources")

        return unique_events

    except Exception as e:
        logger.error(f"Event hub failed: {e}", exc_info=True)
        return load_cache()


# ============================
# Deduplication
# ============================

def deduplicate_events(events):
    """Remove duplicate events. Key: (ticker, date, type) — news uses headline snippet to preserve distinct articles."""
    seen = set()
    unique = []

    for e in events:
        ticker = e.get("ticker", "")
        date   = e.get("date", "")
        etype  = e.get("type", "")
        if etype == "news":
            # Use headline snippet so distinct news articles are kept
            meta = e.get("metadata", {})
            headline = (meta.get("headline", "") if isinstance(meta, dict) else "")[:60]
            key = (etype, ticker, date, headline)
        else:
            key = (ticker, date, etype)

        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# ============================
# Persistence fallback
# ============================

CACHE_FILE = "data/events_cache.json"


def save_cache(events):
    os.makedirs("data", exist_ok=True)

    with open(CACHE_FILE, "w") as f:
        json.dump(events, f, indent=2)


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)

    return []


def load_cache_if_fresh(max_age_seconds=900):
    """Charge le cache fichier seulement s'il est plus récent que max_age_seconds."""
    if not os.path.exists(CACHE_FILE):
        return None
    age = time.time() - os.path.getmtime(CACHE_FILE)
    if age > max_age_seconds:
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        return data if data else None
    except Exception:
        return None


# ============================
# Public API
# ============================

def get_events(tickers=None):
    """
    Get all events, optionally filtered by tickers
    
    Args:
        tickers: list of tickers to fetch company-specific news for
    """
    return build_events(tickers=tickers)


def get_events_by_ticker(ticker):
    """Get events for a specific ticker"""
    events = build_events()

    return [e for e in events if e.get("ticker") == ticker]


def refresh_events(tickers=None):
    """Force refresh events"""
    return build_events(tickers=tickers, force_refresh=True)


if __name__ == "__main__":
    # Test
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    ev = get_events(tickers=test_tickers)
    print(f"Fetched {len(ev)} events")
    print(ev[:3])
