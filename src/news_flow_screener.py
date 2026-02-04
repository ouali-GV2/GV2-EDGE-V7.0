"""
NEWS FLOW SCREENER - Global News → Ticker Mapping
==================================================

Architecture inversée pour détection anticipative:

AVANT (inefficace):
    Pour chaque ticker → chercher ses news → analyser
    Problème: 500 tickers × API calls = lent + rate limits

MAINTENANT (efficace):
    1. Fetch ALL breaking news (market-wide)
    2. NLP filter: garder seulement news à fort impact potentiel
    3. Extract tickers mentionnés
    4. Filter: garder seulement small caps US <$2B
    5. Output: {ticker: [events]} prêt pour scoring

Sources:
- Polygon (via Grok): news ticker-specific + market-wide
- Finnhub: company news + market news
- SEC EDGAR: 8-K filings (material events)

Impact keywords for filtering:
- FDA, approval, trial, phase, breakthrough
- Earnings, beat, miss, guidance, revenue
- Merger, acquisition, buyout, takeover
- Contract, partnership, deal, agreement
- Upgrade, downgrade, price target
- Short squeeze, gamma, options
"""

import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get, safe_post
from config import GROK_API_KEY, FINNHUB_API_KEY

logger = get_logger("NEWS_FLOW_SCREENER")

# Cache pour éviter doublons
news_cache = Cache(ttl=900)  # 15 min

# Keywords pour filtrer les news à fort impact
BULLISH_KEYWORDS = [
    # FDA / Biotech
    'fda', 'approval', 'approved', 'clearance', 'breakthrough', 'designation',
    'trial', 'phase 3', 'phase iii', 'positive', 'efficacy', 'endpoint',
    'pdufa', 'nda', 'bla', 'eua',
    
    # Earnings
    'beat', 'beats', 'exceeded', 'surpass', 'record revenue', 'record earnings',
    'guidance raise', 'raises guidance', 'upside', 'outperform',
    
    # M&A
    'acquisition', 'acquire', 'merger', 'buyout', 'takeover', 'bid',
    'offer', 'deal', 'transaction',
    
    # Contracts
    'contract', 'award', 'partnership', 'agreement', 'collaboration',
    'license', 'milestone', 'order',
    
    # Analyst
    'upgrade', 'price target', 'buy rating', 'outperform',
    
    # Squeeze / Options
    'short squeeze', 'gamma', 'options activity', 'unusual volume',
    
    # General positive
    'surge', 'soar', 'jump', 'spike', 'rally', 'breakout'
]

# Keywords négatifs (à éviter ou inverser)
BEARISH_KEYWORDS = [
    'downgrade', 'miss', 'below', 'guidance cut', 'warning',
    'reject', 'fail', 'negative', 'decline', 'drop', 'plunge',
    'bankruptcy', 'default', 'dilution', 'offering'
]

# Types d'events et leur impact potentiel
EVENT_TYPES = {
    'FDA_APPROVAL': {'keywords': ['fda approv', 'clearance', 'breakthrough'], 'base_impact': 0.9},
    'FDA_TRIAL': {'keywords': ['phase', 'trial', 'endpoint', 'efficacy'], 'base_impact': 0.8},
    'EARNINGS_BEAT': {'keywords': ['beat', 'exceeded', 'record'], 'base_impact': 0.7},
    'GUIDANCE_RAISE': {'keywords': ['guidance raise', 'raises guidance', 'outlook'], 'base_impact': 0.7},
    'MERGER_ACQUISITION': {'keywords': ['acquisition', 'merger', 'buyout', 'takeover'], 'base_impact': 0.85},
    'CONTRACT_WIN': {'keywords': ['contract', 'award', 'deal', 'order'], 'base_impact': 0.6},
    'PARTNERSHIP': {'keywords': ['partnership', 'collaboration', 'agreement'], 'base_impact': 0.5},
    'ANALYST_UPGRADE': {'keywords': ['upgrade', 'price target', 'buy rating'], 'base_impact': 0.4},
    'SHORT_SQUEEZE': {'keywords': ['short squeeze', 'gamma', 'squeeze'], 'base_impact': 0.6},
    'BREAKING_POSITIVE': {'keywords': ['surge', 'soar', 'spike', 'rally'], 'base_impact': 0.5}
}


@dataclass
class NewsEvent:
    """Single news event with extracted data"""
    headline: str
    summary: str
    tickers: List[str]
    event_type: str
    impact_score: float
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    source: str
    published_at: str
    url: Optional[str] = None


@dataclass 
class TickerEvents:
    """All events for a single ticker"""
    ticker: str
    events: List[NewsEvent]
    total_impact: float
    dominant_event_type: str
    event_count: int


# ============================
# NEWS FETCHING
# ============================

def fetch_polygon_news_global(hours_back: int = 6) -> List[Dict]:
    """
    Fetch market-wide news from Polygon via Grok
    
    Returns raw news items for processing
    """
    logger.info(f"Fetching Polygon news (last {hours_back}h)...")
    
    prompt = f"""Execute this Python code to fetch recent market news from Polygon:

```python
from polygon import RESTClient
from datetime import datetime, timedelta

client = RESTClient()

# Get news from last {hours_back} hours
cutoff = (datetime.utcnow() - timedelta(hours={hours_back})).strftime('%Y-%m-%dT%H:%M:%SZ')

# Fetch market-wide news (no ticker filter)
news = list(client.list_ticker_news(
    limit=100,
    order='desc',
    sort='published_utc',
    published_utc_gte=cutoff
))

results = []
for n in news:
    results.append({{
        'title': n.title,
        'tickers': n.tickers if hasattr(n, 'tickers') else [],
        'published': str(n.published_utc),
        'url': n.article_url if hasattr(n, 'article_url') else '',
        'keywords': n.keywords if hasattr(n, 'keywords') else [],
        'description': n.description if hasattr(n, 'description') else ''
    }})

print(results)
```

Return ONLY the raw Python list output, no other text."""

    try:
        payload = {
            "model": "grok-4-1-fast-reasoning",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        r = safe_post(
            "https://api.x.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        
        # Parse the list from response
        list_match = re.search(r'\[.*\]', content, re.DOTALL)
        if list_match:
            news_list = json.loads(list_match.group())
            logger.info(f"Polygon: fetched {len(news_list)} news items")
            return news_list
        
        return []
        
    except Exception as e:
        logger.error(f"Polygon news fetch failed: {e}")
        return []


def fetch_finnhub_general_news() -> List[Dict]:
    """
    Fetch general market news from Finnhub
    """
    logger.info("Fetching Finnhub general news...")
    
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {
            "category": "general",
            "token": FINNHUB_API_KEY
        }
        
        r = safe_get(url, params=params, timeout=10)
        news = r.json()
        
        results = []
        for n in news[:50]:  # Limit to 50 most recent
            results.append({
                'title': n.get('headline', ''),
                'tickers': [],  # Finnhub general news doesn't include tickers
                'published': datetime.fromtimestamp(n.get('datetime', 0)).isoformat(),
                'url': n.get('url', ''),
                'keywords': [],
                'description': n.get('summary', '')
            })
        
        logger.info(f"Finnhub: fetched {len(results)} news items")
        return results
        
    except Exception as e:
        logger.error(f"Finnhub news fetch failed: {e}")
        return []


# ============================
# NLP PROCESSING
# ============================

def filter_high_impact_news(news_items: List[Dict]) -> List[Dict]:
    """
    Filter news to keep only potentially high-impact items
    
    Uses keyword matching for speed, then Grok for deeper analysis
    """
    high_impact = []
    
    for item in news_items:
        title = item.get('title', '').lower()
        desc = item.get('description', '').lower()
        combined = f"{title} {desc}"
        
        # Check for bullish keywords
        bullish_matches = sum(1 for kw in BULLISH_KEYWORDS if kw in combined)
        bearish_matches = sum(1 for kw in BEARISH_KEYWORDS if kw in combined)
        
        # Keep if more bullish than bearish, or if contains key FDA/M&A keywords
        if bullish_matches > bearish_matches or bullish_matches >= 2:
            item['_bullish_score'] = bullish_matches
            item['_bearish_score'] = bearish_matches
            high_impact.append(item)
    
    logger.info(f"Filtered to {len(high_impact)} high-impact news from {len(news_items)}")
    
    return high_impact


def classify_event_type(title: str, description: str) -> Tuple[str, float]:
    """
    Classify news into event type and estimate impact
    
    Returns: (event_type, impact_score)
    """
    combined = f"{title} {description}".lower()
    
    best_match = ('BREAKING_POSITIVE', 0.3)
    
    for event_type, config in EVENT_TYPES.items():
        for keyword in config['keywords']:
            if keyword in combined:
                if config['base_impact'] > best_match[1]:
                    best_match = (event_type, config['base_impact'])
                break
    
    return best_match


def extract_tickers_from_news(news_item: Dict, universe_tickers: Set[str]) -> List[str]:
    """
    Extract valid tickers from news item
    
    Sources:
    1. Explicit tickers in news data
    2. Ticker patterns in title/description
    3. Company name → ticker mapping
    
    Filter: only keep tickers in our small cap universe
    """
    found_tickers = set()
    
    # 1. Explicit tickers from news source
    if news_item.get('tickers'):
        for t in news_item['tickers']:
            if t.upper() in universe_tickers:
                found_tickers.add(t.upper())
    
    # 2. Ticker pattern matching in text
    title = news_item.get('title', '')
    desc = news_item.get('description', '')
    combined = f"{title} {desc}"
    
    # Pattern: $TICKER or (TICKER) or "TICKER:"
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',           # $NVDA
        r'\(([A-Z]{1,5})\)',            # (NVDA)
        r'\b([A-Z]{2,5}):\s',           # NVDA:
        r'\b([A-Z]{2,5})\s(?:stock|shares|inc|corp|ltd)', # NVDA stock
    ]
    
    for pattern in ticker_patterns:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        for match in matches:
            ticker = match.upper()
            if ticker in universe_tickers and len(ticker) >= 2:
                found_tickers.add(ticker)
    
    return list(found_tickers)


def analyze_news_with_grok(news_items: List[Dict], universe_tickers: Set[str]) -> List[NewsEvent]:
    """
    Use Grok for deep NLP analysis of news items
    
    - Extract all tickers mentioned
    - Classify event type
    - Score impact
    - Determine sentiment
    """
    if not news_items:
        return []
    
    logger.info(f"Grok analyzing {len(news_items)} news items...")
    
    # Batch for efficiency
    batch_size = 20
    all_events = []
    
    for i in range(0, len(news_items), batch_size):
        batch = news_items[i:i + batch_size]
        
        # Format news for Grok
        news_text = ""
        for idx, item in enumerate(batch):
            news_text += f"\n[{idx}] {item.get('title', '')}\n"
            if item.get('description'):
                news_text += f"    {item.get('description', '')[:200]}\n"
        
        prompt = f"""Analyze these news items for stock trading signals.

NEWS ITEMS:
{news_text}

For each news item, extract:
1. ALL stock tickers mentioned (format: uppercase, 1-5 letters)
2. Event type: FDA_APPROVAL, FDA_TRIAL, EARNINGS_BEAT, GUIDANCE_RAISE, MERGER_ACQUISITION, CONTRACT_WIN, PARTNERSHIP, ANALYST_UPGRADE, SHORT_SQUEEZE, BREAKING_POSITIVE, or NONE
3. Impact score: 0.0 to 1.0 (how likely to cause +20%+ move in small cap)
4. Sentiment: BULLISH, BEARISH, or NEUTRAL

Return ONLY a JSON array:
[
    {{
        "index": 0,
        "tickers": ["NVDA", "AMD"],
        "event_type": "EARNINGS_BEAT",
        "impact_score": 0.7,
        "sentiment": "BULLISH"
    }}
]

Skip items with NONE event type or impact < 0.3. Only include items with clear bullish catalysts."""

        try:
            payload = {
                "model": "grok-4-1-fast-reasoning",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            r = safe_post(
                "https://api.x.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=45
            )
            
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                analyses = json.loads(json_match.group())
                
                for analysis in analyses:
                    idx = analysis.get('index', 0)
                    if idx < len(batch):
                        original = batch[idx]
                        
                        # Filter tickers to our universe
                        valid_tickers = [
                            t.upper() for t in analysis.get('tickers', [])
                            if t.upper() in universe_tickers
                        ]
                        
                        if valid_tickers and analysis.get('impact_score', 0) >= 0.3:
                            event = NewsEvent(
                                headline=original.get('title', ''),
                                summary=original.get('description', '')[:200],
                                tickers=valid_tickers,
                                event_type=analysis.get('event_type', 'BREAKING_POSITIVE'),
                                impact_score=analysis.get('impact_score', 0.5),
                                sentiment=analysis.get('sentiment', 'BULLISH'),
                                source='grok_nlp',
                                published_at=original.get('published', datetime.utcnow().isoformat()),
                                url=original.get('url')
                            )
                            all_events.append(event)
            
            time.sleep(1)  # Rate limit
            
        except Exception as e:
            logger.error(f"Grok analysis batch failed: {e}")
            continue
    
    logger.info(f"Grok extracted {len(all_events)} valid events")
    
    return all_events


# ============================
# AGGREGATION
# ============================

def aggregate_events_by_ticker(events: List[NewsEvent]) -> Dict[str, TickerEvents]:
    """
    Aggregate all events by ticker
    
    Returns dict: ticker → TickerEvents
    """
    ticker_map = defaultdict(list)
    
    for event in events:
        for ticker in event.tickers:
            ticker_map[ticker].append(event)
    
    results = {}
    
    for ticker, ticker_events in ticker_map.items():
        # Calculate total impact (max, not sum, to avoid over-counting)
        impacts = [e.impact_score for e in ticker_events]
        total_impact = max(impacts) if impacts else 0
        
        # Add bonus for multiple events
        if len(ticker_events) > 1:
            total_impact = min(1.0, total_impact + 0.1 * (len(ticker_events) - 1))
        
        # Find dominant event type
        type_counts = defaultdict(int)
        for e in ticker_events:
            type_counts[e.event_type] += 1
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else 'UNKNOWN'
        
        results[ticker] = TickerEvents(
            ticker=ticker,
            events=ticker_events,
            total_impact=round(total_impact, 3),
            dominant_event_type=dominant_type,
            event_count=len(ticker_events)
        )
    
    return results


# ============================
# MAIN SCREENER FUNCTION
# ============================

def run_news_flow_screener(universe_tickers: List[str], hours_back: int = 6) -> Dict[str, TickerEvents]:
    """
    Main entry point for news flow screening
    
    Flow:
    1. Fetch global news (Polygon + Finnhub)
    2. Filter high-impact news
    3. NLP analysis with Grok
    4. Extract and validate tickers
    5. Aggregate by ticker
    
    Args:
        universe_tickers: List of valid small cap tickers
        hours_back: How many hours to look back
    
    Returns:
        Dict[ticker] -> TickerEvents with all relevant events
    """
    logger.info(f"=" * 60)
    logger.info(f"NEWS FLOW SCREENER - Last {hours_back}h")
    logger.info(f"Universe: {len(universe_tickers)} tickers")
    logger.info(f"=" * 60)
    
    universe_set = set(t.upper() for t in universe_tickers)
    
    # Step 1: Fetch news from all sources
    all_news = []
    
    polygon_news = fetch_polygon_news_global(hours_back)
    all_news.extend(polygon_news)
    
    finnhub_news = fetch_finnhub_general_news()
    all_news.extend(finnhub_news)
    
    logger.info(f"Total raw news: {len(all_news)}")
    
    if not all_news:
        logger.warning("No news fetched")
        return {}
    
    # Step 2: Filter high-impact news
    high_impact = filter_high_impact_news(all_news)
    
    if not high_impact:
        logger.info("No high-impact news found")
        return {}
    
    # Step 3: Deep NLP analysis with Grok
    events = analyze_news_with_grok(high_impact, universe_set)
    
    if not events:
        logger.info("No valid events extracted")
        return {}
    
    # Step 4: Aggregate by ticker
    ticker_events = aggregate_events_by_ticker(events)
    
    # Step 5: Sort by impact
    sorted_tickers = sorted(
        ticker_events.keys(),
        key=lambda t: ticker_events[t].total_impact,
        reverse=True
    )
    
    # Log results
    logger.info(f"\n📊 NEWS FLOW RESULTS:")
    logger.info(f"{'='*50}")
    
    for ticker in sorted_tickers[:20]:  # Top 20
        te = ticker_events[ticker]
        logger.info(
            f"  {ticker}: impact={te.total_impact:.2f}, "
            f"type={te.dominant_event_type}, "
            f"events={te.event_count}"
        )
    
    return ticker_events


def get_events_by_type(ticker_events: Dict[str, TickerEvents]) -> Dict[str, List[str]]:
    """
    Get tickers grouped by event type
    
    Returns: {event_type: [tickers]}
    """
    type_map = defaultdict(list)
    
    for ticker, te in ticker_events.items():
        type_map[te.dominant_event_type].append({
            'ticker': ticker,
            'impact': te.total_impact,
            'event_count': te.event_count
        })
    
    # Sort each type by impact
    for event_type in type_map:
        type_map[event_type].sort(key=lambda x: x['impact'], reverse=True)
    
    return dict(type_map)


def get_calendar_view(ticker_events: Dict[str, TickerEvents]) -> List[Dict]:
    """
    Get chronological view of all events
    
    Returns: List of events sorted by time
    """
    all_events = []
    
    for ticker, te in ticker_events.items():
        for event in te.events:
            all_events.append({
                'ticker': ticker,
                'headline': event.headline,
                'event_type': event.event_type,
                'impact': event.impact_score,
                'published_at': event.published_at,
                'sentiment': event.sentiment
            })
    
    # Sort by time
    all_events.sort(key=lambda x: x['published_at'], reverse=True)
    
    return all_events


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("NEWS FLOW SCREENER - TEST")
    print("=" * 60)
    
    # Test universe
    test_universe = [
        "AAPL", "TSLA", "NVDA", "AMD", "PLTR", "SOFI", "NIO", "LCID",
        "MARA", "RIOT", "COIN", "HOOD", "UPST", "AFRM", "SQ", "PYPL"
    ]
    
    print(f"\nTest universe: {len(test_universe)} tickers")
    
    results = run_news_flow_screener(test_universe, hours_back=6)
    
    print(f"\n📊 Results: {len(results)} tickers with events")
    
    # Show by event type
    by_type = get_events_by_type(results)
    print(f"\n📅 By Event Type:")
    for event_type, tickers in by_type.items():
        print(f"  {event_type}: {[t['ticker'] for t in tickers[:5]]}")
    
    # Show calendar view
    calendar = get_calendar_view(results)
    print(f"\n📰 Latest Events:")
    for event in calendar[:10]:
        print(f"  [{event['event_type']}] {event['ticker']}: {event['headline'][:50]}...")
