"""
SOCIAL BUZZ TRACKER
===================

Track social mention volume (not sentiment, just buzz level)

Sources:
1. Twitter/X via Grok API (mention count)
2. Reddit WallStreetBets scraping (post count)  
3. StockTwits API (message volume)
4. Google Trends (search interest)

Goal: Detect abnormal buzz BEFORE news breaks
(Insiders/smart money often discuss before public)

Scoring: Compare current buzz to 30-day average
Spike >3x = signal
"""

from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get, safe_post

from config import (
    GROK_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    STOCKTWITS_ACCESS_TOKEN
)

logger = get_logger("SOCIAL_BUZZ")

# ============================
# Reddit API Client (PRAW)
# ============================

_reddit_client = None

def get_reddit_client():
    """
    Get authenticated Reddit client using PRAW

    Returns:
        praw.Reddit instance or None if not configured
    """
    global _reddit_client

    if _reddit_client is not None:
        return _reddit_client

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning("Reddit API not configured - set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return None

    try:
        import praw

        _reddit_client = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        logger.info("Reddit client initialized successfully")
        return _reddit_client

    except ImportError:
        logger.warning("praw not installed - run: pip install praw")
        return None
    except Exception as e:
        logger.error(f"Reddit client init failed: {e}")
        return None

cache = Cache(ttl=900)  # 15min cache


# ============================
# 1. Twitter/X Buzz via Grok
# ============================

def get_twitter_buzz_grok(ticker, hours_back=24):
    """
    Use Grok to estimate Twitter mention volume
    
    Grok has access to X/Twitter data and can estimate
    how many times a ticker was mentioned recently.
    
    Args:
        ticker: Stock symbol
        hours_back: Time window to analyze
    
    Returns:
        Estimated mention count
    """
    cache_key = f"twitter_buzz_{ticker}_{hours_back}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    try:
        # Ask Grok to count mentions
        prompt = f"""Count how many times the stock ticker ${ticker} was mentioned on Twitter/X in the last {hours_back} hours. 

Only return a JSON with:
{{"mentions": <number>, "trending": <true/false>}}

Be conservative - only count clear stock ticker mentions, not company name in other contexts."""
        
        payload = {
            "model": "grok-4-1-fast-reasoning",  # FIXED: Updated from grok-beta
            "messages": [
                {"role": "user", "content": prompt}
            ],
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
            timeout=15
        )
        
        data = r.json()
        
        content = data["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        # Grok might wrap in markdown, so extract
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group())
            
            mentions = result.get("mentions", 0)
            trending = result.get("trending", False)
            
            buzz_data = {
                "mentions": mentions,
                "trending": trending,
                "source": "twitter_grok"
            }
            
            cache.set(cache_key, buzz_data)
            
            logger.info(f"{ticker} Twitter buzz: {mentions} mentions")
            
            return buzz_data
    
    except Exception as e:
        logger.warning(f"Grok Twitter buzz failed for {ticker}: {e}")
    
    return {"mentions": 0, "trending": False, "source": "twitter_grok"}


# ============================
# 2. Reddit Multi-Subreddit Search (PRAW)
# ============================

def get_reddit_wsb_buzz(ticker, subreddits=None):
    """
    Search Reddit for ticker mentions using PRAW API

    Uses authenticated API for higher rate limits and reliability.
    Searches multiple stock-related subreddits.

    Args:
        ticker: Stock symbol
        subreddits: List of subreddits to search (default: WSB, stocks, pennystocks)

    Returns:
        Mention count and sentiment data
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "pennystocks", "investing", "smallstreetbets"]

    cache_key = f"reddit_buzz_{ticker}"
    cached = cache.get(cache_key)

    if cached:
        return cached

    # Try PRAW first (authenticated)
    reddit = get_reddit_client()

    if reddit is not None:
        try:
            mention_count = 0
            bullish_count = 0
            bearish_count = 0

            ticker_upper = ticker.upper()
            ticker_with_dollar = f"${ticker_upper}"

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)

                    # Search for ticker mentions in last 24h
                    for post in subreddit.search(
                        f"{ticker_upper} OR {ticker_with_dollar}",
                        time_filter="day",
                        limit=50
                    ):
                        mention_count += 1

                        # Basic sentiment from flair or keywords
                        title_lower = post.title.lower()
                        if any(w in title_lower for w in ["buy", "calls", "moon", "rocket", "bullish", "long"]):
                            bullish_count += 1
                        elif any(w in title_lower for w in ["sell", "puts", "short", "bearish", "crash"]):
                            bearish_count += 1

                except Exception as e:
                    logger.debug(f"Reddit search failed for r/{sub_name}: {e}")
                    continue

            result = {
                "mentions": mention_count,
                "bullish": bullish_count,
                "bearish": bearish_count,
                "sentiment_ratio": bullish_count / max(bearish_count, 1),
                "source": "reddit_praw"
            }

            cache.set(cache_key, result)
            logger.info(f"{ticker} Reddit: {mention_count} mentions (bullish: {bullish_count}, bearish: {bearish_count})")

            return result

        except Exception as e:
            logger.warning(f"PRAW search failed for {ticker}: {e}")

    # Fallback to JSON API (unauthenticated)
    try:
        url = f"https://www.reddit.com/r/wallstreetbets/new.json?limit=100"

        headers = {
            'User-Agent': REDDIT_USER_AGENT or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            logger.warning(f"Reddit returned {r.status_code}")
            return {"mentions": 0, "source": "reddit_fallback"}

        data = r.json()
        posts = data.get("data", {}).get("children", [])

        mention_count = 0
        ticker_upper = ticker.upper()
        ticker_with_dollar = f"${ticker_upper}"

        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "").upper()
            selftext = post_data.get("selftext", "").upper()
            combined = title + " " + selftext

            if ticker_upper in combined or ticker_with_dollar in combined:
                mention_count += 1

        result = {
            "mentions": mention_count,
            "bullish": 0,
            "bearish": 0,
            "sentiment_ratio": 1.0,
            "source": "reddit_fallback"
        }

        cache.set(cache_key, result)
        logger.info(f"{ticker} Reddit (fallback): {mention_count} mentions")

        return result

    except Exception as e:
        logger.warning(f"Reddit fallback failed for {ticker}: {e}")
        return {"mentions": 0, "bullish": 0, "bearish": 0, "sentiment_ratio": 1.0, "source": "reddit_fallback"}


# ============================
# 3. StockTwits API (Authenticated)
# ============================

def get_stocktwits_buzz(ticker):
    """
    Get StockTwits message volume and sentiment

    Uses authenticated API for higher rate limits (200 req/hour free).
    Extracts bullish/bearish sentiment from message labels.

    Args:
        ticker: Stock symbol

    Returns:
        Message volume and sentiment breakdown
    """
    cache_key = f"stocktwits_{ticker}"
    cached = cache.get(cache_key)

    if cached:
        return cached

    try:
        # StockTwits streams API
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

        headers = {}
        if STOCKTWITS_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {STOCKTWITS_ACCESS_TOKEN}"

        r = safe_get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            logger.warning(f"StockTwits returned {r.status_code} for {ticker}")
            return {"messages": 0, "bullish": 0, "bearish": 0, "sentiment_ratio": 1.0, "source": "stocktwits"}

        data = r.json()

        messages = data.get("messages", [])
        message_count = len(messages)

        # Extract sentiment from message labels
        bullish_count = 0
        bearish_count = 0

        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment", {})
            basic = sentiment.get("basic")

            if basic == "Bullish":
                bullish_count += 1
            elif basic == "Bearish":
                bearish_count += 1

        # Get symbol info if available
        symbol_info = data.get("symbol", {})
        watchlist_count = symbol_info.get("watchlist_count", 0)

        result = {
            "messages": message_count,
            "bullish": bullish_count,
            "bearish": bearish_count,
            "sentiment_ratio": bullish_count / max(bearish_count, 1),
            "watchlist_count": watchlist_count,
            "source": "stocktwits"
        }

        cache.set(cache_key, result)

        logger.info(f"{ticker} StockTwits: {message_count} msgs (bullish: {bullish_count}, bearish: {bearish_count})")

        return result

    except Exception as e:
        logger.warning(f"StockTwits failed for {ticker}: {e}")
        return {"messages": 0, "bullish": 0, "bearish": 0, "sentiment_ratio": 1.0, "source": "stocktwits"}


# ============================
# 4. Google Trends
# ============================

def get_google_trends_score(ticker):
    """
    Get Google Trends interest score
    
    Uses pytrends library (free Google Trends API)
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Interest score (0-100)
    """
    cache_key = f"google_trends_{ticker}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    try:
        from pytrends.request import TrendReq
        
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build search term (ticker + stock)
        search_term = f"{ticker} stock"
        
        # Get interest over last 7 days
        pytrends.build_payload([search_term], timeframe='now 7-d')
        
        # Get interest over time
        interest_df = pytrends.interest_over_time()
        
        if interest_df.empty:
            return {"interest": 0, "source": "google_trends"}
        
        # Get average interest
        avg_interest = interest_df[search_term].mean()
        
        result = {
            "interest": int(avg_interest),
            "source": "google_trends"
        }
        
        cache.set(cache_key, result)
        
        logger.info(f"{ticker} Google Trends: {avg_interest:.0f}")
        
        return result
    
    except Exception as e:
        logger.warning(f"Google Trends failed for {ticker}: {e}")
        return {"interest": 0, "source": "google_trends"}


# ============================
# Consolidate All Sources
# ============================

def get_total_buzz_score(ticker):
    """
    Aggregate buzz from all sources
    
    Normalized score 0-1:
    - 0 = no buzz
    - 0.5 = normal buzz
    - 1.0 = viral/trending
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Aggregated buzz score
    """
    # Get data from all sources
    twitter = get_twitter_buzz_grok(ticker, hours_back=24)
    reddit = get_reddit_wsb_buzz(ticker)
    stocktwits = get_stocktwits_buzz(ticker)
    google = get_google_trends_score(ticker)
    
    # Normalize each source (rough thresholds)
    twitter_score = min(1.0, twitter.get("mentions", 0) / 100)  # 100 mentions = max
    reddit_score = min(1.0, reddit.get("mentions", 0) / 10)     # 10 mentions = max
    stocktwits_score = min(1.0, stocktwits.get("messages", 0) / 30)  # 30 messages = max
    google_score = google.get("interest", 0) / 100  # Already 0-100
    
    # Weighted average
    total_score = (
        twitter_score * 0.35 +      # Twitter most important
        reddit_score * 0.25 +       # WSB influential
        stocktwits_score * 0.20 +   # StockTwits niche but useful
        google_score * 0.20         # Google Trends = mainstream interest
    )
    
    # Trending boost
    if twitter.get("trending", False):
        total_score *= 1.3  # 30% boost if trending
    
    # Clamp 0-1
    total_score = max(0, min(1, total_score))
    
    logger.info(f"{ticker} total buzz score: {total_score:.2f}")
    
    return {
        "ticker": ticker,
        "buzz_score": total_score,
        "twitter_mentions": twitter.get("mentions", 0),
        "reddit_mentions": reddit.get("mentions", 0),
        "stocktwits_messages": stocktwits.get("messages", 0),
        "google_interest": google.get("interest", 0),
        "trending": twitter.get("trending", False)
    }


# ============================
# Detect Buzz Spikes
# ============================

def detect_buzz_spike(ticker, threshold=3.0):
    """
    Detect if buzz is abnormally high
    
    Compare current buzz to historical average.
    
    Args:
        ticker: Stock symbol
        threshold: Spike threshold (default 3x = 300%)
    
    Returns:
        bool: True if spike detected
    """
    current_buzz = get_total_buzz_score(ticker)
    current_score = current_buzz["buzz_score"]
    
    # For now, use simple threshold
    # Full implementation would track historical averages
    
    baseline = 0.3  # Assume baseline buzz = 0.3
    
    if current_score >= baseline * threshold:
        logger.warning(f"🚨 BUZZ SPIKE DETECTED: {ticker} (score: {current_score:.2f})")
        return True
    
    return False


# ============================
# Public API
# ============================

def get_buzz_signal(ticker):
    """Get buzz signal for ticker"""
    buzz = get_total_buzz_score(ticker)
    spike = detect_buzz_spike(ticker)
    
    return {
        **buzz,
        "spike_detected": spike,
        "signal": "BUZZ_ALERT" if spike else "NORMAL"
    }


if __name__ == "__main__":
    print("\n📱 SOCIAL BUZZ TRACKER TEST")
    print("=" * 60)
    
    test_tickers = ["AAPL", "TSLA", "GME"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        
        buzz = get_buzz_signal(ticker)
        
        print(f"  Buzz Score: {buzz['buzz_score']:.2f}")
        print(f"  Twitter: {buzz['twitter_mentions']}")
        print(f"  Reddit: {buzz['reddit_mentions']}")
        print(f"  StockTwits: {buzz['stocktwits_messages']}")
        print(f"  Google: {buzz['google_interest']}")
        print(f"  Trending: {buzz['trending']}")
        print(f"  Spike: {buzz['spike_detected']}")
