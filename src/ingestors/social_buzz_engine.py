"""
SOCIAL BUZZ ENGINE V6.1 — ACTIVE (sera remplacé par social_velocity.py — PLAN_AMELIORATION_V9)
================================================================================================

Ce module reste ACTIF et utilisé par scan_scheduler et batch_scheduler.
Il sera remplacé par src.social_velocity.SocialVelocityEngine (V9) une fois ce module créé.
Ne pas supprimer avant que social_velocity.py soit opérationnel.

The raw mention counting in this module is supplemented by
src.social_velocity.SocialVelocityEngine (V9) which measures
acceleration of mentions (1st/2nd derivatives) for earlier detection.

This module remains active for data ingestion (Reddit + StockTwits fetch).
SocialVelocityEngine consumes its data for velocity computation.

Sources:
- Reddit (PRAW API - FREE)
- StockTwits (FREE tier: 200 req/hour)

Métriques:
- mention_count: nombre de mentions récentes
- acceleration: delta vs baseline 24h
- source_diversity: nombre de sources différentes
- sentiment_ratio: bullish/bearish ratio
"""

import os
import math
import asyncio
import aiohttp
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("SOCIAL_BUZZ_ENGINE")


# ============================
# Configuration
# ============================

# Reddit config
REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "pennystocks",
    "smallstreetbets",
    "investing",
]

# StockTwits config
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
STOCKTWITS_RATE_LIMIT = 200  # per hour

# Database
BUZZ_DB = "data/social_buzz.db"

# Thresholds
ACCELERATION_THRESHOLD = 3.0  # 3x baseline = trigger


# ============================
# Data Classes
# ============================

@dataclass
class BuzzMetrics:
    """Social buzz metrics for a ticker"""
    ticker: str
    timestamp: datetime
    # Raw counts
    mention_count_1h: int = 0
    mention_count_24h: int = 0
    # By source
    reddit_mentions: int = 0
    stocktwits_mentions: int = 0
    # Sentiment
    bullish_count: int = 0
    bearish_count: int = 0
    sentiment_ratio: float = 1.0
    # Computed
    source_diversity: int = 0
    acceleration: float = 0.0
    buzz_score: float = 0.0
    # Flags
    is_accelerating: bool = False
    should_trigger_hot: bool = False


# ============================
# Baseline Tracker
# ============================

class BaselineTracker:
    """Tracks 7-day baseline for buzz comparison"""

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize baseline database"""
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(BUZZ_DB, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS buzz_baseline (
                ticker TEXT,
                date TEXT,
                mention_count INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS buzz_history (
                ticker TEXT,
                timestamp TEXT,
                mention_1h INTEGER,
                mention_24h INTEGER,
                reddit INTEGER,
                stocktwits INTEGER,
                bullish INTEGER,
                bearish INTEGER,
                buzz_score REAL
            )
        """)
        self.conn.commit()

    def get_baseline(self, ticker: str) -> float:
        """Get 7-day average daily mentions"""
        ticker = ticker.upper()
        cutoff = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT AVG(mention_count) FROM buzz_baseline
            WHERE ticker = ? AND date >= ?
        """, (ticker, cutoff))

        result = cursor.fetchone()[0]
        return result if result else 0.0

    def update_baseline(self, ticker: str, mention_count: int):
        """Update today's baseline"""
        ticker = ticker.upper()
        today = datetime.utcnow().strftime("%Y-%m-%d")

        self.conn.execute("""
            INSERT OR REPLACE INTO buzz_baseline (ticker, date, mention_count)
            VALUES (?, ?, ?)
        """, (ticker, today, mention_count))
        self.conn.commit()

    def save_metrics(self, metrics: BuzzMetrics):
        """Save buzz metrics to history"""
        self.conn.execute("""
            INSERT INTO buzz_history
            (ticker, timestamp, mention_1h, mention_24h, reddit, stocktwits, bullish, bearish, buzz_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.ticker,
            metrics.timestamp.isoformat(),
            metrics.mention_count_1h,
            metrics.mention_count_24h,
            metrics.reddit_mentions,
            metrics.stocktwits_mentions,
            metrics.bullish_count,
            metrics.bearish_count,
            metrics.buzz_score
        ))
        self.conn.commit()


# ============================
# Social Buzz Engine
# ============================

class SocialBuzzEngine:
    """
    Measures social buzz from Reddit + StockTwits

    Usage:
        engine = SocialBuzzEngine()
        metrics = await engine.get_buzz("AAPL")
        if metrics.should_trigger_hot:
            hot_queue.push("AAPL", reason="social_buzz")
    """

    def __init__(self):
        self.baseline = BaselineTracker()
        self._session = None

        # Reddit client (optional - requires praw)
        self.reddit = None
        self._init_reddit()

    def _init_reddit(self):
        """Initialize Reddit client if credentials available"""
        try:
            import praw

            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")

            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="GV2-EDGE/6.1 Social Buzz Engine"
                )
                logger.info("Reddit client initialized")
            else:
                logger.warning("Reddit credentials not configured")
        except ImportError:
            logger.warning("praw not installed - Reddit disabled")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_buzz(self, ticker: str) -> BuzzMetrics:
        """
        Get social buzz metrics for ticker

        Args:
            ticker: Stock ticker

        Returns:
            BuzzMetrics with all computed values
        """
        ticker = ticker.upper()
        logger.debug(f"Getting buzz for {ticker}")

        # Parallel fetch
        reddit_task = self._fetch_reddit(ticker)
        stocktwits_task = self._fetch_stocktwits(ticker)

        reddit_data, stocktwits_data = await asyncio.gather(
            reddit_task,
            stocktwits_task,
            return_exceptions=True
        )

        # Handle errors
        if isinstance(reddit_data, Exception):
            logger.warning(f"Reddit error: {reddit_data}")
            reddit_data = self._empty_source_data()

        if isinstance(stocktwits_data, Exception):
            logger.warning(f"StockTwits error: {stocktwits_data}")
            stocktwits_data = self._empty_source_data()

        # Aggregate metrics
        metrics = self._calculate_metrics(ticker, reddit_data, stocktwits_data)

        # Save to history
        self.baseline.save_metrics(metrics)
        self.baseline.update_baseline(ticker, metrics.mention_count_24h)

        return metrics

    def _empty_source_data(self) -> Dict:
        """Return empty source data structure"""
        return {
            "mentions_1h": 0,
            "mentions_24h": 0,
            "bullish": 0,
            "bearish": 0
        }

    async def _fetch_reddit(self, ticker: str) -> Dict:
        """Fetch Reddit mentions"""
        if not self.reddit:
            return self._empty_source_data()

        mentions_1h = 0
        mentions_24h = 0
        bullish = 0
        bearish = 0

        now = datetime.utcnow()
        cutoff_1h = now - timedelta(hours=1)
        cutoff_24h = now - timedelta(hours=24)

        # Search patterns
        patterns = [f"${ticker}", ticker.upper()]

        try:
            for subreddit_name in REDDIT_SUBREDDITS:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Check recent posts
                    for submission in subreddit.new(limit=50):
                        created = datetime.utcfromtimestamp(submission.created_utc)

                        if created < cutoff_24h:
                            continue

                        text = f"{submission.title} {submission.selftext}".upper()

                        if any(p in text for p in patterns):
                            mentions_24h += 1

                            if created >= cutoff_1h:
                                mentions_1h += 1

                            # Sentiment from upvote ratio
                            if submission.upvote_ratio > 0.7:
                                bullish += 1
                            elif submission.upvote_ratio < 0.4:
                                bearish += 1

                except Exception as e:
                    logger.debug(f"Subreddit {subreddit_name} error: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Reddit fetch error: {e}")

        return {
            "mentions_1h": mentions_1h,
            "mentions_24h": mentions_24h,
            "bullish": bullish,
            "bearish": bearish
        }

    async def _fetch_stocktwits(self, ticker: str) -> Dict:
        """Fetch StockTwits mentions"""
        url = STOCKTWITS_URL.format(ticker=ticker)

        try:
            session = await self._get_session()

            async with session.get(url, timeout=5) as resp:
                if resp.status != 200:
                    return self._empty_source_data()

                data = await resp.json()

        except Exception as e:
            logger.debug(f"StockTwits error for {ticker}: {e}")
            return self._empty_source_data()

        messages = data.get("messages", [])

        now = datetime.utcnow()
        mentions_1h = 0
        mentions_24h = 0
        bullish = 0
        bearish = 0

        for msg in messages:
            try:
                created_str = msg.get("created_at", "")
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                created = created.replace(tzinfo=None)

                age_hours = (now - created).total_seconds() / 3600

                if age_hours <= 24:
                    mentions_24h += 1

                    if age_hours <= 1:
                        mentions_1h += 1

                    # StockTwits has sentiment labels
                    sentiment = msg.get("entities", {}).get("sentiment", {}).get("basic")
                    if sentiment == "Bullish":
                        bullish += 1
                    elif sentiment == "Bearish":
                        bearish += 1

            except Exception:
                continue

        return {
            "mentions_1h": mentions_1h,
            "mentions_24h": mentions_24h,
            "bullish": bullish,
            "bearish": bearish
        }

    def _calculate_metrics(
        self,
        ticker: str,
        reddit_data: Dict,
        stocktwits_data: Dict
    ) -> BuzzMetrics:
        """Calculate composite buzz metrics"""
        # Aggregate
        mention_1h = reddit_data["mentions_1h"] + stocktwits_data["mentions_1h"]
        mention_24h = reddit_data["mentions_24h"] + stocktwits_data["mentions_24h"]

        bullish = reddit_data["bullish"] + stocktwits_data["bullish"]
        bearish = reddit_data["bearish"] + stocktwits_data["bearish"]

        # Source diversity (0-2)
        diversity = sum([
            1 if reddit_data["mentions_1h"] > 0 else 0,
            1 if stocktwits_data["mentions_1h"] > 0 else 0
        ])

        # Sentiment ratio
        if bearish > 0:
            sentiment_ratio = bullish / bearish
        else:
            sentiment_ratio = bullish if bullish > 0 else 1.0

        # Get baseline and calculate acceleration
        baseline = self.baseline.get_baseline(ticker)
        if baseline > 0:
            # Compare hourly rate vs baseline daily rate
            hourly_baseline = baseline / 24
            acceleration = mention_1h / hourly_baseline if hourly_baseline > 0 else 0
        else:
            # No baseline - high mentions = high acceleration
            acceleration = mention_1h * 5 if mention_1h > 0 else 0

        # Calculate buzz score (0-1)
        buzz_score = self._calculate_buzz_score(
            mention_1h,
            acceleration,
            diversity,
            sentiment_ratio
        )

        # Determine if should trigger hot queue
        is_accelerating = acceleration >= ACCELERATION_THRESHOLD
        should_trigger = is_accelerating and mention_1h >= 3

        return BuzzMetrics(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            mention_count_1h=mention_1h,
            mention_count_24h=mention_24h,
            reddit_mentions=reddit_data["mentions_24h"],
            stocktwits_mentions=stocktwits_data["mentions_24h"],
            bullish_count=bullish,
            bearish_count=bearish,
            sentiment_ratio=sentiment_ratio,
            source_diversity=diversity,
            acceleration=acceleration,
            buzz_score=buzz_score,
            is_accelerating=is_accelerating,
            should_trigger_hot=should_trigger
        )

    def _calculate_buzz_score(
        self,
        mention_count: int,
        acceleration: float,
        diversity: int,
        sentiment_ratio: float
    ) -> float:
        """
        Calculate composite buzz score (0-1)

        Formula:
        - Base: log(mentions + 1) normalized
        - Acceleration boost: if > 2x baseline
        - Diversity bonus: more sources = higher confidence
        - Sentiment adjustment: bullish ratio boost
        """
        # Base score from mentions (log scale)
        # ~50 mentions = 0.5, ~150 mentions = 0.8, ~500 mentions = 1.0
        if mention_count > 0:
            base = min(1.0, math.log(mention_count + 1) / 6.2)
        else:
            base = 0.0

        # Acceleration multiplier (1.0 - 1.5)
        if acceleration >= 5:
            accel_mult = 1.5
        elif acceleration >= 3:
            accel_mult = 1.3
        elif acceleration >= 2:
            accel_mult = 1.15
        else:
            accel_mult = 1.0

        # Diversity bonus (1.0 - 1.2)
        diversity_mult = 1.0 + diversity * 0.1

        # Sentiment adjustment (0.8 - 1.2)
        if sentiment_ratio >= 3:
            sentiment_mult = 1.2
        elif sentiment_ratio >= 2:
            sentiment_mult = 1.1
        elif sentiment_ratio <= 0.5:
            sentiment_mult = 0.8
        else:
            sentiment_mult = 1.0

        score = base * accel_mult * diversity_mult * sentiment_mult

        return min(1.0, max(0.0, score))

    async def get_buzz_batch(self, tickers: List[str]) -> Dict[str, BuzzMetrics]:
        """Get buzz for multiple tickers"""
        results = {}

        # Rate limit - process sequentially with small delay
        for ticker in tickers:
            try:
                results[ticker] = await self.get_buzz(ticker)
                await asyncio.sleep(0.5)  # Be nice to APIs
            except Exception as e:
                logger.warning(f"Buzz error for {ticker}: {e}")

        return results


# ============================
# Convenience Functions
# ============================

_engine_instance = None
_engine_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_buzz_engine() -> SocialBuzzEngine:
    """Get singleton engine instance"""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = SocialBuzzEngine()
    return _engine_instance


async def quick_buzz_check(ticker: str) -> BuzzMetrics:
    """Quick buzz check for single ticker"""
    engine = get_buzz_engine()
    return await engine.get_buzz(ticker)


# ============================
# Module exports
# ============================

__all__ = [
    "SocialBuzzEngine",
    "BuzzMetrics",
    "BaselineTracker",
    "get_buzz_engine",
    "quick_buzz_check",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        engine = SocialBuzzEngine()

        test_tickers = ["AAPL", "TSLA", "GME"]

        print("=" * 60)
        print("SOCIAL BUZZ ENGINE TEST")
        print("=" * 60)

        for ticker in test_tickers:
            print(f"\nChecking {ticker}...")
            metrics = await engine.get_buzz(ticker)

            print(f"  Mentions (1h): {metrics.mention_count_1h}")
            print(f"  Mentions (24h): {metrics.mention_count_24h}")
            print(f"  Reddit: {metrics.reddit_mentions}, StockTwits: {metrics.stocktwits_mentions}")
            print(f"  Sentiment ratio: {metrics.sentiment_ratio:.2f}")
            print(f"  Acceleration: {metrics.acceleration:.2f}x")
            print(f"  Buzz score: {metrics.buzz_score:.2f}")
            print(f"  Trigger hot: {metrics.should_trigger_hot}")

        await engine.close()

    asyncio.run(test())
