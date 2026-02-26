"""
GLOBAL NEWS INGESTOR V6.1
=========================

Ingestion des news globales pour découverte de catalyseurs.

Sources:
- Finnhub General News (FREE tier: 60 req/min)
- SEC EDGAR 8-K (FREE, unlimited)

Rôle:
- Découvrir catalyseurs inattendus
- Alimenter hot_ticker_queue
- Fournir early signals avant scans company-specific

Fréquence:
- REALTIME mode: toutes les 3-5 minutes
- BATCH mode: toutes les 15 minutes

Architecture:
1. Fetch parallel (Finnhub + SEC)
2. Keyword filter (fast, NO LLM)
3. Ticker extraction
4. Push to hot_ticker_queue
5. Optionnel: NLP classification pour EVENT_TYPE
"""

import os
import asyncio
import aiohttp
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import json

from utils.logger import get_logger
from config import FINNHUB_API_KEY

# Import Phase 1 modules
from src.processors.keyword_filter import get_keyword_filter, FilterPriority
from src.processors.ticker_extractor import get_ticker_extractor
from src.ingestors.sec_filings_ingestor import SECIngestor

logger = get_logger("GLOBAL_NEWS_INGESTOR")


# ============================
# Configuration
# ============================

FINNHUB_GENERAL_URL = "https://finnhub.io/api/v1/news"

# Rate limiting
FINNHUB_RATE_LIMIT = 60  # requests per minute
SEC_RATE_LIMIT = 10  # requests per second (SEC is generous)

# Scan intervals (seconds)
REALTIME_INTERVAL = 180  # 3 minutes
BATCH_INTERVAL = 900  # 15 minutes


# ============================
# Data Classes
# ============================

@dataclass
class GlobalNewsItem:
    """Represents a global news item"""
    id: str
    source: str  # "finnhub_general", "sec_8k"
    headline: str
    summary: str
    url: str
    published_at: datetime
    tickers: List[str] = field(default_factory=list)
    filter_priority: Optional[FilterPriority] = None
    filter_category: Optional[str] = None
    event_type: Optional[str] = None
    source_priority: int = 3  # 1=critical, 2=high, 3=moderate


@dataclass
class GlobalScanResult:
    """Result of a global scan"""
    timestamp: datetime
    items_fetched: int
    items_filtered: int
    hot_tickers: List[str]
    news_items: List[GlobalNewsItem]
    errors: List[str] = field(default_factory=list)


# ============================
# Global News Ingestor
# ============================

class GlobalNewsIngestor:
    """
    Ingests global news from multiple sources

    Usage:
        ingestor = GlobalNewsIngestor(universe={"AAPL", "TSLA"})
        result = await ingestor.scan()
        for ticker in result.hot_tickers:
            hot_queue.push(ticker)
    """

    def __init__(self, universe: Set[str] = None):
        self.universe = universe or set()
        self.keyword_filter = get_keyword_filter()
        self.ticker_extractor = get_ticker_extractor(universe)
        self.sec_ingestor = SECIngestor(universe)

        # Track last processed IDs to avoid duplicates
        self.last_finnhub_id = 0
        self.last_sec_id = None

        # HTTP session
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def set_universe(self, universe: Set[str]):
        """Update universe"""
        self.universe = universe
        self.ticker_extractor.set_universe(universe)
        self.sec_ingestor.set_universe(universe)

    async def scan(
        self,
        hours_back: int = 2,
        include_sec: bool = True,
        include_finnhub: bool = True
    ) -> GlobalScanResult:
        """
        Run global news scan

        Args:
            hours_back: How far back to look
            include_sec: Include SEC 8-K filings
            include_finnhub: Include Finnhub general news

        Returns:
            GlobalScanResult with filtered items and hot tickers
        """
        logger.info(f"Starting global scan (hours_back={hours_back})")

        all_items = []
        errors = []

        # Parallel fetch from all sources
        tasks = []

        if include_finnhub:
            tasks.append(self._fetch_finnhub_general())

        if include_sec:
            tasks.append(self._fetch_sec_8k(hours_back))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(str(result))
                logger.error(f"Source fetch error: {result}")
            elif result:
                all_items.extend(result)

        logger.info(f"Fetched {len(all_items)} total items")

        # Apply keyword filter
        filtered_items = []
        for item in all_items:
            text = f"{item.headline} {item.summary}"
            filter_result = self.keyword_filter.filter(text)

            if filter_result.should_process:
                item.filter_priority = filter_result.priority
                item.filter_category = filter_result.category
                filtered_items.append(item)

        logger.info(f"Filtered to {len(filtered_items)} relevant items")

        # Extract tickers and build hot list
        hot_tickers = set()
        for item in filtered_items:
            tickers = self.ticker_extractor.extract_validated(
                f"{item.headline} {item.summary}"
            )
            item.tickers = tickers

            # Add to hot list based on priority
            for ticker in tickers:
                if item.filter_priority in [FilterPriority.CRITICAL, FilterPriority.HIGH]:
                    hot_tickers.add(ticker)

        logger.info(f"Found {len(hot_tickers)} hot tickers")

        return GlobalScanResult(
            timestamp=datetime.utcnow(),
            items_fetched=len(all_items),
            items_filtered=len(filtered_items),
            hot_tickers=list(hot_tickers),
            news_items=filtered_items,
            errors=errors
        )

    async def _fetch_finnhub_general(self) -> List[GlobalNewsItem]:
        """Fetch general news from Finnhub"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return []

        try:
            session = await self._get_session()

            params = {
                "category": "general",
                "token": FINNHUB_API_KEY,
                "minId": self.last_finnhub_id
            }

            async with session.get(FINNHUB_GENERAL_URL, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"Finnhub error: {resp.status}")
                    return []

                data = await resp.json()

            items = []
            for article in data:
                news_id = article.get("id", 0)

                # Update last ID
                if news_id > self.last_finnhub_id:
                    self.last_finnhub_id = news_id

                # Parse datetime
                try:
                    published = datetime.fromtimestamp(article.get("datetime", 0))
                except:
                    published = datetime.utcnow()

                items.append(GlobalNewsItem(
                    id=f"finnhub_{news_id}",
                    source="finnhub_general",
                    headline=article.get("headline", ""),
                    summary=article.get("summary", ""),
                    url=article.get("url", ""),
                    published_at=published,
                    source_priority=3  # Moderate priority
                ))

            logger.debug(f"Fetched {len(items)} Finnhub general news")
            return items

        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")
            return []

    async def _fetch_sec_8k(self, hours_back: int = 2) -> List[GlobalNewsItem]:
        """Fetch SEC 8-K filings"""
        try:
            filings = await self.sec_ingestor.fetch_all_recent(hours_back=hours_back)

            items = []
            for filing in filings:
                items.append(GlobalNewsItem(
                    id=f"sec_{filing.accession_number}",
                    source="sec_8k",
                    headline=f"SEC 8-K: {filing.company_name} - {filing.form_type}",
                    summary=filing.summary,
                    url=filing.url,
                    published_at=filing.filed_date,
                    tickers=[filing.ticker] if filing.ticker else [],
                    source_priority=1  # Critical priority for SEC
                ))

            logger.debug(f"Fetched {len(items)} SEC 8-K filings")
            return items

        except Exception as e:
            logger.error(f"SEC fetch error: {e}")
            return []

    async def scan_continuous(
        self,
        interval_seconds: int = REALTIME_INTERVAL,
        callback=None
    ):
        """
        Run continuous scanning

        Args:
            interval_seconds: Time between scans
            callback: Function to call with results
        """
        logger.info(f"Starting continuous scan (interval={interval_seconds}s)")

        while True:
            try:
                result = await self.scan()

                if callback:
                    await callback(result)

                # Log summary
                if result.hot_tickers:
                    logger.info(f"Hot tickers: {', '.join(result.hot_tickers[:10])}")

            except Exception as e:
                logger.error(f"Scan error: {e}")

            await asyncio.sleep(interval_seconds)


# ============================
# Convenience Functions
# ============================

_ingestor_instance = None
_ingestor_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton

def get_global_ingestor(universe: Set[str] = None) -> GlobalNewsIngestor:
    """Get singleton ingestor instance"""
    global _ingestor_instance
    with _ingestor_lock:
        if _ingestor_instance is None:
            _ingestor_instance = GlobalNewsIngestor(universe)
        elif universe:
            _ingestor_instance.set_universe(universe)
    return _ingestor_instance


async def quick_global_scan(universe: Set[str] = None) -> GlobalScanResult:
    """Quick one-shot global scan"""
    ingestor = get_global_ingestor(universe)
    return await ingestor.scan()


# ============================
# Module exports
# ============================

__all__ = [
    "GlobalNewsIngestor",
    "GlobalNewsItem",
    "GlobalScanResult",
    "get_global_ingestor",
    "quick_global_scan",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        # Test universe
        universe = {"AAPL", "TSLA", "NVDA", "BIOX", "MRNA", "PFE"}

        ingestor = GlobalNewsIngestor(universe)

        print("=" * 60)
        print("GLOBAL NEWS INGESTOR TEST")
        print("=" * 60)

        result = await ingestor.scan(hours_back=4)

        print(f"\nFetched: {result.items_fetched}")
        print(f"Filtered: {result.items_filtered}")
        print(f"Hot tickers: {result.hot_tickers}")

        if result.news_items:
            print("\nTop 5 items:")
            for item in result.news_items[:5]:
                print(f"  [{item.source}] {item.headline[:60]}...")
                print(f"    Tickers: {item.tickers}, Priority: {item.filter_priority}")

        await ingestor.close()

    asyncio.run(test())
