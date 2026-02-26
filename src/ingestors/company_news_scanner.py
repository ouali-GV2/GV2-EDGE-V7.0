"""
COMPANY NEWS SCANNER V6.1
=========================

Scan ciblé des news par ticker (company-specific).

Source: Finnhub Company News API (FREE tier: 60 req/min)

Rôle:
- Scan profond sur tickers "chauds" (hot_ticker_queue)
- Extraction détaillée des catalyseurs
- Classification EVENT_TYPE via NLP
- Alimentation Event Hub

Fréquence (dynamique):
- HOT tickers: 1-2 min
- WARM tickers: 5 min
- NORMAL rotation: 10-15 min

Architecture:
1. Receive ticker from hot_queue or scheduler
2. Fetch Finnhub company news
3. Filter by recency (4-24h)
4. NLP classification (Grok)
5. Create catalyst events
6. Push to Event Hub
"""

import os
import asyncio
import aiohttp
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger
from config import FINNHUB_API_KEY

# Import Phase 1 modules
from src.processors.keyword_filter import get_keyword_filter, FilterPriority
from src.processors.nlp_classifier import get_classifier, ClassificationResult

logger = get_logger("COMPANY_NEWS_SCANNER")


# ============================
# Configuration
# ============================

FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"

# Rate limiting for Finnhub free tier
MAX_REQUESTS_PER_MINUTE = 60
REQUEST_DELAY_MS = 1000  # 1 request per second to be safe

# News recency thresholds (hours)
RECENCY_HOT = 4  # For hot tickers
RECENCY_WARM = 12  # For warm tickers
RECENCY_NORMAL = 24  # For normal rotation


# ============================
# Enums
# ============================

class ScanPriority(Enum):
    """Scan priority levels"""
    HOT = 1  # Scan every 1-2 min
    WARM = 2  # Scan every 5 min
    NORMAL = 3  # Scan every 10-15 min


# ============================
# Data Classes
# ============================

@dataclass
class CompanyNewsItem:
    """Represents a company-specific news item"""
    id: str
    ticker: str
    headline: str
    summary: str
    url: str
    source: str  # News source (Reuters, Bloomberg, etc.)
    published_at: datetime
    # Classification (filled after NLP)
    event_type: Optional[str] = None
    event_impact: float = 0.0
    event_tier: int = 0
    nlp_confidence: float = 0.0
    # Filter info
    filter_priority: Optional[FilterPriority] = None
    filter_category: Optional[str] = None


@dataclass
class CompanyScanResult:
    """Result of scanning a single company"""
    ticker: str
    timestamp: datetime
    priority: ScanPriority
    news_count: int
    catalyst_count: int
    news_items: List[CompanyNewsItem]
    top_catalyst: Optional[CompanyNewsItem] = None
    error: Optional[str] = None


# ============================
# Company News Scanner
# ============================

class CompanyNewsScanner:
    """
    Scans company-specific news from Finnhub

    Usage:
        scanner = CompanyNewsScanner()
        result = await scanner.scan_company("AAPL", priority=ScanPriority.HOT)
        if result.top_catalyst:
            print(f"Catalyst: {result.top_catalyst.event_type}")
    """

    def __init__(self):
        self.keyword_filter = get_keyword_filter()
        self.nlp_classifier = get_classifier()

        # Track last scan times per ticker
        self.last_scan: Dict[str, datetime] = {}

        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(5)
        self.last_request_time = datetime.min

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

    async def _rate_limit(self):
        """Apply rate limiting"""
        now = datetime.utcnow()
        elapsed = (now - self.last_request_time).total_seconds() * 1000

        if elapsed < REQUEST_DELAY_MS:
            await asyncio.sleep((REQUEST_DELAY_MS - elapsed) / 1000)

        self.last_request_time = datetime.utcnow()

    def _get_recency_hours(self, priority: ScanPriority) -> int:
        """Get recency threshold based on priority"""
        if priority == ScanPriority.HOT:
            return RECENCY_HOT
        elif priority == ScanPriority.WARM:
            return RECENCY_WARM
        else:
            return RECENCY_NORMAL

    async def scan_company(
        self,
        ticker: str,
        priority: ScanPriority = ScanPriority.NORMAL,
        classify: bool = True
    ) -> CompanyScanResult:
        """
        Scan news for a specific company

        Args:
            ticker: Stock ticker
            priority: Scan priority (affects recency filter)
            classify: Whether to run NLP classification

        Returns:
            CompanyScanResult with news and catalysts
        """
        logger.debug(f"Scanning {ticker} (priority={priority.name})")

        # Fetch news from Finnhub
        news_items = await self._fetch_company_news(ticker, priority)

        if not news_items:
            return CompanyScanResult(
                ticker=ticker,
                timestamp=datetime.utcnow(),
                priority=priority,
                news_count=0,
                catalyst_count=0,
                news_items=[]
            )

        # Apply keyword filter
        filtered_items = []
        for item in news_items:
            text = f"{item.headline} {item.summary}"
            filter_result = self.keyword_filter.filter(text)

            # Keep all items but mark priority
            item.filter_priority = filter_result.priority
            item.filter_category = filter_result.category

            # Only process non-noise
            if not filter_result.is_noise:
                filtered_items.append(item)

        # NLP classification for catalyst detection
        catalysts = []
        if classify and filtered_items:
            for item in filtered_items:
                try:
                    result = await self.nlp_classifier.classify(
                        item.headline,
                        item.summary
                    )

                    item.event_type = result.event_type
                    item.event_impact = result.impact
                    item.event_tier = result.tier
                    item.nlp_confidence = result.confidence

                    # Track catalysts (tier 1-4)
                    if result.tier > 0 and result.tier <= 4:
                        catalysts.append(item)

                except Exception as e:
                    logger.warning(f"NLP classification failed for {ticker}: {e}")

        # Sort catalysts by impact
        catalysts.sort(key=lambda x: x.event_impact, reverse=True)
        top_catalyst = catalysts[0] if catalysts else None

        # Update last scan time
        self.last_scan[ticker] = datetime.utcnow()

        return CompanyScanResult(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            priority=priority,
            news_count=len(news_items),
            catalyst_count=len(catalysts),
            news_items=filtered_items,
            top_catalyst=top_catalyst
        )

    async def _fetch_company_news(
        self,
        ticker: str,
        priority: ScanPriority
    ) -> List[CompanyNewsItem]:
        """Fetch company news from Finnhub"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return []

        try:
            async with self.request_semaphore:
                await self._rate_limit()

                session = await self._get_session()

                # Date range based on priority
                recency_hours = self._get_recency_hours(priority)
                to_date = datetime.utcnow()
                from_date = to_date - timedelta(hours=recency_hours)

                params = {
                    "symbol": ticker,
                    "from": from_date.strftime("%Y-%m-%d"),
                    "to": to_date.strftime("%Y-%m-%d"),
                    "token": FINNHUB_API_KEY
                }

                async with session.get(
                    FINNHUB_COMPANY_NEWS_URL,
                    params=params,
                    timeout=10
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Finnhub error for {ticker}: {resp.status}")
                        return []

                    data = await resp.json()

            items = []
            cutoff = datetime.utcnow() - timedelta(hours=recency_hours)

            for article in data:
                # Parse datetime
                try:
                    published = datetime.fromtimestamp(article.get("datetime", 0))
                except:
                    continue

                # Filter by recency
                if published < cutoff:
                    continue

                items.append(CompanyNewsItem(
                    id=f"finnhub_co_{article.get('id', 0)}",
                    ticker=ticker,
                    headline=article.get("headline", ""),
                    summary=article.get("summary", ""),
                    url=article.get("url", ""),
                    source=article.get("source", ""),
                    published_at=published
                ))

            logger.debug(f"Fetched {len(items)} news for {ticker}")
            return items

        except Exception as e:
            logger.error(f"Fetch error for {ticker}: {e}")
            return []

    async def scan_batch(
        self,
        tickers: List[str],
        priority: ScanPriority = ScanPriority.NORMAL,
        max_concurrent: int = 5
    ) -> List[CompanyScanResult]:
        """
        Scan multiple tickers in parallel

        Args:
            tickers: List of tickers to scan
            priority: Scan priority for all
            max_concurrent: Max concurrent requests

        Returns:
            List of CompanyScanResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scan_with_limit(ticker):
            async with semaphore:
                return await self.scan_company(ticker, priority)

        tasks = [scan_with_limit(t) for t in tickers]
        return await asyncio.gather(*tasks)

    def should_scan(self, ticker: str, priority: ScanPriority) -> bool:
        """
        Check if ticker should be scanned based on last scan time

        Args:
            ticker: Stock ticker
            priority: Scan priority

        Returns:
            True if enough time has passed since last scan
        """
        last = self.last_scan.get(ticker)
        if not last:
            return True

        # Intervals in seconds
        intervals = {
            ScanPriority.HOT: 90,  # 1.5 min
            ScanPriority.WARM: 300,  # 5 min
            ScanPriority.NORMAL: 600  # 10 min
        }

        elapsed = (datetime.utcnow() - last).total_seconds()
        return elapsed >= intervals[priority]


# ============================
# Convenience Functions
# ============================

_scanner_instance = None
_scanner_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_company_scanner() -> CompanyNewsScanner:
    """Get singleton scanner instance"""
    global _scanner_instance
    with _scanner_lock:
        if _scanner_instance is None:
            _scanner_instance = CompanyNewsScanner()
    return _scanner_instance


async def quick_company_scan(ticker: str) -> CompanyScanResult:
    """Quick one-shot company scan"""
    scanner = get_company_scanner()
    return await scanner.scan_company(ticker, ScanPriority.HOT)


# ============================
# Module exports
# ============================

__all__ = [
    "CompanyNewsScanner",
    "CompanyNewsItem",
    "CompanyScanResult",
    "ScanPriority",
    "get_company_scanner",
    "quick_company_scan",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        scanner = CompanyNewsScanner()

        test_tickers = ["AAPL", "NVDA", "TSLA"]

        print("=" * 60)
        print("COMPANY NEWS SCANNER TEST")
        print("=" * 60)

        for ticker in test_tickers:
            print(f"\nScanning {ticker}...")
            result = await scanner.scan_company(ticker, ScanPriority.HOT)

            print(f"  News count: {result.news_count}")
            print(f"  Catalysts: {result.catalyst_count}")

            if result.top_catalyst:
                c = result.top_catalyst
                print(f"  Top catalyst: {c.event_type} (tier {c.event_tier})")
                print(f"    {c.headline[:60]}...")

        await scanner.close()

    asyncio.run(test())
