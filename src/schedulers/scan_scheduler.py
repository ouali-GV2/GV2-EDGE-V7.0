"""
SCAN SCHEDULER V6.1
===================

Orchestration dynamique des scans selon mode marche.

Modes:
- REALTIME: Marche ouvert (4h-20h ET), scans frequents
- BATCH: Hors marche, traitement complet

Frequences:
- Global scan: 3-5 min
- Hot tickers: 1-2 min
- Warm tickers: 5 min
- Universe rotation: 10-15 min

Architecture:
- Integration avec hot_ticker_queue
- Callbacks pour Event Hub
- Market session awareness (NYSE calendar)
- Graceful shutdown
"""

import asyncio
import threading
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import pytz

from utils.logger import get_logger

# Import ingestors
from src.ingestors.global_news_ingestor import GlobalNewsIngestor, GlobalScanResult
from src.ingestors.company_news_scanner import CompanyNewsScanner, CompanyScanResult, ScanPriority
from src.ingestors.social_buzz_engine import SocialBuzzEngine

# Import schedulers
from src.schedulers.hot_ticker_queue import (
    HotTickerQueue, TickerPriority, TriggerReason, get_hot_queue
)

logger = get_logger("SCAN_SCHEDULER")


# ============================
# Configuration
# ============================

# Timezone
NYC_TZ = pytz.timezone("America/New_York")

# Scan intervals (seconds)
INTERVALS = {
    "global": 180,  # 3 minutes
    "hot": 90,  # 1.5 minutes
    "warm": 300,  # 5 minutes
    "normal": 600,  # 10 minutes
    "buzz_check": 600,  # 10 minutes
}

# Market hours (ET)
PREMARKET_START = time(4, 0)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
AFTERHOURS_END = time(20, 0)


# ============================
# Enums
# ============================

class ScanMode(Enum):
    """Scan mode based on market hours"""
    REALTIME = "realtime"  # Market open (4AM-8PM ET)
    BATCH = "batch"  # Market closed


class MarketSession(Enum):
    """Current market session"""
    PREMARKET = "premarket"
    RTH = "rth"  # Regular Trading Hours
    AFTERHOURS = "afterhours"
    CLOSED = "closed"


# ============================
# Data Classes
# ============================

@dataclass
class SchedulerStats:
    """Scheduler statistics"""
    start_time: datetime
    mode: ScanMode
    session: MarketSession
    global_scans: int = 0
    company_scans: int = 0
    hot_tickers_processed: int = 0
    catalysts_detected: int = 0
    errors: int = 0


# ============================
# Scan Scheduler
# ============================

class ScanScheduler:
    """
    Dynamic scan orchestration

    Usage:
        scheduler = ScanScheduler(universe={"AAPL", "TSLA"})
        scheduler.on_catalyst(lambda c: event_hub.store(c))
        await scheduler.start()
    """

    def __init__(self, universe: Set[str] = None):
        self.universe = universe or set()

        # Ingestors
        self.global_ingestor = GlobalNewsIngestor(universe)
        self.company_scanner = CompanyNewsScanner()
        self.buzz_engine = SocialBuzzEngine()

        # Queue
        self.hot_queue = get_hot_queue()

        # State
        self.running = False
        self.mode = ScanMode.BATCH
        self.session = MarketSession.CLOSED
        self.stats = None

        # Callbacks
        self._on_catalyst_callbacks: List[Callable] = []
        self._on_hot_ticker_callbacks: List[Callable] = []
        self._on_scan_complete_callbacks: List[Callable] = []

        # Last scan times
        self._last_global_scan = datetime.min
        self._last_buzz_check = datetime.min
        self._universe_index = 0

    def set_universe(self, universe: Set[str]):
        """Update universe"""
        self.universe = universe
        self.global_ingestor.set_universe(universe)

    # ============================
    # Callbacks
    # ============================

    def on_catalyst(self, callback: Callable):
        """Register callback for catalyst detection"""
        self._on_catalyst_callbacks.append(callback)

    def on_hot_ticker(self, callback: Callable):
        """Register callback for hot ticker detection"""
        self._on_hot_ticker_callbacks.append(callback)

    def on_scan_complete(self, callback: Callable):
        """Register callback for scan completion"""
        self._on_scan_complete_callbacks.append(callback)

    async def _emit_catalyst(self, catalyst: Any):
        """Emit catalyst to callbacks"""
        for callback in self._on_catalyst_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(catalyst)
                else:
                    callback(catalyst)
            except Exception as e:
                logger.error(f"Catalyst callback error: {e}")

    async def _emit_hot_ticker(self, ticker: str, reason: str):
        """Emit hot ticker to callbacks"""
        for callback in self._on_hot_ticker_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ticker, reason)
                else:
                    callback(ticker, reason)
            except Exception as e:
                logger.error(f"Hot ticker callback error: {e}")

    # ============================
    # Market Session
    # ============================

    def _get_session(self) -> MarketSession:
        """Determine current market session"""
        now = datetime.now(NYC_TZ)
        current_time = now.time()
        weekday = now.weekday()

        # Weekend
        if weekday >= 5:
            return MarketSession.CLOSED

        # Check session
        if current_time < PREMARKET_START:
            return MarketSession.CLOSED
        elif current_time < MARKET_OPEN:
            return MarketSession.PREMARKET
        elif current_time < MARKET_CLOSE:
            return MarketSession.RTH
        elif current_time < AFTERHOURS_END:
            return MarketSession.AFTERHOURS
        else:
            return MarketSession.CLOSED

    def _get_mode(self) -> ScanMode:
        """Determine scan mode"""
        session = self._get_session()
        if session == MarketSession.CLOSED:
            return ScanMode.BATCH
        return ScanMode.REALTIME

    # ============================
    # Main Loop
    # ============================

    async def start(self):
        """Start the scheduler"""
        self.running = True
        self.stats = SchedulerStats(
            start_time=datetime.utcnow(),
            mode=self._get_mode(),
            session=self._get_session()
        )

        logger.info("=" * 60)
        logger.info("SCAN SCHEDULER STARTING")
        logger.info(f"Mode: {self.stats.mode.value}")
        logger.info(f"Session: {self.stats.session.value}")
        logger.info(f"Universe: {len(self.universe)} tickers")
        logger.info("=" * 60)

        try:
            while self.running:
                # Update mode/session
                self.mode = self._get_mode()
                self.session = self._get_session()
                self.stats.mode = self.mode
                self.stats.session = self.session

                if self.mode == ScanMode.REALTIME:
                    await self._realtime_cycle()
                else:
                    await self._batch_cycle()

                # Short sleep between cycles
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            self.stats.errors += 1
        finally:
            await self._cleanup()

    async def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False

    async def _cleanup(self):
        """Cleanup resources"""
        await self.global_ingestor.close()
        await self.company_scanner.close()
        await self.buzz_engine.close()
        logger.info("Scheduler stopped")

    # ============================
    # Realtime Cycle
    # ============================

    async def _realtime_cycle(self):
        """
        Realtime mode cycle:
        1. Global scan (every 3-5 min)
        2. Hot ticker scans (every 1-2 min)
        3. Warm ticker scans (every 5 min)
        4. Universe rotation (every 10-15 min)
        5. Buzz check (every 10 min)
        """
        now = datetime.utcnow()

        # 1. Global scan
        if (now - self._last_global_scan).total_seconds() >= INTERVALS["global"]:
            asyncio.create_task(self._run_global_scan())
            self._last_global_scan = now

        # 2. Hot ticker scans
        hot_tickers = self.hot_queue.get_all_for_scan()
        for ticker in hot_tickers[:10]:  # Limit concurrent
            priority = self.hot_queue.get_priority(ticker)
            if priority == TickerPriority.HOT:
                asyncio.create_task(self._run_company_scan(ticker, ScanPriority.HOT))
                self.hot_queue.mark_scanned(ticker)

        # 3. Warm tickers
        for ticker in hot_tickers[10:20]:
            priority = self.hot_queue.get_priority(ticker)
            if priority == TickerPriority.WARM:
                asyncio.create_task(self._run_company_scan(ticker, ScanPriority.WARM))
                self.hot_queue.mark_scanned(ticker)

        # 4. Universe rotation (background)
        await self._rotate_universe()

        # 5. Buzz check for hot tickers
        if (now - self._last_buzz_check).total_seconds() >= INTERVALS["buzz_check"]:
            asyncio.create_task(self._run_buzz_check())
            self._last_buzz_check = now

    async def _batch_cycle(self):
        """
        Batch mode cycle (off-market):
        - Full universe scan
        - Complete SEC ingestion
        - Buzz baseline updates
        - Generate watchlist
        """
        logger.info("Running batch cycle...")

        # Full global scan
        await self._run_global_scan()

        # Scan universe in chunks
        universe_list = list(self.universe)
        chunk_size = 20

        for i in range(0, len(universe_list), chunk_size):
            chunk = universe_list[i:i + chunk_size]

            for ticker in chunk:
                await self._run_company_scan(ticker, ScanPriority.NORMAL)
                await asyncio.sleep(1)  # Rate limit

            # Check if should switch to realtime
            if self._get_mode() == ScanMode.REALTIME:
                logger.info("Switching to realtime mode")
                return

        # Long sleep in batch mode
        logger.info("Batch cycle complete, sleeping...")
        await asyncio.sleep(1800)  # 30 min

    # ============================
    # Scan Tasks
    # ============================

    async def _run_global_scan(self):
        """Run global news scan"""
        try:
            result = await self.global_ingestor.scan(hours_back=2)
            self.stats.global_scans += 1

            # Process hot tickers
            for ticker in result.hot_tickers:
                self.hot_queue.push(
                    ticker,
                    TickerPriority.HOT,
                    TriggerReason.GLOBAL_CATALYST
                )
                await self._emit_hot_ticker(ticker, "global_catalyst")
                self.stats.hot_tickers_processed += 1

            # Emit catalysts from filtered items
            for item in result.news_items:
                if item.tickers and item.filter_priority:
                    await self._emit_catalyst({
                        "source": item.source,
                        "tickers": item.tickers,
                        "headline": item.headline,
                        "priority": item.filter_priority.name,
                        "category": item.filter_category
                    })
                    self.stats.catalysts_detected += 1

            logger.debug(f"Global scan: {result.items_filtered} items, {len(result.hot_tickers)} hot")

        except Exception as e:
            logger.error(f"Global scan error: {e}")
            self.stats.errors += 1

    async def _run_company_scan(self, ticker: str, priority: ScanPriority):
        """Run company-specific scan"""
        try:
            result = await self.company_scanner.scan_company(ticker, priority)
            self.stats.company_scans += 1

            # Emit top catalyst
            if result.top_catalyst:
                await self._emit_catalyst({
                    "source": "company_news",
                    "ticker": ticker,
                    "event_type": result.top_catalyst.event_type,
                    "impact": result.top_catalyst.event_impact,
                    "tier": result.top_catalyst.event_tier,
                    "headline": result.top_catalyst.headline
                })
                self.stats.catalysts_detected += 1

                # Promote to HOT if high impact
                if result.top_catalyst.event_tier <= 2:
                    self.hot_queue.promote(ticker, TickerPriority.HOT)

            logger.debug(f"Company scan {ticker}: {result.catalyst_count} catalysts")

        except Exception as e:
            logger.error(f"Company scan error for {ticker}: {e}")
            self.stats.errors += 1

    async def _run_buzz_check(self):
        """Check social buzz for hot tickers"""
        try:
            hot_tickers = self.hot_queue.get_hot_tickers()[:10]

            for ticker in hot_tickers:
                metrics = await self.buzz_engine.get_buzz(ticker)

                if metrics.should_trigger_hot:
                    self.hot_queue.push(
                        ticker,
                        TickerPriority.HOT,
                        TriggerReason.SOCIAL_BUZZ,
                        {"buzz_score": metrics.buzz_score, "acceleration": metrics.acceleration}
                    )
                    await self._emit_hot_ticker(ticker, "social_buzz")

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Buzz check error: {e}")
            self.stats.errors += 1

    async def _rotate_universe(self):
        """Rotate through universe for normal scans"""
        if not self.universe:
            return

        universe_list = list(self.universe)

        # Scan a few tickers each cycle
        for _ in range(3):
            if self._universe_index >= len(universe_list):
                self._universe_index = 0

            ticker = universe_list[self._universe_index]
            self._universe_index += 1

            # Skip if already in hot queue
            if ticker not in self.hot_queue:
                asyncio.create_task(
                    self._run_company_scan(ticker, ScanPriority.NORMAL)
                )

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        if not self.stats:
            return {}

        runtime = (datetime.utcnow() - self.stats.start_time).total_seconds()

        return {
            "runtime_seconds": runtime,
            "mode": self.stats.mode.value,
            "session": self.stats.session.value,
            "global_scans": self.stats.global_scans,
            "company_scans": self.stats.company_scans,
            "hot_tickers": self.stats.hot_tickers_processed,
            "catalysts": self.stats.catalysts_detected,
            "errors": self.stats.errors,
            "hot_queue_size": len(self.hot_queue)
        }


# ============================
# Convenience Functions
# ============================

_scheduler_instance = None
_scheduler_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_scheduler(universe: Set[str] = None) -> ScanScheduler:
    """Get singleton scheduler instance"""
    global _scheduler_instance
    with _scheduler_lock:
        if _scheduler_instance is None:
            _scheduler_instance = ScanScheduler(universe)
        elif universe:
            _scheduler_instance.set_universe(universe)
    return _scheduler_instance


# ============================
# Module exports
# ============================

__all__ = [
    "ScanScheduler",
    "ScanMode",
    "MarketSession",
    "SchedulerStats",
    "get_scheduler",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        universe = {"AAPL", "TSLA", "NVDA", "BIOX", "MRNA"}

        scheduler = ScanScheduler(universe)

        # Register callbacks
        scheduler.on_catalyst(lambda c: print(f"CATALYST: {c}"))
        scheduler.on_hot_ticker(lambda t, r: print(f"HOT: {t} ({r})"))

        print("=" * 60)
        print("SCAN SCHEDULER TEST")
        print(f"Mode: {scheduler._get_mode().value}")
        print(f"Session: {scheduler._get_session().value}")
        print("=" * 60)

        # Run for 30 seconds
        try:
            task = asyncio.create_task(scheduler.start())
            await asyncio.sleep(30)
            await scheduler.stop()
            await task
        except Exception as e:
            print(f"Error: {e}")

        print("\nStats:", scheduler.get_stats())

    asyncio.run(test())
