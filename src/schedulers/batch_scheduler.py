"""
BATCH SCHEDULER V6.1
====================

Traitement batch hors-marche (nuits, weekends, jours feries).

Objectifs:
- Ingestion complete SEC filings
- Scan complet universe
- Recalcul Catalyst Score & Repeat Gainer Memory
- Generation watchlists prioritaires
- Cleanup & maintenance

Execution:
- Automatic: Detecte mode BATCH via ScanScheduler
- Manual: python -m src.schedulers.batch_scheduler

Architecture:
- Tasks sequentielles (pas de rate limit concerns)
- Progress tracking
- Resume capability
- Report generation
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger

# Import ingestors
from src.ingestors.sec_filings_ingestor import SECIngestor
from src.ingestors.global_news_ingestor import GlobalNewsIngestor
from src.ingestors.company_news_scanner import CompanyNewsScanner, ScanPriority
from src.ingestors.social_buzz_engine import SocialBuzzEngine

# Import processors
from src.processors.nlp_classifier import get_classifier

logger = get_logger("BATCH_SCHEDULER")


# ============================
# Configuration
# ============================

BATCH_REPORT_DIR = "data/batch_reports"
WATCHLIST_DIR = "data/watchlists"


# ============================
# Enums
# ============================

class BatchTaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================
# Data Classes
# ============================

@dataclass
class BatchTask:
    """Represents a batch task"""
    name: str
    status: BatchTaskStatus = BatchTaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    items_processed: int = 0
    errors: int = 0
    result: Optional[Dict] = None


@dataclass
class BatchReport:
    """Batch run report"""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks: List[BatchTask] = None
    watchlist: List[str] = None
    summary: Dict = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.watchlist is None:
            self.watchlist = []
        if self.summary is None:
            self.summary = {}


# ============================
# Batch Scheduler
# ============================

class BatchScheduler:
    """
    Off-market batch processing

    Usage:
        batch = BatchScheduler(universe={"AAPL", "TSLA"})
        report = await batch.run()
        print(f"Watchlist: {report.watchlist}")
    """

    def __init__(self, universe: Set[str] = None):
        self.universe = universe or set()

        # Ingestors
        self.sec_ingestor = SECIngestor(universe)
        self.global_ingestor = GlobalNewsIngestor(universe)
        self.company_scanner = CompanyNewsScanner()
        self.buzz_engine = SocialBuzzEngine()
        self.nlp_classifier = get_classifier()

        # State
        self.report = None
        self.running = False

        # Create directories
        os.makedirs(BATCH_REPORT_DIR, exist_ok=True)
        os.makedirs(WATCHLIST_DIR, exist_ok=True)

    def set_universe(self, universe: Set[str]):
        """Update universe"""
        self.universe = universe
        self.sec_ingestor.set_universe(universe)
        self.global_ingestor.set_universe(universe)

    async def run(self, hours_back: int = 24) -> BatchReport:
        """
        Run full batch processing

        Args:
            hours_back: How far back to look for data

        Returns:
            BatchReport with results and watchlist
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.report = BatchReport(
            run_id=run_id,
            started_at=datetime.now(timezone.utc)
        )

        self.running = True

        logger.info("=" * 60)
        logger.info(f"BATCH RUN STARTING: {run_id}")
        logger.info(f"Universe: {len(self.universe)} tickers")
        logger.info(f"Hours back: {hours_back}")
        logger.info("=" * 60)

        try:
            # Task 1: SEC Filings Ingestion
            await self._run_task(
                "sec_filings",
                self._task_sec_filings,
                hours_back=hours_back
            )

            # Task 2: Global News Scan
            await self._run_task(
                "global_news",
                self._task_global_news,
                hours_back=hours_back
            )

            # Task 3: Company News Scan (full universe)
            await self._run_task(
                "company_news",
                self._task_company_news,
                hours_back=hours_back
            )

            # Task 4: Social Buzz Baseline
            await self._run_task(
                "social_buzz",
                self._task_social_buzz
            )

            # Task 5: Generate Watchlist
            await self._run_task(
                "generate_watchlist",
                self._task_generate_watchlist
            )

            # Task 6: Cleanup
            await self._run_task(
                "cleanup",
                self._task_cleanup
            )

        except Exception as e:
            logger.error(f"Batch run error: {e}")

        finally:
            self.running = False
            self.report.completed_at = datetime.now(timezone.utc)
            self._generate_summary()
            self._save_report()

        logger.info("=" * 60)
        logger.info(f"BATCH RUN COMPLETED: {run_id}")
        logger.info(f"Duration: {(self.report.completed_at - self.report.started_at).total_seconds():.1f}s")
        logger.info(f"Watchlist: {len(self.report.watchlist)} tickers")
        logger.info("=" * 60)

        return self.report

    async def _run_task(self, name: str, func, **kwargs):
        """Run a batch task with tracking"""
        task = BatchTask(
            name=name,
            status=BatchTaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        self.report.tasks.append(task)

        logger.info(f"Starting task: {name}")

        try:
            result = await func(**kwargs)
            task.status = BatchTaskStatus.COMPLETED
            task.result = result
            logger.info(f"Task completed: {name}")

        except Exception as e:
            task.status = BatchTaskStatus.FAILED
            task.errors += 1
            task.result = {"error": str(e)}
            logger.error(f"Task failed: {name} - {e}")

        finally:
            task.completed_at = datetime.now(timezone.utc)

    # ============================
    # Tasks
    # ============================

    async def _task_sec_filings(self, hours_back: int) -> Dict:
        """Ingest SEC filings"""
        task = self._get_task("sec_filings")

        # Fetch 8-K filings
        filings_8k = await self.sec_ingestor.fetch_8k_filings(hours_back=hours_back)
        task.items_processed += len(filings_8k)

        # Categorize and store
        catalysts = []
        for filing in filings_8k:
            if filing.ticker in self.universe:
                catalysts.append({
                    "ticker": filing.ticker,
                    "type": filing.event_type,
                    "source": "sec_8k",
                    "date": filing.filed_date.isoformat()
                })

        return {
            "filings_8k": len(filings_8k),
            "catalysts": len(catalysts),
            "tickers_affected": list(set(c["ticker"] for c in catalysts))
        }

    async def _task_global_news(self, hours_back: int) -> Dict:
        """Global news scan"""
        task = self._get_task("global_news")

        result = await self.global_ingestor.scan(hours_back=hours_back)
        task.items_processed = result.items_fetched

        return {
            "items_fetched": result.items_fetched,
            "items_filtered": result.items_filtered,
            "hot_tickers": result.hot_tickers
        }

    async def _task_company_news(self, hours_back: int) -> Dict:
        """Scan all companies in universe"""
        task = self._get_task("company_news")

        results = []
        catalysts_found = 0

        for ticker in self.universe:
            try:
                result = await self.company_scanner.scan_company(
                    ticker,
                    ScanPriority.NORMAL
                )
                task.items_processed += 1

                if result.top_catalyst:
                    catalysts_found += 1
                    results.append({
                        "ticker": ticker,
                        "event_type": result.top_catalyst.event_type,
                        "impact": result.top_catalyst.event_impact,
                        "tier": result.top_catalyst.event_tier
                    })

                # Rate limit
                await asyncio.sleep(1)

            except Exception as e:
                task.errors += 1
                logger.warning(f"Company scan error for {ticker}: {e}")

        return {
            "tickers_scanned": task.items_processed,
            "catalysts_found": catalysts_found,
            "top_catalysts": sorted(results, key=lambda x: x["impact"], reverse=True)[:20]
        }

    async def _task_social_buzz(self) -> Dict:
        """Update social buzz baselines"""
        task = self._get_task("social_buzz")

        high_buzz = []

        # Sample of universe for baseline
        sample = list(self.universe)[:50]

        for ticker in sample:
            try:
                metrics = await self.buzz_engine.get_buzz(ticker)
                task.items_processed += 1

                if metrics.buzz_score > 0.5:
                    high_buzz.append({
                        "ticker": ticker,
                        "buzz_score": metrics.buzz_score,
                        "acceleration": metrics.acceleration
                    })

                await asyncio.sleep(0.5)

            except Exception as e:
                task.errors += 1

        return {
            "tickers_checked": task.items_processed,
            "high_buzz": high_buzz
        }

    async def _task_generate_watchlist(self) -> Dict:
        """Generate priority watchlist for market open"""
        task = self._get_task("generate_watchlist")

        watchlist = []
        scores = {}

        # Aggregate scores from previous tasks
        for t in self.report.tasks:
            if t.name == "sec_filings" and t.result:
                for ticker in t.result.get("tickers_affected", []):
                    scores[ticker] = scores.get(ticker, 0) + 30  # SEC = high priority

            if t.name == "global_news" and t.result:
                for ticker in t.result.get("hot_tickers", []):
                    scores[ticker] = scores.get(ticker, 0) + 20

            if t.name == "company_news" and t.result:
                for cat in t.result.get("top_catalysts", []):
                    ticker = cat["ticker"]
                    scores[ticker] = scores.get(ticker, 0) + cat["impact"] * 25

            if t.name == "social_buzz" and t.result:
                for buzz in t.result.get("high_buzz", []):
                    ticker = buzz["ticker"]
                    scores[ticker] = scores.get(ticker, 0) + buzz["buzz_score"] * 10

        # Sort and take top 20
        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        watchlist = [t[0] for t in sorted_tickers[:20]]

        self.report.watchlist = watchlist
        task.items_processed = len(watchlist)

        # Save watchlist to file
        watchlist_file = f"{WATCHLIST_DIR}/watchlist_{self.report.run_id}.json"
        with open(watchlist_file, "w") as f:
            json.dump({
                "date": datetime.now(timezone.utc).isoformat(),
                "tickers": watchlist,
                "scores": dict(sorted_tickers[:20])
            }, f, indent=2)

        return {
            "watchlist_size": len(watchlist),
            "watchlist": watchlist,
            "file": watchlist_file
        }

    async def _task_cleanup(self) -> Dict:
        """Cleanup old data"""
        task = self._get_task("cleanup")

        deleted = 0

        # Clean old batch reports (keep last 7 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        for filename in os.listdir(BATCH_REPORT_DIR):
            if filename.startswith("batch_"):
                try:
                    date_str = filename.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date < cutoff:
                        os.remove(f"{BATCH_REPORT_DIR}/{filename}")
                        deleted += 1
                except:
                    pass

        task.items_processed = deleted

        return {"files_deleted": deleted}

    def _get_task(self, name: str) -> BatchTask:
        """Get task by name"""
        for task in self.report.tasks:
            if task.name == name:
                return task
        return None

    def _generate_summary(self):
        """Generate run summary"""
        completed = sum(1 for t in self.report.tasks if t.status == BatchTaskStatus.COMPLETED)
        failed = sum(1 for t in self.report.tasks if t.status == BatchTaskStatus.FAILED)
        total_items = sum(t.items_processed for t in self.report.tasks)
        total_errors = sum(t.errors for t in self.report.tasks)

        duration = 0
        if self.report.completed_at and self.report.started_at:
            duration = (self.report.completed_at - self.report.started_at).total_seconds()

        self.report.summary = {
            "duration_seconds": duration,
            "tasks_completed": completed,
            "tasks_failed": failed,
            "total_items_processed": total_items,
            "total_errors": total_errors,
            "watchlist_size": len(self.report.watchlist)
        }

    def _save_report(self):
        """Save report to file"""
        report_file = f"{BATCH_REPORT_DIR}/batch_{self.report.run_id}.json"

        report_dict = {
            "run_id": self.report.run_id,
            "started_at": self.report.started_at.isoformat(),
            "completed_at": self.report.completed_at.isoformat() if self.report.completed_at else None,
            "summary": self.report.summary,
            "watchlist": self.report.watchlist,
            "tasks": [
                {
                    "name": t.name,
                    "status": t.status.value,
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "items_processed": t.items_processed,
                    "errors": t.errors,
                    "result": t.result
                }
                for t in self.report.tasks
            ]
        }

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved: {report_file}")

    def get_latest_watchlist(self) -> List[str]:
        """Get most recent watchlist"""
        try:
            files = os.listdir(WATCHLIST_DIR)
            if not files:
                return []

            latest = sorted(files)[-1]
            with open(f"{WATCHLIST_DIR}/{latest}") as f:
                data = json.load(f)
                return data.get("tickers", [])
        except:
            return []


# ============================
# Convenience Functions
# ============================

async def run_batch(universe: Set[str], hours_back: int = 24) -> BatchReport:
    """Quick batch run"""
    scheduler = BatchScheduler(universe)
    return await scheduler.run(hours_back)


def get_latest_watchlist() -> List[str]:
    """Get latest watchlist"""
    scheduler = BatchScheduler()
    return scheduler.get_latest_watchlist()


# ============================
# Module exports
# ============================

__all__ = [
    "BatchScheduler",
    "BatchReport",
    "BatchTask",
    "BatchTaskStatus",
    "run_batch",
    "get_latest_watchlist",
]


# ============================
# CLI
# ============================

if __name__ == "__main__":
    import sys

    async def main():
        # Load universe
        try:
            from src.universe_loader import load_universe
            universe = set(load_universe())
        except:
            universe = {"AAPL", "TSLA", "NVDA", "BIOX", "MRNA", "PFE", "AMD", "INTC"}

        hours_back = int(sys.argv[1]) if len(sys.argv) > 1 else 24

        print("=" * 60)
        print("BATCH SCHEDULER")
        print(f"Universe: {len(universe)} tickers")
        print(f"Hours back: {hours_back}")
        print("=" * 60)

        report = await run_batch(universe, hours_back)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Duration: {report.summary['duration_seconds']:.1f}s")
        print(f"Tasks completed: {report.summary['tasks_completed']}")
        print(f"Items processed: {report.summary['total_items_processed']}")
        print(f"Watchlist: {report.watchlist}")

    asyncio.run(main())
