"""
HOT TICKER QUEUE V6.1
=====================

Queue de priorité pour les tickers "chauds" nécessitant
des scans fréquents.

Triggers pour devenir HOT:
- Pre-Spike Radar >= 2 signaux
- Catalyst détecté en global scan
- Repeat Gainer score élevé (> 0.7)
- Mouvement PM/AH > 5%
- Accélération buzz social > 3x

Priorités:
- HOT: scan toutes les 1-2 min
- WARM: scan toutes les 5 min
- NORMAL: rotation 10-15 min

Architecture:
- Priority heap pour tri automatique
- TTL automatique (tickers expirent après inactivité)
- Promotion/démotion automatique
- Thread-safe pour usage async
"""

import heapq
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import os

from utils.logger import get_logger

logger = get_logger("HOT_TICKER_QUEUE")


# ============================
# Configuration
# ============================

# C6 FIX: Extended TTL (was 1h/30m/15m — too short, hot tickers expired
# before they could spike). Auto-renewal on re-scan keeps active tickers alive.
TTL_HOT = 14400   # 4 hours (was 1h)
TTL_WARM = 7200   # 2 hours (was 30 min)
TTL_NORMAL = 3600  # 1 hour (was 15 min)

# Scan intervals (seconds)
INTERVAL_HOT = 90  # 1.5 min
INTERVAL_WARM = 300  # 5 min
INTERVAL_NORMAL = 600  # 10 min

# Persistence
QUEUE_DB = "data/hot_ticker_queue.db"


# ============================
# Enums
# ============================

class TickerPriority(Enum):
    """Ticker priority levels"""
    HOT = 1
    WARM = 2
    NORMAL = 3

    def __lt__(self, other):
        return self.value < other.value


class TriggerReason(Enum):
    """Reasons for adding to hot queue"""
    PRE_SPIKE_RADAR = "pre_spike_radar"
    GLOBAL_CATALYST = "global_catalyst"
    REPEAT_GAINER = "repeat_gainer"
    EXTENDED_HOURS_MOVE = "extended_hours_move"
    SOCIAL_BUZZ = "social_buzz"
    SEC_FILING = "sec_filing"
    MANUAL = "manual"


# ============================
# Data Classes
# ============================

@dataclass(order=True)
class HotTicker:
    """Represents a hot ticker in the queue"""
    priority: TickerPriority = field(compare=True)
    ticker: str = field(compare=False)
    reason: TriggerReason = field(compare=False)
    added_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    last_scan: Optional[datetime] = field(compare=False, default=None)
    expires_at: datetime = field(compare=False, default=None)
    metadata: Dict = field(compare=False, default_factory=dict)

    def __post_init__(self):
        if self.expires_at is None:
            ttl = {
                TickerPriority.HOT: TTL_HOT,
                TickerPriority.WARM: TTL_WARM,
                TickerPriority.NORMAL: TTL_NORMAL
            }
            self.expires_at = self.added_at + timedelta(seconds=ttl[self.priority])

    def is_expired(self) -> bool:
        """Check if ticker has expired"""
        return datetime.utcnow() > self.expires_at

    def should_scan(self) -> bool:
        """Check if ticker should be scanned now"""
        if self.last_scan is None:
            return True

        intervals = {
            TickerPriority.HOT: INTERVAL_HOT,
            TickerPriority.WARM: INTERVAL_WARM,
            TickerPriority.NORMAL: INTERVAL_NORMAL
        }

        elapsed = (datetime.utcnow() - self.last_scan).total_seconds()
        return elapsed >= intervals[self.priority]


# ============================
# Hot Ticker Queue
# ============================

class HotTickerQueue:
    """
    Priority queue for hot tickers

    Usage:
        queue = HotTickerQueue()
        queue.push("AAPL", TickerPriority.HOT, TriggerReason.PRE_SPIKE_RADAR)
        ticker = queue.pop_next_for_scan()
        if ticker:
            # scan ticker
            queue.mark_scanned(ticker)
    """

    def __init__(self, persist: bool = True):
        self._lock = threading.Lock()
        self._heap: List[Tuple[int, HotTicker]] = []
        self._tickers: Dict[str, HotTicker] = {}
        self._counter = 0  # For stable heap ordering

        self.persist = persist
        if persist:
            self._init_db()
            self._load_from_db()

    def _init_db(self):
        """Initialize persistence database"""
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(QUEUE_DB, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hot_tickers (
                ticker TEXT PRIMARY KEY,
                priority INTEGER,
                reason TEXT,
                added_at TEXT,
                last_scan TEXT,
                expires_at TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def _load_from_db(self):
        """Load queue from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM hot_tickers")

        for row in cursor.fetchall():
            ticker, priority, reason, added_at, last_scan, expires_at, metadata = row

            hot = HotTicker(
                ticker=ticker,
                priority=TickerPriority(priority),
                reason=TriggerReason(reason),
                added_at=datetime.fromisoformat(added_at),
                last_scan=datetime.fromisoformat(last_scan) if last_scan else None,
                expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
                metadata=eval(metadata) if metadata else {}
            )

            # Only load if not expired
            if not hot.is_expired():
                self._tickers[ticker] = hot
                self._counter += 1
                heapq.heappush(self._heap, (hot.priority.value, self._counter, hot))

        logger.info(f"Loaded {len(self._tickers)} hot tickers from DB")

    def _save_to_db(self, hot: HotTicker):
        """Save ticker to database"""
        if not self.persist:
            return

        self.conn.execute("""
            INSERT OR REPLACE INTO hot_tickers
            (ticker, priority, reason, added_at, last_scan, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            hot.ticker,
            hot.priority.value,
            hot.reason.value,
            hot.added_at.isoformat(),
            hot.last_scan.isoformat() if hot.last_scan else None,
            hot.expires_at.isoformat() if hot.expires_at else None,
            str(hot.metadata)
        ))
        self.conn.commit()

    def _remove_from_db(self, ticker: str):
        """Remove ticker from database"""
        if not self.persist:
            return

        self.conn.execute("DELETE FROM hot_tickers WHERE ticker = ?", (ticker,))
        self.conn.commit()

    def push(
        self,
        ticker: str,
        priority: TickerPriority = TickerPriority.HOT,
        reason: TriggerReason = TriggerReason.MANUAL,
        metadata: Dict = None
    ) -> bool:
        """
        Add ticker to hot queue

        Args:
            ticker: Stock ticker
            priority: Priority level
            reason: Why ticker was added
            metadata: Additional context

        Returns:
            True if added, False if already exists at same/higher priority
        """
        ticker = ticker.upper()

        with self._lock:
            existing = self._tickers.get(ticker)

            # If exists at higher priority, don't downgrade
            if existing and existing.priority.value <= priority.value:
                # Refresh TTL
                existing.expires_at = datetime.utcnow() + timedelta(
                    seconds={
                        TickerPriority.HOT: TTL_HOT,
                        TickerPriority.WARM: TTL_WARM,
                        TickerPriority.NORMAL: TTL_NORMAL
                    }[existing.priority]
                )
                self._save_to_db(existing)
                return False

            # Create new hot ticker
            hot = HotTicker(
                ticker=ticker,
                priority=priority,
                reason=reason,
                metadata=metadata or {}
            )

            self._tickers[ticker] = hot
            self._counter += 1
            heapq.heappush(self._heap, (priority.value, self._counter, hot))
            self._save_to_db(hot)

            logger.info(f"Added {ticker} to hot queue: {priority.name} ({reason.value})")
            return True

    def pop_next_for_scan(self) -> Optional[str]:
        """
        Get next ticker that needs scanning

        Returns:
            Ticker string or None if no tickers need scanning
        """
        with self._lock:
            # Clean expired entries
            self._cleanup_expired()

            # Find first ticker that should be scanned
            for _, _, hot in sorted(self._heap):
                if hot.ticker in self._tickers and hot.should_scan():
                    return hot.ticker

            return None

    def get_all_for_scan(self) -> List[str]:
        """
        Get all tickers that need scanning, sorted by priority

        Returns:
            List of ticker strings
        """
        with self._lock:
            self._cleanup_expired()

            result = []
            for _, _, hot in sorted(self._heap):
                if hot.ticker in self._tickers and hot.should_scan():
                    result.append(hot.ticker)

            return result

    def mark_scanned(self, ticker: str, still_active: bool = False):
        """
        Mark ticker as just scanned.

        C6 FIX: If still_active=True, auto-renew TTL so hot tickers
        don't expire while they're still showing activity.
        """
        ticker = ticker.upper()

        with self._lock:
            if ticker in self._tickers:
                hot = self._tickers[ticker]
                hot.last_scan = datetime.utcnow()

                # C6: Auto-renewal — extend TTL if ticker still shows activity
                if still_active:
                    ttl = {
                        TickerPriority.HOT: TTL_HOT,
                        TickerPriority.WARM: TTL_WARM,
                        TickerPriority.NORMAL: TTL_NORMAL
                    }
                    hot.expires_at = datetime.utcnow() + timedelta(seconds=ttl[hot.priority])
                    logger.debug(f"Auto-renewed TTL for {ticker} ({hot.priority.name})")

                self._save_to_db(hot)

    def promote(self, ticker: str, new_priority: TickerPriority):
        """Promote ticker to higher priority"""
        ticker = ticker.upper()

        with self._lock:
            if ticker in self._tickers:
                hot = self._tickers[ticker]
                if new_priority.value < hot.priority.value:
                    hot.priority = new_priority
                    # Refresh TTL
                    hot.expires_at = datetime.utcnow() + timedelta(
                        seconds=TTL_HOT if new_priority == TickerPriority.HOT else TTL_WARM
                    )
                    self._save_to_db(hot)
                    logger.info(f"Promoted {ticker} to {new_priority.name}")

    def demote(self, ticker: str, new_priority: TickerPriority):
        """Demote ticker to lower priority"""
        ticker = ticker.upper()

        with self._lock:
            if ticker in self._tickers:
                hot = self._tickers[ticker]
                if new_priority.value > hot.priority.value:
                    hot.priority = new_priority
                    self._save_to_db(hot)
                    logger.info(f"Demoted {ticker} to {new_priority.name}")

    def remove(self, ticker: str):
        """Remove ticker from queue"""
        ticker = ticker.upper()

        with self._lock:
            if ticker in self._tickers:
                del self._tickers[ticker]
                self._remove_from_db(ticker)
                logger.info(f"Removed {ticker} from hot queue")

    def _cleanup_expired(self):
        """Remove expired tickers"""
        expired = []
        for ticker, hot in self._tickers.items():
            if hot.is_expired():
                expired.append(ticker)

        for ticker in expired:
            del self._tickers[ticker]
            self._remove_from_db(ticker)

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired tickers")

    def get_stats(self) -> Dict:
        """Get queue statistics"""
        with self._lock:
            self._cleanup_expired()

            hot_count = sum(1 for t in self._tickers.values() if t.priority == TickerPriority.HOT)
            warm_count = sum(1 for t in self._tickers.values() if t.priority == TickerPriority.WARM)
            normal_count = sum(1 for t in self._tickers.values() if t.priority == TickerPriority.NORMAL)

            # Reason breakdown
            reasons = {}
            for t in self._tickers.values():
                reasons[t.reason.value] = reasons.get(t.reason.value, 0) + 1

            return {
                "total": len(self._tickers),
                "hot": hot_count,
                "warm": warm_count,
                "normal": normal_count,
                "by_reason": reasons
            }

    def get_hot_tickers(self) -> List[str]:
        """Get all HOT priority tickers"""
        with self._lock:
            return [t for t, h in self._tickers.items() if h.priority == TickerPriority.HOT]

    def get_all_tickers(self) -> List[str]:
        """Get all tickers in queue"""
        with self._lock:
            return list(self._tickers.keys())

    def contains(self, ticker: str) -> bool:
        """Check if ticker is in queue"""
        return ticker.upper() in self._tickers

    def get_priority(self, ticker: str) -> Optional[TickerPriority]:
        """Get priority of ticker"""
        hot = self._tickers.get(ticker.upper())
        return hot.priority if hot else None

    def __len__(self) -> int:
        return len(self._tickers)

    def __contains__(self, ticker: str) -> bool:
        return self.contains(ticker)


# ============================
# Convenience Functions
# ============================

_queue_instance = None


def get_hot_queue() -> HotTickerQueue:
    """Get singleton queue instance"""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = HotTickerQueue()
    return _queue_instance


def add_hot_ticker(
    ticker: str,
    priority: TickerPriority = TickerPriority.HOT,
    reason: TriggerReason = TriggerReason.MANUAL
):
    """Quick add ticker to hot queue"""
    queue = get_hot_queue()
    queue.push(ticker, priority, reason)


# ============================
# Module exports
# ============================

__all__ = [
    "HotTickerQueue",
    "HotTicker",
    "TickerPriority",
    "TriggerReason",
    "get_hot_queue",
    "add_hot_ticker",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    queue = HotTickerQueue(persist=False)

    print("=" * 60)
    print("HOT TICKER QUEUE TEST")
    print("=" * 60)

    # Add tickers
    queue.push("AAPL", TickerPriority.HOT, TriggerReason.PRE_SPIKE_RADAR)
    queue.push("TSLA", TickerPriority.WARM, TriggerReason.GLOBAL_CATALYST)
    queue.push("NVDA", TickerPriority.HOT, TriggerReason.SEC_FILING)
    queue.push("BIOX", TickerPriority.NORMAL, TriggerReason.SOCIAL_BUZZ)

    print(f"\nQueue size: {len(queue)}")
    print(f"Stats: {queue.get_stats()}")

    print("\nTickers for scan:")
    for ticker in queue.get_all_for_scan():
        priority = queue.get_priority(ticker)
        print(f"  {ticker}: {priority.name}")

    # Simulate scan
    print("\nScanning AAPL...")
    queue.mark_scanned("AAPL")

    print("\nAfter scan, tickers for scan:")
    for ticker in queue.get_all_for_scan():
        print(f"  {ticker}")

    # Promote
    print("\nPromoting BIOX to HOT...")
    queue.promote("BIOX", TickerPriority.HOT)
    print(f"BIOX priority: {queue.get_priority('BIOX').name}")
