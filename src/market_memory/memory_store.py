"""
Memory Store - Persistence Layer for Market Memory

Handles storage and retrieval of:
- Missed opportunities
- Trade records
- Learned patterns
- Ticker profiles
- Context scores

Supports multiple backends:
- JSON files (default)
- SQLite database
- In-memory (testing)
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, TypeVar
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StorageBackend(Enum):
    """Available storage backends."""
    JSON = "JSON"
    SQLITE = "SQLITE"
    MEMORY = "MEMORY"


@dataclass
class StoreConfig:
    """Configuration for memory store."""
    backend: StorageBackend = StorageBackend.JSON
    base_path: str = "data/market_memory"
    db_path: str = "data/market_memory.db"

    # Retention
    retention_days: int = 365
    max_records_per_type: int = 100000

    # Auto-save
    auto_save: bool = True
    save_interval_minutes: int = 5

    # Compression
    compress_old_data: bool = True
    compress_after_days: int = 30


class BaseStorage(ABC):
    """Abstract base for storage backends."""

    @abstractmethod
    def save(self, collection: str, key: str, data: Dict) -> bool:
        """Save a record."""
        pass

    @abstractmethod
    def load(self, collection: str, key: str) -> Optional[Dict]:
        """Load a record."""
        pass

    @abstractmethod
    def load_all(self, collection: str) -> List[Dict]:
        """Load all records from a collection."""
        pass

    @abstractmethod
    def delete(self, collection: str, key: str) -> bool:
        """Delete a record."""
        pass

    @abstractmethod
    def query(
        self,
        collection: str,
        filters: Optional[Dict] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Query records with filters."""
        pass

    @abstractmethod
    def count(self, collection: str) -> int:
        """Count records in collection."""
        pass

    @abstractmethod
    def clear(self, collection: str) -> int:
        """Clear all records in collection."""
        pass


class JSONStorage(BaseStorage):
    """JSON file-based storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, Dict[str, Dict]] = {}

    def _get_collection_path(self, collection: str) -> Path:
        """Get path for collection file."""
        return self.base_path / f"{collection}.json"

    def _load_collection(self, collection: str) -> Dict[str, Dict]:
        """Load collection into cache."""
        if collection in self._cache:
            return self._cache[collection]

        path = self._get_collection_path(collection)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._cache[collection] = data
            except Exception as e:
                logger.error(f"Error loading collection {collection}: {e}")
                self._cache[collection] = {}
        else:
            self._cache[collection] = {}

        return self._cache[collection]

    def _save_collection(self, collection: str) -> bool:
        """Save collection to disk."""
        if collection not in self._cache:
            return True

        path = self._get_collection_path(collection)
        try:
            with open(path, 'w') as f:
                json.dump(self._cache[collection], f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving collection {collection}: {e}")
            return False

    def save(self, collection: str, key: str, data: Dict) -> bool:
        """Save a record."""
        coll = self._load_collection(collection)
        coll[key] = data
        return self._save_collection(collection)

    def load(self, collection: str, key: str) -> Optional[Dict]:
        """Load a record."""
        coll = self._load_collection(collection)
        return coll.get(key)

    def load_all(self, collection: str) -> List[Dict]:
        """Load all records."""
        coll = self._load_collection(collection)
        return list(coll.values())

    def delete(self, collection: str, key: str) -> bool:
        """Delete a record."""
        coll = self._load_collection(collection)
        if key in coll:
            del coll[key]
            return self._save_collection(collection)
        return False

    def query(
        self,
        collection: str,
        filters: Optional[Dict] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Query with filters."""
        coll = self._load_collection(collection)
        results = []

        for record in coll.values():
            if filters:
                match = all(
                    record.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue
            results.append(record)
            if len(results) >= limit:
                break

        return results

    def count(self, collection: str) -> int:
        """Count records."""
        coll = self._load_collection(collection)
        return len(coll)

    def clear(self, collection: str) -> int:
        """Clear collection."""
        coll = self._load_collection(collection)
        count = len(coll)
        self._cache[collection] = {}
        self._save_collection(collection)
        return count

    def flush(self) -> None:
        """Flush all cached data to disk."""
        for collection in self._cache:
            self._save_collection(collection)


class SQLiteStorage(BaseStorage):
    """SQLite database storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    collection TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (collection, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON records(collection)
            """)
            conn.commit()

    def save(self, collection: str, key: str, data: Dict) -> bool:
        """Save a record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO records
                    (collection, key, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (collection, key, json.dumps(data, default=str), now, now))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"SQLite save error: {e}")
            return False

    def load(self, collection: str, key: str) -> Optional[Dict]:
        """Load a record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data FROM records
                    WHERE collection = ? AND key = ?
                """, (collection, key))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.error(f"SQLite load error: {e}")
        return None

    def load_all(self, collection: str) -> List[Dict]:
        """Load all records."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data FROM records WHERE collection = ?
                """, (collection,))
                return [json.loads(row[0]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"SQLite load_all error: {e}")
            return []

    def delete(self, collection: str, key: str) -> bool:
        """Delete a record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM records WHERE collection = ? AND key = ?
                """, (collection, key))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"SQLite delete error: {e}")
            return False

    def query(
        self,
        collection: str,
        filters: Optional[Dict] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Query with filters (limited - loads all then filters)."""
        records = self.load_all(collection)

        if filters:
            records = [
                r for r in records
                if all(r.get(k) == v for k, v in filters.items())
            ]

        return records[:limit]

    def count(self, collection: str) -> int:
        """Count records."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM records WHERE collection = ?
                """, (collection,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"SQLite count error: {e}")
            return 0

    def clear(self, collection: str) -> int:
        """Clear collection."""
        count = self.count(collection)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM records WHERE collection = ?
                """, (collection,))
                conn.commit()
        except Exception as e:
            logger.error(f"SQLite clear error: {e}")
        return count


class MemoryStorage(BaseStorage):
    """In-memory storage (for testing)."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Dict]] = {}

    def save(self, collection: str, key: str, data: Dict) -> bool:
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][key] = data
        return True

    def load(self, collection: str, key: str) -> Optional[Dict]:
        return self._data.get(collection, {}).get(key)

    def load_all(self, collection: str) -> List[Dict]:
        return list(self._data.get(collection, {}).values())

    def delete(self, collection: str, key: str) -> bool:
        if collection in self._data and key in self._data[collection]:
            del self._data[collection][key]
            return True
        return False

    def query(
        self,
        collection: str,
        filters: Optional[Dict] = None,
        limit: int = 1000
    ) -> List[Dict]:
        records = self.load_all(collection)
        if filters:
            records = [
                r for r in records
                if all(r.get(k) == v for k, v in filters.items())
            ]
        return records[:limit]

    def count(self, collection: str) -> int:
        return len(self._data.get(collection, {}))

    def clear(self, collection: str) -> int:
        count = self.count(collection)
        self._data[collection] = {}
        return count


class MemoryStore:
    """
    Central persistence layer for market memory.

    Usage:
        store = MemoryStore()

        # Save data
        store.save_miss(miss_record)
        store.save_trade(trade_record)
        store.save_pattern(pattern)

        # Load data
        misses = store.load_misses(ticker="AAPL")
        trades = store.load_trades(since=datetime.now() - timedelta(days=30))

        # Export/Import
        store.export_all("backup.json")
        store.import_all("backup.json")
    """

    # Collection names
    MISSES = "misses"
    TRADES = "trades"
    PATTERNS = "patterns"
    PROFILES = "ticker_profiles"
    SCORES = "context_scores"
    METADATA = "metadata"

    def __init__(self, config: Optional[StoreConfig] = None):
        self.config = config or StoreConfig()

        # Initialize storage backend
        if self.config.backend == StorageBackend.JSON:
            self._storage = JSONStorage(self.config.base_path)
        elif self.config.backend == StorageBackend.SQLITE:
            self._storage = SQLiteStorage(self.config.db_path)
        else:
            self._storage = MemoryStorage()

        # Last save time
        self._last_save = datetime.now()

        logger.info(f"MemoryStore initialized with {self.config.backend.value} backend")

    # Miss records

    def save_miss(self, miss: Dict) -> bool:
        """Save a missed opportunity record."""
        key = miss.get("id", str(datetime.now().timestamp()))
        return self._storage.save(self.MISSES, key, miss)

    def load_miss(self, miss_id: str) -> Optional[Dict]:
        """Load a specific miss."""
        return self._storage.load(self.MISSES, miss_id)

    def load_misses(
        self,
        ticker: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Load misses with optional filters."""
        filters = {}
        if ticker:
            filters["ticker"] = ticker.upper()

        records = self._storage.query(self.MISSES, filters, limit * 2)

        # Apply date filter
        if since:
            since_str = since.isoformat()
            records = [
                r for r in records
                if r.get("signal_time", "") >= since_str
            ]

        return records[:limit]

    # Trade records

    def save_trade(self, trade: Dict) -> bool:
        """Save a trade record."""
        key = trade.get("id", str(datetime.now().timestamp()))
        return self._storage.save(self.TRADES, key, trade)

    def load_trades(
        self,
        ticker: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Load trades with optional filters."""
        filters = {}
        if ticker:
            filters["ticker"] = ticker.upper()

        records = self._storage.query(self.TRADES, filters, limit * 2)

        if since:
            since_str = since.isoformat()
            records = [
                r for r in records
                if r.get("entry_time", "") >= since_str
            ]

        return records[:limit]

    # Patterns

    def save_pattern(self, pattern: Dict) -> bool:
        """Save a learned pattern."""
        key = pattern.get("id", pattern.get("name", str(datetime.now().timestamp())))
        return self._storage.save(self.PATTERNS, key, pattern)

    def load_patterns(
        self,
        pattern_type: Optional[str] = None
    ) -> List[Dict]:
        """Load patterns."""
        filters = {}
        if pattern_type:
            filters["pattern_type"] = pattern_type
        return self._storage.query(self.PATTERNS, filters)

    # Ticker profiles

    def save_profile(self, profile: Dict) -> bool:
        """Save a ticker profile."""
        ticker = profile.get("ticker", "UNKNOWN")
        return self._storage.save(self.PROFILES, ticker, profile)

    def load_profile(self, ticker: str) -> Optional[Dict]:
        """Load a ticker profile."""
        return self._storage.load(self.PROFILES, ticker.upper())

    def load_all_profiles(self) -> List[Dict]:
        """Load all ticker profiles."""
        return self._storage.load_all(self.PROFILES)

    # Context scores (cached)

    def save_score(self, score: Dict) -> bool:
        """Save a context score."""
        ticker = score.get("ticker", "UNKNOWN")
        timestamp = score.get("timestamp", datetime.now().isoformat())
        key = f"{ticker}_{timestamp}"
        return self._storage.save(self.SCORES, key, score)

    # Metadata

    def save_metadata(self, key: str, data: Dict) -> bool:
        """Save metadata."""
        return self._storage.save(self.METADATA, key, data)

    def load_metadata(self, key: str) -> Optional[Dict]:
        """Load metadata."""
        return self._storage.load(self.METADATA, key)

    # Bulk operations

    def export_all(self, filepath: str) -> bool:
        """Export all data to a JSON file."""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "misses": self._storage.load_all(self.MISSES),
                "trades": self._storage.load_all(self.TRADES),
                "patterns": self._storage.load_all(self.PATTERNS),
                "profiles": self._storage.load_all(self.PROFILES),
                "metadata": self._storage.load_all(self.METADATA),
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported all data to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_all(self, filepath: str, merge: bool = True) -> bool:
        """Import data from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if not merge:
                # Clear existing data
                for collection in [self.MISSES, self.TRADES, self.PATTERNS, self.PROFILES]:
                    self._storage.clear(collection)

            # Import each collection
            for miss in data.get("misses", []):
                self.save_miss(miss)

            for trade in data.get("trades", []):
                self.save_trade(trade)

            for pattern in data.get("patterns", []):
                self.save_pattern(pattern)

            for profile in data.get("profiles", []):
                self.save_profile(profile)

            logger.info(f"Imported data from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False

    # Maintenance

    def cleanup_old(self, days: Optional[int] = None) -> Dict[str, int]:
        """Remove records older than N days."""
        days = days or self.config.retention_days
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        removed = {}

        # Cleanup misses
        misses = self._storage.load_all(self.MISSES)
        old_misses = [
            m for m in misses
            if m.get("signal_time", datetime.now().isoformat()) < cutoff
        ]
        for miss in old_misses:
            self._storage.delete(self.MISSES, miss.get("id", ""))
        removed["misses"] = len(old_misses)

        # Cleanup trades
        trades = self._storage.load_all(self.TRADES)
        old_trades = [
            t for t in trades
            if t.get("entry_time", datetime.now().isoformat()) < cutoff
        ]
        for trade in old_trades:
            self._storage.delete(self.TRADES, trade.get("id", ""))
        removed["trades"] = len(old_trades)

        logger.info(f"Cleanup removed: {removed}")
        return removed

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            "backend": self.config.backend.value,
            "misses_count": self._storage.count(self.MISSES),
            "trades_count": self._storage.count(self.TRADES),
            "patterns_count": self._storage.count(self.PATTERNS),
            "profiles_count": self._storage.count(self.PROFILES),
        }

    def flush(self) -> None:
        """Flush any cached data to storage."""
        if hasattr(self._storage, 'flush'):
            self._storage.flush()


# Singleton instance
_store: Optional[MemoryStore] = None


def get_memory_store(config: Optional[StoreConfig] = None) -> MemoryStore:
    """Get singleton MemoryStore instance."""
    global _store
    if _store is None:
        _store = MemoryStore(config)
    return _store
