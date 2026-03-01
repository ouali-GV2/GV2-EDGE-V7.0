"""
Ticker Profile Store — GV2-EDGE V9.0
=====================================
Base SQLite persistante des caractéristiques structurelles par ticker.
Alimentée chaque weekend par ticker_profile_feeder.py.
Lue en intraday à coût zéro (0 API call) par Monster Score et Signal Producer.

Table : ticker_profiles (~3 000 tickers × 21 champs)
DB    : data/ticker_profiles.db
"""

import sqlite3
import threading
import json
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from utils.logger import get_logger

logger = get_logger("TICKER_PROFILE_STORE")

DB_PATH = "data/ticker_profiles.db"

# ============================================================================
# Init
# ============================================================================

def _init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_profiles (
                -- Identity
                ticker              TEXT PRIMARY KEY,

                -- TIER 1 : Capital structure
                market_cap          REAL,
                float_shares        REAL,
                shares_outstanding  REAL,
                insider_pct         REAL,
                institutional_pct   REAL,

                -- TIER 1 : Short squeeze
                short_interest_pct  REAL,
                days_to_cover       REAL,
                borrow_rate         REAL,

                -- TIER 1 : Risk flags
                reverse_split_count INTEGER DEFAULT 0,
                last_reverse_split  TEXT,
                shelf_active        INTEGER DEFAULT 0,
                atm_active          INTEGER DEFAULT 0,
                warrants_outstanding INTEGER DEFAULT 0,
                dilution_tier       TEXT,

                -- TIER 2 : Historical behavior
                top_gainer_count    INTEGER DEFAULT 0,
                avg_move_pct        REAL,
                best_session        TEXT,
                catalyst_affinity   TEXT,
                halt_count          INTEGER DEFAULT 0,

                -- TIER 2 : Technical baseline
                atr_14              REAL,
                avg_daily_volume    REAL,

                -- Meta
                data_quality        REAL DEFAULT 0.0,
                updated_at          TEXT,
                source_flags        TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_float   ON ticker_profiles(float_shares)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_short   ON ticker_profiles(short_interest_pct)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_updated ON ticker_profiles(updated_at)")
        conn.commit()
    logger.info(f"TickerProfileStore initialized: {DB_PATH}")


# ============================================================================
# Store class
# ============================================================================

_COLUMNS = [
    "ticker", "market_cap", "float_shares", "shares_outstanding",
    "insider_pct", "institutional_pct",
    "short_interest_pct", "days_to_cover", "borrow_rate",
    "reverse_split_count", "last_reverse_split",
    "shelf_active", "atm_active", "warrants_outstanding", "dilution_tier",
    "top_gainer_count", "avg_move_pct", "best_session",
    "catalyst_affinity", "halt_count",
    "atr_14", "avg_daily_volume",
    "data_quality", "updated_at", "source_flags",
]

# Tier 1 fields used to compute data_quality score
_QUALITY_FIELDS = [
    "market_cap", "float_shares", "shares_outstanding",
    "short_interest_pct", "days_to_cover", "borrow_rate",
]


class TickerProfileStore:
    """Thread-safe SQLite store for per-ticker strategic profiles."""

    def __init__(self):
        _init_db()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, profile: Dict[str, Any]) -> bool:
        """Insert or replace a ticker profile."""
        ticker = profile.get("ticker", "").upper().strip()
        if not ticker:
            return False

        profile["ticker"] = ticker
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Serialize source_flags dict → JSON string
        if isinstance(profile.get("source_flags"), dict):
            profile["source_flags"] = json.dumps(profile["source_flags"])

        # Compute data_quality (fraction of Tier-1 fields populated)
        filled = sum(1 for f in _QUALITY_FIELDS if profile.get(f) is not None)
        profile["data_quality"] = round(filled / len(_QUALITY_FIELDS), 2)

        # Build INSERT OR REPLACE
        cols = [c for c in _COLUMNS if c in profile]
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        values = [profile[c] for c in cols]

        try:
            with self._lock:
                with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                    conn.execute(
                        f"INSERT OR REPLACE INTO ticker_profiles ({col_names}) VALUES ({placeholders})",
                        values
                    )
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"upsert error for {ticker}: {e}")
            return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return profile for one ticker, or None."""
        ticker = ticker.upper().strip()
        try:
            with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM ticker_profiles WHERE ticker = ?", (ticker,)
                )
                row = cur.fetchone()
            if row is None:
                return None
            return _row_to_dict(row)
        except Exception as e:
            logger.error(f"get error for {ticker}: {e}")
            return None

    def get_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return profiles for a list of tickers as {ticker: profile}."""
        if not tickers:
            return {}
        upper = [t.upper().strip() for t in tickers]
        placeholders = ", ".join("?" for _ in upper)
        try:
            with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    f"SELECT * FROM ticker_profiles WHERE ticker IN ({placeholders})",
                    upper
                )
                rows = cur.fetchall()
            return {row["ticker"]: _row_to_dict(row) for row in rows}
        except Exception as e:
            logger.error(f"get_batch error: {e}")
            return {}

    def get_universe_profiles(self) -> List[Dict[str, Any]]:
        """Return all stored profiles."""
        try:
            with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("SELECT * FROM ticker_profiles")
                return [_row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"get_universe_profiles error: {e}")
            return []

    def get_by_criteria(
        self,
        max_float_m: Optional[float] = None,     # float en millions max
        min_short_pct: Optional[float] = None,
        min_borrow_rate: Optional[float] = None,
        atm_active: Optional[bool] = None,
        max_reverse_splits: Optional[int] = None,
        min_data_quality: float = 0.3,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Filtered query — used by weekend batch and dashboard."""
        where = ["data_quality >= ?"]
        params: list = [min_data_quality]

        if max_float_m is not None:
            where.append("float_shares <= ?")
            params.append(max_float_m * 1_000_000)
        if min_short_pct is not None:
            where.append("short_interest_pct >= ?")
            params.append(min_short_pct)
        if min_borrow_rate is not None:
            where.append("borrow_rate >= ?")
            params.append(min_borrow_rate)
        if atm_active is not None:
            where.append("atm_active = ?")
            params.append(1 if atm_active else 0)
        if max_reverse_splits is not None:
            where.append("reverse_split_count <= ?")
            params.append(max_reverse_splits)

        params.append(limit)
        sql = (
            "SELECT * FROM ticker_profiles WHERE "
            + " AND ".join(where)
            + " ORDER BY data_quality DESC, short_interest_pct DESC LIMIT ?"
        )
        try:
            with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(sql, params)
                return [_row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"get_by_criteria error: {e}")
            return []

    def count(self) -> int:
        """Return number of stored profiles."""
        try:
            with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM ticker_profiles")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def needs_update(self, ticker: str, max_age_days: int = 7) -> bool:
        """Return True if profile is missing or older than max_age_days."""
        profile = self.get(ticker)
        if not profile or not profile.get("updated_at"):
            return True
        try:
            updated = datetime.fromisoformat(profile["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated).days
            return age >= max_age_days
        except Exception:
            return True


# ============================================================================
# Helpers
# ============================================================================

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # Deserialize source_flags JSON string
    if isinstance(d.get("source_flags"), str):
        try:
            d["source_flags"] = json.loads(d["source_flags"])
        except Exception:
            pass
    return d


# ============================================================================
# Singleton
# ============================================================================

_store_instance: Optional[TickerProfileStore] = None
_store_lock = threading.Lock()


def get_ticker_profile_store() -> TickerProfileStore:
    """Return thread-safe singleton TickerProfileStore."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = TickerProfileStore()
    return _store_instance
