"""
SIGNAL LOGGER - Persistent Signal Storage
==========================================

Logs tous les signaux BUY/BUY_STRONG avec timestamp précis
pour analyse historique et audit lead time.

Base de données SQLite simple et robuste.
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from utils.logger import get_logger

logger = get_logger("SIGNAL_LOGGER")

DB_PATH = "data/signals_history.db"


# ============================
# Database initialization
# ============================

def init_db():
    """Create signals table if not exists"""
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # S2-5 FIX: WAL mode — allows concurrent reads while a write is in progress.
    # Default journal mode (DELETE) blocks all readers during writes.
    # WAL is safe for single-process multi-thread use (main + Streamlit dashboard).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # Slightly faster, still crash-safe with WAL
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            monster_score REAL,
            confidence REAL,
            pm_gap REAL,
            event_impact REAL,
            pattern_score REAL,
            pm_transition_score REAL,
            volume_spike REAL,
            momentum REAL,
            entry_price REAL,
            stop_loss REAL,
            shares INTEGER,
            metadata TEXT
        )
    """)
    
    # Index pour performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
        ON signals(ticker, timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON signals(timestamp)
    """)
    
    conn.commit()
    conn.close()
    
    logger.info("Signal database initialized")


# ============================
# Log signal
# ============================

def log_signal(signal_data):
    """
    Log a signal to database
    
    Args:
        signal_data: dict with signal information
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signals (
                timestamp, ticker, signal_type, monster_score,
                confidence, pm_gap, event_impact, pattern_score,
                pm_transition_score, volume_spike, momentum,
                entry_price, stop_loss, shares, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            signal_data.get("ticker", ""),
            signal_data.get("signal", ""),
            signal_data.get("monster_score", 0),
            signal_data.get("confidence", 0),
            signal_data.get("pm_gap", 0),
            signal_data.get("event_impact", 0),
            signal_data.get("pattern_score", 0),
            signal_data.get("pm_transition_score", 0),
            signal_data.get("volume_spike", 0),
            signal_data.get("momentum", 0),
            signal_data.get("entry", 0),
            signal_data.get("stop", 0),
            signal_data.get("shares", 0),
            json.dumps(signal_data.get("metadata", {}))
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged signal: {signal_data.get('signal')} {signal_data.get('ticker')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to log signal: {e}", exc_info=True)
        return False


# ============================
# Query signals
# ============================

def get_signals_for_period(start_date, end_date, signal_type=None):
    """
    Get signals for a date range
    
    Args:
        start_date: datetime or ISO string
        end_date: datetime or ISO string
        signal_type: filter by BUY/BUY_STRONG (optional)
    
    Returns:
        DataFrame with signals
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        
        query = """
            SELECT * FROM signals 
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
        
        df = pd.read_sql(query, conn, params=params)
        
        conn.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to query signals: {e}")
        return pd.DataFrame()


def get_signal_by_ticker_and_date(ticker, date):
    """
    Get earliest signal for a ticker on a specific date
    
    Args:
        ticker: stock ticker
        date: datetime or ISO string
    
    Returns:
        dict with signal data or None
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date[:10]  # Extract date part
        
        cursor.execute("""
            SELECT * FROM signals 
            WHERE ticker = ? 
            AND date(timestamp) = ?
            ORDER BY timestamp ASC
            LIMIT 1
        """, (ticker, date_str))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [
                "id", "timestamp", "ticker", "signal_type", "monster_score",
                "confidence", "pm_gap", "event_impact", "pattern_score",
                "pm_transition_score", "volume_spike", "momentum",
                "entry_price", "stop_loss", "shares", "metadata"
            ]
            return dict(zip(columns, row))
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get signal: {e}")
        return None


def get_all_signals_for_ticker(ticker, limit=100):
    """Get all signals for a ticker (most recent first)"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        df = pd.read_sql("""
            SELECT * FROM signals 
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=[ticker, limit])
        
        conn.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to get ticker signals: {e}")
        return pd.DataFrame()


# ============================
# Statistics
# ============================

def get_signal_stats(days_back=30):
    """Get signal statistics for recent period"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=days_back)).isoformat()
        
        stats = {}
        
        # Total signals
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM signals 
            WHERE timestamp >= ?
        """, (start_date,))
        stats["total_signals"] = cursor.fetchone()[0]
        
        # By type
        cursor.execute("""
            SELECT signal_type, COUNT(*) 
            FROM signals 
            WHERE timestamp >= ?
            GROUP BY signal_type
        """, (start_date,))
        stats["by_type"] = dict(cursor.fetchall())
        
        # Average scores
        cursor.execute("""
            SELECT 
                AVG(monster_score) as avg_monster_score,
                AVG(confidence) as avg_confidence,
                AVG(pattern_score) as avg_pattern_score
            FROM signals 
            WHERE timestamp >= ?
        """, (start_date,))
        row = cursor.fetchone()
        stats["avg_scores"] = {
            "monster_score": row[0] or 0,
            "confidence": row[1] or 0,
            "pattern_score": row[2] or 0
        }
        
        conn.close()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {}


# ============================
# Signal History (for weight_optimizer)
# ============================

def get_signal_history(days_back: int = 30, signal_types: list = None) -> pd.DataFrame:
    """
    Get signal history for a period.

    Required by weight_optimizer.py for weekly weight optimization.

    Args:
        days_back: How many days back to retrieve
        signal_types: Filter by signal types (e.g. ["BUY", "BUY_STRONG"])

    Returns:
        DataFrame with columns: timestamp, ticker, signal_type, monster_score,
        confidence, entry_price, stop_loss, shares, metadata
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)

        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

        query = "SELECT * FROM signals WHERE timestamp >= ?"
        params = [start_date]

        if signal_types:
            placeholders = ", ".join("?" * len(signal_types))
            query += f" AND signal_type IN ({placeholders})"
            params.extend(signal_types)

        query += " ORDER BY timestamp DESC"

        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df

    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        return pd.DataFrame()


# ============================
# Cleanup
# ============================

def cleanup_old_signals(days_to_keep=90):
    """Remove signals older than X days"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now(timezone.utc) - pd.Timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute("""
            DELETE FROM signals 
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted} old signals")
        
        return deleted
        
    except Exception as e:
        logger.error(f"Failed to cleanup signals: {e}")
        return 0


# ============================
# Initialize on import
# ============================

init_db()


if __name__ == "__main__":
    # Test
    test_signal = {
        "ticker": "AAPL",
        "signal": "BUY_STRONG",
        "monster_score": 0.85,
        "confidence": 0.9,
        "pm_gap": 0.05,
        "event_impact": 0.8,
        "pattern_score": 0.7,
        "entry": 150.0,
        "stop": 147.0,
        "shares": 100
    }
    
    log_signal(test_signal)
    
    stats = get_signal_stats(days_back=7)
    print("Stats:", stats)
