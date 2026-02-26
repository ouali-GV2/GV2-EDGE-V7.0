# ============================
# REPEAT GAINER MEMORY
# ============================
# Tracks historical top gainers and calculates repeat probability scores
# Part of GV2-EDGE V6 - Repeat Gainer Layer
#
# Concept: Small caps that have spiked before are more likely to spike again.
# Factors: historical spike count, amplitude, recency, float characteristics

import sqlite3
import math
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

from utils.logger import get_logger
from utils.market_calendar import is_trading_day, get_previous_trading_day

logger = get_logger("REPEAT_GAINER")

# Database path
DB_PATH = "data/repeat_gainers.db"


# ============================
# DATA STRUCTURES
# ============================

@dataclass
class GainerRecord:
    """Record of a single top gainer event"""
    ticker: str
    date: date
    gain_pct: float           # Percentage gain (e.g., 50.0 for +50%)
    volume: int               # Volume on that day
    market_cap: float         # Market cap at time of spike
    float_shares: Optional[float] = None  # Float if available
    catalyst: Optional[str] = None        # Event type if known
    peak_gain_pct: Optional[float] = None # Intraday high gain


@dataclass
class RepeatGainerScore:
    """Calculated repeat gainer score for a ticker"""
    ticker: str
    score: float              # 0-1 normalized score
    spike_count: int          # Number of historical spikes
    avg_amplitude: float      # Average spike amplitude
    max_amplitude: float      # Largest spike
    last_spike_days: int      # Days since last spike
    recency_factor: float     # Decay-adjusted recency (0-1)
    is_repeat_runner: bool    # True if score > threshold


# ============================
# DATABASE INITIALIZATION
# ============================

def init_db():
    """Initialize SQLite database for repeat gainer tracking"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Create gainers history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gainer_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            gain_pct REAL NOT NULL,
            volume INTEGER,
            market_cap REAL,
            float_shares REAL,
            catalyst TEXT,
            peak_gain_pct REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    ''')

    # Create index for fast lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_gainer_ticker ON gainer_history(ticker)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_gainer_date ON gainer_history(date)
    ''')

    # Create daily top gainers snapshot table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_top_gainers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            rank INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            gain_pct REAL NOT NULL,
            volume INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, rank)
        )
    ''')

    conn.commit()
    conn.close()

    logger.info(f"Repeat Gainer database initialized: {DB_PATH}")


# ============================
# RECORD MANAGEMENT
# ============================

def record_gainer(record: GainerRecord) -> bool:
    """
    Record a top gainer event

    Args:
        record: GainerRecord with spike data

    Returns:
        True if recorded successfully
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO gainer_history
            (ticker, date, gain_pct, volume, market_cap, float_shares, catalyst, peak_gain_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.ticker,
            record.date.isoformat(),
            record.gain_pct,
            record.volume,
            record.market_cap,
            record.float_shares,
            record.catalyst,
            record.peak_gain_pct
        ))

        conn.commit()
        conn.close()

        logger.info(f"Recorded gainer: {record.ticker} +{record.gain_pct:.1f}% on {record.date}")
        return True

    except Exception as e:
        logger.error(f"Failed to record gainer {record.ticker}: {e}")
        return False


def record_daily_top_gainers(date: date, gainers: List[Tuple[str, float, int]]) -> bool:
    """
    Record daily top gainers snapshot

    Args:
        date: Trading date
        gainers: List of (ticker, gain_pct, volume) tuples, ranked

    Returns:
        True if recorded successfully
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        for rank, (ticker, gain_pct, volume) in enumerate(gainers, 1):
            cursor.execute('''
                INSERT OR REPLACE INTO daily_top_gainers
                (date, rank, ticker, gain_pct, volume)
                VALUES (?, ?, ?, ?, ?)
            ''', (date.isoformat(), rank, ticker, gain_pct, volume))

        conn.commit()
        conn.close()

        logger.info(f"Recorded {len(gainers)} top gainers for {date}")
        return True

    except Exception as e:
        logger.error(f"Failed to record daily gainers for {date}: {e}")
        return False


def get_ticker_history(ticker: str, days_back: int = 180) -> List[GainerRecord]:
    """
    Get spike history for a ticker

    Args:
        ticker: Stock symbol
        days_back: How far back to look

    Returns:
        List of GainerRecords
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        cutoff = (datetime.now().date() - timedelta(days=days_back)).isoformat()

        cursor.execute('''
            SELECT ticker, date, gain_pct, volume, market_cap, float_shares, catalyst, peak_gain_pct
            FROM gainer_history
            WHERE ticker = ? AND date >= ?
            ORDER BY date DESC
        ''', (ticker, cutoff))

        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append(GainerRecord(
                ticker=row[0],
                date=date.fromisoformat(row[1]),
                gain_pct=row[2],
                volume=row[3],
                market_cap=row[4],
                float_shares=row[5],
                catalyst=row[6],
                peak_gain_pct=row[7]
            ))

        return records

    except Exception as e:
        logger.error(f"Failed to get history for {ticker}: {e}")
        return []


# ============================
# REPEAT GAINER SCORING
# ============================

# Configuration
DECAY_HALF_LIFE_DAYS = 30    # Recency decay half-life
MIN_SPIKE_PCT = 20.0          # Minimum gain to count as spike
REPEAT_RUNNER_THRESHOLD = 0.5 # Score threshold for "repeat runner"
LOOKBACK_DAYS = 180           # How far back to look


def calculate_recency_decay(days_since: int, half_life: int = DECAY_HALF_LIFE_DAYS) -> float:
    """
    Calculate exponential decay factor based on recency

    Args:
        days_since: Days since the event
        half_life: Half-life in days

    Returns:
        Decay factor (0-1), where 1 is most recent
    """
    return math.exp(-0.693 * days_since / half_life)  # ln(2) ≈ 0.693


def calculate_repeat_score(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> RepeatGainerScore:
    """
    Calculate repeat gainer score for a ticker

    Score formula:
    score = (spike_count_weighted × avg_amplitude_factor × recency_factor) / normalization

    Args:
        ticker: Stock symbol
        lookback_days: How far back to look

    Returns:
        RepeatGainerScore with all metrics
    """
    history = get_ticker_history(ticker, lookback_days)

    # Filter for significant spikes only
    spikes = [r for r in history if r.gain_pct >= MIN_SPIKE_PCT]

    if not spikes:
        return RepeatGainerScore(
            ticker=ticker,
            score=0.0,
            spike_count=0,
            avg_amplitude=0.0,
            max_amplitude=0.0,
            last_spike_days=999,
            recency_factor=0.0,
            is_repeat_runner=False
        )

    today = datetime.now().date()

    # Calculate metrics
    spike_count = len(spikes)
    amplitudes = [s.gain_pct for s in spikes]
    avg_amplitude = sum(amplitudes) / len(amplitudes)
    max_amplitude = max(amplitudes)

    # Days since last spike
    last_spike_date = max(s.date for s in spikes)
    last_spike_days = (today - last_spike_date).days

    # Calculate weighted spike count with recency decay
    weighted_count = 0.0
    for spike in spikes:
        days_since = (today - spike.date).days
        weight = calculate_recency_decay(days_since)
        # Also weight by amplitude (bigger spikes = more significant)
        amplitude_weight = min(spike.gain_pct / 100.0, 2.0)  # Cap at 200%
        weighted_count += weight * amplitude_weight

    # Recency factor (0-1) based on last spike
    recency_factor = calculate_recency_decay(last_spike_days)

    # Normalize components
    # Spike count factor: more spikes = higher probability (log scale)
    count_factor = math.log1p(weighted_count) / math.log1p(10)  # Normalize to ~10 spikes
    count_factor = min(count_factor, 1.0)

    # Amplitude factor: bigger average spikes = more explosive ticker
    amplitude_factor = min(avg_amplitude / 100.0, 1.0)  # Cap at 100%

    # Combined score (0-1)
    # Formula: weighted average of factors
    score = (
        count_factor * 0.4 +        # 40% spike frequency
        amplitude_factor * 0.3 +    # 30% spike size
        recency_factor * 0.3        # 30% recency
    )

    # Boost for very recent spikes (within 7 days)
    if last_spike_days <= 7:
        score *= 1.2

    # Clamp to 0-1
    score = max(0.0, min(1.0, score))

    return RepeatGainerScore(
        ticker=ticker,
        score=score,
        spike_count=spike_count,
        avg_amplitude=avg_amplitude,
        max_amplitude=max_amplitude,
        last_spike_days=last_spike_days,
        recency_factor=recency_factor,
        is_repeat_runner=score >= REPEAT_RUNNER_THRESHOLD
    )


def get_repeat_runners(min_score: float = REPEAT_RUNNER_THRESHOLD) -> List[RepeatGainerScore]:
    """
    Get all tickers that qualify as repeat runners

    Args:
        min_score: Minimum score threshold

    Returns:
        List of RepeatGainerScores for qualifying tickers
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        # Get unique tickers with recent activity
        cutoff = (datetime.now().date() - timedelta(days=LOOKBACK_DAYS)).isoformat()

        cursor.execute('''
            SELECT DISTINCT ticker FROM gainer_history
            WHERE date >= ? AND gain_pct >= ?
        ''', (cutoff, MIN_SPIKE_PCT))

        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Calculate scores for each
        runners = []
        for ticker in tickers:
            score = calculate_repeat_score(ticker)
            if score.score >= min_score:
                runners.append(score)

        # Sort by score descending
        runners.sort(key=lambda x: x.score, reverse=True)

        return runners

    except Exception as e:
        logger.error(f"Failed to get repeat runners: {e}")
        return []


# ============================
# BATCH SCORING
# ============================

def get_repeat_scores_batch(tickers: List[str]) -> Dict[str, RepeatGainerScore]:
    """
    Get repeat gainer scores for a batch of tickers

    Args:
        tickers: List of stock symbols

    Returns:
        Dict mapping ticker to RepeatGainerScore
    """
    scores = {}

    for ticker in tickers:
        scores[ticker] = calculate_repeat_score(ticker)

    return scores


def get_repeat_score_boost(ticker: str) -> float:
    """
    Get repeat gainer boost factor for Monster Score

    Used as a multiplier in catalyst scoring.

    Args:
        ticker: Stock symbol

    Returns:
        Boost factor (1.0 = no boost, up to 1.5 for strong repeat runners)
    """
    score = calculate_repeat_score(ticker)

    if not score.is_repeat_runner:
        return 1.0

    # Linear boost from 1.0 to 1.5 based on score
    # score 0.5 = 1.0x, score 1.0 = 1.5x
    boost = 1.0 + (score.score - REPEAT_RUNNER_THRESHOLD) * 1.0

    return min(boost, 1.5)


# ============================
# DATA COLLECTION HELPERS
# ============================

def fetch_and_record_top_gainers():
    """
    Fetch today's top gainers and record them.

    S3-7 FIX: Was a stub returning False. Now wired to TopGainersSource (C8 module)
    which uses IBKR Scanner + Yahoo Finance as sources.
    Called daily after market close (batch_processor.py or weekend_scheduler.py).

    Returns:
        True if at least one gainer was recorded
    """
    import asyncio
    import concurrent.futures

    try:
        from src.top_gainers_source import get_top_gainers_source

        logger.info("Fetching daily top gainers via TopGainersSource...")

        source = get_top_gainers_source()

        # fetch_top_gainers() is async — run it in a dedicated thread so this
        # synchronous function works regardless of whether an event loop is running.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            gainers = _ex.submit(
                asyncio.run,
                source.fetch_top_gainers(min_change_pct=10.0, min_volume=50_000)
            ).result(timeout=30)

        if not gainers:
            logger.warning("TopGainersSource returned no gainers (market closed or API unavailable)")
            return False

        today = date.today()
        recorded = 0

        for g in gainers:
            rec = GainerRecord(
                ticker=g.ticker,
                date=today,
                gain_pct=g.change_pct,
                volume=g.volume,
                market_cap=getattr(g, "market_cap", 0) or 0,
                catalyst=getattr(g, "catalyst_type", None),
            )
            if record_gainer(rec):
                recorded += 1

        logger.info(f"Recorded {recorded}/{len(gainers)} top gainers for {today}")
        return recorded > 0

    except Exception as e:
        logger.error(f"Failed to fetch top gainers: {e}")
        return False


def import_historical_gainers(csv_path: str) -> int:
    """
    Import historical gainers from CSV file

    CSV format: date,ticker,gain_pct,volume,market_cap

    Args:
        csv_path: Path to CSV file

    Returns:
        Number of records imported
    """
    import csv

    try:
        count = 0

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                record = GainerRecord(
                    ticker=row['ticker'],
                    date=date.fromisoformat(row['date']),
                    gain_pct=float(row['gain_pct']),
                    volume=int(row.get('volume', 0)),
                    market_cap=float(row.get('market_cap', 0)),
                    float_shares=float(row['float_shares']) if row.get('float_shares') else None,
                    catalyst=row.get('catalyst')
                )

                if record_gainer(record):
                    count += 1

        logger.info(f"Imported {count} historical gainer records from {csv_path}")
        return count

    except Exception as e:
        logger.error(f"Failed to import from {csv_path}: {e}")
        return 0


# ============================
# STATISTICS
# ============================

def get_database_stats() -> Dict:
    """Get statistics about the repeat gainer database"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        # Total records
        cursor.execute('SELECT COUNT(*) FROM gainer_history')
        total_records = cursor.fetchone()[0]

        # Unique tickers
        cursor.execute('SELECT COUNT(DISTINCT ticker) FROM gainer_history')
        unique_tickers = cursor.fetchone()[0]

        # Date range
        cursor.execute('SELECT MIN(date), MAX(date) FROM gainer_history')
        date_range = cursor.fetchone()

        # Recent activity (last 30 days)
        cutoff = (datetime.now().date() - timedelta(days=30)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM gainer_history WHERE date >= ?', (cutoff,))
        recent_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_records": total_records,
            "unique_tickers": unique_tickers,
            "date_range": {
                "start": date_range[0],
                "end": date_range[1]
            },
            "recent_30d_count": recent_count
        }

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


# ============================
# INITIALIZATION
# ============================

# Initialize database on module import
init_db()


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("\n🔁 REPEAT GAINER MEMORY TEST")
    print("=" * 50)

    # Show database stats
    stats = get_database_stats()
    print(f"\n📊 Database Stats:")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Unique tickers: {stats.get('unique_tickers', 0)}")
    print(f"  Date range: {stats.get('date_range', {})}")
    print(f"  Recent (30d): {stats.get('recent_30d_count', 0)}")

    # Test with sample data
    print("\n📝 Recording sample data...")

    sample_records = [
        GainerRecord("MULN", date(2024, 1, 15), 85.0, 50000000, 100000000, catalyst="EARNINGS_BEAT"),
        GainerRecord("MULN", date(2024, 2, 20), 120.0, 80000000, 150000000, catalyst="M&A_RUMOR"),
        GainerRecord("MULN", date(2024, 3, 10), 45.0, 30000000, 120000000),
        GainerRecord("FFIE", date(2024, 2, 1), 200.0, 100000000, 50000000, catalyst="SHORT_SQUEEZE"),
        GainerRecord("FFIE", date(2024, 3, 5), 150.0, 90000000, 80000000),
        GainerRecord("AAPL", date(2024, 3, 1), 5.0, 50000000, 2500000000000),  # Not a spike (< 20%)
    ]

    for record in sample_records:
        record_gainer(record)

    # Calculate scores
    print("\n📈 Repeat Gainer Scores:")

    for ticker in ["MULN", "FFIE", "AAPL", "NVDA"]:
        score = calculate_repeat_score(ticker)
        status = "🔥 REPEAT RUNNER" if score.is_repeat_runner else ""
        print(f"\n  {ticker}:")
        print(f"    Score: {score.score:.3f} {status}")
        print(f"    Spikes: {score.spike_count}")
        print(f"    Avg amplitude: +{score.avg_amplitude:.1f}%")
        print(f"    Max amplitude: +{score.max_amplitude:.1f}%")
        print(f"    Last spike: {score.last_spike_days} days ago")
        print(f"    Recency factor: {score.recency_factor:.3f}")
        print(f"    Boost factor: {get_repeat_score_boost(ticker):.2f}x")

    # Get all repeat runners
    print("\n🏃 All Repeat Runners:")
    runners = get_repeat_runners()
    for runner in runners[:10]:
        print(f"  {runner.ticker}: {runner.score:.3f} ({runner.spike_count} spikes, avg +{runner.avg_amplitude:.0f}%)")

    print("\n✅ Test complete!")
