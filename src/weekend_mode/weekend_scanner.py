"""
Weekend Scanner - Batch Analysis for Monday Preparation

Runs comprehensive analysis during market-closed hours:
- Full universe screening (not just hot tickers)
- Deep fundamental analysis
- Technical pattern detection
- Sector rotation analysis
- Earnings calendar preparation
- SEC filing batch processing

Designed to run Saturday/Sunday when API limits aren't critical.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
import asyncio
import logging
import concurrent.futures

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of weekend scans."""
    FULL_UNIVERSE = "FULL_UNIVERSE"       # All tradeable tickers
    SECTOR_ROTATION = "SECTOR_ROTATION"   # Sector strength analysis
    EARNINGS_PREP = "EARNINGS_PREP"       # Upcoming earnings
    SEC_FILINGS = "SEC_FILINGS"           # Recent SEC filings
    TECHNICAL_PATTERNS = "TECHNICAL"      # Chart patterns
    MOMENTUM_SCREEN = "MOMENTUM"          # Momentum indicators
    VALUE_SCREEN = "VALUE"                # Value metrics
    INSIDER_ACTIVITY = "INSIDER"          # Form 4 filings
    SHORT_INTEREST = "SHORT_INTEREST"     # Short squeeze candidates
    OPTIONS_FLOW = "OPTIONS_FLOW"         # Unusual options activity
    NEWS_SENTIMENT = "NEWS_SENTIMENT"     # Weekend news analysis
    TICKER_PROFILES = "TICKER_PROFILES"   # Strategic profile update (float, short, dilution)
    CUSTOM = "CUSTOM"                     # User-defined scan


class ScanPriority(Enum):
    """Scan execution priority."""
    CRITICAL = 1    # Must complete before Monday
    HIGH = 2        # Important for Monday prep
    STANDARD = 3    # Nice to have
    LOW = 4         # Background/optional


class ScanStatus(Enum):
    """Status of a scan job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


@dataclass
class ScanResult:
    """Result from a single ticker scan."""
    ticker: str
    scan_type: ScanType
    timestamp: datetime = field(default_factory=datetime.now)

    # Scores (0-100)
    overall_score: float = 0.0
    momentum_score: float = 0.0
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0

    # Flags
    is_candidate: bool = False
    has_catalyst: bool = False
    has_risk_flags: bool = False

    # Details
    signals: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)

    # Raw data for further processing
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "scan_type": self.scan_type.value,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "momentum_score": self.momentum_score,
            "technical_score": self.technical_score,
            "fundamental_score": self.fundamental_score,
            "sentiment_score": self.sentiment_score,
            "is_candidate": self.is_candidate,
            "has_catalyst": self.has_catalyst,
            "signals": self.signals,
            "risk_flags": self.risk_flags,
            "catalysts": self.catalysts,
        }


@dataclass
class ScanJob:
    """A scan job to be executed."""
    id: str
    scan_type: ScanType
    priority: ScanPriority
    created_at: datetime = field(default_factory=datetime.now)

    # Target
    tickers: List[str] = field(default_factory=list)
    universe: str = ""  # "SP500", "NASDAQ100", "ALL", etc.

    # Status
    status: ScanStatus = ScanStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress
    total_tickers: int = 0
    processed_tickers: int = 0
    failed_tickers: int = 0

    # Results
    results: List[ScanResult] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)

    # Config
    config: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_tickers == 0:
            return 0.0
        return (self.processed_tickers / self.total_tickers) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class ScannerConfig:
    """Configuration for weekend scanner."""
    # Concurrency
    max_concurrent_scans: int = 3
    batch_size: int = 50
    rate_limit_delay: float = 0.1  # seconds between API calls

    # Thresholds for candidates
    min_overall_score: float = 60.0
    min_momentum_score: float = 50.0
    min_volume_ratio: float = 1.5  # vs average

    # Universe defaults
    default_universe: str = "TRADEABLE"
    exclude_otc: bool = True
    min_price: float = 1.0
    max_price: float = 500.0
    min_avg_volume: int = 100_000

    # Timeouts
    ticker_timeout: float = 30.0  # seconds per ticker
    job_timeout: float = 3600.0   # seconds per job (1 hour)

    # Retry
    max_retries: int = 3
    retry_delay: float = 5.0


class WeekendScanner:
    """
    Comprehensive scanner for weekend batch analysis.

    Usage:
        scanner = WeekendScanner()

        # Queue scans
        job_id = scanner.queue_scan(
            scan_type=ScanType.MOMENTUM_SCREEN,
            universe="SP500",
            priority=ScanPriority.HIGH
        )

        # Run all queued scans
        await scanner.run_all()

        # Get results
        candidates = scanner.get_candidates()
    """

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()

        # Job queue
        self._jobs: Dict[str, ScanJob] = {}
        self._job_queue: List[str] = []

        # Results cache
        self._results: Dict[str, List[ScanResult]] = {}  # ticker -> results

        # Scan handlers
        self._handlers: Dict[ScanType, Callable] = {}

        # State
        self._running = False
        self._current_job: Optional[str] = None

        # Statistics
        self._stats = {
            "total_scans": 0,
            "successful_scans": 0,
            "failed_scans": 0,
            "total_tickers_scanned": 0,
            "candidates_found": 0,
        }

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default scan handlers."""
        self._handlers[ScanType.MOMENTUM_SCREEN] = self._scan_momentum
        self._handlers[ScanType.TECHNICAL_PATTERNS] = self._scan_technical
        self._handlers[ScanType.EARNINGS_PREP] = self._scan_earnings
        self._handlers[ScanType.SEC_FILINGS] = self._scan_sec_filings
        self._handlers[ScanType.SHORT_INTEREST] = self._scan_short_interest
        self._handlers[ScanType.SECTOR_ROTATION] = self._scan_sector_rotation
        self._handlers[ScanType.NEWS_SENTIMENT] = self._scan_news_sentiment
        self._handlers[ScanType.TICKER_PROFILES] = self._scan_ticker_profiles

    def register_handler(
        self,
        scan_type: ScanType,
        handler: Callable
    ) -> None:
        """Register a custom scan handler."""
        self._handlers[scan_type] = handler

    def queue_scan(
        self,
        scan_type: ScanType,
        tickers: Optional[List[str]] = None,
        universe: str = "",
        priority: ScanPriority = ScanPriority.STANDARD,
        config: Optional[Dict] = None
    ) -> str:
        """
        Queue a scan job.

        Args:
            scan_type: Type of scan to run
            tickers: Specific tickers (if None, uses universe)
            universe: Universe to scan (SP500, NASDAQ100, ALL, etc.)
            priority: Execution priority
            config: Additional scan configuration

        Returns:
            Job ID
        """
        import uuid
        job_id = f"scan_{uuid.uuid4().hex[:8]}"

        job = ScanJob(
            id=job_id,
            scan_type=scan_type,
            priority=priority,
            tickers=tickers or [],
            universe=universe or self.config.default_universe,
            config=config or {}
        )

        self._jobs[job_id] = job
        self._job_queue.append(job_id)

        # Sort queue by priority
        self._job_queue.sort(
            key=lambda x: self._jobs[x].priority.value
        )

        logger.info(f"Queued scan job: {job_id} ({scan_type.value})")

        return job_id

    async def run_all(self) -> Dict[str, ScanJob]:
        """Run all queued scans."""
        self._running = True
        completed_jobs = {}

        try:
            while self._job_queue and self._running:
                job_id = self._job_queue.pop(0)
                job = self._jobs.get(job_id)

                if not job:
                    continue

                self._current_job = job_id

                try:
                    await self._run_job(job)
                    completed_jobs[job_id] = job
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}")
                    job.status = ScanStatus.FAILED
                    job.errors.append(str(e))
                    completed_jobs[job_id] = job

                self._current_job = None

        finally:
            self._running = False

        return completed_jobs

    async def run_job(self, job_id: str) -> Optional[ScanJob]:
        """Run a specific job."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        await self._run_job(job)
        return job

    async def _run_job(self, job: ScanJob) -> None:
        """Execute a scan job."""
        job.status = ScanStatus.RUNNING
        job.started_at = datetime.now()

        logger.info(f"Starting scan job: {job.id} ({job.scan_type.value})")

        # Get tickers to scan
        tickers = job.tickers
        if not tickers:
            tickers = await self._get_universe_tickers(job.universe)

        job.total_tickers = len(tickers)

        # Get handler
        handler = self._handlers.get(job.scan_type)
        if not handler:
            raise ValueError(f"No handler for scan type: {job.scan_type}")

        # Process in batches
        for i in range(0, len(tickers), self.config.batch_size):
            if not self._running:
                job.status = ScanStatus.CANCELLED
                break

            batch = tickers[i:i + self.config.batch_size]

            try:
                results = await self._process_batch(batch, handler, job)
                job.results.extend(results)

                # Track candidates
                for result in results:
                    if result.is_candidate:
                        job.candidates.append(result.ticker)

                job.processed_tickers += len(batch)

            except Exception as e:
                logger.error(f"Batch failed: {e}")
                job.failed_tickers += len(batch)
                job.errors.append(f"Batch {i}: {e}")

            # Rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)

        # Complete
        job.completed_at = datetime.now()
        if job.status == ScanStatus.RUNNING:
            job.status = ScanStatus.COMPLETED

        # Update stats
        self._stats["total_scans"] += 1
        if job.status == ScanStatus.COMPLETED:
            self._stats["successful_scans"] += 1
        else:
            self._stats["failed_scans"] += 1
        self._stats["total_tickers_scanned"] += job.processed_tickers
        self._stats["candidates_found"] += len(job.candidates)

        # Cache results by ticker
        for result in job.results:
            if result.ticker not in self._results:
                self._results[result.ticker] = []
            self._results[result.ticker].append(result)

        logger.info(
            f"Completed scan job: {job.id} - "
            f"{job.processed_tickers}/{job.total_tickers} tickers, "
            f"{len(job.candidates)} candidates"
        )

    async def _process_batch(
        self,
        tickers: List[str],
        handler: Callable,
        job: ScanJob
    ) -> List[ScanResult]:
        """Process a batch of tickers."""
        tasks = [
            asyncio.wait_for(
                handler(ticker, job.config),
                timeout=self.config.ticker_timeout
            )
            for ticker in tickers
        ]

        results = []
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, result in zip(tickers, completed):
            if isinstance(result, Exception):
                logger.debug(f"Scan failed for {ticker}: {result}")
                continue
            if result:
                results.append(result)

        return results

    async def _get_universe_tickers(self, universe: str) -> List[str]:
        """Get real tickers from universe_loader, filtered by universe type."""
        try:
            from src.universe_loader import load_universe
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, load_universe)
            if df is None or df.empty:
                return []
            # Apply market-cap filter for known sub-universes
            if universe in ("SP500", "NASDAQ100") and "market_cap" in df.columns:
                sub = df[df["market_cap"] >= 10_000_000_000]["ticker"].dropna().tolist()
                return sub[:500] if sub else df["ticker"].dropna().tolist()[:500]
            if universe in ("RUSSELL2000", "SMALLCAP") and "market_cap" in df.columns:
                sub = df[
                    (df["market_cap"] >= 100_000_000) &
                    (df["market_cap"] <= 2_000_000_000)
                ]["ticker"].dropna().tolist()
                return sub[:2000] if sub else df["ticker"].dropna().tolist()[:2000]
            # ALL / TRADEABLE / default: full universe
            return df["ticker"].dropna().tolist()
        except Exception as e:
            logger.error(f"_get_universe_tickers({universe}): {e}")
            return []

    # ============================
    # Shared data helpers
    # ============================

    async def _fetch_daily_candles(self, ticker: str, days: int = 60) -> Optional[dict]:
        """
        Fetch daily OHLCV candles from Finnhub (available 24/7, even weekends).
        Returns dict with lists: o, h, l, c, v, t  — or None.
        """
        try:
            from utils.api_guard import pool_safe_get
            from config import FINNHUB_API_KEY
            now_ts = int(datetime.now(timezone.utc).timestamp())
            from_ts = now_ts - days * 86400
            url = "https://finnhub.io/api/v1/stock/candle"
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: pool_safe_get(
                    url,
                    params={
                        "symbol": ticker,
                        "resolution": "D",
                        "from": from_ts,
                        "to": now_ts,
                        "token": FINNHUB_API_KEY,
                    },
                    provider="finnhub",
                )
            )
            if not resp or resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("s") != "ok" or not data.get("c"):
                return None
            return data
        except Exception as e:
            logger.debug(f"Daily candles {ticker}: {e}")
            return None

    # Default scan handlers

    async def _scan_momentum(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """
        Momentum scan using Finnhub daily candles (works on weekends).
        Calculates: 5d return, 20d volume avg, volume spike, RSI-14 proxy.
        """
        result = ScanResult(ticker=ticker, scan_type=ScanType.MOMENTUM_SCREEN)
        try:
            data = await self._fetch_daily_candles(ticker, days=30)
            if not data:
                return result
            closes  = data["c"]
            volumes = data["v"]
            if len(closes) < 6:
                return result
            # 5-day return
            ret_5d = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] else 0
            # Volume spike: last day vs 20d avg
            avg_vol = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)
            vol_spike = (volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
            # Simple RSI-14 proxy
            gains  = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
            losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
            ag = sum(gains[-14:])  / 14 if len(gains) >= 14 else 0
            al = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
            rsi = 100 - (100 / (1 + ag / al)) if al > 0 else 50
            # Score 0-100
            ret_s  = min(40.0, max(0.0, ret_5d * 200))   # +20% 5d = 40pts
            vol_s  = min(40.0, (vol_spike - 1) * 20)     # 3x volume = 40pts
            rsi_s  = min(20.0, max(0.0, (rsi - 50) * 0.4)) if rsi > 50 else 0
            result.momentum_score = ret_s + vol_s + rsi_s
            result.overall_score  = result.momentum_score
            result.data = {"ret_5d": ret_5d, "vol_spike": vol_spike, "rsi": rsi}
            if ret_5d > 0.10:
                result.signals.append(f"RETURN_5D_{ret_5d*100:.1f}%")
            if vol_spike >= 2.0:
                result.signals.append(f"VOL_SPIKE_{vol_spike:.1f}x")
            if rsi > 65:
                result.signals.append(f"RSI_{rsi:.0f}")
            if result.momentum_score >= self.config.min_momentum_score:
                result.is_candidate = True
        except Exception as e:
            logger.debug(f"Momentum scan {ticker}: {e}")
        return result

    async def _scan_technical(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """
        Technical pattern scan using Finnhub daily candles.
        Detects: higher highs/lows, consolidation box, bollinger squeeze.
        """
        result = ScanResult(ticker=ticker, scan_type=ScanType.TECHNICAL_PATTERNS)
        try:
            data = await self._fetch_daily_candles(ticker, days=60)
            if not data:
                return result
            closes  = data["c"]
            highs   = data["h"]
            lows    = data["l"]
            volumes = data["v"]
            if len(closes) < 20:
                return result
            score = 0.0
            # Higher highs + higher lows (uptrend) — last 10 bars
            hh = sum(1 for i in range(1, 10) if highs[-i] > highs[-i-1])
            hl = sum(1 for i in range(1, 10) if lows[-i]  > lows[-i-1])
            if hh >= 6 and hl >= 5:
                score += 30
                result.signals.append("HIGHER_HIGHS_LOWS")
            # Tight consolidation last 5 days (low ATR / range)
            ranges = [(highs[-i] - lows[-i]) / closes[-i] for i in range(1, 6) if closes[-i]]
            avg_range = sum(ranges) / len(ranges) if ranges else 0
            if avg_range < 0.03:
                score += 25
                result.signals.append("TIGHT_CONSOLIDATION")
            # Bollinger squeeze: price inside ±1.5% of 20-day SMA
            sma20 = sum(closes[-20:]) / 20
            dev   = abs(closes[-1] - sma20) / sma20 if sma20 else 1
            if dev < 0.015:
                score += 20
                result.signals.append("BB_SQUEEZE")
            # Volume accumulation: last 5 days avg > 10d avg
            vol5  = sum(volumes[-5:])  / 5
            vol10 = sum(volumes[-10:]) / 10
            if vol10 > 0 and vol5 / vol10 > 1.3:
                score += 25
                result.signals.append("VOL_ACCUMULATION")
            result.technical_score = min(100.0, score)
            result.overall_score   = result.technical_score
            result.data = {"avg_range": avg_range, "dev_sma20": dev, "vol_ratio": vol5/vol10 if vol10 else 0}
            if result.technical_score >= 50:
                result.is_candidate = True
        except Exception as e:
            logger.debug(f"Technical scan {ticker}: {e}")
        return result

    async def _scan_earnings(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """Earnings calendar scan via Finnhub /stock/earnings API."""
        result = ScanResult(ticker=ticker, scan_type=ScanType.EARNINGS_PREP)
        try:
            from utils.api_guard import pool_safe_get
            from config import FINNHUB_API_KEY
            from datetime import datetime, timezone
            url = "https://finnhub.io/api/v1/stock/earnings"
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: pool_safe_get(
                    url,
                    params={"symbol": ticker, "limit": 4, "token": FINNHUB_API_KEY},
                    provider="finnhub",
                )
            )
            if not resp or resp.status_code != 200:
                return result
            earnings = resp.json() or []
            now = datetime.now(timezone.utc)
            for e in earnings:
                date_str = e.get("date") or e.get("period", "")
                if not date_str:
                    continue
                try:
                    edate = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                except Exception:
                    try:
                        edate = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    except Exception:
                        continue
                days_until = (edate - now).days
                if 0 <= days_until <= 14:
                    result.has_catalyst = True
                    result.fundamental_score = max(0, min(100, 100 - days_until * 5))
                    result.overall_score = result.fundamental_score
                    result.signals.append(f"EARNINGS_IN_{days_until}D")
                    result.catalysts.append(f"Earnings: {date_str}")
                    result.is_candidate = True
                    break
                elif 15 <= days_until <= 30:
                    result.fundamental_score = max(0, min(60, 60 - (days_until - 14) * 2))
                    result.overall_score = result.fundamental_score
                    result.signals.append(f"EARNINGS_IN_{days_until}D")
                    result.catalysts.append(f"Earnings: {date_str}")
                    break
        except Exception as e:
            logger.debug(f"Earnings scan {ticker}: {e}")
        return result

    async def _scan_sec_filings(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """Recent 8-K / Form 4 scan via SECIngestor (72h look-back)."""
        result = ScanResult(ticker=ticker, scan_type=ScanType.SEC_FILINGS)
        try:
            from src.ingestors.sec_filings_ingestor import SECIngestor
            import concurrent.futures
            loop = asyncio.get_event_loop()

            def _fetch():
                import asyncio as _aio
                with concurrent.futures.ThreadPoolExecutor(1) as ex:
                    return ex.submit(_aio.run, SECIngestor().fetch_all_recent(hours_back=72)).result(timeout=25)

            filings = await loop.run_in_executor(None, _fetch)
            if not filings:
                return result
            ticker_filings = [
                f for f in filings
                if (getattr(f, "ticker", "") or "").upper() == ticker.upper()
            ]
            if ticker_filings:
                result.has_catalyst = True
                result.fundamental_score = min(100.0, len(ticker_filings) * 30.0)
                result.overall_score = result.fundamental_score
                for f in ticker_filings[:3]:
                    ftype   = getattr(f, "filing_type", "N/A")
                    summary = (getattr(f, "summary", "") or "")[:80]
                    result.signals.append(f"SEC_{ftype}")
                    result.catalysts.append(f"{ftype}: {summary}")
                result.is_candidate = any(
                    getattr(f, "filing_type", "").startswith("8-K")
                    for f in ticker_filings
                )
        except Exception as e:
            logger.debug(f"SEC filings scan {ticker}: {e}")
        return result

    async def _scan_short_interest(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """Short squeeze potential via squeeze_boost (Finnhub short data)."""
        result = ScanResult(ticker=ticker, scan_type=ScanType.SHORT_INTEREST)
        try:
            from src.boosters.squeeze_boost import quick_squeeze_check
            loop = asyncio.get_event_loop()
            sr = await loop.run_in_executor(None, quick_squeeze_check, ticker)
            if not sr or not sr.has_data:
                return result
            result.data = {
                "short_float_pct": sr.short_float_pct,
                "days_to_cover":   sr.days_to_cover,
                "boost_score":     sr.boost_score,
                "squeeze_signal":  sr.squeeze_signal.value if sr.squeeze_signal else "NONE",
            }
            result.fundamental_score = min(100.0, float(sr.short_float_pct or 0) * 2)
            result.overall_score = result.fundamental_score
            if sr.short_float_pct and sr.short_float_pct >= 15:
                result.signals.append(f"SHORT_FLOAT_{sr.short_float_pct:.1f}%")
            if sr.days_to_cover and sr.days_to_cover >= 3:
                result.signals.append(f"DTC_{sr.days_to_cover:.1f}d")
            if sr.boost_score >= 0.5:
                result.is_candidate = True
                result.has_catalyst = True
                result.catalysts.append(f"Short squeeze: {sr.reason}")
        except Exception as e:
            logger.debug(f"Short interest scan {ticker}: {e}")
        return result

    async def _scan_sector_rotation(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """Sector/relative-strength scan via Finnhub basic financials metric."""
        result = ScanResult(ticker=ticker, scan_type=ScanType.SECTOR_ROTATION)
        try:
            from utils.api_guard import pool_safe_get
            from config import FINNHUB_API_KEY
            url = "https://finnhub.io/api/v1/stock/metric"
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: pool_safe_get(
                    url,
                    params={"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY},
                    provider="finnhub",
                )
            )
            if not resp or resp.status_code != 200:
                return result
            metric = (resp.json() or {}).get("metric", {})
            if not metric:
                return result
            price_52w  = float(metric.get("52WeekPriceReturnDaily", 0) or 0)
            rev_growth = float(metric.get("revenueGrowthQuarterlyYoy", 0) or 0)
            score = 0.0
            if price_52w > 50:   score += 40
            elif price_52w > 20: score += 25
            elif price_52w > 0:  score += 10
            if rev_growth > 0.20: score += 30
            elif rev_growth > 0.10: score += 15
            result.fundamental_score = min(100.0, score)
            result.overall_score     = result.fundamental_score
            result.data = {"52w_return_pct": price_52w, "rev_growth_yoy": rev_growth}
            if price_52w > 0:
                result.signals.append(f"52W_RETURN_{price_52w:.0f}%")
            if rev_growth > 0.10:
                result.signals.append(f"REV_GROWTH_{rev_growth*100:.0f}%_YOY")
            if result.overall_score >= 40:
                result.is_candidate = True
        except Exception as e:
            logger.debug(f"Sector rotation scan {ticker}: {e}")
        return result

    async def _scan_news_sentiment(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[ScanResult]:
        """News + social sentiment via CompanyNewsScanner + social_buzz."""
        result = ScanResult(ticker=ticker, scan_type=ScanType.NEWS_SENTIMENT)
        try:
            import concurrent.futures
            loop = asyncio.get_event_loop()
            # Social buzz
            from src.social_buzz import get_total_buzz_score
            buzz = await loop.run_in_executor(None, get_total_buzz_score, ticker)
            buzz_score = float(buzz or 0)
            # Company news
            from src.ingestors.company_news_scanner import CompanyNewsScanner
            scanner = CompanyNewsScanner()
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                scan_res = await loop.run_in_executor(ex, lambda: scanner.scan_company(ticker))
            news_count  = int(getattr(scan_res, "article_count", 0) or 0)
            sentiment   = float(getattr(scan_res, "sentiment_score", 0) or 0)
            # Composite 0-100
            buzz_c      = min(40.0, buzz_score * 40)
            news_c      = min(40.0, news_count * 4)
            sent_c      = min(20.0, max(0.0, sentiment * 20 + 10))
            result.sentiment_score = buzz_c + news_c + sent_c
            result.overall_score   = result.sentiment_score
            result.data = {"buzz": buzz_score, "news_count": news_count, "sentiment": sentiment}
            if buzz_score > 0.5:
                result.signals.append(f"BUZZ_{buzz_score:.2f}")
            if news_count >= 3:
                result.signals.append(f"NEWS_{news_count}_ARTICLES")
            if result.sentiment_score >= 50:
                result.is_candidate = True
                result.has_catalyst  = True
        except Exception as e:
            logger.debug(f"News sentiment scan {ticker}: {e}")
        return result

    # Query methods

    async def _scan_ticker_profiles(self, ticker: str, config: dict) -> Optional["ScanResult"]:
        """Update strategic profile for one ticker (float, short, dilution, history)."""
        try:
            from src.market_memory.ticker_profile_feeder import update_ticker_profile
            profile = await update_ticker_profile(ticker)
            if profile:
                result = ScanResult(ticker=ticker, scan_type=ScanType.TICKER_PROFILES)
                result.overall_score = profile.get("data_quality", 0.0)
                result.is_candidate = False
                result.data["data_quality"] = profile.get("data_quality", 0.0)
                result.data["float_shares"] = profile.get("float_shares")
                result.data["short_interest_pct"] = profile.get("short_interest_pct")
                return result
        except Exception as e:
            logger.debug(f"_scan_ticker_profiles failed for {ticker}: {e}")
        return None

    def get_candidates(
        self,
        scan_type: Optional[ScanType] = None,
        min_score: float = 0.0
    ) -> List[str]:
        """Get candidate tickers from all scans."""
        candidates = set()

        for job in self._jobs.values():
            if scan_type and job.scan_type != scan_type:
                continue

            for result in job.results:
                if result.is_candidate and result.overall_score >= min_score:
                    candidates.add(result.ticker)

        return list(candidates)

    def get_results(
        self,
        ticker: str
    ) -> List[ScanResult]:
        """Get all scan results for a ticker."""
        return self._results.get(ticker.upper(), [])

    def get_job(self, job_id: str) -> Optional[ScanJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[ScanStatus]:
        """Get status of a job."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    def get_pending_jobs(self) -> List[ScanJob]:
        """Get all pending jobs."""
        return [
            self._jobs[jid] for jid in self._job_queue
            if jid in self._jobs
        ]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        if job_id in self._job_queue:
            self._job_queue.remove(job_id)
            if job_id in self._jobs:
                self._jobs[job_id].status = ScanStatus.CANCELLED
            return True
        return False

    def stop(self) -> None:
        """Stop scanner after current batch."""
        self._running = False

    def get_stats(self) -> Dict:
        """Get scanner statistics."""
        return {
            **self._stats,
            "pending_jobs": len(self._job_queue),
            "current_job": self._current_job,
        }

    def clear_results(self) -> None:
        """Clear all cached results."""
        self._results.clear()
        self._jobs.clear()
        self._job_queue.clear()


# Singleton instance
_scanner: Optional[WeekendScanner] = None


def get_weekend_scanner(config: Optional[ScannerConfig] = None) -> WeekendScanner:
    """Get singleton WeekendScanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = WeekendScanner(config)
    return _scanner
