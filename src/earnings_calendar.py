"""
EARNINGS CALENDAR ENGINE — GV2-EDGE V9 (A4)
=============================================

Calendrier earnings systematique pour detecter les 15-20% de top gainers
qui sont earnings-driven.

Sources de donnees (gratuites):
- Finnhub /calendar/earnings — dates + EPS estimate + actual
- SEC EDGAR 10-Q/10-K — proxy filing dates
- Yahoo Finance — calendar + surprise history (scraping)

Integration Monster Score:
- J-7 a J-1: +0.05 a +0.10 boost (anticipation)
- J-Day BMO: +0.15 boost en pre-market
- Post-earnings (beat): +0.20 boost pendant 2h
- Post-earnings (miss): -0.15 penalty

Deploiement: Hetzner CX43 (8 vCPU, 16 Go RAM)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("EARNINGS_CALENDAR")


# ============================================================================
# Configuration
# ============================================================================

EARNINGS_CACHE_TTL = 3600 * 6        # 6 heures de cache
SURPRISE_HISTORY_TTL = 3600 * 24     # 24h pour historique surprises
FINNHUB_EARNINGS_ENDPOINT = "/calendar/earnings"
MAX_DAYS_AHEAD = 14                  # Scan 14 jours a l'avance
BOOST_WINDOW_DAYS = 7               # Boost commence J-7


class EarningsTiming(Enum):
    BMO = "BMO"    # Before Market Open
    AMC = "AMC"    # After Market Close
    UNKNOWN = "UNKNOWN"


class EarningsSurprise(Enum):
    BEAT = "BEAT"
    MISS = "MISS"
    INLINE = "INLINE"
    PENDING = "PENDING"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EarningsEvent:
    """Evenement earnings a venir ou recent."""
    ticker: str
    date: datetime
    timing: EarningsTiming
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    beat_rate: float = 0.5           # Historique beat % (0-1)
    avg_surprise_pct: float = 0.0    # Surprise moyenne historique
    implied_move: float = 0.0        # Move implicite via options IV
    sector: str = ""
    surprise: EarningsSurprise = EarningsSurprise.PENDING
    surprise_pct: float = 0.0        # % surprise vs estimate
    source: str = "finnhub"

    @property
    def days_until(self) -> int:
        """Jours avant l'evenement."""
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        event_date = self.date.replace(hour=0, minute=0, second=0, microsecond=0)
        if event_date.tzinfo is None:
            event_date = event_date.replace(tzinfo=timezone.utc)
        return (event_date - now).days

    @property
    def is_today(self) -> bool:
        return self.days_until == 0

    @property
    def is_past(self) -> bool:
        return self.days_until < 0

    @property
    def has_reported(self) -> bool:
        return self.eps_actual is not None

    def compute_surprise(self) -> None:
        """Calcule la surprise si les donnees sont disponibles."""
        if self.eps_actual is not None and self.eps_estimate is not None:
            if self.eps_estimate != 0:
                self.surprise_pct = ((self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)) * 100
            else:
                self.surprise_pct = 100.0 if self.eps_actual > 0 else -100.0

            if self.surprise_pct > 2.0:
                self.surprise = EarningsSurprise.BEAT
            elif self.surprise_pct < -2.0:
                self.surprise = EarningsSurprise.MISS
            else:
                self.surprise = EarningsSurprise.INLINE


# ============================================================================
# Earnings Calendar Engine
# ============================================================================

class EarningsCalendar:
    """
    Moteur de calendrier earnings complet.

    Sources:
    1. Finnhub /calendar/earnings (primary)
    2. Historical beat rate calculation
    3. Options implied move estimation

    Integration:
    - EventHub pour enrichir les catalysts
    - Monster Score pour boost earnings
    - Anticipation Engine pour pre-positioning
    """

    def __init__(self):
        self._cache = TTLCache(default_ttl=EARNINGS_CACHE_TTL)
        self._events: Dict[str, List[EarningsEvent]] = {}
        self._last_fetch: Optional[datetime] = None
        self._lock = threading.Lock()
        logger.info("EarningsCalendar initialized")

    async def fetch_upcoming(self, days_ahead: int = MAX_DAYS_AHEAD) -> List[EarningsEvent]:
        """
        Fetch earnings a venir depuis Finnhub.

        Args:
            days_ahead: nombre de jours a scanner

        Returns:
            Liste d'EarningsEvent triee par date
        """
        cache_key = f"earnings_upcoming_{days_ahead}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        events = []

        try:
            events = await self._fetch_finnhub_earnings(days_ahead)
        except Exception as e:
            logger.warning(f"Finnhub earnings fetch failed: {e}")

        # Enrichir avec beat rate historique
        for event in events:
            try:
                event.beat_rate = await self._get_beat_probability(event.ticker)
            except Exception:
                pass

        # Trier par date
        events.sort(key=lambda e: e.date)

        # Cache
        self._cache.set(cache_key, events)

        with self._lock:
            for event in events:
                if event.ticker not in self._events:
                    self._events[event.ticker] = []
                self._events[event.ticker].append(event)

        logger.info(f"Fetched {len(events)} upcoming earnings (next {days_ahead} days)")
        return events

    async def _fetch_finnhub_earnings(self, days_ahead: int) -> List[EarningsEvent]:
        """Fetch depuis Finnhub /calendar/earnings."""
        events = []

        try:
            from config import FINNHUB_API_KEY
            if not FINNHUB_API_KEY:
                logger.debug("No Finnhub API key — skipping earnings fetch")
                return events

            import aiohttp

            now = datetime.now(timezone.utc)
            from_date = now.strftime("%Y-%m-%d")
            to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

            url = f"https://finnhub.io/api/v1/calendar/earnings?from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Finnhub earnings API returned {resp.status}")
                        return events

                    data = await resp.json()

            for item in data.get("earningsCalendar", []):
                ticker = item.get("symbol", "")
                if not ticker:
                    continue

                # Parse date
                date_str = item.get("date", "")
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                # Parse timing
                hour = item.get("hour", "")
                if hour == "bmo":
                    timing = EarningsTiming.BMO
                elif hour == "amc":
                    timing = EarningsTiming.AMC
                else:
                    timing = EarningsTiming.UNKNOWN

                event = EarningsEvent(
                    ticker=ticker,
                    date=event_date,
                    timing=timing,
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                    source="finnhub",
                )

                # Compute surprise si deja reporte
                if event.has_reported:
                    event.compute_surprise()

                events.append(event)

        except ImportError:
            logger.debug("aiohttp not available — falling back to sync")
            events = self._fetch_finnhub_sync(days_ahead)
        except Exception as e:
            logger.warning(f"Finnhub earnings error: {e}")

        return events

    def _fetch_finnhub_sync(self, days_ahead: int) -> List[EarningsEvent]:
        """Fallback synchrone pour Finnhub."""
        events = []
        try:
            from config import FINNHUB_API_KEY
            if not FINNHUB_API_KEY:
                return events

            import requests

            now = datetime.now(timezone.utc)
            from_date = now.strftime("%Y-%m-%d")
            to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

            url = f"https://finnhub.io/api/v1/calendar/earnings"
            params = {"from": from_date, "to": to_date, "token": FINNHUB_API_KEY}

            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return events

            data = resp.json()
            for item in data.get("earningsCalendar", []):
                ticker = item.get("symbol", "")
                if not ticker:
                    continue

                date_str = item.get("date", "")
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                hour = item.get("hour", "")
                timing = EarningsTiming.BMO if hour == "bmo" else (
                    EarningsTiming.AMC if hour == "amc" else EarningsTiming.UNKNOWN
                )

                event = EarningsEvent(
                    ticker=ticker,
                    date=event_date,
                    timing=timing,
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                    source="finnhub",
                )
                if event.has_reported:
                    event.compute_surprise()

                events.append(event)

        except Exception as e:
            logger.warning(f"Finnhub sync earnings error: {e}")

        return events

    def get_upcoming(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Retourne les earnings a venir depuis le cache."""
        results = []
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)

        with self._lock:
            for ticker, events in self._events.items():
                for event in events:
                    event_date = event.date
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=timezone.utc)
                    if now <= event_date <= cutoff:
                        results.append(event)

        results.sort(key=lambda e: e.date)
        return results

    def get_today(self) -> List[EarningsEvent]:
        """Earnings aujourd'hui (BMO/AMC)."""
        results = []
        with self._lock:
            for ticker, events in self._events.items():
                for event in events:
                    if event.is_today:
                        results.append(event)
        return results

    def get_today_bmo(self) -> List[EarningsEvent]:
        """Earnings aujourd'hui Before Market Open."""
        return [e for e in self.get_today() if e.timing == EarningsTiming.BMO]

    def get_today_amc(self) -> List[EarningsEvent]:
        """Earnings aujourd'hui After Market Close."""
        return [e for e in self.get_today() if e.timing == EarningsTiming.AMC]

    def get_by_ticker(self, ticker: str) -> List[EarningsEvent]:
        """Tous les earnings pour un ticker."""
        with self._lock:
            return list(self._events.get(ticker.upper(), []))

    async def _get_beat_probability(self, ticker: str) -> float:
        """
        Probabilite de beat basee sur historique.
        Source: Finnhub earnings history ou historical_beat_rate.py
        """
        try:
            from src.historical_beat_rate import get_beat_rate
            rate = get_beat_rate(ticker)
            if rate is not None:
                return rate
        except ImportError:
            pass

        # Default: 60% des small caps battent les estimates
        return 0.60

    def get_beat_probability(self, ticker: str) -> float:
        """Version sync de _get_beat_probability."""
        try:
            from src.historical_beat_rate import get_beat_rate
            rate = get_beat_rate(ticker)
            if rate is not None:
                return rate
        except (ImportError, Exception):
            pass
        return 0.60

    def get_surprise_magnitude(self, ticker: str) -> float:
        """
        Magnitude de surprise attendue basee sur historique + IV.

        Returns:
            float: expected move en % (ex: 15.0 = +/-15%)
        """
        events = self.get_by_ticker(ticker)
        past_events = [e for e in events if e.has_reported]

        if past_events:
            avg_surprise = sum(abs(e.surprise_pct) for e in past_events) / len(past_events)
            return avg_surprise

        # Default small-cap: +/-20% move typique
        return 20.0

    def get_earnings_boost(self, ticker: str) -> Tuple[float, Dict]:
        """
        Calcule le boost Monster Score pour un ticker avec earnings.

        Returns:
            (boost: float, details: dict)

        Boost schedule:
        - J-7 a J-3: +0.05 (anticipation lointaine)
        - J-2 a J-1: +0.10 (anticipation proche)
        - J-Day BMO pre-market: +0.15
        - Post-earnings beat: +0.20 (pendant 2h)
        - Post-earnings miss: -0.15 (pendant 2h)
        """
        events = self.get_by_ticker(ticker)
        if not events:
            return 0.0, {}

        # Trouver l'earnings le plus proche
        now = datetime.now(timezone.utc)
        closest = None
        min_dist = float('inf')

        for event in events:
            dist = abs(event.days_until)
            if dist < min_dist:
                min_dist = dist
                closest = event

        if not closest:
            return 0.0, {}

        days = closest.days_until
        boost = 0.0
        reason = ""

        # Post-earnings (deja reporte)
        if closest.has_reported and -1 <= days <= 0:
            if closest.surprise == EarningsSurprise.BEAT:
                boost = 0.20
                reason = f"EARNINGS_BEAT (+{closest.surprise_pct:.1f}%)"
            elif closest.surprise == EarningsSurprise.MISS:
                boost = -0.15
                reason = f"EARNINGS_MISS ({closest.surprise_pct:.1f}%)"
            else:
                boost = 0.05
                reason = "EARNINGS_INLINE"

        # J-Day (pas encore reporte)
        elif days == 0 and not closest.has_reported:
            if closest.timing == EarningsTiming.BMO:
                boost = 0.15
                reason = "EARNINGS_TODAY_BMO"
            else:
                boost = 0.10
                reason = "EARNINGS_TODAY_AMC"

        # J-1 a J-2
        elif 1 <= days <= 2:
            boost = 0.10
            reason = f"EARNINGS_J-{days}"

        # J-3 a J-7
        elif 3 <= days <= BOOST_WINDOW_DAYS:
            boost = 0.05
            reason = f"EARNINGS_J-{days}"

        # Ajuster par beat rate (high beat rate = plus de boost)
        if boost > 0 and closest.beat_rate > 0.7:
            boost *= 1.0 + (closest.beat_rate - 0.5) * 0.5  # Max +25% du boost

        details = {
            "ticker": ticker,
            "date": closest.date.strftime("%Y-%m-%d"),
            "timing": closest.timing.value,
            "days_until": days,
            "beat_rate": closest.beat_rate,
            "eps_estimate": closest.eps_estimate,
            "boost": round(boost, 4),
            "reason": reason,
        }

        return round(boost, 4), details

    def get_status(self) -> Dict:
        """Status du calendrier earnings."""
        with self._lock:
            total = sum(len(v) for v in self._events.values())
            today = len(self.get_today())
            upcoming_7d = len(self.get_upcoming(7))

        return {
            "total_events_cached": total,
            "tickers_tracked": len(self._events),
            "earnings_today": today,
            "earnings_next_7d": upcoming_7d,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
        }


# ============================================================================
# Singleton
# ============================================================================

_calendar: Optional[EarningsCalendar] = None
_calendar_lock = threading.Lock()


def get_earnings_calendar() -> EarningsCalendar:
    """Get singleton EarningsCalendar instance."""
    global _calendar
    with _calendar_lock:
        if _calendar is None:
            _calendar = EarningsCalendar()
    return _calendar
