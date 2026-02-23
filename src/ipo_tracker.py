"""
IPO/SPO TRACKER — GV2-EDGE V9 (A7)
=====================================

Tracking des IPO day-1 runners et follow-on offerings pour detecter
les 3-5% de top gainers qui sont IPO-driven.

Sources:
- SEC EDGAR S-1/424B — IPO filings (gratuit)
- Finnhub IPO Calendar — upcoming IPOs (free tier)
- SEC EDGAR S-3/424B — Follow-on offerings (gratuit)

Detection:
- IPO day-1 runners (fresh floats, short squeezes)
- Recent IPO runners (days 2-30, lockup awareness)
- Lockup expirations (short squeeze catalysts)
- Follow-on offerings (dilution detection)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple
from enum import Enum

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("IPO_TRACKER")


# ============================================================================
# Configuration
# ============================================================================

IPO_CACHE_TTL = 3600 * 6       # 6h
LOCKUP_PERIOD_DAYS = 180       # Typique: 180 jours
RECENT_IPO_DAYS = 30           # IPO considere "recent" pendant 30 jours
DAY1_BOOST = 0.15              # Boost Monster Score jour-1
RECENT_BOOST_BASE = 0.05       # Boost base pour recent IPO
LOCKUP_BOOST = 0.10            # Boost pre-lockup expiry


class IPOStatus(Enum):
    UPCOMING = "UPCOMING"       # Pas encore listee
    DAY1 = "DAY1"              # Premier jour de trading
    RECENT = "RECENT"          # 2-30 jours
    ESTABLISHED = "ESTABLISHED" # 30+ jours
    PRE_LOCKUP = "PRE_LOCKUP"  # 7 jours avant lockup expiry


class OfferingType(Enum):
    IPO = "IPO"
    SPAC = "SPAC"
    DIRECT_LISTING = "DIRECT_LISTING"
    SPO = "SPO"                # Secondary/Follow-on


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IPOEvent:
    """Evenement IPO/listing."""
    ticker: str
    company_name: str
    ipo_date: date
    offering_type: OfferingType = OfferingType.IPO
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    offer_price: Optional[float] = None
    shares_offered: Optional[int] = None
    float_shares: Optional[int] = None
    market_cap_est: Optional[float] = None
    sector: str = ""
    exchange: str = ""
    underwriters: List[str] = field(default_factory=list)
    lockup_expiry: Optional[date] = None
    source: str = "finnhub"

    def __post_init__(self):
        if self.lockup_expiry is None and self.ipo_date:
            self.lockup_expiry = self.ipo_date + timedelta(days=LOCKUP_PERIOD_DAYS)

    @property
    def status(self) -> IPOStatus:
        today = date.today()
        if self.ipo_date > today:
            return IPOStatus.UPCOMING
        days_since = (today - self.ipo_date).days
        if days_since == 0:
            return IPOStatus.DAY1
        if days_since <= RECENT_IPO_DAYS:
            if self.lockup_expiry and (self.lockup_expiry - today).days <= 7:
                return IPOStatus.PRE_LOCKUP
            return IPOStatus.RECENT
        if self.lockup_expiry and 0 <= (self.lockup_expiry - today).days <= 7:
            return IPOStatus.PRE_LOCKUP
        return IPOStatus.ESTABLISHED

    @property
    def days_since_ipo(self) -> int:
        return (date.today() - self.ipo_date).days

    @property
    def days_to_lockup(self) -> Optional[int]:
        if self.lockup_expiry:
            return (self.lockup_expiry - date.today()).days
        return None


@dataclass
class LockupEvent:
    """Evenement expiration lockup."""
    ticker: str
    lockup_date: date
    ipo_date: date
    shares_locked: Optional[int] = None
    pct_float_locked: Optional[float] = None

    @property
    def days_until(self) -> int:
        return (self.lockup_date - date.today()).days


# ============================================================================
# IPO Tracker Engine
# ============================================================================

class IPOTracker:
    """
    Tracker IPO/SPO avec detection automatique.

    Fonctionnalites:
    - Fetch calendrier IPO (Finnhub + SEC EDGAR)
    - Suivi jour-1 runners (float serree, potentiel squeeze)
    - Detection lockup expirations
    - Boost Monster Score pour IPO/SPO
    """

    def __init__(self):
        self._ipos: Dict[str, IPOEvent] = {}
        self._cache = TTLCache(default_ttl=IPO_CACHE_TTL)
        self._lock = threading.Lock()
        logger.info("IPOTracker initialized")

    async def fetch_upcoming(self, days_ahead: int = 14) -> List[IPOEvent]:
        """Fetch IPO a venir depuis Finnhub."""
        cache_key = f"ipo_upcoming_{days_ahead}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        events = []

        try:
            from config import FINNHUB_API_KEY
            if not FINNHUB_API_KEY:
                return events

            import aiohttp

            now = datetime.now(timezone.utc)
            from_date = now.strftime("%Y-%m-%d")
            to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

            url = f"https://finnhub.io/api/v1/calendar/ipo?from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return events
                    data = await resp.json()

            for item in data.get("ipoCalendar", []):
                ticker = item.get("symbol", "")
                if not ticker:
                    continue

                date_str = item.get("date", "")
                try:
                    ipo_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

                event = IPOEvent(
                    ticker=ticker,
                    company_name=item.get("name", ""),
                    ipo_date=ipo_date,
                    price_range_low=item.get("priceRangeLow"),
                    price_range_high=item.get("priceRangeHigh"),
                    shares_offered=item.get("numberOfShares"),
                    exchange=item.get("exchange", ""),
                    source="finnhub",
                )
                events.append(event)

                with self._lock:
                    self._ipos[ticker] = event

        except ImportError:
            logger.debug("aiohttp not available for IPO fetch")
        except Exception as e:
            logger.warning(f"IPO fetch error: {e}")

        self._cache.set(cache_key, events)
        return events

    def get_recent_ipos(self, days: int = RECENT_IPO_DAYS) -> List[IPOEvent]:
        """IPOs des N derniers jours."""
        cutoff = date.today() - timedelta(days=days)
        with self._lock:
            return [
                ipo for ipo in self._ipos.values()
                if ipo.ipo_date >= cutoff and ipo.status in (IPOStatus.DAY1, IPOStatus.RECENT)
            ]

    def get_upcoming_ipos(self, days: int = 14) -> List[IPOEvent]:
        """IPOs planifiees."""
        with self._lock:
            return [
                ipo for ipo in self._ipos.values()
                if ipo.status == IPOStatus.UPCOMING and ipo.days_since_ipo >= -days
            ]

    def get_day1_ipos(self) -> List[IPOEvent]:
        """IPOs jour-1 (aujourd'hui)."""
        with self._lock:
            return [ipo for ipo in self._ipos.values() if ipo.status == IPOStatus.DAY1]

    def get_lockup_expirations(self, days: int = 14) -> List[LockupEvent]:
        """Expirations lockup dans les N prochains jours."""
        results = []
        today = date.today()
        cutoff = today + timedelta(days=days)

        with self._lock:
            for ipo in self._ipos.values():
                if ipo.lockup_expiry and today <= ipo.lockup_expiry <= cutoff:
                    results.append(LockupEvent(
                        ticker=ipo.ticker,
                        lockup_date=ipo.lockup_expiry,
                        ipo_date=ipo.ipo_date,
                        shares_locked=ipo.shares_offered,
                    ))

        results.sort(key=lambda x: x.lockup_date)
        return results

    def get_ipo_boost(self, ticker: str) -> Tuple[float, Dict]:
        """
        Boost Monster Score pour IPO/SPO.

        Returns:
            (boost, details)

        Boosts:
        - Day 1: +0.15 (fresh float, high volatility)
        - Day 2-7: +0.10 (still very volatile)
        - Day 8-30: +0.05 (recent IPO premium)
        - Pre-lockup (J-7): +0.10 (squeeze potential)
        """
        ticker = ticker.upper()
        with self._lock:
            ipo = self._ipos.get(ticker)

        if not ipo:
            return 0.0, {}

        status = ipo.status
        boost = 0.0
        reason = ""

        if status == IPOStatus.DAY1:
            boost = DAY1_BOOST
            reason = "IPO_DAY1"
        elif status == IPOStatus.RECENT:
            days = ipo.days_since_ipo
            if days <= 7:
                boost = 0.10
                reason = f"IPO_DAY_{days}"
            else:
                boost = RECENT_BOOST_BASE
                reason = f"RECENT_IPO_{days}d"
        elif status == IPOStatus.PRE_LOCKUP:
            boost = LOCKUP_BOOST
            reason = f"PRE_LOCKUP_J-{ipo.days_to_lockup}"

        details = {
            "ticker": ticker,
            "ipo_date": ipo.ipo_date.isoformat(),
            "status": status.value,
            "days_since_ipo": ipo.days_since_ipo,
            "days_to_lockup": ipo.days_to_lockup,
            "boost": boost,
            "reason": reason,
        }

        return boost, details

    def is_recent_ipo(self, ticker: str) -> bool:
        """Check si un ticker est un IPO recent."""
        with self._lock:
            ipo = self._ipos.get(ticker.upper())
            return ipo is not None and ipo.status in (IPOStatus.DAY1, IPOStatus.RECENT)

    def add_ipo(self, event: IPOEvent) -> None:
        """Ajoute manuellement un IPO au tracker."""
        with self._lock:
            self._ipos[event.ticker.upper()] = event

    def get_status(self) -> Dict:
        """Status du tracker IPO."""
        with self._lock:
            day1 = len([i for i in self._ipos.values() if i.status == IPOStatus.DAY1])
            recent = len([i for i in self._ipos.values() if i.status == IPOStatus.RECENT])
            upcoming = len([i for i in self._ipos.values() if i.status == IPOStatus.UPCOMING])

        return {
            "total_tracked": len(self._ipos),
            "day1_ipos": day1,
            "recent_ipos": recent,
            "upcoming_ipos": upcoming,
            "lockup_expirations_14d": len(self.get_lockup_expirations(14)),
        }


# ============================================================================
# Singleton
# ============================================================================

_tracker: Optional[IPOTracker] = None
_tracker_lock = threading.Lock()


def get_ipo_tracker() -> IPOTracker:
    """Get singleton IPOTracker instance."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = IPOTracker()
    return _tracker
