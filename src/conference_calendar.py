"""
CONFERENCE CALENDAR — GV2-EDGE V9 (A6)
========================================

Calendrier des conferences biotech/finance pour detecter les 5-8% de top gainers
qui sont conference-driven (presentations, data readouts, partnerships).

Conferences cles: JPM Healthcare, ASCO, ASH, AACR, AAN, Bio-Europe, Needham.

Source: Calendrier statique + SEC 8-K mentionnant presentations.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional
from enum import Enum

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("CONFERENCE_CALENDAR")


# ============================================================================
# Configuration
# ============================================================================

CONFERENCE_CACHE_TTL = 3600 * 12  # 12 heures
BOOST_DAYS_BEFORE = 5            # Boost commence J-5
BOOST_DAYS_DURING = 0            # Boost pendant la conference
BOOST_DAYS_AFTER = 1             # Boost 1 jour apres


class ConferenceImpact(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Conference:
    """Conference planifiee."""
    name: str
    short_name: str
    start_date: date
    end_date: date
    sector: str
    impact: ConferenceImpact
    location: str = ""
    source_url: str = ""
    presenting_tickers: List[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        today = date.today()
        return self.start_date <= today <= self.end_date

    @property
    def days_until(self) -> int:
        return (self.start_date - date.today()).days

    @property
    def is_upcoming(self) -> bool:
        return 0 < self.days_until <= BOOST_DAYS_BEFORE

    @property
    def is_past(self) -> bool:
        return (date.today() - self.end_date).days > 0


# ============================================================================
# Conference Database 2025-2027
# ============================================================================

def _build_conference_db() -> List[Conference]:
    """
    Base de donnees des conferences majeures.
    Mise a jour annuelle. Les dates sont approximatives
    (les conferences recurrentes tombent generalement aux memes periodes).
    """
    conferences = []

    # --- 2026 ---
    year = 2026

    # JPM Healthcare Conference (Janvier)
    conferences.append(Conference(
        name="JPM Healthcare Conference",
        short_name="JPM",
        start_date=date(year, 1, 12),
        end_date=date(year, 1, 15),
        sector="Healthcare/Biotech",
        impact=ConferenceImpact.CRITICAL,
        location="San Francisco, CA",
    ))

    # Needham Growth Conference (Janvier)
    conferences.append(Conference(
        name="Needham Growth Conference",
        short_name="NEEDHAM",
        start_date=date(year, 1, 13),
        end_date=date(year, 1, 15),
        sector="Tech/Biotech Small-Cap",
        impact=ConferenceImpact.MEDIUM,
        location="New York, NY",
    ))

    # ASCO GI (Janvier)
    conferences.append(Conference(
        name="ASCO Gastrointestinal Cancers Symposium",
        short_name="ASCO_GI",
        start_date=date(year, 1, 22),
        end_date=date(year, 1, 24),
        sector="Oncology",
        impact=ConferenceImpact.HIGH,
        location="San Francisco, CA",
    ))

    # Oppenheimer Healthcare (Mars)
    conferences.append(Conference(
        name="Oppenheimer Annual Healthcare Conference",
        short_name="OPCO",
        start_date=date(year, 3, 16),
        end_date=date(year, 3, 18),
        sector="Healthcare",
        impact=ConferenceImpact.MEDIUM,
        location="New York, NY",
    ))

    # AACR Annual Meeting (Avril)
    conferences.append(Conference(
        name="AACR Annual Meeting",
        short_name="AACR",
        start_date=date(year, 4, 11),
        end_date=date(year, 4, 15),
        sector="Oncology",
        impact=ConferenceImpact.HIGH,
        location="Los Angeles, CA",
    ))

    # AAN Annual Meeting (Avril)
    conferences.append(Conference(
        name="AAN Annual Meeting",
        short_name="AAN",
        start_date=date(year, 4, 25),
        end_date=date(year, 5, 1),
        sector="Neurology",
        impact=ConferenceImpact.HIGH,
        location="Seattle, WA",
    ))

    # ASCO Annual Meeting (Juin)
    conferences.append(Conference(
        name="ASCO Annual Meeting",
        short_name="ASCO",
        start_date=date(year, 5, 29),
        end_date=date(year, 6, 2),
        sector="Oncology",
        impact=ConferenceImpact.CRITICAL,
        location="Chicago, IL",
    ))

    # BIO International (Juin)
    conferences.append(Conference(
        name="BIO International Convention",
        short_name="BIO",
        start_date=date(year, 6, 8),
        end_date=date(year, 6, 11),
        sector="Biotech",
        impact=ConferenceImpact.HIGH,
        location="Boston, MA",
    ))

    # EHA (European Hematology) (Juin)
    conferences.append(Conference(
        name="EHA Annual Congress",
        short_name="EHA",
        start_date=date(year, 6, 12),
        end_date=date(year, 6, 15),
        sector="Hematology",
        impact=ConferenceImpact.HIGH,
        location="Milan, Italy",
    ))

    # ESMO (Septembre)
    conferences.append(Conference(
        name="ESMO Annual Congress",
        short_name="ESMO",
        start_date=date(year, 9, 13),
        end_date=date(year, 9, 17),
        sector="Oncology",
        impact=ConferenceImpact.HIGH,
        location="Barcelona, Spain",
    ))

    # Bio-Europe (Novembre)
    conferences.append(Conference(
        name="Bio-Europe Conference",
        short_name="BIOEU",
        start_date=date(year, 11, 2),
        end_date=date(year, 11, 4),
        sector="Biotech EU",
        impact=ConferenceImpact.MEDIUM,
        location="Munich, Germany",
    ))

    # ASH Annual Meeting (Decembre)
    conferences.append(Conference(
        name="ASH Annual Meeting",
        short_name="ASH",
        start_date=date(year, 12, 5),
        end_date=date(year, 12, 8),
        sector="Hematology",
        impact=ConferenceImpact.CRITICAL,
        location="San Diego, CA",
    ))

    # --- Repetition pour 2027 (memes periodes, +365 jours) ---
    for conf in list(conferences):
        conferences.append(Conference(
            name=conf.name,
            short_name=conf.short_name,
            start_date=conf.start_date.replace(year=2027),
            end_date=conf.end_date.replace(year=2027),
            sector=conf.sector,
            impact=conf.impact,
            location=conf.location,
        ))

    return conferences


# ============================================================================
# Conference Calendar Engine
# ============================================================================

class ConferenceCalendar:
    """
    Moteur calendrier conferences.

    Detecte les conferences actives/proches et fournit un boost
    pour les tickers qui presentent.
    """

    def __init__(self):
        self._conferences = _build_conference_db()
        self._presenter_map: Dict[str, List[str]] = {}  # ticker -> [conference_short_names]
        self._cache = TTLCache(default_ttl=CONFERENCE_CACHE_TTL)
        logger.info(f"ConferenceCalendar initialized — {len(self._conferences)} conferences loaded")

    def register_presenter(self, ticker: str, conference_short_name: str) -> None:
        """Enregistre un ticker comme presenteur a une conference."""
        ticker = ticker.upper()
        if ticker not in self._presenter_map:
            self._presenter_map[ticker] = []
        if conference_short_name not in self._presenter_map[ticker]:
            self._presenter_map[ticker].append(conference_short_name)

        # Ajouter aussi a la conference
        for conf in self._conferences:
            if conf.short_name == conference_short_name:
                if ticker not in conf.presenting_tickers:
                    conf.presenting_tickers.append(ticker)

    def register_presenters_batch(self, presenters: Dict[str, List[str]]) -> None:
        """Enregistre plusieurs presenteurs. {conference_short: [tickers]}"""
        for conf_name, tickers in presenters.items():
            for ticker in tickers:
                self.register_presenter(ticker, conf_name)

    def get_active_conferences(self) -> List[Conference]:
        """Conferences en cours aujourd'hui."""
        return [c for c in self._conferences if c.is_active]

    def get_upcoming_conferences(self, days: int = 7) -> List[Conference]:
        """Conferences dans les X prochains jours."""
        today = date.today()
        cutoff = today + timedelta(days=days)
        return [
            c for c in self._conferences
            if today <= c.start_date <= cutoff
        ]

    def get_presenting_tickers(self, conference_short_name: str) -> List[str]:
        """Tickers avec presentations planifiees."""
        for conf in self._conferences:
            if conf.short_name == conference_short_name:
                return list(conf.presenting_tickers)
        return []

    def get_ticker_conferences(self, ticker: str) -> List[Conference]:
        """Conferences ou un ticker presente."""
        ticker = ticker.upper()
        conf_names = self._presenter_map.get(ticker, [])
        return [c for c in self._conferences if c.short_name in conf_names]

    def get_conference_boost(self, ticker: str) -> tuple:
        """
        Boost Monster Score pour conference.

        Returns:
            (boost: float, details: dict)

        Boost schedule:
        - J-5 a J-1 (upcoming): +0.05 a +0.10
        - J-Day (active, presenting): +0.15
        - J-Day (active, same sector): +0.05
        """
        ticker = ticker.upper()
        boost = 0.0
        details = {}

        # Check si le ticker presente a une conference active/upcoming
        ticker_confs = self.get_ticker_conferences(ticker)

        for conf in ticker_confs:
            if conf.is_active:
                # Conference en cours, ticker presente
                conf_boost = 0.15 if conf.impact == ConferenceImpact.CRITICAL else 0.10
                if conf_boost > boost:
                    boost = conf_boost
                    details = {
                        "conference": conf.name,
                        "status": "ACTIVE_PRESENTER",
                        "impact": conf.impact.value,
                    }
            elif conf.is_upcoming:
                # Conference a venir
                days = conf.days_until
                if days <= 2:
                    conf_boost = 0.10
                elif days <= 5:
                    conf_boost = 0.05
                else:
                    conf_boost = 0.03

                if conf_boost > boost:
                    boost = conf_boost
                    details = {
                        "conference": conf.name,
                        "status": f"UPCOMING_J-{days}",
                        "impact": conf.impact.value,
                    }

        # Check si une conference du meme secteur est active (boost leger)
        if boost == 0:
            active = self.get_active_conferences()
            for conf in active:
                if conf.impact in (ConferenceImpact.CRITICAL, ConferenceImpact.HIGH):
                    # Sector match rudimentaire — le ticker pourrait beneficier
                    sector_boost = 0.03
                    if sector_boost > boost:
                        boost = sector_boost
                        details = {
                            "conference": conf.name,
                            "status": "SECTOR_ACTIVE",
                            "impact": conf.impact.value,
                        }

        return round(boost, 4), details

    def get_status(self) -> Dict:
        """Status du calendrier conferences."""
        active = self.get_active_conferences()
        upcoming = self.get_upcoming_conferences(14)
        return {
            "total_conferences": len(self._conferences),
            "active_now": len(active),
            "active_names": [c.short_name for c in active],
            "upcoming_14d": len(upcoming),
            "upcoming_names": [c.short_name for c in upcoming],
            "registered_presenters": len(self._presenter_map),
        }


# ============================================================================
# Singleton
# ============================================================================

_calendar: Optional[ConferenceCalendar] = None
_calendar_lock = threading.Lock()


def get_conference_calendar() -> ConferenceCalendar:
    """Get singleton ConferenceCalendar instance."""
    global _calendar
    with _calendar_lock:
        if _calendar is None:
            _calendar = ConferenceCalendar()
    return _calendar
