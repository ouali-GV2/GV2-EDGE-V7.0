"""
FDA CALENDAR SCRAPER
====================

Scrape FDA-related events for biotech/pharma stocks:
1. PDUFA dates (FDA decision deadlines)
2. Clinical trial results (Phase I/II/III)
3. Biotech conferences (JPM, ASCO, ASH, etc.)

Sources:
- BiopharmCatalyst.com (free scraping)
- FDA.gov PDUFA calendar
- ClinicalTrials.gov
- Conference websites

Impact: HIGH for biotech small caps (often +50%+ on approvals)
"""

from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("FDA_CALENDAR")

cache = Cache(ttl=3600 * 6)  # 6h cache (FDA calendar updates infrequently)


# ============================
# PDUFA Dates Scraping
# ============================

def scrape_pdufa_dates():
    """
    Scrape PDUFA dates from BiopharmCatalyst
    
    PDUFA = Prescription Drug User Fee Act
    FDA must make decision by this date
    
    Returns:
        List of PDUFA events
    """
    cached = cache.get("pdufa_dates")
    if cached:
        return cached
    
    try:
        # BiopharmCatalyst free calendar
        url = "https://www.biopharmcatalyst.com/calendars/fda-calendar"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"BiopharmCatalyst returned {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pdufa_events = []
        
        # Parse table (structure may vary)
        # This is a simplified parser - may need adjustment
        table = soup.find('table', {'class': 'table'})
        
        if not table:
            logger.warning("No PDUFA table found on page")
            return []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cols = row.find_all('td')
            
            if len(cols) < 4:
                continue
            
            try:
                ticker = cols[0].text.strip()
                drug_name = cols[1].text.strip()
                pdufa_date = cols[2].text.strip()
                indication = cols[3].text.strip() if len(cols) > 3 else ""
                
                # Parse date
                try:
                    event_date = datetime.strptime(pdufa_date, "%m/%d/%Y")
                except:
                    # Try other formats
                    try:
                        event_date = datetime.strptime(pdufa_date, "%Y-%m-%d")
                    except:
                        continue
                
                # Only future dates
                if event_date.date() < datetime.now().date():
                    continue
                
                pdufa_events.append({
                    "ticker": ticker.upper(),
                    "type": "PDUFA",
                    "drug_name": drug_name,
                    "date": event_date.strftime("%Y-%m-%d"),
                    "indication": indication,
                    "impact": 0.9,  # PDUFA = very high impact
                    "category": "FDA_APPROVAL"
                })
            
            except Exception as e:
                logger.debug(f"Error parsing PDUFA row: {e}")
                continue
        
        cache.set("pdufa_dates", pdufa_events)
        
        logger.info(f"Scraped {len(pdufa_events)} PDUFA dates")
        
        return pdufa_events
    
    except Exception as e:
        logger.error(f"PDUFA scraping failed: {e}")
        return []


# ============================
# Clinical Trial Results
# ============================

def scrape_trial_results():
    """
    Scrape upcoming clinical trial results
    
    Phase I: Safety (small group)
    Phase II: Efficacy (larger group)
    Phase III: Large-scale confirmation
    
    Returns:
        List of trial result events
    """
    cached = cache.get("trial_results")
    if cached:
        return cached
    
    try:
        # BiopharmCatalyst clinical trials calendar
        url = "https://www.biopharmcatalyst.com/calendars/clinical-trial-calendar"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"Clinical trials page returned {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        trial_events = []
        
        table = soup.find('table', {'class': 'table'})
        
        if not table:
            logger.warning("No trial table found")
            return []
        
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            
            if len(cols) < 4:
                continue
            
            try:
                ticker = cols[0].text.strip()
                drug_name = cols[1].text.strip()
                phase = cols[2].text.strip()  # Phase I/II/III
                date_str = cols[3].text.strip()
                
                # Parse date
                try:
                    event_date = datetime.strptime(date_str, "%m/%d/%Y")
                except:
                    try:
                        event_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except:
                        # Try fuzzy parsing (Q1 2025, etc.)
                        event_date = parse_fuzzy_date(date_str)
                        if not event_date:
                            continue
                
                # Future only
                if event_date.date() < datetime.now().date():
                    continue
                
                # Impact based on phase
                impact_map = {
                    "Phase I": 0.6,
                    "Phase II": 0.75,
                    "Phase III": 0.85,
                    "Phase 1": 0.6,
                    "Phase 2": 0.75,
                    "Phase 3": 0.85
                }
                
                impact = impact_map.get(phase, 0.7)
                
                trial_events.append({
                    "ticker": ticker.upper(),
                    "type": "TRIAL_RESULT",
                    "drug_name": drug_name,
                    "phase": phase,
                    "date": event_date.strftime("%Y-%m-%d"),
                    "impact": impact,
                    "category": "FDA_TRIAL_RESULT"
                })
            
            except Exception as e:
                logger.debug(f"Error parsing trial row: {e}")
                continue
        
        cache.set("trial_results", trial_events)
        
        logger.info(f"Scraped {len(trial_events)} trial results")
        
        return trial_events
    
    except Exception as e:
        logger.error(f"Trial scraping failed: {e}")
        return []


def parse_fuzzy_date(date_str):
    """
    Parse fuzzy dates like "Q1 2025", "Early 2025", etc.
    
    Returns:
        datetime object or None
    """
    now = datetime.now()
    
    # Q1, Q2, Q3, Q4
    if "Q1" in date_str:
        return datetime(int(date_str.split()[-1]), 3, 15)  # Mid Q1
    elif "Q2" in date_str:
        return datetime(int(date_str.split()[-1]), 6, 15)
    elif "Q3" in date_str:
        return datetime(int(date_str.split()[-1]), 9, 15)
    elif "Q4" in date_str:
        return datetime(int(date_str.split()[-1]), 12, 15)
    
    # Early/Mid/Late
    if "Early" in date_str or "H1" in date_str:
        return datetime(int(date_str.split()[-1]), 3, 1)
    elif "Mid" in date_str:
        return datetime(int(date_str.split()[-1]), 7, 1)
    elif "Late" in date_str or "H2" in date_str:
        return datetime(int(date_str.split()[-1]), 10, 1)
    
    return None


# ============================
# Biotech Conferences
# ============================

def get_biotech_conferences():
    """
    Get major biotech conference dates
    
    Conferences where trial data is often presented:
    - JP Morgan Healthcare Conference (January)
    - ASCO (American Society of Clinical Oncology) (May/June)
    - ASH (American Society of Hematology) (December)
    - AACR (American Association for Cancer Research) (April)
    - EHA (European Hematology Association) (June)
    
    Returns:
        List of conference events
    """
    # Static calendar (updated manually)
    # Can be enhanced with scraping if needed
    
    year = datetime.now().year
    
    conferences = [
        {
            "name": "JP Morgan Healthcare Conference",
            "start_date": f"{year}-01-08",
            "end_date": f"{year}-01-11",
            "location": "San Francisco",
            "impact": 0.7
        },
        {
            "name": "ASCO Annual Meeting",
            "start_date": f"{year}-05-30",
            "end_date": f"{year}-06-03",
            "location": "Chicago",
            "impact": 0.85
        },
        {
            "name": "AACR Annual Meeting",
            "start_date": f"{year}-04-05",
            "end_date": f"{year}-04-10",
            "location": "Various",
            "impact": 0.75
        },
        {
            "name": "EHA Congress",
            "start_date": f"{year}-06-12",
            "end_date": f"{year}-06-15",
            "location": "Europe",
            "impact": 0.7
        },
        {
            "name": "ASH Annual Meeting",
            "start_date": f"{year}-12-07",
            "end_date": f"{year}-12-10",
            "location": "San Diego",
            "impact": 0.8
        }
    ]
    
    # Filter: only upcoming conferences
    upcoming = []
    
    for conf in conferences:
        start = datetime.strptime(conf["start_date"], "%Y-%m-%d")
        end = datetime.strptime(conf["end_date"], "%Y-%m-%d")
        
        # Include if conference starts within next 90 days
        if 0 <= (start.date() - datetime.now().date()).days <= 90:
            upcoming.append({
                "type": "CONFERENCE",
                "name": conf["name"],
                "start_date": conf["start_date"],
                "end_date": conf["end_date"],
                "location": conf["location"],
                "impact": conf["impact"],
                "category": "BIOTECH_CONFERENCE"
            })
    
    logger.info(f"Found {len(upcoming)} upcoming biotech conferences")
    
    return upcoming


# ============================
# Consolidate All FDA Events
# ============================

def get_all_fda_events():
    """
    Get all FDA-related events:
    - PDUFA dates
    - Clinical trial results
    - Biotech conferences
    
    Returns:
        Consolidated list of FDA events
    """
    all_events = []
    
    # PDUFA dates
    pdufa = scrape_pdufa_dates()
    all_events.extend(pdufa)
    
    # Trial results
    trials = scrape_trial_results()
    all_events.extend(trials)
    
    # Conferences
    conferences = get_biotech_conferences()
    all_events.extend(conferences)
    
    logger.info(f"Total FDA events: {len(all_events)} (PDUFA: {len(pdufa)}, Trials: {len(trials)}, Conferences: {len(conferences)})")
    
    return all_events


# ============================
# Filter by ticker
# ============================

def get_fda_events_by_ticker(ticker):
    """Get FDA events for specific ticker"""
    all_events = get_all_fda_events()
    
    ticker_events = [e for e in all_events if e.get("ticker", "").upper() == ticker.upper()]
    
    return ticker_events


# ============================
# Upcoming PDUFA (Next 30 days)
# ============================

def get_upcoming_pdufa(days=30):
    """Get PDUFA dates in next N days"""
    pdufa_events = scrape_pdufa_dates()
    
    cutoff = datetime.now() + timedelta(days=days)
    
    upcoming = []
    
    for event in pdufa_events:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d")
        
        if event_date <= cutoff:
            upcoming.append(event)
    
    return upcoming


# ============================
# V9 — A5: FDA Calendar Complet
# ============================
# FDACalendarEngine: real API integration (OpenFDA, ClinicalTrials.gov, Finnhub)
# 5-tier catalyst scoring, probability estimation, singleton pattern
# Backward-compatible: get_fda_events(ticker), get_fda_boost(ticker)

import asyncio
import aiohttp
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# ============================
# FDA Event Types & Tiers
# ============================

class FDACatalystTier(Enum):
    """
    5-tier FDA catalyst scoring system.

    Each tier defines:
    - The type of FDA event
    - The score boost applied to Monster Score
    - The time window (days before event) during which the boost is active
    """
    TIER_1 = "PDUFA"            # PDUFA decision date: +0.25, J-3 to J-Day
    TIER_2 = "PHASE3_DATA"      # Phase 3 data readout: +0.20, J-7 to J-Day
    TIER_3 = "ADCOM"            # ADCOM meeting: +0.15, J-5 to J-Day
    TIER_4 = "PHASE2_TOPLINE"   # Phase 2 topline: +0.10, J-3 to J-Day
    TIER_5 = "CRL_APPROVAL"     # CRL/Approval news: +0.20, J-Day only


# Tier configuration: (boost, window_days_before, window_days_after)
FDA_TIER_CONFIG = {
    FDACatalystTier.TIER_1: {"boost": 0.25, "window_before": 3, "window_after": 0, "label": "PDUFA Decision"},
    FDACatalystTier.TIER_2: {"boost": 0.20, "window_before": 7, "window_after": 0, "label": "Phase 3 Data Readout"},
    FDACatalystTier.TIER_3: {"boost": 0.15, "window_before": 5, "window_after": 0, "label": "ADCOM Meeting"},
    FDACatalystTier.TIER_4: {"boost": 0.10, "window_before": 3, "window_after": 0, "label": "Phase 2 Topline"},
    FDACatalystTier.TIER_5: {"boost": 0.20, "window_before": 0, "window_after": 0, "label": "CRL/Approval News"},
}


@dataclass
class FDAEvent:
    """
    Represents a single FDA-related catalyst event.

    Used by FDACalendarEngine for scoring and by Multi-Radar Engine
    (Catalyst Radar) for confluence detection.
    """
    ticker: str
    company: str
    drug_name: str
    event_type: str                    # PDUFA, PHASE3_DATA, APPROVAL, CRL, ADCOM, PHASE2_TOPLINE
    date: datetime
    phase: str = ""                    # Phase 1, 2, 3, NDA, BLA
    indication: str = ""               # Disease/condition
    probability: float = 0.5           # Estimated approval probability (0-1)
    historical_precedent: float = 0.5  # Historical approval rate for similar drugs
    tier: Optional[FDACatalystTier] = None
    source: str = ""                   # openfda, clinicaltrials, finnhub, static
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def days_until(self) -> int:
        """Days until this event (negative = past)."""
        return (self.date.date() - datetime.now().date()).days

    @property
    def is_in_window(self) -> bool:
        """Check if current date falls within the scoring window for this event's tier."""
        if not self.tier:
            return False
        config = FDA_TIER_CONFIG.get(self.tier, {})
        days = self.days_until
        return -config.get("window_after", 0) <= days <= config.get("window_before", 7)

    @property
    def boost(self) -> float:
        """Get the boost value if event is in window, else 0."""
        if self.is_in_window and self.tier:
            return FDA_TIER_CONFIG[self.tier]["boost"]
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/alerts."""
        return {
            "ticker": self.ticker,
            "company": self.company,
            "drug_name": self.drug_name,
            "event_type": self.event_type,
            "date": self.date.strftime("%Y-%m-%d"),
            "phase": self.phase,
            "indication": self.indication,
            "probability": round(self.probability, 2),
            "tier": self.tier.value if self.tier else None,
            "days_until": self.days_until,
            "boost": round(self.boost, 3),
            "source": self.source,
        }


# ============================
# FDA Calendar Engine
# ============================

# API endpoints (all free, no key required for OpenFDA and ClinicalTrials.gov)
OPENFDA_DRUG_APPROVALS_URL = "https://api.fda.gov/drug/drugsfda.json"
CLINICAL_TRIALS_V2_URL = "https://clinicaltrials.gov/api/v2/studies"
FINNHUB_FDA_CALENDAR_URL = "https://finnhub.io/api/v1/fda-advisory-committee-calendar"

# Historical approval rates by phase (FDA industry averages)
PHASE_APPROVAL_RATES = {
    "Phase 1": 0.14,
    "Phase 2": 0.29,
    "Phase 3": 0.58,
    "NDA": 0.85,
    "BLA": 0.85,
    "sNDA": 0.90,
}

# Indication-based probability adjustments (oncology is harder, rare disease easier)
INDICATION_PROBABILITY_MODIFIERS = {
    "oncology": -0.10,
    "cancer": -0.10,
    "rare disease": +0.10,
    "orphan": +0.10,
    "breakthrough": +0.15,
    "fast track": +0.10,
    "priority review": +0.10,
    "accelerated approval": +0.12,
}


class FDACalendarEngine:
    """
    V9 A5 — Complete FDA Calendar Engine.

    Integrates 3 real data sources:
    1. OpenFDA API — drug approvals, denials, CRLs (truly free, no key)
    2. ClinicalTrials.gov v2 API — Phase 2/3 trials with imminent results (free)
    3. Finnhub FDA calendar — PDUFA dates (free tier, uses FINNHUB_API_KEY)

    Plus the existing static calendar as fallback.

    Provides:
    - 5-tier FDA catalyst scoring
    - Probability estimation per event
    - Integration with Multi-Radar Engine (Catalyst Radar)
    - Backward-compatible get_fda_boost(ticker) interface
    """

    def __init__(self):
        self._cache = Cache(ttl=3600 * 4)  # 4h cache (FDA data changes slowly)
        self._events: List[FDAEvent] = []
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 3600 * 2  # Refresh every 2 hours
        self._lock = threading.Lock()
        self._finnhub_key = ""
        self._initialized = False

        # Load Finnhub API key
        try:
            from config import FINNHUB_API_KEY
            self._finnhub_key = FINNHUB_API_KEY
        except ImportError:
            import os
            self._finnhub_key = os.getenv("FINNHUB_API_KEY", "")

        logger.info("FDACalendarEngine initialized (OpenFDA + ClinicalTrials.gov + Finnhub)")

    # ============================
    # Data Fetching Methods
    # ============================

    async def fetch_openfda_approvals(self, limit: int = 100) -> List[FDAEvent]:
        """
        Fetch recent drug approvals, denials, and CRLs from OpenFDA API.

        OpenFDA is truly free with no API key required.
        Rate limit: 240 requests/min without key, 120,000/day with key.

        Returns:
            List of FDAEvent for recent FDA actions
        """
        cached = self._cache.get("openfda_approvals")
        if cached is not None:
            return cached

        events = []

        try:
            # Fetch recent submissions with approval dates in the last 90 days
            # and upcoming decisions
            now = datetime.now()
            date_90_ago = (now - timedelta(days=90)).strftime("%Y%m%d")
            date_30_ahead = (now + timedelta(days=30)).strftime("%Y%m%d")

            params = {
                "search": f'submissions.submission_status_date:[{date_90_ago}+TO+{date_30_ahead}]',
                "limit": limit,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPENFDA_DRUG_APPROVALS_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"OpenFDA returned {resp.status}")
                        self._cache.set("openfda_approvals", [], ttl=600)
                        return []

                    data = await resp.json()

            results = data.get("results", [])
            logger.info(f"OpenFDA returned {len(results)} drug records")

            for drug in results:
                try:
                    # Extract ticker from openfda section if available
                    openfda = drug.get("openfda", {})
                    brand_names = openfda.get("brand_name", [])
                    manufacturer = openfda.get("manufacturer_name", ["Unknown"])
                    generic_names = openfda.get("generic_name", [])

                    sponsor = drug.get("sponsor_name", manufacturer[0] if manufacturer else "Unknown")
                    drug_name = brand_names[0] if brand_names else (
                        generic_names[0] if generic_names else "Unknown"
                    )

                    # Process each submission for this drug
                    submissions = drug.get("submissions", [])
                    for sub in submissions:
                        sub_type = sub.get("submission_type", "")
                        sub_status = sub.get("submission_status", "")
                        status_date_str = sub.get("submission_status_date", "")

                        if not status_date_str:
                            continue

                        # Parse date (format: YYYYMMDD)
                        try:
                            event_date = datetime.strptime(status_date_str, "%Y%m%d")
                        except ValueError:
                            continue

                        # Determine event type and tier
                        event_type, tier = self._classify_openfda_action(sub_type, sub_status)
                        if not event_type:
                            continue

                        # Estimate probability based on submission type
                        prob = self._estimate_probability(
                            phase=sub_type, indication="", sub_status=sub_status
                        )

                        fda_event = FDAEvent(
                            ticker="",  # OpenFDA does not provide tickers directly
                            company=sponsor,
                            drug_name=drug_name,
                            event_type=event_type,
                            date=event_date,
                            phase=sub_type,
                            indication=", ".join(drug.get("products", [{}])[0].get("active_ingredients", [{}])[0].get("strength", "") if drug.get("products") else ""),
                            probability=prob,
                            historical_precedent=PHASE_APPROVAL_RATES.get(sub_type, 0.5),
                            tier=tier,
                            source="openfda",
                            raw_data={"submission": sub, "sponsor": sponsor},
                        )
                        events.append(fda_event)

                except Exception as e:
                    logger.debug(f"Error parsing OpenFDA drug record: {e}")
                    continue

            self._cache.set("openfda_approvals", events)
            logger.info(f"OpenFDA: parsed {len(events)} FDA action events")

        except asyncio.TimeoutError:
            logger.warning("OpenFDA API timeout")
            self._cache.set("openfda_approvals", [], ttl=600)
        except Exception as e:
            logger.error(f"OpenFDA fetch failed: {e}")
            self._cache.set("openfda_approvals", [], ttl=600)

        return events

    async def fetch_clinical_trials(self, phase: str = "Phase 3", max_results: int = 50) -> List[FDAEvent]:
        """
        Fetch Phase 2/3 clinical trials with imminent results from ClinicalTrials.gov v2 API.

        The v2 API is truly free with no key required.
        Focuses on trials expected to complete within 90 days.

        Args:
            phase: Trial phase to search for ("Phase 2" or "Phase 3")
            max_results: Maximum number of results

        Returns:
            List of FDAEvent for upcoming trial readouts
        """
        cache_key = f"clinical_trials_{phase.replace(' ', '_')}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        events = []

        try:
            now = datetime.now()
            date_future = (now + timedelta(days=90)).strftime("%Y-%m-%d")
            date_now = now.strftime("%Y-%m-%d")

            # ClinicalTrials.gov v2 API parameters
            # Filter for trials in target phase with primary completion date in near future
            phase_filter = "PHASE3" if "3" in phase else "PHASE2"

            params = {
                "format": "json",
                "query.term": "small cap OR biotech",
                "filter.overallStatus": "ACTIVE_NOT_RECRUITING,COMPLETED",
                "filter.phase": phase_filter,
                "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,"
                          "PrimaryCompletionDate,CompletionDate,LeadSponsorName,"
                          "Condition,InterventionName,StudyType",
                "sort": "PrimaryCompletionDate",
                "pageSize": max_results,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    CLINICAL_TRIALS_V2_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"ClinicalTrials.gov returned {resp.status}")
                        self._cache.set(cache_key, [], ttl=600)
                        return []

                    data = await resp.json()

            studies = data.get("studies", [])
            logger.info(f"ClinicalTrials.gov returned {len(studies)} {phase} studies")

            for study in studies:
                try:
                    proto = study.get("protocolSection", {})
                    id_mod = proto.get("identificationModule", {})
                    status_mod = proto.get("statusModule", {})
                    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
                    conditions_mod = proto.get("conditionsModule", {})
                    interventions_mod = proto.get("armsInterventionsModule", {})
                    design_mod = proto.get("designModule", {})

                    nct_id = id_mod.get("nctId", "")
                    title = id_mod.get("briefTitle", id_mod.get("officialTitle", ""))
                    sponsor_name = sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")

                    # Get completion date
                    completion_info = status_mod.get("primaryCompletionDateStruct", {})
                    if not completion_info:
                        completion_info = status_mod.get("completionDateStruct", {})

                    date_str = completion_info.get("date", "")
                    if not date_str:
                        continue

                    # Parse date (may be YYYY-MM-DD or YYYY-MM or YYYY)
                    event_date = self._parse_ct_date(date_str)
                    if not event_date:
                        continue

                    # Only include trials completing in the future or very recently
                    if event_date < now - timedelta(days=7):
                        continue

                    # Get conditions and interventions
                    conditions = conditions_mod.get("conditions", [])
                    indication = ", ".join(conditions[:3]) if conditions else ""

                    interventions = interventions_mod.get("interventions", [])
                    drug_name = ""
                    if interventions:
                        drug_name = interventions[0].get("name", title[:60])

                    # Get phases
                    phases = design_mod.get("phases", [])
                    phase_str = phases[0] if phases else phase

                    # Determine tier
                    tier = FDACatalystTier.TIER_2 if "3" in phase_str else FDACatalystTier.TIER_4

                    # Estimate probability
                    prob = self._estimate_probability(
                        phase=phase_str, indication=indication, sub_status=""
                    )

                    fda_event = FDAEvent(
                        ticker="",  # ClinicalTrials.gov does not provide tickers
                        company=sponsor_name,
                        drug_name=drug_name if drug_name else title[:60],
                        event_type="PHASE3_DATA" if "3" in phase_str else "PHASE2_TOPLINE",
                        date=event_date,
                        phase=phase_str,
                        indication=indication,
                        probability=prob,
                        historical_precedent=PHASE_APPROVAL_RATES.get(phase, 0.5),
                        tier=tier,
                        source="clinicaltrials",
                        raw_data={"nct_id": nct_id, "title": title},
                    )
                    events.append(fda_event)

                except Exception as e:
                    logger.debug(f"Error parsing ClinicalTrials study: {e}")
                    continue

            self._cache.set(cache_key, events)
            logger.info(f"ClinicalTrials.gov: parsed {len(events)} {phase} events")

        except asyncio.TimeoutError:
            logger.warning("ClinicalTrials.gov API timeout")
            self._cache.set(cache_key, [], ttl=600)
        except Exception as e:
            logger.error(f"ClinicalTrials.gov fetch failed: {e}")
            self._cache.set(cache_key, [], ttl=600)

        return events

    async def fetch_pdufa_dates(self) -> List[FDAEvent]:
        """
        Fetch PDUFA dates from Finnhub FDA advisory committee calendar.

        Uses FINNHUB_API_KEY from config. Falls back to existing static
        scrape_pdufa_dates() if Finnhub is unavailable.

        Returns:
            List of FDAEvent for upcoming PDUFA dates
        """
        cached = self._cache.get("pdufa_dates_v9")
        if cached is not None:
            return cached

        events = []

        # --- Source 1: Finnhub FDA calendar ---
        if self._finnhub_key:
            try:
                now = datetime.now()
                params = {
                    "from": now.strftime("%Y-%m-%d"),
                    "to": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
                    "token": self._finnhub_key,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        FINNHUB_FDA_CALENDAR_URL,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            fda_items = data if isinstance(data, list) else data.get("result", [])

                            for item in fda_items:
                                try:
                                    date_str = item.get("fromDate", item.get("date", ""))
                                    if not date_str:
                                        continue

                                    event_date = datetime.strptime(date_str, "%Y-%m-%d")

                                    # Classify event type
                                    event_name = item.get("eventDescription", item.get("name", "")).lower()
                                    tier = FDACatalystTier.TIER_1  # Default: PDUFA
                                    event_type = "PDUFA"

                                    if "adcom" in event_name or "advisory" in event_name:
                                        tier = FDACatalystTier.TIER_3
                                        event_type = "ADCOM"

                                    ticker = item.get("symbol", item.get("ticker", "")).upper()
                                    company = item.get("companyName", item.get("company", ""))
                                    drug = item.get("drugName", item.get("drug", event_name[:60]))

                                    fda_event = FDAEvent(
                                        ticker=ticker,
                                        company=company,
                                        drug_name=drug,
                                        event_type=event_type,
                                        date=event_date,
                                        phase="NDA",
                                        indication=item.get("indication", ""),
                                        probability=0.75 if tier == FDACatalystTier.TIER_1 else 0.60,
                                        historical_precedent=0.85,
                                        tier=tier,
                                        source="finnhub",
                                        raw_data=item,
                                    )
                                    events.append(fda_event)

                                except Exception as e:
                                    logger.debug(f"Error parsing Finnhub FDA item: {e}")
                                    continue

                            logger.info(f"Finnhub FDA calendar: {len(events)} events")
                        else:
                            logger.warning(f"Finnhub FDA calendar returned {resp.status}")

            except asyncio.TimeoutError:
                logger.warning("Finnhub FDA calendar timeout")
            except Exception as e:
                logger.warning(f"Finnhub FDA calendar failed: {e}")

        # --- Source 2: Fallback to existing static scraper ---
        try:
            static_pdufa = scrape_pdufa_dates()
            for item in static_pdufa:
                # Skip duplicates (same ticker + same date)
                ticker = item.get("ticker", "").upper()
                date_str = item.get("date", "")
                if not date_str:
                    continue

                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                is_dup = any(
                    e.ticker == ticker and e.date.date() == event_date.date()
                    for e in events
                )
                if is_dup:
                    continue

                fda_event = FDAEvent(
                    ticker=ticker,
                    company="",
                    drug_name=item.get("drug_name", ""),
                    event_type="PDUFA",
                    date=event_date,
                    phase="NDA",
                    indication=item.get("indication", ""),
                    probability=0.70,
                    historical_precedent=0.85,
                    tier=FDACatalystTier.TIER_1,
                    source="static",
                    raw_data=item,
                )
                events.append(fda_event)

        except Exception as e:
            logger.debug(f"Static PDUFA fallback failed: {e}")

        self._cache.set("pdufa_dates_v9", events)
        logger.info(f"Total PDUFA events: {len(events)}")

        return events

    # ============================
    # Refresh & Aggregation
    # ============================

    async def refresh_all(self) -> List[FDAEvent]:
        """
        Refresh all FDA data sources in parallel.

        Merges events from OpenFDA, ClinicalTrials.gov (Phase 2 + Phase 3),
        and Finnhub/static PDUFA calendar.

        Returns:
            Complete list of FDAEvent
        """
        import time as _time
        now = _time.time()

        # Rate limit refreshes
        if now - self._last_refresh < self._refresh_interval and self._events:
            return self._events

        try:
            # Fetch all sources in parallel
            results = await asyncio.gather(
                self.fetch_openfda_approvals(),
                self.fetch_clinical_trials(phase="Phase 3"),
                self.fetch_clinical_trials(phase="Phase 2"),
                self.fetch_pdufa_dates(),
                return_exceptions=True,
            )

            all_events = []
            source_names = ["OpenFDA", "Phase3_Trials", "Phase2_Trials", "PDUFA"]

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"FDA source {source_names[i]} failed: {result}")
                    continue
                if isinstance(result, list):
                    all_events.extend(result)

            # Sort by date (soonest first)
            all_events.sort(key=lambda e: e.date)

            with self._lock:
                self._events = all_events
                self._last_refresh = now
                self._initialized = True

            logger.info(
                f"FDA Calendar refreshed: {len(all_events)} total events "
                f"(OpenFDA={len(results[0]) if not isinstance(results[0], Exception) else 0}, "
                f"P3={len(results[1]) if not isinstance(results[1], Exception) else 0}, "
                f"P2={len(results[2]) if not isinstance(results[2], Exception) else 0}, "
                f"PDUFA={len(results[3]) if not isinstance(results[3], Exception) else 0})"
            )

            return all_events

        except Exception as e:
            logger.error(f"FDA Calendar refresh_all failed: {e}")
            return self._events

    def _ensure_initialized(self):
        """Ensure at least one refresh has been done. Non-blocking if already initialized."""
        if self._initialized and self._events:
            return

        try:
            loop = asyncio.get_running_loop()
            # Already in async context — schedule but do not block
            asyncio.ensure_future(self.refresh_all())
        except RuntimeError:
            # No running loop — create one and run synchronously
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.refresh_all())
                loop.close()
            except Exception as e:
                logger.warning(f"FDA Calendar sync init failed: {e}")

    # ============================
    # Scoring Methods
    # ============================

    def score_fda_catalyst(self, ticker: str) -> float:
        """
        Compute a 0-1 FDA catalyst score for a given ticker.

        The score is the maximum boost from all matching events
        that fall within their active window, weighted by probability.

        Args:
            ticker: Stock ticker symbol (e.g. "MRNA")

        Returns:
            float: FDA catalyst score (0.0 to 1.0)
        """
        self._ensure_initialized()

        ticker = ticker.upper().strip()
        if not ticker:
            return 0.0

        # Get events for this ticker
        ticker_events = self._get_ticker_events(ticker)
        if not ticker_events:
            return 0.0

        max_score = 0.0

        for event in ticker_events:
            if not event.is_in_window:
                continue

            # Base boost from tier
            boost = event.boost

            # Weight by probability
            weighted_boost = boost * max(0.3, event.probability)

            max_score = max(max_score, weighted_boost)

        return min(1.0, max_score)

    def get_fda_boost(self, ticker: str) -> Tuple[float, Dict[str, Any]]:
        """
        Get the FDA boost and details for a ticker.

        This is the primary interface for Multi-Radar Engine integration.
        Returns a tuple of (boost_value, details_dict).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (boost: float, details: dict)
            - boost: 0.0 to 0.25 (max tier boost)
            - details: event information for logging/alerts
        """
        self._ensure_initialized()

        ticker = ticker.upper().strip()
        if not ticker:
            return 0.0, {}

        ticker_events = self._get_ticker_events(ticker)
        if not ticker_events:
            return 0.0, {}

        best_boost = 0.0
        best_event = None

        for event in ticker_events:
            if not event.is_in_window:
                continue

            boost = event.boost * max(0.3, event.probability)
            if boost > best_boost:
                best_boost = boost
                best_event = event

        if best_event is None:
            return 0.0, {}

        details = {
            "event_type": best_event.event_type,
            "drug_name": best_event.drug_name,
            "date": best_event.date.strftime("%Y-%m-%d"),
            "days_until": best_event.days_until,
            "tier": best_event.tier.value if best_event.tier else None,
            "tier_label": FDA_TIER_CONFIG[best_event.tier]["label"] if best_event.tier else "",
            "probability": round(best_event.probability, 2),
            "boost": round(best_boost, 3),
            "source": best_event.source,
            "company": best_event.company,
            "indication": best_event.indication,
        }

        return best_boost, details

    def get_upcoming_catalysts(self, days: int = 30) -> List[FDAEvent]:
        """
        Get all FDA catalyst events within the next N days.

        Args:
            days: Number of days ahead to look (default 30)

        Returns:
            List of FDAEvent sorted by date
        """
        self._ensure_initialized()

        cutoff = datetime.now() + timedelta(days=days)
        now = datetime.now()

        upcoming = [
            e for e in self._events
            if now.date() <= e.date.date() <= cutoff.date()
        ]

        # Sort by date, then by tier priority (TIER_1 first)
        tier_priority = {
            FDACatalystTier.TIER_1: 0,
            FDACatalystTier.TIER_2: 1,
            FDACatalystTier.TIER_3: 2,
            FDACatalystTier.TIER_4: 3,
            FDACatalystTier.TIER_5: 4,
        }

        upcoming.sort(key=lambda e: (
            e.date,
            tier_priority.get(e.tier, 5),
        ))

        return upcoming

    def get_events_for_ticker(self, ticker: str) -> List[FDAEvent]:
        """
        Get all FDA events for a specific ticker (past and future).

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of FDAEvent for this ticker
        """
        self._ensure_initialized()
        return self._get_ticker_events(ticker)

    # ============================
    # Internal Helpers
    # ============================

    def _get_ticker_events(self, ticker: str) -> List[FDAEvent]:
        """
        Find events matching a ticker.

        Matches by ticker symbol (exact) or company name (fuzzy substring).
        """
        ticker = ticker.upper().strip()
        matches = []

        with self._lock:
            for event in self._events:
                # Exact ticker match (primary)
                if event.ticker and event.ticker.upper() == ticker:
                    matches.append(event)
                    continue

                # Company name match (secondary, for OpenFDA/ClinicalTrials without tickers)
                # This requires external ticker-to-company mapping — skip for now
                # unless ticker is embedded in raw_data

        return matches

    def _classify_openfda_action(
        self, sub_type: str, sub_status: str
    ) -> Tuple[Optional[str], Optional[FDACatalystTier]]:
        """
        Classify an OpenFDA submission into an event type and tier.

        Args:
            sub_type: Submission type (e.g. "NDA", "BLA", "ANDA", "SUPPL")
            sub_status: Status (e.g. "AP" = approved, "TA" = tentative approval)

        Returns:
            Tuple of (event_type, tier) or (None, None) if not relevant
        """
        status_lower = sub_status.lower() if sub_status else ""
        type_lower = sub_type.lower() if sub_type else ""

        # Approved
        if status_lower in ("ap", "approved"):
            return "APPROVAL", FDACatalystTier.TIER_5

        # Complete Response Letter (denial / more data needed)
        if "crl" in status_lower or "complete response" in status_lower:
            return "CRL", FDACatalystTier.TIER_5

        # Tentative approval
        if status_lower in ("ta",):
            return "APPROVAL", FDACatalystTier.TIER_5

        # NDA/BLA filed = PDUFA upcoming
        if type_lower in ("nda", "bla") and status_lower in ("", "pending", "submitted"):
            return "PDUFA", FDACatalystTier.TIER_1

        return None, None

    def _estimate_probability(
        self, phase: str, indication: str, sub_status: str
    ) -> float:
        """
        Estimate the probability of a positive FDA outcome.

        Uses historical approval rates by phase, adjusted for indication type
        and any special designations.

        Args:
            phase: Trial phase or submission type
            indication: Disease/condition description
            sub_status: Submission status

        Returns:
            Estimated probability (0.0 to 1.0)
        """
        # Base probability from phase
        base_prob = PHASE_APPROVAL_RATES.get(phase, 0.50)

        # Already decided
        if sub_status:
            status_lower = sub_status.lower()
            if status_lower in ("ap", "approved"):
                return 1.0
            if "crl" in status_lower or "refuse" in status_lower:
                return 0.0

        # Adjust for indication
        indication_lower = indication.lower() if indication else ""
        modifier = 0.0
        for keyword, mod in INDICATION_PROBABILITY_MODIFIERS.items():
            if keyword in indication_lower:
                modifier = max(modifier, mod) if mod > 0 else min(modifier, mod)

        prob = base_prob + modifier
        return max(0.05, min(0.95, prob))

    def _parse_ct_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse ClinicalTrials.gov date formats.

        Handles: YYYY-MM-DD, YYYY-MM, YYYY, Month YYYY, Month Day, YYYY
        """
        if not date_str:
            return None

        # Try standard formats
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # Try "Month YYYY" format (e.g. "December 2026")
        try:
            return datetime.strptime(date_str.strip(), "%B %Y")
        except ValueError:
            pass

        # Try "Month Day, YYYY" format (e.g. "December 15, 2026")
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except ValueError:
            pass

        return None

    # ============================
    # Stats & Diagnostics
    # ============================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics for monitoring."""
        with self._lock:
            total = len(self._events)
            by_source = {}
            by_tier = {}
            in_window = 0

            for event in self._events:
                by_source[event.source] = by_source.get(event.source, 0) + 1
                tier_name = event.tier.value if event.tier else "NONE"
                by_tier[tier_name] = by_tier.get(tier_name, 0) + 1
                if event.is_in_window:
                    in_window += 1

            return {
                "total_events": total,
                "in_window": in_window,
                "by_source": by_source,
                "by_tier": by_tier,
                "initialized": self._initialized,
                "last_refresh": datetime.fromtimestamp(self._last_refresh).isoformat()
                if self._last_refresh > 0 else "never",
            }


# ============================
# Singleton
# ============================

_fda_calendar_engine: Optional[FDACalendarEngine] = None
_fda_engine_lock = threading.Lock()


def get_fda_calendar_engine() -> FDACalendarEngine:
    """
    Get or create the singleton FDACalendarEngine instance.

    Thread-safe singleton following the project convention
    (same pattern as get_ibkr(), get_signal_producer(), get_multi_radar_engine()).

    Returns:
        FDACalendarEngine singleton instance
    """
    global _fda_calendar_engine

    if _fda_calendar_engine is None:
        with _fda_engine_lock:
            if _fda_calendar_engine is None:
                _fda_calendar_engine = FDACalendarEngine()
                logger.info("FDACalendarEngine singleton created")

    return _fda_calendar_engine


# ============================
# Backward-Compatible Interface
# ============================
# These functions are used by multi_radar_engine.py (Catalyst Radar)
# and can be called from anywhere that imported the old interface.

def get_fda_events(ticker: str) -> List[Dict[str, Any]]:
    """
    Get FDA events for a specific ticker (backward-compatible interface).

    Used by: src/engines/multi_radar_engine.py (CatalystRadar, line ~580)

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of event dicts (backward-compatible format)
    """
    try:
        engine = get_fda_calendar_engine()
        fda_events = engine.get_events_for_ticker(ticker)

        if fda_events:
            return [e.to_dict() for e in fda_events]

        # Fallback: also check the legacy get_fda_events_by_ticker
        legacy = get_fda_events_by_ticker(ticker)
        return legacy

    except Exception as e:
        logger.debug(f"get_fda_events({ticker}) error: {e}")
        # Final fallback to legacy function
        try:
            return get_fda_events_by_ticker(ticker)
        except Exception:
            return []


def get_fda_boost(ticker: str) -> float:
    """
    Get the FDA boost value for a ticker (backward-compatible interface).

    Used by: src/engines/multi_radar_engine.py (CatalystRadar, line ~583)

    Returns the boost value only (not the details dict). For the full
    interface with details, use get_fda_calendar_engine().get_fda_boost(ticker).

    Args:
        ticker: Stock ticker symbol

    Returns:
        float: FDA boost value (0.0 to 0.25)
    """
    try:
        engine = get_fda_calendar_engine()
        boost, _details = engine.get_fda_boost(ticker)
        return boost
    except Exception as e:
        logger.debug(f"get_fda_boost({ticker}) error: {e}")
        return 0.0


# ============================
# Main (test/demo)
# ============================

if __name__ == "__main__":
    print("\n FDA CALENDAR SCRAPER TEST")
    print("=" * 60)

    # Test PDUFA
    print("\n PDUFA DATES:")
    pdufa = scrape_pdufa_dates()
    for event in pdufa[:5]:
        print(f"  {event['ticker']}: {event['drug_name']} - {event['date']}")

    # Test trials
    print("\n CLINICAL TRIALS:")
    trials = scrape_trial_results()
    for event in trials[:5]:
        print(f"  {event['ticker']}: {event['phase']} - {event['date']}")

    # Test conferences
    print("\n BIOTECH CONFERENCES:")
    conferences = get_biotech_conferences()
    for conf in conferences:
        print(f"  {conf['name']}: {conf['start_date']} - {conf['location']}")

    # Summary
    all_events = get_all_fda_events()
    print(f"\n TOTAL FDA EVENTS (legacy): {len(all_events)}")

    # V9 FDACalendarEngine test
    print("\n" + "=" * 60)
    print(" V9 FDA CALENDAR ENGINE TEST")
    print("=" * 60)

    async def test_engine():
        engine = get_fda_calendar_engine()
        events = await engine.refresh_all()
        print(f"\n Total V9 events: {len(events)}")

        # Show upcoming catalysts
        upcoming = engine.get_upcoming_catalysts(days=30)
        print(f" Upcoming (30 days): {len(upcoming)}")
        for e in upcoming[:10]:
            print(f"  [{e.tier.value if e.tier else '?'}] {e.ticker or e.company[:20]}: "
                  f"{e.drug_name[:30]} - {e.date.strftime('%Y-%m-%d')} "
                  f"(prob={e.probability:.0%}, boost={e.boost:.3f})")

        # Stats
        stats = engine.get_stats()
        print(f"\n Engine stats: {stats}")

    asyncio.run(test_engine())
