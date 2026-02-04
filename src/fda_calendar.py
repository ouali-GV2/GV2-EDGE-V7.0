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


if __name__ == "__main__":
    print("\n🧬 FDA CALENDAR SCRAPER TEST")
    print("=" * 60)
    
    # Test PDUFA
    print("\n📅 PDUFA DATES:")
    pdufa = scrape_pdufa_dates()
    for event in pdufa[:5]:
        print(f"  {event['ticker']}: {event['drug_name']} - {event['date']}")
    
    # Test trials
    print("\n🔬 CLINICAL TRIALS:")
    trials = scrape_trial_results()
    for event in trials[:5]:
        print(f"  {event['ticker']}: {event['phase']} - {event['date']}")
    
    # Test conferences
    print("\n🏛️ BIOTECH CONFERENCES:")
    conferences = get_biotech_conferences()
    for conf in conferences:
        print(f"  {conf['name']}: {conf['start_date']} - {conf['location']}")
    
    # Summary
    all_events = get_all_fda_events()
    print(f"\n📊 TOTAL FDA EVENTS: {len(all_events)}")
