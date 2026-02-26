"""
SEC FILINGS INGESTOR V6.1
=========================

Sources 100% reelles (GRATUIT):
- SEC EDGAR 8-K: Evenements materiels (FDA, contrats, M&A, earnings)
- SEC EDGAR Form 4: Insider trading (boost signal)

API: RSS feeds + Full-text search (aucune cle API requise)
Rate limit: 10 req/sec (SEC est genereux)

Architecture:
- fetch_8k_filings(): Recupere les 8-K recents
- fetch_form4_filings(): Recupere les Form 4 pour un ticker
- categorize_8k(): Map 8-K items vers EVENT_TYPE taxonomy
- parse_8k_content(): Extrait details du filing
"""

import os
import re
import asyncio
import aiohttp
import feedparser
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from xml.etree import ElementTree as ET

from utils.logger import get_logger

logger = get_logger("SEC_INGESTOR")

# ============================
# Configuration
# ============================

SEC_RSS_8K = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=100&output=atom"
SEC_RSS_FORM4 = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom"
SEC_COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count={count}&output=atom"
SEC_FULLTEXT_API = "https://efts.sec.gov/LATEST/search-index"

# CIK to Ticker mapping cache
CIK_CACHE_DB = "data/cik_mapping.db"

# 8-K Item Types -> EVENT_TYPE mapping
SEC_8K_ITEM_MAP = {
    "1.01": "MAJOR_CONTRACT",           # Entry into Material Agreement
    "1.02": "BANKRUPTCY",               # Bankruptcy/Receivership
    "1.03": "BANKRUPTCY",               # Bankruptcy (other)
    "2.01": "MERGER_ACQUISITION",       # Acquisition/Disposition of Assets
    "2.02": "EARNINGS_BEAT",            # Results of Operations
    "2.03": "MAJOR_CONTRACT",           # Creation of Direct Financial Obligation
    "2.04": "RESTRUCTURING",            # Triggering Events
    "2.05": "RESTRUCTURING",            # Costs Associated with Exit
    "2.06": "IMPAIRMENT",               # Material Impairments
    "3.01": "DELISTING",                # Notice of Delisting
    "3.02": "UNREGISTERED_SALE",        # Unregistered Sales of Equity
    "3.03": "CHARTER_AMENDMENT",        # Material Modification of Rights
    "4.01": "AUDITOR_CHANGE",           # Changes in Registrant's Certifying Accountant
    "4.02": "FINANCIAL_RESTATEMENT",    # Non-Reliance on Previously Issued Financials
    "5.01": "CONTROL_CHANGE",           # Changes in Control
    "5.02": "MANAGEMENT_CHANGE",        # Departure/Election of Directors/Officers
    "5.03": "CHARTER_AMENDMENT",        # Amendments to Articles
    "5.04": "SHAREHOLDER_VOTE",         # Temporary Suspension of Trading
    "5.05": "CHARTER_AMENDMENT",        # Amendments to Code of Ethics
    "5.06": "SHELL_COMPANY",            # Change in Shell Company Status
    "5.07": "SHAREHOLDER_VOTE",         # Submission of Matters to Shareholders
    "5.08": "FISCAL_YEAR_CHANGE",       # Shareholder Director Nominations
    "6.01": "ABS_INFO",                 # ABS Informational
    "6.02": "ABS_INFO",                 # Change of Servicer
    "6.03": "ABS_INFO",                 # Change in Credit Enhancement
    "6.04": "ABS_INFO",                 # Failure to Make Distribution
    "6.05": "ABS_INFO",                 # Securities Act Updating Disclosure
    "7.01": "REG_FD",                   # Regulation FD Disclosure
    "8.01": "OTHER_EVENTS",             # Other Events
    "9.01": "FINANCIAL_EXHIBIT",        # Financial Statements and Exhibits
}

# FDA-related keywords for 8-K content analysis
FDA_KEYWORDS = [
    r"\bFDA\s+approv", r"\bFDA\s+clear", r"\bPDUFA\b", r"\bNDA\b", r"\bBLA\b",
    r"\b510\(k\)", r"\bbreakthrough\s+therapy", r"\bfast\s+track",
    r"\bpriority\s+review", r"\bcomplete\s+response", r"\bCRL\b",
    r"\bphase\s+(2|II|3|III)", r"\bclinical\s+trial", r"\bdrug\s+approv"
]

# Positive trial keywords
POSITIVE_TRIAL_KEYWORDS = [
    r"\bpositive\b.*\b(result|data|outcome)", r"\bmet\b.*\bendpoint",
    r"\bsuccess", r"\befficacy\b.*\bdemonstrat", r"\bstatistically\s+significant"
]


# ============================
# Data Classes
# ============================

@dataclass
class SECFiling:
    """Represents a SEC filing"""
    accession_number: str
    form_type: str
    cik: str
    ticker: Optional[str]
    company_name: str
    filed_date: datetime
    accepted_date: datetime
    items: List[str]  # For 8-K: list of item numbers
    event_type: Optional[str]
    event_impact: float
    headline: str
    summary: str
    url: str
    source: str = "sec_edgar"
    source_priority: int = 1  # Level 1 - Critical


@dataclass
class InsiderTransaction:
    """Represents a Form 4 insider transaction"""
    accession_number: str
    cik: str
    ticker: Optional[str]
    company_name: str
    insider_name: str
    insider_title: str
    transaction_date: datetime
    transaction_code: str  # P=Purchase, S=Sale, M=Exercise
    shares: int
    price: float
    value: float
    ownership_type: str  # D=Direct, I=Indirect
    shares_after: int
    url: str


# ============================
# CIK Mapping
# ============================

class CIKMapper:
    """Maps CIK numbers to ticker symbols"""

    def __init__(self):
        self._init_db()
        self._load_sec_mapping()

    def _init_db(self):
        """Initialize SQLite cache"""
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(CIK_CACHE_DB, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cik_mapping (
                cik TEXT PRIMARY KEY,
                ticker TEXT,
                company_name TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def _load_sec_mapping(self):
        """Load SEC's official CIK-ticker mapping"""
        # SEC provides a JSON file with all mappings
        # https://www.sec.gov/files/company_tickers.json
        try:
            import requests
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                cursor = self.conn.cursor()
                for item in data.values():
                    cik = str(item.get("cik_str", "")).zfill(10)
                    ticker = item.get("ticker", "")
                    name = item.get("title", "")
                    cursor.execute("""
                        INSERT OR REPLACE INTO cik_mapping (cik, ticker, company_name, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (cik, ticker, name, datetime.utcnow().isoformat()))
                self.conn.commit()
                logger.info(f"Loaded {len(data)} CIK mappings from SEC")
        except Exception as e:
            logger.warning(f"Failed to load SEC CIK mapping: {e}")

    def get_ticker(self, cik: str) -> Optional[str]:
        """Get ticker for CIK"""
        cik = cik.zfill(10)
        cursor = self.conn.cursor()
        cursor.execute("SELECT ticker FROM cik_mapping WHERE cik = ?", (cik,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for ticker"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT cik FROM cik_mapping WHERE ticker = ?", (ticker.upper(),))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_ticker_by_company(self, company_name: str) -> Optional[str]:
        """
        Fuzzy lookup: find ticker by company name.

        S3-5 FIX: OpenFDA and ClinicalTrials.gov return company/sponsor names,
        not tickers. This method resolves names → tickers via the SEC EDGAR
        CIK mapping (loaded from https://www.sec.gov/files/company_tickers.json).

        Strategy:
          1. Exact match (case-insensitive)
          2. Starts-with match (handles "Pfizer Inc" → "Pfizer")
          3. Contains match (partial, last resort)

        Returns:
            Ticker string or None if not found
        """
        if not company_name:
            return None
        name_upper = company_name.upper().strip()
        cursor = self.conn.cursor()

        # Exact match
        cursor.execute(
            "SELECT ticker FROM cik_mapping WHERE UPPER(company_name) = ? LIMIT 1",
            (name_upper,)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        # Starts-with (handles "Pfizer Inc." → "PFIZER")
        cursor.execute(
            "SELECT ticker FROM cik_mapping WHERE UPPER(company_name) LIKE ? LIMIT 1",
            (name_upper + "%",)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        # Contains (last resort — pick shortest match to avoid false positives)
        cursor.execute(
            "SELECT ticker, company_name FROM cik_mapping WHERE UPPER(company_name) LIKE ? LIMIT 5",
            ("%" + name_upper + "%",)
        )
        rows = cursor.fetchall()
        if rows:
            # Prefer the shortest company_name (closest match)
            rows.sort(key=lambda r: len(r[1]))
            return rows[0][0]

        return None


# Global mapper instance
_cik_mapper = None

def get_cik_mapper() -> CIKMapper:
    global _cik_mapper
    if _cik_mapper is None:
        _cik_mapper = CIKMapper()
    return _cik_mapper


# ============================
# 8-K Filings Fetcher
# ============================

class SEC8KIngestor:
    """Fetches and parses SEC 8-K filings"""

    def __init__(self, universe: set = None):
        self.universe = universe or set()
        self.cik_mapper = get_cik_mapper()
        self.processed_accessions = set()
        self._load_processed()

    def _load_processed(self):
        """Load previously processed accession numbers"""
        cache_file = "data/sec_8k_processed.txt"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.processed_accessions = set(f.read().splitlines())

    def _save_processed(self, accession: str):
        """Save processed accession number"""
        self.processed_accessions.add(accession)
        cache_file = "data/sec_8k_processed.txt"
        with open(cache_file, "a") as f:
            f.write(f"{accession}\n")

    async def fetch_recent_8k(self, hours_back: int = 2) -> List[SECFiling]:
        """
        Fetch recent 8-K filings from SEC RSS feed

        Args:
            hours_back: How many hours back to fetch

        Returns:
            List of SECFiling objects
        """
        filings = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(SEC_RSS_8K, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"SEC RSS error: {resp.status}")
                        return []

                    content = await resp.text()
                    feed = feedparser.parse(content)

                    for entry in feed.entries:
                        filing = self._parse_8k_entry(entry)

                        if filing is None:
                            continue

                        # Skip if already processed
                        if filing.accession_number in self.processed_accessions:
                            continue

                        # Skip if too old
                        if filing.filed_date < cutoff:
                            continue

                        # Skip if not in universe (if universe is set)
                        if self.universe and filing.ticker and filing.ticker not in self.universe:
                            continue

                        # Categorize the filing
                        filing = await self._enrich_filing(filing)

                        filings.append(filing)
                        self._save_processed(filing.accession_number)

                    logger.info(f"Fetched {len(filings)} new 8-K filings")

        except Exception as e:
            logger.error(f"Error fetching 8-K filings: {e}")

        return filings

    def _parse_8k_entry(self, entry: dict) -> Optional[SECFiling]:
        """Parse RSS entry into SECFiling"""
        try:
            # Extract accession number from link
            link = entry.get("link", "")
            accession_match = re.search(r"(\d{10}-\d{2}-\d{6})", link)
            if not accession_match:
                return None

            accession = accession_match.group(1)

            # Parse title for company and form type
            title = entry.get("title", "")
            # Format: "8-K - Company Name (CIK)"
            title_match = re.match(r"([\d\w-]+)\s*-\s*(.+?)\s*\((\d+)\)", title)

            if title_match:
                form_type = title_match.group(1)
                company_name = title_match.group(2).strip()
                cik = title_match.group(3).zfill(10)
            else:
                form_type = "8-K"
                company_name = title
                cik = ""

            # Get ticker from CIK
            ticker = self.cik_mapper.get_ticker(cik) if cik else None

            # Parse dates
            updated = entry.get("updated", "")
            try:
                filed_date = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
            except:
                filed_date = datetime.utcnow()

            # Extract items from summary (if available)
            summary = entry.get("summary", "")
            items = self._extract_items(summary)

            return SECFiling(
                accession_number=accession,
                form_type=form_type,
                cik=cik,
                ticker=ticker,
                company_name=company_name,
                filed_date=filed_date,
                accepted_date=filed_date,
                items=items,
                event_type=None,
                event_impact=0.0,
                headline=f"{company_name} filed {form_type}",
                summary=summary[:500] if summary else "",
                url=link
            )

        except Exception as e:
            logger.warning(f"Error parsing 8-K entry: {e}")
            return None

    def _extract_items(self, text: str) -> List[str]:
        """Extract 8-K item numbers from text"""
        items = []
        # Match patterns like "Item 1.01" or "1.01"
        matches = re.findall(r"(?:Item\s+)?(\d+\.\d{2})", text, re.IGNORECASE)
        for match in matches:
            if match in SEC_8K_ITEM_MAP:
                items.append(match)
        return list(set(items))

    async def _enrich_filing(self, filing: SECFiling) -> SECFiling:
        """Enrich filing with event type and impact"""

        # Determine event type from items
        event_type = self._categorize_filing(filing)
        filing.event_type = event_type

        # Set impact based on event type
        filing.event_impact = self._get_impact(event_type)

        # Update headline
        if event_type:
            filing.headline = f"{filing.company_name}: {event_type.replace('_', ' ').title()}"

        return filing

    def _categorize_filing(self, filing: SECFiling) -> Optional[str]:
        """
        Categorize 8-K filing into EVENT_TYPE taxonomy

        Logic:
        1. Check items first (most reliable)
        2. Check content for FDA keywords
        3. Fall back to general categorization
        """
        text = f"{filing.headline} {filing.summary}".lower()

        # Check for FDA-related content (highest priority)
        for pattern in FDA_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if positive result
                for pos_pattern in POSITIVE_TRIAL_KEYWORDS:
                    if re.search(pos_pattern, text, re.IGNORECASE):
                        return "FDA_TRIAL_POSITIVE"

                if re.search(r"approv|clear", text, re.IGNORECASE):
                    return "FDA_APPROVAL"
                elif re.search(r"breakthrough", text, re.IGNORECASE):
                    return "BREAKTHROUGH_DESIGNATION"
                elif re.search(r"fast\s+track", text, re.IGNORECASE):
                    return "FDA_FAST_TRACK"
                elif re.search(r"PDUFA", text, re.IGNORECASE):
                    return "PDUFA_DECISION"
                else:
                    return "FDA_TRIAL_POSITIVE"

        # Check items
        for item in filing.items:
            if item in SEC_8K_ITEM_MAP:
                mapped = SEC_8K_ITEM_MAP[item]

                # Refine based on content
                if mapped == "EARNINGS_BEAT" and "beat" in text:
                    if re.search(r"(big|significant|strong|exceed)", text):
                        return "EARNINGS_BEAT_BIG"
                    return "EARNINGS_BEAT"

                if mapped == "MERGER_ACQUISITION":
                    if "buyout" in text or "acquire" in text:
                        if "confirmed" in text or "definitive" in text:
                            return "BUYOUT_CONFIRMED"
                        return "MERGER_ACQUISITION"

                if mapped == "MAJOR_CONTRACT":
                    # Check contract value
                    value_match = re.search(r"\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)", text, re.IGNORECASE)
                    if value_match:
                        value = float(value_match.group(1))
                        unit = value_match.group(2).lower()
                        if unit in ("billion", "b"):
                            value *= 1000
                        if value >= 50:  # $50M+ = MAJOR_CONTRACT
                            return "MAJOR_CONTRACT"

                return mapped

        # Default for 8-K with no specific items
        if "7.01" in filing.items or "8.01" in filing.items:
            return "OTHER_EVENTS"

        return None

    def _get_impact(self, event_type: Optional[str]) -> float:
        """Get impact score for event type"""
        impact_map = {
            # TIER 1
            "FDA_APPROVAL": 0.95,
            "PDUFA_DECISION": 0.92,
            "BUYOUT_CONFIRMED": 0.95,
            # TIER 2
            "FDA_TRIAL_POSITIVE": 0.85,
            "BREAKTHROUGH_DESIGNATION": 0.82,
            "FDA_FAST_TRACK": 0.80,
            "MERGER_ACQUISITION": 0.82,
            "EARNINGS_BEAT_BIG": 0.80,
            "MAJOR_CONTRACT": 0.78,
            # TIER 3
            "GUIDANCE_RAISE": 0.70,
            "EARNINGS_BEAT": 0.65,
            "PARTNERSHIP": 0.65,
            # TIER 4+
            "MANAGEMENT_CHANGE": 0.50,
            "RESTRUCTURING": 0.45,
            "OTHER_EVENTS": 0.40,
        }
        return impact_map.get(event_type, 0.35)


# ============================
# Form 4 (Insider) Fetcher
# ============================

class SECForm4Ingestor:
    """Fetches and parses SEC Form 4 insider transactions"""

    def __init__(self):
        self.cik_mapper = get_cik_mapper()

    async def fetch_insider_transactions(
        self,
        ticker: str,
        days_back: int = 30
    ) -> List[InsiderTransaction]:
        """
        Fetch Form 4 filings for a specific ticker

        Args:
            ticker: Stock symbol
            days_back: How many days back to fetch

        Returns:
            List of InsiderTransaction objects
        """
        cik = self.cik_mapper.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return []

        transactions = []
        cutoff = datetime.utcnow() - timedelta(days=days_back)

        try:
            url = SEC_COMPANY_SEARCH.format(cik=cik, form_type="4", count=40)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"SEC Form 4 error: {resp.status}")
                        return []

                    content = await resp.text()
                    feed = feedparser.parse(content)

                    for entry in feed.entries:
                        txn = await self._parse_form4_entry(entry, ticker)

                        if txn and txn.transaction_date >= cutoff:
                            transactions.append(txn)

                    logger.info(f"Fetched {len(transactions)} Form 4 transactions for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching Form 4 for {ticker}: {e}")

        return transactions

    async def _parse_form4_entry(
        self,
        entry: dict,
        ticker: str
    ) -> Optional[InsiderTransaction]:
        """Parse RSS entry into InsiderTransaction by fetching and parsing the XML filing."""
        try:
            link = entry.get("link", "")
            accession_match = re.search(r"(\d{10}-\d{2}-\d{6})", link)
            if not accession_match:
                return None

            accession = accession_match.group(1)

            # Parse title: "4 - Company Name (CIK)"
            title = entry.get("title", "")
            title_match = re.match(r"4\s*-\s*(.+?)\s*\((\d+)\)", title)
            if title_match:
                company_name = title_match.group(1).strip()
                cik = title_match.group(2).zfill(10)
            else:
                company_name = title
                cik = ""

            # Fetch and parse the actual XML document for real transaction data
            xml_url = self._build_form4_xml_url(cik, accession)
            xml_content = await self._fetch_form4_xml(xml_url) if xml_url else None
            if not xml_content:
                return None

            return self._parse_form4_xml(xml_content, accession, cik, ticker, company_name, link)

        except Exception as e:
            logger.warning(f"Error parsing Form 4 entry: {e}")
            return None

    def _build_form4_xml_url(self, cik: str, accession: str) -> Optional[str]:
        """Construct the SEC EDGAR URL for a Form 4 XML document."""
        if not cik or not accession:
            return None
        try:
            cik_int = str(int(cik))  # Remove leading zeros for URL path
            accession_nodash = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{accession_nodash}/{accession}.xml"
            )
        except (ValueError, AttributeError):
            return None

    async def _fetch_form4_xml(self, url: str) -> Optional[str]:
        """Fetch Form 4 XML document from SEC EDGAR."""
        try:
            headers = {"User-Agent": "GV2-EDGE research@gv2edge.com"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        logger.debug(f"Form 4 XML not found (status {resp.status}): {url}")
                        return None
                    return await resp.text()
        except Exception as e:
            logger.debug(f"Form 4 XML fetch error: {e}")
            return None

    def _parse_form4_xml(
        self,
        xml_content: str,
        accession: str,
        cik: str,
        ticker: str,
        company_name: str,
        url: str
    ) -> Optional[InsiderTransaction]:
        """Parse Form 4 XML content into InsiderTransaction."""
        try:
            root = ET.fromstring(xml_content)

            def node_text(parent, tag) -> str:
                """Get text from child tag, checking <value> sub-element first."""
                el = parent.find(tag) if parent is not None else None
                if el is None:
                    return ""
                val = el.find("value")
                return ((val.text if val is not None else el.text) or "").strip()

            # Reporting owner
            insider_name = ""
            insider_title = ""
            owner = root.find("reportingOwner")
            if owner is not None:
                owner_id = owner.find("reportingOwnerId")
                if owner_id is not None:
                    insider_name = node_text(owner_id, "rptOwnerName")
                rel = owner.find("reportingOwnerRelationship")
                if rel is not None:
                    insider_title = node_text(rel, "officerTitle")
                    if not insider_title and node_text(rel, "isDirector") == "1":
                        insider_title = "Director"

            # First non-derivative transaction
            txn_date = datetime.utcnow()
            transaction_code = ""
            shares = 0
            price = 0.0
            shares_after = 0
            ownership_type = "D"

            nd_table = root.find("nonDerivativeTable")
            if nd_table is not None:
                txn = nd_table.find("nonDerivativeTransaction")
                if txn is not None:
                    # Date
                    date_str = node_text(txn, "transactionDate")
                    try:
                        txn_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass

                    # Transaction code (P=Purchase, S=Sale, M=Exercise)
                    coding = txn.find("transactionCoding")
                    if coding is not None:
                        transaction_code = node_text(coding, "transactionCode")

                    # Amounts
                    amounts = txn.find("transactionAmounts")
                    if amounts is not None:
                        try:
                            shares = int(float(node_text(amounts, "transactionShares") or "0"))
                        except (ValueError, TypeError):
                            shares = 0
                        try:
                            price = float(node_text(amounts, "transactionPricePerShare") or "0")
                        except (ValueError, TypeError):
                            price = 0.0

                    # Post-transaction shares
                    post = txn.find("postTransactionAmounts")
                    if post is not None:
                        try:
                            shares_after = int(float(
                                node_text(post, "sharesOwnedFollowingTransaction") or "0"
                            ))
                        except (ValueError, TypeError):
                            shares_after = 0

                    # Ownership type (D=Direct, I=Indirect)
                    own_nature = txn.find("ownershipNature")
                    if own_nature is not None:
                        ownership_type = node_text(own_nature, "directOrIndirectOwnership") or "D"

            return InsiderTransaction(
                accession_number=accession,
                cik=cik,
                ticker=ticker,
                company_name=company_name,
                insider_name=insider_name,
                insider_title=insider_title,
                transaction_date=txn_date,
                transaction_code=transaction_code,
                shares=shares,
                price=price,
                value=shares * price,
                ownership_type=ownership_type,
                shares_after=shares_after,
                url=url
            )

        except Exception as e:
            logger.warning(f"Form 4 XML parse error: {e}")
            return None

    async def get_insider_summary(self, ticker: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get insider activity summary for boost calculation

        Returns:
            {
                "has_recent_buys": bool,
                "total_buy_value": float,
                "unique_buyers": int,
                "boost_multiplier": float (1.0-1.3)
            }
        """
        transactions = await self.fetch_insider_transactions(ticker, days_back)

        # Filter to purchases only
        buys = [t for t in transactions if t.transaction_code == "P"]

        if not buys:
            return {
                "has_recent_buys": False,
                "total_buy_value": 0,
                "unique_buyers": 0,
                "boost_multiplier": 1.0
            }

        total_value = sum(t.value for t in buys)
        unique_buyers = len(set(t.insider_name for t in buys))

        # Calculate boost
        if total_value >= 1_000_000:
            boost = 1.3
        elif total_value >= 500_000:
            boost = 1.2
        elif total_value >= 100_000:
            boost = 1.1
        else:
            boost = 1.05

        # Multiple insiders = extra confidence
        if unique_buyers >= 3:
            boost = min(1.3, boost + 0.1)

        return {
            "has_recent_buys": True,
            "total_buy_value": total_value,
            "unique_buyers": unique_buyers,
            "boost_multiplier": boost
        }


# ============================
# Unified SEC Ingestor
# ============================

class SECIngestor:
    """Unified interface for SEC data"""

    def __init__(self, universe: set = None):
        self.universe = universe or set()
        self.ingestor_8k = SEC8KIngestor(universe)
        self.ingestor_form4 = SECForm4Ingestor()

    async def fetch_all_recent(self, hours_back: int = 2) -> List[SECFiling]:
        """Fetch all recent 8-K filings"""
        return await self.ingestor_8k.fetch_recent_8k(hours_back)

    async def get_insider_boost(self, ticker: str) -> Dict[str, Any]:
        """Get insider activity boost for ticker"""
        return await self.ingestor_form4.get_insider_summary(ticker)

    def set_universe(self, universe: set):
        """Update universe filter"""
        self.universe = universe
        self.ingestor_8k.universe = universe


# ============================
# Module exports
# ============================

__all__ = [
    "SECFiling",
    "InsiderTransaction",
    "SEC8KIngestor",
    "SECForm4Ingestor",
    "SECIngestor",
    "get_cik_mapper"
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        ingestor = SECIngestor()

        print("Fetching recent 8-K filings...")
        filings = await ingestor.fetch_all_recent(hours_back=24)

        for f in filings[:5]:
            print(f"\n{f.ticker or f.cik}: {f.headline}")
            print(f"  Type: {f.event_type}, Impact: {f.event_impact:.2f}")
            print(f"  Items: {f.items}")

    asyncio.run(test())
