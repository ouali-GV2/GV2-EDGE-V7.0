"""
TICKER EXTRACTOR V6.1
=====================

Extrait et valide les tickers depuis du texte.

Features:
- Pattern matching ($AAPL, AAPL, etc.)
- Validation contre l'univers small caps
- Company name -> ticker mapping
- CIK -> ticker mapping (SEC filings)
- Disambiguation (APPLE vs AAPL)

Architecture:
- extract(): Extrait tickers bruts du texte
- validate(): Valide contre l'univers
- map_company(): Map nom d'entreprise vers ticker
"""

import re
import os
import sqlite3
import threading
from typing import List, Set, Optional, Dict, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("TICKER_EXTRACTOR")


# ============================
# Configuration
# ============================

# Common false positives to exclude
FALSE_POSITIVE_TICKERS = {
    # Common words that look like tickers
    "A", "I", "IT", "AT", "ON", "IN", "TO", "BY", "AN", "OR", "AS", "IS",
    "BE", "DO", "GO", "HE", "IF", "ME", "MY", "NO", "OF", "SO", "UP", "US",
    "WE", "AM", "PM", "ET", "CEO", "CFO", "IPO", "FDA", "SEC", "NYSE", "OTC",
    "EPS", "GDP", "ETF", "API", "USA", "USD", "EUR", "GBP", "BTC", "ETH",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS",
    "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY",
    "DID", "LOW", "TOP", "BIG", "END", "FAR", "SAY", "SHE", "TOO", "ANY",
    # Financial terms
    "ATM", "ROI", "P/E", "EV", "EBITDA", "YOY", "QOQ", "MOM", "TTM",
    # Company type suffixes
    "INC", "LLC", "LTD", "PLC", "CORP", "CO",
}

# Ticker pattern: 1-5 uppercase letters, optionally prefixed with $
TICKER_PATTERN = re.compile(r'\$?([A-Z]{1,5})\b')

# Company name patterns (for extraction)
COMPANY_PATTERNS = [
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|PLC)',
    r'(?:company|firm|stock)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
]


# ============================
# Data Classes
# ============================

@dataclass
class ExtractedTicker:
    """Represents an extracted ticker"""
    ticker: str
    source: str  # "pattern", "company_name", "cik"
    confidence: float
    context: str  # Surrounding text
    position: int  # Position in text


# ============================
# Ticker Extractor Class
# ============================

class TickerExtractor:
    """
    Extracts and validates stock tickers from text

    Usage:
        extractor = TickerExtractor(universe={"AAPL", "TSLA", "NVDA"})
        tickers = extractor.extract("FDA approves $AAPL new device")
        # Returns: [ExtractedTicker(ticker="AAPL", ...)]
    """

    def __init__(self, universe: Set[str] = None):
        self.universe = universe or set()
        self.company_map = {}  # company_name -> ticker
        self.cik_map = {}  # cik -> ticker

        self._load_mappings()

    def _load_mappings(self):
        """Load company name and CIK mappings"""
        # Try to load from SEC CIK database
        cik_db = "data/cik_mapping.db"
        if os.path.exists(cik_db):
            try:
                conn = sqlite3.connect(cik_db)
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, company_name, cik FROM cik_mapping")
                for row in cursor.fetchall():
                    ticker, company, cik = row
                    if ticker and company:
                        # Normalize company name
                        normalized = self._normalize_company(company)
                        self.company_map[normalized] = ticker
                    if ticker and cik:
                        self.cik_map[cik] = ticker
                conn.close()
                logger.info(f"Loaded {len(self.company_map)} company mappings")
            except Exception as e:
                logger.warning(f"Failed to load CIK mappings: {e}")

    def _normalize_company(self, name: str) -> str:
        """Normalize company name for matching"""
        # Remove common suffixes
        name = re.sub(r'\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|PLC|Company)$', '', name, flags=re.IGNORECASE)
        # Lowercase and remove extra spaces
        return ' '.join(name.lower().split())

    def set_universe(self, universe: Set[str]):
        """Update universe of valid tickers"""
        self.universe = universe

    def extract(self, text: str, validate: bool = True) -> List[ExtractedTicker]:
        """
        Extract tickers from text

        Args:
            text: Input text (news headline, filing, etc.)
            validate: If True, only return tickers in universe

        Returns:
            List of ExtractedTicker objects
        """
        if not text:
            return []

        results = []
        seen = set()

        # Method 1: Direct ticker patterns ($AAPL, AAPL)
        for match in TICKER_PATTERN.finditer(text):
            ticker = match.group(1).upper()

            if ticker in seen:
                continue

            if ticker in FALSE_POSITIVE_TICKERS:
                continue

            if validate and self.universe and ticker not in self.universe:
                continue

            # Get context
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]

            # Higher confidence if prefixed with $
            has_dollar = text[match.start()] == '$' if match.start() < len(text) else False
            confidence = 0.9 if has_dollar else 0.7

            results.append(ExtractedTicker(
                ticker=ticker,
                source="pattern",
                confidence=confidence,
                context=context,
                position=match.start()
            ))
            seen.add(ticker)

        # Method 2: Company name matching
        for ticker in self._extract_by_company_name(text):
            if ticker in seen:
                continue

            if validate and self.universe and ticker not in self.universe:
                continue

            results.append(ExtractedTicker(
                ticker=ticker,
                source="company_name",
                confidence=0.8,
                context="",
                position=-1
            ))
            seen.add(ticker)

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def _extract_by_company_name(self, text: str) -> List[str]:
        """Extract tickers by matching company names"""
        tickers = []

        # Normalize text
        normalized = self._normalize_company(text)

        # Check against company map
        for company, ticker in self.company_map.items():
            if company in normalized:
                tickers.append(ticker)

        return tickers

    def extract_from_sec(self, cik: str) -> Optional[str]:
        """Get ticker from CIK number"""
        cik = cik.zfill(10) if cik else None
        return self.cik_map.get(cik)

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid"""
        if not ticker:
            return False

        ticker = ticker.upper().strip()

        if ticker in FALSE_POSITIVE_TICKERS:
            return False

        if self.universe and ticker not in self.universe:
            return False

        return True

    def extract_all(self, text: str) -> List[str]:
        """
        Extract all potential tickers (no validation)

        Args:
            text: Input text

        Returns:
            List of ticker strings
        """
        results = self.extract(text, validate=False)
        return [r.ticker for r in results]

    def extract_validated(self, text: str) -> List[str]:
        """
        Extract only validated tickers

        Args:
            text: Input text

        Returns:
            List of ticker strings in universe
        """
        results = self.extract(text, validate=True)
        return [r.ticker for r in results]

    def extract_with_confidence(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract tickers with confidence scores

        Args:
            text: Input text

        Returns:
            List of (ticker, confidence) tuples
        """
        results = self.extract(text, validate=True)
        return [(r.ticker, r.confidence) for r in results]


# ============================
# Batch Processing
# ============================

class BatchTickerExtractor:
    """Process multiple items efficiently"""

    def __init__(self, universe: Set[str] = None):
        self.extractor = TickerExtractor(universe)

    def extract_batch(
        self,
        items: List[Dict],
        text_keys: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Extract tickers from batch of items and group by ticker

        Args:
            items: List of dicts (news items, filings, etc.)
            text_keys: Keys to search for text (default: ["headline", "summary", "text"])

        Returns:
            Dict mapping ticker -> list of items mentioning that ticker
        """
        if text_keys is None:
            text_keys = ["headline", "summary", "text", "title", "content"]

        ticker_items = {}

        for item in items:
            # Combine text from all keys
            text_parts = []
            for key in text_keys:
                if key in item and item[key]:
                    text_parts.append(str(item[key]))

            text = " ".join(text_parts)

            # Extract tickers
            tickers = self.extractor.extract_validated(text)

            # Add to results
            for ticker in tickers:
                if ticker not in ticker_items:
                    ticker_items[ticker] = []
                ticker_items[ticker].append(item)

        return ticker_items

    def filter_by_ticker(
        self,
        items: List[Dict],
        ticker: str,
        text_keys: List[str] = None
    ) -> List[Dict]:
        """
        Filter items to only those mentioning a specific ticker

        Args:
            items: List of dicts
            ticker: Ticker to filter for
            text_keys: Keys to search for text

        Returns:
            List of items mentioning the ticker
        """
        if text_keys is None:
            text_keys = ["headline", "summary", "text", "title", "content"]

        matching = []
        ticker = ticker.upper()

        for item in items:
            text_parts = []
            for key in text_keys:
                if key in item and item[key]:
                    text_parts.append(str(item[key]))

            text = " ".join(text_parts)
            tickers = self.extractor.extract_validated(text)

            if ticker in tickers:
                matching.append(item)

        return matching


# ============================
# Convenience Functions
# ============================

_extractor_instance = None
_extractor_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton

def get_ticker_extractor(universe: Set[str] = None) -> TickerExtractor:
    """Get singleton extractor instance"""
    global _extractor_instance
    with _extractor_lock:
        if _extractor_instance is None:
            _extractor_instance = TickerExtractor(universe)
        elif universe:
            _extractor_instance.set_universe(universe)
    return _extractor_instance


def extract_tickers(text: str, universe: Set[str] = None) -> List[str]:
    """Quick extraction of tickers from text"""
    extractor = get_ticker_extractor(universe)
    return extractor.extract_validated(text) if universe else extractor.extract_all(text)


def has_ticker(text: str, ticker: str) -> bool:
    """Check if text mentions a specific ticker"""
    ticker = ticker.upper()
    tickers = extract_tickers(text)
    return ticker in tickers


# ============================
# Module exports
# ============================

__all__ = [
    "TickerExtractor",
    "BatchTickerExtractor",
    "ExtractedTicker",
    "get_ticker_extractor",
    "extract_tickers",
    "has_ticker",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    # Test with sample universe
    universe = {"AAPL", "TSLA", "NVDA", "BIOX", "MRNA", "PFE", "JNJ"}
    extractor = TickerExtractor(universe)

    test_cases = [
        "FDA approves $AAPL new medical device for diabetes monitoring",
        "NVDA reports record earnings, beats estimates",
        "Pfizer and BioNTech announce partnership for new vaccine",
        "Tesla (TSLA) announces new factory in Texas",
        "BIOX phase 3 trial meets primary endpoint",
        "The CEO said that IT spending will increase",
        "Apple Inc. announces new product line",
    ]

    print("=" * 60)
    print("TICKER EXTRACTOR TEST")
    print("=" * 60)

    for text in test_cases:
        results = extractor.extract(text)
        print(f"\n{text[:50]}...")
        if results:
            for r in results:
                print(f"  -> {r.ticker} ({r.source}, conf={r.confidence:.2f})")
        else:
            print("  -> No tickers found")
