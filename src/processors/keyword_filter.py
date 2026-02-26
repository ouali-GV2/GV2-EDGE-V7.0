"""
KEYWORD FILTER V6.1
===================

Fast pre-filter pour news/filings AVANT le NLP.
Objectif: Reduire les appels Grok API (cost saving)

Architecture:
- CRITICAL_KEYWORDS: Trigger scan immediat (FDA, M&A, etc.)
- HIGH_KEYWORDS: Important mais pas urgent
- NOISE_PATTERNS: Skip ces items (lawsuits, dilution, etc.)

Performance: ~1000 items/seconde (pure regex, no LLM)
"""

import re
import threading
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("KEYWORD_FILTER")


# ============================
# Priority Levels
# ============================

class FilterPriority(Enum):
    """Priority levels for filtered items"""
    CRITICAL = 1    # Immediate scan, TIER 1-2 potential
    HIGH = 2        # Important, TIER 2-3 potential
    MODERATE = 3    # Worth checking, TIER 3-4 potential
    LOW = 4         # Minor, TIER 4-5 potential
    NOISE = 5       # Skip, likely negative/noise


# ============================
# Keyword Patterns
# ============================

# CRITICAL: TIER 1-2 catalysts (immediate hot queue)
CRITICAL_PATTERNS = {
    "fda_approval": [
        r"\bFDA\s+approv(?:al|ed|es)",
        r"\bFDA\s+clear(?:ance|ed)",
        r"\bPDUFA\b.*\b(?:date|decision|approv)",
        r"\bNDA\s+approv",
        r"\bBLA\s+approv",
        r"\b510\(k\)\s+clear",
    ],
    "fda_positive": [
        r"\bphase\s+(?:2|II|3|III)\b.*\b(?:positive|success|met|exceed)",
        r"\bprimary\s+endpoint\s+met",
        r"\bstatistically\s+significant",
        r"\bbreakthrough\s+therapy\s+designat",
        r"\bfast\s+track\s+(?:designat|grant)",
        r"\bpriority\s+review\s+(?:designat|grant)",
    ],
    "buyout_ma": [
        r"\bacquir(?:e|ed|es|ing)\s+(?:by|for)\s+\$",
        r"\bmerger\s+(?:agreement|with)",
        r"\bbuyout\s+(?:offer|deal|confirmed)",
        r"\bgoing\s+private",
        r"\bdefinitive\s+agreement\s+to\s+(?:acquire|merge|be\s+acquired)",
        r"\btakeover\s+(?:bid|offer)",
    ],
    "earnings_beat_big": [
        r"\bEPS\s+(?:of\s+)?\$[\d.]+\s+(?:beat|vs\.?\s+\$[\d.]+\s+expect)",
        r"\b(?:beat|exceed)(?:s|ed)?\s+(?:by|estimates?\s+by)\s+(?:\$[\d.]+|[\d.]+%|\d+\s*(?:cents?|c))",
        r"\brevenue\s+(?:of\s+)?\$[\d.]+[MB]\s+(?:beat|vs\.?\s+\$[\d.]+[MB]\s+expect)",
        r"\bguidance\s+rais(?:e|ed)",
        r"\bupside\s+(?:surprise|beat)",
    ],
    "major_contract": [
        r"\bcontract\s+(?:worth|valued?\s+at|for)\s+\$\d+\s*(?:million|billion|M|B)",
        r"\baward(?:ed)?\s+\$\d+\s*(?:million|billion|M|B)\s+contract",
        r"\bdeal\s+(?:worth|valued?\s+at)\s+\$\d+\s*(?:million|billion|M|B)",
        r"\bpartnership\s+(?:worth|valued?\s+at)\s+\$\d+\s*(?:million|billion|M|B)",
    ],
}

# HIGH: TIER 2-3 catalysts
HIGH_PATTERNS = {
    "earnings_beat": [
        r"\b(?:beat|exceed)(?:s|ed)?\s+(?:estimates?|expectations?|consensus)",
        r"\bEPS\s+(?:beat|above|exceed)",
        r"\brevenue\s+(?:beat|above|exceed)",
        r"\bstrong(?:er)?\s+(?:quarter|earnings|results)",
        r"\brecord\s+(?:revenue|earnings|quarter)",
    ],
    "guidance": [
        r"\b(?:raise|raised|raises|raising)\s+(?:guidance|outlook|forecast)",
        r"\b(?:upward|positive)\s+(?:revision|guidance)",
        r"\bfull[- ]year\s+guidance\s+(?:raised|increased)",
        r"\bforecast\s+(?:raised|increased|above)",
    ],
    "analyst_positive": [
        r"\bprice\s+target\s+(?:raised|increased|to\s+\$\d+)",
        r"\bupgrad(?:e|ed)\s+(?:to|from)\s+(?:buy|outperform|overweight)",
        r"\b(?:initiated?|start(?:ed)?)\s+(?:at|with)\s+(?:buy|outperform)",
    ],
    "partnership": [
        r"\bpartnership\s+(?:with|agreement)",
        r"\bstrategic\s+(?:alliance|partnership|collaboration)",
        r"\blicens(?:e|ing)\s+(?:agreement|deal)",
        r"\bcollaboration\s+(?:agreement|with)",
    ],
    "short_squeeze": [
        r"\bshort\s+(?:squeeze|interest|covering)",
        r"\bhigh(?:ly)?\s+shorted",
        r"\bshort\s+%\s+(?:of\s+)?float",
        r"\bdays\s+to\s+cover",
    ],
}

# MODERATE: TIER 3-4 catalysts
MODERATE_PATTERNS = {
    "product_news": [
        r"\bnew\s+product\s+launch",
        r"\bproduct\s+(?:approval|launch|release)",
        r"\bcommercial\s+launch",
    ],
    "insider_buying": [
        r"\binsider\s+(?:buy|purchase|buying)",
        r"\b(?:CEO|CFO|director)\s+(?:bought|purchased|buys)",
        r"\bform\s+4\s+(?:filing|purchase)",
    ],
    "volume_spike": [
        r"\bunusual\s+(?:volume|activity)",
        r"\bvolume\s+(?:spike|surge)",
        r"\bheavy\s+(?:volume|trading)",
    ],
}

# NOISE: Skip these (negative/irrelevant)
NOISE_PATTERNS = [
    # Negative events
    r"\bprice\s+target\s+(?:cut|lower|reduced)",
    r"\bdowngrad(?:e|ed)\s+(?:to|from)\s+(?:sell|underperform|underweight)",
    r"\b(?:miss|missed|misses)\s+(?:estimates?|expectations?|EPS)",
    r"\blower(?:s|ed)?\s+(?:guidance|outlook|forecast)",
    r"\b(?:warn|warning)\s+(?:about|of|on)",
    r"\b(?:recall|recalled)\s+(?:product|due\s+to)",
    r"\bFDA\s+(?:reject|warning|CRL|complete\s+response)",

    # Dilution
    r"\bstock\s+(?:offering|issuance)",
    r"\bdilut(?:ion|ive)",
    r"\bsecondary\s+offering",
    r"\bshelf\s+registration",
    r"\b(?:ATM|at-the-market)\s+offering",

    # Legal/regulatory issues
    r"\blawsuit\s+(?:filed|against)",
    r"\bclass\s+action",
    r"\bSEC\s+(?:investigation|probe|subpoena)",
    r"\bfraud\s+(?:allegations?|charges?)",
    r"\baccounting\s+(?:irregularities?|issues?)",

    # Management issues
    r"\b(?:CEO|CFO)\s+(?:resign|step(?:s|ped)?\s+down|depart)",
    r"\bdelisting\s+(?:notice|warning)",
    r"\bgoing\s+concern",

    # Generic noise
    r"\bno\s+(?:news|update|change)",
    r"\broutine\s+(?:filing|update)",
]


# ============================
# Filter Result
# ============================

@dataclass
class FilterResult:
    """Result of keyword filtering"""
    passed: bool
    priority: FilterPriority
    matched_category: Optional[str]
    matched_patterns: List[str]
    confidence: float


# ============================
# Keyword Filter Class
# ============================

class KeywordFilter:
    """
    Fast keyword filter for news and filings

    Usage:
        filter = KeywordFilter()
        result = filter.apply("FDA approves ACME drug for diabetes")
        if result.passed:
            print(f"Priority: {result.priority}, Category: {result.matched_category}")
    """

    def __init__(self):
        # Compile all patterns for performance
        self.critical_compiled = self._compile_category(CRITICAL_PATTERNS)
        self.high_compiled = self._compile_category(HIGH_PATTERNS)
        self.moderate_compiled = self._compile_category(MODERATE_PATTERNS)
        self.noise_compiled = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

        logger.info("KeywordFilter initialized with compiled patterns")

    def _compile_category(self, patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for a category"""
        compiled = {}
        for category, pattern_list in patterns.items():
            compiled[category] = [re.compile(p, re.IGNORECASE) for p in pattern_list]
        return compiled

    def apply(self, text: str) -> FilterResult:
        """
        Apply filter to text

        Args:
            text: News headline + summary

        Returns:
            FilterResult with priority and matched patterns
        """
        if not text:
            return FilterResult(
                passed=False,
                priority=FilterPriority.NOISE,
                matched_category=None,
                matched_patterns=[],
                confidence=0.0
            )

        text = text.strip()

        # Check noise patterns first (fast rejection)
        for pattern in self.noise_compiled:
            if pattern.search(text):
                return FilterResult(
                    passed=False,
                    priority=FilterPriority.NOISE,
                    matched_category="noise",
                    matched_patterns=[pattern.pattern],
                    confidence=0.0
                )

        # Check critical patterns
        result = self._check_category(text, self.critical_compiled, FilterPriority.CRITICAL)
        if result.passed:
            return result

        # Check high patterns
        result = self._check_category(text, self.high_compiled, FilterPriority.HIGH)
        if result.passed:
            return result

        # Check moderate patterns
        result = self._check_category(text, self.moderate_compiled, FilterPriority.MODERATE)
        if result.passed:
            return result

        # No match - low priority but not noise
        return FilterResult(
            passed=True,  # Still pass for NLP classification
            priority=FilterPriority.LOW,
            matched_category=None,
            matched_patterns=[],
            confidence=0.3
        )

    def _check_category(
        self,
        text: str,
        compiled: Dict[str, List[re.Pattern]],
        priority: FilterPriority
    ) -> FilterResult:
        """Check text against a category of patterns"""

        for category, patterns in compiled.items():
            matched = []
            for pattern in patterns:
                if pattern.search(text):
                    matched.append(pattern.pattern)

            if matched:
                # Confidence based on number of matches
                confidence = min(1.0, 0.5 + 0.1 * len(matched))

                return FilterResult(
                    passed=True,
                    priority=priority,
                    matched_category=category,
                    matched_patterns=matched,
                    confidence=confidence
                )

        return FilterResult(
            passed=False,
            priority=priority,
            matched_category=None,
            matched_patterns=[],
            confidence=0.0
        )

    def apply_batch(self, items: List[Dict[str, Any]], text_key: str = "text") -> List[Tuple[Dict, FilterResult]]:
        """
        Apply filter to batch of items

        Args:
            items: List of dicts with text field
            text_key: Key for text field (default: "text")

        Returns:
            List of (item, FilterResult) tuples, sorted by priority
        """
        results = []

        for item in items:
            text = item.get(text_key, "") or item.get("headline", "") + " " + item.get("summary", "")
            result = self.apply(text)
            results.append((item, result))

        # Sort by priority (CRITICAL first)
        results.sort(key=lambda x: x[1].priority.value)

        return results

    def filter_passed(self, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Filter items and return only those that passed

        Args:
            items: List of dicts with text field
            text_key: Key for text field

        Returns:
            List of items that passed the filter (with filter_result attached)
        """
        results = self.apply_batch(items, text_key)

        passed = []
        for item, result in results:
            if result.passed:
                item["filter_result"] = {
                    "priority": result.priority.name,
                    "category": result.matched_category,
                    "confidence": result.confidence
                }
                passed.append(item)

        return passed

    def get_hot_triggers(self, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Get only CRITICAL and HIGH priority items (for hot ticker queue)

        Args:
            items: List of dicts with text field
            text_key: Key for text field

        Returns:
            List of high-priority items
        """
        results = self.apply_batch(items, text_key)

        hot = []
        for item, result in results:
            if result.priority in (FilterPriority.CRITICAL, FilterPriority.HIGH):
                item["filter_result"] = {
                    "priority": result.priority.name,
                    "category": result.matched_category,
                    "confidence": result.confidence
                }
                hot.append(item)

        return hot


# ============================
# Convenience Functions
# ============================

_filter_instance = None
_filter_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton

def get_keyword_filter() -> KeywordFilter:
    """Get singleton filter instance"""
    global _filter_instance
    with _filter_lock:
        if _filter_instance is None:
            _filter_instance = KeywordFilter()
    return _filter_instance


def quick_filter(text: str) -> FilterResult:
    """Quick filter for single text"""
    return get_keyword_filter().apply(text)


def is_critical(text: str) -> bool:
    """Check if text contains critical catalyst"""
    result = quick_filter(text)
    return result.priority == FilterPriority.CRITICAL


def is_noise(text: str) -> bool:
    """Check if text is noise/negative"""
    result = quick_filter(text)
    return result.priority == FilterPriority.NOISE


# ============================
# Module exports
# ============================

__all__ = [
    "KeywordFilter",
    "FilterResult",
    "FilterPriority",
    "get_keyword_filter",
    "quick_filter",
    "is_critical",
    "is_noise",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    filter = KeywordFilter()

    test_cases = [
        "FDA approves ACME's new diabetes drug",
        "Phase 3 trial meets primary endpoint with statistical significance",
        "Company to be acquired by BigCorp for $500 million",
        "EPS of $1.50 beats estimates by $0.30",
        "Awarded $200 million government contract",
        "Stock price target lowered to $5 from $10",
        "SEC investigation into accounting practices",
        "Company announces routine quarterly filing",
        "Partnership agreement with leading pharma company",
        "High short interest triggers squeeze potential",
    ]

    print("=" * 60)
    print("KEYWORD FILTER TEST")
    print("=" * 60)

    for text in test_cases:
        result = filter.apply(text)
        status = "PASS" if result.passed else "SKIP"
        print(f"\n[{status}] {result.priority.name}: {text[:50]}...")
        if result.matched_category:
            print(f"       Category: {result.matched_category}")
