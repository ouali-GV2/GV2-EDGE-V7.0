"""
NLP CLASSIFIER V6.1
===================

Grok API - UNIQUEMENT pour classification EVENT_TYPE.
JAMAIS pour sourcing de donnees.

Architecture:
- Input: News item (headline + summary)
- Output: EVENT_TYPE from unified taxonomy (18 types, 5 tiers)
- Cache: TTL 1h pour eviter appels redondants
- Batch: Support pour classification en lot avec rate limiting

Cost optimization:
- Pre-filter avec keyword_filter.py AVANT appel Grok
- Cache responses
- Batch processing
- Skip items deja classifies par SEC 8-K
"""

import os
import re
import json
import hashlib
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

from openai import AsyncOpenAI

from utils.logger import get_logger
from config import GROK_API_KEY, GROK_API_URL, ENABLE_OLLAMA_BATCH

logger = get_logger("NLP_CLASSIFIER")


# ============================
# Configuration
# ============================

# Grok client setup
GROK_CLIENT = AsyncOpenAI(
    api_key=GROK_API_KEY,
    base_url=GROK_API_URL
) if GROK_API_KEY else None

# Cache database
NLP_CACHE_DB = "data/nlp_classifier_cache.db"

# Rate limiting
MAX_CONCURRENT_REQUESTS = 5
REQUEST_DELAY_MS = 100  # Delay between requests


# ============================
# EVENT_TYPE Taxonomy
# ============================

class EventType(Enum):
    """Unified EVENT_TYPE taxonomy V6"""

    # TIER 1 - CRITICAL IMPACT (0.90-1.00)
    FDA_APPROVAL = "FDA_APPROVAL"
    PDUFA_DECISION = "PDUFA_DECISION"
    BUYOUT_CONFIRMED = "BUYOUT_CONFIRMED"

    # TIER 2 - HIGH IMPACT (0.75-0.89)
    FDA_TRIAL_POSITIVE = "FDA_TRIAL_POSITIVE"
    BREAKTHROUGH_DESIGNATION = "BREAKTHROUGH_DESIGNATION"
    FDA_FAST_TRACK = "FDA_FAST_TRACK"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    EARNINGS_BEAT_BIG = "EARNINGS_BEAT_BIG"
    MAJOR_CONTRACT = "MAJOR_CONTRACT"

    # TIER 3 - MODERATE IMPACT (0.60-0.74)
    GUIDANCE_RAISE = "GUIDANCE_RAISE"
    EARNINGS_BEAT = "EARNINGS_BEAT"
    PARTNERSHIP = "PARTNERSHIP"
    PRICE_TARGET_RAISE = "PRICE_TARGET_RAISE"

    # TIER 4 - LOW-MODERATE IMPACT (0.45-0.59)
    ANALYST_UPGRADE = "ANALYST_UPGRADE"
    SHORT_SQUEEZE_SIGNAL = "SHORT_SQUEEZE_SIGNAL"
    UNUSUAL_VOLUME_NEWS = "UNUSUAL_VOLUME_NEWS"

    # TIER 5 - SPECULATIVE (0.30-0.44)
    BUYOUT_RUMOR = "BUYOUT_RUMOR"
    SOCIAL_MEDIA_SURGE = "SOCIAL_MEDIA_SURGE"
    BREAKING_POSITIVE = "BREAKING_POSITIVE"

    # Non-catalyst
    NONE = "NONE"


# Impact scores by event type
EVENT_IMPACT = {
    # TIER 1
    EventType.FDA_APPROVAL: 0.95,
    EventType.PDUFA_DECISION: 0.92,
    EventType.BUYOUT_CONFIRMED: 0.95,
    # TIER 2
    EventType.FDA_TRIAL_POSITIVE: 0.85,
    EventType.BREAKTHROUGH_DESIGNATION: 0.82,
    EventType.FDA_FAST_TRACK: 0.80,
    EventType.MERGER_ACQUISITION: 0.82,
    EventType.EARNINGS_BEAT_BIG: 0.80,
    EventType.MAJOR_CONTRACT: 0.78,
    # TIER 3
    EventType.GUIDANCE_RAISE: 0.70,
    EventType.EARNINGS_BEAT: 0.65,
    EventType.PARTNERSHIP: 0.65,
    EventType.PRICE_TARGET_RAISE: 0.62,
    # TIER 4
    EventType.ANALYST_UPGRADE: 0.55,
    EventType.SHORT_SQUEEZE_SIGNAL: 0.52,
    EventType.UNUSUAL_VOLUME_NEWS: 0.48,
    # TIER 5
    EventType.BUYOUT_RUMOR: 0.42,
    EventType.SOCIAL_MEDIA_SURGE: 0.38,
    EventType.BREAKING_POSITIVE: 0.35,
    # Non-catalyst
    EventType.NONE: 0.0,
}


def get_tier(event_type: EventType) -> int:
    """Get tier number for event type"""
    impact = EVENT_IMPACT.get(event_type, 0)
    if impact >= 0.90:
        return 1
    elif impact >= 0.75:
        return 2
    elif impact >= 0.60:
        return 3
    elif impact >= 0.45:
        return 4
    elif impact >= 0.30:
        return 5
    else:
        return 0


# ============================
# Classification Prompt
# ============================

CLASSIFICATION_PROMPT = """You are a financial news classifier for small-cap US stocks.
Classify the following news into ONE of these EVENT_TYPES.

EVENT TYPES (use EXACTLY these names):

TIER 1 - CRITICAL IMPACT (impact: 0.90-1.00):
- FDA_APPROVAL: Drug/device approved by FDA
- PDUFA_DECISION: FDA PDUFA deadline decision (positive)
- BUYOUT_CONFIRMED: Confirmed acquisition/buyout with price

TIER 2 - HIGH IMPACT (impact: 0.75-0.89):
- FDA_TRIAL_POSITIVE: Positive Phase II/III clinical trial results
- BREAKTHROUGH_DESIGNATION: FDA breakthrough therapy designation
- FDA_FAST_TRACK: FDA fast track designation granted
- MERGER_ACQUISITION: M&A announcement (not yet confirmed)
- EARNINGS_BEAT_BIG: Earnings beat estimates by >20%
- MAJOR_CONTRACT: Contract award worth >$50M

TIER 3 - MODERATE IMPACT (impact: 0.60-0.74):
- GUIDANCE_RAISE: Forward guidance raised/increased
- EARNINGS_BEAT: Standard earnings beat (<20%)
- PARTNERSHIP: Strategic partnership announcement
- PRICE_TARGET_RAISE: Analyst raises price target significantly

TIER 4 - LOW-MODERATE IMPACT (impact: 0.45-0.59):
- ANALYST_UPGRADE: Analyst upgrades rating (Hold to Buy, etc.)
- SHORT_SQUEEZE_SIGNAL: High short interest + covering signals
- UNUSUAL_VOLUME_NEWS: Unusual volume with positive news

TIER 5 - SPECULATIVE (impact: 0.30-0.44):
- BUYOUT_RUMOR: Unconfirmed acquisition rumor
- SOCIAL_MEDIA_SURGE: Viral social media mentions
- BREAKING_POSITIVE: Generic positive breaking news

NONE: Not a catalyst, negative news, or irrelevant

IMPORTANT RULES:
1. Be conservative - only classify as high tier if clearly justified
2. Earnings beats need clear evidence (specific numbers) for EARNINGS_BEAT_BIG
3. FDA news without "approved" or "cleared" is not FDA_APPROVAL
4. Rumors are TIER 5, confirmed deals are TIER 1-2
5. If unsure, use BREAKING_POSITIVE or NONE

Respond with JSON only: {{"event_type": "...", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

NEWS TO CLASSIFY:
Headline: {headline}
Summary: {summary}"""


# ============================
# Classification Result
# ============================

@dataclass
class ClassificationResult:
    """Result of NLP classification"""
    event_type: str
    impact: float
    tier: int
    confidence: float
    reasoning: str
    from_cache: bool = False


# ============================
# Cache
# ============================

class ClassificationCache:
    """SQLite cache for classification results"""

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize cache database"""
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(NLP_CACHE_DB, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS classification_cache (
                hash TEXT PRIMARY KEY,
                event_type TEXT,
                impact REAL,
                tier INTEGER,
                confidence REAL,
                reasoning TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    def _hash(self, headline: str, summary: str) -> str:
        """Generate hash for cache key"""
        text = f"{headline}|{summary}"
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, headline: str, summary: str) -> Optional[ClassificationResult]:
        """Get cached result"""
        hash_key = self._hash(headline, summary)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT event_type, impact, tier, confidence, reasoning, created_at
            FROM classification_cache WHERE hash = ?
        """, (hash_key,))
        row = cursor.fetchone()

        if row:
            # Check if expired (1 hour TTL)
            created = datetime.fromisoformat(row[5])
            if datetime.now(timezone.utc) - created < timedelta(hours=1):
                return ClassificationResult(
                    event_type=row[0],
                    impact=row[1],
                    tier=row[2],
                    confidence=row[3],
                    reasoning=row[4],
                    from_cache=True
                )

        return None

    def set(self, headline: str, summary: str, result: ClassificationResult):
        """Cache result"""
        hash_key = self._hash(headline, summary)
        self.conn.execute("""
            INSERT OR REPLACE INTO classification_cache
            (hash, event_type, impact, tier, confidence, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            hash_key,
            result.event_type,
            result.impact,
            result.tier,
            result.confidence,
            result.reasoning,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def clear_expired(self):
        """Clear expired cache entries"""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        self.conn.execute(
            "DELETE FROM classification_cache WHERE created_at < ?",
            (cutoff,)
        )
        self.conn.commit()


# ============================
# NLP Classifier
# ============================

class NLPClassifier:
    """
    Grok-based NLP classifier for EVENT_TYPE

    Usage:
        classifier = NLPClassifier()
        result = await classifier.classify(
            "FDA approves ACME drug",
            "The FDA has approved ACME's new diabetes drug..."
        )
        print(f"Type: {result.event_type}, Impact: {result.impact}")
    """

    def __init__(self):
        self.client = GROK_CLIENT
        self.cache = ClassificationCache()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._quota_exhausted_until = 0.0  # circuit breaker: skip API until this time

        if not self.client:
            logger.warning("Grok API not configured - classifier will use fallback")

    async def classify(
        self,
        headline: str,
        summary: str = "",
        use_cache: bool = True
    ) -> ClassificationResult:
        """
        Classify a single news item

        Args:
            headline: News headline
            summary: News summary/body (optional)
            use_cache: Whether to use cached results

        Returns:
            ClassificationResult with event_type, impact, tier, confidence
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(headline, summary)
            if cached:
                logger.debug(f"Cache hit for: {headline[:50]}...")
                return cached

        # 1. Ollama local (batch — gratuit, 0 latence réseau)
        if ENABLE_OLLAMA_BATCH:
            ollama_result = self._try_ollama(headline, summary)
            if ollama_result is not None:
                if use_cache:
                    self.cache.set(headline, summary, ollama_result)
                return ollama_result

        # 2. Grok/Groq API (fallback si Ollama down)
        if not self.client:
            return self._fallback_classify(headline, summary)

        import time as _time
        if _time.time() < self._quota_exhausted_until:
            return self._fallback_classify(headline, summary)

        try:
            async with self.semaphore:
                result = await self._call_grok(headline, summary)
            if use_cache:
                self.cache.set(headline, summary, result)
            return result

        except Exception as e:
            import time as _time
            from openai import RateLimitError
            if isinstance(e, RateLimitError):
                self._quota_exhausted_until = _time.time() + 14400
                logger.warning("Grok quota exhausted — using fallback")
            else:
                logger.warning(f"Grok API error [{type(e).__name__}]: {e}")
            return self._fallback_classify(headline, summary)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from response (handles markdown code blocks and reasoning text)"""
        if not text:
            return {}
        # Try direct parse first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Extract JSON object from text (handles  or inline JSON)
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}

    def _try_ollama(self, headline: str, summary: str) -> Optional[ClassificationResult]:
        """Tente la classification via Ollama local (synchrone, batch only)."""
        try:
            from src.nlp_local import call_ollama
            prompt = CLASSIFICATION_PROMPT.format(
                headline=headline,
                summary=summary[:500] if summary else "N/A"
            )
            data = call_ollama(CLASSIFICATION_PROMPT, f"Headline: {headline}\nSummary: {summary[:500] or 'N/A'}")
            if not data:
                return None
            event_type_str = data.get("event_type", "NONE")
            try:
                event_type = EventType[event_type_str]
            except KeyError:
                event_type = EventType.NONE
            impact = EVENT_IMPACT.get(event_type, 0.0)
            return ClassificationResult(
                event_type=event_type.value,
                impact=impact,
                tier=get_tier(event_type),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", "ollama-local"),
                from_cache=False,
            )
        except Exception as e:
            logger.debug(f"Ollama classify error: {e}")
            return None

    async def _call_grok(self, headline: str, summary: str) -> ClassificationResult:
        """Make Grok API call"""
        prompt = CLASSIFICATION_PROMPT.format(
            headline=headline,
            summary=summary[:500] if summary else "N/A"
        )

        response = await self.client.chat.completions.create(
            model="grok-3-fast",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )

        # Parse response (reasoning models may wrap JSON in text)
        raw = response.choices[0].message.content or ""
        data = self._extract_json(raw)

        event_type_str = data.get("event_type", "NONE")

        # Validate event type
        try:
            event_type = EventType[event_type_str]
        except KeyError:
            logger.warning(f"Unknown event type from Grok: {event_type_str}")
            event_type = EventType.NONE

        impact = EVENT_IMPACT.get(event_type, 0.0)
        tier = get_tier(event_type)

        return ClassificationResult(
            event_type=event_type.value,
            impact=impact,
            tier=tier,
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            from_cache=False
        )

    def _fallback_classify(self, headline: str, summary: str) -> ClassificationResult:
        """
        Fallback classification when Grok unavailable
        Uses keyword matching (less accurate but free)
        """
        text = f"{headline} {summary}".lower()

        # Check for TIER 1 keywords
        if "fda approv" in text or "fda clear" in text:
            return ClassificationResult(
                event_type="FDA_APPROVAL",
                impact=0.95,
                tier=1,
                confidence=0.7,
                reasoning="Keyword match: FDA approval",
                from_cache=False
            )

        if "buyout" in text and ("confirmed" in text or "definitive" in text):
            return ClassificationResult(
                event_type="BUYOUT_CONFIRMED",
                impact=0.95,
                tier=1,
                confidence=0.6,
                reasoning="Keyword match: buyout confirmed",
                from_cache=False
            )

        # Check for TIER 2 keywords
        if "phase" in text and ("positive" in text or "met" in text or "success" in text):
            return ClassificationResult(
                event_type="FDA_TRIAL_POSITIVE",
                impact=0.85,
                tier=2,
                confidence=0.6,
                reasoning="Keyword match: positive trial",
                from_cache=False
            )

        if "beat" in text and ("eps" in text or "earnings" in text or "revenue" in text):
            return ClassificationResult(
                event_type="EARNINGS_BEAT",
                impact=0.65,
                tier=3,
                confidence=0.5,
                reasoning="Keyword match: earnings beat",
                from_cache=False
            )

        # Default
        return ClassificationResult(
            event_type="BREAKING_POSITIVE",
            impact=0.35,
            tier=5,
            confidence=0.3,
            reasoning="Fallback: no specific keywords matched",
            from_cache=False
        )

    async def classify_batch(
        self,
        items: List[Dict[str, Any]],
        headline_key: str = "headline",
        summary_key: str = "summary"
    ) -> List[Dict[str, Any]]:
        """
        Classify batch of items

        Args:
            items: List of dicts with headline/summary fields
            headline_key: Key for headline field
            summary_key: Key for summary field

        Returns:
            Items with classification results added
        """
        async def classify_item(item):
            headline = item.get(headline_key, "")
            summary = item.get(summary_key, "")

            result = await self.classify(headline, summary)

            item["classification"] = {
                "event_type": result.event_type,
                "impact": result.impact,
                "tier": result.tier,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "from_cache": result.from_cache
            }

            # Small delay between requests
            await asyncio.sleep(REQUEST_DELAY_MS / 1000)

            return item

        # Process in parallel with rate limiting
        tasks = [classify_item(item) for item in items]
        return await asyncio.gather(*tasks)


# ============================
# Convenience Functions
# ============================

_classifier_instance = None
_classifier_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton

def get_classifier() -> NLPClassifier:
    """Get singleton classifier instance"""
    global _classifier_instance
    with _classifier_lock:
        if _classifier_instance is None:
            _classifier_instance = NLPClassifier()
    return _classifier_instance


async def classify_news(headline: str, summary: str = "") -> ClassificationResult:
    """Quick classification of single news item"""
    classifier = get_classifier()
    return await classifier.classify(headline, summary)


def get_event_impact(event_type: str) -> float:
    """Get impact score for event type string"""
    try:
        et = EventType[event_type]
        return EVENT_IMPACT.get(et, 0.0)
    except KeyError:
        return 0.0


def get_event_tier(event_type: str) -> int:
    """Get tier for event type string"""
    try:
        et = EventType[event_type]
        return get_tier(et)
    except KeyError:
        return 0


# ============================
# Module exports
# ============================

__all__ = [
    "NLPClassifier",
    "ClassificationResult",
    "EventType",
    "EVENT_IMPACT",
    "get_classifier",
    "classify_news",
    "get_event_impact",
    "get_event_tier",
    "get_tier",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        classifier = NLPClassifier()

        test_cases = [
            ("FDA approves ACME's new diabetes drug", "The FDA has granted approval..."),
            ("BIOX Phase 3 trial meets primary endpoint", "Results showed statistical significance..."),
            ("Company to be acquired for $500 million", "Definitive agreement signed..."),
            ("Q4 earnings beat estimates", "EPS of $1.20 vs $1.00 expected..."),
            ("New partnership announced", "Strategic collaboration with major pharma..."),
        ]

        print("=" * 60)
        print("NLP CLASSIFIER TEST")
        print("=" * 60)

        for headline, summary in test_cases:
            result = await classifier.classify(headline, summary)
            print(f"\n{headline[:50]}...")
            print(f"  Type: {result.event_type}")
            print(f"  Impact: {result.impact:.2f}, Tier: {result.tier}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Cache: {result.from_cache}")

    asyncio.run(test())
