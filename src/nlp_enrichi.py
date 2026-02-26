# ============================
# NLP ENRICHI V6.0
# Advanced Sentiment & News Processing
# ============================
#
# V6 Enhancements over basic NLP:
# 1. Enhanced sentiment analysis (bullish/bearish intensity)
# 2. Entity extraction (tickers, people, products, numbers)
# 3. News classification (breaking vs routine, urgency)
# 4. Multi-source sentiment aggregation
# 5. Contextual financial analysis
# 6. Headline vs body sentiment comparison
#
# Architecture: ADDITIVE (works alongside nlp_event_parser.py)

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from config import GROK_API_KEY
from utils.api_guard import safe_post
from utils.logger import get_logger

logger = get_logger("NLP_ENRICHI")

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


# ============================
# ENUMS & CONSTANTS
# ============================

class SentimentDirection(Enum):
    """Sentiment direction classification"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class NewsUrgency(Enum):
    """News urgency classification"""
    BREAKING = "breaking"       # Immediate market impact
    HIGH = "high"              # Same-day impact
    MEDIUM = "medium"          # Near-term relevance
    LOW = "low"                # Background/routine
    STALE = "stale"            # Old news, no urgency


class NewsCategory(Enum):
    """News category classification"""
    FDA_REGULATORY = "fda_regulatory"
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_RATING = "analyst_rating"
    CONTRACT_DEAL = "contract_deal"
    PRODUCT_LAUNCH = "product_launch"
    MANAGEMENT = "management"
    LEGAL = "legal"
    GUIDANCE = "guidance"
    INSIDER_ACTIVITY = "insider_activity"
    SECTOR_NEWS = "sector_news"
    MACRO = "macro"
    OTHER = "other"


# Sentiment direction to numeric score mapping
SENTIMENT_SCORES = {
    SentimentDirection.VERY_BULLISH: 1.0,
    SentimentDirection.BULLISH: 0.7,
    SentimentDirection.SLIGHTLY_BULLISH: 0.4,
    SentimentDirection.NEUTRAL: 0.0,
    SentimentDirection.SLIGHTLY_BEARISH: -0.4,
    SentimentDirection.BEARISH: -0.7,
    SentimentDirection.VERY_BEARISH: -1.0,
}

# Urgency to decay factor (for time-weighted scoring)
URGENCY_DECAY_HOURS = {
    NewsUrgency.BREAKING: 4,     # Decays fast (action needed now)
    NewsUrgency.HIGH: 12,
    NewsUrgency.MEDIUM: 24,
    NewsUrgency.LOW: 48,
    NewsUrgency.STALE: 168,      # Week-long slow decay
}

# Category impact weights (how much this category typically moves stocks)
CATEGORY_IMPACT = {
    NewsCategory.FDA_REGULATORY: 1.0,
    NewsCategory.MERGER_ACQUISITION: 0.95,
    NewsCategory.EARNINGS: 0.85,
    NewsCategory.CONTRACT_DEAL: 0.75,
    NewsCategory.GUIDANCE: 0.72,
    NewsCategory.ANALYST_RATING: 0.65,
    NewsCategory.PRODUCT_LAUNCH: 0.60,
    NewsCategory.INSIDER_ACTIVITY: 0.55,
    NewsCategory.MANAGEMENT: 0.45,
    NewsCategory.LEGAL: 0.40,
    NewsCategory.SECTOR_NEWS: 0.30,
    NewsCategory.MACRO: 0.25,
    NewsCategory.OTHER: 0.20,
}


# ============================
# DATA CLASSES
# ============================

@dataclass
class ExtractedEntity:
    """Extracted entity from text"""
    entity_type: str  # ticker, person, product, number, organization
    value: str
    context: str  # Surrounding context
    confidence: float = 0.8


@dataclass
class SentimentAnalysis:
    """Detailed sentiment analysis result"""
    direction: SentimentDirection
    score: float  # -1.0 to 1.0
    confidence: float  # 0-1
    intensity: str  # weak, moderate, strong

    # Breakdown
    headline_sentiment: float = 0.0
    body_sentiment: float = 0.0

    # Keywords found
    bullish_keywords: List[str] = field(default_factory=list)
    bearish_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "direction": self.direction.value,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "intensity": self.intensity,
            "headline_sentiment": round(self.headline_sentiment, 3),
            "body_sentiment": round(self.body_sentiment, 3),
            "bullish_keywords": self.bullish_keywords[:5],
            "bearish_keywords": self.bearish_keywords[:5],
        }


@dataclass
class EnrichedNews:
    """Fully enriched news item"""
    # Original
    headline: str
    body: str
    source: str
    published_at: datetime
    url: Optional[str] = None

    # Extracted
    tickers: List[str] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)

    # Classification
    category: NewsCategory = NewsCategory.OTHER
    urgency: NewsUrgency = NewsUrgency.MEDIUM

    # Sentiment
    sentiment: Optional[SentimentAnalysis] = None

    # Scoring
    relevance_score: float = 0.5
    impact_score: float = 0.5
    final_score: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "headline": self.headline,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "tickers": self.tickers,
            "category": self.category.value,
            "urgency": self.urgency.value,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "relevance_score": round(self.relevance_score, 3),
            "impact_score": round(self.impact_score, 3),
            "final_score": round(self.final_score, 3),
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple sources"""
    ticker: str

    # Aggregated scores
    overall_sentiment: float  # -1 to 1
    sentiment_confidence: float
    sentiment_direction: SentimentDirection

    # Source breakdown
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    analyst_sentiment: float = 0.0

    # Metadata
    source_count: int = 0
    news_count: int = 0
    time_weighted: bool = True

    # Alert
    sentiment_shift: float = 0.0  # Change from previous period
    is_sentiment_spike: bool = False

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "overall_sentiment": round(self.overall_sentiment, 3),
            "confidence": round(self.sentiment_confidence, 3),
            "direction": self.sentiment_direction.value,
            "news_sentiment": round(self.news_sentiment, 3),
            "social_sentiment": round(self.social_sentiment, 3),
            "analyst_sentiment": round(self.analyst_sentiment, 3),
            "source_count": self.source_count,
            "sentiment_shift": round(self.sentiment_shift, 3),
            "is_sentiment_spike": self.is_sentiment_spike,
        }


# ============================
# KEYWORD DICTIONARIES
# ============================

BULLISH_KEYWORDS = {
    # FDA/Biotech
    "fda approval", "approved", "clearance", "breakthrough therapy",
    "fast track", "priority review", "positive results", "efficacy",
    "met primary endpoint", "statistically significant",

    # Earnings
    "beat", "beats", "exceeded", "surpassed", "record revenue",
    "record earnings", "growth", "outperform", "strong quarter",
    "raised guidance", "raises outlook", "better than expected",

    # M&A
    "acquisition", "acquires", "merger", "buyout", "takeover",
    "premium", "all-cash deal", "strategic acquisition",

    # Contracts
    "contract", "awarded", "wins", "secures", "partnership",
    "agreement", "deal", "collaboration", "joint venture",

    # Analyst
    "upgrade", "upgraded", "buy rating", "outperform rating",
    "price target raised", "bullish", "positive outlook",

    # General
    "surge", "soars", "jumps", "rallies", "skyrockets",
    "breakthrough", "innovative", "disruptive", "leading",
}

BEARISH_KEYWORDS = {
    # FDA/Biotech
    "fda rejection", "rejected", "complete response letter", "crl",
    "failed", "negative results", "safety concerns", "clinical hold",
    "missed endpoint", "discontinued", "terminated",

    # Earnings
    "miss", "misses", "missed", "disappoints", "disappointing",
    "weak", "decline", "declining", "loss", "losses",
    "lowered guidance", "cuts outlook", "below expectations",

    # Legal/Regulatory
    "lawsuit", "litigation", "investigation", "sec probe",
    "fraud", "accounting issues", "restatement", "delisted",

    # Analyst
    "downgrade", "downgraded", "sell rating", "underperform",
    "price target cut", "bearish", "negative outlook",

    # General
    "plunges", "crashes", "tumbles", "drops", "falls",
    "warning", "risk", "bankruptcy", "default", "layoffs",
    "restructuring", "writedown", "impairment",
}

URGENCY_KEYWORDS = {
    "breaking": {"breaking", "just announced", "breaking news", "alert",
                 "just in", "developing", "urgent"},
    "high": {"today", "this morning", "announces", "reported",
             "filed", "released", "confirms"},
}


# ============================
# NLP GROK PROMPTS
# ============================

SENTIMENT_ANALYSIS_PROMPT = """
You are a financial sentiment analysis expert. Analyze the following news/text for sentiment.

Return JSON only:
{
  "direction": "very_bullish|bullish|slightly_bullish|neutral|slightly_bearish|bearish|very_bearish",
  "confidence": 0.0 to 1.0,
  "intensity": "weak|moderate|strong",
  "headline_sentiment": -1.0 to 1.0,
  "body_sentiment": -1.0 to 1.0,
  "bullish_factors": ["list of bullish points"],
  "bearish_factors": ["list of bearish points"],
  "key_numbers": {"revenue": "X", "growth": "Y%", etc.}
}

Context: This is for small-cap US stocks. Focus on market-moving sentiment.
Only output valid JSON. No text outside JSON.
"""

ENTITY_EXTRACTION_PROMPT = """
You are a financial entity extraction engine. Extract entities from the text.

Return JSON only:
{
  "tickers": ["ABC", "XYZ"],
  "people": [{"name": "John Doe", "role": "CEO", "company": "ABC Inc"}],
  "products": ["Product Name"],
  "numbers": [{"type": "revenue|contract|price_target|etc", "value": "X", "context": "..."}],
  "organizations": ["Company Name", "FDA", "SEC"]
}

Only extract entities that are explicitly mentioned.
Only output valid JSON. No text outside JSON.
"""

NEWS_CLASSIFICATION_PROMPT = """
You are a financial news classifier. Classify the following news.

Categories:
- fda_regulatory: FDA approvals, trials, drug news
- earnings: Quarterly results, revenue, EPS
- merger_acquisition: M&A, buyouts, takeovers
- analyst_rating: Upgrades, downgrades, price targets
- contract_deal: New contracts, partnerships, deals
- product_launch: New products, services
- management: CEO changes, board changes
- legal: Lawsuits, investigations
- guidance: Forward guidance changes
- insider_activity: Insider buying/selling
- sector_news: Industry-wide news
- macro: Economic, market-wide news
- other: Doesn't fit above

Urgency:
- breaking: Immediate action, just happened
- high: Same-day relevance
- medium: Near-term relevance
- low: Background info
- stale: Old news

Return JSON only:
{
  "category": "category_name",
  "urgency": "urgency_level",
  "relevance_score": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

Only output valid JSON. No text outside JSON.
"""


# ============================
# GROK API CALLS
# ============================

def _call_grok(prompt: str, text: str, temperature: float = 0.1) -> Optional[Dict]:
    """Make Grok API call with error handling"""
    if not GROK_API_KEY:
        logger.warning("GROK_API_KEY not set")
        return None

    try:
        payload = {
            "model": "grok-3-fast",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": temperature
        }

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        response = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Grok API HTTP {response.status_code}: {response.text[:200]}")
            return None
        result = response.json()

        content = result["choices"][0]["message"]["content"]

        # Clean up potential markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Grok API error: {e}")
        return None


# S5-1 FIX: Unified Grok prompt — replaces 3 separate calls with 1
UNIFIED_NLP_PROMPT = """
You are a financial NLP engine for small-cap US stocks. Analyze the headline and body text.

Return a single JSON object with THREE sections (sentiment, entities, classification):

{
  "sentiment": {
    "direction": "very_bullish|bullish|slightly_bullish|neutral|slightly_bearish|bearish|very_bearish",
    "confidence": 0.0,
    "intensity": "weak|moderate|strong",
    "headline_sentiment": 0.0,
    "body_sentiment": 0.0,
    "bullish_factors": [],
    "bearish_factors": []
  },
  "entities": {
    "tickers": ["ABC"],
    "people": [{"name": "...", "role": "...", "company": "..."}],
    "products": ["Product Name"],
    "numbers": [{"type": "revenue|price_target|contract|etc", "value": "...", "context": "..."}]
  },
  "classification": {
    "category": "fda_regulatory|earnings|merger_acquisition|analyst_rating|contract_deal|product_launch|management|legal|guidance|insider_activity|sector_news|macro|other",
    "urgency": "breaking|high|medium|low|stale",
    "relevance_score": 0.0,
    "reasoning": "brief explanation"
  }
}

Context: Small-cap US stock news. Focus on market-moving events.
Only output valid JSON. No text outside JSON.
"""


def _call_grok_unified(headline: str, body: str) -> Optional[Dict]:
    """
    S5-1 FIX: Single Grok call returning sentiment + entities + classification.
    Reduces API calls from 3 to 1 per news item (-67% Grok usage).
    """
    if not GROK_API_KEY:
        return None

    text = f"Headline: {headline}\n\nBody: {body[:800]}"

    try:
        payload = {
            "model": "grok-3-fast",
            "messages": [
                {"role": "system", "content": UNIFIED_NLP_PROMPT},
                {"role": "user", "content": text}
            ],
            "temperature": 0.1
        }
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        response = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Unified Grok HTTP {response.status_code}")
            return None
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        # Strip markdown fences
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[: content.rfind("```")]
        return json.loads(content.strip())
    except Exception as e:
        logger.debug(f"Unified Grok call failed: {e}")
        return None


# ============================
# KEYWORD-BASED ANALYSIS (Fast)
# ============================

def analyze_sentiment_keywords(text: str) -> Tuple[float, List[str], List[str]]:
    """
    Fast keyword-based sentiment analysis.
    Returns (score, bullish_keywords_found, bearish_keywords_found)
    """
    text_lower = text.lower()

    bullish_found = []
    bearish_found = []

    for kw in BULLISH_KEYWORDS:
        if kw in text_lower:
            bullish_found.append(kw)

    for kw in BEARISH_KEYWORDS:
        if kw in text_lower:
            bearish_found.append(kw)

    # Calculate score
    bullish_count = len(bullish_found)
    bearish_count = len(bearish_found)
    total = bullish_count + bearish_count

    if total == 0:
        return 0.0, [], []

    score = (bullish_count - bearish_count) / total

    return score, bullish_found, bearish_found


def detect_urgency_keywords(text: str, published_at: datetime) -> NewsUrgency:
    """Detect urgency from keywords and timestamp"""
    text_lower = text.lower()

    # Check for breaking keywords
    for kw in URGENCY_KEYWORDS["breaking"]:
        if kw in text_lower:
            return NewsUrgency.BREAKING

    # Check for high urgency keywords
    for kw in URGENCY_KEYWORDS["high"]:
        if kw in text_lower:
            return NewsUrgency.HIGH

    # Time-based urgency
    hours_old = (datetime.now() - published_at).total_seconds() / 3600

    if hours_old < 1:
        return NewsUrgency.HIGH
    elif hours_old < 6:
        return NewsUrgency.MEDIUM
    elif hours_old < 24:
        return NewsUrgency.LOW
    else:
        return NewsUrgency.STALE


def extract_tickers_regex(text: str) -> List[str]:
    """Extract potential ticker symbols using regex"""
    # Pattern: 1-5 uppercase letters, often preceded by $ or in parentheses
    patterns = [
        r'\$([A-Z]{1,5})\b',           # $AAPL format
        r'\(([A-Z]{1,5})\)',            # (AAPL) format
        r'\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|Inc|Corp|Ltd))',  # AAPL stock
    ]

    tickers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        tickers.update(matches)

    # Filter out common false positives
    false_positives = {"CEO", "CFO", "FDA", "SEC", "IPO", "NYSE", "NASDAQ",
                       "ETF", "USD", "USA", "EPS", "YOY", "QOQ", "API"}
    tickers = tickers - false_positives

    return list(tickers)


# ============================
# MAIN ENRICHMENT ENGINE
# ============================

class NLPEnrichi:
    """
    Advanced NLP Enrichment Engine V6

    Provides:
    - Enhanced sentiment analysis
    - Entity extraction
    - News classification
    - Multi-source aggregation
    """

    def __init__(
        self,
        db_path: str = "data/nlp_sentiment.db",
        use_grok: bool = True,
        sentiment_lookback_hours: int = 24
    ):
        """
        Initialize NLP Enrichi engine.

        Args:
            db_path: Path to SQLite database for history
            use_grok: Whether to use Grok API (vs keyword-only)
            sentiment_lookback_hours: Hours to look back for aggregation
        """
        self.db_path = Path(db_path)
        self.use_grok = use_grok and bool(GROK_API_KEY)
        self.lookback_hours = sentiment_lookback_hours

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"NLPEnrichi initialized (grok={self.use_grok})")

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()

            # Sentiment history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    sentiment_score REAL,
                    confidence REAL,
                    direction TEXT,
                    source TEXT,
                    category TEXT,
                    urgency TEXT,
                    headline TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Aggregated sentiment (cached)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_sentiment (
                    ticker TEXT PRIMARY KEY,
                    overall_sentiment REAL,
                    confidence REAL,
                    direction TEXT,
                    news_count INTEGER,
                    source_count INTEGER,
                    last_updated DATETIME
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_ticker
                ON sentiment_history(ticker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp
                ON sentiment_history(timestamp)
            """)

            conn.commit()

    def analyze_sentiment(
        self,
        text: str,
        headline: Optional[str] = None
    ) -> SentimentAnalysis:
        """
        Analyze sentiment of text.

        Uses Grok if available, falls back to keyword analysis.
        """
        # Fast keyword analysis (always run)
        kw_score, bullish_kw, bearish_kw = analyze_sentiment_keywords(text)

        if self.use_grok:
            # Enhanced Grok analysis
            full_text = f"Headline: {headline}\n\nBody: {text}" if headline else text
            grok_result = _call_grok(SENTIMENT_ANALYSIS_PROMPT, full_text)

            if grok_result:
                direction = self._parse_direction(grok_result.get("direction", "neutral"))

                return SentimentAnalysis(
                    direction=direction,
                    score=SENTIMENT_SCORES.get(direction, 0.0),
                    confidence=grok_result.get("confidence", 0.7),
                    intensity=grok_result.get("intensity", "moderate"),
                    headline_sentiment=grok_result.get("headline_sentiment", kw_score),
                    body_sentiment=grok_result.get("body_sentiment", kw_score),
                    bullish_keywords=grok_result.get("bullish_factors", bullish_kw),
                    bearish_keywords=grok_result.get("bearish_factors", bearish_kw),
                )

        # Fallback to keyword-based
        direction = self._score_to_direction(kw_score)
        intensity = "strong" if abs(kw_score) > 0.6 else "moderate" if abs(kw_score) > 0.3 else "weak"

        return SentimentAnalysis(
            direction=direction,
            score=kw_score,
            confidence=0.6,  # Lower confidence for keyword-only
            intensity=intensity,
            headline_sentiment=kw_score,
            body_sentiment=kw_score,
            bullish_keywords=bullish_kw,
            bearish_keywords=bearish_kw,
        )

    def extract_entities(self, text: str) -> Tuple[List[str], List[ExtractedEntity]]:
        """
        Extract entities from text.

        Returns (tickers, other_entities)
        """
        # Fast regex extraction for tickers
        tickers = extract_tickers_regex(text)

        entities = []

        if self.use_grok:
            grok_result = _call_grok(ENTITY_EXTRACTION_PROMPT, text)

            if grok_result:
                # Merge Grok tickers with regex tickers
                grok_tickers = grok_result.get("tickers", [])
                tickers = list(set(tickers + grok_tickers))

                # Extract other entities
                for person in grok_result.get("people", []):
                    entities.append(ExtractedEntity(
                        entity_type="person",
                        value=person.get("name", ""),
                        context=f"{person.get('role', '')} at {person.get('company', '')}",
                        confidence=0.8
                    ))

                for product in grok_result.get("products", []):
                    entities.append(ExtractedEntity(
                        entity_type="product",
                        value=product,
                        context="",
                        confidence=0.7
                    ))

                for number in grok_result.get("numbers", []):
                    entities.append(ExtractedEntity(
                        entity_type="number",
                        value=str(number.get("value", "")),
                        context=number.get("context", ""),
                        confidence=0.9
                    ))

        return tickers, entities

    def classify_news(
        self,
        text: str,
        headline: str,
        published_at: datetime
    ) -> Tuple[NewsCategory, NewsUrgency, float]:
        """
        Classify news category, urgency, and relevance.

        Returns (category, urgency, relevance_score)
        """
        # Fast urgency detection
        urgency = detect_urgency_keywords(headline + " " + text, published_at)

        # Default category detection via keywords
        category = self._detect_category_keywords(text)
        relevance = 0.5

        if self.use_grok:
            full_text = f"Headline: {headline}\n\nBody: {text[:500]}"  # Truncate for efficiency
            grok_result = _call_grok(NEWS_CLASSIFICATION_PROMPT, full_text)

            if grok_result:
                category_str = grok_result.get("category", "other")
                category = self._parse_category(category_str)

                urgency_str = grok_result.get("urgency", "medium")
                urgency = self._parse_urgency(urgency_str)

                relevance = grok_result.get("relevance_score", 0.5)

        return category, urgency, relevance

    def enrich_news(
        self,
        headline: str,
        body: str,
        source: str,
        published_at: datetime,
        url: Optional[str] = None
    ) -> EnrichedNews:
        """
        Fully enrich a news item with sentiment, entities, and classification.
        """
        # S5-1 FIX: Try unified Grok call (1 API call vs 3) when Grok enabled
        unified = None
        if self.use_grok:
            unified = _call_grok_unified(headline, body)

        if unified:
            # Parse all 3 sections from unified response
            s = unified.get("sentiment", {})
            e = unified.get("entities", {})
            c = unified.get("classification", {})

            direction = self._parse_direction(s.get("direction", "neutral"))
            sentiment = SentimentAnalysis(
                direction=direction,
                score=SENTIMENT_SCORES.get(direction, 0.0),
                confidence=float(s.get("confidence", 0.7)),
                intensity=s.get("intensity", "moderate"),
                headline_sentiment=float(s.get("headline_sentiment", 0.0)),
                body_sentiment=float(s.get("body_sentiment", 0.0)),
                bullish_keywords=s.get("bullish_factors", []),
                bearish_keywords=s.get("bearish_factors", []),
            )

            grok_tickers = e.get("tickers", [])
            regex_tickers = extract_tickers_regex(headline + " " + body)
            tickers = list(set(grok_tickers + regex_tickers))
            entities = []
            for person in e.get("people", []):
                entities.append(ExtractedEntity(
                    entity_type="person",
                    value=person.get("name", ""),
                    context=f"{person.get('role', '')} at {person.get('company', '')}",
                    confidence=0.8
                ))

            category = self._parse_category(c.get("category", "other"))
            urgency = self._parse_urgency(c.get("urgency", "medium"))
            relevance = float(c.get("relevance_score", 0.5))
        else:
            # Fallback: separate calls (or keyword-only if Grok not enabled)
            tickers, entities = self.extract_entities(headline + " " + body)
            sentiment = self.analyze_sentiment(body, headline)
            category, urgency, relevance = self.classify_news(body, headline, published_at)

        # Calculate impact score
        category_impact = CATEGORY_IMPACT.get(category, 0.2)
        sentiment_abs = abs(sentiment.score)
        impact_score = (category_impact * 0.5) + (sentiment_abs * 0.3) + (relevance * 0.2)

        # Calculate final score (considering all factors)
        urgency_factor = {
            NewsUrgency.BREAKING: 1.3,
            NewsUrgency.HIGH: 1.15,
            NewsUrgency.MEDIUM: 1.0,
            NewsUrgency.LOW: 0.85,
            NewsUrgency.STALE: 0.6,
        }.get(urgency, 1.0)

        final_score = min(1.0, impact_score * urgency_factor * sentiment.confidence)

        enriched = EnrichedNews(
            headline=headline,
            body=body,
            source=source,
            published_at=published_at,
            url=url,
            tickers=tickers,
            entities=entities,
            category=category,
            urgency=urgency,
            sentiment=sentiment,
            relevance_score=relevance,
            impact_score=impact_score,
            final_score=final_score,
        )

        # Record sentiment for each ticker
        for ticker in tickers:
            self._record_sentiment(ticker, enriched)

        logger.debug(
            f"Enriched: {headline[:50]}... | "
            f"tickers={tickers} cat={category.value} "
            f"sentiment={sentiment.score:.2f} final={final_score:.2f}"
        )

        return enriched

    def aggregate_sentiment(
        self,
        ticker: str,
        hours: Optional[int] = None
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment for a ticker across recent news.
        """
        if hours is None:
            hours = self.lookback_hours

        cutoff = datetime.now() - timedelta(hours=hours)

        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()

                # Get recent sentiment records
                cursor.execute("""
                    SELECT sentiment_score, confidence, direction, source,
                           category, timestamp
                    FROM sentiment_history
                    WHERE ticker = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (ticker, cutoff.isoformat()))

                rows = cursor.fetchall()

                if not rows:
                    return AggregatedSentiment(
                        ticker=ticker,
                        overall_sentiment=0.0,
                        sentiment_confidence=0.0,
                        sentiment_direction=SentimentDirection.NEUTRAL,
                        source_count=0,
                        news_count=0,
                    )

                # Time-weighted aggregation
                total_weight = 0
                weighted_sentiment = 0
                confidence_sum = 0
                sources = set()
                news_sentiment_sum = 0
                news_count = 0

                for row in rows:
                    score, conf, direction, source, category, ts = row

                    # Parse timestamp
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)

                    # Calculate time weight (newer = higher weight)
                    hours_old = (datetime.now() - ts).total_seconds() / 3600
                    time_weight = max(0.1, 1.0 - (hours_old / hours))

                    # Weight by confidence too
                    weight = time_weight * conf

                    weighted_sentiment += score * weight
                    total_weight += weight
                    confidence_sum += conf
                    sources.add(source)

                    if source in ("news", "finnhub", "finnhub_company", "sec_8k", "finnhub_fallback"):
                        news_sentiment_sum += score
                        news_count += 1

                # Calculate aggregates
                overall = weighted_sentiment / total_weight if total_weight > 0 else 0
                avg_confidence = confidence_sum / len(rows) if rows else 0
                direction = self._score_to_direction(overall)

                news_sentiment = news_sentiment_sum / news_count if news_count > 0 else 0

                # Check for sentiment shift (get previous period)
                prev_cutoff = cutoff - timedelta(hours=hours)
                cursor.execute("""
                    SELECT AVG(sentiment_score)
                    FROM sentiment_history
                    WHERE ticker = ? AND timestamp >= ? AND timestamp < ?
                """, (ticker, prev_cutoff.isoformat(), cutoff.isoformat()))

                prev_avg = cursor.fetchone()[0]
                sentiment_shift = overall - prev_avg if prev_avg else 0
                is_spike = abs(sentiment_shift) > 0.3

                result = AggregatedSentiment(
                    ticker=ticker,
                    overall_sentiment=overall,
                    sentiment_confidence=avg_confidence,
                    sentiment_direction=direction,
                    news_sentiment=news_sentiment,
                    social_sentiment=0.0,  # TODO: integrate social
                    analyst_sentiment=0.0,  # TODO: integrate analyst
                    source_count=len(sources),
                    news_count=len(rows),
                    time_weighted=True,
                    sentiment_shift=sentiment_shift,
                    is_sentiment_spike=is_spike,
                )

                # Cache result
                cursor.execute("""
                    INSERT OR REPLACE INTO aggregated_sentiment
                    (ticker, overall_sentiment, confidence, direction,
                     news_count, source_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, overall, avg_confidence, direction.value,
                    len(rows), len(sources), datetime.now().isoformat()
                ))
                conn.commit()

                return result

        except Exception as e:
            logger.error(f"Aggregate sentiment error: {e}")
            return AggregatedSentiment(
                ticker=ticker,
                overall_sentiment=0.0,
                sentiment_confidence=0.0,
                sentiment_direction=SentimentDirection.NEUTRAL,
            )

    def get_sentiment_boost(self, ticker: str) -> float:
        """
        Get sentiment boost multiplier for Monster Score.

        Returns 0.8-1.3 based on sentiment.
        """
        agg = self.aggregate_sentiment(ticker)

        # No data = neutral boost
        if agg.news_count == 0:
            return 1.0

        # Scale sentiment to boost
        # Very bullish (1.0) = 1.3x
        # Neutral (0.0) = 1.0x
        # Very bearish (-1.0) = 0.7x
        boost = 1.0 + (agg.overall_sentiment * 0.3 * agg.sentiment_confidence)

        # Spike bonus
        if agg.is_sentiment_spike and agg.overall_sentiment > 0:
            boost *= 1.1

        return max(0.7, min(1.4, boost))

    def _record_sentiment(self, ticker: str, enriched: EnrichedNews):
        """Record sentiment to database"""
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sentiment_history
                    (ticker, sentiment_score, confidence, direction, source,
                     category, urgency, headline, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    enriched.sentiment.score if enriched.sentiment else 0,
                    enriched.sentiment.confidence if enriched.sentiment else 0,
                    enriched.sentiment.direction.value if enriched.sentiment else "neutral",
                    enriched.source,
                    enriched.category.value,
                    enriched.urgency.value,
                    enriched.headline,
                    enriched.published_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Record sentiment error: {e}")

    def _parse_direction(self, direction_str: str) -> SentimentDirection:
        """Parse direction string to enum"""
        mapping = {
            "very_bullish": SentimentDirection.VERY_BULLISH,
            "bullish": SentimentDirection.BULLISH,
            "slightly_bullish": SentimentDirection.SLIGHTLY_BULLISH,
            "neutral": SentimentDirection.NEUTRAL,
            "slightly_bearish": SentimentDirection.SLIGHTLY_BEARISH,
            "bearish": SentimentDirection.BEARISH,
            "very_bearish": SentimentDirection.VERY_BEARISH,
        }
        return mapping.get(direction_str.lower(), SentimentDirection.NEUTRAL)

    def _score_to_direction(self, score: float) -> SentimentDirection:
        """Convert numeric score to direction"""
        if score >= 0.7:
            return SentimentDirection.VERY_BULLISH
        elif score >= 0.4:
            return SentimentDirection.BULLISH
        elif score >= 0.15:
            return SentimentDirection.SLIGHTLY_BULLISH
        elif score > -0.15:
            return SentimentDirection.NEUTRAL
        elif score > -0.4:
            return SentimentDirection.SLIGHTLY_BEARISH
        elif score > -0.7:
            return SentimentDirection.BEARISH
        else:
            return SentimentDirection.VERY_BEARISH

    def _parse_category(self, category_str: str) -> NewsCategory:
        """Parse category string to enum"""
        mapping = {
            "fda_regulatory": NewsCategory.FDA_REGULATORY,
            "earnings": NewsCategory.EARNINGS,
            "merger_acquisition": NewsCategory.MERGER_ACQUISITION,
            "analyst_rating": NewsCategory.ANALYST_RATING,
            "contract_deal": NewsCategory.CONTRACT_DEAL,
            "product_launch": NewsCategory.PRODUCT_LAUNCH,
            "management": NewsCategory.MANAGEMENT,
            "legal": NewsCategory.LEGAL,
            "guidance": NewsCategory.GUIDANCE,
            "insider_activity": NewsCategory.INSIDER_ACTIVITY,
            "sector_news": NewsCategory.SECTOR_NEWS,
            "macro": NewsCategory.MACRO,
        }
        return mapping.get(category_str.lower(), NewsCategory.OTHER)

    def _parse_urgency(self, urgency_str: str) -> NewsUrgency:
        """Parse urgency string to enum"""
        mapping = {
            "breaking": NewsUrgency.BREAKING,
            "high": NewsUrgency.HIGH,
            "medium": NewsUrgency.MEDIUM,
            "low": NewsUrgency.LOW,
            "stale": NewsUrgency.STALE,
        }
        return mapping.get(urgency_str.lower(), NewsUrgency.MEDIUM)

    def _detect_category_keywords(self, text: str) -> NewsCategory:
        """Detect category from keywords (fallback)"""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["fda", "approval", "trial", "drug", "therapy"]):
            return NewsCategory.FDA_REGULATORY
        if any(kw in text_lower for kw in ["earnings", "revenue", "eps", "quarter"]):
            return NewsCategory.EARNINGS
        if any(kw in text_lower for kw in ["merger", "acquisition", "buyout", "takeover"]):
            return NewsCategory.MERGER_ACQUISITION
        if any(kw in text_lower for kw in ["upgrade", "downgrade", "price target", "rating"]):
            return NewsCategory.ANALYST_RATING
        if any(kw in text_lower for kw in ["contract", "deal", "partnership", "agreement"]):
            return NewsCategory.CONTRACT_DEAL
        if any(kw in text_lower for kw in ["guidance", "outlook", "forecast"]):
            return NewsCategory.GUIDANCE

        return NewsCategory.OTHER


# ============================
# CONVENIENCE FUNCTIONS
# ============================

_nlp_singleton: Optional["NLPEnrichi"] = None
_nlp_singleton_lock = None  # Lazy import threading


def _get_nlp_lock():
    import threading
    global _nlp_singleton_lock
    if _nlp_singleton_lock is None:
        _nlp_singleton_lock = threading.Lock()
    return _nlp_singleton_lock


def get_nlp_enrichi() -> "NLPEnrichi":
    """S5-1 FIX: Thread-safe singleton — avoids creating a new NLPEnrichi per call."""
    global _nlp_singleton
    if _nlp_singleton is None:
        with _get_nlp_lock():
            if _nlp_singleton is None:
                _nlp_singleton = NLPEnrichi()
    return _nlp_singleton


def get_nlp_sentiment_boost(ticker: str, nlp: Optional[NLPEnrichi] = None) -> float:
    """
    Convenience function to get NLP sentiment boost for Monster Score.
    S5-1 FIX: Uses singleton to avoid creating a new instance on every call.
    """
    if nlp is None:
        nlp = get_nlp_enrichi()
    return nlp.get_sentiment_boost(ticker)


def enrich_news_batch(
    news_items: List[Dict],
    nlp: Optional[NLPEnrichi] = None
) -> List[EnrichedNews]:
    """
    Batch enrich multiple news items.
    """
    if nlp is None:
        nlp = NLPEnrichi()

    results = []
    for item in news_items:
        enriched = nlp.enrich_news(
            headline=item.get("headline", ""),
            body=item.get("body", item.get("summary", "")),
            source=item.get("source", "unknown"),
            published_at=datetime.fromisoformat(item["published_at"])
                if isinstance(item.get("published_at"), str)
                else item.get("published_at", datetime.now()),
            url=item.get("url")
        )
        results.append(enriched)

    return results


# ============================
# TESTING
# ============================

if __name__ == "__main__":
    # Initialize engine
    nlp = NLPEnrichi(db_path="data/test_nlp.db", use_grok=False)  # Test without Grok

    # Test news item
    test_headline = "ABCD Receives FDA Approval for Breakthrough Cancer Treatment"
    test_body = """
    ABCD Inc. (NASDAQ: ABCD) announced today that the FDA has granted approval
    for its innovative cancer drug, marking a major breakthrough in treatment options.
    The company expects strong revenue growth following this approval.
    CEO John Smith stated this represents a significant milestone for patients.
    """

    # Enrich
    enriched = nlp.enrich_news(
        headline=test_headline,
        body=test_body,
        source="test",
        published_at=datetime.now()
    )

    print("\n" + "="*60)
    print("NLP ENRICHI V6 TEST")
    print("="*60)
    print(f"\nHeadline: {enriched.headline}")
    print(f"\nExtracted Tickers: {enriched.tickers}")
    print(f"Category: {enriched.category.value}")
    print(f"Urgency: {enriched.urgency.value}")
    print(f"\nSentiment:")
    if enriched.sentiment:
        print(f"  Direction: {enriched.sentiment.direction.value}")
        print(f"  Score: {enriched.sentiment.score:.3f}")
        print(f"  Confidence: {enriched.sentiment.confidence:.3f}")
        print(f"  Intensity: {enriched.sentiment.intensity}")
        print(f"  Bullish Keywords: {enriched.sentiment.bullish_keywords}")
        print(f"  Bearish Keywords: {enriched.sentiment.bearish_keywords}")
    print(f"\nScores:")
    print(f"  Relevance: {enriched.relevance_score:.3f}")
    print(f"  Impact: {enriched.impact_score:.3f}")
    print(f"  Final: {enriched.final_score:.3f}")
    print("="*60)

    # Test aggregation
    agg = nlp.aggregate_sentiment("ABCD")
    print(f"\nAggregated Sentiment for ABCD:")
    print(f"  Overall: {agg.overall_sentiment:.3f}")
    print(f"  Direction: {agg.sentiment_direction.value}")
    print(f"  Boost: {nlp.get_sentiment_boost('ABCD'):.2f}x")
    print("="*60)
