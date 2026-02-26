"""
SQUEEZE BOOST V6.1
==================

Boost signal basé sur le short interest et potentiel squeeze.

Sources:
- Finnhub Short Interest (FREE)
- FINRA short data (via scraping/alternative)
- Calculated metrics

Métriques:
- Short Interest Ratio (SIR): shares short / avg daily volume
- Days to Cover (DTC): même chose, en jours
- Short % of Float: shares short / float
- Cost to Borrow (CTB): si disponible

Thresholds (small caps):
- SIR > 5: Elevated
- SIR > 10: High squeeze potential
- SIR > 20: Extreme

Score:
- 0.0-0.3: Low short interest
- 0.3-0.5: Elevated
- 0.5-0.7: High squeeze potential
- 0.7-1.0: Extreme (GME-like)

Rôle:
- Boost (+5-20%) sur Monster Score si catalyst présent
- PAS un signal standalone (besoin catalyst)
- Confluence obligatoire
"""

import asyncio
import aiohttp
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from config import FINNHUB_API_KEY

logger = get_logger("SQUEEZE_BOOST")


# ============================
# Configuration
# ============================

FINNHUB_SHORT_URL = "https://finnhub.io/api/v1/stock/short-interest"

# Short interest thresholds
SIR_ELEVATED = 5.0      # 5 days to cover
SIR_HIGH = 10.0         # 10 days to cover
SIR_EXTREME = 20.0      # 20+ days = extreme

SHORT_FLOAT_ELEVATED = 15.0   # 15% of float
SHORT_FLOAT_HIGH = 25.0       # 25% of float
SHORT_FLOAT_EXTREME = 40.0    # 40%+ = extreme

# Cache TTL (short data updates bi-weekly)
CACHE_TTL = 43200  # 12 hours


# ============================
# Enums
# ============================

class SqueezeSignal(Enum):
    """Squeeze potential level"""
    NONE = "none"
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


# ============================
# Data Classes
# ============================

@dataclass
class ShortData:
    """Raw short interest data"""
    ticker: str
    date: datetime
    short_interest: int = 0        # Shares short
    avg_volume: int = 0            # Average daily volume
    days_to_cover: float = 0.0     # Short interest ratio
    short_float_pct: float = 0.0   # % of float shorted
    # Optional
    cost_to_borrow: Optional[float] = None
    utilization: Optional[float] = None


@dataclass
class SqueezeBoostResult:
    """Result of squeeze analysis"""
    ticker: str
    timestamp: datetime
    # Raw data
    short_data: Optional[ShortData] = None
    has_data: bool = False
    # Analysis
    days_to_cover: float = 0.0
    short_float_pct: float = 0.0
    squeeze_signal: SqueezeSignal = SqueezeSignal.NONE
    # Scores
    boost_score: float = 0.0
    # Context
    needs_catalyst: bool = True
    reason: str = ""


# ============================
# Squeeze Boost Engine
# ============================

class SqueezeBoostEngine:
    """
    Analyzes short interest for squeeze potential

    Usage:
        engine = SqueezeBoostEngine()
        result = await engine.analyze("GME")
        if result.squeeze_signal == SqueezeSignal.HIGH:
            # Only boost if catalyst present
            if has_catalyst:
                monster_score *= (1 + result.boost_score * 0.20)

    IMPORTANT: Squeeze boost should ONLY be applied when
    there is a catalyst present. Short interest alone is not
    a trading signal.
    """

    def __init__(self):
        self._session = None
        self._cache: Dict[str, SqueezeBoostResult] = {}
        self._last_fetch: Dict[str, datetime] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def analyze(self, ticker: str) -> SqueezeBoostResult:
        """
        Analyze squeeze potential for a ticker

        Args:
            ticker: Stock ticker

        Returns:
            SqueezeBoostResult with boost score
        """
        ticker = ticker.upper()
        logger.debug(f"Analyzing squeeze potential for {ticker}")

        # Check cache
        cached = self._get_cached(ticker)
        if cached:
            return cached

        # Fetch short data
        short_data = await self._fetch_short_data(ticker)

        # Build result
        result = SqueezeBoostResult(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            short_data=short_data,
            has_data=short_data is not None
        )

        if not short_data:
            result.reason = "No short data available"
            self._cache[ticker] = result
            return result

        # Analyze
        result.days_to_cover = short_data.days_to_cover
        result.short_float_pct = short_data.short_float_pct

        # Calculate boost score
        self._calculate_boost(result)

        # Cache result
        self._cache[ticker] = result

        return result

    def _get_cached(self, ticker: str) -> Optional[SqueezeBoostResult]:
        """Get cached result if valid"""
        if ticker not in self._cache:
            return None

        result = self._cache[ticker]
        age = (datetime.utcnow() - result.timestamp).total_seconds()

        if age > CACHE_TTL:
            del self._cache[ticker]
            return None

        return result

    async def _fetch_short_data(self, ticker: str) -> Optional[ShortData]:
        """Fetch short interest from Finnhub"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return None

        try:
            session = await self._get_session()

            # Date range (last 30 days)
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(days=30)

            params = {
                "symbol": ticker,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": FINNHUB_API_KEY
            }

            async with session.get(FINNHUB_SHORT_URL, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"Finnhub short data error for {ticker}: {resp.status}")
                    return None

                data = await resp.json()

            if not data or "data" not in data:
                return None

            # Get most recent entry
            entries = data.get("data", [])
            if not entries:
                return None

            latest = entries[-1]

            short_interest = latest.get("shortInterest", 0)
            avg_volume = latest.get("avgDailyVolume", 1)

            # Calculate days to cover
            dtc = short_interest / avg_volume if avg_volume > 0 else 0

            return ShortData(
                ticker=ticker,
                date=datetime.utcnow(),
                short_interest=short_interest,
                avg_volume=avg_volume,
                days_to_cover=dtc,
                short_float_pct=latest.get("shortPercentFloat", 0)  # Already in % (e.g., 15.3 = 15.3%)
            )

        except Exception as e:
            logger.error(f"Error fetching short data for {ticker}: {e}")
            return None

    def _calculate_boost(self, result: SqueezeBoostResult):
        """Calculate squeeze boost score"""
        dtc = result.days_to_cover
        short_pct = result.short_float_pct

        score = 0.0
        reasons = []

        # Days to cover component
        if dtc >= SIR_EXTREME:
            score += 0.5
            reasons.append(f"Extreme DTC ({dtc:.1f})")
        elif dtc >= SIR_HIGH:
            score += 0.35
            reasons.append(f"High DTC ({dtc:.1f})")
        elif dtc >= SIR_ELEVATED:
            score += 0.2
            reasons.append(f"Elevated DTC ({dtc:.1f})")
        else:
            reasons.append(f"Normal DTC ({dtc:.1f})")

        # Short float component
        if short_pct >= SHORT_FLOAT_EXTREME:
            score += 0.5
            reasons.append(f"Extreme short float ({short_pct:.1f}%)")
        elif short_pct >= SHORT_FLOAT_HIGH:
            score += 0.35
            reasons.append(f"High short float ({short_pct:.1f}%)")
        elif short_pct >= SHORT_FLOAT_ELEVATED:
            score += 0.2
            reasons.append(f"Elevated short float ({short_pct:.1f}%)")

        # Cost to borrow bonus (if available)
        if result.short_data and result.short_data.cost_to_borrow:
            ctb = result.short_data.cost_to_borrow
            if ctb >= 50:  # 50%+ CTB = very hard to borrow
                score += 0.15
                reasons.append(f"High CTB ({ctb:.0f}%)")
            elif ctb >= 20:
                score += 0.1

        # Cap score
        result.boost_score = min(1.0, max(0.0, score))

        # Determine signal level
        if result.boost_score >= 0.7:
            result.squeeze_signal = SqueezeSignal.EXTREME
        elif result.boost_score >= 0.5:
            result.squeeze_signal = SqueezeSignal.HIGH
        elif result.boost_score >= 0.3:
            result.squeeze_signal = SqueezeSignal.ELEVATED
        elif result.boost_score >= 0.1:
            result.squeeze_signal = SqueezeSignal.LOW
        else:
            result.squeeze_signal = SqueezeSignal.NONE

        result.reason = ", ".join(reasons)

    async def analyze_batch(
        self,
        tickers: List[str]
    ) -> Dict[str, SqueezeBoostResult]:
        """Analyze multiple tickers"""
        results = {}

        for ticker in tickers:
            try:
                results[ticker] = await self.analyze(ticker)
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.warning(f"Squeeze analysis error for {ticker}: {e}")

        return results

    def get_high_squeeze_potential(
        self,
        results: Dict[str, SqueezeBoostResult],
        min_signal: SqueezeSignal = SqueezeSignal.ELEVATED
    ) -> List[Dict]:
        """Get tickers with high squeeze potential"""
        signal_order = [
            SqueezeSignal.NONE,
            SqueezeSignal.LOW,
            SqueezeSignal.ELEVATED,
            SqueezeSignal.HIGH,
            SqueezeSignal.EXTREME
        ]

        min_index = signal_order.index(min_signal)
        high_potential = []

        for ticker, result in results.items():
            if signal_order.index(result.squeeze_signal) >= min_index:
                high_potential.append({
                    "ticker": ticker,
                    "boost_score": result.boost_score,
                    "signal": result.squeeze_signal.value,
                    "days_to_cover": result.days_to_cover,
                    "short_float_pct": result.short_float_pct,
                    "reason": result.reason
                })

        high_potential.sort(key=lambda x: x["boost_score"], reverse=True)
        return high_potential


# ============================
# Convenience Functions
# ============================

_engine_instance = None
_engine_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_squeeze_engine() -> SqueezeBoostEngine:
    """Get singleton engine instance"""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = SqueezeBoostEngine()
    return _engine_instance


async def quick_squeeze_check(ticker: str) -> SqueezeBoostResult:
    """Quick squeeze analysis for single ticker"""
    engine = get_squeeze_engine()
    return await engine.analyze(ticker)


def apply_squeeze_boost(
    base_score: float,
    squeeze_result: SqueezeBoostResult,
    has_catalyst: bool = False
) -> float:
    """
    Apply squeeze boost to a base score

    IMPORTANT: Only applies if has_catalyst is True.
    Short interest alone is NOT a trading signal.

    Args:
        base_score: Original score (0-1)
        squeeze_result: SqueezeBoostResult
        has_catalyst: Whether a catalyst is present

    Returns:
        Boosted score (0-1)
    """
    # ONLY boost if catalyst present
    if not has_catalyst:
        return base_score

    if squeeze_result.boost_score <= 0:
        return base_score

    # Max 20% boost for squeeze (only with catalyst)
    boost_factor = 1 + (squeeze_result.boost_score * 0.20)
    return min(1.0, base_score * boost_factor)


# ============================
# Module exports
# ============================

__all__ = [
    "SqueezeBoostEngine",
    "SqueezeBoostResult",
    "SqueezeSignal",
    "ShortData",
    "get_squeeze_engine",
    "quick_squeeze_check",
    "apply_squeeze_boost",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        engine = SqueezeBoostEngine()

        # Test with known high short interest tickers
        test_tickers = ["GME", "AMC", "BBBY", "AAPL"]

        print("=" * 60)
        print("SQUEEZE BOOST ENGINE TEST")
        print("=" * 60)

        for ticker in test_tickers:
            print(f"\nAnalyzing {ticker}...")
            result = await engine.analyze(ticker)

            print(f"  Has data: {result.has_data}")
            print(f"  Days to cover: {result.days_to_cover:.1f}")
            print(f"  Short float %: {result.short_float_pct:.1f}%")
            print(f"  Squeeze signal: {result.squeeze_signal.value}")
            print(f"  Boost score: {result.boost_score:.2f}")
            print(f"  Reason: {result.reason}")

            # Test boost application
            base = 0.65
            # Without catalyst
            no_cat = apply_squeeze_boost(base, result, has_catalyst=False)
            # With catalyst
            with_cat = apply_squeeze_boost(base, result, has_catalyst=True)
            print(f"  Example: {base:.2f} -> {no_cat:.2f} (no cat) / {with_cat:.2f} (with cat)")

        await engine.close()

    asyncio.run(test())
