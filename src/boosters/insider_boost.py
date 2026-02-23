"""
INSIDER BOOST V6.1
==================

Boost signal basé sur l'activité insider (SEC Form 4).

Sources:
- SEC EDGAR Form 4 (FREE API)
- Via SECForm4Ingestor (Phase 1)

Métriques:
- Cluster buying: Multiple insiders achètent
- Size relative: Taille vs compensation historique
- Executive vs Director: CEO/CFO = plus de poids
- Timing: Achat récent (< 48h) = plus pertinent

Score:
- 0.0-0.3: Normal/No activity
- 0.3-0.5: Notable insider buying
- 0.5-0.7: Strong insider signal
- 0.7-1.0: Exceptional cluster buying

Rôle:
- Boost léger (+5-15%) sur Monster Score
- Confluence avec autres signaux
- Pas un signal standalone
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

# Import SEC Form 4 ingestor
from src.ingestors.sec_filings_ingestor import SECForm4Ingestor, InsiderTransaction

logger = get_logger("INSIDER_BOOST")


# ============================
# Configuration
# ============================

# Weights for insider types
INSIDER_WEIGHTS = {
    "CEO": 1.0,
    "CFO": 0.9,
    "COO": 0.85,
    "President": 0.85,
    "Director": 0.6,
    "VP": 0.5,
    "Officer": 0.5,
    "10% Owner": 0.7,
    "Other": 0.3,
}

# Time decay (hours)
TIME_DECAY = {
    6: 1.0,    # < 6h = full weight
    24: 0.9,   # < 24h = 90%
    48: 0.7,   # < 48h = 70%
    72: 0.5,   # < 72h = 50%
    168: 0.3,  # < 1 week = 30%
}

# Size thresholds
MIN_TRANSACTION_VALUE = 10000  # $10k minimum
LARGE_TRANSACTION = 100000    # $100k = notable
HUGE_TRANSACTION = 500000     # $500k = significant

# Cluster thresholds
CLUSTER_MIN_INSIDERS = 2  # 2+ insiders = cluster
CLUSTER_MAX_HOURS = 72    # Within 72h


# ============================
# Enums
# ============================

class InsiderSignal(Enum):
    """Insider signal strength"""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXCEPTIONAL = "exceptional"


# ============================
# Data Classes
# ============================

@dataclass
class InsiderBoostResult:
    """Result of insider analysis"""
    ticker: str
    timestamp: datetime
    # Raw data
    transactions: List[InsiderTransaction]
    buy_count: int = 0
    sell_count: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    # Analysis
    unique_buyers: int = 0
    is_cluster: bool = False
    has_executive: bool = False
    largest_buy: float = 0.0
    # Scores
    boost_score: float = 0.0
    signal: InsiderSignal = InsiderSignal.NONE
    # Confidence
    confidence: float = 0.0
    reason: str = ""


# ============================
# Insider Boost Engine
# ============================

class InsiderBoostEngine:
    """
    Analyzes insider activity for signal boosting

    Usage:
        engine = InsiderBoostEngine()
        result = await engine.analyze("AAPL")
        if result.is_cluster:
            monster_score *= (1 + result.boost_score * 0.15)
    """

    def __init__(self, universe: Set[str] = None):
        self.universe = universe or set()
        self.form4_ingestor = SECForm4Ingestor(universe)

        # Cache for recent analyses
        self._cache: Dict[str, InsiderBoostResult] = {}
        self._cache_ttl = 3600  # 1 hour

    def set_universe(self, universe: Set[str]):
        """Update universe"""
        self.universe = universe
        self.form4_ingestor.set_universe(universe)

    async def analyze(
        self,
        ticker: str,
        hours_back: int = 168  # 1 week
    ) -> InsiderBoostResult:
        """
        Analyze insider activity for a ticker

        Args:
            ticker: Stock ticker
            hours_back: How far back to look

        Returns:
            InsiderBoostResult with boost score
        """
        ticker = ticker.upper()
        logger.debug(f"Analyzing insider activity for {ticker}")

        # Check cache
        cached = self._get_cached(ticker)
        if cached:
            return cached

        # Fetch Form 4 transactions
        transactions = await self.form4_ingestor.fetch_form4(
            ticker,
            hours_back=hours_back
        )

        # Build result
        result = InsiderBoostResult(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            transactions=transactions
        )

        if not transactions:
            result.reason = "No insider activity"
            self._cache[ticker] = result
            return result

        # Analyze transactions
        self._analyze_transactions(result)

        # Calculate boost score
        self._calculate_boost(result)

        # Cache result
        self._cache[ticker] = result

        return result

    def _get_cached(self, ticker: str) -> Optional[InsiderBoostResult]:
        """Get cached result if valid"""
        if ticker not in self._cache:
            return None

        result = self._cache[ticker]
        age = (datetime.utcnow() - result.timestamp).total_seconds()

        if age > self._cache_ttl:
            del self._cache[ticker]
            return None

        return result

    def _analyze_transactions(self, result: InsiderBoostResult):
        """Analyze transaction details"""
        buyers = set()
        has_exec = False
        largest_buy = 0.0

        for tx in result.transactions:
            if tx.transaction_type == "BUY":
                result.buy_count += 1
                result.total_buy_value += tx.value
                buyers.add(tx.insider_name)
                largest_buy = max(largest_buy, tx.value)

                # Check for executive
                title = tx.insider_title.upper()
                if any(t in title for t in ["CEO", "CFO", "COO", "PRESIDENT", "CHIEF"]):
                    has_exec = True

            elif tx.transaction_type == "SELL":
                result.sell_count += 1
                result.total_sell_value += tx.value

        result.unique_buyers = len(buyers)
        result.has_executive = has_exec
        result.largest_buy = largest_buy

        # Check for cluster
        if result.unique_buyers >= CLUSTER_MIN_INSIDERS:
            # Verify transactions are within cluster window
            if result.transactions:
                times = [tx.filed_date for tx in result.transactions if tx.transaction_type == "BUY"]
                if times:
                    min_time = min(times)
                    max_time = max(times)
                    spread_hours = (max_time - min_time).total_seconds() / 3600
                    result.is_cluster = spread_hours <= CLUSTER_MAX_HOURS

    def _calculate_boost(self, result: InsiderBoostResult):
        """Calculate boost score"""
        score = 0.0
        reasons = []

        # No buys = no boost
        if result.buy_count == 0:
            result.signal = InsiderSignal.NONE
            result.reason = "No insider buying"
            return

        # Base score from buy value
        if result.total_buy_value >= HUGE_TRANSACTION:
            score += 0.4
            reasons.append(f"Large buy ${result.total_buy_value/1000:.0f}K")
        elif result.total_buy_value >= LARGE_TRANSACTION:
            score += 0.25
            reasons.append(f"Notable buy ${result.total_buy_value/1000:.0f}K")
        elif result.total_buy_value >= MIN_TRANSACTION_VALUE:
            score += 0.1
            reasons.append(f"Buy ${result.total_buy_value/1000:.0f}K")

        # Cluster bonus
        if result.is_cluster:
            score += 0.2
            reasons.append(f"Cluster ({result.unique_buyers} insiders)")

        # Executive bonus
        if result.has_executive:
            score += 0.15
            reasons.append("Executive buying")

        # Multiple buyers bonus
        if result.unique_buyers >= 3:
            score += 0.1
            reasons.append(f"{result.unique_buyers} unique buyers")

        # Time decay - weight by recency
        if result.transactions:
            recent_buys = [
                tx for tx in result.transactions
                if tx.transaction_type == "BUY"
            ]
            if recent_buys:
                most_recent = max(recent_buys, key=lambda x: x.filed_date)
                age_hours = (datetime.utcnow() - most_recent.filed_date).total_seconds() / 3600

                decay = 1.0
                for threshold, factor in sorted(TIME_DECAY.items()):
                    if age_hours <= threshold:
                        decay = factor
                        break
                else:
                    decay = 0.2  # > 1 week

                score *= decay

        # Net buy/sell ratio adjustment
        if result.total_sell_value > result.total_buy_value * 2:
            score *= 0.5  # Heavy selling offsets buying
            reasons.append("Offset by selling")

        # Cap at 1.0
        result.boost_score = min(1.0, max(0.0, score))

        # Determine signal level
        if result.boost_score >= 0.7:
            result.signal = InsiderSignal.EXCEPTIONAL
        elif result.boost_score >= 0.5:
            result.signal = InsiderSignal.STRONG
        elif result.boost_score >= 0.3:
            result.signal = InsiderSignal.MODERATE
        elif result.boost_score >= 0.1:
            result.signal = InsiderSignal.WEAK
        else:
            result.signal = InsiderSignal.NONE

        # Confidence based on data quality
        result.confidence = min(1.0, len(result.transactions) / 5)
        result.reason = ", ".join(reasons) if reasons else "Minimal activity"

    async def analyze_batch(
        self,
        tickers: List[str],
        hours_back: int = 168
    ) -> Dict[str, InsiderBoostResult]:
        """Analyze multiple tickers"""
        results = {}

        for ticker in tickers:
            try:
                results[ticker] = await self.analyze(ticker, hours_back)
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.warning(f"Insider analysis error for {ticker}: {e}")

        return results

    def get_top_insider_activity(
        self,
        results: Dict[str, InsiderBoostResult],
        min_score: float = 0.3
    ) -> List[Dict]:
        """Get tickers with notable insider activity"""
        notable = []

        for ticker, result in results.items():
            if result.boost_score >= min_score:
                notable.append({
                    "ticker": ticker,
                    "boost_score": result.boost_score,
                    "signal": result.signal.value,
                    "is_cluster": result.is_cluster,
                    "has_executive": result.has_executive,
                    "total_buy_value": result.total_buy_value,
                    "reason": result.reason
                })

        notable.sort(key=lambda x: x["boost_score"], reverse=True)
        return notable


# ============================
# Cluster Detection (A8)
# ============================

def detect_insider_cluster(ticker: str, days: int = 7, min_insiders: int = 3) -> Optional[Dict]:
    """
    Detecte les achats groupes d'insiders via SEC Form 4.

    3+ insiders achetant dans 7 jours = forte conviction interne.
    C'est un signal beaucoup plus fort qu'un seul achat insider.

    Args:
        ticker: symbole
        days: fenetre de detection (default 7 jours)
        min_insiders: minimum d'insiders distincts

    Returns:
        Dict with cluster info or None if no cluster detected:
        {
            "ticker": str,
            "insiders": List[str],   # noms des insiders
            "total_value": float,    # valeur totale des achats
            "avg_value": float,
            "has_executive": bool,   # CEO/CFO/COO dans le cluster
            "cluster_score": float,  # 0-1
            "signal": str,           # "CLUSTER_STRONG" / "CLUSTER_MODERATE"
            "days_span": int,
            "boost": float,          # boost Monster Score (0.10 - 0.20)
        }
    """
    try:
        engine = get_insider_engine()

        # Utiliser analyze() existant pour recuperer les transactions
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si deja dans un event loop, utiliser le cache
                result = None
            else:
                result = loop.run_until_complete(engine.analyze(ticker, hours_back=days * 24))
        except RuntimeError:
            result = None

        if not result or result.buy_count == 0:
            return None

        # Compter les acheteurs uniques
        unique_buyers = result.unique_buyers

        if unique_buyers < min_insiders:
            return None

        # Verifier si c'est un vrai cluster (achats rapproches)
        transactions = result.transactions
        buy_txns = [t for t in transactions if getattr(t, 'is_buy', False) or getattr(t, 'transaction_type', '') == 'P']

        if len(buy_txns) < min_insiders:
            return None

        # Noms des insiders
        insider_names = list(set(getattr(t, 'insider_name', 'Unknown') for t in buy_txns))

        # Valeur totale
        total_value = result.total_buy_value
        avg_value = total_value / max(1, len(buy_txns))

        # Score du cluster
        cluster_score = 0.0

        # Nombre d'insiders (plus il y en a, plus c'est fort)
        insider_component = min(1.0, unique_buyers / 5.0)  # 5 insiders = max
        cluster_score += insider_component * 0.35

        # Valeur totale (plus c'est gros, plus c'est conviction)
        if total_value >= 1_000_000:
            value_component = 1.0
        elif total_value >= 500_000:
            value_component = 0.7
        elif total_value >= 100_000:
            value_component = 0.4
        else:
            value_component = 0.2
        cluster_score += value_component * 0.30

        # Presence de C-suite
        has_exec = result.has_executive
        exec_component = 1.0 if has_exec else 0.3
        cluster_score += exec_component * 0.20

        # Concentration temporelle (plus c'est serre, plus c'est coordonne)
        if hasattr(buy_txns[0], 'transaction_date') and hasattr(buy_txns[-1], 'transaction_date'):
            try:
                first = buy_txns[0].transaction_date
                last = buy_txns[-1].transaction_date
                span = abs((last - first).days) if hasattr(last, 'days') else days
                tight_component = 1.0 - (span / max(1, days))
            except Exception:
                tight_component = 0.5
        else:
            tight_component = 0.5
        cluster_score += tight_component * 0.15

        cluster_score = min(1.0, cluster_score)

        # Signal et boost
        if cluster_score >= 0.7 or (has_exec and unique_buyers >= 4):
            signal = "CLUSTER_STRONG"
            boost = 0.20
        elif cluster_score >= 0.4:
            signal = "CLUSTER_MODERATE"
            boost = 0.15
        else:
            signal = "CLUSTER_WEAK"
            boost = 0.10

        return {
            "ticker": ticker,
            "insiders": insider_names[:5],
            "unique_count": unique_buyers,
            "total_value": total_value,
            "avg_value": avg_value,
            "has_executive": has_exec,
            "cluster_score": round(cluster_score, 4),
            "signal": signal,
            "days_span": days,
            "boost": boost,
        }

    except Exception as e:
        logger.debug(f"Insider cluster detection error {ticker}: {e}")
        return None


# ============================
# Convenience Functions
# ============================

_engine_instance = None


def get_insider_engine(universe: Set[str] = None) -> InsiderBoostEngine:
    """Get singleton engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InsiderBoostEngine(universe)
    elif universe:
        _engine_instance.set_universe(universe)
    return _engine_instance


async def quick_insider_check(ticker: str) -> InsiderBoostResult:
    """Quick insider analysis for single ticker"""
    engine = get_insider_engine()
    return await engine.analyze(ticker)


def apply_insider_boost(base_score: float, boost_result: InsiderBoostResult) -> float:
    """
    Apply insider boost to a base score

    Args:
        base_score: Original score (0-1)
        boost_result: InsiderBoostResult

    Returns:
        Boosted score (0-1)
    """
    if boost_result.boost_score <= 0:
        return base_score

    # Max 15% boost
    boost_factor = 1 + (boost_result.boost_score * 0.15)
    return min(1.0, base_score * boost_factor)


# ============================
# Module exports
# ============================

__all__ = [
    "InsiderBoostEngine",
    "InsiderBoostResult",
    "InsiderSignal",
    "get_insider_engine",
    "quick_insider_check",
    "apply_insider_boost",
    "detect_insider_cluster",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        engine = InsiderBoostEngine()

        test_tickers = ["AAPL", "NVDA", "TSLA"]

        print("=" * 60)
        print("INSIDER BOOST ENGINE TEST")
        print("=" * 60)

        for ticker in test_tickers:
            print(f"\nAnalyzing {ticker}...")
            result = await engine.analyze(ticker)

            print(f"  Transactions: {len(result.transactions)}")
            print(f"  Buys: {result.buy_count} (${result.total_buy_value:,.0f})")
            print(f"  Sells: {result.sell_count} (${result.total_sell_value:,.0f})")
            print(f"  Is cluster: {result.is_cluster}")
            print(f"  Has executive: {result.has_executive}")
            print(f"  Boost score: {result.boost_score:.2f}")
            print(f"  Signal: {result.signal.value}")
            print(f"  Reason: {result.reason}")

            # Test boost application
            base = 0.65
            boosted = apply_insider_boost(base, result)
            print(f"  Example: {base:.2f} -> {boosted:.2f}")

    asyncio.run(test())
