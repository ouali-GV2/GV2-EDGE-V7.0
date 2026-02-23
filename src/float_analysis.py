"""
FLOAT TURNOVER & SHORT SQUEEZE SCORE — GV2-EDGE V9 (A9)
=========================================================

Float analysis et detection de squeezes en formation.
Les plus gros gainers sont souvent des short squeezes.

Concept: Le float turnover mesure combien de fois le float entier
a ete echange. Un turnover > 1.0x avec prix en hausse = squeeze en formation.

Sources:
- IBKR: Shortable shares + fee rate (generic tick 236)
- Finnhub: Short interest (delayed 2 weeks, free)
- Yahoo Finance: Float shares, short % float

Integration Monster Score:
- Remplace le composant squeeze (0.04 weight)
- Si squeeze_score > 0.7 ET volume_zscore > 2.0: boost additionnel +0.10
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("FLOAT_ANALYSIS")


# ============================================================================
# Configuration
# ============================================================================

FLOAT_CACHE_TTL = 3600 * 4       # 4h
SI_CACHE_TTL = 3600 * 24         # 24h (donnees delayed)
SQUEEZE_THRESHOLD = 0.60          # Score minimum pour "squeeze candidate"
HIGH_SI_PCT = 20.0               # Short interest > 20% = significatif
CRITICAL_SI_PCT = 30.0           # SI > 30% = critique
HIGH_DTC = 3.0                   # Days to Cover > 3 = significatif
HIGH_TURNOVER = 1.0              # Float turnover > 1x = significatif
CTB_HARD_THRESHOLD = 100.0       # Cost to borrow > 100% = "hard to borrow"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FloatAnalysis:
    """Analyse float complete pour un ticker."""
    ticker: str
    timestamp: datetime

    # Float data
    float_shares: int = 0
    shares_outstanding: int = 0
    float_pct: float = 0.0         # float / outstanding

    # Short interest
    short_interest: int = 0
    short_pct_float: float = 0.0   # short / float
    days_to_cover: float = 0.0     # short / avg_volume

    # Turnover
    turnover_ratio: float = 0.0     # cumulative_volume / float
    turnover_velocity: float = 0.0  # 1st derivative of turnover

    # Cost to borrow
    cost_to_borrow_pct: float = 0.0
    shortable_shares: int = 0
    borrow_status: str = "UNKNOWN"  # EASY, MEDIUM, HARD, UNAVAILABLE

    # Composite score
    squeeze_score: float = 0.0     # 0-1 composite squeeze probability
    squeeze_signals: list = None   # Active squeeze signals

    # Source tracking
    source: str = "composite"

    def __post_init__(self):
        if self.squeeze_signals is None:
            self.squeeze_signals = []

    @property
    def is_squeeze_candidate(self) -> bool:
        return self.squeeze_score >= SQUEEZE_THRESHOLD

    @property
    def si_level(self) -> str:
        if self.short_pct_float >= CRITICAL_SI_PCT:
            return "CRITICAL"
        elif self.short_pct_float >= HIGH_SI_PCT:
            return "HIGH"
        elif self.short_pct_float >= 10.0:
            return "MEDIUM"
        return "LOW"


# ============================================================================
# Float Analyzer Engine
# ============================================================================

class FloatAnalyzer:
    """
    Analyse float + short interest + turnover pour detection squeeze.

    Pipeline:
    1. Fetch float data (IBKR / Yahoo / cache)
    2. Fetch short interest (Finnhub / IBKR)
    3. Compute turnover intraday (volume / float)
    4. Compute squeeze composite score
    """

    def __init__(self):
        self._float_cache = TTLCache(default_ttl=FLOAT_CACHE_TTL)
        self._si_cache = TTLCache(default_ttl=SI_CACHE_TTL)
        self._intraday_volume: Dict[str, int] = {}
        self._lock = threading.Lock()
        logger.info("FloatAnalyzer initialized")

    def analyze(self, ticker: str, current_price: float = 0, volume_today: int = 0) -> FloatAnalysis:
        """
        Analyse complete float + SI + turnover.

        Args:
            ticker: symbole
            current_price: prix actuel
            volume_today: volume cumule aujourd'hui

        Returns:
            FloatAnalysis avec squeeze_score
        """
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)

        # Fetch float data
        float_shares, shares_outstanding = self._get_float(ticker)

        # Fetch short interest
        si_data = self._get_short_interest(ticker)

        # Fetch borrow data
        borrow_data = self._get_borrow_data(ticker)

        # Build analysis
        analysis = FloatAnalysis(
            ticker=ticker,
            timestamp=now,
            float_shares=float_shares,
            shares_outstanding=shares_outstanding,
            float_pct=(float_shares / shares_outstanding * 100) if shares_outstanding > 0 else 0,
            short_interest=si_data.get("short_interest", 0),
            short_pct_float=si_data.get("short_pct_float", 0),
            days_to_cover=si_data.get("days_to_cover", 0),
            cost_to_borrow_pct=borrow_data.get("fee_rate", 0),
            shortable_shares=borrow_data.get("shortable", 0),
            borrow_status=borrow_data.get("status", "UNKNOWN"),
        )

        # Compute turnover
        if float_shares > 0 and volume_today > 0:
            analysis.turnover_ratio = volume_today / float_shares

        # Compute velocity (changement de turnover)
        prev_volume = self._intraday_volume.get(ticker, 0)
        if prev_volume > 0 and float_shares > 0:
            prev_turnover = prev_volume / float_shares
            analysis.turnover_velocity = analysis.turnover_ratio - prev_turnover

        with self._lock:
            self._intraday_volume[ticker] = volume_today

        # Compute squeeze score
        analysis.squeeze_score = self._compute_squeeze_score(analysis)

        return analysis

    def _get_float(self, ticker: str) -> Tuple[int, int]:
        """Recupere les donnees float."""
        cached = self._float_cache.get(f"float_{ticker}")
        if cached:
            return cached

        float_shares = 0
        outstanding = 0

        # Source 1: Yahoo Finance (gratuit)
        try:
            float_shares, outstanding = self._fetch_yahoo_float(ticker)
        except Exception as e:
            logger.debug(f"Yahoo float error {ticker}: {e}")

        # Source 2: Finnhub company profile (fallback)
        if float_shares == 0:
            try:
                float_shares, outstanding = self._fetch_finnhub_float(ticker)
            except Exception as e:
                logger.debug(f"Finnhub float error {ticker}: {e}")

        result = (float_shares, outstanding)
        self._float_cache.set(f"float_{ticker}", result)
        return result

    def _fetch_yahoo_float(self, ticker: str) -> Tuple[int, int]:
        """Fetch float depuis Yahoo Finance."""
        try:
            import requests

            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "defaultKeyStatistics"}
            headers = {"User-Agent": "Mozilla/5.0"}

            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                return 0, 0

            data = resp.json()
            stats = data.get("quoteSummary", {}).get("result", [{}])[0].get("defaultKeyStatistics", {})

            float_shares = stats.get("floatShares", {}).get("raw", 0)
            outstanding = stats.get("sharesOutstanding", {}).get("raw", 0)

            return int(float_shares), int(outstanding)
        except Exception:
            return 0, 0

    def _fetch_finnhub_float(self, ticker: str) -> Tuple[int, int]:
        """Fetch float depuis Finnhub company profile."""
        try:
            from config import FINNHUB_API_KEY
            if not FINNHUB_API_KEY:
                return 0, 0

            import requests

            url = f"https://finnhub.io/api/v1/stock/profile2"
            params = {"symbol": ticker, "token": FINNHUB_API_KEY}

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return 0, 0

            data = resp.json()
            outstanding = int(data.get("shareOutstanding", 0) * 1_000_000)

            # Finnhub ne fournit pas le float directement
            # Estimation: float ~ 80% des outstanding pour small caps
            float_est = int(outstanding * 0.80) if outstanding > 0 else 0
            return float_est, outstanding
        except Exception:
            return 0, 0

    def _get_short_interest(self, ticker: str) -> Dict:
        """Recupere short interest."""
        cached = self._si_cache.get(f"si_{ticker}")
        if cached:
            return cached

        si_data = {"short_interest": 0, "short_pct_float": 0, "days_to_cover": 0}

        # Finnhub short interest
        try:
            from config import FINNHUB_API_KEY
            if FINNHUB_API_KEY:
                import requests

                url = f"https://finnhub.io/api/v1/stock/short-interest"
                params = {"symbol": ticker, "token": FINNHUB_API_KEY}

                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("data"):
                        latest = data["data"][0]  # Most recent
                        si_data["short_interest"] = latest.get("shortInterest", 0)
        except Exception as e:
            logger.debug(f"SI fetch error {ticker}: {e}")

        # Calculer % float et DTC
        float_shares, _ = self._get_float(ticker)
        if float_shares > 0 and si_data["short_interest"] > 0:
            si_data["short_pct_float"] = (si_data["short_interest"] / float_shares) * 100

        self._si_cache.set(f"si_{ticker}", si_data)
        return si_data

    def _get_borrow_data(self, ticker: str) -> Dict:
        """Recupere donnees cost-to-borrow via IBKR."""
        result = {"fee_rate": 0, "shortable": 0, "status": "UNKNOWN"}

        try:
            from src.ibkr_connector import get_ibkr
            ibkr = get_ibkr()
            if ibkr and ibkr.connected:
                # IBKR generic tick 236 = shortable shares
                quote = ibkr.get_quote(ticker, use_cache=True)
                if quote:
                    shortable = quote.get("shortableShares", 0)
                    fee_rate = quote.get("feeRate", 0)

                    result["shortable"] = int(shortable) if shortable else 0
                    result["fee_rate"] = float(fee_rate) if fee_rate else 0

                    if result["fee_rate"] >= CTB_HARD_THRESHOLD:
                        result["status"] = "HARD"
                    elif result["fee_rate"] >= 20:
                        result["status"] = "MEDIUM"
                    elif result["shortable"] > 0:
                        result["status"] = "EASY"
                    else:
                        result["status"] = "UNAVAILABLE"
        except Exception as e:
            logger.debug(f"Borrow data error {ticker}: {e}")

        return result

    def _compute_squeeze_score(self, analysis: FloatAnalysis) -> float:
        """
        Calcule le score composite de squeeze (0-1).

        Composants:
        - Short % float (30%): Plus le SI est eleve, plus le squeeze est probable
        - Days to cover (15%): Plus il faut de jours, plus la pression est forte
        - Float turnover (25%): Turnover > 1x = shorts en difficulte
        - Cost to borrow (15%): CTB eleve = stress sur les shorts
        - Float size (15%): Petit float = plus volatile

        Signaux additifs:
        - HTB + high SI = +0.10
        - Turnover > 2x + rising price = +0.10
        """
        score = 0.0
        signals = []

        # 1. Short % float (30%)
        si_pct = analysis.short_pct_float
        if si_pct >= CRITICAL_SI_PCT:
            si_component = 1.0
            signals.append(f"CRITICAL_SI({si_pct:.1f}%)")
        elif si_pct >= HIGH_SI_PCT:
            si_component = 0.7 + (si_pct - HIGH_SI_PCT) / (CRITICAL_SI_PCT - HIGH_SI_PCT) * 0.3
            signals.append(f"HIGH_SI({si_pct:.1f}%)")
        elif si_pct >= 10:
            si_component = 0.3 + (si_pct - 10) / (HIGH_SI_PCT - 10) * 0.4
        else:
            si_component = si_pct / 10 * 0.3 if si_pct > 0 else 0

        score += si_component * 0.30

        # 2. Days to cover (15%)
        dtc = analysis.days_to_cover
        if dtc >= 5:
            dtc_component = 1.0
            signals.append(f"HIGH_DTC({dtc:.1f})")
        elif dtc >= HIGH_DTC:
            dtc_component = 0.6 + (dtc - HIGH_DTC) / (5 - HIGH_DTC) * 0.4
        elif dtc > 0:
            dtc_component = dtc / HIGH_DTC * 0.6
        else:
            dtc_component = 0

        score += dtc_component * 0.15

        # 3. Float turnover (25%)
        turnover = analysis.turnover_ratio
        if turnover >= 2.0:
            turnover_component = 1.0
            signals.append(f"EXTREME_TURNOVER({turnover:.1f}x)")
        elif turnover >= HIGH_TURNOVER:
            turnover_component = 0.6 + (turnover - HIGH_TURNOVER) / 1.0 * 0.4
            signals.append(f"HIGH_TURNOVER({turnover:.1f}x)")
        elif turnover > 0:
            turnover_component = turnover / HIGH_TURNOVER * 0.6
        else:
            turnover_component = 0

        score += turnover_component * 0.25

        # 4. Cost to borrow (15%)
        ctb = analysis.cost_to_borrow_pct
        if analysis.borrow_status == "UNAVAILABLE":
            ctb_component = 1.0
            signals.append("NO_SHARES_AVAILABLE")
        elif analysis.borrow_status == "HARD":
            ctb_component = 0.8
            signals.append(f"HARD_TO_BORROW({ctb:.0f}%)")
        elif analysis.borrow_status == "MEDIUM":
            ctb_component = 0.4
        else:
            ctb_component = 0

        score += ctb_component * 0.15

        # 5. Float size (15%)
        float_m = analysis.float_shares / 1_000_000 if analysis.float_shares > 0 else 0
        if 0 < float_m <= 5:
            float_component = 1.0
            signals.append(f"MICRO_FLOAT({float_m:.1f}M)")
        elif float_m <= 15:
            float_component = 0.7
            signals.append(f"SMALL_FLOAT({float_m:.1f}M)")
        elif float_m <= 50:
            float_component = 0.3
        else:
            float_component = 0.1

        score += float_component * 0.15

        # Bonus: HTB + high SI
        if analysis.borrow_status in ("HARD", "UNAVAILABLE") and si_pct >= HIGH_SI_PCT:
            score += 0.10
            signals.append("HTB_HIGH_SI_COMBO")

        # Bonus: Extreme turnover + positive velocity
        if turnover >= 2.0 and analysis.turnover_velocity > 0:
            score += 0.10
            signals.append("TURNOVER_ACCELERATING")

        analysis.squeeze_signals = signals
        return min(1.0, round(score, 4))

    def get_squeeze_candidates(self, min_score: float = SQUEEZE_THRESHOLD) -> List[FloatAnalysis]:
        """Retourne les tickers avec squeeze_score > seuil."""
        # Retourne depuis le cache
        candidates = []
        for key in self._float_cache._cache.keys():
            if key.startswith("analysis_"):
                analysis = self._float_cache.get(key)
                if analysis and analysis.squeeze_score >= min_score:
                    candidates.append(analysis)
        candidates.sort(key=lambda a: a.squeeze_score, reverse=True)
        return candidates

    def compute_turnover_intraday(self, ticker: str, volume_today: int) -> float:
        """Turnover intraday en temps reel."""
        float_shares, _ = self._get_float(ticker)
        if float_shares > 0 and volume_today > 0:
            return volume_today / float_shares
        return 0.0

    def get_squeeze_boost(self, ticker: str, volume_today: int = 0, volume_zscore: float = 0) -> Tuple[float, Dict]:
        """
        Boost Monster Score pour squeeze.

        Returns:
            (boost, details)

        Si squeeze_score > 0.7 ET volume_zscore > 2.0: boost +0.10
        """
        analysis = self.analyze(ticker, volume_today=volume_today)

        boost = 0.0
        if analysis.squeeze_score > 0.7 and volume_zscore > 2.0:
            boost = 0.10
        elif analysis.squeeze_score > 0.5:
            boost = 0.05

        details = {
            "squeeze_score": analysis.squeeze_score,
            "si_pct": analysis.short_pct_float,
            "turnover": analysis.turnover_ratio,
            "borrow_status": analysis.borrow_status,
            "signals": analysis.squeeze_signals,
            "boost": boost,
        }

        return boost, details

    def get_status(self) -> Dict:
        """Status de l'analyseur float."""
        return {
            "tickers_with_volume": len(self._intraday_volume),
            "float_cache_size": len(self._float_cache._cache) if hasattr(self._float_cache, '_cache') else 0,
        }


# ============================================================================
# Singleton
# ============================================================================

_analyzer: Optional[FloatAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_float_analyzer() -> FloatAnalyzer:
    """Get singleton FloatAnalyzer instance."""
    global _analyzer
    with _analyzer_lock:
        if _analyzer is None:
            _analyzer = FloatAnalyzer()
    return _analyzer
