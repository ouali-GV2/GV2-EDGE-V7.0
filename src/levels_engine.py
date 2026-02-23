"""
SUPPORT/RESISTANCE DYNAMIQUES — GV2-EDGE V9 (A11)
=====================================================

Calcul des niveaux S/R en temps reel a partir des donnees IBKR streaming.

Niveaux calcules:
- VWAP (Volume-Weighted Average Price)
- POC (Point of Control — prix le plus echange)
- HOD/LOD (High/Low of Day)
- PM High/Low (Pre-Market High/Low)
- Previous Close/High
- Dynamic S/R (from volume profile)

Usage: Un ticker avec beaucoup de "room to resistance" a plus de
potentiel de hausse = boost Monster Score.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("LEVELS_ENGINE")


# ============================================================================
# Configuration
# ============================================================================

LEVELS_CACHE_TTL = 60            # 1 min (recalcul frequent)
VOLUME_PROFILE_BINS = 50         # Nombre de bins pour le volume profile
MIN_TOUCHES_FOR_LEVEL = 3       # Minimum 3 touches pour valider un S/R
ROOM_BOOST_THRESHOLD = 5.0      # 5% room to resistance = boost


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TechnicalLevels:
    """Niveaux techniques calcules pour un ticker."""
    ticker: str
    timestamp: datetime

    # Key levels
    vwap: float = 0.0
    poc: float = 0.0               # Point of Control
    hod: float = 0.0               # High of Day
    lod: float = 0.0               # Low of Day
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    previous_close: float = 0.0
    previous_high: float = 0.0

    # Dynamic S/R
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    # Distance metrics
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    room_to_resistance_pct: float = 0.0
    distance_from_support_pct: float = 0.0

    # Derived
    daily_range_pct: float = 0.0
    above_vwap: bool = False
    above_poc: bool = False

    @property
    def has_room(self) -> bool:
        """True si le ticker a > 5% de room avant resistance."""
        return self.room_to_resistance_pct >= ROOM_BOOST_THRESHOLD


# ============================================================================
# Volume Profile
# ============================================================================

class VolumeProfile:
    """
    Profil de volume pour un ticker.

    Construit un histogramme volume vs prix pour identifier:
    - POC (prix le plus echange)
    - Value Area (70% du volume)
    - Support/Resistance de volume (niveaux a fort echange)
    """

    def __init__(self):
        self._price_volume: Dict[str, List[Tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def add_trade(self, ticker: str, price: float, volume: int) -> None:
        """Ajoute un trade au profil."""
        ticker = ticker.upper()
        with self._lock:
            if ticker not in self._price_volume:
                self._price_volume[ticker] = []
            self._price_volume[ticker].append((price, volume))

    def compute(self, ticker: str, num_bins: int = VOLUME_PROFILE_BINS) -> Dict:
        """
        Calcule le volume profile.

        Returns:
            {
                "poc": float,
                "value_area_high": float,
                "value_area_low": float,
                "high_volume_nodes": List[float],
                "low_volume_nodes": List[float],
            }
        """
        ticker = ticker.upper()
        with self._lock:
            trades = self._price_volume.get(ticker, [])

        if len(trades) < 10:
            return {"poc": 0, "value_area_high": 0, "value_area_low": 0,
                    "high_volume_nodes": [], "low_volume_nodes": []}

        # Compute bins
        prices = [t[0] for t in trades]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        if price_range <= 0:
            return {"poc": min_price, "value_area_high": min_price, "value_area_low": min_price,
                    "high_volume_nodes": [], "low_volume_nodes": []}

        bin_size = price_range / num_bins

        # Aggregate volume per bin
        bins = defaultdict(int)
        for price, volume in trades:
            bin_idx = min(int((price - min_price) / bin_size), num_bins - 1)
            bins[bin_idx] += volume

        # POC = bin with highest volume
        poc_bin = max(bins.keys(), key=lambda k: bins[k]) if bins else 0
        poc = min_price + (poc_bin + 0.5) * bin_size

        # Value area = 70% of total volume
        total_vol = sum(bins.values())
        target_vol = total_vol * 0.70
        sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)

        cumul = 0
        va_bins = set()
        for bin_idx, vol in sorted_bins:
            va_bins.add(bin_idx)
            cumul += vol
            if cumul >= target_vol:
                break

        if va_bins:
            va_low = min_price + min(va_bins) * bin_size
            va_high = min_price + (max(va_bins) + 1) * bin_size
        else:
            va_low = min_price
            va_high = max_price

        # High volume nodes (potential S/R)
        avg_vol = total_vol / max(1, len(bins))
        hvn = []
        lvn = []
        for bin_idx in range(num_bins):
            price_level = min_price + (bin_idx + 0.5) * bin_size
            vol = bins.get(bin_idx, 0)
            if vol > avg_vol * 1.5:
                hvn.append(round(price_level, 2))
            elif vol < avg_vol * 0.3 and vol > 0:
                lvn.append(round(price_level, 2))

        return {
            "poc": round(poc, 2),
            "value_area_high": round(va_high, 2),
            "value_area_low": round(va_low, 2),
            "high_volume_nodes": hvn,
            "low_volume_nodes": lvn,
        }

    def reset_ticker(self, ticker: str) -> None:
        """Reset le profil d'un ticker (debut de journee)."""
        ticker = ticker.upper()
        with self._lock:
            self._price_volume.pop(ticker, None)


# ============================================================================
# Levels Engine
# ============================================================================

class LevelsEngine:
    """
    Moteur de calcul S/R dynamiques.

    Sources:
    - IBKR Streaming quotes (HOD, LOD, VWAP)
    - Volume Profile (POC, HVN, LVN)
    - Historical data (previous close, previous high)
    - Pre-market data (PM high/low)
    """

    def __init__(self):
        self._volume_profile = VolumeProfile()
        self._levels_cache = TTLCache(default_ttl=LEVELS_CACHE_TTL)
        self._previous_data: Dict[str, Dict] = {}  # ticker -> {close, high}
        self._pm_data: Dict[str, Dict] = {}         # ticker -> {high, low}
        self._lock = threading.Lock()
        logger.info("LevelsEngine initialized")

    def set_previous_data(self, ticker: str, close: float, high: float) -> None:
        """Set previous day close and high."""
        self._previous_data[ticker.upper()] = {"close": close, "high": high}

    def set_premarket_data(self, ticker: str, high: float, low: float) -> None:
        """Set pre-market high and low."""
        self._pm_data[ticker.upper()] = {"high": high, "low": low}

    def feed_trade(self, ticker: str, price: float, volume: int) -> None:
        """Alimente le volume profile avec un trade."""
        self._volume_profile.add_trade(ticker, price, volume)

    def compute_levels(self, ticker: str, current_price: float = 0,
                       hod: float = 0, lod: float = 0, vwap: float = 0) -> TechnicalLevels:
        """
        Calcule les niveaux S/R dynamiques.

        Args:
            ticker: symbole
            current_price: prix actuel
            hod: high of day
            lod: low of day
            vwap: VWAP actuel

        Returns:
            TechnicalLevels complet
        """
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)

        # Volume profile
        vp = self._volume_profile.compute(ticker)

        # Previous data
        prev = self._previous_data.get(ticker, {})
        pm = self._pm_data.get(ticker, {})

        levels = TechnicalLevels(
            ticker=ticker,
            timestamp=now,
            vwap=vwap,
            poc=vp.get("poc", 0),
            hod=hod,
            lod=lod,
            premarket_high=pm.get("high", 0),
            premarket_low=pm.get("low", 0),
            previous_close=prev.get("close", 0),
            previous_high=prev.get("high", 0),
        )

        # Build S/R from all sources
        all_levels = set()

        # Ajouter les niveaux connus
        for lvl in [levels.vwap, levels.poc, levels.hod, levels.lod,
                     levels.premarket_high, levels.premarket_low,
                     levels.previous_close, levels.previous_high]:
            if lvl > 0:
                all_levels.add(round(lvl, 2))

        # Ajouter les HVN du volume profile
        for hvn in vp.get("high_volume_nodes", []):
            all_levels.add(hvn)

        # Classer en support/resistance
        if current_price > 0:
            levels.support_levels = sorted([l for l in all_levels if l < current_price], reverse=True)
            levels.resistance_levels = sorted([l for l in all_levels if l > current_price])

            # Nearest S/R
            if levels.support_levels:
                levels.nearest_support = levels.support_levels[0]
                levels.distance_from_support_pct = ((current_price - levels.nearest_support) / current_price) * 100

            if levels.resistance_levels:
                levels.nearest_resistance = levels.resistance_levels[0]
                levels.room_to_resistance_pct = ((levels.nearest_resistance - current_price) / current_price) * 100

            # Daily range
            if hod > 0 and lod > 0 and lod > 0:
                levels.daily_range_pct = ((hod - lod) / lod) * 100

            # Relative position
            levels.above_vwap = current_price > vwap if vwap > 0 else False
            levels.above_poc = current_price > levels.poc if levels.poc > 0 else False

        return levels

    def get_room_boost(self, ticker: str, current_price: float = 0,
                       hod: float = 0, lod: float = 0, vwap: float = 0) -> Tuple[float, Dict]:
        """
        Boost Monster Score basé sur room to resistance.

        Returns:
            (boost, details)

        Boost:
        - Room > 10%: +0.10
        - Room > 5%: +0.05
        - Above VWAP + above POC: +0.03 additionnel
        """
        levels = self.compute_levels(ticker, current_price, hod, lod, vwap)

        boost = 0.0

        if levels.room_to_resistance_pct >= 10:
            boost = 0.10
        elif levels.room_to_resistance_pct >= ROOM_BOOST_THRESHOLD:
            boost = 0.05

        if levels.above_vwap and levels.above_poc:
            boost += 0.03

        details = {
            "room_to_resistance_pct": round(levels.room_to_resistance_pct, 2),
            "nearest_resistance": levels.nearest_resistance,
            "nearest_support": levels.nearest_support,
            "above_vwap": levels.above_vwap,
            "above_poc": levels.above_poc,
            "support_count": len(levels.support_levels),
            "resistance_count": len(levels.resistance_levels),
            "boost": boost,
        }

        return boost, details

    def reset_day(self, ticker: str) -> None:
        """Reset pour nouveau jour de trading."""
        self._volume_profile.reset_ticker(ticker)
        self._levels_cache._cache.pop(f"levels_{ticker}", None)

    def get_status(self) -> Dict:
        """Status du moteur."""
        return {
            "tickers_with_previous": len(self._previous_data),
            "tickers_with_pm": len(self._pm_data),
        }


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[LevelsEngine] = None
_engine_lock = threading.Lock()


def get_levels_engine() -> LevelsEngine:
    """Get singleton LevelsEngine instance."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = LevelsEngine()
    return _engine
