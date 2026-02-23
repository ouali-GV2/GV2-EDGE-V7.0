"""
CROSS-TICKER CORRELATION / SECTOR MOMENTUM — GV2-EDGE V9 (A13)
=================================================================

Detecte les mouvements sectoriels coordonnes.
Quand 2+ tickers du meme secteur bougent en meme temps,
c'est un signal de sector catalyst (FDA class effect, etc.).

Usage: Boost Monster Score pour les tickers du secteur en mouvement.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("SECTOR_MOMENTUM")


# ============================================================================
# Configuration
# ============================================================================

SECTOR_CACHE_TTL = 120            # 2 min
MIN_MOVERS_FOR_SIGNAL = 2         # 2+ tickers pour signal sectoriel
MIN_MOVE_PCT = 3.0                # 3% move minimum
SECTOR_BOOST_MAX = 0.08           # Boost max pour sector momentum


# ============================================================================
# Sector Mapping
# ============================================================================

SECTOR_MAP = {
    # Biotech sub-sectors
    "BIOTECH_ONCOLOGY": ["oncology", "cancer", "tumor", "immuno-oncology"],
    "BIOTECH_RARE": ["rare disease", "orphan drug", "gene therapy"],
    "BIOTECH_CNS": ["neurology", "cns", "alzheimer", "parkinson", "depression"],
    "BIOTECH_CARDIO": ["cardiovascular", "heart", "cardiac"],
    "BIOTECH_INFECTIOUS": ["infectious", "antiviral", "vaccine", "covid"],
    "BIOTECH_GENERAL": ["biotech", "biopharmaceutical", "drug development"],

    # Tech sub-sectors
    "TECH_AI": ["artificial intelligence", "machine learning", "AI"],
    "TECH_SEMI": ["semiconductor", "chip", "silicon"],
    "TECH_EV": ["electric vehicle", "EV", "battery", "lithium"],
    "TECH_SAAS": ["software", "cloud", "SaaS"],
    "TECH_CYBER": ["cybersecurity", "security"],

    # Other
    "CANNABIS": ["cannabis", "marijuana", "hemp", "CBD"],
    "CRYPTO_RELATED": ["bitcoin", "crypto", "blockchain"],
    "MINING": ["gold", "silver", "mining", "copper"],
    "ENERGY": ["oil", "natural gas", "solar", "wind energy"],
    "RETAIL": ["retail", "e-commerce", "consumer"],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TickerMove:
    """Mouvement d'un ticker."""
    ticker: str
    change_pct: float
    volume_ratio: float = 1.0   # vs 20-day average
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SectorSignal:
    """Signal de mouvement sectoriel."""
    sector: str
    tickers_moving: List[str]
    avg_move_pct: float
    leader: str                   # Ticker avec le plus grand move
    leader_move_pct: float
    confidence: float             # 0-1
    total_volume_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def strength(self) -> str:
        if len(self.tickers_moving) >= 4 and self.avg_move_pct >= 5:
            return "STRONG"
        elif len(self.tickers_moving) >= 3:
            return "MODERATE"
        return "WEAK"


# ============================================================================
# Sector Momentum Engine
# ============================================================================

class SectorMomentum:
    """
    Detecte les mouvements sectoriels coordonnes.

    Pipeline:
    1. Recoie les mouvements de prix des tickers
    2. Groupe par secteur
    3. Detecte quand 2+ tickers du meme secteur bougent ensemble
    4. Calcule un boost pour tous les tickers du secteur
    """

    def __init__(self):
        self._ticker_sectors: Dict[str, str] = {}     # ticker -> sector
        self._recent_moves: Dict[str, List[TickerMove]] = defaultdict(list)  # sector -> moves
        self._cache = TTLCache(default_ttl=SECTOR_CACHE_TTL)
        self._lock = threading.Lock()
        logger.info("SectorMomentum initialized")

    def register_ticker_sector(self, ticker: str, sector: str) -> None:
        """Associe un ticker a un secteur."""
        self._ticker_sectors[ticker.upper()] = sector

    def register_tickers_batch(self, mapping: Dict[str, str]) -> None:
        """Enregistre plusieurs tickers. {ticker: sector}"""
        for ticker, sector in mapping.items():
            self._ticker_sectors[ticker.upper()] = sector

    def record_move(self, ticker: str, change_pct: float, volume_ratio: float = 1.0) -> None:
        """Enregistre un mouvement de prix."""
        ticker = ticker.upper()
        sector = self._ticker_sectors.get(ticker)
        if not sector:
            return

        if abs(change_pct) < MIN_MOVE_PCT:
            return

        move = TickerMove(
            ticker=ticker,
            change_pct=change_pct,
            volume_ratio=volume_ratio,
        )

        with self._lock:
            self._recent_moves[sector].append(move)

            # Garder seulement les 2 dernieres heures
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            self._recent_moves[sector] = [
                m for m in self._recent_moves[sector]
                if m.timestamp >= cutoff
            ]

    def detect_sector_moves(self) -> List[SectorSignal]:
        """
        Detecte les mouvements sectoriels coordonnes.

        Returns:
            Liste de SectorSignal triee par confidence descendante.
        """
        cached = self._cache.get("sector_signals")
        if cached:
            return cached

        signals = []

        with self._lock:
            for sector, moves in self._recent_moves.items():
                if len(moves) < MIN_MOVERS_FOR_SIGNAL:
                    continue

                # Dedup: un seul move par ticker (le plus recent)
                latest_by_ticker: Dict[str, TickerMove] = {}
                for move in moves:
                    existing = latest_by_ticker.get(move.ticker)
                    if not existing or move.timestamp > existing.timestamp:
                        latest_by_ticker[move.ticker] = move

                unique_moves = list(latest_by_ticker.values())

                # Filtrer par direction (tous dans le meme sens)
                bullish = [m for m in unique_moves if m.change_pct > 0]
                bearish = [m for m in unique_moves if m.change_pct < 0]

                # Utiliser le groupe le plus grand
                dominant = bullish if len(bullish) >= len(bearish) else bearish

                if len(dominant) < MIN_MOVERS_FOR_SIGNAL:
                    continue

                # Calculer les metriques
                tickers = [m.ticker for m in dominant]
                avg_move = sum(m.change_pct for m in dominant) / len(dominant)
                total_vol_ratio = sum(m.volume_ratio for m in dominant)

                # Leader = ticker avec le plus grand move
                leader_move = max(dominant, key=lambda m: abs(m.change_pct))

                # Confidence basee sur nombre de movers et magnitude
                confidence = min(1.0, (len(dominant) / 5.0) * (abs(avg_move) / 10.0))

                signal = SectorSignal(
                    sector=sector,
                    tickers_moving=tickers,
                    avg_move_pct=round(avg_move, 2),
                    leader=leader_move.ticker,
                    leader_move_pct=round(leader_move.change_pct, 2),
                    confidence=round(confidence, 4),
                    total_volume_ratio=round(total_vol_ratio, 2),
                )
                signals.append(signal)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        self._cache.set("sector_signals", signals)
        return signals

    def get_sector_boost(self, ticker: str) -> Tuple[float, Dict]:
        """
        Boost Monster Score si le secteur du ticker est en mouvement.

        Returns:
            (boost, details)

        Boost:
        - STRONG sector move: +0.08
        - MODERATE sector move: +0.05
        - WEAK sector move: +0.03
        """
        ticker = ticker.upper()
        sector = self._ticker_sectors.get(ticker)
        if not sector:
            return 0.0, {}

        signals = self.detect_sector_moves()

        for signal in signals:
            if signal.sector == sector:
                if signal.strength == "STRONG":
                    boost = SECTOR_BOOST_MAX
                elif signal.strength == "MODERATE":
                    boost = 0.05
                else:
                    boost = 0.03

                details = {
                    "sector": sector,
                    "strength": signal.strength,
                    "tickers_moving": signal.tickers_moving,
                    "avg_move_pct": signal.avg_move_pct,
                    "leader": signal.leader,
                    "confidence": signal.confidence,
                    "boost": boost,
                }
                return boost, details

        return 0.0, {}

    def get_status(self) -> Dict:
        """Status du moteur."""
        with self._lock:
            active_sectors = len(self._recent_moves)
            total_moves = sum(len(m) for m in self._recent_moves.values())

        return {
            "registered_tickers": len(self._ticker_sectors),
            "active_sectors": active_sectors,
            "total_recent_moves": total_moves,
            "active_signals": len(self.detect_sector_moves()),
        }


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[SectorMomentum] = None
_engine_lock = threading.Lock()


def get_sector_momentum() -> SectorMomentum:
    """Get singleton SectorMomentum instance."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SectorMomentum()
    return _engine
