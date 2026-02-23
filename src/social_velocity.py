"""
SOCIAL VELOCITY ENGINE — GV2-EDGE V9 (A12)
=============================================

Mesure l'acceleration des mentions sociales plutot que le volume brut.
Remplace social_buzz.py comme source primaire pour le composant
social_buzz_score du Monster Score.

Concept: Ce qui compte c'est l'ACCELERATION des mentions, pas le nombre.
Un ticker passant de 2 a 20 mentions/h est plus interessant qu'un ticker
stable a 100 mentions/h.

Sources:
- Reddit (PRAW): r/wallstreetbets, r/pennystocks, r/stocks
- StockTwits: Messages + sentiment labels
- Grok/xAI: Classification NLP
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import deque

from utils.logger import get_logger
from utils.cache import TTLCache

logger = get_logger("SOCIAL_VELOCITY")


# ============================================================================
# Configuration
# ============================================================================

VELOCITY_CACHE_TTL = 300         # 5 min
HISTORY_WINDOW_HOURS = 24        # Historique 24h
SNAPSHOT_INTERVAL_MIN = 15       # Snapshot toutes les 15 min
MAX_SNAPSHOTS = 96               # 24h / 15min = 96 snapshots
SPIKE_ZSCORE = 3.0               # Z-score pour spike de mentions
ACCELERATION_THRESHOLD = 2.0     # 2x acceleration = significatif


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MentionSnapshot:
    """Snapshot des mentions a un instant donne."""
    timestamp: datetime
    reddit_count: int = 0
    stocktwits_count: int = 0
    twitter_count: int = 0
    total_count: int = 0
    sentiment_bullish: int = 0
    sentiment_bearish: int = 0
    sentiment_ratio: float = 0.0   # bullish / (bullish + bearish)


@dataclass
class SocialVelocity:
    """Mesure de velocite sociale pour un ticker."""
    ticker: str
    timestamp: datetime

    # Counts
    mention_count_1h: int = 0
    mention_count_4h: int = 0
    mention_count_24h: int = 0

    # Velocity = 1st derivative (mentions/heure)
    velocity: float = 0.0
    velocity_1h: float = 0.0       # Changement derniere heure
    velocity_4h: float = 0.0       # Changement 4 heures

    # Acceleration = 2nd derivative
    acceleration: float = 0.0

    # Z-score vs baseline 7 jours
    velocity_zscore: float = 0.0

    # Sentiment dynamics
    sentiment_shift: float = 0.0    # Changement sentiment (-1 a +1)
    sentiment_current: float = 0.5  # Sentiment actuel (0=bearish, 1=bullish)

    # Keywords
    top_keywords: List[str] = field(default_factory=list)

    # Composite score
    social_score: float = 0.0       # 0-1

    # Signals
    signals: List[str] = field(default_factory=list)


# ============================================================================
# Social Velocity Engine
# ============================================================================

class SocialVelocityEngine:
    """
    Moteur de velocite sociale.

    Au lieu de compter les mentions (social_buzz.py),
    mesure leur ACCELERATION pour detecter les mouvements naissants.

    Pipeline:
    1. Collecte snapshots periodiques (toutes les 15 min)
    2. Calcule velocite (1st derivative)
    3. Calcule acceleration (2nd derivative)
    4. Compare au baseline (z-score)
    5. Produit un score composite
    """

    def __init__(self):
        self._snapshots: Dict[str, deque] = {}  # ticker -> deque of MentionSnapshot
        self._baselines: Dict[str, Dict] = {}   # ticker -> {mean, std} over 7 days
        self._cache = TTLCache(default_ttl=VELOCITY_CACHE_TTL)
        self._lock = threading.Lock()
        logger.info("SocialVelocityEngine initialized")

    def record_snapshot(self, ticker: str, snapshot: MentionSnapshot) -> None:
        """Enregistre un snapshot de mentions."""
        ticker = ticker.upper()
        with self._lock:
            if ticker not in self._snapshots:
                self._snapshots[ticker] = deque(maxlen=MAX_SNAPSHOTS)
            self._snapshots[ticker].append(snapshot)

    def record_mentions(self, ticker: str, reddit: int = 0, stocktwits: int = 0,
                        twitter: int = 0, bullish: int = 0, bearish: int = 0) -> None:
        """Raccourci pour enregistrer un snapshot."""
        total = reddit + stocktwits + twitter
        ratio = bullish / max(1, bullish + bearish)

        snapshot = MentionSnapshot(
            timestamp=datetime.now(timezone.utc),
            reddit_count=reddit,
            stocktwits_count=stocktwits,
            twitter_count=twitter,
            total_count=total,
            sentiment_bullish=bullish,
            sentiment_bearish=bearish,
            sentiment_ratio=ratio,
        )
        self.record_snapshot(ticker, snapshot)

    def set_baseline(self, ticker: str, mean_mentions_per_hour: float,
                     std_mentions_per_hour: float) -> None:
        """Set baseline historique pour z-score."""
        self._baselines[ticker.upper()] = {
            "mean": mean_mentions_per_hour,
            "std": std_mentions_per_hour,
        }

    def compute_velocity(self, ticker: str) -> SocialVelocity:
        """
        Calcule la velocite sociale pour un ticker.

        Returns:
            SocialVelocity avec velocity, acceleration, z-score, score composite
        """
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)

        # Cached?
        cached = self._cache.get(f"vel_{ticker}")
        if cached:
            return cached

        with self._lock:
            snapshots = list(self._snapshots.get(ticker, []))

        result = SocialVelocity(ticker=ticker, timestamp=now)

        if len(snapshots) < 2:
            self._cache.set(f"vel_{ticker}", result)
            return result

        # Counts par fenetre
        cutoff_1h = now - timedelta(hours=1)
        cutoff_4h = now - timedelta(hours=4)
        cutoff_24h = now - timedelta(hours=24)

        mentions_1h = sum(s.total_count for s in snapshots if s.timestamp >= cutoff_1h)
        mentions_4h = sum(s.total_count for s in snapshots if s.timestamp >= cutoff_4h)
        mentions_24h = sum(s.total_count for s in snapshots if s.timestamp >= cutoff_24h)

        result.mention_count_1h = mentions_1h
        result.mention_count_4h = mentions_4h
        result.mention_count_24h = mentions_24h

        # Velocity = mentions/heure (derniere heure vs moyenne 4h)
        avg_per_hour_4h = mentions_4h / 4.0 if mentions_4h > 0 else 0
        result.velocity_1h = float(mentions_1h)
        result.velocity_4h = avg_per_hour_4h
        result.velocity = mentions_1h - avg_per_hour_4h

        # Acceleration = changement de velocity
        # Comparer derniere heure vs heure precedente
        cutoff_2h = now - timedelta(hours=2)
        mentions_prev_hour = sum(
            s.total_count for s in snapshots
            if cutoff_2h <= s.timestamp < cutoff_1h
        )
        result.acceleration = float(mentions_1h - mentions_prev_hour)

        # Z-score vs baseline
        baseline = self._baselines.get(ticker, {})
        baseline_mean = baseline.get("mean", avg_per_hour_4h)
        baseline_std = baseline.get("std", max(1, avg_per_hour_4h * 0.5))

        if baseline_std > 0:
            result.velocity_zscore = (mentions_1h - baseline_mean) / baseline_std

        # Sentiment dynamics
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_1h]
        older_snapshots = [s for s in snapshots if cutoff_4h <= s.timestamp < cutoff_1h]

        if recent_snapshots:
            current_sentiment = sum(s.sentiment_ratio for s in recent_snapshots) / len(recent_snapshots)
            result.sentiment_current = current_sentiment

            if older_snapshots:
                old_sentiment = sum(s.sentiment_ratio for s in older_snapshots) / len(older_snapshots)
                result.sentiment_shift = current_sentiment - old_sentiment

        # Signals
        signals = []
        if result.velocity_zscore >= SPIKE_ZSCORE:
            signals.append(f"MENTION_SPIKE(z={result.velocity_zscore:.1f})")
        elif result.velocity_zscore >= 2.0:
            signals.append(f"MENTION_SURGE(z={result.velocity_zscore:.1f})")

        if result.acceleration >= ACCELERATION_THRESHOLD * max(1, avg_per_hour_4h):
            signals.append(f"ACCELERATION({result.acceleration:.0f}/h)")

        if result.sentiment_shift > 0.2:
            signals.append(f"SENTIMENT_BULLISH_SHIFT(+{result.sentiment_shift:.2f})")
        elif result.sentiment_shift < -0.2:
            signals.append(f"SENTIMENT_BEARISH_SHIFT({result.sentiment_shift:.2f})")

        result.signals = signals

        # Composite score
        result.social_score = self._compute_social_score(result)

        self._cache.set(f"vel_{ticker}", result)
        return result

    def _compute_social_score(self, v: SocialVelocity) -> float:
        """
        Score composite 0-1 base sur velocite.

        Composants:
        - Velocity z-score (40%): Acceleration vs baseline
        - Acceleration (25%): 2nd derivative
        - Sentiment shift (20%): Changement sentiment
        - Volume brut (15%): Mentions absolues (normalise)
        """
        score = 0.0

        # 1. Velocity z-score (40%)
        z = v.velocity_zscore
        if z >= 4.0:
            z_component = 1.0
        elif z >= 2.0:
            z_component = 0.5 + (z - 2.0) / 2.0 * 0.5
        elif z >= 1.0:
            z_component = 0.2 + (z - 1.0) * 0.3
        elif z > 0:
            z_component = z * 0.2
        else:
            z_component = 0

        score += z_component * 0.40

        # 2. Acceleration (25%)
        if v.acceleration > 0:
            accel_normalized = min(1.0, v.acceleration / max(1, v.velocity_4h * 3))
            score += accel_normalized * 0.25

        # 3. Sentiment shift (20%)
        if v.sentiment_shift > 0:
            sent_component = min(1.0, v.sentiment_shift / 0.5)
            score += sent_component * 0.20
        elif v.sentiment_current > 0.7:
            score += 0.10  # Sentiment already bullish

        # 4. Volume brut normalise (15%)
        # 50 mentions/h = score 1.0 pour small caps
        vol_normalized = min(1.0, v.mention_count_1h / 50.0)
        score += vol_normalized * 0.15

        return min(1.0, round(score, 4))

    def get_top_trending(self, min_score: float = 0.3, limit: int = 20) -> List[SocialVelocity]:
        """Retourne les tickers trending tries par score."""
        results = []
        with self._lock:
            tickers = list(self._snapshots.keys())

        for ticker in tickers:
            vel = self.compute_velocity(ticker)
            if vel.social_score >= min_score:
                results.append(vel)

        results.sort(key=lambda v: v.social_score, reverse=True)
        return results[:limit]

    def get_social_boost(self, ticker: str) -> Tuple[float, Dict]:
        """
        Boost Monster Score pour social velocity.

        Returns:
            (boost, details)
        """
        vel = self.compute_velocity(ticker)

        boost = 0.0
        if vel.social_score >= 0.7:
            boost = 0.08
        elif vel.social_score >= 0.5:
            boost = 0.05
        elif vel.social_score >= 0.3:
            boost = 0.03

        details = {
            "social_score": vel.social_score,
            "velocity_zscore": round(vel.velocity_zscore, 2),
            "acceleration": round(vel.acceleration, 2),
            "mentions_1h": vel.mention_count_1h,
            "sentiment": round(vel.sentiment_current, 2),
            "signals": vel.signals,
            "boost": boost,
        }

        return boost, details

    def get_status(self) -> Dict:
        """Status du moteur."""
        with self._lock:
            n_tickers = len(self._snapshots)
            total_snapshots = sum(len(s) for s in self._snapshots.values())

        return {
            "tickers_tracked": n_tickers,
            "total_snapshots": total_snapshots,
            "baselines_set": len(self._baselines),
        }


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[SocialVelocityEngine] = None
_engine_lock = threading.Lock()


def get_social_velocity_engine() -> SocialVelocityEngine:
    """Get singleton SocialVelocityEngine instance."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SocialVelocityEngine()
    return _engine
