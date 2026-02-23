"""
MULTI-RADAR DETECTION ENGINE — GV2-EDGE V9
============================================

4 radars independants surveillent le marche en parallele.
Chacun s'adapte a la session en cours (AH, PM, RTH, CLOSED).

    RADAR A: FLOW        — Volume, prix, derivees, accumulation
    RADAR B: CATALYST    — News, SEC, FDA, earnings, evenements
    RADAR C: SMART MONEY — Options flow, insider activity
    RADAR D: SENTIMENT   — Social buzz, NLP sentiment, repeat gainers

La Confluence Matrix combine les 4 scores independants pour produire
un signal final plus robuste que le scoring unique (Monster Score V4).

Principes:
- Detection JAMAIS bloquee (coherent avec l'architecture V7/V8)
- Chaque radar est autonome (fallback gracieux si module absent)
- Additif, pas multiplicatif (boosts de confluence)
- Session-adaptatif (poids et sensibilites changent par session)
- Ultra-rapide (buffer reads, pas d'API calls dans le scan)

Architecture:
    SessionAdapter ──→ poids + sensibilites par session
                       │
    ┌──────────────────┼──────────────────────────────┐
    │                  │                              │
    FlowRadar    CatalystRadar   SmartMoneyRadar   SentimentRadar
    (parallel)   (parallel)      (parallel)        (parallel)
    │                  │                │               │
    └──────────────────┼──────────────────────────────┘
                       │
               ConfluenceMatrix ──→ ConfluenceSignal
                       │
                  MultiRadarEngine (orchestrateur)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("MULTI_RADAR")


# ================================================================
# ENUMS & DATACLASSES
# ================================================================

class RadarPriority(Enum):
    """Priorite d'un signal radar individuel."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

    def __lt__(self, other):
        order = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return order.index(self.value) < order.index(other.value)


class AgreementLevel(Enum):
    """Niveau d'accord entre les radars."""
    UNANIMOUS = "UNANIMOUS"     # 4/4 radars voient quelque chose
    STRONG = "STRONG"           # 3/4 radars
    MODERATE = "MODERATE"       # 2/4 radars
    WEAK = "WEAK"               # 1/4 radar
    NONE = "NONE"               # Aucun radar


@dataclass
class RadarResult:
    """Resultat d'un scan par un radar individuel."""
    radar_name: str
    ticker: str
    score: float               # 0.0 - 1.0
    confidence: float          # 0.0 - 1.0
    state: str                 # DORMANT/WATCHING/ACCUMULATING/READY/LAUNCHING/BREAKOUT
    priority: RadarPriority
    signals: List[str]         # Declencheurs detectes
    details: Dict[str, Any]    # Donnees specifiques au radar
    scan_time_ms: float        # Performance tracking

    @property
    def level(self) -> str:
        """Classifie le score en HIGH/MEDIUM/LOW."""
        if self.score >= 0.60:
            return "HIGH"
        elif self.score >= 0.30:
            return "MEDIUM"
        return "LOW"

    @property
    def is_active(self) -> bool:
        return self.score >= 0.15 and self.state != "DORMANT"


@dataclass
class ConfluenceSignal:
    """Signal final apres confluence des 4 radars."""
    ticker: str
    final_score: float
    signal_type: str             # BUY_STRONG, BUY, WATCH, EARLY_SIGNAL, NO_SIGNAL
    agreement: AgreementLevel
    lead_radar: str              # Radar avec le score le plus eleve
    radar_results: Dict[str, RadarResult]
    confluence_bonus: float      # Bonus d'accord multi-radar
    session: str
    session_weights: Dict[str, float]
    estimated_lead_time_min: float
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_actionable(self) -> bool:
        return self.signal_type in ("BUY_STRONG", "BUY")

    def get_active_radars(self) -> List[str]:
        return [name for name, r in self.radar_results.items() if r.is_active]

    def get_summary(self) -> str:
        """Resume lisible pour logs et Telegram."""
        active = self.get_active_radars()
        parts = [
            f"{self.signal_type} {self.ticker} | Score: {self.final_score:.2f}",
            f"Agreement: {self.agreement.value} ({len(active)}/4 radars)",
            f"Lead: {self.lead_radar}",
            f"Session: {self.session}",
        ]
        for name, r in self.radar_results.items():
            if r.is_active:
                parts.append(f"  {name}: {r.score:.2f} [{r.state}] {', '.join(r.signals[:3])}")
        return " | ".join(parts[:4]) + "\n" + "\n".join(parts[4:])

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "final_score": round(self.final_score, 4),
            "signal_type": self.signal_type,
            "agreement": self.agreement.value,
            "lead_radar": self.lead_radar,
            "confluence_bonus": round(self.confluence_bonus, 4),
            "session": self.session,
            "estimated_lead_time_min": self.estimated_lead_time_min,
            "confidence": round(self.confidence, 4),
            "radars": {
                name: {
                    "score": round(r.score, 4),
                    "confidence": round(r.confidence, 4),
                    "state": r.state,
                    "signals": r.signals,
                    "scan_time_ms": round(r.scan_time_ms, 2),
                }
                for name, r in self.radar_results.items()
            },
            "timestamp": self.timestamp.isoformat(),
        }


# ================================================================
# SESSION ADAPTER
# ================================================================

class SessionAdapter:
    """
    Adapte les poids et sensibilites des radars selon la session de marche.

    Logique:
    - After-Hours (16:00-20:00): Catalyst dominant (news flow), Flow reduit
    - Pre-Market (04:00-09:30):  Equilibre (gap confirmation + catalyst)
    - RTH Open (09:30-10:30):    Flow + Smart Money au max (ouverture volatile)
    - RTH Midday (10:30-14:30):  Flow dominant, sentiment reduit
    - RTH Close (14:30-16:00):   Equilibre (anticipation lendemain)
    - Closed/Weekend:            Catalyst + Sentiment (scanning batch)
    """

    # Poids de chaque radar par sous-session
    # Contrainte: somme = 1.0 pour chaque session
    PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
        "AFTER_HOURS": {
            "flow":        {"weight": 0.15, "sensitivity": "LOW"},
            "catalyst":    {"weight": 0.45, "sensitivity": "ULTRA"},
            "smart_money": {"weight": 0.10, "sensitivity": "LOW"},
            "sentiment":   {"weight": 0.30, "sensitivity": "HIGH"},
        },
        "PRE_MARKET": {
            "flow":        {"weight": 0.30, "sensitivity": "HIGH"},
            "catalyst":    {"weight": 0.30, "sensitivity": "HIGH"},
            "smart_money": {"weight": 0.15, "sensitivity": "MEDIUM"},
            "sentiment":   {"weight": 0.25, "sensitivity": "HIGH"},
        },
        "RTH_OPEN": {
            "flow":        {"weight": 0.35, "sensitivity": "ULTRA"},
            "catalyst":    {"weight": 0.20, "sensitivity": "HIGH"},
            "smart_money": {"weight": 0.30, "sensitivity": "ULTRA"},
            "sentiment":   {"weight": 0.15, "sensitivity": "MEDIUM"},
        },
        "RTH_MIDDAY": {
            "flow":        {"weight": 0.40, "sensitivity": "HIGH"},
            "catalyst":    {"weight": 0.20, "sensitivity": "MEDIUM"},
            "smart_money": {"weight": 0.25, "sensitivity": "HIGH"},
            "sentiment":   {"weight": 0.15, "sensitivity": "LOW"},
        },
        "RTH_CLOSE": {
            "flow":        {"weight": 0.30, "sensitivity": "HIGH"},
            "catalyst":    {"weight": 0.30, "sensitivity": "HIGH"},
            "smart_money": {"weight": 0.25, "sensitivity": "HIGH"},
            "sentiment":   {"weight": 0.15, "sensitivity": "MEDIUM"},
        },
        "CLOSED": {
            "flow":        {"weight": 0.05, "sensitivity": "LOW"},
            "catalyst":    {"weight": 0.50, "sensitivity": "HIGH"},
            "smart_money": {"weight": 0.05, "sensitivity": "LOW"},
            "sentiment":   {"weight": 0.40, "sensitivity": "HIGH"},
        },
    }

    # Seuils de sensibilite: plus c'est sensible, plus on detecte tot
    SENSITIVITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
        "ULTRA":    {"min_score": 0.12, "min_confidence": 0.15},
        "HIGH":     {"min_score": 0.20, "min_confidence": 0.25},
        "MEDIUM":   {"min_score": 0.35, "min_confidence": 0.35},
        "LOW":      {"min_score": 0.50, "min_confidence": 0.45},
    }

    def get_sub_session(self) -> str:
        """
        Determine la sous-session actuelle avec granularite RTH.

        Returns:
            AFTER_HOURS, PRE_MARKET, RTH_OPEN, RTH_MIDDAY, RTH_CLOSE, CLOSED
        """
        try:
            from utils.time_utils import (
                is_premarket, is_market_open, is_after_hours,
                minutes_since_open, minutes_before_close
            )

            if is_after_hours():
                return "AFTER_HOURS"
            elif is_premarket():
                return "PRE_MARKET"
            elif is_market_open():
                mins_open = minutes_since_open()
                mins_close = minutes_before_close()
                if mins_open <= 60:
                    return "RTH_OPEN"
                elif mins_close <= 90:
                    return "RTH_CLOSE"
                else:
                    return "RTH_MIDDAY"
            else:
                return "CLOSED"
        except Exception:
            return "CLOSED"

    def get_profile(self, session: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Retourne le profil complet pour la session donnee."""
        if session is None:
            session = self.get_sub_session()
        return self.PROFILES.get(session, self.PROFILES["CLOSED"])

    def get_weights(self, session: Optional[str] = None) -> Dict[str, float]:
        """Retourne les poids des 4 radars pour la session."""
        profile = self.get_profile(session)
        return {name: cfg["weight"] for name, cfg in profile.items()}

    def get_sensitivity(self, radar_name: str, session: Optional[str] = None) -> Dict[str, float]:
        """Retourne les seuils de sensibilite pour un radar dans la session."""
        profile = self.get_profile(session)
        sens_level = profile.get(radar_name, {}).get("sensitivity", "MEDIUM")
        return self.SENSITIVITY_THRESHOLDS.get(sens_level, self.SENSITIVITY_THRESHOLDS["MEDIUM"])

    def build_context(self, session: Optional[str] = None) -> Dict[str, Any]:
        """Construit le contexte complet pour un cycle de scan."""
        if session is None:
            session = self.get_sub_session()
        return {
            "session": session,
            "weights": self.get_weights(session),
            "sensitivities": {
                name: self.get_sensitivity(name, session)
                for name in ("flow", "catalyst", "smart_money", "sentiment")
            },
        }


# ================================================================
# RADAR A: FLOW (Quantitatif — Volume, Prix, Derivees)
# ================================================================

class FlowRadar:
    """
    Radar quantitatif — Detecte les mouvements de prix/volume AVANT le spike.

    Sources (toutes locales, pas d'API calls):
    - AccelerationEngine: velocity, acceleration, z-scores
    - TickerStateBuffer: ring buffer, derivative state
    - SmallCapRadar: ACCUMULATING -> BREAKOUT phases

    Performance: <10ms par ticker (buffer reads only)

    Force:  Detecte l'accumulation 5-15 min avant le mouvement
    Faiblesse: Aveugle sans donnees de volume (AH limite)
    """

    NAME = "flow"

    async def scan(self, ticker: str, session_ctx: Dict) -> RadarResult:
        t0 = time.monotonic()
        signals: List[str] = []
        details: Dict[str, Any] = {}
        score = 0.0
        confidence = 0.0
        state = "DORMANT"
        sensitivity = session_ctx.get("sensitivities", {}).get(self.NAME, {})
        min_score = sensitivity.get("min_score", 0.20)

        # --- 1. Acceleration Engine (40% du score flow) ---
        accel_score_val = 0.0
        accel_state = "DORMANT"
        try:
            from src.engines.acceleration_engine import get_acceleration_engine
            accel_engine = get_acceleration_engine()
            accel_result = accel_engine.score(ticker)
            if accel_result and accel_result.acceleration_score > 0:
                accel_score_val = accel_result.acceleration_score
                accel_state = accel_result.state or "DORMANT"
                accel_confidence = getattr(accel_result, 'confidence', 0.5)

                score += accel_score_val * 0.40
                confidence += accel_confidence * 0.40

                details["acceleration"] = {
                    "score": round(accel_score_val, 4),
                    "state": accel_state,
                    "volume_zscore": round(getattr(accel_result, 'volume_zscore', 0), 2),
                    "accumulation": round(getattr(accel_result, 'accumulation_score', 0), 2),
                    "breakout_readiness": round(getattr(accel_result, 'breakout_readiness', 0), 2),
                }

                if accel_state == "ACCUMULATING":
                    signals.append(f"ACCUMULATING(score={accel_score_val:.2f})")
                elif accel_state == "LAUNCHING":
                    signals.append(f"LAUNCHING(readiness={getattr(accel_result, 'breakout_readiness', 0):.2f})")
                elif accel_state == "BREAKOUT":
                    signals.append(f"BREAKOUT(z={getattr(accel_result, 'volume_zscore', 0):.1f})")
        except Exception as e:
            logger.debug(f"FlowRadar accel error {ticker}: {e}")

        # --- 2. Ticker State Buffer — Derivees brutes (30% du score flow) ---
        try:
            from src.engines.ticker_state_buffer import get_ticker_state_buffer
            buffer = get_ticker_state_buffer()
            ds = buffer.get_derivative_state(ticker)
            if ds and getattr(ds, 'samples', 0) >= 3:
                vol_z = getattr(ds, 'volume_zscore', 0)
                price_vel = getattr(ds, 'price_velocity_pct', 0)
                spread_tight = getattr(ds, 'spread_tightening', False)
                accum = getattr(ds, 'accumulation_score', 0)

                # Volume z-score normalise (max = 4.0)
                vol_component = min(1.0, max(0, vol_z) / 4.0)
                score += vol_component * 0.20
                confidence += min(1.0, getattr(ds, 'confidence', 0.5)) * 0.20

                # Accumulation from derivatives
                score += min(1.0, accum) * 0.10

                details["derivatives"] = {
                    "volume_zscore": round(vol_z, 2),
                    "price_velocity_pct": round(price_vel, 4),
                    "spread_tightening": spread_tight,
                    "accumulation": round(accum, 2),
                    "samples": ds.samples,
                }

                if spread_tight:
                    signals.append("SPREAD_TIGHTENING")
                    score += 0.03

                if vol_z > 2.5:
                    signals.append(f"VOLUME_SURGE(z={vol_z:.1f})")
                elif vol_z > 1.5:
                    signals.append(f"VOLUME_RISING(z={vol_z:.1f})")

                if price_vel > 0.005:
                    signals.append(f"PRICE_MOVING(vel={price_vel:.3f}%/min)")
        except Exception as e:
            logger.debug(f"FlowRadar buffer error {ticker}: {e}")

        # --- 3. SmallCap Radar boost (20% du score flow) ---
        try:
            from src.engines.smallcap_radar import get_smallcap_radar
            radar = get_smallcap_radar()
            radar_boost = radar.get_monster_boost(ticker)
            if radar_boost and radar_boost > 0:
                score += min(0.20, radar_boost)
                confidence += 0.15
                details["smallcap_radar_boost"] = round(radar_boost, 4)
                signals.append(f"RADAR_BOOST(+{radar_boost:.2f})")
        except Exception as e:
            logger.debug(f"FlowRadar radar error {ticker}: {e}")

        # --- 4. Feature Engine — Volume ratio rapide (10% du score flow) ---
        try:
            from src.feature_engine import volume_spike as get_vol_spike, fetch_candles
            # Utilise le cache existant (TTL 30s dans feature_engine)
            df = fetch_candles(ticker, resolution="1", lookback=30)
            if df is not None and len(df) >= 5:
                vol_raw = get_vol_spike(df)
                vol_norm = min(1.0, vol_raw / 5.0)
                score += vol_norm * 0.10
                details["volume_ratio"] = round(vol_raw, 2)
                if vol_raw > 3.0:
                    signals.append(f"VOL_RATIO({vol_raw:.1f}x)")
        except Exception as e:
            logger.debug(f"FlowRadar feature error {ticker}: {e}")

        # --- Normalisation et classification ---
        score = min(1.0, max(0.0, score))
        confidence = min(1.0, max(0.0, confidence))

        # Classification de l'etat du radar flow
        if accel_state in ("BREAKOUT",):
            state = "BREAKOUT"
        elif accel_state in ("LAUNCHING",):
            state = "LAUNCHING"
        elif accel_state in ("ACCUMULATING",) or score >= 0.30:
            state = "ACCUMULATING"
        elif score >= 0.15:
            state = "WATCHING"
        else:
            state = "DORMANT"

        # Priorite
        if state == "BREAKOUT":
            priority = RadarPriority.CRITICAL
        elif state == "LAUNCHING":
            priority = RadarPriority.HIGH
        elif state == "ACCUMULATING":
            priority = RadarPriority.MEDIUM
        elif state == "WATCHING":
            priority = RadarPriority.LOW
        else:
            priority = RadarPriority.NONE

        # Appliquer seuil de sensibilite
        if score < min_score:
            state = "DORMANT"
            priority = RadarPriority.NONE
            signals = []

        scan_ms = (time.monotonic() - t0) * 1000
        return RadarResult(
            radar_name=self.NAME, ticker=ticker,
            score=score, confidence=confidence,
            state=state, priority=priority,
            signals=signals, details=details,
            scan_time_ms=scan_ms,
        )


# ================================================================
# RADAR B: CATALYST (Fondamental — News, Events, Filings)
# ================================================================

class CatalystRadar:
    """
    Radar fondamental — Detecte les catalysts AVANT le mouvement de prix.

    Sources (caches, pas d'API calls frais):
    - EventHub: evenements caches (Finnhub news + company events)
    - CatalystScorerV3: 5 tiers, 18 types, decay temporel
    - AnticipationEngine: signaux WATCH_EARLY existants
    - IBKR News Trigger: alertes news recentes

    Force:  Peut detecter des heures/jours avant le mouvement
    Faiblesse: Latence API, depend des sources externes
    """

    NAME = "catalyst"

    async def scan(self, ticker: str, session_ctx: Dict) -> RadarResult:
        t0 = time.monotonic()
        signals: List[str] = []
        details: Dict[str, Any] = {}
        score = 0.0
        confidence = 0.0
        sensitivity = session_ctx.get("sensitivities", {}).get(self.NAME, {})
        min_score = sensitivity.get("min_score", 0.20)

        # --- 1. Event Hub — Evenements caches (40% du score catalyst) ---
        try:
            from src.event_engine.event_hub import get_events_by_ticker
            events = get_events_by_ticker(ticker)
            if events:
                # Prendre l'evenement le plus impactant
                best_event = None
                best_impact = 0.0
                for evt in events:
                    impact = evt.get("impact", 0) if isinstance(evt, dict) else getattr(evt, "impact", 0)
                    if impact > best_impact:
                        best_impact = impact
                        best_event = evt

                if best_event and best_impact > 0:
                    event_score = min(1.0, best_impact)
                    score += event_score * 0.40
                    confidence += 0.35

                    evt_type = (best_event.get("type", "UNKNOWN") if isinstance(best_event, dict)
                                else getattr(best_event, "type", "UNKNOWN"))
                    headline = (best_event.get("headline", "") if isinstance(best_event, dict)
                                else getattr(best_event, "headline", ""))

                    details["event_hub"] = {
                        "best_type": evt_type,
                        "best_impact": round(best_impact, 3),
                        "event_count": len(events),
                        "headline": headline[:100],
                    }
                    signals.append(f"EVENT({evt_type}:{best_impact:.2f})")
        except Exception as e:
            logger.debug(f"CatalystRadar event_hub error {ticker}: {e}")

        # --- 2. Catalyst Score V3 (30% du score catalyst) ---
        try:
            from src.catalyst_score_v3 import CatalystScorerV3, Catalyst
            scorer = CatalystScorerV3()

            # Recuperer les catalysts existants en cache dans le scorer
            cat_data = scorer.get_ticker_catalysts(ticker) if hasattr(scorer, 'get_ticker_catalysts') else []
            if cat_data:
                cat_result = scorer.score_catalysts(ticker, cat_data)
                if cat_result:
                    cat_score = cat_result.final_score if hasattr(cat_result, 'final_score') else 0.0
                    score += min(1.0, cat_score) * 0.30
                    confidence += 0.25
                    details["catalyst_v3"] = {
                        "score": round(cat_score, 3),
                        "tier": getattr(cat_result, 'tier', 'UNKNOWN'),
                        "count": len(cat_data),
                    }
                    if cat_score > 0.5:
                        signals.append(f"CATALYST_V3(score={cat_score:.2f})")
        except Exception as e:
            logger.debug(f"CatalystRadar catalyst_v3 error {ticker}: {e}")

        # --- 3. Anticipation Engine — Signaux existants (20% du score catalyst) ---
        try:
            from src.anticipation_engine import get_anticipation_engine
            antic_engine = get_anticipation_engine()
            if antic_engine:
                watch_signals = antic_engine.watch_early_signals if hasattr(antic_engine, 'watch_early_signals') else {}
                if ticker in watch_signals:
                    antic_signal = watch_signals[ticker]
                    antic_score = getattr(antic_signal, 'combined_score', 0.0)
                    antic_level = getattr(antic_signal, 'signal_level', None)

                    score += min(1.0, antic_score) * 0.20
                    confidence += 0.20

                    details["anticipation"] = {
                        "score": round(antic_score, 3),
                        "level": str(antic_level),
                        "catalyst_type": getattr(antic_signal, 'catalyst_type', 'UNKNOWN'),
                    }
                    signals.append(f"ANTICIPATION({antic_level}:{antic_score:.2f})")
        except Exception as e:
            logger.debug(f"CatalystRadar anticipation error {ticker}: {e}")

        # --- 4. FDA Calendar check (10% du score catalyst) ---
        try:
            from src.fda_calendar import get_fda_events, get_fda_boost
            fda_events = get_fda_events(ticker) if callable(get_fda_events) else []
            if fda_events:
                fda_boost = get_fda_boost(ticker) if callable(get_fda_boost) else 0
                if fda_boost and fda_boost > 0:
                    score += min(1.0, fda_boost) * 0.10
                    confidence += 0.10
                    signals.append(f"FDA_CATALYST(boost={fda_boost:.2f})")
                    details["fda"] = {"boost": round(fda_boost, 3), "events": len(fda_events)}
        except Exception as e:
            logger.debug(f"CatalystRadar fda error {ticker}: {e}")

        # --- Normalisation et classification ---
        score = min(1.0, max(0.0, score))
        confidence = min(1.0, max(0.0, confidence))

        if score >= 0.70:
            state = "BREAKOUT"
        elif score >= 0.50:
            state = "READY"
        elif score >= 0.30:
            state = "ACCUMULATING"
        elif score >= 0.15:
            state = "WATCHING"
        else:
            state = "DORMANT"

        if state == "BREAKOUT":
            priority = RadarPriority.CRITICAL
        elif state == "READY":
            priority = RadarPriority.HIGH
        elif state == "ACCUMULATING":
            priority = RadarPriority.MEDIUM
        elif state == "WATCHING":
            priority = RadarPriority.LOW
        else:
            priority = RadarPriority.NONE

        if score < min_score:
            state = "DORMANT"
            priority = RadarPriority.NONE
            signals = []

        scan_ms = (time.monotonic() - t0) * 1000
        return RadarResult(
            radar_name=self.NAME, ticker=ticker,
            score=score, confidence=confidence,
            state=state, priority=priority,
            signals=signals, details=details,
            scan_time_ms=scan_ms,
        )


# ================================================================
# RADAR C: SMART MONEY (Options + Insiders)
# ================================================================

class SmartMoneyRadar:
    """
    Radar argent intelligent — Detecte l'activite des institutionnels.

    Sources:
    - Options Flow IBKR (OPRA): call/put ratio, volume vs OI, sweeps
    - Insider Boost (SEC Form 4): achats directeurs
    - Extended Hours: block trades (volume > 10x moyenne)

    Force:  Signaux de haute conviction (skin in the game)
    Faiblesse: Options fermees en AH/PM, insider data delayed
    """

    NAME = "smart_money"

    async def scan(self, ticker: str, session_ctx: Dict) -> RadarResult:
        t0 = time.monotonic()
        signals: List[str] = []
        details: Dict[str, Any] = {}
        score = 0.0
        confidence = 0.0
        sensitivity = session_ctx.get("sensitivities", {}).get(self.NAME, {})
        min_score = sensitivity.get("min_score", 0.35)

        # --- 1. Options Flow IBKR (60% du score smart money) ---
        try:
            from src.options_flow_ibkr import get_options_flow_score
            opt_score, opt_details = get_options_flow_score(ticker)
            if opt_score and opt_score > 0:
                score += min(1.0, opt_score) * 0.60
                confidence += 0.40

                details["options_flow"] = {
                    "score": round(opt_score, 3),
                    "call_ratio": opt_details.get("call_ratio", 0),
                    "volume_oi_ratio": opt_details.get("volume_oi_ratio", 0),
                    "signal_types": opt_details.get("signal_types", []),
                }

                if opt_score > 0.7:
                    signals.append(f"OPTIONS_STRONG(score={opt_score:.2f})")
                elif opt_score > 0.4:
                    signals.append(f"OPTIONS_UNUSUAL(score={opt_score:.2f})")

                # Signaux specifiques
                for sig_type in opt_details.get("signal_types", []):
                    if "SWEEP" in str(sig_type).upper():
                        signals.append("CALL_SWEEP")
                        score += 0.05
                    elif "VOLUME_SPIKE" in str(sig_type).upper():
                        signals.append("OPTIONS_VOL_SPIKE")
        except Exception as e:
            logger.debug(f"SmartMoneyRadar options error {ticker}: {e}")

        # --- 2. Insider Boost — SEC Form 4 (30% du score smart money) ---
        try:
            from src.boosters.insider_boost import get_insider_engine
            insider_eng = get_insider_engine()
            if insider_eng:
                insider_result = insider_eng.check_ticker(ticker) if hasattr(insider_eng, 'check_ticker') else None
                if insider_result is None and hasattr(insider_eng, 'get_boost'):
                    insider_result = insider_eng.get_boost(ticker)

                if insider_result:
                    ins_score = getattr(insider_result, 'boost_score', 0)
                    if not ins_score:
                        ins_score = getattr(insider_result, 'score', 0)

                    if ins_score and ins_score > 0:
                        score += min(1.0, ins_score) * 0.30
                        confidence += 0.25
                        details["insider"] = {
                            "score": round(ins_score, 3),
                            "signal": str(getattr(insider_result, 'signal', 'NONE')),
                        }
                        signals.append(f"INSIDER_BUY(score={ins_score:.2f})")
        except Exception as e:
            logger.debug(f"SmartMoneyRadar insider error {ticker}: {e}")

        # --- 3. Extended Hours — Block trades (10% du score smart money) ---
        try:
            from src.extended_hours_quotes import get_extended_quote
            ext_quote = get_extended_quote(ticker)
            if ext_quote:
                gap_pct = abs(getattr(ext_quote, 'gap_pct', 0) or 0)
                ext_volume = getattr(ext_quote, 'volume', 0) or 0

                if gap_pct > 0.05 and ext_volume > 100_000:
                    block_score = min(1.0, gap_pct * 5)
                    score += block_score * 0.10
                    confidence += 0.10
                    details["extended_hours"] = {
                        "gap_pct": round(gap_pct, 4),
                        "volume": ext_volume,
                    }
                    signals.append(f"GAP({gap_pct*100:.1f}%)+VOLUME")
        except Exception as e:
            logger.debug(f"SmartMoneyRadar extended error {ticker}: {e}")

        # --- Normalisation et classification ---
        score = min(1.0, max(0.0, score))
        confidence = min(1.0, max(0.0, confidence))

        if score >= 0.65:
            state = "LAUNCHING"
        elif score >= 0.40:
            state = "ACCUMULATING"
        elif score >= 0.20:
            state = "WATCHING"
        else:
            state = "DORMANT"

        if state == "LAUNCHING":
            priority = RadarPriority.HIGH
        elif state == "ACCUMULATING":
            priority = RadarPriority.MEDIUM
        elif state == "WATCHING":
            priority = RadarPriority.LOW
        else:
            priority = RadarPriority.NONE

        if score < min_score:
            state = "DORMANT"
            priority = RadarPriority.NONE
            signals = []

        scan_ms = (time.monotonic() - t0) * 1000
        return RadarResult(
            radar_name=self.NAME, ticker=ticker,
            score=score, confidence=confidence,
            state=state, priority=priority,
            signals=signals, details=details,
            scan_time_ms=scan_ms,
        )


# ================================================================
# RADAR D: SENTIMENT (Social + NLP + Repeat Patterns)
# ================================================================

class SentimentRadar:
    """
    Radar sentiment — Capte le buzz et les patterns de repeat gainers.

    Sources:
    - Social Buzz (Reddit WSB + StockTwits): mentions et spike detection
    - NLP Enrichi (Grok): sentiment analysis sur les news
    - Repeat Gainer Memory: historique des runners
    - Extended Hours: gap + momentum AH/PM

    Force:  Capte le buzz retail avant le mouvement
    Faiblesse: Social = bruit, manipulation possible
    """

    NAME = "sentiment"

    async def scan(self, ticker: str, session_ctx: Dict) -> RadarResult:
        t0 = time.monotonic()
        signals: List[str] = []
        details: Dict[str, Any] = {}
        score = 0.0
        confidence = 0.0
        sensitivity = session_ctx.get("sensitivities", {}).get(self.NAME, {})
        min_score = sensitivity.get("min_score", 0.20)

        # --- 1. Social Buzz — Reddit + StockTwits (35% du score sentiment) ---
        try:
            from src.social_buzz import get_buzz_signal, detect_buzz_spike
            buzz_signal = get_buzz_signal(ticker)
            if buzz_signal:
                buzz_score = buzz_signal.get("combined_score", 0) if isinstance(buzz_signal, dict) else getattr(buzz_signal, 'combined_score', 0)
                if buzz_score and buzz_score > 0:
                    score += min(1.0, buzz_score) * 0.25
                    confidence += 0.20

                    details["social_buzz"] = {
                        "score": round(buzz_score, 3),
                        "sources": buzz_signal.get("sources", []) if isinstance(buzz_signal, dict) else [],
                    }

                    if buzz_score > 0.6:
                        signals.append(f"BUZZ_HIGH(score={buzz_score:.2f})")
                    elif buzz_score > 0.3:
                        signals.append(f"BUZZ_RISING(score={buzz_score:.2f})")

            # Detection spike independante
            spike_result = detect_buzz_spike(ticker)
            if spike_result:
                is_spike = spike_result.get("is_spike", False) if isinstance(spike_result, dict) else getattr(spike_result, 'is_spike', False)
                if is_spike:
                    spike_mult = spike_result.get("spike_multiplier", 3.0) if isinstance(spike_result, dict) else 3.0
                    score += 0.10
                    confidence += 0.10
                    signals.append(f"BUZZ_SPIKE({spike_mult:.1f}x)")
                    details["buzz_spike"] = {"multiplier": spike_mult}
        except Exception as e:
            logger.debug(f"SentimentRadar buzz error {ticker}: {e}")

        # --- 2. NLP Sentiment — Grok analysis (25% du score sentiment) ---
        try:
            from src.nlp_enrichi import get_nlp_sentiment_boost
            nlp_boost = get_nlp_sentiment_boost(ticker)
            if nlp_boost and nlp_boost != 0:
                # nlp_boost est un multiplicateur (0.7 = bearish, 1.4 = bullish)
                # Convertir en score 0-1: (boost - 0.7) / (1.4 - 0.7) = normalise
                nlp_score = min(1.0, max(0.0, (nlp_boost - 0.7) / 0.7))
                score += nlp_score * 0.25
                confidence += 0.15

                details["nlp_sentiment"] = {
                    "boost": round(nlp_boost, 3),
                    "score": round(nlp_score, 3),
                    "direction": "BULLISH" if nlp_boost > 1.0 else "BEARISH" if nlp_boost < 1.0 else "NEUTRAL",
                }

                if nlp_boost > 1.2:
                    signals.append(f"NLP_BULLISH(boost={nlp_boost:.2f})")
                elif nlp_boost < 0.85:
                    signals.append(f"NLP_BEARISH(boost={nlp_boost:.2f})")
        except Exception as e:
            logger.debug(f"SentimentRadar nlp error {ticker}: {e}")

        # --- 3. Repeat Gainer Memory (25% du score sentiment) ---
        try:
            from src.repeat_gainer_memory import calculate_repeat_score, get_repeat_score_boost
            repeat_score_data = calculate_repeat_score(ticker)
            if repeat_score_data:
                rep_score = getattr(repeat_score_data, 'score', 0)
                if rep_score and rep_score > 0:
                    score += min(1.0, rep_score) * 0.25
                    confidence += 0.20

                    details["repeat_gainer"] = {
                        "score": round(rep_score, 3),
                        "spike_count": getattr(repeat_score_data, 'spike_count', 0),
                        "is_repeat_runner": getattr(repeat_score_data, 'is_repeat_runner', False),
                    }

                    if getattr(repeat_score_data, 'is_repeat_runner', False):
                        signals.append(f"REPEAT_RUNNER(score={rep_score:.2f})")
                    elif rep_score > 0.3:
                        signals.append(f"REPEAT_HISTORY(score={rep_score:.2f})")
        except Exception as e:
            logger.debug(f"SentimentRadar repeat error {ticker}: {e}")

        # --- 4. Extended Hours Boost (15% du score sentiment) ---
        try:
            from src.extended_hours_quotes import get_extended_hours_boost
            ext_boost, ext_details = get_extended_hours_boost(ticker)
            if ext_boost and ext_boost > 0:
                score += min(1.0, ext_boost / 0.22) * 0.15  # Normalise (max boost = 0.22)
                confidence += 0.10

                details["extended_hours"] = {
                    "boost": round(ext_boost, 3),
                    "gap_pct": ext_details.get("gap_pct", 0),
                    "direction": ext_details.get("direction", "UNKNOWN"),
                }

                if ext_boost > 0.15:
                    signals.append(f"EXT_HOURS_STRONG(boost={ext_boost:.2f})")
                elif ext_boost > 0.05:
                    signals.append(f"EXT_HOURS_GAP(boost={ext_boost:.2f})")
        except Exception as e:
            logger.debug(f"SentimentRadar extended error {ticker}: {e}")

        # --- Normalisation et classification ---
        score = min(1.0, max(0.0, score))
        confidence = min(1.0, max(0.0, confidence))

        if score >= 0.60:
            state = "LAUNCHING"
        elif score >= 0.35:
            state = "ACCUMULATING"
        elif score >= 0.15:
            state = "WATCHING"
        else:
            state = "DORMANT"

        if state == "LAUNCHING":
            priority = RadarPriority.HIGH
        elif state == "ACCUMULATING":
            priority = RadarPriority.MEDIUM
        elif state == "WATCHING":
            priority = RadarPriority.LOW
        else:
            priority = RadarPriority.NONE

        if score < min_score:
            state = "DORMANT"
            priority = RadarPriority.NONE
            signals = []

        scan_ms = (time.monotonic() - t0) * 1000
        return RadarResult(
            radar_name=self.NAME, ticker=ticker,
            score=score, confidence=confidence,
            state=state, priority=priority,
            signals=signals, details=details,
            scan_time_ms=scan_ms,
        )


# ================================================================
# CONFLUENCE MATRIX
# ================================================================

class ConfluenceMatrix:
    """
    Matrice de confluence — Combine les 4 radars en un signal final.

    Principes:
    1. Score pondere par session: final = sum(radar_score * session_weight)
    2. Bonus de confluence quand plusieurs radars convergent
    3. Matrice 2D (flow x catalyst) pour signal de base, modifie par smart_money + sentiment
    4. Un seul radar fort = WATCH max (pas BUY sans confirmation)
    5. Deux radars forts = BUY potentiel
    6. Trois+ radars forts = BUY_STRONG potentiel

    Coherent avec la philosophie: detection jamais bloquee, additif pas multiplicatif.
    """

    # Matrice de base: (flow_level, catalyst_level) -> signal de depart
    # Niveaux: HIGH (score >= 0.60), MEDIUM (0.30-0.60), LOW (< 0.30)
    BASE_SIGNAL_MATRIX: Dict[Tuple[str, str], str] = {
        ("HIGH", "HIGH"):     "BUY_STRONG",
        ("HIGH", "MEDIUM"):   "BUY",
        ("HIGH", "LOW"):      "WATCH",
        ("MEDIUM", "HIGH"):   "BUY",
        ("MEDIUM", "MEDIUM"): "WATCH",
        ("MEDIUM", "LOW"):    "EARLY_SIGNAL",
        ("LOW", "HIGH"):      "WATCH",
        ("LOW", "MEDIUM"):    "EARLY_SIGNAL",
        ("LOW", "LOW"):       "NO_SIGNAL",
    }

    # Bonus de confluence par niveau d'accord
    CONFLUENCE_BONUS: Dict[str, float] = {
        "UNANIMOUS": 0.15,    # 4/4 radars actifs
        "STRONG": 0.10,       # 3/4 radars actifs
        "MODERATE": 0.05,     # 2/4 radars actifs
        "WEAK": 0.00,         # 1/4 radar actif
        "NONE": 0.00,         # Aucun
    }

    # Upgrades possibles par smart money / sentiment
    SIGNAL_ORDER = ["NO_SIGNAL", "EARLY_SIGNAL", "WATCH", "BUY", "BUY_STRONG"]

    def _upgrade_signal(self, signal: str, levels: int = 1) -> str:
        """Monte un signal de N niveaux."""
        idx = self.SIGNAL_ORDER.index(signal)
        new_idx = min(len(self.SIGNAL_ORDER) - 1, idx + levels)
        return self.SIGNAL_ORDER[new_idx]

    def _downgrade_signal(self, signal: str, levels: int = 1) -> str:
        """Descend un signal de N niveaux (mais jamais en dessous de EARLY_SIGNAL si actif)."""
        idx = self.SIGNAL_ORDER.index(signal)
        new_idx = max(0, idx - levels)
        return self.SIGNAL_ORDER[new_idx]

    def evaluate(
        self,
        radar_results: Dict[str, RadarResult],
        session_weights: Dict[str, float],
        session: str,
    ) -> ConfluenceSignal:
        """
        Evalue la confluence de tous les radars et produit un signal final.

        Args:
            radar_results: dict {radar_name: RadarResult}
            session_weights: dict {radar_name: weight} (somme = 1.0)
            session: sous-session actuelle

        Returns:
            ConfluenceSignal avec score final, signal type, agreement level
        """
        flow = radar_results.get("flow")
        catalyst = radar_results.get("catalyst")
        smart_money = radar_results.get("smart_money")
        sentiment = radar_results.get("sentiment")

        # --- 1. Score pondere par session ---
        weighted_score = 0.0
        weighted_confidence = 0.0
        for name, result in radar_results.items():
            w = session_weights.get(name, 0.25)
            weighted_score += result.score * w
            weighted_confidence += result.confidence * w

        # --- 2. Niveau d'accord ---
        active_radars = [name for name, r in radar_results.items() if r.is_active]
        active_count = len(active_radars)

        if active_count >= 4:
            agreement = AgreementLevel.UNANIMOUS
        elif active_count == 3:
            agreement = AgreementLevel.STRONG
        elif active_count == 2:
            agreement = AgreementLevel.MODERATE
        elif active_count == 1:
            agreement = AgreementLevel.WEAK
        else:
            agreement = AgreementLevel.NONE

        # --- 3. Bonus de confluence (additif) ---
        confluence_bonus = self.CONFLUENCE_BONUS.get(agreement.value, 0.0)
        weighted_score = min(1.0, weighted_score + confluence_bonus)

        # --- 4. Signal de base via matrice 2D (flow x catalyst) ---
        flow_level = flow.level if flow else "LOW"
        cat_level = catalyst.level if catalyst else "LOW"
        base_signal = self.BASE_SIGNAL_MATRIX.get(
            (flow_level, cat_level), "NO_SIGNAL"
        )

        # --- 5. Modificateurs smart money et sentiment ---
        final_signal = base_signal

        # Smart Money HIGH = upgrade +1 (forte conviction)
        if smart_money and smart_money.level == "HIGH":
            final_signal = self._upgrade_signal(final_signal, 1)

        # Sentiment HIGH + au moins un autre radar actif = confirme (pas de downgrade)
        if sentiment and sentiment.level == "HIGH" and active_count >= 2:
            if final_signal in ("WATCH", "EARLY_SIGNAL"):
                final_signal = self._upgrade_signal(final_signal, 1)

        # UNANIME (4/4) avec score > 0.50 = minimum BUY
        if agreement == AgreementLevel.UNANIMOUS and weighted_score > 0.50:
            if self.SIGNAL_ORDER.index(final_signal) < self.SIGNAL_ORDER.index("BUY"):
                final_signal = "BUY"

        # Si le score pondere est tres fort mais le signal est bas → ajuster
        if weighted_score >= 0.75 and final_signal not in ("BUY_STRONG", "BUY"):
            final_signal = "BUY"
        elif weighted_score >= 0.55 and final_signal == "NO_SIGNAL":
            final_signal = "WATCH"

        # --- 6. Radar leader ---
        lead_radar = max(radar_results.items(), key=lambda x: x[1].score)[0] if radar_results else "none"

        # --- 7. Estimation du lead time ---
        estimated_lead = self._estimate_lead_time(radar_results, session)

        return ConfluenceSignal(
            ticker=flow.ticker if flow else (catalyst.ticker if catalyst else "???"),
            final_score=round(weighted_score, 4),
            signal_type=final_signal,
            agreement=agreement,
            lead_radar=lead_radar,
            radar_results=radar_results,
            confluence_bonus=round(confluence_bonus, 4),
            session=session,
            session_weights=session_weights,
            estimated_lead_time_min=estimated_lead,
            confidence=round(weighted_confidence, 4),
        )

    def _estimate_lead_time(self, radar_results: Dict[str, RadarResult], session: str) -> float:
        """
        Estime le temps avant le mouvement en minutes.

        Logique:
        - ACCUMULATING (flow) = 5-15 min
        - READY (catalyst) = 2-8 min (catalyst deja chaud)
        - LAUNCHING = 1-3 min
        - BREAKOUT = 0 min (deja en cours)
        - Si seulement catalyst (pas de flow) = 30-120 min
        """
        flow = radar_results.get("flow")
        catalyst = radar_results.get("catalyst")

        flow_state = flow.state if flow else "DORMANT"
        cat_state = catalyst.state if catalyst else "DORMANT"

        if flow_state == "BREAKOUT":
            return 0.0
        elif flow_state == "LAUNCHING":
            return 2.0
        elif flow_state == "ACCUMULATING":
            if cat_state in ("READY", "BREAKOUT"):
                return 5.0   # Catalyst chaud + accumulation = imminent
            return 10.0       # Accumulation seule
        elif cat_state in ("READY", "BREAKOUT"):
            return 15.0       # Catalyst sans flow encore = attente du flow
        elif cat_state == "ACCUMULATING":
            return 30.0       # Catalyst en construction
        elif flow_state == "WATCHING":
            return 20.0

        return 60.0           # Pas de signal clair


# ================================================================
# MULTI-RADAR ENGINE (Orchestrateur)
# ================================================================

class MultiRadarEngine:
    """
    Moteur Multi-Radar — Orchestrateur principal.

    Lance les 4 radars en parallele (asyncio.gather) pour chaque ticker,
    adapte les poids/sensibilites a la session en cours,
    et produit un ConfluenceSignal via la matrice de confluence.

    Usage:
        engine = get_multi_radar_engine()
        signal = await engine.scan_ticker("AAPL")
        signals = await engine.scan_batch(["AAPL", "TSLA", "NVDA"])
    """

    def __init__(self):
        self.flow_radar = FlowRadar()
        self.catalyst_radar = CatalystRadar()
        self.smart_money_radar = SmartMoneyRadar()
        self.sentiment_radar = SentimentRadar()
        self.session_adapter = SessionAdapter()
        self.confluence_matrix = ConfluenceMatrix()

        # Historique des scans (pour tracking tendance)
        self._scan_history: Dict[str, List[ConfluenceSignal]] = {}
        self._scan_count = 0
        self._total_scan_time_ms = 0.0

        # Callbacks pour signaux actionnables
        self._callbacks: List = []

        # Streaming watchlist — tickers a surveiller en temps reel
        self._streaming_watchlist: Dict[str, Dict[str, Any]] = {}

        logger.info("MultiRadarEngine initialized — 4 radars ready")

    def on_signal(self, callback):
        """Enregistre un callback pour les signaux actionnables."""
        self._callbacks.append(callback)

    async def scan_ticker(self, ticker: str, session: Optional[str] = None) -> ConfluenceSignal:
        """
        Scan un ticker avec les 4 radars en parallele.

        Args:
            ticker: symbole a scanner
            session: force une session (sinon auto-detect)

        Returns:
            ConfluenceSignal avec le resultat confluencie
        """
        t0 = time.monotonic()

        # Contexte de session
        if session is None:
            session = self.session_adapter.get_sub_session()
        ctx = self.session_adapter.build_context(session)

        # Lancer les 4 radars en parallele
        results = await asyncio.gather(
            self.flow_radar.scan(ticker, ctx),
            self.catalyst_radar.scan(ticker, ctx),
            self.smart_money_radar.scan(ticker, ctx),
            self.sentiment_radar.scan(ticker, ctx),
            return_exceptions=True,
        )

        # Collecter les resultats (remplacer les exceptions par des resultats vides)
        radar_map: Dict[str, RadarResult] = {}
        radar_names = ["flow", "catalyst", "smart_money", "sentiment"]

        for name, result in zip(radar_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Radar {name} error for {ticker}: {result}")
                radar_map[name] = RadarResult(
                    radar_name=name, ticker=ticker,
                    score=0.0, confidence=0.0,
                    state="DORMANT", priority=RadarPriority.NONE,
                    signals=[], details={"error": str(result)},
                    scan_time_ms=0.0,
                )
            else:
                radar_map[name] = result

        # Confluence
        weights = ctx["weights"]
        signal = self.confluence_matrix.evaluate(radar_map, weights, session)

        # Stats
        total_ms = (time.monotonic() - t0) * 1000
        self._scan_count += 1
        self._total_scan_time_ms += total_ms

        # Historique (garder les 10 derniers par ticker)
        if ticker not in self._scan_history:
            self._scan_history[ticker] = []
        self._scan_history[ticker].append(signal)
        if len(self._scan_history[ticker]) > 10:
            self._scan_history[ticker] = self._scan_history[ticker][-10:]

        # Log si signal actif
        if signal.signal_type not in ("NO_SIGNAL",):
            logger.info(
                f"[{session}] {signal.signal_type} {ticker} | "
                f"Score={signal.final_score:.2f} Agreement={signal.agreement.value} "
                f"Lead={signal.lead_radar} "
                f"Radars: F={radar_map['flow'].score:.2f} C={radar_map['catalyst'].score:.2f} "
                f"SM={radar_map['smart_money'].score:.2f} S={radar_map['sentiment'].score:.2f} "
                f"({total_ms:.0f}ms)"
            )

        # Streaming feedback loop — alimenter IBKR streaming avec les
        # tickers detectes par les radars non-flow (Catalyst, SmartMoney, Sentiment)
        try:
            self._feed_streaming_watchlist(signal)
        except Exception as e:
            logger.debug(f"Streaming feedback error for {ticker}: {e}")

        # Callbacks
        if signal.is_actionable():
            for cb in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(signal)
                    else:
                        cb(signal)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        return signal

    async def scan_batch(self, tickers: List[str], session: Optional[str] = None) -> List[ConfluenceSignal]:
        """
        Scan batch de tickers.

        Les tickers sont traites sequentiellement pour eviter
        de surcharger les APIs, mais les 4 radars de chaque ticker
        tournent en parallele.

        Args:
            tickers: liste de symboles
            session: force une session (sinon auto-detect)

        Returns:
            Liste de ConfluenceSignal, triee par score descendant
        """
        if session is None:
            session = self.session_adapter.get_sub_session()

        results = []
        for ticker in tickers:
            try:
                signal = await self.scan_ticker(ticker, session)
                results.append(signal)
            except Exception as e:
                logger.error(f"scan_batch error {ticker}: {e}")

        # Trier par score descendant
        results.sort(key=lambda s: s.final_score, reverse=True)
        return results

    def get_ticker_trend(self, ticker: str) -> Optional[str]:
        """
        Analyse la tendance d'un ticker basee sur l'historique des scans.

        Returns:
            IMPROVING, STABLE, DEGRADING, ou None si pas assez d'historique
        """
        history = self._scan_history.get(ticker, [])
        if len(history) < 3:
            return None

        recent_scores = [s.final_score for s in history[-5:]]
        if len(recent_scores) < 3:
            return None

        # Regression lineaire simple
        n = len(recent_scores)
        x_mean = (n - 1) / 2
        y_mean = sum(recent_scores) / n
        num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(recent_scores))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0

        if slope > 0.02:
            return "IMPROVING"
        elif slope < -0.02:
            return "DEGRADING"
        return "STABLE"

    def get_session_status(self) -> Dict:
        """Status actuel du moteur multi-radar."""
        session = self.session_adapter.get_sub_session()
        weights = self.session_adapter.get_weights(session)
        avg_ms = (self._total_scan_time_ms / self._scan_count) if self._scan_count > 0 else 0

        return {
            "session": session,
            "weights": weights,
            "scan_count": self._scan_count,
            "avg_scan_time_ms": round(avg_ms, 2),
            "tracked_tickers": len(self._scan_history),
            "radars": {
                "flow": {"name": "Flow (Quantitatif)", "weight": weights.get("flow", 0)},
                "catalyst": {"name": "Catalyst (Fondamental)", "weight": weights.get("catalyst", 0)},
                "smart_money": {"name": "Smart Money (Options+Insiders)", "weight": weights.get("smart_money", 0)},
                "sentiment": {"name": "Sentiment (Social+NLP)", "weight": weights.get("sentiment", 0)},
            },
        }

    def get_top_signals(self, min_score: float = 0.40) -> List[ConfluenceSignal]:
        """Retourne les derniers signaux au-dessus du seuil."""
        top = []
        for ticker, history in self._scan_history.items():
            if history:
                latest = history[-1]
                if latest.final_score >= min_score:
                    top.append(latest)
        top.sort(key=lambda s: s.final_score, reverse=True)
        return top

    # ================================================================
    # STREAMING FEEDBACK LOOP
    # ================================================================
    #
    # Boucle vertueuse:
    #   Radars detectent → ticker ajoute au streaming IBKR
    #   → TickerStateBuffer se remplit → FlowRadar detecte mieux
    #   → Score monte → priorite HOT → scan plus frequent
    #
    # C'est le chaînon manquant: les 3 radars non-flow (Catalyst,
    # SmartMoney, Sentiment) ALIMENTENT le streaming pour que
    # FlowRadar puisse ensuite confirmer avec les donnees temps reel.

    def _feed_streaming_watchlist(self, signal: ConfluenceSignal) -> None:
        """
        Alimente la watchlist streaming a partir des resultats des radars.

        Logique:
        - Si un radar non-flow detecte un candidat → subscribe au streaming
        - Priorite streaming basee sur le signal_type et l'agreement
        - Le streaming remplit le TickerStateBuffer
        - Au prochain cycle, FlowRadar pourra confirmer ou infirmer

        C'est la boucle feedback qui connecte les 4 radars au streaming.
        """
        ticker = signal.ticker

        # Seuil minimum pour justifier un subscribe streaming
        if signal.final_score < 0.15 or signal.signal_type == "NO_SIGNAL":
            return

        # Determiner la priorite streaming
        if signal.signal_type in ("BUY_STRONG",) or signal.agreement in (AgreementLevel.UNANIMOUS,):
            stream_priority = "HOT"
        elif signal.signal_type in ("BUY",) or signal.agreement in (AgreementLevel.STRONG,):
            stream_priority = "HOT"
        elif signal.signal_type in ("WATCH",):
            stream_priority = "WARM"
        else:
            stream_priority = "NORMAL"

        # Identifier quels radars non-flow ont detecte ce ticker
        source_radars = []
        for name in ("catalyst", "smart_money", "sentiment"):
            r = signal.radar_results.get(name)
            if r and r.is_active:
                source_radars.append(name)

        # Si le FlowRadar est deja actif, le streaming est deja en place
        flow = signal.radar_results.get("flow")
        flow_already_active = flow and flow.is_active

        # Mettre a jour la watchlist interne
        self._streaming_watchlist[ticker] = {
            "priority": stream_priority,
            "score": signal.final_score,
            "signal_type": signal.signal_type,
            "source_radars": source_radars,
            "flow_active": flow_already_active,
            "needs_streaming": not flow_already_active and len(source_radars) > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Envoyer au streaming IBKR + HotTickerQueue
        if not flow_already_active and source_radars:
            self._push_to_ibkr_streaming(ticker, stream_priority)
            self._push_to_hot_queue(ticker, stream_priority, source_radars)

    def _push_to_ibkr_streaming(self, ticker: str, priority: str) -> None:
        """
        Subscribe un ticker au streaming IBKR pour surveillance temps reel.

        Le streaming va:
        1. Ouvrir un abonnement L1 persistant (pas de poll-and-cancel)
        2. Recevoir les ticks en ~10ms
        3. Alimenter automatiquement le TickerStateBuffer
        4. FlowRadar pourra lire les derivees au prochain scan
        """
        try:
            from src.ibkr_streaming import get_ibkr_streaming
            streaming = get_ibkr_streaming()

            if not streaming.is_subscribed(ticker):
                count = streaming.subscribe([ticker], priority=priority)
                if count > 0:
                    logger.info(
                        f"STREAMING SUBSCRIBE: {ticker} (priority={priority}) "
                        f"— radar-driven feedback loop"
                    )
                else:
                    logger.debug(f"Streaming subscribe failed for {ticker}")
        except Exception as e:
            logger.debug(f"IBKR streaming not available for {ticker}: {e}")

    def _push_to_hot_queue(self, ticker: str, priority: str, source_radars: List[str]) -> None:
        """
        Ajoute un ticker a la HotTickerQueue pour scan prioritaire.

        La HotTickerQueue controle la frequence de scan:
        - HOT: scan toutes les 90s
        - WARM: scan toutes les 5 min
        - NORMAL: scan toutes les 10 min
        """
        try:
            from src.schedulers.hot_ticker_queue import get_hot_queue

            # Mapper la priorite string vers l'enum
            from src.schedulers.hot_ticker_queue import TickerPriority, TriggerReason

            prio_map = {
                "HOT": TickerPriority.HOT,
                "WARM": TickerPriority.WARM,
                "NORMAL": TickerPriority.NORMAL,
            }
            queue_priority = prio_map.get(priority, TickerPriority.NORMAL)

            # Determiner la raison du trigger
            reason = TriggerReason.GLOBAL_CATALYST
            if "smart_money" in source_radars:
                reason = TriggerReason.PRE_SPIKE_RADAR
            elif "sentiment" in source_radars:
                reason = TriggerReason.SOCIAL_BUZZ

            queue = get_hot_queue()
            queue.push(
                ticker=ticker,
                priority=queue_priority,
                reason=reason,
                metadata={"source": "multi_radar", "radars": source_radars},
            )
            logger.debug(f"HOT_QUEUE: {ticker} → {priority} (from {source_radars})")
        except Exception as e:
            logger.debug(f"HotTickerQueue push failed for {ticker}: {e}")

    def get_streaming_watchlist(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne la watchlist de tickers a surveiller en streaming.

        Utile pour:
        - Savoir quels tickers sont surveilles et pourquoi
        - Debug: voir quels radars ont declenche le streaming
        - Dashboard: afficher les tickers sous surveillance active

        Returns:
            Dict {ticker: {priority, score, source_radars, needs_streaming, ...}}
        """
        return dict(self._streaming_watchlist)

    def get_tickers_needing_streaming(self) -> List[Tuple[str, str]]:
        """
        Retourne les tickers qui ont besoin de streaming IBKR
        mais dont le FlowRadar n'a pas encore de donnees.

        Returns:
            Liste de (ticker, priority) triee par score descendant.
            Ce sont les tickers detectes par Catalyst/SmartMoney/Sentiment
            qui n'ont pas encore de donnees temps reel.
        """
        needing = []
        for ticker, info in self._streaming_watchlist.items():
            if info.get("needs_streaming", False):
                needing.append((ticker, info["priority"], info["score"]))

        # Trier par score descendant
        needing.sort(key=lambda x: x[2], reverse=True)
        return [(t, p) for t, p, _ in needing]


# ================================================================
# SINGLETON
# ================================================================

_engine: Optional[MultiRadarEngine] = None


def get_multi_radar_engine() -> MultiRadarEngine:
    """Retourne l'instance singleton du MultiRadarEngine."""
    global _engine
    if _engine is None:
        _engine = MultiRadarEngine()
    return _engine
