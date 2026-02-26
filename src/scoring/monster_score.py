"""
MONSTER SCORE V4 - Composite Scoring Engine with Acceleration
==============================================================

V4 CHANGES (from V3):
- NEW: Acceleration component (7% weight) from AccelerationEngine V8
  - Replaces static momentum with derivative-based anticipation
  - ACCUMULATING state adds +0.05 to +0.12 boost
  - LAUNCHING state adds +0.08 to +0.15 boost
- CHANGED: Momentum weight reduced from 8% to 4% (velocity replaces it)
- CHANGED: Social buzz weight adjusted from 6% to 3% (lower reliability)
- NEW: Z-score based volume normalization (P9 fix, alongside absolute)
- NEW: Gap quality integration from PM Scanner V8

Calculates a comprehensive score (0-1) for each ticker based on:
- Event impact (earnings, FDA, M&A, etc.)
- Volume spikes (now with z-score normalization)
- Technical patterns
- Pre-market transition
- Acceleration (NEW V4 - replaces pure momentum)
- Momentum (reduced weight)
- Squeeze indicators
- Options flow
- Social buzz (reduced weight)

The score determines signal strength: BUY (0.65+), BUY_STRONG (0.80+)
"""

import json
import os
from typing import Optional

from utils.logger import get_logger
from utils.cache import Cache

from src.event_engine.event_hub import get_events_by_ticker
from src.feature_engine import compute_features
from src.pm_scanner import compute_pm_metrics

# S5-3: CatalystScorerV3 singleton for enhanced event scoring
CATALYST_V3_AVAILABLE = False
try:
    from src.catalyst_score_v3 import (
        get_catalyst_scorer, create_catalyst_from_event, CatalystType
    )
    CATALYST_V3_AVAILABLE = True
except ImportError:
    pass

# Import intelligence modules with graceful fallback
from src.historical_beat_rate import get_earnings_probability
from src.social_buzz import get_buzz_signal, get_total_buzz_score

# V8: Import AccelerationEngine for anticipatory scoring
ACCELERATION_AVAILABLE = False
try:
    from src.engines.acceleration_engine import get_acceleration_engine
    ACCELERATION_AVAILABLE = True
except ImportError:
    pass

# V8: Import SmallCapRadar for radar boost
RADAR_AVAILABLE = False
try:
    from src.engines.smallcap_radar import get_smallcap_radar
    RADAR_AVAILABLE = True
except ImportError:
    pass

# Import config flags
from config import (
    ADVANCED_MONSTER_WEIGHTS,
    ENABLE_OPTIONS_FLOW,
    ENABLE_SOCIAL_BUZZ
)

# Options flow (optional - requires IBKR OPRA subscription)
OPTIONS_FLOW_AVAILABLE = False
if ENABLE_OPTIONS_FLOW:
    try:
        from src.options_flow_ibkr import get_options_flow_score
        OPTIONS_FLOW_AVAILABLE = True
    except ImportError:
        pass

# Extended hours boost (optional - requires IBKR subscription)
EXTENDED_HOURS_AVAILABLE = False
try:
    from src.extended_hours_quotes import get_extended_hours_boost
    EXTENDED_HOURS_AVAILABLE = True
except ImportError:
    pass

logger = get_logger("MONSTER_SCORE")

cache = Cache(ttl=30)

WEIGHT_FILE = "data/monster_score_weights.json"

# S2-4 FIX: Cache weights at module level — avoid file I/O on every tick.
# compute_monster_score() is called once per ticker per cycle (100s of tickers).
# Before: open+read+json.load() per call. After: read once, cached in _weights.
# reload_weights() lets score_optimizer refresh without restarting the process.
_weights: Optional[dict] = None


# ============================
# Load weights (auto-tuning ready)
# ============================

def load_weights() -> dict:
    """
    Load weights from file or use ADVANCED_MONSTER_WEIGHTS as default.
    Cached after first call; use reload_weights() to force refresh.

    Priority:
    1. Custom weights from weekly_audit auto-tuning (if exists)
    2. ADVANCED_MONSTER_WEIGHTS from config (optimized)
    """
    global _weights
    if _weights is not None:
        return _weights

    if os.path.exists(WEIGHT_FILE):
        try:
            with open(WEIGHT_FILE) as f:
                custom_weights = json.load(f)
                logger.info("Using custom auto-tuned weights")
                _weights = custom_weights
                return _weights
        except Exception:
            pass

    # Use optimized weights from config
    logger.info("Using ADVANCED_MONSTER_WEIGHTS from config")
    _weights = ADVANCED_MONSTER_WEIGHTS.copy()
    return _weights


def reload_weights() -> dict:
    """Force reload weights from file (called by score_optimizer after auto-tune)."""
    global _weights
    _weights = None
    return load_weights()


def save_weights(w):
    os.makedirs("data", exist_ok=True)
    with open(WEIGHT_FILE, "w") as f:
        json.dump(w, f, indent=2)


# ============================
# Normalize helpers
# ============================

def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


def normalize(x, scale=1):
    try:
        return clamp(x / scale)
    except:
        return 0


# ============================
# Score computation
# ============================

def compute_event_score(ticker):
    """
    S5-3 FIX: Use CatalystScorerV3 when available for richer event scoring.
    Falls back to event_hub raw boosted_impact if V3 unavailable.
    """
    events = get_events_by_ticker(ticker)

    if not events:
        return 0

    # Base score: best boosted_impact from event_hub (guarded against KeyError)
    base_score = 0.0
    for e in events:
        val = e.get("boosted_impact")
        if val is not None:
            base_score = max(base_score, val)

    if not CATALYST_V3_AVAILABLE:
        return base_score

    # Enhanced score via CatalystScorerV3 singleton
    try:
        scorer = get_catalyst_scorer()
        catalysts = [
            create_catalyst_from_event(
                ticker=ticker,
                event_type=e.get("event_type", "unknown"),
                headline=e.get("headline", ""),
                source=e.get("source", "financial"),
                impact_score=e.get("boosted_impact", 0.5)
            )
            for e in events
        ]
        v3_score = scorer.score_catalysts(ticker, catalysts)
        # Additive: take max of base and V3 final_score
        return max(base_score, v3_score.final_score)
    except Exception as exc:
        logger.debug(f"CatalystV3 score error for {ticker}: {exc}")
        return base_score


def compute_monster_score(ticker, use_advanced=True):
    """
    Compute Monster Score with optional advanced patterns
    
    Args:
        ticker: stock ticker
        use_advanced: if True, use pattern_analyzer and pm_transition
    
    Returns: dict with score and components
    """
    cached = cache.get(f"score_{ticker}")
    if cached:
        return cached

    weights = load_weights()

    # ===== EVENTS =====
    event_score = compute_event_score(ticker)

    # ===== FEATURES =====
    feats = compute_features(ticker, include_advanced=use_advanced)

    if not feats:
        return None

    momentum = normalize(abs(feats["momentum"]), 0.2)
    volume = normalize(feats["volume_spike"], 5)
    squeeze = normalize(feats["squeeze_proxy"], 10)

    # ===== V8: ACCELERATION ENGINE =====
    acceleration_score_value = 0
    acceleration_boost = 0
    acceleration_state = "DORMANT"
    if ACCELERATION_AVAILABLE:
        try:
            accel_engine = get_acceleration_engine()
            accel_data = accel_engine.score(ticker)
            acceleration_score_value = accel_data.acceleration_score
            acceleration_boost = accel_data.get_monster_score_boost()
            acceleration_state = accel_data.state

            if acceleration_score_value > 0.2:
                logger.debug(
                    f"{ticker} V8 acceleration: {acceleration_score_value:.2f} "
                    f"state={acceleration_state} boost={acceleration_boost:+.3f}"
                )
        except Exception as e:
            logger.debug(f"Acceleration engine error for {ticker}: {e}")

    # ===== PREMARKET =====
    pm = compute_pm_metrics(ticker)
    
    # ===== ADVANCED PATTERNS (if enabled) =====
    pattern_score = 0
    pm_transition_score = 0
    
    if use_advanced:
        try:
            # Get full dataframe for pattern analysis
            from src.feature_engine import fetch_candles
            df = fetch_candles(ticker)
            
            if df is not None and len(df) >= 20:
                # Pattern Analyzer
                from src.pattern_analyzer import compute_pattern_score as calc_pattern
                pattern_data = calc_pattern(ticker, df, pm)
                pattern_score = pattern_data.get("pattern_score", 0)
                
                # PM Transition Analyzer
                if pm:
                    from src.pm_transition import compute_pm_transition_score
                    transition_data = compute_pm_transition_score(ticker, df, pm)
                    pm_transition_score = transition_data.get("pm_transition_score", 0)
        
        except Exception as e:
            logger.warning(f"Advanced scoring error for {ticker}: {e}")
            # Continue with basic scoring
    
    # ===== INTELLIGENCE MODULES V3 (FULLY INTEGRATED) =====
    # Options flow and social buzz are now CORE components (not just boosts)
    options_flow_score = 0
    social_buzz_score = 0
    beat_rate_boost = 0
    extended_hours_boost = 0

    try:
        # ----- OPTIONS FLOW (10% weight) -----
        if OPTIONS_FLOW_AVAILABLE and ENABLE_OPTIONS_FLOW:
            try:
                opt_score, opt_details = get_options_flow_score(ticker)
                options_flow_score = clamp(opt_score)  # 0-1
                if options_flow_score > 0.3:
                    logger.debug(f"{ticker} options flow: {options_flow_score:.2f}")
            except Exception as e:
                logger.debug(f"Options flow error for {ticker}: {e}")

        # ----- SOCIAL BUZZ (6% weight) -----
        if ENABLE_SOCIAL_BUZZ:
            try:
                buzz_data = get_total_buzz_score(ticker)
                social_buzz_score = clamp(buzz_data.get("buzz_score", 0))  # 0-1
                if social_buzz_score > 0.3:
                    logger.debug(f"{ticker} social buzz: {social_buzz_score:.2f}")
            except Exception as e:
                logger.debug(f"Social buzz error for {ticker}: {e}")

        # ----- BEAT RATE BOOST (additive for earnings events) -----
        events = get_events_by_ticker(ticker)
        has_earnings = any("earning" in e.get("type", "").lower() for e in events)

        if has_earnings:
            try:
                earnings_prob = get_earnings_probability(ticker)
                if earnings_prob and earnings_prob > 0.6:
                    # High probability of beat = boost score (max +0.15)
                    beat_rate_boost = (earnings_prob - 0.5) * 0.3
                    logger.info(f"{ticker} beat probability: {earnings_prob*100:.0f}% → +{beat_rate_boost:.2f}")
            except Exception as e:
                logger.debug(f"Beat rate error for {ticker}: {e}")

        # ----- EXTENDED HOURS BOOST (additive for gap plays) -----
        if EXTENDED_HOURS_AVAILABLE:
            try:
                eh_boost, eh_details = get_extended_hours_boost(ticker)
                if eh_boost > 0:
                    extended_hours_boost = eh_boost  # Max +0.22
                    logger.debug(f"{ticker} extended hours: +{eh_boost:.2f}")
            except Exception as e:
                logger.debug(f"Extended hours error for {ticker}: {e}")

    except Exception as e:
        logger.warning(f"Intelligence modules error for {ticker}: {e}")

    # ===== FINAL SCORE V4 (UNIFIED WEIGHTED SCORE WITH ACCELERATION) =====
    # V4: Added acceleration component, reduced momentum and social_buzz
    base_score = (
        weights.get("event", 0.25) * event_score +
        weights.get("volume", 0.17) * volume +
        weights.get("pattern", 0.17) * pattern_score +
        weights.get("pm_transition", 0.13) * pm_transition_score +
        weights.get("acceleration", 0.07) * acceleration_score_value +  # V8 NEW
        weights.get("momentum", 0.04) * momentum +         # V8: Reduced from 0.08
        weights.get("squeeze", 0.04) * squeeze +
        weights.get("options_flow", 0.10) * options_flow_score +
        weights.get("social_buzz", 0.03) * social_buzz_score   # V8: Reduced from 0.06
    )

    # Add conditional boosts (beat rate + extended hours + V8 acceleration)
    total_boost = beat_rate_boost + extended_hours_boost + acceleration_boost
    final_score = base_score + total_boost

    # Clamp 0-1
    final_score = clamp(final_score)

    details = {
        "monster_score": final_score,
        "base_score": round(base_score, 4),
        "intelligence_boost": round(total_boost, 4),
        "version": "V4",  # V8 marker
        "components": {
            "event": round(event_score * weights.get("event", 0.25), 4),
            "volume": round(volume * weights.get("volume", 0.17), 4),
            "pattern": round(pattern_score * weights.get("pattern", 0.17), 4),
            "pm_transition": round(pm_transition_score * weights.get("pm_transition", 0.13), 4),
            "acceleration": round(acceleration_score_value * weights.get("acceleration", 0.07), 4),  # V8
            "momentum": round(momentum * weights.get("momentum", 0.04), 4),
            "squeeze": round(squeeze * weights.get("squeeze", 0.04), 4),
            "options_flow": round(options_flow_score * weights.get("options_flow", 0.10), 4),
            "social_buzz": round(social_buzz_score * weights.get("social_buzz", 0.03), 4),
            "beat_rate_boost": round(beat_rate_boost, 4),
            "extended_hours_boost": round(extended_hours_boost, 4),
            "acceleration_boost": round(acceleration_boost, 4),  # V8
        },
        "raw_scores": {
            "event": round(event_score, 4),
            "volume": round(volume, 4),
            "pattern": round(pattern_score, 4),
            "pm_transition": round(pm_transition_score, 4),
            "acceleration": round(acceleration_score_value, 4),  # V8
            "acceleration_state": acceleration_state,              # V8
            "momentum": round(momentum, 4),
            "squeeze": round(squeeze, 4),
            "options_flow": round(options_flow_score, 4),
            "social_buzz": round(social_buzz_score, 4)
        }
    }

    cache.set(f"score_{ticker}", details)

    return details


# ============================
# Batch helper
# ============================

def score_many(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        s = compute_monster_score(t)
        if s:
            results[t] = s

    logger.info(f"Scored {len(results)} tickers")

    return results


if __name__ == "__main__":
    print(compute_monster_score("AAPL"))
