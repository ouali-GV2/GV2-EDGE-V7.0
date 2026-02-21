"""
ENSEMBLE ENGINE - DEPRECATED (V9)
==================================

This module is DEPRECATED since V9.0.
Use src.engines.multi_radar_engine.MultiRadarEngine (Confluence Matrix) instead.

Kept for backward compatibility with legacy edge_cycle() in main.py.
"""

import warnings
from utils.logger import get_logger

logger = get_logger("ENSEMBLE_ENGINE")

_DEPRECATION_WARNED = False


def _warn_deprecated():
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        warnings.warn(
            "ensemble_engine is deprecated since V9.0. "
            "Use src.engines.multi_radar_engine.MultiRadarEngine instead.",
            DeprecationWarning,
            stacklevel=3
        )
        logger.warning("DEPRECATED: ensemble_engine.py - use MultiRadarEngine (V9+)")
        _DEPRECATION_WARNED = True


# ============================
# Soft confluence logic (DEPRECATED)
# ============================

def apply_confluence(signal):
    """
    DEPRECATED: Use MultiRadarEngine.scan_ticker() instead.
    Soft ensemble: doesn't block signals, only boosts or soft reduces confidence.
    """
    _warn_deprecated()

    score = signal["monster_score"]
    components = signal.get("components", {})

    boost = 1.0

    # Event strong catalyst
    if components.get("event", 0) > 0.6:
        boost += 0.15

    # Strong momentum + volume
    if components.get("momentum", 0) > 0.6 and components.get("volume", 0) > 0.6:
        boost += 0.15

    # Squeeze pressure
    if components.get("squeeze", 0) > 0.7:
        boost += 0.1

    # Premarket gap strong
    if components.get("pm_gap", 0) > 0.5:
        boost += 0.1

    new_confidence = min(1.0, signal["confidence"] * boost)

    signal["confidence"] = new_confidence
    signal["ensemble_boost"] = round(boost, 2)

    return signal


def apply_many(signals):
    enhanced = []

    for s in signals:
        enhanced.append(apply_confluence(s))

    logger.info(f"Confluence applied to {len(enhanced)} signals")

    return enhanced


if __name__ == "__main__":
    test = {
        "monster_score": 0.7,
        "confidence": 0.7,
        "components": {
            "event": 0.8,
            "momentum": 0.7,
            "volume": 0.8,
            "squeeze": 0.6,
            "pm_gap": 0.7
        }
    }

    print(apply_confluence(test))
