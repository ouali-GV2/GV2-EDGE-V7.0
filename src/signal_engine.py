"""
SIGNAL ENGINE - DEPRECATED (V8)
================================

This module is DEPRECATED since V7.0.
Use src.engines.signal_producer.SignalProducer instead.

This file is kept ONLY for backward compatibility with:
- main.py
- backtests/backtest_engine_edge.py
- tests/test_pipeline.py

All functions delegate to the V7 Monster Score. For new code,
use SignalProducer directly.
"""

import warnings

from utils.logger import get_logger
from utils.data_validator import validate_signal

from src.scoring.monster_score import compute_monster_score

from config import (
    BUY_THRESHOLD,
    BUY_STRONG_THRESHOLD
)

logger = get_logger("SIGNAL_ENGINE")

_DEPRECATION_WARNED = False


def _warn_deprecated():
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        warnings.warn(
            "signal_engine is deprecated since V7.0. "
            "Use src.engines.signal_producer.SignalProducer instead.",
            DeprecationWarning,
            stacklevel=3
        )
        logger.warning("DEPRECATED: signal_engine.py - use SignalProducer (V7+)")
        _DEPRECATION_WARNED = True


def generate_signal(ticker):
    """DEPRECATED: Use SignalProducer.detect() instead."""
    _warn_deprecated()

    score_data = compute_monster_score(ticker)

    if not score_data:
        return None

    score = score_data["monster_score"]

    if score >= BUY_STRONG_THRESHOLD:
        signal_type = "BUY_STRONG"
    elif score >= BUY_THRESHOLD:
        signal_type = "BUY"
    else:
        signal_type = "HOLD"

    confidence = min(1.0, score)

    signal = {
        "ticker": ticker,
        "signal": signal_type,
        "confidence": confidence,
        "monster_score": score,
        "components": score_data["components"]
    }

    if not validate_signal(signal):
        return None

    return signal


def generate_many(tickers, limit=None):
    """DEPRECATED: Use SignalProducer.detect_batch() instead."""
    _warn_deprecated()

    signals = []

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        s = generate_signal(t)
        if s:
            signals.append(s)

    logger.info(f"Generated {len(signals)} signals")

    return signals
