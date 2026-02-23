"""
PORTFOLIO ENGINE - DEPRECATED (V8)
====================================

This module is DEPRECATED since V8.0.
Use src.engines.order_computer.OrderComputer for position sizing instead.

Kept for backward compatibility with legacy edge_cycle() in main.py
and backtests/backtest_engine_edge.py.
"""

import warnings
from utils.logger import get_logger
from utils.api_guard import safe_get
from utils.time_utils import market_session

from config import (
    RISK_BUY,
    RISK_BUY_STRONG,
    ATR_MULTIPLIER_STOP,
    FINNHUB_API_KEY,
    MANUAL_CAPITAL
)

logger = get_logger("PORTFOLIO_ENGINE")

_DEPRECATION_WARNED = False


def _warn_deprecated():
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        warnings.warn(
            "portfolio_engine is deprecated since V8.0. "
            "Use src.engines.order_computer.OrderComputer instead.",
            DeprecationWarning,
            stacklevel=3
        )
        logger.warning("DEPRECATED: portfolio_engine.py - use OrderComputer (V8+)")
        _DEPRECATION_WARNED = True

FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote"


# ============================
# Price fetch
# ============================

def fetch_price(ticker):
    params = {
        "symbol": ticker,
        "token": FINNHUB_API_KEY
    }

    r = safe_get(FINNHUB_QUOTE, params=params)
    return r.json().get("c")


# ============================
# ATR estimation (light)
# ============================

def estimate_atr(features, price=None):
    """
    Estime l'ATR (Average True Range) en dollars
    Utilise la volatilité normalisée et multiplie par le prix
    """
    # Volatilité normalisée (0-1)
    vol_normalized = abs(features.get("volatility", 0.05))
    
    # Convertir en pourcentage réel (0-1 -> 0-5%)
    vol_pct = vol_normalized * 0.05
    
    # Si on a un prix, calculer l'ATR en dollars
    # Sinon, utiliser un proxy de 2% du prix moyen
    if price:
        atr_dollars = price * vol_pct
    else:
        # Proxy: 2% de volatilité sur un prix moyen de $10
        atr_dollars = 10 * 0.02
    
    # Minimum ATR pour éviter divisions par zéro
    return max(atr_dollars, 0.10)


# ============================
# Position sizing
# ============================

def compute_position(signal, features, capital=None):
    """DEPRECATED: Use OrderComputer.compute_order() instead."""
    _warn_deprecated()
    if capital is None:
        capital = MANUAL_CAPITAL
    
    ticker = signal["ticker"]
    signal_type = signal["signal"]
    
    # Déterminer le risque selon le type de signal
    risk_pct = RISK_BUY_STRONG if signal_type == "BUY_STRONG" else RISK_BUY
    
    price = fetch_price(ticker)

    if not price or price <= 0:
        logger.warning(f"Invalid price for {ticker}: {price}")
        return None

    atr = estimate_atr(features, price)  # Passer le prix pour meilleur calcul

    stop_distance = atr * ATR_MULTIPLIER_STOP

    risk_amount = capital * risk_pct

    shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0

    entry = price
    stop = price - stop_distance

    position = {
        "ticker": ticker,
        "signal": signal_type,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "shares": shares,
        "risk_amount": round(risk_amount, 2),
        "risk_pct": risk_pct,
        "session": market_session(),
        "monster_score": signal.get("monster_score", 0),
        "confidence": signal.get("confidence", 0)
    }

    return position


# ============================
# Trailing stop logic (simple)
# ============================

def update_trailing_stop(position, new_price):
    risk = position["entry"] - position["stop"]

    # trail when price moves 1R in favor
    if new_price - position["entry"] >= risk:
        new_stop = new_price - risk
        position["stop"] = max(position["stop"], new_stop)

    return position


# ============================
# Process signal (main entry point)
# ============================

def process_signal(signal, capital=None):
    """DEPRECATED: Use OrderComputer.compute_order() instead."""
    _warn_deprecated()
    from src.feature_engine import compute_features
    
    ticker = signal["ticker"]
    
    # Récupérer les features pour calculer l'ATR
    features = compute_features(ticker)
    
    if not features:
        logger.warning(f"No features available for {ticker}, cannot compute position")
        return None
    
    # Calculer la position
    position = compute_position(signal, features, capital)
    
    if not position:
        logger.warning(f"Could not compute position for {ticker}")
        return None
    
    # Vérifier que la position est valide
    if position["shares"] <= 0:
        logger.warning(f"Invalid position size for {ticker}: {position['shares']} shares")
        return None
    
    logger.info(
        f"Trade plan for {ticker}: {position['shares']} shares @ ${position['entry']} "
        f"(stop: ${position['stop']}, risk: ${position['risk_amount']})"
    )
    
    return position


if __name__ == "__main__":
    print("Portfolio engine ready")
