# ============================
# MOCK DATA FOR TESTING
# ============================
"""
Ce fichier contient des données simulées pour tester GV2-EDGE
sans avoir besoin d'accès aux APIs réelles.
"""

import pandas as pd
from datetime import datetime, timedelta

# ============================
# MOCK UNIVERSE
# ============================

def get_mock_universe():
    """
    Retourne un univers fictif de small caps
    """
    return pd.DataFrame({
        "ticker": ["MOCK", "TEST", "DEMO", "FAKE", "SIMU"],
        "market_cap": [500_000_000, 800_000_000, 1_200_000_000, 300_000_000, 900_000_000],
        "price": [5.50, 12.30, 8.75, 3.20, 15.80],
        "avg_volume": [500_000, 750_000, 1_200_000, 350_000, 900_000]
    })


# ============================
# MOCK PRICE DATA (OHLCV)
# ============================

def get_mock_candles(ticker, lookback=120):
    """
    Retourne des données OHLCV fictives
    Simule un mouvement momentum haussier
    """
    import numpy as np
    
    # Prix de base selon le ticker
    base_prices = {
        "MOCK": 5.50,
        "TEST": 12.30,
        "DEMO": 8.75,
        "FAKE": 3.20,
        "SIMU": 15.80
    }
    
    base = base_prices.get(ticker, 10.0)
    
    # Simuler un trend haussier avec volatilité
    prices = []
    current = base * 0.85  # Commence 15% plus bas
    
    for i in range(lookback):
        # Trend haussier avec bruit
        trend = 0.003  # +0.3% par période en moyenne
        noise = np.random.normal(0, 0.015)  # Volatilité 1.5%
        
        current = current * (1 + trend + noise)
        prices.append(current)
    
    # Créer OHLCV
    data = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    }
    
    for price in prices:
        o = price * np.random.uniform(0.995, 1.005)
        h = price * np.random.uniform(1.005, 1.025)
        l = price * np.random.uniform(0.975, 0.995)
        c = price
        v = np.random.randint(300_000, 2_000_000)
        
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(l)
        data["close"].append(c)
        data["volume"].append(v)
    
    return pd.DataFrame(data)


# ============================
# MOCK EVENTS
# ============================

def get_mock_events(ticker=None):
    """
    Retourne des événements fictifs
    """
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    all_events = [
        {
            "ticker": "MOCK",
            "type": "FDA_APPROVAL",
            "date": today,
            "impact": 0.85,
            "boosted_impact": 0.85 * 1.3,  # today boost
            "headline": "Mock Corp receives FDA approval for new drug"
        },
        {
            "ticker": "MOCK",
            "type": "EARNINGS",
            "date": tomorrow,
            "impact": 0.70,
            "boosted_impact": 0.70 * 1.15,  # tomorrow boost
            "headline": "Mock Corp to report Q4 earnings"
        },
        {
            "ticker": "TEST",
            "type": "M&A",
            "date": today,
            "impact": 0.90,
            "boosted_impact": 0.90 * 1.3,
            "headline": "Test Inc announces acquisition talks"
        },
        {
            "ticker": "DEMO",
            "type": "ANALYST_UPGRADE",
            "date": next_week,
            "impact": 0.60,
            "boosted_impact": 0.60 * 1.05,  # week boost
            "headline": "Demo Corp upgraded to Buy by major analyst"
        },
        {
            "ticker": "FAKE",
            "type": "PRODUCT_LAUNCH",
            "date": today,
            "impact": 0.75,
            "boosted_impact": 0.75 * 1.3,
            "headline": "Fake Ltd launches revolutionary product"
        }
    ]
    
    if ticker:
        return [e for e in all_events if e["ticker"] == ticker]
    
    return all_events


# ============================
# MOCK FEATURES
# ============================

def get_mock_features(ticker):
    """
    Retourne des features techniques fictives
    Simule un ticker avec bon momentum
    """
    # Différents profils selon le ticker
    profiles = {
        "MOCK": {  # Très fort momentum + volume
            "momentum": 0.85,
            "volume_spike": 0.90,
            "vwap_dev": 0.75,
            "volatility": 0.60,
            "squeeze_proxy": 0.80,
            "breakout": 1.0,
            "strong_green": 1.0
        },
        "TEST": {  # Bon momentum moyen
            "momentum": 0.70,
            "volume_spike": 0.65,
            "vwap_dev": 0.55,
            "volatility": 0.45,
            "squeeze_proxy": 0.60,
            "breakout": 1.0,
            "strong_green": 1.0
        },
        "DEMO": {  # Momentum faible
            "momentum": 0.40,
            "volume_spike": 0.35,
            "vwap_dev": 0.25,
            "volatility": 0.30,
            "squeeze_proxy": 0.35,
            "breakout": 0.0,
            "strong_green": 0.0
        },
        "FAKE": {  # Très volatil
            "momentum": 0.65,
            "volume_spike": 0.80,
            "vwap_dev": 0.65,
            "volatility": 0.85,
            "squeeze_proxy": 0.40,
            "breakout": 1.0,
            "strong_green": 1.0
        },
        "SIMU": {  # Équilibré
            "momentum": 0.60,
            "volume_spike": 0.55,
            "vwap_dev": 0.50,
            "volatility": 0.40,
            "squeeze_proxy": 0.55,
            "breakout": 1.0,
            "strong_green": 0.0
        }
    }
    
    return profiles.get(ticker, profiles["DEMO"])


# ============================
# MOCK PRE-MARKET DATA
# ============================

def get_mock_pm_data(ticker):
    """
    Retourne des données pre-market fictives
    """
    pm_profiles = {
        "MOCK": {
            "gap_pct": 0.125,  # +12.5% gap
            "pm_high": 6.20,
            "pm_low": 5.85,
            "pm_momentum": 0.060,
            "pm_volume": 450_000,
            "pm_liquid": True
        },
        "TEST": {
            "gap_pct": 0.085,  # +8.5% gap
            "pm_high": 13.35,
            "pm_low": 12.80,
            "pm_momentum": 0.043,
            "pm_volume": 280_000,
            "pm_liquid": True
        },
        "DEMO": {
            "gap_pct": 0.025,  # +2.5% gap
            "pm_high": 9.00,
            "pm_low": 8.75,
            "pm_momentum": 0.029,
            "pm_volume": 120_000,
            "pm_liquid": True
        },
        "FAKE": {
            "gap_pct": 0.095,  # +9.5% gap
            "pm_high": 3.55,
            "pm_low": 3.30,
            "pm_momentum": 0.076,
            "pm_volume": 180_000,
            "pm_liquid": True
        },
        "SIMU": {
            "gap_pct": 0.045,  # +4.5% gap
            "pm_high": 16.50,
            "pm_low": 15.90,
            "pm_momentum": 0.038,
            "pm_volume": 320_000,
            "pm_liquid": True
        }
    }
    
    return pm_profiles.get(ticker, None)


# ============================
# MOCK SOCIAL SENTIMENT
# ============================

def get_mock_sentiment(ticker):
    """
    Retourne un sentiment social fictif
    """
    sentiments = {
        "MOCK": {"score": 0.85, "mentions": 450, "trend": "bullish"},
        "TEST": {"score": 0.70, "mentions": 320, "trend": "bullish"},
        "DEMO": {"score": 0.45, "mentions": 85, "trend": "neutral"},
        "FAKE": {"score": 0.75, "mentions": 220, "trend": "bullish"},
        "SIMU": {"score": 0.60, "mentions": 180, "trend": "neutral"}
    }
    
    return sentiments.get(ticker, {"score": 0.50, "mentions": 0, "trend": "neutral"})


# ============================
# MOCK NEWS BUZZ
# ============================

def get_mock_buzz(ticker):
    """
    Retourne un buzz news fictif
    """
    buzz = {
        "MOCK": {"articles": 15, "sentiment": 0.80, "velocity": "high"},
        "TEST": {"articles": 8, "sentiment": 0.65, "velocity": "medium"},
        "DEMO": {"articles": 3, "sentiment": 0.50, "velocity": "low"},
        "FAKE": {"articles": 12, "sentiment": 0.70, "velocity": "high"},
        "SIMU": {"articles": 6, "sentiment": 0.55, "velocity": "medium"}
    }
    
    return buzz.get(ticker, {"articles": 0, "sentiment": 0.50, "velocity": "low"})


# ============================
# MOCK LIVE QUOTE
# ============================

def get_mock_quote(ticker):
    """
    Retourne un quote en temps réel fictif
    """
    base_prices = {
        "MOCK": 5.50,
        "TEST": 12.30,
        "DEMO": 8.75,
        "FAKE": 3.20,
        "SIMU": 15.80
    }
    
    price = base_prices.get(ticker, 10.0)
    
    return {
        "c": price,  # current price
        "h": price * 1.05,  # high
        "l": price * 0.95,  # low
        "o": price * 0.98,  # open
        "pc": price * 0.95,  # previous close
        "v": 850_000  # volume
    }


# ============================
# EXPECTED RESULTS
# ============================

def get_expected_signals():
    """
    Signaux attendus pour valider la logique
    """
    return {
        "MOCK": {
            "expected_signal": "BUY_STRONG",
            "expected_score_range": (0.80, 1.0),
            "reason": "Strong events + momentum + PM gap + volume"
        },
        "TEST": {
            "expected_signal": "BUY",
            "expected_score_range": (0.65, 0.80),
            "reason": "Good events + decent momentum"
        },
        "DEMO": {
            "expected_signal": "HOLD",
            "expected_score_range": (0.0, 0.65),
            "reason": "Weak momentum + weak events"
        },
        "FAKE": {
            "expected_signal": "BUY",
            "expected_score_range": (0.65, 0.80),
            "reason": "Strong events + high volatility"
        },
        "SIMU": {
            "expected_signal": "HOLD",
            "expected_score_range": (0.50, 0.70),
            "reason": "Moderate across the board"
        }
    }


if __name__ == "__main__":
    # Test rapide
    print("=== MOCK UNIVERSE ===")
    print(get_mock_universe())
    
    print("\n=== MOCK EVENTS FOR 'MOCK' ===")
    print(get_mock_events("MOCK"))
    
    print("\n=== MOCK FEATURES FOR 'MOCK' ===")
    print(get_mock_features("MOCK"))
    
    print("\n=== EXPECTED SIGNALS ===")
    import json
    print(json.dumps(get_expected_signals(), indent=2))
