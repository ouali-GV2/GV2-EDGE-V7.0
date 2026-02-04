"""
PATTERN ANALYZER - Advanced Pattern Detection for GV2-EDGE
===========================================================

Détecte les patterns structurels qui précèdent les +50%/+100% movers:
- PM Break + RTH Retest
- Tight Consolidation Explosive
- Higher Lows Progressive
- Flag/Pennant Continuation
- Volume Profile Analysis
- Bollinger Squeeze

Philosophie: Structure de marché > IA lourde
"""

import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("PATTERN_ANALYZER")
cache = Cache(ttl=60)

# ============================
# VOLUME PROFILE ANALYSIS
# ============================

def volume_accumulation(df, window=20):
    """
    Détecte accumulation progressive
    Volume croissant + prix dans range tight = setup squeeze
    
    Returns: 0-1 score
    """
    if len(df) < window + 10:
        return 0
    
    recent = df.iloc[-window:]
    
    # Volume trend (régression linéaire simple)
    volumes = recent["volume"].values
    x = np.arange(len(volumes))
    
    # Pente volume (croissant = positif)
    vol_slope = np.polyfit(x, volumes, 1)[0]
    vol_mean = volumes.mean()
    vol_trend = vol_slope / vol_mean if vol_mean > 0 else 0
    
    # Price range tightness
    highs = recent["high"]
    lows = recent["low"]
    closes = recent["close"]
    
    avg_range = ((highs - lows) / closes).mean()
    
    # Score accumulation
    # Volume croissant (vol_trend > 0) + range tight (< 3%)
    if vol_trend > 0 and avg_range < 0.03:
        score = min(1.0, vol_trend * 20)  # Normaliser
        return score
    
    return 0


def volume_climax(df):
    """
    Détecte volume climax (spike soudain 5x+)
    Indicateur de pré-explosion ou capitulation
    
    Returns: 0-1 score
    """
    if len(df) < 53:
        return 0
    
    # Volume des 3 dernières candles
    recent_vol = df["volume"].iloc[-3:].mean()
    
    # Volume moyen 50 candles (excluant les 3 dernières)
    avg_vol = df["volume"].iloc[-53:-3].mean()
    
    if avg_vol <= 0:
        return 0
    
    ratio = recent_vol / avg_vol
    
    # Climax si > 5x
    if ratio >= 5:
        return 1.0
    elif ratio >= 3:
        return 0.7
    elif ratio >= 2:
        return 0.4
    
    return min(1.0, ratio / 5)


def volume_profile_bullish(df, window=20):
    """
    Volume profile bullish:
    - Plus de volume sur candles vertes
    - Volume croissant sur les ups
    
    Returns: 0-1 score
    """
    if len(df) < window:
        return 0
    
    recent = df.iloc[-window:]
    
    # Séparer candles vertes et rouges
    green_candles = recent[recent["close"] > recent["open"]]
    red_candles = recent[recent["close"] <= recent["open"]]
    
    if len(green_candles) == 0:
        return 0
    
    vol_green = green_candles["volume"].mean()
    vol_red = red_candles["volume"].mean() if len(red_candles) > 0 else 1
    
    # Ratio volume vert vs rouge
    if vol_red <= 0:
        return 1.0
    
    ratio = vol_green / vol_red
    
    # Bullish si vert > rouge
    if ratio >= 2.0:
        return 1.0
    elif ratio >= 1.5:
        return 0.7
    elif ratio > 1.0:
        return 0.4
    
    return 0


# ============================
# STRUCTURE PATTERNS
# ============================

def higher_lows_pattern(df, window=10, min_touches=3):
    """
    Détecte série de higher lows (accumulation)
    Pattern bullish classique
    
    Returns: 0-1 score
    """
    if len(df) < window + 5:
        return 0
    
    recent = df.iloc[-window:]
    lows = recent["low"].values
    
    # Trouver les lows locaux (creux)
    local_lows = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            local_lows.append(lows[i])
    
    if len(local_lows) < min_touches:
        return 0
    
    # Vérifier si les lows sont progressivement plus hauts
    higher_count = 0
    for i in range(1, len(local_lows)):
        if local_lows[i] > local_lows[i-1]:
            higher_count += 1
    
    # Score basé sur % de higher lows
    if len(local_lows) > 1:
        score = higher_count / (len(local_lows) - 1)
        return score
    
    return 0


def tight_consolidation(df, window=15, max_range_pct=0.03):
    """
    Détecte consolidation tight (coil)
    Range < 3% pendant 15+ candles = squeeze
    
    Returns: 0-1 score
    """
    if len(df) < window:
        return 0
    
    recent = df.iloc[-window:]
    
    high = recent["high"].max()
    low = recent["low"].min()
    close = recent["close"].iloc[-1]
    
    if close <= 0:
        return 0
    
    range_pct = (high - low) / close
    
    # Score inversé: plus tight = meilleur score
    if range_pct <= max_range_pct:
        # Parfait squeeze
        tightness = 1 - (range_pct / max_range_pct)
        return max(0.8, tightness)
    
    # Pas assez tight
    return 0


def flag_pennant_pattern(df, spike_threshold=0.10, consol_window=10):
    """
    Détecte Flag/Pennant après spike initial
    Pattern continuation classique
    
    Returns: 0-1 score
    """
    if len(df) < consol_window + 10:
        return 0
    
    # Vérifier spike initial (10-20 candles avant)
    spike_window = df.iloc[-(consol_window + 20):-(consol_window)]
    
    if len(spike_window) < 5:
        return 0
    
    spike_move = (spike_window["close"].iloc[-1] - spike_window["close"].iloc[0]) / spike_window["close"].iloc[0]
    
    # Pas de spike initial
    if spike_move < spike_threshold:
        return 0
    
    # Vérifier consolidation après spike
    consol = df.iloc[-consol_window:]
    consol_range = (consol["high"].max() - consol["low"].min()) / consol["close"].iloc[-1]
    
    # Consolidation tight après spike
    if consol_range < 0.05:  # < 5% range
        return 0.9
    elif consol_range < 0.08:
        return 0.6
    
    return 0


# ============================
# BOLLINGER SQUEEZE
# ============================

def bollinger_squeeze(df, window=20, squeeze_threshold=0.5):
    """
    Détecte compression Bollinger Bands
    Bandes serrées = volatilité basse = explosion imminente
    
    Returns: 0-1 score
    """
    if len(df) < window + 50:
        return 0
    
    # Calcul Bollinger Bands
    closes = df["close"]
    sma = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Largeur des bandes (bandwidth)
    bandwidth = (upper - lower) / sma
    
    # Bandwidth actuel
    current_bw = bandwidth.iloc[-1]
    
    # Bandwidth moyen sur 50 candles
    avg_bw = bandwidth.iloc[-50:].mean()
    
    if avg_bw <= 0:
        return 0
    
    # Ratio squeeze
    squeeze_ratio = current_bw / avg_bw
    
    # Squeeze si bandwidth actuel < moyenne
    if squeeze_ratio < 0.5:
        return 1.0  # Squeeze fort
    elif squeeze_ratio < 0.7:
        return 0.7
    elif squeeze_ratio < 0.9:
        return 0.4
    
    return 0


# ============================
# MOMENTUM ACCELERATION
# ============================

def momentum_acceleration(df, window=10):
    """
    Mesure accélération du momentum
    Dérivée 1 = vitesse, Dérivée 2 = accélération
    Accélération positive = pression acheteuse croissante
    
    Returns: 0-1 score
    """
    if len(df) < window + 2:
        return 0
    
    prices = df["close"].iloc[-window:]
    
    # 1ère dérivée (velocity)
    velocity = prices.diff()
    
    # 2ème dérivée (acceleration)
    acceleration = velocity.diff()
    
    # Accélération récente
    recent_accel = acceleration.iloc[-3:].mean()
    
    # Normaliser autour de 0
    # Positif = bullish, négatif = bearish
    if recent_accel > 0:
        # Accélération positive et croissante
        if acceleration.iloc[-1] > acceleration.iloc[-2]:
            return min(1.0, recent_accel * 100)
        else:
            return min(0.7, recent_accel * 100)
    
    return 0


# ============================
# VOLUME SQUEEZE DETECTION
# ============================

def volume_squeeze_score(df, compression_window=10):
    """
    Détecte compression volume → expansion
    Volume faible pendant consolidation puis spike = explosion
    
    Returns: 0-1 score
    """
    if len(df) < compression_window + 50:
        return 0
    
    # Volume récent (compression phase)
    recent_vol = df["volume"].iloc[-compression_window:]
    
    # Volume moyen 50 candles (avant compression)
    avg_vol = df["volume"].iloc[-(compression_window + 50):-compression_window].mean()
    
    if avg_vol <= 0:
        return 0
    
    # Phase compression: volume récent < moyenne
    compression_ratio = recent_vol.mean() / avg_vol
    compression_phase = compression_ratio < 0.6
    
    # Phase expansion: dernière candle volume spike
    last_vol = df["volume"].iloc[-1]
    expansion = last_vol > avg_vol * 2
    
    # Setup parfait: compression puis expansion
    if compression_phase and expansion:
        return 1.0
    elif compression_phase:
        return 0.6  # En compression, attendre expansion
    elif expansion:
        return 0.4  # Expansion sans compression préalable
    
    return 0


# ============================
# PM + RTH COMBINED PATTERNS
# ============================

def pm_rth_continuation_pattern(df, pm_data):
    """
    Pattern PM break + RTH continuation
    PM gap up + RTH retest PM high + continuation
    
    Requires:
    - df: full dataframe with PM + RTH data
    - pm_data: dict with pm_high, pm_low, gap_pct
    
    Returns: 0-1 score
    """
    if not pm_data or len(df) < 20:
        return 0
    
    pm_high = pm_data.get("pm_high", 0)
    gap_pct = pm_data.get("gap_pct", 0)
    
    if pm_high <= 0 or gap_pct <= 0:
        return 0
    
    # RTH data (après 9:30 AM)
    # Pour simplifier, on utilise les dernières candles
    rth_candles = df.iloc[-15:]  # ~15 min RTH
    
    # Check si retest PM high
    lows = rth_candles["low"]
    highs = rth_candles["high"]
    
    # Touché PM high
    touched_pm_high = any((lows <= pm_high) & (highs >= pm_high))
    
    if not touched_pm_high:
        return 0
    
    # Continuation après retest
    current_price = df["close"].iloc[-1]
    
    if current_price > pm_high * 1.02:  # +2% au-dessus PM high
        return 0.9
    elif current_price > pm_high:
        return 0.6
    
    return 0.3  # Retest mais pas encore continuation


# ============================
# MASTER PATTERN SCORE
# ============================

def compute_pattern_score(ticker, df, pm_data=None):
    """
    Calcule score pattern global (0-1) - SIMPLIFIED V2
    Garde seulement les patterns les plus performants
    
    Removed:
    - bollinger_squeeze (corrélé avec tight_consolidation)
    - momentum_acceleration (bruit sur 1min candles)
    - volume_accumulation (merge dans volume_profile_bullish)
    - flag_pennant (trop rare en PM/early RTH)
    
    Returns: dict avec score total et détails
    """
    if df is None or len(df) < 20:
        return {"pattern_score": 0, "details": {}}
    
    # PATTERNS SIMPLIFIÉS (seulement les meilleurs)
    patterns = {
        # Volume Profile (35%) - INCREASED
        "volume_climax": volume_climax(df) * 0.20,         # ↑ Most important
        "volume_profile_bullish": volume_profile_bullish(df) * 0.15,
        
        # Structure (35%) - INCREASED
        "higher_lows": higher_lows_pattern(df) * 0.15,
        "tight_consolidation": tight_consolidation(df) * 0.20,  # ↑ Key pattern
        
        # Volume Squeeze (30%)
        "volume_squeeze": volume_squeeze_score(df) * 0.30,  # ↑ INCREASED
        
        # PM + RTH (if PM data available)
        "pm_rth_continuation": 0  # Will be added below if PM data exists
    }
    
    # Add PM pattern if data available
    if pm_data:
        pm_score = pm_rth_continuation_pattern(df, pm_data)
        # Replace volume_squeeze weight partially
        patterns["volume_squeeze"] = volume_squeeze_score(df) * 0.15  # Reduced
        patterns["pm_rth_continuation"] = pm_score * 0.35  # MAJOR weight
    
    # Score total
    total_score = sum(patterns.values())
    
    # Clamp 0-1
    total_score = max(0, min(1, total_score))
    
    return {
        "pattern_score": total_score,
        "details": patterns
    }


# ============================
# STRUCTURE STRENGTH SCORE
# ============================

def compute_structure_strength(df):
    """
    Score de qualité de la structure (0-1)
    Critères:
    - Higher lows
    - Consolidation tight
    - Volume profile bullish
    - Support/resistance clear
    - Momentum acceleration
    
    Returns: 0-1 score
    """
    if df is None or len(df) < 20:
        return 0
    
    components = {
        "higher_lows": higher_lows_pattern(df),
        "tight_range": tight_consolidation(df),
        "volume_bullish": volume_profile_bullish(df),
        "momentum_accel": momentum_acceleration(df),
        "squeeze": bollinger_squeeze(df)
    }
    
    # Moyenne pondérée
    weights = {
        "higher_lows": 0.25,
        "tight_range": 0.20,
        "volume_bullish": 0.20,
        "momentum_accel": 0.20,
        "squeeze": 0.15
    }
    
    score = sum(components[k] * weights[k] for k in components)
    
    return max(0, min(1, score))


# ============================
# BATCH HELPER
# ============================

def analyze_patterns_batch(tickers_data, pm_data_dict=None):
    """
    Analyse patterns pour plusieurs tickers
    
    Args:
        tickers_data: dict {ticker: dataframe}
        pm_data_dict: dict {ticker: pm_data}
    
    Returns: dict {ticker: pattern_score_data}
    """
    results = {}
    
    for ticker, df in tickers_data.items():
        pm_data = pm_data_dict.get(ticker) if pm_data_dict else None
        
        pattern_data = compute_pattern_score(ticker, df, pm_data)
        results[ticker] = pattern_data
    
    logger.info(f"Analyzed patterns for {len(results)} tickers")
    
    return results


if __name__ == "__main__":
    # Test basique
    print("Pattern Analyzer module loaded")
    print("Available patterns:")
    print("  - Volume Accumulation")
    print("  - Volume Climax")
    print("  - Higher Lows")
    print("  - Tight Consolidation")
    print("  - Flag/Pennant")
    print("  - Bollinger Squeeze")
    print("  - Momentum Acceleration")
    print("  - PM+RTH Continuation")
