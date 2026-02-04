"""
PM TRANSITION ANALYZER - Timing Optimal PM→RTH
===============================================

Optimise le timing d'entrée lors de la transition PM→RTH:
- Position dans le range PM
- Qualité du retest PM high
- Force momentum PM
- Timing entrée RTH optimal

Objectif: Entrer au MEILLEUR moment, ni trop tôt (fakeout), ni trop tard (déjà parti)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("PM_TRANSITION")
cache = Cache(ttl=30)

# ============================
# PM POSITION ANALYSIS
# ============================

def pm_position_in_range(pm_data):
    """
    Calcule où se trouve le prix dans le range PM
    
    Position > 0.8 = près du high (bullish)
    Position < 0.2 = près du low (bearish)
    Position 0.4-0.6 = milieu de range (neutre)
    
    Args:
        pm_data: dict avec pm_high, pm_low, last (current price)
    
    Returns: 
        position (0-1)
        strength_score (0-1)
    """
    if not pm_data:
        return 0.5, 0
    
    pm_high = pm_data.get("pm_high", 0)
    pm_low = pm_data.get("pm_low", 0)
    current = pm_data.get("last", 0)
    
    if pm_high <= 0 or pm_low <= 0 or current <= 0:
        return 0.5, 0
    
    pm_range = pm_high - pm_low
    
    if pm_range <= 0:
        return 0.5, 0
    
    # Position relative (0 = low, 1 = high)
    position = (current - pm_low) / pm_range
    position = max(0, min(1, position))
    
    # Strength score selon position
    if position >= 0.8:
        # Très proche du high = bullish fort
        strength = 0.9
    elif position >= 0.6:
        # Au-dessus milieu = bullish
        strength = 0.7
    elif position >= 0.4:
        # Milieu de range = neutre
        strength = 0.4
    elif position >= 0.2:
        # En-dessous milieu = faible
        strength = 0.2
    else:
        # Près du low = très faible
        strength = 0.1
    
    return position, strength


def pm_momentum_strength(pm_data):
    """
    Mesure la force du momentum PM
    Combine: gap %, range %, volume
    
    Args:
        pm_data: dict avec gap_pct, pm_high, pm_low, pm_volume
    
    Returns: 0-1 score
    """
    if not pm_data:
        return 0
    
    gap = abs(pm_data.get("gap_pct", 0))
    pm_high = pm_data.get("pm_high", 0)
    pm_low = pm_data.get("pm_low", 0)
    volume = pm_data.get("pm_volume", 0)
    pm_liquid = pm_data.get("pm_liquid", False)
    
    # Calcul range %
    if pm_low > 0:
        range_pct = (pm_high - pm_low) / pm_low
    else:
        range_pct = 0
    
    score = 0
    
    # Gap fort (> 5%)
    if gap >= 0.10:
        score += 0.40  # Gap énorme
    elif gap >= 0.05:
        score += 0.30  # Gap fort
    elif gap >= 0.03:
        score += 0.20  # Gap moyen
    
    # Range PM (> 3%)
    if range_pct >= 0.05:
        score += 0.30  # Range large
    elif range_pct >= 0.03:
        score += 0.20  # Range moyen
    
    # Volume
    if pm_liquid:
        score += 0.30  # Volume suffisant
    elif volume > 0:
        score += 0.15  # Un peu de volume
    
    return min(1.0, score)


def pm_gap_quality(pm_data):
    """
    Évalue la qualité du gap PM
    - Gap propre sans noise
    - Gap soutenu (pas de retour immédiat)
    
    Returns: 0-1 score
    """
    if not pm_data:
        return 0
    
    gap_pct = pm_data.get("gap_pct", 0)
    pm_high = pm_data.get("pm_high", 0)
    pm_low = pm_data.get("pm_low", 0)
    current = pm_data.get("last", 0)
    
    if gap_pct <= 0:
        return 0  # Gap down ou pas de gap
    
    # Gap size score
    if gap_pct >= 0.10:
        gap_score = 1.0
    elif gap_pct >= 0.05:
        gap_score = 0.8
    elif gap_pct >= 0.03:
        gap_score = 0.6
    else:
        gap_score = 0.3
    
    # Gap hold score (prix maintenu près du high)
    if pm_high > 0 and current > 0:
        hold_ratio = current / pm_high
        
        if hold_ratio >= 0.95:
            hold_score = 1.0  # Maintenu au high
        elif hold_ratio >= 0.85:
            hold_score = 0.7
        elif hold_ratio >= 0.70:
            hold_score = 0.4
        else:
            hold_score = 0.2  # Retracé fort
    else:
        hold_score = 0.5
    
    # Score combiné
    quality = (gap_score * 0.6) + (hold_score * 0.4)
    
    return quality


# ============================
# RTH RETEST ANALYSIS
# ============================

def pm_retest_quality(df, pm_data, rth_start_idx=None):
    """
    Analyse la qualité du retest du PM high en RTH
    
    Retest propre = 
    - Touche PM high
    - Rebond immédiat
    - Volume faible sur retest
    - Continuation après
    
    Args:
        df: full dataframe (PM + RTH)
        pm_data: dict avec pm_high
        rth_start_idx: index de début RTH (si connu)
    
    Returns: 0-1 score
    """
    if df is None or not pm_data or len(df) < 10:
        return 0
    
    pm_high = pm_data.get("pm_high", 0)
    
    if pm_high <= 0:
        return 0
    
    # Si pas de RTH start idx, utiliser dernières 20 candles
    if rth_start_idx is None:
        rth_candles = df.iloc[-20:]
    else:
        rth_candles = df.iloc[rth_start_idx:]
    
    if len(rth_candles) < 5:
        return 0
    
    lows = rth_candles["low"].values
    highs = rth_candles["high"].values
    closes = rth_candles["close"].values
    volumes = rth_candles["volume"].values
    
    # Tolérance pour "touché PM high" (±0.5%)
    tolerance = pm_high * 0.005
    
    # Check si PM high a été touché
    touched_indices = []
    for i in range(len(rth_candles)):
        if lows[i] <= (pm_high + tolerance) and highs[i] >= (pm_high - tolerance):
            touched_indices.append(i)
    
    if not touched_indices:
        return 0  # Pas de retest
    
    # Analyser le premier retest
    first_touch = touched_indices[0]
    
    # Volume sur retest
    retest_vol = volumes[first_touch]
    avg_vol = volumes.mean()
    
    vol_ratio = retest_vol / avg_vol if avg_vol > 0 else 1
    
    # Retest propre = volume faible (< 0.7x moyenne)
    if vol_ratio < 0.7:
        vol_score = 1.0
    elif vol_ratio < 1.0:
        vol_score = 0.6
    else:
        vol_score = 0.3
    
    # Continuation après retest
    if first_touch < len(closes) - 1:
        continuation_price = closes[-1]
        retest_price = closes[first_touch]
        
        if continuation_price > pm_high * 1.02:
            continuation_score = 1.0  # +2% au-dessus
        elif continuation_price > pm_high:
            continuation_score = 0.7
        else:
            continuation_score = 0.3
    else:
        continuation_score = 0.5
    
    # Score combiné
    quality = (vol_score * 0.4) + (continuation_score * 0.6)
    
    return quality


def rth_momentum_confirmation(df, pm_data, lookback=10):
    """
    Vérifie confirmation momentum en RTH
    - Prix > PM high
    - Volume soutenu
    - Momentum positif
    
    Returns: 0-1 score
    """
    if df is None or not pm_data or len(df) < lookback:
        return 0
    
    pm_high = pm_data.get("pm_high", 0)
    
    if pm_high <= 0:
        return 0
    
    recent = df.iloc[-lookback:]
    
    current_price = recent["close"].iloc[-1]
    current_vol = recent["volume"].iloc[-1]
    avg_vol = recent["volume"].mean()
    
    # Prix au-dessus PM high
    if current_price > pm_high * 1.05:
        price_score = 1.0  # +5% au-dessus
    elif current_price > pm_high * 1.02:
        price_score = 0.8
    elif current_price > pm_high:
        price_score = 0.6
    else:
        price_score = 0.2
    
    # Volume soutenu
    if avg_vol > 0:
        vol_ratio = current_vol / avg_vol
        
        if vol_ratio >= 1.5:
            vol_score = 1.0
        elif vol_ratio >= 1.0:
            vol_score = 0.7
        else:
            vol_score = 0.4
    else:
        vol_score = 0.5
    
    # Momentum (simple: prix monte)
    first_price = recent["close"].iloc[0]
    if first_price > 0:
        mom = (current_price - first_price) / first_price
        
        if mom >= 0.03:
            mom_score = 1.0
        elif mom >= 0.01:
            mom_score = 0.7
        elif mom > 0:
            mom_score = 0.4
        else:
            mom_score = 0.1
    else:
        mom_score = 0.5
    
    # Score combiné
    confirmation = (price_score * 0.4) + (vol_score * 0.3) + (mom_score * 0.3)
    
    return confirmation


# ============================
# FAKEOUT DETECTION
# ============================

def detect_pm_fakeout(df, pm_data, min_hold_candles=5):
    """
    Détecte fakeout PM (break puis retour rapide)
    
    Fakeout = 
    - Break PM high
    - Retour sous PM high en < 5 candles
    - Volume faible sur break
    
    Returns: 
        is_fakeout (bool)
        confidence (0-1)
    """
    if df is None or not pm_data or len(df) < min_hold_candles + 5:
        return False, 0
    
    pm_high = pm_data.get("pm_high", 0)
    
    if pm_high <= 0:
        return False, 0
    
    recent = df.iloc[-min_hold_candles - 5:]
    
    # Trouver si break PM high
    breaks = recent["high"] > pm_high
    
    if not breaks.any():
        return False, 0  # Pas de break
    
    # Index du premier break
    first_break_idx = breaks.idxmax()
    
    # Candles après le break
    after_break = df.loc[first_break_idx:]
    
    if len(after_break) < min_hold_candles:
        return False, 0  # Pas assez de données
    
    # Check si retour sous PM high rapidement
    closes_after = after_break["close"].values[:min_hold_candles]
    
    failed_hold = any(closes_after < pm_high * 0.98)  # -2% sous PM high
    
    if failed_hold:
        # Volume sur break
        break_vol = after_break["volume"].iloc[0]
        avg_vol = recent["volume"].mean()
        
        vol_ratio = break_vol / avg_vol if avg_vol > 0 else 1
        
        # Fakeout plus probable si volume faible
        if vol_ratio < 0.8:
            confidence = 0.9
        elif vol_ratio < 1.2:
            confidence = 0.7
        else:
            confidence = 0.4
        
        return True, confidence
    
    return False, 0


# ============================
# ENTRY TIMING SCORE
# ============================

def compute_entry_timing_score(df, pm_data):
    """
    Score timing optimal d'entrée (0-1)
    
    Combine:
    - PM setup quality (0-0.3)
    - RTH confirmation (0-0.4)
    - Momentum acceleration (0-0.3)
    
    Returns: 0-1 score
    """
    if df is None or not pm_data:
        return 0
    
    # 1. PM Setup Quality (30%)
    pm_quality = pm_gap_quality(pm_data)
    pm_momentum = pm_momentum_strength(pm_data)
    pm_position, pm_strength = pm_position_in_range(pm_data)
    
    pm_setup_score = (pm_quality * 0.4 + pm_momentum * 0.3 + pm_strength * 0.3) * 0.3
    
    # 2. RTH Confirmation (40%)
    retest = pm_retest_quality(df, pm_data)
    momentum_conf = rth_momentum_confirmation(df, pm_data)
    
    # Détecter fakeout
    is_fakeout, fakeout_conf = detect_pm_fakeout(df, pm_data)
    
    if is_fakeout:
        # Pénaliser si fakeout détecté
        rth_conf_score = momentum_conf * (1 - fakeout_conf * 0.5) * 0.4
    else:
        rth_conf_score = ((retest * 0.5) + (momentum_conf * 0.5)) * 0.4
    
    # 3. Momentum Acceleration (30%)
    # Utiliser simple momentum sur dernières candles
    if len(df) >= 10:
        recent = df.iloc[-10:]
        first = recent["close"].iloc[0]
        last = recent["close"].iloc[-1]
        
        if first > 0:
            mom_accel = (last - first) / first
            
            if mom_accel >= 0.05:
                accel_score = 1.0
            elif mom_accel >= 0.03:
                accel_score = 0.8
            elif mom_accel >= 0.01:
                accel_score = 0.5
            elif mom_accel > 0:
                accel_score = 0.3
            else:
                accel_score = 0.1
        else:
            accel_score = 0.5
    else:
        accel_score = 0.5
    
    accel_score *= 0.3
    
    # Score total
    total_score = pm_setup_score + rth_conf_score + accel_score
    
    return max(0, min(1, total_score))


# ============================
# PM TRANSITION MASTER SCORE
# ============================

def compute_pm_transition_score(ticker, df, pm_data):
    """
    Score global de transition PM→RTH (0-1)
    
    Returns: dict avec score et détails
    """
    if df is None or not pm_data:
        return {"pm_transition_score": 0, "details": {}}
    
    # Composants
    pm_position, pm_strength = pm_position_in_range(pm_data)
    pm_mom = pm_momentum_strength(pm_data)
    pm_gap_qual = pm_gap_quality(pm_data)
    
    retest_qual = pm_retest_quality(df, pm_data)
    rth_conf = rth_momentum_confirmation(df, pm_data)
    
    is_fakeout, fakeout_conf = detect_pm_fakeout(df, pm_data)
    
    entry_timing = compute_entry_timing_score(df, pm_data)
    
    details = {
        "pm_position": pm_position,
        "pm_strength": pm_strength,
        "pm_momentum": pm_mom,
        "pm_gap_quality": pm_gap_qual,
        "retest_quality": retest_qual,
        "rth_confirmation": rth_conf,
        "is_fakeout": is_fakeout,
        "fakeout_confidence": fakeout_conf if is_fakeout else 0,
        "entry_timing": entry_timing
    }
    
    # Score global = entry timing (déjà combine tout)
    total_score = entry_timing
    
    return {
        "pm_transition_score": total_score,
        "details": details
    }


if __name__ == "__main__":
    print("PM Transition Analyzer module loaded")
    print("Features:")
    print("  - PM Position in Range")
    print("  - PM Momentum Strength")
    print("  - PM Gap Quality")
    print("  - RTH Retest Analysis")
    print("  - Fakeout Detection")
    print("  - Entry Timing Optimization")
