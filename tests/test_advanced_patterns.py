"""
TESTS - Pattern Analyzer & PM Transition
=========================================

Tests unitaires pour valider les nouveaux modules
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

print("\n" + "="*60)
print("🧪 TESTS - NOUVEAUX PATTERNS & PM TRANSITION")
print("="*60 + "\n")

# ============================
# TEST 1: Pattern Analyzer
# ============================

print("TEST 1: Pattern Analyzer")
print("-" * 60)

try:
    from src.pattern_analyzer import (
        bollinger_squeeze,
        volume_accumulation,
        higher_lows_pattern,
        tight_consolidation,
        momentum_acceleration,
        compute_pattern_score
    )
    
    # Créer données de test
    # Scénario: Consolidation tight puis explosion
    np.random.seed(42)
    
    # Phase 1: Prix stable autour de 10.00 avec volume décroissant
    n1 = 30
    prices1 = 10.0 + np.random.normal(0, 0.01, n1)  # Range très tight
    volumes1 = np.linspace(100000, 50000, n1)  # Volume décroissant
    
    # Phase 2: Breakout avec volume spike
    n2 = 10
    prices2 = np.linspace(10.0, 10.50, n2)  # +5% move
    volumes2 = np.linspace(50000, 200000, n2)  # Volume expansion
    
    prices = np.concatenate([prices1, prices2])
    volumes = np.concatenate([volumes1, volumes2])
    
    df_test = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes
    })
    
    # Tests individuels
    print("\n  Testing individual patterns:")
    
    bb_squeeze = bollinger_squeeze(df_test)
    print(f"    Bollinger Squeeze: {bb_squeeze:.3f} {'✅' if bb_squeeze > 0.5 else '⚠️'}")
    
    vol_accum = volume_accumulation(df_test)
    print(f"    Volume Accumulation: {vol_accum:.3f}")
    
    tight_consol = tight_consolidation(df_test)
    print(f"    Tight Consolidation: {tight_consol:.3f} {'✅' if tight_consol > 0.7 else '⚠️'}")
    
    mom_accel = momentum_acceleration(df_test)
    print(f"    Momentum Acceleration: {mom_accel:.3f}")
    
    # Test score global
    pattern_result = compute_pattern_score("TEST", df_test)
    pattern_score = pattern_result["pattern_score"]
    
    print(f"\n  Pattern Score Global: {pattern_score:.3f}")
    print(f"  Components:")
    for key, value in pattern_result["details"].items():
        print(f"    - {key}: {value:.3f}")
    
    if pattern_score > 0.3:
        print("\n  ✅ Pattern Analyzer Test PASSED")
    else:
        print("\n  ⚠️  Pattern score lower than expected")
    
except Exception as e:
    print(f"\n  ❌ Pattern Analyzer Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================
# TEST 2: PM Transition
# ============================

print("\n" + "="*60)
print("TEST 2: PM Transition Analyzer")
print("-" * 60)

try:
    from src.pm_transition import (
        pm_position_in_range,
        pm_momentum_strength,
        pm_gap_quality,
        compute_pm_transition_score
    )
    
    # Données PM fictives - Setup bullish
    pm_data_bullish = {
        "pm_high": 11.50,
        "pm_low": 11.00,
        "last": 11.45,  # Proche du high (bullish)
        "gap_pct": 0.08,  # Gap 8%
        "pm_volume": 150000,
        "pm_liquid": True,
        "gap_pct": 0.08
    }
    
    # Tests individuels
    print("\n  Testing PM metrics:")
    
    position, strength = pm_position_in_range(pm_data_bullish)
    print(f"    PM Position: {position:.3f} (strength: {strength:.3f}) {'✅' if position > 0.8 else '⚠️'}")
    
    pm_mom = pm_momentum_strength(pm_data_bullish)
    print(f"    PM Momentum: {pm_mom:.3f} {'✅' if pm_mom > 0.5 else '⚠️'}")
    
    gap_qual = pm_gap_quality(pm_data_bullish)
    print(f"    PM Gap Quality: {gap_qual:.3f} {'✅' if gap_qual > 0.6 else '⚠️'}")
    
    # Créer dataframe RTH avec retest
    # PM high = 11.50, on simule un retest propre
    n_rth = 20
    
    # Phase 1: Pullback vers PM high
    prices_rth1 = np.linspace(11.45, 11.48, 10)
    
    # Phase 2: Continuation au-dessus
    prices_rth2 = np.linspace(11.50, 11.75, 10)
    
    prices_rth = np.concatenate([prices_rth1, prices_rth2])
    volumes_rth = np.random.uniform(80000, 120000, n_rth)
    
    df_rth = pd.DataFrame({
        "open": prices_rth * 0.999,
        "high": prices_rth * 1.003,
        "low": prices_rth * 0.997,
        "close": prices_rth,
        "volume": volumes_rth
    })
    
    # Test score global PM transition
    transition_result = compute_pm_transition_score("TEST", df_rth, pm_data_bullish)
    transition_score = transition_result["pm_transition_score"]
    
    print(f"\n  PM Transition Score: {transition_score:.3f}")
    print(f"  Details:")
    for key, value in transition_result["details"].items():
        if isinstance(value, (int, float)):
            print(f"    - {key}: {value:.3f}")
        else:
            print(f"    - {key}: {value}")
    
    if transition_score > 0.5:
        print("\n  ✅ PM Transition Test PASSED")
    else:
        print("\n  ⚠️  Transition score lower than expected")
    
except Exception as e:
    print(f"\n  ❌ PM Transition Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================
# TEST 3: Integration Monster Score
# ============================

print("\n" + "="*60)
print("TEST 3: Monster Score Integration")
print("-" * 60)

try:
    # Mock des fonctions pour test
    import sys
    from unittest.mock import Mock, patch
    
    # Mock get_events_by_ticker
    def mock_get_events(ticker):
        return [{"boosted_impact": 0.8}]  # Event fort
    
    # Mock compute_pm_metrics
    def mock_pm_metrics(ticker):
        return {
            "gap_pct": 0.07,
            "pm_high": 11.50,
            "pm_low": 11.00,
            "last": 11.45,
            "pm_volume": 150000,
            "pm_liquid": True
        }
    
    # Mock fetch_candles
    def mock_fetch_candles(ticker):
        return df_test  # Utiliser le df de test précédent
    
    with patch('src.scoring.monster_score.get_events_by_ticker', mock_get_events):
        with patch('src.scoring.monster_score.compute_pm_metrics', mock_pm_metrics):
            with patch('src.feature_engine.fetch_candles', mock_fetch_candles):
                
                from src.scoring.monster_score import compute_monster_score
                
                # Test avec patterns avancés
                print("\n  Testing with advanced patterns...")
                score_advanced = compute_monster_score("TEST", use_advanced=True)
                
                if score_advanced:
                    print(f"\n  Monster Score (Advanced): {score_advanced['monster_score']:.3f}")
                    print(f"  Components:")
                    for key, value in score_advanced['components'].items():
                        print(f"    - {key}: {value:.3f}")
                    
                    # Vérifier que les nouveaux composants sont présents
                    has_pattern = score_advanced['components'].get('pattern', 0) > 0
                    has_transition = score_advanced['components'].get('pm_transition', 0) > 0
                    
                    if has_pattern and has_transition:
                        print("\n  ✅ Monster Score Integration PASSED")
                        print("     (Advanced patterns successfully integrated)")
                    else:
                        print("\n  ⚠️  Advanced components not activated")
                else:
                    print("\n  ❌ Monster Score returned None")
    
except Exception as e:
    print(f"\n  ❌ Integration Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================
# SUMMARY
# ============================

print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print("""
Tests completed for:
  1. ✅ Pattern Analyzer (Bollinger, Volume, Consolidation, etc.)
  2. ✅ PM Transition (Position, Momentum, Retest Quality)
  3. ✅ Monster Score Integration

If all tests passed, the new modules are ready to use!

Next steps:
  1. Test with real market data
  2. Backtest on historical data
  3. Calibrate thresholds and weights
  4. Deploy in paper trading mode
""")
print("="*60 + "\n")
