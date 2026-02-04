"""
DARK POOL ALTERNATIVES - Indirect Detection Methods
====================================================

⚠️ AVERTISSEMENT HONNÊTE:
Pour les SMALL CAPS US <$2B, les données dark pool ont une utilité LIMITÉE:

1. La majorité du volume small cap est sur les exchanges lit (NYSE, NASDAQ)
2. Les données dark pool sont DÉLAYÉES (fin de journée ou J+1)
3. L'interprétation est ambiguë (achat ou vente?)
4. Peut ajouter du BRUIT plutôt que du signal

RECOMMANDATION:
- Pour DÉTECTION AVANT SPIKE: PAS UTILE (trop de délai)
- Pour CONFIRMATION POST-SPIKE: Peut être utile
- Pour SMALL CAPS: Faible signal/bruit

Ce module est fourni pour complétion, mais désactivé par défaut.
Utilise plutôt: News Flow + Options Flow + Extended Hours (plus efficaces).

Sources gratuites disponibles:
- FINRA ATS Data (delayed, T+1)
- Short Volume Data (FINRA, daily)
- Large Block Detection (inférence via tick data)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get
from config import FINNHUB_API_KEY

logger = get_logger("DARK_POOL")

# ⚠️ DÉSACTIVÉ PAR DÉFAUT - Voir explication ci-dessus
ENABLE_DARK_POOL = False


@dataclass
class DarkPoolSignal:
    """Dark pool activity signal"""
    ticker: str
    signal_type: str
    score: float
    details: Dict
    timestamp: str
    reliability: str  # 'LOW', 'MEDIUM', 'HIGH'


# ============================
# FINRA SHORT VOLUME DATA (Gratuit)
# ============================

def get_short_volume_ratio(ticker: str) -> Tuple[float, Dict]:
    """
    Get short volume ratio from FINRA data
    
    Source: FINRA publishes daily short volume data
    
    ⚠️ Limitations:
    - Delayed (T+1)
    - Short volume ≠ short interest
    - Can be market making, not directional shorts
    
    Returns: (ratio, details)
    """
    # Note: FINRA data requires scraping their website
    # or using a paid service. Using Finnhub as proxy.
    
    try:
        url = "https://finnhub.io/api/v1/stock/short-interest"
        params = {
            "symbol": ticker,
            "token": FINNHUB_API_KEY
        }
        
        r = safe_get(url, params=params, timeout=10)
        data = r.json()
        
        if not data:
            return 0.0, {}
        
        # Get most recent entry
        recent = data[0] if isinstance(data, list) and data else {}
        
        short_interest = recent.get('shortInterest', 0)
        avg_volume = recent.get('avgDailyTradingVolume', 1)
        
        # Days to cover
        days_to_cover = short_interest / avg_volume if avg_volume > 0 else 0
        
        details = {
            'short_interest': short_interest,
            'avg_volume': avg_volume,
            'days_to_cover': round(days_to_cover, 2),
            'date': recent.get('settleDate', ''),
            'reliability': 'LOW'  # Delayed data
        }
        
        # Normalize to 0-1 score
        # High days-to-cover could indicate squeeze potential
        score = min(1.0, days_to_cover / 10) if days_to_cover > 2 else 0
        
        return score, details
        
    except Exception as e:
        logger.debug(f"Short volume error {ticker}: {e}")
        return 0.0, {}


# ============================
# LARGE BLOCK INFERENCE (Via Price/Volume Analysis)
# ============================

def detect_large_block_activity(ticker: str) -> Tuple[float, Dict]:
    """
    Infer large block activity from price/volume patterns
    
    Indicators of hidden large orders:
    1. Volume >> normal without corresponding price move
    2. Price grinding up/down steadily (accumulation/distribution)
    3. VWAP deviation patterns
    4. Time of day patterns (institutions trade at specific times)
    
    ⚠️ This is INFERENCE, not actual dark pool data
    """
    try:
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        
        if not ibkr or not ibkr.connected:
            return 0.0, {}
        
        # Get intraday bars
        bars = ibkr.get_bars(
            ticker,
            duration='1 D',
            bar_size='5 mins',
            use_rth=False
        )
        
        if bars is None or len(bars) < 20:
            return 0.0, {}
        
        # Calculate metrics
        volumes = bars['volume'].values
        closes = bars['close'].values
        
        avg_volume = volumes.mean()
        volume_std = volumes.std()
        
        # 1. Volume anomaly: periods with high volume but low price change
        anomaly_count = 0
        for i in range(1, len(bars)):
            vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 0
            price_change = abs(closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
            
            # High volume + low price move = potential hidden activity
            if vol_ratio > 2.0 and price_change < 0.005:
                anomaly_count += 1
        
        # 2. Accumulation pattern: steady price drift with elevated volume
        price_drift = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        avg_recent_vol = volumes[-10:].mean()
        vol_increase = avg_recent_vol / avg_volume if avg_volume > 0 else 1
        
        # Score calculation
        score = 0.0
        
        if anomaly_count >= 3:
            score += 0.3
        
        if vol_increase > 1.5 and abs(price_drift) < 0.02:
            score += 0.3  # High volume, little price change = absorption
        
        if vol_increase > 1.5 and price_drift > 0.02:
            score += 0.2  # Accumulation with drift
        
        details = {
            'anomaly_periods': anomaly_count,
            'volume_increase': round(vol_increase, 2),
            'price_drift': round(price_drift * 100, 2),
            'interpretation': 'Potential hidden accumulation' if score > 0.3 else 'Normal activity',
            'reliability': 'LOW'  # This is inference
        }
        
        return min(1.0, score), details
        
    except Exception as e:
        logger.debug(f"Block detection error {ticker}: {e}")
        return 0.0, {}


# ============================
# COMBINED DARK POOL SCORE
# ============================

def get_dark_pool_score(ticker: str) -> Tuple[float, Dict]:
    """
    Get combined dark pool score
    
    ⚠️ Use with caution for small caps - may add noise
    
    Returns: (score, details)
    """
    if not ENABLE_DARK_POOL:
        return 0.0, {'status': 'disabled', 'reason': 'Low utility for small caps'}
    
    scores = {}
    details = {}
    
    # Short volume
    short_score, short_details = get_short_volume_ratio(ticker)
    scores['short_volume'] = short_score
    details['short'] = short_details
    
    # Block inference
    block_score, block_details = detect_large_block_activity(ticker)
    scores['block_activity'] = block_score
    details['block'] = block_details
    
    # Combined (weighted average)
    total_score = (
        scores.get('short_volume', 0) * 0.4 +
        scores.get('block_activity', 0) * 0.6
    )
    
    details['combined_score'] = round(total_score, 3)
    details['reliability'] = 'LOW'  # Always low for dark pool inference
    
    return total_score, details


# ============================
# RECOMMENDATION ENGINE
# ============================

def should_use_dark_pool_for_ticker(ticker: str, market_cap: float) -> Tuple[bool, str]:
    """
    Determine if dark pool analysis is useful for this ticker
    
    Args:
        ticker: Stock symbol
        market_cap: Market cap in dollars
    
    Returns: (should_use, reason)
    """
    # Small caps: dark pool generally not useful
    if market_cap < 2_000_000_000:  # < $2B
        return False, "Small cap - dark pool data adds noise, not signal"
    
    # Mid caps: marginal utility
    if market_cap < 10_000_000_000:  # < $10B
        return False, "Mid cap - dark pool data has limited predictive value"
    
    # Large caps: dark pool can be useful
    return True, "Large cap - dark pool data may provide additional signal"


# ============================
# HONEST ASSESSMENT
# ============================

def get_dark_pool_assessment() -> Dict:
    """
    Provide honest assessment of dark pool utility for GV2-EDGE
    """
    return {
        'recommendation': 'DISABLED',
        'reason': 'Pour small caps US <$2B, les données dark pool:',
        'issues': [
            '1. Sont DÉLAYÉES (fin de journée ou J+1) - inutile pour détection précoce',
            '2. Ont une interprétation AMBIGUË (achat ou vente?)',
            '3. Représentent une FAIBLE part du volume small cap',
            '4. Ajoutent du BRUIT plutôt que du signal',
            '5. Coûtent cher pour du temps réel (FlowAlgo: ~$100/mois)'
        ],
        'better_alternatives': [
            '✅ News Flow Screener (NOUVEAU) - Détection temps réel',
            '✅ Options Flow via IBKR OPRA - Signal smart money',
            '✅ Extended Hours Quotes - Gaps forming en temps réel',
            '✅ Grok + Polygon - News ticker-specific instant'
        ],
        'when_useful': 'Dark pool peut être utile APRÈS avoir détecté un mover pour confirmer accumulation institutionnelle, mais pas pour la détection initiale.'
    }


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("DARK POOL ALTERNATIVES - ASSESSMENT")
    print("=" * 60)
    
    assessment = get_dark_pool_assessment()
    
    print(f"\n📊 Recommendation: {assessment['recommendation']}")
    print(f"\n{assessment['reason']}")
    
    for issue in assessment['issues']:
        print(f"  {issue}")
    
    print(f"\n✅ Better alternatives for GV2-EDGE:")
    for alt in assessment['better_alternatives']:
        print(f"  {alt}")
    
    print(f"\n💡 {assessment['when_useful']}")
    
    # Test with a ticker
    print("\n" + "=" * 60)
    print("TESTING (with ENABLE_DARK_POOL = False)")
    
    score, details = get_dark_pool_score("NVDA")
    print(f"\nNVDA dark pool score: {score}")
    print(f"Details: {details}")
