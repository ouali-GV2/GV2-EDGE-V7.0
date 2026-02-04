"""
OPTIONS FLOW MONITOR (IBKR)
===========================

Detect unusual options activity via IBKR:
- High volume compared to open interest
- Large block trades
- Unusual call/put ratio
- Aggressive bid/ask (sweeps)

Indicators of smart money positioning before moves.

Data source: IBKR market data (Level 1 sufficient)
"""

from datetime import datetime, timedelta
import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("OPTIONS_FLOW")

cache = Cache(ttl=300)  # 5min cache


# ============================
# IBKR Options Data
# ============================

def get_options_data_ibkr(ticker):
    """
    Get options data from IBKR
    
    IBKR provides:
    - Options chains
    - Volume by strike/expiry
    - Open interest
    - Bid/ask
    
    Args:
        ticker: Stock symbol
    
    Returns:
        dict with options data
    """
    try:
        from src.ibkr_connector import get_ibkr
        
        ibkr = get_ibkr()
        
        if not ibkr or not ibkr.connected:
            logger.warning("IBKR not connected, cannot fetch options")
            return None
        
        from ib_insync import Stock, Option
        
        # Get stock contract
        stock = Stock(ticker, 'SMART', 'USD')
        
        # Request options chains
        # Get next 2 monthly expirations
        chains = ibkr.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        
        if not chains:
            logger.warning(f"No options chains for {ticker}")
            return None
        
        # Take first chain (usually most liquid)
        chain = chains[0]
        
        # Get expirations (next 30 days)
        today = datetime.now().date()
        nearby_expirations = [
            exp for exp in chain.expirations 
            if datetime.strptime(exp, "%Y%m%d").date() <= today + timedelta(days=30)
        ][:2]  # Next 2 expirations
        
        if not nearby_expirations:
            return None
        
        options_data = []
        
        for expiration in nearby_expirations:
            # Get strikes near current price (ATM options)
            # Request market data for a few strikes
            
            # This is simplified - full implementation would:
            # 1. Get current stock price
            # 2. Select strikes within 10% of current price
            # 3. Request market data for each option
            # 4. Analyze volume vs open interest
            
            # For now, return basic structure
            options_data.append({
                "expiration": expiration,
                "strikes_analyzed": 0,
                "total_call_volume": 0,
                "total_put_volume": 0,
                "call_put_ratio": 0
            })
        
        return {
            "ticker": ticker,
            "expirations": nearby_expirations,
            "data": options_data
        }
    
    except Exception as e:
        logger.error(f"Options data fetch failed for {ticker}: {e}")
        return None


# ============================
# Unusual Options Activity Detection
# ============================

def detect_unusual_options(ticker):
    """
    Detect unusual options activity
    
    Signals:
    - Volume > 2x open interest (unusual activity)
    - Call/Put ratio >> 1 (bullish positioning)
    - Large blocks (>100 contracts single trade)
    
    Args:
        ticker: Stock symbol
    
    Returns:
        dict with unusual activity score
    """
    cache_key = f"options_unusual_{ticker}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    # Get options data from IBKR
    options_data = get_options_data_ibkr(ticker)
    
    if not options_data:
        return {
            "ticker": ticker,
            "unusual_activity": False,
            "score": 0,
            "reason": "No options data available"
        }
    
    # Analyze for unusual activity
    # This is a simplified version - full implementation would:
    # 1. Compare current volume to 20-day avg volume
    # 2. Check volume vs open interest ratio
    # 3. Detect large block trades
    # 4. Analyze call/put skew
    
    # For demo purposes:
    result = {
        "ticker": ticker,
        "unusual_activity": False,
        "score": 0,
        "call_put_ratio": 0,
        "reason": "Analysis in progress"
    }
    
    cache.set(cache_key, result)
    
    return result


# ============================
# Simplified Options Flow Score
# ============================

def calculate_options_flow_score(ticker):
    """
    Calculate options flow score (0-1)
    
    Note: This is a simplified implementation.
    Full implementation requires:
    - Real-time options data subscription
    - Historical options volume database
    - Block trade detection
    
    For now, returns neutral score unless unusual activity detected.
    
    Returns:
        float: 0-1 score
    """
    unusual = detect_unusual_options(ticker)
    
    if unusual and unusual.get("unusual_activity"):
        return min(1.0, unusual.get("score", 0))
    
    return 0.5  # Neutral (no signal)


# ============================
# Public API
# ============================

def get_options_signal(ticker):
    """
    Get options flow signal for ticker
    
    Returns:
        dict with signal info
    """
    score = calculate_options_flow_score(ticker)
    unusual = detect_unusual_options(ticker)
    
    return {
        "ticker": ticker,
        "options_score": score,
        "unusual_activity": unusual.get("unusual_activity", False),
        "reason": unusual.get("reason", ""),
        "impact": "bullish" if score > 0.6 else "neutral" if score >= 0.4 else "bearish"
    }


# ============================
# NOTE: Options Data Complexity
# ============================

"""
IMPORTANT NOTE:

Full options flow analysis requires:

1. Real-time options data subscription (additional cost on IBKR)
2. Historical options volume database
3. Open interest tracking
4. Block trade detection algorithms
5. Greeks calculation (delta, gamma, etc.)

Current implementation is SIMPLIFIED and provides:
- Basic framework
- IBKR integration structure
- Placeholder for future enhancement

For production use, consider:
- Subscribing to CBOE options data
- Using services like Unusual Whales (paid)
- Building historical options database

For GV2-EDGE purposes, options flow is OPTIONAL and can provide:
+5-10% additional edge when implemented fully.

For now, this module returns neutral scores to avoid false signals.
"""


if __name__ == "__main__":
    print("\n📊 OPTIONS FLOW MONITOR TEST")
    print("=" * 60)
    
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    
    for ticker in test_tickers:
        signal = get_options_signal(ticker)
        
        print(f"\n{ticker}:")
        print(f"  Score: {signal['options_score']:.2f}")
        print(f"  Impact: {signal['impact']}")
        print(f"  Unusual: {signal['unusual_activity']}")
        print(f"  Reason: {signal['reason']}")
