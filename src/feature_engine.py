import pandas as pd
from datetime import datetime
from utils.cache import Cache
from utils.logger import get_logger
from utils.data_validator import validate_features
from utils.api_guard import safe_get
from config import FINNHUB_API_KEY, USE_IBKR_DATA

logger = get_logger("FEATURE_ENGINE")
cache = Cache(ttl=60)

FINNHUB_CANDLE = "https://finnhub.io/api/v1/stock/candle"

# IBKR connector (if enabled)
ibkr_connector = None

if USE_IBKR_DATA:
    try:
        from src.ibkr_connector import get_ibkr
        ibkr_connector = get_ibkr()
        if ibkr_connector.connected:
            logger.info("✅ IBKR connector active for market data")
        else:
            logger.warning("⚠️ IBKR not connected, falling back to Finnhub")
            ibkr_connector = None
    except Exception as e:
        logger.warning(f"⚠️ IBKR connector unavailable: {e}, using Finnhub")
        ibkr_connector = None

# ============================
# Helpers
# ============================

def clamp(x, low, high):
    return max(low, min(high, x))

def normalize_ratio(x, max_val):
    return clamp(x / max_val, 0, 1)

# ============================
# Price fetch (IBKR or Finnhub)
# ============================

def fetch_candles(ticker, resolution="1", lookback=120):
    """
    Fetch historical candles from IBKR (if available) or Finnhub
    
    Args:
        ticker: Stock symbol
        resolution: Candle size ("1" = 1 min)
        lookback: Minutes to look back
    
    Returns:
        DataFrame with OHLCV data
    """
    # Try IBKR first (if enabled and connected)
    if ibkr_connector and ibkr_connector.connected:
        try:
            # Convert lookback to IBKR duration
            if lookback <= 120:
                duration = '2 D'  # 2 days to ensure coverage
            elif lookback <= 1440:
                duration = '1 W'
            else:
                duration = '1 M'
            
            # Convert resolution to bar size
            bar_size = '1 min' if resolution == "1" else f"{resolution} mins"
            
            df = ibkr_connector.get_bars(ticker, duration=duration, bar_size=bar_size)
            
            if df is not None and not df.empty:
                # Take only last 'lookback' bars
                df = df.tail(lookback)
                
                # Ensure required columns
                df = df.rename(columns={'date': 'timestamp'})
                
                logger.debug(f"✅ IBKR: Fetched {len(df)} bars for {ticker}")
                return df
            else:
                logger.debug(f"⚠️ IBKR returned no data for {ticker}, trying Finnhub")
        
        except Exception as e:
            logger.debug(f"⚠️ IBKR fetch failed for {ticker}: {e}, trying Finnhub")
    
    # Fallback to Finnhub
    try:
        now = int(datetime.utcnow().timestamp())
        start = now - lookback * 60

        params = {
            "symbol": ticker,
            "resolution": resolution,
            "from": start,
            "to": now,
            "token": FINNHUB_API_KEY
        }

        r = safe_get(FINNHUB_CANDLE, params=params)
        data = r.json()

        if data.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })
        
        logger.debug(f"✅ Finnhub: Fetched {len(df)} bars for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"❌ Both IBKR and Finnhub failed for {ticker}: {e}")
        return None

# ============================
# Indicators (raw)
# ============================

def compute_vwap(df):
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum()
    return pv / vol

def raw_momentum(df, n=5):
    base = df["close"].iloc[-n]
    if base <= 0:
        return 0
    return (df["close"].iloc[-1] - base) / base

def raw_volume_spike(df):
    recent = df["volume"].iloc[-1]
    avg = df["volume"].iloc[:-1].mean()
    if avg <= 0:
        return 0
    return recent / avg

def raw_vwap_dev(df):
    vwap = compute_vwap(df).iloc[-1]
    if vwap <= 0:
        return 0
    return (df["close"].iloc[-1] - vwap) / vwap

def raw_volatility(df):
    return df["close"].pct_change().std()

# ============================
# Normalized features
# ============================

def momentum(df):
    raw = raw_momentum(df)
    return normalize_ratio(abs(raw), 0.15)   # 15% move = strong

def volume_spike(df):
    raw = raw_volume_spike(df)
    return normalize_ratio(raw, 8)            # cap at 8x avg volume

def vwap_deviation(df):
    raw = raw_vwap_dev(df)
    return clamp(raw, -0.05, 0.05) / 0.05     # normalize -1 → 1

def volatility(df):
    raw = raw_volatility(df)
    return normalize_ratio(raw, 0.05)         # 5% vol = high

# ============================
# Squeeze proxy (normalized)
# ============================

def squeeze_proxy(df):
    mom = abs(raw_momentum(df))
    vol = raw_volatility(df)
    if vol <= 0:
        return 0
    raw = mom / vol
    return normalize_ratio(raw, 10)           # cap extreme squeezes

# ============================
# Pattern flags
# ============================

def breakout_high(df, window=20):
    if len(df) < window + 1:
        return 0
    high = df["high"].iloc[-window:-1].max()
    return float(df["close"].iloc[-1] > high)

def strong_green(df):
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    if o <= 0:
        return 0
    return float(c > o * 1.015)   # 1.5% candle

# ============================
# Feature builder
# ============================

def compute_features(ticker, include_advanced=True):
    """
    Compute features with optional advanced patterns
    
    Args:
        ticker: stock ticker
        include_advanced: if True, include pattern analyzer features
    
    Returns: dict of features
    """
    cached = cache.get(f"feat_{ticker}")
    if cached:
        return cached

    try:
        df = fetch_candles(ticker)

        # C9 FIX: Reduced from 20 to 5 bars minimum
        # 20 bars was too restrictive — many small-cap tickers have limited
        # intraday data especially in pre-market. 5 bars is enough for
        # momentum, volume spike, and VWAP calculations.
        if df is None or len(df) < 5:
            return None

        feats = {
            "momentum": momentum(df),
            "volume_spike": volume_spike(df),
            "vwap_dev": vwap_deviation(df),
            "volatility": volatility(df),
            "squeeze_proxy": squeeze_proxy(df),
            "breakout": breakout_high(df),
            "strong_green": strong_green(df),
        }
        
        # Add advanced pattern features if requested
        if include_advanced:
            try:
                from src.pattern_analyzer import (
                    bollinger_squeeze,
                    volume_accumulation,
                    higher_lows_pattern,
                    tight_consolidation,
                    momentum_acceleration
                )
                
                feats["bollinger_squeeze"] = bollinger_squeeze(df)
                feats["volume_accumulation"] = volume_accumulation(df)
                feats["higher_lows"] = higher_lows_pattern(df)
                feats["tight_consolidation"] = tight_consolidation(df)
                feats["momentum_accel"] = momentum_acceleration(df)
                
            except Exception as e:
                logger.warning(f"Advanced patterns error for {ticker}: {e}")
                # Continue without advanced features

        if not validate_features(feats):
            return None

        cache.set(f"feat_{ticker}", feats)
        return feats

    except Exception as e:
        logger.error(f"Feature error {ticker}: {e}")
        return None

# ============================
# Batch helper
# ============================

def compute_many(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        f = compute_features(t)
        if f:
            results[t] = f

    logger.info(f"Computed features for {len(results)} tickers")
    return results


if __name__ == "__main__":
    print(compute_features("AAPL"))
