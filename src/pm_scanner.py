"""
PM SCANNER V8 - Pre-Market Scanner with Correct Gap Calculation
================================================================

V8 FIXES (from REVIEW):
- P1 CRITICAL: Fixed gap_pct to use prev_close (overnight gap) instead of pm_open (intra-PM momentum)
- Added gap classification: NEGLIGIBLE / EXPLOITABLE / EXTENDED / OVEREXTENDED
- Sweet spot prioritization for 3-8% gaps (highest probability of continuation)
- Separated intra-PM momentum from overnight gap for cleaner signals
"""

from utils.logger import get_logger
from utils.api_guard import safe_get
from utils.cache import Cache
from utils.time_utils import is_premarket

from config import FINNHUB_API_KEY, PM_MIN_VOLUME, USE_IBKR_DATA

from datetime import datetime

logger = get_logger("PM_SCANNER")

cache = Cache(ttl=30)

# V8: Gap classification zones (sweet spot = EXPLOITABLE)
GAP_ZONES = {
    "NEGLIGIBLE": (0.0, 0.03),      # <3% - noise, ignore
    "EXPLOITABLE": (0.03, 0.08),     # 3-8% - sweet spot, highest continuation probability
    "EXTENDED": (0.08, 0.15),        # 8-15% - tradeable but fading risk
    "OVEREXTENDED": (0.15, float("inf")),  # >15% - high fade risk, reduce size
}

# V8: Gap quality scores (higher = better risk/reward for continuation plays)
GAP_QUALITY_SCORES = {
    "NEGLIGIBLE": 0.1,
    "EXPLOITABLE": 1.0,    # Best zone
    "EXTENDED": 0.6,
    "OVEREXTENDED": 0.25,
}


def classify_gap(gap_pct_abs: float) -> str:
    """Classify gap magnitude into quality zone (V8)"""
    for zone, (lo, hi) in GAP_ZONES.items():
        if lo <= gap_pct_abs < hi:
            return zone
    return "OVEREXTENDED"

FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote"

# IBKR connector with lazy loading
_ibkr_connector = None
_ibkr_init_attempted = False


def _get_ibkr_connector():
    """
    Lazy loading for IBKR connector.
    Only initializes once, caches result (including failures).
    """
    global _ibkr_connector, _ibkr_init_attempted

    if _ibkr_init_attempted:
        return _ibkr_connector

    _ibkr_init_attempted = True

    if not USE_IBKR_DATA:
        return None

    try:
        from src.ibkr_connector import get_ibkr
        _ibkr_connector = get_ibkr()
        if _ibkr_connector and _ibkr_connector.connected:
            logger.info("PM Scanner: IBKR connector initialized")
        else:
            logger.debug("PM Scanner: IBKR not connected, using Finnhub fallback")
            _ibkr_connector = None
    except ImportError:
        logger.debug("PM Scanner: IBKR connector not available")
        _ibkr_connector = None
    except Exception as e:
        logger.debug(f"PM Scanner: IBKR init failed: {e}")
        _ibkr_connector = None

    return _ibkr_connector


# ============================
# Fetch live quote (IBKR or Finnhub)
# ============================

def fetch_quote(ticker):
    """Fetch quote from IBKR (if available) or Finnhub fallback"""

    # Try IBKR first (lazy load)
    ibkr = _get_ibkr_connector()
    if ibkr and ibkr.connected:
        try:
            quote = ibkr.get_quote(ticker)
            if quote:
                # Convert IBKR format to Finnhub-like format
                return {
                    "o": quote.get("open"),
                    "h": quote.get("high"),
                    "l": quote.get("low"),
                    "c": quote.get("last"),
                    "pc": quote.get("close"),  # previous close
                    "v": quote.get("volume")
                }
        except Exception as e:
            logger.debug(f"IBKR quote failed for {ticker}: {e}")
    
    # Fallback to Finnhub
    params = {
        "symbol": ticker,
        "token": FINNHUB_API_KEY
    }

    r = safe_get(FINNHUB_QUOTE, params=params)
    return r.json()


# ============================
# PM Metrics
# ============================

def compute_pm_metrics(ticker):
    if not is_premarket():
        return None

    cached = cache.get(f"pm_{ticker}")
    if cached:
        return cached

    try:
        q = fetch_quote(ticker)

        pm_open = q.get("o")
        pm_high = q.get("h")
        pm_low = q.get("l")
        last = q.get("c")
        prev_close = q.get("pc")  # V8 FIX: Use previous close for true overnight gap
        volume = q.get("v", 0)

        if not pm_open or not last:
            return None

        # V8 FIX (P1 CRITICAL): True overnight gap uses prev_close, NOT pm_open
        # Old (WRONG): gap_pct = (last - pm_open) / pm_open  ← intra-PM momentum
        # New (CORRECT): gap_pct = (last - prev_close) / prev_close  ← true gap
        if prev_close and prev_close > 0:
            gap_pct = (last - prev_close) / prev_close
        else:
            # Fallback if prev_close unavailable (should be rare)
            gap_pct = (last - pm_open) / pm_open
            logger.warning(f"{ticker}: prev_close unavailable, using pm_open fallback")

        # V8: Separate intra-PM momentum (useful metric, but distinct from gap)
        intra_pm_momentum = (last - pm_open) / pm_open if pm_open > 0 else 0

        # V8: PM range momentum (high-low spread)
        pm_range = (pm_high - pm_low) / pm_low if pm_low and pm_low > 0 else 0

        liquid = volume >= PM_MIN_VOLUME

        # V8: Gap classification and quality score
        gap_zone = classify_gap(abs(gap_pct))
        gap_quality = GAP_QUALITY_SCORES.get(gap_zone, 0.1)
        gap_direction = "UP" if gap_pct > 0 else "DOWN" if gap_pct < 0 else "FLAT"

        metrics = {
            "gap_pct": gap_pct,
            "gap_zone": gap_zone,              # V8: NEGLIGIBLE/EXPLOITABLE/EXTENDED/OVEREXTENDED
            "gap_quality": gap_quality,          # V8: 0-1 quality score (1.0 = sweet spot)
            "gap_direction": gap_direction,      # V8: UP/DOWN/FLAT
            "prev_close": prev_close,            # V8: Preserved for downstream use
            "intra_pm_momentum": intra_pm_momentum,  # V8: Separated from gap
            "pm_high": pm_high,
            "pm_low": pm_low,
            "pm_momentum": pm_range,
            "pm_volume": volume,
            "pm_liquid": liquid
        }

        cache.set(f"pm_{ticker}", metrics)

        if gap_zone == "EXPLOITABLE" and gap_direction == "UP" and liquid:
            logger.info(f"🎯 {ticker} SWEET SPOT gap: {gap_pct:+.1%} ({gap_zone}) vol={volume:,}")

        return metrics

    except Exception as e:
        logger.error(f"PM scan error {ticker}: {e}")
        return None


# ============================
# Batch scan
# ============================

def scan_premarket(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        m = compute_pm_metrics(t)
        if m:
            results[t] = m

    logger.info(f"PM scanned {len(results)} tickers")

    return results


if __name__ == "__main__":
    print(compute_pm_metrics("AAPL"))
