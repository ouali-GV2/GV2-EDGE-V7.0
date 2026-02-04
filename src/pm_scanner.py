from utils.logger import get_logger
from utils.api_guard import safe_get
from utils.cache import Cache
from utils.time_utils import is_premarket

from config import FINNHUB_API_KEY, PM_MIN_VOLUME, USE_IBKR_DATA

from datetime import datetime

logger = get_logger("PM_SCANNER")

cache = Cache(ttl=30)

FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote"

# IBKR connector (if enabled)
ibkr_connector = None

if USE_IBKR_DATA:
    try:
        from src.ibkr_connector import get_ibkr
        ibkr_connector = get_ibkr()
    except:
        pass


# ============================
# Fetch live quote (IBKR or Finnhub)
# ============================

def fetch_quote(ticker):
    """Fetch quote from IBKR (if available) or Finnhub"""
    
    # Try IBKR first
    if ibkr_connector and ibkr_connector.connected:
        try:
            quote = ibkr_connector.get_quote(ticker)
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
        volume = q.get("v", 0)

        if not pm_open or not last:
            return None

        gap_pct = (last - pm_open) / pm_open

        momentum = (pm_high - pm_low) / pm_low if pm_low else 0

        liquid = volume >= PM_MIN_VOLUME

        metrics = {
            "gap_pct": gap_pct,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "pm_momentum": momentum,
            "pm_volume": volume,
            "pm_liquid": liquid
        }

        cache.set(f"pm_{ticker}", metrics)

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
