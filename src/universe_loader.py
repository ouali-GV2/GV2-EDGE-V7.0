"""
UNIVERSE LOADER V2 - Enhanced Small Caps Filter
================================================

Construit l'univers des small caps US avec filtres anti-manipulation:
- Market cap filtering
- OTC Markets exclusion (OTCBB, Pink Sheets, etc.)
- Penny stocks INCLUS (< $1 autorisé sur NASDAQ/NYSE)
- Low float exclusion (float > 5M shares)
- Volume filtering (avg > 500K)
- SPAC & shell company exclusion
- Proper error handling & caching

Note: Les penny stocks listés sur les exchanges majeurs (NASDAQ, NYSE, AMEX)
sont inclus car ils ont des obligations de reporting SEC. Seuls les OTC
Markets sont exclus car moins réglementés.
"""

import os
import time
import pandas as pd

from config import (
    FINNHUB_API_KEY, 
    MAX_MARKET_CAP,
    MIN_PRICE,
    MAX_PRICE,
    MIN_AVG_VOLUME,
    EXCLUDE_OTC
)
from utils.api_guard import safe_get
from utils.cache import Cache
from utils.logger import get_logger

logger = get_logger("UNIVERSE_LOADER_V2")

cache = Cache(ttl=60 * 60)  # 1h cache


FINNHUB_BASE = "https://finnhub.io/api/v1"


# ============================
# Finnhub Symbol Fetching
# ============================

def fetch_finnhub_symbols():
    """Fetch all US stock symbols from Finnhub"""
    url = f"{FINNHUB_BASE}/stock/symbol"
    params = {
        "exchange": "US",
        "token": FINNHUB_API_KEY
    }

    r = safe_get(url, params=params, timeout=10)
    data = r.json()

    df = pd.DataFrame(data)

    # Keep common stocks only (exclude ADRs, preferred, etc.)
    if "type" in df.columns:
        df = df[df["type"] == "Common Stock"]

    return df[["symbol", "displaySymbol", "description"]]


# ============================
# Market Data Fetching
# ============================

def fetch_stock_profile(ticker):
    """
    Fetch comprehensive stock profile
    
    Returns:
        dict with market_cap, price, shares_outstanding, etc.
    """
    try:
        url = f"{FINNHUB_BASE}/stock/profile2"
        params = {
            "symbol": ticker,
            "token": FINNHUB_API_KEY
        }

        r = safe_get(url, params=params, timeout=5)
        profile = r.json()

        # Extract key data
        return {
            "ticker": ticker,
            "market_cap": profile.get("marketCapitalization", 0),
            "shares_outstanding": profile.get("shareOutstanding", 0),
            "name": profile.get("name", ""),
            "ipo": profile.get("ipo", ""),
            "country": profile.get("country", ""),
            "exchange": profile.get("exchange", "")
        }

    except Exception as e:
        logger.warning(f"Profile fetch failed for {ticker}: {e}")
        return None


def fetch_quote(ticker):
    """
    Fetch current price and volume
    
    Returns:
        dict with price, volume
    """
    try:
        url = f"{FINNHUB_BASE}/quote"
        params = {
            "symbol": ticker,
            "token": FINNHUB_API_KEY
        }

        r = safe_get(url, params=params, timeout=5)
        quote = r.json()

        return {
            "ticker": ticker,
            "price": quote.get("c", 0),  # current price
            "volume": quote.get("v", 0)   # volume
        }

    except Exception as e:
        logger.warning(f"Quote fetch failed for {ticker}: {e}")
        return None


# ============================
# Enhanced Filtering
# ============================

def is_spac_or_shell(ticker, name, ipo_date):
    """
    Detect SPACs and shell companies
    
    Heuristics:
    - Name contains "Acquisition", "Holdings", "Capital"
    - IPO very recent (<6 months)
    - Empty/generic name
    """
    if not name:
        return True  # No name = suspicious
    
    name_lower = name.lower()
    
    # SPAC keywords
    spac_keywords = [
        "acquisition",
        "capital corp",
        "holdings inc",
        "special purpose",
        "blank check"
    ]
    
    for keyword in spac_keywords:
        if keyword in name_lower:
            return True
    
    # Very recent IPO (potential SPAC pre-merger)
    if ipo_date:
        try:
            from datetime import datetime
            ipo_dt = datetime.strptime(ipo_date, "%Y-%m-%d")
            months_since_ipo = (datetime.utcnow() - ipo_dt).days / 30
            
            if months_since_ipo < 6:
                # Very new stock = risky
                return True
        except:
            pass
    
    return False


def filter_universe(stocks_df):
    """
    Apply comprehensive filters to raw stock list
    
    Filters:
    1. Market cap < MAX_MARKET_CAP
    2. Price between MIN_PRICE and MAX_PRICE
    3. Volume > MIN_AVG_VOLUME
    4. Exclude OTC (if enabled)
    5. Exclude SPACs/shells
    6. Shares outstanding > 5M (low float exclusion)
    """
    logger.info(f"Filtering {len(stocks_df)} raw stocks...")
    
    filtered = stocks_df.copy()
    
    # 1. Market cap filter
    filtered = filtered[
        (filtered["market_cap"] > 0) & 
        (filtered["market_cap"] <= MAX_MARKET_CAP)
    ]
    logger.info(f"After market cap filter: {len(filtered)}")
    
    # 2. Price filter (penny stocks inclus, MIN_PRICE ~= 0)
    filtered = filtered[
        (filtered["price"] >= MIN_PRICE) &
        (filtered["price"] <= MAX_PRICE)
    ]
    logger.info(f"After price filter: {len(filtered)}")
    
    # 3. Volume filter
    filtered = filtered[filtered["volume"] >= MIN_AVG_VOLUME]
    logger.info(f"After volume filter: {len(filtered)}")
    
    # 4. OTC exclusion
    if EXCLUDE_OTC:
        # OTC exchanges typically have "OTC" in exchange name
        filtered = filtered[~filtered["exchange"].str.contains("OTC", na=False, case=False)]
        logger.info(f"After OTC exclusion: {len(filtered)}")
    
    # 5. Low float exclusion (shares outstanding > 5M)
    filtered = filtered[filtered["shares_outstanding"] >= 5.0]  # in millions
    logger.info(f"After low float exclusion: {len(filtered)}")
    
    # 6. SPAC/shell exclusion
    filtered["is_spac"] = filtered.apply(
        lambda row: is_spac_or_shell(row["ticker"], row["name"], row["ipo"]),
        axis=1
    )
    filtered = filtered[~filtered["is_spac"]]
    filtered = filtered.drop(columns=["is_spac"])
    logger.info(f"After SPAC/shell exclusion: {len(filtered)}")
    
    return filtered


# ============================
# Build Universe (Enhanced)
# ============================

def build_universe_v2():
    """
    Build enhanced small caps universe
    
    Process:
    1. Fetch all US symbols
    2. Get profiles (market cap, shares, etc.)
    3. Get quotes (price, volume)
    4. Apply comprehensive filters
    5. Return clean universe
    """
    logger.info("=" * 60)
    logger.info("BUILDING UNIVERSE V2 (ENHANCED)")
    logger.info("=" * 60)

    # Step 1: Get symbols
    symbols_df = fetch_finnhub_symbols()
    tickers = symbols_df["symbol"].tolist()
    
    logger.info(f"Fetched {len(tickers)} raw US symbols")

    # Step 2 & 3: Get profiles and quotes
    # (Limit to avoid rate limits on free tier)
    # In production, batch this or use premium tier
    
    stocks_data = []
    
    for i, ticker in enumerate(tickers[:500]):  # Limit for demo
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{min(500, len(tickers))}")
        
        profile = fetch_stock_profile(ticker)
        if not profile:
            continue
        
        quote = fetch_quote(ticker)
        if not quote:
            continue
        
        # Merge data
        stock_data = {**profile, **quote}
        stocks_data.append(stock_data)
        
        time.sleep(0.15)  # Rate limit respect

    stocks_df = pd.DataFrame(stocks_data)
    
    logger.info(f"Fetched data for {len(stocks_df)} stocks")

    # Step 4: Filter
    universe = filter_universe(stocks_df)

    # Step 5: Sort by market cap
    universe = universe.sort_values("market_cap", ascending=True)

    logger.info(f"Final universe: {len(universe)} small caps")
    logger.info("=" * 60)

    return universe[["ticker", "market_cap", "price", "volume", "shares_outstanding", "name"]]


# ============================
# Main Loader (cached)
# ============================

def load_universe(force_refresh=False):
    """Load universe with caching"""
    cached = cache.get("universe_v2")

    if cached is not None and not force_refresh:
        logger.info(f"Universe loaded from cache ({len(cached)} tickers)")
        return cached

    try:
        universe = build_universe_v2()
        cache.set("universe_v2", universe)

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        universe.to_csv("data/universe.csv", index=False)

        logger.info(f"Universe built and saved: {len(universe)} tickers")

        return universe

    except Exception as e:
        logger.error(f"Universe build failed: {e}", exc_info=True)

        # Fallback to saved CSV
        if os.path.exists("data/universe.csv"):
            logger.warning("Loading universe from fallback CSV")
            return pd.read_csv("data/universe.csv")

        raise RuntimeError("No universe available")


# ============================
# Quick helper
# ============================

def get_tickers(limit=None):
    """Get ticker list from universe"""
    df = load_universe()
    tickers = df["ticker"].tolist()

    if limit:
        tickers = tickers[:limit]

    return tickers


if __name__ == "__main__":
    u = load_universe(force_refresh=True)
    print(u.head(10))
    print(f"\nTotal tickers: {len(u)}")
    print(f"Market cap range: ${u['market_cap'].min():.0f}M - ${u['market_cap'].max():.0f}M")
    print(f"Price range: ${u['price'].min():.2f} - ${u['price'].max():.2f}")

