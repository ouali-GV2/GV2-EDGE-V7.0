"""
UNIVERSE LOADER V3 - Single API Call, Wide Coverage
====================================================

Construit l'univers des small caps US en 1 seul appel API Finnhub.

Principe architectural:
    Univers LARGE et MINIMAL a la construction.
    Filtres dynamiques au moment du scan (Feature Engine / Monster Score).

Le loader repond uniquement a la question:
    "Quels tickers sont des common stocks US cotes sur un exchange majeur
     (non OTC), dans la tranche small cap ?"

Tout le reste (volume, market cap, liquidite) est verifie en temps reel
par le Feature Engine et le Monster Score -- pas en amont.

Resultat attendu: ~2500-3500 tickers, rebuild en < 5 secondes, 1 appel API.

Refresh policy:
- Rebuild complet: 1 fois par semaine (dimanche soir)
- Cache memoire: TTL 24h (l'univers ne change pas en intraday)
- Fallback: si le rebuild echoue, charger data/universe.csv sans bloquer
"""

import os
import re
import pandas as pd

from config import FINNHUB_API_KEY
from utils.api_guard import safe_get
from utils.cache import Cache
from utils.logger import get_logger

logger = get_logger("UNIVERSE_LOADER_V3")

# 24h cache -- l'univers des symboles cotes ne change pas en intraday
cache = Cache(ttl=60 * 60 * 24)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# OTC exchange keywords to exclude
_OTC_KEYWORDS = re.compile(r"OTC|Pink|Grey|Expert", re.IGNORECASE)

# Ticker suffix patterns for warrants, rights, units
# W, WS = warrants, R = rights, U = units
_BAD_SUFFIX = re.compile(r"[WR]$|WS$|U$")

# Characters that indicate non-standard securities
_BAD_CHARS = re.compile(r"[./+$]")


# ============================
# Finnhub Symbol Fetching
# ============================

def fetch_finnhub_symbols():
    """
    Fetch all US stock symbols from Finnhub in a single API call.

    Returns DataFrame with columns: symbol, displaySymbol, description,
    type, exchange, currency.
    """
    url = f"{FINNHUB_BASE}/stock/symbol"
    params = {
        "exchange": "US",
        "token": FINNHUB_API_KEY
    }

    r = safe_get(url, params=params, timeout=15)
    data = r.json()

    df = pd.DataFrame(data)
    logger.info(f"Finnhub returned {len(df)} US symbols")
    return df


# ============================
# Static Filtering (0 API calls)
# ============================

def filter_universe(symbols_df):
    """
    Apply static filters on Finnhub symbol data. Zero additional API calls.

    Filters applied:
    1. type == "Common Stock" -- exclut ETFs, warrants, preferred, ADRs
    2. Exchange non-OTC: exclure si exchange contient OTC, Pink, Grey, Expert
    3. Symbole propre: exclure si ticker contient . / + $
       ou se termine par W, R, U, WS (warrants, rights, units)

    Filters NOT applied (delegated to Feature Engine / Monster Score):
    - Volume (un ticker calme aujourd'hui peut spiquer demain)
    - Market cap (verifie en temps reel)
    - Price (verifie en temps reel)
    """
    if symbols_df.empty:
        logger.warning("Empty symbols DataFrame, nothing to filter")
        return pd.DataFrame(columns=["ticker", "exchange", "name"])

    filtered = symbols_df.copy()
    initial_count = len(filtered)

    # 1. Common Stock only
    if "type" in filtered.columns:
        filtered = filtered[filtered["type"] == "Common Stock"]
    logger.info(f"After Common Stock filter: {len(filtered)} "
                f"(removed {initial_count - len(filtered)})")

    # 2. Exclude OTC exchanges
    if "exchange" in filtered.columns:
        before = len(filtered)
        filtered = filtered[
            ~filtered["exchange"].str.contains(_OTC_KEYWORDS, na=False)
        ]
        logger.info(f"After OTC exclusion: {len(filtered)} "
                    f"(removed {before - len(filtered)})")

    # 3. Clean ticker symbols
    if "symbol" in filtered.columns:
        before = len(filtered)
        # Remove tickers with special characters
        mask_chars = filtered["symbol"].str.contains(_BAD_CHARS, na=False)
        # Remove warrants/rights/units by suffix
        mask_suffix = filtered["symbol"].str.match(
            r".*(" + _BAD_SUFFIX.pattern + r")", na=False
        )
        filtered = filtered[~mask_chars & ~mask_suffix]
        logger.info(f"After symbol cleanup: {len(filtered)} "
                    f"(removed {before - len(filtered)})")

    # Build output DataFrame with standardized column names
    result = pd.DataFrame({
        "ticker": filtered["symbol"].values,
        "exchange": filtered.get("exchange", pd.Series(dtype=str)).values
                    if "exchange" in filtered.columns
                    else [""] * len(filtered),
        "name": filtered.get("description", pd.Series(dtype=str)).values
                if "description" in filtered.columns
                else [""] * len(filtered),
    })

    # Drop any rows with empty/NaN ticker
    result = result.dropna(subset=["ticker"])
    result = result[result["ticker"].str.len() > 0]

    logger.info(f"Final universe: {len(result)} tickers "
                f"(from {initial_count} raw symbols)")
    return result.reset_index(drop=True)


# ============================
# Build Universe (1 API call)
# ============================

def build_universe():
    """
    Build small caps universe in a single API call.

    Target: ~2500-3500 tickers, < 5 seconds.
    """
    logger.info("=" * 60)
    logger.info("BUILDING UNIVERSE V3 (SINGLE API CALL)")
    logger.info("=" * 60)

    symbols_df = fetch_finnhub_symbols()
    universe = filter_universe(symbols_df)

    logger.info(f"Universe built: {len(universe)} tickers")
    logger.info("=" * 60)
    return universe


# ============================
# Main Loader (cached)
# ============================

def load_universe(force_refresh=False):
    """
    Load universe with caching.

    Signature preserved for backward compatibility with:
    main.py, daily_audit, weekly_audit, backtest, watch_list,
    afterhours_scanner, batch_scheduler.

    Returns DataFrame with at minimum: ticker, exchange, name
    """
    cached = cache.get("universe_v3")

    if cached is not None and not force_refresh:
        logger.info(f"Universe loaded from cache ({len(cached)} tickers)")
        return cached

    try:
        universe = build_universe()
        cache.set("universe_v3", universe)

        # Save to CSV for fallback
        os.makedirs("data", exist_ok=True)
        universe.to_csv("data/universe.csv", index=False)
        logger.info(f"Universe built and saved: {len(universe)} tickers")

        return universe

    except Exception as e:
        logger.error(f"Universe build failed: {e}", exc_info=True)

        # Fallback to saved CSV
        if os.path.exists("data/universe.csv"):
            try:
                fallback = pd.read_csv("data/universe.csv")
                if (not fallback.empty
                        and "ticker" in fallback.columns
                        and len(fallback) > 0):
                    logger.warning(
                        f"Loading universe from fallback CSV "
                        f"({len(fallback)} tickers)"
                    )
                    return fallback
                else:
                    logger.warning("Fallback CSV is empty or malformed")
            except Exception as csv_err:
                logger.warning(f"Fallback CSV read failed: {csv_err}")

        logger.error("No universe available - will retry next cycle")
        return pd.DataFrame(columns=["ticker", "exchange", "name"])


# ============================
# Quick helper
# ============================

def get_tickers(limit=None):
    """Get ticker list from universe."""
    df = load_universe()
    tickers = df["ticker"].tolist()

    if limit:
        tickers = tickers[:limit]

    return tickers


if __name__ == "__main__":
    u = load_universe(force_refresh=True)
    print(u.head(20))
    print(f"\nTotal tickers: {len(u)}")
    if "exchange" in u.columns:
        print(f"\nExchanges: {u['exchange'].value_counts().to_dict()}")
    # Validation checks
    assert len(u) > 0, "Universe is empty!"
    assert "ticker" in u.columns, "Missing ticker column!"
    assert not u["ticker"].isna().any(), "NaN tickers found!"
    otc_count = u["exchange"].str.contains("OTC", na=False).sum()
    assert otc_count == 0, f"Found {otc_count} OTC tickers!"
    warrant_count = u["ticker"].str.match(r".*[WR]$|.*WS$").sum()
    print(f"Warrant-like tickers remaining: {warrant_count}")
    print("\nAll validation checks passed!")
