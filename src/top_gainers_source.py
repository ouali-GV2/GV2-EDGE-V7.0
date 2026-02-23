"""
TOP GAINERS EXTERNAL SOURCE V8
================================

C8 FIX (P10): Aucune source externe de top gainers.

Le systeme detectait les movers uniquement via son propre scoring, sans
reference aux listes publiques de top gainers. Resultat : des top gainers
evidents pouvaient etre manques si le scoring ne les capturait pas.

Ce module ajoute 2 sources externes :
1. IBKR Scanner (Market Scanner) — temps reel, pas de rate limit
2. Yahoo Finance (yfinance) — gratuit, backup

Les top gainers externes sont injectes dans la Hot Ticker Queue pour
scanning prioritaire.

Usage:
    source = TopGainersSource()
    gainers = await source.fetch_top_gainers()
    for g in gainers:
        hot_queue.push(g.ticker, TickerPriority.HOT, TriggerReason.EXTERNAL_GAINER)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get
from config import USE_IBKR_DATA

logger = get_logger("TOP_GAINERS_SOURCE")

# Cache to avoid hammering sources
_cache = Cache(ttl=300)  # 5 min cache


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExternalGainer:
    """A top gainer from an external source."""
    ticker: str
    change_pct: float           # % change today
    price: float
    volume: int
    source: str                  # "ibkr" or "yahoo"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    market_cap: Optional[float] = None

    def is_small_cap(self) -> bool:
        """Check if this fits our small-cap criteria."""
        if self.market_cap and self.market_cap > 2_000_000_000:
            return False
        if self.price < 0.50 or self.price > 50.0:
            return False
        return True


# ============================================================================
# Top Gainers Source
# ============================================================================

class TopGainersSource:
    """
    External top gainers fetcher.

    Combines IBKR Scanner and Yahoo Finance for comprehensive coverage.
    """

    def __init__(self):
        self._ibkr_available = USE_IBKR_DATA
        self._last_fetch: Optional[datetime] = None
        self._fetch_count = 0

    async def fetch_top_gainers(
        self,
        min_change_pct: float = 5.0,
        max_price: float = 50.0,
        min_volume: int = 100_000,
    ) -> List[ExternalGainer]:
        """
        Fetch top gainers from all available sources.

        Args:
            min_change_pct: Minimum % change to include
            max_price: Maximum price filter
            min_volume: Minimum volume filter

        Returns:
            List of ExternalGainer, sorted by change_pct descending
        """
        # Check cache
        cached = _cache.get("top_gainers")
        if cached:
            return cached

        gainers = []

        # Source 1: IBKR Scanner
        if self._ibkr_available:
            ibkr_gainers = self._fetch_ibkr_gainers(min_change_pct, max_price)
            gainers.extend(ibkr_gainers)

        # Source 2: Yahoo Finance (backup/supplement)
        yahoo_gainers = self._fetch_yahoo_gainers(min_change_pct, max_price)
        gainers.extend(yahoo_gainers)

        # Deduplicate (prefer IBKR data)
        seen = set()
        unique_gainers = []
        for g in gainers:
            if g.ticker not in seen:
                seen.add(g.ticker)
                unique_gainers.append(g)

        # Filter
        filtered = [
            g for g in unique_gainers
            if g.change_pct >= min_change_pct
            and g.volume >= min_volume
            and g.is_small_cap()
        ]

        # Sort by change %
        filtered.sort(key=lambda g: g.change_pct, reverse=True)

        # Cache
        _cache.set("top_gainers", filtered)

        self._last_fetch = datetime.utcnow()
        self._fetch_count += 1

        logger.info(
            f"Fetched {len(filtered)} top gainers "
            f"(IBKR: {len([g for g in filtered if g.source == 'ibkr'])}, "
            f"Yahoo: {len([g for g in filtered if g.source == 'yahoo'])})"
        )

        return filtered

    def _fetch_ibkr_gainers(
        self,
        min_change_pct: float,
        max_price: float,
    ) -> List[ExternalGainer]:
        """
        Fetch top gainers via IBKR Market Scanner.

        Uses scannerSubscribe for US equities sorted by % change.
        """
        gainers = []

        try:
            from src.ibkr_connector import get_ibkr
            ibkr = get_ibkr()

            if not ibkr or not ibkr.connected:
                logger.debug("IBKR not connected for scanner")
                return []

            # Request top gainers scan
            # ScannerSubscription for "TOP_PERC_GAIN" instrument type "STK"
            scan_results = ibkr.run_scanner(
                instrument="STK",
                location="STK.US.MAJOR",
                scan_code="TOP_PERC_GAIN",
                above_price=0.50,
                below_price=max_price,
                above_volume=50000,
                number_of_rows=50,
            )

            if scan_results:
                for result in scan_results:
                    ticker = result.get("symbol", "")
                    change_pct = result.get("change_pct", 0)
                    price = result.get("last", 0)
                    volume = result.get("volume", 0)
                    market_cap = result.get("market_cap")

                    if ticker and change_pct >= min_change_pct:
                        gainers.append(ExternalGainer(
                            ticker=ticker,
                            change_pct=change_pct,
                            price=price,
                            volume=volume,
                            source="ibkr",
                            market_cap=market_cap,
                        ))

                logger.info(f"IBKR Scanner: {len(gainers)} top gainers")

        except Exception as e:
            logger.debug(f"IBKR scanner error: {e}")

        return gainers

    def _fetch_yahoo_gainers(
        self,
        min_change_pct: float,
        max_price: float,
    ) -> List[ExternalGainer]:
        """
        Fetch top gainers via Yahoo Finance (yfinance).

        Free, no API key needed. Good backup source.
        """
        gainers = []

        try:
            import yfinance as yf

            # Yahoo Finance day gainers
            # Use the screener for US small cap gainers
            tickers_data = yf.Tickers(
                " ".join([])  # Empty — we use the screener instead
            )

            # Alternative: scrape Yahoo Finance gainers page
            url = "https://finance.yahoo.com/gainers"
            try:
                r = safe_get(url, timeout=10)
                # Parse response for ticker data
                # Yahoo returns HTML, we need to extract the data
                # For simplicity, use the query API instead
                query_url = (
                    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                    "?formatted=true&lang=en-US&region=US&scrIds=day_gainers&count=50"
                )
                r = safe_get(query_url, timeout=10)
                data = r.json()

                quotes = (
                    data.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
                )

                for quote in quotes:
                    ticker = quote.get("symbol", "")
                    change_pct = quote.get("regularMarketChangePercent", {})
                    if isinstance(change_pct, dict):
                        change_pct = change_pct.get("raw", 0)

                    price = quote.get("regularMarketPrice", {})
                    if isinstance(price, dict):
                        price = price.get("raw", 0)

                    volume = quote.get("regularMarketVolume", {})
                    if isinstance(volume, dict):
                        volume = volume.get("raw", 0)

                    market_cap = quote.get("marketCap", {})
                    if isinstance(market_cap, dict):
                        market_cap = market_cap.get("raw", 0)

                    if (
                        ticker
                        and change_pct >= min_change_pct
                        and 0.50 <= price <= max_price
                    ):
                        gainers.append(ExternalGainer(
                            ticker=ticker,
                            change_pct=round(change_pct, 2),
                            price=round(price, 2),
                            volume=int(volume),
                            source="yahoo",
                            market_cap=market_cap if market_cap else None,
                        ))

                logger.info(f"Yahoo Finance: {len(gainers)} top gainers")

            except Exception as e:
                logger.debug(f"Yahoo screener query error: {e}")

        except ImportError:
            logger.debug("yfinance not installed, skipping Yahoo source")
        except Exception as e:
            logger.debug(f"Yahoo gainers error: {e}")

        return gainers

    def get_stats(self) -> Dict:
        """Get source statistics."""
        return {
            "ibkr_available": self._ibkr_available,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "fetch_count": self._fetch_count,
        }


# ============================================================================
# Singleton
# ============================================================================

_source_instance: Optional[TopGainersSource] = None


def get_top_gainers_source() -> TopGainersSource:
    """Get singleton TopGainersSource instance."""
    global _source_instance
    if _source_instance is None:
        _source_instance = TopGainersSource()
    return _source_instance


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "TopGainersSource",
    "ExternalGainer",
    "get_top_gainers_source",
]
