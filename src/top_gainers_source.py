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

import threading
import time
from datetime import datetime, timedelta, timezone
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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

        self._last_fetch = datetime.now(timezone.utc)
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

        S4-2 FIX: `ibkr.run_scanner()` did not exist. Use ib-insync's
        `ScannerSubscription` + `IB.reqScannerSubscription()` directly.
        The scanner subscription is synchronous in ib-insync when called from
        a thread that has access to the running event loop.
        """
        gainers = []

        try:
            from src.ibkr_connector import get_ibkr
            ibkr = get_ibkr()

            if not ibkr or not ibkr.connected:
                logger.debug("IBKR not connected for scanner")
                return []

            # Import ib-insync types
            from ib_insync import ScannerSubscription

            sub = ScannerSubscription(
                instrument="STK",
                locationCode="STK.US.MAJOR",
                scanCode="TOP_PERC_GAIN",
                abovePrice=0.50,
                belowPrice=max_price,
                aboveVolume=50_000,
                numberOfRows=50,
            )

            # reqScannerSubscription returns a list of ScanData that populates
            # asynchronously via the ib-insync event loop. Run in executor thread
            # and give it 3 s to populate.
            scan_data = ibkr.ib.reqScannerSubscription(sub)
            ibkr.ib.sleep(3)  # let the event loop deliver scanner callbacks
            ibkr.ib.cancelScannerSubscription(scan_data)

            for item in scan_data:
                contract = item.contractDetails.contract
                ticker = contract.symbol
                if not ticker:
                    continue

                # Fetch live quote for change_pct / volume
                quote = ibkr.get_quote(ticker)
                if not quote:
                    continue

                price = quote.get("last") or quote.get("close") or 0
                change_pct = quote.get("change_pct", 0)
                volume = quote.get("volume", 0)

                if change_pct >= min_change_pct and 0.50 <= price <= max_price:
                    gainers.append(ExternalGainer(
                        ticker=ticker,
                        change_pct=round(change_pct, 2),
                        price=round(price, 2),
                        volume=int(volume),
                        source="ibkr",
                    ))

            logger.info(f"IBKR Scanner: {len(gainers)} top gainers")

        except ImportError:
            logger.debug("ib-insync ScannerSubscription not available, skipping IBKR scanner")
        except Exception as e:
            logger.debug(f"IBKR scanner error: {e}")

        return gainers

    def _fetch_yahoo_gainers(
        self,
        min_change_pct: float,
        max_price: float,
    ) -> List[ExternalGainer]:
        """
        Fetch top gainers via Yahoo Finance.

        S4-2 FIX: Previous code used query1.finance.yahoo.com (deprecated/404) and
        passed formatted=true which wraps numbers in {raw, fmt} dicts making parsing
        fragile. Now uses query2 + formatted=false for plain numeric values.
        """
        gainers = []

        try:
            # S4-2 FIX: query2 + formatted=false → plain numbers, no dict wrapping
            query_url = (
                "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
                "?formatted=false&lang=en-US&region=US&scrIds=day_gainers&count=50"
            )
            # Mimic browser to avoid 403
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
            }
            r = safe_get(query_url, timeout=15, headers=headers)

            if r.status_code != 200:
                logger.debug(f"Yahoo screener HTTP {r.status_code}")
                return gainers

            data = r.json()
            quotes = (
                data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )

            for quote in quotes:
                ticker = quote.get("symbol", "")
                # formatted=false → plain float, no dict wrapping
                change_pct = float(quote.get("regularMarketChangePercent") or 0)
                price = float(quote.get("regularMarketPrice") or 0)
                volume = int(quote.get("regularMarketVolume") or 0)
                market_cap = quote.get("marketCap")
                market_cap = float(market_cap) if market_cap else None

                if (
                    ticker
                    and change_pct >= min_change_pct
                    and 0.50 <= price <= max_price
                ):
                    gainers.append(ExternalGainer(
                        ticker=ticker,
                        change_pct=round(change_pct, 2),
                        price=round(price, 2),
                        volume=volume,
                        source="yahoo",
                        market_cap=market_cap,
                    ))

            logger.info(f"Yahoo Finance: {len(gainers)} top gainers")

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
_source_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_top_gainers_source() -> TopGainersSource:
    """Get singleton TopGainersSource instance."""
    global _source_instance
    with _source_lock:  # S4-1 FIX
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
