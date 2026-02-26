"""
EXTENDED HOURS QUOTES - After-Hours & Pre-Market via IBKR
==========================================================

Récupère les quotes en dehors des heures de marché régulières (RTH).

Heures de trading US:
- Pre-Market: 04:00 - 09:30 ET
- Regular (RTH): 09:30 - 16:00 ET  
- After-Hours: 16:00 - 20:00 ET

Avec tes abonnements IBKR (NYSE, NASDAQ, BATS L1):
✅ Pre-Market quotes disponibles
✅ After-Hours quotes disponibles
✅ Volumes extended hours

Configuration IBKR requise:
- "Allow connections from localhost only" dans API settings
- Extended hours trading enabled (même en read-only)

Usage:
- Détecter gaps forming en after-hours
- Suivre momentum pre-market
- Confirmer breakouts avant RTH
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from utils.cache import Cache
from utils.time_utils import is_after_hours, is_premarket, get_market_session
from config import USE_IBKR_DATA

logger = get_logger("EXTENDED_HOURS")

# Cache
extended_cache = Cache(ttl=60)  # 1 min cache for real-time


@dataclass
class ExtendedQuote:
    """Quote with extended hours data"""
    ticker: str
    session: str  # 'PRE', 'RTH', 'POST', 'CLOSED'
    
    # Current prices
    last: float
    bid: float
    ask: float
    
    # Volume
    volume: int
    extended_volume: int  # Volume in extended session only
    
    # Reference prices
    prev_close: float     # Yesterday's close
    rth_close: float      # Today's RTH close (for after-hours)
    rth_open: float       # Today's RTH open
    
    # Calculated
    gap_pct: float        # Gap from prev_close
    change_pct: float     # Change from reference (rth_close or prev_close)
    
    # Metadata
    timestamp: str
    is_halted: bool = False


class ExtendedHoursScanner:
    """
    Scanner for extended hours quotes via IBKR
    """
    
    def __init__(self):
        self.ibkr = None
        self.ib = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize IBKR connection"""
        if not USE_IBKR_DATA:
            logger.warning("IBKR disabled in config")
            return
        
        try:
            from src.ibkr_connector import get_ibkr
            self.ibkr = get_ibkr()
            
            if self.ibkr and self.ibkr.connected:
                self.ib = self.ibkr.ib
                logger.info("✅ Extended Hours Scanner connected")
            else:
                logger.warning("⚠️ IBKR not connected")
                
        except Exception as e:
            logger.error(f"Extended Hours init failed: {e}")
    
    def get_extended_quote(self, ticker: str, use_cache: bool = True) -> Optional[ExtendedQuote]:
        """
        Get quote with extended hours data
        
        Includes:
        - Current bid/ask/last
        - Extended hours volume
        - Gap calculation
        """
        cache_key = f"ext_{ticker}"
        
        if use_cache:
            cached = extended_cache.get(cache_key)
            if cached:
                return cached
        
        # S2-7 FIX: Route directly to ibkr_connector.get_quote() instead of
        # reqMktData() + time.sleep(0.5) + ticker(). The connector already wraps
        # the same L1 data with caching and is non-blocking (cache hit = 0ms).
        # Fallback path (_fallback_quote) uses the same logic — unify to it.
        quote = self._fallback_quote(ticker)
        if quote:
            extended_cache.set(cache_key, quote)
        return quote

        # --- legacy reqMktData path (kept for reference, no longer reachable) ---
        try:  # noqa: unreachable
            from ib_insync import Stock

            contract = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            self.ib.reqMktData(contract, genericTickList='', snapshot=False, regulatorySnapshot=False)
            time.sleep(0.5)  # blocking — replaced above
            ticker_data = self.ib.ticker(contract)

            if not ticker_data:
                self.ib.cancelMktData(contract)
                return None

            bars = self.ib.reqHistoricalData(
                contract, endDateTime='', durationStr='2 D',
                barSizeSetting='1 day', whatToShow='TRADES', useRTH=True, formatDate=1
            )

            prev_close = bars[-2].close if len(bars) >= 2 else 0
            rth_close = bars[-1].close if bars else 0
            rth_open = bars[-1].open if bars else 0

            session = get_market_session()
            last = ticker_data.last or ticker_data.close or 0
            bid = ticker_data.bid or 0
            ask = ticker_data.ask or 0
            volume = ticker_data.volume or 0

            if prev_close > 0:
                gap_pct = (last - prev_close) / prev_close
            else:
                gap_pct = 0

            if session == 'POST' and rth_close > 0:
                change_pct = (last - rth_close) / rth_close
            elif session == 'PRE' and prev_close > 0:
                change_pct = (last - prev_close) / prev_close
            else:
                change_pct = gap_pct

            extended_volume = volume

            quote = ExtendedQuote(
                ticker=ticker,
                session=session,
                last=last,
                bid=bid,
                ask=ask,
                volume=volume,
                extended_volume=extended_volume,
                prev_close=prev_close,
                rth_close=rth_close,
                rth_open=rth_open,
                gap_pct=round(gap_pct, 4),
                change_pct=round(change_pct, 4),
                timestamp=datetime.utcnow().isoformat(),
                is_halted=ticker_data.halted if hasattr(ticker_data, 'halted') else False
            )
            
            self.ib.cancelMktData(contract)
            
            extended_cache.set(cache_key, quote)
            
            return quote
            
        except Exception as e:
            logger.error(f"Extended quote error {ticker}: {e}")
            return self._fallback_quote(ticker)
    
    def _fallback_quote(self, ticker: str) -> Optional[ExtendedQuote]:
        """Fallback quote: IBKR poll → Finnhub WS → None.

        Priority:
          1. IBKR connector get_quote() — primary (cache hit ~0ms)
          2. Finnhub WS screener — used when IBKR is disconnected or has no data
             (pre-market gaps, after-hours, IBKR outage)
          3. None — no data available
        """
        # ── 1. IBKR ───────────────────────────────────────────────────────────
        if self.ibkr:
            try:
                quote = self.ibkr.get_quote(ticker, use_cache=False)

                if quote and quote.get('last', 0) > 0:
                    session = get_market_session()
                    prev_close = quote.get('close', 0)
                    last = quote.get('last', 0)
                    gap_pct = (last - prev_close) / prev_close if prev_close > 0 else 0

                    return ExtendedQuote(
                        ticker=ticker,
                        session=session,
                        last=last,
                        bid=quote.get('bid', 0),
                        ask=quote.get('ask', 0),
                        volume=quote.get('volume', 0),
                        extended_volume=quote.get('volume', 0),
                        prev_close=prev_close,
                        rth_close=prev_close,
                        rth_open=quote.get('open', 0),
                        gap_pct=round(gap_pct, 4),
                        change_pct=round(gap_pct, 4),
                        timestamp=datetime.utcnow().isoformat()
                    )
            except Exception as e:
                logger.debug(f"IBKR fallback error {ticker}: {e}")

        # ── 2. Finnhub WS screener ────────────────────────────────────────────
        # Works during pre-market and after-hours when Finnhub WS is connected.
        try:
            from src.finnhub_ws_screener import get_finnhub_ws_screener
            ws = get_finnhub_ws_screener()
            ws_state = ws.get_ticker_state(ticker)

            if ws_state and ws_state.last_price > 0:
                session = get_market_session()
                last = ws_state.last_price
                volume = int(ws_state.volume_5min)

                return ExtendedQuote(
                    ticker=ticker,
                    session=session,
                    last=last,
                    bid=0.0,
                    ask=0.0,
                    volume=volume,
                    extended_volume=volume,
                    prev_close=0.0,
                    rth_close=0.0,
                    rth_open=0.0,
                    gap_pct=0.0,       # prev_close unknown without IBKR
                    change_pct=0.0,
                    timestamp=datetime.utcnow().isoformat(),
                )
        except Exception as _ws_e:
            logger.debug(f"Finnhub WS fallback error {ticker}: {_ws_e}")

        return None
    
    def scan_extended_movers(
        self, 
        tickers: List[str],
        min_gap_pct: float = 0.03,
        min_volume: int = 10000
    ) -> List[ExtendedQuote]:
        """
        Scan for movers in extended hours
        
        Args:
            tickers: List of tickers to scan
            min_gap_pct: Minimum gap % to include (default 3%)
            min_volume: Minimum extended volume
        
        Returns:
            List of ExtendedQuote for movers, sorted by gap
        """
        movers = []
        
        logger.info(f"🔍 Scanning {len(tickers)} tickers for extended hours movers...")
        
        for ticker in tickers:
            try:
                quote = self.get_extended_quote(ticker)
                
                if not quote:
                    continue
                
                # Check filters
                if abs(quote.gap_pct) >= min_gap_pct and quote.volume >= min_volume:
                    movers.append(quote)
                    logger.info(
                        f"📈 {ticker}: gap={quote.gap_pct*100:+.1f}%, "
                        f"vol={quote.volume:,}"
                    )
                
            except Exception as e:
                logger.debug(f"Scan error {ticker}: {e}")
                continue
        
        # Sort by absolute gap
        movers.sort(key=lambda q: abs(q.gap_pct), reverse=True)
        
        logger.info(f"Found {len(movers)} extended hours movers")
        
        return movers


# ============================
# HELPER FUNCTIONS
# ============================

def get_extended_quote(ticker: str) -> Optional[ExtendedQuote]:
    """Convenience function to get extended quote"""
    scanner = ExtendedHoursScanner()
    return scanner.get_extended_quote(ticker)


def scan_afterhours_gaps(tickers: List[str], min_gap: float = 0.03) -> List[ExtendedQuote]:
    """Scan for after-hours gaps"""
    if not is_after_hours():
        logger.warning("Not in after-hours session")
        return []
    
    scanner = ExtendedHoursScanner()
    return scanner.scan_extended_movers(tickers, min_gap_pct=min_gap)


def scan_premarket_gaps(tickers: List[str], min_gap: float = 0.03) -> List[ExtendedQuote]:
    """Scan for pre-market gaps"""
    if not is_premarket():
        logger.warning("Not in pre-market session")
        return []
    
    scanner = ExtendedHoursScanner()
    return scanner.scan_extended_movers(tickers, min_gap_pct=min_gap)


def detect_gap_forming(ticker: str, threshold: float = 0.03) -> Tuple[bool, Dict]:
    """
    Detect if a significant gap is forming
    
    Returns: (is_gapping, details)
    """
    scanner = ExtendedHoursScanner()
    quote = scanner.get_extended_quote(ticker)
    
    if not quote:
        return False, {}
    
    is_gapping = abs(quote.gap_pct) >= threshold
    
    details = {
        'ticker': ticker,
        'session': quote.session,
        'gap_pct': quote.gap_pct,
        'last': quote.last,
        'prev_close': quote.prev_close,
        'volume': quote.volume,
        'is_gapping': is_gapping,
        'direction': 'UP' if quote.gap_pct > 0 else 'DOWN'
    }
    
    return is_gapping, details


# ============================
# INTEGRATION WITH ANTICIPATION ENGINE
# ============================

def get_extended_hours_boost(ticker: str) -> Tuple[float, Dict]:
    """
    Calculate boost score for extended hours activity
    
    Used by Monster Score to boost tickers showing early movement
    
    Returns: (boost_score, details)
    """
    quote = get_extended_quote(ticker)
    
    if not quote:
        return 0.0, {}
    
    boost = 0.0
    details = {}
    
    # Gap boost (max +0.15)
    gap_abs = abs(quote.gap_pct)
    if gap_abs >= 0.03:
        gap_boost = min(0.15, gap_abs * 1.5)
        boost += gap_boost
        details['gap_boost'] = round(gap_boost, 3)
    
    # Volume boost (max +0.05)
    if quote.volume >= 50000:
        vol_boost = min(0.05, quote.volume / 1000000)
        boost += vol_boost
        details['volume_boost'] = round(vol_boost, 3)
    
    # Direction confirmation (extra +0.02 if gap is positive)
    if quote.gap_pct > 0.03:
        boost += 0.02
        details['direction'] = 'UP'
    
    details['total_boost'] = round(boost, 3)
    details['session'] = quote.session
    details['gap_pct'] = quote.gap_pct
    details['volume'] = quote.volume
    
    return boost, details


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("EXTENDED HOURS SCANNER - TEST")
    print("=" * 60)
    
    from utils.time_utils import get_market_session
    
    session = get_market_session()
    print(f"\nCurrent session: {session}")
    
    test_tickers = ["AAPL", "TSLA", "NVDA", "AMD"]
    
    print(f"\nTesting with: {test_tickers}")
    
    scanner = ExtendedHoursScanner()
    
    for ticker in test_tickers:
        quote = scanner.get_extended_quote(ticker)
        
        if quote:
            print(f"\n{ticker}:")
            print(f"  Session: {quote.session}")
            print(f"  Last: ${quote.last:.2f}")
            print(f"  Gap: {quote.gap_pct*100:+.2f}%")
            print(f"  Volume: {quote.volume:,}")
        else:
            print(f"\n{ticker}: No data")
