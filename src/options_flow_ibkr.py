"""
OPTIONS FLOW DETECTOR - Unusual Options Activity via IBKR
==========================================================

Détecte les signaux d'options inhabituels qui précèdent souvent
des mouvements explosifs sur small caps.

Requiert: Abonnement IBKR OPRA (Options Price Reporting Authority) L1

Ce que OPRA L1 donne:
- Last price
- Bid/Ask
- Volume
- Open Interest (delayed)

Signaux détectés:
1. VOLUME SPIKE: Volume options >> Open Interest (smart money loading)
2. CALL SWEEP: Gros volume calls OTM achetés sur l'ask (bullish)
3. PUT/CALL RATIO: Ratio anormalement bas (bullish sentiment)
4. NEAR-TERM FOCUS: Concentration sur expirations proches (imminent move)
5. STRIKE CLUSTERING: Plusieurs gros trades sur même strike (target price)

Limitations avec L1:
- Pas de trade-by-trade flow (besoin L2)
- Pas de direction (buy vs sell)
- Open Interest delayed (J-1)

Workaround:
- Comparer volume vs OI pour détecter accumulation
- Analyser bid/ask spread pour estimer direction
- Focus sur options near-the-money et proches expirations
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from utils.logger import get_logger
from utils.cache import Cache
from config import USE_IBKR_DATA

logger = get_logger("OPTIONS_FLOW")

# Cache
options_cache = Cache(ttl=300)  # 5 min


@dataclass
class OptionContract:
    """Single option contract data"""
    symbol: str
    underlying: str
    expiry: str
    strike: float
    option_type: str  # CALL or PUT
    last: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_vol: Optional[float] = None


@dataclass
class OptionsFlowSignal:
    """Unusual options activity signal"""
    ticker: str
    signal_type: str  # VOLUME_SPIKE, CALL_SWEEP, LOW_PC_RATIO, etc.
    score: float      # 0.0 to 1.0
    details: Dict
    timestamp: str


# ============================
# IBKR OPTIONS DATA
# ============================

class IBKROptionsScanner:
    """
    Scanner d'options via IBKR avec abonnement OPRA L1
    """
    
    def __init__(self):
        self.ibkr = None
        self.ib = None  # ib_insync connection
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
                self.ib = self.ibkr.ib  # Access underlying ib_insync
                logger.info("✅ IBKR Options Scanner connected")
            else:
                logger.warning("⚠️ IBKR not connected")
                
        except Exception as e:
            logger.error(f"IBKR Options init failed: {e}")
    
    def get_option_chain(self, ticker: str) -> List[OptionContract]:
        """
        Get option chain for a ticker
        
        Returns list of OptionContract for near-term expirations
        """
        if not self.ib:
            return []
        
        cache_key = f"chain_{ticker}"
        cached = options_cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from ib_insync import Stock, Option
            
            # Create stock contract
            stock = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            # Get option chains (expirations + strikes)
            chains = self.ib.reqSecDefOptParams(
                stock.symbol, '', stock.secType, stock.conId
            )
            
            if not chains:
                return []
            
            # Find nearest expirations (next 30 days)
            today = datetime.now()
            max_expiry = today + timedelta(days=30)
            
            contracts = []
            
            for chain in chains:
                if chain.exchange != 'SMART':
                    continue
                
                for expiry in chain.expirations[:3]:  # Next 3 expirations
                    expiry_date = datetime.strptime(expiry, '%Y%m%d')
                    if expiry_date > max_expiry:
                        continue
                    
                    # Get current stock price for ATM strikes
                    stock_price = self._get_stock_price(ticker)
                    if not stock_price:
                        continue
                    
                    # Get strikes around current price (±20%)
                    atm_strikes = [
                        s for s in chain.strikes
                        if stock_price * 0.8 <= s <= stock_price * 1.2
                    ]
                    
                    for strike in atm_strikes[:10]:  # Limit strikes
                        for right in ['C', 'P']:  # Calls and Puts
                            try:
                                opt = Option(
                                    ticker, expiry, strike, right, 'SMART'
                                )
                                self.ib.qualifyContracts(opt)
                                
                                # Request market data
                                self.ib.reqMktData(opt, '', False, False)
                                time.sleep(0.1)  # OK: runs in executor thread, not event loop
                                
                                ticker_data = self.ib.ticker(opt)
                                
                                if ticker_data and ticker_data.last:
                                    contract = OptionContract(
                                        symbol=f"{ticker}{expiry}{right}{strike}",
                                        underlying=ticker,
                                        expiry=expiry,
                                        strike=strike,
                                        option_type='CALL' if right == 'C' else 'PUT',
                                        last=ticker_data.last or 0,
                                        bid=ticker_data.bid or 0,
                                        ask=ticker_data.ask or 0,
                                        volume=ticker_data.volume or 0,
                                        open_interest=0,  # Need separate request
                                        implied_vol=ticker_data.impliedVolatility
                                    )
                                    contracts.append(contract)
                                
                                self.ib.cancelMktData(opt)
                                
                            except Exception as e:
                                logger.debug(f"Option contract error: {e}")
                                continue
            
            options_cache.set(cache_key, contracts)
            logger.info(f"Fetched {len(contracts)} option contracts for {ticker}")
            
            return contracts
            
        except Exception as e:
            logger.error(f"Option chain fetch failed for {ticker}: {e}")
            return []
    
    def _get_stock_price(self, ticker: str) -> Optional[float]:
        """Get current stock price"""
        if self.ibkr:
            quote = self.ibkr.get_quote(ticker, use_cache=True)
            if quote:
                return quote.get('last', 0)
        return None
    
    def get_options_summary(self, ticker: str) -> Dict:
        """
        Get summarized options data for a ticker.

        Returns volume-based metrics only (OI ratio disabled for reliability):
        - call_volume: Total call contracts volume
        - put_volume: Total put contracts volume
        - total_volume: Combined volume
        - put_call_ratio: P/C ratio for sentiment
        - max_volume_strike: Strike with highest volume
        - max_volume_type: CALL or PUT
        - max_volume_expiry: Expiration date
        - contracts_count: Number of contracts analyzed

        Note: Volume/OI ratio is NOT calculated because OI is delayed (J-1)
        and unreliable for real-time small cap analysis.
        """
        contracts = self.get_option_chain(ticker)

        if not contracts:
            return {}

        # Calculate volume metrics
        call_volume = sum(c.volume for c in contracts if c.option_type == 'CALL')
        put_volume = sum(c.volume for c in contracts if c.option_type == 'PUT')
        total_volume = call_volume + put_volume

        # Find max volume contract (highest activity)
        max_vol_contract = max(contracts, key=lambda c: c.volume) if contracts else None

        # Calculate P/C ratio (lower = more bullish)
        if call_volume > 0:
            pc_ratio = put_volume / call_volume
        else:
            pc_ratio = 999 if put_volume > 0 else 1.0  # Default to neutral if no data

        return {
            'ticker': ticker,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'total_volume': total_volume,
            'put_call_ratio': round(pc_ratio, 3),
            'max_volume_strike': max_vol_contract.strike if max_vol_contract else 0,
            'max_volume_type': max_vol_contract.option_type if max_vol_contract else '',
            'max_volume_expiry': max_vol_contract.expiry if max_vol_contract else '',
            'contracts_count': len(contracts)
            # Note: avg_volume_oi_ratio removed - OI is delayed and unreliable
        }


# ============================
# SIGNAL DETECTION
# ============================

def detect_options_signals(ticker: str, summary: Dict) -> List[OptionsFlowSignal]:
    """
    Detect unusual options activity signals based on VOLUME and CONCENTRATION only.

    Note: Volume/OI ratio is DISABLED because OI is delayed (J-1) and unreliable
    for real-time small cap analysis. We focus on absolute volume and concentration
    which are more stable indicators.

    Signal Types:
    1. HIGH_CALL_VOLUME - Significant call volume (absolute)
    2. LOW_PC_RATIO - Bullish sentiment (call/put imbalance)
    3. CALL_CONCENTRATION - Heavy buying at specific strikes
    4. HIGH_OPTIONS_VOLUME - Overall unusual activity

    Returns list of signals with scores
    """
    signals = []

    if not summary:
        return signals

    now = datetime.utcnow().isoformat()

    # Get key metrics
    call_volume = summary.get('call_volume', 0)
    put_volume = summary.get('put_volume', 0)
    total_volume = summary.get('total_volume', 0)

    # 1. HIGH CALL VOLUME (absolute, not ratio-based)
    # For small caps, 5000+ call contracts is significant
    if call_volume >= 5000:
        # Score: 5K = 0.5, 10K = 0.75, 20K+ = 1.0
        score = min(1.0, 0.5 + (call_volume - 5000) / 30000)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type='HIGH_CALL_VOLUME',
            score=score,
            details={
                'call_volume': call_volume,
                'put_volume': put_volume,
                'interpretation': 'Significant call buying activity detected'
            },
            timestamp=now
        ))

    # 2. LOW PUT/CALL RATIO: Bullish sentiment
    pc_ratio = summary.get('put_call_ratio', 1)
    if pc_ratio < 0.5 and total_volume > 1000:  # Require minimum volume for reliability
        # Lower P/C = more bullish. Score: 0.5 P/C = 0.0, 0.0 P/C = 1.0
        score = min(1.0, (0.5 - pc_ratio) * 2)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type='LOW_PC_RATIO',
            score=score,
            details={
                'put_call_ratio': round(pc_ratio, 3),
                'call_volume': call_volume,
                'put_volume': put_volume,
                'interpretation': 'Strong bullish sentiment - calls dominate puts'
            },
            timestamp=now
        ))

    # 3. CALL CONCENTRATION: Heavy buying at specific strikes
    if summary.get('max_volume_type') == 'CALL' and total_volume > 0:
        call_pct = call_volume / total_volume if total_volume > 0 else 0

        # 70%+ call concentration is significant
        if call_pct >= 0.70:
            signals.append(OptionsFlowSignal(
                ticker=ticker,
                signal_type='CALL_CONCENTRATION',
                score=min(1.0, call_pct),  # 70% = 0.7, 100% = 1.0
                details={
                    'call_percentage': round(call_pct * 100, 1),
                    'max_strike': summary.get('max_volume_strike', 0),
                    'max_expiry': summary.get('max_volume_expiry', ''),
                    'interpretation': 'Heavy call concentration at specific strike (potential target)'
                },
                timestamp=now
            ))

    # 4. HIGH TOTAL VOLUME: Unusual overall activity
    # For small caps, 10K+ total options volume is notable
    if total_volume >= 10000:
        # Score: 10K = 0.5, 25K = 0.75, 50K+ = 1.0
        score = min(1.0, 0.5 + (total_volume - 10000) / 80000)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type='HIGH_OPTIONS_VOLUME',
            score=score,
            details={
                'total_volume': total_volume,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'interpretation': 'High options activity - smart money potentially positioning'
            },
            timestamp=now
        ))

    return signals


# ============================
# SINGLETON
# ============================

# S2-6 FIX: IBKROptionsScanner singleton — avoids creating a new IBKR connection
# on every get_options_flow_score() call (called once per ticker per cycle).
# Before: N tickers × new IBKROptionsScanner() = N connection attempts.
# After:  one shared scanner instance reused across all tickers.
_scanner_singleton: Optional["IBKROptionsScanner"] = None
_scanner_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def _get_options_scanner() -> "IBKROptionsScanner":
    global _scanner_singleton
    with _scanner_lock:
        if _scanner_singleton is None:
            _scanner_singleton = IBKROptionsScanner()
    return _scanner_singleton


# ============================
# MAIN SCANNER
# ============================

def scan_options_flow(tickers: List[str]) -> Dict[str, List[OptionsFlowSignal]]:
    """
    Scan multiple tickers for options flow signals

    Returns: {ticker: [signals]}
    """
    scanner = _get_options_scanner()
    
    if not scanner.ib:
        logger.warning("IBKR not available for options scanning")
        return {}
    
    logger.info(f"🔍 Scanning options flow for {len(tickers)} tickers...")
    
    results = {}
    
    for ticker in tickers:
        try:
            # Get options summary
            summary = scanner.get_options_summary(ticker)
            
            if not summary:
                continue
            
            # Detect signals
            signals = detect_options_signals(ticker, summary)
            
            if signals:
                results[ticker] = signals
                
                # Log significant signals
                for sig in signals:
                    if sig.score >= 0.5:
                        logger.info(
                            f"📊 {ticker}: {sig.signal_type} "
                            f"(score: {sig.score:.2f})"
                        )
            
        except Exception as e:
            logger.debug(f"Options scan error {ticker}: {e}")
            continue
    
    logger.info(f"Options flow: {len(results)} tickers with signals")
    
    return results


def get_options_flow_score(ticker: str) -> Tuple[float, Dict]:
    """
    Get combined options flow score for a single ticker.

    Returns: (score, details)
    """
    scanner = _get_options_scanner()  # S2-6: Use singleton

    if not scanner.ib:
        return 0.0, {}
    
    try:
        summary = scanner.get_options_summary(ticker)
        signals = detect_options_signals(ticker, summary)
        
        if not signals:
            return 0.0, {'status': 'no_signals'}
        
        # Combined score (max of individual signals)
        max_score = max(s.score for s in signals)
        
        # Build details
        details = {
            'signals': [s.signal_type for s in signals],
            'max_score': max_score,
            'summary': summary
        }
        
        return max_score, details
        
    except Exception as e:
        logger.debug(f"Options flow score error {ticker}: {e}")
        return 0.0, {'error': str(e)}


# ============================
# SIMPLIFIED VERSION (Without full chain)
# ============================

def quick_options_check(ticker: str) -> Dict:
    """
    Quick options activity check without full chain
    
    Uses IBKR's built-in scanner capabilities
    """
    try:
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        
        if not ibkr or not ibkr.connected:
            return {}
        
        # Get basic quote which includes some options data
        quote = ibkr.get_quote(ticker, use_cache=False)
        
        if not quote:
            return {}
        
        # Check for unusual activity indicators
        volume = quote.get('volume', 0)
        avg_volume = quote.get('avg_volume', 0)
        
        result = {
            'ticker': ticker,
            'has_unusual_volume': volume > avg_volume * 2 if avg_volume > 0 else False,
            'volume_ratio': volume / avg_volume if avg_volume > 0 else 0
        }
        
        return result
        
    except Exception as e:
        logger.debug(f"Quick options check error {ticker}: {e}")
        return {}


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIONS FLOW DETECTOR - TEST")
    print("=" * 60)
    
    test_tickers = ["NVDA", "AMD", "TSLA"]
    
    print(f"\nTesting with: {test_tickers}")
    
    results = scan_options_flow(test_tickers)
    
    print(f"\n📊 Results:")
    for ticker, signals in results.items():
        print(f"\n  {ticker}:")
        for sig in signals:
            print(f"    - {sig.signal_type}: {sig.score:.2f}")
            print(f"      {sig.details.get('interpretation', '')}")
