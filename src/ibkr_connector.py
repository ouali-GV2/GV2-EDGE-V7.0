"""
IBKR CONNECTOR - Level 1 Market Data Integration
================================================

Connects to Interactive Brokers (read-only) for real-time Level 1 data.

Level 1 provides:
- Real-time prices (last, bid, ask)
- Volume & VWAP
- Pre-market & After-hours data
- Historical bars (unlimited)

Replaces Finnhub for superior data quality on small caps.

Requirements:
- IB Gateway or TWS running
- Market data subscription (Level 1)
- ib_insync library: pip install ib_insync
"""

from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("IBKR_CONNECTOR")

cache = Cache(ttl=5)  # 5 second cache for quotes


class IBKRConnector:
    """
    IBKR connection for real-time Level 1 market data
    
    Usage:
        ibkr = IBKRConnector()
        ibkr.connect()
        quote = ibkr.get_quote('AAPL')
        bars = ibkr.get_bars('AAPL', duration='1 D', bar_size='5 mins')
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, readonly=True):
        """
        Initialize IBKR connector
        
        Args:
            host: IB Gateway/TWS host (default: localhost)
            port: 7497 (paper), 7496 (live), 4001/4002 (gateway)
            client_id: Unique client ID (1-999)
            readonly: True for read-only mode (safety)
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.readonly = readonly
        self.connected = False
        
        # Cache for contracts
        self.contract_cache = {}
    
    def connect(self, timeout=10):
        """
        Connect to IBKR
        
        Returns:
            bool: True if connected
        """
        try:
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id,
                readonly=self.readonly,
                timeout=timeout
            )
            self.connected = True
            logger.info(f"✅ Connected to IBKR ({self.host}:{self.port}) - READ ONLY")
            return True
            
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def _get_contract(self, ticker):
        """
        Get or create contract for ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
        
        Returns:
            Contract object
        """
        if ticker in self.contract_cache:
            return self.contract_cache[ticker]
        
        # Create stock contract
        contract = Stock(ticker, 'SMART', 'USD')
        
        # Qualify contract (get full details from IBKR)
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self.contract_cache[ticker] = qualified[0]
                return qualified[0]
        except:
            pass
        
        # Fallback: unqualified contract
        self.contract_cache[ticker] = contract
        return contract
    
    # ============================
    # REAL-TIME QUOTES (Level 1)
    # ============================
    
    def get_quote(self, ticker, use_cache=True):
        """
        Get real-time Level 1 quote
        
        Args:
            ticker: Stock symbol
            use_cache: Use 5-second cache
        
        Returns:
            dict with quote data or None
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return None
        
        # Check cache
        cache_key = f"quote_{ticker}"
        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        try:
            contract = self._get_contract(ticker)
            
            # Request market data
            ticker_data = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for data (max 2 seconds)
            self.ib.sleep(2)
            
            # Extract Level 1 data
            quote = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                
                # Prices
                "last": ticker_data.last if ticker_data.last == ticker_data.last else None,
                "bid": ticker_data.bid if ticker_data.bid == ticker_data.bid else None,
                "ask": ticker_data.ask if ticker_data.ask == ticker_data.ask else None,
                "close": ticker_data.close if ticker_data.close == ticker_data.close else None,
                
                # Sizes
                "bid_size": ticker_data.bidSize if hasattr(ticker_data, 'bidSize') else None,
                "ask_size": ticker_data.askSize if hasattr(ticker_data, 'askSize') else None,
                
                # Volume & VWAP
                "volume": ticker_data.volume if ticker_data.volume == ticker_data.volume else None,
                "vwap": ticker_data.vwap if hasattr(ticker_data, 'vwap') and ticker_data.vwap == ticker_data.vwap else None,
                
                # Daily stats
                "open": ticker_data.open if hasattr(ticker_data, 'open') else None,
                "high": ticker_data.high if ticker_data.high == ticker_data.high else None,
                "low": ticker_data.low if ticker_data.low == ticker_data.low else None,
            }
            
            # Calculate spread
            if quote['bid'] and quote['ask']:
                quote['spread'] = quote['ask'] - quote['bid']
                quote['spread_pct'] = (quote['spread'] / quote['last']) * 100 if quote['last'] else None
            
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
            
            # Cache quote
            cache.set(cache_key, quote)
            
            return quote
            
        except Exception as e:
            logger.error(f"Failed to get quote for {ticker}: {e}")
            return None
    
    # ============================
    # HISTORICAL BARS
    # ============================
    
    def get_bars(self, ticker, duration='1 D', bar_size='5 mins', use_rth=False):
        """
        Get historical bars (IBKR has unlimited historical data)
        
        Args:
            ticker: Stock symbol
            duration: '1 D', '2 D', '1 W', '1 M'
            bar_size: '1 min', '5 mins', '15 mins', '1 hour', '1 day'
            use_rth: Only regular trading hours (9:30-16:00)
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return None
        
        try:
            contract = self._get_contract(ticker)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=use_rth,  # False = include pre/post market
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No bars returned for {ticker}")
                return None
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Add ticker column
            df['ticker'] = ticker
            
            logger.info(f"Retrieved {len(df)} bars for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get bars for {ticker}: {e}")
            return None
    
    # ============================
    # PRE-MARKET / AFTER-HOURS
    # ============================
    
    def get_premarket_data(self, ticker):
        """
        Get pre-market data (4:00 AM - 9:30 AM ET)
        
        Returns:
            dict with PM high, low, volume
        """
        # Get bars for today pre-market only
        bars = self.get_bars(ticker, duration='1 D', bar_size='1 min', use_rth=False)
        
        if bars is None or bars.empty:
            return None
        
        # Filter pre-market (4:00-9:30 AM ET)
        today = datetime.now().date()
        
        # Convert to ET timezone
        bars['datetime'] = pd.to_datetime(bars['date'])
        
        # Filter pre-market hours
        pm_bars = bars[
            (bars['datetime'].dt.date == today) &
            (bars['datetime'].dt.hour >= 4) &
            (bars['datetime'].dt.hour < 9) |
            ((bars['datetime'].dt.hour == 9) & (bars['datetime'].dt.minute < 30))
        ]
        
        if pm_bars.empty:
            return None
        
        return {
            "ticker": ticker,
            "pm_high": pm_bars['high'].max(),
            "pm_low": pm_bars['low'].min(),
            "pm_volume": pm_bars['volume'].sum(),
            "pm_open": pm_bars.iloc[0]['open'],
            "pm_close": pm_bars.iloc[-1]['close'],
            "pm_bars_count": len(pm_bars)
        }
    
    # ============================
    # ACCOUNT INFO (Read-Only)
    # ============================
    
    def get_account_capital(self):
        """
        Get available capital from IBKR account
        
        Returns:
            float: Available cash
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return 0
        
        try:
            # Request account summary
            account_summary = self.ib.accountSummary()
            
            for item in account_summary:
                if item.tag == 'TotalCashValue':
                    capital = float(item.value)
                    logger.info(f"IBKR Account Capital: ${capital:,.2f}")
                    return capital
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get account capital: {e}")
            return 0
    
    def get_account_info(self):
        """
        Get full account information
        
        Returns:
            dict with account details
        """
        if not self.connected:
            return None
        
        try:
            summary = self.ib.accountSummary()
            
            info = {}
            for item in summary:
                info[item.tag] = item.value
            
            return {
                "total_cash": float(info.get('TotalCashValue', 0)),
                "net_liquidation": float(info.get('NetLiquidation', 0)),
                "buying_power": float(info.get('BuyingPower', 0)),
                "available_funds": float(info.get('AvailableFunds', 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    # ============================
    # UNIVERSE SCANNING
    # ============================
    
    def scan_top_gainers(self, num_results=100):
        """
        Scan for top % gainers (IBKR Scanner)
        
        Args:
            num_results: Number of results to return
        
        Returns:
            List of ticker symbols
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return []
        
        try:
            # Create scanner subscription
            scanner = ScannerSubscription()
            scanner.instrument = 'STK'
            scanner.locationCode = 'STK.US'
            scanner.scanCode = 'TOP_PERC_GAIN'
            
            # Filters for small caps
            scanner.abovePrice = 1.0
            scanner.belowPrice = 20.0
            scanner.aboveVolume = 500000
            
            # Request scan
            scan_data = self.ib.reqScannerData(scanner)
            
            # Extract tickers
            tickers = [item.contract.symbol for item in scan_data[:num_results]]
            
            logger.info(f"Scanner found {len(tickers)} top gainers")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Scanner failed: {e}")
            return []
    
    # ============================
    # HEALTH CHECK
    # ============================
    
    def is_healthy(self):
        """
        Check if connection is healthy
        
        Returns:
            bool
        """
        if not self.connected:
            return False
        
        try:
            # Test with simple request
            self.ib.reqCurrentTime()
            return True
        except:
            self.connected = False
            return False


# ============================
# SINGLETON INSTANCE
# ============================

_ibkr_instance = None

def get_ibkr():
    """Get or create singleton IBKR instance"""
    global _ibkr_instance
    
    if _ibkr_instance is None:
        _ibkr_instance = IBKRConnector()
        _ibkr_instance.connect()
    
    return _ibkr_instance


# ============================
# TESTING
# ============================

if __name__ == "__main__":
    print("Testing IBKR Connector (Level 1)...")
    
    # Connect
    ibkr = IBKRConnector()
    
    if ibkr.connect():
        print("✅ Connected to IBKR\n")
        
        # Test quote
        print("Testing real-time quote:")
        quote = ibkr.get_quote('AAPL')
        if quote:
            print(f"  AAPL: Last=${quote['last']}, Bid=${quote['bid']}, Ask=${quote['ask']}")
            print(f"  Spread: ${quote['spread']:.2f} ({quote['spread_pct']:.2f}%)")
            print(f"  Volume: {quote['volume']:,}")
            print()
        
        # Test historical bars
        print("Testing historical bars:")
        bars = ibkr.get_bars('AAPL', '1 D', '5 mins')
        if bars is not None:
            print(f"  Retrieved {len(bars)} bars")
            print(f"  Latest: {bars.iloc[-1]['date']} - Close: ${bars.iloc[-1]['close']}")
            print()
        
        # Test account
        print("Testing account info:")
        capital = ibkr.get_account_capital()
        print(f"  Available capital: ${capital:,.2f}")
        print()
        
        # Disconnect
        ibkr.disconnect()
        print("✅ Test complete")
    else:
        print("❌ Connection failed")
