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

V7.1 Enhancements:
- Automatic reconnection with exponential backoff
- Heartbeat thread (reqCurrentTime every 30s)
- Connection state machine (DISCONNECTED / CONNECTING / CONNECTED / RECONNECTING / FAILED)
- Uptime tracking and statistics
- Alert callback on connection events

Requirements:
- IB Gateway or TWS running
- Market data subscription (Level 1)
- ib_insync library: pip install ib_insync
"""

import time
import threading
from enum import Enum

from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.cache import Cache

from config import (
    USE_IBKR_DATA,
    IBKR_HOST,
    IBKR_PORT,
    IBKR_CLIENT_ID,
)

logger = get_logger("IBKR_CONNECTOR")

cache = Cache(ttl=5)  # 5 second cache for quotes


# ============================
# CONNECTION STATES
# ============================

class ConnectionState(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class IBKRConnector:
    """
    IBKR connection for real-time Level 1 market data
    with automatic reconnection and health monitoring.

    Usage:
        ibkr = IBKRConnector()
        ibkr.connect()
        quote = ibkr.get_quote('AAPL')
        bars = ibkr.get_bars('AAPL', duration='1 D', bar_size='5 mins')
    """

    # Reconnection backoff delays (seconds)
    RECONNECT_DELAYS = [0, 2, 5, 15, 30]
    MAX_RECONNECT_ATTEMPTS = 5
    HEARTBEAT_INTERVAL = 30  # seconds

    def __init__(self, host=None, port=None, client_id=None, readonly=True):
        """
        Initialize IBKR connector

        Args:
            host: IB Gateway/TWS host (default: from config)
            port: 7497 (paper), 7496 (live), 4001/4002 (gateway)
            client_id: Unique client ID (1-999)
            readonly: True for read-only mode (safety)
        """
        self.ib = IB()
        self.host = host or IBKR_HOST
        self.port = port or IBKR_PORT
        self.client_id = client_id or IBKR_CLIENT_ID
        self.readonly = readonly
        self.connected = False

        # Connection state machine
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()

        # Reconnection tracking
        self._reconnect_attempts = 0

        # Heartbeat
        self._heartbeat_thread = None
        self._heartbeat_running = False
        self._last_heartbeat = 0
        self._heartbeat_latency_ms = 0

        # Statistics
        self._connected_since = None
        self._total_disconnections = 0
        self._total_reconnections = 0
        self._last_disconnection_time = None
        self._last_downtime_seconds = 0

        # Alert callback (set by system_guardian or main)
        self._on_state_change = None

        # Cache for contracts
        self.contract_cache = {}

        # Register ib_insync disconnect handler
        self.ib.disconnectedEvent += self._on_disconnect

    # ============================
    # CONNECTION STATE
    # ============================

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: ConnectionState):
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        if old_state != new_state:
            logger.info(f"IBKR state: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                try:
                    self._on_state_change(old_state, new_state)
                except Exception as e:
                    logger.debug(f"State change callback error: {e}")

    def set_state_change_callback(self, callback):
        """Set callback for connection state changes: callback(old_state, new_state)"""
        self._on_state_change = callback

    # ============================
    # CONNECTION STATS
    # ============================

    def get_connection_stats(self) -> dict:
        """Get connection statistics for monitoring and dashboard"""
        uptime_seconds = 0
        if self._connected_since and self.connected:
            uptime_seconds = (datetime.now() - self._connected_since).total_seconds()

        return {
            "state": self.state.value,
            "connected": self.connected,
            "host": f"{self.host}:{self.port}",
            "connected_since": self._connected_since.isoformat() if self._connected_since else None,
            "uptime_seconds": uptime_seconds,
            "heartbeat_latency_ms": self._heartbeat_latency_ms,
            "last_heartbeat": self._last_heartbeat,
            "total_disconnections": self._total_disconnections,
            "total_reconnections": self._total_reconnections,
            "last_downtime_seconds": self._last_downtime_seconds,
            "reconnect_attempts": self._reconnect_attempts,
        }

    # ============================
    # CONNECT / DISCONNECT
    # ============================

    def connect(self, timeout=10):
        """
        Connect to IBKR

        Returns:
            bool: True if connected
        """
        self._set_state(ConnectionState.CONNECTING)

        try:
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                readonly=self.readonly,
                timeout=timeout
            )
            self.connected = True
            self._connected_since = datetime.now()
            self._reconnect_attempts = 0
            self._set_state(ConnectionState.CONNECTED)
            self._start_heartbeat()
            logger.info(f"Connected to IBKR ({self.host}:{self.port}) - READ ONLY")
            return True

        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            self.connected = False
            self._set_state(ConnectionState.DISCONNECTED)
            return False

    def disconnect(self):
        """Disconnect from IBKR cleanly"""
        self._stop_heartbeat()
        if self.connected:
            try:
                self.ib.disconnect()
            except Exception:
                pass
            self.connected = False
            self._set_state(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from IBKR")

    def _on_disconnect(self):
        """Called by ib_insync when connection drops unexpectedly"""
        was_connected = self.connected
        self.connected = False

        if was_connected:
            self._total_disconnections += 1
            self._last_disconnection_time = datetime.now()
            logger.warning("IBKR connection lost unexpectedly")
            self._set_state(ConnectionState.RECONNECTING)

            # Launch reconnection in background thread
            threading.Thread(
                target=self._reconnect_loop,
                daemon=True,
                name="ibkr-reconnect"
            ).start()

    # ============================
    # AUTOMATIC RECONNECTION
    # ============================

    def _reconnect_loop(self):
        """
        Attempt reconnection with exponential backoff.
        Runs in a background thread.
        """
        self._stop_heartbeat()
        self._reconnect_attempts = 0

        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            self._reconnect_attempts = attempt + 1
            delay = self.RECONNECT_DELAYS[min(attempt, len(self.RECONNECT_DELAYS) - 1)]

            if delay > 0:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.MAX_RECONNECT_ATTEMPTS} in {delay}s...")
                time.sleep(delay)
            else:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.MAX_RECONNECT_ATTEMPTS}...")

            try:
                # Create fresh IB instance (old one may be in bad state)
                self.ib = IB()
                self.ib.disconnectedEvent += self._on_disconnect

                self.ib.connect(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    readonly=self.readonly,
                    timeout=15
                )

                self.connected = True
                self._total_reconnections += 1
                self._reconnect_attempts = 0
                self.contract_cache.clear()

                # Calculate downtime
                if self._last_disconnection_time:
                    self._last_downtime_seconds = (datetime.now() - self._last_disconnection_time).total_seconds()

                self._connected_since = datetime.now()
                self._set_state(ConnectionState.CONNECTED)
                self._start_heartbeat()

                logger.info(
                    f"IBKR reconnected after {self._last_downtime_seconds:.0f}s downtime "
                    f"(attempt {attempt + 1})"
                )
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                continue

        # All attempts exhausted
        self._set_state(ConnectionState.FAILED)
        logger.error(
            f"IBKR reconnection FAILED after {self.MAX_RECONNECT_ATTEMPTS} attempts. "
            f"Falling back to Finnhub."
        )

    def force_reconnect(self):
        """Force a reconnection attempt (callable from dashboard or guardian)"""
        logger.info("Force reconnect requested")
        self._stop_heartbeat()
        self.connected = False

        try:
            self.ib.disconnect()
        except Exception:
            pass

        self._set_state(ConnectionState.RECONNECTING)
        threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="ibkr-force-reconnect"
        ).start()

    # ============================
    # HEARTBEAT
    # ============================

    def _start_heartbeat(self):
        """Start heartbeat monitoring thread"""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="ibkr-heartbeat"
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop heartbeat thread"""
        self._heartbeat_running = False

    def _heartbeat_loop(self):
        """Periodic heartbeat to detect silent disconnections"""
        while self._heartbeat_running and self.connected:
            try:
                start = time.time()
                self.ib.reqCurrentTime()
                elapsed_ms = (time.time() - start) * 1000

                self._last_heartbeat = time.time()
                self._heartbeat_latency_ms = elapsed_ms

                if elapsed_ms > 2000:
                    logger.warning(f"IBKR heartbeat slow: {elapsed_ms:.0f}ms")

            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                # Connection likely dead - trigger reconnection
                if self.connected:
                    self.connected = False
                    self._on_disconnect()
                break

            time.sleep(self.HEARTBEAT_INTERVAL)

    # ============================
    # ENSURE CONNECTED (guard)
    # ============================

    def _ensure_connected(self) -> bool:
        """
        Check connection before any data request.
        Returns True if connected, False if not available.
        """
        if self.connected and self.state == ConnectionState.CONNECTED:
            return True

        if self.state in (ConnectionState.RECONNECTING, ConnectionState.FAILED):
            logger.debug(f"IBKR {self.state.value}, request skipped")
            return False

        return False

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
        if not self._ensure_connected():
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
        if not self._ensure_connected():
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
        if not self._ensure_connected():
            return 0

        try:
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
        if not self._ensure_connected():
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
        if not self._ensure_connected():
            return []

        try:
            scanner = ScannerSubscription()
            scanner.instrument = 'STK'
            scanner.locationCode = 'STK.US'
            scanner.scanCode = 'TOP_PERC_GAIN'

            scanner.abovePrice = 1.0
            scanner.belowPrice = 20.0
            scanner.aboveVolume = 500000

            scan_data = self.ib.reqScannerData(scanner)
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

        # Check heartbeat freshness (stale if > 2 intervals without heartbeat)
        if self._last_heartbeat > 0:
            stale_threshold = self.HEARTBEAT_INTERVAL * 2
            if (time.time() - self._last_heartbeat) > stale_threshold:
                logger.warning("IBKR heartbeat stale")
                return False

        try:
            self.ib.reqCurrentTime()
            return True
        except:
            self.connected = False
            return False


# ============================
# SINGLETON INSTANCE
# ============================

_ibkr_instance = None
_ibkr_lock = threading.Lock()


def get_ibkr():
    """Get or create singleton IBKR instance"""
    global _ibkr_instance

    if not USE_IBKR_DATA:
        return None

    with _ibkr_lock:
        if _ibkr_instance is None:
            _ibkr_instance = IBKRConnector()
            _ibkr_instance.connect()

    return _ibkr_instance


# ============================
# TESTING
# ============================

if __name__ == "__main__":
    print("Testing IBKR Connector (Level 1 + Reconnection)...")

    ibkr = IBKRConnector()

    if ibkr.connect():
        print("Connected to IBKR\n")

        print("Connection stats:")
        stats = ibkr.get_connection_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()

        print("Testing real-time quote:")
        quote = ibkr.get_quote('AAPL')
        if quote:
            print(f"  AAPL: Last=${quote['last']}, Bid=${quote['bid']}, Ask=${quote['ask']}")
            print(f"  Volume: {quote['volume']:,}")
            print()

        print("Testing historical bars:")
        bars = ibkr.get_bars('AAPL', '1 D', '5 mins')
        if bars is not None:
            print(f"  Retrieved {len(bars)} bars")
            print()

        print("Testing account info:")
        capital = ibkr.get_account_capital()
        print(f"  Available capital: ${capital:,.2f}")
        print()

        ibkr.disconnect()
        print("Test complete")
    else:
        print("Connection failed")
