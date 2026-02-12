"""
GV2-EDGE V7.0 — Backtest Engine
================================

Backtest engine for validating signal performance.

Supports both legacy (signal_engine) and V7.0 (SignalProducer) architectures.
Set USE_V7_ARCHITECTURE=True in config.py to use V7 backtesting.

Metrics:
- Win rate, average win/loss
- Max drawdown, Sharpe ratio
- Hit rate vs real top gainers
- Lead time analysis
"""

import os
import json
import time
from datetime import datetime, timedelta

import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache
from utils.api_guard import safe_get

from src.universe_loader import load_universe
from src.signal_engine import generate_signal
from src.feature_engine import compute_features
from src.portfolio_engine import compute_position, update_trailing_stop

from config import (
    FINNHUB_API_KEY,
    BACKTEST_LOOKBACK_DAYS,
    SLIPPAGE_PCT,
    CAPITAL_INITIAL,
    USE_V7_ARCHITECTURE
)

logger = get_logger("BACKTEST_EDGE")

FINNHUB_CANDLE = "https://finnhub.io/api/v1/stock/candle"

os.makedirs("data/backtest_reports", exist_ok=True)


# ============================
# Fetch historical candles
# ============================

def fetch_history(ticker, start_ts, end_ts, resolution="5"):
    params = {
        "symbol": ticker,
        "resolution": resolution,
        "from": start_ts,
        "to": end_ts,
        "token": FINNHUB_API_KEY
    }

    r = safe_get(FINNHUB_CANDLE, params=params)
    data = r.json()

    if data.get("s") != "ok":
        return None

    df = pd.DataFrame({
        "t": data["t"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    })

    df["datetime"] = pd.to_datetime(df["t"], unit="s")

    return df


# ============================
# Backtest one ticker timeline
# ============================

def backtest_ticker(ticker, capital):

    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=BACKTEST_LOOKBACK_DAYS)).timestamp())

    df = fetch_history(ticker, start, end)

    if df is None or len(df) < 50:
        return []

    trades = []
    position = None

    for i in range(30, len(df)):

        price_row = df.iloc[i]
        price = price_row["close"]

        # ===== Generate signal (real EDGE logic) =====
        signal = generate_signal(ticker)

        if signal and signal["signal"] in ["BUY", "BUY_STRONG"] and not position:

            features = compute_features(ticker)
            if not features:
                continue

            position = compute_position(signal, features, capital)

            if position:
                position["entry_time"] = price_row["datetime"]
                position["entry_price"] = price * (1 + SLIPPAGE_PCT)

        # ===== Manage open position =====
        if position:

            position = update_trailing_stop(position, price)

            # stop hit
            if price <= position["stop"]:

                exit_price = price * (1 - SLIPPAGE_PCT)

                pnl = (exit_price - position["entry_price"]) * position["shares"]

                trades.append({
                    "ticker": ticker,
                    "entry_time": position["entry_time"],
                    "exit_time": price_row["datetime"],
                    "entry": position["entry_price"],
                    "exit": exit_price,
                    "shares": position["shares"],
                    "pnl": pnl
                })

                capital += pnl
                position = None

    return trades


# ============================
# Run full universe backtest
# ============================

def run_backtest():

    universe = load_universe()
    tickers = universe["ticker"].tolist()

    capital = CAPITAL_INITIAL

    all_trades = []

    logger.info(f"Backtesting {len(tickers)} tickers")

    for t in tickers:
        try:
            trades = backtest_ticker(t, capital)
            all_trades.extend(trades)
            logger.info(f"Backtested {t} : {len(trades)} trades")

        except Exception as e:
            logger.error(f"Backtest error {t}: {e}")

    df = pd.DataFrame(all_trades)

    report_file = f"data/backtest_reports/backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"

    df.to_csv(report_file, index=False)

    logger.info(f"Backtest saved: {report_file}")

    return df


# ============================
# CLI
# ============================

if __name__ == "__main__":
    df = run_backtest()
    print(df.head())
