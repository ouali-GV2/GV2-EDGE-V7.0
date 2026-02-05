import os
import pandas as pd
from datetime import datetime

from utils.logger import get_logger

BACKTEST_DIR = "data/backtest_reports"

logger = get_logger("REGIME_TEST")


# ============================
# Detect market regime (simple & robust)
# ============================

def detect_regime(returns):
    """
    Simple regime classifier:
    - bull: positive mean return
    - bear: negative mean return
    - high_vol: high std dev
    """

    mean_r = returns.mean()
    vol = returns.std()

    if vol > abs(mean_r) * 2:
        return "HIGH_VOL"

    if mean_r > 0:
        return "BULL"

    return "BEAR"


# ============================
# Load all backtests
# ============================

def load_backtests():
    dfs = []

    for f in os.listdir(BACKTEST_DIR):
        if f.startswith("backtest_") and f.endswith(".csv"):
            df = pd.read_csv(f"{BACKTEST_DIR}/{f}")
            dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


# ============================
# Regime analysis
# ============================

def run_regime_test():

    trades = load_backtests()

    if trades is None or trades.empty:
        logger.warning("No backtest data found")
        return None

    trades["return"] = trades["pnl"]

    regimes = {}

    # group by month (proxy for regime segments)
    trades["month"] = pd.to_datetime(trades["exit_time"]).dt.to_period("M")

    for m, df in trades.groupby("month"):

        regime = detect_regime(df["return"])

        if regime not in regimes:
            regimes[regime] = []

        regimes[regime].append({
            "month": str(m),
            "total_pnl": df["return"].sum(),
            "winrate": (df["return"] > 0).mean(),
            "trades": len(df)
        })

    # summarize
    summary = {}

    for regime, rows in regimes.items():
        rdf = pd.DataFrame(rows)

        summary[regime] = {
            "avg_pnl": round(rdf["total_pnl"].mean(), 2),
            "avg_winrate": round(rdf["winrate"].mean(), 3),
            "avg_trades": int(rdf["trades"].mean())
        }

    out_path = f"{BACKTEST_DIR}/regime_test_summary.json"
    pd.Series(summary).to_json(out_path, indent=2)

    logger.info(f"Regime test saved: {out_path}")

    return summary


# ============================
# CLI
# ============================

if __name__ == "__main__":
    print(run_regime_test())
