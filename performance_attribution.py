import os
import json
import pandas as pd

from utils.logger import get_logger

ATTRIB_FILE = "data/backtest_reports/performance_attribution.json"
TRADE_FILES_DIR = "data/backtest_reports"

logger = get_logger("PERF_ATTRIBUTION")

os.makedirs("data/backtest_reports", exist_ok=True)


# ============================
# Load all backtest trades
# ============================

def load_all_trades():
    dfs = []

    for file in os.listdir(TRADE_FILES_DIR):
        if file.startswith("backtest_") and file.endswith(".csv"):
            path = os.path.join(TRADE_FILES_DIR, file)
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        logger.warning("No backtest trades found")
        return None

    return pd.concat(dfs, ignore_index=True)


# ============================
# Attribution core
# ============================

def compute_attribution(trades_df):
    """
    Requires backtest trades to include:
    event_score, momentum_score, volume_score,
    squeeze_score, pm_gap_score, pnl
    """

    components = [
        "event_score",
        "momentum_score",
        "volume_score",
        "vwap_score",
        "squeeze_score",
        "pm_gap_score"
    ]

    attribution = {}

    for comp in components:
        if comp not in trades_df.columns:
            continue

        # correlation proxy as contribution
        corr = trades_df[comp].corr(trades_df["pnl"])

        if pd.isna(corr):
            corr = 0

        attribution[comp.replace("_score", "")] = round(float(corr), 4)

    return attribution


# ============================
# Save attribution
# ============================

def save_attribution(data):
    with open(ATTRIB_FILE, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Attribution saved: {ATTRIB_FILE}")


# ============================
# Main runner
# ============================

def run_attribution():

    trades = load_all_trades()

    if trades is None or trades.empty:
        logger.warning("No trades for attribution")
        return None

    attribution = compute_attribution(trades)

    save_attribution(attribution)

    logger.info(f"Attribution results: {attribution}")

    return attribution


# ============================
# CLI
# ============================

if __name__ == "__main__":
    result = run_attribution()
    print(result)
