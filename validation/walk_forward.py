from datetime import datetime, timedelta, timezone
import pandas as pd
import os

from utils.logger import get_logger
from backtests.backtest_engine_edge import run_backtest

logger = get_logger("WALK_FORWARD")

WALK_WINDOW_DAYS = 30      # durée backtest par segment
STEP_DAYS = 15            # pas d’avancement
TOTAL_MONTHS = 6          # profondeur validation

REPORT_DIR = "data/backtest_reports"


# ============================
# Walk forward core
# ============================

def run_walk_forward():

    logger.info("Starting walk-forward validation")

    results = []

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=TOTAL_MONTHS * 30)

    current = start_date

    while current + timedelta(days=WALK_WINDOW_DAYS) <= end_date:

        period_start = current
        period_end = current + timedelta(days=WALK_WINDOW_DAYS)

        logger.info(
            f"Testing period {period_start.date()} → {period_end.date()}"
        )

        # Backtest EDGE (uses historical windows internally)
        df_trades = run_backtest()

        if df_trades is None or df_trades.empty:
            logger.warning("No trades in this window")
            current += timedelta(days=STEP_DAYS)
            continue

        total_pnl = df_trades["pnl"].sum()
        winrate = (df_trades["pnl"] > 0).mean()

        results.append({
            "start": period_start.strftime("%Y-%m-%d"),
            "end": period_end.strftime("%Y-%m-%d"),
            "total_pnl": round(total_pnl, 2),
            "winrate": round(winrate, 3),
            "trades": len(df_trades)
        })

        current += timedelta(days=STEP_DAYS)

    df = pd.DataFrame(results)

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f"{REPORT_DIR}/walk_forward.csv"
    df.to_csv(path, index=False)

    logger.info(f"Walk-forward results saved: {path}")

    return df


# ============================
# CLI
# ============================

if __name__ == "__main__":
    print(run_walk_forward())
