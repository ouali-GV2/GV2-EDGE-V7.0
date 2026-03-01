import os
from datetime import datetime, timezone

from utils.logger import get_logger

from validation.walk_forward import run_walk_forward
from validation.regime_test import run_regime_test
from validation.stress_test import run_stress_test
from validation.monte_carlo import run_monte_carlo
from validation.report_generator import generate_report

logger = get_logger("VALIDATION_ENGINE")

REPORT_DIR = "data/backtest_reports"


# ============================
# Main validation pipeline
# ============================

def run_full_validation():

    logger.info("Starting full EDGE validation suite...")

    results = {}

    logger.info("Running walk-forward...")
    results["walk_forward"] = run_walk_forward()

    logger.info("Running regime test...")
    results["regime_test"] = run_regime_test()

    logger.info("Running stress test...")
    results["stress_test"] = run_stress_test()

    logger.info("Running Monte Carlo...")
    results["monte_carlo"] = run_monte_carlo()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report_file = f"{REPORT_DIR}/validation_summary_{timestamp}.json"

    os.makedirs(REPORT_DIR, exist_ok=True)

    generate_report(results, report_file)

    logger.info(f"Validation completed. Report saved: {report_file}")

    return results


# ============================
# CLI
# ============================

if __name__ == "__main__":
    res = run_full_validation()
    print(res)
