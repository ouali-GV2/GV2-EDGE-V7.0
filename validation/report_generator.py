import json
from datetime import datetime, timezone

from utils.logger import get_logger

logger = get_logger("REPORT_GENERATOR")


# ============================
# Generate consolidated report
# ============================

def generate_report(results_dict, output_path):

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "summary": {}
    }

    # Walk forward
    wf = results_dict.get("walk_forward")
    if wf is not None and not wf.empty:
        report["summary"]["walk_forward"] = {
            "avg_pnl": round(float(wf["total_pnl"].mean()), 2),
            "avg_winrate": round(float(wf["winrate"].mean()), 3),
            "segments": len(wf)
        }

    # Regime test
    rt = results_dict.get("regime_test")
    if rt:
        report["summary"]["regime_performance"] = rt

    # Stress test
    st = results_dict.get("stress_test")
    if st:
        report["summary"]["stress_metrics"] = st

    # Monte Carlo
    mc = results_dict.get("monte_carlo")
    if mc:
        report["summary"]["monte_carlo"] = mc

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save readable text report
    txt_path = output_path.replace(".json", ".txt")

    with open(txt_path, "w") as f:
        f.write("GV2-EDGE VALIDATION REPORT\n")
        f.write("=" * 40 + "\n\n")

        for section, content in report["summary"].items():
            f.write(f"{section.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(json.dumps(content, indent=2))
            f.write("\n\n")

    logger.info(f"Validation report saved: {output_path}")
    logger.info(f"Readable report saved: {txt_path}")

    return report


# ============================
# CLI test
# ============================

if __name__ == "__main__":
    sample = {
        "walk_forward": None,
        "regime_test": {"BULL": {"avg_pnl": 100}},
        "stress_test": {"max_drawdown_real": -500},
        "monte_carlo": {"avg_final_equity": 2000}
    }

    print(generate_report(sample, "data/backtest_reports/test_report.json"))
