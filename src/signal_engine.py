from utils.logger import get_logger
from utils.data_validator import validate_signal

from src.scoring.monster_score import compute_monster_score

from config import (
    BUY_THRESHOLD,
    BUY_STRONG_THRESHOLD
)

logger = get_logger("SIGNAL_ENGINE")


# ============================
# Signal logic
# ============================

def generate_signal(ticker):
    score_data = compute_monster_score(ticker)

    if not score_data:
        return None

    score = score_data["monster_score"]

    if score >= BUY_STRONG_THRESHOLD:
        signal_type = "BUY_STRONG"

    elif score >= BUY_THRESHOLD:
        signal_type = "BUY"

    else:
        signal_type = "HOLD"

    # Confidence scaled from score
    confidence = min(1.0, score)

    signal = {
        "ticker": ticker,
        "signal": signal_type,
        "confidence": confidence,
        "monster_score": score,
        "components": score_data["components"]
    }

    if not validate_signal(signal):
        return None

    return signal


# ============================
# Batch helper
# ============================

def generate_many(tickers, limit=None):
    signals = []

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        s = generate_signal(t)
        if s:
            signals.append(s)

    logger.info(f"Generated {len(signals)} signals")

    return signals


if __name__ == "__main__":
    print(generate_signal("AAPL"))
