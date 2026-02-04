from utils.logger import get_logger

logger = get_logger("ENSEMBLE_ENGINE")


# ============================
# Soft confluence logic
# ============================

def apply_confluence(signal):
    """
    Soft ensemble:
    doesn't block signals
    only boosts or soft reduces confidence
    """

    score = signal["monster_score"]
    components = signal.get("components", {})

    boost = 1.0

    # Event strong catalyst
    if components.get("event", 0) > 0.6:
        boost += 0.15

    # Strong momentum + volume
    if components.get("momentum", 0) > 0.6 and components.get("volume", 0) > 0.6:
        boost += 0.15

    # Squeeze pressure
    if components.get("squeeze", 0) > 0.7:
        boost += 0.1

    # Premarket gap strong
    if components.get("pm_gap", 0) > 0.5:
        boost += 0.1

    new_confidence = min(1.0, signal["confidence"] * boost)

    signal["confidence"] = new_confidence
    signal["ensemble_boost"] = round(boost, 2)

    return signal


def apply_many(signals):
    enhanced = []

    for s in signals:
        enhanced.append(apply_confluence(s))

    logger.info(f"Confluence applied to {len(enhanced)} signals")

    return enhanced


if __name__ == "__main__":
    test = {
        "monster_score": 0.7,
        "confidence": 0.7,
        "components": {
            "event": 0.8,
            "momentum": 0.7,
            "volume": 0.8,
            "squeeze": 0.6,
            "pm_gap": 0.7
        }
    }

    print(apply_confluence(test))
