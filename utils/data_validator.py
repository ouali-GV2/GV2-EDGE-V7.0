from utils.logger import get_logger

logger = get_logger("DATA_VALIDATOR")


def validate_price_data(price):
    """
    Validate live price structure
    """
    if price is None:
        return False

    required_keys = ["open", "high", "low", "close", "volume"]

    for k in required_keys:
        if k not in price:
            logger.warning(f"Missing price field: {k}")
            return False

        if price[k] is None:
            logger.warning(f"Null price field: {k}")
            return False

    if price["close"] <= 0:
        logger.warning("Invalid close price")
        return False

    if price["volume"] < 0:
        logger.warning("Invalid volume")
        return False

    return True


def validate_event(event):
    """
    Validate parsed NLP event
    """
    required = ["ticker", "type", "impact", "date"]

    for r in required:
        if r not in event:
            logger.warning(f"Invalid event missing {r}")
            return False

    # S5-6 FIX: bearish events have negative impact (-1.0 to 0.0)
    if event["impact"] < -1.0 or event["impact"] > 1.0:
        logger.warning("Event impact out of range")
        return False

    return True


def validate_features(features):
    """
    Validate feature dict
    """
    if not isinstance(features, dict):
        return False

    if len(features) == 0:
        logger.warning("Empty features dict")
        return False

    for k, v in features.items():
        if v is None:
            logger.warning(f"Feature {k} is None")
            return False

        if isinstance(v, (int, float)) and abs(v) > 1e6:
            logger.warning(f"Feature {k} value suspicious: {v}")
            return False

    return True


def validate_signal(signal):
    """
    Validate trading signal structure
    """
    required = ["ticker", "signal", "confidence", "monster_score"]

    for r in required:
        if r not in signal:
            logger.warning(f"Signal missing {r}")
            return False

    # S4-7 FIX: WATCH_EARLY renamed to EARLY_SIGNAL in V9; NO_SIGNAL is valid (no trade).
    valid_signals = ["BUY", "BUY_STRONG", "WATCH", "EARLY_SIGNAL", "NO_SIGNAL", "HOLD",
                     "WATCH_EARLY"]  # keep legacy alias for backward compat
    if signal["signal"] not in valid_signals:
        logger.warning(f"Unknown signal type: {signal['signal']}. Valid types: {valid_signals}")
        return False

    if not (0 <= signal["confidence"] <= 1):
        logger.warning("Confidence out of range")
        return False

    return True
