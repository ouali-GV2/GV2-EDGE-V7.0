import time
import psutil
import requests

from utils.logger import get_logger
from alerts.telegram_alerts import send_system_alert, send_ibkr_connection_alert
from utils.api_guard import safe_get

from config import (
    FINNHUB_API_KEY,
    GROK_API_KEY,
    USE_IBKR_DATA,
)

logger = get_logger("SYSTEM_GUARDIAN")

CHECK_INTERVAL = 60  # seconds

FINNHUB_PING = "https://finnhub.io/api/v1/quote"
GROK_PING = "https://api.x.ai/v1/models"

# Track IBKR state for edge-triggered alerts (only alert on change)
_last_ibkr_state = None


# ============================
# System health
# ============================

def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    return {
        "cpu": cpu,
        "ram": ram,
        "disk": disk
    }


# ============================
# API health
# ============================

def check_finnhub():
    try:
        params = {"symbol": "AAPL", "token": FINNHUB_API_KEY}
        r = safe_get(FINNHUB_PING, params=params)
        return r.status_code == 200
    except:
        return False


def check_grok():
    try:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
        r = requests.get(GROK_PING, headers=headers, timeout=5)
        return r.status_code == 200
    except:
        return False


# ============================
# IBKR health
# ============================

def check_ibkr():
    """
    Check IBKR connection health.

    Returns:
        dict with state, connected, latency, details
        or None if IBKR is disabled
    """
    if not USE_IBKR_DATA:
        return None

    try:
        from src.ibkr_connector import get_ibkr

        ibkr = get_ibkr()
        if ibkr is None:
            return {"state": "DISABLED", "connected": False, "latency_ms": 0}

        stats = ibkr.get_connection_stats()

        return {
            "state": stats["state"],
            "connected": stats["connected"],
            "latency_ms": stats["heartbeat_latency_ms"],
            "uptime_seconds": stats["uptime_seconds"],
            "disconnections": stats["total_disconnections"],
            "reconnections": stats["total_reconnections"],
            "last_downtime": stats["last_downtime_seconds"],
        }

    except Exception as e:
        logger.warning(f"IBKR health check failed: {e}")
        return {"state": "ERROR", "connected": False, "latency_ms": 0}


# ============================
# Alert logic
# ============================

def analyze_health():
    global _last_ibkr_state

    stats = get_system_stats()

    if stats["cpu"] > 85:
        send_system_alert(f"High CPU usage: {stats['cpu']}%")

    if stats["ram"] > 85:
        send_system_alert(f"High RAM usage: {stats['ram']}%")

    if stats["disk"] > 90:
        send_system_alert(f"Disk almost full: {stats['disk']}%")

    if not check_finnhub():
        send_system_alert("Finnhub API unreachable")

    if not check_grok():
        send_system_alert("Grok API unreachable")

    # IBKR health check with edge-triggered alerts
    ibkr_status = check_ibkr()

    ibkr_msg = ""
    if ibkr_status:
        current_state = ibkr_status["state"]

        # Alert only on state transitions (not every 60s)
        if _last_ibkr_state is not None and current_state != _last_ibkr_state:
            if current_state == "CONNECTED" and _last_ibkr_state in ("RECONNECTING", "FAILED", "DISCONNECTED"):
                send_ibkr_connection_alert(
                    status="reconnected",
                    details={
                        "downtime_seconds": ibkr_status["last_downtime"],
                        "reconnections": ibkr_status["reconnections"],
                    }
                )
            elif current_state == "RECONNECTING":
                send_ibkr_connection_alert(
                    status="disconnected",
                    details={"state": current_state}
                )
            elif current_state == "FAILED":
                send_ibkr_connection_alert(
                    status="failed",
                    details={
                        "disconnections": ibkr_status["disconnections"],
                    }
                )

        _last_ibkr_state = current_state

        # Warn on high latency
        if ibkr_status["connected"] and ibkr_status["latency_ms"] > 2000:
            send_system_alert(
                f"IBKR latency high: {ibkr_status['latency_ms']:.0f}ms",
                level="warning"
            )

        ibkr_msg = f" | IBKR {current_state} ({ibkr_status['latency_ms']:.0f}ms)"

    logger.info(
        f"Health OK | CPU {stats['cpu']}% RAM {stats['ram']}% DISK {stats['disk']}%{ibkr_msg}"
    )


# ============================
# Main loop
# ============================

def run_guardian():

    logger.info("System Guardian started")

    # Wire up IBKR state change callback for real-time alerts
    if USE_IBKR_DATA:
        try:
            from src.ibkr_connector import get_ibkr, ConnectionState
            ibkr = get_ibkr()
            if ibkr:
                def _on_ibkr_state_change(old_state, new_state):
                    if new_state == ConnectionState.RECONNECTING:
                        send_ibkr_connection_alert("disconnected", {"state": new_state.value})
                    elif new_state == ConnectionState.CONNECTED and old_state == ConnectionState.RECONNECTING:
                        stats = ibkr.get_connection_stats()
                        send_ibkr_connection_alert("reconnected", {
                            "downtime_seconds": stats["last_downtime_seconds"],
                            "reconnections": stats["total_reconnections"],
                        })
                    elif new_state == ConnectionState.FAILED:
                        send_ibkr_connection_alert("failed", {})

                ibkr.set_state_change_callback(_on_ibkr_state_change)
                logger.info("IBKR state change callback registered")
        except Exception as e:
            logger.warning(f"Could not register IBKR callback: {e}")

    while True:
        try:
            analyze_health()
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Guardian crash: {e}")
            send_system_alert(f"Guardian crashed: {e}")
            time.sleep(10)


# ============================
# CLI
# ============================

if __name__ == "__main__":
    run_guardian()
