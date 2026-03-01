import time
import psutil
import requests
from datetime import datetime, timezone

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
# Track since when IBKR is disconnected (for auto force_reconnect)
_ibkr_disconnected_since: float = 0.0

# S4-4 FIX: Per-condition cooldown to prevent Telegram flood.
# Key = condition label, value = last alert timestamp (epoch seconds).
_alert_cooldowns: dict = {}
_COOLDOWN_SECONDS = {
    "cpu_high": 600,       # max 1 alert / 10 min for CPU spikes
    "ram_high": 600,
    "disk_full": 1800,     # max 1 alert / 30 min for disk
    "finnhub_down": 300,
    "grok_down": 300,
    "ibkr_latency": 300,
    "ibkr_reconnect": 300,  # force_reconnect() max 1 fois / 5 min
}
_DEFAULT_COOLDOWN = 300  # fallback for unlisted conditions


def _cooldown_ok(condition: str) -> bool:
    """Return True if enough time has elapsed since the last alert for this condition."""
    import time as _time
    now = _time.monotonic()
    last = _alert_cooldowns.get(condition, 0.0)
    limit = _COOLDOWN_SECONDS.get(condition, _DEFAULT_COOLDOWN)
    if now - last >= limit:
        _alert_cooldowns[condition] = now
        return True
    return False


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
    global _last_ibkr_state, _ibkr_disconnected_since

    stats = get_system_stats()

    # S4-4 FIX: gate each alert through per-condition cooldown
    if stats["cpu"] > 85 and _cooldown_ok("cpu_high"):
        send_system_alert(f"High CPU usage: {stats['cpu']}%")

    if stats["ram"] > 85 and _cooldown_ok("ram_high"):
        send_system_alert(f"High RAM usage: {stats['ram']}%")

    if stats["disk"] > 90 and _cooldown_ok("disk_full"):
        send_system_alert(f"Disk almost full: {stats['disk']}%")

    if not check_finnhub() and _cooldown_ok("finnhub_down"):
        send_system_alert("Finnhub API unreachable")

    if not check_grok() and _cooldown_ok("grok_down"):
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

        # Auto force_reconnect si IBKR déconnecté depuis > 5 min
        if current_state in ("DISCONNECTED", "FAILED"):
            now = time.monotonic()
            if _ibkr_disconnected_since == 0.0:
                _ibkr_disconnected_since = now
            elif (now - _ibkr_disconnected_since) > 300 and _cooldown_ok("ibkr_reconnect"):
                logger.warning(
                    f"IBKR {current_state} depuis >{(now - _ibkr_disconnected_since)/60:.0f}min "
                    f"— force_reconnect() automatique"
                )
                try:
                    from src.ibkr_connector import get_ibkr
                    ibkr_inst = get_ibkr()
                    if ibkr_inst:
                        ibkr_inst.force_reconnect()
                except Exception as _re:
                    logger.warning(f"force_reconnect failed: {_re}")
        else:
            _ibkr_disconnected_since = 0.0

        # Warn on high latency (cooldown so we don't flood on sustained high latency)
        if ibkr_status["connected"] and ibkr_status["latency_ms"] > 2000 and _cooldown_ok("ibkr_latency"):
            send_system_alert(
                f"IBKR latency high: {ibkr_status['latency_ms']:.0f}ms",
                level="warning"
            )

        ibkr_msg = f" | IBKR {current_state} ({ibkr_status['latency_ms']:.0f}ms)"

        # Write status snapshot for dashboard (separate process — no shared memory)
        try:
            import json as _json, os as _os
            _status_path = "data/ibkr_status.json"
            _os.makedirs("data", exist_ok=True)
            _up = ibkr_status.get("uptime_seconds", 0) or 0
            _uptime_str = (f"{_up/3600:.1f}h" if _up >= 3600
                           else f"{_up/60:.0f}m" if _up >= 60
                           else f"{_up:.0f}s")
            _payload = {
                "connected":  ibkr_status["connected"],
                "state":      current_state,
                "latency_ms": ibkr_status.get("latency_ms", 0),
                "uptime":     _uptime_str if ibkr_status["connected"] else None,
                "latency":    f"{ibkr_status.get('latency_ms',0):.0f}ms" if ibkr_status["connected"] else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(_status_path, "w") as _f:
                _json.dump(_payload, _f)
        except Exception:
            pass

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
