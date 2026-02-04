import time
import psutil
import requests

from utils.logger import get_logger
from alerts.telegram_alerts import send_system_alert
from utils.api_guard import safe_get

from config import (
    FINNHUB_API_KEY,
    GROK_API_KEY
)

logger = get_logger("SYSTEM_GUARDIAN")

CHECK_INTERVAL = 60  # seconds

FINNHUB_PING = "https://finnhub.io/api/v1/quote"
GROK_PING = "https://api.x.ai/v1/models"


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
# Alert logic
# ============================

def analyze_health():

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

    logger.info(
        f"Health OK | CPU {stats['cpu']}% RAM {stats['ram']}% DISK {stats['disk']}%"
    )


# ============================
# Main loop
# ============================

def run_guardian():

    logger.info("System Guardian started 🛡")

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
