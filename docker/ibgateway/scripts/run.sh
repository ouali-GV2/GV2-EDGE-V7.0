#!/bin/bash
# ============================================
# IB Gateway Headless Launcher
# ============================================
# Starts Xvfb (virtual display) then IB Gateway via IBC.
# Handles daily IB reset (~23:45 ET) gracefully.
# ============================================

set -e

echo "=========================================="
echo " IB Gateway Headless Launcher"
echo "=========================================="

# ---- Validate required env vars ----
if [ -z "$IBKR_USER" ]; then
    echo "ERROR: IBKR_USER not set. Check your .env file."
    exit 1
fi

if [ -z "$IBKR_PASS" ]; then
    echo "ERROR: IBKR_PASS not set. Check your .env file."
    exit 1
fi

TRADING_MODE="${TRADING_MODE:-paper}"
echo "Trading mode: $TRADING_MODE"

# ---- Inject credentials into IBC config ----
# IBC reads IbLoginId/IbPassword from config.ini
# We substitute them at runtime from env vars (never stored on disk)
sed -i "s/^IbLoginId=.*/IbLoginId=${IBKR_USER}/" "$IBC_INI"
sed -i "s/^IbPassword=.*/IbPassword=${IBKR_PASS}/" "$IBC_INI"
sed -i "s/^TradingMode=.*/TradingMode=${TRADING_MODE}/" "$IBC_INI"

# ---- Determine API port based on trading mode ----
if [ "$TRADING_MODE" = "live" ]; then
    API_PORT=4001
else
    API_PORT=4002
fi
echo "API port: $API_PORT"

# ---- Start Xvfb (virtual framebuffer) ----
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
sleep 2

# Verify Xvfb is running
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID: $XVFB_PID)"

export DISPLAY=:99

# ---- Start IB Gateway via IBC ----
echo "Starting IB Gateway via IBC..."
echo "  IBC path:  $IBC_PATH"
echo "  TWS path:  $TWS_PATH"
echo "  Settings:  $TWS_SETTINGS_PATH"

# IBC start command
# -g = Gateway mode (not TWS)
# --tws-path = IB Gateway installation dir
# --tws-settings-path = JTS settings dir
# --ibc-path = IBC installation dir
# --ibc-ini = IBC config file
"${IBC_PATH}/scripts/ibcstart.sh" \
    -g \
    --tws-path="${TWS_PATH}" \
    --tws-settings-path="${TWS_SETTINGS_PATH}" \
    --ibc-path="${IBC_PATH}" \
    --ibc-ini="${IBC_INI}" \
    --mode="${TRADING_MODE}" \
    --on2fatimeout=exit &

IBC_PID=$!
echo "IBC started (PID: $IBC_PID)"

# ---- Wait for API port to become available ----
echo "Waiting for IB Gateway API on port $API_PORT..."
MAX_WAIT=180
WAITED=0

while ! nc -z localhost "$API_PORT" 2>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: IB Gateway API not available after ${MAX_WAIT}s"
        echo "Check credentials and 2FA settings."
        exit 1
    fi
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${WAITED}s / ${MAX_WAIT}s)"
    fi
done

echo "=========================================="
echo " IB Gateway API ready on port $API_PORT"
echo " Uptime monitoring active"
echo "=========================================="

# ---- Keep container running ----
# Monitor both Xvfb and IBC processes
# If either dies, the container exits and Docker restarts it
while true; do
    if ! kill -0 $XVFB_PID 2>/dev/null; then
        echo "ERROR: Xvfb crashed, exiting for restart..."
        exit 1
    fi

    if ! kill -0 $IBC_PID 2>/dev/null; then
        echo "WARN: IBC process exited, exiting for restart..."
        exit 1
    fi

    # Check API port still up
    if ! nc -z localhost "$API_PORT" 2>/dev/null; then
        echo "WARN: API port $API_PORT not responding"
        # Don't exit immediately - IB does daily resets
        # Wait 60s then check again
        sleep 60
        if ! nc -z localhost "$API_PORT" 2>/dev/null; then
            echo "ERROR: API port $API_PORT still down after 60s, exiting for restart..."
            exit 1
        fi
    fi

    sleep 30
done
