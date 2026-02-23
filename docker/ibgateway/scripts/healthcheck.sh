#!/bin/bash
# ============================================
# IB Gateway Docker Healthcheck
# ============================================
# Called by Docker HEALTHCHECK at 30s intervals.
# Checks that the API port is accepting connections.
# ============================================

TRADING_MODE="${TRADING_MODE:-paper}"

if [ "$TRADING_MODE" = "live" ]; then
    API_PORT=4001
else
    API_PORT=4002
fi

# Check if API port is open and accepting connections
if nc -z localhost "$API_PORT" 2>/dev/null; then
    exit 0
else
    exit 1
fi
