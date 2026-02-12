#!/bin/bash
# GV2-EDGE V7.0 Entrypoint
# Launches both the trading engine and the Streamlit dashboard

set -e

echo "=== GV2-EDGE V7.0 Starting ==="

# Start Streamlit dashboard in background
echo "Starting Streamlit dashboard on port 8501..."
streamlit run dashboards/streamlit_dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    &

STREAMLIT_PID=$!

# Start main trading engine in foreground
echo "Starting trading engine (main.py)..."
python main.py &
MAIN_PID=$!

# Wait for either process to exit
wait -n $STREAMLIT_PID $MAIN_PID

# If one exits, stop the other
echo "=== Process exited, shutting down ==="
kill $STREAMLIT_PID $MAIN_PID 2>/dev/null || true
wait
