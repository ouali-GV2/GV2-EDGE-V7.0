# DEPLOYMENT GUIDE - GV2-EDGE V9.0

Complete guide to install, configure, and deploy GV2-EDGE on a production server.

**V9.0 Architecture**: Multi-Radar Detection + IBKR Streaming + Detection/Execution Separation

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Installation](#2-local-installation)
3. [API Configuration](#3-api-configuration)
4. [IBKR Setup](#4-ibkr-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Server Deployment (Hetzner/Linux)](#6-server-deployment)
7. [Docker Deployment](#7-docker-deployment)
8. [Monitoring & Maintenance](#8-monitoring--maintenance)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| RAM | 4 GB | 16 GB |
| CPU | 2 cores | 8 cores |
| Storage | 20 GB SSD | 160 GB SSD |
| Python | 3.10+ | 3.11+ |

### Required Accounts & Subscriptions

| Service | Purpose | Cost |
|---------|---------|------|
| **Interactive Brokers** | Real-time market data (Level 1) | ~$10/month |
| **Grok API (x.ai)** | NLP analysis, news parsing | Pay-per-use |
| **Finnhub** | Fallback data, earnings calendar | Free tier OK |
| **Telegram** | Alert notifications | Free |

### Optional Subscriptions

| Service | Purpose | Cost |
|---------|---------|------|
| IBKR OPRA L1 | Options flow data | ~$1.50/month |

---

## 2. Local Installation

### 2.1 Clone Repository

```bash
git clone https://github.com/your-org/GV2-EDGE-V9.0.git
cd GV2-EDGE-V9.0
```

### 2.2 Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 2.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4 Verify Installation

```bash
python -c "import ib_insync; import streamlit; import pandas; print('All dependencies OK')"
```

---

## 3. API Configuration

### 3.1 Create Environment File (Required)

Create a `.env` file from the template:

```bash
cp .env.example .env
chmod 600 .env  # Restrict permissions
```

### 3.2 Configure API Keys

Edit `.env` with your API keys:

```bash
# ========= REQUIRED =========

# Grok API (x.ai) - Required for NLP & Twitter/X access
GROK_API_KEY=xai-your-api-key-here

# Finnhub - Required for fallback data & earnings calendar
FINNHUB_API_KEY=your-finnhub-key

# Telegram Bot - Required for alerts
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-100123456789

# ========= IBKR (Recommended) =========

IBKR_HOST=127.0.0.1
IBKR_PORT=4002  # 4002=paper, 4001=live

# ========= SOCIAL BUZZ (Optional but Recommended) =========

# Reddit API - Get at https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
REDDIT_USER_AGENT=GV2-EDGE/1.0

# StockTwits API - Get at https://api.stocktwits.com/developers
STOCKTWITS_ACCESS_TOKEN=your-stocktwits-token
```

**IMPORTANT:** Never commit `.env` to git (already in `.gitignore`)

### 3.3 Get API Keys

#### Grok API (x.ai) - REQUIRED
1. Visit https://x.ai/api
2. Sign up and create an API key
3. Add to `.env` as `GROK_API_KEY`
4. **Note:** This also provides Twitter/X data access

#### Finnhub - REQUIRED
1. Visit https://finnhub.io/
2. Sign up for free account
3. Copy API key from dashboard
4. Add to `.env` as `FINNHUB_API_KEY`

#### Telegram Bot - REQUIRED
1. Message @BotFather on Telegram
2. Send `/newbot` and follow prompts
3. Copy the bot token to `TELEGRAM_BOT_TOKEN`
4. Create a group/channel and add the bot
5. Get chat ID via `https://api.telegram.org/bot<TOKEN>/getUpdates`
6. Add chat ID to `TELEGRAM_CHAT_ID`

#### Reddit API - OPTIONAL
1. Visit https://www.reddit.com/prefs/apps
2. Click "Create App" → Select "script"
3. Fill in name and redirect URI (http://localhost:8080)
4. Copy `client_id` (under app name) and `client_secret`
5. Add to `.env` as `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`

#### StockTwits API - OPTIONAL
1. Visit https://api.stocktwits.com/developers
2. Create developer account
3. Create an application
4. Copy access token to `STOCKTWITS_ACCESS_TOKEN`

### 3.4 Social Buzz Sources & Weights

| Source | Weight | API Required |
|--------|--------|--------------|
| Twitter/X | 45% | `GROK_API_KEY` |
| Reddit | 30% | `REDDIT_CLIENT_ID` + `SECRET` |
| StockTwits | 25% | `STOCKTWITS_ACCESS_TOKEN` |
| Google Trends | 0% | Disabled (unreliable) |

**Note:** If Reddit/StockTwits APIs are not configured, the system falls back to public endpoints with reduced reliability.

### 3.5 Test API Connections

```bash
# Test Finnhub
python -c "
from config import FINNHUB_API_KEY
import requests
r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=AAPL&token={FINNHUB_API_KEY}')
print('Finnhub:', 'OK' if r.status_code == 200 else 'FAILED')
"

# Test Telegram
python -c "
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import requests
r = requests.post(
    f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
    json={'chat_id': TELEGRAM_CHAT_ID, 'text': 'GV2-EDGE test message'}
)
print('Telegram:', 'OK' if r.status_code == 200 else 'FAILED')
"

# Test Social Buzz (all sources)
python src/social_buzz.py
```

---

## 4. IBKR Setup

### 4.1 Install IB Gateway

#### Linux
```bash
# Download IB Gateway
wget https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh

# Make executable and install
chmod +x ibgateway-stable-standalone-linux-x64.sh
./ibgateway-stable-standalone-linux-x64.sh
```

#### Windows/Mac
Download from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

### 4.2 Configure IB Gateway

1. Launch IB Gateway: `~/Jts/ibgateway/*/ibgateway`
2. Login with IBKR credentials
3. Go to **Configure > Settings > API**:

| Setting | Value |
|---------|-------|
| Enable ActiveX and Socket Clients | ✅ Checked |
| Read-Only API | ✅ Checked |
| Socket Port | `4001` (live) or `4002` (paper) |
| Allow connections from localhost only | ✅ Checked |
| Master API client ID | Leave empty |

4. Click **OK** and restart Gateway

### 4.3 Configure GV2-EDGE for IBKR

Edit `config.py`:

```python
# ========= IBKR CONNECTION =========
USE_IBKR_DATA = True

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002   # 4002 = paper, 4001 = live
IBKR_CLIENT_ID = 1
```

### 4.4 Test IBKR Connection

```bash
python src/ibkr_connector.py
```

Expected output:
```
Testing IBKR Connector (Level 1)...
✅ Connected to IBKR (127.0.0.1:4002) - READ ONLY
Testing real-time quote:
  AAPL: Last=$XXX.XX, Bid=$XXX.XX, Ask=$XXX.XX
```

---

## 5. Running the Pipeline

### 5.1 Start Main Pipeline

```bash
# Activate venv
source venv/bin/activate

# Run main pipeline
python main.py
```

### 5.2 Start Dashboard

```bash
streamlit run dashboards/streamlit_dashboard.py --server.port 8501
```

Access at: http://localhost:8501

### 5.3 Run Audits

```bash
# Daily audit
python daily_audit.py

# Weekly audit
python weekly_deep_audit.py

# Specific date
python daily_audit.py --date 2024-01-15
```

### 5.4 Run Backtest

```bash
python backtests/backtest_engine_edge.py
```

---

## 6. Server Deployment

### 6.1 Server Setup (Ubuntu 22.04)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip git screen nginx

# Create dedicated user
sudo useradd -m -s /bin/bash gv2edge
sudo passwd gv2edge

# Switch to user
sudo su - gv2edge
```

### 6.2 Install Application

```bash
cd /home/gv2edge

# Clone repository
git clone https://github.com/your-org/GV2-EDGE-V9.0.git app
cd app

# Setup venv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure API keys
nano config.py
```

### 6.3 Create Systemd Service (Main Pipeline)

```bash
sudo nano /etc/systemd/system/gv2edge.service
```

Content:
```ini
[Unit]
Description=GV2-EDGE Trading Pipeline
After=network.target

[Service]
Type=simple
User=gv2edge
WorkingDirectory=/home/gv2edge/app
Environment=PATH=/home/gv2edge/app/venv/bin
ExecStart=/home/gv2edge/app/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/gv2edge/app/data/logs/service.log
StandardError=append:/home/gv2edge/app/data/logs/service_error.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gv2edge
sudo systemctl start gv2edge
sudo systemctl status gv2edge
```

### 6.4 Create Systemd Service (Dashboard)

```bash
sudo nano /etc/systemd/system/gv2edge-dashboard.service
```

Content:
```ini
[Unit]
Description=GV2-EDGE Dashboard
After=network.target

[Service]
Type=simple
User=gv2edge
WorkingDirectory=/home/gv2edge/app
Environment=PATH=/home/gv2edge/app/venv/bin
ExecStart=/home/gv2edge/app/venv/bin/streamlit run dashboards/streamlit_dashboard.py --server.port 8501 --server.address 127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gv2edge-dashboard
sudo systemctl start gv2edge-dashboard
```

### 6.5 Nginx Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/gv2edge
```

Content:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Dashboard
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # WebSocket for Streamlit
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/gv2edge /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6.6 SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 6.7 IB Gateway Headless Mode

For running IB Gateway without display:

```bash
# Install Xvfb
sudo apt install xvfb

# Create startup script
nano /home/gv2edge/start_ibgateway.sh
```

Content:
```bash
#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
sleep 2
~/Jts/ibgateway/*/ibgateway &
```

Make executable:
```bash
chmod +x /home/gv2edge/start_ibgateway.sh
```

### 6.8 Cron Jobs

```bash
crontab -e
```

Add:
```cron
# Daily audit at 20:30 UTC (after market close)
30 20 * * * /home/gv2edge/app/venv/bin/python /home/gv2edge/app/daily_audit.py >> /home/gv2edge/app/data/logs/cron_audit.log 2>&1

# Weekly deep audit Friday 22:00 UTC
0 22 * * 5 /home/gv2edge/app/venv/bin/python /home/gv2edge/app/weekly_deep_audit.py >> /home/gv2edge/app/data/logs/cron_weekly.log 2>&1

# Restart IB Gateway daily at 03:00 UTC (maintenance window)
0 3 * * * /home/gv2edge/start_ibgateway.sh >> /home/gv2edge/app/data/logs/ibgateway.log 2>&1

# Clean old logs weekly
0 4 * * 0 find /home/gv2edge/app/data/logs -name "*.log" -mtime +30 -delete
```

---

## 7. Docker Deployment

### 7.1 Build Image

```bash
docker build -t gv2edge:latest .
```

### 7.2 Run with Docker Compose

```bash
docker-compose up -d
```

### 7.3 View Logs

```bash
docker-compose logs -f gv2edge
```

### 7.4 Stop

```bash
docker-compose down
```

---

## 8. Monitoring & Maintenance

### 8.1 Check Service Status

```bash
# Main pipeline
sudo systemctl status gv2edge

# Dashboard
sudo systemctl status gv2edge-dashboard

# View logs
journalctl -u gv2edge -f
```

### 8.2 Application Logs

```bash
# Real-time logs
tail -f /home/gv2edge/app/data/logs/gv2edge.log

# Error logs
tail -f /home/gv2edge/app/data/logs/service_error.log
```

### 8.3 System Health

```bash
# Run health check
python monitoring/system_guardian.py

# Check resource usage
htop
df -h
```

### 8.4 Database Maintenance

```bash
# Backup signals database
cp data/signals_history.db data/backups/signals_$(date +%Y%m%d).db

# Vacuum database (reduce size)
sqlite3 data/signals_history.db "VACUUM;"
```

### 8.5 Update Application

```bash
cd /home/gv2edge/app
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart gv2edge
sudo systemctl restart gv2edge-dashboard
```

---

## 9. Troubleshooting

### IBKR Connection Failed

```bash
# Check if IB Gateway is running
pgrep -f ibgateway

# Check port availability
netstat -tlnp | grep 4002

# Test connection manually
python -c "
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
print('Connected:', ib.isConnected())
ib.disconnect()
"
```

**Common fixes:**
- Ensure IB Gateway is running and logged in
- Check API settings in Gateway (port, read-only mode)
- Verify firewall allows localhost connections

### No Signals Generated

1. Check universe loaded:
```bash
wc -l data/universe.csv
```

2. Check market session:
```bash
python -c "from utils.time_utils import market_session; print(market_session())"
```

3. Verify IBKR data subscription active

4. Check logs for errors:
```bash
grep -i error data/logs/gv2edge.log | tail -20
```

### Dashboard Not Loading

```bash
# Check if port in use
lsof -i :8501

# Restart dashboard
sudo systemctl restart gv2edge-dashboard

# Check logs
journalctl -u gv2edge-dashboard -n 50
```

### Telegram Alerts Not Working

```bash
# Test bot token
curl "https://api.telegram.org/bot<TOKEN>/getMe"

# Test sending message
python -c "
from alerts.telegram_alerts import send_telegram_message
send_telegram_message('Test from GV2-EDGE')
"
```

### High Memory Usage

```bash
# Check process memory
ps aux | grep python

# Restart services
sudo systemctl restart gv2edge
```

### API Rate Limits

If hitting rate limits:
1. Increase cache TTL in `utils/cache.py`
2. Reduce scan frequency in `config.py`
3. Check `utils/api_guard.py` for retry settings

---

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Grok API key configured
- [ ] Finnhub API key configured
- [ ] Telegram bot configured
- [ ] IB Gateway installed and running
- [ ] IBKR connection tested (`python src/ibkr_connector.py`)
- [ ] Main pipeline runs (`python main.py`)
- [ ] Dashboard accessible (`streamlit run dashboards/streamlit_dashboard.py`)
- [ ] Alerts working (test Telegram)
- [ ] Cron jobs configured (audits)
- [ ] Logs rotating properly

---

## Support

- **Documentation:** See `README.md`, `CLAUDE.md`, `README_TRADER.md`
- **Issues:** Report bugs via GitHub Issues
- **Logs:** Always include relevant log files when reporting issues

---

*Last updated: 2026-02-21 - GV2-EDGE V9.0*
