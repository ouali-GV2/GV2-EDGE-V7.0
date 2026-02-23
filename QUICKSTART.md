# GV2-EDGE V9.0 - Quick Start Guide

## Installation en 5 Minutes

### 1. Extraction

```bash
git clone <repo_url>
cd GV2-EDGE-V7
```

### 2. Environnement Python

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration APIs (Variables d'environnement)

Creer un fichier `.env` a la racine :

```bash
cp .env.example .env
nano .env  # ou votre editeur prefere
```

Remplir les valeurs :

```bash
# ========= OBLIGATOIRE =========
GROK_API_KEY=xai-YOUR_KEY_HERE
FINNHUB_API_KEY=YOUR_FINNHUB_KEY
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID

# ========= IBKR (recommande) =========
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=paper, 7496=live

# ========= SOCIAL BUZZ (optionnel) =========
REDDIT_CLIENT_ID=YOUR_REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET=YOUR_REDDIT_SECRET
STOCKTWITS_ACCESS_TOKEN=YOUR_STOCKTWITS_TOKEN
```

### 4. IBKR Gateway/TWS (si utilise)

1. Ouvrir IB Gateway ou TWS
2. API Settings :
   - Enable Socket Clients
   - Read-Only API
   - Port: 7497 (paper) ou 7496 (live)
   - Trusted IP: 127.0.0.1

### 5. Lancement

```bash
python main.py
```

---

## Architecture V9.0

```
Detection (Layer 1) -> JAMAIS bloque
     |
Order (Layer 2) -> TOUJOURS calcule
     |
Execution (Layer 3) -> Seul point de blocage
     |
Multi-Radar V9 -> 4 radars paralleles + confluence
     |
IBKR Streaming V9 -> Temps reel ~10ms
     |
Output -> TOUS signaux visibles (avec raisons si bloques)
```

---

## Verification Rapide

```bash
# Test connexion IBKR
python src/ibkr_connector.py

# Test V7 engines
python -c "from src.engines.signal_producer import get_signal_producer; print('SignalProducer OK')"
python -c "from src.engines.order_computer import get_order_computer; print('OrderComputer OK')"
python -c "from src.engines.execution_gate import get_execution_gate; print('ExecutionGate OK')"

# Test Risk Guard
python -c "from src.risk_guard import get_unified_guard; print('RiskGuard OK')"

# Test Market Memory
python -c "from src.market_memory import is_market_memory_stable; print(is_market_memory_stable())"

# Test V9 Multi-Radar
python -c "from src.engines.multi_radar_engine import get_multi_radar_engine; print('MultiRadar OK')"

# Test V9 IBKR Streaming
python -c "from src.ibkr_streaming import get_ibkr_streaming; print('Streaming OK')"
```

---

## Recevoir les Alertes

1. Creer un bot Telegram via @BotFather
2. Recuperer le token
3. Envoyer un message au bot
4. Recuperer votre chat_id via `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Ajouter dans `.env`

---

## Configuration V9 (config.py)

```python
# V7.0 Architecture
USE_V7_ARCHITECTURE = True      # Utilise le nouveau pipeline

# Execution Gate
DAILY_TRADE_LIMIT = 5           # Max trades/jour
MAX_POSITION_PCT = 0.10         # Max 10% par position

# Pre-Halt Engine
ENABLE_PRE_HALT_ENGINE = True   # Detection pre-halt

# Risk Guard
ENABLE_RISK_GUARD = True        # Assessment risques

# Market Memory
ENABLE_MARKET_MEMORY = True     # MRP/EP (auto-activation)

# V9.0 Multi-Radar
ENABLE_MULTI_RADAR = True

# V9.0 Streaming
ENABLE_IBKR_STREAMING = True
```

---

## Le Systeme Tourne Automatiquement

| Session | Horaire (ET) | Action |
|---------|--------------|--------|
| After-Hours | 16:00-20:00 | Detection anticipative |
| Pre-Market | 04:00-09:30 | V7 cycle + confirmation |
| RTH | 09:30-16:00 | V7 cycle every 3 min |
| Daily Audit | 20:30 UTC | Rapport performance |

---

## Premiers Signaux V9

Attendez les alertes Telegram :

| Signal | Signification |
|--------|---------------|
| **BUY_STRONG** | Opportunite majeure (si autorise) |
| **BUY** | Signal confirme (si autorise) |
| **WATCH** | Potentiel detecte (observation) |
| **BUY (BLOCKED)** | Detecte mais bloque (avec raison) |

### Raisons de Blocage

- `DAILY_TRADE_LIMIT` - Max trades atteint
- `CAPITAL_INSUFFICIENT` - Pas assez de capital
- `PRE_HALT_HIGH` - Risque de halt
- `DILUTION_HIGH` - Risque de dilution
- `COMPLIANCE_HIGH` - Risque compliance

---

## Dashboard

```bash
streamlit run dashboards/streamlit_dashboard.py
```

Affiche:
- V7 Modules status
- Signals (allowed + blocked)
- Execution Gate stats
- Market Memory status

---

## Documentation

- `README.md` : Documentation complete
- `README_DEV.md` : Architecture technique V7
- `README_TRADER.md` : Guide trading V7
- `DEPLOYMENT.md` : Deploiement serveur

---

## Important

- **Mode READ ONLY** : Le systeme ne passe JAMAIS d'ordres
- **Decision humaine** : Vous decidez d'entrer ou non
- **Transparence** : TOUS les signaux visibles (meme bloques)
- **Apprentissage** : Blocked signals alimentent Market Memory
- **Securite** : Ne jamais commiter le fichier `.env`

---

**Happy Trading with V9.0!**

*Detection JAMAIS bloquee. Multi-Radar. Temps reel.*
