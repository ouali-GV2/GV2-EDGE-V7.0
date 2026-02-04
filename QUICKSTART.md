# 🚀 GV2-EDGE V5.1 - Quick Start Guide

## ⏱️ Installation en 5 Minutes

### 1. Extraction

```bash
unzip GV2-EDGE-V5.1-COMPLETE.zip
cd GV2-EDGE-V2-ENHANCED
```

### 2. Environnement Python

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration APIs

Éditer `config.py` :

```python
# Grok API (X.AI) - OBLIGATOIRE
GROK_API_KEY = "xai-YOUR_KEY_HERE"

# Finnhub - OBLIGATOIRE (gratuit)
FINNHUB_API_KEY = "YOUR_FINNHUB_KEY"

# Telegram - OBLIGATOIRE
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# IBKR - OPTIONNEL mais recommandé
USE_IBKR_DATA = True
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # 7497=paper, 7496=live
```

### 4. IBKR Gateway/TWS (si utilisé)

1. Ouvrir IB Gateway ou TWS
2. API Settings :
   - ✅ Enable Socket Clients
   - ✅ Read-Only API
   - Port: 7497 (paper) ou 7496 (live)
   - Trusted IP: 127.0.0.1

### 5. Lancement

```bash
python main.py
```

---

## ✅ Vérification Rapide

```bash
# Test connexion IBKR
python src/ibkr_connector.py

# Test Grok API
python src/social_buzz.py

# Test News Flow
python src/news_flow_screener.py
```

---

## 📱 Recevoir les Alertes

1. Créer un bot Telegram via @BotFather
2. Récupérer le token
3. Envoyer un message au bot
4. Récupérer votre chat_id via `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Configurer dans `config.py`

---

## ⏰ Le Système Tourne Automatiquement

| Session | Horaire (ET) | Action |
|---------|--------------|--------|
| After-Hours | 16:00-20:00 | Détection anticipative |
| Pre-Market | 04:00-09:30 | Confirmation + signaux |
| RTH | 09:30-16:00 | Monitoring |
| Daily Audit | 20:30 UTC | Rapport performance |

---

## 🚨 Premiers Signaux

Attendez les alertes Telegram :
- **WATCH_EARLY** : Catalyst détecté (surveiller)
- **BUY** : Signal confirmé (entry)
- **BUY_STRONG** : Opportunité majeure (entry immédiate)

---

## 📚 Documentation

- `README.md` : Documentation complète
- `README_DEV.md` : Architecture technique
- `README_TRADER.md` : Guide trading
- `IBKR_LEVEL1_GUIDE.md` : Configuration IBKR

---

## ⚠️ Important

- **Mode READ ONLY** : Le système ne passe JAMAIS d'ordres
- **Décision humaine** : Vous décidez d'entrer ou non
- **Risk management** : Toujours utiliser des stops

---

**Happy Trading! 🚀**
