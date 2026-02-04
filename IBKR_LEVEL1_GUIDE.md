# 🎯 IBKR Integration Guide - GV2-EDGE V5.1

## 📊 Tes Abonnements IBKR

| Abonnement | Type | Utilisation GV2-EDGE |
|------------|------|---------------------|
| **OPRA** | Options L1 | ✅ Options Flow Detection |
| **NYSE (Network A/CTA)** | Stocks L1 | ✅ Quotes + Extended Hours |
| **NASDAQ (Network C/UTP)** | Stocks L1 | ✅ Quotes + Extended Hours |
| **NYSE American, BATS, ARCA, IEX** | Stocks L1 | ✅ Small caps coverage |

---

## ✅ Ce que Level 1 fournit

### Données Actions (NYSE/NASDAQ/BATS)

```
✅ Prix temps réel (Last, Bid, Ask)
✅ Spread réel (Ask - Bid)
✅ Volume journalier
✅ Pre-market data (4:00-9:30 AM)
✅ After-hours data (16:00-20:00)
✅ Historical bars (illimités)
✅ Daily stats (Open, High, Low, Close)
```

### Données Options (OPRA L1)

```
✅ Last price options
✅ Bid/Ask options
✅ Volume options
✅ Open Interest (delayed J-1)
✅ Greeks (calculés)
```

---

## 🔧 Configuration IBKR

### Option A: IB Gateway (Recommandé)

1. Télécharger IB Gateway sur ibkr.com
2. Lancer et se connecter
3. **Configure → Settings → API** :
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API
   - Port: `4001` (paper) ou `4002` (live)
   - Trusted IPs: `127.0.0.1`
4. Cliquer "Apply"

### Option B: TWS (Trader Workstation)

1. Lancer TWS et se connecter
2. **Edit → Global Configuration → API → Settings** :
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API (IMPORTANT!)
   - Port: `7497` (paper) ou `7496` (live)
   - Trusted IPs: `127.0.0.1`

### config.py

```python
USE_IBKR_DATA = True
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497   # ou 4001 pour Gateway
IBKR_CLIENT_ID = 1
```

---

## 📈 Extended Hours (After-Hours & Pre-Market)

### Activation

Tes abonnements NYSE/NASDAQ L1 incluent les extended hours.

Pour vérifier dans TWS :
1. **Edit → Global Configuration → API → Settings**
2. Vérifier que "Allow connections from localhost only" est coché

### Horaires Extended Hours

| Session | Horaire ET | Disponibilité |
|---------|-----------|---------------|
| Pre-Market | 04:00-09:30 | ✅ Avec tes abonnements |
| RTH | 09:30-16:00 | ✅ Standard |
| After-Hours | 16:00-20:00 | ✅ Avec tes abonnements |

### Code GV2-EDGE

```python
from src.extended_hours_quotes import (
    get_extended_quote,
    scan_afterhours_gaps,
    scan_premarket_gaps
)

# Get quote avec session info
quote = get_extended_quote("NVDA")
print(f"Session: {quote.session}")  # PRE, RTH, POST
print(f"Gap: {quote.gap_pct*100:.1f}%")

# Scan gaps after-hours
gaps = scan_afterhours_gaps(tickers, min_gap=0.03)
```

---

## 📊 Options Flow (OPRA L1)

### Ce que tu peux détecter

| Signal | Méthode | Interprétation |
|--------|---------|----------------|
| Volume Spike | Volume >> Open Interest | Smart money loading |
| Low P/C Ratio | Put/Call < 0.5 | Bullish sentiment |
| Call Concentration | 70%+ volume en calls | Target price identifié |

### Limitations OPRA L1

```
❌ Pas de trade-by-trade (besoin L2)
❌ Pas de direction (buy vs sell at ask/bid)
❌ Open Interest delayed (J-1)
```

### Code GV2-EDGE

```python
from src.options_flow_ibkr import (
    scan_options_flow,
    get_options_flow_score
)

# Scan options flow sur plusieurs tickers
signals = scan_options_flow(["NVDA", "AMD", "TSLA"])

# Score pour un ticker
score, details = get_options_flow_score("NVDA")
print(f"Options score: {score:.2f}")
print(f"Signals: {details.get('signals', [])}")
```

---

## 🔍 IBKR Radar (Anomaly Detection)

Le module `anticipation_engine.py` utilise IBKR pour :

1. **Volume Spike** : Volume > 3x moyenne
2. **Gap Detection** : Gap > 3% vs previous close
3. **Volatility Surge** : Range > 2x normal

```python
from src.anticipation_engine import run_ibkr_radar

anomalies = run_ibkr_radar(tickers)
for a in anomalies:
    print(f"{a.ticker}: {a.anomaly_type} (score: {a.score:.2f})")
```

---

## ⚠️ Troubleshooting

### "Connection refused"

```
Vérifier:
1. IB Gateway/TWS est lancé
2. Le bon port dans config.py
3. Trusted IPs inclut 127.0.0.1
```

### "Not connected"

```
Vérifier:
1. Logged in dans TWS/Gateway
2. Paper trading vs Live (ports différents)
3. Client ID unique (pas d'autre connexion)
```

### "No market data"

```
Vérifier:
1. Abonnements actifs dans Account Management
2. Ticker existe (pas OTC)
3. Market ouvert (ou extended hours activé)
```

### Test de connexion

```bash
python -c "
from src.ibkr_connector import get_ibkr
ibkr = get_ibkr()
print(f'Connected: {ibkr.connected if ibkr else False}')
if ibkr and ibkr.connected:
    quote = ibkr.get_quote('AAPL')
    print(f'AAPL: {quote}')
"
```

---

## 📋 Checklist Avant Lancement

- [ ] IB Gateway/TWS lancé et connecté
- [ ] Port correct dans config.py
- [ ] Read-Only API activé
- [ ] Trusted IPs configuré
- [ ] Test connexion OK
- [ ] Abonnements OPRA + NYSE + NASDAQ actifs

---

## 🚀 Performance Tips

1. **Utiliser IB Gateway** plutôt que TWS (moins de RAM)
2. **Cache activé** : `utils/cache.py` évite les calls redondants
3. **Rate limiting** : Pause 0.1s entre les requêtes
4. **Reconnexion auto** : Géré par `ibkr_connector.py`

---

**Version:** 5.1.0  
**Last Updated:** 2026-02-03
