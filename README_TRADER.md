# 📊 GV2-EDGE V5.3 — Trader Guide

## 🎯 Objectif

GV2-EDGE détecte les top gainers small caps US **AVANT** leurs hausses majeures (+50% à +500%).

**Cible** : Small caps < $2B market cap, hors OTC

---

## 🆕 Nouveautés V5.3

### Monster Score - Nouvelles Composantes

Le score inclut maintenant **8 facteurs** pondérés :

| Composante | Poids | Description |
|------------|-------|-------------|
| Event | 25% | Catalysts (earnings, FDA, M&A) |
| Volume | 17% | Volume spikes vs moyenne |
| Pattern | 17% | Patterns techniques (consolidation, flags) |
| PM Transition | 13% | Qualité transition pre-market → RTH |
| **Options Flow** | **10%** | Activité options (volume, concentration calls) |
| Momentum | 8% | Momentum prix |
| **Social Buzz** | **6%** | Mentions Twitter, Reddit, StockTwits |
| Squeeze | 4% | Bollinger squeeze |

### Impact pour le Trading

- **Options Flow élevé** (>0.5) = Smart money potentiel
- **Social Buzz spike** (>0.7) = Attention retail croissante
- Ces facteurs peuvent confirmer ou renforcer un signal

---

## 🚦 Signaux (du plus précoce au plus confirmé)

### 👀 WATCH_EARLY (NOUVEAU V5)
- **Quand** : Catalyst détecté en after-hours/pre-market
- **Signification** : Potentiel en formation, pas encore confirmé
- **Action** : Surveiller, préparer entry
- **Sizing** : Aucun (attendre upgrade)

### 📊 BUY
- **Quand** : Score 0.65-0.79 + confirmation technique
- **Signification** : Setup solide, probabilité élevée
- **Action** : Entry standard
- **Sizing** : Position normale (2% risk)

### 🚨 BUY_STRONG
- **Quand** : Score 0.80+ + catalyst fort + confirmation
- **Signification** : Opportunité majeure
- **Action** : Entry immédiate
- **Sizing** : Position max (3% risk)

### ⏸️ HOLD
- **Signification** : Pas d'opportunité claire
- **Action** : Ignorer

---

## ⏰ Timeline de Détection V5.1

```
16:00-20:00 ET │ AFTER-HOURS
              │ ├─ News Flow Screener actif
              │ ├─ Extended Hours gaps détectés
              │ ├─ Options Flow analysé
              │ └─ Signaux: WATCH_EARLY
              │
04:00-09:30 ET │ PRE-MARKET
              │ ├─ Confirmation des gaps
              │ ├─ Volume PM analysé
              │ ├─ Upgrades: WATCH_EARLY → BUY
              │ └─ Signaux: BUY, BUY_STRONG
              │
09:30-16:00 ET │ RTH (Regular Trading Hours)
              │ ├─ Monitoring positions
              │ ├─ Breakout confirmation
              │ └─ Signaux: BUY_STRONG (tardifs)
```

---

## 📱 Alertes Telegram

### Format WATCH_EARLY
```
👀 WATCH_EARLY: NVDA

📊 Score: 0.55
├─ Catalyst: EARNINGS_BEAT
├─ Impact: 0.7
└─ Urgency: MEDIUM

📰 "NVIDIA beats Q4 expectations..."

⏰ Session: AFTER-HOURS
💡 Action: Surveiller PM confirmation
```

### Format BUY
```
📊 BUY: NVDA

📊 Monster Score: 0.72
├─ Technical: 0.65
├─ Fundamental: 0.78
└─ AH Boost: +0.05

📅 Catalyst: EARNINGS_BEAT
📈 PM Gap: +5.2%

💰 Trade Plan:
├─ Entry: $152.50
├─ Stop: $148.20 (-2.8%)
├─ Target 1: $165 (+8.2%)
└─ Risk: 2% capital

⏰ Execute: PM OPEN
```

### Format BUY_STRONG
```
🚨 BUY_STRONG: NVDA

📊 Monster Score: 0.85
├─ Technical: 0.80
├─ Fundamental: 0.88
└─ Options Flow: BULLISH

📅 Catalyst: FDA_APPROVAL
📈 PM Gap: +12.5%
🔥 Volume: 5x average

💰 Trade Plan:
├─ Entry: $165.00 (MARKET)
├─ Stop: $158.00 (-4.2%)
├─ Target: $200+ (+21%)
└─ Risk: 3% capital (MAX)

⏰ Execute: IMMEDIATELY
```

---

## 🎯 Stratégie d'Entrée Recommandée

### Pour WATCH_EARLY
1. **Ne pas entrer** immédiatement
2. Mettre le ticker en watchlist
3. Attendre confirmation PM :
   - Gap > 3%
   - Volume PM élevé
   - Prix tient au-dessus du gap
4. Si confirmé → entry sur upgrade à BUY

### Pour BUY
1. Entry au prix indiqué (limit order)
2. Stop-loss obligatoire
3. Sizing : 2% du capital à risque
4. Target : selon plan

### Pour BUY_STRONG
1. Entry immédiate (market order OK)
2. Stop-loss plus large (volatilité)
3. Sizing : jusqu'à 3% du capital à risque
4. Trailing stop recommandé

---

## 📊 Catalysts par Impact

| Type | Impact Typique | Timing |
|------|----------------|--------|
| FDA_APPROVAL | +50% à +200% | Immédiat |
| MERGER/ACQUISITION | +30% à +100% | 1-3 jours |
| EARNINGS_BEAT | +20% à +80% | PM/RTH open |
| GUIDANCE_RAISE | +15% à +50% | PM/RTH open |
| CONTRACT_WIN | +10% à +40% | Variable |
| ANALYST_UPGRADE | +5% à +20% | Variable |

---

## ⚠️ Risk Management

### Règles d'Or
1. **Stop-loss toujours** : Jamais de position sans stop
2. **Max 5 positions** : Diversification obligatoire
3. **Max 3% risk/trade** : Même sur BUY_STRONG
4. **Cut losses fast** : Si stop touché, sortir sans hésiter

### Sizing par Signal

| Signal | Risk Max | Position Typique |
|--------|----------|------------------|
| WATCH_EARLY | 0% | Pas de position |
| BUY | 2% | $2k sur $100k |
| BUY_STRONG | 3% | $3k sur $100k |

---

## 📈 Performance Attendue

| Métrique | Cible V5.1 |
|----------|-----------|
| Hit Rate | 50-65% |
| Early Catch (>2h avant) | 60-75% |
| Avg Win | +45-80% |
| Avg Loss | -8-15% |
| Win/Loss Ratio | 3:1 |
| Lead Time | 6-12h |

---

## 🔔 Sessions Clés

### After-Hours (16:00-20:00 ET)
- **Focus** : Détection précoce
- **Alertes** : WATCH_EARLY
- **Action** : Préparer watchlist

### Pre-Market (04:00-09:30 ET)
- **Focus** : Confirmation + entry
- **Alertes** : BUY, BUY_STRONG
- **Action** : Exécuter trades

### RTH (09:30-16:00 ET)
- **Focus** : Gestion positions
- **Alertes** : BUY_STRONG (rares)
- **Action** : Trailing stops, targets

---

---

## 📊 Interpréter les Composantes V5.3

### Options Flow (10%)

| Score | Signification | Action |
|-------|---------------|--------|
| 0.0-0.3 | Activité normale | Neutre |
| 0.3-0.6 | Activité légèrement élevée | Surveiller |
| 0.6-0.8 | Activité inhabituelle | Confirme le signal |
| 0.8-1.0 | Activité très élevée (smart money?) | Renforce confiance |

**Signaux positifs** :
- `HIGH_CALL_VOLUME` : Volume calls >= 5000
- `LOW_PC_RATIO` : Put/Call < 0.5 (bullish)
- `CALL_CONCENTRATION` : 70%+ du volume en calls

### Social Buzz (6%)

| Score | Signification | Action |
|-------|---------------|--------|
| 0.0-0.3 | Buzz normal | Neutre |
| 0.3-0.5 | Buzz croissant | Surveiller |
| 0.5-0.7 | Buzz élevé | Attention retail |
| 0.7-1.0 | Viral/Trending | Prudence (late?) |

**Sources (V5.3)** :
| Source | Poids | Notes |
|--------|-------|-------|
| Twitter/X | 45% | Via Grok API - Real-time |
| Reddit | 30% | PRAW API - WSB, stocks, pennystocks |
| StockTwits | 25% | API - Sentiment labels (Bullish/Bearish) |
| Google Trends | 0% | **Désactivé** (pytrends instable) |

**Sentiment Boost** : +15% si Reddit ou StockTwits montrent un sentiment fortement bullish (ratio > 2.0)

---

## 🔗 Ressources

- **Installation** : Voir `DEPLOYMENT.md`
- **Architecture** : Voir `README_DEV.md`
- **Configuration** : Voir `config.py`
- **Dashboard** : `streamlit run dashboards/streamlit_dashboard.py`

---

**Version:** 5.3.1
**Last Updated:** 2026-02-05
