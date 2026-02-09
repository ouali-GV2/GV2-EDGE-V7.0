# GV2-EDGE V6.0 - Anticipation Multi-Layer System

**Version 6.0 - Full Anticipation Intelligence**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()

---

## Nouveautes V6.0

### Architecture Anticipation Multi-Couches

**5 nouveaux modules d'intelligence pour detecter les mouvements AVANT tout le monde:**

| Couche | Module | Description |
|--------|--------|-------------|
| 1 | **Market Calendar US** | Jours feries NYSE, demi-seances, ajustement volumes |
| 2 | **Repeat Gainer Memory** | Tracking historique des repeat runners avec decay |
| 3 | **Pre-Spike Radar** | Detection acceleration AVANT spike (4 signaux) |
| 4 | **Catalyst Score V3** | Scoring par type, temporal decay, confluence |
| 5 | **NLP Enrichi** | Sentiment avance, entites, categories, urgence |

### Unified EVENT_TYPE Taxonomy

**18 types d'evenements en 5 tiers:**

```
TIER 1 - CRITICAL (0.90-1.00):
  FDA_APPROVAL, PDUFA_DECISION, BUYOUT_CONFIRMED

TIER 2 - HIGH (0.75-0.89):
  FDA_TRIAL_POSITIVE, BREAKTHROUGH_DESIGNATION, FDA_FAST_TRACK,
  MERGER_ACQUISITION, EARNINGS_BEAT_BIG, MAJOR_CONTRACT

TIER 3 - MODERATE (0.60-0.74):
  GUIDANCE_RAISE, EARNINGS_BEAT, PARTNERSHIP, PRICE_TARGET_RAISE

TIER 4 - LOW-MOD (0.45-0.59):
  ANALYST_UPGRADE, SHORT_SQUEEZE_SIGNAL, UNUSUAL_VOLUME_NEWS

TIER 5 - SPECULATIVE (0.30-0.44):
  BUYOUT_RUMOR, SOCIAL_MEDIA_SURGE, BREAKING_POSITIVE
```

### Pre-Spike Radar (Nouveau)

Detection d'acceleration AVANT le spike (pas le niveau, la derivee):

| Signal | Description |
|--------|-------------|
| Volume Acceleration | Taux de changement volume croissant |
| Bid-Ask Tightening | Spread qui se resserre |
| Price Compression | Bollinger squeeze avant breakout |
| Dark Pool Activity | Activite inhabituelle hors marche |

**Alert Levels:** NONE < WATCH < ELEVATED < HIGH

### Catalyst Score V3 (Nouveau)

- **Type Weighting**: FDA > Earnings > Contract
- **Temporal Decay**: Half-life 24h (events frais > anciens)
- **Quality Assessment**: Fiabilite source + confirmation multi-sources
- **Confluence Scoring**: Plusieurs catalysts = score plus eleve
- **Performance Tracking**: Apprentissage historique

### NLP Enrichi (Nouveau)

- **Sentiment Avance**: Bullish/Bearish avec intensite et confiance
- **Entity Extraction**: Tickers, personnes, produits, chiffres cles
- **13 Categories News**: FDA_REGULATORY, MERGER_ACQUISITION, EARNINGS...
- **5 Niveaux Urgence**: BREAKING, HIGH, MEDIUM, LOW, STALE
- **Multi-Source Aggregation**: Time-weighted sentiment

---

## Vue d'Ensemble

**GV2-EDGE** est un systeme automatise de trading momentum concu pour detecter **tres tot** les top gainers small caps du marche americain, idealement **avant ou au tout debut** de leurs hausses majeures (+50%, +100%, +200%).

### Objectif Principal

> Capter les mouvements explosifs **3 a 60 jours avant** qu'ils ne se produisent, avec un systeme rapide, robuste et oriente performance reelle.

### Ce que GV2-EDGE fait

- Predit les mouvements 7-60 jours a l'avance (via calendar events & intelligence)
- Anticipe les setups 1-3 jours avant (via historical beat rate & social buzz)
- Detecte en temps reel pendant le pre-market (4:00-9:30 AM)
- Alerte via Telegram avec plans de trade complets
- S'ameliore continuellement via audits automatiques

---

## Architecture V6.0

```
                    GV2-EDGE V6.0
            Anticipation Multi-Layer System

+----------------------------------------------------------+
|  COUCHE 1: MARKET CALENDAR US                            |
|  utils/market_calendar.py                                |
|  - Jours feries NYSE 2024-2027                          |
|  - Demi-seances (early close)                           |
|  - Ajustement volumes pour comparaison                  |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  COUCHE 2: REPEAT GAINER MEMORY                          |
|  src/repeat_gainer_memory.py                             |
|  - Historique des top gainers                           |
|  - Score "repeat runner" avec decay                      |
|  - Boost multiplicateur Monster Score (1.5x max)        |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  COUCHE 3: PRE-SPIKE RADAR                               |
|  src/pre_spike_radar.py                                  |
|  - Volume acceleration (derivee)                        |
|  - Options acceleration                                  |
|  - Buzz acceleration                                     |
|  - Technical compression                                 |
|  - Alert levels: NONE < WATCH < ELEVATED < HIGH         |
|  - Boost anticipatif (1.4x max)                         |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  COUCHE 4: CATALYST SCORE V3                             |
|  src/catalyst_score_v3.py                                |
|  - Type weighting (5 tiers)                             |
|  - Temporal decay (half-life 24h)                       |
|  - Quality assessment                                    |
|  - Confluence multi-catalyst                            |
|  - Historical performance tracking                       |
|  - Boost multiplicateur (1.6x max)                      |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  COUCHE 5: NLP ENRICHI                                   |
|  src/nlp_enrichi.py                                      |
|  - Enhanced sentiment analysis                           |
|  - Entity extraction                                     |
|  - 13 news categories                                    |
|  - 5 urgency levels                                      |
|  - Multi-source aggregation                              |
|  - Boost multiplicateur (0.7x - 1.4x)                   |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  SIGNAL ENGINE                                           |
|  - WATCH_EARLY: Catalyst detecte, potentiel             |
|  - BUY: Score 0.65+ confirme                            |
|  - BUY_STRONG: Score 0.80+ avec catalyst fort           |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  OUTPUT & AUDIT                                          |
|  - Telegram Alerts V6 (emojis EVENT_TYPE, tiers)        |
|  - Daily Audit V6 (module performance)                  |
|  - Weekly Audit V6 (trend analysis)                     |
|  - Dashboard V6 (module status)                         |
+----------------------------------------------------------+
```

---

## Performance

### Metrics Attendues V6

| Metrique | Valeur Cible | Notes |
|----------|--------------|-------|
| **Hit Rate** | **70-80%** | Ameliore par Pre-Spike Radar + Catalyst V3 |
| **Early Catch Rate** | **60-70%** | Detection >2h avant explosion |
| **Avg Lead Time** | **8-40 jours** | WATCH signals (calendar prediction) |
| **Optimal Lead Time** | **4-8 heures** | BUY_STRONG (PM 4:00-9:30) |
| **False Positive Rate** | **20-30%** | Reduit par NLP Enrichi |
| **Max Drawdown** | **<15%** | Protection capital |

---

## Installation

### Prerequis

- **Python 3.8+**
- **IBKR Account** (paper trading ou live)
- **IB Gateway ou TWS** installe et configure
- **API Keys:** Grok (X.AI), Finnhub, Telegram

### Installation Rapide

```bash
# 1. Clone le projet
git clone <repo_url>
cd GV2-EDGE-V6

# 2. Environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Editer .env avec vos API keys
```

---

## Configuration

### Environment Variables (.env)

```bash
# Required
GROK_API_KEY=xai-...
FINNHUB_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# IBKR
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=paper, 7496=live

# Optional Social APIs
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
STOCKTWITS_ACCESS_TOKEN=...
```

---

## Utilisation

### Demarrage

```bash
source venv/bin/activate
python main.py
```

### Workflow Automatique

```
03:00 AM UTC  -> Generate daily WATCH list
04:00-09:30 ET -> Pre-market scanning
09:30-16:00 ET -> Regular market monitoring
16:00-20:00 ET -> After-hours catalyst scanning
20:30 UTC     -> Daily Audit V6
Friday 22:00  -> Weekly Deep Audit V6
```

---

## Documentation

| Fichier | Description |
|---------|-------------|
| `README.md` | Vue d'ensemble (ce fichier) |
| `README_DEV.md` | Guide developpeur V6 |
| `README_TRADER.md` | Guide trader V6 |
| `DEPLOYMENT.md` | Guide deploiement |
| `QUICKSTART.md` | Demarrage rapide |

---

## Changelog V6.0

- **Market Calendar US**: NYSE holidays + early closes
- **Repeat Gainer Memory**: Historical spike tracking with decay
- **Pre-Spike Radar**: 4-signal acceleration detection before spike
- **Catalyst Score V3**: Type weighting + temporal decay + confluence
- **NLP Enrichi**: Advanced sentiment with entity extraction
- **Unified EVENT_TYPE Taxonomy**: 18 types in 5 tiers
- **Telegram Alerts V6**: Emojis par tier, Pre-Spike alerts, Repeat badges
- **Daily/Weekly Audit V6**: V6 module performance tracking
- **Dashboard V6**: V6 modules status section

---

**GV2-EDGE V6.0 - Anticipation Multi-Layer System**

*Detectez les top gainers AVANT tout le monde.*

---

**Version:** 6.0.0
**Last Updated:** 2026-02-09
**Status:** Production Ready
