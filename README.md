# GV2-EDGE V7.0 - Detection/Execution Separation Architecture

**Version 7.0 - Full Signal Pipeline with Risk Management**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()

---

## Nouveautes V7.0

### Architecture Detection/Execution Separation

**Principe fondamental: Detection JAMAIS bloquee, Execution uniquement limitee**

```
ANCIEN (V6):
  Signal blocked? -> No signal visible

NOUVEAU (V7):
  Detection -> ALWAYS produces signal (visible)
  Order -> ALWAYS calculated (visible)
  Execution -> ONLY layer with limits (transparent)
```

### 3-Layer Pipeline

| Layer | Module | Role | Bloque? |
|-------|--------|------|---------|
| 1 | **SignalProducer** | Detection de tous les signaux | JAMAIS |
| 2 | **OrderComputer** | Calcul d'ordre (size, stop, target) | JAMAIS |
| 3 | **ExecutionGate** | Application des limites | OUI (seul) |

### Nouveaux Modules V7

| Module | Description |
|--------|-------------|
| `src/engines/signal_producer.py` | Production de signaux (UnifiedSignal) |
| `src/engines/order_computer.py` | Calcul d'ordres (ProposedOrder) |
| `src/engines/execution_gate.py` | Gate d'execution (ExecutionDecision) |
| `src/risk_guard/` | Unified risk assessment (dilution, compliance, halt) |
| `src/market_memory/` | MRP/EP context enrichment |
| `src/pre_halt_engine.py` | Detection pre-halt risk |
| `src/ibkr_news_trigger.py` | Early news alerts via keywords |
| `src/api_pool/` | Multi-key API management |

### Market Memory (MRP/EP)

Contexte historique pour chaque signal:

- **MRP (Missed Recovery Potential)**: Score base sur les signaux manques precedents
- **EP (Edge Probability)**: Probabilite de succes basee sur patterns similaires
- **Auto-activation**: MRP/EP s'activent uniquement quand les donnees sont stables

```python
# Thresholds d'activation
MIN_TOTAL_MISSES = 50
MIN_TRADES_RECORDED = 30
MIN_PATTERNS_LEARNED = 10
MIN_TICKER_PROFILES = 20
```

### Pre-Halt Engine

Detection proactive du risque de halt:

| State | Risk Level | Action |
|-------|------------|--------|
| NORMAL | Low | Execute normal |
| ELEVATED | Medium | Reduce size 50% |
| HIGH | High | Block execution |

### Risk Guard

Assessment unifie des risques:

- **Dilution Detector**: ATM offerings, shelf registrations
- **Compliance Checker**: Delisting risk, SEC issues
- **Halt Monitor**: Current and imminent halts

---

## Vue d'Ensemble

**GV2-EDGE** est un systeme automatise de trading momentum concu pour detecter **tres tot** les top gainers small caps du marche americain.

### Objectif Principal

> Capter les mouvements explosifs **3 a 60 jours avant** qu'ils ne se produisent, avec transparence totale sur les signaux detectes vs executes.

### Ce que GV2-EDGE V7 fait

- Detecte TOUS les signaux (jamais bloque au niveau detection)
- Calcule TOUS les ordres (visible meme si non execute)
- Applique les limites UNIQUEMENT a l'execution
- Montre les raisons de blocage (transparence)
- Track les signaux manques pour apprentissage (Market Memory)
- Alerte via Telegram avec plans de trade complets

---

## Architecture V7.0

```
                    GV2-EDGE V7.0
        Detection/Execution Separation Architecture

+----------------------------------------------------------+
|  LAYER 1: SIGNAL PRODUCER (Detection - Never Blocked)    |
|  src/engines/signal_producer.py                          |
|  - Monster Score computation                             |
|  - Signal type determination (BUY/BUY_STRONG/WATCH)     |
|  - Pre-spike state evaluation                           |
|  Output: UnifiedSignal                                   |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  ENRICHMENT: MARKET MEMORY (Context - Informational)     |
|  src/market_memory/                                      |
|  - MRP (Missed Recovery Potential)                       |
|  - EP (Edge Probability)                                 |
|  - Auto-activates when data stable                       |
|  Output: context_mrp, context_ep, context_active         |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  LAYER 2: ORDER COMPUTER (Always Computed)               |
|  src/engines/order_computer.py                           |
|  - Position size calculation                             |
|  - Stop-loss placement                                   |
|  - Price target computation                              |
|  Output: ProposedOrder added to signal                   |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  RISK ASSESSMENT: UNIFIED GUARD (Informational)          |
|  src/risk_guard/                                         |
|  - Dilution risk (ATM, offerings)                        |
|  - Compliance risk (delisting, SEC)                      |
|  - Halt status (current/imminent)                        |
|  Output: RiskFlags for ExecutionGate                     |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  LAYER 3: EXECUTION GATE (Only Blocking Layer)           |
|  src/engines/execution_gate.py                           |
|  - Daily trade limit check                               |
|  - Capital sufficiency check                             |
|  - Risk flags evaluation                                 |
|  - Pre-halt state evaluation                             |
|  Output: ExecutionDecision (ALLOW/BLOCK + reasons)       |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|  OUTPUT (All Signals Visible)                            |
|  - Telegram Alerts (including blocked with reasons)      |
|  - Signal Logger (full history)                          |
|  - Market Memory (track misses for learning)             |
+----------------------------------------------------------+
```

---

## Configuration V7.0

### config.py - Nouvelles Options

```python
# V7.0 Architecture
USE_V7_ARCHITECTURE = True

# Execution Gate
DAILY_TRADE_LIMIT = 5
MAX_POSITION_PCT = 0.10
MAX_TOTAL_EXPOSURE = 0.80

# Pre-Halt Engine
ENABLE_PRE_HALT_ENGINE = True
PRE_HALT_VOLATILITY_THRESHOLD = 3.0

# Risk Guard
ENABLE_RISK_GUARD = True
RISK_BLOCK_ON_CRITICAL = True

# Market Memory
ENABLE_MARKET_MEMORY = True
MARKET_MEMORY_MIN_MISSES = 50
```

---

## Performance

### Metrics V7

| Metrique | Valeur Cible | Notes |
|----------|--------------|-------|
| **Detection Rate** | **100%** | Tous les signaux detectes (jamais bloques) |
| **Execution Rate** | **60-80%** | Signaux autorises par ExecutionGate |
| **Hit Rate** | **70-80%** | Top gainers detectes |
| **Early Catch Rate** | **60-70%** | Detection >2h avant explosion |
| **MRP/EP Correlation** | **>0.6** | Quand actif |

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
cd GV2-EDGE-V7

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

## Utilisation

### Demarrage

```bash
source venv/bin/activate
python main.py
```

### Workflow V7

```
03:00 AM UTC  -> Generate daily WATCH list
04:00-09:30 ET -> Pre-market: V7 cycle (detection + execution)
09:30-16:00 ET -> RTH: V7 cycle every 3 min
16:00-20:00 ET -> After-hours: Anticipation scanning
20:30 UTC     -> Daily Audit V7
Friday 22:00  -> Weekly Deep Audit V7
```

---

## Documentation

| Fichier | Description |
|---------|-------------|
| `README.md` | Vue d'ensemble (ce fichier) |
| `README_DEV.md` | Guide developpeur V7 |
| `README_TRADER.md` | Guide trader V7 |
| `DEPLOYMENT.md` | Guide deploiement |
| `QUICKSTART.md` | Demarrage rapide |

---

## Changelog V7.0

- **Detection/Execution Separation**: 3-layer pipeline
- **SignalProducer**: Detection never blocked
- **OrderComputer**: Orders always computed
- **ExecutionGate**: Only blocking layer with transparency
- **UnifiedSignal**: Complete signal state object
- **Risk Guard**: Unified risk assessment
- **Pre-Halt Engine**: Proactive halt risk detection
- **Market Memory**: MRP/EP context enrichment
- **IBKR News Trigger**: Keyword-based early alerts
- **API Pool**: Multi-key management
- **Missed Tracker**: Learn from blocked signals

---

**GV2-EDGE V7.0 - Detection/Execution Separation Architecture**

*Detectez TOUS les signaux. Controlez l'execution.*

---

**Version:** 7.0.0
**Last Updated:** 2026-02-12
**Status:** Production Ready
