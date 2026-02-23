# GV2-EDGE V9.0 - Documentation Complete

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture du Système](#architecture-du-système)
3. [Installation et Configuration](#installation-et-configuration)
4. [Modules Principaux](#modules-principaux)
5. [Pipeline de Génération de Signaux](#pipeline-de-génération-de-signaux)
6. [Types de Signaux](#types-de-signaux)
7. [Monster Score - Calcul du Score](#monster-score---calcul-du-score)
8. [Gestion des Risques (Risk Guard)](#gestion-des-risques-risk-guard)
9. [Market Memory - Système d'Apprentissage](#market-memory---système-dapprentissage)
10. [Weekend Mode - Préparation Monday](#weekend-mode---préparation-monday)
11. [API Pool - Gestion Multi-Clés](#api-pool---gestion-multi-clés)
12. [Sources de Données](#sources-de-données)
13. [Configuration Avancée](#configuration-avancée)
14. [Alertes Telegram](#alertes-telegram)
15. [Maintenance et Monitoring](#maintenance-et-monitoring)

---

## Vue d'Ensemble

### Qu'est-ce que GV2-EDGE?

GV2-EDGE est un système automatisé de détection de momentum conçu pour identifier les actions small-cap US ($50M - $2B market cap) susceptibles de connaître des gains majeurs (50% à 500%+) **AVANT** que le marché général ne les reconnaisse.

### Philosophie

| Horizon | Objectif | Méthode |
|---------|----------|---------|
| 7-60 jours | Détection précoce | Calendrier événements (FDA, earnings) |
| 1-3 jours | Anticipation | Patterns historiques + buzz social |
| 4-8 heures | Capture temps réel | Pre-market scanning |

### Ce que GV2-EDGE fait

- ✅ Détecte les signaux de trading (BUY, BUY_STRONG)
- ✅ Calcule les niveaux d'entrée, stop-loss, et taille de position
- ✅ Envoie des alertes Telegram en temps réel
- ✅ Analyse les catalyseurs (FDA, earnings, M&A, etc.)
- ✅ Évalue les risques (dilution, delisting, halts)
- ✅ Apprend de l'historique pour s'améliorer

### Ce que GV2-EDGE ne fait PAS

- ❌ Passer des ordres automatiquement
- ❌ Gérer ton compte broker
- ❌ Garantir des profits

---

## Architecture du Système

### Structure des Répertoires

```
GV2-EDGE-V9.0/
├── main.py                    # Point d'entree principal
├── config.py                  # Configuration centralisee
├── .env                       # Variables d'environnement (API keys)
│
├── src/                       # Code source principal
│   ├── engines/               # Moteurs V7/V8/V9 (coeur du systeme)
│   │   ├── signal_producer.py        # Layer 1: Detection V8
│   │   ├── order_computer.py         # Layer 2: Calcul ordres
│   │   ├── execution_gate.py         # Layer 3: Gate execution
│   │   ├── acceleration_engine.py    # V8: Derivees + z-scores
│   │   ├── smallcap_radar.py         # V8: Radar anticipatif
│   │   ├── ticker_state_buffer.py    # V8: Ring buffer 120pts
│   │   └── multi_radar_engine.py     # V9: 4 radars + confluence
│   ├── ibkr_streaming.py     # V9: Streaming temps reel IBKR
│   ├── finnhub_ws_screener.py # V8: WebSocket Finnhub
│   ├── top_gainers_source.py  # V8: Source top gainers
│   ├── models/                # Types et structures de donnees
│   ├── risk_guard/            # Gestion des risques (V8)
│   ├── market_memory/         # Systeme d'apprentissage (V7)
│   ├── weekend_mode/          # Mode weekend (V7)
│   ├── api_pool/              # Gestion multi-cles API (V7)
│   ├── event_engine/          # Detection d'evenements
│   ├── ingestors/             # Ingestion de donnees
│   ├── processors/            # Traitement de donnees
│   ├── schedulers/            # Planification des taches
│   ├── monitors/              # Monitoring pipeline
│   ├── scoring/               # Calcul des scores
│   ├── boosters/              # Boosters additifs (insider, squeeze)
│   └── social_engine/         # Analyse sociale
│
├── utils/                     # Utilitaires
├── alerts/                    # Systeme d'alertes
├── monitoring/                # Surveillance systeme
├── data/                      # Donnees runtime
└── logs/                      # Fichiers de log
```

### Flux Principal

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN.PY                                  │
│                     (Point d'entrée)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET SESSION LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│  AFTER-HOURS (16:00-20:00 ET)                                   │
│  └─ News Flow Screener → Options Flow → Anticipation Engine     │
│                                                                  │
│  PRE-MARKET (04:00-09:30 ET)                                    │
│  └─ Anticipation Scan → Signal Upgrades (WATCH → BUY)           │
│                                                                  │
│  REGULAR (09:30-16:00 ET)                                       │
│  └─ Edge Cycle (toutes les 3 min) → Génération Signaux          │
│                                                                  │
│  CLOSED                                                          │
│  └─ Sleep → Audits → Weekend Mode                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EDGE CYCLE                                  │
├─────────────────────────────────────────────────────────────────┤
│  Pour chaque ticker dans l'univers:                             │
│  1. Signal Producer    → Génère le signal                       │
│  2. Risk Guard         → Évalue les risques                     │
│  3. Order Computer     → Calcule entry/stop/size                │
│  4. Context Scorer     → Ajuste selon l'historique              │
│  5. Execution Gate     → Applique les limites                   │
│  6. Telegram Alert     → Notifie si signal                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation et Configuration

### Prérequis

- Python 3.10+
- Compte Interactive Brokers (pour les données temps réel)
- Clés API: Finnhub, Grok (optionnel)
- Bot Telegram (pour les alertes)

### Installation

```bash
# Cloner le repository
git clone <repo-url>
cd GV2-EDGE-V5.1

# Installer les dépendances
pip install -r requirements.txt

# Copier le template de configuration
cp .env.example .env

# Éditer .env avec tes clés API
nano .env
```

### Configuration .env

```bash
# OBLIGATOIRE - Données marché
FINNHUB_API_KEY=your_finnhub_key

# OBLIGATOIRE - Alertes
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# OPTIONNEL - NLP avancé
GROK_API_KEY=your_grok_key

# OPTIONNEL - Données IBKR (recommandé)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497

# OPTIONNEL - Social
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
STOCKTWITS_ACCESS_TOKEN=your_stocktwits_token
```

### Lancer le Système

```bash
# Mode normal
python main.py

# Avec logs détaillés
python main.py --debug

# En arrière-plan
nohup python main.py > output.log 2>&1 &
```

---

## Modules Principaux

### 1. Signal Producer V8 (`src/engines/signal_producer.py`)

**Role:** Layer 1 — Detection illimitee de signaux. Ne bloque JAMAIS.

**Integration V8/V9:**
- AccelerationEngine : detecte ACCUMULATING/LAUNCHING avant le mouvement
- SmallCapRadar : phases ACCUMULATING → PRE_LAUNCH → LAUNCHING → BREAKOUT
- Multi-Radar V9 : 4 radars paralleles (Flow, Catalyst, Smart Money, Sentiment)

### 1bis. Signal Engine Legacy (`src/signal_engine.py`)

**Role:** [LEGACY] Module original, delegue au SignalProducer V8.

**Output:**
```python
{
    "ticker": "AAPL",
    "signal": "BUY_STRONG",
    "confidence": 0.85,
    "monster_score": 0.82,
    "components": {
        "event": 0.9,
        "momentum": 0.7,
        "volume": 0.8,
        ...
    }
}
```

### 2. Universe Loader (`src/universe_loader.py`)

**Rôle:** Construit l'univers des tickers à scanner.

**Filtres appliqués:**
| Critère | Valeur | Raison |
|---------|--------|--------|
| Market Cap | $50M - $2B | Small caps avec potentiel |
| Prix | $1 - $20 | Évite penny stocks et blue chips |
| Volume moyen | > 500K | Liquidité suffisante |
| Exchange | NYSE, NASDAQ | Pas d'OTC |

**Résultat:** ~300-500 tickers qualifiés

### 3. Portfolio Engine (`src/portfolio_engine.py`)

**Rôle:** Calcule la taille de position et les niveaux de stop.

**Calcul de position:**
```
Stop Distance = ATR × 2.0
Shares = (Capital × Risk%) / Stop Distance

Exemple:
- Capital: $10,000
- Risk: 2%
- ATR: $0.50
- Stop Distance: $1.00
- Shares = ($10,000 × 0.02) / $1.00 = 200 shares
```

**Output:**
```python
{
    "ticker": "XYZ",
    "entry": 5.00,
    "stop": 4.00,
    "shares": 200,
    "risk_amount": 200.00  # $200 max loss
}
```

### 4. Event Engine (`src/event_engine/`)

**Rôle:** Détecte et classifie les catalyseurs.

**Sources:**
- Finnhub (news, earnings calendar)
- SEC EDGAR (8-K filings)
- FDA Calendar (PDUFA dates)

**Classification des événements:**

| Tier | Impact | Exemples |
|------|--------|----------|
| 1 | 0.90-1.00 | FDA Approval, Buyout Confirmed |
| 2 | 0.75-0.89 | Positive Trial, M&A, Big Earnings Beat |
| 3 | 0.60-0.74 | Guidance Raise, Partnership |
| 4 | 0.45-0.59 | Analyst Upgrade, Short Squeeze Setup |
| 5 | 0.30-0.44 | Rumor, Social Surge |

### 5. Anticipation Engine (`src/anticipation_engine.py`)

**Rôle:** Détecte les opportunités en after-hours et pre-market.

**Fonctionnement:**
1. Scan des news après la clôture
2. Détection des gaps pre-market
3. Analyse du flow d'options
4. Upgrade des signaux WATCH → BUY si confirmés

---

## Pipeline de Génération de Signaux

### Étape par Étape

```
1. CHARGEMENT UNIVERS
   └─ 300-500 tickers small-cap qualifiés

2. POUR CHAQUE TICKER:

   ┌─────────────────────────────────────────────────────────┐
   │ A. DÉTECTION ÉVÉNEMENTS                                 │
   │    ├─ Fetch news (Finnhub)                              │
   │    ├─ Fetch earnings calendar                           │
   │    ├─ Fetch FDA calendar                                │
   │    ├─ Classification NLP                                │
   │    └─ Score impact: 0.0 - 1.0                          │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ B. CALCUL FEATURES                                      │
   │    ├─ Momentum (accélération prix)                      │
   │    ├─ Volume spike ratio                                │
   │    ├─ VWAP deviation                                    │
   │    ├─ Bollinger squeeze                                 │
   │    └─ ATR volatility                                    │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ C. MÉTRIQUES PRE-MARKET                                 │
   │    ├─ Gap % vs close                                    │
   │    ├─ PM volume                                         │
   │    ├─ PM → RTH transition                               │
   │    └─ Position dans range                               │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ D. ANALYSE PATTERNS                                     │
   │    ├─ Consolidation, higher lows, flags                 │
   │    ├─ Volume accumulation/climax                        │
   │    └─ Bollinger squeeze compression                     │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ E. MONSTER SCORE                                        │
   │    Weighted sum de tous les composants                  │
   │    Score final: 0.0 - 1.0                              │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ F. DÉTERMINATION SIGNAL                                 │
   │    ├─ Score >= 0.80 → BUY_STRONG                       │
   │    ├─ Score >= 0.65 → BUY                              │
   │    └─ Score < 0.65  → HOLD (pas d'alerte)              │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ G. RISK GUARD CHECK                                     │
   │    ├─ Dilution risk?                                    │
   │    ├─ Compliance issue?                                 │
   │    ├─ Halt imminent?                                    │
   │    └─ Action: ALLOW / REDUCE / BLOCK                    │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ H. CONTEXT SCORER (Market Memory)                       │
   │    ├─ MRP: Similar misses outcome                       │
   │    ├─ EP: Historical edge probability                   │
   │    ├─ Ajustement score: -30 à +30                      │
   │    └─ Multiplicateur taille: 0.25x à 1.5x              │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ I. EXECUTION GATE                                       │
   │    ├─ Limite 5 trades/jour?                            │
   │    ├─ Capital disponible?                               │
   │    ├─ Position max atteinte?                            │
   │    └─ EXÉCUTÉ ou BLOQUÉ (mais toujours visible)        │
   └─────────────────────────────────────────────────────────┘
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │ J. ALERTE TELEGRAM                                      │
   │    Si signal BUY ou BUY_STRONG:                        │
   │    → Envoie notification avec tous les détails          │
   └─────────────────────────────────────────────────────────┘
```

---

## Types de Signaux

### WATCH_EARLY 👀

**Quand:** Catalyseur détecté mais score < 0.65

**Signification:** Opportunité potentielle en formation

**Action recommandée:** Surveiller pour upgrade

**Exemple d'alerte:**
```
👀 WATCH_EARLY - XYZ
Score: 0.58
Catalyst: FDA_TRIAL_POSITIVE
Note: Monitoring for confirmation
```

### BUY ✅

**Quand:** Monster score >= 0.65

**Signification:** Setup confirmé, prêt à entrer

**Action recommandée:** Entrée standard

**Sizing:** 2% du capital à risque

**Exemple d'alerte:**
```
✅ BUY - XYZ
Score: 0.72
Entry: $5.25
Stop: $4.75 (-9.5%)
Shares: 150
Risk: $75 (2%)
Catalyst: EARNINGS_BEAT
```

### BUY_STRONG 🚀

**Quand:** Monster score >= 0.80

**Signification:** Opportunité majeure

**Action recommandée:** Entrée immédiate

**Sizing:** 2.5% du capital à risque

**Exemple d'alerte:**
```
🚀 BUY_STRONG - XYZ
Score: 0.85
Entry: $5.25
Stop: $4.50 (-14.3%)
Shares: 200
Risk: $150 (2.5%)
Catalyst: FDA_APPROVAL
Urgency: IMMEDIATE
```

---

## Monster Score - Calcul du Score

### Formule

```
Monster Score = Σ (Weight × Component Score)
```

### Composants et Poids

| Composant | Poids | Description |
|-----------|-------|-------------|
| **Event** | 25% | Impact du catalyseur (FDA, earnings, etc.) |
| **Volume** | 17% | Spike de volume vs moyenne |
| **Pattern** | 17% | Patterns techniques (flags, consolidation) |
| **PM Transition** | 13% | Momentum pre-market → regular |
| **Options Flow** | 10% | Activite options inhabituelle |
| **Acceleration** | 7% | [V8] Derivees + z-scores |
| **Momentum** | 4% | [Reduit V8] Velocite prix |
| **Squeeze** | 4% | Compression Bollinger Bands |
| **Social Buzz** | 3% | [Reduit V8] Mentions reseaux sociaux |
| **Total** | **100%** | |

### Exemple de Calcul

```
XYZ Corp:
- Event: 0.90 (FDA approval) × 0.25 = 0.225
- Volume: 0.80 (4x average) × 0.17 = 0.136
- Pattern: 0.70 (bull flag) × 0.17 = 0.119
- PM Trans: 0.60 × 0.13 = 0.078
- Options: 0.85 (unusual calls) × 0.10 = 0.085
- Momentum: 0.65 × 0.08 = 0.052
- Social: 0.50 × 0.06 = 0.030
- Squeeze: 0.40 × 0.04 = 0.016

Monster Score = 0.741 → BUY Signal
```

### Confluence Boost

Si plusieurs composants sont forts simultanément:
- Event > 0.6 AND Momentum > 0.6 AND Volume > 0.6 → +15%
- Squeeze > 0.7 → +10%
- PM Gap > 0.5 → +10%

Score final plafonné à 1.0

---

## Gestion des Risques (Risk Guard)

### Architecture

```
src/risk_guard/
├── unified_guard.py        # Orchestrateur central
├── dilution_detector.py    # Risque dilution
├── compliance_checker.py   # Risque delisting
└── halt_monitor.py         # Risque de halt
```

### 1. Dilution Detector

**Détecte:**
- S-3 shelf registrations
- 424B prospectus (offering imminent)
- ATM programs actifs
- PIPE deals
- Toxic financing (variable rate converts)

**Actions:**

| Risque | Score | Action |
|--------|-------|--------|
| Active Offering | 70+ | BLOCK |
| Toxic Financing | 70+ | BLOCK |
| ATM Active | 45-69 | REDUCE (x0.25) |
| Recent S-3 | 25-44 | REDUCE (x0.50) |

### 2. Compliance Checker

**Détecte:**
- Prix < $1 pendant 30+ jours (NASDAQ rule)
- Deficiency notices
- Delisting warnings
- Reverse split pending

**Actions:**

| Risque | Score | Action |
|--------|-------|--------|
| Delisting Pending | 70+ | BLOCK |
| Hearing Scheduled | 50-69 | REDUCE (x0.25) |
| Deficiency Notice | 25-49 | REDUCE (x0.50) |

### 3. Halt Monitor

**Détecte:**
- Prix proche du LULD band
- Volatilité extrême
- News pending probable
- Historique de halts fréquents

**Actions:**

| Risque | Probabilité | Action |
|--------|-------------|--------|
| Imminent | >80% | BLOCK |
| High | 50-80% | REDUCE (x0.25) |
| Elevated | 25-50% | REDUCE (x0.50) |

### Utilisation

```python
from src.risk_guard import get_unified_guard

guard = get_unified_guard()
assessment = await guard.assess("MULN", current_price=0.45)

if assessment.is_blocked:
    print(f"BLOCKED: {assessment.block_reasons}")
    # Output: BLOCKED: ['ACTIVE_OFFERING', 'DELISTING_RISK']
else:
    adjusted_size = base_size * assessment.position_multiplier
    # Output: adjusted_size = 100 * 0.25 = 25 shares
```

---

## Market Memory - Système d'Apprentissage

### Architecture

```
src/market_memory/
├── missed_tracker.py     # Track des opportunités manquées
├── pattern_learner.py    # Apprentissage des patterns
├── context_scorer.py     # Scoring contextuel (MRP/EP)
└── memory_store.py       # Persistence (JSON/SQLite)
```

### 1. Missed Tracker

**Fonction:** Track ce que tu as raté et pourquoi.

**Exemple:**
```
Signal: AAPL BUY_STRONG @ $150
Raison du miss: DAILY_TRADE_LIMIT
───────────────────────────────
3 jours plus tard:
Prix: $165 (+10%)
Outcome: WIN
Lesson: "Missed $750 gain due to trade limit"
```

**Utilité:**
- Identifier les patterns de miss
- Ajuster les priorités de signaux
- Améliorer la stratégie

### 2. Pattern Learner

**Fonction:** Apprend de ton historique de trades.

**Output par ticker:**
```
AAPL Profile:
├── Total trades: 47
├── Win rate: 67%
├── Avg gain: +4.2%
├── Avg loss: -2.1%
├── Best time: MORNING (9:30-11:30)
├── Best day: Tuesday
├── Avg hold: 4.2 hours
├── Flag: FAVORABLE ✓
└── Trend: IMPROVING ↑
```

### 3. Context Scorer (MRP/EP)

**MRP - Missed Recovery Potential:**
```
"Sur les 20 derniers miss similaires pour AAPL:
 - 14 sont devenus des winners (70%)
 - MRP Score: 72/100"

→ Recommandation: Considérer override du block
```

**EP - Edge Probability:**
```
Base win rate: 55%
+ Ticker bonus: +5 (historique favorable)
+ Time bonus: +8 (c'est le matin)
+ Pattern bonus: +3
─────────────────
EP Score: 71/100
```

**Output final:**
```python
context = scorer.score("AAPL", "BUY_STRONG", 75.0, 150.0)

context.signal_adjustment  # +12 points
context.size_multiplier    # 1.15x
context.action             # "EXECUTE"
context.reasoning          # ["67% win rate", "Optimal time"]
```

---

## Weekend Mode - Préparation Monday

### Architecture

```
src/weekend_mode/
├── weekend_scheduler.py   # Orchestration des phases
├── weekend_scanner.py     # Scan full universe
├── batch_processor.py     # Calculs lourds
└── monday_prep.py         # Génération watchlist
```

### Phases d'Exécution

```
VENDREDI 16:00 ─► MARKET_CLOSE
                  └─ Cleanup des données

VENDREDI 18:00 ─► FRIDAY_EVENING
                  └─ Backfill data historique
                  └─ Calculs lourds (pas de rate limit)

SAMEDI ─────────► SATURDAY
                  └─ Scan 8000+ tickers
                  └─ Analyse technique complète
                  └─ Check tous les SEC filings

DIMANCHE AM ────► SUNDAY_MORNING
                  └─ Analyse rotation sectorielle

DIMANCHE PM ────► SUNDAY_AFTERNOON
                  └─ Analyse sentiment news weekend

DIMANCHE SOIR ──► SUNDAY_EVENING
                  └─ Génération Monday Prep
                  └─ Tri par priorité

LUNDI 04:00 ────► PRE_MARKET
                  └─ Cache warming
                  └─ Notification "Ready"
```

### Output Monday Prep

```python
monday_plan = prep.get_current_plan()

monday_plan.primary_focus    # Top 10 tickers (score > 75)
# ['NVDA', 'AMD', 'TSLA', 'META', 'AAPL']

monday_plan.secondary_focus  # 20 suivants (score 60-75)
# ['MSFT', 'GOOGL', 'AMZN', ...]

monday_plan.avoid_list       # À ne pas toucher
# ['MULN', 'BBBY', 'AMC']

monday_plan.market_bias      # "BULLISH" / "BEARISH" / "NEUTRAL"
monday_plan.sector_leaders   # Secteurs forts
monday_plan.earnings_today   # Earnings du jour
```

### Utilisation

```python
from src.weekend_mode import get_weekend_scheduler

scheduler = get_weekend_scheduler()

# Créer le plan weekend
plan = scheduler.create_plan()

# Exécuter (tourne tout le weekend)
await scheduler.execute_plan(plan)

# Lundi matin: récupérer la watchlist
from src.weekend_mode import get_monday_prep
prep = get_monday_prep()
watchlist = prep.get_current_plan()

print(f"Focus today: {watchlist.primary_focus}")
```

---

## API Pool - Gestion Multi-Clés

### Architecture

```
src/api_pool/
├── pool_manager.py      # Orchestrateur central
├── key_registry.py      # Stockage des clés
├── request_router.py    # Routage intelligent
└── usage_tracker.py     # Tracking usage/limites
```

### Problème Résolu

```
AVANT (1 clé):
09:31 - Scan 50 tickers = 50 calls
09:31 - News check = 10 calls
09:31 - RATE LIMITED ❌
09:32 - Signal urgent... pas de data

APRÈS (pool de clés):
Clé 1 + Clé 2 + Clé 3 = 180 calls/min
Requêtes CRITICAL = quota réservé (30%)
Jamais de rate limit sur les urgences
```

### Niveaux de Priorité

| Priorité | Exemple | Quota |
|----------|---------|-------|
| CRITICAL | Halt check, execution-blocking | 30% réservé |
| HIGH | Hot ticker, breaking news | Best available |
| STANDARD | Normal scan | Round-robin |
| LOW | Background tasks | Any available |
| BATCH | Weekend processing | Least loaded |

### Configuration

```python
# Dans config.py ou .env
FINNHUB_API_KEYS = [
    "key1_xxxxxx",
    "key2_xxxxxx",
    "key3_xxxxxx"
]
```

### Utilisation

```python
from src.api_pool import get_pool_manager, Priority

pool = get_pool_manager()

# Requête critique (halt check)
async with pool.acquire("finnhub", "HALT_CHECK", Priority.CRITICAL) as key:
    response = await fetch(url, headers={"Token": key})

# Requête standard
async with pool.acquire("finnhub", "NEWS", Priority.STANDARD) as key:
    response = await fetch(url, headers={"Token": key})
```

---

## Sources de Données

### Données de Marché

| Source | Usage | Coût | Rate Limit |
|--------|-------|------|------------|
| **IBKR Level 1** | Quotes temps réel, bars | ~$10/mois | Illimité |
| **Finnhub** | Fallback data, news | Gratuit | 60/min |

### News et Catalyseurs

| Source | Données | Coût |
|--------|---------|------|
| **Finnhub** | Company news, press releases | Gratuit |
| **SEC EDGAR** | 8-K filings, corporate actions | Gratuit |
| **FDA Calendar** | PDUFA dates, trial results | Manuel |

### Sentiment Social

| Source | Poids | Configuration |
|--------|-------|---------------|
| **Twitter/X** | 45% | Via Grok API |
| **Reddit WSB** | 30% | REDDIT_CLIENT_ID/SECRET |
| **StockTwits** | 25% | STOCKTWITS_ACCESS_TOKEN |

### Options

| Source | Données | Coût |
|--------|---------|------|
| **IBKR OPRA** | Volume, bid/ask, IV | ~$1.50/mois |

---

## Configuration Avancée

### config.py - Paramètres Clés

#### Capital et Risque

```python
MANUAL_CAPITAL = 10000        # Capital de trading
RISK_BUY = 0.02               # 2% risque par BUY
RISK_BUY_STRONG = 0.025       # 2.5% risque par BUY_STRONG
MAX_OPEN_POSITIONS = 5        # Max positions simultanées
ATR_MULTIPLIER_STOP = 2.0     # Stop = entry - (ATR × 2)
```

#### Filtres Univers

```python
MAX_MARKET_CAP = 2_000_000_000  # $2B max
MIN_PRICE = 1.0                 # $1 min
MAX_PRICE = 20                  # $20 max
MIN_AVG_VOLUME = 500_000        # 500K volume min
EXCLUDE_OTC = True              # Pas d'OTC
```

#### Seuils de Signaux

```python
BUY_THRESHOLD = 0.65            # Score min pour BUY
BUY_STRONG_THRESHOLD = 0.80     # Score min pour BUY_STRONG
```

#### Poids Monster Score

```python
ADVANCED_MONSTER_WEIGHTS = {
    "event": 0.25,
    "volume": 0.17,
    "pattern": 0.17,
    "pm_transition": 0.13,
    "options_flow": 0.10,
    "momentum": 0.08,
    "social_buzz": 0.06,
    "squeeze": 0.04
}
```

#### Intervalles de Scan

```python
FULL_UNIVERSE_SCAN_INTERVAL = 300   # 5 min
EVENT_SCAN_INTERVAL = 600           # 10 min
PM_SCAN_INTERVAL = 60               # 1 min (premarket)
```

#### Features Optionnelles

```python
ENABLE_OPTIONS_FLOW = True
ENABLE_SOCIAL_BUZZ = True
ENABLE_PRE_SPIKE_RADAR = True
ENABLE_CATALYST_V3 = True
```

---

## Alertes Telegram

### Configuration

```bash
# .env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### Format des Alertes

#### BUY Signal
```
✅ BUY - XYZ

Score: 0.72 ████████░░
Entry: $5.25
Stop: $4.75 (-9.5%)
Target: $6.30 (+20%)

Shares: 150
Risk: $75 (2%)

Catalyst: EARNINGS_BEAT
Components:
├─ Event: 0.85
├─ Volume: 0.78
├─ Pattern: 0.65
└─ Momentum: 0.70

Session: PRE-MARKET
Time: 08:45:32 ET
```

#### BUY_STRONG Signal
```
🚀 BUY_STRONG - XYZ

Score: 0.87 █████████░
Entry: $5.25
Stop: $4.50 (-14.3%)
Target: $7.35 (+40%)

Shares: 200
Risk: $150 (2.5%)

Catalyst: FDA_APPROVAL
Urgency: IMMEDIATE ⚡

Components:
├─ Event: 0.95 ⭐
├─ Volume: 0.88
├─ Pattern: 0.82
├─ Options: 0.90 ⭐
└─ Momentum: 0.75

Session: PRE-MARKET
Time: 07:15:22 ET
```

#### Risk Warning
```
⚠️ RISK ALERT - XYZ

Signal: BUY blocked
Reason: DILUTION_RISK

Risk Flags:
├─ Active S-3 filing (2 days ago)
├─ ATM program: $50M capacity
└─ Recent insider selling

Recommendation: AVOID

Details: Check SEC filings
```

---

## Maintenance et Monitoring

### Logs

```bash
# Logs en temps réel
tail -f logs/gv2edge.log

# Logs d'erreur
tail -f logs/error.log

# Rechercher un ticker
grep "AAPL" logs/gv2edge.log
```

### Audits Automatiques

| Audit | Timing | Contenu |
|-------|--------|---------|
| Daily | 20:30 UTC | Performance du jour, signals générés |
| Weekly | Vendredi 22:00 UTC | Analyse profonde, ajustement poids |

### Health Checks

```python
# monitoring/system_guardian.py
# Vérifie automatiquement:
- Connexion API
- Taux d'erreur
- Latence
- Usage mémoire
```

### Commandes Utiles

```bash
# Status du système
python -c "from src.api_pool import get_pool_manager; print(get_pool_manager().get_stats())"

# Vérifier les positions (si IBKR connecté)
python -c "from src.ibkr_connector import get_connector; print(get_connector().get_positions())"

# Forcer un scan
python -c "from src.signal_engine import generate_many; generate_many(['AAPL', 'TSLA', 'NVDA'])"
```

### Troubleshooting

| Problème | Cause Probable | Solution |
|----------|----------------|----------|
| Pas de signaux | Rate limited | Vérifier API keys, ajouter au pool |
| Signaux retardés | IBKR déconnecté | Vérifier TWS/Gateway |
| Alertes manquantes | Bot Telegram | Vérifier token et chat_id |
| Score toujours bas | Pas de catalyseurs | Normal si marché calme |

---

## Resume des Modules V9.0

| Module | Fichiers | Fonction |
|--------|----------|----------|
| `engines/` | 7 | Detection V8 + Acceleration + Multi-Radar V9 |
| `api_pool/` | 5 | Gestion multi-cles API |
| `risk_guard/` | 5 | Evaluation des risques V8 (MIN-based) |
| `weekend_mode/` | 5 | Preparation weekend/Monday |
| `market_memory/` | 5 | Apprentissage historique |
| `ibkr_streaming.py` | 1 | V9: Streaming temps reel IBKR |
| `finnhub_ws_screener.py` | 1 | V8: WebSocket Finnhub |
| `top_gainers_source.py` | 1 | V8: Source top gainers |

---

## Contact et Support

- **Issues:** GitHub Issues
- **Logs:** `data/logs/gv2edge.log`
- **Config:** `config.py` et `.env`

---

*Documentation generee pour GV2-EDGE V9.0*
*Derniere mise a jour: 2026-02-21*
