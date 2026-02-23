# GV2-EDGE — System Reference & Development Guide

> **Version** : V9.0 (Architecture V7.0 + Acceleration V8 + Multi-Radar V9)
> **Derniere mise a jour** : 2026-02-21
> **Deploiement** : Hetzner CX43 (8 vCPU, 16 Go RAM, 160 Go SSD, headless Linux) + IBKR Gateway
> **Langage** : Python 3.11+ (asyncio + threading)

---

## 1. VISION

GV2-EDGE est un systeme de **detection anticipative** des top gainers small-cap US intraday.

**Objectif principal** : Identifier les actions susceptibles de gagner +30% a +300% en une journee, **AVANT** que le mouvement ne commence, en analysant les catalysts, le volume, les options, le sentiment social et les patterns techniques.

**Philosophie fondamentale** :
- La **detection ne s'arrete JAMAIS** — seule l'execution est limitee
- Le trader voit **TOUS** les signaux, meme ceux bloques par les limites
- Le systeme apprend de ses erreurs (Market Memory) et s'ameliore
- **Anticipation > Reaction** : detecter les signaux AVANT le spike, pas apres

**Cible de marche** :
- Small caps US (market cap < $2B)
- Prix $0.50 - $20
- Catalysts : FDA, M&A, Earnings, Contracts, Short Squeeze
- Sessions : Pre-market (04:00-09:30 ET), RTH (09:30-16:00 ET), After-hours (16:00-20:00 ET)

---

## 2. ARCHITECTURE

### 2.1 Pipeline Principal V7.0 (3 couches)

```
COUCHE 1 — SIGNAL PRODUCER (detection illimitee)
    Produit des signaux 24/7, jamais bloque
    Input:  MonsterScore + Catalyst + AccelerationState + PreSpike
    Output: UnifiedSignal (BUY_STRONG / BUY / WATCH / EARLY_SIGNAL / NO_SIGNAL)

COUCHE 2 — ORDER COMPUTER (calcul systematique)
    Calcule un ordre pour CHAQUE signal BUY/BUY_STRONG
    Input:  UnifiedSignal + prix + ATR + capital
    Output: ProposedOrder (shares, entry, stop, targets, R/R)

COUCHE 3 — EXECUTION GATE (limites et autorisations)
    SEULE couche avec des limites d'execution
    Input:  ProposedOrder + RiskFlags + AccountState
    Output: ExecutionDecision (ALLOWED / REDUCED / BLOCKED / DELAYED)
    Le signal original est TOUJOURS preserve (signal_preserved=True)
```

### 2.2 Cycle Principal (`main.py`)

```
run_edge() — Boucle infinie
│
├── After-Hours (16:00-20:00 ET) — MODE ANTICIPATION
│   ├── News Flow Screener (SEC + Finnhub)
│   ├── Extended Hours Gap Scan
│   ├── Options Flow Scan
│   ├── Anticipation Engine (mode "afterhours")
│   └── After-Hours Scanner
│   Sleep: 600s (10 min)
│
├── Pre-Market (04:00-09:30 ET) — MODE CONFIRMATION
│   ├── Anticipation Engine (mode "premarket")
│   ├── Signal Upgrades (WATCH_EARLY → BUY)
│   └── edge_cycle_v7() [pipeline complet]
│   Sleep: 300s (5 min)
│
├── RTH (09:30-16:00 ET) — MODE EXECUTION
│   ├── Clear expired signals
│   └── edge_cycle_v7() [pipeline complet]
│   Sleep: 180s (3 min)
│
└── Market Closed — MODE IDLE
    Sleep: 900s (15 min)
```

### 2.3 Pipeline V7 par ticker (`process_ticker_v7`)

```
Pour chaque ticker dans l'univers:
  1. compute_monster_score(ticker)        → score 0.0-1.0
  2. compute_features(ticker)             → features techniques
  3. get_ibkr().get_quote(ticker)         → prix temps reel
  4. producer.detect(DetectionInput)       → SignalType (Layer 1)
  5. pre_halt.assess(ticker)              → PreHaltState
  6. enrich_signal_with_context()         → MRP/EP (Market Memory)
  7. computer.compute_order()             → ProposedOrder (Layer 2)
  8. guard.assess(ticker)                 → RiskFlags
  9. gate.evaluate()                      → ExecutionDecision (Layer 3)
  10. handle_signal_result()              → Log + Telegram + Market Memory
```

### 2.4 Multi-Radar Engine V9 (`src/engines/multi_radar_engine.py`)

Architecture de detection multi-angle : 4 radars independants tournent **en parallele** (`asyncio.gather`) pour chaque ticker, chacun s'adapte a la session en cours.

```
               SessionAdapter (6 sous-sessions)
                       │
    ┌──────────────────┼─────────────────────────────┐
    │                  │               │              │
 FLOW RADAR     CATALYST RADAR   SMART MONEY    SENTIMENT
 (quantitatif)  (fondamental)    (options+ins)  (social+NLP)
    │                  │               │              │
    │  asyncio.gather — les 4 tournent simultanement  │
    └──────────────────┼─────────────────────────────┘
                       │
               CONFLUENCE MATRIX (matrice 2D + modifiers)
                       │
               ConfluenceSignal → signal_type + agreement + lead_radar
```

**Les 4 Radars :**

| Radar | Sources (modules existants) | Ce qu'il detecte | Performance |
|-------|---------------------------|-------------------|-------------|
| **FLOW** | AccelerationEngine, TickerStateBuffer, SmallCapRadar, FeatureEngine | Accumulation volume, derivees, breakout readiness | <10ms (buffer reads) |
| **CATALYST** | EventHub, CatalystScorerV3, AnticipationEngine, FDA Calendar | Catalysts, news, SEC filings | <50ms (cache reads) |
| **SMART MONEY** | Options Flow IBKR (OPRA), InsiderBoost (SEC Form 4), Extended Hours | Options inhabituelles, achats insiders | <100ms |
| **SENTIMENT** | Social Buzz (Reddit+StockTwits), NLP Enrichi (Grok), Repeat Gainer Memory | Buzz social, sentiment, repeat runners | <200ms |

**Adaptation par session :**

| Sous-session | Flow | Catalyst | Smart Money | Sentiment | Logique |
|-------------|------|----------|-------------|-----------|---------|
| AFTER_HOURS (16-20h) | 15% LOW | **45% ULTRA** | 10% LOW | 30% HIGH | News dominant, flow limite |
| PRE_MARKET (04-09:30) | 30% HIGH | 30% HIGH | 15% MED | 25% HIGH | Equilibre gap+catalyst |
| RTH_OPEN (09:30-10:30) | **35% ULTRA** | 20% HIGH | **30% ULTRA** | 15% MED | Max flow+options |
| RTH_MIDDAY (10:30-14:30) | **40% HIGH** | 20% MED | 25% HIGH | 15% LOW | Flow dominant |
| RTH_CLOSE (14:30-16h) | 30% HIGH | 30% HIGH | 25% HIGH | 15% MED | Anticipation lendemain |
| CLOSED/WEEKEND | 5% LOW | **50% HIGH** | 5% LOW | **40% HIGH** | Scanning batch |

**Confluence Matrix :**

```
                    Catalyst Level
                    HIGH (>=0.6)    MEDIUM (0.3-0.6)   LOW (<0.3)
Flow    HIGH        BUY_STRONG      BUY                WATCH
Level   MEDIUM      BUY             WATCH              EARLY_SIGNAL
        LOW         WATCH           EARLY_SIGNAL       NO_SIGNAL

Modifiers:
  Smart Money HIGH → upgrade +1 niveau
  Sentiment HIGH + 2+ radars actifs → upgrade +1 niveau
  4/4 radars actifs (UNANIMOUS) → +0.15 bonus, minimum BUY si score > 0.50
  3/4 radars actifs (STRONG) → +0.10 bonus
  2/4 radars actifs (MODERATE) → +0.05 bonus
```

**Singleton :** `get_multi_radar_engine()`
**Config :** `ENABLE_MULTI_RADAR = True` dans `config.py`

---

## 3. MODULES

### 3.1 Carte des modules

```
GV2-EDGE-V7.0/
├── main.py                              # Point d'entree, boucle principale
├── config.py                            # Configuration globale (464 lignes)
├── daily_audit.py                       # Audit quotidien (hit rate, lead time)
├── weekly_deep_audit.py                 # Audit hebdomadaire avance
├── performance_attribution.py           # Attribution des trades backtest
│
├── src/
│   ├── signal_engine.py                 # [LEGACY] Moteur signal (delegue a V7)
│   ├── anticipation_engine.py           # Detection precoce (IBKR radar + V6.1 ingestors)
│   ├── afterhours_scanner.py            # Scan post-marche (16h-20h ET)
│   ├── catalyst_score_v3.py             # Score catalyst ponderes (5 tiers, 18 types)
│   ├── dark_pool_alternatives.py        # Detection dark pool indirecte
│   ├── ensemble_engine.py               # Confluence douce (boost, jamais bloque)
│   ├── extended_hours_quotes.py         # Quotes pre/post marche + gap boost
│   ├── fda_calendar.py                  # Calendrier FDA (PDUFA, phases, conferences)
│   ├── feature_engine.py               # Calcul features techniques (Finnhub candles)
│   ├── historical_beat_rate.py          # Taux de beat earnings historique
│   ├── ibkr_connector.py               # Connexion IBKR Level 1 (reconnexion auto)
│   ├── ibkr_news_trigger.py            # Alertes rapides news IBKR (trigger, pas score)
│   ├── news_flow_screener.py           # Screener news global (SEC + Finnhub → tickers)
│   ├── nlp_enrichi.py                  # NLP avance (sentiment, entites, urgence)
│   ├── options_flow.py                 # Options flow monitor
│   ├── options_flow_ibkr.py            # Detection options inhabituelles (4 signaux)
│   ├── pattern_analyzer.py             # Patterns techniques (flag, squeeze, HLs)
│   ├── pm_scanner.py                   # Scanner pre-marche V8 (gap zones)
│   ├── pm_transition.py                # Analyse transition PM → RTH
│   ├── portfolio_engine.py             # Sizing positions (ATR-based)
│   ├── pre_halt_engine.py              # Evaluation risque halt (LOW/MEDIUM/HIGH)
│   ├── pre_spike_radar.py              # Signaux precurseurs de spike
│   ├── repeat_gainer_memory.py         # Memoire des repeat runners (SQLite)
│   ├── signal_logger.py                # Log persistant signaux (SQLite)
│   ├── social_buzz.py                  # Tracker buzz social (Reddit, StockTwits)
│   ├── universe_loader.py              # Chargement univers V3 (1 API call, ~3000 tickers)
│   ├── watch_list.py                   # Watchlist calendrier (J-7 → J-Day)
│   ├── finnhub_ws_screener.py          # [C1] WebSocket streaming Finnhub (remplace polling)
│   ├── top_gainers_source.py           # [C8] Source externe top gainers (IBKR + Yahoo)
│   └── ibkr_streaming.py              # [V9] Streaming temps reel IBKR (event-driven, ~10ms)
│
│   ├── engines/                         # Moteurs V7/V8/V9 (coeur du systeme)
│   │   ├── signal_producer.py           # LAYER 1: Detection illimitee V8
│   │   ├── order_computer.py            # LAYER 2: Calcul ordres systematique
│   │   ├── execution_gate.py            # LAYER 3: Gate execution (9 checks)
│   │   ├── acceleration_engine.py       # V8: Derivees + z-scores (anticipation)
│   │   ├── smallcap_radar.py            # V8: Radar small-cap (ACCUMULATING → BREAKOUT)
│   │   ├── ticker_state_buffer.py       # V8: Ring buffer (120 snapshots/ticker)
│   │   └── multi_radar_engine.py        # [V9] 4 radars paralleles + confluence matrix
│   │
│   ├── scoring/                         # Systeme de scoring
│   │   ├── monster_score.py             # Monster Score V4 (9 composants)
│   │   └── score_optimizer.py           # Auto-tuning poids
│   │
│   ├── models/                          # Types de donnees
│   │   ├── signal_types.py              # SignalType, PreSpikeState, ProposedOrder, etc.
│   │   └── __init__.py                  # Exports modeles
│   │
│   ├── event_engine/                    # Gestion evenements/catalysts
│   │   ├── event_hub.py                 # Hub central catalysts (cache 15 min)
│   │   └── nlp_event_parser.py          # Extraction NLP via Grok (18 types, 5 tiers)
│   │
│   ├── ingestors/                       # Ingestion donnees V6.1 (100% real APIs)
│   │   ├── sec_filings_ingestor.py      # SEC EDGAR 8-K + Form 4 (GRATUIT)
│   │   ├── global_news_ingestor.py      # News globale Finnhub + SEC
│   │   ├── company_news_scanner.py      # News par ticker (Finnhub company)
│   │   └── social_buzz_engine.py        # Reddit + StockTwits buzz
│   │
│   ├── processors/                      # Traitement donnees V6.1
│   │   ├── keyword_filter.py            # Pre-filtre regex rapide (1000 items/s)
│   │   ├── ticker_extractor.py          # Extraction + validation tickers
│   │   └── nlp_classifier.py            # Classification EVENT_TYPE via Grok
│   │
│   ├── risk_guard/                      # Gestion risques V8
│   │   ├── dilution_detector.py         # Detection dilution (4 tiers V8)
│   │   ├── compliance_checker.py        # Conformite exchange + delisting
│   │   ├── halt_monitor.py              # Monitoring halts (LULD, news pending)
│   │   └── unified_guard.py             # Orchestrateur risque (MIN-based V8)
│   │
│   ├── market_memory/                   # Apprentissage V7
│   │   ├── missed_tracker.py            # Track opportunites ratees
│   │   ├── pattern_learner.py           # Apprentissage patterns historiques
│   │   ├── context_scorer.py            # Scores MRP/EP contextuels
│   │   └── memory_store.py              # Persistence (JSON/SQLite)
│   │
│   ├── schedulers/                      # Orchestration scans
│   │   ├── hot_ticker_queue.py          # File prioritaire (HOT/WARM/NORMAL + TTL)
│   │   ├── scan_scheduler.py            # Scheduler dynamique (REALTIME/BATCH)
│   │   └── batch_scheduler.py           # Traitement batch hors-marche
│   │
│   ├── monitors/                        # Monitoring pipeline
│   │   └── pipeline_monitor.py          # Health checks API + alertes
│   │
│   ├── boosters/                        # Boosters additifs
│   │   ├── insider_boost.py             # Boost SEC Form 4 (+5-15%)
│   │   └── squeeze_boost.py             # Boost short interest (+5-20%)
│   │
│   ├── social_engine/                   # Moteur social
│   │   ├── grok_sentiment.py            # Sentiment via Grok API
│   │   └── news_buzz.py                 # Buzz news par ticker
│   │
│   ├── weekend_mode/                    # Mode week-end
│   │   ├── weekend_scanner.py           # Scan batch complet
│   │   ├── monday_prep.py              # Preparation watchlist lundi
│   │   ├── batch_processor.py           # Calcul lourd (backfill, training)
│   │   └── weekend_scheduler.py         # Orchestration week-end
│   │
│   └── api_pool/                        # Pool multi-cles API
│       ├── key_registry.py              # Registre cles + tiers + quotas
│       ├── pool_manager.py              # Rotation + health + cooldown
│       ├── request_router.py            # Routage par priorite (CRITICAL → BATCH)
│       └── usage_tracker.py             # Tracking usage + health score
│
├── alerts/
│   └── telegram_alerts.py              # Alertes Telegram V7 (emojis, badges, blocked)
│
├── backtests/
│   └── backtest_engine_edge.py          # Moteur backtest (V7 + legacy)
│
├── dashboards/
│   └── streamlit_dashboard.py           # Dashboard Streamlit + Plotly
│
├── monitoring/
│   └── system_guardian.py               # Guardian systeme (CPU, RAM, API health)
│
├── validation/                          # Suite de validation
│   ├── walk_forward.py                  # Walk-forward (30j fenetre, 15j step)
│   ├── regime_test.py                   # Detection regimes (bull/bear/vol)
│   ├── stress_test.py                   # Stress test resultats
│   ├── monte_carlo.py                   # Analyse Monte Carlo
│   ├── report_generator.py              # Rapport consolide
│   └── validation_engine.py             # Orchestrateur validation
│
├── tests/
│   ├── mock_data.py                     # Donnees mock (pas d'API)
│   ├── test_pipeline.py                 # Test pipeline complet
│   └── test_advanced_patterns.py        # Tests patterns + PM transition
│
└── utils/
    ├── api_guard.py                     # Safe API calls + retry + backoff
    ├── cache.py                         # TTL Cache in-memory
    ├── data_validator.py                # Validation donnees
    ├── logger.py                        # Logging rotatif
    ├── market_calendar.py               # Calendrier NYSE (2024-2027)
    └── time_utils.py                    # Sessions marche (ET timezone)
```

### 3.2 Dependances entre modules

```
Universe Loader ──→ Main Loop ──→ Process Ticker V7
                                    │
                    ┌───────────────┤
                    ▼               ▼
            Feature Engine    Monster Score V4
            (Finnhub candles) (9 composants ponderes)
                    │               │
                    └───────┬───────┘
                            ▼
                    Signal Producer V8 (Layer 1)
                    + AccelerationEngine
                    + SmallCapRadar
                            │
                            ▼
                    Pre-Halt Engine
                    Market Memory (MRP/EP)
                            │
                            ▼
                    Order Computer (Layer 2)
                            │
                            ▼
                    Risk Guard V8
                    (Dilution + Compliance + Halt)
                            │
                            ▼
                    Execution Gate (Layer 3)
                            │
                    ┌───────┴───────┐
                    ▼               ▼
            Signal Logger    Telegram Alert
            (SQLite)         (async)
```

---

## 4. MONSTER SCORE V4

### 4.1 Formule

```
base_score = (
    0.25 x event_score          +   # Catalysts (FDA, M&A, Earnings)
    0.17 x volume_spike         +   # Volume z-score normalise
    0.17 x pattern_score        +   # Patterns techniques
    0.13 x pm_transition_score  +   # Qualite transition PM→RTH
    0.10 x options_flow_score   +   # Activite options inhabituelle
    0.07 x acceleration_score   +   # [V8] Derivees + z-scores
    0.04 x momentum             +   # [Reduit V8] Velocite prix
    0.04 x squeeze              +   # Compression Bollinger
    0.03 x social_buzz_score        # [Reduit V8] Buzz social
)

final_score = clamp(base_score + beat_rate_boost + extended_hours_boost + acceleration_boost, 0, 1)
```

### 4.2 Seuils signaux

| Signal | Score | Condition supplementaire |
|--------|-------|--------------------------|
| BUY_STRONG | >= 0.80 | — |
| BUY | >= 0.65 | — |
| WATCH | >= 0.50 | — |
| EARLY_SIGNAL | >= 0.40 | + catalyst OU pre_spike != DORMANT |
| NO_SIGNAL | < 0.40 | — |

### 4.3 Boosts additifs

| Boost | Source | Max |
|-------|--------|-----|
| Beat rate | Historique earnings | +0.15 |
| Extended hours | Gap + volume AH/PM | +0.22 |
| Acceleration V8 | ACCUMULATING/LAUNCHING | +0.15 |
| Insider | SEC Form 4 | +15% base_score |
| Squeeze | Short interest | +20% base_score |

---

## 5. SIGNAL TYPES & ENUMS

### 5.1 Detection (Layer 1)

```python
SignalType:
  BUY_STRONG    # Score >= 0.80, haute confiance
  BUY           # Score >= 0.65, signal standard
  WATCH         # Score >= 0.50, a surveiller
  EARLY_SIGNAL  # Score >= 0.40, pre-spike (V8: ACCUMULATING)
  NO_SIGNAL     # Score < 0.40

PreSpikeState:
  DORMANT       # Pas d'activite
  CHARGING      # Accumulation momentum
  READY         # Pret a bouger
  LAUNCHING     # Mouvement demarre
  EXHAUSTED     # Mouvement termine

PreHaltState:
  LOW           # Pas de restriction
  MEDIUM        # Taille reduite / confirmation requise
  HIGH          # Execution bloquee/retardee
```

### 5.2 Execution (Layer 3)

```python
ExecutionStatus:
  EXECUTE_ALLOWED   # Execution complete
  EXECUTE_REDUCED   # Taille reduite (size_multiplier < 1.0)
  EXECUTE_BLOCKED   # Bloque (signal visible, execution non)
  EXECUTE_DELAYED   # En attente (marche ferme, etc.)
  ALERT_ONLY        # Notification uniquement

BlockReason:
  DAILY_TRADE_LIMIT, CAPITAL_INSUFFICIENT, POSITION_LIMIT
  DILUTION_HIGH, COMPLIANCE_HIGH, DELISTING_HIGH, PRE_HALT_HIGH
  MARKET_CLOSED, CIRCUIT_BREAKER, BROKER_DISCONNECTED
  PENNY_STOCK_RISK, MANUAL_BLOCK
```

### 5.3 Acceleration V8

```python
AccelerationState (strings, pas d'enum):
  "DORMANT"       # Pas d'activite
  "ACCUMULATING"  # Volume monte, prix stable → SIGNAL LE PLUS PRECOCE
  "LAUNCHING"     # Prix commence a bouger avec volume
  "BREAKOUT"      # Breakout confirme (prix + volume forts)
  "EXHAUSTED"     # Momentum decelerant

SmallCap Radar Phases:
  ACCUMULATING    # 5-15 min avant le mouvement
  PRE_LAUNCH      # 1-5 min avant
  LAUNCHING       # Au debut du mouvement
  BREAKOUT        # Confirme (apres +5%)
  DORMANT         # Rien de special
```

---

## 6. SOURCES DE DONNEES

### 6.1 Donnees de marche

**Abonnements IBKR actifs :**
- L1 US Equities — Network A (NYSE), Network B (ARCA/BATS), Network C (NASDAQ)
- OPRA — Options US (flux complet)

| Source | Type | Rate Limit | Cout | Statut |
|--------|------|-----------|------|--------|
| IBKR Streaming | Event-driven (pendingTickersEvent) | Illimite | Abonnement actif | **SOURCE PRIMAIRE V9** |
| IBKR Level 1 | TCP socket (ib-insync) | Illimite | Abonnement actif | Fallback (poll mode) |
| IBKR OPRA | Options chain via ib-insync | Illimite | Abonnement actif | **SOURCE PRIMAIRE** |
| Finnhub REST | Polling HTTP | 60 req/min (free) | Gratuit | Fallback |
| Finnhub WebSocket | Streaming (wss://ws.finnhub.io) | Illimite (1 conn) | Gratuit | Fallback streaming |

### 6.2 News & Catalysts

| Source | Contenu | Rate Limit | Cout |
|--------|---------|-----------|------|
| SEC EDGAR | 8-K filings, Form 4 | 10 req/s | Gratuit |
| Finnhub News | General + company news | 60 req/min | Gratuit |
| FDA Calendar | PDUFA, trials, conferences | Manuel | Gratuit |

### 6.3 Sentiment & NLP

| Source | Contenu | Rate Limit | Cout |
|--------|---------|-----------|------|
| Grok/xAI | Classification EVENT_TYPE | 10 req/min (free) | Gratuit |
| Reddit PRAW | Mentions WSB + subreddits | 60 req/min | Gratuit |
| StockTwits | Messages + sentiment labels | 200 req/min | Gratuit |

### 6.4 Pool API (`src/api_pool/`)

Rotation multi-cles avec priorites :
- **CRITICAL** (30% reserve) : Pre-halt, execution-blocking
- **HIGH** : Hot tickers, breaking news
- **STANDARD** : Scans normaux
- **LOW/BATCH** : Traitement hors-marche

---

## 7. RISK GUARD V8

### 7.1 Composants

| Module | Detection | Fichier |
|--------|-----------|---------|
| Dilution Detector | S-3, 424B, ATM, PIPE, toxic | `risk_guard/dilution_detector.py` |
| Compliance Checker | Deficiency, delisting, $1 rule | `risk_guard/compliance_checker.py` |
| Halt Monitor | LULD, vol spike, news pending | `risk_guard/halt_monitor.py` |
| Unified Guard | Orchestrateur central | `risk_guard/unified_guard.py` |

### 7.2 Corrections V8

| Probleme V7 | Fix V8 |
|-------------|--------|
| S-3 = CRITICAL = bloque 90j | 4 tiers : ACTIVE → SHELF_RECENT → SHELF_DORMANT → CAPACITY_ONLY |
| Multiplicatif 0.5 x 0.5 x 0.25 = 0.06 | MIN-based : min(0.5, 0.5, 0.25) = 0.25 |
| Pas de plancher → position 0.0 | Planchers : penny 0.10, standard 0.25 |
| Top gainers bloques par dilution | Momentum override : catalyst + vol + prix = -50% penalty |

---

## 8. SCHEDULING & SCAN INTERVALS

### 8.1 Intervalles

| Scan | Intervalle | Contexte |
|------|-----------|----------|
| Global news | 180s (3 min) | scan_scheduler.py |
| Hot tickers | 90s (1.5 min) | Tickers prioritaires |
| Warm tickers | 300s (5 min) | Tickers secondaires |
| Universe rotation | 600s (10 min) | 3 tickers aleatoires/cycle |
| Social buzz | 600s (10 min) | Reddit + StockTwits |
| SmallCap Radar | 5s | Buffer local (pas d'API) |

### 8.2 Hot Ticker Queue

| Priorite | TTL | Intervalle scan |
|----------|-----|-----------------|
| HOT | 3600s (1h) | 90s |
| WARM | 1800s (30 min) | 300s |
| NORMAL | 900s (15 min) | 600s |

Triggers pour devenir HOT :
- Pre-Spike Radar >= 2 signaux
- Catalyst detecte dans scan global
- Repeat Gainer score > 0.7
- Mouvement PM/AH > 5%
- Buzz social acceleration > 3x

---

## 9. CONFIGURATION CLES (`config.py`)

### 9.1 Feature Flags

| Flag | Default | Ligne | Description |
|------|---------|-------|-------------|
| `USE_V7_ARCHITECTURE` | True | 357 | Pipeline V7 unifie |
| `USE_IBKR_DATA` | **True** | 17 | Donnees IBKR (L1 + OPRA actifs) |
| `ENABLE_ACCELERATION_ENGINE` | True | 441 | V8 derivees + z-scores |
| `ENABLE_SMALLCAP_RADAR` | True | 457 | V8 radar anticipatif |
| `ENABLE_PRE_HALT_ENGINE` | True | 372 | Evaluation risque halt |
| `ENABLE_RISK_GUARD` | True | 414 | Guard risque unifie |
| `ENABLE_MARKET_MEMORY` | True | 429 | Apprentissage MRP/EP |
| `ENABLE_OPTIONS_FLOW` | True | 236 | Analyse options |
| `ENABLE_SOCIAL_BUZZ` | True | 244 | Tracking buzz social |
| `ENABLE_NLP_ENRICHI` | True | 328 | NLP avance (Grok) |
| `ENABLE_CATALYST_V3` | True | 282 | Score catalyst V3 |
| `ENABLE_PRE_SPIKE_RADAR` | True | 304 | Radar pre-spike |

### 9.2 Limites Execution

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| `DAILY_TRADE_LIMIT` | 5 | Trades max par jour |
| `MAX_POSITION_PCT` | 10% | Max capital par position |
| `MAX_TOTAL_EXPOSURE` | 80% | Exposition totale max |
| `MIN_ORDER_USD` | $100 | Ordre minimum |
| `RISK_BUY` | 2% | Risque par trade standard |
| `RISK_BUY_STRONG` | 2.5% | Risque par trade fort |

### 9.3 Stops

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| `ATR_MULTIPLIER_STOP` | 2.0 | Stop loss en ATR |
| `ATR_MULTIPLIER_TRAIL` | 1.5 | Trailing stop en ATR |

---

## 10. ETAT ACTUEL & PROBLEMES CONNUS

### 10.1 Problemes identifies (PLAN_CORRECTION_COVERAGE.md)

| ID | Probleme | Severite | Statut |
|----|----------|----------|--------|
| P1 | Finnhub 60 req/min → 6% couverture/cycle | CRITIQUE | CORRIGE (C1) |
| P2 | Feature Engine exige 20 bars minimum | HAUT | CORRIGE (C9) |
| P3 | NO_SIGNAL si score < 0.40 sans catalyst | HAUT | CORRIGE (C3) |
| P4 | SmallCap Radar filtre volume 500K redondant | MOYEN | CORRIGE (C2) |
| P5 | Batch limits durs dans anticipation_engine | MOYEN | CORRIGE (C7) |
| P6 | Aucun flux streaming (tout est polling) | CRITIQUE | CORRIGE (C1) |
| P7 | Acceleration Engine : 3+ samples minimum | MOYEN | CORRIGE (C4) |
| P8 | Alert cooldown fixe 2 min | MOYEN | CORRIGE (C5) |
| P9 | Hot ticker TTL 1h expire | MOYEN | CORRIGE (C6) |
| P10 | Aucune source externe top gainers | MOYEN | CORRIGE (C8) |

### 10.2 Corrections prevues (PLAN_CORRECTION_COVERAGE.md)

| ID | Correction | Fichier modifie | Statut |
|----|-----------|-----------------|--------|
| C1 | WebSocket Screener (Finnhub WS) | `src/finnhub_ws_screener.py` (NEW) | FAIT |
| C2 | Supprimer filtres SmallCap Radar | `src/engines/smallcap_radar.py` | FAIT |
| C3 | Score plancher adaptatif (vol z > 2.5) | `src/engines/signal_producer.py` | FAIT |
| C4 | Warm-up accelere (2 samples) | `src/engines/acceleration_engine.py` | FAIT |
| C5 | Alert cooldown adaptatif (15-120s) | `src/engines/acceleration_engine.py` | FAIT |
| C6 | Hot Ticker TTL 4h + auto-renewal | `src/schedulers/hot_ticker_queue.py` | FAIT |
| C7 | Relever batch limits anticipation | `src/anticipation_engine.py` | FAIT |
| C8 | Source externe top gainers (IBKR+Yahoo) | `src/top_gainers_source.py` (NEW) | FAIT |
| C9 | Feature Engine 5 bars minimum | `src/feature_engine.py` | FAIT |

### 10.3 Fichiers de reference

| Document | Contenu |
|----------|---------|
| `PLAN_TRANSFORMATION_V8.md` | Plan transformation V7 → V8 |
| `PLAN_CORRECTION_COVERAGE.md` | Corrections couverture 100% top gainers |
| `PLAN_AMELIORATION_V9.md` | Plan amelioration V9 complet (21 ameliorations, 5 sprints) |
| `CLAUDE.md` | Ce fichier — reference systeme |

---

## 11. CONVENTIONS DE DEVELOPPEMENT

### 11.1 Architecture

- **Detection JAMAIS bloquee** — seule l'execution a des limites
- **Additif, pas multiplicatif** — les boosts s'ajoutent, les penalites utilisent MIN
- **Fallback systematique** — IBKR → Finnhub → Cache → Default
- **Singleton pattern** — `get_ibkr()`, `get_ibkr_streaming()`, `get_signal_producer()`, `get_multi_radar_engine()`
- **Multi-Radar V9** — 4 radars paralleles (asyncio.gather), confluence matrix, session-adaptatif
- **Async + Thread-safe** — asyncio pour I/O, threading pour background

### 11.2 Sources de donnees

- **100% APIs reelles** — jamais de simulation LLM pour les donnees
- **Grok = classification UNIQUEMENT** — pas de sourcing de donnees
- **SEC EDGAR = gratuit et illimite** — privilegier pour les filings
- **Finnhub free tier** — gerer les 60 req/min via pool_manager

### 11.3 Scoring

- **Monster Score** : composant ponderes normalises 0-1 + boosts additifs
- **Catalyst Score** : 5 tiers x 18 types + recency decay + quality + confluence
- **Signal Producer** : score ajuste + boosts V8 (acceleration, pre-spike, repeat gainer)
- **Multi-Radar V9** : 4 scores independants → confluence matrix → signal final (plus robuste que score unique)

### 11.4 Risk

- **V8 MIN-based** — `min(risk1, risk2, risk3)` pas `risk1 * risk2 * risk3`
- **Momentum override** — catalyst + volume + prix = penalty reduite 50%
- **Planchers** — penny stock 0.10, standard 0.25 (jamais 0.0 sauf toxique)
- **Block UNIQUEMENT** sur : delisting, halt actif, toxic financing, block manuel

### 11.5 Tests & Validation

- `tests/test_pipeline.py` — test complet avec mock data
- `validation/validation_engine.py` — walk-forward, regime, stress, Monte Carlo
- `daily_audit.py` — hit rate, lead time, miss rate quotidiens
- `weekly_deep_audit.py` — analyse profonde hebdomadaire

### 11.6 Deploiement

- **Serveur** : Hetzner CX43 — 8 vCPU, 16 Go RAM, 160 Go SSD (headless Linux 4.4.0)
- **IBKR** : Actif (USE_IBKR_DATA=True) — L1 US Equities (Network A/B/C) + OPRA Options US
- **Fallback** : Finnhub REST si IBKR Gateway deconnecte
- **Alertes** : Telegram (token + chat_id via .env)
- **Logs** : `data/logs/` avec rotation
- **Persistance** : SQLite (`data/*.db`) + JSON + CSV

---

## 12. GLOSSAIRE

| Terme | Definition |
|-------|-----------|
| **Monster Score** | Score composite 0-1 combinant 9 composants ponderes |
| **Signal Producer** | Layer 1 — detection illimitee de signaux |
| **Order Computer** | Layer 2 — calcul systematique d'ordres |
| **Execution Gate** | Layer 3 — seule couche avec limites |
| **Acceleration Engine** | V8 — detection par derivees (velocite + acceleration) |
| **SmallCap Radar** | V8 — radar anticipatif (ACCUMULATING → BREAKOUT) |
| **Ticker State Buffer** | V8 — ring buffer 120 snapshots par ticker |
| **Hot Ticker Queue** | File prioritaire avec TTL et promotion/demotion |
| **Anticipation Engine** | Scan multi-source (IBKR + SEC + Finnhub + NLP) |
| **Market Memory** | Apprentissage : MRP (recovery potential), EP (edge probability) |
| **Risk Guard** | Evaluation risque : dilution + compliance + halt |
| **Unified Guard** | Orchestrateur risque V8 (MIN-based, momentum override) |
| **Pre-Halt Engine** | Detection risque halt (LULD, vol spike, news) |
| **Catalyst Score V3** | Score evenement : 5 tiers, 18 types, temporal decay |
| **Beat Rate** | Probabilite de beat earnings basee sur historique |
| **PM Transition** | Qualite de la transition pre-market → RTH |
| **Gap Zone** | Classification gap : NEGLIGIBLE / EXPLOITABLE / EXTENDED / OVEREXTENDED |
| **RTH** | Regular Trading Hours (09:30-16:00 ET) |
| **IBKR Streaming** | V9 — streaming temps reel event-driven (~10ms latence) |
| **Multi-Radar Engine** | V9 — 4 radars paralleles (Flow, Catalyst, Smart Money, Sentiment) + Confluence Matrix |
| **Flow Radar** | V9 — radar quantitatif (AccelerationEngine + Buffer + SmallCapRadar) |
| **Catalyst Radar** | V9 — radar fondamental (EventHub + CatalystV3 + AnticipationEngine + FDA) |
| **Smart Money Radar** | V9 — radar options + insiders (OPRA + SEC Form 4) |
| **Sentiment Radar** | V9 — radar social (Reddit + StockTwits + NLP Grok + Repeat Gainers) |
| **Confluence Matrix** | V9 — matrice 2D (Flow x Catalyst) + modifiers (Smart Money + Sentiment) |
| **Session Adapter** | V9 — adapte poids et sensibilites des radars par sous-session (6 modes) |
| **PDUFA** | Prescription Drug User Fee Act — date limite decision FDA |
