# GV2-EDGE V9.0 - Radar Anticipatif Small-Caps Intraday

**Version 9.0 - Multi-Radar Anticipatory Detection Architecture**

---

## 1. Presentation du Projet

### Objectif Strategique

GV2-EDGE est un systeme de detection et d'execution automatise concu pour identifier les top gainers small-caps du marche americain **avant** qu'ils n'apparaissent sur les screeners publics. Il cible specifiquement les actions a faible capitalisation ($50M-$2B) affichant des signaux precurseurs de mouvements explosifs (+10% a +100% intraday).

Le systeme ne trade pas sur des indicateurs retardes. Il detecte les **phases d'accumulation** -- periodes ou le volume accelere alors que le prix reste stable -- qui precedent statistiquement les breakouts de 3 a 15 minutes.

### Positionnement

GV2-EDGE n'est pas un screener classique. Les differences fondamentales :

| Screener Classique | GV2-EDGE V9 |
|---|---|
| Detecte les actions qui **bougent deja** (+5%, +10%) | Detecte les actions qui **vont bouger** (accumulation) |
| Seuils fixes (volume > 5x, prix > 20%) | Z-scores adaptatifs vs baseline 20 jours par ticker |
| Signal binaire (passe/ne passe pas) | 4 phases progressives (ACCUMULATING → LAUNCHING) |
| Pas de gestion de risque integree | Pipeline complet : detection → sizing → execution → risk |
| Aucune memoire | Market Memory : apprend des signaux manques |
| Angle unique (prix ou volume) | V9 Multi-Radar : 4 radars paralleles (Flow, Catalyst, Smart Money, Sentiment) avec confluence matrix |

### Univers Cible

- **Capitalisation** : $50M - $2B (small caps)
- **Prix** : $0.50 - $20.00
- **Volume moyen** : > 500K actions/jour
- **Exclusions** : OTC, ADR non liquides
- **Taille de l'univers actif** : 50-200 hot tickers (rotation dynamique)

---

## 2. Architecture Globale

### Pipeline 3 Couches (Separation Detection/Execution)

Le principe fondamental de GV2-EDGE : **la detection ne s'arrete jamais, seule l'execution est limitee.** Un signal bloque par le Risk Guard reste visible pour le trader et est archive pour l'apprentissage.

```
┌────────────────────────────────────────────────────────────────┐
│ COUCHE 0 (V8) : RADAR ANTICIPATIF                              │
│                                                                │
│ TickerStateBuffer ──► AccelerationEngine ──► SmallCapRadar     │
│ (ring buffer 120pts)   (derivees, z-scores)   (4 phases)       │
│                                                                │
│ Produit : AccelerationScore + RadarBlip                        │
│ Latence : <500ms pour 200 tickers (lecture buffer, pas d'API)  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ COUCHE 0.5 (V9) : MULTI-RADAR ENGINE                          │
│                                                                │
│ 4 radars paralleles (asyncio.gather):                          │
│   Flow Radar ──► Catalyst Radar ──► Smart Money ──► Sentiment  │
│                                                                │
│ Confluence Matrix (2D + modifiers) → ConfluenceSignal          │
│ Session-adaptatif (6 sous-sessions)                            │
│ Latence : <200ms (cache reads)                                 │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ COUCHE 1 : SIGNAL PRODUCER (Detection - JAMAIS bloquee)        │
│ src/engines/signal_producer.py                                 │
│                                                                │
│ Entrees : MonsterScore V4, AccelerationScore, Catalyst, Buzz   │
│ Sorties : UnifiedSignal (BUY_STRONG / BUY / WATCH / EARLY)    │
│                                                                │
│ Seuils : BUY_STRONG ≥ 0.80 | BUY ≥ 0.65 | WATCH ≥ 0.50      │
│ V8 : ACCUMULATING + score ≥ 0.34 → EARLY_SIGNAL               │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ COUCHE 2 : ORDER COMPUTER (Toujours calcule)                   │
│ src/engines/order_computer.py                                  │
│                                                                │
│ Entrees : UnifiedSignal + MarketContext                        │
│ Sorties : ProposedOrder (size, stop, targets, R/R)             │
│                                                                │
│ Sizing : Risk-based (2% par trade), ATR-adjusted               │
│ Stops  : ATR x2.0 ou structure pre-market                      │
│ Targets: [1.5R, 2.5R, 4.0R] echelonnes                        │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ COUCHE 3 : EXECUTION GATE (Seule couche avec limites)          │
│ src/engines/execution_gate.py                                  │
│                                                                │
│ 9 verifications sequentielles :                                │
│ 1. Limite trades/jour    2. Capital disponible                 │
│ 3. Concentration position 4. Pre-Halt risk                     │
│ 5. Risk Guard flags       6. Heures de marche                  │
│ 7. Circuit breaker        8. Connexion broker                  │
│ 9. P&L journalier                                              │
│                                                                │
│ Sorties : ExecutionDecision (ALLOW / REDUCE / BLOCK + raison)  │
│ Garantie : le signal original est TOUJOURS preserve            │
└────────────────────────────────────────────────────────────────┘
```

### Flow Complet d'un Ticker

```
1. Donnees brutes (prix, volume, bid/ask) → TickerStateBuffer (push toutes les ~60s)
2. AccelerationEngine lit le buffer → calcule derivees, z-scores → detecte ACCUMULATING
3. SmallCapRadar scan → produit RadarBlip (priorite, phase, temps estime)
3.5. Multi-Radar V9 : 4 radars paralleles (Flow, Catalyst, Smart Money, Sentiment)
     → Confluence Matrix (2D + modifiers) → ConfluenceSignal (agreement + lead_radar)
4. MonsterScore V4 consolide : event(25%) + volume(17%) + pattern(17%)
   + pm_transition(13%) + acceleration(7%) + momentum(4%) + squeeze(4%)
   + options_flow(10%) + social_buzz(3%) + boosts conditionnels
5. SignalProducer ajuste le score (catalyst boost, pre-spike boost, V8 accel boost)
   → determine SignalType → cree UnifiedSignal
6. OrderComputer calcule le ProposedOrder (sizing, stop, targets)
7. Risk Guard evalue les risques (dilution tier, compliance, halt)
8. ExecutionGate decide : ALLOW, REDUCE, ou BLOCK (avec raisons)
9. Signal complete visible pour le trader, meme si bloque
```

---

## 3. Modules Principaux

### 3.1 SmallCap Radar (V8)

**Fichier** : `src/engines/smallcap_radar.py`

**Role** : Orchestrateur de la detection anticipative. Combine les signaux du TickerStateBuffer et de l'AccelerationEngine pour identifier les top gainers potentiels avant que le mouvement ne soit visible.

**Donnees utilisees** :
- Buffer de snapshots temps reel (prix, volume, spread, VWAP)
- Z-scores vs baseline 20 jours
- Contexte ticker (catalyst, repeat gainer, float, gap quality)

**4 phases de detection** :

| Phase | Temps avant move | Volume z-score | Prix | Priorite |
|---|---|---|---|---|
| ACCUMULATING | 5-15 min | > 1.5 | Stable (<2%) | MEDIUM |
| PRE_LAUNCH | 1-5 min | > 2.0 | Tick up | HIGH |
| LAUNCHING | 0-2 min | > 2.5 | Accelere | CRITICAL |
| BREAKOUT | En cours | > 2.5 | z > 2.0 | CRITICAL |

**Impact** : Boost MonsterScore de +0.05 (ACCUMULATING) a +0.18 (BREAKOUT).

### 3.2 Acceleration Engine (V8)

**Fichier** : `src/engines/acceleration_engine.py`

**Role** : Remplace la detection reactive V7 (valeurs instantanees) par un tracking de derivees (velocite, acceleration) et des z-scores adaptatifs.

**Metriques** :
- Prix : velocite (%/min), acceleration (%/min²)
- Volume : velocite, acceleration, ratio vs baseline
- Spread : velocite, detection de tightening (indicateur institutionnel)
- Composite : accumulation_score (0-1), breakout_readiness (0-1)

**Innovation cle** : La detection de l'etat ACCUMULATING -- volume qui accelere pendant que le prix reste stable -- est le signal le plus precoce avant un breakout. Ce pattern precede typiquement les top gainers de 3 a 15 minutes.

### 3.3 Ticker State Buffer (V8)

**Fichier** : `src/engines/ticker_state_buffer.py`

**Role** : Ring buffer circulaire qui stocke les snapshots temporels de chaque ticker. Fournit les donnees brutes pour le calcul des derivees.

**Specifications** :
- Capacite : 120 entrees par ticker (2 heures a 1 min d'intervalle)
- Memoire : ~50 octets/snapshot x 120 x 200 tickers = ~1.2 MB
- Performance : O(1) append, O(n) derivees (n ≤ 120)
- Thread-safe pour single-writer/multi-reader (deque Python)

### 3.4 Monster Score V4

**Fichier** : `src/scoring/monster_score.py`

**Role** : Score composite (0-1) determinant la force d'un signal. V4 ajoute le composant acceleration et reduit le poids des composants a faible fiabilite.

**Ponderation V4** :

| Composant | Poids | Source | Fiabilite |
|---|---|---|---|
| Event (Catalyst) | 25% | EventHub, CatalystScore V3 | Haute |
| Volume | 17% | Finnhub/IBKR, feature_engine | Haute |
| Pattern | 17% | pattern_analyzer (OHLCV) | Moyenne |
| PM Transition | 13% | pm_transition, pm_scanner V8 | Moyenne |
| **Acceleration** | **7%** | **AccelerationEngine V8** | **Haute (V8)** |
| Options Flow | 10% | options_flow_ibkr | Haute (si dispo) |
| Momentum | 4% | feature_engine | Moyenne |
| Squeeze | 4% | Bollinger bandwidth | Basse |
| Social Buzz | 3% | Reddit, StockTwits | Basse |

**Boosts conditionnels** (additifs) :
- Beat rate : jusqu'a +0.15 (earnings a forte probabilite)
- Extended hours : jusqu'a +0.22 (gap play detection)
- **Acceleration V8** : jusqu'a +0.18 (etat BREAKOUT)

### 3.5 Risk Guard Progressif (V8)

**Fichier** : `src/risk_guard/unified_guard.py`

**Role** : Evaluation unifiee des risques. V8 corrige le sur-blocage des small-caps et introduit le momentum override.

**Composants** :
- **Dilution Detector** (`dilution_detector.py`) : 4 tiers au lieu du blocage binaire
- **Compliance Checker** (`compliance_checker.py`) : Deficience bid, delisting
- **Halt Monitor** (`halt_monitor.py`) : Halts actuels et imminents (LULD)

**Corrections V8 critiques** :

| Probleme V7 | Correction V8 |
|---|---|
| Multiplicatif : 0.5 x 0.5 x 0.25 = 0.06 | Mode MIN : min(0.5, 0.5, 0.25) = 0.25 |
| S-3 shelf = blocage 90 jours | 4 tiers : ACTIVE(0.20x) / RECENT(0.60x) / DORMANT(0.85x) / CAPACITY(1.0x) |
| Penny stock +40% = bloque | Momentum override : catalyst + vol z>2 = -50% penalite |
| Minimum 0.0 (blocage total) | Planchers : 0.10 (penny) / 0.25 (standard) |
| Blocage sur tout CRITICAL | Blocage UNIQUEMENT : delisting, halt, toxic financing |

### 3.6 PM Scanner (V8)

**Fichier** : `src/pm_scanner.py`

**Role** : Analyse pre-market avec calcul de gap corrige. V8 utilise le `prev_close` (vrai gap overnight) au lieu du `pm_open` (intra-PM momentum).

**Classification des gaps V8** :

| Zone | Range | Qualite | Probabilite continuation |
|---|---|---|---|
| NEGLIGIBLE | < 3% | 0.1 | Bruit, ignorer |
| **EXPLOITABLE** | **3-8%** | **1.0** | **Meilleur risk/reward** |
| EXTENDED | 8-15% | 0.6 | Tradeable, risque de fade |
| OVEREXTENDED | > 15% | 0.25 | Fort risque de fade |

### 3.7 Options Flow

**Fichier** : `src/options_flow.py`, `src/options_flow_ibkr.py`

**Role** : Analyse des flux d'options pour detecter l'activite smart money. Poids de 10% dans MonsterScore.

**Etat actuel** : Actif avec abonnement IBKR OPRA. Detecte via le Smart Money Radar V9 :
- Volume inhabituel de calls vs baseline
- Concentration d'options sur strikes specifiques
- Ratio put/call anormal

### 3.8 Catalyst Engine

**Fichier** : `src/catalog_score_v3.py`

**Role** : Score et boost des catalyseurs (FDA, earnings, M&A, contrats). Decay temporel avec boost de proximite pour evenements futurs.

**Types de catalyseurs et boosts** :

| Type | Boost SignalProducer | Fiabilite |
|---|---|---|
| FDA_APPROVAL | +0.15 | Haute (PDUFA dates) |
| BUYOUT | +0.15 | Haute |
| MERGER_ACQUISITION | +0.12 | Haute |
| FDA_TRIAL_RESULT | +0.12 | Moyenne (dates fuzzy) |
| EARNINGS_BEAT | +0.10 | Haute |
| CONTRACT_WIN | +0.10 | Moyenne |
| PARTNERSHIP | +0.08 | Moyenne |
| ANALYST_UPGRADE | +0.05 | Basse |

### 3.9 Social & Buzz Engine

**Fichiers** : `src/social_buzz.py`, `src/ingestors/social_buzz_engine.py`

**Role** : Aggregation du sentiment et de l'acceleration des mentions sur Reddit (PRAW), StockTwits (API), et Twitter/X (estimation Grok).

**Limitation connue** : Grok ne peut pas acceder aux donnees Twitter/X en temps reel. Il estime les mentions a partir de ses donnees d'entrainement. Le poids a ete reduit a 3% dans V8 pour refleter cette fiabilite limitee.

### 3.10 Market Memory

**Fichier** : `src/market_memory/`

**Role** : Systeme d'apprentissage base sur les signaux manques (MRP) et les patterns historiques (EP). Informatif uniquement, ne bloque pas l'execution.

**Activation** : Necessite 50+ misses, 30+ trades, 10+ patterns, 20+ profils (1-2 semaines de warm-up).

### 3.11 Multi-Radar Engine (V9)

**Fichier** : `src/engines/multi_radar_engine.py`

**Role** : Architecture de detection multi-angle avec 4 radars independants operant en parallele, chacun adapte a la session en cours.

**4 Radars :**
| Radar | Sources | Detection |
|-------|---------|-----------|
| **FLOW** | AccelerationEngine, TickerStateBuffer, SmallCapRadar | Accumulation volume, derivees, breakout readiness |
| **CATALYST** | EventHub, CatalystScorerV3, AnticipationEngine, FDA | Catalysts, news, SEC filings |
| **SMART MONEY** | Options Flow IBKR (OPRA), InsiderBoost (SEC Form 4) | Options inhabituelles, achats insiders |
| **SENTIMENT** | Social Buzz (Reddit+StockTwits), NLP Enrichi (Grok) | Buzz social, sentiment, repeat runners |

**Confluence Matrix** : Matrice 2D (Flow x Catalyst) + modifiers (Smart Money, Sentiment) produisant un ConfluenceSignal avec signal_type, agreement level, et lead_radar.

**Adaptation par session** : 6 sous-sessions (AFTER_HOURS, PRE_MARKET, RTH_OPEN, RTH_MIDDAY, RTH_CLOSE, CLOSED) avec poids dynamiques pour chaque radar.

### 3.12 IBKR Streaming (V9)

**Fichier** : `src/ibkr_streaming.py`

**Role** : Streaming temps reel event-driven remplacant le pattern poll-and-cancel. Latence ~10ms vs 2000ms auparavant.

**Fonctionnalites** :
- Subscriptions persistantes (max 200 concurrentes)
- Event callbacks: on_quote(), on_event()
- Auto-detection: VOLUME_SPIKE, PRICE_SURGE, SPREAD_TIGHTENING, NEW_HIGH
- Feed automatique du TickerStateBuffer
- Integration HotTickerQueue (auto-promotion sur events)

---

## 4. Fonctionnement Intraday

### Cycle de Scan

```
03:00 UTC     Batch processing : SEC filings, global news, social baseline
04:00 ET      Pre-market : PM Scanner V8 actif, gap detection
04:00-09:30   Pre-market cycle : scan toutes les 60s
              - SmallCap Radar V8 : scan buffer toutes les 5s
              - MonsterScore V4 : recalcul sur hot tickers
              - Risk Guard V8 : evaluation continue
09:30 ET      Market open : transition PM → RTH
09:30-16:00   RTH cycle : scan toutes les 3 min
              - AccelerationEngine : derivees continues
              - SignalProducer V8 : detection ACCUMULATING → BREAKOUT
              - ExecutionGate : decisions d'execution
16:00-20:00   After-hours : anticipation scanning, news monitoring
20:30 UTC     Daily Audit : performance, auto-tuning poids
Vendredi      Weekly Deep Audit : analyse complete, ajustements
```

### Latence Cible

| Operation | Latence | Bottleneck |
|---|---|---|
| Radar scan (200 tickers) | < 500ms | Lecture buffer (CPU) |
| MonsterScore (1 ticker) | 50-100ms | SQLite queries |
| Anticipation Engine | 5-25s | Finnhub rate limit |
| News Flow Screener | 2-5s | Grok rate limit |
| FDA Calendar (cache) | < 1ms | Cache hit |
| **Full pipeline (1 ticker)** | **250-400ms** | **API calls** |
| **Full universe (200 tickers)** | **30-60s** | **Rate limits** |

### Promotion/Demotion des Tickers

Le `HotTickerQueue` gere la priorite de scan :

| Priorite | Intervalle scan | Critere |
|---|---|---|
| HOT | 90 secondes | Volume z > 2.0, catalyseur actif, ACCUMULATING |
| WARM | 5 minutes | Volume z > 1.0, mention sociale en hausse |
| NORMAL | 10 minutes | Univers standard |

**Promotion** : Un ticker passe HOT quand le SmallCap Radar detecte une phase ACCUMULATING ou superieure, quand un catalyseur est active, ou quand le volume z-score depasse 2.0.

**Demotion** : Retour a NORMAL apres TTL expire (defaut: 30 min pour HOT) ou si l'etat passe a EXHAUSTED.

### Detection d'Acceleration (V8)

La detection fonctionne en pipeline continu :

```
Toutes les ~60s :
  1. Push snapshot dans TickerStateBuffer (prix, volume, bid, ask)
  2. AccelerationEngine.score() :
     - Calcule velocite prix (%/min)
     - Calcule acceleration prix (%/min²)
     - Calcule velocite volume (shares/min change)
     - Calcule z-scores vs baseline 20j
     - Detecte accumulation (vol ↑ + prix stable)
     - Evalue breakout_readiness (0-1)
  3. Classifie etat : DORMANT → ACCUMULATING → LAUNCHING → BREAKOUT → EXHAUSTED

Toutes les ~5s :
  4. SmallCapRadar.scan() : lit les etats, produit les RadarBlips
  5. RadarBlips alimentent le scoring et le SignalProducer
```

---

## 5. Installation & Setup

### Prerequis

| Composant | Version | Obligatoire |
|---|---|---|
| Python | 3.8+ | Oui |
| IBKR Gateway/TWS | Latest | Non (fallback Finnhub) |
| Finnhub API key | Free tier | Oui |
| Grok API key | Free tier | Recommande |
| Reddit API credentials | PRAW | Recommande |
| Telegram Bot | BotFather | Recommande |

### Installation

```bash
# Clone
git clone <repo_url>
cd GV2-EDGE-V8.0

# Environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Dependances
pip install -r requirements.txt

# Configuration
cp .env.example .env
```

### Variables d'Environnement

```bash
# API Keys (obligatoires)
FINNHUB_API_KEY=votre_cle_finnhub

# API Keys (recommandees)
GROK_API_KEY=votre_cle_grok
REDDIT_CLIENT_ID=votre_id_reddit
REDDIT_CLIENT_SECRET=votre_secret_reddit
STOCKTWITS_ACCESS_TOKEN=votre_token_stocktwits

# IBKR (optionnel - si Gateway/TWS disponible)
USE_IBKR_DATA=False
IBKR_HOST=127.0.0.1
IBKR_PORT=7497      # 7497=paper, 7496=live, 4001/4002=Gateway
IBKR_CLIENT_ID=1

# Alertes (recommande)
TELEGRAM_BOT_TOKEN=votre_token_telegram
TELEGRAM_CHAT_ID=votre_chat_id
```

### Configuration (config.py)

Le fichier `config.py` centralise tous les parametres. Les plus importants :

```python
# Capital
MANUAL_CAPITAL = 1000           # Capital initial (mode manuel)
USE_IBKR_CAPITAL = False        # Auto-detection via IBKR

# Risk par trade
RISK_BUY = 0.02                 # 2% du capital par trade BUY
RISK_BUY_STRONG = 0.025         # 2.5% par trade BUY_STRONG

# Positions
MAX_OPEN_POSITIONS = 5          # Maximum simultane
MAX_POSITION_PCT = 0.10         # 10% max par position

# Architecture
USE_V7_ARCHITECTURE = True      # Pipeline 3 couches

# V8 modules
ENABLE_ACCELERATION_ENGINE = True
ENABLE_SMALLCAP_RADAR = True
RADAR_SENSITIVITY = "HIGH"      # ULTRA / HIGH / STANDARD
```

### Demarrage

```bash
source venv/bin/activate
python main.py
```

Le systeme detecte automatiquement la session de marche (pre-market, RTH, after-hours) et ajuste son cycle de scan.

### Deploiement Headless (Hetzner, VPS)

Pour un serveur sans interface graphique :

```python
# config.py
USE_IBKR_DATA = False           # Pas d'IBKR sans xvfb
# Les donnees viennent de Finnhub (fallback automatique)
```

IBKR Gateway necessite un serveur X (ou `xvfb`). Sur un VPS headless, le systeme bascule automatiquement sur Finnhub pour les quotes et les donnees historiques.

---

## 6. Configuration Strategique

### Parametres Capital

| Parametre | Defaut | Description |
|---|---|---|
| `MANUAL_CAPITAL` | 1000 | Capital en USD |
| `RISK_BUY` | 0.02 | Risque par trade BUY (2%) |
| `RISK_BUY_STRONG` | 0.025 | Risque par trade BUY_STRONG (2.5%) |
| `MAX_OPEN_POSITIONS` | 5 | Positions ouvertes max |
| `MAX_POSITION_PCT` | 0.10 | Taille max par position (10%) |
| `MAX_TOTAL_EXPOSURE` | 0.80 | Exposition totale max (80%) |
| `MIN_ORDER_USD` | 100 | Taille minimale d'ordre |

### Parametres Risk Guard V8

| Parametre | Defaut | Description |
|---|---|---|
| `RISK_APPLY_COMBINED_MULTIPLIERS` | **False** | V8: mode MIN (pas multiplicatif) |
| `RISK_ENABLE_MOMENTUM_OVERRIDE` | **True** | V8: momentum reduit les penalites |
| `RISK_MIN_POSITION_MULTIPLIER` | 0.10 | Plancher penny stocks |
| `RISK_BLOCK_ON_CRITICAL` | True | Bloquer sur risque critique |
| `RISK_BLOCK_ON_ACTIVE_OFFERING` | True | Bloquer sur offre active |
| `RISK_BLOCK_ON_DELISTING` | True | Bloquer sur delisting |

### Parametres Scoring V4

| Parametre | Defaut | Description |
|---|---|---|
| `BUY_THRESHOLD` | 0.65 | MonsterScore minimum pour BUY |
| `BUY_STRONG_THRESHOLD` | 0.80 | MonsterScore minimum pour BUY_STRONG |
| `ADVANCED_MONSTER_WEIGHTS` | Dict | Poids V4 (avec acceleration 7%) |
| `AUTO_TUNING_ENABLED` | True | Auto-ajustement hebdomadaire |

### Parametres Radar V8

| Parametre | Defaut | Description |
|---|---|---|
| `ENABLE_SMALLCAP_RADAR` | True | Activer le radar anticipatif |
| `RADAR_SENSITIVITY` | "HIGH" | Sensibilite (ULTRA/HIGH/STANDARD) |
| `RADAR_SCAN_INTERVAL` | 5 | Secondes entre scans radar |
| `TICKER_BUFFER_MAX_SNAPSHOTS` | 120 | Taille buffer (2h a 1min) |
| `ACCEL_VOLUME_ZSCORE_THRESHOLD` | 1.5 | Z-score volume minimum |
| `ACCEL_ACCUMULATION_MIN` | 0.30 | Score accumulation minimum |

### Sensibilite du Radar

| Mode | Accumulation min | Volume z min | Conf. min | Usage |
|---|---|---|---|---|
| ULTRA | 0.20 | 1.0 | 0.20 | Maximum de detections, plus de faux positifs |
| **HIGH** | **0.30** | **1.5** | **0.30** | **Equilibre recommande** |
| STANDARD | 0.45 | 2.0 | 0.40 | Conservateur, moins de faux positifs |

---

## 7. Performance & Scalabilite

### Architecture Async

Le systeme utilise `asyncio` pour les operations I/O-bound (appels API, requetes base de donnees). Les operations CPU-bound (calcul de derivees, scoring) sont executees de maniere synchrone car elles sont rapides (< 1ms par ticker).

```
Parallelisme reel :
- Finnhub API : semaphore de 5 requetes simultanees
- SEC EDGAR : 10 req/sec (non throttle)
- Reddit PRAW : sequentiel (60 req/min)
- Grok : 10 req/min (free tier) - bottleneck principal

Operations paralleles (asyncio.gather) :
- Catalyst Score + Pre-Spike Radar + News Flow + Social Buzz
- Objectif V8 : 30-60s sequentiel → 15-25s parallele
```

### API Pool Manager

**Fichier** : `src/api_pool/`

Le pool manager (`pool_manager.py`, `key_registry.py`, `request_router.py`, `usage_tracker.py`) est implemente mais **pas encore connecte aux ingestors** (chaque ingestor hardcode sa propre cle). La connexion est prevue en Sprint futur pour multiplier le throughput :

| Provider | Free Tier | Avec Pool (3 cles) |
|---|---|---|
| Finnhub | 60 req/min | 180 req/min |
| Grok | 10 req/min | 30 req/min |

### Cache TTL

| Module | TTL | Justification |
|---|---|---|
| MonsterScore | 30s | Compromis fraicheur/performance |
| PM Scanner | 30s | Donnees pre-market |
| Pre-Spike Radar | 300s (5 min) | Donnees historiques stables |
| FDA Calendar | 3600s (1h) | Dates changent rarement |
| Dilution Detector | 3600s (1h) | Filings SEC quotidiens |
| **Ticker State Buffer** | **N/A** | **Ring buffer, pas de TTL** |

### Limites de Scalabilite

| Metrique | Valeur Actuelle | Limite |
|---|---|---|
| Tickers scannes activement | 50-200 | Optimise |
| Tickers univers complet | 500-1000 | Faisable (rotation) |
| Tickers > 2000 | Non supporte | Rate limits API |
| Latence scan radar (200 tickers) | < 500ms | Buffer reads uniquement |
| Latence full pipeline (1 ticker) | 250-400ms | API calls |
| Rotation univers complet | 19-45 min | Acceptable pour swing |
| Memoire buffer (200 tickers) | ~1.2 MB | Negligeable |

---

## 8. Limitations & Risques

### Donnees Sociales Partielles

**Grok/Twitter** : Grok ne peut pas acceder aux donnees Twitter/X en temps reel. Il estime les comptages de mentions a partir de ses donnees d'entrainement. Le poids social_buzz a ete reduit a 3% dans V4 pour refleter cette limitation.

**Reddit** : Keyword matching simple ("buy", "moon", "rocket" = bullish). Pas de gestion de la negation ("don't buy"), pas de detection du sarcasme. Source PRAW temps reel, mais analyse basique.

**StockTwits** : Seule source avec labels de sentiment natifs. Fiabilite moderee.

### Dependance API

Le systeme depend de Finnhub comme source de donnees primaire. Si Finnhub est indisponible, le pipeline entier stagne (sauf si IBKR est configure).

**Mitigation** : Le pool manager est implemente (mais pas connecte). La prochaine iteration connectera les ingestors au pool pour rotation automatique et fallback.

### Risque Penny Stocks

Malgre les corrections V8 (momentum override, planchers minimaux, classification dilution 4 tiers), les penny stocks restent intrinsequement risques :

- **Spread eleve** : Slippage de 0.5% a 1.0% sur small caps
- **Manipulation** : Pump & dump non detectable par analyse technique seule
- **Halt** : LULD halts frequents sur mouvements > 10% en 5 minutes
- **Liquidite** : Volume suffisant pour entrer, pas toujours pour sortir

Le systeme est concu pour **detecter et dimensionner** ces risques, pas pour les eliminer.

### Systeme Reactif par Defaut

Malgre l'ajout de l'AccelerationEngine V8, la detection reste partiellement reactive :

- Les derivees necessitent des donnees historiques (3+ snapshots minimum)
- Le warm-up du buffer prend 3-5 minutes apres le demarrage
- Les catalyseurs sont detectes via news (latence 30 min - 4h apres filing SEC)
- Aucune donnee de flux d'ordres institutionnel (Level 3 / dark pool)

L'etat ACCUMULATING est le signal le plus precoce disponible avec les donnees accessibles, mais il ne constitue pas une prediction. C'est une detection d'anomalie statistique.

### Composants Absents

| Feature | Status | Impact |
|---|---|---|
| Halt detection temps reel | Absent | Risque d'achat de stocks haltes |
| IPO calendar | Absent | Aveugle aux nouvelles cotations |
| VWAP/volume profile | Absent | Analyse de volume incomplete |
| Sector rotation | Absent | Ne peut exploiter les themes sectoriels |
| ML ranking | Non implemente | Scoring base sur regles uniquement |

---

## 9. Roadmap

### Phase 1 (Complete) - Corrections Critiques V8

- [x] Fix gap calculation pm_scanner.py (prev_close vs pm_open)
- [x] Fix Risk Guard sur-blocage (mode MIN, momentum override, planchers)
- [x] Fix dilution confusion (4 tiers : ACTIVE/RECENT/DORMANT/CAPACITY)
- [x] Deprecier signal_engine.py obsolete

### Phase 2 (Complete) - Moteur Anticipatif

- [x] TickerStateBuffer (ring buffer, derivees, z-scores)
- [x] AccelerationEngine (detection ACCUMULATING, velocite, acceleration)
- [x] SmallCapRadar (4 phases, radar score, estimation temps)
- [x] MonsterScore V4 (composant acceleration 7%, poids ajustes)
- [x] SignalProducer V8 (integration acceleration, badges, boost)

### Phase 3 (Partiel) - Robustesse

- [x] Connecter API Pool Manager aux ingestors (throughput x3)
- [ ] Seuils z-score adaptatifs par classe de market cap (P9)
- [ ] Pattern analyzer connecte au TickerStateBuffer
- [x] Pipeline parallele (asyncio.gather via Multi-Radar V9)
- [ ] Cache TTL adaptatif par priorite ticker (CRITICAL=5s, HOT=10s)

### Phase 3.5 (V9 - Complete) - Multi-Radar & Streaming

- [x] Multi-Radar Engine (4 radars paralleles + confluence matrix)
- [x] IBKR Streaming Engine (event-driven, ~10ms latence)
- [x] Finnhub WebSocket Screener (streaming bulk)
- [x] Source externe top gainers (IBKR + Yahoo)
- [x] Session Adapter (6 sous-sessions)
- [x] Confluence Matrix (2D + modifiers)

### Phase 4 (A venir) - Intelligence

- [ ] Market Memory connecte au scoring (pas juste informatif)
- [ ] Seuils d'activation reduits (20 misses, 10 trades, 5 patterns)
- [ ] Detecteur de regime simple (SPY 20-day trend → bull/bear)
- [ ] Conferences FDA dynamiques (scraping mensuel)
- [ ] NLP contextuel (bigrams au lieu d'unigrams)

### Phase 5 (Futur) - ML & Scale

- [ ] ML ranking : gradient boosted model sur features V8
- [ ] Halt detection via Finnhub websocket
- [ ] Options flow relatif (call_volume / avg_20d au lieu de seuils absolus)
- [ ] VWAP profile via IBKR Level 1
- [ ] SQLite backend avec WHERE clauses (au lieu de SELECT * + filtre Python)
- [ ] Backend Redis pour le TickerStateBuffer (multi-process)

---

## Structure du Projet

```
GV2-EDGE-V8.0/
├── config.py                         # Configuration globale V8
├── main.py                           # Point d'entree, boucle principale
├── daily_audit.py                    # Audit quotidien
├── weekly_deep_audit.py              # Audit hebdomadaire, auto-tuning
├── performance_attribution.py        # Attribution P&L
│
├── src/
│   ├── engines/                      # Pipeline 3 couches + V8
│   │   ├── signal_producer.py        # Couche 1 : Detection V8
│   │   ├── order_computer.py         # Couche 2 : Calcul d'ordres
│   │   ├── execution_gate.py         # Couche 3 : Execution
│   │   ├── ticker_state_buffer.py    # V8 : Ring buffer
│   │   ├── acceleration_engine.py    # V8 : Derivees, z-scores
│   │   └── smallcap_radar.py         # V8 : Radar anticipatif
│   │
│   ├── scoring/
│   │   ├── monster_score.py          # Score composite V4
│   │   └── score_optimizer.py        # Auto-tuning poids
│   │
│   ├── risk_guard/
│   │   ├── unified_guard.py          # Orchestrateur risque V8
│   │   ├── dilution_detector.py      # Dilution 4 tiers V8
│   │   ├── compliance_checker.py     # Delisting, bid deficiency
│   │   └── halt_monitor.py           # LULD, halts
│   │
│   ├── models/
│   │   └── signal_types.py           # Types unifies V7
│   │
│   ├── market_memory/                # Apprentissage
│   │   ├── memory_store.py
│   │   ├── context_scorer.py         # MRP/EP
│   │   ├── pattern_learner.py
│   │   └── missed_tracker.py
│   │
│   ├── api_pool/                     # Pool API (implemente, pas connecte)
│   │   ├── pool_manager.py
│   │   ├── key_registry.py
│   │   ├── request_router.py
│   │   └── usage_tracker.py
│   │
│   ├── ingestors/                    # Ingestion donnees
│   │   ├── sec_filings_ingestor.py
│   │   ├── company_news_scanner.py
│   │   ├── global_news_ingestor.py
│   │   └── social_buzz_engine.py
│   │
│   ├── schedulers/                   # Planification
│   │   ├── batch_scheduler.py
│   │   ├── scan_scheduler.py
│   │   └── hot_ticker_queue.py
│   │
│   ├── event_engine/
│   │   ├── event_hub.py
│   │   └── nlp_event_parser.py
│   │
│   ├── social_engine/
│   │   ├── grok_sentiment.py
│   │   └── news_buzz.py
│   │
│   ├── anticipation_engine.py
│   ├── afterhours_scanner.py
│   ├── catalog_score_v3.py
│   ├── extended_hours_quotes.py
│   ├── fda_calendar.py
│   ├── feature_engine.py
│   ├── ibkr_connector.py
│   ├── news_flow_screener.py
│   ├── nlp_enrichi.py
│   ├── options_flow.py
│   ├── options_flow_ibkr.py
│   ├── pattern_analyzer.py
│   ├── pm_scanner.py                 # V8 : Gap corrige
│   ├── pm_transition.py
│   ├── pre_halt_engine.py
│   ├── pre_spike_radar.py
│   ├── repeat_gainer_memory.py
│   ├── signal_engine.py              # DEPRECATED (V8)
│   ├── signal_logger.py
│   ├── social_buzz.py
│   ├── universe_loader.py
│   └── watch_list.py
│
├── data/                             # Donnees persistantes (SQLite, JSON, logs)
└── utils/                            # Utilitaires (logger, cache, API guard)
```

---

## Conventions Techniques

- **Scoring** : Tous les scores sont normalises 0-1
- **Z-scores** : Utilises pour l'anomaly detection (V8), baseline 20 jours
- **Singletons** : Les modules principaux exposent un `get_*()` pour l'instance unique
- **Async** : Les operations I/O sont async, le scoring est synchrone
- **Cache** : TTL-based via `utils/cache.py`, pas de cache distribue
- **Logs** : Module `utils/logger.py`, rotation 10 MB, 5 backups
- **Persistence** : SQLite pour market_memory, JSON pour les poids et configs

---

**GV2-EDGE V8.0 - Anticipatory Small-Cap Radar**

*Detection anticipative. Execution controlee. Apprentissage continu.*

**Version** : 8.0.0
**Architecture** : Detection/Execution Separation + Anticipatory Radar
**Derniere mise a jour** : 2026-02-17
