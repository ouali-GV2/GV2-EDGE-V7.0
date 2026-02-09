# 📘 GV2-EDGE V6.0 — Developer Documentation

## 🎯 Objectif

Ce document explique :
- L'architecture technique V6.0 (Anticipation Multi-Couches)
- Le rôle de chaque module
- Les flux de données et le scoring
- Comment étendre le système

---

## 🆕 Changements V6.0

### Nouvelles Couches d'Anticipation

**1. Market Calendar US** (`utils/market_calendar.py`)
- Gestion complète des jours fériés NYSE (2024-2027)
- Demi-séances (early close days)
- Ajustement des volumes pour comparaison
- Fonctions: `is_trading_day()`, `is_early_close()`, `get_previous_trading_day()`

**2. Repeat Gainer Memory** (`src/repeat_gainer_memory.py`)
- Tracking historique des top gainers
- Score de "repeat runner" avec decay temporel
- Boost multiplicateur pour Monster Score
- Database SQLite pour persistance

**3. Pre-Spike Radar** (`src/pre_spike_radar.py`)
- Détection d'accélération AVANT le spike (pas le niveau, la dérivée)
- 4 signaux: Volume, Options, Buzz, Technical compression
- Confluence scoring: plus de signaux = plus haute probabilité
- Alert levels: NONE → WATCH → ELEVATED → HIGH
- Boost anticipatif pour Monster Score (jusqu'à 1.4x)

**4. Catalyst Score V3** (`src/catalyst_score_v3.py`)
- Pondération par type de catalyst (FDA > Earnings > Contract)
- Temporal decay: événements frais > événements anciens (half-life 24h)
- Quality assessment: fiabilité source + confirmation multi-sources
- Confluence multi-catalyst: plusieurs catalysts = score plus élevé
- Historical performance tracking: apprentissage des performances passées
- Alert levels: NONE → LOW → MEDIUM → HIGH → CRITICAL
- Boost multiplicateur pour Monster Score (jusqu'à 1.6x)

**5. NLP Enrichi** (`src/nlp_enrichi.py`)
- Analyse sentiment avancée: bullish/bearish avec intensité et confiance
- Extraction d'entités: tickers, personnes, produits, chiffres clés
- Classification news: 13 catégories + 5 niveaux d'urgence
- Agrégation multi-sources: time-weighted sentiment across sources
- Détection spike sentiment: alerte si changement > 30%
- Database SQLite pour historique sentiment
- Boost multiplicateur: 0.7x (bearish) à 1.4x (bullish)

### Catégories de News (NLP Enrichi)

```python
# Impact décroissant
FDA_REGULATORY     # 1.00 - FDA approvals, trials
MERGER_ACQUISITION # 0.95 - M&A, buyouts
EARNINGS           # 0.85 - Quarterly results
CONTRACT_DEAL      # 0.75 - Contracts, partnerships
GUIDANCE           # 0.72 - Forward guidance
ANALYST_RATING     # 0.65 - Upgrades, downgrades
PRODUCT_LAUNCH     # 0.60 - New products
INSIDER_ACTIVITY   # 0.55 - Insider buying/selling
MANAGEMENT         # 0.45 - CEO changes
LEGAL              # 0.40 - Lawsuits
SECTOR_NEWS        # 0.30 - Industry news
MACRO              # 0.25 - Economic news
```

### Niveaux d'Urgence

```python
BREAKING  # Immediate action, just happened (decay: 4h)
HIGH      # Same-day relevance (decay: 12h)
MEDIUM    # Near-term relevance (decay: 24h)
LOW       # Background info (decay: 48h)
STALE     # Old news (decay: 168h)
```

### Unified EVENT_TYPE Taxonomy (V6)

```python
# TIER 1 - CRITICAL IMPACT (0.90-1.00)
FDA_APPROVAL           # Drug/device approved by FDA
PDUFA_DECISION         # FDA PDUFA deadline decision
BUYOUT_CONFIRMED       # Confirmed acquisition/buyout

# TIER 2 - HIGH IMPACT (0.75-0.89)
FDA_TRIAL_POSITIVE     # Positive Phase II/III results
BREAKTHROUGH_DESIGNATION # FDA breakthrough therapy
FDA_FAST_TRACK         # FDA fast track granted
MERGER_ACQUISITION     # M&A announcement
EARNINGS_BEAT_BIG      # Earnings beat >20%
MAJOR_CONTRACT         # Contract award >$50M

# TIER 3 - MODERATE IMPACT (0.60-0.74)
GUIDANCE_RAISE         # Forward guidance increased
EARNINGS_BEAT          # Standard earnings beat
PARTNERSHIP            # Strategic partnership
PRICE_TARGET_RAISE     # Analyst PT increase

# TIER 4 - LOW-MODERATE IMPACT (0.45-0.59)
ANALYST_UPGRADE        # Analyst rating upgrade
SHORT_SQUEEZE_SIGNAL   # Short interest spike + covering
UNUSUAL_VOLUME_NEWS    # Abnormal volume with news

# TIER 5 - SPECULATIVE (0.30-0.44)
BUYOUT_RUMOR           # Unconfirmed acquisition rumor
SOCIAL_MEDIA_SURGE     # Viral social media mentions
BREAKING_POSITIVE      # Generic positive breaking news
```

### Monster Score V3 - Nouveau Système de Poids

```python
ADVANCED_MONSTER_WEIGHTS = {
    "event": 0.25,          # Catalysts (earnings, FDA, M&A)
    "volume": 0.17,         # Volume spikes
    "pattern": 0.17,        # Technical patterns
    "pm_transition": 0.13,  # PM→RTH transition
    "momentum": 0.08,       # Price momentum
    "squeeze": 0.04,        # Bollinger squeeze
    "options_flow": 0.10,   # NEW: Options activity (volume + concentration)
    "social_buzz": 0.06,    # NEW: Social media buzz
}
# Total = 100%
# + Repeat Gainer Boost (up to 1.5x multiplier)
```

### Options Flow - Changements

- Volume/OI ratio **DÉSACTIVÉ** (OI delayed J-1, peu fiable)
- Nouveaux signaux basés sur volume absolu:
  - `HIGH_CALL_VOLUME` : >= 5000 contracts
  - `LOW_PC_RATIO` : Put/Call < 0.5
  - `CALL_CONCENTRATION` : 70%+ calls
  - `HIGH_OPTIONS_VOLUME` : >= 10,000 total

---

## 🧱 Architecture V6.0

```
main.py
│
├── 📅 MARKET CALENDAR (NEW V6)
│   └── utils/market_calendar.py      # NYSE holidays, early closes
│
├── 🔁 REPEAT GAINER MEMORY (NEW V6)
│   └── src/repeat_gainer_memory.py   # Historical spike tracking
│
├── ⚡ PRE-SPIKE RADAR (NEW V6)
│   └── src/pre_spike_radar.py        # Acceleration detection before spike
│       ├── Volume acceleration       # Derivative of volume (not level)
│       ├── Options acceleration      # Call momentum increasing
│       ├── Buzz acceleration         # Social mentions picking up
│       └── Technical compression     # Squeeze before breakout
│
├── 🎯 CATALYST SCORE V3 (NEW V6)
│   └── src/catalyst_score_v3.py      # Enhanced event-based scoring
│       ├── Type weighting            # FDA > Earnings > Contract > etc.
│       ├── Temporal decay            # Fresh events > old events
│       ├── Quality assessment        # Source reliability + confirmation
│       ├── Confluence scoring        # Multiple catalysts = higher score
│       └── Performance tracking      # Learn from historical data
│
├── 🧠 NLP ENRICHI (NEW V6)
│   └── src/nlp_enrichi.py            # Advanced sentiment & news processing
│       ├── Enhanced sentiment        # Bullish/bearish with intensity
│       ├── Entity extraction         # Tickers, people, products, numbers
│       ├── News classification       # 13 categories + 5 urgency levels
│       ├── Multi-source aggregation  # Time-weighted sentiment
│       └── Sentiment spike detection # Alert on 30%+ change
│
├── 🎯 ANTICIPATION ENGINE (V5)
│   ├── src/anticipation_engine.py      # Orchestrateur principal
│   ├── src/news_flow_screener.py       # NEWS → NLP → Tickers
│   ├── src/options_flow_ibkr.py        # Options via OPRA L1
│   ├── src/extended_hours_quotes.py    # After-hours/Pre-market
│   └── src/dark_pool_alternatives.py   # Évaluation (désactivé)
│
├── 📊 DATA LAYER
│   ├── src/universe_loader.py          # Univers small caps
│   ├── src/ibkr_connector.py           # IBKR API (READ ONLY)
│   └── utils/cache.py                  # Cache système
│
├── 📅 EVENT LAYER
│   ├── src/event_engine/event_hub.py   # Agrégation events
│   ├── src/event_engine/nlp_event_parser.py  # NLP Grok
│   ├── src/fda_calendar.py             # FDA/Biotech events
│   └── src/historical_beat_rate.py     # Earnings prediction
│
├── 📈 ANALYSIS LAYER
│   ├── src/feature_engine.py           # Features techniques
│   ├── src/pattern_analyzer.py         # Patterns detection
│   ├── src/pm_scanner.py               # Pre-market scanner
│   ├── src/pm_transition.py            # PM→RTH transition
│   └── src/social_buzz.py              # Social sentiment
│
├── 🎯 SCORING LAYER
│   ├── src/scoring/monster_score.py    # Score principal
│   ├── src/ensemble_engine.py          # Confluence
│   └── src/signal_engine.py            # BUY/BUY_STRONG/WATCH
│
├── 💰 PORTFOLIO LAYER
│   ├── src/portfolio_engine.py         # Risk management
│   └── src/watch_list.py               # Watch list gestion
│
├── 📤 OUTPUT LAYER
│   ├── alerts/telegram_alerts.py       # Telegram notifications
│   ├── src/signal_logger.py            # SQLite persistence
│   └── dashboards/streamlit_dashboard.py
│
└── 🔍 AUDIT LAYER
    ├── daily_audit.py                  # Audit quotidien
    ├── weekly_deep_audit.py            # Audit hebdomadaire
    └── performance_attribution.py      # Attribution performance
```

---

## 🔄 Flow Principal V5.1

### After-Hours (16:00-20:00 ET)

```
1. News Flow Screener
   └── Fetch ALL news (Polygon + Finnhub)
   └── NLP filter (keywords bullish)
   └── Grok analysis (extract tickers + impact)
   └── Output: {ticker: events}

2. Extended Hours Gaps
   └── IBKR quotes extended hours
   └── Detect gaps > 3%
   └── Output: [ExtendedQuote]

3. Options Flow
   └── IBKR OPRA L1 data
   └── Volume vs OI analysis
   └── P/C ratio analysis
   └── Output: {ticker: signals}

4. Anticipation Engine
   └── IBKR Radar (anomalies)
   └── Grok+Polygon (catalysts)
   └── Generate WATCH_EARLY / BUY signals
```

### Pre-Market (04:00-09:30 ET)

```
1. Signal Upgrades
   └── Check WATCH_EARLY signals
   └── PM confirmation (gap, volume, momentum)
   └── Upgrade to BUY if confirmed

2. Regular Edge Cycle
   └── Feature extraction
   └── Monster Score
   └── Signal generation
```

---

## 📦 Modules Clés

### anticipation_engine.py

**Rôle** : Orchestrateur principal de l'anticipation

```python
# Classes principales
class AnticipationState      # État global (suspects, signals)
class Anomaly               # Anomalie détectée par IBKR
class CatalystEvent         # Catalyst détecté par Grok
class AnticipationSignal    # Signal final

# Fonctions principales
run_ibkr_radar(tickers)           # Scan large IBKR
analyze_with_grok_polygon(tickers) # Analyse Grok ciblée
generate_signals(anomalies, catalysts)  # Génération signaux
run_anticipation_scan(universe, mode)   # Entry point
```

### news_flow_screener.py

**Rôle** : Scanner news global → mapping tickers

```python
# Flow inversé (efficace)
fetch_polygon_news_global()    # Toutes les news
filter_high_impact_news()      # Filtre keywords
analyze_news_with_grok()       # NLP extraction tickers
aggregate_events_by_ticker()   # Groupement par ticker

# Entry point
run_news_flow_screener(universe, hours_back=6)
```

### options_flow_ibkr.py (V5.3 - Updated)

**Rôle** : Détection options via IBKR OPRA L1 (volume-based)

```python
# Signaux détectés (V5.3 - Volume based, NO OI ratio)
HIGH_CALL_VOLUME    # Call volume >= 5000 contracts
LOW_PC_RATIO        # Put/Call < 0.5 (bullish)
CALL_CONCENTRATION  # 70%+ calls
HIGH_OPTIONS_VOLUME # Total volume >= 10k

# NOTE: Volume/OI ratio DISABLED (OI is delayed J-1)

# Entry points
scan_options_flow(tickers)      # Batch scan
get_options_flow_score(ticker)  # Single ticker score
```

**Impact V5.3** : 10% du Monster Score (composante core)

### extended_hours_quotes.py

**Rôle** : Quotes after-hours et pre-market

```python
# Data structure
@dataclass
class ExtendedQuote:
    ticker, session, last, bid, ask
    volume, extended_volume
    prev_close, rth_close, rth_open
    gap_pct, change_pct

# Entry points
get_extended_quote(ticker)
scan_afterhours_gaps(tickers, min_gap=0.03)
scan_premarket_gaps(tickers, min_gap=0.03)
get_extended_hours_boost(ticker)  # Pour Monster Score
```

---

## 🔧 Configuration

### Environment Variables (.env)

**IMPORTANT:** API keys are now loaded from environment variables for security.

```bash
# Create .env file from template
cp .env.example .env

# Required variables
GROK_API_KEY=xai-...           # x.ai API (NLP + Twitter/X)
FINNHUB_API_KEY=...            # Market data fallback
TELEGRAM_BOT_TOKEN=...         # Alerts
TELEGRAM_CHAT_ID=...           # Alerts

# IBKR (recommended)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                 # 7497=paper, 7496=live

# Social Buzz APIs (optional but recommended)
REDDIT_CLIENT_ID=...           # Reddit PRAW
REDDIT_CLIENT_SECRET=...
STOCKTWITS_ACCESS_TOKEN=...    # StockTwits
```

### config.py - Key Settings

```python
# Signal thresholds
BUY_THRESHOLD = 0.65
BUY_STRONG_THRESHOLD = 0.80

# Universe filters
MAX_MARKET_CAP = 2_000_000_000  # $2B
MIN_PRICE = 1.0
MAX_PRICE = 20.0

# Social Buzz (V5.3)
ENABLE_SOCIAL_BUZZ = True
ENABLE_GOOGLE_TRENDS = False   # Disabled (pytrends unreliable)
SOCIAL_BUZZ_SOURCES = ["twitter", "reddit", "stocktwits"]
```

### Social Buzz Sources (V5.3)

| Source | Weight | API | Notes |
|--------|--------|-----|-------|
| Twitter/X | 45% | `GROK_API_KEY` | Real-time, institutional leaks |
| Reddit | 30% | `REDDIT_*` | PRAW authenticated (WSB, stocks, pennystocks) |
| StockTwits | 25% | `STOCKTWITS_*` | Dedicated traders, sentiment labels |
| Google Trends | 0% | N/A | **Disabled** (pytrends rate limited) |

---

## 🧪 Tests

```bash
# Test anticipation engine
python src/anticipation_engine.py

# Test news flow screener
python src/news_flow_screener.py

# Test options flow
python src/options_flow_ibkr.py

# Test extended hours
python src/extended_hours_quotes.py

# Test pipeline complet
python tests/test_pipeline.py
```

---

## 📊 Logs

```
data/logs/
├── anticipation_engine.log
├── news_flow_screener.log
├── options_flow.log
├── extended_hours.log
├── monster_score.log
├── signal_engine.log
└── ...
```

---

## 🚀 Ajouter un Nouveau Module

1. Créer `src/nouveau_module.py`
2. Ajouter import dans `main.py`
3. Intégrer dans la boucle appropriée (AH/PM/RTH)
4. Ajouter tests dans `tests/`
5. Documenter dans ce README

---

## ⚠️ Règles Critiques

1. **IBKR READ ONLY** : Jamais d'ordres automatiques
2. **Grok Rate Limits** : Max ~300 calls/heure
3. **Cache** : Utiliser `utils/cache.py` pour éviter calls redondants
4. **Logs** : Toujours logger avec `utils/logger.py`

---

---

## 📊 Flux de Données Scoring V5.3

```
Universe Loader (300-500 tickers)
        ↓
Feature Engine + Event Hub + PM Scanner
        ↓
Pattern Analyzer + Options Flow + Social Buzz
        ↓
Monster Score V3 (8 composantes pondérées)
├── event (25%)
├── volume (17%)
├── pattern (17%)
├── pm_transition (13%)
├── options_flow (10%)  ← NEW CORE
├── momentum (8%)
├── social_buzz (6%)    ← NEW CORE
└── squeeze (4%)
        ↓
Signal Engine (BUY/BUY_STRONG/WATCH_EARLY)
        ↓
Portfolio Engine (risk management)
        ↓
Output (Telegram + SQLite + Dashboard)
```

---

**Version:** 6.0.0
**Last Updated:** 2026-02-05
