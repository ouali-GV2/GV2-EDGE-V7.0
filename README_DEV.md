# 📘 GV2-EDGE V5.1 — Developer Documentation

## 🎯 Objectif

Ce document explique :
- L'architecture technique V5 (Anticipation Engine)
- Le rôle de chaque module
- Les flux de données
- Comment étendre le système

---

## 🧱 Architecture V5.1

```
main.py
│
├── 🎯 ANTICIPATION ENGINE (NEW V5)
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

### options_flow_ibkr.py

**Rôle** : Détection options via OPRA L1

```python
# Signaux détectés
VOLUME_SPIKE      # Volume >> Open Interest
LOW_PC_RATIO      # Put/Call < 0.5
CALL_CONCENTRATION # 70%+ calls
HIGH_OPTIONS_VOLUME # >10k volume

# Entry point
scan_options_flow(tickers)
get_options_flow_score(ticker)
```

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

### config.py - Variables Clés

```python
# APIs
GROK_API_KEY = "xai-..."
FINNHUB_API_KEY = "..."
TELEGRAM_BOT_TOKEN = "..."

# IBKR
USE_IBKR_DATA = True
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # 7497=paper, 7496=live

# Seuils signaux
BUY_THRESHOLD = 0.65
BUY_STRONG_THRESHOLD = 0.80

# Universe
MAX_MARKET_CAP = 2_000_000_000  # $2B
MIN_PRICE = 1.0
MAX_PRICE = 50.0
```

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

**Version:** 5.1.0  
**Last Updated:** 2026-02-03
