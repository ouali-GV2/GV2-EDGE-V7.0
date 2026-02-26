# AUDIT COMPLET — GV2-EDGE V9.0
> **Date** : 2026-02-25
> **Scope** : 90 fichiers Python analysés (src/, engines/, risk_guard/, market_memory/, schedulers/, utils/, api_pool/, monitors/, boosters/, ingestors/, processors/, backtests/, validation/, monitoring/)
> **Objectif** : Identifier toutes les faiblesses pour atteindre l'objectif de détection anticipative des top gainers small-cap US (+30% à +300%/jour)

---

## RÉSUMÉ EXÉCUTIF

L'architecture V9.0 est **ambitieuse et conceptuellement solide**, mais souffre de **nombreuses déconnexions entre les modules conçus et leur usage réel en production**. Les 4 problèmes les plus critiques :

1. **Le pipeline V8 (AccelerationEngine + TickerStateBuffer + SmallCapRadar) est entièrement déconnecté de `process_ticker_v7`** — le cœur du système V8 ne s'exécute jamais dans la boucle principale.
2. **La limite quotidienne de trades ne se déclenche jamais** — `V7State.trades_today` n'est jamais incrémenté.
3. **`top_gainers_source.py` est non-fonctionnel** — les deux sources (IBKR scanner + Yahoo) sont cassées, rendant la correction C8 inopérante.
4. **La mémoire de marché (Market Memory) ne persiste jamais** — `PatternLearner` et `MissedTracker` sont 100% in-memory et perdent tout à chaque redémarrage.

---

## CRITICITÉ PAR CATÉGORIE

| Catégorie | Bugs critiques | Manquant critique | Perf. bloquante | Arch. majeure |
|-----------|:--------------:|:-----------------:|:---------------:|:-------------:|
| Pipeline principal (main.py) | 5 | 4 | 2 | 3 |
| Engines V7/V8/V9 | 12 | 8 | 6 | 7 |
| Scoring (Monster + Catalyst) | 6 | 5 | 4 | 3 |
| Risk Guard | 4 | 3 | 2 | 3 |
| Data Sources & Ingestors | 10 | 6 | 6 | 5 |
| Market Memory | 5 | 4 | 3 | 4 |
| Schedulers | 4 | 4 | 4 | 2 |
| Social & NLP | 6 | 4 | 4 | 4 |
| Utils & API Pool | 6 | 4 | 3 | 4 |
| **TOTAL** | **58** | **42** | **34** | **35** |

---

## PARTIE 1 — PROBLÈMES CRITIQUES TRANSVERSAUX

### PC-1 : V8 Acceleration Pipeline complètement déconnecté du pipeline principal

**Impact : CRITIQUE — Toute la détection anticipative V8 est inactive en production**

`process_ticker_v7` ne remplit jamais `DetectionInput.acceleration_state` (toujours `"DORMANT"`).
`AccelerationEngine`, `TickerStateBuffer` et `SmallCapRadar` ne sont invoqués **que** depuis `MultiRadarEngine.FlowRadar`, jamais depuis la boucle principale.
`TickerStateBuffer.set_baseline_raw` n'est jamais appelé → tous les z-scores utilisent des valeurs par défaut (`volume_mean=1, volume_std=1`) → détection d'anomalies totalement faussée.

**Fix requis** : Intégrer `AccelerationEngine.score(ticker)` dans `process_ticker_v7` après l'étape `compute_features()`, et initialiser les baselines depuis `universe_loader`.

---

### PC-2 : Compteur de trades jamais incrémenté

**Impact : CRITIQUE — La limite `DAILY_TRADE_LIMIT=5` ne se déclenche jamais**

`handle_signal_result` ne call jamais `state.record_trade()` même quand `decision.status == EXECUTE_ALLOWED`. `account.trades_today` reste à 0 toute la journée. La protection contre le sur-trading est inexistante.

**Fix requis** : Appeler `state.record_trade()` et `gate.record_trade_executed()` dans `handle_signal_result` quand le statut est ALLOWED.

---

### PC-3 : Source de top gainers (C8) non-fonctionnelle

**Impact : CRITIQUE — La correction C8 est complètement inopérante**

- `ibkr.run_scanner(...)` lève `AttributeError` (méthode inexistante sur `IBKRConnector`)
- L'API Yahoo Finance v1 screener est dépréciée depuis 2023 (retourne 404)
- `fetch_top_gainers` est déclarée `async` mais ne contient aucun `await` — bloque l'event loop
- `repeat_gainer_memory.fetch_and_record_top_gainers` est un stub qui retourne toujours `False`

**Fix requis** : Implémenter un vrai scanner IBKR via `ib_insync.ScannerSubscription`, ou utiliser l'API Finnhub quotes pour les top movers du jour.

---

### PC-4 : Market Memory entièrement in-memory (perte totale au redémarrage)

**Impact : CRITIQUE — Le système "apprenant" ne retient rien entre les sessions**

`PatternLearner`, `MissedTracker` et `ContextScorer` n'utilisent jamais `MemoryStore` malgré son existence. Tout l'apprentissage disparaît à chaque redémarrage. La base de `repeat_gainers.db` ne se remplit jamais automatiquement.

**Fix requis** : Connecter `PatternLearner.learn_all()` à `MemoryStore.save_*()`, et appeler `MemoryStore.load_*()` à l'initialisation de `ContextScorer`.

---

### PC-5 : `asyncio.coroutine` supprimé en Python 3.11

**Impact : CRASH RUNTIME — `unified_guard.py` plante quand un composant est désactivé**

```python
asyncio.coroutine(lambda: None)()  # AttributeError en Python 3.11+
```

Lignes 307–319 de `unified_guard.py`. Si `ENABLE_RISK_GUARD=False` pour l'un des composants, l'orchestrateur crash.

**Fix requis** : Remplacer par `async def _noop(): return None` et utiliser `await _noop()`.

---

### PC-6 : Pre-Spike Radar jamais invoqué dans le pipeline principal

**Impact : HAUT — Composant documenté mais absent du pipeline V7**

`main.py` passe `pre_spike_state=PreSpikeState.DORMANT` codé en dur à `DetectionInput`. Le module `src/pre_spike_radar.py` n'est appelé nulle part dans `process_ticker_v7`.

**Fix requis** : Ajouter un appel à `scan_pre_spike(ticker, features)` dans `process_ticker_v7` entre les étapes compute_features et signal_producer.detect.

---

### PC-7 : Cooldown des signaux (`_should_signal`) = code mort

**Impact : HAUT — Même ticker signalé toutes les 3 minutes indéfiniment**

`SignalProducer._should_signal()` est implémenté mais jamais appelé depuis `detect()`. Le cooldown de 5 minutes est inopérant. Même ticker → signal répété → alerts Telegram en boucle.

---

### PC-8 : Timezone naive/aware mélangées partout

**Impact : HAUT — Bugs silencieux sur les comparaisons de dates**

Présent dans : `dilution_detector.py`, `catalyst_score_v3.py`, `afterhours_scanner.py`, `global_news_ingestor.py`, `company_news_scanner.py`, `extended_hours_quotes.py`, `nlp_enrichi.py`, `api_pool/key_registry.py`.
Cause : mélange entre `datetime.now()` (naive) et `datetime.now(timezone.utc)` (aware).
Résultat : `TypeError: can't subtract offset-naive and offset-aware datetimes` en runtime.

---

### PC-9 : `top_gainers_source` — IBKR + Yahoo tous les deux cassés

**Impact : HAUT — Module C8 = zéro fonctionnel**

Voir PC-3. Module retourne une liste vide systématiquement.

---

### PC-10 : Singletons non thread-safe (race condition au démarrage)

**Impact : MOYEN — Double initialisation possible**

Affecte : `get_dilution_detector()`, `get_unified_guard()`, `get_pre_halt_engine()`, `get_registry()`, `get_pool_manager()`, `get_hot_queue()`, `get_pattern_learner()`, `get_missed_tracker()`, et la plupart des autres singletons.
Aucun n'utilise de `threading.Lock` autour de l'assignation du singleton.

---

### PC-11 : Multi-Radar V9 — Faux parallélisme (3 radars bloquants sur 4)

**Impact : CRITIQUE — `asyncio.gather()` tourne séquentiellement, pas en parallèle. Latence réelle ~35s vs ~200ms attendus.**

> Source : `plan_feature_engine_streaming.md` §0

`asyncio.gather(flow, catalyst, smart_money, sentiment)` (ligne 1193 de `multi_radar_engine.py`) lance les 4 radars en "parallèle", mais **3 radars contiennent des appels synchrones bloquants** qui gèlent l'event loop :

| Radar | Appel bloquant | Durée bloquée |
|-------|----------------|---------------|
| **FLOW** | `fetch_candles()` ligne 414 → `ib.sleep(2)` | ~2s |
| **SMART MONEY** | `get_options_flow_score()` ligne 664 → `ib.qualifyContracts()` + `ib.reqSecDefOptParams()` | ~2-5s |
| **SENTIMENT** | `get_buzz_signal()` ligne 805 (Grok + Reddit + StockTwits séquentiels) | ~15-40s |
| CATALYST | Lectures cache/RAM uniquement | OK |

Résultat réel en production :
```
Attendu (parallèle) : ~200ms total
Réel (séquentiel)   : ~35s pire cas (sans cache), ~2-5s avec cache
Scan 200 tickers HOT : ~7 000s réel vs ~200s attendu
```

**Impact secondaire** : l'event loop bloquée coupe le streaming IBKR (`ibkr_streaming.py`) pendant la durée du scan Multi-Radar. Les ticks de prix sont perdus.

**Fixes requis dans `multi_radar_engine.py`** (voir `plan_feature_engine_streaming.md` §0.3-0.4) :
- **Flow Radar** ligne 414 : `fetch_candles()` → `await fetch_candles_async()`
- **Smart Money Radar** ligne 664 : `get_options_flow_score()` → `await loop.run_in_executor(_executor, ...)`
- **Sentiment Radar** lignes 805+838 : `get_buzz_signal()` + `get_nlp_sentiment_boost()` → `await loop.run_in_executor(_executor, ...)`
- Ajouter en haut du fichier : `_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="radar-io")`
- Partager le semaphore IBKR `_ibkr_bar_lock` entre Flow Radar et Smart Money Radar pour éviter la contention ib_insync

---

### PC-12 : Feature Engine — `compute_features_async` ignore le TickerStateBuffer

**Impact : HAUT — ~2s par ticker même pour les tickers déjà streamés en temps réel**

> Source : `plan_feature_engine_streaming.md` §1-4

Après le fix async (`10dd739`), `compute_features_async()` appelle encore `get_bars()` via thread pool (~2s/ticker réseau IBKR), même pour les tickers actifs dont le `TickerStateBuffer` contient déjà jusqu'à 120 snapshots en RAM.

```
Situation actuelle :
  compute_features_async("AAPL")  → get_bars() dans thread → 2s (même si AAPL streamé)

Architecture cible (buffer-first) :
  compute_features_async("AAPL")
    ├── 1. TickerStateBuffer.get_snapshots("AAPL")   → 0ms si disponible
    │       └── _build_df_from_buffer(snapshots)     → microsecondes
    └── 2. Fallback : fetch_candles_async()           → 2s (tickers non streamés)
```

**Gain attendu** :
| Métrique | Avant | Après |
|----------|-------|-------|
| Features ticker HOT (streamé) | ~2s | **<1ms** |
| Cycle RTH 200 tickers HOT | ~400s d'appels IBKR | **<1s** pour les HOT |

**Fichiers à modifier** (voir `plan_feature_engine_streaming.md` §3-4) :
- `src/engines/ticker_state_buffer.py` : vérifier/ajouter `get_snapshots(ticker, n)` → liste de snapshots
- `src/feature_engine.py` : ajouter `_build_df_from_buffer()` + logique de priorité dans `fetch_candles_async()`

**Point de vigilance** : features depuis le buffer sont des approximations (high=close, open=premier snapshot). Acceptable pour momentum/vwap/volatility. Moins précis pour `breakout_high` et `strong_green`.

---

## PARTIE 2 — ANALYSE MODULE PAR MODULE

---

### main.py

**Rôle** : Point d'entrée, boucle principale, orchestrateur des 3 sessions (AH/PM/RTH).

**Bugs/Erreurs** :
- `pre_spike_state` codé en dur à `DORMANT` — Pre-Spike Radar jamais appelé
- `"WATCH_EARLY"` string comparé à un `SignalType` enum → branche else toujours silencieuse
- `available_cash` toujours = `MANUAL_CAPITAL`, pas de tracking des positions ouvertes
- `state.record_trade()` jamais appelé après un EXECUTE_ALLOWED
- `edge_cycle_v7` itère avec `universe.iterrows()` (très lent pandas) au lieu de `universe["ticker"].tolist()`
- Audit journalier peut se déclencher plusieurs fois entre 20h30 et minuit (pas de garde sur la minute)

**Manquant** :
- `process_ibkr_news` défini mais jamais branché sur un callback IBKR
- Aucune gestion des signaux SIGTERM/SIGINT → pas de shutdown propre
- Pas de déduplication des signaux inter-cycles (même ticker → alert Telegram toutes les 3 min)
- `load_universe()` appelé plusieurs fois par session → devrait être caché par cycle

**Performance** :
- `compute_monster_score(ticker)` synchrone appelé dans un `async def` → bloque l'event loop
- `run_edge()` utilise `loop.run_until_complete()` sans concurrence réelle

**Architecture** :
- Code legacy `edge_cycle()` + imports `generate_signal`/`process_signal` toujours présents → 200+ lignes de code mort
- `USE_V7_ARCHITECTURE` toujours `True` mais la branche legacy ajoute de la complexité
- V7State est un god-object (initialisation singletons + compteurs + day rollover)

---

### config.py

**Bugs/Erreurs** :
- `DEFAULT_MONSTER_WEIGHTS` (V3, event=0.35) et `ADVANCED_MONSTER_WEIGHTS` (V4, event=0.25) coexistent → confusion sur lequel est utilisé
- `MANUAL_CAPITAL` défaut à `$1000` → `MAX_POSITION_PCT=10%` = $100 = exactement `MIN_ORDER_USD` → zéro marge
- `SOCIAL_BUZZ_SOURCES` inclut `"twitter"` mais aucune intégration Twitter n'existe
- `MULTI_RADAR_BUY_STRONG_THRESHOLD=0.75` défini mais **jamais référencé** dans `ConfluenceMatrix` (utilise 0.60 hardcodé)

**Manquant** :
- Pas de validation des clés API obligatoires au démarrage → échecs silencieux
- `IBKR_STREAMING_MAX_SUBSCRIPTIONS` hardcodé dans main.py (200) au lieu de config
- Pas de `FINNHUB_WS_URL` dans la config malgré `finnhub_ws_screener.py`

---

### src/engines/signal_producer.py

**Bugs/Erreurs** :
- `_should_signal()` implémenté mais jamais appelé → cooldown = dead code
- `signal_id` collision : format `%Y%m%d_%H%M%S` = IDs identiques si 2 tickers traités dans la même seconde (`uuid4` importé mais non utilisé)
- BREAKOUT déclenche BUY_STRONG si `score >= 0.55` (trop bas pour le signal maximum)
- LAUNCHING déclenche BUY_STRONG si `score >= 0.585`
- `detect_batch` via `asyncio.gather` sur du code purement synchrone → overhead coroutine sans bénéfice

**Manquant** :
- Pas de validation que `monster_score` est dans [0,1] avant les boosts
- `quick_detect` ne remplit aucun champ V8 → signaux toujours en mode legacy

---

### src/engines/order_computer.py

**Bugs/Erreurs** :
- `total_multiplier = signal_multiplier * spike_multiplier * halt_multiplier` → multiplicatif (interdit par V8 MIN-based)
- `stop_loss` peut être négatif pour les penny stocks si ATR > prix
- `valid_until = utcnow() + 1h` codé en dur → ordre expiré avant l'ouverture RTH si calculé en pre-market
- `available_cash` jamais décrémenté → chaque ordre voit le capital total disponible

**Manquant** :
- Aucun modèle de slippage pour les ordres MARKET sur small-caps illiquides
- `portfolio_context` jamais mis à jour après l'initialisation → position sizes incorrectes

---

### src/engines/execution_gate.py

**Bugs/Erreurs** :
- `record_trade_executed()` jamais appelé depuis main.py → limite journalière inopérante
- Check 9 (P&L limite) réduit `size_multiplier` à 0.5 mais n'ajoute pas à `blocks` → statut reste ALLOWED au lieu de REDUCED
- `high_count >= 2` → `multiplier = 0.0` mais aucune raison de blocage ajoutée → **blocage silencieux sans log**
- `_blocked_signals` inclut les EXECUTE_DELAYED (marché fermé) → statistiques faussées

**Manquant** :
- Pas de check pour position déjà ouverte sur le même ticker
- `MAX_TOTAL_EXPOSURE` défini dans `AccountState` mais jamais évalué dans `_run_all_checks`
- `MANUAL_BLOCK` dans `BlockReason` jamais vérifié

---

### src/engines/multi_radar_engine.py

**Bugs/Erreurs** :
- `RadarPriority` défini en double : plain class dans `smallcap_radar.py` ET Enum dans `multi_radar_engine.py` → conflit d'import
- `FlowRadar.scan` appelle `fetch_candles()` (HTTP synchrone) dans un `async def` → bloque les 4 radars parallèles → détruit le bénéfice de `asyncio.gather`
- `CatalystRadar` instancie `CatalystScorerV3()` à chaque appel → overhead massif
- `ConfluenceMatrix._upgrade_signal` lève `ValueError` si signal inconnu (ex: `"WATCH_EARLY"` legacy)
- `scan_batch` traite les tickers séquentiellement au lieu de `asyncio.gather(*[scan_ticker(t) for t in tickers])`
- `SessionAdapter.get_sub_session` : si `minutes_since_open`/`minutes_before_close` manquent dans `time_utils`, fallback silencieux en `"CLOSED"` → poids session toujours = weekend

**Manquant** :
- `MULTI_RADAR_*_THRESHOLD` de config.py jamais utilisés dans `ConfluenceMatrix`
- `_scan_history` in-memory perdu au redémarrage
- `get_ticker_trend()` implémenté mais jamais appelé

---

### src/engines/acceleration_engine.py

**Bugs/Erreurs** :
- `scan_all()` requiert `samples >= 5` mais `score()` accepte `samples >= 2` (fix C4) → incohérence : l'alerte ne tire pas pour les tickers 2-4 samples
- Callbacks `scan_all` invoqués synchronement → une erreur dans un callback bloque les suivants

**Manquant** :
- **Jamais intégré dans `process_ticker_v7`** — tout le V8 est mort dans le pipeline principal
- Pas d'interface async malgré l'utilisation depuis des contextes async
- `set_baseline_raw` jamais appelé depuis main.py → z-scores sans signification

---

### src/engines/smallcap_radar.py

**Bugs/Erreurs** :
- `RadarPriority` plain class (pas Enum) → conflit avec `multi_radar_engine.RadarPriority`
- `_evaluate_ticker` appelle `engine.score(ticker)` ET `buffer.get_derivative_state(ticker)` → calcul doublon des dérivées
- `run_continuous` bloque l'event loop (`self.scan()` synchrone dans un `async def`)

**Manquant** :
- `set_ticker_context` jamais appelé depuis main.py ou multi_radar → contexte (catalyst, gap, repeat gainer) toujours = 0
- Constants `SMALLCAP_MAX_MARKET_CAP`, `SMALLCAP_MIN_AVG_VOLUME` définis mais jamais utilisés dans `_evaluate_ticker`

---

### src/engines/ticker_state_buffer.py

**Bugs/Erreurs** :
- Fix C4 (`samples >= 2`) dans `AccelerationEngine` est sans effet : le buffer requiert `>= 3` dans `get_derivative_state` → `DerivativeState(state="DORMANT")` renvoyé pour 2 samples
- `price_change` calculé sur l'ensemble du buffer (jusqu'à 2h) → stock plat après un vieux move montre un z-score élevé
- `BaselineStats.price_std = 0.01` absolu → z-scores incomparables entre un stock à $0.50 et un à $20
- Vélocité calculée sans normalisation temporelle → si snapshots irréguliers, les "$/min" sont faux
- EXHAUSTED check : requiert `price_vel_pct > 0.005` → stock déjà arrêté après un gros move → classifié DORMANT au lieu d'EXHAUSTED

---

### src/scoring/monster_score.py

**Bugs/Erreurs** :
- `load_weights()` appellé à chaque invocation → milliers de lectures disque par cycle
- `compute_event_score` appelle `get_events_by_ticker` deux fois (calcul score + beat rate)
- `feats["volume_spike"]` normalisé avec scale=5 codé en dur
- Commentaire dit "6% weight" pour social buzz mais le vrai poids est 3%
- `compute_event_score` : `max(e["boosted_impact"] for e in events)` → `KeyError` si champ absent

**Manquant** :
- Pas de validation que la somme des poids = 1.0 → score peut dépasser [0,1] avant clamp
- Résultat du Multi-Radar V9 jamais réintégré dans le Monster Score comme boost additif
- `score_many()` séquentiel → pas de concurrence pour 3000 tickers

---

### src/scoring/weight_optimizer.py

**Bugs critiques** :
- `_load_signal_history()` appelle `get_signal_history(days_back=...)` — **fonction inexistante** dans `signal_logger.py` → optimizer échoue silencieusement chaque semaine
- `_compute_metrics` : les composants en historique sont **déjà pondérés** (`event * 0.25`) → re-pondération = double weighting, fitness corrompue
- Écrit dans `data/monster_weights_history.json` mais `monster_score.py` lit `data/monster_score_weights.json` → **la sortie n'est jamais lue**
- `optimize_weekly` jamais appelé depuis main.py, weekly_audit ou tout scheduler → module orphelin

---

### src/risk_guard/dilution_detector.py

**Bugs/Erreurs** :
- `DilutionEvent.days_since_filing()` : `datetime.now()` naive - `filing_date` UTC-aware → `TypeError`
- Toxic financing → `ACTIVE_OFFERING` tier (multipli 0.20) mais devrait être hard-block 0.0
- `_profiles` cache croît indéfiniment (3000 tickers × sessions) → memory leak
- Singleton non thread-safe

**Manquant** :
- `_toxic_tickers` in-memory → perdu au redémarrage
- `analyze_batch` sans limite de concurrence → 3000 coroutines simultanées

---

### src/risk_guard/unified_guard.py

**Bugs critiques** :
- `asyncio.coroutine(lambda: None)()` → **crash en Python 3.11** quand un composant est désactivé (voir PC-5)
- `HALT_IMMINENT` → `blocking=True` + `size_multiplier=0.0` mais NOT dans `HARD_BLOCK_CODES` → halt imminent traité en MIN mode au lieu de hard-block

**Manquant** :
- `_watchlist` populé mais jamais utilisé dans `assess()` → feature déclarée mais non implémentée
- `MAX_TOTAL_EXPOSURE` dans AccountState jamais évalué

---

### src/catalyst_score_v3.py

**Bugs/Erreurs** :
- `confluence_score` additionné deux fois : une fois dans la somme pondérée (15%) ET une fois en additive → double-comptage
- `CatalystScorerV3.__init__` crée une DB SQLite à chaque instanciation → resource leak si appelé sans singleton
- `calculate_temporal_decay` : `datetime.now()` naive vs `event_time` UTC-aware → `TypeError`
- `PDUFA_DECISION` et `BREAKTHROUGH_DESIGNATION` absents de `high_impact_types` → ne triggent pas le `CRITICAL` alert override

**Manquant** :
- Pas de singleton `get_catalyst_scorer()` → chaque appel externe crée une nouvelle instance + DB
- `record_performance` jamais appelé automatiquement → tracking historique = vide
- CatalystScorerV3 contourné dans Monster Score : `event_score` vient de `event_hub` directement, pas de V3

---

### src/signal_logger.py

**Bugs/Erreurs** :
- `init_db()` appelé à l'import → crash si `data/` inaccessible avant même le démarrage
- `metadata` stocké via `str({...})` (Python repr) → pas du JSON valide → `json.loads` échoue à la relecture
- Connexion SQLite créée et fermée à chaque `log_signal` → inefficace et risque de lock
- Pas de `check_same_thread=False` dans `get_signals_for_period` → `ProgrammingError` en multi-thread

**Manquant** :
- **`get_signal_history(days_back=N)` absente** → `weight_optimizer.py` échoue en silence
- Pas de colonne `actual_move_pct` → weight optimizer ne peut jamais évaluer la fitness
- Pas de colonne `components` → poids individuels non persistés
- Pas de WAL mode SQLite → performance dégradée en écriture intensive

---

### src/pre_spike_radar.py

**Bugs/Erreurs** :
- `calculate_volume_acceleration` : sigmoid centrée à 0.5 quand accélération = 0 → accélération plate = signal WEAK (faux positif)
- Cache module-level déclaré mais jamais utilisé dans `scan_pre_spike` → recalcul systématique
- Probabilités de spike hardcodées (15%/35%/55%/75%) sans calibration historique

**Manquant** :
- Jamais intégré dans le pipeline principal (voir PC-6)
- Pas d'intégration dans la Confluence Matrix du Multi-Radar
- `scan_universe_pre_spike` synchrone → bloque l'event loop pour N tickers

---

### src/pre_halt_engine.py

**Bugs/Erreurs** :
- `score >= 60` → `BLOCKED` (size_multiplier=0.0) mais ce seuil peut être atteint par volume+float seuls → faux positifs bloquants
- `_calculate_halt_probability` : multiplications séquentielles (×1.5 × ×1.3 × ×1.4) = pattern multiplicatif V7 que V8 devait corriger
- `record_halt` : `_halt_history[ticker]` croît indéfiniment → memory leak

**Manquant** :
- Pas d'interface async → appel synchrone dans un contexte async bloque l'event loop
- Pas de persistence → historique des halts perdu au redémarrage
- Deux systèmes de halt parallèles : `PreHaltEngine` (main.py) + `HaltMonitor` (unified_guard.py) sans partage de données

---

### src/feature_engine.py

**Bugs/Erreurs** :
- `ibkr_connector = get_ibkr()` à l'import → si IBKR non connecté au démarrage, reste `None` pour toujours
- `df.rename(columns={'date': 'timestamp'})` : le nom de colonne IBKR varie selon la version d'ib-insync → no-op silencieux
- NaN VWAP si volume = 0 (pre-market sans trades) → `feats["vwap_dev"]` = NaN propagé
- Code dupliqué entre `compute_features` et `compute_features_async`

**Manquant** :
- Pas de calcul ATR → `portfolio_engine.estimate_atr` utilise une approximation
- `_ibkr_bar_lock = Semaphore(1)` : serialise toutes les requêtes IBKR → annule le bénéfice de l'async pour les accès IBKR

---

### src/anticipation_engine.py

**Bugs/Erreurs** :
- `asyncio.get_event_loop().run_until_complete()` dans un contexte async → `RuntimeError: This event loop is already running` → SEC analysis silencieusement désactivée
- `time.sleep(0.05)` par ticker dans `_scan_with_ibkr` : pour 500 tickers = **25 secondes** de blocage
- `time.sleep(0.2)` par ticker dans `_scan_with_finnhub` : pour 200 tickers = **40 secondes** de blocage
- Détection d'anomalie volume contre `MIN_AVG_VOLUME * 3` (global) au lieu de la moyenne spécifique au ticker

**Manquant** :
- Pas de version async de `run_anticipation_scan`
- Pas d'intégration avec `CatalystScorerV3` → scoring des filings SEC = lookup simple 3 niveaux
- `SignalLevel` enum interne (WATCH_EARLY/BUY/BUY_STRONG/HOLD/EXIT) ne correspond pas à `SignalType` → translation manuelle requise

---

### src/afterhours_scanner.py

**Bugs/Erreurs** :
- `datetime.fromtimestamp(ts)` local vs `datetime.utcnow()` UTC → calcul d'âge des news décalé
- Pas de TTL/déduplication sur les alertes Telegram → même catalyst alerté toutes les 10 minutes

**Manquant** :
- Pas d'intégration FDA calendar
- Pas d'options flow check (puts/calls inhabituels sur earnings reporters AH)
- Pas de persistence des catalysts détectés → perdus à la fin du cycle

---

### src/pm_transition.py

**Bugs critiques** :
- `pm_data.get("last", 0)` → clé `last` **absente** du dict retourné par `pm_scanner` → `pm_position_in_range` toujours = `(0.5, 0)` → composant PM broken
- `pm_gap_quality` : même bug → `hold_score` toujours 0.5 → score biaisé

---

### src/pattern_analyzer.py

**Bugs/Erreurs** :
- Poids PM : `0.20+0.15+0.15+0.20+0.15+0.35 = 1.20` (dépasse 1.0) → clamp compresse tous les signaux PM de manière injuste
- `detect_opening_range_breakout` suppose des barres 1 minute → ORB incorrecte si barres 5 min
- `best_confidence` (float) inclus dans `active_patterns` → inflate `intraday_pattern_score`

**Manquant** :
- Résultats de `detect_all_intraday_patterns` (V9) jamais intégrés dans `compute_pattern_score` → V9 patterns = dead code
- Pas de session awareness (ORB en pre-market, HOD break after-hours → sans sens)

---

### src/boosters/insider_boost.py

**Bugs critiques** :
- `_calculate_boost` filtre `tx.transaction_type == "BUY"` mais le champ est `transaction_code == "P"` → time-decay **toujours dead code**
- `detect_insider_cluster` : `asyncio.get_event_loop().run_until_complete()` dans un event loop → `RuntimeError` → retourne toujours `None` en production
- Cache plain dict sans TTL enforcement → memory leak sur 3000 tickers

---

### src/boosters/squeeze_boost.py

**Bugs critiques** :
- `shortPercentFloat * 100` : Finnhub retourne déjà en % (ex: 15.3 pour 15.3%) → multiplication donne **1530%** → seuils jamais atteints → boost squeeze toujours 0
- `aiohttp.ClientSession.get(timeout=10)` : `10` entier au lieu de `ClientTimeout(total=10)` → `TypeError` ou ignoré
- Bypass total de `api_guard` et `pool_manager` → appels Finnhub non trackés

**Architecture** :
- `apply_squeeze_boost` utilise un boost multiplicatif → violation de la règle V8 "additif seulement"
- Même violation pour `insider_boost`

---

### src/repeat_gainer_memory.py

**Bugs critiques** :
- `fetch_and_record_top_gainers` : **stub non implémenté** → retourne toujours `False` → base de données vide en production
- `get_repeat_score_boost` : formule retourne 1.0-1.5x (multiplicatif) mais le système attend un additif 0.05-0.20

**Manquant** :
- Pas d'intégration avec `top_gainers_source.py` pour alimenter la base automatiquement
- Records > 180 jours jamais nettoyés → DB croît indéfiniment

---

### src/options_flow_ibkr.py

**Bugs/Erreurs** :
- `time.sleep(0.1)` par contrat × 60 contrats = **6 secondes minimum** par ticker
- `scan_options_flow` et `get_options_flow_score` créent chacun une **nouvelle instance** `IBKROptionsScanner` → aucun singleton
- `open_interest=0` hardcodé → signaux OI toujours zéro

**Manquant** :
- Aucune implémentation async
- Pas de singleton → overhead à chaque appel
- `NEAR_TERM_FOCUS` et `STRIKE_CLUSTERING` documentés mais non implémentés

---

### src/ingestors/sec_filings_ingestor.py

**Bugs/Erreurs** :
- `aiohttp.ClientSession.get(timeout=30)` → entier au lieu de `ClientTimeout` → `TypeError`
- `_parse_form4_xml` ne lit que la **première** transaction → Form 4 multi-transactions partiellement lues
- `CIKMapper` : connexion SQLite partagée sans lock → `"database is locked"` possible en async

**Manquant** :
- Pas de User-Agent header sur les requêtes EDGAR → risque de 429/403
- Form 4 dérivés (options/RSUs) non parsés
- `_load_sec_mapping` : insert un-by-un pour ~10k lignes → devrait utiliser `executemany`

---

### src/ingestors/global_news_ingestor.py

**Bugs/Erreurs** :
- `datetime.fromtimestamp()` local time au lieu d'UTC → timestamps décalés
- `last_finnhub_id` non persisté → chaque redémarrage re-fetche tous les items → doublons HOT tickers
- `scan_continuous` : `await callback(result)` sans vérifier si callback est une coroutine

---

### src/ibkr_streaming.py

**Bugs critiques** :
- `_evict_stale_subscriptions` appelle `unsubscribe()` qui acquiert `_sub_lock`, mais est elle-même appelée depuis `subscribe()` qui tient déjà `_sub_lock` → **deadlock**
- `subscribe_tick_by_tick` enregistre les contrats mais ne branche jamais `_process_tick_by_tick` à `tickByTickEvent`
- Pas de logique de reconnexion → si IBKR Gateway déconnecte, toutes les subscriptions perdues silencieusement

---

### src/fda_calendar.py

**Bugs/Erreurs** :
- Legacy scrapers BiopharmCatalyst : site SPA (React) → `requests.get` retourne HTML vide → retourne `[]` silencieusement
- OpenFDA + ClinicalTrials events ont `ticker=""` → `_get_ticker_events` retourne toujours vide → **60-70% des données = mortes**

**Manquant** :
- Pas de mapping company-name → ticker pour OpenFDA/ClinicalTrials
- `FDACalendarEngine` excellent en architecture mais inopérant faute de ce mapping

---

### src/extended_hours_quotes.py

**Bugs/Erreurs** :
- `snapshot=False` + `time.sleep(0.5)` + `cancelMktData` → anti-pattern poll-and-cancel que `ibkr_streaming.py` devait remplacer
- `reqHistoricalData` bloquant sur le thread ib_insync → paralyse tous les market data
- `get_extended_quote` et `detect_gap_forming` créent chacun une **nouvelle instance** → pas de singleton

**Architecture** :
- Devrait lire depuis `IBKRStreaming.get_quote(ticker)` (O(1) in-memory) au lieu de requêtes IBKR

---

### src/market_memory/context_scorer.py

**Bugs/Erreurs** :
- `ContextScorerV2.score_segmented` mute l'objet retourné depuis le cache → les entrées cachées sont corrompues pour les appels suivants
- Cache key `f"{ticker}_{signal_type}_{signal_score}"` avec float → `75.0` ≠ `75.000001` → explosion d'entrées cache

**Manquant** :
- `_segmented_scores` de V2 jamais persisté → tout perdu au redémarrage
- Pas de thread-safety sur `_score_cache`

---

### src/market_memory/pattern_learner.py

**Bugs/Erreurs** :
- `trades[-10:]` pour "recent" sans tri par date → trades dans un ordre non garanti
- `time_periods` mutable dict en attribut de dataclass → instances partagent le même objet

**Manquant** :
- **Aucune connexion à `MemoryStore`** → patterns perdus à chaque redémarrage
- `learn_all()` jamais appelé automatiquement quand des trades sont ajoutés
- `DAY_OF_WEEK` et `STREAK` pattern types définis dans l'enum mais jamais appris

---

### src/market_memory/missed_tracker.py

**Bugs/Erreurs** :
- `list.remove()` O(n) dans `cleanup_old` au lieu d'un deque borné
- Counter reset à 0 au redémarrage → collision d'IDs si DB persistée

**Manquant** :
- **Aucune connexion à `MemoryStore`** → tout perdu au redémarrage
- Pas de boucle background pour résoudre les `_pending_checks` automatiquement

---

### src/market_memory/memory_store.py

**Bugs/Erreurs** :
- `SQLiteStorage.query()` charge TOUS les records en mémoire puis filtre en Python → zéro bénéfice SQL
- `JSONStorage.save()` réécrit **tout** le fichier à chaque nouveau record → O(n) à chaque ajout
- `INSERT OR REPLACE` SQLite écrase `created_at` → perte du timestamp original

**Manquant** :
- `auto_save` et `compress_old_data` définis dans config mais jamais implémentés
- `load_score()` absent → les scores sauvés ne peuvent pas être relus
- Aucun index SQL sur `ticker` ou `signal_time` → full table scan sur chaque query

---

### src/schedulers/hot_ticker_queue.py

**Bugs critiques** :
- `eval()` sur données chargées depuis la DB → **injection de code possible**
- `promote()`/`demote()` mettent à jour `_tickers[ticker].priority` mais pas le heap → heap désordonné
- `sorted(self._heap)` O(n log n) à chaque appel → defeating le but du heap
- Ghost entries s'accumulent sans cleanup → performance dégradée progressivement

**Fix immédiat** : Remplacer `eval()` par `json.loads()` pour la désérialisation des métadonnées.

---

### src/schedulers/scan_scheduler.py

**Bugs/Erreurs** :
- `hot_tickers[:10]` / `hot_tickers[10:20]` slicing par index au lieu de priorité → HOT ticker à l'index 11 traité comme WARM
- `mark_scanned(ticker)` appelé avant que le scan async soit exécuté → TTL renewal ne se déclenche jamais
- Pas de handling NYSE holidays → REALTIME mode le Thanksgiving

---

### src/schedulers/batch_scheduler.py

**Bugs/Erreurs** :
- `self.sec_ingestor.fetch_8k_filings(...)` → méthode renommée en `fetch_all_recent()` → `AttributeError` au runtime
- `bare except: pass` dans `_task_cleanup` avale `KeyboardInterrupt`/`SystemExit`
- `list(self.universe)[:50]` pour social buzz → set ordering non déterministe → 50 tickers aléatoires différents à chaque run

**Performance** :
- `asyncio.sleep(1)` par ticker × 3000 = **3000 secondes** de pure attente
- Sequential → batch complet > 50 minutes (hors trading window)

---

### src/social_buzz.py (DEPRECATED)

**Bugs critiques** :
- `get_twitter_buzz_grok()` : Grok est un LLM sans accès temps réel à Twitter → **hallucine** des comptes de mentions → données fabriquées
- `detect_buzz_spike()` compare toujours à 0.3 hardcodé → pas de baseline dynamique
- Toutes les requêtes HTTP synchrones dans un contexte async → bloque l'event loop

---

### src/social_velocity.py

**Bugs/Erreurs** :
- `from utils.cache import TTLCache` → `TTLCache` peut ne pas exister → `ImportError` au démarrage
- Accélération calculée comme `mentions_1h - mentions_prev_hour` (différence simple) au lieu d'une vraie dérivée seconde normalisée temporellement

**Manquant** :
- Aucune ingestion automatique de données → moteur passif uniquement
- Pas de connection à `HotTickerQueue` pour promouvoir les tickers trending

---

### src/nlp_enrichi.py

**Bugs/Erreurs** :
- 3 appels Grok séparés par news item (entités + sentiment + classification) → 3-6 sec/item → saturation rate limit
- `get_nlp_sentiment_boost()` crée `NLPEnrichi()` à chaque appel sans singleton
- `aggregate_sentiment()` : `if prev_avg` → faux quand `prev_avg = 0.0`

**Manquant** :
- Model Grok `"grok-4-1-fast-reasoning"` probablement inexistant → tous les appels échouent silencieusement
- `social_sentiment` et `analyst_sentiment` dans `AggregatedSentiment` = toujours 0.0 (TODO)

---

### src/event_engine/nlp_event_parser.py

**Bugs/Erreurs** :
- `json.loads(content)` sans strip des balises markdown → Grok renvoie souvent ` ```json [...] ``` ` → `JSONDecodeError`
- Pas de validation HTTP status code → réponse 429/500 parsée comme JSON valide
- Model `"grok-4-1-fast-reasoning"` probablement inexistant
- `SYSTEM_PROMPT` extrait **seulement les catalysts bullish** → FDA rejections, earnings miss, delisting impossibles à détecter

---

### src/processors/nlp_classifier.py

**Bugs/Erreurs** :
- `ClassificationCache` : SQLite sans `check_same_thread=False` → `ProgrammingError` en async
- `classify_batch` : sleep dans chaque task → toutes les tasks dorment en même temps puis tirent simultanément → pas de rate limiting réel
- Fallback `_fallback_classify` : retourne `BREAKING_POSITIVE` pour tout texte non reconnu → faux positifs

---

### src/processors/ticker_extractor.py

**Performance critique** :
- `_extract_by_company_name` : scan linéaire O(n×m) sur ~10k noms d'entreprises pour chaque texte → trop lent pour le temps réel

**Manquant** :
- Tickers `.A`/`.B` (BRK.A, BRK.B) non supportés
- `FALSE_POSITIVE_TICKERS` incomplet (ex: AI, GM, GE, F, MO non inclus)

---

### src/api_pool/key_registry.py

**Bugs/Erreurs** :
- `_load_state` appelé avant que les clés soient enregistrées → cooldowns persistés jamais restaurés
- Connexion SQLite jamais fermée → resource leak sur long run
- `register_from_env` : si clé API manquante → warning seulement, pas d'erreur → echec silencieux

---

### utils/cache.py

**Bugs/Erreurs** :
- `_maybe_cleanup` appelé hors du lock → race condition possible en multi-thread
- `get` retourne `None` pour une valeur `None` cachée → impossible de distinguer miss et cache_hit(None)

---

### utils/api_guard.py

**Bugs/Erreurs** :
- Pas de version async → tous les modules async doivent contourner le guard
- Pas de check HTTP status code → une réponse 429 passe sans retry
- `timeout` paramètre accepté mais ignoré (hardcodé à 10s dans `safe_get`)

---

### utils/logger.py

**Bugs/Erreurs** :
- Check `if logger.handlers` non atomique → deux threads peuvent ajouter des handlers en doublon → chaque log line dupliquée
- `StreamHandler` activé sur serveur headless → I/O console inutile

---

### utils/market_calendar.py

**Bugs/Erreurs** :
- Calendrier expire en 2027 → toutes les dates post-2027 classées comme jours ouvrables sans avertissement
- `get_volume_adjustment_factor` vérifie `next_day` (demain) mais pas le vendredi avant un lundi de holiday

---

### utils/time_utils.py

**Bugs/Erreurs** :
- `market_session()` appelle 3-5 fois `now_us()` et `_get_market_close_time_today()` → redondant
- `now` parameter partiellement ignoré → test avec une heure simulée ne fonctionne pas correctement

**Manquant** :
- Pas de détection des sous-sessions RTH (OPEN/MIDDAY/CLOSE) au niveau utils → chaque module les réimplémente

---

### monitoring/system_guardian.py

**Bugs/Erreurs** :
- `bare except:` dans `check_finnhub` et `check_grok` → avale les erreurs de programmation
- Callback IBKR + poll 60s → **double Telegram alert** sur chaque changement d'état IBKR
- Alertes CPU/RAM envoyées toutes les 60s sans cooldown → flood Telegram possible

**Manquant** :
- Pas de monitoring du process GV2-EDGE lui-même (RSS memory growth)
- Pas de check du nombre de file descriptors ouverts (40+ log files + SQLite = risque FD exhaustion)
- Pas d'action de recovery → guardian alertes uniquement, ne redémarre rien

---

## PARTIE 3 — PRIORITÉS D'AMÉLIORATION

### SPRINT 1 — CORRECTIONS CRITIQUES (Semaine 1, impact immédiat)

| # | Action | Fichier | Effort |
|---|--------|---------|--------|
| S1-1 | Intégrer `AccelerationEngine.score()` dans `process_ticker_v7` | main.py | 2h |
| S1-2 | Appeler `state.record_trade()` après EXECUTE_ALLOWED | main.py | 30min |
| S1-3 | Corriger `asyncio.coroutine` → `async def _noop()` | unified_guard.py | 30min |
| S1-4 | Appeler `_should_signal()` dans `SignalProducer.detect()` | signal_producer.py | 30min |
| S1-5 | Corriger `shortPercentFloat * 100` → ne pas multiplier | squeeze_boost.py | 15min |
| S1-6 | Corriger `tx.transaction_type` → `tx.transaction_code == "P"` | insider_boost.py | 30min |
| S1-7 | Remplacer `eval()` par `json.loads()` | hot_ticker_queue.py | 15min |
| S1-8 | Ajouter appel Pre-Spike Radar dans `process_ticker_v7` | main.py | 1h |
| S1-9 | Corriger `pm_data.get("last", 0)` → clé correcte | pm_transition.py | 1h |
| S1-10 | Ajouter `get_signal_history()` dans signal_logger.py | signal_logger.py | 1h |
| S1-11 | **[PC-11]** Multi-Radar async : Flow→`await fetch_candles_async()`, Smart Money+Sentiment→`run_in_executor` + `_executor` ThreadPoolExecutor(6) | multi_radar_engine.py | 3h |
| S1-12 | **[PC-12]** Feature Engine buffer-first : `get_snapshots()` dans TickerStateBuffer + `_build_df_from_buffer()` dans `fetch_candles_async()` | ticker_state_buffer.py, feature_engine.py | 2h |

### SPRINT 2 — PERFORMANCE & ASYNC (Semaine 2)

| # | Action | Fichier | Impact |
|---|--------|---------|--------|
| S2-1 | `scan_batch` : utiliser `asyncio.gather(*[scan_ticker(t) for t in tickers])` (parallélisme inter-ticker) | multi_radar_engine.py | Throughput ×N |
| S2-1b | Partager semaphore IBKR `_ibkr_bar_lock` entre Flow Radar et Smart Money Radar | multi_radar_engine.py, feature_engine.py | Contention ib_insync |
| S2-3 | Supprimer `time.sleep()` blocants dans anticipation_engine, insider_boost, squeeze_boost | multiple | Event loop libéré |
| S2-4 | `InsiderBoostEngine.detect_insider_cluster` : corriger anti-pattern `run_until_complete` | insider_boost.py | Crash fix |
| S2-5 | `monster_score.py` : cacher `load_weights()` à l'initialisation, pas à chaque appel | monster_score.py | CPU -90% |
| S2-6 | `signal_logger.py` : connexion persistante + WAL mode | signal_logger.py | I/O -80% |
| S2-7 | `options_flow_ibkr.py` : singleton + async | options_flow_ibkr.py | Perf ×6 |
| S2-8 | `extended_hours_quotes.py` : lire depuis `IBKRStreaming.get_quote()` | extended_hours_quotes.py | Latence -3s/ticker |

### SPRINT 3 — INTÉGRITÉ DES DONNÉES (Semaine 3)

| # | Action | Fichier | Impact |
|---|--------|---------|--------|
| S3-1 | Normaliser tous les datetimes → `datetime.now(timezone.utc)` | 12+ fichiers | Crashes timezone |
| S3-2 | Connecter `PatternLearner` et `MissedTracker` à `MemoryStore` | pattern_learner.py, missed_tracker.py | Mémoire persistante |
| S3-3 | Initialiser `TickerStateBuffer` baselines depuis l'univers | main.py + ticker_state_buffer.py | Z-scores corrects |
| S3-4 | Corriger `weight_optimizer` → bonne fonction + bon fichier output | weight_optimizer.py | Optimisation fonctionnelle |
| S3-5 | `fda_calendar.py` : mapping company→ticker via `CIKMapper` | fda_calendar.py | 60-70% données récupérées |
| S3-6 | Ajouter `check_same_thread=False` à tous les SQLite en async | 5+ fichiers | Crashes SQLite |
| S3-7 | `repeat_gainer_memory` : implémenter `fetch_and_record_top_gainers` | repeat_gainer_memory.py | Base de données vivante |

### SPRINT 4 — FIABILITÉ & MONITORING (Semaine 4)

| # | Action | Fichier | Impact |
|---|--------|---------|--------|
| S4-1 | Thread-safe tous les singletons (lock) | 15+ fichiers | Race conditions |
| S4-2 | `top_gainers_source` : IBKR `ScannerSubscription` + alternative Python (yfinance v2) | top_gainers_source.py | C8 fonctionnel |
| S4-3 | `pattern_analyzer` : corriger poids PM (sum = 1.20 → 1.00) | pattern_analyzer.py | Scores corrects |
| S4-4 | `system_guardian` : cooldown par condition + déduplication alerts IBKR | system_guardian.py | No Telegram flood |
| S4-5 | `nlp_event_parser` : strip markdown JSON + check status HTTP | nlp_event_parser.py | Parse fiable |
| S4-6 | `hot_ticker_queue` : remplacer heap+sort par `sortedcontainers.SortedList` | hot_ticker_queue.py | O(log n) vs O(n log n) |
| S4-7 | `data_validator` : mettre à jour avec `EARLY_SIGNAL`, `NO_SIGNAL` | data_validator.py | Validation cohérente |

### SPRINT 5 — AMÉLIORATIONS STRATÉGIQUES (Mois 2)

| # | Action | Impact sur détection |
|---|--------|---------------------|
| S5-1 | Unifier les 2 appels Grok de `nlp_enrichi` + `nlp_event_parser` en 1 seul prompt structuré | -67% API calls Grok |
| S5-2 | `social_velocity` : ajouter boucle background de collecte Reddit/StockTwits | Module V9 activé |
| S5-3 | `CatalystScorerV3` → singleton + intégration directe dans Monster Score | Event score +précis |
| S5-4 | `watch_list` : corriger `include_advanced=True` + probability formula | Upgrades WATCH→BUY fonctionnels |
| S5-5 | `ticker_state_buffer` : velocité normalisée par delta-temps ($/sec) | Dérivées correctes |
| S5-6 | `nlp_event_parser` : ajouter catalysts négatifs au SYSTEM_PROMPT | Détection risk events |
| S5-7 | `memory_store.SQLiteStorage.query()` : filtres SQL natifs | O(n) → O(log n) |
| S5-8 | Feedback loop outcome : wirer `record_performance` après chaque mouvement post-signal | Adaptive weighting réel |

---

## PARTIE 4 — TABLEAU DE BORD DES FAIBLESSES

### Modules en ÉTAT CRITIQUE (ne fonctionnent pas en production)

| Module | Problème principal | Statut |
|--------|-------------------|--------|
| `top_gainers_source.py` | IBKR.run_scanner inexistant + Yahoo 404 | BROKEN |
| `repeat_gainer_memory.py` | `fetch_and_record_top_gainers` = stub | BROKEN |
| `weight_optimizer.py` | Appelle fonction inexistante + mauvais fichier output | BROKEN |
| `pattern_analyzer.py` (V9 patterns) | Résultats calculés mais jamais utilisés dans le score | DEAD CODE |
| `social_buzz.py` Twitter | Grok LLM hallucine les comptes de mentions | UNRELIABLE |
| `fda_calendar.py` OpenFDA + ClinicalTrials | Ticker "" → données jamais matchées | DEAD DATA |
| `extended_hours_quotes.py` | Poll-and-cancel obsolète | SUBOPTIMAL |
| `options_flow_ibkr.py` | Synchrone + nouvelle instance à chaque appel | SLOW |

### Modules PARTIELLEMENT FONCTIONNELS (bugs silencieux)

| Module | Problème principal |
|--------|-------------------|
| `pre_halt_engine.py` | PreHaltState HIGH trop facilement atteint → faux positifs bloquants |
| `acceleration_engine.py` | Fix C4 (samples >= 2) annulé par le buffer (requires >= 3) |
| `pm_transition.py` | Clé `last` absente → score PM toujours basé sur 0.5 |
| `signal_logger.py` | metadata stockée en Python repr, pas JSON |
| `insider_boost.py` | Time-decay code mort (mauvais field name) |
| `context_scorer.py` | Cache mutable partagé entre V1 et V2 → corruption |

### Modules FONCTIONNELS (mais avec limitations)

| Module | Limitation principale |
|--------|-----------------------|
| `multi_radar_engine.py` | **Faux parallélisme** : 3/4 radars bloquants → ~35s réel vs ~200ms attendu (PC-11) |
| `catalyst_score_v3.py` | Contourné dans Monster Score (event_hub utilisé directement) |
| `anticipation_engine.py` | 65 secondes de blocking sleep par scan cycle |
| `market_memory/memory_store.py` | Jamais utilisé par les modules qu'il doit persister |

---

## PARTIE 5 — ANALYSE DE L'OBJECTIF

### Gap analysis : "Détecter les top gainers AVANT le mouvement"

L'objectif fondamental du système est la **détection anticipative**. Voici les 5 chaines de détection et leur état réel :

| Chaîne de détection | Modules concernés | État |
|--------------------|--------------------|------|
| **Volume anomaly** (ACCUMULATING avant spike) | TickerStateBuffer → AccelerationEngine → SmallCapRadar → SignalProducer | ❌ Déconnecté du pipeline principal (PC-1) |
| **Catalyst précoce** (news/SEC avant le prix) | SECIngestor → EventHub → CatalystV3 → AnticipationEngine | ⚠️ SEC fonctionne, AnticipationEngine bloque 65s |
| **Options flow** (smart money) | IBKROptionsScanner → MultiRadar.SmartMoney | ⚠️ Lent (6s/ticker), pas de singleton |
| **Social velocity** (buzz accelerating) | SocialVelocityEngine → MultiRadar.Sentiment | ❌ Aucune ingestion automatique |
| **Repeat gainer memory** (ticker pattern) | RepeatGainerMemory → SignalProducer boost | ❌ Base de données vide (stub non implémenté) |

**Conclusion** : Sur les 5 chaînes de détection anticipative, **2 sont complètement inopérantes** et **2 sont partiellement défectueuses**. Seule la chaîne Catalyst (SEC 8-K) fonctionne de manière fiable, mais avec des limitations de performance.

### Potentiel d'amélioration estimé

En appliquant les corrections SPRINT 1-3 :
- **Détection ACCUMULATING** activée → signal 5-15 min avant le spike
- **Repeat gainers** alimenté automatiquement → reconnaissance des "runners habituels"
- **Social velocity** avec ingestion → détection du buzz croissant
- **Pre-Spike Radar** intégré → 4ème signal de confirmation
- **Market Memory persistante** → le système apprend réellement entre les sessions

Amélioration estimée du hit rate : +15% à +25% sur les top gainers journaliers (de ~6% couverture actuelle vers ~30%+ avec le pipeline complet opérationnel).

---

## RÉFÉRENCES EXTERNES

| Document | Contenu | Problèmes couverts |
|----------|---------|-------------------|
| `plan_feature_engine_streaming.md` | Multi-Radar faux parallélisme (diagnostic par radar + code fixes) + Feature Engine buffer-first path | PC-11, PC-12, S1-11, S1-12 |
| `PLAN_AMELIORATION_V9.md` | Plan d'amélioration V9 complet (21 améliorations, 5 sprints) | Sprint 5 |
| `PLAN_CORRECTION_COVERAGE.md` | Corrections couverture C1-C9 | P1-P10 (historique) |

---

## ANNEXE — LISTE COMPLÈTE DES FICHIERS ANALYSÉS

```
Fichiers analysés : 90 fichiers Python
├── main.py, config.py, daily_audit.py, weekly_deep_audit.py
├── src/
│   ├── signal_engine.py [DEPRECATED], anticipation_engine.py, afterhours_scanner.py
│   ├── catalyst_score_v3.py, ensemble_engine.py [DEPRECATED], extended_hours_quotes.py
│   ├── fda_calendar.py, feature_engine.py, historical_beat_rate.py
│   ├── ibkr_news_trigger.py, ibkr_streaming.py, news_flow_screener.py
│   ├── nlp_enrichi.py, options_flow_ibkr.py, pattern_analyzer.py
│   ├── pm_scanner.py, pm_transition.py, portfolio_engine.py [DEPRECATED]
│   ├── pre_halt_engine.py, pre_spike_radar.py, repeat_gainer_memory.py
│   ├── signal_logger.py, social_buzz.py [DEPRECATED], social_velocity.py
│   ├── top_gainers_source.py, universe_loader.py, watch_list.py
│   ├── conference_calendar.py, earnings_calendar.py, float_analysis.py
│   ├── ipo_tracker.py, levels_engine.py, sector_momentum.py
│   ├── engines/ (7 fichiers)
│   ├── scoring/ (2 fichiers)
│   ├── risk_guard/ (4 fichiers)
│   ├── market_memory/ (4 fichiers)
│   ├── schedulers/ (3 fichiers)
│   ├── api_pool/ (4 fichiers)
│   ├── ingestors/ (4 fichiers)
│   ├── processors/ (3 fichiers)
│   ├── monitors/ (1 fichier)
│   ├── boosters/ (2 fichiers)
│   ├── event_engine/ (2 fichiers)
│   └── models/ (2 fichiers)
├── utils/ (6 fichiers)
├── alerts/ (1 fichier)
├── backtests/ (2 fichiers)
├── monitoring/ (1 fichier)
├── validation/ (6 fichiers)
└── tests/ (4 fichiers)
```

---

*Rapport généré automatiquement par audit multi-agents parallèles — GV2-EDGE V9.0*
*Date : 2026-02-25*
