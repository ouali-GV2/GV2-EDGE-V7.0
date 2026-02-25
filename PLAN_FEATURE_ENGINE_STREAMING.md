# PLAN — Multi-Radar Async + Feature Engine Streaming

> **Statut** : À faire (refactor de fond)
> **Priorité** : CRITIQUE
> **Risque** : MOYEN (format DataFrame à adapter, tests obligatoires)
> **Prérequis** : Tests d'intégration avec IBKR Gateway actif
> **Dernière mise à jour** : 2026-02-25

---

## 0. Multi-Radar V9 — Faux parallélisme (CRITIQUE)

### 0.1 Problème

`asyncio.gather()` (ligne 1193 de `multi_radar_engine.py`) lance les 4 radars en "parallèle", mais **3 des 4 radars contiennent des appels synchrones bloquants** qui gèlent l'event loop. Le résultat : les radars s'exécutent séquentiellement, pas en parallèle.

```
asyncio.gather(flow, catalyst, smart_money, sentiment)

Ce qui DEVRAIT se passer (parallèle vrai) :
  t=0ms      t=100ms     t=200ms
  ├─ FLOW ──────────────────────────┤  (simultané)
  ├─ CATALYST ──┤                      (simultané)
  ├─ SMART ────────────────────┤       (simultané)
  ├─ SENTIMENT ────────────────────┤   (simultané)
  Total : ~200ms

Ce qui se passe RÉELLEMENT (séquentiel par blocs) :
  t=0    t=2s        t=5s          t=35s
  ├ FLOW ┤           │             │
  │      ├ CATALYST ┤│             │    ← tourne seulement quand Flow libère
  │      │          ├┤ SMART      ┤│   ← re-bloque avec IBKR options
  │      │          ││            ├┤ SENTIMENT ──────────────────────┤
  Total : ~35s+ (pire cas, sans cache)
```

### 0.2 Diagnostic par radar

#### RADAR A : FLOW — BLOQUANT (~2s)

```
multi_radar_engine.py ligne 414 :
  df = fetch_candles(ticker, resolution="1", lookback=30)    ← SYNCHRONE
        └── ibkr_connector.get_bars()
              └── ib.sleep(2)  ← bloque l'event loop 2s
```

Le fix async qu'on a fait dans `main.py` (`await compute_features_async()`) ne s'applique **pas ici** — le multi-radar importe directement `fetch_candles` (synchrone), pas `fetch_candles_async`.

#### RADAR B : CATALYST — OK (non-bloquant)

```
- get_events_by_ticker()              ← cache mémoire     (0ms)
- CatalystScorerV3.score_catalysts()  ← calcul CPU        (<1ms)
- anticipation_engine.watch_early_signals ← dict RAM       (0ms)
- get_fda_events()                    ← fichier JSON       (<1ms)
```

Seul radar correctement non-bloquant. Lectures de cache et calculs purs.

#### RADAR C : SMART MONEY — BLOQUANT (~2-5s)

```
multi_radar_engine.py ligne 664 :
  get_options_flow_score(ticker)                              ← SYNCHRONE
    └── IBKROptionsScanner().get_options_summary()
          └── get_option_chain()
                ├── ib.qualifyContracts(stock)                ← IBKR blocking
                └── ib.reqSecDefOptParams(symbol, ...)        ← IBKR blocking
```

Même problème que Flow : ib_insync bloque l'event loop pendant les requêtes options OPRA.

#### RADAR D : SENTIMENT — BLOQUANT (~15-40s pire cas)

```
multi_radar_engine.py ligne 805 :
  get_buzz_signal(ticker)
    └── get_total_buzz_score(ticker)
          ├── get_twitter_buzz_grok()    ← API Grok xAI (safe_post, timeout=15s)
          ├── get_reddit_wsb_buzz()      ← API Reddit PRAW (search 5 subreddits, ~5-10s)
          └── get_stocktwits_buzz()      ← API StockTwits (safe_get, timeout=10s)

multi_radar_engine.py ligne 838 :
  get_nlp_sentiment_boost(ticker)
    └── aggregate_sentiment()            ← SQLite read (rapide, OK)
```

3 appels HTTP synchrones séquentiels. Pire cas sans cache : 15s + 10s + 10s = **35 secondes**. Normalement atténué par le cache TTL, mais au premier appel pour un ticker c'est un mur.

### 0.3 Corrections nécessaires dans `multi_radar_engine.py`

| Radar | Ligne | Appel bloquant | Fix |
|---|---|---|---|
| **FLOW** | 414 | `fetch_candles()` | Remplacer par `await fetch_candles_async()` |
| **SMART MONEY** | 664 | `get_options_flow_score()` | Wrapper dans `await loop.run_in_executor(_executor, ...)` |
| **SENTIMENT** | 805 | `get_buzz_signal()` | Wrapper dans `await loop.run_in_executor(_executor, ...)` |
| **SENTIMENT** | 838 | `get_nlp_sentiment_boost()` | Wrapper dans `await loop.run_in_executor(_executor, ...)` |
| **CATALYST** | — | Aucun | Pas de changement |

### 0.4 Code des corrections

#### Fix Flow Radar (ligne 412-414)

```python
# AVANT (bloquant) :
from src.feature_engine import volume_spike as get_vol_spike, fetch_candles
df = fetch_candles(ticker, resolution="1", lookback=30)

# APRÈS (non-bloquant) :
from src.feature_engine import volume_spike as get_vol_spike, fetch_candles_async
df = await fetch_candles_async(ticker, resolution="1", lookback=30)
```

#### Fix Smart Money Radar (ligne 662-664)

```python
# AVANT (bloquant) :
from src.options_flow_ibkr import get_options_flow_score
opt_score, opt_details = get_options_flow_score(ticker)

# APRÈS (non-bloquant) :
from src.options_flow_ibkr import get_options_flow_score
loop = asyncio.get_event_loop()
opt_score, opt_details = await loop.run_in_executor(
    _executor, get_options_flow_score, ticker
)
```

#### Fix Sentiment Radar (lignes 805 et 838)

```python
# AVANT (bloquant) :
buzz_signal = get_buzz_signal(ticker)
# ...
nlp_boost = get_nlp_sentiment_boost(ticker)

# APRÈS (non-bloquant) :
loop = asyncio.get_event_loop()
buzz_signal = await loop.run_in_executor(_executor, get_buzz_signal, ticker)
# ...
nlp_boost = await loop.run_in_executor(_executor, get_nlp_sentiment_boost, ticker)
```

#### Infrastructure à ajouter en haut de `multi_radar_engine.py`

```python
import concurrent.futures

# Thread pool partagé pour les appels bloquants dans les radars
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=6,
    thread_name_prefix="radar-io"
)
```

### 0.5 Points de vigilance spécifiques au multi-radar

**Semaphore IBKR** : Le Flow Radar et le Smart Money Radar utilisent tous les deux ib_insync. Si les deux tournent en parallèle dans des threads séparés, il faut que le `_ibkr_bar_lock` (semaphore) du `feature_engine.py` protège aussi les appels du Smart Money Radar. Options :
- Importer et réutiliser `_ibkr_bar_lock` de `feature_engine.py`
- Ou créer un semaphore global dans `ibkr_connector.py` partagé par tous les modules

**Cache social** : `social_buzz.py` a des caches TTL internes (cache mémoire). Après le premier appel bloquant (~35s), les appels suivants retournent depuis le cache en <1ms. Le problème est surtout au premier scan d'un ticker.

**`_executor` partagé vs séparé** : Utiliser un `ThreadPoolExecutor` séparé pour le multi-radar (6 workers) plutôt que réutiliser celui du `feature_engine.py` (4 workers), pour ne pas créer de contention entre les deux systèmes.

### 0.6 Gain attendu après fix multi-radar

| Métrique | Avant (séquentiel) | Après (parallèle vrai) |
|---|---|---|
| Scan 1 ticker (sans cache) | ~35-40s | **~15s** (limité par le plus lent = Grok) |
| Scan 1 ticker (avec cache) | ~2-5s | **<10ms** (tous les radars en cache) |
| Scan 200 tickers HOT | ~7000s | **~200s** (parallélisme inter-ticker aussi) |
| Event loop libre pendant scan | Non | **Oui** (streaming continue) |

---

## 1. Problème résiduel — Feature Engine

Le fix async (`10dd739`) a éliminé le blocage de l'event loop, mais `compute_features_async()` appelle encore `get_bars()` via un thread pool, ce qui prend **~2 secondes par ticker** même pour les tickers déjà streamés en temps réel.

```
Situation actuelle (après fix async) :
  compute_features_async("AAPL")
    └── fetch_candles_async()
          └── get_bars() dans thread worker   ← 2s même si AAPL est streamé
```

Le `TickerStateBuffer` contient déjà jusqu'à 120 snapshots en RAM pour les tickers actifs. Ces données pourraient alimenter les features **en quelques microsecondes** sans aucun appel réseau.

---

## 2. Architecture cible

```
compute_features_async("AAPL")
  │
  ├── 1. TickerStateBuffer.get_snapshots("AAPL")   ← 0ms, RAM
  │       └── Si len(snapshots) >= 5 :
  │               └── _build_df_from_buffer(snapshots)
  │                       └── compute features → DONE (microsecondes)
  │
  └── 2. Seulement si buffer vide ou insuffisant :
          └── fetch_candles_async()                ← 2s (fallback actuel)
                └── get_bars() dans thread
```

---

## 3. Fichiers à modifier

| Fichier | Modification |
|---|---|
| `src/engines/ticker_state_buffer.py` | Exposer `get_snapshots(ticker, n)` → liste de snapshots |
| `src/feature_engine.py` | Ajouter `_build_df_from_buffer()` + logique de priorité |
| `src/feature_engine.py` | Modifier `fetch_candles_async()` pour tenter le buffer d'abord |
| `tests/test_pipeline.py` | Tester les deux chemins (buffer vs get_bars) |

---

## 4. Détail des changements

### 4.1 `ticker_state_buffer.py` — Exposer les snapshots

Vérifier que la méthode `get_snapshots(ticker, n=120)` existe et retourne une liste de dicts ou d'objets avec les champs :
- `timestamp`, `price` (last), `volume`, `bid`, `ask`

Si seul `get_state(ticker)` existe, ajouter :

```python
def get_snapshots(self, ticker: str, n: int = 120) -> list:
    """Retourne les n derniers snapshots du ring buffer pour ce ticker."""
    if ticker not in self._buffers:
        return []
    buf = self._buffers[ticker]
    # buf est un deque ou une liste circulaire — adapter selon l'implémentation
    return list(buf)[-n:]
```

### 4.2 `feature_engine.py` — Builder DataFrame depuis le buffer

```python
def _build_df_from_buffer(snapshots: list) -> Optional[pd.DataFrame]:
    """
    Construit un DataFrame OHLCV approximatif depuis les snapshots streaming.

    Limitation : le streaming L1 donne last/bid/ask/volume — pas de vraies
    bougies OHLC. On approxime : open=first_last, high=max_last,
    low=min_last, close=last_last, volume=volume_delta.

    Acceptable pour momentum/vwap/volatility. Moins précis pour breakout_high.
    """
    if not snapshots or len(snapshots) < 5:
        return None

    try:
        df = pd.DataFrame([{
            "open":   s.get("price", 0),
            "high":   s.get("price", 0),
            "low":    s.get("price", 0),
            "close":  s.get("price", 0),
            "volume": s.get("volume", 0),
        } for s in snapshots])

        # Les snapshots sont des points de prix, pas des bougies.
        # Grouper par fenêtre de 1 min pour avoir des bougies propres.
        # Si pas de timestamp dans le snapshot, utiliser tel quel.

        if df["close"].isna().all() or (df["close"] == 0).all():
            return None

        return df

    except Exception:
        return None
```

### 4.3 `feature_engine.py` — Modifier `fetch_candles_async()`

```python
async def fetch_candles_async(ticker, resolution="1", lookback=120):
    """
    Priorité :
      1. TickerStateBuffer (0ms, streaming déjà en RAM)
      2. IBKR get_bars() dans thread (~2s)
      3. Finnhub REST dans thread (~500ms)
    """
    # --- Priorité 1 : Buffer streaming ---
    try:
        from src.engines.ticker_state_buffer import get_ticker_state_buffer
        buf = get_ticker_state_buffer()
        snapshots = buf.get_snapshots(ticker, n=lookback)
        df = _build_df_from_buffer(snapshots)
        if df is not None and len(df) >= 5:
            logger.debug(f"⚡ Buffer: features pour {ticker} ({len(df)} snapshots)")
            return df
    except Exception as e:
        logger.debug(f"Buffer unavailable for {ticker}: {e}")

    # --- Priorité 2 & 3 : fetch réseau (comportement actuel) ---
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        fetch_candles,
        ticker, resolution, lookback
    )
```

---

## 5. Points de vigilance

### 5.1 Format des snapshots dans TickerStateBuffer

Vérifier les champs exacts dans `src/engines/ticker_state_buffer.py` avant d'écrire `_build_df_from_buffer()`. Le snapshot peut s'appeler `price`, `last`, `last_price`, etc.

### 5.2 Précision des features depuis le buffer

| Feature | Qualité depuis buffer | Notes |
|---|---|---|
| `momentum` | ✅ Bonne | Série de prix suffisante |
| `volume_spike` | ✅ Bonne | Volume cumulatif disponible |
| `vwap_deviation` | ✅ Bonne | close × volume en streaming |
| `volatility` | ✅ Bonne | std des pct_change |
| `breakout_high` | ⚠️ Approximée | `high` = `close` (pas de vraie bougie) |
| `strong_green` | ⚠️ Approximée | `open` = premier snapshot de la série |
| `bollinger_squeeze` | ✅ Bonne | Basé sur std des prix |

### 5.3 Tickers non-streamés

Un ticker dans l'univers mais pas encore abonné au streaming n'a pas de buffer. Le fallback vers `get_bars()` reste indispensable.

### 5.4 ib_insync dans un thread worker

Risque résiduel : ib_insync utilise asyncio en interne. Si son event loop interne est dans le thread principal et qu'on l'appelle depuis un worker thread, des race conditions subtiles sont possibles. À monitorer en production (logs `feature-io-*`).

---

## 6. Tests à écrire avant de merger

```python
# tests/test_feature_engine_streaming.py

def test_build_df_from_buffer_basic():
    snapshots = [{"price": 10.0 + i*0.01, "volume": 1000} for i in range(20)]
    df = _build_df_from_buffer(snapshots)
    assert df is not None
    assert len(df) >= 5
    assert set(["open","high","low","close","volume"]).issubset(df.columns)

def test_build_df_from_buffer_empty():
    assert _build_df_from_buffer([]) is None
    assert _build_df_from_buffer([{"price": 0}] * 3) is None  # < 5 snapshots

async def test_fetch_candles_async_uses_buffer_first():
    # Peupler le buffer avec des snapshots mock
    buf = get_ticker_state_buffer()
    for i in range(20):
        buf.update("TEST", {"price": 10.0 + i*0.01, "volume": 500})

    # fetch_candles_async doit retourner depuis le buffer, pas via réseau
    df = await fetch_candles_async("TEST")
    assert df is not None
    # Vérifier qu'aucun appel réseau n'a été fait (mock get_bars + safe_get)

async def test_compute_features_async_buffer_path():
    # Test end-to-end avec buffer
    buf = get_ticker_state_buffer()
    for i in range(30):
        buf.update("MOCK", {"price": 5.0 + i*0.05, "volume": 2000})

    feats = await compute_features_async("MOCK", include_advanced=False)
    assert feats is not None
    assert "momentum" in feats
    assert "volume_spike" in feats
```

---

## 7. Gain attendu

| Métrique | Avant | Après |
|---|---|---|
| Features d'un ticker HOT (streamé) | ~2s (get_bars) | **<1ms** (buffer RAM) |
| Features d'un ticker COLD (non streamé) | ~2s | ~2s (fallback inchangé) |
| Cycle complet RTH avec 200 tickers HOT | ~400s IBKR calls | **<1s** pour les HOT |
| Précision features | Vraies bougies OHLC | Approximation (high=close) |

---

## 8. Ordre d'implémentation recommandé

### Phase 1 — Multi-Radar async (CRITIQUE, faire en premier)

1. **Ajouter** `_executor = ThreadPoolExecutor(max_workers=6)` en haut de `multi_radar_engine.py`
2. **Fix Flow Radar** : `fetch_candles()` → `await fetch_candles_async()` (ligne 414)
3. **Fix Smart Money Radar** : `get_options_flow_score()` → `await run_in_executor()` (ligne 664)
4. **Fix Sentiment Radar** : `get_buzz_signal()` + `get_nlp_sentiment_boost()` → `await run_in_executor()` (lignes 805, 838)
5. **Vérifier** le semaphore IBKR : s'assurer que `_ibkr_bar_lock` protège aussi les appels options du Smart Money Radar
6. **Tester** : vérifier que `asyncio.gather()` produit un vrai parallélisme (mesurer `scan_time_ms` de chaque radar)
7. **Monitorer** les logs `radar-io-*` en production

### Phase 2 — Feature Engine streaming (HAUT, après Phase 1)

1. **Lire** `src/engines/ticker_state_buffer.py` en entier → noter le format exact des snapshots
2. **Écrire** `_build_df_from_buffer()` + tests unitaires (sans IBKR)
3. **Modifier** `fetch_candles_async()` avec la logique de priorité (buffer RAM d'abord)
4. **Tester** en déployant sur Hetzner avec IBKR Gateway actif
5. **Monitorer** les logs `feature-io-*` pour détecter des anomalies de thread
6. **Valider** que les hit rates du daily_audit restent stables (pas de régression features)
