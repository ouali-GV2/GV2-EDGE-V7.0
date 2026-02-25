# PLAN — Feature Engine : Priorisation Streaming sur get_bars()

> **Statut** : À faire (refactor de fond)
> **Priorité** : HAUT
> **Risque** : MOYEN (format DataFrame à adapter, tests obligatoires)
> **Prérequis** : Tests d'intégration avec IBKR Gateway actif

---

## 1. Problème résiduel

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

1. **Lire** `src/engines/ticker_state_buffer.py` en entier → noter le format exact des snapshots
2. **Écrire** `_build_df_from_buffer()` + tests unitaires (sans IBKR)
3. **Modifier** `fetch_candles_async()` avec la logique de priorité
4. **Tester** en déployant sur Hetzner avec IBKR Gateway actif
5. **Monitorer** les logs `feature-io-*` pour détecter des anomalies de thread
6. **Valider** que les hit rates du daily_audit restent stables (pas de régression features)
