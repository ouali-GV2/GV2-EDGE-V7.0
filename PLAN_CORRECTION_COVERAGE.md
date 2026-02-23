# GV2-EDGE V8 — Plan de Correction : Couverture 100% Top Gainers

> **Objectif** : Éliminer tous les points de perte qui empêchent la détection précoce de TOUS les top gainers potentiels, sans liste limitée.
> **Date** : 2026-02-17
> **Branche** : `claude/review-hetzner-compatibility-d3HOb`
> **Base** : Audit complet du pipeline (signal_producer, feature_engine, smallcap_radar, acceleration_engine, anticipation_engine, scan_scheduler, hot_ticker_queue)
> **Statut final** : TOUTES LES 9 CORRECTIONS IMPLEMENTEES ET VALIDEES (2026-02-21)
> **Supersede par** : PLAN_AMELIORATION_V9.md (21 ameliorations supplementaires)

---

## DIAGNOSTIC — Les 10 Points de Perte Identifiés

### P1. Goulot Finnhub 60 req/min (CRITIQUE)
- **Fichier** : `src/feature_engine.py:100` (`safe_get(FINNHUB_CANDLE, ...)`)
- **Impact** : Chaque ticker = 1 appel API candle. 60 req/min = 180 tickers/cycle (3 min).
  Sur un univers de ~3000 tickers → **94% de l'univers jamais scanné dans un cycle**.
- **Calcul** : 3000 tickers / 60 req/min = 50 min pour un scan complet.
  Cycle RTH = 3 min → seulement 180 tickers = **6% de couverture par cycle**.

### P2. Feature Engine exige 20 bars minimum
- **Fichier** : `src/feature_engine.py:222` (`if df is None or len(df) < 20: return None`)
- **Impact** : Tickers récents, IPOs, faible historique → features = None → jamais scorés.
- **Conséquence** : Un IPO qui spike +80% jour 1 n'a pas 20 bars de 5-min → invisible.

### P3. Signal Producer : seuil NO_SIGNAL à 0.40
- **Fichier** : `src/signal_producer.py:62` (`EARLY_SIGNAL: 0.40`)
- **Plus** : Ligne 363-365 — EARLY_SIGNAL exige AUSSI (catalyst OU pre_spike != DORMANT)
- **Impact** : Un ticker à monster_score 0.38 avec volume z-score 3.0 mais sans catalyst connu = **NO_SIGNAL**.
  Il faut score >= 0.40 ET (catalyst OU pre-spike actif).
- **Conséquence** : Les movers "surprise" sans catalyst identifié dans nos sources sont ratés.

### P4. SmallCap Radar : filtres volume/float redondants et contradictoires
- **Fichier** : `src/engines/smallcap_radar.py:57-61`
  ```
  SMALLCAP_MAX_MARKET_CAP = 2_000_000_000   # $2B max
  SMALLCAP_MIN_PRICE = 0.50                  # $0.50 min
  SMALLCAP_MAX_PRICE = 20.00                 # $20 max
  SMALLCAP_MIN_AVG_VOLUME = 500_000          # 500K daily min ← PROBLÈME
  SMALLCAP_MAX_FLOAT = 100_000_000           # 100M float max
  ```
- **Impact** : Le `universe_loader V3` a supprimé le filtre volume pour laisser le Feature Engine gérer dynamiquement. Mais le SmallCap Radar **re-filtre en dur à 500K volume**.
- **Conséquence** : Un ticker à 150K volume moyen qui accumule silencieusement avant un catalyst FDA :
  - ✅ Inclus dans l'univers (V3)
  - ❌ Rejeté par SmallCap Radar (P4)
  - ❌ Probablement jamais scanné par Feature Engine (P1)

### P5. Anticipation Engine : batch limits durs
- **Fichier** : `src/anticipation_engine.py`
  - Ligne 260 : `scan_tickers = tickers[:100]` — Finnhub scan = 100 max
  - Ligne 421 : `for ticker in tickers[:20]` — Company news = 20 max
  - Ligne 489 : `for ticker in tickers[:10]` — Fallback news = 10 max
  - Ligne 711 : `catalysts = analyze_with_real_sources(list(suspects)[:30])` — Suspects = 30 max
  - Ligne 716 : `high_priority = [...score >= 0.5][:10]` — RTH = 10 max
- **Impact** : Même si l'univers est large, l'anticipation ne traite que les N premiers.
  Le ticker #101 n'est jamais scanné par Finnhub. Le ticker #21 n'a jamais de news.

### P6. Aucun flux streaming — tout est polling per-ticker
- **Fichier** : Audit complet — ZÉRO WebSocket dans le codebase
  - ❌ Aucune lib WebSocket dans `requirements.txt`
  - ❌ Aucun fichier `*stream*`, `*websocket*`, `*realtime*`
  - ❌ `config.py` : 0 setting WebSocket/streaming
- **Architecture** : 100% REST polling avec `requests.get()` + `time.sleep()` delays
- **Impact** : Impossible de détecter un spike en temps réel sur 3000 tickers avec du polling.
  Un ticker peut monter de +20% entre deux scans sans être détecté.

### P7. Acceleration Engine : 3 samples minimum, 5 pour alerter
- **Fichier** : `src/acceleration_engine.py`
  - Ligne 206 : `if ds.samples < 3 or ds.confidence < 0.2: return` (scoring)
  - Ligne 242 : `if ds.samples < 5 or ds.confidence < 0.3: continue` (alerting)
  - `ticker_state_buffer.py:263` : `confidence = min(1.0, n / 20)` (pleine conf. à 20 samples)
- **Impact** : Avec polling toutes les ~60s, il faut 3 min pour scorer, 5 min pour alerter.
  Un gainer qui fait +30% en 2 minutes ne sera alerté qu'après 5 minutes.

### P8. Alert cooldown 2 minutes par ticker
- **Fichier** : `src/acceleration_engine.py:60` (`ALERT_COOLDOWN_SECONDS = 120`)
- **Impact** : Un ticker qui accélère rapidement (phase 1 → phase 2 → breakout en 3 min) :
  - Alerte ACCUMULATING envoyée à t=0
  - Alerte BREAKOUT **bloquée** jusqu'à t=2 min (cooldown)
  - Signal retardé de 2 min minimum

### P9. Hot Ticker TTL : expire et disparaît après 1h
- **Fichier** : `src/schedulers/hot_ticker_queue.py:46-48`
  ```
  TTL_HOT = 3600     # 1 heure
  TTL_WARM = 1800    # 30 min
  TTL_NORMAL = 900   # 15 min
  ```
- **Fichier** : `hot_ticker_queue.py:108-110` — `is_expired()` → supprimé silencieusement
- **Impact** : Un runner multi-heures (ex: DWAC +300% sur 4h) expire à 1h, revient à rotation normale (10 min interval), et ses mouvements rapides ne sont plus détectés en hot.

### P10. Aucune source externe temps-réel de top gainers
- **Architecture globale** : Tout dépend de la détection interne (scan news → catalyst → score)
- **Impact** : Si un ticker spike +50% sans catalyst dans nos sources (Finnhub news, SEC, Reddit), il est invisible jusqu'au prochain scan Feature Engine qui le touche (~1 chance sur 16 par cycle).

---

## CORRECTIONS — Plan d'Implémentation Détaillé

---

### CORRECTION C1 : Finnhub WebSocket Bulk Screener (élimine P1, P6)

**Priorité** : P0 — CRITIQUE — Fondation de tout le reste

**Objectif** : Observer TOUS les tickers en temps réel à 0 coût API, puis n'utiliser les 60 req/min REST que sur les candidats chauds.

#### C1.1 — Nouveau module `src/websocket_screener.py`

```
Fichier : src/websocket_screener.py (NOUVEAU)
Dépendances : websockets (à ajouter dans requirements.txt)
```

**Architecture** :
```
Finnhub WebSocket (wss://ws.finnhub.io)
    │
    ├── Subscribe: tous les tickers de l'univers
    │   (pas de limite de tickers sur le free tier WS)
    │
    ├── Message handler: {type: "trade", data: [{s, p, v, t}, ...]}
    │   │
    │   ├── Agrégation 1-min bars en mémoire (dict de deques)
    │   │   pour chaque ticker: prix, volume, VWAP en temps réel
    │   │
    │   ├── Détection anomalie VOLUME (z-score > 1.5 vs 20-day avg)
    │   │   → ticker ajouté à hot_candidates
    │   │
    │   ├── Détection anomalie PRIX (> 3% en 5 min)
    │   │   → ticker ajouté à hot_candidates
    │   │
    │   └── Détection ACCUMULATION (volume croissant, prix stable)
    │       → ticker ajouté à warm_candidates
    │
    └── Output: hot_candidates (set), warm_candidates (set)
        Taille typique: 50-150 tickers
        Rafraîchi en continu (temps réel)
```

**Interactions** :
- Le `scan_scheduler.py` lit `hot_candidates` au lieu de scanner l'univers complet
- Le `feature_engine.py` ne traite QUE les candidats → 50-150 appels REST/cycle ≤ 60 req/min
- Le `acceleration_engine.py` reçoit des snapshots en temps réel du WebSocket → plus de latence 60s

#### C1.2 — Modifications `requirements.txt`

```
Ligne à ajouter :
websockets>=12.0
```

#### C1.3 — Modifications `config.py`

```python
# Section WebSocket Screener (NOUVEAU)
WS_FINNHUB_URL = "wss://ws.finnhub.io"
WS_RECONNECT_DELAYS = [0, 2, 5, 15, 30]      # Backoff en secondes
WS_VOLUME_ZSCORE_THRESHOLD = 1.5               # Seuil anomalie volume
WS_PRICE_CHANGE_THRESHOLD = 0.03               # 3% en 5 min
WS_ACCUMULATION_VOLUME_RATIO = 1.5             # Volume/avg ratio pour accumulation
WS_ACCUMULATION_PRICE_MAX_CHANGE = 0.01        # Prix stable = < 1% variation
WS_BAR_INTERVAL_SECONDS = 60                   # Agrégation 1-min bars
WS_BASELINE_WINDOW_DAYS = 20                   # Fenêtre pour baseline volume
```

#### C1.4 — Modifications `src/schedulers/scan_scheduler.py`

```
Lignes à modifier :

Ligne 298-303 (hot tickers) :
  AVANT : hot_tickers = self.hot_queue.get_all_for_scan()
  APRÈS : hot_tickers = self.ws_screener.get_hot_candidates() | self.hot_queue.get_all_for_scan()

Ligne 312-313 (universe rotation) :
  AVANT : await self._rotate_universe()  # 3 tickers aléatoires
  APRÈS : warm = self.ws_screener.get_warm_candidates()
          await self._scan_warm_candidates(warm)  # Tous les warm, pas 3 aléatoires

Ligne 450 (rotation count) :
  AVANT : for _ in range(3)  # 3 tickers seulement
  APRÈS : # Supprimé — remplacé par warm_candidates du WebSocket
```

#### C1.5 — Modifications `src/feature_engine.py`

```
Ligne 270 (compute_many) :
  Le paramètre limit n'est plus nécessaire car les tickers arrivent
  déjà pré-filtrés par le WebSocket screener.
  Laisser limit=None par défaut (pas de changement de code, juste d'usage).
```

#### C1.6 — Modifications `main.py`

```
Démarrage du WebSocket screener dans la boucle principale :
  AVANT : while True → polling loop
  APRÈS :
    1. Lancer ws_screener en tant que tâche asyncio
    2. La polling loop continue pour les REST calls
    3. ws_screener alimente hot_ticker_queue en continu
```

---

### CORRECTION C2 : Supprimer filtres redondants SmallCap Radar (élimine P4)

**Priorité** : P1 — RAPIDE — Changement de config

#### C2.1 — Modifications `src/engines/smallcap_radar.py`

```
Ligne 60 :
  AVANT : SMALLCAP_MIN_AVG_VOLUME = 500_000
  APRÈS : SMALLCAP_MIN_AVG_VOLUME = 0  # Géré dynamiquement par Feature Engine

Ligne 61 :
  AVANT : SMALLCAP_MAX_FLOAT = 100_000_000
  APRÈS : SMALLCAP_MAX_FLOAT = 500_000_000  # Relax : 500M float (couvre mid-caps explosives)

Ligne 59 :
  AVANT : SMALLCAP_MAX_PRICE = 20.00
  APRÈS : SMALLCAP_MAX_PRICE = 50.00  # Couvre les mid-caps qui breakout au-dessus de $20
```

**Justification** :
- Le `universe_loader V3` a déjà supprimé le filtre volume — le Radar doit être aligné.
- Le Feature Engine + Monster Score gèrent la qualité dynamiquement.
- Un ticker à $22 avec 80K volume moyen qui accumule avant une FDA ne doit pas être exclu.

---

### CORRECTION C3 : Score plancher adaptatif (élimine P3)

**Priorité** : P1 — MOYEN — Modification signal_producer

#### C3.1 — Modifications `src/signal_producer.py`

```
Lignes 363-365 (EARLY_SIGNAL condition) :
  AVANT :
    if adjusted_score >= 0.40 and (has_catalyst or pre_spike_state != "DORMANT"):
        return SignalType.EARLY_SIGNAL

  APRÈS :
    # Seuil adaptatif : si anomalie volume forte, abaisser le plancher
    volume_anomaly = input_data.volume_zscore > 2.5
    effective_threshold = 0.30 if volume_anomaly else 0.40

    if adjusted_score >= effective_threshold:
        # Avec anomalie volume, pas besoin de catalyst connu
        if volume_anomaly or has_catalyst or pre_spike_state != "DORMANT":
            return SignalType.EARLY_SIGNAL
```

**Nouvelle logique** :
```
┌─────────────────────────────────────────────────────────┐
│ Condition EARLY_SIGNAL (avant)                          │
│   score >= 0.40 AND (catalyst OR pre_spike != DORMANT)  │
├─────────────────────────────────────────────────────────┤
│ Condition EARLY_SIGNAL (après)                          │
│   SI volume_zscore > 2.5:                               │
│     score >= 0.30  (plancher abaissé)                   │
│     (pas besoin de catalyst — le volume EST le signal)  │
│   SINON:                                                │
│     score >= 0.40 AND (catalyst OR pre_spike != DORMANT)│
│     (inchangé)                                          │
└─────────────────────────────────────────────────────────┘
```

**Justification** :
- Volume z-score > 2.5 = événement statistiquement rare (< 0.6% des cas).
- Si le volume explose, quelqu'un sait quelque chose. Pas besoin d'attendre un catalyst textuel.
- Score 0.30 reste suffisamment haut pour éviter les faux positifs.

#### C3.2 — Nouveau badge signal_producer

```
Ligne ~450 (badges) — Ajouter :
  if volume_anomaly and adjusted_score < 0.40:
      badges.append("⚡ Vol-Override")  # Indique que le seuil a été abaissé
```

---

### CORRECTION C4 : Accélerer le warm-up Acceleration Engine (élimine P7)

**Priorité** : P2 — MOYEN

#### C4.1 — Modifications `src/acceleration_engine.py`

```
Ligne 206 (scoring minimum) :
  AVANT : if ds.samples < 3 or ds.confidence < 0.2: return
  APRÈS : if ds.samples < 2 or ds.confidence < 0.1: return
  NOTE  : Avec le WebSocket (C1), les samples arrivent toutes les secondes,
          pas toutes les 60s. 2 samples = 2 secondes.

Ligne 242 (alerting minimum) :
  AVANT : if ds.samples < 5 or ds.confidence < 0.3: continue
  APRÈS : if ds.samples < 3 or ds.confidence < 0.15: continue
  NOTE  : Idem — 3 samples WebSocket = 3 secondes vs 5 minutes avant.
```

#### C4.2 — Modifications `src/engines/ticker_state_buffer.py`

```
Ligne 263 (confidence ramp-up) :
  AVANT : confidence = min(1.0, n / 20)       # Pleine confiance à 20 samples
  APRÈS : confidence = min(1.0, n / 10)       # Pleine confiance à 10 samples
  NOTE  : Avec WebSocket, 10 samples = 10 secondes. Confiance full en 10s vs 20 min.

Ligne 189 (dormant check) :
  AVANT : if not buf or len(buf) < 3: return DORMANT
  APRÈS : if not buf or len(buf) < 2: return DORMANT
```

**Garde-fou** : Ces changements ne sont valides QUE si C1 (WebSocket) est implémenté.
Avec le polling actuel (60s), réduire à 2 samples = 2 minutes est acceptable mais moins impactant.

---

### CORRECTION C5 : Réduire alert cooldown (élimine P8)

**Priorité** : P2 — RAPIDE

#### C5.1 — Modifications `src/acceleration_engine.py`

```
Ligne 60 :
  AVANT : ALERT_COOLDOWN_SECONDS = 120   # 2 minutes
  APRÈS : ALERT_COOLDOWN_SECONDS = 45    # 45 secondes

Alternative — cooldown adaptatif par état :
  ALERT_COOLDOWNS = {
      "ACCUMULATING": 120,    # Moins urgent, garder 2 min
      "LAUNCHING":     45,    # Réduire à 45s
      "PRE_LAUNCH":    30,    # 30s — le ticker est en pré-breakout
      "BREAKOUT":      15,    # 15s — phase critique, alerter rapidement
  }
```

**Justification** :
- Un ticker en BREAKOUT peut bouger de +10% en 30 secondes.
- Un cooldown de 2 min pendant un breakout = 2 min de retard sur l'alerte de montée.
- Un cooldown adaptatif garde le filtre anti-spam pour ACCUMULATING (lent) et accélère pour BREAKOUT (rapide).

---

### CORRECTION C6 : Étendre Hot Ticker TTL (élimine P9)

**Priorité** : P2 — RAPIDE

#### C6.1 — Modifications `src/schedulers/hot_ticker_queue.py`

```
Lignes 46-48 :
  AVANT :
    TTL_HOT = 3600      # 1 heure
    TTL_WARM = 1800     # 30 min
    TTL_NORMAL = 900    # 15 min

  APRÈS :
    TTL_HOT = 14400     # 4 heures (couvre un runner intraday complet)
    TTL_WARM = 3600     # 1 heure
    TTL_NORMAL = 1800   # 30 min
```

#### C6.2 — Auto-renewal basé sur activité

```
Ajouter dans hot_ticker_queue.py, méthode mark_scanned() :

  Ligne 320 (après mise à jour last_scan) :
    # Auto-renouveler TTL si le ticker est encore actif
    if hot.priority == TickerPriority.HOT:
        hot.expires_at = datetime.utcnow() + timedelta(seconds=TTL_HOT)
```

**Justification** :
- Un runner comme DWAC (+300% en 4h) ne doit pas expirer à 1h.
- Avec auto-renewal, tant qu'on le scanne (= tant qu'il est actif), il reste HOT.
- Il n'expire que s'il n'est PAS scanné pendant 4h (= vraiment mort).

---

### CORRECTION C7 : Relever les batch limits Anticipation Engine (élimine P5)

**Priorité** : P2 — RAPIDE

#### C7.1 — Modifications `src/anticipation_engine.py`

```
Ligne 260 :
  AVANT : scan_tickers = tickers[:100]
  APRÈS : scan_tickers = tickers[:300]
  NOTE  : Avec WebSocket (C1), les tickers arrivent pré-filtrés.
          300 est le max réaliste en post-WebSocket.

Ligne 421 :
  AVANT : for ticker in tickers[:20]
  APRÈS : for ticker in tickers[:50]
  NOTE  : Company news est critique pour catalyst detection.

Ligne 489 :
  AVANT : for ticker in tickers[:10]
  APRÈS : for ticker in tickers[:30]

Ligne 711 :
  AVANT : catalysts = analyze_with_real_sources(list(suspects)[:30])
  APRÈS : catalysts = analyze_with_real_sources(list(suspects)[:80])

Ligne 716 :
  AVANT : high_priority = [a.ticker for a in anomalies if a.score >= 0.5][:10]
  APRÈS : high_priority = [a.ticker for a in anomalies if a.score >= 0.4][:25]
  NOTE  : Seuil abaissé de 0.5 à 0.4 (aligné avec C3)
```

**Garde-fou** : Les rate limits Finnhub (60 req/min) sont gérés par `api_pool/pool_manager.py` qui throttle automatiquement. Relever les batch limits ne cause pas de rate limiting car le pool gère la file d'attente.

---

### CORRECTION C8 : Source externe top gainers (élimine P10)

**Priorité** : P3 — MOYEN — Filet de sécurité

#### C8.1 — Nouveau module `src/ingestors/external_gainers_ingestor.py`

```
Fichier : src/ingestors/external_gainers_ingestor.py (NOUVEAU)
```

**Sources possibles (par ordre de fiabilité)** :
1. **Finnhub `/stock/market-status`** + scan IBKR scanner (déjà disponible)
2. **Yahoo Finance screener** (gratuit, pas de clé API)
3. **Finviz screener** (gratuit, scraping)

**Logique** :
```python
async def fetch_external_gainers() -> List[str]:
    """
    Récupère les top gainers depuis des sources externes.
    Complémente la détection interne.
    Retourne les tickers qui NE SONT PAS déjà dans hot_ticker_queue.
    """
    gainers = set()

    # Source 1: IBKR Scanner (si connecté)
    if ibkr.is_connected():
        ibkr_gainers = ibkr.scan_top_gainers(limit=50)
        gainers.update(ibkr_gainers)

    # Source 2: Yahoo Finance (fallback, gratuit)
    yf_gainers = await scrape_yahoo_gainers(limit=50)
    gainers.update(yf_gainers)

    # Filtrer ceux déjà en hot queue
    new_gainers = gainers - set(hot_queue.get_all_tickers())

    # Forcer les nouveaux en tête de scan
    for ticker in new_gainers:
        hot_queue.push(ticker, TickerPriority.WARM, source="external_gainer")

    return list(new_gainers)
```

#### C8.2 — Intégration dans `scan_scheduler.py`

```
Ajouter un cycle external_gainers_check toutes les 2 minutes :

Dans INTERVALS (ligne 54-60) :
  "external_gainers": 120   # Toutes les 2 min

Dans _realtime_cycle() (après ligne 318) :
  if time_for_external_gainers:
      new = await fetch_external_gainers()
      for t in new:
          self.hot_queue.push(t, TickerPriority.WARM, source="external")
```

---

### CORRECTION C9 : Feature Engine — réduire le minimum bars (élimine P2)

**Priorité** : P3 — RAPIDE

#### C9.1 — Modifications `src/feature_engine.py`

```
Ligne 222 :
  AVANT : if df is None or len(df) < 20: return None
  APRÈS : if df is None or len(df) < 5: return None
  NOTE  : 5 bars minimum au lieu de 20. Les features qui nécessitent 20 bars
          (comme les moyennes mobiles longues) seront calculées avec des valeurs
          dégradées mais le ticker ne sera pas exclu.
```

#### C9.2 — Features dégradées pour historique court

```
Ajouter un flag dans le dict features retourné :

  features["short_history"] = len(df) < 20
  features["bar_count"] = len(df)

Le Monster Score peut utiliser ce flag pour ajuster la confiance :
  if features.get("short_history"):
      confidence *= 0.7  # Confiance réduite mais pas nulle
```

**Justification** :
- Un IPO qui spike +80% jour 1 a peut-être 5 bars de 5-min.
- Avec 5 bars : on calcule le volume ratio, le prix relatif, la volatilité court-terme.
- Les features long-terme (SMA20, Bollinger) seront NaN → confiance réduite mais ticker visible.

---

## ORDRE D'IMPLÉMENTATION

```
Phase 1 — Quick wins (1-2 heures)
├── C2 : SmallCap Radar filtres    → 3 lignes à modifier
├── C5 : Alert cooldown            → 1 ligne ou petit dict
├── C6 : Hot ticker TTL            → 3 lignes + 2 lignes auto-renewal
└── C9 : Feature Engine 20→5 bars  → 1 ligne + flag

Phase 2 — Score adaptatif (2-3 heures)
├── C3 : Score plancher adaptatif  → ~15 lignes signal_producer
└── C7 : Batch limits anticipation → 5 lignes

Phase 3 — WebSocket Screener (4-6 heures)
└── C1 : Finnhub WebSocket module  → Nouveau module + intégrations
    ├── C1.1 : websocket_screener.py
    ├── C1.2 : requirements.txt
    ├── C1.3 : config.py
    ├── C1.4 : scan_scheduler.py
    └── C1.6 : main.py

Phase 4 — Source externe (2-3 heures)
├── C8 : External gainers ingestor → Nouveau module
└── C4 : Warm-up accéléré          → Après C1 (dépend du WebSocket)
```

---

## MATRICE DE COUVERTURE AVANT/APRÈS

| Métrique | AVANT (V7) | APRÈS (V8 corrigé) |
|----------|-----------|---------------------|
| Couverture par cycle (3 min) | 6% (180/3000) | **100%** (WebSocket temps réel) |
| Latence détection anomalie vol | 3-50 min | **< 5 secondes** |
| Tickers low-volume pré-spike | ❌ Exclus (500K filtre) | ✅ Inclus (filtre supprimé) |
| IPOs / historique court | ❌ Exclus (20 bars) | ✅ Inclus (5 bars, conf. réduite) |
| Movers sans catalyst connu | ❌ NO_SIGNAL | ✅ EARLY_SIGNAL si vol z > 2.5 |
| Runner multi-heures (>1h) | ❌ Expire du hot queue | ✅ Auto-renewal TTL 4h |
| Alerte breakout latence | 2 min cooldown fixe | **15s** cooldown adaptatif |
| Warm-up nouvel entrant | 3-5 min (3-5 polls) | **2-3s** (2-3 WS samples) |
| Source externe gainers | ❌ Aucune | ✅ IBKR scanner + Yahoo |
| Batch limit anticipation | 10-30 tickers | **25-80 tickers** |

---

## RISQUES ET MITIGATIONS

| Risque | Mitigation |
|--------|-----------|
| WebSocket Finnhub free tier limité en symboles | Vérifier docs Finnhub ; fallback = polling amélioré |
| Faux positifs avec score 0.30 | Badge "Vol-Override" + confiance réduite → ne monte pas au-dessus de EARLY_SIGNAL |
| Overload Feature Engine avec trop de candidats | WebSocket pré-filtre à 50-150 max ; pool_manager throttle le reste |
| Yahoo scraping bloqué | Fallback sur IBKR scanner seul ; Feature Engine continue en autonome |
| Warm-up trop rapide → bruit | Confiance plafonnée à 0.5 pour les 5 premiers samples WS |

---

## FICHIERS IMPACTÉS (résumé)

| Fichier | Corrections |
|---------|-------------|
| `src/websocket_screener.py` | **NOUVEAU** — C1.1 |
| `src/ingestors/external_gainers_ingestor.py` | **NOUVEAU** — C8.1 |
| `requirements.txt` | C1.2 — ajouter `websockets>=12.0` |
| `config.py` | C1.3 — section WebSocket |
| `src/signal_producer.py` | C3.1, C3.2 — seuil adaptatif |
| `src/engines/smallcap_radar.py` | C2.1 — filtres relaxés |
| `src/acceleration_engine.py` | C4.1, C5.1 — warm-up + cooldown |
| `src/engines/ticker_state_buffer.py` | C4.2 — confidence ramp |
| `src/schedulers/hot_ticker_queue.py` | C6.1, C6.2 — TTL + auto-renewal |
| `src/anticipation_engine.py` | C7.1 — batch limits |
| `src/feature_engine.py` | C9.1, C9.2 — minimum bars |
| `src/schedulers/scan_scheduler.py` | C1.4, C8.2 — WebSocket + external |
| `main.py` | C1.6 — démarrage WebSocket |

---

## STATUT FINAL

**Toutes les 9 corrections (C1-C9) ont ete implementees et integrees dans le pipeline V9.**

| Correction | Statut | Date |
|-----------|--------|------|
| C1 - WebSocket Screener | ✅ FAIT | 2026-02-18 |
| C2 - SmallCap Radar filtres | ✅ FAIT | 2026-02-17 |
| C3 - Score plancher adaptatif | ✅ FAIT | 2026-02-17 |
| C4 - Warm-up accelere | ✅ FAIT | 2026-02-17 |
| C5 - Alert cooldown adaptatif | ✅ FAIT | 2026-02-17 |
| C6 - Hot Ticker TTL 4h | ✅ FAIT | 2026-02-17 |
| C7 - Batch limits anticipation | ✅ FAIT | 2026-02-17 |
| C8 - Source externe top gainers | ✅ FAIT | 2026-02-18 |
| C9 - Feature Engine 5 bars | ✅ FAIT | 2026-02-17 |

Le systeme a ensuite ete etendu avec le PLAN_AMELIORATION_V9.md (21 ameliorations, 5 sprints).
