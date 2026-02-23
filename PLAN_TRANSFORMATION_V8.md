# GV2-EDGE V8.0 - Plan de Transformation Complet

> **Objectif**: Radar anticipatif small caps intraday, détection ultra-précoce des top gainers
> **Latence cible**: < 30-45 secondes sur tickers HOT
> **Date**: 2026-02-17
> **Base**: Analyse de 78 fichiers source (REVIEW_GV2_EDGE_V7.md)
> **Statut final** : PLAN V8 COMPLETE. Supersede par V9 (Multi-Radar + Streaming).
> **Voir aussi** : PLAN_AMELIORATION_V9.md, PLAN_CORRECTION_COVERAGE.md

---

## PHASE 1 - Transformation du Radar (Réactif → Anticipatif)

### 1.1 Nouveau module: `src/acceleration_engine.py`

Remplace la logique snapshot actuelle par un tracking continu de dérivées.

**Nouvelles métriques à implémenter:**

```python
@dataclass
class AccelerationMetrics:
    # Dérivée première (vélocité)
    price_velocity: float        # d(price)/dt sur 1-5 min
    volume_velocity: float       # d(volume)/dt sur 1-5 min

    # Dérivée seconde (accélération)
    price_acceleration: float    # d²(price)/dt²
    volume_acceleration: float   # d²(volume)/dt²

    # Z-scores (anomalies relatives)
    volume_zscore: float         # (current_vol - mean_20d) / std_20d
    momentum_zscore: float       # (current_mom - mean_20d) / std_20d
    spread_zscore: float         # (current_spread - mean) / std

    # Float velocity proxy
    turnover_rate: float         # volume_5min / float_shares
    turnover_acceleration: float # d(turnover_rate)/dt
```

**Stockage rolling obligatoire** - tableau circulaire de 120 entrées (2h à 1min):

```python
class TickerStateBuffer:
    """Buffer circulaire par ticker pour tracking de dérivées"""
    def __init__(self, max_size=120):
        self.prices: deque[float]    # maxlen=120
        self.volumes: deque[int]
        self.timestamps: deque[datetime]
        self.spreads: deque[float]

    def push(self, price, volume, spread, ts): ...
    def velocity(self, field, window=5) -> float: ...
    def acceleration(self, field, window=5) -> float: ...
    def zscore(self, field, baseline_window=60) -> float: ...
```

### 1.2 Refactoring de `src/feature_engine.py`

**Fichier**: `src/feature_engine.py`

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 130-134 | `raw_momentum(df, n=5)` = snapshot 5 barres | Ajouter `momentum_velocity(buf)` = d(momentum)/dt sur buffer |
| 136-141 | `raw_volume_spike(df)` = ratio instantané | Ajouter `volume_acceleration(buf)` = dérivée seconde du volume |
| 156-158 | `momentum(df)` normalisé à 0.15 fixe | `momentum_adaptive(buf)` normalisé par vol historique du ticker |
| 160-162 | `volume_spike(df)` cap à 8x fixe | `volume_anomaly(buf)` = z-score vs baseline 20j |
| 176-182 | `squeeze_proxy` = mom/vol | `squeeze_velocity(buf)` = taux de compression en cours |
| 205-264 | `compute_features()` cache 60s | Réduire cache à 5-10s pour tickers HOT, garder 60s pour NORMAL |

**Nouvelles fonctions à ajouter:**

```python
def compute_acceleration_features(ticker: str, buf: TickerStateBuffer) -> dict:
    """Features basées sur le buffer rolling, pas un snapshot DataFrame"""
    return {
        "price_velocity": buf.velocity("prices", window=5),
        "price_acceleration": buf.acceleration("prices", window=5),
        "volume_velocity": buf.velocity("volumes", window=5),
        "volume_acceleration": buf.acceleration("volumes", window=5),
        "volume_zscore": buf.zscore("volumes", baseline_window=60),
        "momentum_zscore": buf.zscore("prices", baseline_window=60),
        "turnover_rate": buf.volumes[-1] / get_float_shares(ticker),
    }
```

### 1.3 Refactoring de `src/pre_spike_radar.py`

**Fichier**: `src/pre_spike_radar.py`

Le module a déjà la bonne structure (accélération, confluence). Le problème est qu'il n'est **jamais alimenté en données temps réel**.

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 81-82 | `ACCELERATION_WINDOW_MINUTES=30`, `LOOKBACK_PERIODS=6` | Réduire à `WINDOW=10`, `PERIODS=10` pour 1-min granularité |
| 89-187 | `calculate_volume_acceleration()` attend `historical_volumes` en paramètre | Connecter directement au `TickerStateBuffer` pour auto-alimentation |
| 567-663 | `scan_pre_spike()` attend des dicts optionnels | Refactorer pour lire le buffer automatiquement par ticker |
| 71-74 | Seuils fixes (0.3, 0.25, 0.2, 0.4) | Rendre adaptatifs: `threshold = base × (1 / volatility_percentile)` |

**Nouveau pipeline d'alimentation:**

```
TickerStateBuffer (1-min updates)
    → acceleration_engine calcule dérivées
    → pre_spike_radar.scan_pre_spike() consomme automatiquement
    → Résultat intégré dans monster_score en < 10ms
```

### 1.4 Nouveaux signaux anticipatifs

Ajouter à `src/models/signal_types.py`:

```python
class PreSpikeState(Enum):
    DORMANT = "DORMANT"
    ACCUMULATING = "ACCUMULATING"  # NOUVEAU: volume croissant, prix stable
    CHARGING = "CHARGING"
    READY = "READY"
    LAUNCHING = "LAUNCHING"
    EXHAUSTED = "EXHAUSTED"
```

Le nouvel état `ACCUMULATING` se déclenche quand:
- `volume_acceleration > 0` pendant 3+ périodes consécutives
- `price_velocity ≈ 0` (< 1% mouvement)
- `spread` se resserre

C'est l'indicateur le plus précoce: le volume monte AVANT que le prix bouge.

---

## PHASE 2 - Refonte de la Hiérarchisation HOT/WARM/COLD

### 2.1 Refactoring de `src/schedulers/hot_ticker_queue.py`

**Fichier**: `src/schedulers/hot_ticker_queue.py`

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 46-48 | TTL fixes: HOT=3600, WARM=1800, NORMAL=900 | TTL dynamique: `ttl = base_ttl × activity_multiplier` |
| 51-53 | Intervalles fixes: HOT=90s, WARM=300s, NORMAL=600s | **HOT=15-30s**, WARM=60-120s, NORMAL=300s |
| 63-68 | 3 niveaux: HOT, WARM, NORMAL | 4 niveaux: **CRITICAL, HOT, WARM, COLD** |
| 226-277 | `push()` ne promeut pas automatiquement | Ajouter auto-promotion basée sur score composite |

**Nouveau système de priorité à 4 niveaux:**

```python
class TickerPriority(Enum):
    CRITICAL = 0   # NOUVEAU: spike imminent, scan toutes les 15s
    HOT = 1        # Signal fort, scan toutes les 30s
    WARM = 2       # Surveillance active, scan toutes les 2min
    COLD = 3       # Rotation background, scan toutes les 5min

INTERVALS = {
    TickerPriority.CRITICAL: 15,   # 15 secondes
    TickerPriority.HOT: 30,        # 30 secondes (objectif latence cible)
    TickerPriority.WARM: 120,      # 2 minutes
    TickerPriority.COLD: 300,      # 5 minutes
}
```

### 2.2 Promotion/Démotion Dynamique

**Nouveau module: `src/schedulers/priority_manager.py`**

Règles de promotion automatique:

```python
PROMOTION_RULES = {
    # COLD → WARM
    "cold_to_warm": [
        ("volume_zscore", ">", 1.5),           # Volume 1.5σ au-dessus
        ("catalyst_detected", "==", True),       # Tout catalyste
        ("buzz_acceleration", ">", 0),           # Buzz en croissance
    ],
    # WARM → HOT
    "warm_to_hot": [
        ("volume_zscore", ">", 2.0),            # Volume 2σ au-dessus
        ("pre_spike_confluence", ">=", 2),       # 2+ signaux pré-spike
        ("price_acceleration", ">", 0),          # Prix en accélération
        ("monster_score", ">=", 0.50),           # Score minimum
    ],
    # HOT → CRITICAL
    "hot_to_critical": [
        ("volume_zscore", ">", 3.0),            # Volume 3σ (extrême)
        ("pre_spike_confluence", ">=", 3),       # 3+ signaux convergents
        ("price_velocity", ">", 0.02),           # +2% en cours
        ("turnover_rate", ">", 0.05),            # 5% du float échangé
    ],
}

DEMOTION_RULES = {
    # CRITICAL → HOT (après 5min sans accélération)
    "critical_to_hot": ("time_since_last_signal", ">", 300),
    # HOT → WARM (après 15min sans activité)
    "hot_to_warm": ("time_since_last_signal", ">", 900),
    # WARM → COLD (après 30min sans événement)
    "warm_to_cold": ("time_since_last_signal", ">", 1800),
}
```

### 2.3 Refactoring du Scheduler

**Fichier**: `src/schedulers/scan_scheduler.py`

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 54-60 | `INTERVALS` = global 180s, hot 90s | Aligner sur nouveaux intervalles (15s-300s) |
| 255 | `await asyncio.sleep(5)` entre cycles | `await asyncio.sleep(1)` - cycle 1s pour CRITICAL |
| 298-303 | `hot_tickers[:10]` limité à 10 | CRITICAL: tous. HOT: top 20. WARM: top 10 |
| 306-310 | `hot_tickers[10:20]` warm limité | Slot dédié WARM séparé du HOT |
| 342 | `await asyncio.sleep(1)` rate limit batch | `asyncio.gather()` parallel pour batch |
| 449-461 | `range(3)` - 3 tickers par rotation | `range(10)` - 10 tickers par rotation COLD |

**Nouvelle architecture du cycle:**

```python
async def _realtime_cycle(self):
    now = datetime.utcnow()

    # CRITICAL: scan TOUS les tickers CRITICAL (toutes les 15s)
    critical = self.hot_queue.get_by_priority(TickerPriority.CRITICAL)
    if critical:
        await asyncio.gather(*[
            self._fast_scan(t) for t in critical
        ])

    # HOT: scan les ready-to-scan (toutes les 30s)
    hot = self.hot_queue.get_ready_for_scan(TickerPriority.HOT)
    if hot:
        await asyncio.gather(*[
            self._standard_scan(t) for t in hot[:20]
        ])

    # WARM: rotation (toutes les 2min)
    warm = self.hot_queue.get_ready_for_scan(TickerPriority.WARM)
    for t in warm[:10]:
        asyncio.create_task(self._standard_scan(t))

    # COLD: background rotation (toutes les 5min)
    await self._rotate_cold_universe(batch_size=10)

    # Global news: toutes les 3min (inchangé)
    if self._should_global_scan(now):
        asyncio.create_task(self._run_global_scan())
```

---

## PHASE 3 - Refonte du Scoring

### 3.1 Refactoring de `src/scoring/monster_score.py`

**Fichier**: `src/scoring/monster_score.py`

**Nouveaux poids V8 (accélération-centrés):**

```python
ANTICIPATIVE_WEIGHTS = {
    "event": 0.20,               # ↓ 25→20% (toujours important)
    "volume_acceleration": 0.18, # NOUVEAU: remplace "volume" snapshot
    "pattern": 0.12,             # ↓ 17→12% (confirmation, pas anticipation)
    "pm_transition": 0.10,       # ↓ 13→10%
    "momentum_velocity": 0.10,   # NOUVEAU: remplace "momentum" snapshot
    "squeeze_velocity": 0.08,    # ↑ 4→8% (squeeze = signal anticipatif)
    "options_flow": 0.10,        # = 10% (inchangé si disponible)
    "social_acceleration": 0.07, # NOUVEAU: remplace "social_buzz" snapshot
    "turnover_anomaly": 0.05,    # NOUVEAU: float velocity proxy
}
```

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 58 | `cache = Cache(ttl=30)` | `cache_hot = Cache(ttl=5)`, `cache_normal = Cache(ttl=30)` |
| 103-107 | `normalize(x, scale)` avec seuils fixes | `normalize_adaptive(x, ticker_baseline)` avec z-score |
| 148 | `momentum = normalize(abs(feats["momentum"]), 0.2)` | `momentum = zscore_normalize(accel_feats["momentum_zscore"])` |
| 149 | `volume = normalize(feats["volume_spike"], 5)` | `volume = zscore_normalize(accel_feats["volume_zscore"])` |
| 150 | `squeeze = normalize(feats["squeeze_proxy"], 10)` | `squeeze = accel_feats["squeeze_velocity"]` normalisé |
| 238-247 | Score base = somme pondérée de snapshots | Score = somme pondérée de **dérivées et z-scores** |

**Nouvelle fonction de normalisation adaptative:**

```python
def zscore_normalize(zscore: float, cap: float = 4.0) -> float:
    """
    Convertit un z-score en score 0-1
    z=0 → 0.5, z=2 → ~0.84, z=3 → ~0.95, z≥4 → 1.0

    Avantage vs normalize(): adaptatif au comportement propre du ticker
    """
    capped = max(-cap, min(cap, zscore))
    return 1 / (1 + math.exp(-capped))
```

### 3.2 Intégration d'anomalies comportementales

**Nouveau module: `src/scoring/anomaly_detector.py`**

```python
class BehavioralAnomaly:
    """Détecte les patterns anormaux vs comportement historique du ticker"""

    def detect(self, ticker: str, buf: TickerStateBuffer) -> AnomalyReport:
        return AnomalyReport(
            volume_anomaly=self._volume_anomaly(ticker, buf),     # z-score 20j
            spread_anomaly=self._spread_anomaly(ticker, buf),     # compression anormale
            activity_anomaly=self._activity_anomaly(ticker, buf), # trades/min vs baseline
            time_anomaly=self._time_anomaly(ticker, buf),         # activité hors horaires normales
        )

    def _volume_anomaly(self, ticker, buf) -> float:
        """Z-score du volume actuel vs moyenne mobile 20 jours"""
        current = buf.volumes[-1]
        baseline = self.get_baseline(ticker, "volume", days=20)
        return (current - baseline.mean) / max(baseline.std, 1)
```

### 3.3 Score contextuel par classe de titre

```python
class TickerClass(Enum):
    MICRO_CAP = "micro_cap"     # < 50M
    SMALL_CAP = "small_cap"     # 50M - 300M
    MID_SMALL = "mid_small"     # 300M - 2B

THRESHOLDS_BY_CLASS = {
    TickerClass.MICRO_CAP: {
        "volume_zscore_significant": 1.5,  # Plus bas car vol erratique
        "momentum_significant": 0.05,       # 5% = significatif
        "squeeze_significant": 3.0,
    },
    TickerClass.SMALL_CAP: {
        "volume_zscore_significant": 2.0,
        "momentum_significant": 0.03,
        "squeeze_significant": 5.0,
    },
    TickerClass.MID_SMALL: {
        "volume_zscore_significant": 2.5,
        "momentum_significant": 0.02,
        "squeeze_significant": 7.0,
    },
}
```

---

## PHASE 4 - Correction du Risk Guard

### 4.1 Suppression du blocage binaire

**Fichier**: `src/engines/execution_gate.py`

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 324-356 | Penny stock: ANY HIGH → `multiplier=0.0` | Penny stock: HIGH → `multiplier=0.15` minimum |
| 326-329 | `dilution_risk == "HIGH"` → 0.0 | `dilution_risk == "HIGH"` → 0.15 (position micro, pas blocage) |
| 346-348 | 2 MEDIUM → 0.0 pour penny | 2 MEDIUM → 0.35 pour penny |
| 354-356 | ALL LOW → 0.75 pour penny | ALL LOW → 0.85 pour penny (moins punitif) |

**Nouvelle matrice progressive:**

```python
def _check_risk_flags_v8(self, risk_flags: RiskFlags) -> tuple:
    is_penny = risk_flags.current_price < PENNY_STOCK_THRESHOLD

    # JAMAIS de blocage total sauf delisting confirmé ou halt actif
    MIN_MULTIPLIER = 0.10 if is_penny else 0.25

    if is_penny:
        risk_map = {
            "HIGH": 0.15,    # Micro position, pas blocage
            "MEDIUM": 0.50,
            "LOW": 0.85,
        }
    else:
        risk_map = {
            "HIGH": 0.30,
            "MEDIUM": 0.70,
            "LOW": 1.00,
        }

    # Prendre le MINIMUM des multiplicateurs (pas multiplication)
    multiplier = min(
        risk_map[risk_flags.dilution_risk],
        risk_map[risk_flags.compliance_risk],
        risk_map[risk_flags.delisting_risk],
    )

    # Plancher absolu
    multiplier = max(multiplier, MIN_MULTIPLIER)

    # EXCEPTION: blocage total UNIQUEMENT si:
    # - Delisting CONFIRMÉ (pas risque, confirmé)
    # - Halt actif en cours
    # - Toxic financing PROUVÉ (pas shelf, prouvé)
    if risk_flags.delisting_confirmed or risk_flags.halt_active:
        multiplier = 0.0

    return blocks, multiplier
```

### 4.2 Distinction dilution potentielle vs active

**Fichier**: `src/risk_guard/dilution_detector.py`

Créer 3 niveaux de dilution au lieu de 1:

```python
class DilutionStatus(Enum):
    ACTIVE_OFFERING = "active_offering"     # Secondary/ATM en cours → HIGH risk
    SHELF_RECENT = "shelf_recent"           # S-3 < 30j, pas d'offre → MEDIUM risk
    SHELF_DORMANT = "shelf_dormant"         # S-3 > 30j, pas d'offre → LOW risk
    CAPACITY_ONLY = "capacity_only"         # S-3 vieux, pas utilisé → NEGLIGIBLE

# Mapping vers multiplicateurs
DILUTION_MULTIPLIERS = {
    DilutionStatus.ACTIVE_OFFERING: 0.20,   # -80% position (pas blocage)
    DilutionStatus.SHELF_RECENT: 0.60,      # -40%
    DilutionStatus.SHELF_DORMANT: 0.85,     # -15%
    DilutionStatus.CAPACITY_ONLY: 1.00,     # Pas d'impact
}
```

### 4.3 Risk Guard adaptatif pour top gainers

**Nouveau: override momentum pour penny stocks en breakout**

```python
def _apply_momentum_override(self, multiplier: float, ticker_data: dict) -> float:
    """
    Si un penny stock montre un breakout catalysé RÉEL,
    réduire la pénalité de risque.
    """
    has_catalyst = ticker_data.get("catalyst_type") is not None
    price_up_pct = ticker_data.get("price_change_pct", 0)
    volume_zscore = ticker_data.get("volume_zscore", 0)

    # Breakout catalysé: catalyst + prix en hausse + volume anormal
    if has_catalyst and price_up_pct > 5 and volume_zscore > 2.0:
        # Réduire la pénalité de 50%
        adjusted = multiplier + (1.0 - multiplier) * 0.50
        return min(1.0, adjusted)

    return multiplier
```

### 4.4 Refonte multiplicateurs unified_guard

**Fichier**: `src/risk_guard/unified_guard.py`

Remplacer le mode multiplicatif (qui effondre vers 0) par le mode MIN:

```python
# AVANT (config actuelle):
# apply_combined_multipliers = True → 0.5 × 0.5 × 0.25 = 0.0625

# APRÈS:
# Mode MIN: prendre le pire multiplicateur, pas le produit
multiplier = min(
    dilution_multiplier,
    compliance_multiplier,
    halt_multiplier
)
# 0.5, 0.5, 0.25 → min = 0.25 (pas 0.0625)
```

---

## PHASE 5 - Correction Gap & Pré-Market

### 5.1 Correction du calcul de gap

**Fichier**: `src/pm_scanner.py`

| Ligne | Actuel | Transformation |
|-------|--------|----------------|
| 104-108 | Utilise `o`, `h`, `l`, `c` sans `pc` | Utiliser `pc` (previous close) pour le vrai gap |
| 113 | `gap_pct = (last - pm_open) / pm_open` | `gap_pct = (last - prev_close) / prev_close` |
| 115 | `momentum = (pm_high - pm_low) / pm_low` | Garder (range intra-PM correct) |
| 119-126 | Pas de `prev_close` dans metrics | Ajouter `prev_close`, `true_gap`, `gap_category` |

**Code corrigé:**

```python
def compute_pm_metrics(ticker):
    q = fetch_quote(ticker)

    prev_close = q.get("pc")  # Previous close (disponible mais ignoré actuellement)
    last = q.get("c")
    pm_high = q.get("h")
    pm_low = q.get("l")
    volume = q.get("v", 0)

    if not prev_close or not last:
        return None

    # VRAI gap overnight (corrigé)
    true_gap_pct = (last - prev_close) / prev_close

    # Gap PM high (max potentiel)
    gap_high_pct = (pm_high - prev_close) / prev_close if pm_high else true_gap_pct

    # Catégorisation du gap
    abs_gap = abs(true_gap_pct)
    if abs_gap < 0.03:
        gap_category = "NEGLIGIBLE"
    elif abs_gap < 0.08:
        gap_category = "EXPLOITABLE"     # Sweet spot pour entrée
    elif abs_gap < 0.15:
        gap_category = "EXTENDED"        # Possible mais risqué
    else:
        gap_category = "OVEREXTENDED"    # Trop tard, fade probable
```

### 5.2 Intégration volume pré-market

```python
def compute_pm_metrics_v8(ticker):
    # ... gap calculation ...

    # Volume pré-market relatif
    pm_volume = q.get("v", 0)
    avg_pm_volume = get_baseline(ticker, "pm_volume", days=20)
    pm_volume_ratio = pm_volume / max(avg_pm_volume, 1)

    # Score de liquidité PM
    pm_liquidity_score = min(1.0, pm_volume_ratio / 3.0)  # 3x avg = max score

    # Score composite PM
    pm_score = compute_pm_composite(
        true_gap_pct=true_gap_pct,
        gap_category=gap_category,
        pm_volume_ratio=pm_volume_ratio,
        pm_liquidity_score=pm_liquidity_score,
        pm_momentum=(pm_high - pm_low) / pm_low if pm_low else 0,
    )
```

### 5.3 Priorisation des gaps exploitables

```python
def compute_pm_composite(true_gap_pct, gap_category, pm_volume_ratio,
                          pm_liquidity_score, pm_momentum):
    """
    Score PM pondéré qui PRIORISE les gaps exploitables (3-8%)
    et PÉNALISE les gaps overextended (>15%)
    """
    # Courbe en cloche: pic à 5-8%, décroît avant et après
    gap_abs = abs(true_gap_pct)

    if gap_abs < 0.02:
        gap_score = gap_abs / 0.02 * 0.3    # Faible signal
    elif gap_abs < 0.05:
        gap_score = 0.3 + (gap_abs - 0.02) / 0.03 * 0.5  # Montée
    elif gap_abs < 0.08:
        gap_score = 0.8 + (gap_abs - 0.05) / 0.03 * 0.2  # Pic (0.8-1.0)
    elif gap_abs < 0.15:
        gap_score = 1.0 - (gap_abs - 0.08) / 0.07 * 0.4  # Déclin
    else:
        gap_score = 0.6 - (gap_abs - 0.15) / 0.10 * 0.3  # Pénalité

    gap_score = max(0, min(1.0, gap_score))

    # Composite
    return (
        gap_score * 0.40 +
        pm_volume_ratio_score(pm_volume_ratio) * 0.30 +
        pm_liquidity_score * 0.15 +
        min(1.0, pm_momentum / 0.10) * 0.15
    )
```

---

## PHASE 6 - Refonte Performance & Async

### 6.1 Suppression des goulots d'étranglement

**Fichier**: `src/schedulers/scan_scheduler.py`

| Ligne | Actuel (goulot) | Transformation |
|-------|-----------------|----------------|
| 255 | `asyncio.sleep(5)` entre cycles | `asyncio.sleep(1)` pour cycle CRITICAL |
| 299 | `hot_tickers[:10]` hardcodé | Dynamique: `hot_tickers[:max(20, len(critical)*2)]` |
| 342 | `asyncio.sleep(1)` dans boucle batch | `asyncio.gather()` par chunks de 20 |
| 436 | `asyncio.sleep(0.5)` entre buzz checks | `asyncio.gather()` parallèle |

### 6.2 Activation du pool API

**Fichier**: `src/ingestors/company_news_scanner.py` et autres ingestors

Remplacer les clés API hardcodées par le pool manager existant:

```python
# AVANT (chaque ingestor):
from config import FINNHUB_API_KEY

# APRÈS:
from src.api_pool.pool_manager import get_pool_manager
pool = get_pool_manager()
key = pool.get_key("finnhub")  # Rotation automatique entre clés
```

**Impact**: Avec 3-4 clés Finnhub, le throughput passe de 60 req/min à 180-240 req/min.

### 6.3 Cache TTL adaptatif par priorité

```python
class AdaptiveCache:
    """Cache dont le TTL dépend de la priorité du ticker"""

    def get_ttl(self, ticker: str) -> int:
        priority = get_hot_queue().get_priority(ticker)
        return {
            TickerPriority.CRITICAL: 5,    # 5 secondes
            TickerPriority.HOT: 10,        # 10 secondes
            TickerPriority.WARM: 30,       # 30 secondes
            TickerPriority.COLD: 60,       # 60 secondes
            None: 60,                       # Pas dans la queue
        }.get(priority, 60)
```

### 6.4 Batch API optimisé

Remplacer les appels séquentiels par des batches parallèles:

```python
async def fetch_quotes_batch(tickers: List[str]) -> Dict[str, dict]:
    """Fetch multiple quotes en parallèle avec semaphore"""
    sem = asyncio.Semaphore(20)  # 20 requêtes simultanées (vs 5 actuellement)

    async def fetch_one(ticker):
        async with sem:
            return ticker, await async_fetch_quote(ticker)

    results = await asyncio.gather(*[fetch_one(t) for t in tickers])
    return {t: q for t, q in results if q}
```

### 6.5 Suppression des limites artificielles

**Fichier**: `src/schedulers/scan_scheduler.py`

```python
# SUPPRIMER:
# Ligne 299: hot_tickers[:10]  → hot_tickers[:50]
# Ligne 306: hot_tickers[10:20] → géré par get_ready_for_scan()
# Ligne 450: range(3)          → range(20)
```

---

## PHASE 7 - Roadmap V8 Complète

### 7.1 Liste des réparations classées par impact stratégique

| # | Réparation | Impact Stratégique | Difficulté | Sprint |
|---|-----------|-------------------|------------|--------|
| R1 | Corriger gap PM (`prev_close`) | CRITIQUE | Facile (5 lignes) | 1 |
| R2 | Réduire cache tickers HOT (30s→5s) | HAUT | Facile | 1 |
| R3 | Supprimer blocage binaire penny stocks | CRITIQUE | Moyen | 1 |
| R4 | Distinction dilution potentielle/active | HAUT | Moyen | 1 |
| R5 | Remplacer multipliers multiplicatifs par MIN | HAUT | Facile | 1 |
| R6 | Ajouter `TickerStateBuffer` (rolling) | CRITIQUE | Moyen | 2 |
| R7 | Implémenter `acceleration_engine.py` | CRITIQUE | Complexe | 2 |
| R8 | Ajouter 4ème niveau CRITICAL dans queue | HAUT | Moyen | 2 |
| R9 | Réduire intervalles scan HOT (90s→30s) | HAUT | Facile | 2 |
| R10 | Activer pool API (pool_manager) | HAUT | Facile | 2 |
| R11 | Paralléliser scans (asyncio.gather) | HAUT | Moyen | 3 |
| R12 | Supprimer limites hardcodées ([:10], [:50]) | HAUT | Facile | 3 |
| R13 | Remplacer normalize() par zscore_normalize() | CRITIQUE | Complexe | 3 |
| R14 | Nouveaux poids Monster Score V8 | CRITIQUE | Moyen | 3 |
| R15 | Ajouter état ACCUMULATING à PreSpikeState | MOYEN | Facile | 3 |
| R16 | Implémenter anomaly_detector.py | HAUT | Complexe | 4 |
| R17 | Priority manager (promotion/démotion auto) | HAUT | Complexe | 4 |
| R18 | PM composite avec courbe gap en cloche | MOYEN | Moyen | 4 |
| R19 | Score contextuel par TickerClass | MOYEN | Moyen | 4 |
| R20 | Momentum override pour top gainers | MOYEN | Facile | 4 |
| R21 | Cache TTL adaptatif par priorité | MOYEN | Moyen | 5 |
| R22 | Fetch quotes batch parallèle | MOYEN | Moyen | 5 |
| R23 | Refactorer pre_spike_radar alimentation auto | MOYEN | Complexe | 5 |

### 7.2 Ordre optimal d'implémentation

```
SPRINT 1 (Urgences - 2-3 jours)
├── R1: Corriger gap PM prev_close          [5 lignes, impact immédiat]
├── R2: Réduire cache HOT 30s→5s            [1 ligne, gain latence]
├── R3: Supprimer blocage binaire           [30 lignes, débloquer penny stocks]
├── R4: Distinction dilution pot./active    [50 lignes, moins de faux blocages]
└── R5: Multiplicateurs MIN au lieu de ×    [10 lignes, fin effondrement vers 0]

SPRINT 2 (Fondations anticipatives - 5-7 jours)
├── R6: TickerStateBuffer                   [nouveau module, 200 lignes]
├── R7: acceleration_engine.py              [nouveau module, 300 lignes]
├── R8: Niveau CRITICAL dans queue          [modifier enum + intervalles]
├── R9: Intervalles HOT 90s→30s             [constantes]
└── R10: Activer pool API                   [rewire imports]

SPRINT 3 (Scoring anticipatif - 5-7 jours)
├── R11: Paralléliser scans                 [refactorer scheduler]
├── R12: Supprimer limites hardcodées       [5 changements simples]
├── R13: zscore_normalize()                 [refactorer monster_score]
├── R14: Nouveaux poids V8                  [mise à jour config]
└── R15: État ACCUMULATING                  [ajout enum + détection]

SPRINT 4 (Intelligence avancée - 5-7 jours)
├── R16: anomaly_detector.py               [nouveau module, 250 lignes]
├── R17: priority_manager.py               [nouveau module, 200 lignes]
├── R18: PM composite courbe cloche        [refactorer pm_scanner]
├── R19: Score par TickerClass             [classification + seuils]
└── R20: Momentum override                 [30 lignes dans execution_gate]

SPRINT 5 (Optimisation - 3-5 jours)
├── R21: Cache TTL adaptatif               [nouveau AdaptiveCache]
├── R22: Batch fetch parallèle             [async gather partout]
└── R23: Pre-spike alimentation auto       [rewire pre_spike_radar]
```

### 7.3 Latence cible après transformation

```
AVANT (V7 actuel):
┌──────────────────────────────────────┐
│ Ticker HOT → Monster Score: 30s cache │
│ + Feature Engine: 60s cache           │
│ + Scan Scheduler: 90s interval        │
│ = LATENCE TOTALE: 60-180 secondes     │
└──────────────────────────────────────┘

APRÈS (V8 cible):
┌──────────────────────────────────────┐
│ Ticker CRITICAL → Buffer: temps réel  │
│ + Accel Engine: calcul < 10ms         │
│ + Cache: 5s TTL                       │
│ + Scan Scheduler: 15s interval        │
│ = LATENCE TOTALE: 15-30 secondes      │
│                                        │
│ Ticker HOT → Buffer: temps réel       │
│ + Cache: 10s TTL                      │
│ + Scan Scheduler: 30s interval        │
│ = LATENCE TOTALE: 30-45 secondes      │
└──────────────────────────────────────┘
```

### 7.4 Métriques de succès V8

| Métrique | V7 Actuel | V8 Cible | Mesure |
|----------|-----------|----------|--------|
| Latence detection HOT | 60-180s | < 30-45s | Time from data → signal |
| Latence detection CRITICAL | N/A | < 15-30s | Nouveau tier |
| Taux blocage penny stocks | ~80% | < 20% | Signaux bloqués/total |
| Faux positifs dilution | ~60% | < 15% | S-3 shelf mal classés |
| Signaux anticipatifs (avant +5%) | ~0% | > 30% | Signaux émis avant move |
| Univers scanné/cycle | 13-15 tickers | 50-100 tickers | Tickers/cycle 5s |
| Throughput API | 60 req/min | 180+ req/min | Pool de clés |
| Accélération détectée | NON | OUI | volume_acceleration tracking |

---

## Résumé Exécutif

**23 réparations organisées en 5 sprints de 3-7 jours chacun.**

Les 5 premières réparations (Sprint 1) sont réalisables en 2-3 jours et débloquent immédiatement:
- Le gap PM correct (R1)
- Les penny stocks en breakout (R3-R5)
- La réactivité des tickers HOT (R2)

Les sprints 2-3 construisent les fondations du radar anticipatif (buffer rolling, accélération, z-scores).

Les sprints 4-5 ajoutent l'intelligence avancée (anomalies comportementales, promotion dynamique).

**Résultat final**: Un système capable de détecter un buildup de volume + compression technique + buzz social AVANT le mouvement de prix, avec une latence de 15-45 secondes sur les tickers prioritaires.

---

## STATUT FINAL (2026-02-21)

**Le plan V8 a ete integralement implemente:**

- PHASE 1 (Radar Anticipatif) : ✅ FAIT — AccelerationEngine + TickerStateBuffer + SmallCapRadar
- PHASE 2 (Hierarchisation) : ✅ FAIT — Hot Ticker Queue TTL 4h + auto-renewal
- PHASE 3 (Scoring) : ✅ FAIT — Monster Score V4 + z-scores adaptatifs
- PHASE 4 (Risk Guard) : ✅ FAIT — MIN-based + momentum override + planchers
- PHASE 5 (Gap/Pre-Market) : ✅ FAIT — prev_close corrige + gap zones
- PHASE 6 (Async/Performance) : ✅ FAIT — asyncio.gather + pool API
- PHASE 7 (23 reparations) : ✅ FAIT — Toutes les reparations R1-R23

Le systeme a ensuite ete etendu avec V9 :
- Multi-Radar Engine (4 radars paralleles + confluence matrix)
- IBKR Streaming (event-driven, ~10ms latence)
- Finnhub WebSocket Screener
- Session Adapter (6 sous-sessions)

Voir `PLAN_AMELIORATION_V9.md` pour les details V9.
