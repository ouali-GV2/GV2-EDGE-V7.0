# PLAN D'AMELIORATION V9 — GV2-EDGE

> **Objectif** : Detecter 95%+ des top gainers small-cap US (+30% a +300%) AVANT que le mouvement ne commence
> **Date** : 2026-02-18
> **Architecture cible** : V9.0 (V8 Acceleration + Streaming + Catalysts complets + Intelligence adaptive)
> **Deploiement** : Hetzner CX43 (8 vCPU, 16 Go RAM, 160 Go SSD)

---

## TABLE DES MATIERES

1. [Diagnostic actuel](#1-diagnostic-actuel)
2. [Axes d'amelioration](#2-axes-damelioration)
3. [Sprint 1 — Fondations temps reel](#3-sprint-1--fondations-temps-reel-priorite-critique)
4. [Sprint 2 — Intelligence catalysts](#4-sprint-2--intelligence-catalysts-priorite-haute)
5. [Sprint 3 — Detection avancee](#5-sprint-3--detection-avancee-priorite-haute)
6. [Sprint 4 — Apprentissage et optimisation](#6-sprint-4--apprentissage-et-optimisation-priorite-moyenne)
7. [Sprint 5 — Nettoyage et consolidation](#7-sprint-5--nettoyage-et-consolidation-priorite-moyenne)
8. [Matrice impact/effort](#8-matrice-impacteffort)
9. [Metriques de succes](#9-metriques-de-succes)
10. [Risques et mitigations](#10-risques-et-mitigations)

---

## 1. DIAGNOSTIC ACTUEL

### 1.1 Forces du systeme

| Force | Module | Impact |
|-------|--------|--------|
| Detection anticipative V8 | AccelerationEngine + SmallCapRadar | Detecte ACCUMULATING 5-15 min avant breakout |
| Architecture 3 couches | SignalProducer/OrderComputer/ExecutionGate | Detection jamais bloquee |
| Risk Guard MIN-based V8 | UnifiedGuard + Dilution/Compliance/Halt | Evite le sur-blocage V7 |
| IBKR L1 complet | ibkr_connector.py | Network A/B/C + OPRA |
| Hot Ticker Queue TTL 4h | hot_ticker_queue.py | Garde les tickers actifs |
| Catalyst Score V3 | catalyst_score_v3.py | 5 tiers, 18 types, decay temporel |

### 1.2 Faiblesses identifiees

| # | Faiblesse | Impact | Cause racine |
|---|-----------|--------|--------------|
| F1 | **Pas de streaming IBKR** | Latence 2s/ticker (poll-and-cancel) | ibkr_connector.py sleep(2) + cancel |
| F2 | **Earnings calendar absent** | Rate ~40% des earnings surprises | Pas de source systematique |
| F3 | **FDA calendar partiel** | Seulement les PDUFA manuels | fda_calendar.py incomplet |
| F4 | **Pas de conference tracking** | Rate JPM, ASH, ASCO, etc. | Aucun module |
| F5 | **Float/turnover non calcule** | Ne detecte pas les squeezes en formation | Pas de suivi float turnover |
| F6 | **10 modules morts/redondants** | ~3000 lignes de dette technique | Code legacy V6 jamais nettoye |
| F7 | **Pas de pattern intraday avance** | Rate les VWAP reclaims, ORB, etc. | pattern_analyzer.py basique |
| F8 | **Aucune detection S/R dynamique** | Pas de niveaux support/resistance temps reel | Non implemente |
| F9 | **Market Memory limité** | MRP/EP sans segmentation par catalyst type | context_scorer.py trop generique |
| F10 | **Pas de scoring float/SI** | Short Interest non integre au Monster Score | squeeze_boost.py basique |
| F11 | **Pas de backtest V8** | Impossible de valider acceleration_engine historiquement | backtest_engine_edge.py = V7 |
| F12 | **Pas d'IPO/SPO tracker** | Rate les IPO day 1 runners | Aucun module |
| F13 | **Social buzz sans velocity** | Compte les mentions sans mesurer l'acceleration | social_buzz.py statique |
| F14 | **Telegram alerts basiques** | Pas de graphiques, pas de contexte enrichi | telegram_alerts.py texte seulement |

### 1.3 Couverture actuelle des top gainers

```
Sources de detection actuelles:
  [OK] Monster Score V4 (9 composants)     → Score composite
  [OK] Acceleration Engine V8              → Derivees + z-scores
  [OK] SmallCap Radar V8                   → ACCUMULATING → BREAKOUT
  [OK] Pre-Spike Radar                     → Signaux precurseurs
  [OK] Options Flow IBKR (OPRA)            → Activite options inhabituelles
  [OK] SEC EDGAR (8-K + Form 4)            → Catalysts reglementaires
  [OK] Finnhub News                        → News generales
  [OK] Social Buzz (Reddit + StockTwits)   → Mentions sociales

  [MANQUE] Earnings Calendar reel          → 15-20% des top gainers
  [MANQUE] FDA Calendar complet            → 8-12% des top gainers
  [MANQUE] Conference Calendar             → 5-8% des top gainers
  [MANQUE] IPO/SPO Tracker                 → 3-5% des top gainers
  [MANQUE] Float Turnover / SI dynamique   → Detection squeeze 5-10%
  [MANQUE] IBKR Streaming temps reel       → Latence critique
  [MANQUE] Patterns intraday avances       → Timing d'entree

Estimation couverture actuelle: ~65-70% des top gainers
Objectif V9: 95%+
```

---

## 2. AXES D'AMELIORATION

### Vue d'ensemble

```
AXE 1: TEMPS REEL (Sprint 1)
├── A1: IBKR Streaming Engine              ✅ FAIT (ibkr_streaming.py)
├── A2: Integration Streaming → Pipeline   ✅ FAIT
└── A3: Tick-by-tick pour hot tickers      ✅ FAIT

AXE 2: INTELLIGENCE CATALYSTS (Sprint 2)
├── A4: Earnings Calendar Engine                       ✅ FAIT
├── A5: FDA Calendar Complet (OpenFDA + ClinicalTrials) ✅ FAIT
├── A6: Conference Calendar (JPM, ASH, ASCO, Bio-Europe...) ✅ FAIT
├── A7: IPO/SPO Tracker                                ✅ FAIT
└── A8: SEC Insider Clustering                         ✅ FAIT

AXE 3: DETECTION AVANCEE (Sprint 3)
├── A9: Float Turnover & Short Squeeze Score           ✅ FAIT
├── A10: Patterns Intraday Avances (VWAP, ORB, HoD)   ✅ FAIT
├── A11: Support/Resistance Dynamiques                 ✅ FAIT
├── A12: Social Velocity Engine (acceleration mentions) ✅ FAIT
└── A13: Cross-Ticker Correlation (sector momentum)    ✅ FAIT

AXE 4: APPRENTISSAGE (Sprint 4)
├── A14: Market Memory V2 (segmente par catalyst type) ✅ FAIT
├── A15: Backtest Engine V8 (validation acceleration)  ✅ FAIT
├── A16: Auto-Tuning Weights (Monster Score adaptatif) ✅ FAIT
└── A17: Telegram Alerts Enrichis (graphiques + contexte) ✅ FAIT

AXE 5: NETTOYAGE (Sprint 5)
├── A18: Supprimer modules morts                       ✅ FAIT
├── A19: Migrer modules deprecated                     ✅ FAIT
├── A20: Fusionner doublons                            ✅ FAIT
└── A21: Tests unitaires complets                      ✅ FAIT
```

---

## 3. SPRINT 1 — FONDATIONS TEMPS REEL (Priorite CRITIQUE)

### A1: IBKR Streaming Engine ✅ FAIT

**Fichier**: `src/ibkr_streaming.py` (nouveau)
**Statut**: IMPLEMENTE

Remplace le pattern poll-and-cancel (2s latence) par du streaming event-driven (~10ms).

```
Avant:  reqMktData() → sleep(2s) → read → cancelMktData()  [LENT]
Apres:  reqMktData() → pendingTickersEvent callback         [TEMPS REEL]
```

Fonctionnalites:
- Subscriptions persistantes (max 200 concurrentes)
- Event callbacks: on_quote(), on_event()
- Auto-detection: VOLUME_SPIKE, PRICE_SURGE, SPREAD_TIGHTENING, NEW_HIGH
- Feed automatique du TickerStateBuffer pour AccelerationEngine V8
- Integration HotTickerQueue (auto-promotion sur events)
- Integration SmallCapRadar (scan immediat sur events critiques)

---

### A2: Integration Streaming dans le Pipeline Principal ✅ FAIT

**Fichier**: `main.py` (modifier)
**Statut**: IMPLEMENTE
**Effort**: 1 jour
**Impact**: CRITIQUE

Integrer `ibkr_streaming.py` dans la boucle principale `run_edge()`:

```python
# Au demarrage (dans run_edge):
from src.ibkr_streaming import start_ibkr_streaming

streaming = start_ibkr_streaming(
    tickers=universe,          # Univers initial
    connect_hot_queue=True,    # Auto-promotion
    connect_radar=True,        # Feed SmallCapRadar
)

# Dans edge_cycle_v7:
# - Remplacer ibkr.get_quote() par streaming.get_quote() pour les hot tickers
# - Garder ibkr.get_quote() comme fallback pour les tickers non-subscribes
```

**Pipeline cible:**
```
IBKR Streaming (10ms) ──→ TickerStateBuffer ──→ AccelerationEngine V8
                                                        │
                                                        ▼
                                                SmallCapRadar (5s scan)
                                                        │
                                                        ▼
                                                SignalProducer V8 (Layer 1)
                                                        │
                                                        ▼
                                                OrderComputer (Layer 2)
                                                        │
                                                        ▼
                                                ExecutionGate (Layer 3)
```

---

### A3: Tick-by-Tick pour Hot Tickers ✅ FAIT

**Fichier**: `src/ibkr_streaming.py` (etendre)
**Statut**: IMPLEMENTE
**Effort**: 0.5 jour
**Impact**: HAUT

Pour les tickers en etat LAUNCHING/BREAKOUT, passer de L1 streaming a tick-by-tick:

```python
# ib_insync supporte reqTickByTickData:
self._ib.reqTickByTickData(contract, 'AllLast', 0, False)
# → Chaque trade individuel avec timestamp, prix, taille, exchange
```

Avantage: Detecter les block trades institutionnels (taille > 10x moyenne) qui precedent souvent les spikes.

---

## 4. SPRINT 2 — INTELLIGENCE CATALYSTS (Priorite HAUTE)

### A4: Earnings Calendar Engine ✅ FAIT

**Fichier**: `src/earnings_calendar.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: CRITIQUE (15-20% des top gainers sont earnings-driven)

**Sources de donnees (gratuites):**

| Source | API | Donnees |
|--------|-----|---------|
| SEC EDGAR XBRL | REST (gratuit, 10 req/s) | Dates officielles de reporting |
| Finnhub Earnings | REST (60 req/min free) | Dates + estimations EPS |
| Yahoo Finance | Scraping (gratuit) | Calendar + surprise history |

**Fonctionnalites:**

```python
class EarningsCalendar:
    def get_upcoming(self, days_ahead=7) -> List[EarningsEvent]:
        """Tickers avec earnings dans les X prochains jours"""

    def get_today(self) -> List[EarningsEvent]:
        """Earnings aujourd'hui (BMO/AMC)"""

    def get_beat_probability(self, ticker) -> float:
        """Probabilite de beat basee sur historique"""

    def get_surprise_magnitude(self, ticker) -> float:
        """Magnitude de surprise attendue (volatilite implicite)"""

@dataclass
class EarningsEvent:
    ticker: str
    date: datetime
    timing: str           # BMO (Before Market Open) / AMC (After Market Close)
    eps_estimate: float
    revenue_estimate: float
    beat_rate: float      # Historical beat %
    avg_surprise_pct: float
    implied_move: float   # From options IV
    sector: str
```

**Integration Monster Score:**
- J-7 a J-1: +0.05 a +0.10 boost (anticipation)
- J-Day BMO: +0.15 boost en pre-market
- Post-earnings (beat): +0.20 boost pendant 2h
- Post-earnings (miss): -0.15 penalty

**Sources reelles implementables:**
1. **Finnhub `/calendar/earnings`**: Date + EPS estimate + actual (free tier)
2. **SEC EDGAR**: 10-Q/10-K filing dates comme proxy
3. **Yahoo Finance screener**: Scraping earnings calendar page

---

### A5: FDA Calendar Complet ✅ FAIT

**Fichier**: `src/fda_calendar.py` (refactorer completement)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: HAUT (8-12% des top gainers sont FDA-driven)

**Probleme actuel**: fda_calendar.py a un calendrier statique avec ~20 PDUFA manuels.

**Sources de donnees reelles:**

| Source | Type | Contenu | Cout |
|--------|------|---------|------|
| OpenFDA API | REST (gratuit) | Approvals, warnings, recalls | Gratuit |
| ClinicalTrials.gov | REST (gratuit) | Phases, resultats essais | Gratuit |
| SEC EDGAR 8-K | REST (gratuit) | Annonces FDA dans filings | Gratuit |
| Finnhub FDA Calendar | REST (free tier) | PDUFA dates | Gratuit |

**Fonctionnalites:**

```python
class FDACalendarEngine:
    # Sources
    async def fetch_openfda_approvals(self) -> List[FDAEvent]:
        """OpenFDA drug approvals, denials, CRLs"""

    async def fetch_clinical_trials(self, phase="Phase 3") -> List[TrialEvent]:
        """ClinicalTrials.gov — essais Phase 2/3 avec resultats imminents"""

    async def fetch_pdufa_dates(self) -> List[PDUFAEvent]:
        """PDUFA dates (deadline decisions FDA)"""

    # Scoring
    def score_fda_catalyst(self, ticker) -> float:
        """Score 0-1 base sur proximite/phase/probabilite"""

    # Integration
    def get_upcoming_catalysts(self, days=30) -> List[FDAEvent]:
        """Tous les catalysts FDA a venir"""

@dataclass
class FDAEvent:
    ticker: str
    company: str
    drug_name: str
    event_type: str       # PDUFA, PHASE3_DATA, APPROVAL, CRL, ADCOM
    date: datetime
    phase: str            # Phase 1, 2, 3, NDA, BLA
    indication: str       # Maladie/condition
    probability: float    # Probabilite approbation estimee
    historical_precedent: float  # Taux approbation similar drugs
```

**FDA Catalyst Tiers:**
| Tier | Event | Boost | Fenetre |
|------|-------|-------|---------|
| TIER_1 | PDUFA (decision date) | +0.25 | J-3 a J-Day |
| TIER_2 | Phase 3 data readout | +0.20 | J-7 a J-Day |
| TIER_3 | ADCOM meeting | +0.15 | J-5 a J-Day |
| TIER_4 | Phase 2 topline | +0.10 | J-3 a J-Day |
| TIER_5 | CRL/Approval news | +0.20 | J-Day seulement |

---

### A6: Conference Calendar Biotech/Finance ✅ FAIT

**Fichier**: `src/conference_calendar.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 1.5 jours
**Impact**: HAUT (5-8% des top gainers sont conference-driven)

**Conferences cles pour small-cap movers:**

| Conference | Mois | Impact | Secteur |
|------------|------|--------|---------|
| JPM Healthcare | Janvier | CRITICAL | Biotech/Pharma |
| ASCO Annual | Juin | CRITICAL | Oncologie |
| ASH Annual | Decembre | HAUT | Hematologie |
| AAN Annual | Avril | HAUT | Neurologie |
| AACR Annual | Avril | HAUT | Oncologie |
| Bio-Europe | Novembre | MOYEN | Biotech EU |
| Needham Growth | Janvier | MOYEN | Tech small-cap |
| Oppenheimer Healthcare | Mars | MOYEN | Healthcare |

**Fonctionnalites:**

```python
class ConferenceCalendar:
    def get_active_conferences(self) -> List[Conference]:
        """Conferences en cours ou dans les 7 prochains jours"""

    def get_presenting_tickers(self, conference_id) -> List[str]:
        """Tickers avec presentations planifiees"""

    def get_conference_boost(self, ticker) -> float:
        """Boost pour Monster Score (0-0.15)"""

@dataclass
class Conference:
    name: str
    start_date: datetime
    end_date: datetime
    sector: str
    impact_level: str     # CRITICAL, HIGH, MEDIUM
    presenting_tickers: List[str]
    source_url: str
```

**Source de donnees:** Scraping des sites de conferences + SEC 8-K mentionnant presentations.

---

### A7: IPO/SPO Tracker ✅ FAIT

**Fichier**: `src/ipo_tracker.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 1 jour
**Impact**: MOYEN (3-5% des top gainers sont IPO day-1 ou follow-on)

**Sources:**

| Source | Donnees | Cout |
|--------|---------|------|
| SEC EDGAR S-1/424B | IPO filings | Gratuit |
| Finnhub IPO Calendar | Upcoming IPOs | Free tier |
| SEC EDGAR S-3/424B | Follow-on offerings | Gratuit |

```python
class IPOTracker:
    def get_recent_ipos(self, days=30) -> List[IPOEvent]:
        """IPOs des 30 derniers jours (day 1-30 runners)"""

    def get_upcoming_ipos(self, days=14) -> List[IPOEvent]:
        """IPOs planifiees (pre-IPO plays)"""

    def get_lockup_expirations(self, days=14) -> List[LockupEvent]:
        """Expirations de lockup (short squeeze catalysts)"""
```

---

### A8: SEC Insider Clustering ✅ FAIT

**Fichier**: `src/boosters/insider_boost.py` (etendre)
**Statut**: IMPLEMENTE
**Effort**: 0.5 jour
**Impact**: MOYEN

Ameliorer le module existant pour detecter les **clusters d'achats insiders** (3+ insiders achetant dans la meme semaine = signal fort).

```python
def detect_insider_cluster(ticker, days=7) -> Optional[InsiderCluster]:
    """
    Detecte les achats groupes d'insiders via SEC Form 4.
    3+ insiders achetant dans 7 jours = forte conviction interne.
    """
```

---

## 5. SPRINT 3 — DETECTION AVANCEE (Priorite HAUTE)

### A9: Float Turnover & Short Squeeze Score ✅ FAIT

**Fichier**: `src/float_analysis.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: CRITIQUE (les plus gros gainers sont souvent des squeezes)

**Concept**: Le float turnover mesure combien de fois le float entier a ete echange. Un turnover > 1.0x avec prix en hausse = squeeze en formation.

```python
@dataclass
class FloatAnalysis:
    ticker: str
    float_shares: int          # Nombre d'actions en circulation libre
    short_interest: int        # Actions shortees
    short_pct_float: float     # % du float shorte
    days_to_cover: float       # Jours pour couvrir les shorts
    turnover_ratio: float      # Volume cumule / float
    turnover_velocity: float   # Vitesse du turnover (1st derivative)
    squeeze_score: float       # 0-1 composite squeeze probability
    cost_to_borrow: str        # EASY, MEDIUM, HARD (si disponible)

class FloatAnalyzer:
    def analyze(self, ticker) -> FloatAnalysis:
        """Analyse complete float + SI + turnover"""

    def get_squeeze_candidates(self) -> List[FloatAnalysis]:
        """Tickers avec squeeze_score > 0.6"""

    def compute_turnover_intraday(self, ticker, volume_today, float_shares) -> float:
        """Turnover intraday en temps reel"""
```

**Integration Monster Score:**
- `squeeze_score` remplace le composant `squeeze` actuel (0.04 weight)
- Si squeeze_score > 0.7 ET volume_zscore > 2.0: boost additionnel +0.10

**Sources:**
- IBKR: Shortable shares + fee rate (via `reqMktData` generic tick 236)
- Finnhub: Short interest (delayed 2 weeks, free)
- SEC: Short interest filings (bi-mensuel)
- Yahoo Finance: Float shares, short % float

---

### A10: Patterns Intraday Avances ✅ FAIT

**Fichier**: `src/pattern_analyzer.py` (refactorer)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: HAUT

**Patterns actuels** (basiques): Flag, squeeze Bollinger, higher lows.

**Patterns a ajouter:**

| Pattern | Description | Pouvoir predictif |
|---------|-------------|-------------------|
| **VWAP Reclaim** | Prix repasse au-dessus du VWAP avec volume | HAUT — confirmation de force |
| **Opening Range Breakout (ORB)** | Break du range des 15 premieres minutes | HAUT — setup classique |
| **HOD Break** | Nouveau High of Day avec volume | CRITIQUE — continuation |
| **Red-to-Green** | Passage de negatif a positif avec volume | HAUT — reversal |
| **Consolidation Box** | Prix range tight (< 3% range) avec volume croissant | HAUT — pre-breakout |
| **Volume Shelf** | Support de volume visible (3+ tests) | MOYEN — floor |
| **Parabolic Setup** | 3+ higher lows avec acceleration | HAUT — pre-parabolic |

```python
class IntradayPatternAnalyzer:
    def detect_all(self, ticker, bars_1min) -> List[IntradayPattern]:
        """Detecte tous les patterns intraday sur barres 1-min"""

    def detect_vwap_reclaim(self, bars) -> Optional[IntradayPattern]:
    def detect_orb(self, bars, first_n_minutes=15) -> Optional[IntradayPattern]:
    def detect_hod_break(self, bars) -> Optional[IntradayPattern]:
    def detect_red_to_green(self, bars) -> Optional[IntradayPattern]:
    def detect_consolidation_box(self, bars) -> Optional[IntradayPattern]:

@dataclass
class IntradayPattern:
    pattern_type: str
    confidence: float     # 0-1
    trigger_price: float  # Prix de declenchement
    invalidation: float   # Prix d'invalidation (stop)
    target: float         # Cible estimee
```

---

### A11: Support/Resistance Dynamiques ✅ FAIT

**Fichier**: `src/levels_engine.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 1.5 jours
**Impact**: HAUT

Calcule les niveaux S/R en temps reel a partir des donnees IBKR:

```python
class LevelsEngine:
    def compute_levels(self, ticker) -> TechnicalLevels:
        """Calcule S/R dynamiques"""

@dataclass
class TechnicalLevels:
    ticker: str
    # Key levels
    vwap: float
    poc: float               # Point of Control (prix le plus echange)
    hod: float               # High of Day
    lod: float               # Low of Day
    premarket_high: float
    premarket_low: float
    previous_close: float
    previous_high: float

    # Dynamic S/R
    support_levels: List[float]    # Niveaux de support (ordonnes)
    resistance_levels: List[float] # Niveaux de resistance (ordonnes)

    # Distance au prochain niveau
    nearest_support: float
    nearest_resistance: float
    room_to_resistance_pct: float  # % de potentiel avant resistance
```

**Usage:** Un ticker avec beaucoup de "room to resistance" a plus de potentiel de hausse = boost Monster Score.

---

### A12: Social Velocity Engine ✅ FAIT

**Fichier**: `src/social_velocity.py` (nouveau, remplace social_buzz.py + social_buzz_engine.py)
**Statut**: IMPLEMENTE
**Effort**: 1.5 jours
**Impact**: MOYEN-HAUT

**Probleme actuel:** social_buzz.py compte les mentions statiquement. Ce qui compte c'est l'**acceleration** des mentions.

```python
class SocialVelocityEngine:
    def compute_velocity(self, ticker) -> SocialVelocity:
        """Mesure acceleration des mentions sociales"""

@dataclass
class SocialVelocity:
    ticker: str
    mention_count_1h: int      # Mentions derniere heure
    mention_count_4h: int      # Mentions 4 heures
    velocity: float            # 1st derivative (mentions/heure)
    acceleration: float        # 2nd derivative
    velocity_zscore: float     # Z-score vs baseline 7 jours
    sentiment_shift: float     # Changement sentiment (-1 a +1)
    top_keywords: List[str]    # Mots-cles dominants
    social_score: float        # 0-1 composite
```

**Integration:** Remplace le composant `social_buzz_score` (0.03 weight) dans Monster Score avec une mesure basee sur la velocite plutot que le volume brut.

---

### A13: Cross-Ticker Correlation (Sector Momentum) ✅ FAIT

**Fichier**: `src/sector_momentum.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 1 jour
**Impact**: MOYEN

Quand 2+ tickers du meme secteur bougent en meme temps, c'est un signal de sector catalyst:

```python
class SectorMomentum:
    def detect_sector_move(self) -> List[SectorSignal]:
        """Detecte mouvements sectoriels coordonnes"""

    def get_sector_boost(self, ticker) -> float:
        """Boost si le secteur du ticker est en mouvement"""

@dataclass
class SectorSignal:
    sector: str                # Biotech, Cannabis, EV, etc.
    tickers_moving: List[str]  # Tickers en mouvement
    avg_move_pct: float        # Mouvement moyen
    leader: str                # Ticker leader du mouvement
    confidence: float
```

---

## 6. SPRINT 4 — APPRENTISSAGE ET OPTIMISATION (Priorite MOYENNE)

### A14: Market Memory V2 ✅ FAIT

**Fichier**: `src/market_memory/context_scorer.py` (refactorer)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: HAUT

**Amelioration:** Segmenter MRP/EP par type de catalyst au lieu de traiter tous les signaux de maniere identique.

```python
# V1 (actuel): Un seul MRP/EP par ticker
mrp = get_mrp(ticker)  # 0.65

# V2: MRP/EP segmente par catalyst type
mrp_earnings = get_mrp(ticker, catalyst="EARNINGS")    # 0.80
mrp_fda = get_mrp(ticker, catalyst="FDA")              # 0.45
mrp_squeeze = get_mrp(ticker, catalyst="SQUEEZE")       # 0.70
```

Un ticker peut avoir un excellent historique sur les earnings mais mauvais sur FDA. Le systeme doit apprendre cette nuance.

---

### A15: Backtest Engine V8 ✅ FAIT

**Fichier**: `backtests/backtest_engine_v8.py` (nouveau)
**Statut**: IMPLEMENTE
**Effort**: 3 jours
**Impact**: CRITIQUE pour validation

Le backtest actuel (`backtest_engine_edge.py`) utilise le pipeline V7. Il faut un backtest qui valide:

```python
class BacktestEngineV8:
    def run(self, start_date, end_date) -> BacktestResult:
        """
        Backtest complet V8:
        1. Replay historical ticks into TickerStateBuffer
        2. Run AccelerationEngine on replayed data
        3. Verify ACCUMULATING states precede actual top gainers
        4. Measure lead time (minutes before breakout)
        5. Calculate hit rate, false positive rate, average lead time
        """

@dataclass
class BacktestResult:
    total_top_gainers: int        # Vrais top gainers dans la periode
    detected_before_move: int     # Detectes AVANT le move (ACCUMULATING)
    detected_during_move: int     # Detectes PENDANT (LAUNCHING/BREAKOUT)
    missed: int                   # Rates completement
    false_positives: int          # ACCUMULATING sans suite
    avg_lead_time_minutes: float  # Temps moyen avant le move
    hit_rate: float               # detected / total
    precision: float              # detected / (detected + false_positives)
```

---

### A16: Auto-Tuning Weights ✅ FAIT

**Fichier**: `src/scoring/weight_optimizer.py` (nouveau, remplace score_optimizer.py mort)
**Statut**: IMPLEMENTE
**Effort**: 2 jours
**Impact**: MOYEN-HAUT

Optimisation automatique des poids du Monster Score basee sur les resultats:

```python
class WeightOptimizer:
    def optimize_weekly(self) -> Dict[str, float]:
        """
        Optimise les 9 poids du Monster Score:
        - Maximise: detection rate des vrais top gainers
        - Minimise: faux positifs (score > 0.65 mais pas de move)
        - Methode: Bayesian optimization sur historique 30 jours
        - Contrainte: sum(weights) = 1.0, 0.01 <= weight <= 0.40
        """

    def get_current_weights(self) -> Dict[str, float]:
    def get_weight_history(self) -> List[Dict]:
```

---

### A17: Telegram Alerts Enrichis

**Fichier**: `alerts/telegram_alerts.py` (etendre)
**Effort**: 1 jour
**Impact**: MOYEN

Ajouter du contexte aux alertes Telegram:

```
Actuel:
  BUY_STRONG AAPL | Score: 0.85 | Entry: $150.00

V9 (enrichi):
  🚨 BUY_STRONG AAPL | Score: 0.85
  ├── Entry: $150.00 | Stop: $147.50 | Target: $157.50
  ├── Catalyst: EARNINGS BMO (beat 85% historique)
  ├── Float: 15.2M | SI: 18.5% | DTC: 3.2
  ├── Radar: LAUNCHING (vol z=3.2, price vel +0.5%/min)
  ├── Pattern: VWAP Reclaim + HOD Break
  ├── Room to resistance: 8.5% ($162.75)
  └── MRP: 0.78 | EP: 0.82 (earnings-specific)
```

---

## 7. SPRINT 5 — NETTOYAGE ET CONSOLIDATION (Priorite MOYENNE)

### A18: Supprimer modules morts (zero imports)

| Module | Lignes | Action |
|--------|--------|--------|
| `src/dark_pool_alternatives.py` | ~200 | SUPPRIMER |
| `src/social_engine/news_buzz.py` | ~80 | SUPPRIMER |
| `src/scoring/score_optimizer.py` | ~150 | SUPPRIMER (remplace par A16) |
| `src/options_flow.py` | ~300 | SUPPRIMER (remplace par options_flow_ibkr.py) |

**Effort**: 0.5 jour
**Risque**: ZERO (aucun import dans le codebase)

---

### A19: Migrer modules deprecated

| Module legacy | Remplace par | Importe dans |
|--------------|-------------|-------------|
| `src/signal_engine.py` | `engines/signal_producer.py` | main.py, backtest, tests |
| `src/ensemble_engine.py` | `scoring/monster_score.py` | main.py, tests |
| `src/portfolio_engine.py` | `engines/order_computer.py` | main.py, backtest, tests |

**Effort**: 1 jour (modifier les imports dans main.py, backtest, tests)
**Risque**: FAIBLE (les modules V7 sont deja les vrais executeurs)

---

### A20: Fusionner doublons

| Doublon A | Doublon B | Action |
|-----------|-----------|--------|
| `src/social_buzz.py` | `src/ingestors/social_buzz_engine.py` | Fusionner en `src/social_velocity.py` (A12) |
| `src/social_engine/grok_sentiment.py` | Violation "Grok = classification only" | SUPPRIMER ou refactorer |

**Effort**: 1 jour

---

### A21: Tests unitaires complets

**Fichier**: `tests/` (etendre)
**Effort**: 2 jours

Tests a ajouter:
- `test_ibkr_streaming.py` — Mock ib_insync, test event processing
- `test_acceleration_engine.py` — Test ACCUMULATING detection
- `test_earnings_calendar.py` — Mock API, test scoring
- `test_fda_calendar.py` — Mock OpenFDA, test tiers
- `test_float_analysis.py` — Test squeeze detection
- `test_monster_score_v4.py` — Test poids et boosts
- `test_signal_producer_v8.py` — Test pipeline complet avec mocks

---

## 8. MATRICE IMPACT/EFFORT

```
IMPACT ▲
       │
  CRIT │  A1✅  A2   A4   A9        A15
       │
  HAUT │  A3   A5   A6   A10  A11   A14
       │
  MOY  │  A7   A8   A12  A13  A16   A17
       │
  BAS  │  A18  A19  A20  A21
       │
       └──────────────────────────────────→ EFFORT
          0.5j  1j   1.5j  2j   3j
```

### Ordre de priorite recommande:

| Rang | ID | Amelioration | Effort | Impact | ROI |
|------|-----|-------------|--------|--------|-----|
| 1 | A2 | Integration streaming pipeline | 1j | CRITIQUE | MAX |
| 2 | A4 | Earnings Calendar Engine | 2j | CRITIQUE | TRES HAUT |
| 3 | A9 | Float Turnover & Squeeze Score | 2j | CRITIQUE | TRES HAUT |
| 4 | A5 | FDA Calendar Complet | 2j | HAUT | HAUT |
| 5 | A10 | Patterns Intraday Avances | 2j | HAUT | HAUT |
| 6 | A3 | Tick-by-tick hot tickers | 0.5j | HAUT | HAUT |
| 7 | A6 | Conference Calendar | 1.5j | HAUT | HAUT |
| 8 | A11 | Support/Resistance dynamiques | 1.5j | HAUT | MOYEN |
| 9 | A12 | Social Velocity Engine | 1.5j | MOYEN-HAUT | MOYEN |
| 10 | A14 | Market Memory V2 | 2j | HAUT | MOYEN |
| 11 | A18-20 | Nettoyage modules | 2.5j | MOYEN | HAUT |
| 12 | A7 | IPO/SPO Tracker | 1j | MOYEN | MOYEN |
| 13 | A15 | Backtest Engine V8 | 3j | CRITIQUE | MOYEN |
| 14 | A16 | Auto-Tuning Weights | 2j | MOYEN-HAUT | MOYEN |
| 15 | A17 | Telegram Enrichis | 1j | MOYEN | MOYEN |
| 16 | A8 | SEC Insider Clustering | 0.5j | MOYEN | MOYEN |
| 17 | A13 | Cross-Ticker Correlation | 1j | MOYEN | BAS |
| 18 | A21 | Tests complets | 2j | MOYEN | MOYEN |

---

## 9. METRIQUES DE SUCCES

### 9.1 KPIs principaux

| Metrique | Actuel (V8) | Cible V9 | Methode de mesure |
|----------|-------------|----------|-------------------|
| **Detection Rate** | ~65-70% | **95%+** | % top gainers detectes (daily_audit.py) |
| **Lead Time** | ~3-8 min | **10-20 min** | Temps moyen avant +5% move |
| **Precision** | ~40-50% | **65%+** | % signaux qui menent a un move reel |
| **Latence L1** | 2000ms | **<50ms** | Temps entre tick IBKR et detection |
| **Couverture catalysts** | ~60% | **90%+** | % des catalyst types couverts |
| **Faux positifs BUY** | ~30% | **<15%** | % BUY/BUY_STRONG sans move |

### 9.2 KPIs secondaires

| Metrique | Actuel | Cible V9 |
|----------|--------|----------|
| Uptime IBKR | 95% | 99%+ |
| Tickers surveilles temps reel | ~50 (poll) | 200 (streaming) |
| Cycle complet pipeline | 180s | 5s (streaming-driven) |
| Earnings coverage | 0% | 95%+ |
| FDA coverage | 30% | 90%+ |
| Float data coverage | 0% | 80%+ |

### 9.3 Validation

Chaque amelioration sera validee par:
1. **daily_audit.py** — Hit rate / lead time quotidien
2. **weekly_deep_audit.py** — Analyse profonde hebdomadaire
3. **Backtest V8** (A15) — Validation historique
4. **Walk-forward** (validation/) — Robustesse out-of-sample

---

## 10. RISQUES ET MITIGATIONS

| Risque | Probabilite | Impact | Mitigation |
|--------|------------|--------|------------|
| IBKR streaming instable | MOYEN | HAUT | Fallback automatique vers poll mode |
| Rate limits API (Finnhub, SEC) | HAUT | MOYEN | Pool API multi-cles + cache agressif |
| Donnees earnings incorrectes | MOYEN | HAUT | Cross-validation 2+ sources |
| Faux positifs ACCUMULATING | HAUT | MOYEN | Seuils adaptatifs + Market Memory V2 |
| Surcharge CPU/RAM (Hetzner) | FAIBLE | HAUT | Monitoring via system_guardian.py |
| IBKR max subscriptions depassees | MOYEN | MOYEN | Eviction stale + priorite HOT |
| Overfitting weights | MOYEN | MOYEN | Walk-forward validation + contraintes |

---

## RESUME EXECUTIF

**GV2-EDGE V9** vise a passer de **~65-70%** a **95%+** de detection des top gainers small-cap US, avec un lead time de **10-20 minutes** avant le mouvement.

Les 5 ameliorations les plus impactantes:
1. **IBKR Streaming** (A1-A3): Latence 2000ms → <50ms
2. **Earnings Calendar** (A4): +15-20% de couverture catalysts
3. **Float/Squeeze Analysis** (A9): Detection squeezes en formation
4. **FDA Calendar complet** (A5): +8-12% de couverture catalysts
5. **Patterns Intraday** (A10): Timing d'entree precis

**Effort total estime**: ~30 jours de developpement
**Budget API additionnel**: $0 (toutes les sources sont gratuites)
**Risque de regression**: FAIBLE (architecture additive, detection jamais bloquee)
