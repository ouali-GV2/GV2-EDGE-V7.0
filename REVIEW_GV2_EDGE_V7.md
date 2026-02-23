# GV2-EDGE V7.0 - Revue Approfondie Complète

> **Date**: 2026-02-17
> **Scope**: 78 fichiers source analysés (5 axes d'analyse parallèles)
> **Verdict global**: Système RÉACTIF avec capacité d'apprentissage. Adapté au swing trading catalyst-driven sur 50-200 tickers. NON adapté au scalping intraday ou à un univers de 2000+ tickers.

---

## Table des Matières

1. [Architecture Globale (3 Couches)](#1-architecture-globale)
2. [Monster Score & Scoring](#2-monster-score--scoring)
3. [Modules de Détection](#3-modules-de-détection)
4. [Composants Core (PM, Social, Market Memory)](#4-composants-core)
5. [Options Flow & Risk Guard](#5-options-flow--risk-guard)
6. [Performance & Scalabilité](#6-performance--scalabilité)
7. [Problèmes Critiques Identifiés](#7-problèmes-critiques)
8. [Matrice des Scores par Module](#8-matrice-des-scores)
9. [Conclusions & Recommandations](#9-conclusions)

---

## 1. Architecture Globale

### Design 3 Couches (Clean Separation)

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: SIGNAL PRODUCER (signal_producer.py)           │
│ - Input: monster_score, pre_spike, catalyst, social     │
│ - Output: UnifiedSignal (BUY_STRONG/BUY/WATCH/EARLY)   │
│ - Action: NE BLOQUE JAMAIS, détection uniquement        │
│ - Seuils: 0.65 (BUY), 0.80 (BUY_STRONG)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ (si actionnable)
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: ORDER COMPUTER (order_computer.py)             │
│ - Input: UnifiedSignal, contexte marché                 │
│ - Output: ProposedOrder (attaché au signal)             │
│ - Action: CALCULE TOUJOURS l'ordre théorique            │
│ - Sizing: Risk-based, ATR-adjusted, multi-facteur       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: EXECUTION GATE (execution_gate.py)             │
│ - Input: UnifiedSignal + ProposedOrder                  │
│ - Output: ExecutionDecision                             │
│ - Action: APPLIQUE LES LIMITES UNIQUEMENT               │
│ - 9 vérifications séquentielles, 11 raisons de blocage  │
└─────────────────────────────────────────────────────────┘
```

**Forces**: Séparation propre, signal toujours préservé, bon risk management dans le sizing.

**Faiblesses**: 250-400ms de latence minimum, réactif (pas anticipatif), cache 30s tue les opportunités sub-minute.

---

## 2. Monster Score & Scoring

### Structure des Poids

| Composant | Poids | Contribution Réelle |
|-----------|-------|---------------------|
| Event (Catalyst) | 0.25 | HAUTE - les catalystes drivent les mouvements |
| Volume | 0.17 | HAUTE - confirme la demande |
| Pattern | 0.17 | MOYENNE - dépend du pattern_analyzer |
| PM Transition | 0.13 | MOYENNE - biais pre-market uniquement |
| Options Flow | 0.10 | HAUTE (SI disponible) - forward-looking |
| Momentum | 0.08 | MOYENNE - indicateur retardé |
| Social Buzz | 0.06 | BASSE-MOYENNE - sentiment uniquement |
| Squeeze | 0.04 | BASSE - occurrence rare |

### Verdict: Accélération/Vélocité

**ZÉRO ACCÉLÉRATION/VÉLOCITÉ IMPLÉMENTÉE**

```python
# Ce qui existe (monster_score.py:148-150):
momentum = normalize(abs(feats["momentum"]), 0.2)  # Valeur ABSOLUE
volume = normalize(feats["volume_spike"], 5)        # Seuil FIXE

# Ce qui DEVRAIT exister:
current_mom = (price[-1] - price[-5]) / price[-5]
prev_mom = (price[-10] - price[-15]) / price[-15]
acceleration = current_mom - prev_mom  # Dérivée seconde
```

**Impact**: Le système détecte les actions qui BOUGENT DÉJÀ, pas celles qui VONT bouger.

### Seuils Absolus vs Anomalies Relatives

- **Momentum > 20%** = fort → FIXE (pas adaptatif au titre)
- **Volume > 5x** = spike → FIXE
- **Squeeze > 10** = extrême → FIXE
- **Aucun z-score**, aucun percentile, aucune comparaison sectorielle
- Un mouvement de 20% sur un biotech à $0.50 est traité identiquement à 20% sur AAPL à $500

### Score Optimizer

- Learning rate conservateur (5%), max change ±10%
- Nécessite `performance_attribution.json` (feedback loop non visible)
- Ignore les interactions entre composants (event + volume corrélés)
- Pas de détection de régime (bull/bear)

---

## 3. Modules de Détection

### 3.1 Anticipation Engine

**Question clé**: Détecte-t-il AVANT +5% ou seulement APRÈS les mouvements visibles ?

**RÉPONSE: UNIQUEMENT APRÈS**

- `gap_pct = (last - close) / close` → analyse réactive de quotes
- Volume spike détecté quand DÉJÀ en spike (>3x)
- PM upgrade vérifie des gaps PM déjà formés
- **Aucun indicateur forward-looking**

**Latence**: 5-25 secondes par scan complet (Finnhub API rate limit)

### 3.2 Pre-Spike Radar

**Question**: Mesure-t-il la Float Velocity ?

**RÉPONSE: NON**

Ce qu'il mesure réellement:
- Volume acceleration (dérivée 2nde du volume) ✓
- Options momentum (ratio calls) ✓
- Buzz acceleration (taux de croissance mentions) ✓
- Compression technique (Bollinger bandwidth) ✓

Ce qu'il NE mesure PAS:
- Float velocity (données microstructure nécessaires)
- Flux d'ordres institutionnels
- Distinction retail vs institutionnel vs algo

**"75% spike probability"** → NON VALIDÉ, aucun backtesting montré.

### 3.3 Catalyst Score V3

- Boost range: 1.0x à 1.6x sur Monster Score
- Decay temporel avec plancher à 5% (anciens catalystes ne meurent jamais)
- Boost de proximité pour événements FUTURS uniquement
- **Limitation**: Poids statiques, pas adaptatifs au régime de marché

### 3.4 FDA Calendar

| Source | Réel ? | Fiabilité |
|--------|--------|-----------|
| PDUFA dates (BiopharmCatalyst) | 95% réel (scraping) | Fragile (dépend du HTML) |
| Trial results | 70% réel (dates fuzzy) | "Q1 2025" → 15 Mars |
| Conferences | 0% réel (hardcodé statique) | Ne se met jamais à jour |

**Risque**: Source unique → si BiopharmCatalyst down, aucun fallback.

### 3.5 News Flow Screener

- Sources 100% réelles: SEC EDGAR 8-K + Finnhub
- Pas de simulation Polygon-via-Grok
- **Latence**: 2-5 secondes (Grok rate limit 300/hr)
- Détecte les 8-K ~30 min après filing
- News typiquement 1-4h d'ancienneté

### 3.6 NLP Enrichi

- Sentiment boost: 0.7x à 1.4x
- Keyword-based fragile: "risk" marqué bearish même dans "risk-free returns"
- Grok sentiment non validé (aucun backtesting)
- Cross-reference social/analyst marqué "TODO"

### 3.7 After-Hours Scanner

- RÉACTIF UNIQUEMENT: détecte les earnings APRÈS publication
- Utile pour entrée next-day, pas pour anticipation
- Impact = `min(1.0, abs(beat_pct))` → crude

### Latence Totale Stack de Détection

| Module | Latence | Bottleneck |
|--------|---------|------------|
| Catalyst Score V3 | 50-100ms | SQLite queries |
| Anticipation Engine | 5-25s | Finnhub rate limit |
| Pre-Spike Radar | 1-2s | Données externes |
| FDA Calendar | 15-20s (1er run), 0ms (cache) | Web scraping |
| News Flow Screener | 2-5s | Grok rate limit (300/hr) |
| NLP Enrichi | 100-200ms | Grok API |
| Watch List | <100ms | Traitement événements |
| After-Hours Scanner | <1s | API calls |
| **TOTAL FULL STACK** | **30-60 secondes** | Grok + Finnhub |

---

## 4. Composants Core

### 4.1 PM Scanner - PROBLÈME CRITIQUE

**Le calcul du gap est FONDAMENTALEMENT CASSÉ**:

```python
# Ce qui est implémenté (pm_scanner.py:113):
gap_pct = (last - pm_open) / pm_open  # Calcule momentum intra-PM

# Ce qui DEVRAIT être:
gap_pct = (pm_high - yesterday_close) / yesterday_close  # Vrai gap overnight
```

- `prev_close` n'est PAS utilisé malgré sa disponibilité dans la réponse Finnhub
- Aucune priorisation des gaps modérés (3-8%)
- Aucune pénalisation des gaps extrêmes (>10%, >20%)

### 4.2 PM Transition - Confirmation, Pas Prédiction

| Question | Constat |
|----------|---------|
| Détecte AVANT breakout ? | NON - Analyse les 20 dernières bougies |
| Détecte AU début du breakout ? | NON - Fakeout detection regarde les breaks déjà échoués |
| Analyse intra-minute réelle ? | MINIMAL - Pattern matching OHLCV uniquement |
| Valeur prédictive | RÉACTIVE - Données rétrospectives |

### 4.3 Extended Hours Quotes

- Gap calculation CORRECT (`prev_close` utilisé) ✓
- Volume EH conflate avec volume total ✗
- Boost gap: linéaire 1.5x, cap à +0.15 (pas de sweet-spot)

### 4.4 Analyse Sociale - 80% PLACEHOLDER

| Composant | Vrai NLP ? | Données Temps Réel ? | Niveau Placeholder |
|-----------|------------|---------------------|--------------------|
| Grok Twitter | NON (prompt estimation) | NON (training data cutoff) | ÉLEVÉ |
| Reddit | NON (keyword matching) | OUI (PRAW temps réel) | MOYEN |
| StockTwits | PARTIEL (labels natifs) | OUI (API temps réel) | BAS |
| Grok Sentiment | NON (prompt analysis) | NON (training data) | ÉLEVÉ |

**Grok ne peut PAS accéder aux données Twitter/X en temps réel** - il ESTIME les comptages de mentions basé sur ses données d'entraînement.

**Reddit**: Simple keyword matching (`"buy", "calls", "moon", "rocket"` = bullish). Aucune:
- Analyse de force de sentiment
- Compréhension du contexte
- Gestion de la négation ("don't buy")
- Détection du sarcasme

### 4.5 Market Memory

**Exigences d'activation (TOUTES requises)**:
1. 50+ signaux manqués tracés
2. 30+ trades enregistrés
3. 10+ patterns appris
4. 20+ profils ticker

**Période de warm-up: 1-2 SEMAINES minimum**

- MRP (Missed Recovery Potential): Utile - "a-t-on bien fait de bloquer ?"
- EP (Edge Probability): Modérément utile - probabilité basée sur patterns
- Les deux sont INFORMATIONNELS UNIQUEMENT - ne bloquent/modifient pas l'exécution

### 4.6 Memory Store

- Backend JSON (défaut): Charge complète en mémoire à chaque save
- Backend SQLite: Query load ALL puis filtre en mémoire (défait l'efficacité DB pour >100k records)
- Pas de contrôle d'accès concurrent sur JSON

---

## 5. Options Flow & Risk Guard

### 5.1 Options Flow - DÉCORATIF

**Status: Placeholder fonctionnel, pas structural**

- `options_flow.py`: Retourne score neutre 0.5 par défaut
- `options_flow_ibkr.py`: Plus développé mais seuils ABSOLUS uniquement
- Poids: 10% du monster_score (même que social_buzz)
- **NE déclenche JAMAIS** de signal BUY/BUY_STRONG indépendamment
- Seuils fixes: 5K calls = signal, identique pour penny stock à $0.50 et mega-cap à $150

**Manquant**:
- Comparaison historique (baseline 20j/30j)
- Seuils ajustés à la volatilité
- Détection d'anomalie relative

### 5.2 Risk Guard - SUR-BLOCAGE SYSTÉMATIQUE

#### Matrice de Risque

| Niveau | Multiplicateur Position | Blocage ? |
|--------|------------------------|-----------|
| CRITICAL | 0.0 | OUI |
| HIGH | 0.25 | NON (mais -75%) |
| ELEVATED | 0.50 | NON |
| MODERATE | 1.0 | NON |
| LOW | 1.0 | NON |

#### Problème 1: Blocage binaire penny stocks

**TOUT penny stock sous $1 pendant 30 jours = BLOQUÉ**, même si les fondamentaux s'améliorent.
- Bid price deficiency → CRITICAL → 0x multiplier
- Delisting risk → 0x
- Toxic financing → 0x

#### Problème 2: Confusion dilution potentielle vs active

```python
# Détecte un S-3 shelf registration (CAPACITÉ de lever des fonds)
# Le traite comme risque de dilution ACTUEL
# Bloque le trading pendant 90 jours

# Exemple: Entreprise dépose S-3 le 1er Jan
# Février: Stock fait un beat massif, remonte à $5
# SYSTÈME BLOQUE TOUJOURS car S-3 est "récent" (<90 jours)
```

Pas de distinction entre:
- Secondary offering annoncé (risque réel)
- S-3 sur étagère (risque potentiel)
- ATM actif (risque quotidien)

#### Problème 3: Multiplication catastrophique

Avec `apply_combined_multipliers = True` (défaut):

```
Penny stock avec:
- Bid price deficiency: 0.5x
- S-3 filing récent: 0.5x
- ATM élevé: 0.25x
→ Résultat: 0.5 × 0.5 × 0.25 = 0.0625 (~94% réduction)
→ Effectivement BLOQUÉ
```

#### Problème 4: Biais anti top-gainers small-cap

Un penny stock en hausse de 40% sur 10x volume:
```
halt_prob = 50 (LULD) + 25 (volume) + 20 (low float) + 10 (spread) = 105% → cap 100%
→ Position multiplier = 0.0 (BLOQUÉ)
```

**Résultat**: Le système bloque EXACTEMENT les stocks qu'un trader small-cap voudrait trader.

---

## 6. Performance & Scalabilité

### Async Réel vs Fake

**Les opérations "async" sont PARTIELLEMENT FAUSSES**:
- `batch_scheduler.py`: `asyncio.sleep(1)` dans une boucle séquentielle
- Semaphores limitent à 5-10 requêtes simultanées (pas 2000-3000)
- Social buzz: Traitement séquentiel avec 0.5s sleep entre tickers

### Limites Artificielles sur l'Univers

1. **Batch processor**: `tickers[:50]` → Limite dure de 50 tickers pour company news
2. **Buzz engine**: `list(self.universe)[:50]` → 50 tickers seulement pour social buzz
3. **Scan scheduler**: 3 tickers par cycle → univers complet de 3000 tickers en 8+ heures
4. **Rotation complète**: 19-45 minutes pour un scan complet de l'univers

### Estimation de Latence pour 2000-3000 Tickers

- Company news: 1 req/sec × 2500 tickers = **41 minutes**
- Hot ticker cycle: 10 tickers à la fois → 2500 ÷ 10 = **250 minutes (4+ heures)**

**VERDICT**: Le système est optimisé pour 50-200 hot tickers, PAS pour un univers de 2000-3000.

### Rate Limiting Bottleneck

| Composant | Rate Limit | Concurrence | Throughput Effectif |
|-----------|-----------|-------------|---------------------|
| Finnhub | 60 req/min | 5 (semaphore) | ~300 req/min |
| SEC Edgar | 10 req/sec | Illimité | 10 req/sec |
| Grok NLP | 10 req/min free | 5 | ~10 req/min |
| Reddit | 60 req/min | Séquentiel | 60 req/min |

**Bottleneck principal**: Grok API à 10 req/min → ne peut classifier que 10 items/min.

### Composants Manquants

| Feature | Status | Impact |
|---------|--------|--------|
| Candlestick data fetch | **MANQUANT** | Pattern analyzer non-fonctionnel |
| PM/AH price fetcher | **MANQUANT** | Ne peut détecter les PM spikes |
| IPO calendar | **MANQUANT** | Aveugle aux nouvelles cotations |
| Halt detection | **MANQUANT** | Risque d'achat de stocks haltés |
| Options flow analysis | **MANQUANT** | Pas de détection smart money |
| VWAP/profile analysis | **MANQUANT** | Volume profile incomplet |
| Sector rotation | **MANQUANT** | Ne peut exploiter les thèmes sectoriels |

### API Pool Manager - Non Utilisé

`api_pool/` contient un pool manager complet (key_registry, request_router, pool_manager) mais **jamais appelé dans les ingestors**. Chaque ingestor hardcode sa propre clé API.

---

## 7. Problèmes Critiques

### Severity CRITICAL

1. **Gap calculation cassé** dans `pm_scanner.py` - calcule momentum intra-PM au lieu du gap overnight réel. Nourrit des métriques fausses en amont.

2. **Analyse sociale ~80% placeholder** - Grok ne peut pas accéder aux données Twitter/X en temps réel. Reddit = keyword matching sans NLP.

3. **Risk Guard sur-bloque les top gainers small-cap** - Multiplicateurs cumulatifs + seuils fixes = blocage automatique des penny stocks en breakout.

4. **Confusion dilution potentielle/active** - Un simple S-3 shelf = blocage 90 jours sans distinction avec une offre réelle.

5. **Système fondamentalement RÉACTIF** - Zéro tracking d'accélération/vélocité. Détecte APRÈS les mouvements, pas avant.

### Severity HIGH

6. **Scalabilité limitée à 50-200 tickers** - Limites hardcodées, async partiellement fake, 4+ heures pour scanner 3000 tickers.

7. **Pattern analyzer scaffolding** - Structure en place mais pas de source de données candlestick.

8. **Market Memory nécessite 1-2 semaines de warm-up** avant d'être utile (50+ misses, 30+ trades requis).

9. **Seuils absolus fixes** - Pas de normalisation par volatilité, cap de marché, ou régime.

10. **Single provider dependency** - Si Finnhub tombe, le pipeline entier stagne.

### Severity MODERATE

11. **Cache 30 secondes** trop agressif pour la détection intraday sub-minute.

12. **Conferences FDA hardcodées** - Liste statique qui ne se met jamais à jour.

13. **Ensemble engine trop simple** - Multiplicateur de confiance qui peut gonfler artificiellement des signaux faibles.

14. **`signal_engine.py` obsolète** - Module legacy qui devrait être déprécié en faveur de `signal_producer.py`.

15. **SQLite backend** charge tous les records en mémoire avant de filtrer (pas de WHERE clause).

---

## 8. Matrice des Scores

| Module | Architecture | Fonctionnalité | Fiabilité | Production-Ready |
|--------|-------------|----------------|-----------|------------------|
| Monster Score | 8/10 | 5/10 | 6/10 | 5/10 |
| Signal Producer (L1) | 9/10 | 7/10 | 7/10 | 7/10 |
| Order Computer (L2) | 8/10 | 7/10 | 7/10 | 7/10 |
| Execution Gate (L3) | 8/10 | 6/10 | 7/10 | 6/10 |
| PM Scanner | 6/10 | 3/10 | 4/10 | 3/10 |
| PM Transition | 7/10 | 5/10 | 5/10 | 4/10 |
| Anticipation Engine | 7/10 | 4/10 | 5/10 | 4/10 |
| Pre-Spike Radar | 7/10 | 5/10 | 4/10 | 3/10 |
| Social Buzz | 5/10 | 2/10 | 3/10 | 2/10 |
| Grok Sentiment | 5/10 | 2/10 | 3/10 | 2/10 |
| News Flow Screener | 8/10 | 7/10 | 6/10 | 6/10 |
| NLP Enrichi | 6/10 | 4/10 | 4/10 | 4/10 |
| FDA Calendar | 6/10 | 6/10 | 4/10 | 4/10 |
| Options Flow | 5/10 | 2/10 | 3/10 | 2/10 |
| Risk Guard (Unified) | 7/10 | 5/10 | 5/10 | 4/10 |
| Dilution Detector | 7/10 | 4/10 | 4/10 | 3/10 |
| Market Memory | 8/10 | 6/10 | 5/10 | 4/10 |
| Feature Engine | 7/10 | 6/10 | 6/10 | 5/10 |
| Async/Scalabilité | 2/10 | 1/10 | 3/10 | 2/10 |

**Score Global**: **Architecture 7/10, Fonctionnalité 4/10, Production 4/10**

---

## 9. Conclusions

### Ce que le système fait BIEN

1. Architecture 3 couches propre avec séparation des responsabilités
2. Le signal est toujours préservé à travers toutes les couches
3. Bon risk management dans le sizing (ATR-based, multi-facteur)
4. Intégration multi-sources de données (IBKR, Finnhub, Reddit, StockTwits)
5. Système d'apprentissage (MissedTracker + PatternLearner) pour amélioration continue
6. News Flow Screener avec sources 100% réelles (SEC EDGAR + Finnhub)
7. Scoring catalyst bien structuré avec decay temporel

### Ce que le système ne fait PAS

1. **Prédire les spikes avant qu'ils commencent**
2. **Mesurer la float velocity** (pas de données microstructure)
3. **Anticiper les earnings surprises**
4. **Analyser le flux d'options en temps réel**
5. **Détecter l'accumulation institutionnelle**
6. **S'adapter aux régimes de marché** (bull/bear/consolidation)
7. **Scaler à 2000+ tickers** en temps raisonnable

### Profil d'Utilisation Recommandé

Le système est adapté pour:
- **Swing trading catalyst-driven** (timeframe journalier)
- **50-200 hot tickers** surveillés activement
- **Entrées basées sur des événements connus** (FDA, earnings)
- **Confirmation de mouvements** (pas anticipation)

Le système N'EST PAS adapté pour:
- Scalping intraday (latence trop élevée)
- Trading momentum pur (réactif, pas anticipatif)
- Univers large (>500 tickers simultanés)
- Penny stocks en breakout (sur-blocage systématique)
- Arbitrage sentiment temps réel (social analysis placeholder)

---

## 10. STATUT DE CORRECTION (2026-02-21)

> **Note** : Cette revue a ete realisee sur V7.0. Les problemes identifies ont ete corriges dans V8/V9.

### Problemes critiques — CORRIGES

| # | Probleme V7 | Correction | Version |
|---|------------|------------|---------|
| 1 | Gap calculation casse (pm_scanner.py) | prev_close utilise, gap zones implementees | V8 |
| 2 | Analyse sociale ~80% placeholder | Social Velocity Engine + NLP ameliore | V9 |
| 3 | Risk Guard sur-bloque les top gainers | MIN-based + momentum override + planchers | V8 |
| 4 | Confusion dilution potentielle/active | 4 tiers : ACTIVE/SHELF_RECENT/SHELF_DORMANT/CAPACITY | V8 |
| 5 | Systeme fondamentalement REACTIF | AccelerationEngine + SmallCapRadar + Multi-Radar V9 | V8/V9 |

### Problemes haute severite — CORRIGES

| # | Probleme V7 | Correction | Version |
|---|------------|------------|---------|
| 6 | Scalabilite limitee 50-200 tickers | IBKR Streaming (200 subs) + Finnhub WS (illimite) | V9 |
| 7 | Pattern analyzer scaffolding | Patterns intraday avances (VWAP, ORB, HoD) | V9 |
| 8 | Market Memory warm-up 1-2 semaines | Market Memory V2 segmente par catalyst type | V9 |
| 9 | Seuils absolus fixes | Z-scores adaptatifs + normalisation par baseline 20j | V8 |
| 10 | Single provider dependency | Fallback IBKR → Finnhub WS → Finnhub REST → Cache | V8/V9 |

### Score apres corrections V9

| Module | V7 | V9 (estime) |
|--------|-----|-------------|
| Architecture | 7/10 | 9/10 |
| Fonctionnalite | 4/10 | 8/10 |
| Production-Ready | 4/10 | 7/10 |

Voir `PLAN_TRANSFORMATION_V8.md`, `PLAN_CORRECTION_COVERAGE.md`, et `PLAN_AMELIORATION_V9.md` pour les details complets.
