# GV2-EDGE V6.0 - Trader Guide

## Objectif

GV2-EDGE detecte les top gainers small caps US **AVANT** leurs hausses majeures (+50% a +500%).

**Cible** : Small caps < $2B market cap, hors OTC

---

## Nouveautes V6.0

### 5 Couches d'Anticipation

| Couche | Module | Impact Trading |
|--------|--------|----------------|
| 1 | Market Calendar | Evite faux signaux jours feries |
| 2 | Repeat Gainer | Badge "serial runner" = sizing adapte |
| 3 | Pre-Spike Radar | Detection acceleration avant spike |
| 4 | Catalyst Score V3 | Scoring par tier (FDA > Earnings) |
| 5 | NLP Enrichi | Sentiment + urgence temps reel |

### EVENT_TYPE Tiers

```
TIER 1 - CRITICAL (impact 0.90-1.00):
  FDA_APPROVAL, PDUFA_DECISION, BUYOUT_CONFIRMED
  -> Action: Entry immediate, sizing max

TIER 2 - HIGH (impact 0.75-0.89):
  FDA_TRIAL_POSITIVE, BREAKTHROUGH_DESIGNATION, FDA_FAST_TRACK,
  MERGER_ACQUISITION, EARNINGS_BEAT_BIG, MAJOR_CONTRACT
  -> Action: Entry rapide, sizing standard+

TIER 3 - MODERATE (impact 0.60-0.74):
  GUIDANCE_RAISE, EARNINGS_BEAT, PARTNERSHIP, PRICE_TARGET_RAISE
  -> Action: Attendre confirmation PM

TIER 4 - LOW-MOD (impact 0.45-0.59):
  ANALYST_UPGRADE, SHORT_SQUEEZE_SIGNAL, UNUSUAL_VOLUME_NEWS
  -> Action: Watchlist seulement

TIER 5 - SPECULATIVE (impact 0.30-0.44):
  BUYOUT_RUMOR, SOCIAL_MEDIA_SURGE, BREAKING_POSITIVE
  -> Action: Prudence, rumors non confirmees
```

---

## Signaux V6

### WATCH_EARLY

- **Quand**: Catalyst detecte en after-hours/pre-market
- **Signification**: Potentiel en formation
- **V6 Features**: Pre-Spike Radar level, NLP sentiment
- **Action**: Surveiller, preparer entry
- **Sizing**: Aucun (attendre upgrade)

### BUY

- **Quand**: Score 0.65-0.79 + confirmation technique
- **Signification**: Setup solide
- **V6 Features**: Catalyst tier afiche, Repeat badge si applicable
- **Action**: Entry standard
- **Sizing**: 2% risk

### BUY_STRONG

- **Quand**: Score 0.80+ + catalyst TIER 1-2
- **Signification**: Opportunite majeure
- **V6 Features**: Full V6 intelligence display
- **Action**: Entry immediate
- **Sizing**: 3% risk max

---

## Alertes Telegram V6

### Format Signal V6

```
[SIGNAL_EMOJI] GV2-EDGE V6.0 SIGNAL

Ticker: NVDA
Signal: BUY_STRONG
Monster Score: 0.85
Confidence: 0.92

--- V6 Intelligence ---
[EVENT_EMOJI] Event: FDA_APPROVAL
TIER 1 - CRITICAL (impact: 0.95)
Catalyst Score V3: 0.88
NLP Sentiment: VERY_BULLISH
Pre-Spike Radar: 3/4 signals
REPEAT GAINER (4 past spikes)

--- Position ---
Entry: $152.50
Stop: $148.20
Shares: 45
Risk: $193.50
```

### Alert Pre-Spike Radar

```
PRE-SPIKE RADAR ALERT

Ticker: BIOX
HIGH (3/4 signals)

Active Signals:
[CHECK] Volume Acceleration
[CHECK] Bid-Ask Tightening
[CHECK] Price Compression
[X] Dark Pool Activity

Acceleration Score: 0.78
Monster Score: 0.72

ACTION: Monitor closely for entry
```

### Alert Repeat Gainer

```
REPEAT GAINER DETECTED

Ticker: MARA
SERIAL RUNNER

Historical Spikes: 7
Avg Spike: +65.2%
Last Spike: 2026-01-15
Volatility Score: 0.82

Current Monster Score: 0.75

WARNING: Known for explosive moves - size appropriately
```

---

## Strategie par Tier

### TIER 1 (FDA_APPROVAL, BUYOUT_CONFIRMED)

1. **Entry**: Immediate sur alerte
2. **Sizing**: Max (3% risk)
3. **Stop**: Large (volatilite FDA)
4. **Target**: +50% minimum
5. **Timing**: Market order OK

### TIER 2 (FDA_TRIAL_POSITIVE, EARNINGS_BEAT_BIG)

1. **Entry**: PM open ou early RTH
2. **Sizing**: Standard+ (2.5% risk)
3. **Stop**: ATR-based
4. **Target**: +30-50%
5. **Timing**: Limit preferred

### TIER 3 (EARNINGS_BEAT, PARTNERSHIP)

1. **Entry**: Attendre confirmation PM
2. **Sizing**: Standard (2% risk)
3. **Stop**: Tight
4. **Target**: +20-30%
5. **Timing**: PM confirmation required

### TIER 4-5 (ANALYST_UPGRADE, RUMORS)

1. **Entry**: Watchlist only
2. **Sizing**: Reduit si entry
3. **Stop**: Tres tight
4. **Target**: +10-20%
5. **Timing**: Wait for upgrade to TIER 3+

---

## Pre-Spike Radar Interpretation

| Level | Signals | Signification | Action |
|-------|---------|---------------|--------|
| NONE | 0/4 | Pas d'acceleration | Ignorer |
| WATCH | 1/4 | Debut d'activite | Surveiller |
| ELEVATED | 2/4 | Acceleration probable | Preparer entry |
| HIGH | 3-4/4 | Spike imminent | Entry aggressive |

### Signaux Pre-Spike

| Signal | Interpretation |
|--------|----------------|
| Volume Acceleration | Smart money accumulating |
| Bid-Ask Tightening | Liquidity providers positioning |
| Price Compression | Volatility squeeze before breakout |
| Dark Pool Activity | Institutional interest |

---

## Repeat Gainer Badges

| Badge | Spikes | Sizing |
|-------|--------|--------|
| KNOWN MOVER | 2 | Standard |
| HOT REPEAT | 3-4 | Standard+ |
| SERIAL RUNNER | 5+ | Adapte (volatil) |

**Warning**: Serial runners = moves violents dans les 2 sens

---

## Timeline Detection V6

```
16:00-20:00 ET | AFTER-HOURS
             | - News Flow + NLP Enrichi actif
             | - Pre-Spike Radar scanning
             | - Catalyst Score V3 calculating
             | - Signaux: WATCH_EARLY

04:00-09:30 ET | PRE-MARKET
             | - PM confirmation gaps
             | - Pre-Spike level update
             | - Repeat Gainer check
             | - Upgrades: WATCH_EARLY -> BUY
             | - Signaux: BUY, BUY_STRONG

09:30-16:00 ET | RTH
             | - Monitoring positions
             | - Trailing stops
             | - Late BUY_STRONG (rares)
```

---

## Risk Management V6

### Regles d'Or

1. **Stop-loss toujours**: Jamais de position sans stop
2. **Max 5 positions**: Diversification obligatoire
3. **Sizing par tier**: TIER 1 = max, TIER 5 = min
4. **Repeat Gainer warning**: Size down si serial runner

### Sizing par Signal + Tier

| Signal | TIER 1-2 | TIER 3 | TIER 4-5 |
|--------|----------|--------|----------|
| WATCH_EARLY | 0% | 0% | 0% |
| BUY | 2.5% | 2% | 1.5% |
| BUY_STRONG | 3% | 2.5% | 2% |

---

## Performance Attendue V6

| Metrique | Cible V6 |
|----------|----------|
| Hit Rate | 70-80% |
| Early Catch (>2h) | 60-70% |
| Avg Win | +50-90% |
| Avg Loss | -8-12% |
| Win/Loss Ratio | 4:1 |
| Lead Time | 8-24h |

---

## Dashboard V6

Le dashboard affiche maintenant:

- **V6 Modules Status**: Catalyst V3, Pre-Spike, Repeat Gainer, NLP
- **Signals avec badges V6**: Tier, Pre-Spike level, Repeat status
- **Monster Score radar**: Includes V6 components

```bash
streamlit run dashboards/streamlit_dashboard.py
```

---

**Version:** 6.0.0
**Last Updated:** 2026-02-09
