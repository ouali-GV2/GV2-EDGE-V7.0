# GV2-EDGE V9.0 - Trader Guide

## Objectif

GV2-EDGE detecte les top gainers small caps US **AVANT** leurs hausses majeures (+50% a +500%).

**Cible** : Small caps < $2B market cap, hors OTC

---

## Nouveautes V9.0

### Multi-Radar V9

**4 radars independants analysent chaque ticker en parallele :**

| Radar | Ce qu'il detecte | Latence |
|-------|-----------------|---------|
| **FLOW** | Accumulation volume, derivees, breakout readiness | <10ms |
| **CATALYST** | Catalysts, news, SEC filings, FDA, earnings | <50ms |
| **SMART MONEY** | Options inhabituelles, achats insiders | <100ms |
| **SENTIMENT** | Buzz social, sentiment NLP, repeat runners | <200ms |

### Evolution V7 → V8 → V9

| V7 | V8 | V9 |
|----|----|----|
| Score unique (Monster Score) | + AccelerationEngine + SmallCapRadar | + 4 radars paralleles + confluence matrix |
| Poll mode (2s latence) | Finnhub WebSocket | IBKR Streaming (~10ms) |
| Seuils fixes | Z-scores adaptatifs | 6 sous-sessions adaptatives |
| Risk multiplicatif | Risk MIN-based | Momentum override + planchers |

### Transparence Totale

**Le systeme montre TOUS les signaux, meme ceux bloques**

| Avant (V6) | Apres (V7+) |
|------------|-------------|
| Signal bloque = invisible | Signal bloque = visible avec raison |
| Pas de tracking des misses | Tracking complet pour apprentissage |
| Blocage a plusieurs niveaux | Blocage uniquement a l'execution |

### 3 Couches de Pipeline

| Couche | Module | Bloque? | Visible? |
|--------|--------|---------|----------|
| 1 | SignalProducer | NON | OUI |
| 2 | OrderComputer | NON | OUI |
| 3 | ExecutionGate | OUI | OUI (avec raison) |

### Metriques V9

| Metrique | Description |
|----------|-------------|
| **MRP** | Missed Recovery Potential - score base sur misses precedents |
| **EP** | Edge Probability - probabilite de succes |
| **Pre-Halt State** | Risque de halt (NORMAL/ELEVATED/HIGH) |
| **Block Reasons** | Pourquoi l'execution est bloquee |
| **Confluence** | Nombre de radars en accord (UNANIMOUS/STRONG/MODERATE) |
| **Lead Radar** | Quel radar a detecte en premier |

---

## Signaux V9

### BUY_STRONG

- **Quand**: Score 0.80+ + catalyst fort
- **Execution**: Si autorise par ExecutionGate
- **MRP/EP**: Affiche si actif
- **Action**: Entry immediate si autorise
- **Sizing**: 3% risk max

### BUY

- **Quand**: Score 0.65-0.79 + confirmation technique
- **Execution**: Si autorise par ExecutionGate
- **Action**: Entry standard si autorise
- **Sizing**: 2% risk

### WATCH

- **Quand**: Potentiel detecte, pas encore confirme
- **Execution**: Non (observation seulement)
- **Action**: Surveiller, preparer entry
- **Sizing**: 0% (pas d'entry)

### Signal BLOQUE

- **Quand**: ExecutionGate refuse l'execution
- **Visible**: OUI (avec raisons)
- **Action**: Comprendre la raison, noter pour apprentissage
- **Raisons possibles**:
  - `DAILY_TRADE_LIMIT` - Max trades du jour atteint
  - `CAPITAL_INSUFFICIENT` - Pas assez de capital
  - `PRE_HALT_HIGH` - Risque de halt eleve
  - `DILUTION_HIGH` - Risque de dilution
  - `COMPLIANCE_HIGH` - Risque compliance

---

## Alertes Telegram V9

### Format Signal Autorise

```
[SIGNAL_EMOJI] GV2-EDGE V7.0 SIGNAL

Ticker: NVDA
Signal: BUY_STRONG
Monster Score: 0.85

--- V9 Intelligence ---
Pre-Halt: NORMAL
MRP: 72 | EP: 68

--- Position ---
Entry: $152.50
Stop: $148.20
Shares: 45
Risk: $193.50

V7.0 | NORMAL
```

### Format Signal Bloque

```
[NO_ENTRY] GV2-EDGE V7.0 SIGNAL

Ticker: BIOX
Signal: BUY (BLOCKED)
Monster Score: 0.72

--- V9 Intelligence ---
Pre-Halt: ELEVATED

BLOCKED: DAILY_TRADE_LIMIT, PRE_HALT_ELEVATED

Signal detecte mais execution bloquee
```

---

## Pre-Halt State

| State | Signification | Action |
|-------|---------------|--------|
| NORMAL | Pas de risque detecte | Execute normal |
| ELEVATED | Volatilite anormale | Taille reduite 50% |
| HIGH | Risque de halt imminent | Execution bloquee |

### Indicateurs Pre-Halt

- Volatilite > 3x moyenne
- Mouvement prix > 15% intraday
- Keywords halt dans news
- Volume anormal

---

## MRP/EP (Market Memory)

### MRP (Missed Recovery Potential)

Score 0-100 base sur les signaux manques precedents:
- MRP eleve = ce ticker a souvent ete manque et a bien performe
- Utilise pour ajuster la confiance

### EP (Edge Probability)

Score 0-100 base sur patterns similaires:
- EP eleve = patterns similaires ont bien performe
- Utilise pour sizing

### Activation

MRP/EP s'activent UNIQUEMENT quand Market Memory a assez de donnees:
- 50+ misses tracked
- 30+ trades recorded
- 10+ patterns learned
- 20+ ticker profiles

Avant activation: `context_active = False`

---

## Raisons de Blocage

### DAILY_TRADE_LIMIT

- **Cause**: Max 5 trades/jour atteint
- **Solution**: Attendre demain
- **Note**: Signal track pour Market Memory

### CAPITAL_INSUFFICIENT

- **Cause**: Pas assez de cash disponible
- **Solution**: Liberer du capital
- **Note**: Position existante bloque les fonds

### PRE_HALT_HIGH

- **Cause**: Risque de halt detecte
- **Solution**: Attendre stabilisation
- **Note**: Protege contre halt losses

### DILUTION_HIGH

- **Cause**: ATM offering detecte
- **Solution**: Eviter ou reduire taille
- **Note**: Risk Guard protection

### COMPLIANCE_HIGH

- **Cause**: Risque delisting/SEC
- **Solution**: Eviter
- **Note**: Protection compliance

---

## Timeline V9

```
16:00-20:00 ET | AFTER-HOURS
             | - Detection anticipative
             | - News scanning
             | - Signaux: WATCH, BUY (rare)

04:00-09:30 ET | PRE-MARKET
             | - Confirmation PM
             | - V7 cycle (detection + execution)
             | - Signaux: BUY, BUY_STRONG

09:30-16:00 ET | RTH
             | - V7 cycle every 3 min
             | - Tous signaux possibles
             | - Execution Gate active

20:30 UTC     | DAILY AUDIT
             | - Performance du jour
             | - Blocked vs Allowed ratio
```

---

## Risk Management V9

### Regles d'Or

1. **Stop-loss toujours**: Jamais de position sans stop
2. **Max 5 trades/jour**: Applique par ExecutionGate
3. **Max 10% par position**: Protection capital
4. **Pre-Halt respect**: Si HIGH, pas d'entry
5. **Comprendre les blocks**: Apprendre des raisons

### Sizing par Signal

| Signal | Condition | Sizing |
|--------|-----------|--------|
| BUY_STRONG | Autorise | 3% risk |
| BUY_STRONG | Pre-Halt ELEVATED | 1.5% risk |
| BUY | Autorise | 2% risk |
| BUY | Pre-Halt ELEVATED | 1% risk |
| BLOCKED | - | 0% |

---

## Dashboard V9

Le dashboard affiche:

- **V9 Modules Status**: SignalProducer, OrderComputer, ExecutionGate, RiskGuard, MultiRadar
- **Multi-Radar**: Status des 4 radars, confluence agreements
- **Execution Stats**: Allowed vs Blocked ratio
- **Block Reasons**: Distribution des raisons
- **Market Memory Status**: MRP/EP activation state (segmente par catalyst)
- **Pre-Halt Alerts**: Tickers avec risque halt
- **IBKR Streaming**: Subscriptions actives, events/seconde

```bash
streamlit run dashboards/streamlit_dashboard.py
```

---

## Performance V9

| Metrique | V7 | Cible V9 |
|----------|-----|----------|
| Detection Rate | ~65-70% | **95%+** |
| Lead Time | ~3-8 min | **10-20 min** |
| Precision | ~40-50% | **65%+** |
| Latence L1 | 2000ms | **<50ms** |
| Couverture catalysts | ~60% | **90%+** |
| Hit Rate | 70-80% | **80%+** |

---

## Conseils V9

1. **Lisez les raisons de blocage** - Elles sont la pour vous proteger
2. **Respectez Pre-Halt** - Ne forcez pas si HIGH
3. **Utilisez MRP/EP** - Quand actif, c'est un edge supplementaire
4. **Track vos misses** - Ils alimentent Market Memory
5. **Attendez l'activation** - MRP/EP dormants au debut = normal
6. **Observez la confluence** - 4/4 radars = UNANIMOUS = forte conviction
7. **Notez le lead radar** - Indique l'angle de detection dominant

---

**Version:** 9.0.0
**Last Updated:** 2026-02-21
