# GV2-EDGE V9.0 - Developer Documentation

## Objectif

Ce document explique:
- L'architecture technique V9.0 (V7 Detection/Execution + V8 Acceleration + V9 Multi-Radar)
- Le role de chaque module
- Les flux de donnees et le scoring
- Comment etendre le systeme

---

## Architecture V7.0

### Principe Fondamental

**Detection JAMAIS bloquee, Execution UNIQUEMENT limitee**

```
AVANT (V6): if blocked: return None  # Signal invisible
APRES (V7): signal = produce()       # TOUJOURS visible
            order = compute()         # TOUJOURS calcule
            decision = gate()         # Seul point de blocage
```

### 3-Layer Pipeline

```python
# Layer 1: SignalProducer (Detection - Never Blocked)
signal = await producer.detect(detection_input)
# -> UnifiedSignal with signal_type, monster_score

# Enrichment: Market Memory (Informational)
enrich_signal_with_context(signal)
# -> Adds context_mrp, context_ep, context_active

# Layer 2: OrderComputer (Always Computed)
signal = computer.compute_order(signal, market_context)
# -> Adds ProposedOrder (size, stop, target)

# Risk Assessment: UnifiedGuard (Informational)
risk_flags = await guard.assess(ticker, price, volatility)
# -> RiskFlags for ExecutionGate

# Layer 3: ExecutionGate (Only Blocking Layer)
signal = gate.evaluate(signal, risk_flags)
# -> Adds ExecutionDecision (ALLOW/BLOCK + reasons)
```

---

## Structure des Modules

### src/engines/ - Core Pipeline

```
engines/
├── __init__.py
├── signal_producer.py        # Layer 1: Detection V8
├── order_computer.py         # Layer 2: Order calculation
├── execution_gate.py         # Layer 3: Limits
├── acceleration_engine.py    # V8: Derivees, z-scores
├── ticker_state_buffer.py    # V8: Ring buffer 120pts
├── smallcap_radar.py         # V8: Radar anticipatif (4 phases)
└── multi_radar_engine.py     # V9: 4 radars paralleles + confluence
```

#### signal_producer.py

```python
@dataclass
class DetectionInput:
    ticker: str
    current_price: float
    monster_score: float
    catalyst_score: float
    pre_spike_state: PreSpikeState
    catalyst_type: Optional[str]
    catalyst_confidence: float
    volume_ratio: float
    price_change_pct: float
    market_session: str

class SignalProducer:
    async def detect(self, input: DetectionInput) -> UnifiedSignal:
        """NEVER blocks - always produces a signal"""
        signal_type = self._classify_signal(input)
        return UnifiedSignal(
            ticker=input.ticker,
            signal_type=signal_type,
            monster_score=input.monster_score,
            # ... more fields
        )
```

#### order_computer.py

```python
@dataclass
class ProposedOrder:
    side: str                 # "BUY"
    size_shares: int          # Calculated position size
    size_usd: float           # Dollar amount
    price_target: float       # Entry price
    stop_loss: float          # Stop-loss price
    take_profit: float        # Take-profit price
    confidence: float         # Order confidence

class OrderComputer:
    def compute_order(self, signal: UnifiedSignal, market: MarketContext) -> UnifiedSignal:
        """ALWAYS computes order - never blocks"""
        order = self._calculate_order(signal, market)
        signal.proposed_order = order
        return signal
```

#### execution_gate.py

```python
@dataclass
class ExecutionDecision:
    allowed: bool
    size_multiplier: float    # 1.0 = full, 0.5 = half, 0.0 = blocked
    blocked_by: List[BlockReason]
    block_message: str

class ExecutionGate:
    def evaluate(self, signal: UnifiedSignal, risk_flags: RiskFlags) -> UnifiedSignal:
        """ONLY layer that can block execution"""
        decision = self._evaluate_limits(signal, risk_flags)
        signal.execution = decision
        return signal
```

### V9 Modules

#### multi_radar_engine.py (V9)

```python
class MultiRadarEngine:
    """4 radars paralleles avec confluence matrix, session-adaptatif"""

    async def scan(self, ticker, session) -> ConfluenceSignal:
        # 4 radars en parallele via asyncio.gather
        flow, catalyst, smart_money, sentiment = await asyncio.gather(
            self.flow_radar.scan(ticker),
            self.catalyst_radar.scan(ticker),
            self.smart_money_radar.scan(ticker),
            self.sentiment_radar.scan(ticker),
        )
        return self.confluence_matrix.evaluate(flow, catalyst, smart_money, sentiment)
```

#### ibkr_streaming.py (V9)

```python
class IBKRStreamingEngine:
    """Streaming temps reel event-driven (~10ms latence)"""

    # Remplace poll-and-cancel (2s) par pendingTickersEvent callback
    # Subscriptions persistantes (max 200 concurrentes)
    # Auto-detection: VOLUME_SPIKE, PRICE_SURGE, SPREAD_TIGHTENING
    # Feed automatique TickerStateBuffer + HotTickerQueue
```

### src/models/signal_types.py - Data Models

```python
@dataclass
class UnifiedSignal:
    # Core
    ticker: str
    signal_type: SignalType
    monster_score: float
    timestamp: datetime

    # Pre-states
    pre_spike_state: PreSpikeState
    pre_halt_state: PreHaltState

    # MRP/EP Context
    context_mrp: Optional[float]
    context_ep: Optional[float]
    context_confidence: Optional[float]
    context_active: bool

    # Order (Layer 2)
    proposed_order: Optional[ProposedOrder]

    # Execution (Layer 3)
    execution: Optional[ExecutionDecision]

    def is_actionable(self) -> bool:
        """Signal worth considering (not HOLD/AVOID)"""
        return self.signal_type in [SignalType.BUY, SignalType.BUY_STRONG, SignalType.WATCH]

    def is_executable(self) -> bool:
        """Allowed by ExecutionGate"""
        return self.execution and self.execution.allowed
```

### src/risk_guard/ - Risk Assessment

```
risk_guard/
├── __init__.py
├── unified_guard.py       # Main entry point
├── dilution_detector.py   # ATM, offerings
├── compliance_checker.py  # Delisting, SEC
└── halt_monitor.py        # Current/imminent halts
```

```python
class UnifiedGuard:
    async def assess(self, ticker: str, current_price: float, volatility: float) -> Assessment:
        dilution = await self.dilution_detector.check(ticker)
        compliance = await self.compliance_checker.check(ticker)
        halt = await self.halt_monitor.check(ticker)

        return Assessment(
            dilution_profile=dilution,
            compliance_profile=compliance,
            halt_status=halt,
            flags=self._compute_flags(dilution, compliance, halt)
        )
```

### src/market_memory/ - MRP/EP Context

```
market_memory/
├── __init__.py
├── context_scorer.py      # MRP/EP calculation
├── missed_tracker.py      # Track blocked signals
├── pattern_learner.py     # Pattern analysis
└── memory_store.py        # Persistence
```

```python
# context_scorer.py

def is_market_memory_stable() -> Tuple[bool, Dict]:
    """Check if enough data to activate MRP/EP"""
    stats = get_memory_status()

    stable = (
        stats["total_misses"] >= MIN_TOTAL_MISSES and
        stats["trades_recorded"] >= MIN_TRADES_RECORDED and
        stats["patterns_learned"] >= MIN_PATTERNS_LEARNED and
        stats["ticker_profiles"] >= MIN_TICKER_PROFILES
    )

    return stable, stats

def enrich_signal_with_context(signal: UnifiedSignal, force_enable: bool = False) -> None:
    """Enrich signal with MRP/EP (O(1) lookup)"""
    is_stable, _ = is_market_memory_stable()

    if not is_stable and not force_enable:
        signal.context_active = False
        return

    # Calculate MRP/EP
    signal.context_mrp = calculate_mrp(signal.ticker)
    signal.context_ep = calculate_ep(signal.ticker, signal.signal_type)
    signal.context_confidence = get_data_confidence(signal.ticker)
    signal.context_active = True
```

### src/pre_halt_engine.py - Halt Risk Detection

```python
class ExecutionRecommendation(Enum):
    EXECUTE = "EXECUTE"       # Normal execution
    WAIT = "WAIT"             # Wait for clarity
    REDUCE = "REDUCE"         # Reduce position size
    POST_HALT = "POST_HALT"   # Wait for post-halt
    BLOCKED = "BLOCKED"       # Do not execute

class PreHaltEngine:
    def assess(self, ticker: str, current_price: float, volatility: float) -> PreHaltAssessment:
        # Check volatility spike
        # Check price move threshold
        # Check news keywords
        # Return state + recommendation
        pass
```

### src/ibkr_news_trigger.py - Early News Alerts

```python
HALT_KEYWORDS = ["halt", "pending news", "acquired", "buyout", "merger", "fda approval"]
SPIKE_KEYWORDS = ["surge", "jump", "spike", "rally", "unusual volume"]
RISK_KEYWORDS = ["dilution", "offering", "sec investigation", "delisting"]

class IBKRNewsTrigger:
    def process_headline(self, ticker: str, headline: str) -> Optional[NewsTrigger]:
        """Process headline for early alerts - NOT for scoring"""
        level = self._detect_trigger_level(headline)

        if level >= TriggerLevel.MEDIUM:
            return NewsTrigger(
                ticker=ticker,
                headline=headline,
                trigger_level=level,
                trigger_type=self._classify_type(headline),
                recommended_actions=self._get_actions(level)
            )
        return None
```

---

## Flow Complet V7

### main.py - edge_cycle_v7()

```python
async def process_ticker_v7(ticker: str, state: V7State) -> Optional[UnifiedSignal]:
    # 1. Get monster score and features
    score_data = compute_monster_score(ticker)
    features = compute_features(ticker)

    # 2. Build detection input
    detection_input = DetectionInput(
        ticker=ticker,
        current_price=current_price,
        monster_score=score_data["monster_score"],
        # ...
    )

    # 3. LAYER 1: Signal Producer (never blocked)
    signal = await state.producer.detect(detection_input)

    # 4. Pre-Halt Engine (sets pre_halt_state)
    halt_assessment = state.pre_halt.assess(ticker, price, volatility)
    signal.pre_halt_state = halt_assessment.pre_halt_state

    # 5. MRP/EP Enrichment
    enrich_signal_with_context(signal)

    # 6. LAYER 2: Order Computer (always computed)
    signal = state.computer.compute_order(signal, market_context)

    # 7. Risk Guard (get flags)
    risk_flags = await state.guard.assess(ticker, price, volatility)

    # 8. LAYER 3: Execution Gate (only blocking layer)
    signal = state.gate.evaluate(signal, risk_flags)

    return signal
```

---

## Tests

```bash
# Test V7 signal producer
python -c "from src.engines.signal_producer import get_signal_producer; print('OK')"

# Test order computer
python -c "from src.engines.order_computer import get_order_computer; print('OK')"

# Test execution gate
python -c "from src.engines.execution_gate import get_execution_gate; print('OK')"

# Test risk guard
python -c "from src.risk_guard import get_unified_guard; print('OK')"

# Test market memory
python -c "from src.market_memory import is_market_memory_stable; print(is_market_memory_stable())"

# Test full pipeline
python tests/test_pipeline.py

# Test V9 Multi-Radar
python -c "from src.engines.multi_radar_engine import get_multi_radar_engine; print('MultiRadar OK')"

# Test IBKR Streaming
python -c "from src.ibkr_streaming import get_ibkr_streaming; print('Streaming OK')"
```

---

## Ajouter un Nouveau Module

1. Creer le module dans `src/`
2. Definir l'interface (input/output dataclasses)
3. Integrer dans `main.py` au bon endroit du pipeline
4. Ajouter tests dans `tests/`
5. Documenter dans ce README

### Regles V7

- Detection ne bloque JAMAIS
- Order est TOUJOURS calcule
- Seul ExecutionGate peut bloquer
- Raisons de blocage TOUJOURS visibles
- Signaux bloques alimentent Market Memory
- Multi-Radar V9 : 4 radars paralleles, confluence matrix, session-adaptatif
- IBKR Streaming V9 : event-driven, fallback automatique vers poll mode
- Detection jamais arretee, meme si un radar est indisponible

---

## Configuration

### config.py - V7 Settings

```python
# V7.0 Architecture
USE_V7_ARCHITECTURE = True

# Execution Gate
DAILY_TRADE_LIMIT = 5
MAX_POSITION_PCT = 0.10
MAX_TOTAL_EXPOSURE = 0.80
MIN_ORDER_USD = 100

# Pre-Halt Engine
ENABLE_PRE_HALT_ENGINE = True
PRE_HALT_VOLATILITY_THRESHOLD = 3.0
PRE_HALT_PRICE_MOVE_THRESHOLD = 0.15

# Risk Guard
ENABLE_RISK_GUARD = True
RISK_BLOCK_ON_CRITICAL = True
RISK_BLOCK_ON_ACTIVE_OFFERING = True

# Market Memory
ENABLE_MARKET_MEMORY = True
MARKET_MEMORY_MIN_MISSES = 50
MARKET_MEMORY_MIN_TRADES = 30
MARKET_MEMORY_MIN_PATTERNS = 10

# V9.0 Multi-Radar
ENABLE_MULTI_RADAR = True
MULTI_RADAR_MIN_AGREEMENT = 2  # Minimum 2 radars actifs

# V9.0 IBKR Streaming
ENABLE_IBKR_STREAMING = True
IBKR_MAX_SUBSCRIPTIONS = 200
```

---

## Regles Critiques

1. **Detection JAMAIS bloquee** - SignalProducer produit toujours
2. **Order TOUJOURS calcule** - OrderComputer calcule toujours
3. **Execution seul point de blocage** - ExecutionGate unique gate
4. **Transparence totale** - Raisons de blocage visibles
5. **Apprentissage continu** - Misses alimentent Market Memory
6. **IBKR READ ONLY** - Jamais d'ordres automatiques

---

**Version:** 9.0.0
**Last Updated:** 2026-02-21
