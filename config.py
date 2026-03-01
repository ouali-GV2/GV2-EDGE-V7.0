# ============================
# GV2-EDGE GLOBAL CONFIG
# ============================

import os

# ========= API KEYS =========

GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ========= IBKR CONNECTION =========

# Use IBKR for real-time market data (Level 1)
# IBKR Gateway actif sur Hetzner CX43 (via xvfb)
# Abonnements : L1 US Equities (Network A/B/C) + OPRA Options US
# Fallback automatique vers Finnhub si Gateway deconnecte
USE_IBKR_DATA = os.getenv("USE_IBKR_DATA", "True").lower() in ("true", "1", "yes")

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))   # 7497 = paper trading, 7496 = live, 4001/4002 = Gateway
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

# Note: Level 1 subscription provides:
# - Real-time prices (last, bid, ask)
# - Volume & VWAP
# - Pre-market & After-hours data
# - Historical bars (unlimited)
# This is SUFFICIENT for GV2-EDGE (Level 2 not needed)

# Telegram — legacy single-channel (kept for backwards compatibility)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Telegram — dual-channel routing (V9)
# SIGNALS channel: trading alerts (signals, catalysts, pre-spike, multi-radar...)
# SYSTEM  channel: operational alerts (IBKR, CPU/RAM, audits, session changes...)
# Fallback to legacy BOT_TOKEN/CHAT_ID if not set
TELEGRAM_SIGNALS_TOKEN   = os.getenv("TELEGRAM_SIGNALS_TOKEN",  TELEGRAM_BOT_TOKEN)
TELEGRAM_SIGNALS_CHAT_ID = os.getenv("TELEGRAM_SIGNALS_CHAT_ID", TELEGRAM_CHAT_ID)
TELEGRAM_SYSTEM_TOKEN    = os.getenv("TELEGRAM_SYSTEM_TOKEN",   TELEGRAM_BOT_TOKEN)
TELEGRAM_SYSTEM_CHAT_ID  = os.getenv("TELEGRAM_SYSTEM_CHAT_ID",  TELEGRAM_CHAT_ID)

# ========= REDDIT API =========
# Get credentials at: https://www.reddit.com/prefs/apps

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "GV2-EDGE/1.0")

# ========= STOCKTWITS API =========
# Get API key at: https://api.stocktwits.com/developers

STOCKTWITS_ACCESS_TOKEN = os.getenv("STOCKTWITS_ACCESS_TOKEN", "")

# ============================
# TRADING CAPITAL
# ============================

# Option 1 — Manual (recommended start)
MANUAL_CAPITAL = int(os.getenv("MANUAL_CAPITAL", "1000"))

# Option 2 — Auto from IBKR
USE_IBKR_CAPITAL = False  

# ============================
# RISK SETTINGS
# ============================

RISK_BUY = 0.02           # 2% risk per trade
RISK_BUY_STRONG = 0.025  # 2.5% risk per trade

MAX_OPEN_POSITIONS = 5

# ============================
# STOPS & TRAILING
# ============================

ATR_MULTIPLIER_STOP = 2.0
ATR_MULTIPLIER_TRAIL = 1.5

USE_STRUCTURE_STOP_PM = True

# ============================
# SCAN TIMING (seconds)
# ============================

FULL_UNIVERSE_SCAN_INTERVAL = 300     # 5 min
EVENT_SCAN_INTERVAL = 600             # 10 min
PM_SCAN_INTERVAL = 60                 # 1 min

# ============================
# MARKET SESSIONS (US TIME)
# ============================

PREMARKET_START = "04:00"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
AFTER_HOURS_END = "20:00"

# ============================
# UNIVERSE FILTERS (ENHANCED V2)
# ============================

MAX_MARKET_CAP = 2_000_000_000   # 2B small caps
MIN_PRICE = 0.0001               # Penny stocks inclus (OTC exclus séparément)
MAX_PRICE = 20
MIN_AVG_VOLUME = 500_000         # ↑ Increased from 300K (better liquidity)

EXCLUDE_OTC = True

# ============================
# EVENT BOOST SETTINGS
# ============================

EVENT_PROXIMITY_BOOST = {
    "today": 1.5,
    "tomorrow": 1.3,
    "week": 1.1
}

EVENT_PROXIMITY_DAYS = 7  # Jours pour considérer un événement comme proche

# ============================
# MONSTER SCORE WEIGHTS (OPTIMIZED V2)
# ============================

# Poids BASIQUES (sans patterns avancés)
DEFAULT_MONSTER_WEIGHTS = {
    "event": 0.35,        # Poids des événements (FDA, M&A, Earnings, etc.)
    "momentum": 0.20,     # Poids du momentum prix
    "volume": 0.15,       # Poids du spike de volume
    "vwap": 0.10,        # Poids de la déviation VWAP
    "squeeze": 0.10,     # Poids du squeeze (momentum/volatilité)
    "pm_gap": 0.10       # Poids du gap pre-market
}
# Note: La somme DOIT faire 1.0 pour cohérence

# Poids OPTIMISÉS V4 (avec acceleration engine V8)
# V8 CHANGES:
# - NEW: acceleration (7%) - derivative-based anticipatory detection
# - REDUCED: momentum (8% → 4%) - replaced by acceleration derivatives
# - REDUCED: social_buzz (6% → 3%) - low reliability (80% placeholder per review)
ADVANCED_MONSTER_WEIGHTS = {
    "event": 0.25,          # Catalysts toujours importants
    "volume": 0.17,         # Confirmation essentielle
    "pattern": 0.17,        # Patterns structurels
    "pm_transition": 0.13,  # Timing PM→RTH
    "acceleration": 0.07,   # V8 NEW: Anticipatory detection (velocity + accel)
    "momentum": 0.04,       # ↓ 8% → 4% (redundant with acceleration)
    "squeeze": 0.04,        # Low priority
    "options_flow": 0.10,   # Options activity
    "social_buzz": 0.03,    # ↓ 6% → 3% (low reliability per review)
}
# Monster Score V4 — Total = 1.0 (event25 + vol17 + pattern17 + pmtrans13 + optflow10 + accel7 + mom4 + sq4 + social3 = 100%)

# ============================
# PATTERN ANALYZER SETTINGS
# ============================

USE_ADVANCED_PATTERNS = True  # Activer patterns avancés
PATTERN_MIN_CANDLES = 20      # Minimum de candles pour analyse

# Seuils patterns
TIGHT_CONSOLIDATION_MAX_RANGE = 0.03  # 3% max pour consolidation tight
HIGHER_LOWS_MIN_TOUCHES = 3           # Minimum de higher lows
FLAG_PENNANT_SPIKE_THRESHOLD = 0.10   # 10% spike minimum pour flag
BOLLINGER_SQUEEZE_THRESHOLD = 0.5     # Ratio bandwidth pour squeeze

# ============================
# PM TRANSITION SETTINGS
# ============================

PM_RETEST_TOLERANCE = 0.005        # ±0.5% tolérance pour retest PM high
PM_FAKEOUT_MIN_HOLD_CANDLES = 5   # Candles minimum pour éviter fakeout
PM_STRONG_POSITION_THRESHOLD = 0.8 # Position dans range PM (80%+ = bullish)

# ============================
# AUTO TUNING
# ============================

AUTO_TUNING_ENABLED = True

TUNING_STEP = 0.02
TUNING_MIN_WEIGHT = 0.05
TUNING_MAX_WEIGHT = 0.50

# ============================
# SLIPPAGE SIMULATION (BACKTEST) - REALISTIC V2
# ============================

BASE_SLIPPAGE_PCT = 0.5        # ↑ Increased to 0.5% (realistic for small caps)
LOW_LIQUIDITY_SLIPPAGE = 1.0   # ↑ Increased to 1.0% (low float stocks)

# ============================
# LOGGING
# ============================

LOG_LEVEL = "INFO"
LOG_ROTATION_MB = 10
LOG_BACKUPS = 5

# ============================
# PERFORMANCE SAFETY
# ============================

MAX_CPU_USAGE = 85   # %
MAX_RAM_USAGE = 80   # %

# ============================
# DASHBOARD
# ============================

DASHBOARD_REFRESH_SECONDS = 5

# ============================
# BACKTEST
# ============================

BACKTEST_SLIPPAGE_ENABLED = True
BACKTEST_REALISTIC_STOPS = True

# ============================
# MISC
# ============================

DEBUG_MODE = False

# ============================
# SIGNAL THRESHOLDS
# ============================

BUY_THRESHOLD = 0.65          # Score minimum pour signal BUY
BUY_STRONG_THRESHOLD = 0.80   # Score minimum pour signal BUY_STRONG

# ============================
# PRE-MARKET SETTINGS
# ============================

PM_MIN_VOLUME = 50000         # Volume minimum en pre-market pour liquidité

# ============================
# OPTIONS FLOW SETTINGS (NEW V3)
# ============================

ENABLE_OPTIONS_FLOW = True        # Enable options flow analysis
OPTIONS_FLOW_MIN_VOLUME = 5000    # Minimum call volume for signal
OPTIONS_FLOW_MIN_TOTAL = 10000    # Minimum total options volume

# ============================
# SOCIAL BUZZ SETTINGS (NEW V3)
# ============================

ENABLE_SOCIAL_BUZZ = True                # Enable social sentiment tracking
SOCIAL_BUZZ_LOOKBACK_HOURS = 24          # Hours to look back for buzz
SOCIAL_BUZZ_SPIKE_THRESHOLD = 3.0        # 3x normal = spike

# Active sources and weights (total = 100%)
# Twitter/X (via Grok): 45% - Real-time, institutional leaks
# Reddit (WSB + others): 30% - Retail sentiment, meme stocks
# StockTwits: 25% - Dedicated stock traders
SOCIAL_BUZZ_SOURCES = ["twitter", "reddit", "stocktwits"]

# Google Trends - DISABLED by default (pytrends is unreliable)
# Set to True to enable (may cause rate limiting issues)
ENABLE_GOOGLE_TRENDS = False

# ============================
# REPEAT GAINER MEMORY (V6)
# ============================

ENABLE_REPEAT_GAINER = True           # Enable repeat gainer tracking
REPEAT_GAINER_LOOKBACK_DAYS = 180     # How far back to look for spikes
REPEAT_GAINER_MIN_SPIKE_PCT = 20.0    # Minimum gain % to count as spike
REPEAT_GAINER_DECAY_HALF_LIFE = 30    # Recency decay half-life in days
REPEAT_GAINER_THRESHOLD = 0.5         # Score threshold for "repeat runner"
REPEAT_GAINER_MAX_BOOST = 1.5         # Maximum boost multiplier

# ============================
# MARKET CALENDAR (V6)
# ============================

# Volume adjustment for comparison
VOLUME_ADJUST_EARLY_CLOSE = 0.6       # Volume factor for early close days
VOLUME_ADJUST_FRIDAY = 0.9            # Volume factor for Fridays
VOLUME_ADJUST_PRE_HOLIDAY = 0.75      # Volume factor for day before holiday

# ============================
# CATALYST SCORE V3 (V6)
# ============================

ENABLE_CATALYST_V3 = True             # Enable enhanced catalyst scoring

# Temporal decay settings
CATALYST_DECAY_HALF_LIFE_HOURS = 24   # Hours for score to decay by 50%
CATALYST_MIN_SCORE_THRESHOLD = 0.1    # Minimum score to consider

# Boost settings
CATALYST_MAX_BOOST = 1.6              # Maximum boost multiplier for Monster Score

# Alert level thresholds
CATALYST_ALERT_CRITICAL = 0.8         # Score threshold for CRITICAL alert
CATALYST_ALERT_HIGH = 0.6             # Score threshold for HIGH alert
CATALYST_ALERT_MEDIUM = 0.4           # Score threshold for MEDIUM alert

# Performance tracking
CATALYST_TRACK_PERFORMANCE = True     # Track catalyst performance for learning
CATALYST_PERFORMANCE_LOOKBACK = 90    # Days to look back for performance stats

# ============================
# PRE-SPIKE RADAR (V6)
# ============================

ENABLE_PRE_SPIKE_RADAR = True         # Enable pre-spike detection

# Acceleration thresholds (0-1)
PRE_SPIKE_VOLUME_THRESHOLD = 0.3      # Min volume acceleration
PRE_SPIKE_OPTIONS_THRESHOLD = 0.25    # Min options acceleration
PRE_SPIKE_BUZZ_THRESHOLD = 0.2        # Min buzz acceleration
PRE_SPIKE_SQUEEZE_THRESHOLD = 0.4     # Min technical compression

# Confluence settings
PRE_SPIKE_MIN_CONFLUENCE_WATCH = 2    # Signals needed for WATCH alert
PRE_SPIKE_MIN_CONFLUENCE_HIGH = 3     # Signals needed for HIGH alert
PRE_SPIKE_MIN_PROBABILITY = 0.4       # Min probability for high-priority list

# Time windows
PRE_SPIKE_WINDOW_MINUTES = 30         # Window for acceleration calculation
PRE_SPIKE_LOOKBACK_PERIODS = 6        # Number of periods to analyze

# Boost settings
PRE_SPIKE_MAX_BOOST = 1.4             # Maximum boost multiplier for Monster Score

# ============================
# NLP ENRICHI (V6)
# ============================

ENABLE_NLP_ENRICHI = True             # Enable advanced NLP processing

# Sentiment settings
NLP_USE_GROK = True                   # Use Grok API for deep analysis (vs keyword-only)
NLP_SENTIMENT_LOOKBACK_HOURS = 24     # Hours to look back for sentiment aggregation
NLP_SENTIMENT_SPIKE_THRESHOLD = 0.3   # Sentiment change threshold for spike alert

# Boost settings
NLP_SENTIMENT_MAX_BOOST = 1.4         # Maximum sentiment boost multiplier
NLP_SENTIMENT_MIN_BOOST = 0.7         # Minimum sentiment boost (bearish penalty)

# Urgency decay (hours for urgency to lose impact)
NLP_URGENCY_DECAY_BREAKING = 4        # Breaking news decays fast
NLP_URGENCY_DECAY_HIGH = 12
NLP_URGENCY_DECAY_MEDIUM = 24
NLP_URGENCY_DECAY_LOW = 48

# Category impact weights (override defaults)
# Higher = more impact on final score
NLP_CATEGORY_FDA_WEIGHT = 1.0
NLP_CATEGORY_MA_WEIGHT = 0.95
NLP_CATEGORY_EARNINGS_WEIGHT = 0.85
NLP_CATEGORY_CONTRACT_WEIGHT = 0.75

# ============================
# V7.0 ARCHITECTURE (NEW)
# ============================

# Enable V7.0 unified signal pipeline
USE_V7_ARCHITECTURE = True            # Use SignalProducer → OrderComputer → ExecutionGate

# ============================
# EXECUTION GATE (V7)
# ============================

DAILY_TRADE_LIMIT = 5                 # Max trades per day
MAX_POSITION_PCT = 0.10               # Max 10% of capital per position
MAX_TOTAL_EXPOSURE = 0.80             # Max 80% total exposure
MIN_ORDER_USD = 100                   # Minimum order size

# ============================
# PRE-HALT ENGINE (V7)
# ============================

ENABLE_PRE_HALT_ENGINE = True         # Enable pre-halt risk assessment

# Halt risk thresholds
PRE_HALT_VOLATILITY_THRESHOLD = 3.0   # Volatility multiplier for elevated risk
PRE_HALT_PRICE_MOVE_THRESHOLD = 0.15  # 15% move triggers elevated risk
PRE_HALT_NEWS_KEYWORDS = [
    "halt", "pending news", "acquired", "buyout", "merger",
    "fda approval", "breakthrough", "sec investigation"
]

# Position adjustments by halt risk
PRE_HALT_SIZE_MULTIPLIER_LOW = 1.0
PRE_HALT_SIZE_MULTIPLIER_MEDIUM = 0.5
PRE_HALT_SIZE_MULTIPLIER_HIGH = 0.0   # Block execution

# ============================
# IBKR NEWS TRIGGER (V7)
# ============================

ENABLE_IBKR_NEWS_TRIGGER = True       # Enable IBKR news-based early alerts

# Trigger keywords by category
NEWS_HALT_KEYWORDS = [
    "halt", "halted", "pending news", "news pending",
    "acquired", "acquisition", "buyout", "merger",
    "fda approval", "fda clearance", "breakthrough therapy"
]

NEWS_SPIKE_KEYWORDS = [
    "surge", "surging", "spike", "spiking", "soar", "soaring",
    "jump", "jumping", "rally", "unusual volume", "breakout"
]

NEWS_RISK_KEYWORDS = [
    "dilution", "offering", "shelf registration", "atm program",
    "sec investigation", "delisting", "compliance", "going concern"
]

# ============================
# RISK GUARD (V7)
# ============================

ENABLE_RISK_GUARD = True              # Enable unified risk assessment

# Block conditions
RISK_BLOCK_ON_CRITICAL = True         # Block on CRITICAL risk level
RISK_BLOCK_ON_ACTIVE_OFFERING = True  # Block on active stock offering
RISK_BLOCK_ON_DELISTING = True        # Block on delisting risk
RISK_BLOCK_ON_HALT = True             # Block on current halt

# Position multipliers
RISK_MIN_POSITION_MULTIPLIER = 0.10   # Minimum 10% of intended size

# ============================
# TICKER PROFILE STORE (V9)
# ============================

ENABLE_TICKER_PROFILES = True         # Enable per-ticker strategic profile DB
TICKER_PROFILES_DB = "data/ticker_profiles.db"
TICKER_PROFILES_UPDATE_DAYS = 7       # Days between full profile refreshes

# ============================
# MARKET MEMORY MRP/EP (V7)
# ============================

ENABLE_MARKET_MEMORY = True           # Enable MRP/EP context enrichment

# Auto-activation thresholds (MRP/EP stay inactive until these are met)
MARKET_MEMORY_MIN_MISSES = 50         # Minimum tracked misses
MARKET_MEMORY_MIN_TRADES = 30         # Minimum recorded trades
MARKET_MEMORY_MIN_PATTERNS = 10       # Minimum learned patterns
MARKET_MEMORY_MIN_PROFILES = 20       # Minimum ticker profiles

# ============================
# V8: ACCELERATION ENGINE
# ============================

ENABLE_ACCELERATION_ENGINE = True      # Enable V8 anticipatory detection

# TickerStateBuffer settings
TICKER_BUFFER_MAX_SNAPSHOTS = 120      # 2 hours at 1-min intervals
TICKER_BUFFER_DERIVATIVE_WINDOW = 5    # Samples for derivative calculation

# Acceleration detection thresholds (z-score based)
ACCEL_VOLUME_ZSCORE_THRESHOLD = 1.5    # Volume z-score for "interesting"
ACCEL_VOLUME_ZSCORE_STRONG = 2.5       # Volume z-score for "strong signal"
ACCEL_ACCUMULATION_MIN = 0.30          # Min accumulation score to flag
ACCEL_BREAKOUT_READINESS_MIN = 0.50    # Min readiness for launch alert

# ============================
# V8: SMALLCAP RADAR
# ============================

ENABLE_SMALLCAP_RADAR = True           # Enable V8 small-cap radar
RADAR_SENSITIVITY = "HIGH"             # ULTRA / HIGH / STANDARD
RADAR_SCAN_INTERVAL = 5               # Seconds between scans (fast: buffer reads only)

# Risk Guard V8 overrides
RISK_APPLY_COMBINED_MULTIPLIERS = False  # V8: Use MIN mode (not multiplicative)
RISK_ENABLE_MOMENTUM_OVERRIDE = True     # V8: Allow momentum to reduce risk penalties

# ============================
# V9: MULTI-RADAR ENGINE
# ============================

ENABLE_MULTI_RADAR = True                 # Enable V9 multi-radar detection (4 radars paralleles)

# Seuils de signal confluence
MULTI_RADAR_BUY_STRONG_THRESHOLD = 0.75   # Score confluence minimum pour BUY_STRONG
MULTI_RADAR_BUY_THRESHOLD = 0.55          # Score confluence minimum pour BUY
MULTI_RADAR_WATCH_THRESHOLD = 0.35        # Score confluence minimum pour WATCH
MULTI_RADAR_EARLY_THRESHOLD = 0.20        # Score confluence minimum pour EARLY_SIGNAL

# Bonus de confluence (recompense l'accord entre radars)
MULTI_RADAR_UNANIMOUS_BONUS = 0.15        # 4/4 radars actifs
MULTI_RADAR_STRONG_BONUS = 0.10           # 3/4 radars actifs
MULTI_RADAR_MODERATE_BONUS = 0.05         # 2/4 radars actifs
