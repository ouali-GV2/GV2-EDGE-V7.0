# ============================
# GV2-EDGE GLOBAL CONFIG
# ============================

# ========= API KEYS =========

GROK_API_KEY = "YOUR_GROK_API_KEY"
FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"

# ========= IBKR CONNECTION =========

# Use IBKR for real-time market data (Level 1)
USE_IBKR_DATA = True  # Set to False to use Finnhub instead

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497   # 7497 = paper trading, 7496 = live, 4001/4002 = Gateway
IBKR_CLIENT_ID = 1

# Note: Level 1 subscription provides:
# - Real-time prices (last, bid, ask)
# - Volume & VWAP
# - Pre-market & After-hours data
# - Historical bars (unlimited)
# This is SUFFICIENT for GV2-EDGE (Level 2 not needed)

# Telegram
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# ============================
# TRADING CAPITAL
# ============================

# Option 1 — Manual (recommended start)
MANUAL_CAPITAL = 1000  

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
MIN_PRICE = 1.0                  # ↑ Increased from 0.5 (avoid extreme penny stocks)
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

# Poids OPTIMISÉS V2 (avec patterns + PM transition)
# Basé sur l'expérience momentum trading et feedback loops
ADVANCED_MONSTER_WEIGHTS = {
    "event": 0.30,          # ↑ Increased (catalysts = ROI principal)
    "volume": 0.20,         # ↑ Increased (volume = confirmation essentielle)
    "pattern": 0.20,        # = Patterns structurels OK
    "pm_transition": 0.15,  # = Timing PM→RTH crucial
    "momentum": 0.10,       # ↓ Reduced (momentum suit, ne prédit pas)
    "squeeze": 0.05,        # ↓ Reduced (low priority, corrélé avec patterns)
    # vwap removed (déjà corrélé au price action)
}
# Total = 1.0

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
