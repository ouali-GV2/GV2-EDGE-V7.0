"""
ANTICIPATION ENGINE - Early Detection of Future Top Gainers
============================================================

Architecture hybride pour détecter les top gainers AVANT leur spike:

COUCHE 1: IBKR Radar (low-cost, large coverage)
- Scan large basse fréquence (30-60 min) sur 300-500 tickers
- Détecte: volume spikes, gaps anormaux, volatilité inhabituelle
- Filtre primaire → génère liste "suspects"

COUCHE 2: V6.1 Ingestors (100% sources réelles)
- SEC EDGAR 8-K filings (gratuit, temps réel)
- Finnhub company news (ciblé par ticker)
- NLP classification via Grok (événements seulement)
- Pas de simulation API via LLM

COUCHE 3: Finnhub (fallback + supplementary)
- Backup si IBKR indisponible
- News générales complémentaires

TIMELINE:
16:00-20:00 ET → After-hours catalyst scan
20:00-04:00 ET → Overnight watch
04:00-09:30 ET → Pre-market confirmation
09:30-16:00 ET → RTH monitoring

SIGNAUX:
- WATCH_EARLY: Catalyst détecté, potentiel en formation
- BUY: Confirmation technique (volume, momentum, PM move)
- BUY_STRONG: Breakout/spike confirmé

Objectif: Entrer AVANT que le mover soit visible par tous
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from utils.logger import get_logger
from utils.cache import Cache
from utils.time_utils import is_after_hours, is_premarket, is_market_open
from utils.api_guard import safe_get, safe_post

from config import (
    GROK_API_KEY,
    FINNHUB_API_KEY,
    USE_IBKR_DATA,
    MAX_MARKET_CAP,
    MIN_PRICE,
    MAX_PRICE,
    MIN_AVG_VOLUME
)

logger = get_logger("ANTICIPATION_ENGINE")


# ============================
# ENUMS & DATA CLASSES
# ============================

class SignalLevel(Enum):
    WATCH_EARLY = "WATCH_EARLY"   # Catalyst détecté, potentiel en formation
    BUY = "BUY"                    # Confirmation technique
    BUY_STRONG = "BUY_STRONG"      # Breakout confirmé
    HOLD = "HOLD"                  # Pas d'action
    EXIT = "EXIT"                  # Sortir


@dataclass
class Anomaly:
    """Anomaly detected by IBKR/Finnhub radar"""
    ticker: str
    anomaly_type: str  # GAP, VOLUME_SPIKE, VOLATILITY
    score: float       # 0.0 - 1.0
    details: Dict
    source: str        # ibkr or finnhub
    timestamp: str


@dataclass
class CatalystEvent:
    """Catalyst event detected by V6.1 ingestors (SEC + Finnhub)"""
    ticker: str
    event_type: str    # FDA_APPROVAL, EARNINGS_BEAT, MERGER, etc.
    impact_score: float
    headline: str
    summary: str
    source: str        # sec_8k, finnhub_company, etc.
    timestamp: str


@dataclass
class AnticipationSignal:
    """Combined anticipation signal"""
    ticker: str
    signal_level: SignalLevel
    combined_score: float
    urgency: str       # HIGH, MEDIUM, LOW
    
    # Technical (from anomaly)
    technical_score: float
    anomaly_type: Optional[str]
    
    # Fundamental (from catalyst)
    fundamental_score: float
    catalyst_type: Optional[str]
    catalyst_summary: str
    
    # Metadata
    detection_time: str
    status: str        # PENDING, CONFIRMED, EXPIRED


# ============================
# STATE MANAGEMENT
# ============================

class AnticipationState:
    """Centralized state for anticipation engine"""
    
    def __init__(self):
        self.suspects: Set[str] = set()
        self.watch_early_signals: Dict[str, AnticipationSignal] = {}
        self.anomaly_cache = Cache(ttl=1800)   # 30 min
        self.news_cache = Cache(ttl=900)       # 15 min
        
        # Rate limiting for Grok
        self.grok_calls: List[datetime] = []
        self.GROK_CALLS_PER_HOUR = 300
        
        # Scan timestamps
        self.last_radar_scan: Optional[datetime] = None
        self.last_grok_scan: Optional[datetime] = None
    
    def can_call_grok(self) -> bool:
        """Check Grok rate limit"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        self.grok_calls = [t for t in self.grok_calls if t > hour_ago]
        return len(self.grok_calls) < self.GROK_CALLS_PER_HOUR
    
    def record_grok_call(self):
        """Record a Grok API call"""
        self.grok_calls.append(datetime.utcnow())
    
    def add_suspects(self, tickers: List[str]):
        """Add tickers to suspects list"""
        self.suspects.update(tickers)
    
    def get_suspects(self) -> List[str]:
        """Get current suspects"""
        return list(self.suspects)
    
    def clear_suspects(self):
        """Clear suspects list"""
        self.suspects.clear()
    
    def add_watch_signal(self, signal: AnticipationSignal):
        """Add/update a WATCH_EARLY signal"""
        self.watch_early_signals[signal.ticker] = signal
    
    def get_watch_signals(self) -> List[AnticipationSignal]:
        """Get all WATCH_EARLY signals"""
        return list(self.watch_early_signals.values())
    
    def remove_watch_signal(self, ticker: str):
        """Remove a WATCH_EARLY signal"""
        self.watch_early_signals.pop(ticker, None)
        self.suspects.discard(ticker)


# Global state instance
_state = AnticipationState()


# ============================
# COUCHE 1: IBKR RADAR
# ============================

def run_ibkr_radar(tickers: List[str]) -> List[Anomaly]:
    """
    IBKR Radar: Large coverage anomaly detection
    
    Detects:
    - Volume spikes (>3x average)
    - Price gaps (>3%)
    - Volatility surges
    - Unusual after-hours activity
    
    Returns: List of anomalies
    """
    anomalies = []
    
    logger.info(f"🔍 IBKR RADAR: Scanning {len(tickers)} tickers...")
    
    try:
        if USE_IBKR_DATA:
            from src.ibkr_connector import get_ibkr
            ibkr = get_ibkr()
            
            if ibkr and ibkr.connected:
                anomalies = _scan_with_ibkr(tickers, ibkr)
            else:
                logger.warning("IBKR not connected, using Finnhub")
                anomalies = _scan_with_finnhub(tickers)
        else:
            anomalies = _scan_with_finnhub(tickers)
        
    except Exception as e:
        logger.error(f"IBKR Radar failed: {e}")
        anomalies = _scan_with_finnhub(tickers)
    
    # Add suspects from anomalies
    suspects = [a.ticker for a in anomalies if a.score >= 0.3]
    _state.add_suspects(suspects)
    
    logger.info(f"📊 IBKR RADAR: Found {len(anomalies)} anomalies, {len(suspects)} suspects")
    
    _state.last_radar_scan = datetime.utcnow()
    
    return anomalies


def _scan_with_ibkr(tickers: List[str], ibkr) -> List[Anomaly]:
    """Scan using IBKR data"""
    anomalies = []
    
    for ticker in tickers:
        try:
            quote = ibkr.get_quote(ticker, use_cache=True)
            
            if not quote:
                continue
            
            anomaly = _detect_anomaly_from_quote(ticker, quote, "ibkr")
            
            if anomaly:
                anomalies.append(anomaly)
            
            time.sleep(0.05)  # Small delay
            
        except Exception as e:
            logger.debug(f"IBKR scan error {ticker}: {e}")
            continue
    
    return anomalies


def _scan_with_finnhub(tickers: List[str]) -> List[Anomaly]:
    """Scan using Finnhub data (fallback)"""
    anomalies = []
    
    # C7 FIX: Raised from 100 to 200 (IBKR primary, Finnhub fallback)
    scan_tickers = tickers[:200]
    
    for ticker in scan_tickers:
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker, "token": FINNHUB_API_KEY}
            
            r = safe_get(url, params=params, timeout=5)
            data = r.json()
            
            quote = {
                "last": data.get("c", 0),
                "close": data.get("pc", 0),
                "high": data.get("h", 0),
                "low": data.get("l", 0),
                "volume": data.get("v", 0)
            }
            
            anomaly = _detect_anomaly_from_quote(ticker, quote, "finnhub")
            
            if anomaly:
                anomalies.append(anomaly)
            
            time.sleep(0.2)  # Finnhub rate limit
            
        except Exception as e:
            logger.debug(f"Finnhub scan error {ticker}: {e}")
            continue
    
    return anomalies


def _detect_anomaly_from_quote(ticker: str, quote: dict, source: str) -> Optional[Anomaly]:
    """Analyze quote for anomalies"""
    
    last = quote.get("last", 0)
    close = quote.get("close", 0)
    high = quote.get("high", 0)
    low = quote.get("low", 0)
    volume = quote.get("volume", 0)
    
    if not last or not close or close <= 0:
        return None
    
    # Price filter
    if not (MIN_PRICE <= last <= MAX_PRICE):
        return None
    
    anomaly_type = None
    score = 0.0
    details = {}
    
    # 1. GAP DETECTION (>3%)
    gap_pct = (last - close) / close
    
    if abs(gap_pct) >= 0.03:
        anomaly_type = "GAP"
        score += min(0.5, abs(gap_pct) * 5)  # Max 0.5 from gap
        details["gap_pct"] = round(gap_pct * 100, 2)
    
    # 2. VOLUME SPIKE
    if volume > MIN_AVG_VOLUME * 3:
        vol_ratio = volume / MIN_AVG_VOLUME
        
        if anomaly_type:
            anomaly_type += "+VOLUME"
        else:
            anomaly_type = "VOLUME_SPIKE"
        
        score += min(0.3, vol_ratio / 20)  # Max 0.3 from volume
        details["volume_ratio"] = round(vol_ratio, 2)
    
    # 3. VOLATILITY
    if high > 0 and low > 0:
        intraday_range = (high - low) / low
        
        if intraday_range >= 0.05:
            if anomaly_type:
                anomaly_type += "+VOLATILITY"
            else:
                anomaly_type = "VOLATILITY"
            
            score += min(0.2, intraday_range * 2)  # Max 0.2 from volatility
            details["range_pct"] = round(intraday_range * 100, 2)
    
    # Return if significant
    if anomaly_type and score >= 0.2:
        return Anomaly(
            ticker=ticker,
            anomaly_type=anomaly_type,
            score=round(min(1.0, score), 3),
            details=details,
            source=source,
            timestamp=datetime.utcnow().isoformat()
        )
    
    return None


# ============================
# COUCHE 2: V6.1 INGESTORS (SOURCES RÉELLES)
# ============================

def analyze_with_real_sources(tickers: List[str]) -> List[CatalystEvent]:
    """
    V6.1 Ingestors: Targeted catalyst analysis using REAL sources

    Sources (100% réelles, pas de simulation LLM):
    - SEC EDGAR 8-K filings (gratuit)
    - Finnhub company news

    Returns: List of catalyst events
    """
    if not tickers:
        return []

    logger.info(f"📰 V6.1 INGESTORS: Analyzing {len(tickers)} tickers...")

    events = []

    try:
        # Import V6.1 ingestors
        from src.ingestors.sec_filings_ingestor import SECFilingsIngestor
        from src.ingestors.company_news_scanner import CompanyNewsScanner, ScanPriority

        sec_ingestor = SECFilingsIngestor(set(tickers))
        company_scanner = CompanyNewsScanner()

        # 1. Fetch SEC 8-K filings for these tickers
        import asyncio

        async def fetch_sec_catalysts():
            filings = await sec_ingestor.fetch_8k_filings(hours_back=24)
            return [f for f in filings if f.ticker in tickers]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                sec_filings = []
            else:
                sec_filings = loop.run_until_complete(fetch_sec_catalysts())
        except RuntimeError:
            # No event loop, create new one
            sec_filings = asyncio.run(fetch_sec_catalysts())

        # Convert SEC filings to CatalystEvents
        for filing in sec_filings:
            events.append(CatalystEvent(
                ticker=filing.ticker,
                event_type=filing.event_type or "SEC_8K",
                impact_score=_score_sec_filing(filing),
                headline=f"SEC 8-K: {filing.form_type}",
                summary=filing.description or "",
                source="sec_8k",
                timestamp=filing.filed_date.isoformat() if filing.filed_date else datetime.utcnow().isoformat()
            ))

        # 2. Fetch Finnhub company news for each ticker
        async def fetch_company_news():
            results = []
            # C7 FIX: Raised from 20 to 50 (IBKR has no rate limit,
            # Finnhub limit is managed by api_pool)
            for ticker in tickers[:50]:
                try:
                    result = await company_scanner.scan_company(ticker, ScanPriority.HOT, classify=False)
                    if result.news_items:
                        results.append((ticker, result))
                    await asyncio.sleep(0.5)  # Rate limit
                except Exception as e:
                    logger.debug(f"Company scan error {ticker}: {e}")
            return results

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                company_results = []
            else:
                company_results = loop.run_until_complete(fetch_company_news())
        except RuntimeError:
            company_results = asyncio.run(fetch_company_news())

        # Convert company news to CatalystEvents
        for ticker, result in company_results:
            if result.top_catalyst:
                cat = result.top_catalyst
                events.append(CatalystEvent(
                    ticker=ticker,
                    event_type=cat.event_type or "NEWS",
                    impact_score=cat.event_impact if cat.event_impact else 0.5,
                    headline=cat.headline,
                    summary=cat.summary[:200] if cat.summary else "",
                    source="finnhub_company",
                    timestamp=cat.published_at.isoformat() if cat.published_at else datetime.utcnow().isoformat()
                ))

    except ImportError as e:
        logger.warning(f"V6.1 ingestors not available: {e}")
        # Fallback to basic Finnhub
        events = _fallback_finnhub_news(tickers)
    except Exception as e:
        logger.error(f"V6.1 ingestor error: {e}")
        events = _fallback_finnhub_news(tickers)

    _state.last_grok_scan = datetime.utcnow()

    logger.info(f"📊 V6.1 INGESTORS: Found {len(events)} catalysts")

    return events


def _score_sec_filing(filing) -> float:
    """Score SEC filing impact based on type"""
    # High impact 8-K items
    high_impact = {"FDA_APPROVAL", "MERGER_ACQUISITION", "EARNINGS_BEAT", "MAJOR_CONTRACT"}
    medium_impact = {"MANAGEMENT_CHANGE", "RESTRUCTURING", "GUIDANCE"}

    event_type = filing.event_type or ""

    if event_type in high_impact:
        return 0.8
    elif event_type in medium_impact:
        return 0.6
    else:
        return 0.4


def _fallback_finnhub_news(tickers: List[str]) -> List[CatalystEvent]:
    """Fallback: Direct Finnhub news fetch"""
    events = []

    # C7 FIX: Raised from 10 to 25 (Finnhub fallback)
    for ticker in tickers[:25]:
        try:
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "to": datetime.utcnow().strftime("%Y-%m-%d"),
                "token": FINNHUB_API_KEY
            }

            r = safe_get(url, params=params, timeout=10)
            news = r.json()

            if news and len(news) > 0:
                top_news = news[0]
                events.append(CatalystEvent(
                    ticker=ticker,
                    event_type="NEWS",
                    impact_score=0.5,
                    headline=top_news.get("headline", "")[:100],
                    summary=top_news.get("summary", "")[:200],
                    source="finnhub_fallback",
                    timestamp=datetime.utcnow().isoformat()
                ))

            time.sleep(0.5)

        except Exception as e:
            logger.debug(f"Finnhub fallback error {ticker}: {e}")

    return events


# ============================
# SIGNAL GENERATION
# ============================

def generate_signals(
    anomalies: List[Anomaly],
    catalysts: List[CatalystEvent]
) -> List[AnticipationSignal]:
    """
    Generate WATCH_EARLY signals by combining technical + fundamental
    
    Scoring:
    - Technical (anomaly): 30% weight
    - Fundamental (catalyst): 70% weight
    """
    signals = []
    
    # Build lookups
    anomaly_map = {a.ticker: a for a in anomalies}
    catalyst_map = {c.ticker: c for c in catalysts}
    
    all_tickers = set(anomaly_map.keys()) | set(catalyst_map.keys())
    
    for ticker in all_tickers:
        anomaly = anomaly_map.get(ticker)
        catalyst = catalyst_map.get(ticker)
        
        # Calculate scores
        tech_score = anomaly.score if anomaly else 0
        fund_score = catalyst.impact_score if catalyst else 0
        
        # Combined score (70% fundamental, 30% technical)
        combined = (tech_score * 0.3) + (fund_score * 0.7)
        
        # Need minimum threshold
        if combined < 0.3:
            continue
        
        # Determine signal level
        if combined >= 0.7 and fund_score >= 0.6:
            signal_level = SignalLevel.BUY
        else:
            signal_level = SignalLevel.WATCH_EARLY
        
        # Determine urgency
        if combined >= 0.7:
            urgency = "HIGH"
        elif combined >= 0.5:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        signal = AnticipationSignal(
            ticker=ticker,
            signal_level=signal_level,
            combined_score=round(combined, 3),
            urgency=urgency,
            technical_score=tech_score,
            anomaly_type=anomaly.anomaly_type if anomaly else None,
            fundamental_score=fund_score,
            catalyst_type=catalyst.event_type if catalyst else None,
            catalyst_summary=catalyst.summary if catalyst else "",
            detection_time=datetime.utcnow().isoformat(),
            status="PENDING"
        )
        
        signals.append(signal)
        _state.add_watch_signal(signal)
        
        logger.info(
            f"📡 {signal_level.value}: {ticker} "
            f"(score: {combined:.2f}, urgency: {urgency})"
        )
    
    return signals


def check_signal_upgrades() -> List[AnticipationSignal]:
    """
    Check WATCH_EARLY signals for upgrade to BUY/BUY_STRONG
    
    Upgrade conditions:
    - PM gap forming (>3%)
    - Volume confirmation
    - Price holding/trending up
    """
    upgrades = []
    
    watch_signals = _state.get_watch_signals()
    
    if not watch_signals:
        return upgrades
    
    for signal in watch_signals:
        if signal.signal_level != SignalLevel.WATCH_EARLY:
            continue
        
        try:
            # Get PM metrics
            from src.pm_scanner import compute_pm_metrics
            pm = compute_pm_metrics(signal.ticker)
            
            if not pm:
                continue
            
            # Check upgrade conditions
            gap_ok = pm.get("gap_pct", 0) >= 0.03
            volume_ok = pm.get("pm_liquid", False)
            momentum_ok = pm.get("pm_momentum", 0) > 0
            
            upgrade_score = sum([gap_ok, volume_ok, momentum_ok]) / 3
            
            if upgrade_score >= 0.66:
                # Upgrade signal
                if signal.combined_score >= 0.7 and upgrade_score >= 0.9:
                    signal.signal_level = SignalLevel.BUY_STRONG
                else:
                    signal.signal_level = SignalLevel.BUY
                
                signal.status = "CONFIRMED"
                
                upgrades.append(signal)
                
                logger.info(
                    f"⬆️ UPGRADE: {signal.ticker} → {signal.signal_level.value} "
                    f"(PM gap: {pm.get('gap_pct', 0)*100:.1f}%)"
                )
        
        except Exception as e:
            logger.debug(f"Upgrade check failed {signal.ticker}: {e}")
            continue
    
    return upgrades


# ============================
# MAIN ORCHESTRATION
# ============================

def run_anticipation_scan(universe: List[str], mode: str = "auto") -> Dict:
    """
    Main entry point for anticipation scanning
    
    Modes:
    - auto: Determine based on market hours
    - afterhours: Deep catalyst scan
    - premarket: Confirmation scan
    - intraday: Light monitoring
    
    Returns: Scan results dict
    """
    # Determine mode
    if mode == "auto":
        if is_after_hours():
            mode = "afterhours"
        elif is_premarket():
            mode = "premarket"
        elif is_market_open():
            mode = "intraday"
        else:
            mode = "idle"
    
    logger.info(f"🔄 ANTICIPATION SCAN - Mode: {mode.upper()}")
    
    results = {
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat(),
        "anomalies": [],
        "catalysts": [],
        "new_signals": [],
        "upgrades": [],
        "suspects_count": 0
    }
    
    if mode == "idle":
        logger.info("Market closed, minimal activity")
        return results
    
    # STEP 1: IBKR Radar (always run on large universe)
    logger.info("Step 1: IBKR Radar...")
    anomalies = run_ibkr_radar(universe)
    results["anomalies"] = [asdict(a) for a in anomalies]
    results["suspects_count"] = len(_state.get_suspects())
    
    # STEP 2: V6.1 Ingestors on suspects (SEC + Finnhub)
    suspects = _state.get_suspects()
    
    if mode in ["afterhours", "premarket"] and suspects:
        logger.info(f"Step 2: V6.1 Ingestors on {len(suspects)} suspects...")
        # C7 FIX: Raised from 30 to 80 suspects
        catalysts = analyze_with_real_sources(list(suspects)[:80])
        results["catalysts"] = [asdict(c) for c in catalysts]

    elif mode == "intraday" and suspects:
        # During RTH, only high-priority
        # C7 FIX: Raised from 10 to 30 high-priority tickers
        high_priority = [a.ticker for a in anomalies if a.score >= 0.5][:30]
        if high_priority:
            logger.info(f"Step 2: V6.1 Ingestors (intraday) on {len(high_priority)} high-priority...")
            catalysts = analyze_with_real_sources(high_priority)
            results["catalysts"] = [asdict(c) for c in catalysts]
    else:
        catalysts = []
    
    # STEP 3: Generate signals
    if anomalies or catalysts:
        logger.info("Step 3: Generating signals...")
        new_signals = generate_signals(anomalies, catalysts)
        results["new_signals"] = [asdict(s) for s in new_signals]
    
    # STEP 4: Check upgrades (PM/RTH only)
    if mode in ["premarket", "intraday"]:
        logger.info("Step 4: Checking upgrades...")
        upgrades = check_signal_upgrades()
        results["upgrades"] = [asdict(u) for u in upgrades]
    
    # Summary
    logger.info(
        f"📊 SCAN COMPLETE: "
        f"{len(results['anomalies'])} anomalies, "
        f"{len(results['catalysts'])} catalysts, "
        f"{len(results['new_signals'])} signals, "
        f"{len(results['upgrades'])} upgrades"
    )
    
    return results


# ============================
# PUBLIC API
# ============================

def get_anticipation_engine():
    """Get the state manager (for compatibility)"""
    return _state


def get_watch_early_signals() -> List[AnticipationSignal]:
    """Get current WATCH_EARLY signals"""
    return [s for s in _state.get_watch_signals() 
            if s.signal_level == SignalLevel.WATCH_EARLY]


def get_buy_signals() -> List[AnticipationSignal]:
    """Get current BUY/BUY_STRONG signals"""
    return [s for s in _state.get_watch_signals() 
            if s.signal_level in [SignalLevel.BUY, SignalLevel.BUY_STRONG]]


def get_all_active_signals() -> List[AnticipationSignal]:
    """Get all active signals"""
    return _state.get_watch_signals()


def clear_expired_signals(max_age_hours: int = 24):
    """Remove signals older than max_age_hours"""
    now = datetime.utcnow()
    
    for signal in list(_state.watch_early_signals.values()):
        signal_time = datetime.fromisoformat(signal.detection_time)
        age = (now - signal_time).total_seconds() / 3600
        
        if age > max_age_hours:
            _state.remove_watch_signal(signal.ticker)
            logger.info(f"🧹 Expired: {signal.ticker}")


def get_engine_status() -> Dict:
    """Get engine status"""
    return {
        "suspects_count": len(_state.suspects),
        "watch_signals_count": len(_state.watch_early_signals),
        "grok_calls_last_hour": len(_state.grok_calls),
        "grok_remaining": _state.GROK_CALLS_PER_HOUR - len(_state.grok_calls),
        "last_radar_scan": _state.last_radar_scan.isoformat() if _state.last_radar_scan else None,
        "last_grok_scan": _state.last_grok_scan.isoformat() if _state.last_grok_scan else None
    }


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("ANTICIPATION ENGINE TEST")
    print("=" * 60)
    
    test_universe = ["AAPL", "TSLA", "NVDA", "AMD", "PLTR", "SOFI", "NIO", "LCID"]
    
    print(f"\nTesting with {len(test_universe)} tickers...")
    
    results = run_anticipation_scan(test_universe, mode="afterhours")
    
    print(f"\nResults:")
    print(f"  Mode: {results['mode']}")
    print(f"  Anomalies: {len(results['anomalies'])}")
    print(f"  Catalysts: {len(results['catalysts'])}")
    print(f"  New signals: {len(results['new_signals'])}")
    
    print(f"\nEngine status: {get_engine_status()}")
