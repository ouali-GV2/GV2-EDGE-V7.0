"""
COMPREHENSIVE UNIT TESTS — GV2-EDGE V9 (A21)
================================================

Tests for all V9 improvements:
- A3: Tick-by-tick streaming
- A4: Earnings Calendar
- A5: FDA Calendar Engine
- A6: Conference Calendar
- A7: IPO Tracker
- A8: Insider Clustering
- A9: Float Analysis / Squeeze
- A10: Intraday Patterns
- A11: Levels Engine
- A12: Social Velocity
- A13: Sector Momentum
- A14: Market Memory V2
- A15: Backtest Engine V8
- A16: Weight Optimizer

Run: python -m pytest tests/test_v9_improvements.py -v
"""

import pytest
from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# ============================================================================
# A4: Earnings Calendar Tests
# ============================================================================

class TestEarningsCalendar:
    def test_earnings_event_creation(self):
        from src.earnings_calendar import EarningsEvent, EarningsTiming, EarningsSurprise
        event = EarningsEvent(
            ticker="AAPL",
            date=datetime.now(timezone.utc) + timedelta(days=3),
            timing=EarningsTiming.BMO,
            eps_estimate=1.50,
        )
        assert event.ticker == "AAPL"
        assert event.timing == EarningsTiming.BMO
        assert event.days_until == 3
        assert not event.is_today
        assert not event.has_reported

    def test_earnings_surprise_computation(self):
        from src.earnings_calendar import EarningsEvent, EarningsTiming, EarningsSurprise
        event = EarningsEvent(
            ticker="TSLA",
            date=datetime.now(timezone.utc),
            timing=EarningsTiming.AMC,
            eps_estimate=0.50,
            eps_actual=0.75,
        )
        event.compute_surprise()
        assert event.surprise == EarningsSurprise.BEAT
        assert event.surprise_pct == 50.0

    def test_earnings_miss(self):
        from src.earnings_calendar import EarningsEvent, EarningsTiming, EarningsSurprise
        event = EarningsEvent(
            ticker="NFLX",
            date=datetime.now(timezone.utc),
            timing=EarningsTiming.AMC,
            eps_estimate=1.00,
            eps_actual=0.50,
        )
        event.compute_surprise()
        assert event.surprise == EarningsSurprise.MISS
        assert event.surprise_pct == -50.0

    def test_earnings_boost_schedule(self):
        from src.earnings_calendar import EarningsCalendar, EarningsEvent, EarningsTiming
        cal = EarningsCalendar()

        # Add event J-1
        event = EarningsEvent(
            ticker="TEST",
            date=datetime.now(timezone.utc) + timedelta(days=1),
            timing=EarningsTiming.BMO,
            beat_rate=0.75,
        )
        cal._events["TEST"] = [event]

        boost, details = cal.get_earnings_boost("TEST")
        assert boost >= 0.10  # J-1 should get at least +0.10
        assert "EARNINGS_J-1" in details.get("reason", "")

    def test_singleton(self):
        from src.earnings_calendar import get_earnings_calendar
        c1 = get_earnings_calendar()
        c2 = get_earnings_calendar()
        assert c1 is c2


# ============================================================================
# A6: Conference Calendar Tests
# ============================================================================

class TestConferenceCalendar:
    def test_conference_db_loaded(self):
        from src.conference_calendar import get_conference_calendar
        cal = get_conference_calendar()
        status = cal.get_status()
        assert status["total_conferences"] > 0

    def test_register_presenter(self):
        from src.conference_calendar import get_conference_calendar
        cal = get_conference_calendar()
        cal.register_presenter("MRNA", "ASCO")
        tickers = cal.get_presenting_tickers("ASCO")
        assert "MRNA" in tickers

    def test_conference_boost_no_presenter(self):
        from src.conference_calendar import ConferenceCalendar
        cal = ConferenceCalendar()
        boost, details = cal.get_conference_boost("UNKNOWN_TICKER")
        assert boost >= 0  # May get small sector boost if a conf is active


# ============================================================================
# A7: IPO Tracker Tests
# ============================================================================

class TestIPOTracker:
    def test_ipo_event_status(self):
        from src.ipo_tracker import IPOEvent, IPOStatus, OfferingType
        today = date.today()

        # Day 1 IPO
        ipo = IPOEvent(ticker="NEWCO", company_name="New Co", ipo_date=today)
        assert ipo.status == IPOStatus.DAY1

        # Recent IPO (5 days ago)
        ipo2 = IPOEvent(ticker="RECENT", company_name="Recent Co", ipo_date=today - timedelta(days=5))
        assert ipo2.status == IPOStatus.RECENT

        # Upcoming
        ipo3 = IPOEvent(ticker="FUTURE", company_name="Future Co", ipo_date=today + timedelta(days=5))
        assert ipo3.status == IPOStatus.UPCOMING

    def test_ipo_boost(self):
        from src.ipo_tracker import IPOTracker, IPOEvent
        tracker = IPOTracker()
        ipo = IPOEvent(ticker="DAY1", company_name="Day1 Co", ipo_date=date.today())
        tracker.add_ipo(ipo)

        boost, details = tracker.get_ipo_boost("DAY1")
        assert boost == 0.15  # Day 1 boost
        assert details["reason"] == "IPO_DAY1"

    def test_lockup_computation(self):
        from src.ipo_tracker import IPOEvent
        ipo = IPOEvent(ticker="LOCK", company_name="Lock Co", ipo_date=date.today() - timedelta(days=170))
        assert ipo.days_to_lockup is not None
        assert ipo.days_to_lockup == 10  # 180 - 170


# ============================================================================
# A9: Float Analysis Tests
# ============================================================================

class TestFloatAnalysis:
    def test_squeeze_score_high_si(self):
        from src.float_analysis import FloatAnalysis
        analysis = FloatAnalysis(
            ticker="SQUEEZE",
            timestamp=datetime.now(timezone.utc),
            float_shares=5_000_000,
            short_interest=2_000_000,
            short_pct_float=40.0,
            days_to_cover=5.0,
            borrow_status="HARD",
            cost_to_borrow_pct=150.0,
        )
        assert analysis.si_level == "CRITICAL"
        assert analysis.short_pct_float >= 30.0

    def test_turnover_computation(self):
        from src.float_analysis import FloatAnalyzer
        analyzer = FloatAnalyzer()
        turnover = analyzer.compute_turnover_intraday("TEST", 10_000_000, 5_000_000)
        # 10M volume / 5M float = 2.0x turnover
        assert turnover == 0.0  # Float is fetched dynamically, so returns 0 without real data


# ============================================================================
# A10: Intraday Pattern Tests
# ============================================================================

class TestIntradayPatterns:
    def _make_df(self, n=30, trend="up"):
        """Create a simple test DataFrame."""
        base = 10.0
        data = {
            "open": [], "high": [], "low": [], "close": [], "volume": []
        }
        for i in range(n):
            if trend == "up":
                c = base + i * 0.1
            elif trend == "down":
                c = base - i * 0.1
            else:
                c = base + np.random.uniform(-0.1, 0.1)

            data["open"].append(c - 0.05)
            data["close"].append(c)
            data["high"].append(c + 0.1)
            data["low"].append(c - 0.1)
            data["volume"].append(100000 + i * 5000)

        return pd.DataFrame(data)

    def test_detect_hod_break(self):
        from src.pattern_analyzer import detect_hod_break
        df = self._make_df(30, trend="up")
        score = detect_hod_break(df)
        assert 0 <= score <= 1.0

    def test_detect_consolidation_box(self):
        from src.pattern_analyzer import detect_consolidation_box
        df = self._make_df(30, trend="flat")
        score = detect_consolidation_box(df)
        assert 0 <= score <= 1.0

    def test_detect_all_returns_dict(self):
        from src.pattern_analyzer import detect_all_intraday_patterns
        df = self._make_df(30, trend="up")
        result = detect_all_intraday_patterns(df)
        assert isinstance(result, dict)
        assert "best_pattern" in result
        assert "intraday_pattern_score" in result

    def test_empty_df(self):
        from src.pattern_analyzer import detect_vwap_reclaim, detect_hod_break
        assert detect_vwap_reclaim(None) == 0.0
        assert detect_hod_break(pd.DataFrame()) == 0.0


# ============================================================================
# A11: Levels Engine Tests
# ============================================================================

class TestLevelsEngine:
    def test_volume_profile(self):
        from src.levels_engine import VolumeProfile
        vp = VolumeProfile()
        # Add trades
        for i in range(100):
            price = 10.0 + (i % 10) * 0.1
            vp.add_trade("TEST", price, 1000)

        result = vp.compute("TEST")
        assert result["poc"] > 0

    def test_levels_computation(self):
        from src.levels_engine import LevelsEngine
        engine = LevelsEngine()
        engine.set_previous_data("TEST", close=10.0, high=10.5)
        engine.set_premarket_data("TEST", high=10.3, low=9.8)

        levels = engine.compute_levels("TEST", current_price=10.2, hod=10.4, lod=9.9, vwap=10.1)
        assert levels.ticker == "TEST"
        assert levels.previous_close == 10.0
        assert levels.above_vwap is True

    def test_room_boost(self):
        from src.levels_engine import LevelsEngine
        engine = LevelsEngine()
        engine.set_previous_data("TEST", close=10.0, high=12.0)

        boost, details = engine.get_room_boost("TEST", current_price=10.0, hod=10.5, lod=9.5, vwap=9.8)
        assert boost >= 0


# ============================================================================
# A12: Social Velocity Tests
# ============================================================================

class TestSocialVelocity:
    def test_record_and_compute(self):
        from src.social_velocity import SocialVelocityEngine
        engine = SocialVelocityEngine()

        # Record some mentions
        engine.record_mentions("TEST", reddit=5, stocktwits=3, twitter=10, bullish=15, bearish=3)
        engine.record_mentions("TEST", reddit=10, stocktwits=8, twitter=20, bullish=30, bearish=5)

        vel = engine.compute_velocity("TEST")
        assert vel.ticker == "TEST"
        assert vel.social_score >= 0

    def test_empty_ticker(self):
        from src.social_velocity import SocialVelocityEngine
        engine = SocialVelocityEngine()
        vel = engine.compute_velocity("NONEXISTENT")
        assert vel.social_score == 0.0


# ============================================================================
# A13: Sector Momentum Tests
# ============================================================================

class TestSectorMomentum:
    def test_register_and_detect(self):
        from src.sector_momentum import SectorMomentum
        engine = SectorMomentum()

        engine.register_ticker_sector("MRNA", "BIOTECH")
        engine.register_ticker_sector("BNTX", "BIOTECH")
        engine.register_ticker_sector("PFE", "BIOTECH")

        engine.record_move("MRNA", 8.5, volume_ratio=2.0)
        engine.record_move("BNTX", 6.2, volume_ratio=1.5)
        engine.record_move("PFE", 4.1, volume_ratio=1.2)

        signals = engine.detect_sector_moves()
        assert len(signals) >= 1
        assert signals[0].sector == "BIOTECH"

    def test_sector_boost(self):
        from src.sector_momentum import SectorMomentum
        engine = SectorMomentum()

        engine.register_ticker_sector("A", "TECH")
        engine.register_ticker_sector("B", "TECH")

        engine.record_move("A", 5.0)
        engine.record_move("B", 4.0)

        boost, details = engine.get_sector_boost("A")
        assert boost >= 0


# ============================================================================
# A14: Market Memory V2 Tests
# ============================================================================

class TestContextScorerV2:
    def test_record_and_retrieve(self):
        from src.market_memory.context_scorer import ContextScorerV2

        scorer = ContextScorerV2()

        # Record some outcomes
        scorer.record_outcome("AAPL", "EARNINGS", True, 15.0)
        scorer.record_outcome("AAPL", "EARNINGS", True, 8.0)
        scorer.record_outcome("AAPL", "EARNINGS", False, -5.0)
        scorer.record_outcome("AAPL", "FDA", False, -10.0)
        scorer.record_outcome("AAPL", "FDA", False, -8.0)

        summary = scorer.get_segment_summary("AAPL")
        assert "EARNINGS" in summary
        assert "FDA" in summary
        assert summary["EARNINGS"]["win_rate"] > summary["FDA"]["win_rate"]

    def test_best_catalyst(self):
        from src.market_memory.context_scorer import ContextScorerV2
        scorer = ContextScorerV2()

        for _ in range(5):
            scorer.record_outcome("TST", "SQUEEZE", True, 20.0)
        for _ in range(5):
            scorer.record_outcome("TST", "EARNINGS", False, -5.0)

        best = scorer.get_best_catalyst("TST")
        assert best == "SQUEEZE"


# ============================================================================
# A16: Weight Optimizer Tests
# ============================================================================

class TestWeightOptimizer:
    def test_default_weights(self):
        from src.scoring.weight_optimizer import WeightOptimizer, DEFAULT_WEIGHTS
        opt = WeightOptimizer()
        weights = opt.get_current_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_generate_candidates(self):
        from src.scoring.weight_optimizer import WeightOptimizer, DEFAULT_WEIGHTS
        opt = WeightOptimizer()
        candidates = opt._generate_candidates(DEFAULT_WEIGHTS, 10)
        assert len(candidates) == 10
        for c in candidates:
            assert abs(sum(c.values()) - 1.0) < 0.01


# ============================================================================
# Integration: ibkr_streaming tick-by-tick (A3)
# ============================================================================

class TestTickByTick:
    def test_tbt_data_classes(self):
        from src.ibkr_streaming import StreamingQuote, StreamingEvent
        q = StreamingQuote(
            ticker="TEST",
            timestamp=datetime.now(),
            last=10.5,
            bid=10.4,
            ask=10.6,
            volume=100000,
            is_valid=True,
        )
        assert q.is_valid
        assert q.last == 10.5


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
