"""
USAGE TRACKER V7.0
==================

Tracking temps réel de l'utilisation des clés API.

Responsabilités:
- Comptage des appels (sliding window)
- Tracking de latence
- Détection des rate limits
- Métriques de santé
- Quota disponible en temps réel

Architecture:
- Sliding window pour calls/minute
- Rolling average pour latence
- Counters pour erreurs
- Persistance optionnelle
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from utils.logger import get_logger

logger = get_logger("USAGE_TRACKER")


# ============================================================================
# Configuration
# ============================================================================

# Sliding window sizes
WINDOW_MINUTE = 60          # 60 seconds
WINDOW_HOUR = 3600          # 1 hour
WINDOW_DAY = 86400          # 24 hours

# Health thresholds
ERROR_RATE_WARNING = 0.05   # 5% error rate = warning
ERROR_RATE_CRITICAL = 0.10  # 10% error rate = critical
LATENCY_WARNING_MS = 2000   # 2s = warning
LATENCY_CRITICAL_MS = 5000  # 5s = critical


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CallRecord:
    """Record of a single API call"""
    timestamp: float          # Unix timestamp
    latency_ms: float
    success: bool
    error_type: Optional[str] = None


@dataclass
class KeyMetrics:
    """Aggregated metrics for a key"""
    key_id: str

    # Call counts (computed from window)
    calls_last_minute: int = 0
    calls_last_hour: int = 0
    calls_last_day: int = 0

    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Errors
    errors_last_minute: int = 0
    errors_last_hour: int = 0
    error_rate: float = 0.0
    consecutive_errors: int = 0

    # Rate limits
    rate_limits_hit: int = 0
    last_rate_limit: Optional[datetime] = None

    # Health
    health_score: float = 1.0  # 0.0 - 1.0
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None

    # Quota
    quota_per_minute: int = 60
    quota_remaining: int = 60
    quota_usage_pct: float = 0.0


# ============================================================================
# Usage Tracker
# ============================================================================

class UsageTracker:
    """
    Tracks API key usage in real-time

    Usage:
        tracker = UsageTracker()

        # Record a call
        tracker.record_call("FH_A", latency_ms=125, success=True)

        # Check available quota
        remaining = tracker.get_quota_remaining("FH_A", quota=60)

        # Get metrics
        metrics = tracker.get_metrics("FH_A")
    """

    def __init__(self):
        # Call history per key (sliding window)
        self._calls: Dict[str, Deque[CallRecord]] = {}

        # Quota tracking
        self._quotas: Dict[str, int] = {}  # key_id -> quota_per_minute

        # Consecutive error tracking
        self._consecutive_errors: Dict[str, int] = {}

        # Last success/error timestamps
        self._last_success: Dict[str, datetime] = {}
        self._last_error: Dict[str, str] = {}

        # Rate limit tracking
        self._rate_limits: Dict[str, int] = {}
        self._last_rate_limit: Dict[str, datetime] = {}

        # Thread safety
        self._lock = Lock()

    def set_quota(self, key_id: str, quota_per_minute: int):
        """Set quota for a key"""
        self._quotas[key_id] = quota_per_minute

    def record_call(
        self,
        key_id: str,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """
        Record an API call

        Args:
            key_id: API key identifier
            latency_ms: Call latency in milliseconds
            success: Whether call succeeded
            error_type: Error type if failed (e.g., "RATE_LIMIT", "TIMEOUT")
        """
        with self._lock:
            # Initialize if needed
            if key_id not in self._calls:
                self._calls[key_id] = deque(maxlen=10000)  # Keep last 10k calls

            # Create record
            record = CallRecord(
                timestamp=time.time(),
                latency_ms=latency_ms,
                success=success,
                error_type=error_type
            )

            self._calls[key_id].append(record)

            # Update consecutive errors
            if success:
                self._consecutive_errors[key_id] = 0
                self._last_success[key_id] = datetime.utcnow()
            else:
                self._consecutive_errors[key_id] = self._consecutive_errors.get(key_id, 0) + 1
                self._last_error[key_id] = error_type or "UNKNOWN"

                # Track rate limits specifically
                if error_type == "RATE_LIMIT":
                    self._rate_limits[key_id] = self._rate_limits.get(key_id, 0) + 1
                    self._last_rate_limit[key_id] = datetime.utcnow()

    def get_calls_in_window(
        self,
        key_id: str,
        window_seconds: int
    ) -> List[CallRecord]:
        """Get calls within a time window"""
        if key_id not in self._calls:
            return []

        cutoff = time.time() - window_seconds
        return [c for c in self._calls[key_id] if c.timestamp >= cutoff]

    def get_calls_last_minute(self, key_id: str) -> int:
        """Get call count in last minute"""
        return len(self.get_calls_in_window(key_id, WINDOW_MINUTE))

    def get_calls_last_hour(self, key_id: str) -> int:
        """Get call count in last hour"""
        return len(self.get_calls_in_window(key_id, WINDOW_HOUR))

    def get_quota_remaining(self, key_id: str, quota: int = None) -> int:
        """
        Get remaining quota for this minute

        Args:
            key_id: API key identifier
            quota: Quota per minute (uses stored if not provided)

        Returns:
            Remaining calls allowed this minute
        """
        if quota is None:
            quota = self._quotas.get(key_id, 60)

        calls_this_minute = self.get_calls_last_minute(key_id)
        return max(0, quota - calls_this_minute)

    def get_quota_usage_pct(self, key_id: str, quota: int = None) -> float:
        """Get quota usage percentage"""
        if quota is None:
            quota = self._quotas.get(key_id, 60)

        if quota <= 0:
            return 0.0

        calls_this_minute = self.get_calls_last_minute(key_id)
        return calls_this_minute / quota

    def get_metrics(self, key_id: str, quota: int = None) -> KeyMetrics:
        """
        Get comprehensive metrics for a key

        Args:
            key_id: API key identifier
            quota: Quota per minute (uses stored if not provided)
        """
        if quota is None:
            quota = self._quotas.get(key_id, 60)

        # Get calls in different windows
        calls_minute = self.get_calls_in_window(key_id, WINDOW_MINUTE)
        calls_hour = self.get_calls_in_window(key_id, WINDOW_HOUR)
        calls_day = self.get_calls_in_window(key_id, WINDOW_DAY)

        # Calculate latencies
        latencies = [c.latency_ms for c in calls_hour if c.success]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
        p95_idx = int(len(latencies_sorted) * 0.95)
        p95 = latencies_sorted[p95_idx] if latencies_sorted else 0.0
        max_lat = max(latencies) if latencies else 0.0

        # Calculate errors
        errors_minute = sum(1 for c in calls_minute if not c.success)
        errors_hour = sum(1 for c in calls_hour if not c.success)
        error_rate = errors_hour / len(calls_hour) if calls_hour else 0.0

        # Calculate health score
        health = self._calculate_health_score(
            error_rate=error_rate,
            avg_latency=avg_latency,
            consecutive_errors=self._consecutive_errors.get(key_id, 0)
        )

        # Quota info
        remaining = max(0, quota - len(calls_minute))
        usage_pct = len(calls_minute) / quota if quota > 0 else 0.0

        return KeyMetrics(
            key_id=key_id,
            calls_last_minute=len(calls_minute),
            calls_last_hour=len(calls_hour),
            calls_last_day=len(calls_day),
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            max_latency_ms=max_lat,
            errors_last_minute=errors_minute,
            errors_last_hour=errors_hour,
            error_rate=error_rate,
            consecutive_errors=self._consecutive_errors.get(key_id, 0),
            rate_limits_hit=self._rate_limits.get(key_id, 0),
            last_rate_limit=self._last_rate_limit.get(key_id),
            health_score=health,
            last_success=self._last_success.get(key_id),
            last_error=self._last_error.get(key_id),
            quota_per_minute=quota,
            quota_remaining=remaining,
            quota_usage_pct=usage_pct
        )

    def _calculate_health_score(
        self,
        error_rate: float,
        avg_latency: float,
        consecutive_errors: int
    ) -> float:
        """Calculate health score (0.0 - 1.0)"""
        score = 1.0

        # Penalize error rate
        if error_rate >= ERROR_RATE_CRITICAL:
            score -= 0.5
        elif error_rate >= ERROR_RATE_WARNING:
            score -= 0.2

        # Penalize high latency
        if avg_latency >= LATENCY_CRITICAL_MS:
            score -= 0.3
        elif avg_latency >= LATENCY_WARNING_MS:
            score -= 0.1

        # Penalize consecutive errors
        if consecutive_errors >= 5:
            score -= 0.3
        elif consecutive_errors >= 3:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def is_rate_limited(self, key_id: str, quota: int = None) -> bool:
        """Check if key is currently rate limited"""
        return self.get_quota_remaining(key_id, quota) <= 0

    def is_healthy(self, key_id: str) -> bool:
        """Check if key is healthy"""
        metrics = self.get_metrics(key_id)
        return metrics.health_score >= 0.5

    def get_best_key(
        self,
        key_ids: List[str],
        min_quota: int = 1
    ) -> Optional[str]:
        """
        Get the best key from a list based on health and quota

        Args:
            key_ids: List of key IDs to choose from
            min_quota: Minimum quota required

        Returns:
            Best key ID or None if none available
        """
        candidates = []

        for key_id in key_ids:
            metrics = self.get_metrics(key_id)

            # Skip if not enough quota
            if metrics.quota_remaining < min_quota:
                continue

            # Skip if unhealthy
            if metrics.health_score < 0.3:
                continue

            candidates.append((key_id, metrics))

        if not candidates:
            return None

        # Sort by: health_score DESC, quota_remaining DESC, avg_latency ASC
        candidates.sort(key=lambda x: (
            -x[1].health_score,
            -x[1].quota_remaining,
            x[1].avg_latency_ms
        ))

        return candidates[0][0]

    def get_all_metrics(self) -> Dict[str, KeyMetrics]:
        """Get metrics for all tracked keys"""
        return {key_id: self.get_metrics(key_id) for key_id in self._calls.keys()}

    def reset_key(self, key_id: str):
        """Reset tracking for a key"""
        with self._lock:
            if key_id in self._calls:
                self._calls[key_id].clear()
            self._consecutive_errors.pop(key_id, None)
            self._rate_limits.pop(key_id, None)

    def cleanup_old_data(self, max_age_seconds: int = WINDOW_DAY):
        """Remove old call records"""
        cutoff = time.time() - max_age_seconds

        with self._lock:
            for key_id in self._calls:
                # Filter out old records
                self._calls[key_id] = deque(
                    (c for c in self._calls[key_id] if c.timestamp >= cutoff),
                    maxlen=10000
                )


# ============================================================================
# Convenience Functions
# ============================================================================

_tracker_instance = None
_tracker_lock = Lock()  # S4-1 FIX: thread-safe singleton


def get_tracker() -> UsageTracker:
    """Get singleton tracker instance"""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = UsageTracker()
    return _tracker_instance


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "UsageTracker",
    "KeyMetrics",
    "CallRecord",
    "get_tracker",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("USAGE TRACKER TEST")
    print("=" * 60)

    tracker = UsageTracker()
    tracker.set_quota("FH_A", 60)
    tracker.set_quota("FH_B", 60)

    # Simulate calls
    print("\nSimulating API calls...")

    for i in range(50):
        # FH_A: mostly successful, low latency
        tracker.record_call(
            "FH_A",
            latency_ms=100 + random.randint(0, 100),
            success=random.random() > 0.05  # 5% error rate
        )

        # FH_B: some issues
        success = random.random() > 0.15  # 15% error rate
        tracker.record_call(
            "FH_B",
            latency_ms=200 + random.randint(0, 300),
            success=success,
            error_type="RATE_LIMIT" if not success and random.random() > 0.5 else None
        )

    # Get metrics
    print("\nMetrics for FH_A:")
    metrics_a = tracker.get_metrics("FH_A")
    print(f"  Calls/min: {metrics_a.calls_last_minute}")
    print(f"  Avg latency: {metrics_a.avg_latency_ms:.1f}ms")
    print(f"  Error rate: {metrics_a.error_rate*100:.1f}%")
    print(f"  Health: {metrics_a.health_score:.2f}")
    print(f"  Quota remaining: {metrics_a.quota_remaining}/{metrics_a.quota_per_minute}")

    print("\nMetrics for FH_B:")
    metrics_b = tracker.get_metrics("FH_B")
    print(f"  Calls/min: {metrics_b.calls_last_minute}")
    print(f"  Avg latency: {metrics_b.avg_latency_ms:.1f}ms")
    print(f"  Error rate: {metrics_b.error_rate*100:.1f}%")
    print(f"  Health: {metrics_b.health_score:.2f}")
    print(f"  Rate limits hit: {metrics_b.rate_limits_hit}")

    # Test best key selection
    best = tracker.get_best_key(["FH_A", "FH_B"])
    print(f"\nBest key: {best}")
