"""
POOL MANAGER V7.0
=================

Gestionnaire central du pool de clés API.

Point d'entrée unique pour toutes les requêtes API.
Orchestre Registry, Tracker et Router.

Usage:
    pool = get_pool_manager()

    # Get a key for a request
    async with pool.acquire("finnhub", "COMPANY_NEWS", Priority.HIGH) as key:
        response = await fetch(url, headers={"X-Finnhub-Token": key})

    # Or manual usage
    key_info = pool.get_key("finnhub", "COMPANY_NEWS")
    if key_info:
        response = await fetch(url, headers={"X-Finnhub-Token": key_info.key})
        pool.release(key_info.key_id, success=True, latency_ms=125)

Features:
- Automatic key selection
- Usage tracking
- Cooldown management
- Health monitoring
- Graceful degradation
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

from utils.logger import get_logger

from .key_registry import (
    KeyRegistry,
    APIKeyConfig,
    KeyStatus,
    get_registry,
    setup_default_keys
)
from .usage_tracker import UsageTracker, KeyMetrics, get_tracker
from .request_router import RequestRouter, RoutingResult, Priority

logger = get_logger("POOL_MANAGER")


# ============================================================================
# Configuration
# ============================================================================

# Default cooldown on rate limit (seconds)
DEFAULT_RATE_LIMIT_COOLDOWN = 60

# Default cooldown on error (seconds)
DEFAULT_ERROR_COOLDOWN = 30

# Consecutive errors before longer cooldown
CONSECUTIVE_ERROR_THRESHOLD = 3
EXTENDED_COOLDOWN = 300  # 5 minutes


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KeyAcquisition:
    """Result of acquiring a key"""
    success: bool
    key_id: Optional[str] = None
    key: Optional[str] = None          # Actual API key value
    provider: Optional[str] = None

    # Failure info
    reason: Optional[str] = None
    retry_after: Optional[int] = None

    # Tracking
    acquired_at: datetime = None

    def __post_init__(self):
        if self.acquired_at is None:
            self.acquired_at = datetime.utcnow()


# ============================================================================
# Pool Manager
# ============================================================================

class APIPoolManager:
    """
    Central API key pool manager

    Usage:
        pool = APIPoolManager()
        pool.setup()  # Load keys from environment

        # Acquire and use a key
        key_info = pool.get_key("finnhub", "COMPANY_NEWS", Priority.HIGH)
        if key_info.success:
            try:
                response = api_call(key_info.key)
                pool.release(key_info.key_id, success=True, latency_ms=120)
            except RateLimitError:
                pool.release(key_info.key_id, success=False, error="RATE_LIMIT")
            except Exception as e:
                pool.release(key_info.key_id, success=False, error=str(e))

        # Or use context manager
        async with pool.acquire("finnhub", "COMPANY_NEWS") as key:
            if key:
                response = await api_call(key)
    """

    def __init__(self):
        self.registry: Optional[KeyRegistry] = None
        self.tracker: Optional[UsageTracker] = None
        self.router: Optional[RequestRouter] = None

        self._initialized = False
        self._active_acquisitions: Dict[str, datetime] = {}

    def setup(self, auto_load_keys: bool = True):
        """
        Initialize the pool manager

        Args:
            auto_load_keys: Whether to load keys from environment variables
        """
        if self._initialized:
            return

        self.registry = get_registry()
        self.tracker = get_tracker()

        if auto_load_keys:
            setup_default_keys()

        self.router = RequestRouter(self.registry, self.tracker)

        # Sync quotas from registry to tracker
        for key in self.registry.list_keys():
            key_config = self.registry.get_key(key["id"])
            if key_config:
                self.tracker.set_quota(key["id"], key_config.quota_per_minute)

        self._initialized = True
        logger.info("Pool manager initialized")

    def _ensure_initialized(self):
        """Ensure manager is initialized"""
        if not self._initialized:
            self.setup()

    def register_key(self, config: APIKeyConfig):
        """Register a new API key"""
        self._ensure_initialized()
        self.registry.register_key(config)
        self.tracker.set_quota(config.id, config.quota_per_minute)

    def get_key(
        self,
        provider: str,
        task_type: str = None,
        priority: Priority = Priority.STANDARD
    ) -> KeyAcquisition:
        """
        Get an API key for a request

        Args:
            provider: API provider name
            task_type: Type of task/role
            priority: Request priority

        Returns:
            KeyAcquisition with key info or failure reason
        """
        self._ensure_initialized()

        # Route to best key
        result = self.router.route(provider, task_type, priority)

        if not result.success:
            return KeyAcquisition(
                success=False,
                reason=result.reason,
                retry_after=result.retry_after_seconds
            )

        # Track acquisition
        self._active_acquisitions[result.key_id] = datetime.utcnow()

        return KeyAcquisition(
            success=True,
            key_id=result.key_id,
            key=result.key.key,
            provider=provider
        )

    def release(
        self,
        key_id: str,
        success: bool,
        latency_ms: float = 0,
        error: str = None
    ):
        """
        Release a key after use and record metrics

        Args:
            key_id: Key identifier
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            error: Error type if failed (e.g., "RATE_LIMIT", "TIMEOUT")
        """
        self._ensure_initialized()

        # Record usage
        self.tracker.record_call(key_id, latency_ms, success, error)

        # Remove from active
        self._active_acquisitions.pop(key_id, None)

        # Handle errors
        if not success:
            self._handle_error(key_id, error)

    def _handle_error(self, key_id: str, error: str):
        """Handle key error with appropriate cooldown"""

        key = self.registry.get_key(key_id)
        if not key:
            return

        metrics = self.tracker.get_metrics(key_id)

        # Rate limit error
        if error == "RATE_LIMIT":
            self.registry.set_cooldown(key_id, DEFAULT_RATE_LIMIT_COOLDOWN)
            logger.warning(f"Key {key_id} rate limited, cooldown {DEFAULT_RATE_LIMIT_COOLDOWN}s")
            return

        # Consecutive errors
        if metrics.consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
            self.registry.set_cooldown(key_id, EXTENDED_COOLDOWN)
            self.registry.set_status(key_id, KeyStatus.DEGRADED, error)
            logger.warning(f"Key {key_id} degraded after {metrics.consecutive_errors} consecutive errors")
            return

        # Single error - short cooldown
        if error:
            self.registry.set_cooldown(key_id, DEFAULT_ERROR_COOLDOWN)

    @asynccontextmanager
    async def acquire(
        self,
        provider: str,
        task_type: str = None,
        priority: Priority = Priority.STANDARD,
        timeout: float = 5.0
    ):
        """
        Context manager for acquiring a key

        Usage:
            async with pool.acquire("finnhub", "COMPANY_NEWS") as key:
                if key:
                    response = await fetch(url, headers={"Token": key})

        Args:
            provider: API provider name
            task_type: Type of task/role
            priority: Request priority
            timeout: Max time to wait for a key

        Yields:
            API key string or None if unavailable
        """
        self._ensure_initialized()

        key_info = None
        start_time = time.time()

        # Try to acquire, with retry on failure
        while (time.time() - start_time) < timeout:
            key_info = self.get_key(provider, task_type, priority)

            if key_info.success:
                break

            if key_info.retry_after and key_info.retry_after > 0:
                wait_time = min(key_info.retry_after, timeout - (time.time() - start_time))
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            else:
                break  # No point retrying

        if not key_info or not key_info.success:
            yield None
            return

        # Track timing
        acquire_start = time.time()

        try:
            yield key_info.key
        except Exception as e:
            # Record error
            latency = (time.time() - acquire_start) * 1000
            error_type = "RATE_LIMIT" if "rate" in str(e).lower() else "ERROR"
            self.release(key_info.key_id, success=False, latency_ms=latency, error=error_type)
            raise
        else:
            # Record success (caller should call release with actual latency)
            # If not released by caller, we release here with estimated latency
            if key_info.key_id in self._active_acquisitions:
                latency = (time.time() - acquire_start) * 1000
                self.release(key_info.key_id, success=True, latency_ms=latency)

    def get_status(self) -> Dict[str, Any]:
        """Get pool status"""
        self._ensure_initialized()

        return {
            "initialized": self._initialized,
            "registry": self.registry.get_status(),
            "router": self.router.get_stats(),
            "active_acquisitions": len(self._active_acquisitions),
            "keys": self._get_keys_summary()
        }

    def _get_keys_summary(self) -> Dict[str, Any]:
        """Get summary of all keys"""
        summary = {}

        for key_config in self.registry.list_keys():
            key_id = key_config["id"]
            metrics = self.tracker.get_metrics(key_id)

            summary[key_id] = {
                "provider": key_config["provider"],
                "status": key_config["status"],
                "health": metrics.health_score,
                "quota_remaining": metrics.quota_remaining,
                "calls_last_hour": metrics.calls_last_hour,
                "error_rate": metrics.error_rate
            }

        return summary

    def get_metrics(self, key_id: str) -> Optional[KeyMetrics]:
        """Get metrics for a specific key"""
        self._ensure_initialized()
        key = self.registry.get_key(key_id)
        if not key:
            return None
        return self.tracker.get_metrics(key_id, key.quota_per_minute)

    def get_all_metrics(self) -> Dict[str, KeyMetrics]:
        """Get metrics for all keys"""
        self._ensure_initialized()
        return self.tracker.get_all_metrics()

    def disable_key(self, key_id: str):
        """Disable a key"""
        self._ensure_initialized()
        self.registry.disable_key(key_id)

    def enable_key(self, key_id: str):
        """Enable a key"""
        self._ensure_initialized()
        self.registry.enable_key(key_id)

    def clear_cooldown(self, key_id: str):
        """Clear cooldown for a key"""
        self._ensure_initialized()
        self.registry.clear_cooldown(key_id)


# ============================================================================
# Singleton Instance
# ============================================================================

_pool_instance = None
_pool_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_pool_manager() -> APIPoolManager:
    """Get singleton pool manager instance"""
    global _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            _pool_instance = APIPoolManager()
    return _pool_instance


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "APIPoolManager",
    "KeyAcquisition",
    "Priority",
    "get_pool_manager",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import os

    # Set test environment variables
    os.environ["FINNHUB_API_KEY"] = "test_finnhub_key"
    os.environ["XAI_API_KEY"] = "test_grok_key"

    async def test():
        print("=" * 60)
        print("POOL MANAGER TEST")
        print("=" * 60)

        pool = APIPoolManager()
        pool.setup()

        # Test key acquisition
        print("\n--- Test 1: Get key ---")
        key_info = pool.get_key("finnhub", "COMPANY_NEWS", Priority.HIGH)
        print(f"  Success: {key_info.success}")
        print(f"  Key ID: {key_info.key_id}")
        print(f"  Has key: {bool(key_info.key)}")

        if key_info.success:
            # Simulate usage
            pool.release(key_info.key_id, success=True, latency_ms=150)
            print("  Released successfully")

        # Test context manager
        print("\n--- Test 2: Context manager ---")
        async with pool.acquire("finnhub", "GLOBAL_NEWS") as key:
            if key:
                print(f"  Acquired key: {key[:10]}...")
            else:
                print("  No key available")

        # Test rate limit handling
        print("\n--- Test 3: Rate limit simulation ---")
        key_info = pool.get_key("finnhub")
        if key_info.success:
            pool.release(key_info.key_id, success=False, error="RATE_LIMIT")
            print(f"  Key {key_info.key_id} should be in cooldown")

            # Check status
            status = pool.get_status()
            print(f"  Key status: {status['keys'].get(key_info.key_id, {}).get('status')}")

        # Pool status
        print("\n--- Pool Status ---")
        status = pool.get_status()
        print(f"  Registry: {status['registry']}")
        print(f"  Router stats: {status['router']}")

    asyncio.run(test())
