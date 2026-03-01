"""
KEY REGISTRY V7.0
=================

Gestion centralisée des clés API.

Responsabilités:
- Stockage sécurisé des clés
- Configuration par provider/tier/role
- État de santé des clés
- Persistance et chargement

Providers supportés:
- Finnhub (news, quotes)
- Grok/xAI (NLP classification)
- Reddit (social buzz)
- StockTwits (social buzz)

Architecture:
- Clés chargées depuis variables d'environnement
- Configuration depuis YAML ou dict
- État persisté en SQLite
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

from utils.logger import get_logger

logger = get_logger("KEY_REGISTRY")


# ============================================================================
# Configuration
# ============================================================================

REGISTRY_DB = "data/api_keys.db"

# Default quotas by provider and tier
DEFAULT_QUOTAS = {
    "finnhub": {
        "free": {"per_minute": 60, "per_day": None},
        "starter": {"per_minute": 300, "per_day": None},
        "premium": {"per_minute": 600, "per_day": None},
    },
    "grok": {
        "free": {"per_minute": 10, "per_day": 100},
        "starter": {"per_minute": 60, "per_day": 1000},
        "premium": {"per_minute": 200, "per_day": None},
    },
    "groq": {
        "free": {"per_minute": 30, "per_day": 14400},
        "starter": {"per_minute": 100, "per_day": None},
    },
    "reddit": {
        "free": {"per_minute": 60, "per_day": 1000},
    },
    "stocktwits": {
        "free": {"per_minute": 200, "per_day": None},
    },
}


# ============================================================================
# Enums
# ============================================================================

class KeyStatus(Enum):
    """API key status"""
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    ERROR = "error"


class TaskRole(Enum):
    """Task roles for key assignment"""
    # Critical - dedicated keys
    CRITICAL = "CRITICAL"
    HOT_TICKERS = "HOT_TICKERS"
    PRE_HALT = "PRE_HALT"

    # Standard
    COMPANY_NEWS = "COMPANY_NEWS"
    GLOBAL_NEWS = "GLOBAL_NEWS"
    NLP_CLASSIFY = "NLP_CLASSIFY"
    SOCIAL_BUZZ = "SOCIAL_BUZZ"

    # Background
    BATCH = "BATCH"
    BACKUP = "BACKUP"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class APIKeyConfig:
    """Configuration for an API key"""
    # Identity
    id: str                            # Unique ID: "FH_A", "GROK_1", etc.
    provider: str                      # "finnhub", "grok", "reddit", "stocktwits"
    key: str                           # Actual API key (from env var)

    # Tier and quotas
    tier: str = "free"                 # "free", "starter", "premium"
    quota_per_minute: int = 60
    quota_per_day: Optional[int] = None

    # Role assignment
    roles: List[str] = field(default_factory=list)
    priority: int = 1                  # 1 = highest priority for role

    # State
    enabled: bool = True
    status: KeyStatus = KeyStatus.ACTIVE
    cooldown_until: Optional[datetime] = None

    # Metadata
    account_email: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def is_available(self) -> bool:
        """Check if key is available for use"""
        if not self.enabled:
            return False
        if self.status in [KeyStatus.DISABLED, KeyStatus.ERROR]:
            return False
        if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
            return False
        return True

    def has_role(self, role: str) -> bool:
        """Check if key has a specific role"""
        return role in self.roles or "ALL" in self.roles

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding sensitive key)"""
        return {
            "id": self.id,
            "provider": self.provider,
            "tier": self.tier,
            "quota_per_minute": self.quota_per_minute,
            "quota_per_day": self.quota_per_day,
            "roles": self.roles,
            "priority": self.priority,
            "enabled": self.enabled,
            "status": self.status.value,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }


# ============================================================================
# Key Registry
# ============================================================================

class KeyRegistry:
    """
    Central registry for API keys

    Usage:
        registry = KeyRegistry()

        # Register keys
        registry.register_key(APIKeyConfig(
            id="FH_A",
            provider="finnhub",
            key=os.environ["FINNHUB_KEY_A"],
            tier="free",
            roles=["HOT_TICKERS", "COMPANY_NEWS"],
            priority=1
        ))

        # Get keys for a task
        keys = registry.get_keys_for_role("finnhub", "HOT_TICKERS")

        # Update key state
        registry.set_cooldown("FH_A", seconds=60)
    """

    def __init__(self, db_path: str = REGISTRY_DB):
        self.db_path = db_path
        self._keys: Dict[str, APIKeyConfig] = {}
        self._by_provider: Dict[str, List[str]] = {}

        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialize database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS key_state (
                key_id TEXT PRIMARY KEY,
                status TEXT,
                cooldown_until TEXT,
                last_error TEXT,
                error_count INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def _load_state(self):
        """Load key states from database"""
        cursor = self.conn.execute("SELECT * FROM key_state")
        for row in cursor.fetchall():
            key_id = row[0]
            if key_id in self._keys:
                key = self._keys[key_id]
                key.status = KeyStatus(row[1]) if row[1] else KeyStatus.ACTIVE
                key.cooldown_until = datetime.fromisoformat(row[2]) if row[2] else None

    def register_key(self, config: APIKeyConfig):
        """Register an API key"""
        if not config.key:
            logger.warning(f"Key {config.id} has no API key value, skipping")
            return

        # Apply default quotas if not specified
        if config.provider in DEFAULT_QUOTAS:
            tier_quotas = DEFAULT_QUOTAS[config.provider].get(config.tier, {})
            if config.quota_per_minute == 60:  # Default value
                config.quota_per_minute = tier_quotas.get("per_minute", 60)
            if config.quota_per_day is None:
                config.quota_per_day = tier_quotas.get("per_day")

        self._keys[config.id] = config

        # Index by provider
        if config.provider not in self._by_provider:
            self._by_provider[config.provider] = []
        if config.id not in self._by_provider[config.provider]:
            self._by_provider[config.provider].append(config.id)

        logger.info(f"Registered key {config.id} for {config.provider} ({config.tier})")

    def register_from_env(self, configs: List[Dict]):
        """
        Register keys from environment variables

        Example config:
        [
            {
                "id": "FH_A",
                "provider": "finnhub",
                "env_var": "FINNHUB_KEY_A",
                "tier": "free",
                "roles": ["HOT_TICKERS", "COMPANY_NEWS"]
            }
        ]
        """
        for cfg in configs:
            env_var = cfg.get("env_var", "")
            key_value = os.environ.get(env_var, "")

            if not key_value:
                logger.warning(f"Environment variable {env_var} not set for key {cfg.get('id')}")
                continue

            config = APIKeyConfig(
                id=cfg.get("id", ""),
                provider=cfg.get("provider", ""),
                key=key_value,
                tier=cfg.get("tier", "free"),
                roles=cfg.get("roles", []),
                priority=cfg.get("priority", 1),
                account_email=cfg.get("account_email", ""),
                notes=cfg.get("notes", "")
            )
            self.register_key(config)

    def get_key(self, key_id: str) -> Optional[APIKeyConfig]:
        """Get a specific key by ID"""
        return self._keys.get(key_id)

    def get_keys(self, provider: str) -> List[APIKeyConfig]:
        """Get all keys for a provider"""
        key_ids = self._by_provider.get(provider, [])
        return [self._keys[kid] for kid in key_ids if kid in self._keys]

    def get_available_keys(self, provider: str) -> List[APIKeyConfig]:
        """Get available (non-cooldown) keys for a provider"""
        return [k for k in self.get_keys(provider) if k.is_available()]

    def get_keys_for_role(
        self,
        provider: str,
        role: str,
        only_available: bool = True
    ) -> List[APIKeyConfig]:
        """Get keys assigned to a specific role"""
        keys = self.get_keys(provider)

        # Filter by role
        role_keys = [k for k in keys if k.has_role(role)]

        # Filter by availability
        if only_available:
            role_keys = [k for k in role_keys if k.is_available()]

        # Sort by priority
        role_keys.sort(key=lambda k: k.priority)

        return role_keys

    def set_cooldown(self, key_id: str, seconds: int = 60):
        """Set a key into cooldown"""
        if key_id not in self._keys:
            return

        key = self._keys[key_id]
        key.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        key.status = KeyStatus.COOLDOWN

        self._save_state(key_id)
        logger.info(f"Key {key_id} in cooldown for {seconds}s")

    def clear_cooldown(self, key_id: str):
        """Clear cooldown for a key"""
        if key_id not in self._keys:
            return

        key = self._keys[key_id]
        key.cooldown_until = None
        key.status = KeyStatus.ACTIVE

        self._save_state(key_id)

    def set_status(self, key_id: str, status: KeyStatus, error: str = ""):
        """Set key status"""
        if key_id not in self._keys:
            return

        key = self._keys[key_id]
        key.status = status

        self._save_state(key_id, error=error)

    def enable_key(self, key_id: str):
        """Enable a key"""
        if key_id in self._keys:
            self._keys[key_id].enabled = True
            self._keys[key_id].status = KeyStatus.ACTIVE
            self._save_state(key_id)

    def disable_key(self, key_id: str):
        """Disable a key"""
        if key_id in self._keys:
            self._keys[key_id].enabled = False
            self._keys[key_id].status = KeyStatus.DISABLED
            self._save_state(key_id)

    def _save_state(self, key_id: str, error: str = ""):
        """Save key state to database"""
        if key_id not in self._keys:
            return

        key = self._keys[key_id]

        self.conn.execute("""
            INSERT OR REPLACE INTO key_state
            (key_id, status, cooldown_until, last_error, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            key_id,
            key.status.value,
            key.cooldown_until.isoformat() if key.cooldown_until else None,
            error,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def get_status(self) -> Dict:
        """Get registry status"""
        status = {
            "total_keys": len(self._keys),
            "by_provider": {},
        }

        for provider, key_ids in self._by_provider.items():
            keys = [self._keys[kid] for kid in key_ids if kid in self._keys]
            status["by_provider"][provider] = {
                "total": len(keys),
                "available": sum(1 for k in keys if k.is_available()),
                "in_cooldown": sum(1 for k in keys if k.status == KeyStatus.COOLDOWN),
                "disabled": sum(1 for k in keys if not k.enabled),
            }

        return status

    def list_keys(self) -> List[Dict]:
        """List all keys (without sensitive data)"""
        return [k.to_dict() for k in self._keys.values()]


# ============================================================================
# Convenience Functions
# ============================================================================

_registry_instance = None
_registry_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_registry() -> KeyRegistry:
    """Get singleton registry instance"""
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = KeyRegistry()
    return _registry_instance


def setup_default_keys():
    """
    Setup default keys from environment variables

    Expected env vars:
    - FINNHUB_API_KEY (or FINNHUB_KEY_A, FINNHUB_KEY_B, etc.)
    - XAI_API_KEY (or GROK_KEY_A, etc.)
    - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
    """
    registry = get_registry()

    # Finnhub keys
    # FH_A: WebSocket dedicated — NOT in REST rotation (WS_ONLY role)
    # FH_B → HOT REST (priority 1), FH_C → standard REST (priority 2)
    # FH_D → batch (priority 3), FH_E → automatic backup (priority 4)
    # FH_MAIN → legacy fallback for safe_get() calls with hardcoded token (priority 5)
    finnhub_envs = [
        ("FINNHUB_API_KEY", "FH_MAIN", ["ALL"],                          5),
        ("FINNHUB_KEY_A",   "FH_A",    ["WS_ONLY"],                      0),  # WebSocket only
        ("FINNHUB_KEY_B",   "FH_B",    ["HOT_TICKERS", "COMPANY_NEWS"],  1),
        ("FINNHUB_KEY_C",   "FH_C",    ["GLOBAL_NEWS", "PRE_HALT"],      2),
        ("FINNHUB_KEY_D",   "FH_D",    ["BATCH"],                        3),
        ("FINNHUB_KEY_E",   "FH_E",    ["BACKUP", "ALL"],                4),
    ]

    for env_var, key_id, roles, prio in finnhub_envs:
        key_value = os.environ.get(env_var)
        if key_value:
            registry.register_key(APIKeyConfig(
                id=key_id,
                provider="finnhub",
                key=key_value,
                tier="free",
                roles=roles,
                priority=prio
            ))

    # Groq (groq.com) — principal NLP (Llama 3.3 70B, 14 400 req/jour free)
    groq_envs = [
        ("GROQ_API_KEY",   "GROQ_MAIN", ["ALL", "NLP_CLASSIFY"], 1),
        ("GROQ_KEY_B",     "GROQ_B",    ["NLP_CLASSIFY"],         2),
    ]
    for env_var, key_id, roles, prio in groq_envs:
        key_value = os.environ.get(env_var)
        if key_value:
            registry.register_key(APIKeyConfig(
                id=key_id,
                provider="groq",
                key=key_value,
                tier="free",
                roles=roles,
                priority=prio
            ))

    # Grok/xAI keys — fallback NLP
    # GROK_A → CRITICAL/HOT NLP (priority 1)
    # GROK_B → STANDARD NLP (priority 2)
    # GROK_C → BATCH NLP (priority 3)
    # GROK_MAIN → legacy fallback (priority 4)
    grok_envs = [
        ("XAI_API_KEY",  "GROK_MAIN", ["ALL"],                      4),
        ("GROK_KEY_A",   "GROK_A",    ["NLP_CLASSIFY", "CRITICAL"],  1),
        ("GROK_KEY_B",   "GROK_B",    ["NLP_CLASSIFY"],              2),
        ("GROK_KEY_C",   "GROK_C",    ["BATCH"],                     3),
    ]

    for env_var, key_id, roles, prio in grok_envs:
        key_value = os.environ.get(env_var)
        if key_value:
            registry.register_key(APIKeyConfig(
                id=key_id,
                provider="grok",
                key=key_value,
                tier="starter",
                roles=roles,
                priority=prio
            ))

    logger.info(f"Default keys setup complete: {registry.get_status()}")
    return registry


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "KeyRegistry",
    "APIKeyConfig",
    "KeyStatus",
    "TaskRole",
    "get_registry",
    "setup_default_keys",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KEY REGISTRY TEST")
    print("=" * 60)

    registry = KeyRegistry(db_path="data/test_keys.db")

    # Register test keys
    registry.register_key(APIKeyConfig(
        id="FH_TEST_A",
        provider="finnhub",
        key="test_key_a",
        tier="free",
        roles=["HOT_TICKERS", "COMPANY_NEWS"],
        priority=1
    ))

    registry.register_key(APIKeyConfig(
        id="FH_TEST_B",
        provider="finnhub",
        key="test_key_b",
        tier="free",
        roles=["GLOBAL_NEWS", "BATCH"],
        priority=2
    ))

    registry.register_key(APIKeyConfig(
        id="GROK_TEST",
        provider="grok",
        key="test_grok_key",
        tier="starter",
        roles=["NLP_CLASSIFY"],
        priority=1
    ))

    print("\nRegistered keys:")
    for key in registry.list_keys():
        print(f"  {key['id']}: {key['provider']} ({key['tier']}) - {key['roles']}")

    print("\nKeys for HOT_TICKERS role:")
    for key in registry.get_keys_for_role("finnhub", "HOT_TICKERS"):
        print(f"  {key.id} (priority: {key.priority})")

    # Test cooldown
    registry.set_cooldown("FH_TEST_A", seconds=30)
    print(f"\nAfter cooldown on FH_TEST_A:")
    print(f"  Available: {[k.id for k in registry.get_available_keys('finnhub')]}")

    print(f"\nRegistry status: {registry.get_status()}")
