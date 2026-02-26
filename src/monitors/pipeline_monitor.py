"""
PIPELINE MONITOR V6.1
=====================

Monitoring santé du pipeline de news/catalysts.

Responsabilités:
- Health checks des APIs (SEC, Finnhub, Reddit, StockTwits)
- Métriques de performance (latence, throughput)
- Alertes automatiques (Slack/Discord webhook)
- Dashboard status

Métriques trackées:
- API response times
- Rate limit usage
- Error rates
- News flow volume
- Catalyst detection rate

Alertes:
- API down > 5 min
- Error rate > 10%
- News flow drop > 80%
- Rate limit > 90%

Architecture:
- Background monitoring thread
- SQLite for metrics persistence
- Webhook notifications
- Status endpoint for dashboard
"""

import asyncio
import aiohttp
import sqlite3
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger
from config import FINNHUB_API_KEY

logger = get_logger("PIPELINE_MONITOR")


# ============================
# Configuration
# ============================

MONITOR_DB = "data/pipeline_monitor.db"
METRICS_RETENTION_DAYS = 7

# Check intervals (seconds)
HEALTH_CHECK_INTERVAL = 300  # 5 min
METRICS_FLUSH_INTERVAL = 60  # 1 min

# Alert thresholds
ALERT_ERROR_RATE = 0.10  # 10%
ALERT_LATENCY_MS = 5000  # 5 sec
ALERT_NEWS_DROP = 0.80   # 80% drop
ALERT_RATE_LIMIT = 0.90  # 90% used

# API endpoints for health checks
HEALTH_ENDPOINTS = {
    "sec_edgar": "https://www.sec.gov/cgi-bin/browse-edgar",
    "finnhub": "https://finnhub.io/api/v1/quote?symbol=AAPL",
    "stocktwits": "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
}


# ============================
# Enums
# ============================

class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================
# Data Classes
# ============================

@dataclass
class ComponentHealth:
    """Health status of a component"""
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    last_check: datetime = None
    last_success: datetime = None
    error_count: int = 0
    error_message: str = ""


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    timestamp: datetime
    # Throughput
    news_fetched: int = 0
    news_filtered: int = 0
    catalysts_detected: int = 0
    # Latency
    avg_fetch_latency_ms: float = 0.0
    avg_classify_latency_ms: float = 0.0
    # Errors
    api_errors: int = 0
    classify_errors: int = 0
    # Rate limits
    finnhub_remaining: int = 0
    sec_remaining: int = 0


@dataclass
class Alert:
    """Monitor alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict = None
    acknowledged: bool = False


# ============================
# Pipeline Monitor
# ============================

class PipelineMonitor:
    """
    Monitors pipeline health and performance

    Usage:
        monitor = PipelineMonitor()
        monitor.set_webhook("https://hooks.slack.com/...")
        await monitor.start()

        # Record metrics
        monitor.record_fetch(items=50, latency_ms=120)
        monitor.record_error("finnhub", "Rate limit exceeded")

        # Check health
        health = await monitor.check_health()
        print(health)
    """

    def __init__(self):
        self._session = None
        self._running = False
        self._webhook_url = None

        # Component health
        self._component_health: Dict[str, ComponentHealth] = {}

        # Current metrics buffer
        self._metrics_buffer = PipelineMetrics(timestamp=datetime.utcnow())

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Alerts history
        self._alerts: List[Alert] = []
        self._alert_counter = 0

        # Initialize DB
        self._init_db()

    def _init_db(self):
        """Initialize metrics database"""
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(MONITOR_DB, check_same_thread=False)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TEXT PRIMARY KEY,
                news_fetched INTEGER,
                news_filtered INTEGER,
                catalysts_detected INTEGER,
                avg_fetch_latency_ms REAL,
                avg_classify_latency_ms REAL,
                api_errors INTEGER,
                classify_errors INTEGER
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                level TEXT,
                component TEXT,
                message TEXT,
                details TEXT,
                acknowledged INTEGER
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS health_history (
                timestamp TEXT,
                component TEXT,
                status TEXT,
                latency_ms REAL,
                error_message TEXT,
                PRIMARY KEY (timestamp, component)
            )
        """)

        self.conn.commit()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close resources"""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        self.conn.close()

    def set_webhook(self, url: str):
        """Set webhook URL for alerts"""
        self._webhook_url = url
        logger.info(f"Webhook configured: {url[:30]}...")

    def on_alert(self, callback: Callable):
        """Register alert callback"""
        self._alert_callbacks.append(callback)

    # ============================
    # Health Checks
    # ============================

    async def check_health(self) -> Dict[str, ComponentHealth]:
        """Run health checks on all components"""
        logger.debug("Running health checks...")

        tasks = []
        for name, url in HEALTH_ENDPOINTS.items():
            tasks.append(self._check_endpoint(name, url))

        await asyncio.gather(*tasks, return_exceptions=True)

        return self._component_health.copy()

    async def _check_endpoint(self, name: str, url: str):
        """Check single endpoint health"""
        start = datetime.utcnow()

        # Add API key for Finnhub
        if name == "finnhub" and FINNHUB_API_KEY:
            url = f"{url}&token={FINNHUB_API_KEY}"

        try:
            session = await self._get_session()

            async with session.get(url, timeout=10) as resp:
                latency = (datetime.utcnow() - start).total_seconds() * 1000

                if resp.status == 200:
                    status = HealthStatus.HEALTHY
                elif resp.status == 429:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.DEGRADED

                health = ComponentHealth(
                    name=name,
                    status=status,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    last_success=datetime.utcnow() if status == HealthStatus.HEALTHY else None
                )

        except asyncio.TimeoutError:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.DOWN,
                latency_ms=10000,
                last_check=datetime.utcnow(),
                error_message="Timeout"
            )

        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.DOWN,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

        # Update stored health
        prev = self._component_health.get(name)
        self._component_health[name] = health

        # Save to history
        self._save_health(health)

        # Check for status change
        if prev and prev.status != health.status:
            if health.status == HealthStatus.DOWN:
                await self._create_alert(
                    AlertLevel.ERROR,
                    name,
                    f"{name} is DOWN: {health.error_message}"
                )
            elif health.status == HealthStatus.HEALTHY and prev.status == HealthStatus.DOWN:
                await self._create_alert(
                    AlertLevel.INFO,
                    name,
                    f"{name} recovered"
                )

    def _save_health(self, health: ComponentHealth):
        """Save health to history"""
        self.conn.execute("""
            INSERT OR REPLACE INTO health_history
            (timestamp, component, status, latency_ms, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            health.name,
            health.status.value,
            health.latency_ms,
            health.error_message
        ))
        self.conn.commit()

    # ============================
    # Metrics Recording
    # ============================

    def record_fetch(self, items: int, latency_ms: float):
        """Record fetch operation"""
        self._metrics_buffer.news_fetched += items
        # Rolling average
        if self._metrics_buffer.avg_fetch_latency_ms == 0:
            self._metrics_buffer.avg_fetch_latency_ms = latency_ms
        else:
            self._metrics_buffer.avg_fetch_latency_ms = (
                self._metrics_buffer.avg_fetch_latency_ms * 0.9 + latency_ms * 0.1
            )

    def record_filter(self, items_in: int, items_out: int):
        """Record filter operation"""
        self._metrics_buffer.news_filtered += items_out

    def record_classify(self, catalysts: int, latency_ms: float):
        """Record classification operation"""
        self._metrics_buffer.catalysts_detected += catalysts
        if self._metrics_buffer.avg_classify_latency_ms == 0:
            self._metrics_buffer.avg_classify_latency_ms = latency_ms
        else:
            self._metrics_buffer.avg_classify_latency_ms = (
                self._metrics_buffer.avg_classify_latency_ms * 0.9 + latency_ms * 0.1
            )

    def record_error(self, component: str, error: str):
        """Record an error"""
        self._metrics_buffer.api_errors += 1

        # Update component health
        if component in self._component_health:
            self._component_health[component].error_count += 1
            self._component_health[component].error_message = error

        logger.warning(f"Error recorded for {component}: {error}")

    def record_rate_limit(self, component: str, remaining: int, limit: int):
        """Record rate limit status"""
        if component == "finnhub":
            self._metrics_buffer.finnhub_remaining = remaining
        elif component == "sec":
            self._metrics_buffer.sec_remaining = remaining

        # Alert if near limit
        usage = 1 - (remaining / limit) if limit > 0 else 1
        if usage >= ALERT_RATE_LIMIT:
            asyncio.create_task(self._create_alert(
                AlertLevel.WARNING,
                component,
                f"Rate limit {usage*100:.0f}% used ({remaining} remaining)"
            ))

    # ============================
    # Alerts
    # ============================

    async def _create_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        details: Dict = None
    ):
        """Create and dispatch an alert"""
        self._alert_counter += 1
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            timestamp=datetime.utcnow(),
            level=level,
            component=component,
            message=message,
            details=details
        )

        self._alerts.append(alert)

        # Save to DB
        self.conn.execute("""
            INSERT INTO alerts (id, timestamp, level, component, message, details, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.timestamp.isoformat(),
            alert.level.value,
            alert.component,
            alert.message,
            json.dumps(alert.details) if alert.details else None,
            0
        ))
        self.conn.commit()

        # Log
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(level, logger.info)

        log_fn(f"[ALERT] {component}: {message}")

        # Callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Webhook
        if self._webhook_url and level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            await self._send_webhook(alert)

    async def _send_webhook(self, alert: Alert):
        """Send alert to webhook"""
        try:
            session = await self._get_session()

            payload = {
                "text": f"*[{alert.level.value.upper()}]* {alert.component}\n{alert.message}",
                "attachments": [{
                    "color": {
                        AlertLevel.INFO: "#36a64f",
                        AlertLevel.WARNING: "#ffcc00",
                        AlertLevel.ERROR: "#ff0000",
                        AlertLevel.CRITICAL: "#8b0000"
                    }.get(alert.level, "#808080"),
                    "fields": [
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }]
            }

            async with session.post(self._webhook_url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    logger.warning(f"Webhook failed: {resp.status}")

        except Exception as e:
            logger.error(f"Webhook error: {e}")

    # ============================
    # Main Loop
    # ============================

    async def start(self):
        """Start background monitoring"""
        self._running = True
        logger.info("Pipeline monitor starting...")

        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_flush_loop())

    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                await self.check_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _metrics_flush_loop(self):
        """Background metrics flush loop"""
        while self._running:
            await asyncio.sleep(METRICS_FLUSH_INTERVAL)

            try:
                self._flush_metrics()
            except Exception as e:
                logger.error(f"Metrics flush error: {e}")

    def _flush_metrics(self):
        """Flush metrics buffer to database"""
        metrics = self._metrics_buffer

        self.conn.execute("""
            INSERT OR REPLACE INTO metrics
            (timestamp, news_fetched, news_filtered, catalysts_detected,
             avg_fetch_latency_ms, avg_classify_latency_ms, api_errors, classify_errors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp.isoformat(),
            metrics.news_fetched,
            metrics.news_filtered,
            metrics.catalysts_detected,
            metrics.avg_fetch_latency_ms,
            metrics.avg_classify_latency_ms,
            metrics.api_errors,
            metrics.classify_errors
        ))
        self.conn.commit()

        # Reset buffer
        self._metrics_buffer = PipelineMetrics(timestamp=datetime.utcnow())

        # Cleanup old data
        cutoff = (datetime.utcnow() - timedelta(days=METRICS_RETENTION_DAYS)).isoformat()
        self.conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
        self.conn.execute("DELETE FROM health_history WHERE timestamp < ?", (cutoff,))
        self.conn.commit()

    # ============================
    # Status & Reports
    # ============================

    def get_status(self) -> Dict:
        """Get current pipeline status"""
        health_summary = {}
        for name, health in self._component_health.items():
            health_summary[name] = {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "last_check": health.last_check.isoformat() if health.last_check else None
            }

        # Recent alerts
        recent_alerts = [
            {
                "id": a.id,
                "level": a.level.value,
                "component": a.component,
                "message": a.message,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self._alerts[-10:]
        ]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": self._get_overall_status().value,
            "components": health_summary,
            "metrics": {
                "news_fetched": self._metrics_buffer.news_fetched,
                "catalysts_detected": self._metrics_buffer.catalysts_detected,
                "avg_fetch_latency_ms": self._metrics_buffer.avg_fetch_latency_ms,
                "api_errors": self._metrics_buffer.api_errors
            },
            "recent_alerts": recent_alerts
        }

    def _get_overall_status(self) -> HealthStatus:
        """Determine overall pipeline status"""
        if not self._component_health:
            return HealthStatus.UNKNOWN

        statuses = [h.status for h in self._component_health.values()]

        if any(s == HealthStatus.DOWN for s in statuses):
            return HealthStatus.DOWN
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get historical metrics"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metrics WHERE timestamp >= ? ORDER BY timestamp
        """, (cutoff,))

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ============================
# Convenience Functions
# ============================

_monitor_instance = None
_monitor_lock = threading.Lock()  # S4-1 FIX: thread-safe singleton


def get_monitor() -> PipelineMonitor:
    """Get singleton monitor instance"""
    global _monitor_instance
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = PipelineMonitor()
    return _monitor_instance


async def quick_health_check() -> Dict:
    """Quick health check"""
    monitor = get_monitor()
    await monitor.check_health()
    return monitor.get_status()


# ============================
# Module exports
# ============================

__all__ = [
    "PipelineMonitor",
    "PipelineMetrics",
    "ComponentHealth",
    "Alert",
    "HealthStatus",
    "AlertLevel",
    "get_monitor",
    "quick_health_check",
]


# ============================
# Test
# ============================

if __name__ == "__main__":
    async def test():
        monitor = PipelineMonitor()

        # Register callback
        monitor.on_alert(lambda a: print(f"ALERT: {a.level.value} - {a.message}"))

        print("=" * 60)
        print("PIPELINE MONITOR TEST")
        print("=" * 60)

        # Run health check
        print("\nRunning health checks...")
        health = await monitor.check_health()

        for name, h in health.items():
            print(f"  {name}: {h.status.value} ({h.latency_ms:.0f}ms)")

        # Record some metrics
        monitor.record_fetch(50, 120)
        monitor.record_filter(50, 35)
        monitor.record_classify(5, 450)
        monitor.record_error("test", "Test error")

        # Get status
        print("\nCurrent status:")
        status = monitor.get_status()
        print(f"  Overall: {status['overall_status']}")
        print(f"  Metrics: {status['metrics']}")

        await monitor.close()

    asyncio.run(test())
