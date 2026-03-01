import time
import threading
import collections
import requests
from functools import wraps
from urllib.parse import urlparse

from utils.logger import get_logger

logger = get_logger("API_GUARD")
api_log = get_logger("API_MONITOR")   # → data/logs/api_monitor.log

# ============================
# Grok token bucket (8 req/min — Grok free tier = 10/min)
# ============================
_grok_tb_lock = threading.Lock()
_grok_tb_timestamps: collections.deque = collections.deque()
_GROK_TB_MAX    = 8     # max requests per window
_GROK_TB_WINDOW = 60.0  # sliding window in seconds


def _grok_token_bucket_wait(timeout: float = 70.0) -> bool:
    """
    Block until a Grok API slot is available (max `timeout` seconds).
    Returns True if a slot was acquired, False if timeout expired.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _grok_tb_lock:
            now = time.time()
            # Evict timestamps outside the sliding window
            while _grok_tb_timestamps and _grok_tb_timestamps[0] < now - _GROK_TB_WINDOW:
                _grok_tb_timestamps.popleft()
            if len(_grok_tb_timestamps) < _GROK_TB_MAX:
                _grok_tb_timestamps.append(now)
                return True
            # Calculate wait until oldest slot expires
            wait = _GROK_TB_WINDOW - (now - _grok_tb_timestamps[0]) + 0.05
        time.sleep(min(wait, 1.0))
    return False


# ============================
# HELPERS
# ============================

def _short_url(url: str) -> str:
    """Return domain+path only — no query params (hides API keys)."""
    try:
        p = urlparse(url)
        return f"{p.netloc}{p.path}"
    except Exception:
        return url[:80]


def _infer_provider(url: str) -> str:
    """Infer provider name from URL."""
    u = url.lower()
    if "finnhub" in u:    return "finnhub"
    if "sec.gov" in u:    return "sec"
    if "x.ai" in u or "grok" in u: return "grok"
    if "reddit" in u:     return "reddit"
    if "stocktwits" in u: return "stocktwits"
    if "yahoo" in u:      return "yahoo"
    if "ibkr" in u or "interactivebrokers" in u: return "ibkr"
    return "http"


def _log_api(method: str, url: str, status: int, latency_ms: float,
             provider: str = "", key_id: str = "", error: str = "", note: str = ""):
    """
    Write one structured line to api_monitor.log.

    Format:
      METHOD | PROVIDER | endpoint | STATUS_TAG | 42ms [| key=XX] [| note]
    """
    endpoint = _short_url(url)
    prov = provider or _infer_provider(url)

    if error:
        tag = f"ERR={error[:60]}"
    elif status == 429:
        tag = "RATE_LIMIT(429)"
    elif status == 403:
        tag = "FORBIDDEN(403)"
    elif status >= 500:
        tag = f"SERVER_ERR({status})"
    elif status >= 400:
        tag = f"CLIENT_ERR({status})"
    elif status == 0:
        tag = "TIMEOUT/CONN"
    else:
        tag = f"OK({status})"

    parts = [method, prov, endpoint, tag, f"{latency_ms:.0f}ms"]
    if key_id:
        parts.append(f"key={key_id}")
    if note:
        parts.append(note)

    msg = " | ".join(parts)

    if error or status >= 500:
        api_log.error(msg)
    elif status in (429, 403) or (status >= 400 and status != 200):
        api_log.warning(msg)
    else:
        api_log.info(msg)


# ============================
# DECORATOR
# ============================

class APIGuardException(Exception):
    pass


def safe_api_call(
    retries=3,
    timeout=10,
    backoff=2,
    allowed_exceptions=(requests.exceptions.RequestException,)
):
    """
    Decorator for safe API calls with retry + exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait = 1

            while attempt < retries:
                try:
                    return func(*args, **kwargs)

                except allowed_exceptions as e:
                    attempt += 1
                    logger.warning(
                        f"API error on {func.__name__} attempt {attempt}/{retries}: {e}"
                    )
                    if attempt >= retries:
                        logger.error(f"API failed permanently: {func.__name__}")
                        raise APIGuardException(str(e))
                    time.sleep(wait)
                    wait *= backoff

                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise

        return wrapper
    return decorator


# ============================
# SIMPLE SAFE REQUEST (legacy — single key, backoff on error)
# ============================

@safe_api_call()
def safe_get(url, params=None, headers=None, timeout=10):
    t0 = time.time()
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        latency = (time.time() - t0) * 1000
        _log_api("GET", url, r.status_code, latency)
        return r
    except Exception as e:
        latency = (time.time() - t0) * 1000
        _log_api("GET", url, 0, latency, error=str(e)[:80])
        raise


@safe_api_call()
def safe_post(url, json=None, headers=None, timeout=10):
    t0 = time.time()
    try:
        r = requests.post(url, json=json, headers=headers, timeout=timeout)
        latency = (time.time() - t0) * 1000
        _log_api("POST", url, r.status_code, latency)
        return r
    except Exception as e:
        latency = (time.time() - t0) * 1000
        _log_api("POST", url, 0, latency, error=str(e)[:80])
        raise


# ============================
# POOL-AWARE SAFE REQUEST
# On HTTP 429 → release key as RATE_LIMIT → switch to next key immediately (no sleep)
# Falls back to safe_get/safe_post if pool is unavailable or all keys exhausted
# ============================

def pool_safe_get(
    url,
    params=None,
    headers=None,
    timeout=10,
    provider="finnhub",
    task_type=None,
    priority=None,
    max_switches=5,
):
    """
    Pool-aware GET request.

    On HTTP 429 → release key as RATE_LIMIT → switch to next key immediately (no sleep).
    Falls back to safe_get() if pool is unavailable or all keys are exhausted.

    Key injection:
      - Finnhub: params["token"] = key  (query param)
      - Grok/xAI: headers["Authorization"] = "Bearer <key>"

    Args:
        url:          Request URL (do NOT include ?token= — pool injects it)
        params:       Query params without the API token
        headers:      HTTP headers without Authorization
        timeout:      Request timeout in seconds
        provider:     "finnhub" or "grok"
        task_type:    Pool role (e.g. "COMPANY_NEWS", "NLP_CLASSIFY")
        priority:     Priority enum value (defaults to STANDARD)
        max_switches: Max key rotations before fallback to default key
    """
    try:
        from src.api_pool.pool_manager import get_pool_manager
        from src.api_pool.request_router import Priority as P
        pool = get_pool_manager()
    except Exception:
        # Pool unavailable — use legacy single-key path
        return safe_get(url, params=params, headers=headers, timeout=timeout)

    if priority is None:
        from src.api_pool.request_router import Priority as P
        priority = P.STANDARD

    for attempt in range(max_switches):
        acq = pool.get_key(provider, task_type, priority)

        if not acq.success:
            logger.warning(
                f"Pool: no key available for {provider}/{task_type} (attempt {attempt + 1})"
            )
            api_log.warning(
                f"GET | {provider} | {_short_url(url)} | NO_KEY | 0ms | attempt={attempt+1}"
            )
            break

        merged_params = dict(params or {})
        merged_headers = dict(headers or {})

        if provider == "finnhub":
            merged_params["token"] = acq.key
        elif provider in ("grok", "xai"):
            merged_headers["Authorization"] = f"Bearer {acq.key}"

        t0 = time.time()
        try:
            r = requests.get(url, params=merged_params, headers=merged_headers, timeout=timeout)
            latency_ms = (time.time() - t0) * 1000

            if r.status_code == 429:
                logger.warning(
                    f"Pool: 429 on {acq.key_id} for {provider}/{task_type}"
                    f" — switching key (attempt {attempt + 1}/{max_switches})"
                )
                _log_api("GET", url, 429, latency_ms, provider=provider,
                         key_id=acq.key_id, note=f"key_switch {attempt+1}/{max_switches}")
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="RATE_LIMIT")
                continue  # No sleep — immediate retry with next key

            if r.status_code == 403:
                _log_api("GET", url, 403, latency_ms, provider=provider, key_id=acq.key_id)
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="FORBIDDEN")
                break  # 403 = permanent for this key — no retry

            _log_api("GET", url, r.status_code, latency_ms,
                     provider=provider, key_id=acq.key_id)
            pool.release(acq.key_id, success=True, latency_ms=latency_ms)
            return r

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            _log_api("GET", url, 0, latency_ms, provider=provider,
                     key_id=acq.key_id, error=str(e)[:80])
            pool.release(acq.key_id, success=False, latency_ms=latency_ms, error=str(e)[:100])
            raise

    # All key attempts exhausted — fallback to legacy safe_get
    logger.warning(
        f"Pool: all {max_switches} key attempts exhausted for {provider}"
        f" — fallback to default key"
    )
    api_log.warning(
        f"GET | {provider} | {_short_url(url)} | POOL_EXHAUSTED | 0ms | fallback"
    )
    return safe_get(url, params=params, headers=headers, timeout=timeout)


def pool_safe_post(
    url,
    json=None,
    headers=None,
    timeout=10,
    provider="grok",
    task_type=None,
    priority=None,
    max_switches=3,
):
    """
    Pool-aware POST request.

    On HTTP 429 → release key as RATE_LIMIT → switch to next key immediately (no sleep).
    Falls back to safe_post() if pool is unavailable or all keys are exhausted.

    Key injection:
      - Grok/xAI: headers["Authorization"] = "Bearer <key>"
      - Finnhub: headers["X-Finnhub-Token"] = key

    Args:
        url:          Request URL
        json:         JSON request body
        headers:      HTTP headers without Authorization (pool injects Bearer token)
        timeout:      Request timeout in seconds
        provider:     "grok" or "finnhub"
        task_type:    Pool role (e.g. "NLP_CLASSIFY", "CRITICAL")
        priority:     Priority enum value (defaults to STANDARD)
        max_switches: Max key rotations before fallback to default key
    """
    try:
        from src.api_pool.pool_manager import get_pool_manager
        from src.api_pool.request_router import Priority as P
        pool = get_pool_manager()
    except Exception:
        return safe_post(url, json=json, headers=headers, timeout=timeout)

    if priority is None:
        from src.api_pool.request_router import Priority as P
        priority = P.STANDARD

    # Grok token bucket — enforce 8 req/min (Groq a son propre circuit breaker dans _call_groq)
    if provider in ("grok", "xai"):
        if not _grok_token_bucket_wait(timeout=70.0):
            api_log.warning(
                f"POST | {provider} | {_short_url(url)} | TOKEN_BUCKET_TIMEOUT | 0ms"
            )
            return safe_post(url, json=json, headers=headers, timeout=timeout)

    for attempt in range(max_switches):
        acq = pool.get_key(provider, task_type, priority)

        if not acq.success:
            logger.warning(
                f"Pool: no key available for {provider}/{task_type} (attempt {attempt + 1})"
            )
            api_log.warning(
                f"POST | {provider} | {_short_url(url)} | NO_KEY | 0ms | attempt={attempt+1}"
            )
            break

        merged_headers = dict(headers or {})

        if provider in ("grok", "xai", "groq"):
            merged_headers["Authorization"] = f"Bearer {acq.key}"
        elif provider == "finnhub":
            merged_headers["X-Finnhub-Token"] = acq.key

        t0 = time.time()
        try:
            r = requests.post(url, json=json, headers=merged_headers, timeout=timeout)
            latency_ms = (time.time() - t0) * 1000

            if r.status_code == 429:
                logger.warning(
                    f"Pool: 429 on {acq.key_id} for {provider}/{task_type}"
                    f" — switching key (attempt {attempt + 1}/{max_switches})"
                )
                _log_api("POST", url, 429, latency_ms, provider=provider,
                         key_id=acq.key_id, note=f"key_switch {attempt+1}/{max_switches}")
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="RATE_LIMIT")
                continue  # No sleep — immediate retry with next key

            if r.status_code == 403:
                _log_api("POST", url, 403, latency_ms, provider=provider, key_id=acq.key_id)
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="FORBIDDEN")
                break  # 403 = permanent for this key — no retry

            _log_api("POST", url, r.status_code, latency_ms,
                     provider=provider, key_id=acq.key_id)
            pool.release(acq.key_id, success=True, latency_ms=latency_ms)
            return r

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            _log_api("POST", url, 0, latency_ms, provider=provider,
                     key_id=acq.key_id, error=str(e)[:80])
            pool.release(acq.key_id, success=False, latency_ms=latency_ms, error=str(e)[:100])
            raise

    # All key attempts exhausted — fallback to legacy safe_post
    logger.warning(
        f"Pool: all {max_switches} key attempts exhausted for {provider}"
        f" — fallback to default key"
    )
    api_log.warning(
        f"POST | {provider} | {_short_url(url)} | POOL_EXHAUSTED | 0ms | fallback"
    )
    return safe_post(url, json=json, headers=headers, timeout=timeout)
