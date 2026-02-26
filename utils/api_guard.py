import time
import requests
from functools import wraps

from utils.logger import get_logger

logger = get_logger("API_GUARD")


class APIGuardException(Exception):
    pass


def safe_api_call(
    retries=3,
    timeout=10,
    backoff=2,
    allowed_exceptions=(requests.exceptions.RequestException,)
):
    """
    Decorator for safe API calls with retry + exponential backoff
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
                        f"API error on {func.__name__} attempt {attempt}/{retries} : {e}"
                    )

                    if attempt >= retries:
                        logger.error(f"API failed permanently: {func.__name__}")
                        raise APIGuardException(str(e))

                    time.sleep(wait)
                    wait *= backoff

                except Exception as e:
                    logger.error(f"Unexpected error in API call {func.__name__}: {e}")
                    raise

        return wrapper

    return decorator


# ============================
# SIMPLE SAFE REQUEST (legacy — single key, backoff on error)
# ============================

@safe_api_call()
def safe_get(url, params=None, headers=None, timeout=10):
    return requests.get(url, params=params, headers=headers, timeout=timeout)


@safe_api_call()
def safe_post(url, json=None, headers=None, timeout=10):
    return requests.post(url, json=json, headers=headers, timeout=timeout)


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
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="RATE_LIMIT")
                continue  # No sleep — immediate retry with next key

            pool.release(acq.key_id, success=True, latency_ms=latency_ms)
            return r

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            pool.release(acq.key_id, success=False, latency_ms=latency_ms, error=str(e)[:100])
            raise

    # All key attempts exhausted — fallback to legacy safe_get
    logger.warning(
        f"Pool: all {max_switches} key attempts exhausted for {provider}"
        f" — fallback to default key"
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

    for attempt in range(max_switches):
        acq = pool.get_key(provider, task_type, priority)

        if not acq.success:
            logger.warning(
                f"Pool: no key available for {provider}/{task_type} (attempt {attempt + 1})"
            )
            break

        merged_headers = dict(headers or {})

        if provider in ("grok", "xai"):
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
                pool.release(acq.key_id, success=False, latency_ms=latency_ms, error="RATE_LIMIT")
                continue  # No sleep — immediate retry with next key

            pool.release(acq.key_id, success=True, latency_ms=latency_ms)
            return r

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            pool.release(acq.key_id, success=False, latency_ms=latency_ms, error=str(e)[:100])
            raise

    # All key attempts exhausted — fallback to legacy safe_post
    logger.warning(
        f"Pool: all {max_switches} key attempts exhausted for {provider}"
        f" — fallback to default key"
    )
    return safe_post(url, json=json, headers=headers, timeout=timeout)
