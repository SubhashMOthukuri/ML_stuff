"""
Retry decorator for flaky external calls (Redis, Oracle Vault, HTTP).

Usage:
    from src.utils.retry import retry

    @retry(attempts=3, delay=0.5, exceptions=(redis.RedisError,))
    def get_cached(key: str):
        return redis_client.get(key)
"""

import functools
import logging
import time
from typing import Type

logger = logging.getLogger(__name__)


def retry(
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    silent: bool = False,
):
    """
    Retry a function on failure with exponential backoff.

    Args:
        attempts: Maximum number of tries (including the first).
        delay:    Initial wait in seconds between retries.
        backoff:  Multiplier applied to delay after each failure (2.0 = doubles).
        exceptions: Only retry on these exception types.
        silent:   If True, log at DEBUG instead of WARNING.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == attempts:
                        raise
                    log = logger.debug if silent else logger.warning
                    log(
                        "Retry %d/%d for %s — %s: %s. Retrying in %.1fs…",
                        attempt, attempts, func.__qualname__,
                        type(exc).__name__, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
