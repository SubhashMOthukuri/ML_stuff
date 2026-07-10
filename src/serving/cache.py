"""
Redis-backed prediction cache.

Key:  nyc:pred:<md5(sorted_json_of_input)>
TTL:  CACHE_TTL_SECONDS env var (default 5 min)

Graceful degradation: if Redis is down, every call is a cache miss —
the API continues working, just without caching.
"""

import hashlib
import json
import logging
from datetime import date

from src.core.config import settings

logger = logging.getLogger(__name__)


class PredictionCache:

    def __init__(self):
        self._client = None
        self._hits   = 0
        self._misses = 0
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                socket_connect_timeout=2,
                socket_timeout=1,
                decode_responses=True,
            )
            self._client.ping()
            logger.info("PredictionCache connected to Redis")
        except Exception as exc:
            logger.warning("Redis unavailable — cache disabled (%s)", exc)
            self._client = None

    @staticmethod
    def _key(raw: dict) -> str:
        # Include today's date so checkin_date-relative features (days_to_checkin)
        # don't return a stale cached result from a prior day.
        keyed = {**raw, "_date": date.today().isoformat()}
        canonical = json.dumps(keyed, sort_keys=True)
        return f"nyc:pred:{hashlib.md5(canonical.encode()).hexdigest()}"

    def get(self, raw: dict) -> dict | None:
        if not self._client:
            return None
        try:
            val = self._client.get(self._key(raw))
            if val:
                self._hits += 1
                return json.loads(val)
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None

    def set(self, raw: dict, result: dict, ttl: int | None = None) -> None:
        if not self._client:
            return
        try:
            self._client.setex(self._key(raw), ttl or settings.cache_ttl_seconds, json.dumps(result))
        except Exception as exc:
            logger.debug("Cache set failed (prediction still served): %s", exc)

    def stats(self) -> dict:
        total = self._hits + self._misses
        base  = {
            "connected":   self._client is not None,
            "hits":        self._hits,
            "misses":      self._misses,
            "hit_rate_pct": round(self._hits / max(total, 1) * 100, 1),
        }
        if self._client:
            try:
                info = self._client.info("stats")
                base["redis_hits"]   = info.get("keyspace_hits", 0)
                base["redis_misses"] = info.get("keyspace_misses", 0)
            except Exception:
                pass
        return base
