"""
Dead Letter Queue — stores failed / timed-out requests for retry or alerting.

Backend: Redis List (LPUSH on write, LRANGE on read).
Fallback: in-memory list when Redis is unavailable.

Entry schema:
  {
    "timestamp":   "2026-06-26T18:00:00+00:00",
    "request_id":  "uuid",
    "endpoint":    "/predict",
    "status_code": 500,
    "error":       "...",
    "payload":     { ...original input... }
  }

Retry worker (separate process) would BRPOP from nyc:dlq and replay.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DLQ_KEY  = "nyc:dlq"
DLQ_MAX  = int(os.getenv("DLQ_MAX_SIZE", "10000"))


class DeadLetterQueue:

    def __init__(self):
        self._client   = None
        self._fallback: list[dict] = []
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                socket_connect_timeout=2,
                socket_timeout=1,
                decode_responses=True,
            )
            self._client.ping()
            logger.info("DeadLetterQueue connected to Redis (key=%s)", DLQ_KEY)
        except Exception as exc:
            logger.warning("DLQ Redis unavailable — using in-memory fallback (%s)", exc)
            self._client = None

    # ── writes ────────────────────────────────────────────────────────────────

    def push(
        self,
        payload:     dict,
        error:       str,
        endpoint:    str,
        request_id:  str  = "",
        status_code: int  = 500,
    ) -> None:
        entry = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "request_id":  request_id,
            "endpoint":    endpoint,
            "status_code": status_code,
            "error":       error,
            "payload":     payload,
        }
        if self._client:
            try:
                pipe = self._client.pipeline()
                pipe.lpush(DLQ_KEY, json.dumps(entry))
                pipe.ltrim(DLQ_KEY, 0, DLQ_MAX - 1)   # cap at DLQ_MAX entries
                pipe.execute()
                return
            except Exception as exc:
                logger.error("DLQ push to Redis failed: %s", exc)
        # in-memory fallback
        self._fallback.insert(0, entry)
        if len(self._fallback) > DLQ_MAX:
            self._fallback = self._fallback[:DLQ_MAX]

    # ── reads ─────────────────────────────────────────────────────────────────

    def peek(self, n: int = 50) -> list[dict]:
        if self._client:
            try:
                raw = self._client.lrange(DLQ_KEY, 0, n - 1)
                return [json.loads(r) for r in raw]
            except Exception:
                pass
        return self._fallback[:n]

    def size(self) -> int:
        if self._client:
            try:
                return self._client.llen(DLQ_KEY)
            except Exception:
                pass
        return len(self._fallback)

    def clear(self) -> int:
        """Delete all entries. Returns count removed."""
        n = self.size()
        if self._client:
            try:
                self._client.delete(DLQ_KEY)
            except Exception:
                pass
        self._fallback.clear()
        return n

    def stats(self) -> dict:
        return {
            "redis_connected": self._client is not None,
            "size":            self.size(),
            "key":             DLQ_KEY,
        }
