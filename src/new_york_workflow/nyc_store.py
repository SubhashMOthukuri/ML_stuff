"""
SQLite request store — logs every prediction for offline retraining.

Schema is append-only. Every row is one /predict call with:
  - full input features (JSON)
  - predicted price
  - whether it was served from cache

To retrain on production traffic:
  store = RequestStore()
  rows  = store.export_for_retraining()   # list of dicts, ready for pd.DataFrame

Swap DB_PATH to a Postgres DSN for production:
  export PREDICTION_DB=postgresql://user:pass@host/db
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("PREDICTION_DB", "data/predictions.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    borough         TEXT,
    room_type       TEXT,
    accommodates    INTEGER,
    bedrooms        REAL,
    bathrooms       REAL,
    predicted_price REAL    NOT NULL,
    log_price       REAL    NOT NULL,
    cache_hit       INTEGER NOT NULL DEFAULT 0,
    features_json   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_borough   ON predictions(borough);
CREATE INDEX IF NOT EXISTS idx_ts        ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_room_type ON predictions(room_type);
"""


class RequestStore:

    _lock = threading.Lock()

    def __init__(self, db_path: Path = DB_PATH):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("RequestStore ready: %s", self._path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ── writes ────────────────────────────────────────────────────────────────

    def log_prediction(
        self,
        request_id: str,
        raw: dict,
        result: dict,
        cache_hit: bool = False,
    ) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO predictions
                   (request_id, timestamp, borough, room_type, accommodates,
                    bedrooms, bathrooms, predicted_price, log_price, cache_hit, features_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    raw.get("borough"),
                    raw.get("room_type"),
                    raw.get("accommodates"),
                    raw.get("bedrooms"),
                    raw.get("bathrooms"),
                    result["price_usd"],
                    result["log_price"],
                    int(cache_hit),
                    json.dumps(raw),
                ),
            )

    # ── reads ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            row = c.execute("""
                SELECT
                    COUNT(*)                                     AS total,
                    SUM(cache_hit)                               AS cache_hits,
                    ROUND(AVG(predicted_price), 2)               AS avg_price,
                    ROUND(MIN(predicted_price), 2)               AS min_price,
                    ROUND(MAX(predicted_price), 2)               AS max_price
                FROM predictions
            """).fetchone()
        return dict(row)

    def clear(self, before: str | None = None) -> int:
        """
        Delete prediction rows. If `before` is an ISO-8601 timestamp string,
        only rows older than that are removed. Otherwise all rows are deleted.
        Returns the number of rows removed.
        """
        with self._lock, self._conn() as c:
            if before:
                c.execute("DELETE FROM predictions WHERE timestamp < ?", (before,))
            else:
                c.execute("DELETE FROM predictions")
            n = c.execute("SELECT changes()").fetchone()[0]
        logger.info("Cleared %d prediction rows (before=%s)", n, before or "all")
        return n

    def export_for_retraining(self, limit: int = 50_000) -> list[dict]:
        """
        Returns up to `limit` prediction rows with features unpacked.
        Limit is pushed into SQL — never loads the whole table into memory.
        """
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT request_id, timestamp, predicted_price, log_price, cache_hit, features_json "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "request_id":      r["request_id"],
                "timestamp":       r["timestamp"],
                "predicted_price": r["predicted_price"],
                "log_price":       r["log_price"],
                "cache_hit":       bool(r["cache_hit"]),
                **json.loads(r["features_json"]),
            })
        return out
