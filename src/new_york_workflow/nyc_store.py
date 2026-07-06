"""
Append-only prediction log — SQLite for local dev, Postgres for production.

Detects connection type from PREDICTION_DB:
  SQLite:   data/predictions.db      (default — local dev / single-node)
  Postgres: postgresql://user:pw@host:5432/db  (production via RDS)

To retrain on production traffic:
  store = RequestStore()
  rows  = store.export_for_retraining()   # list[dict] → pd.DataFrame

Switch by setting PREDICTION_DB env var (injected from AWS Secrets Manager in prod).
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.prediction_db

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    listing_id      TEXT,
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
CREATE INDEX IF NOT EXISTS idx_borough    ON predictions(borough);
CREATE INDEX IF NOT EXISTS idx_ts         ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_room_type  ON predictions(room_type);
CREATE INDEX IF NOT EXISTS idx_listing_id ON predictions(listing_id);
"""

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    request_id      TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    listing_id      TEXT,
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
CREATE INDEX IF NOT EXISTS idx_borough    ON predictions(borough);
CREATE INDEX IF NOT EXISTS idx_ts         ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_room_type  ON predictions(room_type);
CREATE INDEX IF NOT EXISTS idx_listing_id ON predictions(listing_id);
"""

# Only needed for SQLite databases predating the listing_id column.
_MIGRATION_SQLITE = """
ALTER TABLE predictions ADD COLUMN listing_id TEXT;
CREATE INDEX IF NOT EXISTS idx_listing_id ON predictions(listing_id);
"""


class RequestStore:

    def __init__(self, db_path: str = DB_PATH):
        self._lock   = threading.Lock()
        self._path   = str(db_path)
        self._is_pg  = self._path.startswith(("postgresql://", "postgres://"))

        if not self._is_pg:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()
        logger.info("RequestStore ready: %s", "postgres" if self._is_pg else self._path)

    # ── connection helpers ─────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield an open connection; commit on success, rollback on exception."""
        if self._is_pg:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self._path)
            conn.cursor_factory = psycopg2.extras.RealDictCursor
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _q(self, sql: str) -> str:
        """Rewrite SQLite '?' placeholders to Postgres '%s' style."""
        return sql.replace("?", "%s") if self._is_pg else sql

    def _execute(self, conn, sql: str, params: tuple = ()):
        """Run a single statement; returns cursor with Postgres, connection with SQLite."""
        if self._is_pg:
            cur = conn.cursor()
            cur.execute(self._q(sql), params)
            return cur
        else:
            return conn.execute(self._q(sql), params)

    def _fetchall(self, result) -> list[dict]:
        """Normalise rows from both drivers to list[dict]."""
        rows = result.fetchall()
        return [dict(r) for r in rows]

    def _fetchone_scalar(self, result):
        """Return the first column of the first row (for COUNT, changes, etc.)."""
        row = result.fetchone()
        if row is None:
            return 0
        return row[0] if not isinstance(row, dict) else list(row.values())[0]

    # ── schema init ───────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            if self._is_pg:
                self._init_schema_pg()
            else:
                self._init_schema_sqlite()

    def _init_schema_pg(self) -> None:
        import psycopg2
        conn = psycopg2.connect(self._path)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                for stmt in _SCHEMA_PG.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
        finally:
            conn.close()

    def _init_schema_sqlite(self) -> None:
        with sqlite3.connect(self._path, check_same_thread=False) as conn:
            conn.executescript(_SCHEMA_SQLITE)
            # migrate databases predating the listing_id column
            cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            if "listing_id" not in cols:
                for stmt in _MIGRATION_SQLITE.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            conn.execute(stmt)
                        except Exception:
                            pass

    # ── writes ────────────────────────────────────────────────────────────────

    def log_prediction(
        self,
        request_id: str,
        raw: dict,
        result: dict,
        cache_hit: bool = False,
        listing_id: str | None = None,
    ) -> None:
        """Append one prediction row — called on every /predict cache miss."""
        sql = """INSERT INTO predictions
                   (request_id, timestamp, listing_id, borough, room_type, accommodates,
                    bedrooms, bathrooms, predicted_price, log_price, cache_hit, features_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
        params = (
            request_id,
            datetime.now(timezone.utc).isoformat(),
            listing_id,
            raw.get("borough"),
            raw.get("room_type"),
            raw.get("accommodates"),
            raw.get("bedrooms"),
            raw.get("bathrooms"),
            result["price_usd"],
            result["log_price"],
            int(cache_hit),
            json.dumps(raw),
        )
        with self._lock, self._conn() as conn:
            self._execute(conn, sql, params)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get_listings_pending_ground_truth(self) -> list[dict]:
        """Return rows with a listing_id not yet in the ground_truth table."""
        sql = """SELECT request_id, listing_id, predicted_price, timestamp
                   FROM predictions
                   WHERE listing_id IS NOT NULL
                   AND listing_id NOT IN (
                       SELECT listing_id FROM ground_truth
                   )
                   ORDER BY timestamp DESC"""
        with self._lock, self._conn() as conn:
            result = self._execute(conn, sql)
            return self._fetchall(result)

    def count(self) -> int:
        """Total rows in predictions table."""
        with self._lock, self._conn() as conn:
            cur = self._execute(conn, "SELECT COUNT(*) FROM predictions")
            return self._fetchone_scalar(cur)

    def recent(self, n: int = 20) -> list[dict]:
        """Last n predictions ordered by insertion."""
        with self._lock, self._conn() as conn:
            result = self._execute(
                conn, "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
            )
            return self._fetchall(result)

    def stats(self) -> dict:
        """Aggregate statistics across all predictions."""
        sql = """SELECT
                    COUNT(*)                                     AS total,
                    SUM(cache_hit)                               AS cache_hits,
                    ROUND(AVG(predicted_price)::numeric, 2)      AS avg_price,
                    ROUND(MIN(predicted_price)::numeric, 2)      AS min_price,
                    ROUND(MAX(predicted_price)::numeric, 2)      AS max_price
                FROM predictions"""
        sql_sqlite = """SELECT
                    COUNT(*)                                     AS total,
                    SUM(cache_hit)                               AS cache_hits,
                    ROUND(AVG(predicted_price), 2)               AS avg_price,
                    ROUND(MIN(predicted_price), 2)               AS min_price,
                    ROUND(MAX(predicted_price), 2)               AS max_price
                FROM predictions"""
        with self._lock, self._conn() as conn:
            result = self._execute(conn, sql if self._is_pg else sql_sqlite)
            rows = self._fetchall(result)
            return rows[0] if rows else {}

    def clear(self, before: str | None = None) -> int:
        """Delete rows — all rows if before is None, else rows older than before (ISO-8601)."""
        if before:
            sql    = "DELETE FROM predictions WHERE timestamp < ?"
            params = (before,)
        else:
            sql    = "DELETE FROM predictions"
            params = ()

        with self._lock, self._conn() as conn:
            cur = self._execute(conn, sql, params)
            n   = cur.rowcount
        logger.info("Cleared %d prediction rows (before=%s)", n, before or "all")
        return n

    def export_for_retraining(self, limit: int = 50_000) -> list[dict]:
        """Return up to limit rows with features unpacked — ready for pd.DataFrame(rows)."""
        sql = (
            "SELECT request_id, timestamp, predicted_price, log_price, cache_hit, features_json "
            "FROM predictions ORDER BY id DESC LIMIT ?"
        )
        with self._lock, self._conn() as conn:
            result = self._execute(conn, sql, (limit,))
            rows   = self._fetchall(result)

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
