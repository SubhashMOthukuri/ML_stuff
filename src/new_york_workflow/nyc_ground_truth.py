"""
Ground truth store — tracks actual vs predicted prices for production accuracy.

Same SQLite file as the prediction store (predictions.db) so joins between
predictions and ground_truth are free with no cross-file queries.

Lifecycle:
  1. /predict logs listing_id alongside the prediction (nyc_store.py)
  2. scripts/fetch_ground_truth.py runs as a cron job:
       - downloads fresh InsideAirbnb NYC snapshot
       - joins on listing_id → gets actual price
       - writes rows here
  3. GET /ground-truth/stats reads this table → production MAE
  4. If production MAE > training MAE + threshold → push alert → trigger retrain
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
CREATE TABLE IF NOT EXISTS ground_truth (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id      TEXT    NOT NULL,
    request_id      TEXT,
    predicted_price REAL    NOT NULL,
    actual_price    REAL    NOT NULL,
    abs_error       REAL    NOT NULL,
    pct_error       REAL    NOT NULL,
    borough         TEXT,
    room_type       TEXT,
    fetched_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gt_listing ON ground_truth(listing_id);
CREATE INDEX IF NOT EXISTS idx_gt_fetched        ON ground_truth(fetched_at);
CREATE INDEX IF NOT EXISTS idx_gt_borough        ON ground_truth(borough);
"""


class GroundTruthStore:

    def __init__(self, db_path: Path = DB_PATH):
        self._lock = threading.Lock()
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("GroundTruthStore ready: %s", self._path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ── writes ────────────────────────────────────────────────────────────────

    def insert(
        self,
        listing_id:      str,
        predicted_price: float,
        actual_price:    float,
        request_id:      str | None = None,
        borough:         str | None = None,
        room_type:       str | None = None,
    ) -> bool:
        """
        Insert one ground truth row. Skips silently if listing_id already exists
        (UNIQUE constraint) — safe to call from the batch job on repeated runs.
        Returns True if inserted, False if skipped.
        """
        abs_error = abs(predicted_price - actual_price)
        pct_error = abs_error / max(actual_price, 1.0) * 100
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    """INSERT OR IGNORE INTO ground_truth
                       (listing_id, request_id, predicted_price, actual_price,
                        abs_error, pct_error, borough, room_type, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        listing_id,
                        request_id,
                        round(predicted_price, 2),
                        round(actual_price, 2),
                        round(abs_error, 2),
                        round(pct_error, 2),
                        borough,
                        room_type,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                inserted = c.execute("SELECT changes()").fetchone()[0]
            return bool(inserted)
        except Exception as exc:
            logger.error("GroundTruthStore.insert failed: %s", exc)
            return False

    # ── reads ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            row = c.execute("""
                SELECT
                    COUNT(*)                            AS n_evaluated,
                    ROUND(AVG(abs_error), 2)            AS mae_dollar,
                    ROUND(AVG(pct_error), 2)            AS mape_pct,
                    ROUND(MIN(abs_error), 2)            AS min_error,
                    ROUND(MAX(abs_error), 2)            AS max_error,
                    ROUND(AVG(actual_price), 2)         AS avg_actual_price,
                    ROUND(AVG(predicted_price), 2)      AS avg_predicted_price,
                    MIN(fetched_at)                     AS first_evaluation,
                    MAX(fetched_at)                     AS last_evaluation
                FROM ground_truth
            """).fetchone()
            by_borough = {}
            for r in c.execute("""
                SELECT borough,
                       COUNT(*)              AS n,
                       ROUND(AVG(abs_error),2) AS mae_dollar
                FROM ground_truth
                WHERE borough IS NOT NULL
                GROUP BY borough ORDER BY mae_dollar DESC
            """).fetchall():
                by_borough[r["borough"]] = {"n": r["n"], "mae_dollar": r["mae_dollar"]}

        return {**dict(row), "by_borough": by_borough}

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                """SELECT listing_id, predicted_price, actual_price,
                          abs_error, pct_error, borough, room_type, fetched_at
                   FROM ground_truth ORDER BY fetched_at DESC LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def error_distribution(self) -> dict:
        """Bucket abs_error into bands for the frontend histogram."""
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT abs_error FROM ground_truth").fetchall()
        if not rows:
            return {}
        buckets = {"<$10": 0, "$10-25": 0, "$25-50": 0, "$50-100": 0, ">$100": 0}
        for r in rows:
            e = r["abs_error"]
            if   e < 10:   buckets["<$10"]    += 1
            elif e < 25:   buckets["$10-25"]  += 1
            elif e < 50:   buckets["$25-50"]  += 1
            elif e < 100:  buckets["$50-100"] += 1
            else:          buckets[">$100"]   += 1
        n = len(rows)
        return {k: {"count": v, "pct": round(v / n * 100, 1)} for k, v in buckets.items()}


