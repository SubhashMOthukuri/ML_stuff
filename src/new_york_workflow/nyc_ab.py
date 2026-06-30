"""
A/B test for NYC Airbnb price prediction models.

Unlike shadow testing (challenger hidden from users), A/B routing sends a
configurable percentage of real traffic to the challenger — users see its output.

When ground truth arrives via fetch_ground_truth.py, GET /ab/stats JOINs
ab_predictions with ground_truth on listing_id to compute per-arm MAE.
This gives a real-world accuracy comparison, not just output similarity.

Lifecycle:
  1. Shadow stats say "promote" → POST /ab/start?split_pct=20
  2. 20% of non-cached requests see challenger predictions
  3. Ground truth accumulates over days via scripts/fetch_ground_truth.py
  4. GET /ab/stats → per-arm MAE + recommendation
  5. POST /ab/promote → move to canary deployment (POST /canary/start)
     OR POST /ab/stop  → revert to champion, keep challenger for later
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH    = Path(os.getenv("PREDICTION_DB", "data/predictions.db"))
STATE_PATH = Path("data/ab_state.json")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ab_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id         TEXT    NOT NULL,
    request_id      TEXT,
    listing_id      TEXT,
    arm             TEXT    NOT NULL,
    predicted_price REAL    NOT NULL,
    borough         TEXT,
    room_type       TEXT,
    ts              TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ab_test ON ab_predictions(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_lid  ON ab_predictions(listing_id);
CREATE INDEX IF NOT EXISTS idx_ab_arm  ON ab_predictions(arm);
"""

MIN_GT_SAMPLES = 30    # ground-truth rows per arm needed before recommendation
PROMOTE_DELTA  = 2.0   # challenger MAE must beat champion by >= $2
REJECT_DELTA   = -5.0  # challenger MAE worse by >= $5 → reject


class ABTest:
    """
    Routes a configurable % of non-cached traffic to the challenger model.
    Uses hash(request_id) % 100 for deterministic, stateless arm assignment.

    ab_predictions lives in the same predictions.db as ground_truth, so the
    accuracy JOIN (ab_predictions ⟕ ground_truth on listing_id) costs nothing.
    State survives API restarts via data/ab_state.json.
    """

    def __init__(self, db_path: Path = DB_PATH, state_path: Path = STATE_PATH):
        self._lock       = threading.Lock()
        self._path       = Path(db_path)
        self._state_path = Path(state_path)
        self._state: dict = {}
        self._init_db()
        self._load_state()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                self._state = json.loads(self._state_path.read_text())
                if self._state.get("active"):
                    logger.info(
                        "ABTest: resumed test %s (split=%.0f%%)",
                        self._state["test_id"], self._state["split_pct"],
                    )
                return
            except Exception:
                pass
        self._state = {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2))

    # ── properties ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return bool(self._state.get("active"))

    @property
    def test_id(self) -> str | None:
        return self._state.get("test_id")

    # ── control ────────────────────────────────────────────────────────────────

    def start(self, split_pct: float) -> dict:
        """Activate A/B test: `split_pct`% of non-cached traffic → challenger."""
        if not (1 <= split_pct <= 50):
            return {"success": False, "error": "split_pct must be 1–50 (keep risk bounded)"}
        tid = datetime.now(timezone.utc).strftime("ab_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        with self._lock:
            if self.active:
                return {"success": False, "error": f"Test {self.test_id} already active — stop it first"}
            self._state = {
                "active":     True,
                "test_id":    tid,
                "split_pct":  split_pct,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()
        logger.info("ABTest: started %s — %.0f%% → challenger", tid, split_pct)
        return {"success": True, "test_id": tid, "split_pct": split_pct}

    def stop(self) -> dict:
        """Stop routing to challenger. Challenger files stay on disk."""
        tid = self.test_id
        if not tid:
            return {"success": False, "error": "No A/B test has been started"}
        with self._lock:
            self._state["active"]   = False
            self._state["ended_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
        logger.info("ABTest: stopped %s", tid)
        return {"success": True, "test_id": tid, "message": "A/B test stopped. Champion serves all traffic."}

    def promote(self) -> dict:
        """Stop A/B test and signal readiness for canary deployment."""
        self.stop()
        return {
            "success": True,
            "message": "A/B test stopped. Start canary deployment to roll out gradually.",
            "next": "POST /canary/start",
        }

    def reject(self) -> dict:
        """Stop A/B test without promoting. Use POST /shadow/reject to discard challenger."""
        self.stop()
        return {
            "success": True,
            "message": "A/B test stopped. Challenger not promoted. POST /shadow/reject to discard it.",
        }

    # ── routing ────────────────────────────────────────────────────────────────

    def route(self, request_id: str) -> str:
        """Returns 'challenger' for split_pct% of IDs, 'champion' otherwise."""
        if not self.active:
            return "champion"
        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        return "challenger" if h < self._state["split_pct"] else "champion"

    # ── logging ────────────────────────────────────────────────────────────────

    def log_prediction(
        self,
        request_id:      str,
        listing_id:      str | None,
        arm:             str,
        predicted_price: float,
        borough:         str | None = None,
        room_type:       str | None = None,
    ) -> None:
        if not self.active or not self.test_id:
            return
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    """INSERT INTO ab_predictions
                       (test_id, request_id, listing_id, arm, predicted_price, borough, room_type, ts)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        self.test_id, request_id, listing_id, arm,
                        round(predicted_price, 2), borough, room_type,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as exc:
            logger.debug("ABTest.log_prediction suppressed: %s", exc)

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Per-arm accuracy via JOIN with ground_truth on listing_id.
        Ground truth only matches requests that supplied a listing_id.
        """
        tid = self.test_id
        base = {
            "active":     self.active,
            "test_id":    tid,
            "split_pct":  self._state.get("split_pct"),
            "started_at": self._state.get("started_at"),
            "ended_at":   self._state.get("ended_at"),
        }

        if not tid:
            return {**base, "message": "No A/B test has been run. POST /ab/start?split_pct=20 to begin."}

        with self._lock, self._conn() as c:
            counts = {
                r["arm"]: r["n"]
                for r in c.execute(
                    "SELECT arm, COUNT(*) AS n FROM ab_predictions WHERE test_id=? GROUP BY arm",
                    (tid,),
                ).fetchall()
            }
            accuracy_rows = c.execute(
                """
                SELECT ab.arm,
                       COUNT(*)                                                       AS n_gt,
                       ROUND(AVG(ABS(gt.actual_price - ab.predicted_price)), 2)       AS mae,
                       ROUND(AVG(ABS(gt.actual_price - ab.predicted_price) /
                                 MAX(gt.actual_price, 1.0) * 100), 2)                AS mape_pct,
                       ROUND(MIN(ABS(gt.actual_price - ab.predicted_price)), 2)       AS min_error,
                       ROUND(MAX(ABS(gt.actual_price - ab.predicted_price)), 2)       AS max_error
                FROM ab_predictions ab
                JOIN ground_truth gt ON gt.listing_id = ab.listing_id
                WHERE ab.test_id = ?
                GROUP BY ab.arm
                """,
                (tid,),
            ).fetchall()

        accuracy = {r["arm"]: dict(r) for r in accuracy_rows}
        champ = accuracy.get("champion",   {})
        chal  = accuracy.get("challenger", {})

        champ_n   = champ.get("n_gt", 0)
        chal_n    = chal.get("n_gt",  0)
        champ_mae = champ.get("mae")
        chal_mae  = chal.get("mae")

        if champ_n < MIN_GT_SAMPLES or chal_n < MIN_GT_SAMPLES:
            recommendation = "insufficient_data"
            rec_reason = (
                f"Need {MIN_GT_SAMPLES}+ ground-truth rows per arm. "
                f"Have champion={champ_n}, challenger={chal_n}. "
                "Predictions must include listing_id to match ground truth."
            )
        elif champ_mae is None or chal_mae is None:
            recommendation = "insufficient_data"
            rec_reason = "No ground-truth matches yet. Ensure predictions include listing_id."
        else:
            delta = champ_mae - chal_mae   # positive = challenger is better
            if delta >= PROMOTE_DELTA:
                recommendation = "promote"
                rec_reason = f"Challenger MAE ${chal_mae} beats champion ${champ_mae} by ${delta:.2f}"
            elif delta <= REJECT_DELTA:
                recommendation = "reject"
                rec_reason = f"Challenger MAE ${chal_mae} is ${abs(delta):.2f} worse than champion ${champ_mae}"
            else:
                recommendation = "monitor"
                rec_reason = (
                    f"Difference ${abs(delta):.2f} (challenger {'better' if delta > 0 else 'worse'}) "
                    f"below promote threshold ${PROMOTE_DELTA}. Keep collecting."
                )

        return {
            **base,
            "predictions_served": {
                "champion":   counts.get("champion",   0),
                "challenger": counts.get("challenger", 0),
            },
            "ground_truth_accuracy": {
                "champion":   champ,
                "challenger": chal,
            },
            "recommendation":        recommendation,
            "recommendation_reason": rec_reason,
            "thresholds": {
                "min_gt_samples_per_arm":            MIN_GT_SAMPLES,
                "promote_if_challenger_better_by_usd": PROMOTE_DELTA,
                "reject_if_challenger_worse_by_usd":   abs(REJECT_DELTA),
            },
            "actions": {
                "promote": "POST /ab/promote → stop test, then POST /canary/start",
                "reject":  "POST /ab/reject  → stop test, POST /shadow/reject to discard challenger",
            },
        }
