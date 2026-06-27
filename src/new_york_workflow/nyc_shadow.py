"""
Shadow deployment for A/B model evaluation.

The challenger model runs alongside the champion on every cache-miss request:
  - Champion result → returned to user (zero impact)
  - Challenger result → logged to SQLite only (never seen by user)
  - Inference runs in a background thread pool → no added latency

Lifecycle:
  1. retrain.py runs → gate FAILS → saves challenger.onnx + challenger_scaler.pkl
  2. API restarts → ShadowPredictor loads challenger.onnx → shadow mode active
  3. Traffic flows in → both models predict → comparisons accumulate in SQLite
  4. GET /shadow/stats → see head-to-head metrics (bias, agreement, per-borough)
  5. POST /shadow/promote → swap challenger into champion position → restart API
     OR
     POST /shadow/reject → discard challenger.onnx → shadow mode disabled

Recommendation thresholds (conservative, tunable):
  promote   n>=100 AND |mean_diff|<=5% AND agreement_within_10pct>=90%
  reject    |mean_diff|>15%  (challenger systematically off)
  monitor   everything else (keep collecting data)
"""

import json
import logging
import pickle
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as rt

logger = logging.getLogger(__name__)

BASE_DIR         = Path(__file__).resolve().parents[2]
MODEL_DIR        = BASE_DIR / "models" / "nyc"
CHALLENGER_ONNX  = MODEL_DIR / "challenger.onnx"
CHALLENGER_SCALER = MODEL_DIR / "challenger_scaler.pkl"
SHADOW_DB        = BASE_DIR / "data" / "shadow_comparisons.db"

# Agreement bands for stats
TIGHT_BAND_PCT = 5.0    # ±5%  → "tight" agreement
LOOSE_BAND_PCT = 10.0   # ±10% → "loose" agreement

# Promotion / rejection thresholds
PROMOTE_MIN_N           = 100
PROMOTE_MAX_BIAS_PCT    = 5.0
PROMOTE_MIN_AGREE_10    = 0.90
REJECT_BIAS_PCT         = 15.0


class ShadowPredictor:
    """
    Loads challenger.onnx and runs it in a fire-and-forget thread on every
    non-cached prediction.  Thread-safe for use with preload_app=True gunicorn.

    Usage:
        shadow = ShadowPredictor()
        shadow.inject_champion(predictor)   # share feature engineering + feature list

        # Inside /predict, after champion inference:
        shadow.run_async(raw, result["price_usd"], request_id, borough, room_type)
    """

    def __init__(self, db_path: Path = SHADOW_DB):
        self._session: rt.InferenceSession | None = None
        self._input_name = ""
        self._challenger_scaler = None
        self._champion = None          # injected after init
        self._session_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")
        self._db = db_path
        self._db_lock = threading.Lock()
        self._init_db()
        self._load()

    # ── setup ──────────────────────────────────────────────────────────────────

    def inject_champion(self, predictor) -> None:
        """
        Share the champion predictor's feature engineering method and feature list.
        Must be called before run_async() — typically in the API lifespan startup.
        """
        self._champion = predictor

    def _load(self) -> None:
        """Load challenger.onnx + challenger_scaler.pkl if they exist."""
        if not CHALLENGER_ONNX.exists():
            logger.info("Shadow: no challenger.onnx — shadow deployment disabled")
            return

        try:
            sess = rt.InferenceSession(
                str(CHALLENGER_ONNX),
                providers=["CPUExecutionProvider"],   # challenger always on CPU; champion can use CoreML/CUDA
            )
            input_name = sess.get_inputs()[0].name

            challenger_scaler = None
            if CHALLENGER_SCALER.exists():
                with open(CHALLENGER_SCALER, "rb") as f:
                    challenger_scaler = pickle.load(f)
                logger.info("Shadow: loaded challenger_scaler.pkl")
            else:
                logger.warning(
                    "Shadow: challenger_scaler.pkl missing — will fall back to champion's scaler "
                    "(minor accuracy difference expected)"
                )

            with self._session_lock:
                self._session          = sess
                self._input_name       = input_name
                self._challenger_scaler = challenger_scaler

            logger.info("Shadow deployment ACTIVE — challenger.onnx loaded")
        except Exception as exc:
            logger.error("Shadow: failed to load challenger.onnx: %s", exc)

    def reload(self) -> None:
        """Hot-reload challenger model files (called after promote/reject)."""
        with self._session_lock:
            self._session          = None
            self._challenger_scaler = None
        self._load()

    @property
    def active(self) -> bool:
        return self._session is not None and self._champion is not None

    # ── async inference ────────────────────────────────────────────────────────

    def run_async(
        self,
        raw: dict,
        champion_price: float,
        request_id: str,
        borough: str,
        room_type: str,
    ) -> None:
        """
        Submit challenger inference to the thread pool.  Returns immediately.
        Safe to call unconditionally — no-ops if shadow is inactive.
        """
        if not self.active:
            return
        self._executor.submit(
            self._infer_and_log,
            raw, champion_price, request_id, borough, room_type,
        )

    def _infer_and_log(
        self,
        raw: dict,
        champion_price: float,
        request_id: str,
        borough: str,
        room_type: str,
    ) -> None:
        """Runs in background thread: engineer → scale → infer → write to DB."""
        try:
            with self._session_lock:
                sess             = self._session
                input_name       = self._input_name
                challenger_scaler = self._challenger_scaler

            if sess is None or self._champion is None:
                return

            # Reuse champion's _engineer() — same feature transformation logic
            row = self._champion._engineer(raw)
            df  = pd.DataFrame([row])[self._champion.feature_list]

            # Challenger scaler if available, otherwise fall back to champion's
            scaler = challenger_scaler if challenger_scaler is not None else self._champion.scaler
            X_scaled = scaler.transform(df).astype(np.float32)

            out = sess.run(None, {input_name: X_scaled})
            challenger_log  = float(np.array(out[0]).flat[0])
            challenger_price = float(np.expm1(challenger_log))

            diff_pct = (challenger_price - champion_price) / max(champion_price, 1.0) * 100

            self._write(
                request_id       = request_id,
                champion_price   = champion_price,
                challenger_price = challenger_price,
                diff_pct         = diff_pct,
                borough          = borough,
                room_type        = room_type,
                features_json    = raw,
            )
        except Exception as exc:
            logger.debug("Shadow inference error (suppressed): %s", exc)

    # ── SQLite ─────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._db.parent.mkdir(parents=True, exist_ok=True)
        with self._db_lock, sqlite3.connect(str(self._db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_comparisons (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    request_id        TEXT,
                    champion_price    REAL,
                    challenger_price  REAL,
                    diff_pct          REAL,
                    borough           TEXT,
                    room_type         TEXT,
                    features_json     TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_ts "
                "ON shadow_comparisons(timestamp)"
            )

    def _write(
        self,
        request_id: str,
        champion_price: float,
        challenger_price: float,
        diff_pct: float,
        borough: str,
        room_type: str,
        features_json: dict,
    ) -> None:
        with self._db_lock, sqlite3.connect(str(self._db)) as conn:
            conn.execute(
                """INSERT INTO shadow_comparisons
                   (timestamp, request_id, champion_price, challenger_price,
                    diff_pct, borough, room_type, features_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    request_id,
                    round(champion_price, 2),
                    round(challenger_price, 2),
                    round(diff_pct, 4),
                    borough,
                    room_type,
                    json.dumps(features_json),
                ),
            )

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self, window_hours: int = 168) -> dict:
        """
        Head-to-head comparison over the last `window_hours` of shadow traffic.
        Returns per-borough and per-room-type breakdowns plus a recommendation.
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()

        with self._db_lock, sqlite3.connect(str(self._db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT champion_price, challenger_price, diff_pct, borough, room_type
                   FROM shadow_comparisons WHERE timestamp >= ?""",
                (since,),
            ).fetchall()
            total_all = conn.execute(
                "SELECT COUNT(*) FROM shadow_comparisons"
            ).fetchone()[0]

        n = len(rows)
        base = {
            "active":           self.active,
            "challenger_path":  str(CHALLENGER_ONNX) if CHALLENGER_ONNX.exists() else None,
            "window_hours":     window_hours,
            "n_comparisons":    n,
            "total_all_time":   total_all,
        }

        if n == 0:
            return {
                **base,
                "message": (
                    "No shadow comparisons yet — run some predictions first."
                    if self.active
                    else "Shadow deployment disabled. Run retrain.py to create a challenger."
                ),
            }

        champs = [r["champion_price"]   for r in rows]
        chals  = [r["challenger_price"] for r in rows]
        diffs  = [r["diff_pct"]         for r in rows]

        mean_diff  = sum(diffs) / n
        var_diff   = sum((d - mean_diff) ** 2 for d in diffs) / n
        std_diff   = var_diff ** 0.5
        agree_5    = sum(1 for d in diffs if abs(d) <= TIGHT_BAND_PCT)  / n
        agree_10   = sum(1 for d in diffs if abs(d) <= LOOSE_BAND_PCT) / n

        # Per-dimension breakdowns
        borough_stats   = _breakdown(rows, "borough",   "diff_pct")
        room_type_stats = _breakdown(rows, "room_type", "diff_pct")

        # Distribution of diff_pct in buckets
        buckets = {"<-15%": 0, "-15%_to_-5%": 0, "-5%_to_5%": 0, "5%_to_15%": 0, ">15%": 0}
        for d in diffs:
            if   d < -15:  buckets["<-15%"]        += 1
            elif d < -5:   buckets["-15%_to_-5%"]  += 1
            elif d <=  5:  buckets["-5%_to_5%"]    += 1
            elif d <= 15:  buckets["5%_to_15%"]    += 1
            else:          buckets[">15%"]          += 1

        # Recommendation
        if n < PROMOTE_MIN_N:
            recommendation = "insufficient_data"
            rec_reason = f"Need at least {PROMOTE_MIN_N} comparisons (have {n})"
        elif abs(mean_diff) > REJECT_BIAS_PCT:
            recommendation = "reject"
            rec_reason = f"|mean_diff| {abs(mean_diff):.1f}% > {REJECT_BIAS_PCT}% threshold"
        elif agree_10 >= PROMOTE_MIN_AGREE_10 and abs(mean_diff) <= PROMOTE_MAX_BIAS_PCT:
            recommendation = "promote"
            rec_reason = (
                f"agreement_10pct={agree_10:.0%} >= {PROMOTE_MIN_AGREE_10:.0%} "
                f"AND |mean_diff|={abs(mean_diff):.1f}% <= {PROMOTE_MAX_BIAS_PCT}%"
            )
        else:
            recommendation = "monitor"
            rec_reason = "Thresholds not yet met — keep collecting comparisons"

        return {
            **base,
            "champion_mean_usd":       round(sum(champs) / n, 2),
            "challenger_mean_usd":     round(sum(chals)  / n, 2),
            "mean_diff_pct":           round(mean_diff, 2),
            "std_diff_pct":            round(std_diff, 2),
            "agreement_within_5pct":   round(agree_5,  4),
            "agreement_within_10pct":  round(agree_10, 4),
            "diff_distribution":       {k: round(v / n, 4) for k, v in buckets.items()},
            "by_borough":              borough_stats,
            "by_room_type":            room_type_stats,
            "recommendation":          recommendation,
            "recommendation_reason":   rec_reason,
            "actions": {
                "promote": "POST /shadow/promote  — swap challenger into production",
                "reject":  "DELETE /shadow/comparisons + remove challenger.onnx manually",
            },
        }

    # ── promotion / rejection ──────────────────────────────────────────────────

    def promote_challenger(self) -> dict:
        """
        Swap challenger into champion position:
          1. Backup champion ONNX → previous.onnx
          2. challenger.onnx     → nyc_xgb_model.onnx   (new champion model)
          3. challenger_scaler.pkl → nyc_scaler.pkl      (new champion scaler)
          4. Remove challenger artefacts → shadow disabled
          5. Disable in-memory shadow session

        The API must be restarted for the new champion to serve traffic.
        """
        champion_onnx   = MODEL_DIR / "nyc_xgb_model.onnx"
        champion_scaler = MODEL_DIR / "nyc_scaler.pkl"
        previous_onnx   = MODEL_DIR / "previous.onnx"
        previous_scaler = MODEL_DIR / "previous_scaler.pkl"

        if not CHALLENGER_ONNX.exists():
            return {"success": False, "error": "challenger.onnx not found — nothing to promote"}

        try:
            # 1. Backup current champion
            if champion_onnx.exists():
                shutil.copy2(champion_onnx, previous_onnx)
            if champion_scaler.exists():
                shutil.copy2(champion_scaler, previous_scaler)
            logger.info("Shadow promote: backed up champion → previous.onnx + previous_scaler.pkl")

            # 2. Swap challenger into champion slot
            shutil.copy2(CHALLENGER_ONNX, champion_onnx)
            if CHALLENGER_SCALER.exists():
                shutil.copy2(CHALLENGER_SCALER, champion_scaler)
            logger.info("Shadow promote: challenger.onnx → nyc_xgb_model.onnx")

            # 3. Remove challenger artefacts
            CHALLENGER_ONNX.unlink()
            if CHALLENGER_SCALER.exists():
                CHALLENGER_SCALER.unlink()
            logger.info("Shadow promote: challenger artefacts removed — shadow now disabled")

            # 4. Disable in-memory session
            with self._session_lock:
                self._session          = None
                self._challenger_scaler = None

            return {
                "success": True,
                "message": (
                    "Challenger promoted to champion. "
                    "Run 'docker compose restart api' to serve the new model."
                ),
                "previous_backup": {
                    "onnx":   str(previous_onnx),
                    "scaler": str(previous_scaler),
                },
                "rollback": "Copy previous.onnx → nyc_xgb_model.onnx and restart to undo.",
            }
        except Exception as exc:
            logger.error("Shadow promote failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def reject_challenger(self) -> dict:
        """
        Discard challenger.onnx + challenger_scaler.pkl without promoting.
        Shadow deployment is disabled until next retrain.
        """
        removed = []
        for path in (CHALLENGER_ONNX, CHALLENGER_SCALER):
            if path.exists():
                path.unlink()
                removed.append(path.name)

        with self._session_lock:
            self._session          = None
            self._challenger_scaler = None

        return {
            "success": True,
            "removed": removed,
            "message": "Challenger rejected. Shadow deployment disabled until next retrain.",
        }

    def clear_comparisons(self) -> int:
        """Delete all rows from shadow_comparisons. Returns count deleted."""
        with self._db_lock, sqlite3.connect(str(self._db)) as conn:
            cur = conn.execute("DELETE FROM shadow_comparisons")
            return cur.rowcount


# ── helpers ────────────────────────────────────────────────────────────────────

def _breakdown(rows, group_col: str, value_col: str) -> dict:
    """Group rows by `group_col` and compute mean/std of `value_col`."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        key = r[group_col] or "unknown"
        groups.setdefault(key, []).append(r[value_col])
    out = {}
    for key, vals in sorted(groups.items()):
        n = len(vals)
        mean = sum(vals) / n
        out[key] = {
            "n":             n,
            "mean_diff_pct": round(mean, 2),
            "std_diff_pct":  round((sum((v - mean) ** 2 for v in vals) / n) ** 0.5, 2),
        }
    return out
