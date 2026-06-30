"""
Canary deployment: gradual traffic rollout with automated health checks.

After shadow + A/B testing confirm the challenger is better, canary rolls it
out to increasing percentages of real traffic:
  1% → 5% → 25% → 100%

At each stage a background health monitor evaluates challenger requests:
  - error_rate  < 5%    (hard failure threshold)
  - p99 latency < 2000ms

If the stage is healthy for MIN_STAGE_DURATION_S (60s) with enough samples,
it auto-advances. If health checks fail, it auto-rolls-back to 0% (champion).
At 100% it promotes challenger → champion via file swap, then disables itself.

Lifecycle:
  1. POST /canary/start        → stage 0 (1%)
  2. Traffic flows             → health accumulates in canary_health.db
  3. GET /canary/status        → current stage, health, history
  4. (auto) advance 1→5→25→100 OR rollback if unhealthy
  5. POST /canary/advance      → skip health wait, advance manually
  6. POST /canary/rollback     → force back to champion
  7. At 100%: challenger.onnx → nyc_xgb_model.onnx; restart API
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parents[2]
MODEL_DIR   = BASE_DIR / "models" / "nyc"
STATE_PATH  = BASE_DIR / "data" / "canary_state.json"
HEALTH_DB   = BASE_DIR / "data" / "canary_health.db"

CHALLENGER_ONNX   = MODEL_DIR / "challenger.onnx"
CHALLENGER_SCALER = MODEL_DIR / "challenger_scaler.pkl"
CHAMPION_ONNX     = MODEL_DIR / "nyc_xgb_model.onnx"
CHAMPION_SCALER   = MODEL_DIR / "nyc_scaler.pkl"
PREVIOUS_ONNX     = MODEL_DIR / "previous.onnx"
PREVIOUS_SCALER   = MODEL_DIR / "previous_scaler.pkl"

STAGES = [1, 5, 25, 100]

ERROR_RATE_THRESHOLD    = 0.05   # 5%  error rate → rollback during canary stages
LATENCY_P99_MS          = 2000   # 2s  p99 latency → rollback during canary stages
MIN_STAGE_SAMPLES       = 20     # requests at stage before health matters
MIN_STAGE_DURATION_S    = 60     # healthy for 60s → auto-advance
HEALTH_CHECK_INTERVAL_S = 15     # background thread cadence

# Post-promotion: watch the promoted model for 1 hour. If error rate jumps 3x the
# pre-promotion canary baseline, auto-revert files and push an alert. Persists across
# API restarts via canary_state.json so a restart to serve the new model doesn't lose context.
POST_PROMOTION_WINDOW_S      = 3600   # monitoring window after promotion
POST_PROMOTION_ROLLBACK_MULT = 3.0    # error rate multiplier that triggers auto-revert
POST_PROMOTION_MIN_SAMPLES   = 30     # minimum samples before revert can fire

_SCHEMA = """
CREATE TABLE IF NOT EXISTS canary_health (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL,
    stage_pct  INTEGER NOT NULL,
    arm        TEXT    NOT NULL,
    latency_ms REAL    NOT NULL,
    success    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ch_ts ON canary_health(ts);
"""


class HealthResult(NamedTuple):
    ok:          bool
    error_rate:  float
    p99_latency: float
    n_samples:   int
    reason:      str


class CanaryDeployment:
    """
    Manages gradual rollout: routes traffic by hash(request_id), records health
    per request, and runs a background thread that auto-advances or auto-rolls back.
    Persists state across API restarts via data/canary_state.json.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._state: dict = {}
        self._shadow = None    # injected ShadowPredictor for challenger inference
        self._stop_event    = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._init_db()
        self._load_state()
        if self.active or self.post_promotion_active:
            self._start_monitor()

    # ── setup ──────────────────────────────────────────────────────────────────

    def inject(self, shadow) -> None:
        """Share the ShadowPredictor so canary can call predict_sync()."""
        self._shadow = shadow

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(HEALTH_DB), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        HEALTH_DB.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _load_state(self) -> None:
        if STATE_PATH.exists():
            try:
                self._state = json.loads(STATE_PATH.read_text())
                if self._state.get("active"):
                    logger.info(
                        "Canary: resumed at %d%% (started %s)",
                        self._state["current_pct"], self._state.get("started_at", "?"),
                    )
                elif self.post_promotion_active:
                    logger.info("Canary: post-promotion monitoring window active (promoted %s)",
                                self._state.get("promoted_at", "?"))
                return
            except Exception:
                pass
        self._state = {}

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self._state, indent=2))

    # ── properties ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return bool(self._state.get("active"))

    @property
    def current_pct(self) -> int:
        return int(self._state.get("current_pct", 0))

    @property
    def post_promotion_active(self) -> bool:
        """True while the 1-hour post-promotion monitoring window is open."""
        if self._state.get("post_promotion_resolved") or not self._state.get("promoted_at"):
            return False
        try:
            promoted_at = datetime.fromisoformat(
                self._state["promoted_at"].replace("Z", "+00:00")
            )
            elapsed = (datetime.now(timezone.utc) - promoted_at).total_seconds()
            return elapsed < POST_PROMOTION_WINDOW_S
        except Exception:
            return False

    # ── control ────────────────────────────────────────────────────────────────

    def start(self) -> dict:
        """Begin canary at stage 0 (1% of traffic → challenger)."""
        if self.active:
            return {"success": False, "error": f"Canary already active at {self.current_pct}%"}
        if not CHALLENGER_ONNX.exists():
            return {"success": False, "error": "challenger.onnx not found — run retrain.py first"}

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._state = {
                "active":           True,
                "current_pct":      STAGES[0],
                "stage_idx":        0,
                "started_at":       now,
                "stage_started_at": now,
                "history":          [],
            }
            self._save_state()

        self._start_monitor()
        logger.info("Canary: started at %d%%", STAGES[0])
        return {
            "success":     True,
            "current_pct": STAGES[0],
            "stages":      STAGES,
            "message":     f"Canary started. {STAGES[0]}% of traffic now routed to challenger.",
        }

    def abort(self) -> dict:
        """Stop canary and return 100% traffic to champion. Challenger stays on disk."""
        return self.rollback(reason="aborted")

    def rollback(self, reason: str = "manual") -> dict:
        """Roll back to champion. challenger.onnx preserved for inspection."""
        with self._lock:
            prev = self._state.get("current_pct", 0)
            self._state.setdefault("history", []).append({
                "event": "rollback", "from_pct": prev,
                "reason": reason, "ts": datetime.now(timezone.utc).isoformat(),
            })
            self._state.update({
                "active":       False,
                "current_pct":  0,
                "ended_at":     datetime.now(timezone.utc).isoformat(),
                "ended_reason": f"rollback:{reason}",
            })
            self._save_state()
        self._stop_monitor()
        logger.warning("Canary: ROLLED BACK from %d%% — %s", prev, reason)
        return {
            "success":            True,
            "rolled_back_from":   f"{prev}%",
            "reason":             reason,
            "message":            "Rolled back to champion. challenger.onnx preserved.",
        }

    def advance(self, manual: bool = False) -> dict:
        """Advance to the next stage. At 100%, promotes challenger to champion."""
        with self._lock:
            if not self._state.get("active"):
                return {"success": False, "error": "Canary not active"}

            idx     = self._state["stage_idx"]
            cur_pct = STAGES[idx]

            if idx + 1 >= len(STAGES):
                return self._promote_locked()

            next_idx = idx + 1
            next_pct = STAGES[next_idx]
            now      = datetime.now(timezone.utc).isoformat()

            self._state.setdefault("history", []).append({
                "event": "advance", "from_pct": cur_pct,
                "to_pct": next_pct, "manual": manual, "ts": now,
            })
            self._state.update({
                "stage_idx":        next_idx,
                "current_pct":      next_pct,
                "stage_started_at": now,
            })
            self._save_state()

        logger.info("Canary: advanced %d%% → %d%%", cur_pct, next_pct)

        if next_pct == 100:
            return self._promote()

        return {"success": True, "previous_pct": cur_pct, "current_pct": next_pct}

    def _promote(self) -> dict:
        with self._lock:
            return self._promote_locked()

    def _promote_locked(self) -> dict:
        """File swap: challenger → champion. Starts post-promotion monitoring window."""
        try:
            if CHAMPION_ONNX.exists():
                shutil.copy2(CHAMPION_ONNX, PREVIOUS_ONNX)
            if CHAMPION_SCALER.exists():
                shutil.copy2(CHAMPION_SCALER, PREVIOUS_SCALER)

            shutil.copy2(CHALLENGER_ONNX, CHAMPION_ONNX)
            if CHALLENGER_SCALER.exists():
                shutil.copy2(CHALLENGER_SCALER, CHAMPION_SCALER)

            CHALLENGER_ONNX.unlink()
            if CHALLENGER_SCALER.exists():
                CHALLENGER_SCALER.unlink()

            # Snapshot the baseline error rate from the final canary stage health check
            # (lock is already held — read directly from DB to avoid deadlock with health_check())
            pre_error_rate = self._read_stage_error_rate_locked()

            now = datetime.now(timezone.utc).isoformat()
            self._state.setdefault("history", []).append({"event": "promoted", "ts": now})
            self._state.update({
                "active":                    False,
                "current_pct":               100,
                "ended_at":                  now,
                "ended_reason":              "promoted",
                "promoted_at":               now,
                "pre_promotion_error_rate":  pre_error_rate,
                "post_promotion_resolved":   False,
            })
            self._save_state()
            # Monitor thread continues — _tick_post_promotion() takes over
            logger.info(
                "Canary: challenger PROMOTED — restart API to serve new model. "
                "Post-promotion monitoring active for %ds (baseline error_rate=%.1f%%).",
                POST_PROMOTION_WINDOW_S, pre_error_rate * 100,
            )
            return {
                "success": True,
                "message": (
                    "Challenger promoted to champion. Run 'docker compose restart api'. "
                    f"Post-promotion monitoring active for {POST_PROMOTION_WINDOW_S // 60} min."
                ),
                "previous_backup":            {"onnx": str(PREVIOUS_ONNX), "scaler": str(PREVIOUS_SCALER)},
                "post_promotion_window_s":    POST_PROMOTION_WINDOW_S,
                "pre_promotion_error_rate":   pre_error_rate,
            }
        except Exception as exc:
            logger.error("Canary promote failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def _read_stage_error_rate_locked(self) -> float:
        """Read the current stage error rate directly (called while lock is held, no DB contention)."""
        try:
            since = datetime.fromtimestamp(
                time.time() - MIN_STAGE_DURATION_S * 4, tz=timezone.utc
            ).isoformat()
            with self._conn() as c:
                rows = c.execute(
                    "SELECT success FROM canary_health WHERE arm='challenger' AND stage_pct=? AND ts >= ?",
                    (self.current_pct, since),
                ).fetchall()
            if not rows:
                return 0.0
            return sum(1 for r in rows if not r["success"]) / len(rows)
        except Exception:
            return 0.0

    # ── routing ────────────────────────────────────────────────────────────────

    def route(self, request_id: str) -> str:
        """Deterministically returns 'challenger' for current_pct% of request IDs."""
        if not self.active:
            return "champion"
        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        return "challenger" if h < self.current_pct else "champion"

    # ── health ─────────────────────────────────────────────────────────────────

    def record_post_promotion(self, latency_ms: float, success: bool) -> None:
        """Log one /predict outcome during the post-promotion monitoring window."""
        if not self.post_promotion_active:
            return
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO canary_health (ts, stage_pct, arm, latency_ms, success) VALUES (?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), 100, "post_promotion", latency_ms, int(success)),
                )
        except Exception:
            pass

    def record_result(self, arm: str, latency_ms: float, success: bool) -> None:
        """Log one request outcome for the health monitor."""
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO canary_health (ts, stage_pct, arm, latency_ms, success) VALUES (?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), self.current_pct, arm, latency_ms, int(success)),
                )
        except Exception:
            pass

    def health_check(self, window_s: int = 120) -> HealthResult:
        """Evaluate challenger health over the last `window_s` seconds at current stage."""
        since = datetime.fromtimestamp(time.time() - window_s, tz=timezone.utc).isoformat()

        with self._conn() as c:
            rows = c.execute(
                """SELECT latency_ms, success FROM canary_health
                   WHERE arm='challenger' AND stage_pct=? AND ts >= ?""",
                (self.current_pct, since),
            ).fetchall()

        n = len(rows)
        if n < MIN_STAGE_SAMPLES:
            return HealthResult(
                ok=True, error_rate=0.0, p99_latency=0.0, n_samples=n,
                reason=f"Collecting samples ({n}/{MIN_STAGE_SAMPLES} minimum)",
            )

        errors     = sum(1 for r in rows if not r["success"])
        error_rate = errors / n
        latencies  = sorted(r["latency_ms"] for r in rows)
        p99        = latencies[min(n - 1, int(n * 0.99))]

        if error_rate > ERROR_RATE_THRESHOLD:
            return HealthResult(
                ok=False, error_rate=error_rate, p99_latency=p99, n_samples=n,
                reason=f"error_rate {error_rate:.1%} exceeds {ERROR_RATE_THRESHOLD:.0%} threshold",
            )
        if p99 > LATENCY_P99_MS:
            return HealthResult(
                ok=False, error_rate=error_rate, p99_latency=p99, n_samples=n,
                reason=f"p99 latency {p99:.0f}ms exceeds {LATENCY_P99_MS}ms threshold",
            )
        return HealthResult(ok=True, error_rate=error_rate, p99_latency=p99, n_samples=n, reason="healthy")

    # ── background monitor ─────────────────────────────────────────────────────

    def _start_monitor(self) -> None:
        self._stop_event.clear()
        t = threading.Thread(target=self._monitor_loop, daemon=True, name="canary-monitor")
        t.start()
        self._monitor_thread = t
        logger.info("Canary: health monitor started (interval=%ds)", HEALTH_CHECK_INTERVAL_S)

    def _stop_monitor(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(timeout=HEALTH_CHECK_INTERVAL_S):
            if self.active:
                try:
                    self._tick()
                except Exception as exc:
                    logger.error("Canary monitor tick error: %s", exc)
            elif self.post_promotion_active:
                try:
                    self._tick_post_promotion()
                except Exception as exc:
                    logger.error("Post-promotion monitor tick error: %s", exc)
            else:
                break
        logger.info("Canary: health monitor stopped")

    def _tick_post_promotion(self) -> None:
        """Check post-promotion error rate; revert files + alert if 3× baseline is exceeded."""
        promoted_at = self._state.get("promoted_at", "")
        try:
            since = datetime.fromisoformat(promoted_at.replace("Z", "+00:00")).isoformat()
        except Exception:
            return

        with self._conn() as c:
            rows = c.execute(
                "SELECT success FROM canary_health WHERE arm='post_promotion' AND ts >= ?",
                (since,),
            ).fetchall()

        n = len(rows)
        if n < POST_PROMOTION_MIN_SAMPLES:
            return   # not enough data yet

        error_rate = sum(1 for r in rows if not r["success"]) / n
        baseline   = float(self._state.get("pre_promotion_error_rate", 0.0))

        # Guard against a near-zero baseline masking a genuine spike — floor at 1%
        effective_baseline = max(baseline, 0.01)
        threshold          = effective_baseline * POST_PROMOTION_ROLLBACK_MULT

        if error_rate <= threshold:
            # Window expired without a spike — resolve cleanly
            if not self.post_promotion_active:
                with self._lock:
                    self._state["post_promotion_resolved"] = True
                    self._save_state()
                logger.info("Post-promotion window closed — no spike detected.")
            return

        # Spike detected → revert files + push alert
        logger.warning(
            "Post-promotion SPIKE: error_rate=%.1f%% > %.1f%% (%.0fx baseline=%.1f%%) "
            "over %d requests. Reverting to previous champion.",
            error_rate * 100, threshold * 100, POST_PROMOTION_ROLLBACK_MULT,
            effective_baseline * 100, n,
        )
        self._revert_to_previous(error_rate=error_rate, n_samples=n)

    def _revert_to_previous(self, error_rate: float, n_samples: int) -> None:
        """Swap previous.onnx back to nyc_xgb_model.onnx and push an alert."""
        try:
            if PREVIOUS_ONNX.exists():
                shutil.copy2(PREVIOUS_ONNX, CHAMPION_ONNX)
            if PREVIOUS_SCALER.exists():
                shutil.copy2(PREVIOUS_SCALER, CHAMPION_SCALER)
        except Exception as exc:
            logger.error("Post-promotion revert file swap failed: %s", exc)

        with self._lock:
            self._state.update({
                "post_promotion_resolved":  True,
                "post_promotion_reverted":  True,
                "post_promotion_revert_ts": datetime.now(timezone.utc).isoformat(),
            })
            self._state.setdefault("history", []).append({
                "event":      "post_promotion_revert",
                "error_rate": round(error_rate, 4),
                "n_samples":  n_samples,
                "ts":         datetime.now(timezone.utc).isoformat(),
            })
            self._save_state()

        # Push alert via nyc_alerts so it surfaces at GET /alerts
        try:
            sys.path.insert(0, str(BASE_DIR))
            from src.new_york_workflow.nyc_alerts import alerts
            alerts.push_once(
                alert_type = "post_promotion_revert",
                message    = (
                    f"Post-promotion auto-revert triggered: error_rate={error_rate:.1%} "
                    f"exceeded {POST_PROMOTION_ROLLBACK_MULT:.0f}× baseline. "
                    f"Previous champion files restored. "
                    f"Run 'docker compose restart api' to serve the reverted model."
                ),
                severity   = "critical",
                details    = {
                    "error_rate":   round(error_rate, 4),
                    "n_samples":    n_samples,
                    "multiplier":   POST_PROMOTION_ROLLBACK_MULT,
                    "baseline":     self._state.get("pre_promotion_error_rate"),
                    "action_needed": "docker compose restart api",
                },
            )
        except Exception as exc:
            logger.error("Failed to push post-promotion revert alert: %s", exc)

    def _tick(self) -> None:
        health = self.health_check()

        if not health.ok:
            logger.warning("Canary: UNHEALTHY at %d%% — %s", self.current_pct, health.reason)
            self.rollback(reason=health.reason)
            return

        stage_started = self._state.get("stage_started_at")
        if not stage_started:
            return

        try:
            elapsed = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(stage_started.replace("Z", "+00:00"))
            ).total_seconds()
        except Exception:
            return

        if elapsed >= MIN_STAGE_DURATION_S and health.n_samples >= MIN_STAGE_SAMPLES:
            logger.info(
                "Canary: %d%% healthy for %.0fs (%d samples) — auto-advancing",
                self.current_pct, elapsed, health.n_samples,
            )
            self.advance(manual=False)

    # ── status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        health = self.health_check() if self.active else None

        elapsed_s = None
        stage_started = self._state.get("stage_started_at")
        if stage_started and self.active:
            try:
                elapsed_s = round((
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(stage_started.replace("Z", "+00:00"))
                ).total_seconds())
            except Exception:
                pass

        return {
            "active":              self.active,
            "current_pct":        self.current_pct,
            "stages":             STAGES,
            "stage_idx":          self._state.get("stage_idx", 0),
            "started_at":         self._state.get("started_at"),
            "stage_started_at":   stage_started,
            "elapsed_at_stage_s": elapsed_s,
            "ended_reason":       self._state.get("ended_reason"),
            "challenger_exists":  CHALLENGER_ONNX.exists(),
            "post_promotion": {
                "active":           self.post_promotion_active,
                "promoted_at":      self._state.get("promoted_at"),
                "resolved":         self._state.get("post_promotion_resolved", False),
                "reverted":         self._state.get("post_promotion_reverted", False),
                "baseline_error_rate": self._state.get("pre_promotion_error_rate"),
                "window_s":         POST_PROMOTION_WINDOW_S,
                "rollback_mult":    POST_PROMOTION_ROLLBACK_MULT,
            } if self._state.get("promoted_at") else None,
            "health": {
                "ok":            health.ok,
                "error_rate_pct": round(health.error_rate * 100, 2),
                "p99_latency_ms": health.p99_latency,
                "n_samples":      health.n_samples,
                "reason":         health.reason,
            } if health else None,
            "thresholds": {
                "error_rate_pct":       ERROR_RATE_THRESHOLD * 100,
                "p99_latency_ms":       LATENCY_P99_MS,
                "min_stage_samples":    MIN_STAGE_SAMPLES,
                "min_stage_duration_s": MIN_STAGE_DURATION_S,
            },
            "history": self._state.get("history", [])[-10:],
        }
