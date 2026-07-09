"""
Production-level tests for A/B testing and canary deployment.

Coverage:
  ABTest (ab.py)
    - Routing: determinism, distribution accuracy, inactive guard
    - Lifecycle: start, stop, promote, reject, restart, double-start guard
    - log_prediction: writes when active, no-op when stopped (bug fix verified),
                      no-op when never started
    - stats: no-test, no-data, insufficient ground truth, promote/monitor/reject
             recommendations, per-arm counts and accuracy via GT join
    - State persistence across instance restarts

  CanaryDeployment (canary.py)
    - Routing: determinism, distribution at 1% / 25%, inactive guard
    - Lifecycle: start (with/without challenger.onnx), advance through all stages,
                 manual advance, rollback, abort, double-start guard
    - Health check: insufficient samples, all-healthy, error-rate breach,
                    latency breach, p99 calculation (bug fix verified),
                    stage isolation (only checks current stage)
    - Promotion: file swap, backup creation, challenger removal
    - Status: shape, history entries
    - State persistence across instance restarts

  ShadowPredictor.predict_sync (shadow.py)
    - Returns None when shadow inactive
    - Returns correct dict shape when active (mocked session)
    - Returns None on inference exception (fallback path)
    - Price formatting is correct

  API endpoints (api.py)
    - /ab/start: 400 without challenger, 400 if already active
    - /ab/stats: idle shape, no-test message
    - /ab/stop: 400 when not active
    - /canary/start: 400 without challenger.onnx
    - /canary/status: idle shape
    - /canary/advance: 400 when not active
    - /canary/rollback: 400 when not active

All unit tests use tmp_path (isolated temp SQLite / JSON files).
No production DB, model files, or network calls are touched.
"""

import json
import sqlite3
import sys
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.ab import ABTest, MIN_GT_SAMPLES, PROMOTE_DELTA, REJECT_DELTA
from src.serving.canary import (
    CanaryDeployment,
    HealthResult,
    STAGES,
    ERROR_RATE_THRESHOLD,
    LATENCY_P99_MS,
    MIN_STAGE_SAMPLES,
    POST_PROMOTION_MIN_SAMPLES,
    POST_PROMOTION_ROLLBACK_MULT,
    POST_PROMOTION_WINDOW_S,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ab(tmp_path):
    """
    ABTest backed by isolated temp DB and state file.
    Also initialises GroundTruthStore against the same DB so the ground_truth
    table exists — required for stats() to run its JOIN without error.
    """
    from src.serving.ground_truth import GroundTruthStore
    db = tmp_path / "predictions.db"
    GroundTruthStore(db_path=db)   # creates ground_truth table in the shared DB
    return ABTest(db_path=db, state_path=tmp_path / "ab_state.json")


@pytest.fixture
def ab_active(ab):
    """ABTest already started at 20% split."""
    ab.start(20)
    return ab


@pytest.fixture
def canary_mod(monkeypatch, tmp_path):
    """
    Patches all module-level path constants in nyc_canary before any instance
    is created. Returns the patched module so tests can call CanaryDeployment().
    """
    import src.serving.canary as mod

    monkeypatch.setattr(mod, "STATE_PATH",        tmp_path / "canary_state.json")
    monkeypatch.setattr(mod, "HEALTH_DB",         tmp_path / "canary_health.db")
    monkeypatch.setattr(mod, "CHALLENGER_ONNX",   tmp_path / "challenger.onnx")
    monkeypatch.setattr(mod, "CHALLENGER_SCALER", tmp_path / "challenger_scaler.pkl")
    monkeypatch.setattr(mod, "CHAMPION_ONNX",     tmp_path / "nyc_xgb_model.onnx")
    monkeypatch.setattr(mod, "CHAMPION_SCALER",   tmp_path / "nyc_scaler.pkl")
    monkeypatch.setattr(mod, "PREVIOUS_ONNX",     tmp_path / "previous.onnx")
    monkeypatch.setattr(mod, "PREVIOUS_SCALER",   tmp_path / "previous_scaler.pkl")
    return mod


@pytest.fixture
def canary(canary_mod):
    """Fresh CanaryDeployment with isolated paths and no challenger.onnx."""
    c = canary_mod.CanaryDeployment()
    yield c
    c._stop_monitor()


@pytest.fixture
def canary_with_challenger(canary_mod, tmp_path):
    """CanaryDeployment with a fake challenger.onnx on disk."""
    (tmp_path / "challenger.onnx").write_bytes(b"fake-onnx")
    c = canary_mod.CanaryDeployment()
    yield c
    c._stop_monitor()


@pytest.fixture
def started_canary(canary_with_challenger):
    """Canary already started at stage 0 (1%)."""
    canary_with_challenger.start()
    return canary_with_challenger


# ═══════════════════════════════════════════════════════════════════════════════
# ABTest — routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestABRouting:

    def test_returns_champion_when_inactive(self, ab):
        assert ab.route("any-request-id") == "champion"

    def test_deterministic_same_id_always_same_arm(self, ab_active):
        rid = "stable-request-id"
        arms = {ab_active.route(rid) for _ in range(10)}
        assert len(arms) == 1, "same request_id must always map to the same arm"

    def test_distribution_approximates_split_pct(self, ab):
        ab.start(20)
        rids = [str(uuid.uuid4()) for _ in range(2000)]
        challenger_count = sum(1 for r in rids if ab.route(r) == "challenger")
        pct = challenger_count / len(rids) * 100
        assert 15 <= pct <= 25, f"expected ~20% challenger, got {pct:.1f}%"

    def test_distribution_at_1pct(self, ab):
        ab.start(1)
        rids = [str(uuid.uuid4()) for _ in range(2000)]
        challenger_count = sum(1 for r in rids if ab.route(r) == "challenger")
        assert challenger_count < 60, "1% split should rarely select challenger"

    def test_distribution_at_50pct(self, ab):
        ab.start(50)
        rids = [str(uuid.uuid4()) for _ in range(2000)]
        champion_count   = sum(1 for r in rids if ab.route(r) == "champion")
        challenger_count = sum(1 for r in rids if ab.route(r) == "challenger")
        assert abs(champion_count - challenger_count) < 200

    def test_route_consistent_across_restarts(self, tmp_path):
        """Same request_id maps to same arm even after ABTest is re-instantiated."""
        db    = tmp_path / "predictions.db"
        state = tmp_path / "ab_state.json"
        ab1 = ABTest(db_path=db, state_path=state)
        ab1.start(30)
        rid = "consistent-id"
        arm1 = ab1.route(rid)

        ab2 = ABTest(db_path=db, state_path=state)
        arm2 = ab2.route(rid)
        assert arm1 == arm2


# ═══════════════════════════════════════════════════════════════════════════════
# ABTest — lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestABLifecycle:

    def test_start_returns_success(self, ab):
        result = ab.start(20)
        assert result["success"] is True

    def test_start_activates_test(self, ab):
        ab.start(20)
        assert ab.active is True

    def test_start_sets_test_id(self, ab):
        ab.start(20)
        assert ab.test_id is not None
        assert ab.test_id.startswith("ab_")

    def test_start_when_already_active_returns_error(self, ab_active):
        result = ab_active.start(30)
        assert result["success"] is False
        assert "already active" in result["error"]

    def test_split_pct_below_1_rejected(self, ab):
        assert ab.start(0)["success"] is False
        assert ab.start(0.5)["success"] is False

    def test_split_pct_above_50_rejected(self, ab):
        assert ab.start(51)["success"] is False
        assert ab.start(100)["success"] is False

    def test_split_pct_at_boundary_values(self, ab):
        assert ab.start(1)["success"] is True
        ab.stop()
        assert ab.start(50)["success"] is True

    def test_stop_deactivates_test(self, ab_active):
        ab_active.stop()
        assert ab_active.active is False

    def test_stop_preserves_test_id(self, ab_active):
        tid = ab_active.test_id
        ab_active.stop()
        assert ab_active.test_id == tid

    def test_stop_when_never_started_returns_error(self, ab):
        """Bug fix: stop() on a fresh instance must not return success=True."""
        result = ab.stop()
        assert result["success"] is False
        assert "error" in result

    def test_stop_when_never_started_does_not_create_state_file(self, tmp_path):
        state = tmp_path / "ab_state.json"
        ab = ABTest(db_path=tmp_path / "p.db", state_path=state)
        ab.stop()
        assert not state.exists()

    def test_promote_stops_test(self, ab_active):
        ab_active.promote()
        assert ab_active.active is False

    def test_reject_stops_test(self, ab_active):
        ab_active.reject()
        assert ab_active.active is False

    def test_start_after_stop_creates_new_test_id(self, tmp_path):
        """
        Two consecutive tests must have distinct IDs so their ground-truth rows
        do not mix in the JOIN. test_id includes a uuid suffix to guarantee this
        even when both runs happen within the same second.
        """
        db    = tmp_path / "p.db"
        state = tmp_path / "state.json"
        ab = ABTest(db_path=db, state_path=state)
        ab.start(20)
        tid1 = ab.test_id
        ab.stop()
        ab.start(20)
        assert ab.test_id != tid1, (
            "each test run must have a unique ID — "
            "duplicate IDs would corrupt the ground-truth JOIN"
        )

    def test_state_persists_across_instances(self, tmp_path):
        db    = tmp_path / "p.db"
        state = tmp_path / "state.json"
        ab1 = ABTest(db_path=db, state_path=state)
        ab1.start(25)
        tid = ab1.test_id

        ab2 = ABTest(db_path=db, state_path=state)
        assert ab2.active is True
        assert ab2.test_id == tid


# ═══════════════════════════════════════════════════════════════════════════════
# ABTest — log_prediction
# ═══════════════════════════════════════════════════════════════════════════════

class TestABLogPrediction:

    def _count_rows(self, ab) -> int:
        with sqlite3.connect(str(ab._path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM ab_predictions").fetchone()[0]

    def test_log_writes_row_when_active(self, ab_active):
        ab_active.log_prediction("req-1", "listing-1", "champion", 150.0)
        assert self._count_rows(ab_active) == 1

    def test_log_records_correct_arm(self, ab_active):
        ab_active.log_prediction("req-1", "listing-1", "challenger", 180.0)
        with sqlite3.connect(str(ab_active._path)) as conn:
            row = conn.execute("SELECT arm FROM ab_predictions").fetchone()
        assert row[0] == "challenger"

    def test_log_records_correct_price(self, ab_active):
        ab_active.log_prediction("req-1", "listing-1", "champion", 123.45)
        with sqlite3.connect(str(ab_active._path)) as conn:
            price = conn.execute("SELECT predicted_price FROM ab_predictions").fetchone()[0]
        assert price == pytest.approx(123.45, abs=0.01)

    def test_log_with_no_listing_id_still_inserts(self, ab_active):
        ab_active.log_prediction("req-1", None, "champion", 100.0)
        assert self._count_rows(ab_active) == 1

    def test_log_is_noop_when_test_stopped(self, ab_active):
        """Bug fix: log_prediction() must not write rows after stop()."""
        ab_active.stop()
        ab_active.log_prediction("req-1", "listing-1", "champion", 150.0)
        assert self._count_rows(ab_active) == 0

    def test_log_is_noop_when_never_started(self, ab):
        ab.log_prediction("req-1", "listing-1", "champion", 150.0)
        assert self._count_rows(ab) == 0

    def test_multiple_logs_accumulate(self, ab_active):
        for i in range(5):
            ab_active.log_prediction(f"req-{i}", f"listing-{i}", "champion", 100.0 + i)
        assert self._count_rows(ab_active) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# ABTest — stats
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_ab_gt(tmp_path, champion_mae_error: float, challenger_mae_error: float,
                n_per_arm: int = MIN_GT_SAMPLES):
    """
    Seed an AB test and matching ground truth rows so stats() has data to JOIN.
    champion_mae_error / challenger_mae_error are the per-listing abs errors.
    Returns (ab, ground_truth_store).
    """
    from src.serving.ground_truth import GroundTruthStore

    db    = tmp_path / "predictions.db"
    state = tmp_path / "ab_state.json"
    ab = ABTest(db_path=db, state_path=state)
    ab.start(50)
    gt = GroundTruthStore(db_path=db)

    actual = 150.0
    for i in range(n_per_arm):
        lid_champ = f"champ-{i}"
        lid_chal  = f"chal-{i}"
        rid_champ = str(uuid.uuid4())
        rid_chal  = str(uuid.uuid4())

        champ_pred = actual - champion_mae_error
        chal_pred  = actual - challenger_mae_error

        ab.log_prediction(rid_champ, lid_champ, "champion",   champ_pred)
        ab.log_prediction(rid_chal,  lid_chal,  "challenger", chal_pred)

        gt.insert(lid_champ, predicted_price=champ_pred, actual_price=actual,
                  request_id=rid_champ)
        gt.insert(lid_chal,  predicted_price=chal_pred,  actual_price=actual,
                  request_id=rid_chal)

    return ab, gt


class TestABStats:

    def test_stats_no_test_returns_no_test_message(self, ab):
        result = ab.stats()
        assert "message" in result
        assert result["test_id"] is None

    def test_stats_active_false_when_inactive(self, ab):
        assert ab.stats()["active"] is False

    def test_stats_active_true_when_running(self, ab_active):
        assert ab_active.stats()["active"] is True

    def test_stats_zero_predictions_when_no_requests(self, ab_active):
        s = ab_active.stats()
        assert s["predictions_served"]["champion"]   == 0
        assert s["predictions_served"]["challenger"] == 0

    def test_stats_insufficient_gt_recommendation(self, ab_active):
        ab_active.log_prediction("r1", "l1", "champion",   100.0)
        ab_active.log_prediction("r2", "l2", "challenger", 100.0)
        assert ab_active.stats()["recommendation"] == "insufficient_data"

    def test_stats_promote_when_challenger_clearly_better(self, tmp_path):
        # champion error=$30, challenger error=$10 → delta=$20 > PROMOTE_DELTA=$2
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=30.0, challenger_mae_error=10.0)
        result = ab.stats()
        assert result["recommendation"] == "promote", result["recommendation_reason"]

    def test_stats_reject_when_challenger_clearly_worse(self, tmp_path):
        # champion error=$10, challenger error=$20 → delta=-$10 < REJECT_DELTA=-$5
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=10.0, challenger_mae_error=20.0)
        result = ab.stats()
        assert result["recommendation"] == "reject", result["recommendation_reason"]

    def test_stats_monitor_when_difference_too_small(self, tmp_path):
        # champion error=$15, challenger error=$14 → delta=$1 < PROMOTE_DELTA=$2
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=15.0, challenger_mae_error=14.0)
        result = ab.stats()
        assert result["recommendation"] == "monitor", result["recommendation_reason"]

    def test_stats_counts_predictions_per_arm(self, ab_active):
        for i in range(3):
            ab_active.log_prediction(f"r-c-{i}", f"l-c-{i}", "champion",   100.0)
        for i in range(5):
            ab_active.log_prediction(f"r-ch-{i}", f"l-ch-{i}", "challenger", 100.0)
        s = ab_active.stats()
        assert s["predictions_served"]["champion"]   == 3
        assert s["predictions_served"]["challenger"] == 5

    def test_stats_gt_accuracy_has_expected_keys(self, tmp_path):
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=20.0, challenger_mae_error=10.0)
        s = ab.stats()
        for arm in ("champion", "challenger"):
            d = s["ground_truth_accuracy"][arm]
            assert "n_gt"     in d
            assert "mae"      in d
            assert "mape_pct" in d

    def test_stats_correct_mae_values(self, tmp_path):
        # champion error=$20, challenger error=$5 → exact MAE should match
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=20.0, challenger_mae_error=5.0)
        s = ab.stats()
        champ_mae = s["ground_truth_accuracy"]["champion"]["mae"]
        chal_mae  = s["ground_truth_accuracy"]["challenger"]["mae"]
        assert champ_mae == pytest.approx(20.0, abs=0.5)
        assert chal_mae  == pytest.approx(5.0,  abs=0.5)

    def test_stats_insufficient_gt_below_min_samples(self, tmp_path):
        ab, _ = _seed_ab_gt(tmp_path, champion_mae_error=30.0, challenger_mae_error=5.0,
                             n_per_arm=MIN_GT_SAMPLES - 1)
        assert ab.stats()["recommendation"] == "insufficient_data"

    def test_stats_thresholds_are_present(self, ab_active):
        s = ab_active.stats()
        t = s["thresholds"]
        assert "min_gt_samples_per_arm"              in t
        assert "promote_if_challenger_better_by_usd" in t
        assert "reject_if_challenger_worse_by_usd"   in t


# ═══════════════════════════════════════════════════════════════════════════════
# CanaryDeployment — routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryRouting:

    def test_returns_champion_when_inactive(self, canary):
        assert canary.route("any-id") == "champion"

    def test_deterministic_same_id_always_same_arm(self, started_canary):
        rid  = "stable-canary-id"
        arms = {started_canary.route(rid) for _ in range(10)}
        assert len(arms) == 1

    def test_distribution_at_1pct_very_few_challenger(self, started_canary):
        assert started_canary.current_pct == 1
        rids = [str(uuid.uuid4()) for _ in range(2000)]
        n_chal = sum(1 for r in rids if started_canary.route(r) == "challenger")
        assert n_chal < 60, f"expected ~1% challenger, got {n_chal/20:.1f}%"

    def test_distribution_at_25pct(self, started_canary):
        started_canary.advance(manual=True)  # 1% → 5%
        started_canary.advance(manual=True)  # 5% → 25%
        assert started_canary.current_pct == 25
        rids   = [str(uuid.uuid4()) for _ in range(2000)]
        n_chal = sum(1 for r in rids if started_canary.route(r) == "challenger")
        pct    = n_chal / len(rids) * 100
        assert 19 <= pct <= 31, f"expected ~25% challenger, got {pct:.1f}%"

    def test_all_champion_after_rollback(self, started_canary):
        started_canary.rollback()
        rids = [str(uuid.uuid4()) for _ in range(500)]
        assert all(started_canary.route(r) == "champion" for r in rids)

    def test_routing_consistent_across_instances(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"fake")
        c1 = canary_mod.CanaryDeployment()
        c1.start()
        rid  = "must-be-stable"
        arm1 = c1.route(rid)

        c2 = canary_mod.CanaryDeployment()
        arm2 = c2.route(rid)
        assert arm1 == arm2
        c1._stop_monitor()
        c2._stop_monitor()


# ═══════════════════════════════════════════════════════════════════════════════
# CanaryDeployment — lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryLifecycle:

    def test_start_fails_without_challenger_onnx(self, canary):
        result = canary.start()
        assert result["success"] is False
        assert "challenger.onnx" in result["error"]

    def test_start_succeeds_with_challenger_onnx(self, canary_with_challenger):
        result = canary_with_challenger.start()
        assert result["success"] is True
        assert canary_with_challenger.active is True

    def test_start_begins_at_first_stage(self, canary_with_challenger):
        canary_with_challenger.start()
        assert canary_with_challenger.current_pct == STAGES[0]

    def test_start_when_already_active_returns_error(self, started_canary):
        result = started_canary.start()
        assert result["success"] is False
        assert "already active" in result["error"]

    def test_advance_moves_to_next_stage(self, started_canary):
        started_canary.advance(manual=True)
        assert started_canary.current_pct == STAGES[1]

    def test_advance_through_all_stages_up_to_last(self, started_canary, tmp_path):
        """Advance through 1%→5%→25%. At 100% the advance triggers a promote
        which needs real files — stop before that stage in this test."""
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"champion")
        started_canary.advance(manual=True)  # 1→5
        assert started_canary.current_pct == STAGES[1]
        started_canary.advance(manual=True)  # 5→25
        assert started_canary.current_pct == STAGES[2]

    def test_rollback_deactivates_canary(self, started_canary):
        started_canary.rollback()
        assert started_canary.active is False

    def test_rollback_resets_pct_to_zero(self, started_canary):
        started_canary.rollback()
        assert started_canary.current_pct == 0

    def test_rollback_preserves_challenger_file(self, canary_with_challenger, tmp_path):
        canary_with_challenger.start()
        canary_with_challenger.rollback()
        assert (tmp_path / "challenger.onnx").exists()

    def test_rollback_records_reason_in_history(self, started_canary):
        started_canary.rollback(reason="test-reason")
        history = started_canary.status()["history"]
        assert any(e["event"] == "rollback" and e["reason"] == "test-reason"
                   for e in history)

    def test_abort_deactivates_canary(self, started_canary):
        started_canary.abort()
        assert started_canary.active is False

    def test_advance_when_not_active_returns_error(self, canary):
        result = canary.advance(manual=True)
        assert result["success"] is False

    def test_state_persists_across_instances(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"fake")
        c1 = canary_mod.CanaryDeployment()
        c1.start()
        c1.advance(manual=True)   # 1% → 5%
        assert c1.current_pct == STAGES[1]
        c1._stop_monitor()

        c2 = canary_mod.CanaryDeployment()
        assert c2.active is True
        assert c2.current_pct == STAGES[1]
        c2._stop_monitor()


# ═══════════════════════════════════════════════════════════════════════════════
# CanaryDeployment — health check
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_health(canary, n_success: int, n_error: int, latency_ms: float = 50.0,
                 spike_ms: float | None = None):
    """Insert n_success + n_error challenger health rows at current stage."""
    for _ in range(n_success):
        canary.record_result("challenger", latency_ms, success=True)
    for _ in range(n_error):
        canary.record_result("challenger", latency_ms, success=False)
    if spike_ms is not None:
        canary.record_result("challenger", spike_ms, success=True)


class TestCanaryHealthCheck:

    def test_insufficient_samples_returns_ok(self, started_canary):
        _seed_health(started_canary, n_success=MIN_STAGE_SAMPLES - 1, n_error=0)
        h = started_canary.health_check()
        assert h.ok is True
        assert h.n_samples < MIN_STAGE_SAMPLES

    def test_all_healthy_returns_ok(self, started_canary):
        _seed_health(started_canary, n_success=MIN_STAGE_SAMPLES, n_error=0)
        h = started_canary.health_check()
        assert h.ok is True
        assert h.error_rate == 0.0

    def test_high_error_rate_returns_not_ok(self, started_canary):
        # 6 errors out of 100 = 6% > 5% threshold
        _seed_health(started_canary, n_success=94, n_error=6)
        h = started_canary.health_check()
        assert h.ok is False
        assert "error_rate" in h.reason

    def test_error_rate_exactly_at_threshold_is_ok(self, started_canary):
        # 5 errors out of 100 = 5% = threshold (not strictly greater)
        _seed_health(started_canary, n_success=95, n_error=5)
        h = started_canary.health_check()
        assert h.ok is True

    def test_high_latency_p99_returns_not_ok(self, started_canary):
        # 99 fast requests + 1 spike above threshold
        # p99 fix: index = min(99, int(100*0.99)) = min(99,99) = 99 → spike value
        _seed_health(started_canary, n_success=99, n_error=0, latency_ms=100.0,
                     spike_ms=LATENCY_P99_MS + 1.0)
        h = started_canary.health_check()
        assert h.ok is False
        assert "latency" in h.reason

    def test_p99_fix_spike_at_index_99_is_caught(self, started_canary):
        """
        Regression test for the off-by-one p99 bug.

        Old formula: latencies[max(0, int(n*0.99)-1)] with n=100
          → index 98 → misses spike at index 99 → health passes (WRONG)

        Fixed formula: latencies[min(n-1, int(n*0.99))] with n=100
          → index 99 → catches spike → health fails (CORRECT)
        """
        _seed_health(started_canary, n_success=99, n_error=0, latency_ms=50.0,
                     spike_ms=LATENCY_P99_MS + 500.0)
        h = started_canary.health_check()
        assert h.ok is False, (
            "p99 health check should catch latency spike at index 99 (last element of 100)"
        )

    def test_latency_at_threshold_exactly_is_ok(self, started_canary):
        _seed_health(started_canary, n_success=99, n_error=0, latency_ms=50.0,
                     spike_ms=float(LATENCY_P99_MS))
        h = started_canary.health_check()
        assert h.ok is True

    def test_health_only_checks_challenger_arm(self, started_canary):
        """Champion errors must not affect challenger health check."""
        for _ in range(50):
            started_canary.record_result("champion", 50.0, success=False)
        _seed_health(started_canary, n_success=MIN_STAGE_SAMPLES, n_error=0)
        h = started_canary.health_check()
        assert h.ok is True

    def test_health_only_checks_current_stage(self, started_canary):
        """Rows from a previous stage must not pollute the current stage's health."""
        # Record errors at stage 1 (1%)
        for _ in range(50):
            started_canary.record_result("challenger", 50.0, success=False)

        # Advance to stage 2 (5%) — previous errors are at stage_pct=1
        started_canary.advance(manual=True)
        # All healthy at stage 2
        _seed_health(started_canary, n_success=MIN_STAGE_SAMPLES, n_error=0)
        h = started_canary.health_check()
        assert h.ok is True, "previous stage errors should not affect current stage health"

    def test_health_check_returns_named_tuple_shape(self, started_canary):
        h = started_canary.health_check()
        assert hasattr(h, "ok")
        assert hasattr(h, "error_rate")
        assert hasattr(h, "p99_latency")
        assert hasattr(h, "n_samples")
        assert hasattr(h, "reason")


# ═══════════════════════════════════════════════════════════════════════════════
# CanaryDeployment — promotion (file swap)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryPromotion:

    def test_advance_to_100_promotes_challenger(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"new-model")
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"old-model")
        c = canary_mod.CanaryDeployment()
        c.start()
        c.advance(manual=True)  # 1→5
        c.advance(manual=True)  # 5→25
        c.advance(manual=True)  # 25→100 → promote
        c._stop_monitor()

        champion = (tmp_path / "nyc_xgb_model.onnx").read_bytes()
        assert champion == b"new-model", "champion should now contain challenger bytes"

    def test_promote_removes_challenger_onnx(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"new-model")
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"old-model")
        c = canary_mod.CanaryDeployment()
        c.start()
        c.advance(manual=True)
        c.advance(manual=True)
        c.advance(manual=True)
        c._stop_monitor()

        assert not (tmp_path / "challenger.onnx").exists()

    def test_promote_creates_previous_backup(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"new-model")
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"old-model")
        c = canary_mod.CanaryDeployment()
        c.start()
        c.advance(manual=True)
        c.advance(manual=True)
        c.advance(manual=True)
        c._stop_monitor()

        backup = (tmp_path / "previous.onnx").read_bytes()
        assert backup == b"old-model", "previous.onnx should contain old champion bytes"

    def test_promote_deactivates_canary(self, canary_mod, tmp_path):
        (tmp_path / "challenger.onnx").write_bytes(b"new-model")
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"old-model")
        c = canary_mod.CanaryDeployment()
        c.start()
        for _ in range(3):
            c.advance(manual=True)
        c._stop_monitor()

        assert c.active is False

    def test_promote_fails_gracefully_without_challenger(self, canary):
        """Promote when challenger.onnx doesn't exist must not crash."""
        canary._state = {"active": True, "stage_idx": 3, "current_pct": 100}
        result = canary._promote()
        assert result["success"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# CanaryDeployment — status
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryStatus:

    def test_status_inactive_shape(self, canary):
        s = canary.status()
        assert s["active"] is False
        assert s["current_pct"] == 0
        assert "stages" in s
        assert "thresholds" in s
        assert s["health"] is None

    def test_status_active_shape(self, started_canary):
        s = started_canary.status()
        assert s["active"] is True
        assert s["current_pct"] == STAGES[0]
        assert s["health"] is not None
        assert "ok" in s["health"]

    def test_status_history_grows_on_advance(self, started_canary):
        started_canary.advance(manual=True)
        history = started_canary.status()["history"]
        assert len(history) >= 1
        assert history[-1]["event"] == "advance"

    def test_status_history_capped_at_10(self, started_canary):
        # Rollback and restart multiple times to accumulate history
        for _ in range(15):
            started_canary._state.setdefault("history", []).append({"event": "fake"})
        history = started_canary.status()["history"]
        assert len(history) <= 10

    def test_status_challenger_exists_reflects_file(self, started_canary, tmp_path):
        s = started_canary.status()
        assert s["challenger_exists"] is True
        (tmp_path / "challenger.onnx").unlink()
        s2 = started_canary.status()
        assert s2["challenger_exists"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowPredictor.predict_sync
# ═══════════════════════════════════════════════════════════════════════════════

def _make_shadow_with_mock_session(log_price: float = 4.6):
    """
    Build a ShadowPredictor instance with all ONNX plumbing replaced by mocks.
    Avoids needing real model files. Returns (shadow, mock_champion).
    """
    from src.serving.shadow import ShadowPredictor

    shadow = ShadowPredictor.__new__(ShadowPredictor)
    shadow._session_lock = threading.Lock()
    shadow._db_lock      = threading.Lock()

    mock_sess = MagicMock()
    mock_sess.run.return_value = [np.array([[log_price]], dtype=np.float32)]
    shadow._session          = mock_sess
    shadow._input_name       = "float_input"
    shadow._challenger_scaler = None

    mock_champ = MagicMock()
    mock_champ._engineer.return_value = {"feat_a": 1.0, "feat_b": 2.0}
    mock_champ.feature_list           = ["feat_a", "feat_b"]
    mock_champ.scaler                 = MagicMock()
    mock_champ.scaler.transform.return_value = np.zeros((1, 2), dtype=np.float32)
    shadow._champion = mock_champ

    return shadow, mock_champ


class TestShadowPredictSync:

    def test_returns_none_when_session_is_none(self):
        from src.serving.shadow import ShadowPredictor
        s = ShadowPredictor.__new__(ShadowPredictor)
        s._session_lock = threading.Lock()
        s._session      = None
        s._champion     = MagicMock()
        assert s.predict_sync({}) is None

    def test_returns_none_when_champion_is_none(self):
        from src.serving.shadow import ShadowPredictor
        s = ShadowPredictor.__new__(ShadowPredictor)
        s._session_lock = threading.Lock()
        s._session      = MagicMock()
        s._champion     = None
        assert s.predict_sync({}) is None

    def test_returns_dict_with_required_keys(self):
        shadow, _ = _make_shadow_with_mock_session(log_price=4.6)
        result = shadow.predict_sync({"accommodates": 2})
        assert result is not None
        assert "price_usd"  in result
        assert "price_str"  in result
        assert "log_price"  in result

    def test_price_usd_is_expm1_of_log_price(self):
        log_price = 4.6
        shadow, _ = _make_shadow_with_mock_session(log_price=log_price)
        result = shadow.predict_sync({})
        expected = float(np.expm1(log_price))
        assert result["price_usd"] == pytest.approx(expected, abs=0.01)

    def test_price_str_includes_dollar_sign(self):
        shadow, _ = _make_shadow_with_mock_session()
        result = shadow.predict_sync({})
        assert result["price_str"].startswith("$")
        assert "/night" in result["price_str"]

    def test_log_price_is_rounded(self):
        shadow, _ = _make_shadow_with_mock_session(log_price=4.123456789)
        result = shadow.predict_sync({})
        assert len(str(result["log_price"]).split(".")[-1]) <= 4

    def test_returns_none_on_inference_exception(self):
        shadow, _ = _make_shadow_with_mock_session()
        shadow._session.run.side_effect = RuntimeError("ONNX exploded")
        assert shadow.predict_sync({}) is None

    def test_returns_none_on_engineer_exception(self):
        shadow, mock_champ = _make_shadow_with_mock_session()
        mock_champ._engineer.side_effect = ValueError("bad feature")
        assert shadow.predict_sync({}) is None

    def test_uses_challenger_scaler_when_available(self):
        shadow, mock_champ = _make_shadow_with_mock_session()
        mock_challenger_scaler = MagicMock()
        mock_challenger_scaler.transform.return_value = np.zeros((1, 2), dtype=np.float32)
        shadow._challenger_scaler = mock_challenger_scaler

        shadow.predict_sync({"accommodates": 2})
        mock_challenger_scaler.transform.assert_called_once()
        mock_champ.scaler.transform.assert_not_called()

    def test_falls_back_to_champion_scaler_when_challenger_scaler_missing(self):
        shadow, mock_champ = _make_shadow_with_mock_session()
        shadow._challenger_scaler = None

        shadow.predict_sync({})
        mock_champ.scaler.transform.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoints — A/B and canary (integration via TestClient)
# ═══════════════════════════════════════════════════════════════════════════════

class TestABEndpoints:
    """
    Tests against the live FastAPI app (full model loaded).
    A/B start requires shadow.active (challenger.onnx), which does not exist
    in the test environment, so we test the 400 path and idle paths.
    """

    def test_ab_stats_returns_200(self, client):
        assert client.get("/ab/stats").status_code == 200

    def test_ab_stats_idle_has_expected_shape(self, client):
        body = client.get("/ab/stats").json()
        assert "active" in body
        assert body["active"] is False

    def test_ab_start_without_challenger_returns_400(self, client):
        r = client.post("/ab/start?split_pct=20")
        assert r.status_code == 400

    def test_ab_stop_when_not_active_returns_400(self, client):
        r = client.post("/ab/stop")
        assert r.status_code == 400

    def test_ab_promote_returns_200_and_next_hint(self, client):
        r = client.post("/ab/promote")
        body = r.json()
        assert r.status_code == 200
        assert "next" in body

    def test_ab_reject_returns_200(self, client):
        assert client.post("/ab/reject").status_code == 200


class TestCanaryEndpoints:
    """
    Tests against the live FastAPI app.
    Canary start requires challenger.onnx, which does not exist in the test
    environment — we test the 400 path and all idle/control paths.
    """

    def test_canary_status_returns_200(self, client):
        assert client.get("/canary/status").status_code == 200

    def test_canary_status_inactive_shape(self, client):
        body = client.get("/canary/status").json()
        assert "active"      in body
        assert "stages"      in body
        assert "thresholds"  in body
        assert body["health"] is None

    def test_canary_start_without_challenger_returns_400(self, client):
        r = client.post("/canary/start")
        assert r.status_code == 400

    def test_canary_advance_when_not_active_returns_400(self, client):
        r = client.post("/canary/advance")
        assert r.status_code == 400

    def test_canary_rollback_when_not_active_returns_400(self, client):
        r = client.post("/canary/rollback")
        assert r.status_code == 400

    def test_canary_abort_returns_200_when_not_active(self, client):
        # abort is idempotent — should not 400 when inactive
        r = client.post("/canary/abort")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROMOTION MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostPromotionMonitor:
    """
    Tests for the 1-hour post-promotion window that auto-reverts files if the
    new champion's error rate jumps 3× the canary baseline.
    """

    def _make_promoted_canary(self, canary_mod, tmp_path, pre_error_rate=0.02):
        """Set up a CanaryDeployment whose state.json looks like a just-promoted canary."""
        import json
        from datetime import datetime, timezone
        state = {
            "active": False,
            "current_pct": 100,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "ended_reason": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "pre_promotion_error_rate": pre_error_rate,
            "post_promotion_resolved": False,
            "history": [],
        }
        (tmp_path / "canary_state.json").write_text(json.dumps(state))
        # Create dummy previous files so revert can copy them
        (tmp_path / "previous.onnx").write_bytes(b"prev-onnx")
        (tmp_path / "nyc_xgb_model.onnx").write_bytes(b"new-onnx")
        return CanaryDeployment()

    def test_post_promotion_active_immediately_after_promotion(self, canary_mod, tmp_path):
        c = self._make_promoted_canary(canary_mod, tmp_path)
        assert c.post_promotion_active is True

    def test_post_promotion_inactive_when_no_promoted_at(self, canary):
        assert canary.post_promotion_active is False

    def test_post_promotion_resolved_when_flag_set(self, canary_mod, tmp_path):
        import json
        from datetime import datetime, timezone
        state = {
            "active": False, "current_pct": 100,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "pre_promotion_error_rate": 0.01,
            "post_promotion_resolved": True,
        }
        (tmp_path / "canary_state.json").write_text(json.dumps(state))
        c = CanaryDeployment()
        assert c.post_promotion_active is False

    def test_record_post_promotion_no_op_when_inactive(self, canary):
        # Should not raise even when monitoring is off
        canary.record_post_promotion(50.0, success=True)

    def test_record_post_promotion_writes_when_active(self, canary_mod, tmp_path):
        import sqlite3
        c = self._make_promoted_canary(canary_mod, tmp_path)
        assert c.post_promotion_active
        c.record_post_promotion(42.0, success=True)
        with sqlite3.connect(str(tmp_path / "canary_health.db")) as conn:
            rows = conn.execute("SELECT arm FROM canary_health").fetchall()
        assert any(r[0] == "post_promotion" for r in rows)

    def test_tick_post_promotion_no_revert_below_threshold(self, canary_mod, tmp_path):
        c = self._make_promoted_canary(canary_mod, tmp_path, pre_error_rate=0.02)
        # All successes → 0% error rate, well below 0.02*3=6% threshold
        for _ in range(POST_PROMOTION_MIN_SAMPLES):
            c.record_post_promotion(30.0, success=True)
        c._tick_post_promotion()
        assert not c._state.get("post_promotion_reverted")

    def test_tick_post_promotion_reverts_on_spike(self, canary_mod, tmp_path):
        c = self._make_promoted_canary(canary_mod, tmp_path, pre_error_rate=0.02)
        # Feed spiked requests — 50% error_rate >> 0.02 * 3 = 0.06 threshold
        for i in range(POST_PROMOTION_MIN_SAMPLES):
            c.record_post_promotion(30.0, success=(i % 2 == 0))
        # Suppress alert push (alerts not wired in unit context)
        with patch("src.serving.canary.CanaryDeployment._revert_to_previous",
                   wraps=c._revert_to_previous) as spy:
            c._tick_post_promotion()
        assert c._state.get("post_promotion_reverted") is True

    def test_revert_swaps_files(self, canary_mod, tmp_path):
        c = self._make_promoted_canary(canary_mod, tmp_path, pre_error_rate=0.01)
        prev_content = (tmp_path / "previous.onnx").read_bytes()
        c._revert_to_previous(error_rate=0.5, n_samples=50)
        # Champion should now contain what was in previous.onnx
        assert (tmp_path / "nyc_xgb_model.onnx").read_bytes() == prev_content

    def test_status_includes_post_promotion_section(self, canary_mod, tmp_path):
        c = self._make_promoted_canary(canary_mod, tmp_path)
        s = c.status()
        assert s["post_promotion"] is not None
        assert s["post_promotion"]["active"] is True
        assert "baseline_error_rate" in s["post_promotion"]
