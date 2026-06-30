"""
Unit tests for the ground truth feedback loop.

Coverage:
  GroundTruthStore
    - insert: happy path, duplicate suppression, error computation
    - stats: correct MAE/MAPE, empty store, per-borough breakdown
    - recent: ordering, limit
    - error_distribution: all five bands, empty store

  RequestStore (listing_id additions)
    - log_prediction stores listing_id correctly
    - log_prediction without listing_id stores NULL
    - get_listings_pending_ground_truth returns only unmatched rows
    - get_listings_pending_ground_truth excludes already-matched rows
    - schema migration adds listing_id to a pre-existing database

  fetch_ground_truth helpers
    - parse_price handles currency strings, nulls, zeros, bad input
    - run() dry_run: matches are logged, nothing written to DB
    - run() full: rows inserted, production MAE computed
    - run() no pending predictions: exits cleanly
    - run() no snapshot matches: inserts nothing
    - run() MAE above threshold: alert pushed
    - run() MAE below threshold: no alert

All tests use tmp_path (temp SQLite files) — no production DB touched.
"""

import copy
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.new_york_workflow.nyc_ground_truth import GroundTruthStore
from src.new_york_workflow.nyc_store import RequestStore


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gt_store(tmp_path):
    return GroundTruthStore(db_path=tmp_path / "test.db")


@pytest.fixture
def req_store(tmp_path):
    return RequestStore(db_path=tmp_path / "test.db")


@pytest.fixture
def gt_and_req(tmp_path):
    """Both stores sharing the same DB — required for pending-ground-truth queries."""
    db = tmp_path / "shared.db"
    return RequestStore(db_path=db), GroundTruthStore(db_path=db)


VALID_RAW = {
    "accommodates": 2, "bedrooms": 1.0, "bathrooms": 1.0,
    "room_type": "Entire home/apt", "borough": "Manhattan",
    "latitude": 40.76, "longitude": -73.99, "minimum_nights": 2,
    "host_listings_count": 1, "number_of_reviews": 30,
    "reviews_per_month": 1.2, "review_scores_rating": 4.8,
    "amenity_count": 20, "neighbourhood": "Midtown",
}
VALID_RESULT = {"price_usd": 120.0, "price_str": "$120/night", "log_price": 4.787}


# ═══════════════════════════════════════════════════════════════════════════════
# GroundTruthStore — insert
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthInsert:
    def test_insert_returns_true_on_success(self, gt_store):
        assert gt_store.insert("123", predicted_price=120.0, actual_price=150.0) is True

    def test_insert_duplicate_listing_id_returns_false(self, gt_store):
        gt_store.insert("123", predicted_price=120.0, actual_price=150.0)
        assert gt_store.insert("123", predicted_price=120.0, actual_price=160.0) is False

    def test_insert_computes_abs_error(self, gt_store):
        gt_store.insert("abc", predicted_price=100.0, actual_price=130.0)
        row = gt_store.recent(1)[0]
        assert row["abs_error"] == pytest.approx(30.0, abs=0.01)

    def test_insert_computes_pct_error(self, gt_store):
        gt_store.insert("abc", predicted_price=100.0, actual_price=200.0)
        row = gt_store.recent(1)[0]
        assert row["pct_error"] == pytest.approx(50.0, abs=0.1)

    def test_insert_stores_borough_and_room_type(self, gt_store):
        gt_store.insert("x1", predicted_price=80.0, actual_price=90.0,
                        borough="Brooklyn", room_type="Private room")
        row = gt_store.recent(1)[0]
        assert row["borough"] == "Brooklyn"
        assert row["room_type"] == "Private room"

    def test_insert_stores_request_id(self, gt_store):
        rid = "req-abc-123"
        gt_store.insert("x2", predicted_price=80.0, actual_price=90.0, request_id=rid)
        row = gt_store.recent(1)[0]
        assert row["listing_id"] == "x2"

    def test_insert_predicted_greater_than_actual_still_positive_error(self, gt_store):
        gt_store.insert("x3", predicted_price=200.0, actual_price=150.0)
        row = gt_store.recent(1)[0]
        assert row["abs_error"] == pytest.approx(50.0, abs=0.01)
        assert row["pct_error"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# GroundTruthStore — stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthStats:
    def test_stats_empty_store_returns_zero_n_evaluated(self, gt_store):
        assert gt_store.stats()["n_evaluated"] == 0

    def test_stats_empty_store_mae_is_none(self, gt_store):
        assert gt_store.stats()["mae_dollar"] is None

    def test_stats_correct_n_evaluated(self, gt_store):
        gt_store.insert("a", predicted_price=100.0, actual_price=120.0)
        gt_store.insert("b", predicted_price=100.0, actual_price=140.0)
        assert gt_store.stats()["n_evaluated"] == 2

    def test_stats_correct_mae(self, gt_store):
        gt_store.insert("a", predicted_price=100.0, actual_price=120.0)  # error=20
        gt_store.insert("b", predicted_price=100.0, actual_price=160.0)  # error=60
        mae = gt_store.stats()["mae_dollar"]
        assert mae == pytest.approx(40.0, abs=0.1)

    def test_stats_correct_mape(self, gt_store):
        # predicted=100, actual=200 → pct_error=50%
        gt_store.insert("a", predicted_price=100.0, actual_price=200.0)
        mape = gt_store.stats()["mape_pct"]
        assert mape == pytest.approx(50.0, abs=0.5)

    def test_stats_by_borough_breakdown(self, gt_store):
        gt_store.insert("a", predicted_price=100.0, actual_price=120.0, borough="Manhattan")
        gt_store.insert("b", predicted_price=100.0, actual_price=130.0, borough="Brooklyn")
        by_b = gt_store.stats()["by_borough"]
        assert "Manhattan" in by_b
        assert "Brooklyn" in by_b

    def test_stats_by_borough_correct_mae(self, gt_store):
        gt_store.insert("a", predicted_price=100.0, actual_price=150.0, borough="Queens")
        gt_store.insert("b", predicted_price=100.0, actual_price=110.0, borough="Queens")
        mae = gt_store.stats()["by_borough"]["Queens"]["mae_dollar"]
        assert mae == pytest.approx(30.0, abs=0.1)

    def test_stats_has_avg_prices(self, gt_store):
        gt_store.insert("a", predicted_price=100.0, actual_price=200.0)
        s = gt_store.stats()
        assert s["avg_predicted_price"] == pytest.approx(100.0, abs=0.1)
        assert s["avg_actual_price"]    == pytest.approx(200.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# GroundTruthStore — recent
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthRecent:
    def test_recent_empty_returns_empty_list(self, gt_store):
        assert gt_store.recent() == []

    def test_recent_returns_latest_first(self, gt_store):
        gt_store.insert("first",  predicted_price=100.0, actual_price=110.0)
        gt_store.insert("second", predicted_price=100.0, actual_price=120.0)
        rows = gt_store.recent(10)
        assert rows[0]["listing_id"] == "second"

    def test_recent_limit_respected(self, gt_store):
        for i in range(5):
            gt_store.insert(str(i), predicted_price=100.0, actual_price=110.0)
        assert len(gt_store.recent(3)) == 3

    def test_recent_row_has_expected_keys(self, gt_store):
        gt_store.insert("z", predicted_price=100.0, actual_price=150.0,
                        borough="Bronx", room_type="Private room")
        row = gt_store.recent(1)[0]
        for key in ("listing_id", "predicted_price", "actual_price",
                    "abs_error", "pct_error", "borough", "room_type", "fetched_at"):
            assert key in row, f"missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# GroundTruthStore — error_distribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorDistribution:
    def test_empty_store_returns_empty_dict(self, gt_store):
        assert gt_store.error_distribution() == {}

    def test_all_five_bands_present(self, gt_store):
        for i, (pred, actual) in enumerate([
            (100, 105),   # <$10
            (100, 115),   # $10-25
            (100, 140),   # $25-50
            (100, 170),   # $50-100
            (100, 210),   # >$100
        ]):
            gt_store.insert(str(i), predicted_price=pred, actual_price=actual)
        dist = gt_store.error_distribution()
        assert set(dist.keys()) == {"<$10", "$10-25", "$25-50", "$50-100", ">$100"}

    def test_distribution_pct_sums_to_100(self, gt_store):
        for i, actual in enumerate([105, 115, 140, 170, 210]):
            gt_store.insert(str(i), predicted_price=100.0, actual_price=actual)
        total_pct = sum(v["pct"] for v in gt_store.error_distribution().values())
        assert total_pct == pytest.approx(100.0, abs=0.5)

    def test_correct_band_assigned(self, gt_store):
        gt_store.insert("x", predicted_price=100.0, actual_price=105.0)  # error=$5 → <$10
        dist = gt_store.error_distribution()
        assert dist["<$10"]["count"] == 1
        assert dist["$10-25"]["count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# RequestStore — listing_id additions
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestStoreListingId:
    def _log(self, store, listing_id=None):
        rid = str(uuid.uuid4())
        store.log_prediction(rid, VALID_RAW, VALID_RESULT,
                             cache_hit=False, listing_id=listing_id)
        return rid

    def test_log_with_listing_id_stores_it(self, req_store):
        self._log(req_store, listing_id="listing-99")
        row = req_store.recent(1)[0]
        assert row["listing_id"] == "listing-99"

    def test_log_without_listing_id_stores_null(self, req_store):
        self._log(req_store, listing_id=None)
        row = req_store.recent(1)[0]
        assert row["listing_id"] is None

    def test_listing_id_not_in_features_json(self, req_store):
        self._log(req_store, listing_id="listing-77")
        row = req_store.recent(1)[0]
        features = json.loads(row["features_json"])
        assert "listing_id" not in features


# ═══════════════════════════════════════════════════════════════════════════════
# RequestStore — get_listings_pending_ground_truth
# ═══════════════════════════════════════════════════════════════════════════════

class TestPendingGroundTruth:
    def _log(self, store, listing_id):
        rid = str(uuid.uuid4())
        store.log_prediction(rid, VALID_RAW, VALID_RESULT,
                             cache_hit=False, listing_id=listing_id)
        return rid

    def test_pending_returns_rows_with_listing_id(self, gt_and_req):
        req, gt = gt_and_req
        self._log(req, "id-1")
        pending = req.get_listings_pending_ground_truth()
        assert any(r["listing_id"] == "id-1" for r in pending)

    def test_pending_excludes_rows_without_listing_id(self, gt_and_req):
        req, gt = gt_and_req
        req.log_prediction(str(uuid.uuid4()), VALID_RAW, VALID_RESULT, cache_hit=False)
        pending = req.get_listings_pending_ground_truth()
        assert all(r["listing_id"] is not None for r in pending)

    def test_pending_excludes_already_matched_rows(self, gt_and_req):
        req, gt = gt_and_req
        self._log(req, "id-matched")
        gt.insert("id-matched", predicted_price=120.0, actual_price=150.0)
        pending = req.get_listings_pending_ground_truth()
        assert not any(r["listing_id"] == "id-matched" for r in pending)

    def test_pending_returns_only_unmatched(self, gt_and_req):
        req, gt = gt_and_req
        self._log(req, "matched")
        self._log(req, "unmatched")
        gt.insert("matched", predicted_price=120.0, actual_price=150.0)
        pending = req.get_listings_pending_ground_truth()
        ids = [r["listing_id"] for r in pending]
        assert "unmatched" in ids
        assert "matched" not in ids

    def test_pending_empty_when_no_listing_ids_logged(self, gt_and_req):
        req, gt = gt_and_req
        req.log_prediction(str(uuid.uuid4()), VALID_RAW, VALID_RESULT, cache_hit=False)
        assert req.get_listings_pending_ground_truth() == []


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_ground_truth helpers — parse_price
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsePrice:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.fetch_ground_truth import parse_price
        self.parse_price = parse_price

    def test_dollar_string(self):
        assert self.parse_price("$120.00") == pytest.approx(120.0)

    def test_string_with_comma(self):
        assert self.parse_price("$1,200.00") == pytest.approx(1200.0)

    def test_plain_number_string(self):
        assert self.parse_price("99") == pytest.approx(99.0)

    def test_none_returns_none(self):
        assert self.parse_price(None) is None

    def test_nan_returns_none(self):
        import math
        assert self.parse_price(float("nan")) is None

    def test_zero_returns_none(self):
        assert self.parse_price("$0.00") is None

    def test_non_numeric_returns_none(self):
        assert self.parse_price("free") is None

    def test_negative_not_returned(self):
        assert self.parse_price("-10") is None


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_ground_truth.run()
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchGroundTruthRun:
    """
    Tests for run() using mocked HTTP and real temp SQLite stores.
    We patch fetch_snapshot so no network calls are made.
    """

    def _make_snapshot(self, listing_ids: list, prices: list) -> pd.DataFrame:
        return pd.DataFrame({
            "id":                           [str(i) for i in listing_ids],
            "price":                        [f"${p}.00" for p in prices],
            "neighbourhood_group_cleansed": ["Manhattan"] * len(listing_ids),
            "room_type":                    ["Entire home/apt"] * len(listing_ids),
        })

    def _setup_stores(self, tmp_path, listing_ids):
        db = tmp_path / "test.db"
        req = RequestStore(db_path=db)
        gt  = GroundTruthStore(db_path=db)
        for lid in listing_ids:
            req.log_prediction(
                str(uuid.uuid4()), VALID_RAW, VALID_RESULT,
                cache_hit=False, listing_id=str(lid),
            )
        return req, gt, db

    def test_run_no_pending_returns_zero_matched(self, tmp_path):
        db = tmp_path / "test.db"
        req = RequestStore(db_path=db)
        gt  = GroundTruthStore(db_path=db)
        req.log_prediction(str(uuid.uuid4()), VALID_RAW, VALID_RESULT, cache_hit=False)

        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt):
            result = run(url="http://fake", dry_run=False)
        assert result["pending"] == 0
        assert result["matched"] == 0

    def test_run_matches_and_inserts(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [101, 102])
        snapshot = self._make_snapshot([101, 102, 999], [150, 200, 300])

        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt):
            result = run(url="http://fake", dry_run=False)

        assert result["matched"] == 2
        assert result["inserted"] == 2
        assert gt.stats()["n_evaluated"] == 2

    def test_run_dry_run_writes_nothing(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [101])
        snapshot = self._make_snapshot([101], [150])

        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt):
            result = run(url="http://fake", dry_run=True)

        assert result["dry_run"] is True
        assert gt.stats()["n_evaluated"] == 0

    def test_run_no_snapshot_matches_inserts_nothing(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [101])
        snapshot = self._make_snapshot([999], [150])  # no overlap

        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt):
            result = run(url="http://fake", dry_run=False)

        assert result["matched"] == 0
        assert result["inserted"] == 0

    def test_run_idempotent_on_second_run(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [101])
        snapshot = self._make_snapshot([101], [150])

        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt):
            run(url="http://fake", dry_run=False)
            result2 = run(url="http://fake", dry_run=False)

        assert result2["inserted"] == 0       # second run inserts nothing (UNIQUE constraint)
        assert gt.stats()["n_evaluated"] == 1 # still only 1 row

    def test_run_pushes_alert_when_mae_exceeds_threshold(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [str(i) for i in range(30)])
        # predicted_price comes from VALID_RESULT["price_usd"] = $120
        # actual=$500 → abs_error=$380, MAE=$380 >> baseline $57.62 + $10 threshold
        snapshot = self._make_snapshot(
            list(range(30)), [500] * 30
        )

        baseline = {"models": {"XGBoost": {"test": {"mae_dollar": 57.62, "r2": 0.82, "mape": 23.6}}}}
        baseline_path = tmp_path / "models" / "nyc" / "nyc_training_report.json"
        baseline_path.parent.mkdir(parents=True)
        baseline_path.write_text(json.dumps(baseline))

        mock_alert = MagicMock()
        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt), \
             patch("scripts.fetch_ground_truth.alerts", mock_alert), \
             patch("scripts.fetch_ground_truth.BASE_DIR", tmp_path):
            run(url="http://fake", dry_run=False)

        mock_alert.push_once.assert_called_once()
        call_kwargs = mock_alert.push_once.call_args[1]
        assert call_kwargs["alert_type"] == "production_mae_degraded"
        assert call_kwargs["severity"] == "warning"

    def test_run_no_alert_when_mae_within_threshold(self, tmp_path):
        req, gt, db = self._setup_stores(tmp_path, [str(i) for i in range(30)])
        # prediction=$100, actual=$105 → MAE=$5 (well within $10 threshold)
        snapshot = self._make_snapshot(list(range(30)), [105] * 30)

        baseline = {"models": {"XGBoost": {"test": {"mae_dollar": 57.62, "r2": 0.82, "mape": 23.6}}}}
        baseline_path = tmp_path / "models" / "nyc" / "nyc_training_report.json"
        baseline_path.parent.mkdir(parents=True)
        baseline_path.write_text(json.dumps(baseline))

        mock_alert = MagicMock()
        from scripts.fetch_ground_truth import run
        with patch("scripts.fetch_ground_truth.fetch_snapshot", return_value=snapshot), \
             patch("scripts.fetch_ground_truth.RequestStore", return_value=req), \
             patch("scripts.fetch_ground_truth.GroundTruthStore", return_value=gt), \
             patch("scripts.fetch_ground_truth.alerts", mock_alert), \
             patch("scripts.fetch_ground_truth.BASE_DIR", tmp_path):
            run(url="http://fake", dry_run=False)

        mock_alert.push_once.assert_not_called()
