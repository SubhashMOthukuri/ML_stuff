"""
Unit tests for nyc_data_validator.py — all three pipeline gates.

Each test builds the minimal DataFrame needed to trigger (or pass) a specific
check, so failures pinpoint exactly which rule broke rather than which function.

No network calls, no model files, no production DB touched.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.data_validator import (
    DataQualityError,
    DataQualityReport,
    BOROUGHS,
    ROOM_TYPES,
    _REQUIRED_ENGINEERED,
    save_baseline,
    load_baseline,
    validate_clean_listings,
    validate_engineered_features,
    validate_raw_snapshot,
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid DataFrame factories
# ─────────────────────────────────────────────────────────────────────────────

N = 12_000   # above all minimum-row-count thresholds


def _raw_df(n: int = N, **overrides) -> pd.DataFrame:
    """Minimal valid raw InsideAirbnb snapshot."""
    df = pd.DataFrame({
        "id":                           [str(i) for i in range(n)],
        "price":                        ["$120.00"] * n,
        "neighbourhood_group_cleansed": ["Brooklyn"] * n,
        "room_type":                    ["Entire home/apt"] * n,
        "latitude":                     [40.68] * n,
        "longitude":                    [-73.95] * n,
    })
    for col, val in overrides.items():
        df[col] = val
    return df


def _clean_df(n: int = N, **overrides) -> pd.DataFrame:
    """Minimal valid cleaned listings DataFrame."""
    df = pd.DataFrame({
        "price":                        [120.0] * n,
        "accommodates":                 [2] * n,
        "bedrooms":                     [1.0] * n,
        "bathrooms":                    [1.0] * n,
        "host_is_superhost":            [0] * n,
        "host_listings_count":          [1.0] * n,
        "latitude":                     [40.68] * n,
        "longitude":                    [-73.95] * n,
        "minimum_nights":               [2.0] * n,
        "number_of_reviews":            [30.0] * n,
        "reviews_per_month":            [1.2] * n,
        "review_scores_rating":         [4.8] * n,
        "amenity_count":                [25.0] * n,
        "is_private_bath":              [1] * n,
        "neighbourhood_group_cleansed": ["Brooklyn"] * n,
    })
    for col, val in overrides.items():
        df[col] = val
    return df


def _engineered_df(n: int = N, **overrides) -> pd.DataFrame:
    """Minimal valid engineered feature matrix."""
    base = {
        "log_price":          [4.79] * n,
        "log_accommodates":   [1.10] * n,
        "log_bedrooms":       [0.69] * n,
        "log_bathrooms":      [0.69] * n,
        "log_amenity_count":  [3.26] * n,
        "log_host_listings_count": [0.69] * n,
        "log_number_of_reviews":   [3.43] * n,
        "log_number_of_reviews_ltm": [2.20] * n,
        "log_reviews_per_month":   [0.79] * n,
        "log_minimum_nights":      [1.10] * n,
        "log_minimum_nights_avg_ntm": [1.10] * n,
        "accommodates_sq":    [4.0] * n,
        "bedrooms_sq":        [1.0] * n,
        "accommodates_x_review_scores_rating":        [9.6] * n,
        "bedrooms_x_review_scores_rating":            [4.8] * n,
        "host_listings_count_x_review_scores_rating": [4.8] * n,
        "accommodates_x_host_is_superhost":           [0.0] * n,
        "minimum_nights_x_accommodates":              [4.0] * n,
        "reviews_per_bedroom":      [30.0] * n,
        "reviews_per_accommodates": [15.0] * n,
        "host_density":             [0.5] * n,
        "review_score_std":         [0.12] * n,
        "dist_from_midtown":        [5.2] * n,
        "borough_Brooklyn":         [1] * n,
        "borough_Manhattan":        [0] * n,
        "borough_Queens":           [0] * n,
        "borough_Staten Island":    [0] * n,
        # extra cols present in real CSV
        "price":                [120.0] * n,
        "accommodates":         [2.0] * n,
        "neighbourhood_cleansed": ["Williamsburg"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — validate_raw_snapshot
# ═════════════════════════════════════════════════════════════════════════════

class TestRawSnapshotPasses:
    def test_valid_df_passes(self):
        assert validate_raw_snapshot(_raw_df()).passed

    def test_all_boroughs_accepted(self):
        for b in BOROUGHS:
            df = _raw_df(neighbourhood_group_cleansed=b)
            assert validate_raw_snapshot(df).passed, f"Borough {b!r} rejected"

    def test_all_room_types_accepted(self):
        for rt in ROOM_TYPES:
            df = _raw_df(room_type=rt)
            assert validate_raw_snapshot(df).passed, f"Room type {rt!r} rejected"


class TestRawSnapshotFailures:
    def test_too_few_rows_is_error(self):
        report = validate_raw_snapshot(_raw_df(n=5_000))
        assert not report.passed
        assert any("10,000" in e or "10000" in e for e in report.errors)

    def test_missing_id_column_is_error(self):
        df = _raw_df().drop(columns=["id"])
        report = validate_raw_snapshot(df)
        assert not report.passed
        assert any("id" in e for e in report.errors)

    def test_missing_price_column_is_error(self):
        df = _raw_df().drop(columns=["price"])
        report = validate_raw_snapshot(df)
        assert not report.passed
        assert any("price" in e for e in report.errors)

    def test_unknown_borough_is_error(self):
        df = _raw_df(neighbourhood_group_cleansed="New Jersey")
        report = validate_raw_snapshot(df)
        assert not report.passed
        assert any("neighbourhood_group_cleansed" in e for e in report.errors)

    def test_unknown_room_type_is_error(self):
        df = _raw_df(room_type="Tent")
        report = validate_raw_snapshot(df)
        assert not report.passed

    def test_latitude_out_of_range_is_error(self):
        df = _raw_df(latitude=51.5)   # London
        report = validate_raw_snapshot(df)
        assert not report.passed

    def test_longitude_out_of_range_is_error(self):
        df = _raw_df(longitude=-0.12)  # London
        report = validate_raw_snapshot(df)
        assert not report.passed

    def test_majority_null_price_is_error(self):
        prices = [None] * 9_000 + ["$120.00"] * 3_000
        df = _raw_df(price=prices)
        report = validate_raw_snapshot(df)
        assert not report.passed
        assert any("price" in e and "null" in e.lower() for e in report.errors)

    def test_raise_if_failed_raises_data_quality_error(self):
        report = validate_raw_snapshot(_raw_df(n=100))
        with pytest.raises(DataQualityError):
            report.raise_if_failed()

    def test_passed_report_does_not_raise(self):
        validate_raw_snapshot(_raw_df()).raise_if_failed()   # must not raise


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — validate_clean_listings
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanListingsPasses:
    def test_valid_df_passes(self):
        assert validate_clean_listings(_clean_df()).passed

    def test_zero_bedrooms_studio_passes(self):
        assert validate_clean_listings(_clean_df(bedrooms=0.0)).passed

    def test_high_accommodates_passes(self):
        assert validate_clean_listings(_clean_df(accommodates=16)).passed


class TestCleanListingsFailures:
    def test_negative_price_is_error(self):
        df = _clean_df(price=-10.0)
        assert not validate_clean_listings(df).passed

    def test_price_above_10000_is_error(self):
        df = _clean_df(price=12_000.0)
        assert not validate_clean_listings(df).passed

    def test_accommodates_zero_is_error(self):
        df = _clean_df(accommodates=0)
        assert not validate_clean_listings(df).passed

    def test_accommodates_above_16_is_error(self):
        df = _clean_df(accommodates=20)
        assert not validate_clean_listings(df).passed

    def test_invalid_superhost_value_is_error(self):
        df = _clean_df(host_is_superhost=2)
        assert not validate_clean_listings(df).passed

    def test_latitude_out_of_nyc_is_error(self):
        df = _clean_df(latitude=34.05)   # Los Angeles
        assert not validate_clean_listings(df).passed

    def test_longitude_out_of_nyc_is_error(self):
        df = _clean_df(longitude=-118.24)  # Los Angeles
        assert not validate_clean_listings(df).passed

    def test_unknown_borough_is_error(self):
        df = _clean_df(neighbourhood_group_cleansed="New Jersey")
        assert not validate_clean_listings(df).passed

    def test_too_few_rows_is_error(self):
        df = _clean_df(n=3_000)
        assert not validate_clean_listings(df).passed

    def test_high_null_rate_in_price_is_error(self, tmp_path):
        prices = [None] * 200 + [120.0] * (N - 200)   # >1% nulls
        df = _clean_df(price=prices)
        assert not validate_clean_listings(df).passed

    def test_prices_mostly_out_of_sane_range_is_error(self):
        # 90% of prices above $2000 → parse failure simulation
        prices = [5_000.0] * int(N * 0.9) + [120.0] * int(N * 0.1)
        df = _clean_df(price=prices)
        assert not validate_clean_listings(df).passed

    def test_review_scores_above_5_is_error(self):
        df = _clean_df(review_scores_rating=5.5)
        assert not validate_clean_listings(df).passed

    def test_minimum_nights_above_1095_is_error(self):
        df = _clean_df(minimum_nights=1200.0)
        assert not validate_clean_listings(df).passed


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — validate_engineered_features
# ═════════════════════════════════════════════════════════════════════════════

class TestEngineeredPasses:
    def test_valid_df_passes(self):
        assert validate_engineered_features(_engineered_df()).passed


class TestEngineeredFailures:
    def test_missing_log_feature_is_error(self):
        df = _engineered_df().drop(columns=["log_price"])
        report = validate_engineered_features(df)
        assert not report.passed
        assert any("log_price" in e for e in report.errors)

    def test_missing_interaction_feature_is_error(self):
        df = _engineered_df().drop(columns=["accommodates_x_review_scores_rating"])
        assert not validate_engineered_features(df).passed

    def test_missing_ratio_feature_is_error(self):
        df = _engineered_df().drop(columns=["dist_from_midtown"])
        assert not validate_engineered_features(df).passed

    def test_missing_borough_dummy_is_error(self):
        df = _engineered_df().drop(columns=["borough_Brooklyn"])
        assert not validate_engineered_features(df).passed

    def test_inf_in_numeric_col_is_error(self):
        df = _engineered_df()
        df["log_price"] = np.inf
        assert not validate_engineered_features(df).passed

    def test_nan_in_required_col_is_error(self):
        df = _engineered_df()
        df.loc[df.index[:500], "log_price"] = np.nan   # >0.1% null rate
        assert not validate_engineered_features(df).passed

    def test_log_price_out_of_range_is_error(self):
        # All rows have log_price=15 (way above 10.5 ceiling)
        df = _engineered_df(log_price=15.0)
        assert not validate_engineered_features(df).passed

    def test_negative_review_score_std_is_error(self):
        df = _engineered_df(review_score_std=-1.0)
        assert not validate_engineered_features(df).passed

    def test_too_few_rows_is_error(self):
        df = _engineered_df(n=3_000)
        assert not validate_engineered_features(df).passed

    def test_all_required_cols_checked(self):
        # Dropping each required column in turn must produce a failure
        base = _engineered_df()
        for col in _REQUIRED_ENGINEERED:
            if col in base.columns:
                df = base.drop(columns=[col])
                report = validate_engineered_features(df)
                assert not report.passed, f"Dropping '{col}' should fail but passed"


# ═════════════════════════════════════════════════════════════════════════════
# DataQualityReport behaviour
# ═════════════════════════════════════════════════════════════════════════════

class TestDataQualityReport:
    def test_passed_report_has_no_errors(self):
        r = validate_raw_snapshot(_raw_df())
        assert r.passed
        assert r.errors == []

    def test_failed_report_raise_if_failed_raises(self):
        r = DataQualityReport(stage="test", passed=False, row_count=0,
                              errors=["something broke"])
        with pytest.raises(DataQualityError, match="something broke"):
            r.raise_if_failed()

    def test_passed_report_raise_if_failed_is_silent(self):
        r = DataQualityReport(stage="test", passed=True, row_count=100)
        r.raise_if_failed()  # must not raise

    def test_report_has_timestamp(self):
        r = validate_raw_snapshot(_raw_df())
        assert r.timestamp   # non-empty ISO string

    def test_report_stats_contains_row_count(self):
        r = validate_raw_snapshot(_raw_df(n=N))
        assert r.stats["row_count"] == N

    def test_report_stats_contains_price_median(self):
        r = validate_raw_snapshot(_raw_df())
        assert "price_median" in r.stats
        assert r.stats["price_median"] == pytest.approx(120.0, abs=1.0)


# ═════════════════════════════════════════════════════════════════════════════
# save_baseline / load_baseline
# ═════════════════════════════════════════════════════════════════════════════

class TestBaseline:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "baseline.json"
        df   = _clean_df()
        save_baseline(df, stage="clean_listings", path=path)
        bl   = load_baseline(path=path)
        assert "clean_listings" in bl
        assert bl["clean_listings"]["row_count"] == N

    def test_save_appends_without_overwriting_other_stages(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline(_raw_df(),   stage="raw_snapshot",   path=path)
        save_baseline(_clean_df(), stage="clean_listings", path=path)
        bl = load_baseline(path=path)
        assert "raw_snapshot"   in bl
        assert "clean_listings" in bl

    def test_saved_baseline_includes_price_median(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline(_clean_df(price=150.0), stage="clean_listings", path=path)
        bl = load_baseline(path=path)
        assert bl["clean_listings"]["price_median"] == pytest.approx(150.0, abs=1.0)

    def test_load_returns_empty_dict_when_no_file(self, tmp_path):
        bl = load_baseline(path=tmp_path / "nonexistent.json")
        assert bl == {}

    def test_distribution_warning_on_price_drift(self, tmp_path):
        path = tmp_path / "baseline.json"
        # Save baseline with median $120
        save_baseline(_clean_df(price=120.0), stage="clean_listings", path=path)

        # Validate data where median is $300 — >25% drift → warning
        import unittest.mock as mock
        with mock.patch(
            "src.training.data_validator.BASELINE_PATH", path
        ):
            report = validate_clean_listings(_clean_df(price=300.0))

        assert any("Price median" in w or "price" in w.lower() for w in report.warnings)

    def test_no_baseline_produces_warning_not_error(self, tmp_path):
        import unittest.mock as mock
        nonexistent = tmp_path / "no_baseline.json"
        with mock.patch(
            "src.training.data_validator.BASELINE_PATH", nonexistent
        ):
            report = validate_clean_listings(_clean_df())

        assert report.passed                    # still passes (warning only)
        assert any("baseline" in w.lower() for w in report.warnings)
