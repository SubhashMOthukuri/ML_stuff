"""
Production data quality gates for the NYC Airbnb pipeline.

Three validation stages gate each pipeline step:
  Stage 1 — raw_snapshot     : InsideAirbnb listings.csv before any transformation
  Stage 2 — clean_listings   : after 1_data_cleaning.py, before feature engineering
  Stage 3 — engineered_feats : after 2_feature_engineering.py, before model training

Each stage produces a DataQualityReport with:
  · Hard errors   → report.passed = False  → raise DataQualityError to stop pipeline
  · Soft warnings → logged but pipeline continues

Distribution checks compare current stats against a saved baseline JSON
(models/nyc/data_quality_baseline.json). On first run (no baseline), only
structural checks run; call save_baseline() after a known-good run.

Usage:
    from src.new_york_workflow.nyc_data_validator import (
        validate_raw_snapshot, validate_clean_listings,
        validate_engineered_features, save_baseline,
    )

    report = validate_raw_snapshot(df)
    report.raise_if_failed()   # stops pipeline if schema or row-count fails

    # after a good run, save baseline for future distribution checks:
    save_baseline(df, stage="raw_snapshot")
"""

import json
import logging
import os
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.exceptions import DataQualityError

import numpy as np
import pandas as pd

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from pandera.errors import SchemaError, SchemaErrors

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

BOROUGHS   = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}
ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}

_ROOT          = Path(__file__).resolve().parents[2]
BASELINE_PATH  = _ROOT / "models" / "nyc" / "data_quality_baseline.json"

# Engineered feature columns that MUST be present before training
_REQUIRED_ENGINEERED = [
    "log_price", "log_accommodates", "log_bedrooms", "log_bathrooms",
    "log_amenity_count", "log_host_listings_count", "log_number_of_reviews",
    "log_number_of_reviews_ltm", "log_reviews_per_month",
    "log_minimum_nights", "log_minimum_nights_avg_ntm",
    "accommodates_sq", "bedrooms_sq",
    "accommodates_x_review_scores_rating", "bedrooms_x_review_scores_rating",
    "host_listings_count_x_review_scores_rating",
    "accommodates_x_host_is_superhost", "minimum_nights_x_accommodates",
    "reviews_per_bedroom", "reviews_per_accommodates", "host_density",
    "review_score_std", "dist_from_midtown",
    "borough_Brooklyn", "borough_Manhattan",
    "borough_Queens", "borough_Staten Island",
]


# ── report ────────────────────────────────────────────────────────────────────


@dataclass
class DataQualityReport:
    stage:     str
    passed:    bool
    row_count: int
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)
    stats:     dict      = field(default_factory=dict)
    timestamp: str       = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataQualityError(self.stage, self.errors)

    def log_summary(self) -> None:
        badge = "✓ PASSED" if self.passed else "✗ FAILED"
        logger.info(
            "Data quality [%s] %s | rows=%d  errors=%d  warnings=%d",
            self.stage, badge, self.row_count, len(self.errors), len(self.warnings),
        )
        for e in self.errors:
            logger.error("  [ERROR] %s", e)
        for w in self.warnings:
            logger.warning("  [WARN]  %s", w)


# ── Pandera schemas ───────────────────────────────────────────────────────────

# Stage 1: raw InsideAirbnb CSV (90+ columns; we only gate the ones we use)
_RAW_SCHEMA = DataFrameSchema(
    {
        "id": Column(nullable=False),
        "neighbourhood_group_cleansed": Column(
            str,
            Check.isin(list(BOROUGHS)),
            nullable=False,
        ),
        "room_type": Column(
            str,
            Check.isin(list(ROOM_TYPES)),
            nullable=False,
        ),
        "latitude":  Column(float, Check.in_range(40.4, 41.0), coerce=True),
        "longitude": Column(float, Check.in_range(-74.5, -73.5), coerce=True),
    },
    coerce=False,
)

# Stage 2: cleaned listings (output of 1_data_cleaning.py)
_CLEAN_SCHEMA = DataFrameSchema(
    {
        "price":               Column(float, Check.in_range(3, 50_000)),
        "accommodates":        Column(float, Check.in_range(1, 16),  coerce=True),
        "bedrooms":            Column(float, Check.in_range(0, 20),  nullable=False),
        "bathrooms":           Column(float, Check.in_range(0, 20),  nullable=False),
        "host_is_superhost":   Column(int,   Check.isin([0, 1]),     coerce=True),
        "host_listings_count": Column(float, Check.ge(1),            coerce=True),
        "latitude":            Column(float, Check.in_range(40.4, 41.0)),
        "longitude":           Column(float, Check.in_range(-74.5, -73.5)),
        "minimum_nights":      Column(float, Check.in_range(1, 1095), coerce=True),
        "number_of_reviews":   Column(float, Check.ge(0),            coerce=True),
        "reviews_per_month":   Column(float, Check.ge(0),            nullable=False),
        "review_scores_rating":Column(float, Check.in_range(0, 5),   nullable=False),
        "amenity_count":       Column(float, Check.in_range(0, 500), coerce=True),
        "is_private_bath":     Column(int,   Check.isin([0, 1]),     coerce=True),
        "neighbourhood_group_cleansed": Column(
            str, Check.isin(list(BOROUGHS)), nullable=False,
        ),
    },
    coerce=False,
)

# Stage 3: engineered features (output of 2_feature_engineering.py)
_ENGINEERED_SCHEMA = DataFrameSchema(
    {
        "log_price":         Column(float, Check.in_range(1.0, 10.5)),
        "dist_from_midtown": Column(float, Check.in_range(0, 120)),
        "review_score_std":  Column(float, Check.ge(0)),
        "accommodates_sq":   Column(checks=[Check.ge(0)]),
        "bedrooms_sq":       Column(checks=[Check.ge(0)]),
        "borough_Brooklyn":  Column(coerce=True),
        "borough_Manhattan": Column(coerce=True),
        "borough_Queens":    Column(coerce=True),
    },
    coerce=False,
)


# ── internal helpers ──────────────────────────────────────────────────────────

def _apply_schema(df: pd.DataFrame, schema: DataFrameSchema,
                  errors: list, warnings: list) -> None:
    """Run Pandera validation; append failures into errors list (lazy mode)."""
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            schema.validate(df, lazy=True)
        except SchemaErrors as exc:
            for row in exc.failure_cases.itertuples(index=False):
                col   = getattr(row, "column", "?")
                check = getattr(row, "check", "?")
                val   = getattr(row, "failure_case", "?")
                errors.append(f"Column '{col}': check '{check}' failed (sample: {val!r})")
        except SchemaError as exc:
            errors.append(str(exc))


def _check_required_cols(df: pd.DataFrame, required: list[str],
                         errors: list) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")


def _check_null_rates(df: pd.DataFrame, cols: list[str],
                      hard_pct: float, soft_pct: float,
                      errors: list, warnings: list) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        pct = df[col].isna().mean() * 100
        if pct >= hard_pct:
            errors.append(
                f"Column '{col}': {pct:.1f}% nulls ≥ hard threshold {hard_pct}%"
            )
        elif pct >= soft_pct:
            warnings.append(f"Column '{col}': {pct:.1f}% nulls (elevated, threshold {soft_pct}%)")


def _check_inf(df: pd.DataFrame, errors: list) -> None:
    num = df.select_dtypes(include=[np.number])
    inf_cols = [c for c in num.columns if np.isinf(num[c]).any()]
    if inf_cols:
        errors.append(f"Infinite values in columns: {inf_cols}")


def _collect_stats(df: pd.DataFrame, stage: str) -> dict:
    stats: dict = {"row_count": len(df), "col_count": len(df.columns)}
    if "price" in df.columns:
        try:
            prices = pd.to_numeric(
                df["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
                errors="coerce",
            ).dropna()
            stats.update({
                "price_median": round(float(prices.median()), 2),
                "price_p25":    round(float(prices.quantile(0.25)), 2),
                "price_p75":    round(float(prices.quantile(0.75)), 2),
                "price_null_pct": round(df["price"].isna().mean() * 100, 2),
            })
        except Exception:
            pass
    if "log_price" in df.columns:
        stats.update({
            "log_price_mean":   round(float(df["log_price"].mean()), 4),
            "log_price_std":    round(float(df["log_price"].std()), 4),
        })
    return stats


def _distribution_warnings(stats: dict, baseline: dict,
                            stage: str, warnings: list) -> None:
    """Soft checks: compare current stats against saved baseline."""
    ref = baseline.get(stage)
    if not ref:
        warnings.append(
            f"No baseline for stage '{stage}' — "
            "run save_baseline() after a known-good data run"
        )
        return

    # Row count: warn if < 80 % of baseline or > 3× baseline
    ref_rows = ref.get("row_count")
    cur_rows = stats.get("row_count", 0)
    if ref_rows:
        ratio = cur_rows / ref_rows
        if ratio < 0.80:
            warnings.append(
                f"Row count {cur_rows:,} is {(1-ratio)*100:.0f}% below baseline "
                f"({ref_rows:,}) — possible partial download or aggressive filtering"
            )
        elif ratio > 3.0:
            warnings.append(
                f"Row count {cur_rows:,} is {ratio:.1f}× baseline ({ref_rows:,}) "
                "— unexpected data growth"
            )

    # Price median: warn if >25% drift
    ref_med = ref.get("price_median")
    cur_med = stats.get("price_median")
    if ref_med and cur_med:
        drift = abs(cur_med - ref_med) / ref_med * 100
        if drift > 25:
            warnings.append(
                f"Price median drifted {drift:.1f}% from baseline "
                f"(current ${cur_med:.0f}, expected ${ref_med:.0f}) "
                "— possible pricing data corruption or major market shift"
            )

    # log_price mean: warn if >0.3 (≈ 35% price shift in log space)
    ref_lm = ref.get("log_price_mean")
    cur_lm = stats.get("log_price_mean")
    if ref_lm and cur_lm:
        drift = abs(cur_lm - ref_lm)
        if drift > 0.30:
            warnings.append(
                f"log_price mean shifted by {drift:.3f} from baseline "
                f"(current {cur_lm:.3f}, expected {ref_lm:.3f}) "
                "— distribution anomaly before training"
            )


def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text())
        except Exception:
            return {}
    return {}


# ── public API ────────────────────────────────────────────────────────────────

def validate_raw_snapshot(df: pd.DataFrame) -> DataQualityReport:
    """
    Gate 1: raw InsideAirbnb listings.csv before any transformation.
    Called from scripts/fetch_ground_truth.py and 1_data_cleaning.py.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Structural: required columns present
    _check_required_cols(df, ["id", "price", "neighbourhood_group_cleansed",
                               "room_type", "latitude", "longitude"], errors)

    # Schema: types, ranges, allowed values
    _apply_schema(df, _RAW_SCHEMA, errors, warnings)

    # Row count: minimum 10 000 (a full NYC snapshot has 35 000+)
    if len(df) < 10_000:
        errors.append(
            f"Only {len(df):,} rows — expected ≥ 10,000. "
            "Likely a partial download or wrong URL."
        )

    # Null rate on price: >50% means something broke upstream
    _check_null_rates(df, ["price"], hard_pct=50, soft_pct=10, errors=errors, warnings=warnings)

    # Distribution checks vs baseline
    stats = _collect_stats(df, "raw_snapshot")
    _distribution_warnings(stats, _load_baseline(), "raw_snapshot", warnings)

    passed = len(errors) == 0
    report = DataQualityReport(
        stage="raw_snapshot", passed=passed,
        row_count=len(df), errors=errors, warnings=warnings, stats=stats,
    )
    report.log_summary()
    return report


def validate_clean_listings(df: pd.DataFrame) -> DataQualityReport:
    """
    Gate 2: cleaned listings after 1_data_cleaning.py, before feature engineering.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Schema: types, ranges, categorical values
    _apply_schema(df, _CLEAN_SCHEMA, errors, warnings)

    # No nulls allowed — cleaning should have handled everything
    critical = ["price", "accommodates", "bedrooms", "bathrooms",
                "latitude", "longitude", "minimum_nights"]
    _check_null_rates(df, critical, hard_pct=1, soft_pct=0.1,
                      errors=errors, warnings=warnings)

    # Row count: cleaning drops outliers but should keep the bulk
    if len(df) < 8_000:
        errors.append(
            f"Only {len(df):,} rows after cleaning — expected ≥ 8,000. "
            "Aggressive filtering or bad upstream data."
        )

    # Price range sanity: at least 80% of listings should be in $10–$2,000
    if "price" in df.columns:
        pct_in_range = df["price"].between(10, 2_000).mean() * 100
        if pct_in_range < 80:
            errors.append(
                f"Only {pct_in_range:.1f}% of prices fall in $10–$2,000 "
                "(expected ≥ 80%) — price parsing may have failed."
            )

    stats = _collect_stats(df, "clean_listings")
    _distribution_warnings(stats, _load_baseline(), "clean_listings", warnings)

    passed = len(errors) == 0
    report = DataQualityReport(
        stage="clean_listings", passed=passed,
        row_count=len(df), errors=errors, warnings=warnings, stats=stats,
    )
    report.log_summary()
    return report


def validate_engineered_features(df: pd.DataFrame) -> DataQualityReport:
    """
    Gate 3: engineered feature matrix after 2_feature_engineering.py, before training.
    Catches missing features, NaN/Inf values, and distribution anomalies.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # All engineered features must be present
    _check_required_cols(df, _REQUIRED_ENGINEERED, errors)

    # Schema checks on key columns (only if columns are present)
    present_cols = set(df.columns)
    schema_cols = {k: v for k, v in _ENGINEERED_SCHEMA.columns.items()
                   if k in present_cols}
    if schema_cols:
        _apply_schema(df, DataFrameSchema(schema_cols), errors, warnings)

    # No NaN or Inf in numeric columns
    _check_null_rates(df, [c for c in _REQUIRED_ENGINEERED if c in df.columns],
                      hard_pct=0.1, soft_pct=0.01, errors=errors, warnings=warnings)
    _check_inf(df, errors)

    # Row count
    if len(df) < 8_000:
        errors.append(
            f"Only {len(df):,} engineered rows — expected ≥ 8,000. "
            "Feature engineering may have dropped too many rows."
        )

    # log_price must exist and be numeric (it's the training target)
    if "log_price" in df.columns:
        if df["log_price"].isna().any():
            errors.append("log_price contains NaN — target column is corrupt")
        n_outside = (~df["log_price"].between(1.0, 10.5)).sum()
        if n_outside > len(df) * 0.01:
            warnings.append(
                f"{n_outside} rows have log_price outside [2.0, 10.5] "
                "(>1% of data) — check price outlier handling"
            )

    stats = _collect_stats(df, "engineered_features")
    _distribution_warnings(stats, _load_baseline(), "engineered_features", warnings)

    passed = len(errors) == 0
    report = DataQualityReport(
        stage="engineered_features", passed=passed,
        row_count=len(df), errors=errors, warnings=warnings, stats=stats,
    )
    report.log_summary()
    return report


# ── baseline management ───────────────────────────────────────────────────────

def save_baseline(df: pd.DataFrame, stage: str,
                  path: Path = BASELINE_PATH) -> None:
    """
    Persist stats from a known-good DataFrame as the distribution baseline.
    Call this once after verifying a clean data run; future runs compare against it.

    stage: "raw_snapshot" | "clean_listings" | "engineered_features"
    """
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass

    existing[stage] = _collect_stats(df, stage)
    existing[stage]["saved_at"] = datetime.now(timezone.utc).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))
    logger.info("Baseline saved for stage '%s' → %s", stage, path)


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    """Return the full baseline dict (all stages)."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}
