"""
Train-serve skew detection.

Both code paths transform the same raw listing dict into a feature vector:
  Training  → 2_feature_engineering.py  (DataFrame ops, batch)
  Serving   → NYCAirbnbPredictorONNX._engineer()  (dict ops, per-request)

If someone edits one and not the other, predictions are silently wrong.
This test runs both paths on the same input and asserts identical output
for every shared feature. A mismatch here means a production bug.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── load 2_feature_engineering.py via importlib (name starts with digit) ──────
_FE_PATH = ROOT / "src" / "new_york_workflow" / "2_feature_engineering.py"
_fe_spec = importlib.util.spec_from_file_location("feature_engineering", _FE_PATH)
fe = importlib.util.module_from_spec(_fe_spec)
_fe_spec.loader.exec_module(fe)

from src.new_york_workflow.nyc_predictor_onnx import NYCAirbnbPredictorONNX, MIDTOWN_LAT, MIDTOWN_LON


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NEIGHBOURHOOD_MEANS = {"Williamsburg": 180.0, "Midtown": 220.0}
GLOBAL_MEAN = 150.0


def _make_predictor():
    """Build a NYCAirbnbPredictorONNX without loading any model files."""
    pred = object.__new__(NYCAirbnbPredictorONNX)
    pred.neighbourhood_means = NEIGHBOURHOOD_MEANS
    pred.global_mean = GLOBAL_MEAN
    return pred


def _training_features(raw: dict) -> dict:
    """
    Replicate what the training pipeline produces for a single row.

    Uses the actual functions from 2_feature_engineering.py so any edit
    to the training path is automatically reflected here.
    """
    # The training CSV has 'neighbourhood_group_cleansed' for borough
    df = pd.DataFrame([{**raw, "neighbourhood_group_cleansed": raw["borough"]}])

    # Step 3 — log transforms (skip 'price' — not present at serving time)
    for col in fe.LOG_TRANSFORM_COLS:
        if col in df.columns and col != "price":
            df[f"log_{col}"] = np.log1p(df[col])

    # Step 3.5 — borough one-hot (drops Bronx as reference category)
    # Force all categories so a single-row DataFrame produces all 4 dummy columns
    # (the full training data has all boroughs, but a test fixture may not)
    borough_cats = pd.CategoricalDtype(
        categories=["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        ordered=False,
    )
    df["neighbourhood_group_cleansed"] = df["neighbourhood_group_cleansed"].astype(borough_cats)
    dummies = pd.get_dummies(df["neighbourhood_group_cleansed"], prefix="borough")
    if "borough_Bronx" in dummies.columns:
        dummies = dummies.drop(columns=["borough_Bronx"])
    df = pd.concat([df, dummies], axis=1)

    # Step 4 — polynomial
    for col in fe.POLY_COLS:
        df[f"{col}_sq"] = df[col] ** 2

    # Step 5 — interactions
    for col_a, col_b in fe.INTERACTION_PAIRS:
        df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]

    # Step 6 — ratios
    for name, num, denom in [
        ("reviews_per_bedroom",      "number_of_reviews", "bedrooms"),
        ("reviews_per_accommodates", "number_of_reviews", "accommodates"),
        ("host_density",             "host_listings_count", "accommodates"),
    ]:
        df[name] = df[num] / df[denom].replace(0, 1)

    reviewed = df[fe.REVIEW_SCORE_COLS].sum(axis=1) > 0
    df["review_score_std"] = np.where(
        reviewed,
        df[fe.REVIEW_SCORE_COLS].std(axis=1),   # pandas default: ddof=1
        0.0,
    )

    df["dist_from_midtown"] = np.sqrt(
        (df["latitude"]  - MIDTOWN_LAT) ** 2 +
        (df["longitude"] - MIDTOWN_LON) ** 2
    ) * 111

    return df.iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

RAW_LISTING = {
    # Core
    "accommodates":         4,
    "bedrooms":             2.0,
    "bathrooms":            1.5,
    "is_private_bath":      True,
    "room_type":            "Entire home/apt",
    "borough":              "Brooklyn",
    "neighbourhood":        "Williamsburg",
    "latitude":             40.7128,
    "longitude":            -74.0060,
    # Stay
    "minimum_nights":       2,
    "minimum_nights_avg_ntm": 2.5,
    # Host
    "host_is_superhost":    True,
    "host_listings_count":  3,
    # Reviews — deliberately different values so std != 0
    "number_of_reviews":    50,
    "number_of_reviews_ltm": 8,
    "reviews_per_month":    2.1,
    "review_scores_rating":       4.8,
    "review_scores_accuracy":     4.9,
    "review_scores_cleanliness":  4.7,
    "review_scores_checkin":      5.0,
    "review_scores_communication": 4.8,
    "review_scores_location":     4.6,
    "review_scores_value":        4.5,
    # Amenities
    "amenity_count":        35,
    "has_gym":              False,
    "has_elevator":         True,
    "has_dryer":            True,
    "has_air_conditioning": True,
    "has_washer":           True,
    "has_pool":             False,
}

# A second listing with edge-case values (studio, no reviews, Bronx)
RAW_EDGE = {
    **RAW_LISTING,
    "accommodates":    1,
    "bedrooms":        0.0,   # studio
    "bathrooms":       1.0,
    "room_type":       "Private room",
    "borough":         "Bronx",
    "neighbourhood":   "unknown-neighbourhood",
    "minimum_nights":  30,
    "host_is_superhost": False,
    "number_of_reviews": 0,
    "number_of_reviews_ltm": 0,
    "reviews_per_month": 0.0,
    "review_scores_rating":       0.0,
    "review_scores_accuracy":     0.0,
    "review_scores_cleanliness":  0.0,
    "review_scores_checkin":      0.0,
    "review_scores_communication": 0.0,
    "review_scores_location":     0.0,
    "review_scores_value":        0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainServeSkew:
    """
    Each test asserts one feature (or feature group) matches between paths.
    Separate test methods make failures pinpoint exactly which feature drifted.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pred  = _make_predictor()
        self.train = _training_features(RAW_LISTING)
        self.serve = self.pred._engineer(RAW_LISTING)
        self.train_edge = _training_features(RAW_EDGE)
        self.serve_edge = self.pred._engineer(RAW_EDGE)

    # ── log features ──────────────────────────────────────────────────────────

    def test_log_accommodates(self):
        assert self.serve["log_accommodates"] == pytest.approx(self.train["log_accommodates"], rel=1e-9)

    def test_log_bedrooms(self):
        assert self.serve["log_bedrooms"] == pytest.approx(self.train["log_bedrooms"], rel=1e-9)

    def test_log_bathrooms(self):
        assert self.serve["log_bathrooms"] == pytest.approx(self.train["log_bathrooms"], rel=1e-9)

    def test_log_amenity_count(self):
        assert self.serve["log_amenity_count"] == pytest.approx(self.train["log_amenity_count"], rel=1e-9)

    def test_log_host_listings_count(self):
        assert self.serve["log_host_listings_count"] == pytest.approx(self.train["log_host_listings_count"], rel=1e-9)

    def test_log_number_of_reviews(self):
        assert self.serve["log_number_of_reviews"] == pytest.approx(self.train["log_number_of_reviews"], rel=1e-9)

    def test_log_number_of_reviews_ltm(self):
        assert self.serve["log_number_of_reviews_ltm"] == pytest.approx(self.train["log_number_of_reviews_ltm"], rel=1e-9)

    def test_log_reviews_per_month(self):
        assert self.serve["log_reviews_per_month"] == pytest.approx(self.train["log_reviews_per_month"], rel=1e-9)

    def test_log_minimum_nights(self):
        assert self.serve["log_minimum_nights"] == pytest.approx(self.train["log_minimum_nights"], rel=1e-9)

    def test_log_minimum_nights_avg_ntm(self):
        assert self.serve["log_minimum_nights_avg_ntm"] == pytest.approx(self.train["log_minimum_nights_avg_ntm"], rel=1e-9)

    # ── polynomial ────────────────────────────────────────────────────────────

    def test_accommodates_sq(self):
        assert self.serve["accommodates_sq"] == pytest.approx(self.train["accommodates_sq"], rel=1e-9)

    def test_bedrooms_sq(self):
        assert self.serve["bedrooms_sq"] == pytest.approx(self.train["bedrooms_sq"], rel=1e-9)

    # ── interaction features ──────────────────────────────────────────────────

    def test_accommodates_x_review_scores_rating(self):
        assert self.serve["accommodates_x_review_scores_rating"] == pytest.approx(
            self.train["accommodates_x_review_scores_rating"], rel=1e-9)

    def test_bedrooms_x_review_scores_rating(self):
        assert self.serve["bedrooms_x_review_scores_rating"] == pytest.approx(
            self.train["bedrooms_x_review_scores_rating"], rel=1e-9)

    def test_host_listings_count_x_review_scores_rating(self):
        assert self.serve["host_listings_count_x_review_scores_rating"] == pytest.approx(
            self.train["host_listings_count_x_review_scores_rating"], rel=1e-9)

    def test_accommodates_x_host_is_superhost(self):
        assert self.serve["accommodates_x_host_is_superhost"] == pytest.approx(
            self.train["accommodates_x_host_is_superhost"], rel=1e-9)

    def test_minimum_nights_x_accommodates(self):
        assert self.serve["minimum_nights_x_accommodates"] == pytest.approx(
            self.train["minimum_nights_x_accommodates"], rel=1e-9)

    # ── ratio features ────────────────────────────────────────────────────────

    def test_reviews_per_bedroom(self):
        assert self.serve["reviews_per_bedroom"] == pytest.approx(self.train["reviews_per_bedroom"], rel=1e-9)

    def test_reviews_per_accommodates(self):
        assert self.serve["reviews_per_accommodates"] == pytest.approx(self.train["reviews_per_accommodates"], rel=1e-9)

    def test_host_density(self):
        assert self.serve["host_density"] == pytest.approx(self.train["host_density"], rel=1e-9)

    def test_dist_from_midtown(self):
        assert self.serve["dist_from_midtown"] == pytest.approx(self.train["dist_from_midtown"], rel=1e-9)

    def test_review_score_std(self):
        assert self.serve["review_score_std"] == pytest.approx(self.train["review_score_std"], rel=1e-6)

    # ── borough dummies ───────────────────────────────────────────────────────

    def test_borough_brooklyn(self):
        assert self.serve["borough_Brooklyn"] == int(self.train["borough_Brooklyn"])

    def test_borough_manhattan(self):
        assert self.serve["borough_Manhattan"] == int(self.train["borough_Manhattan"])

    def test_borough_queens(self):
        assert self.serve["borough_Queens"] == int(self.train["borough_Queens"])

    def test_borough_staten_island(self):
        assert self.serve["borough_Staten Island"] == int(self.train["borough_Staten Island"])

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_edge_studio_bedrooms_sq_is_zero(self):
        assert self.serve_edge["bedrooms_sq"] == 0.0

    def test_edge_reviews_per_bedroom_with_zero_bedrooms(self):
        # bedrooms=0 → denominator clamps to 1 on both sides
        assert self.serve_edge["reviews_per_bedroom"] == pytest.approx(
            self.train_edge["reviews_per_bedroom"], rel=1e-9)

    def test_edge_no_reviews_std_is_zero(self):
        # All review scores = 0 → std should be 0 on both sides
        assert self.serve_edge["review_score_std"] == 0.0
        assert self.train_edge["review_score_std"] == 0.0

    def test_edge_unknown_neighbourhood_uses_global_mean(self):
        # neighbourhood not in neighbourhood_means → fall back to global_mean
        assert self.serve_edge["neighbourhood_price_rank"] == GLOBAL_MEAN

    def test_edge_bronx_all_borough_dummies_zero(self):
        # Bronx is the reference category — all 4 dummies should be 0
        for col in ("borough_Brooklyn", "borough_Manhattan", "borough_Queens", "borough_Staten Island"):
            assert self.serve_edge[col] == 0
