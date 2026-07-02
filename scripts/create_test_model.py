"""
Create a minimal but structurally correct set of model artefacts for CI.

Why this exists:
  models/ is gitignored (binary files, too large to track).
  The load-test CI job needs a live API, which needs a real ONNX model.
  This script synthesises one from random data in < 10 seconds so CI
  never has to wait for a full retrain.

Usage:
  python scripts/create_test_model.py          # writes to models/nyc/
  python scripts/create_test_model.py --force  # overwrite even if already present
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "nyc"

# Exact 58 features the predictor expects — must match nyc_feature_list.pkl
FEATURE_LIST = [
    "accommodates", "bedrooms", "host_is_superhost", "host_listings_count",
    "latitude", "longitude", "minimum_nights", "minimum_nights_avg_ntm",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
    "room_type_Entire home/apt", "room_type_Hotel room",
    "room_type_Private room", "room_type_Shared room",
    "bathrooms", "is_private_bath", "amenity_count",
    "has_gym", "has_elevator", "has_dryer", "has_air_conditioning",
    "has_washer", "has_pool",
    "log_host_listings_count", "log_number_of_reviews", "log_number_of_reviews_ltm",
    "log_reviews_per_month", "log_minimum_nights", "log_minimum_nights_avg_ntm",
    "log_accommodates", "log_bedrooms", "log_bathrooms", "log_amenity_count",
    "borough_Brooklyn", "borough_Manhattan", "borough_Queens", "borough_Staten Island",
    "accommodates_sq", "bedrooms_sq",
    "accommodates_x_review_scores_rating", "bedrooms_x_review_scores_rating",
    "host_listings_count_x_review_scores_rating", "accommodates_x_host_is_superhost",
    "minimum_nights_x_accommodates", "reviews_per_bedroom",
    "reviews_per_accommodates", "host_density", "review_score_std",
    "dist_from_midtown", "neighbourhood_price_rank",
]

# Synthetic NYC neighbourhoods with plausible log-price means (~$150/night = log 5.0)
NEIGHBOURHOOD_MEANS = {
    "Hell's Kitchen": 5.12, "Williamsburg": 5.05, "Astoria": 4.92,
    "Harlem": 4.88, "Bushwick": 4.79, "Crown Heights": 4.82,
    "Upper West Side": 5.25, "Chelsea": 5.30, "Midtown": 5.40,
    "Park Slope": 5.10,
}
GLOBAL_MEAN = 5.14


def _make_onnx(model: XGBRegressor, n_features: int, out_path: Path) -> None:
    """Convert XGBoost → ONNX via onnxmltools (same path as retrain.py)."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

    initial_type = [("input", OnnxFloat([None, n_features]))]
    onnx_model   = convert_xgboost(model, initial_types=initial_type)
    out_path.write_bytes(onnx_model.SerializeToString())


def create(force: bool = False) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = MODEL_DIR / "nyc_xgb_model.onnx"
    if onnx_path.exists() and not force:
        log.info("Model artefacts already exist — skipping (use --force to overwrite)")
        return

    n = len(FEATURE_LIST)
    rng = np.random.default_rng(42)

    # ── 1. Synthetic training data ─────────────────────────────────────────────
    log.info("Generating synthetic training data (%d features)…", n)
    X = rng.standard_normal((500, n)).astype(np.float32)
    # Target: log-price in plausible range (4.5–5.8 ≈ $90–$330/night)
    y = rng.uniform(4.5, 5.8, size=500)

    # ── 2. Scaler (fitted on synthetic data — just needs correct shape) ────────
    log.info("Fitting StandardScaler…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── 3. XGBoost (minimal, fast) ────────────────────────────────────────────
    log.info("Training minimal XGBoost…")
    model = XGBRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_scaled, y)

    # ── 4. Convert to ONNX ────────────────────────────────────────────────────
    log.info("Converting to ONNX → %s", onnx_path)
    _make_onnx(model, n, onnx_path)

    # ── 5. Persist supporting artefacts ───────────────────────────────────────
    pkl_path = MODEL_DIR / "nyc_xgb_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    scaler_path = MODEL_DIR / "nyc_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    fl_path = MODEL_DIR / "nyc_feature_list.pkl"
    with open(fl_path, "wb") as f:
        pickle.dump(FEATURE_LIST, f)

    nm_path = MODEL_DIR / "nyc_neighbourhood_means.pkl"
    with open(nm_path, "wb") as f:
        pickle.dump({"means": NEIGHBOURHOOD_MEANS, "global_mean": GLOBAL_MEAN}, f)

    report = {
        "timestamp": "ci-test-model",
        "target": "log_price",
        "n_features": n,
        "feature_cols": FEATURE_LIST,
        "best_model": "XGBoost",
        "assumption_check": {},
        "models": {
            # Structure must match what NYCAirbnbPredictorONNX.model_info() reads:
            # self._report["models"]["XGBoost"]["test"]["r2"] etc.
            "XGBoost": {
                "test": {
                    "r2":        0.01,   # intentionally low — synthetic data
                    "mae_dollar": 99.0,
                    "mape":       50.0,
                },
                "params": {},
            }
        },
    }
    (MODEL_DIR / "nyc_training_report.json").write_text(json.dumps(report, indent=2))

    # Ridge and RF stubs (predictor may look for these)
    for stub in ("nyc_ridge_model.pkl", "nyc_rf_model.pkl"):
        stub_path = MODEL_DIR / stub
        if not stub_path.exists():
            with open(stub_path, "wb") as f:
                pickle.dump(model, f)

    log.info("Done — test model artefacts written to %s", MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing artefacts")
    args = parser.parse_args()
    create(force=args.force)
