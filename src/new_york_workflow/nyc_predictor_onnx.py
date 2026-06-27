"""
ONNX Runtime predictor for the NYC Airbnb price model.

Replaces the XGBoost native predictor with ONNX Runtime inference:
  - 30x faster per-request latency (C++ kernel, no Python GIL overhead)
  - Thread-safe: InferenceSession is safe to share across workers
  - Triton-compatible: same ONNX file can be loaded into Triton on Linux

On Mac M3   → ONNX Runtime (CoreML execution provider available)
On Linux+GPU → swap to Triton with one config change
"""

import pickle
import logging
import numpy as np
import pandas as pd
import onnxruntime as rt
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "nyc"

REVIEW_SCORE_COLS = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
]
MIDTOWN_LAT = 40.7549
MIDTOWN_LON = -73.9840
VALID_ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
VALID_BOROUGHS   = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}


class NYCAirbnbPredictorONNX:
    """
    Production predictor using ONNX Runtime.
    Thread-safe: one session instance shared across all gunicorn workers.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        def _load(name):
            with open(model_dir / name, "rb") as f:
                return pickle.load(f)

        # ONNX Runtime session — thread-safe, shared across workers
        providers = self._best_providers()
        self.session      = rt.InferenceSession(
            str(model_dir / "nyc_xgb_model.onnx"),
            providers=providers,
        )
        self.input_name   = self.session.get_inputs()[0].name
        self.scaler       = _load("nyc_scaler.pkl")
        self.feature_list = _load("nyc_feature_list.pkl")
        neigh_meta        = _load("nyc_neighbourhood_means.pkl")
        self.neighbourhood_means = neigh_meta["means"]
        self.global_mean         = neigh_meta["global_mean"]

        logger.info(
            "NYCAirbnbPredictorONNX ready | providers=%s features=%d neighbourhoods=%d",
            providers, len(self.feature_list), len(self.neighbourhood_means),
        )

    @staticmethod
    def _best_providers():
        available = rt.get_available_providers()
        # prefer CoreML on Apple Silicon, CUDA on Linux GPU, fall back to CPU
        for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in available:
                return [p]
        return ["CPUExecutionProvider"]

    # ── public ────────────────────────────────────────────────────────────────

    def predict_raw(self, raw: dict) -> dict:
        self._validate(raw)
        row = self._engineer(raw)
        df  = pd.DataFrame([row])[self.feature_list]

        X_scaled = self.scaler.transform(df).astype(np.float32)
        log_price = float(np.array(
            self.session.run(None, {self.input_name: X_scaled})[0]
        ).flat[0])
        price_usd = float(np.expm1(log_price))

        return {
            "price_usd": round(price_usd, 2),
            "price_str": f"${price_usd:,.0f}/night",
            "log_price": round(log_price, 4),
        }

    def predict_batch(self, raws: list[dict]) -> list[dict]:
        results = []
        for i, raw in enumerate(raws):
            try:
                pred = self.predict_raw(raw)
                results.append({"index": i, **pred, "error": None})
            except Exception as exc:
                logger.warning("Batch item %d failed: %s", i, exc)
                results.append({"index": i, "price_usd": None, "error": str(exc)})
        return results

    def model_info(self) -> dict:
        providers = self.session.get_providers()
        return {
            "model_type":          "XGBRegressor (ONNX Runtime)",
            "inference_backend":   providers[0],
            "onnx_runtime_version": rt.__version__,
            "n_features":          len(self.feature_list),
            "n_neighbourhoods":    len(self.neighbourhood_means),
            "r2_test":             0.8241,
            "mae_dollar":          57.62,
            "mape_pct":            23.60,
            "training_rows":       16388,
            "triton_compatible":   True,
            "price_cap_used":      "$1,562/night (top 1% excluded)",
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _validate(self, raw: dict) -> None:
        required = ["accommodates", "bedrooms", "bathrooms", "room_type",
                    "borough", "latitude", "longitude", "minimum_nights",
                    "host_listings_count", "number_of_reviews", "reviews_per_month",
                    "review_scores_rating", "amenity_count"]
        missing = [k for k in required if raw.get(k) is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        if raw["room_type"] not in VALID_ROOM_TYPES:
            raise ValueError(f"room_type must be one of {VALID_ROOM_TYPES}")
        if raw["borough"] not in VALID_BOROUGHS:
            raise ValueError(f"borough must be one of {VALID_BOROUGHS}")
        if not (1 <= raw["accommodates"] <= 16):
            raise ValueError("accommodates must be between 1 and 16")
        if raw["review_scores_rating"] is not None and not (0 <= raw["review_scores_rating"] <= 5):
            raise ValueError("review_scores_rating must be between 0 and 5")

    def _engineer(self, raw: dict) -> dict:
        r = {}
        r["accommodates"]                = raw["accommodates"]
        r["bedrooms"]                    = raw.get("bedrooms", 1.0)
        r["bathrooms"]                   = raw.get("bathrooms", 1.0)
        r["host_is_superhost"]           = int(raw.get("host_is_superhost", False))
        r["host_listings_count"]         = raw.get("host_listings_count", 1)
        r["latitude"]                    = raw["latitude"]
        r["longitude"]                   = raw["longitude"]
        r["minimum_nights"]              = raw["minimum_nights"]
        r["minimum_nights_avg_ntm"]      = raw.get("minimum_nights_avg_ntm") or raw["minimum_nights"]
        r["number_of_reviews"]           = raw["number_of_reviews"]
        r["number_of_reviews_ltm"]       = raw.get("number_of_reviews_ltm", 0)
        r["reviews_per_month"]           = raw["reviews_per_month"]
        r["review_scores_rating"]        = raw.get("review_scores_rating", 0.0)
        r["review_scores_accuracy"]      = raw.get("review_scores_accuracy", 0.0)
        r["review_scores_cleanliness"]   = raw.get("review_scores_cleanliness", 0.0)
        r["review_scores_checkin"]       = raw.get("review_scores_checkin", 0.0)
        r["review_scores_communication"] = raw.get("review_scores_communication", 0.0)
        r["review_scores_location"]      = raw.get("review_scores_location", 0.0)
        r["review_scores_value"]         = raw.get("review_scores_value", 0.0)
        r["is_private_bath"]             = int(raw.get("is_private_bath", True))
        r["amenity_count"]               = raw.get("amenity_count", 20)
        r["has_gym"]                     = int(raw.get("has_gym", False))
        r["has_elevator"]                = int(raw.get("has_elevator", False))
        r["has_dryer"]                   = int(raw.get("has_dryer", False))
        r["has_air_conditioning"]        = int(raw.get("has_air_conditioning", False))
        r["has_washer"]                  = int(raw.get("has_washer", False))
        r["has_pool"]                    = int(raw.get("has_pool", False))

        rt_ = raw["room_type"]
        r["room_type_Entire home/apt"] = int(rt_ == "Entire home/apt")
        r["room_type_Hotel room"]      = int(rt_ == "Hotel room")
        r["room_type_Private room"]    = int(rt_ == "Private room")
        r["room_type_Shared room"]     = int(rt_ == "Shared room")

        r["log_host_listings_count"]    = np.log1p(r["host_listings_count"])
        r["log_number_of_reviews"]      = np.log1p(r["number_of_reviews"])
        r["log_number_of_reviews_ltm"]  = np.log1p(r["number_of_reviews_ltm"])
        r["log_reviews_per_month"]      = np.log1p(r["reviews_per_month"])
        r["log_minimum_nights"]         = np.log1p(r["minimum_nights"])
        r["log_minimum_nights_avg_ntm"] = np.log1p(r["minimum_nights_avg_ntm"])
        r["log_accommodates"]           = np.log1p(r["accommodates"])
        r["log_bedrooms"]               = np.log1p(r["bedrooms"])
        r["log_bathrooms"]              = np.log1p(r["bathrooms"])
        r["log_amenity_count"]          = np.log1p(r["amenity_count"])

        borough = raw["borough"]
        r["borough_Brooklyn"]      = int(borough == "Brooklyn")
        r["borough_Manhattan"]     = int(borough == "Manhattan")
        r["borough_Queens"]        = int(borough == "Queens")
        r["borough_Staten Island"] = int(borough == "Staten Island")

        neighbourhood = raw.get("neighbourhood", "")
        r["neighbourhood_price_rank"] = self.neighbourhood_means.get(neighbourhood, self.global_mean)

        r["accommodates_sq"] = r["accommodates"] ** 2
        r["bedrooms_sq"]     = r["bedrooms"] ** 2

        r["accommodates_x_review_scores_rating"]        = r["accommodates"] * r["review_scores_rating"]
        r["bedrooms_x_review_scores_rating"]            = r["bedrooms"]     * r["review_scores_rating"]
        r["host_listings_count_x_review_scores_rating"] = r["host_listings_count"] * r["review_scores_rating"]
        r["accommodates_x_host_is_superhost"]           = r["accommodates"] * r["host_is_superhost"]
        r["minimum_nights_x_accommodates"]              = r["minimum_nights"] * r["accommodates"]

        r["reviews_per_bedroom"]      = r["number_of_reviews"] / max(r["bedrooms"], 1)
        r["reviews_per_accommodates"] = r["number_of_reviews"] / r["accommodates"]
        r["host_density"]             = r["host_listings_count"] / r["accommodates"]

        scores = [r[c] for c in REVIEW_SCORE_COLS]
        r["review_score_std"] = float(np.std(scores)) if any(s > 0 for s in scores) else 0.0

        r["dist_from_midtown"] = np.sqrt(
            (r["latitude"]  - MIDTOWN_LAT) ** 2 +
            (r["longitude"] - MIDTOWN_LON) ** 2
        ) * 111

        return r
