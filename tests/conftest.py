"""
Shared fixtures for the NYC Airbnb API integration test suite.

TestClient spins up the full FastAPI lifespan (model loading) once per
session, so every test hits the same predictor instance — same as production.

In CI, model files are not committed to the repo. When they are absent the
`client` fixture substitutes a stub predictor so integration tests can still
exercise routing, auth, caching, and business logic without the binary artefacts.
"""

import os
import sys
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TEST_API_KEY = "test-api-key-for-pytest"
os.environ.setdefault("VALID_API_KEYS", _TEST_API_KEY)

_MODEL_FILE = Path(__file__).resolve().parents[1] / "models/nyc/nyc_xgb_model.onnx"


def _make_stub_predictor() -> MagicMock:
    """Minimal predictor stub used when model files are absent (CI)."""
    stub = MagicMock()
    stub.feature_list = []
    stub.scaler = MagicMock()
    stub._engineer = MagicMock(return_value={})
    stub._get_shap_explainer = MagicMock()
    stub.model_info.return_value = {
        "model_type":           "XGBRegressor (stub)",
        "inference_backend":    "stub",
        "onnx_runtime_version": "0",
        "n_features":           67,
        "n_neighbourhoods":     0,
        "r2_test":              0.80,
        "mae_dollar":           55.0,
        "training_report":      {},
    }
    stub.predict_raw.return_value = {
        "price_usd":              150.0,
        "price_low":              127.5,
        "price_high":             172.5,
        "price_str":              "$150/night",
        "log_price":              5.01,
        "model":                  "XGBoost (stub)",
        "r2_test":                0.80,
        "request_id":             str(uuid.uuid4()),
        "cache_hit":              False,
        "business_rules_applied": [],
    }
    stub.explain_raw.return_value = []
    return stub


@pytest.fixture(scope="session")
def client():
    """Single TestClient shared across the whole test session.

    When model binaries are present (local dev), uses the real predictor.
    When absent (CI), patches NYCAirbnbPredictorONNX with a stub so the full
    API stack (routing, auth, cache, business rules) is still exercised.
    """
    from src.serving.api import app

    if not _MODEL_FILE.exists():
        from unittest.mock import patch
        stub = _make_stub_predictor()
        with patch("src.serving.api.NYCAirbnbPredictorONNX", return_value=stub):
            with TestClient(app, headers={"X-API-Key": _TEST_API_KEY}) as c:
                yield c
    else:
        with TestClient(app, headers={"X-API-Key": _TEST_API_KEY}) as c:
            yield c


@pytest.fixture
def valid_listing():
    """
    Minimal valid listing that satisfies every required field.
    Adjust individual fields in each test as needed.
    """
    return {
        "accommodates":              2,
        "bedrooms":                  1.0,
        "bathrooms":                 1.0,
        "is_private_bath":           True,
        "room_type":                 "Entire home/apt",
        "borough":                   "Manhattan",
        "neighbourhood":             "Hell's Kitchen",
        "latitude":                  40.7638,
        "longitude":                 -73.9918,
        "minimum_nights":            2,
        "host_is_superhost":         False,
        "host_listings_count":       1,
        "number_of_reviews":         30,
        "number_of_reviews_ltm":     10,
        "reviews_per_month":         1.2,
        "review_scores_rating":      4.8,
        "review_scores_accuracy":    4.9,
        "review_scores_cleanliness": 4.7,
        "review_scores_checkin":     4.9,
        "review_scores_communication": 5.0,
        "review_scores_location":    4.9,
        "review_scores_value":       4.6,
        "amenity_count":             30,
        "has_gym":                   False,
        "has_elevator":              True,
        "has_dryer":                 True,
        "has_air_conditioning":      True,
        "has_washer":                True,
        "has_pool":                  False,
    }
