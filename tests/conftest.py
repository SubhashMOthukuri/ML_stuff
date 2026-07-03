"""
Shared fixtures for the NYC Airbnb API integration test suite.

TestClient spins up the full FastAPI lifespan (model loading) once per
session, so every test hits the same predictor instance — same as production.
"""

import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set test API key BEFORE the app module is imported (module-level VALID_API_KEYS is
# evaluated at import time). Empty env = auth disabled; this enables it for auth tests.
_TEST_API_KEY = "test-api-key-for-pytest"
os.environ.setdefault("VALID_API_KEYS", _TEST_API_KEY)


@pytest.fixture(scope="session")
def client():
    """Single TestClient shared across the whole test session."""
    from src.new_york_workflow.nyc_api import app
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
