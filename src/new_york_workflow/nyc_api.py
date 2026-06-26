"""
FastAPI serving layer for the NYC Airbnb price prediction model.

Production features:
  - Per-endpoint latency tracking (avg, min, max, p95) via HTTP middleware
  - Request / error counts and error-rate
  - Prediction value distribution
  - Structured JSON logging (set LOG_FORMAT=json in environment)
  - Graceful startup/shutdown via FastAPI lifespan

Run:
    uvicorn src.new_york_workflow.nyc_api:app --host 0.0.0.0 --port 8001 --reload

Or directly:
    python src/new_york_workflow/nyc_api.py
"""

import logging
import sys
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

# ── resolve project root so we can import from src/ regardless of CWD ────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.logger_config import setup_logging
from src.metrics import metrics
from src.new_york_workflow.nyc_predictor import NYCAirbnbPredictor

setup_logging()
logger = logging.getLogger(__name__)

predictor: NYCAirbnbPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Startup — loading NYC Airbnb model artifacts")
    predictor = NYCAirbnbPredictor()
    info = predictor.model_info()
    logger.info(
        "Model ready | type=%s  features=%d  R²=%.4f",
        info["model_type"], info["n_features"], info["r2_test"],
    )
    yield
    logger.info("Shutdown — cleaning up")


app = FastAPI(
    title="NYC Airbnb Price Prediction API",
    description=(
        "Production serving layer for the NYC Airbnb XGBoost price model "
        "(R²=0.82 on 20,485 core-market listings). "
        "Submit raw listing fields — feature engineering is applied server-side."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── middleware: record latency + errors for EVERY request ────────────────────

@app.middleware("http")
async def observe_requests(request: Request, call_next) -> Response:
    endpoint = request.url.path
    metrics.record_request(endpoint)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            metrics.record_error(endpoint)
        return response
    except Exception:
        metrics.record_error(endpoint)
        raise
    finally:
        metrics.record_latency(endpoint, (time.perf_counter() - t0) * 1000)


# ── schemas ──────────────────────────────────────────────────────────────────

class ListingFeatures(BaseModel):
    """Raw listing fields — identical to what you'd pull from Airbnb's API."""

    # Capacity
    accommodates: int        = Field(..., ge=1, le=16,  description="Number of guests the listing accommodates")
    bedrooms:     float      = Field(1.0, ge=0, le=20,  description="Number of bedrooms")
    bathrooms:    float      = Field(1.0, ge=0, le=20,  description="Number of bathrooms")
    is_private_bath: bool    = Field(True,               description="Private bathroom (True) vs shared")

    # Listing type
    room_type: str           = Field(..., description="Entire home/apt | Private room | Shared room | Hotel room")
    borough:   str           = Field(..., description="Manhattan | Brooklyn | Queens | Bronx | Staten Island")
    neighbourhood: str       = Field("",  description="Airbnb neighbourhood name (optional — improves accuracy)")

    # Location
    latitude:  float         = Field(..., ge=40.4, le=41.0, description="NYC latitude")
    longitude: float         = Field(..., ge=-74.3, le=-73.6, description="NYC longitude")

    # Stay constraints
    minimum_nights:          int   = Field(1,    ge=1)
    minimum_nights_avg_ntm:  Optional[float] = Field(None, description="Optional — defaults to minimum_nights")

    # Host
    host_is_superhost:    bool  = Field(False)
    host_listings_count:  int   = Field(1, ge=1)

    # Review stats
    number_of_reviews:    int   = Field(0, ge=0)
    number_of_reviews_ltm: int  = Field(0, ge=0, description="Reviews in last 12 months")
    reviews_per_month:    float = Field(0.0, ge=0)
    review_scores_rating:        float = Field(0.0, ge=0, le=5)
    review_scores_accuracy:      float = Field(0.0, ge=0, le=5)
    review_scores_cleanliness:   float = Field(0.0, ge=0, le=5)
    review_scores_checkin:       float = Field(0.0, ge=0, le=5)
    review_scores_communication: float = Field(0.0, ge=0, le=5)
    review_scores_location:      float = Field(0.0, ge=0, le=5)
    review_scores_value:         float = Field(0.0, ge=0, le=5)

    # Amenities
    amenity_count:       int  = Field(20, ge=0, description="Total number of amenities listed")
    has_gym:             bool = False
    has_elevator:        bool = False
    has_dryer:           bool = False
    has_air_conditioning: bool = False
    has_washer:          bool = False
    has_pool:            bool = False


class PredictionResponse(BaseModel):
    price_usd: float  = Field(..., description="Predicted nightly price in USD")
    price_str: str    = Field(..., description="Human-readable price, e.g. $175/night")
    log_price: float  = Field(..., description="Internal log1p prediction value")
    model:     str    = Field(..., description="Model type used for this prediction")
    r2_test:   float  = Field(..., description="Model R² on held-out test set")


class BatchPredictionResponse(BaseModel):
    predictions: list
    succeeded:   int
    requested:   int


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_response(result: dict) -> PredictionResponse:
    metrics.record_prediction(result["price_usd"])
    return PredictionResponse(
        price_usd = result["price_usd"],
        price_str = result["price_str"],
        log_price = result["log_price"],
        model     = "XGBoost",
        r2_test   = 0.8241,
    )


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "NYC Airbnb Price Prediction",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    """Liveness + readiness — includes uptime, request count, and error rate."""
    m = metrics.summary()
    return {
        "status":             "healthy",
        "uptime":             m["uptime_human"],
        "total_requests":     m["totals"]["requests"],
        "total_predictions":  m["predictions"]["total"],
        "error_rate":         m["totals"]["error_rate"],
        "model_r2":           predictor.model_info()["r2_test"],
    }


@app.get("/metrics")
def get_metrics():
    """
    Real-time observability dashboard.

    Returns per-endpoint:
      - request count, error count, error rate
      - latency avg / min / max / p95 (ms) — p95 available after 20+ requests

    Plus aggregate prediction value distribution (min / mean / max of USD prices).
    """
    return metrics.summary()


@app.get("/model-info")
def model_info():
    """Static metadata about the loaded model — version, features, training stats."""
    return predictor.model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingFeatures):
    """
    Predict the nightly price for a single Airbnb listing.

    Submit raw listing fields — the server applies the full feature engineering
    pipeline (log transforms, borough dummies, neighbourhood target encoding,
    polynomial and interaction features) before calling the XGBoost model.

    Returns predicted price in USD. Inverse log1p is applied automatically.
    """
    try:
        result = predictor.predict_raw(listing.model_dump())
        logger.info(
            "predict | %s %s %dbed → %s",
            listing.borough, listing.room_type, listing.bedrooms, result["price_str"],
        )
        return _to_response(result)
    except ValueError as exc:
        logger.warning("Invalid input: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(listings: List[ListingFeatures]):
    """
    Batch predict for up to 100 listings.

    Failed items are returned with `error` set and `price_usd` null —
    one bad row does not fail the whole batch.
    """
    if len(listings) > 100:
        raise HTTPException(status_code=422, detail="Batch size exceeds maximum of 100")
    try:
        raw_list = [l.model_dump() for l in listings]
        results  = predictor.predict_batch(raw_list)
        for r in results:
            if r["price_usd"] is not None:
                metrics.record_prediction(r["price_usd"])
        return BatchPredictionResponse(
            predictions = results,
            succeeded   = sum(1 for r in results if r["error"] is None),
            requested   = len(listings),
        )
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.new_york_workflow.nyc_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_config=None,  # our logger_config handles it
    )