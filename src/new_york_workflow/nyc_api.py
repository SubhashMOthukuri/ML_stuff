"""
NYC Airbnb Price Prediction API — production serving layer.

Stack:
  - ONNX Runtime inference   (31x faster, no GIL)
  - Redis prediction cache   (MD5-keyed, 5-min TTL, ~40-50% hit rate in prod)
  - SQLite request store     (append-only log → retraining pipeline)
  - Redis Dead Letter Queue  (5xx / timeouts pushed for retry / alerting)
  - OpenTelemetry tracing    (request IDs, spans, OTLP export)
  - slowapi rate limiting    (per-IP, 60/min on /predict)
  - CORS                     (React frontend on :5173)

Run locally:
  PYTHONPATH=. python -m uvicorn src.new_york_workflow.nyc_api:app --port 8001 --reload

Run production:
  PYTHONPATH=. gunicorn -c gunicorn.conf.py src.new_york_workflow.nyc_api:app

Environment variables:
  RATE_LIMIT          requests per minute per IP   (default: 60/minute)
  OTLP_ENDPOINT       OpenTelemetry collector URL  (default: console)
  REDIS_HOST          Redis host                   (default: localhost)
  REDIS_PORT          Redis port                   (default: 6379)
  CACHE_TTL_SECONDS   Cache entry lifetime         (default: 300)
  PREDICTION_DB       SQLite path or Postgres DSN  (default: data/predictions.db)
  DLQ_MAX_SIZE        Max DLQ entries in Redis     (default: 10000)
"""

import logging
import sys
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.logger_config import setup_logging
from src.metrics import metrics
from src.new_york_workflow.nyc_predictor_onnx import NYCAirbnbPredictorONNX
from src.new_york_workflow.nyc_cache import PredictionCache
from src.new_york_workflow.nyc_store import RequestStore
from src.new_york_workflow.nyc_dlq import DeadLetterQueue
from src.new_york_workflow.nyc_drift import DriftMonitor
from src.new_york_workflow.nyc_alerts import alerts as alert_store
from src.new_york_workflow.nyc_shadow import ShadowPredictor

setup_logging()
logger = logging.getLogger(__name__)

# ── OpenTelemetry ─────────────────────────────────────────────────────────────

def _setup_tracing():
    provider = TracerProvider()
    otlp_endpoint = os.getenv("OTLP_ENDPOINT")
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        logger.info("OpenTelemetry → OTLP: %s", otlp_endpoint)
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry → console exporter (set OTLP_ENDPOINT for collector)")
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)

tracer = _setup_tracing()

# ── rate limiter ──────────────────────────────────────────────────────────────

RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
limiter    = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ── singletons (populated at startup) ────────────────────────────────────────

predictor: NYCAirbnbPredictorONNX = None
cache:     PredictionCache        = None
store:     RequestStore           = None
dlq:       DeadLetterQueue        = None
drift:     DriftMonitor           = None
shadow:    ShadowPredictor        = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, cache, store, dlq, drift, shadow

    logger.info("Startup — initialising services")
    predictor = NYCAirbnbPredictorONNX()
    cache      = PredictionCache()
    store      = RequestStore()
    dlq        = DeadLetterQueue()
    drift      = DriftMonitor()
    shadow     = ShadowPredictor()
    shadow.inject_champion(predictor)   # share _engineer() + feature_list + scaler

    info = predictor.model_info()
    logger.info(
        "Ready | backend=%s  R²=%.4f  cache=%s  store=%s  dlq=%s  shadow=%s",
        info["inference_backend"], info["r2_test"],
        "redis" if cache._client else "disabled",
        store._path,
        "redis" if dlq._client else "in-memory",
        "active" if shadow.active else "disabled",
    )
    yield
    logger.info("Shutdown")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Airbnb Price Prediction API",
    description="ONNX Runtime · Redis cache · SQLite request store · DLQ · OpenTelemetry",
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

FastAPIInstrumentor.instrument_app(app)


# ── middleware: request ID + latency + DLQ on 5xx ────────────────────────────

@app.middleware("http")
async def observe(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("http.request_id", request_id)
        span.set_attribute("http.client_ip", get_remote_address(request))

    endpoint = request.url.path
    metrics.record_request(endpoint)
    t0 = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - t0) * 1000
    metrics.record_latency(endpoint, latency_ms)

    if response.status_code >= 500:
        metrics.record_error(endpoint)
        # /predict pushes to DLQ itself (with full payload); skip here to avoid duplicates.
        # For all other endpoints we push a minimal entry since we have no body.
        if dlq and endpoint != "/predict":
            dlq.push(
                payload     = {"url": str(request.url), "method": request.method},
                error       = f"HTTP {response.status_code}",
                endpoint    = endpoint,
                request_id  = request_id,
                status_code = response.status_code,
            )

    response.headers["X-Request-ID"] = request_id
    return response


# ── schemas ───────────────────────────────────────────────────────────────────

class ListingFeatures(BaseModel):
    accommodates:                int   = Field(..., ge=1, le=16)
    bedrooms:                    float = Field(1.0, ge=0, le=20)
    bathrooms:                   float = Field(1.0, ge=0, le=20)
    is_private_bath:             bool  = True
    room_type:                   str   = Field(..., description="Entire home/apt | Private room | Shared room | Hotel room")
    borough:                     str   = Field(..., description="Manhattan | Brooklyn | Queens | Bronx | Staten Island")
    neighbourhood:               str   = ""
    latitude:                    float = Field(..., ge=40.4, le=41.0)
    longitude:                   float = Field(..., ge=-74.3, le=-73.6)
    minimum_nights:              int   = Field(1, ge=1)
    minimum_nights_avg_ntm:      Optional[float] = None
    host_is_superhost:           bool  = False
    host_listings_count:         int   = Field(1, ge=1)
    number_of_reviews:           int   = Field(0, ge=0)
    number_of_reviews_ltm:       int   = Field(0, ge=0)
    reviews_per_month:           float = Field(0.0, ge=0)
    review_scores_rating:        float = Field(0.0, ge=0, le=5)
    review_scores_accuracy:      float = Field(0.0, ge=0, le=5)
    review_scores_cleanliness:   float = Field(0.0, ge=0, le=5)
    review_scores_checkin:       float = Field(0.0, ge=0, le=5)
    review_scores_communication: float = Field(0.0, ge=0, le=5)
    review_scores_location:      float = Field(0.0, ge=0, le=5)
    review_scores_value:         float = Field(0.0, ge=0, le=5)
    amenity_count:               int   = Field(20, ge=0)
    has_gym:                     bool  = False
    has_elevator:                bool  = False
    has_dryer:                   bool  = False
    has_air_conditioning:        bool  = False
    has_washer:                  bool  = False
    has_pool:                    bool  = False


class PredictionResponse(BaseModel):
    price_usd:  float
    price_str:  str
    log_price:  float
    model:      str
    r2_test:    float
    request_id: str
    cache_hit:  bool


class BatchPredictionResponse(BaseModel):
    predictions: list
    succeeded:   int
    requested:   int


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "NYC Airbnb Price Prediction",
        "version": "3.0.0",
        "backend": "ONNX Runtime",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    m    = metrics.summary()
    info = predictor.model_info()
    return {
        "status":            "healthy",
        "uptime":            m["uptime_human"],
        "total_requests":    m["totals"]["requests"],
        "total_predictions": m["predictions"]["total"],
        "error_rate":        m["totals"]["error_rate"],
        "model_r2":          info["r2_test"],
        "inference_backend": info["inference_backend"],
        "cache":             cache.stats(),
        "dlq_size":          dlq.size(),
        "store_rows":        store.count(),
    }


@app.get("/health/ready")
def readiness():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.get("/health/live")
def liveness():
    return {"alive": True}


@app.get("/metrics")
def get_metrics():
    return metrics.summary()


@app.get("/model-info")
def model_info():
    """
    Current champion model metadata.
    Primary source: MLflow Model Registry (alias 'champion').
    Fallback: local nyc_training_report.json when MLflow is unreachable.
    """
    import json as _json
    base = predictor.model_info()   # always include ONNX / local info

    mlflow_uri  = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{_ROOT / 'mlflow.db'}")
    mlflow_url  = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
    model_name  = "nyc-airbnb-xgboost"

    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient(tracking_uri=mlflow_uri)
        mv     = client.get_model_version_by_alias(model_name, "champion")
        run    = client.get_run(mv.run_id)
        return {
            **base,
            "registry": {
                "source":        "mlflow",
                "model_name":    model_name,
                "version":       mv.version,
                "run_id":        mv.run_id,
                "run_name":      run.info.run_name,
                "registered_at": mv.creation_timestamp,
                "git_sha":       mv.tags.get("git_sha", "—"),
                "data_date":     mv.tags.get("data_date", "—"),
                "metrics":       {k: round(v, 4) for k, v in run.data.metrics.items()},
                "mlflow_ui":     f"{mlflow_url}/#/models/{model_name}/versions/{mv.version}",
            }
        }
    except Exception as exc:
        # Graceful fallback — MLFLOW_TRACKING_URI may not be running locally
        report_path = _ROOT / "models" / "nyc" / "nyc_training_report.json"
        registry_fallback = {"source": "local_report", "error": str(exc)}
        if report_path.exists():
            rpt = _json.loads(report_path.read_text())
            registry_fallback.update({
                "timestamp":  rpt.get("timestamp"),
                "best_model": rpt.get("best_model"),
                "n_features": rpt.get("n_features"),
                "metrics":    rpt.get("models", {}).get("XGBoost", {}).get("test", {}),
            })
        return {**base, "registry": registry_fallback}


@app.get("/cache/stats")
def cache_stats():
    """Redis cache hit/miss counters and connection status."""
    return cache.stats()


@app.get("/dlq")
def dlq_peek(n: int = 20):
    """
    Inspect the dead letter queue — most recent failed requests first.
    Use ?n=N to control how many entries are returned (max 200).
    """
    n = min(n, 200)
    return {
        "size":    dlq.size(),
        "showing": n,
        "entries": dlq.peek(n),
    }


@app.delete("/dlq")
def dlq_clear():
    """Flush all DLQ entries. Returns count removed."""
    removed = dlq.clear()
    return {"removed": removed}


@app.get("/training-data")
def training_data(limit: int = 1000):
    """
    Returns prediction logs for offline retraining.
    Each row has all input features + predicted_price + timestamp.
    Pipe this to your training script to do continuous retraining.
    """
    limit = min(limit, 50_000)
    rows  = store.export_for_retraining(limit=limit)
    return {
        "count":  len(rows),
        "schema": "request_id, timestamp, predicted_price, log_price, cache_hit, ...all input fields",
        "rows":   rows,
    }


@app.get("/training-data/stats")
def training_data_stats():
    return store.stats()


@app.delete("/predictions")
def clear_predictions(before: str | None = None):
    """
    Delete prediction rows used for drift monitoring.
    Pass ?before=<ISO timestamp> to prune only rows older than that date.
    Omit to clear all rows (useful after load tests pollute the drift window).
    """
    n = store.clear(before=before)
    return {"deleted": n}


@app.get("/drift")
def drift_check(window_hours: int = 168, push_alert: bool = False):
    """
    Run drift check over the last `window_hours` of production traffic (default 7 days).
    Status: 'ok' | 'warning' | 'critical' | 'insufficient_data' | 'no_baseline'
    PSI > 0.10 → warning, PSI > 0.20 → critical.

    Alerts are only pushed when push_alert=true (explicit check) to prevent a page
    load / background poll from flooding the alert log as a side-effect of a GET.
    Alerts are deduplicated: push_once() ensures at most one pending alert per type.
    """
    report = drift.check(window_hours=window_hours)

    if push_alert and report.status in ("critical", "warning"):
        alert_store.push_once(
            alert_type=f"drift_{report.status}",
            message=f"{'Critical drift' if report.status == 'critical' else 'Drift warning'} "
                    f"detected over last {window_hours}h",
            severity=report.status,
            details=report.to_dict(),
        )

    return report.to_dict()


@app.get("/alerts")
def get_alerts(pending_only: bool = False, limit: int = 50):
    """
    Pending production alerts: drift, retraining gate failures, DLQ spikes.
    Use ?pending_only=true to see only unacknowledged alerts.
    """
    entries = alert_store.get_pending() if pending_only else alert_store.get_all(limit=limit)
    return {"stats": alert_store.stats(), "alerts": entries}


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged so it stops appearing in pending."""
    ok = alert_store.acknowledge(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"acknowledged": alert_id}


@app.post("/alerts/acknowledge-all")
def acknowledge_all_alerts():
    """Acknowledge every pending alert at once. Returns the count cleared."""
    n = alert_store.acknowledge_all()
    return {"acknowledged_count": n}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(RATE_LIMIT)
def predict(listing: ListingFeatures, request: Request):
    """
    Predict nightly price. Checks Redis cache first; runs ONNX inference on miss.
    Full request payload is logged to SQLite for retraining. 5xx pushes to DLQ.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    raw        = listing.model_dump()

    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("listing.borough",      listing.borough)
        span.set_attribute("listing.room_type",    listing.room_type)
        span.set_attribute("listing.accommodates", listing.accommodates)

        # ── 1. cache lookup ───────────────────────────────────────────────
        cached = cache.get(raw)
        if cached:
            span.set_attribute("cache.hit", True)
            store.log_prediction(request_id, raw, cached, cache_hit=True)
            metrics.record_prediction(cached["price_usd"])
            logger.info("predict | rid=%s CACHE HIT %s → %s", request_id[:8], listing.borough, cached["price_str"])
            return PredictionResponse(
                **cached,
                model      = "XGBoost (ONNX Runtime)",
                r2_test    = 0.8241,
                request_id = request_id,
                cache_hit  = True,
            )

        # ── 2. ONNX inference ─────────────────────────────────────────────
        try:
            result = predictor.predict_raw(raw)
        except ValueError as exc:
            span.set_attribute("error", str(exc))
            logger.warning("Validation rid=%s: %s", request_id[:8], exc)
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            span.set_attribute("error", str(exc))
            logger.error("Inference error rid=%s: %s", request_id[:8], exc, exc_info=True)
            # Push full payload to DLQ so we can retry or alert
            dlq.push(
                payload     = raw,
                error       = str(exc),
                endpoint    = "/predict",
                request_id  = request_id,
                status_code = 500,
            )
            raise HTTPException(status_code=500, detail=str(exc))

        # ── 3. write-through: cache + store ───────────────────────────────
        cache.set(raw, result)
        store.log_prediction(request_id, raw, result, cache_hit=False)
        metrics.record_prediction(result["price_usd"])

        # ── 4. shadow: run challenger alongside champion (fire-and-forget) ────
        shadow.run_async(
            raw           = raw,
            champion_price = result["price_usd"],
            request_id    = request_id,
            borough       = listing.borough,
            room_type     = listing.room_type,
        )

        span.set_attribute("cache.hit",            False)
        span.set_attribute("prediction.price_usd", result["price_usd"])
        span.set_attribute("shadow.active",        shadow.active)
        logger.info(
            "predict | rid=%s MISS %s %s %.1fbed → %s",
            request_id[:8], listing.borough, listing.room_type,
            listing.bedrooms, result["price_str"],
        )

        return PredictionResponse(
            **result,
            model      = "XGBoost (ONNX Runtime)",
            r2_test    = 0.8241,
            request_id = request_id,
            cache_hit  = False,
        )


@app.post("/predict-batch")
@limiter.limit("20/minute")
def predict_batch(listings: List[ListingFeatures], request: Request):
    """Batch predict up to 100 listings. Stricter rate limit: 20/minute."""
    if len(listings) > 100:
        raise HTTPException(status_code=422, detail="Batch size exceeds maximum of 100")

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    with tracer.start_as_current_span("predict_batch") as span:
        span.set_attribute("batch.size", len(listings))
        results = []
        for i, listing in enumerate(listings):
            raw    = listing.model_dump()
            cached = cache.get(raw)
            if cached:
                results.append({"index": i, **cached, "cache_hit": True, "error": None})
                store.log_prediction(f"{request_id}:{i}", raw, cached, cache_hit=True)
                metrics.record_prediction(cached["price_usd"])
            else:
                try:
                    result = predictor.predict_raw(raw)
                    cache.set(raw, result)
                    store.log_prediction(f"{request_id}:{i}", raw, result, cache_hit=False)
                    metrics.record_prediction(result["price_usd"])
                    results.append({"index": i, **result, "cache_hit": False, "error": None})
                except Exception as exc:
                    logger.warning("Batch item %d failed: %s", i, exc)
                    results.append({"index": i, "price_usd": None, "cache_hit": False, "error": str(exc)})

        succeeded = sum(1 for r in results if r["error"] is None)
        span.set_attribute("batch.succeeded", succeeded)
        return BatchPredictionResponse(predictions=results, succeeded=succeeded, requested=len(listings))


@app.get("/shadow/stats")
def shadow_stats(window_hours: int = 168):
    """
    Head-to-head comparison between champion and challenger over the last
    `window_hours` (default 7 days).

    Returns per-borough / per-room-type breakdowns, agreement rates, and
    a recommendation: 'promote' | 'monitor' | 'reject' | 'insufficient_data'.

    Shadow deployment is only active when models/nyc/challenger.onnx exists.
    Create one by running  python scripts/retrain.py  and letting the gate fail.
    """
    return shadow.stats(window_hours=window_hours)


@app.post("/shadow/promote")
def shadow_promote():
    """
    Promote the challenger to champion (file swap — no traffic change yet):
      1. Backs up current champion → models/nyc/previous.onnx
      2. Copies challenger.onnx  → models/nyc/nyc_xgb_model.onnx
      3. Copies challenger_scaler.pkl → models/nyc/nyc_scaler.pkl
      4. Removes challenger artefacts → shadow disabled

    Restart the API after this to serve the new champion:
      docker compose restart api
    """
    result = shadow.promote_challenger()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/shadow/reject")
def shadow_reject():
    """
    Discard the challenger without promoting it.
    challenger.onnx is deleted — shadow deployment disabled until next retrain.
    """
    return shadow.reject_challenger()


@app.delete("/shadow/comparisons")
def shadow_clear():
    """Delete all shadow comparison rows from the database."""
    deleted = shadow.clear_comparisons()
    return {"deleted": deleted}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.new_york_workflow.nyc_api:app", host="0.0.0.0", port=8001, reload=False, log_config=None)
