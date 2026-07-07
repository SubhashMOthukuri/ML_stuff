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
  VALID_API_KEYS              Comma-separated valid API keys for X-API-Key auth (default: unset → auth disabled)
  GROUND_TRUTH_INGEST_TOKEN  Shared secret for POST /ground-truth/ingest (default: unset → endpoint disabled)
"""

import logging
import sys
import time
import uuid

import structlog
from structlog.contextvars import clear_contextvars, bind_contextvars

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional, List

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import settings
from src.logger_config import setup_logging
from src.metrics import (
    metrics,
    prom_canary_traffic_pct,
    prom_active_alerts,
    prom_dlq_size,
)
from src.new_york_workflow.nyc_predictor_onnx import NYCAirbnbPredictorONNX
from src.new_york_workflow.nyc_cache import PredictionCache
from src.new_york_workflow.nyc_store import RequestStore
from src.new_york_workflow.nyc_dlq import DeadLetterQueue
from src.new_york_workflow.nyc_drift import DriftMonitor
from src.new_york_workflow.nyc_alerts import alerts as alert_store
from src.new_york_workflow.nyc_shadow import ShadowPredictor
from src.new_york_workflow.nyc_ground_truth import GroundTruthStore
from src.new_york_workflow.nyc_ab import ABTest
from src.new_york_workflow.nyc_canary import CanaryDeployment
from scripts.fetch_ground_truth import run as run_ground_truth_ingest, INSIDEAIRBNB_URL

setup_logging()
logger = structlog.get_logger(__name__)

# ── OpenTelemetry ─────────────────────────────────────────────────────────────

def _setup_tracing():
    provider = TracerProvider()
    otlp_endpoint = settings.otlp_endpoint
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

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ── price sanity bounds ───────────────────────────────────────────────────────
# NYC Airbnb market range. Predictions outside this range indicate model failure.
_PRICE_MIN_USD = 10.0
_PRICE_MAX_USD = 5_000.0

# ── API key auth ──────────────────────────────────────────────────────────────

_AUTH_EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/")


# ── admin auth (gates /ground-truth/ingest) ───────────────────────────────────

def _verify_ingest_token(x_ingest_token: Optional[str] = Header(None)) -> None:
    if not settings.ground_truth_ingest_token:
        raise HTTPException(status_code=503, detail="GROUND_TRUTH_INGEST_TOKEN not configured on server")
    if x_ingest_token != settings.ground_truth_ingest_token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ingest-Token header")

def _guard_price(price_usd: float, request_id: str) -> None:
    """Raise 422 if predicted price is outside the valid NYC Airbnb market range."""
    if price_usd < _PRICE_MIN_USD or price_usd > _PRICE_MAX_USD:
        logger.error(
            "price_guard FAIL rid=%s price=%.2f range=[%.0f, %.0f]",
            request_id[:8], price_usd, _PRICE_MIN_USD, _PRICE_MAX_USD,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Predicted price ${price_usd:.2f} is outside valid range "
                   f"[${_PRICE_MIN_USD:.0f}, ${_PRICE_MAX_USD:.0f}] — possible model failure",
        )


# ── singletons (populated at startup) ────────────────────────────────────────

predictor:    NYCAirbnbPredictorONNX = None
cache:        PredictionCache        = None
store:        RequestStore           = None
dlq:          DeadLetterQueue        = None
drift:        DriftMonitor           = None
shadow:       ShadowPredictor        = None
ground_truth: GroundTruthStore       = None
ab_test:      ABTest                 = None
canary:       CanaryDeployment       = None
_r2_test:     float                  = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, cache, store, dlq, drift, shadow, ground_truth, ab_test, canary, _r2_test

    logger.info("Startup — initialising services")
    predictor    = NYCAirbnbPredictorONNX()
    cache        = PredictionCache()
    store        = RequestStore()
    dlq          = DeadLetterQueue()
    drift        = DriftMonitor()
    shadow       = ShadowPredictor()
    ground_truth = GroundTruthStore()
    ab_test      = ABTest()
    canary       = CanaryDeployment()
    shadow.inject_champion(predictor)   # share _engineer() + feature_list + scaler
    canary.inject(shadow)               # share shadow predictor for challenger inference

    info     = predictor.model_info()
    _r2_test = info["r2_test"] or 0.0
    logger.info(
        "Ready | backend=%s  R²=%.4f  cache=%s  store=%s  dlq=%s  shadow=%s  ab=%s  canary=%s",
        info["inference_backend"], _r2_test,
        "redis" if cache._client else "disabled",
        store.safe_label,
        "redis" if dlq._client else "in-memory",
        "active" if shadow.active else "disabled",
        f"active({ab_test.test_id})" if ab_test.active else "idle",
        f"active({canary.current_pct}%)" if canary.active else "idle",
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
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

FastAPIInstrumentor.instrument_app(app)

# Prometheus: instrument HTTP request count + latency automatically.
# We expose /metrics manually below so we can refresh live gauges first.
_prom_instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/health.*", "/docs.*", "/openapi.*", "/redoc.*", "/metrics.*"],
)
_prom_instrumentator.instrument(app)


# ── middleware: request ID + latency + DLQ on 5xx ────────────────────────────
# Registered first → innermost → runs AFTER api_key_auth rejects or passes through.

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


# ── middleware: API key auth ──────────────────────────────────────────────────
# Registered AFTER observe → middle layer (LIFO stack).
# Unauthenticated requests short-circuit here; observe never records them.

@app.middleware("http")
async def api_key_auth(request: Request, call_next) -> Response:
    path = request.url.path
    api_keys = settings.api_keys_set
    exempt = (
        not api_keys
        or path == "/"
        or path.startswith("/health")
        or path.startswith("/docs")
        or path.startswith("/openapi")
        or path.startswith("/redoc")
        or path.startswith("/metrics")
    )
    if exempt:
        return await call_next(request)
    if request.headers.get("X-API-Key", "") not in api_keys:
        return Response(
            content='{"detail":"Invalid or missing X-API-Key header"}',
            status_code=401,
            media_type="application/json",
        )
    return await call_next(request)


# ── middleware: structured log context ────────────────────────────────────────
# Registered LAST → outermost (LIFO stack) → runs FIRST on every request.
# Binds trace_id / method / path into structlog contextvars so every log line
# emitted during the request automatically carries these fields.

@app.middleware("http")
async def request_context_middleware(request: Request, call_next) -> Response:
    clear_contextvars()
    bind_contextvars(
        trace_id=request.headers.get("X-Trace-Id", str(uuid.uuid4())[:8]),
        method=request.method,
        path=request.url.path,
    )
    return await call_next(request)


# ── schemas ───────────────────────────────────────────────────────────────────

class ListingFeatures(BaseModel):
    listing_id:                  Optional[str] = Field(None, description="InsideAirbnb listing ID — enables ground truth feedback loop")
    accommodates:                int   = Field(..., ge=1, le=16)
    bedrooms:                    float = Field(1.0, ge=0, le=20)
    bathrooms:                   float = Field(1.0, ge=0, le=20)
    is_private_bath:             bool  = True
    room_type:                   Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    borough:                     Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
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
    checkin_date:                Optional[str] = Field(None, description="Check-in date (YYYY-MM-DD) — used to compute days_to_checkin")


class PredictionResponse(BaseModel):
    price_usd:               float
    price_low:               float
    price_high:              float
    price_str:               str
    log_price:               float
    model:                   str
    r2_test:                 float
    request_id:              str
    cache_hit:               bool
    business_rules_applied:  Optional[list] = None


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


@app.get("/metrics/summary")
def get_metrics():
    return metrics.summary()


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint — updates live gauges then returns text/plain."""
    # Refresh gauges from live state right before Prometheus reads them
    if canary is not None:
        prom_canary_traffic_pct.set(canary.current_pct if canary.active else 0)
    if dlq is not None:
        prom_dlq_size.set(dlq.size() if hasattr(dlq, "size") else 0)
    alert_stats = alert_store.stats().get("by_severity", {})
    for sev, count in alert_stats.items():
        prom_active_alerts.labels(severity=sev).set(count)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    """
    Current champion model metadata.
    Primary source: MLflow Model Registry (alias 'champion').
    Fallback: local nyc_training_report.json when MLflow is unreachable.
    """
    import json as _json
    base = predictor.model_info()   # always include ONNX / local info

    mlflow_uri  = settings.mlflow_tracking_uri
    mlflow_url  = settings.mlflow_ui_url
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


@app.get("/ground-truth/stats")
def ground_truth_stats():
    """
    Production accuracy measured against real InsideAirbnb prices.
    Populated by scripts/fetch_ground_truth.py (run monthly after each snapshot).
    """
    stats = ground_truth.stats()
    recent = ground_truth.recent(n=10)
    dist   = ground_truth.error_distribution()

    baseline_mae = None
    try:
        import json as _json
        bp = _ROOT / "models" / "nyc" / "nyc_training_report.json"
        if bp.exists():
            rpt = _json.loads(bp.read_text())
            baseline_mae = rpt.get("models", {}).get("XGBoost", {}).get("test", {}).get("mae_dollar")
    except Exception:
        pass

    return {
        "stats":            stats,
        "baseline_mae":     baseline_mae,
        "mae_drift":        round(stats["mae_dollar"] - baseline_mae, 2) if stats["mae_dollar"] and baseline_mae else None,
        "error_distribution": dist,
        "recent":           recent,
    }


@app.post("/ground-truth/ingest", dependencies=[Depends(_verify_ingest_token)])
def ground_truth_ingest(url: str = INSIDEAIRBNB_URL, dry_run: bool = False):
    """
    Trigger scripts/fetch_ground_truth.py against the live predictions DB this
    API instance is running against — no SSH/file-copy needed to reach prod data.

    Requires header: X-Ingest-Token: <GROUND_TRUTH_INGEST_TOKEN>
    Intended caller: a scheduled GitHub Actions job (monthly, after InsideAirbnb
    publishes a fresh snapshot). See .github/workflows/ground-truth.yml.
    """
    try:
        result = run_ground_truth_ingest(url=url, dry_run=dry_run)
    except Exception as e:
        logger.error("Ground truth ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    return result


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
@limiter.limit(settings.rate_limit)
def predict(listing: ListingFeatures, request: Request):
    """
    Predict nightly price. Checks Redis cache first; runs ONNX inference on miss.
    Full request payload is logged to SQLite for retraining. 5xx pushes to DLQ.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    listing_id = listing.listing_id
    raw        = listing.model_dump(exclude={"listing_id"})

    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("listing.borough",      listing.borough)
        span.set_attribute("listing.room_type",    listing.room_type)
        span.set_attribute("listing.accommodates", listing.accommodates)
        if listing_id:
            span.set_attribute("listing.id", listing_id)

        # ── 1. cache lookup ───────────────────────────────────────────────
        cached = cache.get(raw)
        if cached:
            span.set_attribute("cache.hit", True)
            store.log_prediction(request_id, raw, cached, cache_hit=True, listing_id=listing_id)
            metrics.record_prediction_full(cached["price_usd"], arm="champion", cache_hit=True)
            logger.info("predict | rid=%s CACHE HIT %s → %s", request_id[:8], listing.borough, cached["price_str"])
            return PredictionResponse(
                **cached,
                model      = "XGBoost (ONNX Runtime)",
                r2_test    = _r2_test,
                request_id = request_id,
                cache_hit  = True,
            )

        # ── 2. ONNX inference ─────────────────────────────────────────────
        _infer_t0 = time.perf_counter()
        try:
            result = predictor.predict_raw(raw)
        except ValueError as exc:
            span.set_attribute("error", str(exc))
            logger.warning("Validation rid=%s: %s", request_id[:8], exc)
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            span.set_attribute("error", str(exc))
            logger.error("Inference error rid=%s: %s", request_id[:8], exc, exc_info=True)
            dlq.push(
                payload     = raw,
                error       = str(exc),
                endpoint    = "/predict",
                request_id  = request_id,
                status_code = 500,
            )
            canary.record_post_promotion((time.perf_counter() - _infer_t0) * 1000, success=False)
            raise HTTPException(status_code=500, detail=str(exc))

        _infer_ms = (time.perf_counter() - _infer_t0) * 1000

        # ── 3. write-through: cache + store ───────────────────────────────
        cache.set(raw, result)
        store.log_prediction(request_id, raw, result, cache_hit=False, listing_id=listing_id)
        metrics.record_prediction(result["price_usd"])  # in-memory only; prom recorded after routing below

        # ── 4. routing: canary → A/B → shadow (priority order) ──────────────
        # Canary and A/B show challenger output to real users.
        # Shadow runs challenger hidden (fire-and-forget) as a fallback.
        # Cache hits always serve champion — only cache misses are routed.
        routed_arm = "champion"

        if canary.active:
            routed_arm = canary.route(request_id)
            t0 = time.time()
            if routed_arm == "challenger":
                chal = shadow.predict_sync(raw)
                latency_ms = (time.time() - t0) * 1000
                if chal:
                    result = chal
                    canary.record_result("challenger", latency_ms, success=True)
                else:
                    canary.record_result("challenger", latency_ms, success=False)
                    routed_arm = "champion"   # fallback
            else:
                canary.record_result("champion", (time.time() - t0) * 1000, success=True)

        elif ab_test.active:
            routed_arm = ab_test.route(request_id)
            if routed_arm == "challenger":
                chal = shadow.predict_sync(raw)
                if chal:
                    result = chal
                else:
                    routed_arm = "champion"   # fallback; log as champion
            ab_test.log_prediction(
                request_id      = request_id,
                listing_id      = listing_id,
                arm             = routed_arm,
                predicted_price = result["price_usd"],
                borough         = listing.borough,
                room_type       = listing.room_type,
            )

        else:
            # Shadow: challenger runs hidden, champion result served to user
            shadow.run_async(
                raw            = raw,
                champion_price = result["price_usd"],
                request_id     = request_id,
                borough        = listing.borough,
                room_type      = listing.room_type,
            )

        # ── 5. post-promotion monitoring ──────────────────────────────────
        # After canary promotes challenger → champion, the monitor watches the
        # new model's error rate for POST_PROMOTION_WINDOW_S seconds. If error
        # rate jumps 3× the canary baseline, it auto-reverts files + alerts.
        canary.record_post_promotion(_infer_ms, success=True)

        # Prometheus: record prediction with final arm (known only after routing)
        from src.metrics import prom_predictions_total, prom_prediction_price_usd
        prom_predictions_total.labels(arm=routed_arm, cache_hit="false").inc()
        prom_prediction_price_usd.observe(result["price_usd"])

        _guard_price(result["price_usd"], request_id)

        span.set_attribute("cache.hit",            False)
        span.set_attribute("prediction.price_usd", result["price_usd"])
        span.set_attribute("shadow.active",        shadow.active)
        span.set_attribute("routed_arm",           routed_arm)
        logger.info(
            "predict | rid=%s MISS %s %s %.1fbed → %s",
            request_id[:8], listing.borough, listing.room_type,
            listing.bedrooms, result["price_str"],
        )

        return PredictionResponse(
            **result,
            model      = "XGBoost (ONNX Runtime)",
            r2_test    = _r2_test,
            request_id = request_id,
            cache_hit  = False,
        )


@app.post("/predict-batch")
@limiter.limit(settings.rate_limit)
def predict_batch(listings: List[dict], request: Request):
    """Batch predict up to 100 listings. Stricter rate limit: 20/minute."""
    if len(listings) > 100:
        raise HTTPException(status_code=422, detail="Batch size exceeds maximum of 100")

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    with tracer.start_as_current_span("predict_batch") as span:
        span.set_attribute("batch.size", len(listings))
        results = []
        for i, raw_item in enumerate(listings):
            try:
                listing = ListingFeatures.model_validate(raw_item)
            except ValidationError as exc:
                err = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()
                )
                logger.warning("Batch item %d validation error: %s", i, err)
                results.append({"index": i, "price_usd": None, "cache_hit": False, "error": err})
                continue

            lid = listing.listing_id
            raw = listing.model_dump(exclude={"listing_id"})
            cached = cache.get(raw)
            if cached:
                results.append({"index": i, **cached, "cache_hit": True, "error": None})
                store.log_prediction(f"{request_id}:{i}", raw, cached, cache_hit=True, listing_id=lid)
                metrics.record_prediction(cached["price_usd"])
            else:
                try:
                    result = predictor.predict_raw(raw)
                    cache.set(raw, result)
                    store.log_prediction(f"{request_id}:{i}", raw, result, cache_hit=False, listing_id=lid)
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


# ── A/B test ──────────────────────────────────────────────────────────────────

@app.post("/ab/start")
def ab_start(split_pct: float = 20):
    """
    Start an A/B test: route `split_pct`% of non-cached traffic to the challenger.
    Users in the challenger arm see the challenger's price — unlike shadow, which
    keeps the challenger hidden.

    Ground truth (actual prices from InsideAirbnb) accumulates over days via
    scripts/fetch_ground_truth.py. Use GET /ab/stats to see per-arm MAE.

    split_pct must be 1–50 (capped to limit user exposure while testing).
    Requires challenger.onnx to exist (run retrain.py first).
    """
    if not shadow.active:
        raise HTTPException(
            status_code=400,
            detail="Challenger not loaded — challenger.onnx missing or shadow not active. Run retrain.py first.",
        )
    result = ab_test.start(split_pct=split_pct)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/ab/stats")
def ab_stats():
    """
    Per-arm accuracy comparison for the current (or most recent) A/B test.

    Uses ground truth JOIN: ab_predictions ⟕ ground_truth on listing_id.
    Ground truth only matches predictions that included listing_id and whose
    listing has since been scraped by scripts/fetch_ground_truth.py.

    Returns MAE per arm + recommendation: promote | monitor | reject | insufficient_data.
    """
    return ab_test.stats()


@app.post("/ab/stop")
def ab_stop():
    """Stop the A/B test. Champion serves 100% of traffic. Challenger stays on disk."""
    result = ab_test.stop()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Not active"))
    return result


@app.post("/ab/promote")
def ab_promote():
    """
    Stop the A/B test and move to canary deployment.
    Call POST /canary/start after this to begin the gradual rollout.
    """
    return ab_test.promote()


@app.post("/ab/reject")
def ab_reject():
    """
    Stop the A/B test without promoting the challenger.
    Use POST /shadow/reject afterwards to delete challenger.onnx.
    """
    return ab_test.reject()


# ── Canary deployment ──────────────────────────────────────────────────────────

@app.post("/canary/start")
def canary_start():
    """
    Begin canary deployment: 1% of traffic → challenger.

    A background health monitor checks error rate and p99 latency every 15s.
    If healthy for 60s with 20+ samples, it auto-advances: 1% → 5% → 25% → 100%.
    If unhealthy at any stage, it auto-rolls-back to champion.

    At 100%, challenger.onnx is promoted to nyc_xgb_model.onnx automatically.
    Restart the API ('docker compose restart api') to serve the promoted model.
    """
    result = canary.start()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/canary/status")
def canary_status():
    """
    Current canary stage, health metrics, and event history.
    Shows error rate and p99 latency for challenger at the current stage.
    """
    return canary.status()


@app.post("/canary/advance")
def canary_advance():
    """
    Manually advance to the next canary stage, skipping the health-wait.
    At 100%, promotes challenger to champion and disables canary.
    """
    if not canary.active:
        raise HTTPException(status_code=400, detail="Canary not active. POST /canary/start first.")
    return canary.advance(manual=True)


@app.post("/canary/rollback")
def canary_rollback():
    """
    Immediately roll back to champion (0% canary traffic).
    challenger.onnx is preserved on disk for inspection.
    """
    if not canary.active:
        raise HTTPException(status_code=400, detail="Canary not active.")
    return canary.rollback(reason="manual")


@app.post("/canary/abort")
def canary_abort():
    """Stop the canary without rolling back or promoting. Equivalent to rollback."""
    return canary.abort()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.new_york_workflow.nyc_api:app", host="0.0.0.0", port=8001, reload=False, log_config=None)
