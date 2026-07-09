# Architecture — NYC Airbnb Price Prediction System

This document describes what we built, why each piece exists, and how they connect.
Read this before changing any core module — it prevents you from breaking implicit contracts
between components.

---

## What This System Does

A user submits a listing's attributes (bedrooms, borough, amenities, reviews).
The system returns a predicted nightly price in USD with an optional SHAP explanation
of which features drove the prediction.

Behind that simple interaction is a full ML lifecycle: training pipeline, ONNX serving,
Redis cache, A/B testing, drift detection, nightly retraining, and Kubernetes deployment.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER TRAFFIC                               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Nginx (infra/nginx) │  SSL termination
                    │   Port 443 / 80       │  per-IP rate limit
                    │   /api/* → API        │  static frontend
                    └───────────┬──────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   FastAPI (src/new_york_workflow/   │
              │   nyc_api.py) — gunicorn 2 workers  │
              │   Port 8001                          │
              └──────┬──────────┬───────────────────┘
                     │          │
          ┌──────────▼──┐  ┌────▼───────────────────────────────────┐
          │ Redis 7      │  │ ONNX Runtime (nyc_predictor_onnx.py)  │
          │ Cache + DLQ  │  │ XGBoost model, SHAP TreeExplainer      │
          │ Batch queue  │  │ 58 features, log_price → expm1 → $$$   │
          └──────────────┘  └────────────────────────────────────────┘
                     │
          ┌──────────▼──────────────────────┐
          │ Postgres / SQLite               │
          │ predictions.db — request log    │
          │ shadow_comparisons.db — A/B     │
          │ canary_health.db — canary state  │
          └─────────────────────────────────┘
```

---

## Directory Guide

### `src/new_york_workflow/`

The core of the system. Two distinct concerns live here:

**Serving layer** (imported by the API at runtime):
| File | What it does |
|------|-------------|
| `nyc_api.py` | All FastAPI routes. /v1/predict, /v1/predict-batch, /health, /drift, /metrics, rollout endpoints |
| `nyc_predictor_onnx.py` | ONNX Runtime inference + SHAP explanations. Holds the model in memory. |
| `nyc_cache.py` | Redis prediction cache. MD5 key from input, 5-min TTL. ~40% hit rate in prod. |
| `nyc_store.py` | Append-only SQLite/Postgres request log. Feeds retraining pipeline. |
| `nyc_batch.py` | Async batch jobs. Redis BRPOP queue, daemon worker thread, per-job TTL 1h. |
| `nyc_drift.py` | PSI-based drift detection vs baseline_stats.json. Called by `/drift` and nightly CI. |
| `nyc_shadow.py` | Shadow deployment. Runs challenger alongside champion, logs comparison. |
| `nyc_ab.py` | A/B test lifecycle. Traffic split, winner selection, promotion. |
| `nyc_canary.py` | Canary rollout. Gradual traffic shift with health-gate rollback. |
| `nyc_ground_truth.py` | Join predictions vs actual InsideAirbnb prices. Live MAE tracking. |
| `nyc_dlq.py` | Dead letter queue. 5xx failures pushed to Redis list for retry/inspection. |
| `nyc_alerts.py` | Alert store + Slack webhook delivery. warning/critical severity. |
| `nyc_data_validator.py` | Schema + distribution checks run before each retrain. |

**Training pipeline** (`pipeline/` subfolder, run by `scripts/retrain.py`):
| File | What it does |
|------|-------------|
| `pipeline/1_data_cleaning.py` | Raw InsideAirbnb CSV → clean DataFrame. Drops nulls, parses price, encodes categoricals. |
| `pipeline/2_feature_engineering.py` | Log transforms, borough dummies, polynomials, interaction terms, target encoding. |
| `pipeline/3_train_model.py` | Ridge / RF / XGBoost benchmark + hyperparameter search. Saves models + training report. |

### `src/` (shared utilities)

| File | What it does |
|------|-------------|
| `config.py` | Pydantic Settings — reads all env vars, provides typed defaults. Single source of truth. |
| `metrics.py` | Prometheus counters/histograms. Imported by nyc_api.py. |
| `logger_config.py` | structlog setup (JSON format in prod, pretty in dev). |
| `exceptions.py` | Custom exception types shared across modules. |
| `utils/retry.py` | Exponential backoff retry decorator. |
| `utils/timing.py` | Context-manager timing utility. |

### `scripts/` — operational scripts

| File | What it does |
|------|-------------|
| `retrain.py` | Full retrain pipeline. Downloads data, runs pipeline/, validates, exports ONNX, MLflow register. |
| `export_onnx.py` | One-off: convert existing .pkl model to ONNX without retraining. |
| `fetch_ground_truth.py` | Joins predictions vs actual InsideAirbnb prices, writes to ground_truth DB. |
| `register_baseline.py` | One-time setup: register initial champion model in MLflow as v1. |
| `drift_check.py` | CLI wrapper for DriftMonitor. Used by Kubernetes CronJob. |
| `create_test_model.py` | Creates synthetic ONNX model for CI when S3 is unavailable. |
| `eks_step_upgrade.sh` | One-minor-version EKS upgrade step. Called by eks-upgrade.yml workflow. |

### `tests/`

| Path | What it tests |
|------|-------------|
| `test_nyc_api.py` | Full API contract tests (131 tests). Auth, rate limit, cache, batch, SHAP, rollout endpoints. |
| `test_ab_canary.py` | A/B test and canary state machine correctness. |
| `test_data_validator.py` | Data quality check functions. |
| `test_ground_truth.py` | Ground truth join + MAE calculation. |
| `test_store_dsn_masking.py` | DSN credentials never appear in logs. |
| `test_train_serve_skew.py` | Feature list matches between training pipeline and predictor. |
| `load/smoke.js` | k6 load test. Runs on push to main via load-test.yml. |
| `load/load_test.py` | Python load test for local latency profiling. |

### `infra/`

| Path | What it is |
|------|-----------|
| `aws/` | Terraform: EKS cluster, ECR repos, RDS Postgres, S3 bucket, Secrets Manager, VPC |
| `oracle/` | Terraform: Oracle Cloud (OCI) compute — legacy, not used in primary deployment |
| `helm/nyc-airbnb/` | Helm chart for Kubernetes deployment. Values, templates for all K8s resources. |
| `k8s/` | Standalone Kubernetes manifests: ArgoCD app, ExternalSecret, ClusterSecretStore |
| `nginx/` | Nginx config (SSL termination, routing, rate limiting) + self-signed dev certs |
| `monitoring/` | Observability config: Grafana dashboards, Prometheus scrape config, Fluent Bit pipelines |

### `docs/`

| File | What it is |
|------|-----------|
| `ARCHITECTURE.md` | This file. System overview, component map, directory guide. |
| `benchmark.md` | Full model evaluation: dataset stats, model comparison, feature engineering decisions. |
| `fix.md` | Historical: all 22 problems found and how they were solved, start to finish. |

### `frontend/`

React/Vite SPA. Components:

| Component | What it renders |
|-----------|----------------|
| `PredictPage.jsx` | Main prediction form. Calls /v1/predict, shows result + SHAP chart. |
| `ShapChart.jsx` | Horizontal bar chart of SHAP feature attributions. |
| `DriftPage.jsx` | Calls /drift, shows PSI per feature and prediction shift. |
| `ShadowPage.jsx` | Shadow/A/B/canary status and controls. |
| `OpsPage.jsx` | Cache stats, DLQ contents, ground truth MAE. |
| `AlertsPage.jsx` | Active alerts with severity and timestamps. |

Nginx config at `frontend/nginx.conf` proxies `/api/*` → API service with `X-API-Key` injected
from Kubernetes secret — browser never sees the API key.

---

## Key Design Decisions

### Why ONNX instead of native XGBoost?

ONNX Runtime is 31× faster than native XGBoost inference and releases the Python GIL,
so multiple Gunicorn workers run inference truly in parallel. The tradeoff is that
model export adds a step to the retraining pipeline (`scripts/export_onnx.py`).

### Why Redis cache?

~40% of production requests are repeat queries (same listing, same parameters).
MD5-keyed cache with 5-min TTL eliminates ONNX inference for those. Redis also
serves the async batch queue — jobs are enqueued and claimed via BRPOP (atomic,
only one worker claims each job). AOF persistence ensures jobs survive Redis restarts.

### Why log_price as target?

Raw price is right-skewed (skew=14.21). Log-transforming makes predictions
multiplicative — which is how hosts price ("20% premium for Manhattan"). The API
applies `np.expm1()` to convert back to dollars before returning.

### Why minimum_nights_avg_ntm fallback = 25.0?

NYC's 30-day minimum-nights law skews the training distribution of this feature
toward 25+ nights. Using the raw `minimum_nights` input (e.g. 7) instead of
the training mean causes predictions to be inflated by 30–40%. The API uses the
training mean as a safe fallback when this field is not provided.

### Why shadow → A/B → canary rollout?

Each stage answers a different question:
- **Shadow**: does the new model produce sane outputs? (no user impact)
- **A/B**: is the new model statistically better on live traffic? (controlled split)
- **Canary**: does the new model stay healthy at scale? (gradual traffic increase)

Skipping stages risks a bad model reaching 100% of users before problems are detected.

### Why PSI for drift detection?

Population Stability Index measures how much the distribution of a feature has
shifted between training and production. PSI < 0.1 = stable, 0.1–0.2 = warning,
> 0.2 = critical. It's model-agnostic, interpretable, and fast to compute.
Threshold-based alerts (Slack) fire automatically when any feature exceeds 0.2.

### Why not Kafka for batch jobs?

The batch queue has one producer type and one consumer type, jobs have a 1-hour TTL,
and the expected backlog is hundreds of jobs, not millions. Redis BRPOP with AOF
persistence handles this correctly at orders of magnitude lower cost and complexity.
Kafka would add a multi-broker cluster and schema registry for no practical gain here.

---

## Critical Invariants (do not break)

1. **`app.include_router(v1_router)` must be the LAST statement in nyc_api.py** (before `if __name__`). FastAPI copies routes at include time — any `@v1_router.*` decorator after the `include_router` call is silently ignored, returning 404.

2. **`neighbourhood_price_rank` encoding is fitted on training data only.** Re-fitting on all data (train + test) leaks test distribution into the features and inflates R² artificially.

3. **`minimum_nights_avg_ntm` fallback = 25.0, not `minimum_nights`.** Using the user's raw minimum_nights input here causes prediction inflation for short-stay listings due to NYC law skewing the training distribution.

4. **Redis `allkeys-lru` + batch jobs**: LRU eviction can remove batch jobs under memory pressure. If this becomes a problem in production, move batch job keys to a separate Redis instance with `noeviction` policy. Do not disable `allkeys-lru` on the main instance — the prediction cache depends on it.

5. **SHAP pre-warm runs at startup** (`lifespan()` in nyc_api.py). If removed, the first `/v1/predict?explain=true` request will time out (TreeExplainer init takes 2–4s on t3.small).
