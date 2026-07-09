# NYC Airbnb Price Prediction

Production ML service that predicts nightly Airbnb prices for NYC listings.
XGBoost model (R²=0.82, MAE=$58) served via ONNX Runtime, with SHAP explanations,
Redis caching, shadow/A/B/canary rollout, drift detection, and automated retraining.

See [`newyork_benchmark.md`](newyork_benchmark.md) for full model evaluation and
[`MODEL_CARD.md`](MODEL_CARD.md) for model governance details.

---

## Architecture

```
React UI ──► Nginx (SSL, rate limit, /api proxy)
                      │
                      ▼
           FastAPI (gunicorn, 2 workers)
                      │
       ┌──────────────┼──────────────────┐
       ▼              ▼                  ▼
  Redis cache    ONNX Runtime       Postgres / SQLite
  (MD5-keyed,   (champion model)   (predictions, A/B results,
   5-min TTL)         │              ground truth, DLQ)
                      ▼
           shadow / A/B / canary
           challenger evaluation
```

**Observability stack:** structlog → Fluent Bit → Datadog (logs) ·
Prometheus + Grafana (metrics) · Tempo + Loki (distributed tracing) ·
Slack webhooks (drift/alert delivery)

---

## Quick Start

### Local (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env          # edit REDIS_HOST, MLFLOW_TRACKING_URI as needed
uvicorn src.new_york_workflow.nyc_api:app --reload --port 8001
```

### Full stack (Redis + MLflow + API + Nginx)

```bash
docker compose up --build
```

- API: `http://localhost:8001`
- UI via Nginx: `https://localhost`
- MLflow: `http://localhost:5000`
- Frontend dev server: `cd frontend && npm install && npm run dev`

---

## API Reference

All prediction endpoints are versioned under `/v1`. Health and observability
endpoints are at root (used by Kubernetes probes and internal tooling).

### Prediction

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/predict` | Single prediction. Pass `?explain=true` for SHAP feature attributions. |
| `POST` | `/v1/predict-batch` | Async batch (up to 50 listings). Returns `job_id` immediately (202). |
| `GET`  | `/v1/predict-batch/{job_id}` | Poll batch status and partial results. |
| `GET`  | `/v1/model-info` | Champion model version, R², MAE, feature count. |

**Auth:** Set `VALID_API_KEYS` env var to enable `X-API-Key` header auth.
Unset = auth disabled (useful for local dev).

**Rate limit:** 100 requests/min per IP on `/v1/predict` (configurable via `RATE_LIMIT`).

Example single prediction:

```bash
curl -X POST http://localhost:8001/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "accommodates": 2,
    "bedrooms": 1,
    "room_type": "Entire home/apt",
    "neighbourhood": "Williamsburg",
    "borough": "Brooklyn",
    "review_scores_rating": 4.8,
    "amenity_count": 25
  }'
```

### Health & Ops

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Combined liveness + readiness. |
| `GET` | `/health/live` | Liveness probe (K8s). |
| `GET` | `/health/ready` | Readiness probe (K8s). |
| `GET` | `/metrics` | Prometheus scrape endpoint. |
| `GET` | `/drift` | Run drift check. `?window_hours=168&push_alert=true` for weekly with Slack. |
| `GET` | `/cache/stats` | Redis cache hit rate and size. |
| `GET` | `/dlq` | Dead letter queue contents (failed predictions). |
| `GET` | `/ground-truth/stats` | Live production MAE vs training baseline. |
| `POST` | `/ground-truth/ingest` | Trigger ground truth join (auth: `X-Ingest-Token`). |

### Model Rollout

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/shadow/promote` | Promote challenger from shadow to A/B. |
| `POST` | `/shadow/reject` | Reject challenger, keep champion. |
| `POST` | `/ab/start` | Start A/B test with configured traffic split. |
| `POST` | `/ab/promote` | Promote A/B winner to champion. |
| `POST` | `/ab/reject` | End A/B test without promotion. |
| `POST` | `/canary/start` | Begin canary rollout (e.g. 5% traffic). |
| `POST` | `/canary/advance` | Increase canary traffic percentage. |
| `POST` | `/canary/rollback` | Roll canary back to champion. |

---

## Repo Layout

```
src/new_york_workflow/
  1_data_cleaning.py         # InsideAirbnb CSV → cleaned DataFrame
  2_feature_engineering.py   # feature construction (58 features)
  3_train_model.py           # Ridge / RF / XGBoost benchmark + hyperparameter search
  nyc_api.py                 # FastAPI serving layer (all endpoints)
  nyc_predictor_onnx.py      # ONNX inference + SHAP explanations
  nyc_cache.py               # Redis prediction cache
  nyc_store.py               # SQLite request store (append-only)
  nyc_batch.py               # async batch job queue (Redis BRPOP)
  nyc_shadow.py              # shadow deployment logic
  nyc_ab.py                  # A/B test lifecycle
  nyc_canary.py              # canary rollout
  nyc_drift.py               # PSI-based drift detection
  nyc_dlq.py                 # dead letter queue
  nyc_alerts.py              # alert store + Slack delivery
  nyc_ground_truth.py        # ground truth store + MAE tracking
  nyc_data_validator.py      # data quality checks for retrain pipeline

scripts/
  retrain.py                 # full retrain pipeline: download → clean → train → gate → ONNX
  export_onnx.py             # export existing PKL model to ONNX
  fetch_ground_truth.py      # join predictions against actual InsideAirbnb prices
  register_baseline.py       # register initial champion in MLflow
  create_test_model.py       # generate synthetic model for CI (when S3 unavailable)
  load_test.py               # latency/throughput benchmark
  drift_check.py             # standalone CLI for drift check (used by K8s CronJob)
  eks_step_upgrade.sh        # EKS one-minor-version upgrade step (used by eks-upgrade workflow)

.github/workflows/
  deploy.yml                 # build → push ECR → helm upgrade on push to main
  nightly.yml                # nightly drift check → conditional retrain
  ground-truth.yml           # monthly: call POST /ground-truth/ingest on prod
  load-test.yml              # k6 smoke test, latency regression gate
  eks-upgrade.yml            # manual EKS 1.31 → 1.35 upgrade (dry-run by default)

infra/
  helm/nyc-airbnb/           # Helm chart (deployment, service, HPA, PDB, NetworkPolicy, etc.)
  aws/                       # Terraform: EKS cluster, ECR, S3, Secrets Manager

models/nyc/                  # gitignored — synced from S3 at deploy time
  nyc_xgb_model.onnx         # champion model (ONNX)
  nyc_scaler.pkl             # StandardScaler fitted on training data
  nyc_feature_list.pkl       # ordered feature names for inference
  nyc_neighbourhood_means.pkl # per-neighbourhood mean log_price (target encoding)
  nyc_training_report.json   # metrics, best params, feature importances
  baseline_stats.pkl         # baseline feature distributions (PSI reference)

frontend/                    # React/Vite prediction UI
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host for cache + DLQ |
| `REDIS_PORT` | `6379` | Redis port |
| `CACHE_TTL_SECONDS` | `300` | Cache entry lifetime (seconds) |
| `PREDICTION_DB` | `data/predictions.db` | SQLite path or Postgres DSN |
| `DLQ_MAX_SIZE` | `10000` | Max dead letter queue entries |
| `RATE_LIMIT` | `60/minute` | Per-IP rate limit on `/v1/predict` |
| `LOG_LEVEL` | `info` | Log level (debug/info/warning/error) |
| `LOG_FORMAT` | `json` | Log format (json or text) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `OTLP_ENDPOINT` | *(unset)* | OpenTelemetry collector URL; unset = console |
| `VALID_API_KEYS` | *(unset)* | Comma-separated API keys; unset = auth disabled |
| `GROUND_TRUTH_INGEST_TOKEN` | *(unset)* | Token for POST /ground-truth/ingest |
| `SLACK_WEBHOOK_URL` | *(unset)* | Incoming webhook for drift/alert delivery |
| `SLACK_CRITICAL_CHANNEL` | *(unset)* | Override channel for critical alerts |
| `CORS_ALLOW_ORIGINS` | `*` | Allowed CORS origins |
| `OTEL_SERVICE_NAME` | `nyc-airbnb` | Service name in traces |

See [`.env.example`](.env.example) for the complete list.

---

## Testing

```bash
pytest tests/                          # full suite (131 tests)
python scripts/load_test.py            # latency / throughput benchmark
```

CI runs the test suite on every push. When `models/nyc/` is absent (no S3 access),
`tests/conftest.py` injects a stub predictor so all API contract tests still run.

---

## Production Deployment

### Prerequisites

- AWS account with EKS cluster `nyc-airbnb` in `us-east-2`
- ECR repos: `nyc-airbnb` (API) and `nyc-airbnb-frontend`
- S3 bucket `nyc-airbnb-models-698172256228` with model artefacts at `nyc/`
- Secrets Manager secret `nyc-airbnb-secrets` with keys:
  `PREDICTION_DB`, `SHADOW_DB_URL`, `MLFLOW_TRACKING_URI`,
  `VALID_API_KEYS`, `SLACK_WEBHOOK_URL`

### Required GitHub Actions Secrets

| Secret | Used by | Description |
|--------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | deploy, nightly, eks-upgrade | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | deploy, nightly, eks-upgrade | AWS credentials |
| `GH_PAT` | eks-upgrade | GitHub PAT for pushing Terraform version bump |
| `MLFLOW_TRACKING_URI` | nightly retrain | MLflow tracking server URL |
| `PROD_API_URL` | nightly, ground-truth | Live API URL (e.g. `http://abc.us-east-2.elb.amazonaws.com:8001`) |
| `PROD_API_KEY` | nightly, ground-truth | Valid API key for X-API-Key auth |
| `GROUND_TRUTH_INGEST_TOKEN` | ground-truth | Shared secret for POST /ground-truth/ingest |

### Deploy

Push to `main` — the deploy workflow builds Docker images, pushes to ECR, and
runs `helm upgrade` automatically. To deploy manually:

```bash
helm upgrade nyc-airbnb infra/helm/nyc-airbnb/ \
  --install \
  --namespace default \
  --values infra/helm/nyc-airbnb/values.yaml
```

### EKS Upgrade (1.31 → 1.35)

AWS extended support for K8s 1.31 ends 2026-11-26. Upgrade path:

1. Go to GitHub Actions → **EKS Upgrade 1.31 → 1.35**
2. Run with `dry_run=true` first to verify
3. Re-run with `dry_run=false` to apply

The workflow steps through 1.31 → 1.32 → 1.33 → 1.34 → 1.35 automatically
(AWS requires one minor version at a time). Each step upgrades the control
plane, then EKS add-ons, then the managed node group.

---

## Retraining & Model Management

### Automatic (nightly)

The nightly workflow:
1. Calls `GET /drift?window_hours=168&push_alert=true` on the prod API
2. If drift is `critical`, triggers the retrain job
3. Retrain: downloads latest InsideAirbnb snapshot → cleans → trains → validates
   gate (R² must exceed baseline) → exports ONNX → pushes to S3

### Manual retraining

```bash
python scripts/retrain.py                      # full pipeline
python scripts/retrain.py --no-download        # use existing local data
python scripts/retrain.py --force              # skip gate, always promote
python scripts/retrain.py --rollback <version> # roll champion back to a prior MLflow run
```

### Model rollout workflow

New model enters as **challenger**. Promotion path:

```
challenger → shadow (100% traffic, compare silently)
           → A/B   (configurable traffic split, compare on live metrics)
           → canary (gradual rollout: 5% → 25% → 50% → 100%)
           → champion
```

Use the `/shadow/*`, `/ab/*`, and `/canary/*` endpoints to advance or roll back.

### Ground truth tracking

```bash
python scripts/fetch_ground_truth.py           # join predictions vs actual prices
```

Runs monthly in production via `ground-truth.yml` (calls `POST /ground-truth/ingest`
on the live API, so it has access to `predictions.db` without SSH access).
Check live MAE vs training baseline at `GET /ground-truth/stats`.

---

## Monitoring

| Signal | Where |
|--------|-------|
| Structured logs (JSON) | Datadog — service `nyc-airbnb`, env `prod` |
| Request metrics | Grafana → Prometheus (`/metrics` scrape) |
| Distributed traces | Tempo + Loki |
| Drift alerts | Slack `#incidents` (critical) or default channel (warning) |
| Retrain events | GitHub Issues (label `drift`) + Slack |

Prometheus metrics include: prediction latency histograms, cache hit/miss counters,
canary traffic split, active alert count, DLQ size.

---

## Security

- API pod runs as non-root (UID 1000), read-only root filesystem, all Linux capabilities dropped
- NetworkPolicy: ingress on 8001 only; egress to Redis (6379), DNS (53), AWS (443) only
- Secrets injected from AWS Secrets Manager via External Secrets Operator (not in env files)
- `terraform.tfvars` is gitignored — never commit it
- Frontend nginx runs as root (port 80 binding) — switching to `nginxinc/nginx-unprivileged` is a tracked TODO

---

## Known Limitations

See [`MODEL_CARD.md`](MODEL_CARD.md) for model-level limitations.

- **Luxury listings excluded:** Listings above the 99th price percentile (~$1,562/night) are excluded from training. The model is unreliable for ultra-luxury properties.
- **No booking demand data:** Occupancy rate and seasonal demand are not available from InsideAirbnb — including them would improve R² by an estimated +3–4%.
- **NYC only:** Trained exclusively on New York City listings. Not transferable to other cities without retraining.
- **Frontend security context:** nginx image requires root to bind port 80. Switching to `nginxinc/nginx-unprivileged` (port 8080) is a security team item.
- **IRSA not configured:** External Secrets Operator uses static IAM keys. Migrating to IRSA (IAM Roles for Service Accounts) is the correct long-term approach.
