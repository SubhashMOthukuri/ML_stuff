# NYC Airbnb Price Prediction

Production ML service that predicts nightly Airbnb prices in NYC. ONNX Runtime inference, Redis caching, shadow/A/B/canary model rollout, automated drift detection and retraining.

See [`newyork_benchmark.md`](newyork_benchmark.md) for model performance, and `Desktop/files/` for the full engineering protocol (architecture, tradeoffs, code patterns).

## Quick Start

```bash
# Local dev (no Docker)
pip install -r requirements.txt
cp .env.example .env
uvicorn src.new_york_workflow.nyc_api:app --reload --port 8001

# Full stack (Redis + MLflow + API + Nginx)
docker compose up --build
```

API: `http://localhost:8001` (direct) or `https://localhost` (via Nginx).
Frontend: `cd frontend && npm install && npm run dev`.

## Architecture

```
React UI ─▶ Nginx (SSL, rate limit) ─▶ FastAPI (gunicorn, 4 workers)
                                              │
                          ┌───────────────────┼────────────────────┐
                          ▼                   ▼                    ▼
                    Redis cache         ONNX Runtime          SQLite stores
                    (predictions)       (champion model)      (requests, A/B,
                                              │                 ground truth, DLQ)
                                              ▼
                                  shadow / A/B / canary
                                  challenger evaluation
```

Nightly GitHub Actions job (`.github/workflows/nightly.yml`) checks production
drift (PSI) and triggers `scripts/retrain.py` on critical drift — trains a
challenger, validates against gates, exports ONNX, and registers it in MLflow
under the `challenger` alias for promotion via shadow → A/B → canary.

## Key Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Single prediction (cache → ONNX inference) |
| `POST /predict-batch` | Batch predictions |
| `GET /health`, `/health/ready`, `/health/live` | Liveness/readiness probes |
| `GET /model-info`, `/drift`, `/cache/stats`, `/dlq` | Observability |
| `POST /shadow/promote`, `/shadow/reject` | Shadow deployment decisions |
| `POST /ab/start`, `/ab/promote`, `/ab/reject` | A/B test lifecycle |
| `POST /canary/start`, `/canary/advance`, `/canary/rollback` | Canary rollout |
| `GET /ground-truth/stats` | Production MAE vs training baseline |
| `POST /ground-truth/ingest` | Trigger fetch_ground_truth.py remotely (auth: `X-Ingest-Token`) — called monthly by `.github/workflows/ground-truth.yml` |

Full route list in [`src/new_york_workflow/nyc_api.py`](src/new_york_workflow/nyc_api.py).

## Repo Layout

```
src/new_york_workflow/   # training pipeline (1_data_cleaning → 4_review_sentiment)
                          # + serving layer (nyc_api, nyc_predictor_onnx, nyc_cache,
                          #   nyc_shadow, nyc_ab, nyc_canary, nyc_ground_truth, nyc_drift, nyc_dlq)
scripts/                  # retrain.py, fetch_ground_truth.py, export_onnx.py, load_test.py
tests/                    # pytest suite
models/nyc/                # champion model (.onnx), scaler, baseline stats
frontend/                  # React/Vite prediction UI
Desktop/files/              # engineering protocol — read before changing code
```

## Environment Variables

See [`.env.example`](.env.example) for the full list (Redis, MLflow, rate
limiting, cache TTL). `docker-compose.yml` sets these for the containerized
stack automatically.

## Testing

```bash
pytest tests/
python scripts/load_test.py        # latency/throughput under load
```

## Retraining & Ground Truth

```bash
python scripts/retrain.py                       # full pipeline: download, clean, train, gate, export
python scripts/retrain.py --no-download          # use existing local data
python scripts/retrain.py --rollback <version>   # roll champion back to a prior MLflow version
python scripts/fetch_ground_truth.py             # manual run: join predictions against actual prices
```

In production, ground truth ingestion runs automatically — `.github/workflows/ground-truth.yml`
calls `POST /ground-truth/ingest` on the live API monthly, so it has access to the
real `predictions.db` without SSH or file copying. Set `GROUND_TRUTH_INGEST_TOKEN`
in `.env` (server) and as a GitHub Actions secret of the same name, plus `PROD_API_URL`.
