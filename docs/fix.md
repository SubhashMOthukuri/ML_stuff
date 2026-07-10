# Fix Log — NYC Airbnb Price Prediction API

All problems encountered and how they were resolved, in the order they were found.

---

## 1. SHAP: 502 error on first explain request

**Problem:** First `POST /predict?explain=true` returned 502. SHAP `TreeExplainer` initialization takes 2–4 seconds on a t3.small — the request timed out before it finished.

**Fix:** Pre-warm the explainer in a daemon thread at startup so it's ready before the first request arrives.

```python
# nyc_api.py — lifespan()
from threading import Thread
def _prewarm_shap():
    try:
        predictor._get_shap_explainer()
    except Exception as exc:
        logger.warning("SHAP pre-warm failed: %s", exc)
Thread(target=_prewarm_shap, daemon=True).start()
```

---

## 2. SHAP: `is_private_bath` appearing in the explanation chart

**Problem:** `is_private_bath` showed up in the SHAP top-features chart in the UI. Users don't control this field at booking time — it's a host-side property and should not be surfaced.

**Fix:** Added a `_HIDDEN` set in `explain_raw()` that filters out all internal/non-user-controllable features before returning SHAP values.

```python
# nyc_predictor_onnx.py
_HIDDEN = {
    "is_private_bath", "minimum_nights_avg_ntm", "log_minimum_nights_avg_ntm",
    "minimum_minimum_nights", "days_to_checkin", "is_peak_season",
    "is_weekend", "month", "day_of_week",
}
for feat, sv in zip(self.feature_list, shap_values):
    if feat in _HIDDEN:
        continue
```

---

## 3. Prediction inflated for 7-night minimum listings

**Problem:** Listings with a 7-night minimum were predicted 30–40% higher than expected. The fallback for `minimum_nights_avg_ntm` was using `raw["minimum_nights"]` (the user's input, e.g. 7). The model was trained with a distribution centered around 25 nights (NYC's 30-day minimum law skews the training data). Passing 7 instead of the training mean caused the feature to pull the prediction in the wrong direction.

**Fix:** Changed the fallback to the training mean of 25.0 instead of the user's raw input.

```python
# nyc_predictor_onnx.py
mna = raw.get("minimum_nights_avg_ntm") or 25.0  # training mean, not raw minimum_nights
```

---

## 4. UI: Error message and old result showing at the same time

**Problem:** When a user submitted a new prediction after a previous successful one, if the new request errored, the old result stayed visible alongside the new error — confusing double state.

**Fix:** Reset result to null at the start of every submission.

```jsx
// PredictPage.jsx — handleSubmit()
setResult(null)  // clear stale result before new request
setError(null)
```

---

## 5. UI: Rating slider showed "No reviews yet" instead of "Unrated"

**Problem:** The rating slider label said "No reviews yet" at zero which was too long and didn't match the field semantics.

**Fix:** Changed the label format.

```jsx
// ListingForm.jsx
v === 0 ? 'Unrated' : `${v.toFixed(1)} ★`
```

---

## 6. CI: Model files missing from S3 — all tests errored

**Problem:** CI pipeline failed with `ONNXRuntime NoSuchFile` on every test. The S3 bucket `nyc-airbnb-models-698172256228` existed but only had a `training-data/` prefix — the `nyc/` model prefix had never been uploaded.

**Fix:** Uploaded model files manually, then CI's `aws s3 sync` step worked.

```bash
aws s3 sync models/nyc/ s3://nyc-airbnb-models-698172256228/nyc/
```

---

## 7. CI: `tests/conftest.py` change didn't trigger deploy workflow

**Problem:** After fixing conftest.py, CI did not re-run because `tests/` is not in the deploy workflow's path filter (`src/**`, `frontend/**`, `requirements*.txt`, `infra/helm/**`).

**Fix:** Touched `requirements.txt` to trigger the workflow, and added a stub predictor to conftest so CI can run all tests even when model binaries are absent.

```python
# conftest.py — stub used in CI when models/nyc/nyc_xgb_model.onnx is missing
def _make_stub_predictor() -> MagicMock:
    stub = MagicMock()
    stub.model_info.return_value = {"n_features": 67, "r2_test": 0.80, ...}
    stub.predict_raw.return_value = {"price_usd": 150.0, ...}
    stub.explain_raw.return_value = []
    return stub
```

---

## 8. CI: `test_n_features_is_58` failing

**Problem:** Model was retrained with 67 features but the test still asserted 58.

**Fix:** Updated the assertion.

```python
# test_nyc_api.py
assert client.get("/v1/model-info").json()["n_features"] == 67
```

---

## 9. Async batch endpoint — sync design blocking workers

**Problem:** Original `/predict-batch` was synchronous — it blocked the entire HTTP connection while processing all listings. With 50 listings this could take 30+ seconds and tie up a Gunicorn worker.

**Fix:** Replaced with a Redis-backed async design. POST enqueues immediately and returns a job ID (202). A daemon thread per worker uses `BRPOP` (atomic, only one worker claims each job) to process jobs. Client polls `GET /predict-batch/{job_id}` for results. Partial results are written after every listing so polling shows live progress.

```
POST /v1/predict-batch  → enqueue → 202 + job_id  (instant)
GET  /v1/predict-batch/{job_id}  → poll status + results
```

Key properties: `BRPOP` is atomic so only one Gunicorn worker claims each job. Per-listing error isolation — one bad listing doesn't fail the whole batch. Results expire in Redis after 1 hour (TTL).

---

## 10. deploy.yml: sed replacing both image tags with backend SHA

**Problem:** The CI step that updates `values.yaml` used:
```bash
sed -i "s|tag: .*|tag: $TAG|g" infra/helm/nyc-airbnb/values.yaml
```
This matched ALL lines containing `tag:` — including both the API image tag (2-space indent) and the frontend image tag (4-space indent). Both got overwritten with the backend commit SHA.

**Fix:** Use indentation-specific sed patterns to target each tag independently.

```bash
sed -i "s|^  tag: .*|  tag: $API_TAG|" infra/helm/nyc-airbnb/values.yaml
sed -i "s|^    tag: .*|    tag: $FE_TAG|" infra/helm/nyc-airbnb/values.yaml
```

---

## 11. `instant_bookable` field silently dropped

**Problem:** `instant_bookable` was accepted by the API but not included in `ListingFeatures`, so it was silently ignored and never passed to the model.

**Fix:** Added the field with a default of `False`.

```python
# nyc_api.py — ListingFeatures
instant_bookable: bool = False
```

---

## 12. Kubernetes: no securityContext on API pod

**Problem:** The API pod ran as root with no privilege restrictions — any container escape would give root access to the node.

**Fix:** Added pod and container securityContext to `deployment.yaml`.

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000      # matches appuser created in Dockerfile
    fsGroup: 1000
  containers:
    - name: api
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: data
          mountPath: /app/data
  volumes:
    - name: tmp
      emptyDir: {}
    - name: data
      emptyDir: {}
```

Also added `terminationGracePeriodSeconds: 90` so batch workers have time to drain on shutdown.

---

## 13. Kubernetes: no NetworkPolicy

**Problem:** No network policy existed — any pod in the cluster could talk to the API pod on any port.

**Fix:** Created `networkpolicy.yaml` restricting ingress to port 8001 and egress to Redis (6379), DNS (53), and AWS APIs (443) only.

---

## 14. API: no version prefix on routes

**Problem:** All routes were at root (`/predict`, `/predict-batch`, etc.). No ability to introduce a v2 contract without breaking existing callers.

**Fix:** Added `APIRouter(prefix="/v1")` for the user-facing prediction endpoints. Ops/infra endpoints (`/health`, `/metrics`, `/dlq`, `/drift`, etc.) stay at root since Kubernetes probes and internal tooling depend on them.

```python
# nyc_api.py
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/predict", ...)
@v1_router.post("/predict-batch", ...)
@v1_router.get("/predict-batch/{job_id}", ...)
@v1_router.get("/model-info", ...)

app.include_router(v1_router)  # must be after all route decorators
```

Frontend updated to call `/api/v1/predict` and `/api/v1/model-info`. All 131 tests updated and passing.

---

## 15. Kubernetes: PDB blocked node drains permanently

**Problem:** `pdb.yaml` had `minAvailable: 1` but `hpa.minReplicas` was also 1. During a node drain, K8s cannot evict the pod because that would bring available pods below 1. Node maintenance hangs forever.

**Fix:** Changed to `maxUnavailable: 1` which allows drains to proceed. To restore full disruption protection, raise `hpa.minReplicas` to 2 — then `minAvailable: 1` becomes safe again.

---

## 16. Frontend: no liveness or readiness probes

**Problem:** The frontend Deployment had no health probes. Kubernetes had no way to detect a crashed nginx container — it stayed in the load balancer rotation and served errors silently.

**Fix:** Added `httpGet /` probes on port 80 to `frontend.yaml`. Also added an HPA (1–3 replicas, 70% CPU target) since the frontend had no autoscaling at all.

---

## 17. Gunicorn: no worker memory leak guard

**Problem:** `gunicorn.conf.py` had no `max_requests` setting. Long-running Gunicorn workers slowly accumulate memory from Python GC fragmentation and leaked objects. Over days of traffic, workers grow until the pod OOM-kills.

**Fix:** Added `max_requests = 1000` and `max_requests_jitter = 100`. Workers recycle after 1000 requests; jitter prevents all workers restarting simultaneously.

---

## 18. Nightly drift check: deprecated `::set-output` syntax

**Problem:** `nightly.yml` used `print(f'::set-output name=drift_status::...')` which GitHub deprecated and stopped supporting on newer runner images — output passing between jobs silently fails.

**Fix:** Replaced with the current `$GITHUB_OUTPUT` file append pattern.

```python
with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print(f'drift_status={report.status}', file=fh)
    print(f'drift_alerts={alerts_str}', file=fh)
```

---

## 19. Drift cronjob: no runaway job protection

**Problem:** `drift-cronjob.yaml` had no `activeDeadlineSeconds` — a hung drift job (e.g. waiting on a locked DB) would run forever and block all future scheduled runs (`concurrencyPolicy: Forbid`). Completed pod history also accumulated indefinitely.

**Fix:** Added `activeDeadlineSeconds: 600` (kills after 10 min) and `ttlSecondsAfterFinished: 86400` (cleans up completed pods after 24 h).

---

## 20. ServiceMonitor: hardcoded Prometheus label

**Problem:** `servicemonitor.yaml` had `release: kube-prometheus` hardcoded. The Prometheus operator only scrapes ServiceMonitors whose labels match its `serviceMonitorSelector`. If the cluster uses a different Prometheus release name, scraping silently never starts.

**Fix:** Made the label configurable via `values.prometheusRelease` with `kube-prometheus` as the default.

```yaml
labels:
  release: {{ .Values.prometheusRelease | default "kube-prometheus" }}
```

---

## 21. Drift monitoring in CI used empty SQLite — results were meaningless

**Problem:** The nightly drift-check job created an empty SQLite database in the CI runner and ran `DriftMonitor` against it. This always returned `insufficient_data` — the CI result told you nothing about actual production drift.

**Fix:** Replaced the local Python drift check with a direct call to the production API's `/drift?window_hours=168&push_alert=true` endpoint. The API runs against its own production Postgres (injected via Secrets Manager) and also fires Slack alerts automatically when drift is critical. CI parses the JSON response and passes `drift_status` to the retrain job.

Two new GitHub secrets are required:
- `PROD_API_URL` — the backend LoadBalancer hostname (e.g. `http://abc.us-east-2.elb.amazonaws.com:8001`)
- `PROD_API_KEY` — a valid API key (same value as `VALID_API_KEYS` in Secrets Manager)

If either secret is missing or the prod API is unreachable, the job logs a warning and skips gracefully — it does not trigger a false retrain.

---

## 22. EKS cluster on 1.31 — AWS support ends 2026-11-26

**Problem:** Terraform `infra/aws/main.tf` had `version = "1.31"`. AWS extended support for K8s 1.31 ends 2026-11-26. After that date the cluster becomes unsupported.

**Fix:** Created `.github/workflows/eks-upgrade.yml` and `scripts/eks_step_upgrade.sh`. AWS EKS requires upgrading one minor version at a time, so the workflow steps through: `1.31 → 1.32 → 1.33 → 1.34 → 1.35`. Each step:
1. Upgrades the control plane and waits for `ACTIVE`
2. Upgrades EKS add-ons (`vpc-cni`, `kube-proxy`, `coredns`) to the default version for that K8s version
3. Upgrades the managed node group and waits for `ACTIVE`

The workflow is **manual-trigger only** and **dry-run by default** — you must explicitly set `dry_run=false` to make real changes. After all four steps complete, it commits the Terraform version bump to `1.35` automatically.

**How to run:**
1. Go to GitHub Actions → "EKS Upgrade 1.31 → 1.35"
2. Click "Run workflow" — leave dry_run=`true` first to see the plan
3. If the plan looks correct, re-run with dry_run=`false`

---

## 23. `src/core/config.py` — `_ROOT` pointed to `src/`, not the project root

**Problem:** `_ROOT = Path(__file__).resolve().parents[1]` resolves to `src/`. Every path anchored to it (MLflow SQLite DB, alerts JSON) landed inside the package tree instead of the project root.

**Fix:** `parents[2]` — one level further up.

```python
_ROOT = Path(__file__).resolve().parents[2]   # ML_stuff/ project root
```

---

## 24. `src/core/config.py` — `alerts_file` default was a bare relative path

**Problem:** `Field("models/nyc/alerts.json", ...)` resolves relative to wherever the process is launched from. Running from a subdirectory silently wrote alerts to the wrong location.

**Fix:** Anchor to `_ROOT` (which was itself fixed in #23).

```python
alerts_file: Path = Field(_ROOT / "models/nyc/alerts.json", ...)
```

---

## 25. `src/core/config.py` — `log_format` accepted any string

**Problem:** `LOG_FORMAT=jsn` or any typo silently passed validation, then caused a mismatch in `setup_logging()`.

**Fix:** Validator rejects values that aren't `'json'` or `'text'`.

```python
@field_validator("log_format")
@classmethod
def _lower_log_format(cls, v: str) -> str:
    v = v.lower()
    if v not in {"json", "text"}:
        raise ValueError(f"LOG_FORMAT must be 'json' or 'text', got '{v}'")
    return v
```

---

## 26. `src/core/config.py` — AWS account ID hard-coded in `mlflow_artifact_root` default

**Problem:** Default value `"s3://nyc-airbnb-models-698172256228/mlflow/"` embedded the AWS account number in source code and tied the repo to a single deployment.

**Fix:** Default changed to `""` with a description that says to set it in production.

---

## 27. `src/core/exceptions.py` — `DataQualityError` message silently dropped all errors after the first

**Problem:** `errors[0] if errors else 'unknown'` — if Pandera returned 5 validation errors, only the first appeared in the log message. The rest were in `details` but invisible unless structured logs were queried.

**Fix:** Message now includes the total count and a `(+N more)` suffix.

```python
n = len(errors)
first = errors[0] if errors else "unknown"
suffix = f" (+{n - 1} more)" if n > 1 else ""
super().__init__(
    f"Data quality gate failed at stage '{stage}' [{n} error(s)]: {first}{suffix}",
    details={"stage": stage, "errors": errors},
)
```

---

## 28. `src/core/exceptions.py` — no `StoreError` for DB failures

**Problem:** SQLite / Postgres errors from the prediction store and shadow store propagated as raw `psycopg2.Error` or `sqlite3.Error` — not caught by any `NYCBaseError` handler in the API layer.

**Fix:** Added `StoreError(operation, reason)` to the hierarchy. Store modules should wrap DB exceptions in this type.

---

## 29. `src/core/logging.py` — duplicate `add_log_level` processor

**Problem:** `_SHARED_PROCESSORS` contained both `structlog.stdlib.add_log_level` and `structlog.processors.add_log_level`. The native one is redundant when routing through a stdlib handler and could produce a duplicated `level` field in JSON output.

**Fix:** Removed `structlog.processors.add_log_level`; kept the stdlib variant.

---

## 30. `src/core/logging.py` — `_SERVICE` and `_ENV` were dead code

**Problem:** `settings.dd_service` and `settings.dd_env` were assigned to module-level constants but never injected into log records. Datadog relies on `service` and `env` fields being present on every structured log line.

**Fix:** Added `_add_service_context` processor that stamps both fields via `setdefault` on every record.

```python
def _add_service_context(logger, method, event_dict):
    event_dict.setdefault("service", _SERVICE)
    event_dict.setdefault("env", _ENV)
    return event_dict
```

---

## 31. `src/core/logging.py` — idempotency check broke under pytest

**Problem:** `if root.handlers: return` — pytest adds its own handler to the root logger before any test runs, so `setup_logging()` returned immediately without configuring structlog. Tests ran with structlog in its unconfigured default state.

**Fix:** Replaced the handler-presence check with a module-level `_CONFIGURED` boolean flag.

---

## 32. `src/core/metrics.py` — `list.pop(0)` for rolling latency buffer was O(n)

**Problem:** `record_latency()` maintained a bounded list by calling `samples.pop(0)` when the list exceeded 1000 entries. `list.pop(0)` shifts all remaining elements — O(n) per eviction.

**Fix:** `defaultdict(lambda: deque(maxlen=1000))`. `deque` auto-evicts from the left in O(1) and needs no explicit length check.

---

## 33. `src/core/metrics.py` — split `record_prediction` / `record_prediction_full` was a partial-update footgun

**Problem:** Two public methods existed: `record_prediction(price)` (in-memory only) and `record_prediction_full(price, arm, cache_hit)` (in-memory + Prometheus). A caller using the wrong one silently skipped the Prometheus counter and histogram.

**Fix:** Merged into a single `record_prediction(price_usd, arm, cache_hit)` that always updates both. The in-memory-only path is now private (`_record_prediction_inmem`). `api.py` updated to use two consolidated call sites — one on cache hit, one after routing when the final arm is known.

---

## 34. `src/serving/predictor.py` — SHAP `_get_shap_explainer()` race condition under concurrent load

**Problem:** The lazy-init check `if self._shap_explainer is not None` had no lock. Multiple concurrent `/predict?explain=true` requests could all find `None` at the same time and each start loading the XGBoost pkl + initialising `TreeExplainer` (2–4 s each) in parallel. On a t3.small this risks OOM and redundant CPU work.

**Fix:** Added `threading.Lock()` to the predictor. Fast path (already initialised) checks without acquiring the lock. Lock is only acquired on the first init, with a double-checked locking pattern (`if self._shap_explainer is None` inside the lock).

---

## 35. `src/serving/predictor.py` — duplicate keys in `_FEATURE_LABELS` dict

**Problem:** `instant_bookable` appeared twice (lines 152 and 153) and `is_weekend` appeared twice (lines 151 and 187). Python silently uses the last definition — the first is dead code. If someone ever assigned different labels to the duplicates, only one would take effect with no error.

**Fix:** Removed the two stale duplicates, leaving one canonical entry for each at the end of the dict.

---

## 36. `src/serving/predictor.py` — `datetime.now()` used server-local time instead of NYC timezone

**Problem:** `is_weekend`, `month`, `is_peak_season`, and `day_of_week` were computed with `datetime.now()` which uses the server's local timezone. In production (UTC server), a Friday 11 pm EST request becomes Saturday 3 am UTC — flipping `is_weekend` from True to False. The model was trained on NYC data so inference must use NYC Eastern Time.

**Fix:** Replaced with `datetime.now(tz=ZoneInfo("America/New_York"))`. Also made `checkin_date` parsing timezone-aware so `(checkin - now).days` doesn't raise on mixed-aware/naive subtraction. Removed the unused `timezone` import that was left from an earlier draft.

---

## 37. `src/serving/api.py` — DLQ double-push on `/v1/predict` errors

**Problem:** The `observe` middleware guard was `endpoint != "/predict"`. The actual path is `/v1/predict`. Because `/v1/predict != "/predict"` is True, whenever the predict handler pushed to DLQ itself (inference error) and then raised HTTP 500, the middleware also pushed — resulting in two DLQ entries per inference failure.

**Fix:** Changed guard to `endpoint != "/v1/predict"`.

---

## 38. `src/serving/api.py` — `_guard_price` ran after `metrics.record_prediction`

**Problem:** An out-of-range predicted price triggered an HTTP 422, but only after `record_prediction` had already incremented Prometheus counters and the in-memory histogram with the invalid value. Metrics contained garbage data from model failures.

**Fix:** Swapped order — `_guard_price` now runs before `record_prediction`.

---

## 39. `src/serving/api.py` — `/health` endpoint crashed on `None` before startup completed

**Problem:** `/health` called `predictor.model_info()`, `cache.stats()`, `store.count()` without a None guard. Any health check during the startup window (before `lifespan()` yielded) produced `AttributeError: 'NoneType' object has no attribute 'model_info'`. K8s would see a 500 from a pod that wasn't yet ready.

**Fix:** Added `if predictor is None: raise HTTPException(503, ...)` guard, mirroring the existing pattern in `/health/ready`. Other singletons accessed defensively with `if cache else None`.

---

## 40. `src/serving/api.py` — global singletons typed as concrete types but initialised to `None`

**Problem:** `predictor: NYCAirbnbPredictorONNX = None` is a type annotation lie. Type checkers see the type as non-Optional and mark downstream `is None` guards as "unreachable". Static analysis and IDE null-safety tooling trust the annotation and will not flag actual None dereferences.

**Fix:** Added `# type: ignore[assignment]` at each declaration with an explanatory comment: lifespan guarantees all singletons are set before any request is served, so the usage-site type is correct and the pre-startup None is a transient initialisation detail.

---

## 41. `src/serving/cache.py` — stale cached result when `checkin_date` spans days

**Problem:** Cache keys were `md5(raw_inputs)`. `days_to_checkin` is computed as `(checkin_date - today)` inside `_engineer()`, so the same `checkin_date` input produces a different feature value every day. A cached prediction from 45 days out would be served the next day when the true value is 44 days out, silently using stale temporal features.

**Fix:** Added today's date (`date.today().isoformat()`) to the key material under the `_date` key. Cache entries now expire naturally both by TTL and by day rollover.

---

## 42. `src/serving/cache.py` — `set()` silently swallowed all serialisation exceptions

**Problem:** `except Exception: pass` meant any failure (non-serialisable result field, Redis timeout mid-write) produced zero log output. The prediction was still served, but debugging why cache hit rate was 0% was difficult.

**Fix:** Changed to `except Exception as exc: logger.debug("Cache set failed: %s", exc)`.

---

## 43. `src/serving/store.py` — SQLite migration caught all exceptions, masking real errors

**Problem:** `ALTER TABLE ADD COLUMN listing_id` raises `sqlite3.OperationalError("duplicate column name")` when the column exists — this is expected and should be silently ignored. But bare `except Exception: pass` also silently swallowed disk-full errors, syntax errors in the migration SQL, and corrupted-database errors.

**Fix:** Scoped the catch to `sqlite3.OperationalError` only.

---

## 44. `src/serving/batch.py` — jobs that crashed during processing stayed as `"processing"` forever

**Problem:** `BatchWorker._loop()` wrapped `_process()` in `try/except` and logged the exception, but left the job document with `status="processing"`. Pollers calling `GET /predict-batch/{job_id}` would keep getting `"processing"` until the 1-hour TTL expired, then receive a 404 with no indication of failure.

**Fix:** The loop's except block now calls `self._store._patch(job_id, {"status": "failed"})` before re-logging, so pollers see a terminal state immediately.

---

## 45. `src/serving/drift.py` — Postgres DSN corrupted by wrapping in `Path()`

**Problem:** `self._db = Path(db_path or settings.prediction_db)`. On Linux/macOS, `Path("postgresql://user:pw@host/db")` normalises the double slash to a single slash, producing `"postgresql:/user:pw@host/db"`. That malformed URI causes `psycopg2.connect()` to fail. Drift monitoring silently returned `[]` in production Postgres mode, meaning drift was never actually computed.

**Fix:** Store as raw string: `self._db = str(db_path) if db_path else settings.prediction_db`.

---

## 46. `src/serving/drift.py` — psycopg2 connection leaked on query exception

**Problem:** The Postgres branch in `_load_recent()` called `conn.close()` at the end but not in a `finally` block. If `cur.execute()` or `fetchall()` raised, the connection was leaked to GC.

**Fix:** Added `try/finally` around the query block for both Postgres and SQLite paths.

---

## 47. `src/serving/drift.py` — `_categorical_psi` used `list.count()` in a loop — O(n × k)

**Problem:** For each of k categories, `current_values.count(cat)` did a full linear scan of the n-element list. For `borough` (5 categories) with 1000 samples, this was 5000 comparisons when 1000 would suffice.

**Fix:** Pre-compute `Counter(current_values)` once (O(n)), then use `.get()` per category (O(1)).

---

## 48. `src/serving/ground_truth.py` — Postgres DSN wrapped in `Path()`, silently creating a local SQLite junk file

**Problem:** `GroundTruthStore.__init__` called `self._path = Path(db_path)` where `db_path` defaults to `settings.prediction_db`. In production, `PREDICTION_DB=postgresql://...`, so `Path("postgresql://user:pw@host/db")` normalised the double-slash to `"postgresql:/user:pw@host/db"`. `sqlite3.connect()` then opened a file at that local path instead of connecting to Postgres. All ground truth data was written to a random junk file — never queryable, never persisted. The bug was silent because SQLite happily creates files it doesn't find.

**Fix:** Detect Postgres DSN prefix; fall back to a known local SQLite file (`data/ground_truth.db`) and emit a `logger.warning` so the operator can configure a separate ground truth DB. `GroundTruthStore` is intentionally SQLite-only (no Postgres support).

---

## 49. `src/serving/ground_truth.py` — `error_distribution()` fetched all rows to Python for bucketing

**Problem:** `error_distribution()` ran `SELECT abs_error FROM ground_truth` (all rows), then iterated them in Python with if/elif to bucket. At 100 k ground truth rows, this transferred all rows over the SQLite IPC boundary and did O(n) Python comparisons when a single SQL aggregation would produce the same result with one pass.

**Fix:** Replaced with a single SQL query using `SUM(abs_error < 10)`, `SUM(abs_error >= 10 AND abs_error < 25)`, etc. — the DB engine does one scan and returns 6 integers.

---

## 50. `src/serving/shadow.py` — `_SHADOW_DB_DEFAULT` dead code

**Problem:** `_SHADOW_DB_DEFAULT = str(BASE_DIR / "data" / "shadow_comparisons.db")` was defined at module level but never used. `ShadowPredictor.__init__` uses `settings.shadow_db_url_effective` instead. The stale constant was misleading — it implied shadow comparisons go to a separate file, whereas the actual default shares the prediction DB.

**Fix:** Removed the unused constant.

---

## 51. `src/serving/ab.py` — Postgres DSN corrupted by `Path()` wrapping

**Problem:** `db_path: Path = DB_PATH = settings.prediction_db`. In production, `PREDICTION_DB=postgresql://...`, so `Path("postgresql://user:pw@host/db")` normalises the double-slash to a single slash. `sqlite3.connect("postgresql:/user:pw@host/db")` then opens a local file at that malformed path instead of connecting to Postgres. `ABTest` is SQLite-only, so all `ab_predictions` and `ground_truth` JOIN queries ran against a junk local file — the A/B test was silently broken in production.

**Fix:** Added Postgres DSN detection in `__init__`; falls back to `data/ab.db` local SQLite with a `logger.warning`.

---

## 52. `src/serving/ab.py` — `STATE_PATH` relative to process CWD instead of project root

**Problem:** `STATE_PATH = Path("data/ab_state.json")` is a relative path. In Docker the CWD is typically `/app`, but in tests it may be the project root or a temp dir. This caused the state file to land in a different location depending on how the process was launched, so A/B test state would not survive restarts in some environments.

**Fix:** Changed to `_BASE_DIR / "data" / "ab_state.json"` where `_BASE_DIR = Path(__file__).resolve().parents[2]` is the project root.

---

## 53. `src/serving/canary.py` — double-promote race in `advance()`

**Problem:** When advancing from stage 25%→100%, `advance()` updated `stage_idx=3, current_pct=100` inside the lock and saved state, then released the lock, then called `self._promote()` (which re-acquires the lock). Between the lock release and re-acquire, a concurrent `advance()` call could see `idx=3`, trigger the `idx+1 >= len(STAGES)` branch, and call `_promote_locked()` a second time. The second promotion attempt would `shutil.copy2(CHALLENGER_ONNX, ...)` on a file already unlinked by the first promotion, raising `FileNotFoundError`. State would be corrupt (still `active=True`).

**Fix:** Moved `_promote_locked()` call inside the `with self._lock:` block of `advance()` for the `next_pct == 100` case, so promotion is fully atomic with the state update.

---

## 54. `src/serving/canary.py` — `sys.path.insert(0, BASE_DIR)` in background thread

**Problem:** `_revert_to_previous()` called `sys.path.insert(0, str(BASE_DIR))` before importing `src.serving.alerts`. This permanently prepended `BASE_DIR` to `sys.path` on every call — if the post-promotion spike detector fired multiple times, duplicates accumulated at the front of `sys.path`, slowing all future imports. Mutating `sys.path` from a background monitoring thread is also not thread-safe with the import machinery.

**Fix:** Removed `sys.path.insert`. `src.serving.alerts` is already importable as part of the package — no path manipulation needed.
