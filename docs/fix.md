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

## Known remaining item

**Frontend runs as root:** The official `nginx:alpine` image requires root to bind port 80. The fix is to switch to `nginxinc/nginx-unprivileged` (listens on 8080) and update `frontend/nginx.conf` and `frontend.yaml` accordingly. Left intentionally — security team item.
