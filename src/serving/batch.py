"""
Async batch prediction queue — Redis-backed.

Architecture
────────────
  POST /predict-batch  →  validate → enqueue → return job_id immediately
  GET  /predict-batch/{job_id}  →  poll status + partial/full results

Redis layout
────────────
  batch:queue          LIST   — LPUSH to enqueue, BRPOP to consume (atomic)
  batch:job:{job_id}   STRING — JSON job document, TTL 1 hour

Worker
──────
  One daemon thread per Gunicorn worker process.
  BRPOP is atomic, so only one process claims each job even with 2+ workers.
  Partial results are written after every listing so polling shows progress.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

QUEUE_KEY      = "batch:queue"
JOB_PREFIX     = "batch:job:"
JOB_TTL        = 3600   # results live for 1 hour
MAX_BATCH_SIZE = 50


# ── job store ─────────────────────────────────────────────────────────────────

class BatchJobStore:
    """Redis-backed store for batch job documents."""

    def __init__(self, redis_client):
        self._r = redis_client

    @property
    def available(self) -> bool:
        return self._r is not None

    def create(self, listings: list[dict], explain: bool = False) -> str:
        job_id = str(uuid.uuid4())
        doc = {
            "job_id":       job_id,
            "status":       "queued",   # queued | processing | done | failed
            "total":        len(listings),
            "done":         0,
            "explain":      explain,
            "created_at":   datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "results":      [],
            "errors":       [],
        }
        # Pipeline: write job doc + enqueue atomically
        pipe = self._r.pipeline()
        pipe.setex(f"{JOB_PREFIX}{job_id}", JOB_TTL, json.dumps(doc))
        pipe.lpush(QUEUE_KEY, json.dumps({
            "job_id":   job_id,
            "listings": listings,
            "explain":  explain,
        }))
        pipe.execute()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        raw = self._r.get(f"{JOB_PREFIX}{job_id}")
        return json.loads(raw) if raw else None

    def _patch(self, job_id: str, updates: dict) -> None:
        key = f"{JOB_PREFIX}{job_id}"
        raw = self._r.get(key)
        if raw:
            doc = json.loads(raw)
            doc.update(updates)
            self._r.setex(key, JOB_TTL, json.dumps(doc))


# ── worker ────────────────────────────────────────────────────────────────────

class BatchWorker:
    """
    Daemon thread that drains the batch queue.

    One instance per Gunicorn worker process — BRPOP is atomic so only one
    process claims each job even when multiple workers run in parallel.
    Individual listing failures never fail the whole job; each error is
    recorded in doc["errors"] and processing continues.
    """

    def __init__(self, redis_client, job_store: BatchJobStore, predictor):
        self._r       = redis_client
        self._store   = job_store
        self._pred    = predictor
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="batch-worker"
        )
        self._thread.start()
        logger.info("BatchWorker started")

    def stop(self) -> None:
        self._stop.set()

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._r.brpop(QUEUE_KEY, timeout=2)
                if item is None:
                    continue
                _, payload_bytes = item
                payload = json.loads(payload_bytes)
                try:
                    self._process(payload)
                except Exception as exc:
                    logger.error("BatchWorker job crash job=%s: %s", payload.get("job_id", "?")[:8], exc, exc_info=True)
                    self._store._patch(payload["job_id"], {"status": "failed"})
            except Exception as exc:
                logger.error("BatchWorker queue error: %s", exc, exc_info=True)

    def _process(self, payload: dict) -> None:
        job_id   = payload["job_id"]
        listings = payload["listings"]
        explain  = payload.get("explain", False)

        self._store._patch(job_id, {"status": "processing"})
        logger.info("batch start job=%s n=%d explain=%s", job_id[:8], len(listings), explain)

        results: list[dict] = []
        errors:  list[dict] = []

        for i, raw in enumerate(listings):
            lid = raw.get("listing_id")     # don't mutate; predict_raw ignores extra keys
            try:
                pred = self._pred.predict_raw(raw)
                entry: dict = {
                    "index":                  i,
                    "listing_id":             lid,
                    "status":                 "ok",
                    "price_usd":              pred["price_usd"],
                    "price_low":              pred["price_low"],
                    "price_high":             pred["price_high"],
                    "price_str":              pred["price_str"],
                    "business_rules_applied": pred.get("business_rules_applied", []),
                }
                if explain:
                    try:
                        entry["top_features"] = self._pred.explain_raw(raw, pred["price_usd"])
                    except Exception as shap_exc:
                        logger.warning("batch SHAP failed item=%d: %s", i, shap_exc)
                results.append(entry)
            except Exception as exc:
                logger.warning("batch item=%d error: %s", i, exc)
                errors.append({
                    "index":      i,
                    "listing_id": lid,
                    "status":     "error",
                    "detail":     str(exc),
                })

            # Write partial results after every item — polling sees live progress
            self._store._patch(job_id, {
                "done":    i + 1,
                "results": results,
                "errors":  errors,
            })

        self._store._patch(job_id, {
            "status":       "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            "batch done job=%s succeeded=%d errors=%d",
            job_id[:8], len(results), len(errors),
        )
