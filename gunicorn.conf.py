"""
Gunicorn configuration for NYC Airbnb Price Prediction API.

Usage:
    PYTHONPATH=. gunicorn -c gunicorn.conf.py src.new_york_workflow.nyc_api:app

Scale:
    workers = (2 × CPU_count) + 1  is the standard formula.
    Each uvicorn worker holds its own ONNX InferenceSession in memory.
    ONNX Runtime sessions are thread-safe so a single worker handles
    many concurrent async requests without blocking.
"""

import multiprocessing
import os

# ── server socket ─────────────────────────────────────────────────────────────
bind    = "0.0.0.0:8001"
backlog = 2048

# ── worker pool ───────────────────────────────────────────────────────────────
# uvicorn.workers.UvicornWorker = async ASGI workers (FastAPI is ASGI)
worker_class  = "uvicorn.workers.UvicornWorker"
# Cap at 4 on the 12 GB ARM free-tier node. preload_app shares the ONNX model
# via copy-on-write, but each worker still needs ~1.5 GB for its own heap.
workers       = int(os.getenv("GUNICORN_WORKERS", min(multiprocessing.cpu_count() * 2 + 1, 4)))
threads       = 1        # uvicorn workers are async — no thread pool needed
worker_connections = 1000

# ── timeouts ──────────────────────────────────────────────────────────────────
timeout       = 30       # kill worker if request takes > 30 s
keepalive     = 5        # keep alive between Nginx pings
graceful_timeout = 10    # SIGTERM grace period before SIGKILL

# ── logging ───────────────────────────────────────────────────────────────────
accesslog     = "-"      # stdout
errorlog      = "-"      # stderr
loglevel      = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" rid=%({X-Request-ID}i)s'

# ── process ───────────────────────────────────────────────────────────────────
proc_name     = "nyc-api"
preload_app   = True     # load model once in master, fork to workers (saves memory)
max_requests        = 1000   # recycle worker after N requests — guards against slow memory leaks
max_requests_jitter = 100    # randomize so workers don't all restart simultaneously
