"""
In-memory runtime metrics for the prediction API.

Tracks per-endpoint request counts, error counts, latency, and prediction
value distribution. Thread-safe via a single lock.

For production at scale swap this for prometheus-fastapi-instrumentator —
the /metrics response shape stays the same so dashboards don't change.
"""

import threading
import time
from collections import defaultdict


class _Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        # per-endpoint counters
        self._requests: dict[str, int] = defaultdict(int)
        self._errors:   dict[str, int] = defaultdict(int)

        # per-endpoint latency (ms) — keep last 1000 samples per endpoint
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._MAX_SAMPLES = 1000

        # prediction value distribution (all endpoints combined)
        self._pred_count = 0
        self._pred_sum   = 0.0
        self._pred_min   = float("inf")
        self._pred_max   = float("-inf")

    # ------------------------------------------------------------------ writes

    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self._requests[endpoint] += 1

    def record_error(self, endpoint: str) -> None:
        with self._lock:
            self._errors[endpoint] += 1

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            samples = self._latencies[endpoint]
            samples.append(latency_ms)
            if len(samples) > self._MAX_SAMPLES:
                samples.pop(0)

    def record_prediction(self, value: float) -> None:
        with self._lock:
            self._pred_count += 1
            self._pred_sum   += value
            if value < self._pred_min:
                self._pred_min = value
            if value > self._pred_max:
                self._pred_max = value

    # ------------------------------------------------------------------ reads

    def summary(self) -> dict:
        with self._lock:
            uptime_s = time.time() - self._start_time

            endpoint_stats = {}
            all_endpoints = set(self._requests) | set(self._errors)
            for ep in all_endpoints:
                reqs = self._requests[ep]
                errs = self._errors[ep]
                lats = self._latencies[ep]
                endpoint_stats[ep] = {
                    "requests":   reqs,
                    "errors":     errs,
                    "error_rate": f"{(errs / reqs * 100):.1f}%" if reqs else "0.0%",
                    "latency_ms": {
                        "avg": round(sum(lats) / len(lats), 1) if lats else None,
                        "min": round(min(lats), 1) if lats else None,
                        "max": round(max(lats), 1) if lats else None,
                        "p95": round(sorted(lats)[int(len(lats) * 0.95)], 1) if len(lats) >= 5 else None,
                    },
                }

            pred_stats = {
                "total":   self._pred_count,
                "mean":    round(self._pred_sum / self._pred_count, 4) if self._pred_count else None,
                "min":     round(self._pred_min, 4) if self._pred_count else None,
                "max":     round(self._pred_max, 4) if self._pred_count else None,
            }

            total_requests = sum(self._requests.values())
            total_errors   = sum(self._errors.values())

            return {
                "uptime_seconds": round(uptime_s, 1),
                "uptime_human":   _fmt_uptime(uptime_s),
                "totals": {
                    "requests":   total_requests,
                    "errors":     total_errors,
                    "error_rate": f"{(total_errors / total_requests * 100):.1f}%" if total_requests else "0.0%",
                },
                "endpoints":   endpoint_stats,
                "predictions": pred_stats,
            }


def _fmt_uptime(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# module-level singleton — import this everywhere
metrics = _Metrics()
