"""
Runtime metrics — dual-track:

  1. In-memory store  → served as JSON via GET /metrics/summary
  2. prometheus_client → scraped as Prometheus text via GET /metrics
                         → Prometheus → Grafana dashboards

Custom ML metrics exposed to Prometheus:
  nyc_predictions_total          counter  {arm, cache_hit}
  nyc_prediction_price_usd       histogram
  nyc_canary_traffic_pct         gauge
  nyc_active_alerts              gauge    {severity}
  nyc_dlq_size                   gauge
  nyc_psi_score                  gauge    {feature, feature_type}
  nyc_drift_status               gauge    (0=ok, 1=warning, 2=critical)
"""

import threading
import time
from collections import defaultdict, deque

from prometheus_client import Counter, Gauge, Histogram

# ── Prometheus metric objects (module-level singletons) ───────────────────────

prom_predictions_total = Counter(
    "nyc_predictions_total",
    "Total predictions served",
    ["arm", "cache_hit"],
)

prom_prediction_price_usd = Histogram(
    "nyc_prediction_price_usd",
    "Predicted nightly price in USD",
    buckets=[50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000],
)

prom_canary_traffic_pct = Gauge(
    "nyc_canary_traffic_pct",
    "Current canary traffic percentage (0–100)",
)

prom_active_alerts = Gauge(
    "nyc_active_alerts",
    "Unacknowledged alerts by severity",
    ["severity"],
)

prom_dlq_size = Gauge(
    "nyc_dlq_size",
    "Dead letter queue depth",
)

prom_psi_score = Gauge(
    "nyc_psi_score",
    "Population Stability Index per feature (0=no drift, 0.1=warning, 0.2=critical)",
    ["feature", "feature_type"],
)

prom_drift_status = Gauge(
    "nyc_drift_status",
    "Overall drift status: 0=ok, 1=warning, 2=critical",
)


# ── In-memory store (JSON /metrics/summary) ───────────────────────────────────

class _Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        self._requests: dict[str, int]            = defaultdict(int)
        self._errors:   dict[str, int]            = defaultdict(int)
        self._latencies: dict[str, deque[float]]  = defaultdict(lambda: deque(maxlen=1000))

        self._pred_count = 0
        self._pred_sum   = 0.0
        self._pred_min   = float("inf")
        self._pred_max   = float("-inf")

    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self._requests[endpoint] += 1

    def record_error(self, endpoint: str) -> None:
        with self._lock:
            self._errors[endpoint] += 1

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self._latencies[endpoint].append(latency_ms)

    def _record_prediction_inmem(self, price_usd: float) -> None:
        self._pred_count += 1
        self._pred_sum   += price_usd
        if price_usd < self._pred_min:
            self._pred_min = price_usd
        if price_usd > self._pred_max:
            self._pred_max = price_usd

    def record_prediction(self, price_usd: float, arm: str, cache_hit: bool) -> None:
        """Update in-memory stats and Prometheus metrics."""
        with self._lock:
            self._record_prediction_inmem(price_usd)
        prom_predictions_total.labels(
            arm=arm, cache_hit="true" if cache_hit else "false"
        ).inc()
        prom_prediction_price_usd.observe(price_usd)

    def summary(self) -> dict:
        with self._lock:
            uptime_s = time.time() - self._start_time

            endpoint_stats = {}
            for ep in self._requests.keys() | self._errors.keys():
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
                "total": self._pred_count,
                "mean":  round(self._pred_sum / self._pred_count, 4) if self._pred_count else None,
                "min":   round(self._pred_min, 4) if self._pred_count else None,
                "max":   round(self._pred_max, 4) if self._pred_count else None,
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


metrics = _Metrics()
