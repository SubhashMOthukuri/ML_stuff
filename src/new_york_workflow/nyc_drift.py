"""
Production drift monitor.

Compares recent prediction traffic (from SQLite store) against the
training baseline (baseline_stats.json).

Two drift signals:
  1. Feature drift   — PSI > 0.20 on any monitored feature
  2. Prediction drift — rolling mean shifts > 15% from training mean

PSI (Population Stability Index):
  < 0.10  no drift
  0.10–0.20  minor drift  (warning)
  > 0.20  major drift     (alert → consider retraining)

Usage:
    from src.new_york_workflow.nyc_drift import DriftMonitor
    monitor = DriftMonitor()
    report  = monitor.check()
    # report.status in ('ok', 'warning', 'critical')
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[2]
MODEL_DIR  = BASE_DIR / "models" / "nyc"
BASELINE   = MODEL_DIR / "baseline_stats.json"

MONITOR_FEATURES = [
    "accommodates", "bedrooms", "bathrooms", "minimum_nights",
    "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "amenity_count",
]


@dataclass
class DriftReport:
    checked_at:      str
    window_hours:    int
    n_samples:       int
    status:          str          # 'ok' | 'warning' | 'critical' | 'insufficient_data'
    prediction_drift: dict
    feature_drift:   dict
    categorical_drift: dict
    alerts:          list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checked_at":       self.checked_at,
            "window_hours":     self.window_hours,
            "n_samples":        self.n_samples,
            "status":           self.status,
            "prediction_drift": self.prediction_drift,
            "feature_drift":    self.feature_drift,
            "categorical_drift": self.categorical_drift,
            "alerts":           self.alerts,
        }


class DriftMonitor:

    def __init__(self, db_path: Path | None = None, baseline_path: Path = BASELINE):
        import os
        self._db   = Path(db_path or os.getenv("PREDICTION_DB", "data/predictions.db"))
        self._base = self._load_baseline(baseline_path)

    @staticmethod
    def _load_baseline(path: Path) -> dict | None:
        if not path.exists():
            logger.warning("Baseline stats not found at %s — run register_baseline.py first", path)
            return None
        return json.loads(path.read_text())

    # ── public ────────────────────────────────────────────────────────────────

    def check(self, window_hours: int = 168) -> DriftReport:
        """
        Run drift check over the last `window_hours` of production traffic.
        Default: 168h = 7 days.
        """
        now = datetime.now(timezone.utc)
        ts  = (now - timedelta(hours=window_hours)).isoformat()

        rows = self._load_recent(ts)
        n    = len(rows)

        if self._base is None:
            return DriftReport(
                checked_at=now.isoformat(), window_hours=window_hours, n_samples=n,
                status="no_baseline", prediction_drift={}, feature_drift={},
                categorical_drift={}, alerts=["Baseline not registered — run register_baseline.py"],
            )

        min_samples = self._base["drift_thresholds"]["min_samples_for_check"]
        if n < min_samples:
            return DriftReport(
                checked_at=now.isoformat(), window_hours=window_hours, n_samples=n,
                status="insufficient_data", prediction_drift={}, feature_drift={},
                categorical_drift={}, alerts=[f"Only {n} samples in window (need {min_samples})"],
            )

        report_alerts = []
        thresholds    = self._base["drift_thresholds"]

        # 1. Prediction drift
        pred_drift = self._prediction_drift(rows, thresholds["prediction_mean_pct"])
        if pred_drift.get("alert"):
            report_alerts.append(pred_drift["alert"])

        # 2. Feature drift (PSI)
        feat_drift = {}
        for feat in MONITOR_FEATURES:
            if feat not in self._base["feature_dist"]:
                continue
            values = [r.get(feat) for r in rows if r.get(feat) is not None]
            if not values:
                continue
            psi = self._psi(values, self._base["feature_dist"][feat])
            status = "ok" if psi < 0.10 else ("warning" if psi < 0.20 else "critical")
            feat_drift[feat] = {"psi": round(psi, 4), "status": status}
            if psi >= thresholds["feature_psi"]:
                report_alerts.append(f"Feature drift: {feat} PSI={psi:.3f} (threshold {thresholds['feature_psi']})")

        # 3. Categorical drift (PSI on proportions)
        cat_drift = {}
        if "categorical_dist" in self._base:
            for col, baseline_props in self._base["categorical_dist"].items():
                col_vals = [r.get(col) for r in rows if r.get(col) is not None]
                if not col_vals:
                    continue
                psi = self._categorical_psi(col_vals, baseline_props)
                status = "ok" if psi < 0.10 else ("warning" if psi < 0.20 else "critical")
                cat_drift[col] = {"psi": round(psi, 4), "status": status}
                if psi >= thresholds["categorical_psi"]:
                    report_alerts.append(f"Categorical drift: {col} PSI={psi:.3f}")

        # Overall status
        all_psi = [v["psi"] for v in feat_drift.values()] + [v["psi"] for v in cat_drift.values()]
        if report_alerts and any(p >= 0.20 for p in all_psi):
            overall = "critical"
        elif report_alerts:
            overall = "warning"
        else:
            overall = "ok"

        return DriftReport(
            checked_at=now.isoformat(), window_hours=window_hours, n_samples=n,
            status=overall, prediction_drift=pred_drift,
            feature_drift=feat_drift, categorical_drift=cat_drift,
            alerts=report_alerts,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _load_recent(self, since_iso: str) -> list[dict]:
        if not self._db.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT features_json, predicted_price, borough, room_type FROM predictions "
                "WHERE timestamp >= ? AND cache_hit = 0",
                (since_iso,),
            ).fetchall()
            conn.close()
            out = []
            for r in rows:
                try:
                    rec = json.loads(r["features_json"])
                    rec["_predicted_price"] = r["predicted_price"]
                    rec["borough"]   = r["borough"]   or rec.get("borough")
                    rec["room_type"] = r["room_type"] or rec.get("room_type")
                    out.append(rec)
                except Exception:
                    pass
            return out
        except Exception as exc:
            logger.error("Error loading predictions for drift check: %s", exc)
            return []

    def _prediction_drift(self, rows: list[dict], threshold_pct: float) -> dict:
        prices      = [r["_predicted_price"] for r in rows if r.get("_predicted_price") is not None]
        baseline    = self._base["price_dist"]
        if not prices:
            return {}
        current_mean = sum(prices) / len(prices)
        baseline_mean = baseline["mean"]
        pct_shift = (current_mean - baseline_mean) / max(baseline_mean, 1)
        alert = None
        if abs(pct_shift) > threshold_pct:
            alert = (f"Prediction mean shifted {pct_shift*100:+.1f}% "
                     f"(${current_mean:.0f} vs baseline ${baseline_mean:.0f})")
        return {
            "current_mean":   round(current_mean, 2),
            "baseline_mean":  baseline_mean,
            "shift_pct":      round(pct_shift * 100, 2),
            "threshold_pct":  threshold_pct * 100,
            "status":         "alert" if alert else "ok",
            "alert":          alert,
        }

    @staticmethod
    def _psi(current_values: list[float], baseline_stats: dict) -> float:
        """
        PSI using percentile-derived bins with correct handling of tied percentiles.

        Bins are half-open (lo, hi] — closed on the right — which matches the CDF
        interpretation P(lo < v ≤ hi). When consecutive percentiles share the same
        value (common for discrete/concentrated features, e.g. accommodates where
        p25=p50=2), we take the highest CDF at each unique edge so the expected
        proportions always sum to 1.0 and PSI ≈ 0 for same-distribution data.

        Floor: 0.0001 prevents log(0) on empty bins.
        """
        anchors = [
            (float("-inf"),          0.00),
            (baseline_stats["p10"], 0.10),
            (baseline_stats["p25"], 0.25),
            (baseline_stats["p50"], 0.50),
            (baseline_stats["p75"], 0.75),
            (baseline_stats["p90"], 0.90),
            (float("inf"),          1.00),
        ]

        # unique edge → highest CDF at that edge (resolves ties like p25=p50=2)
        edge_cdf: dict[float, float] = {}
        for val, cdf in anchors:
            edge_cdf[val] = max(edge_cdf.get(val, 0.0), cdf)

        unique_edges = sorted(edge_cdf.keys())

        # bins: (lower_exclusive, upper_inclusive, expected_proportion)
        bins: list[tuple] = []
        for i in range(len(unique_edges) - 1):
            lo = unique_edges[i]
            hi = unique_edges[i + 1]
            expected = edge_cdf[hi] - edge_cdf[lo]
            if expected > 0:
                bins.append((lo, hi, expected))

        if not bins:
            return 0.0

        FLOOR  = 0.0001
        n      = max(len(current_values), 1)
        counts = [0] * len(bins)
        for v in current_values:
            placed = False
            for idx, (lo, hi, _) in enumerate(bins):
                if lo < v <= hi:   # half-open (lo, hi]
                    counts[idx] += 1
                    placed = True
                    break
            if not placed:
                counts[0] += 1    # only for v = -inf or NaN; rare fallback

        psi = 0.0
        for count, (_, _, expected) in zip(counts, bins):
            a = max(count / n, FLOOR)
            e = max(expected, FLOOR)
            psi += (a - e) * math.log(a / e)
        return max(psi, 0.0)

    @staticmethod
    def _categorical_psi(current_values: list[str], baseline_props: dict) -> float:
        """PSI for categorical features. Floor at 0.0001 to prevent log(0) blowup."""
        n    = max(len(current_values), 1)
        FLOOR = 0.0001
        psi  = 0.0
        for cat, expected_p in baseline_props.items():
            actual_p = max(current_values.count(cat) / n, FLOOR)
            e        = max(expected_p, FLOOR)
            psi     += (actual_p - e) * math.log(actual_p / e)
        return max(psi, 0.0)
