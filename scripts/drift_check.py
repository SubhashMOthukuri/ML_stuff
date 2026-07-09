"""
Nightly drift check — run as a Kubernetes CronJob.

Pulls last N hours of predictions from Postgres, computes PSI against the
training baseline, and fires a Slack alert (via AlertStore) if any feature or
the prediction mean drifts beyond threshold.

Exit codes:
  0 — ok, warning, or insufficient data (Slack sent for warning; CronJob shows Success)
  1 — critical drift detected (Slack sent; CronJob shows Failed → visible in k8s)

Usage:
  python scripts/drift_check.py                  # 24-hour window
  python scripts/drift_check.py --window-hours 168   # 7-day window
  python scripts/drift_check.py --dry-run        # print report, no Slack
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.drift import DriftMonitor
from src.serving.alerts import AlertStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Data drift check")
    parser.add_argument("--window-hours", type=int, default=24,
                        help="Look-back window in hours (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report without firing Slack alerts")
    args = parser.parse_args()

    monitor = DriftMonitor()
    alert_store = AlertStore()

    logger.info("Running drift check (window=%dh, dry_run=%s)", args.window_hours, args.dry_run)
    report = monitor.check(window_hours=args.window_hours)

    print(json.dumps(report.to_dict(), indent=2))

    if report.status in ("no_baseline", "insufficient_data"):
        logger.warning("Drift check skipped: %s (%d samples)", report.status, report.n_samples)
        return 0  # not a failure — just not enough data yet

    if report.status == "ok":
        logger.info("Drift check PASSED — no significant drift detected (%d samples)", report.n_samples)
        return 0

    # warning or critical
    severity = "critical" if report.status == "critical" else "warning"
    summary_lines = report.alerts or [f"Drift status: {report.status}"]
    message = "; ".join(summary_lines)

    details = {
        "window_hours":  report.window_hours,
        "n_samples":     report.n_samples,
        "status":        report.status,
        "checked_at":    report.checked_at,
    }
    # Add top drifting feature to details for quick Slack scan
    if report.feature_drift:
        worst = max(report.feature_drift.items(), key=lambda x: x[1]["psi"])
        details["worst_feature"] = f"{worst[0]} PSI={worst[1]['psi']:.3f}"
    if report.prediction_drift:
        details["prediction_shift"] = f"{report.prediction_drift.get('shift_pct', 0):+.1f}%"

    if args.dry_run:
        logger.info("DRY RUN — would fire %s alert: %s", severity, message)
    else:
        alert_id = alert_store.push_once(
            alert_type=f"drift_{severity}",
            message=message,
            severity=severity,
            details=details,
        )
        if alert_id:
            logger.warning("Alert fired (id=%s severity=%s): %s", alert_id, severity, message)
        else:
            logger.info("Alert suppressed — unacknowledged drift_%s already open", severity)

    return 1 if severity == "critical" else 0


if __name__ == "__main__":
    sys.exit(main())
