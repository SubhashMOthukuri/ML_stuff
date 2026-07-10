"""
Alert store — writes drift/validation alerts to a JSON file and
optionally posts to Slack.

Severity routing:
  critical → Slack (immediate ping)
  warning  → Slack (if SLACK_WEBHOOK_URL set)
  info     → file only

Set SLACK_WEBHOOK_URL to enable Slack delivery.
Set SLACK_CRITICAL_CHANNEL to override the channel for critical alerts
(e.g. #incidents); defaults to whatever the webhook's default channel is.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import settings

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {
    "critical": ":rotating_light:",
    "warning":  ":warning:",
    "info":     ":information_source:",
}


class AlertStore:

    def __init__(self, path: Path | None = None):
        self._lock = threading.Lock()
        self._path = Path(path or settings.alerts_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(json.dumps([]))

    def _read(self) -> list[dict]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return []

    @staticmethod
    def _build_entry(alert_type: str, message: str, severity: str, details: dict | None) -> tuple[str, dict]:
        alert_id = str(uuid.uuid4())[:8]
        return alert_id, {
            "id":           alert_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "type":         alert_type,
            "severity":     severity,
            "message":      message,
            "details":      details or {},
            "acknowledged": False,
        }

    def _post_slack(self, alert_type: str, message: str, severity: str,
                    alert_id: str, details: dict) -> None:
        webhook_url      = settings.slack_webhook_url
        critical_channel = settings.slack_critical_channel
        if not webhook_url:
            return
        if severity == "info":
            return
        try:
            import requests
            emoji = _SEVERITY_EMOJI.get(severity, ":bell:")
            payload: dict = {
                "text": f"{emoji} *[{severity.upper()}]* `{alert_type}` — {message}",
                "attachments": [{
                    "color": "#FF0000" if severity == "critical" else "#FFA500",
                    "fields": [{"title": k, "value": str(v), "short": True}
                               for k, v in details.items()],
                    "footer": f"id:{alert_id} | nyc-airbnb-api",
                }],
            }
            if severity == "critical" and critical_channel:
                payload["channel"] = critical_channel
            resp = requests.post(webhook_url, json=payload, timeout=5)
            if not resp.ok:
                logger.error("Slack webhook returned %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.error("Failed to post Slack alert: %s", exc)

    def push(self, alert_type: str, message: str, severity: str = "warning",
             details: dict | None = None) -> str:
        severity = severity.lower()
        alert_id, entry = self._build_entry(alert_type, message, severity, details)
        with self._lock:
            alerts = self._read()
            alerts.append(entry)
            self._path.write_text(json.dumps(alerts[-500:], indent=2))
        logger.warning("ALERT [%s] %s: %s", severity.upper(), alert_type, message)
        self._post_slack(alert_type, message, severity, alert_id, details or {})
        return alert_id

    def push_once(self, alert_type: str, message: str, severity: str = "warning",
                  details: dict | None = None) -> str | None:
        """Atomic deduplicated push — suppresses if an unacknowledged alert of the same type exists."""
        severity = severity.lower()
        alert_id, entry = self._build_entry(alert_type, message, severity, details)
        with self._lock:
            alerts = self._read()
            if any(not a["acknowledged"] and a["type"] == alert_type for a in alerts):
                return None
            alerts.append(entry)
            self._path.write_text(json.dumps(alerts[-500:], indent=2))
        logger.warning("ALERT [%s] %s: %s", severity.upper(), alert_type, message)
        self._post_slack(alert_type, message, severity, alert_id, details or {})
        return alert_id

    def get_all(self, limit: int = 100) -> list[dict]:
        return self._read()[-limit:]

    def get_pending(self) -> list[dict]:
        return [a for a in self._read() if not a["acknowledged"]]

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            alerts = self._read()
            for a in alerts:
                if a["id"] == alert_id:
                    a["acknowledged"] = True
                    self._path.write_text(json.dumps(alerts, indent=2))
                    return True
        return False

    def acknowledge_all(self) -> int:
        """Mark every pending alert acknowledged. Returns the count cleared."""
        with self._lock:
            alerts = self._read()
            n = 0
            for a in alerts:
                if not a["acknowledged"]:
                    a["acknowledged"] = True
                    n += 1
            if n:
                self._path.write_text(json.dumps(alerts, indent=2))
        return n

    def stats(self) -> dict:
        alerts = self._read()
        pending = [a for a in alerts if not a["acknowledged"]]
        by_sev  = {}
        for a in pending:
            by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
        return {
            "total":   len(alerts),
            "pending": len(pending),
            "by_severity": by_sev,
        }


# module-level singleton
alerts = AlertStore()
