"""
Alert store — writes drift/validation alerts to a JSON file.

Alerts are append-only. The API serves them via GET /alerts.
In production, replace push() with a Slack/PagerDuty webhook call.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ALERTS_PATH = Path(os.getenv("ALERTS_FILE", "models/nyc/alerts.json"))


class AlertStore:
    _lock = threading.Lock()

    def __init__(self, path: Path = ALERTS_PATH):
        self._path = Path(path)
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

    def push(self, alert_type: str, message: str, severity: str = "warning",
             details: dict | None = None) -> str:
        alert_id, entry = self._build_entry(alert_type, message, severity, details)
        with self._lock:
            alerts = self._read()
            alerts.append(entry)
            self._path.write_text(json.dumps(alerts[-500:], indent=2))
        logger.warning("ALERT [%s] %s: %s", severity.upper(), alert_type, message)
        return alert_id

    def push_once(self, alert_type: str, message: str, severity: str = "warning",
                  details: dict | None = None) -> str | None:
        """Atomic deduplicated push — suppresses if an unacknowledged alert of the same type exists."""
        alert_id, entry = self._build_entry(alert_type, message, severity, details)
        with self._lock:
            alerts = self._read()
            if any(not a["acknowledged"] and a["type"] == alert_type for a in alerts):
                return None
            alerts.append(entry)
            self._path.write_text(json.dumps(alerts[-500:], indent=2))
        logger.warning("ALERT [%s] %s: %s", severity.upper(), alert_type, message)
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
