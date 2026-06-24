"""
Centralized logging setup.

Call setup_logging() once at application startup (in lifespan or __main__).

Env vars:
  LOG_LEVEL  — DEBUG / INFO / WARNING / ERROR  (default: INFO)
  LOG_FORMAT — json / text                     (default: text)

In production set LOG_FORMAT=json so log aggregators (Datadog, CloudWatch,
ELK) can parse fields directly.
"""

import logging
import json
import os
import time


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line — machine-parseable in production."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging() -> None:
    """Configure root logger once. Safe to call multiple times (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = os.getenv("LOG_FORMAT", "text").lower()
    if fmt == "json":
        formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
