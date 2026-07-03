"""
Structured logging via structlog.

Dev  (LOG_FORMAT=text) → colorized console output
Prod (LOG_FORMAT=json) → one JSON object per line → Fluent Bit → Datadog / ELK

All stdlib logging.getLogger() calls in other modules are also structured via
ProcessorFormatter — no changes needed in those modules.

Usage in application code:
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("prediction served", arm="challenger", latency_ms=12.4)

Env vars:
    LOG_LEVEL  — DEBUG / INFO / WARNING / ERROR  (default: INFO)
    LOG_FORMAT — json / text                     (default: text)
    DD_SERVICE — service name tag                (default: nyc-api)
    DD_ENV     — environment tag                 (default: dev)
"""

import logging

import structlog

from src.config import settings

_SERVICE = settings.dd_service
_ENV     = settings.dd_env

# Processors run on every log record — both structlog and stdlib loggers.
_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,        # trace_id, path, method from middleware
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
    structlog.processors.add_log_level,             # normalise level field
]


def setup_logging() -> None:
    """Configure root logger + structlog once. Safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return  # idempotent

    log_format = settings.log_format
    level      = getattr(logging, settings.log_level, logging.INFO)

    renderer = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    # structlog configuration — wraps output through stdlib handler below.
    structlog.configure(
        processors=_SHARED_PROCESSORS + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # stdlib handler that structlog (and all logging.getLogger() callers) emit to.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
