"""
Central configuration — single source of truth for every env variable.

Why this exists:
  Previously each module called os.getenv() with its own default values,
  meaning the same variable could have different defaults in different files.
  Now every module imports `settings` and reads from one place.

Usage:
  from src.core.config import settings
  host = settings.redis_host
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_host: str             = Field("localhost", description="Redis hostname")
    redis_port: int             = Field(6379,        description="Redis port")
    redis_password: str         = Field("",          description="Redis password (empty = no auth)")
    cache_ttl_seconds: int      = Field(300,         description="Prediction cache TTL in seconds")

    # ── Database stores ────────────────────────────────────────────────────────
    prediction_db: str          = Field("data/predictions.db", description="SQLite path or Postgres DSN for predictions")
    shadow_db_url: str          = Field("",          description="Postgres DSN for shadow comparisons (defaults to prediction_db)")
    dlq_max_size: int           = Field(10000,       description="Max DLQ entries before eviction")

    # ── API security ───────────────────────────────────────────────────────────
    rate_limit: str             = Field("60/minute", description="slowapi rate limit string")
    valid_api_keys: str         = Field("",          description="Comma-separated valid API keys")
    ground_truth_ingest_token: str = Field("",       description="Shared secret for /ground-truth/ingest")

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str              = Field("INFO",      description="Python logging level")
    log_format: str             = Field("text",      description="'json' for prod, 'text' for local dev")

    # ── Service identity ───────────────────────────────────────────────────────
    dd_service: str             = Field("nyc-api",   description="Service name tag (Datadog / OTel)")
    dd_env: str                 = Field("dev",       description="Environment tag: dev / staging / prod")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str    = Field(
        f"sqlite:///{_ROOT / 'mlflow.db'}",
        description="MLflow tracking URI (Postgres DSN injected in production)",
    )
    mlflow_artifact_root: str   = Field(
        "s3://nyc-airbnb-models-698172256228/mlflow/",
        description="MLflow default artifact root — S3 bucket for production model artifacts",
    )
    mlflow_ui_url: str          = Field("http://localhost:5000", description="MLflow UI URL")

    # ── Alerts ─────────────────────────────────────────────────────────────────
    alerts_file: Path           = Field("models/nyc/alerts.json", description="JSON alert log path")
    slack_webhook_url: str      = Field("",          description="Slack incoming webhook URL")
    slack_critical_channel: str = Field("",          description="Override channel for critical alerts")

    # ── CORS ───────────────────────────────────────────────────────────────────
    cors_allow_origins: str     = Field(
        "http://localhost:5173,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # ── Tracing (OpenTelemetry) ────────────────────────────────────────────────
    otlp_endpoint: str | None   = Field(None,        description="OTLP gRPC endpoint — None disables tracing")

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def _lower_log_format(cls, v: str) -> str:
        return v.lower()

    # ── Derived helpers ────────────────────────────────────────────────────────

    @property
    def shadow_db_url_effective(self) -> str:
        """Shadow DB connection — falls back to prediction_db if SHADOW_DB_URL is unset."""
        return self.shadow_db_url if self.shadow_db_url else self.prediction_db

    @property
    def api_keys_set(self) -> set[str]:
        """Parsed set of valid API keys — empty set means auth disabled."""
        if not self.valid_api_keys:
            return set()
        return {k.strip() for k in self.valid_api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys_set)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


# Module-level singleton — import this everywhere
settings = Settings()
