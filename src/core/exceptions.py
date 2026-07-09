"""
Custom exception hierarchy for the NYC Airbnb API.

Why custom exceptions?
  Built-in exceptions (ValueError, RuntimeError) give no context about
  WHERE or WHY something failed. Custom exceptions let us:
    1. Catch specific failure types without broad except clauses
    2. Attach structured data (feature name, model arm, listing id)
    3. Map cleanly to HTTP status codes in the API layer

Hierarchy:
  NYCBaseError
  ├── PredictionError       — model inference failures
  │   ├── ModelNotLoadedError
  │   └── FeatureEngineeringError
  ├── CacheError            — Redis read/write failures
  ├── AuthError             — API key / token failures
  ├── ConfigError           — missing or invalid configuration at startup
  ├── VaultError            — Oracle Vault secret fetch failures
  └── DataQualityError      — input validation failures (moved from nyc_data_validator)
"""


class NYCBaseError(Exception):
    """Root exception — catch this to handle any app-level error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details})"


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionError(NYCBaseError):
    """Base for all model inference failures."""


class ModelNotLoadedError(PredictionError):
    """ONNX model artefacts not found or failed to load at startup."""

    def __init__(self, path: str):
        super().__init__(
            f"Model not loaded — artefact missing: {path}",
            details={"path": path},
        )


class FeatureEngineeringError(PredictionError):
    """Input could not be transformed into model features."""

    def __init__(self, feature: str, reason: str):
        super().__init__(
            f"Feature engineering failed for '{feature}': {reason}",
            details={"feature": feature, "reason": reason},
        )


# ── Cache ─────────────────────────────────────────────────────────────────────

class CacheError(NYCBaseError):
    """Redis read, write, or connection failure."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Cache {operation} failed: {reason}",
            details={"operation": operation, "reason": reason},
        )


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthError(NYCBaseError):
    """Invalid or missing API key / ingest token."""

    def __init__(self, reason: str = "Invalid or missing API key"):
        super().__init__(reason)


# ── Config ────────────────────────────────────────────────────────────────────

class ConfigError(NYCBaseError):
    """Required configuration is missing or invalid at startup.

    Raise this during lifespan startup so the container crashes fast
    instead of serving broken predictions.
    """

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Configuration error — '{field}': {reason}",
            details={"field": field, "reason": reason},
        )


# ── Vault ─────────────────────────────────────────────────────────────────────

class VaultError(NYCBaseError):
    """Oracle Vault secret could not be fetched."""

    def __init__(self, secret_name: str, reason: str):
        super().__init__(
            f"Vault fetch failed for secret '{secret_name}': {reason}",
            details={"secret_name": secret_name, "reason": reason},
        )


# ── Data quality ──────────────────────────────────────────────────────────────

class DataQualityError(NYCBaseError):
    """Input or snapshot data failed Pandera validation gates."""

    def __init__(self, stage: str, errors: list[str]):
        super().__init__(
            f"Data quality gate failed at stage '{stage}': {errors[0] if errors else 'unknown'}",
            details={"stage": stage, "errors": errors},
        )
