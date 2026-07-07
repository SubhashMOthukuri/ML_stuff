"""RequestStore must never leak the raw Postgres DSN (with password) into logs."""

import sys
from unittest.mock import MagicMock

from src.new_york_workflow.nyc_store import RequestStore

PG_DSN = "postgresql://nyc_app:Rama1998@nyc-rds.abc.us-east-2.rds.amazonaws.com:5432/nyc"


def _mock_pg_store() -> RequestStore:
    """Build a RequestStore against a fake Postgres DSN without a real connection.

    psycopg2 is imported lazily inside nyc_store.py and isn't installed in
    this dev environment (SQLite is used locally) — stub it via sys.modules
    rather than patching an attribute that doesn't exist yet.
    """
    mock_conn = MagicMock()
    fake_psycopg2 = MagicMock()
    fake_psycopg2.connect.return_value = mock_conn
    fake_psycopg2.extras = MagicMock()
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_psycopg2.extras
    return RequestStore(db_path=PG_DSN)


def test_safe_label_masks_postgres_password():
    store = _mock_pg_store()
    assert store.safe_label == "postgres"
    assert "Rama1998" not in store.safe_label


def test_safe_label_passes_through_sqlite_path(tmp_path):
    db_file = tmp_path / "predictions.db"
    store = RequestStore(db_path=str(db_file))
    assert store.safe_label == str(db_file)


def test_startup_log_never_contains_password(caplog):
    with caplog.at_level("INFO"):
        _mock_pg_store()
    leaked = [r.message for r in caplog.records if "Rama1998" in r.getMessage()]
    assert leaked == [], f"password leaked in log records: {leaked}"
