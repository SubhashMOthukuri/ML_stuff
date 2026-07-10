"""RequestStore must never leak the raw Postgres DSN (with password) into logs."""

import sys
from unittest.mock import MagicMock

import pytest

from src.serving.store import RequestStore

PG_DSN = "postgresql://nyc_app:test-secret-pw@nyc-rds.abc.us-east-2.rds.amazonaws.com:5432/nyc"
_TEST_PW = "test-secret-pw"


@pytest.fixture()
def mock_psycopg2(monkeypatch):
    """Stub psycopg2 in sys.modules for the duration of one test, then restore."""
    mock_conn = MagicMock()
    fake_psycopg2 = MagicMock()
    fake_psycopg2.connect.return_value = mock_conn
    fake_psycopg2.extras = MagicMock()
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_psycopg2.extras)
    return fake_psycopg2


def test_safe_label_masks_postgres_password(mock_psycopg2):
    store = RequestStore(db_path=PG_DSN)
    assert store.safe_label == "postgres"
    assert _TEST_PW not in store.safe_label


def test_safe_label_passes_through_sqlite_path(tmp_path):
    db_file = tmp_path / "predictions.db"
    store = RequestStore(db_path=str(db_file))
    assert store.safe_label == str(db_file)


def test_startup_log_never_contains_password(mock_psycopg2, caplog):
    with caplog.at_level("INFO"):
        RequestStore(db_path=PG_DSN)
    leaked = [r.message for r in caplog.records if _TEST_PW in r.getMessage()]
    assert leaked == [], f"password leaked in log records: {leaked}"
