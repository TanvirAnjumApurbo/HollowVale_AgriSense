"""Offline tests for the Neon persistence foundation (memory/db.py).

No live Postgres is required: these tests cover configuration resolution,
graceful degradation when the database is missing or unreachable, and the
guarantee that credentials never appear in emitted error text. Live
connectivity is exercised separately by `python -m memory.db` wherever a real
DATABASE_URL is configured (local .env or Streamlit Cloud).
"""

import pytest

from memory import db


@pytest.fixture(autouse=True)
def clean_db_state(monkeypatch):
    """Isolate each test: no ambient DATABASE_URL, no cached pool/latches."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.close_pool()
    yield
    db.close_pool()


def test_unconfigured_is_fully_degraded_and_silent():
    assert db.get_database_url() is None
    assert db.is_configured() is False
    assert db.get_pool() is None
    assert db.ensure_schema() is False
    assert db.ping() is False
    assert db.execute("SELECT 1") is False
    assert db.fetch_all("SELECT 1") is None
    assert db.fetch_one("SELECT 1") is None
    assert db.status() == {
        "configured": False,
        "connected": False,
        "schema_ready": False,
    }


def test_blank_url_counts_as_unconfigured(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "   ")
    assert db.get_database_url() is None
    assert db.is_configured() is False


def test_unreachable_database_degrades_without_leaking_credentials(
    monkeypatch, capsys
):
    # Port 9 (discard) refuses fast; the password below must never surface.
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://agrisense:sup3rsecret@127.0.0.1:9/agrisense",
    )
    monkeypatch.setenv("AGRISENSE_DB_TIMEOUT", "2")

    assert db.is_configured() is True
    assert db.ping() is False
    assert db.ensure_schema() is False

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "sup3rsecret" not in combined


def test_ensure_schema_backs_off_after_failure(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://agrisense:pw@127.0.0.1:9/agrisense",
    )
    monkeypatch.setenv("AGRISENSE_DB_TIMEOUT", "2")

    assert db.ensure_schema() is False
    # Within the backoff window the retry must return immediately (no second
    # connect attempt): well under the 2s acquisition timeout.
    import time

    start = time.monotonic()
    assert db.ensure_schema() is False
    assert time.monotonic() - start < 0.5


def test_redact_masks_password_but_keeps_context():
    noisy = (
        'connection to "postgresql://alice:s3cret@db.example.com/agri" failed'
    )
    cleaned = db.redact(noisy)
    assert "s3cret" not in cleaned
    assert "alice" in cleaned
    assert "db.example.com" in cleaned


def test_redact_masks_full_configured_url(monkeypatch):
    url = "postgresql://bob:hunter2@neon.host/dbname"
    monkeypatch.setenv("DATABASE_URL", url)
    assert "hunter2" not in db.redact(f"error while connecting to {url}")


def test_schema_statements_cover_all_persistence_tables():
    ddl = " ".join(db.SCHEMA_STATEMENTS)
    for table in ("users", "conversations", "messages", "agent_usage"):
        assert table in ddl
    # Idempotency: every CREATE must be IF NOT EXISTS.
    for statement in db.SCHEMA_STATEMENTS:
        assert "IF NOT EXISTS" in statement
