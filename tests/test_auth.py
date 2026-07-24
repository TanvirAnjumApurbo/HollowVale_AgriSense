"""Offline tests for memory/auth.py.

Password hashing/verification and credential validation are pure and fully
tested here. Database-dependent paths (create_user, authenticate, sessions)
are exercised in their degraded no-database mode: they must fail closed,
gracefully, and without exceptions. Live end-to-end auth is verified against
the real Neon database via the app / `python -m memory.auth`.
"""

import pytest

from memory import auth, db


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.close_pool()
    yield
    db.close_pool()


def test_hash_and_verify_roundtrip():
    stored = auth.hash_password("correct horse battery")
    assert auth.verify_password("correct horse battery", stored)
    assert not auth.verify_password("wrong password", stored)


def test_hashes_are_salted_and_never_plaintext():
    a = auth.hash_password("samepassword")
    b = auth.hash_password("samepassword")
    assert a != b
    assert "samepassword" not in a
    assert a.startswith("pbkdf2_sha256$")


def test_verify_rejects_malformed_stored_values_without_raising():
    for bad in (None, "", "plaintext", "a$b$c", "pbkdf2_sha256$x$zz$zz"):
        assert auth.verify_password("anything", bad) is False


@pytest.mark.parametrize(
    "username,password,ok",
    [
        ("alice", "secret1", True),
        ("al", "secret1", False),  # too short a username
        ("a" * 33, "secret1", False),  # too long
        ("bad name", "secret1", False),  # space
        (".dotfirst", "secret1", False),  # must start alnum
        ("alice", "tiny", False),  # short password
        ("farmer.rahim-01", "secret1", True),
    ],
)
def test_validate_credentials(username, password, ok):
    error = auth.validate_credentials(username, password)
    assert (error is None) is ok


def test_create_user_without_database_fails_closed():
    user, error = auth.create_user("alice", "secret123")
    assert user is None
    assert "unavailable" in error.lower()


def test_authenticate_without_database_returns_none():
    assert auth.authenticate("alice", "secret123") is None


def test_sessions_without_database_fail_closed():
    assert auth.create_session(1) is None
    assert auth.get_session_user("sometoken") is None
    assert auth.get_session_user(None) is None
    auth.delete_session("sometoken")  # must not raise


def test_token_hashing_is_deterministic_and_opaque():
    token = "raw-session-token"
    digest = auth._hash_token(token)
    assert digest == auth._hash_token(token)
    assert token not in digest
    assert len(digest) == 64
