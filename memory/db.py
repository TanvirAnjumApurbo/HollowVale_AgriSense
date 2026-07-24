"""Neon PostgreSQL foundation for AgriSense persistence.

Resolution and behavior mirror the repo's existing conventions:

- ``DATABASE_URL`` resolves Streamlit secrets first, then environment
  variables (same defensive pattern as ``agent/llm.py::_get_secret``), so the
  app works both on Streamlit Cloud and from a local ``.env``.
- One process-wide connection pool survives Streamlit reruns (module state
  persists across reruns; no per-rerun reconnect).  ``check_connection`` on
  checkout transparently replaces connections Neon closed while idle.
- ``ensure_schema()`` is idempotent (``CREATE TABLE IF NOT EXISTS``) and safe
  to call on every rerun: it latches after the first success and backs off
  after a failure so a down database cannot add connect latency to each rerun.
- Every public function degrades gracefully: with no ``DATABASE_URL``
  configured, or the database unreachable, it returns ``None``/``False``
  instead of raising, and any error text passes through ``redact()`` so
  credentials never reach logs or the UI.

Run ``python -m memory.db`` for a deterministic connectivity/schema
self-check (prints status only -- never the connection string).
"""

import os
import re
import sys
import threading
import time

# All DDL is idempotent; executed in one transaction by ensure_schema().
# The full target schema is created up front (users, conversations, messages,
# usage) so later phases add code, not migrations.
SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL DEFAULT 'New chat',
        farmer_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
        session_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
        ON conversations (user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        conversation_id BIGINT NOT NULL
            REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        trace JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages (conversation_id, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_sessions (
        token_hash TEXT PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        expires_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
        ON auth_sessions (user_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_usage (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        used_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_usage_user_time
        ON agent_usage (user_id, used_at)
    """,
)

_FAILURE_BACKOFF_SECONDS = 60.0

_pool = None
_pool_url = None
_pool_lock = threading.Lock()
_schema_ready = False
_schema_failed_at = 0.0


def _get_secret(key, default=None):
    """Streamlit secrets first, then env vars (see agent/llm.py for why the
    st.secrets probe must be wrapped defensively)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def get_database_url():
    """The configured connection string, or None. Never log/display this."""
    url = _get_secret("DATABASE_URL")
    return url.strip() if isinstance(url, str) and url.strip() else None


def is_configured():
    return get_database_url() is not None


def redact(text):
    """Strip credentials from any text destined for logs or the UI."""
    text = re.sub(r"://([^:@/\s]+):([^@\s]+)@", r"://\1:***@", str(text))
    url = get_database_url()
    if url:
        text = text.replace(url, "postgresql://***")
    return text


def _warn(context, exc):
    print(f"[memory.db] {context}: {redact(exc)}", file=sys.stderr)


def get_pool():
    """The process-wide connection pool, or None when unconfigured/broken."""
    global _pool, _pool_url
    url = get_database_url()
    if not url:
        return None
    with _pool_lock:
        if _pool is not None and _pool_url == url and not _pool.closed:
            return _pool
        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                url,
                min_size=0,
                max_size=4,
                open=False,
                timeout=float(os.environ.get("AGRISENSE_DB_TIMEOUT", "10")),
                check=ConnectionPool.check_connection,
                kwargs={"autocommit": True, "connect_timeout": 8},
            )
            pool.open()
        except Exception as exc:
            _warn("could not create connection pool", exc)
            return None
        _pool, _pool_url = pool, url
        return pool


def close_pool():
    """Close and forget the pool (tests / URL rotation)."""
    global _pool, _pool_url, _schema_ready, _schema_failed_at
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
        _pool = None
        _pool_url = None
        _schema_ready = False
        _schema_failed_at = 0.0


def execute(sql, params=None):
    """Run one statement. True on success, False when the DB is unavailable."""
    pool = get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(sql, params)
        return True
    except Exception as exc:
        _warn("execute failed", exc)
        return False


def fetch_all(sql, params=None):
    """Rows as list-of-dicts; [] for an empty result, None when unavailable."""
    pool = get_pool()
    if pool is None:
        return None
    try:
        from psycopg.rows import dict_row

        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return cur.fetchall()
    except Exception as exc:
        _warn("query failed", exc)
        return None


def fetch_one(sql, params=None):
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def ping():
    """True when a round-trip to the database succeeds."""
    row = fetch_one("SELECT 1 AS ok")
    return bool(row and row.get("ok") == 1)


def ensure_schema(force=False):
    """Create all persistence structures idempotently.

    Latches after the first success (so per-rerun calls are free) and backs
    off for a minute after a failure (so a down database cannot stall every
    Streamlit rerun on connect timeouts). Returns True when the schema is
    known to be in place.
    """
    global _schema_ready, _schema_failed_at
    if _schema_ready and not force:
        return True
    if (
        not force
        and _schema_failed_at
        and time.monotonic() - _schema_failed_at < _FAILURE_BACKOFF_SECONDS
    ):
        return False

    pool = get_pool()
    if pool is None:
        _schema_failed_at = time.monotonic()
        return False
    try:
        with pool.connection() as conn:
            with conn.transaction():
                for statement in SCHEMA_STATEMENTS:
                    conn.execute(statement)
        _schema_ready = True
        _schema_failed_at = 0.0
        return True
    except Exception as exc:
        _warn("schema initialization failed", exc)
        _schema_failed_at = time.monotonic()
        return False


def status():
    """Sanitized snapshot for self-checks and diagnostics (no secrets)."""
    result = {
        "configured": is_configured(),
        "connected": False,
        "schema_ready": False,
    }
    if result["configured"]:
        result["connected"] = ping()
        if result["connected"]:
            result["schema_ready"] = ensure_schema()
    return result


if __name__ == "__main__":
    # Deterministic self-check, in the style of `python -m tools.agronomy`.
    from dotenv import load_dotenv

    load_dotenv()
    snapshot = status()
    print(f"configured   : {snapshot['configured']}")
    print(f"connected    : {snapshot['connected']}")
    print(f"schema_ready : {snapshot['schema_ready']}")
    if snapshot["schema_ready"]:
        rows = fetch_all(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
            """
        )
        names = ", ".join(r["table_name"] for r in rows or [])
        print(f"tables       : {names}")
        assert {"users", "conversations", "messages", "agent_usage"} <= {
            r["table_name"] for r in rows or []
        }, "expected persistence tables missing"
        print("SELF-CHECK OK")
    elif not snapshot["configured"]:
        print("DATABASE_URL is not set (Streamlit secrets or env). "
              "Running in session-only mode is expected without it.")
        print("SELF-CHECK OK (degraded mode)")
    else:
        print("SELF-CHECK FAILED: configured but could not connect/init.")
        sys.exit(1)
