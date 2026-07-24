"""SQLite-backed transaction + simulated-balance store for bdapps CaaS.

Payments are stateless HTTP calls keyed by a transaction *reference*, unlike
the Streamlit chat which is keyed by browser session -- so this store is
separate from ``memory/session_store.py``. It is the persistence seam for the
sidecar: swap SQLite for Postgres here and neither the client nor the server
changes.

The database path is read live from settings, so tests can point it at a temp
file. A partial index is harmless here (unlike the RAG store); the file is
gitignored via the ``*.sqlite3``/db rules.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from bdapps.config import get_settings


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    path = get_settings().db_path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if absent. Idempotent; safe to call on every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                reference     TEXT PRIMARY KEY,
                msisdn        TEXT NOT NULL,
                amount        REAL NOT NULL,
                currency      TEXT NOT NULL,
                description   TEXT,
                status        TEXT NOT NULL,
                status_code   TEXT,
                status_detail TEXT,
                provider_ref  TEXT,
                balance_after REAL,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                request_json  TEXT,
                response_json TEXT,
                notify_json   TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sim_balances (
                msisdn  TEXT PRIMARY KEY,
                balance REAL NOT NULL
            )
            """
        )


def create_transaction(reference, msisdn, amount, currency, description, request_json=None):
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO transactions
                (reference, msisdn, amount, currency, description, status,
                 created_at, updated_at, request_json)
            VALUES (?, ?, ?, ?, ?, 'INITIATED', ?, ?, ?)
            """,
            (reference, msisdn, float(amount), currency, description, now, now, request_json),
        )
    return get_transaction(reference)


def update_transaction(reference, **fields):
    """Update selected columns on a transaction; always bumps updated_at."""
    allowed = {
        "status",
        "status_code",
        "status_detail",
        "provider_ref",
        "balance_after",
        "response_json",
        "notify_json",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    sets["updated_at"] = _now_iso()
    columns = ", ".join(f"{k} = ?" for k in sets)
    values = list(sets.values()) + [reference]
    with _connect() as conn:
        conn.execute(f"UPDATE transactions SET {columns} WHERE reference = ?", values)
    return get_transaction(reference)


def get_transaction(reference):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM transactions WHERE reference = ?", (reference,)
        ).fetchone()
    return dict(row) if row else None


def list_transactions(limit=25):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


# --- Simulated operator balances (simulate mode only) ---------------------

def ensure_subscriber(msisdn, initial_balance):
    """Seed a test subscriber's balance on first contact; return current balance."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT balance FROM sim_balances WHERE msisdn = ?", (msisdn,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sim_balances (msisdn, balance) VALUES (?, ?)",
                (msisdn, float(initial_balance)),
            )
            return float(initial_balance)
        return float(row["balance"])


def get_balance(msisdn):
    with _connect() as conn:
        row = conn.execute(
            "SELECT balance FROM sim_balances WHERE msisdn = ?", (msisdn,)
        ).fetchone()
    return float(row["balance"]) if row else None


def debit(msisdn, amount):
    """Atomically deduct amount if funds suffice.

    Returns (ok, balance_after). ok is False with the unchanged balance when
    the subscriber has insufficient funds.
    """
    amount = float(amount)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT balance FROM sim_balances WHERE msisdn = ?", (msisdn,)
        ).fetchone()
        balance = float(row["balance"]) if row else 0.0
        if row is None or balance < amount:
            conn.execute("ROLLBACK")
            return False, balance
        new_balance = round(balance - amount, 2)
        conn.execute(
            "UPDATE sim_balances SET balance = ? WHERE msisdn = ?", (new_balance, msisdn)
        )
        conn.execute("COMMIT")
    return True, new_balance
