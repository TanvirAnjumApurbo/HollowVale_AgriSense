"""User accounts and login sessions backed by the Neon database.

Design (deliberately self-contained -- no OAuth, no external identity
provider, no new dependencies):

- Passwords are hashed with stdlib PBKDF2-HMAC-SHA256 (per-user random salt,
  600k iterations -- OWASP's current PBKDF2 recommendation) and compared in
  constant time. Plain-text passwords are never stored or logged.
- A login creates a random 256-bit session token. Only the SHA-256 digest of
  the token is stored; the raw token lives in the browser (URL query param)
  and expires server-side after ``AUTH_SESSION_DAYS`` (default 7).
- Every function degrades gracefully when the database is unavailable
  (returns None / (False, message)) -- consistent with memory/db.py.

CLI for demo users (reads .env for DATABASE_URL):

    python -m memory.auth create <username> <password>
    python -m memory.auth list
"""

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from memory import db

PBKDF2_ITERATIONS = 600_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$")
MIN_PASSWORD_LEN = 6

# Verified against when a username doesn't exist, so the response time of
# authenticate() doesn't reveal which usernames are registered.
_DUMMY_HASH = None


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """Constant-time verification; False (never an exception) on bad input."""
    try:
        algo, iterations, salt_hex, hash_hex = str(stored).split("$")
        if algo != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(candidate, bytes.fromhex(hash_hex))
    except Exception:
        return False


def validate_credentials(username, password):
    """Return an error message for unusable credentials, else None."""
    if not _USERNAME_RE.match(username or ""):
        return (
            "Username must be 3-32 characters: letters, digits, dot, "
            "underscore or hyphen, starting with a letter or digit."
        )
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    return None


def create_user(username, password):
    """Create an account. Returns (user_dict, None) or (None, error_message)."""
    username = (username or "").strip()
    error = validate_credentials(username, password)
    if error:
        return None, error
    if not db.ensure_schema():
        return None, "The account database is unavailable. Please try again."
    existing = db.fetch_one(
        "SELECT id FROM users WHERE lower(username) = lower(%s)", (username,)
    )
    if existing:
        return None, "That username is already taken."
    row = db.fetch_one(
        """
        INSERT INTO users (username, password_hash)
        VALUES (%s, %s)
        RETURNING id, username
        """,
        (username, hash_password(password)),
    )
    if not row:
        return None, "Could not create the account. Please try again."
    return {"id": row["id"], "username": row["username"]}, None


def authenticate(username, password):
    """Return the user dict for correct credentials, else None."""
    global _DUMMY_HASH
    username = (username or "").strip()
    if not username or not password:
        return None
    row = db.fetch_one(
        """
        SELECT id, username, password_hash
        FROM users WHERE lower(username) = lower(%s)
        """,
        (username,),
    )
    if not row:
        # Burn comparable time for unknown usernames (anti-enumeration).
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password(secrets.token_hex(8))
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_days():
    try:
        return max(1, int(os.environ.get("AUTH_SESSION_DAYS", "7")))
    except ValueError:
        return 7


def create_session(user_id):
    """Create a login session; returns the raw token, or None on DB failure."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=_session_days())
    # Opportunistic cleanup keeps the table from accumulating dead sessions.
    db.execute("DELETE FROM auth_sessions WHERE expires_at < now()")
    ok = db.execute(
        """
        INSERT INTO auth_sessions (token_hash, user_id, expires_at)
        VALUES (%s, %s, %s)
        """,
        (_hash_token(token), user_id, expires),
    )
    return token if ok else None


def get_session_user(token):
    """Resolve a raw token to its user, or None (expired/unknown/DB down)."""
    if not token:
        return None
    row = db.fetch_one(
        """
        SELECT u.id, u.username
        FROM auth_sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = %s AND s.expires_at > now()
        """,
        (_hash_token(token),),
    )
    return {"id": row["id"], "username": row["username"]} if row else None


def delete_session(token):
    if token:
        db.execute(
            "DELETE FROM auth_sessions WHERE token_hash = %s",
            (_hash_token(token),),
        )


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "create":
        user, error = create_user(args[1], args[2])
        if user:
            print(f"Created user '{user['username']}' (id={user['id']}).")
        else:
            print(f"FAILED: {error}")
            sys.exit(1)
    elif len(args) == 1 and args[0] == "list":
        rows = db.fetch_all(
            "SELECT id, username, created_at FROM users ORDER BY id"
        )
        if rows is None:
            print("Database unavailable (is DATABASE_URL set?).")
            sys.exit(1)
        for row in rows:
            print(f"{row['id']:>4}  {row['username']}  ({row['created_at']:%Y-%m-%d})")
        print(f"{len(rows)} user(s).")
    else:
        print("Usage: python -m memory.auth create <username> <password>")
        print("       python -m memory.auth list")
        sys.exit(2)
