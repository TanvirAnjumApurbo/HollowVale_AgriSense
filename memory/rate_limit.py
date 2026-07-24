"""Database-backed usage limiting for agent executions.

Counts real agent runs per authenticated user in a rolling 24-hour window,
stored in Neon (``agent_usage``), so refreshing the browser or re-logging
cannot reset usage. A global cap protects the API budget against many
accounts at once.

Configuration (Streamlit secrets or env vars; values <= 0 disable a check):

- ``RATE_LIMIT_PER_USER_PER_DAY``  (default 20)
- ``RATE_LIMIT_GLOBAL_PER_DAY``    (default 200)

Failure posture: if the database is unavailable the check fails OPEN
(``tracked=False``) -- a Neon outage must not brick the demo -- and guests
(possible only when no DATABASE_URL is configured at all) are never tracked.
"""

from datetime import timedelta

from memory import db
from memory.db import _get_secret

DEFAULT_PER_USER = 20
DEFAULT_GLOBAL = 200

_WINDOW_SQL = "used_at > now() - interval '24 hours'"


def _limit(name, default):
    raw = _get_secret(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_limits():
    return {
        "per_user": _limit("RATE_LIMIT_PER_USER_PER_DAY", DEFAULT_PER_USER),
        "global": _limit("RATE_LIMIT_GLOBAL_PER_DAY", DEFAULT_GLOBAL),
    }


def _retry_at(user_id, limit):
    """When the earliest of the last ``limit`` requests ages out of the
    window, one slot frees up. None when it cannot be determined."""
    if user_id is None:
        row = db.fetch_one(
            f"""
            SELECT used_at FROM agent_usage
            WHERE {_WINDOW_SQL}
            ORDER BY used_at DESC OFFSET %s LIMIT 1
            """,
            (max(0, limit - 1),),
        )
    else:
        row = db.fetch_one(
            f"""
            SELECT used_at FROM agent_usage
            WHERE user_id = %s AND {_WINDOW_SQL}
            ORDER BY used_at DESC OFFSET %s LIMIT 1
            """,
            (user_id, max(0, limit - 1)),
        )
    return row["used_at"] + timedelta(hours=24) if row else None


def check(user_id):
    """Quota snapshot for a user. Always safe to call.

    Returns a dict with: allowed, tracked, used, limit, remaining (None when
    unlimited), retry_at (aware datetime when blocked), blocked_by
    (None | 'user' | 'global'), global_used, global_limit.
    """
    limits = get_limits()
    result = {
        "allowed": True,
        "tracked": True,
        "used": 0,
        "limit": limits["per_user"],
        "remaining": None,
        "retry_at": None,
        "blocked_by": None,
        "global_used": 0,
        "global_limit": limits["global"],
    }
    if user_id is None or not db.is_configured():
        result["tracked"] = False
        return result

    used_row = db.fetch_one(
        f"SELECT count(*) AS n FROM agent_usage WHERE user_id = %s AND {_WINDOW_SQL}",
        (user_id,),
    )
    if used_row is None:  # database unreachable: fail open, untracked
        result["tracked"] = False
        return result
    result["used"] = used_row["n"]

    if limits["per_user"] > 0:
        result["remaining"] = max(0, limits["per_user"] - result["used"])
        if result["used"] >= limits["per_user"]:
            result["allowed"] = False
            result["blocked_by"] = "user"
            result["retry_at"] = _retry_at(user_id, limits["per_user"])
            return result

    if limits["global"] > 0:
        global_row = db.fetch_one(
            f"SELECT count(*) AS n FROM agent_usage WHERE {_WINDOW_SQL}"
        )
        result["global_used"] = global_row["n"] if global_row else 0
        if result["global_used"] >= limits["global"]:
            result["allowed"] = False
            result["blocked_by"] = "global"
            result["retry_at"] = _retry_at(None, limits["global"])
    return result


def record(user_id):
    """Count one real agent execution. No-op (False) for guests / no DB."""
    if user_id is None:
        return False
    return db.execute(
        "INSERT INTO agent_usage (user_id) VALUES (%s)", (user_id,)
    )
