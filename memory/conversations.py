"""Per-user conversation persistence: transcripts, traces, and agent state.

Each conversation row owns the two structures that ARE the agent's memory --
``farmer_profile`` and ``session_facts`` (weather, ranking, chosen crop,
projection, calendar) -- as JSONB, plus its messages. Assistant messages
carry the visible tool-call trace slice for their turn, so reopening a
conversation restores both the transcript and the full agent working state.

Every read is scoped by ``user_id`` so one user can never load another
user's conversations. All functions degrade gracefully (None/False/[])
when the database is unavailable, consistent with memory/db.py.
"""

import json
import re

from psycopg.types.json import Jsonb

from memory import db


def _jsonb(obj):
    # default=str mirrors the orchestrator's own json.dumps of tool results.
    return Jsonb(obj, dumps=lambda o: json.dumps(o, default=str))


def derive_title(text, max_len=48):
    """A short deterministic title from the first user message."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return "New chat"
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut + "…"


def list_conversations(user_id):
    """[{id, title, updated_at}] newest-first; [] for guests or DB failure."""
    if user_id is None:
        return []
    rows = db.fetch_all(
        """
        SELECT id, title, updated_at
        FROM conversations
        WHERE user_id = %s
        ORDER BY updated_at DESC
        """,
        (user_id,),
    )
    return rows or []


def create_conversation(user_id, first_message=None):
    """Create a conversation titled from the first message; id or None."""
    if user_id is None:
        return None
    row = db.fetch_one(
        """
        INSERT INTO conversations (user_id, title)
        VALUES (%s, %s)
        RETURNING id
        """,
        (user_id, derive_title(first_message)),
    )
    return row["id"] if row else None


def get_conversation(user_id, conversation_id):
    """Owner-scoped conversation row (title + agent state), or None."""
    if user_id is None or conversation_id is None:
        return None
    return db.fetch_one(
        """
        SELECT id, title, farmer_profile, session_facts
        FROM conversations
        WHERE id = %s AND user_id = %s
        """,
        (conversation_id, user_id),
    )


def load_messages(user_id, conversation_id):
    """Ordered [{role, content, trace}] for an owned conversation, else None."""
    if get_conversation(user_id, conversation_id) is None:
        return None
    return db.fetch_all(
        """
        SELECT role, content, trace
        FROM messages
        WHERE conversation_id = %s
        ORDER BY id
        """,
        (conversation_id,),
    )


def append_turn(user_id, conversation_id, user_message, assistant_message, trace):
    """Persist one completed turn (user + assistant + trace) in one
    transaction, so a half-saved turn can never appear on reload."""
    if user_id is None or conversation_id is None:
        return False
    pool = db.get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            with conn.transaction():
                owned = conn.execute(
                    "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                ).fetchone()
                if not owned:
                    return False
                conn.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (%s, 'user', %s)
                    """,
                    (conversation_id, user_message),
                )
                conn.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, trace)
                    VALUES (%s, 'assistant', %s, %s)
                    """,
                    (conversation_id, assistant_message, _jsonb(trace or [])),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %s",
                    (conversation_id,),
                )
        return True
    except Exception as exc:
        print(f"[memory.conversations] append_turn failed: {db.redact(exc)}")
        return False


def save_state(user_id, conversation_id, farmer_profile, session_facts):
    """Persist the agent's working memory for follow-ups after reopening."""
    if user_id is None or conversation_id is None:
        return False
    return db.execute(
        """
        UPDATE conversations
        SET farmer_profile = %s, session_facts = %s, updated_at = now()
        WHERE id = %s AND user_id = %s
        """,
        (
            _jsonb(farmer_profile or {}),
            _jsonb(session_facts or {}),
            conversation_id,
            user_id,
        ),
    )
