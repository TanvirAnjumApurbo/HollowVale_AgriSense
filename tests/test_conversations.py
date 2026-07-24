"""Offline tests for memory/conversations.py.

Pure logic (title derivation) is fully covered; database-dependent functions
are exercised in degraded mode (no DATABASE_URL) and for guest users
(user_id None): they must return empty/None/False without raising. Live
round-trips are verified against the real Neon database on deployment.
"""

import pytest

from memory import conversations as convs, db


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.close_pool()
    yield
    db.close_pool()


def test_derive_title_short_message_passes_through():
    assert convs.derive_title("Plan my boro season") == "Plan my boro season"


def test_derive_title_collapses_whitespace():
    assert convs.derive_title("  hello\n\n  world\t ") == "hello world"


def test_derive_title_truncates_at_word_boundary_with_ellipsis():
    text = (
        "I have two acres of clay loam in Rangpur and I want to know "
        "what to plant this season with a budget of 80000 taka"
    )
    title = convs.derive_title(text)
    assert len(title) <= 49  # 48 + ellipsis char
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")
    assert title[:-1] in text  # a clean prefix, not chopped mid-word


def test_derive_title_empty_falls_back():
    assert convs.derive_title("") == "New chat"
    assert convs.derive_title(None) == "New chat"
    assert convs.derive_title("   \n ") == "New chat"


def test_guest_user_is_fully_short_circuited():
    assert convs.list_conversations(None) == []
    assert convs.create_conversation(None, "hi") is None
    assert convs.get_conversation(None, 1) is None
    assert convs.load_messages(None, 1) is None
    assert convs.append_turn(None, 1, "u", "a", []) is False
    assert convs.save_state(None, 1, {}, {}) is False


def test_no_database_degrades_without_raising():
    assert convs.list_conversations(42) == []
    assert convs.create_conversation(42, "first message") is None
    assert convs.get_conversation(42, 1) is None
    assert convs.load_messages(42, 1) is None
    assert convs.append_turn(42, 1, "u", "a", [{"tool": "x"}]) is False
    assert convs.save_state(42, 1, {"location": "Rangpur"}, {}) is False
