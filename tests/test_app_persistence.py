"""AppTest coverage of app.py's conversation persistence logic.

The Neon layer (memory.conversations) is replaced by an in-memory fake and
run_turn by a deterministic stub, so these tests exercise the real app.py
wiring -- boot-time restore, conversation creation on first message,
save-exactly-once per turn, agent-state save/restore, and per-user isolation
-- without any database, network, or LLM.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

from memory import conversations as convs, db

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")

ALICE = {"id": 1, "username": "alice"}
BOB = {"id": 2, "username": "bob"}


class FakeConvStore:
    """In-memory stand-in for the Neon-backed conversations module."""

    def __init__(self):
        self.rows = {}  # id -> {user_id, title, profile, facts, messages}
        self.next_id = 1
        self.clock = 0
        self.append_calls = 0

    def list_conversations(self, user_id):
        rows = [
            {"id": cid, "title": r["title"], "updated_at": r["updated"]}
            for cid, r in self.rows.items()
            if r["user_id"] == user_id
        ]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)

    def create_conversation(self, user_id, first_message=None):
        if user_id is None:
            return None
        cid = self.next_id
        self.next_id += 1
        self.clock += 1
        self.rows[cid] = {
            "user_id": user_id,
            "title": convs.derive_title(first_message),
            "profile": {},
            "facts": {},
            "messages": [],
            "updated": self.clock,
        }
        return cid

    def get_conversation(self, user_id, conversation_id):
        row = self.rows.get(conversation_id)
        if not row or row["user_id"] != user_id:
            return None
        return {
            "id": conversation_id,
            "title": row["title"],
            "farmer_profile": row["profile"],
            "session_facts": row["facts"],
        }

    def load_messages(self, user_id, conversation_id):
        if self.get_conversation(user_id, conversation_id) is None:
            return None
        return list(self.rows[conversation_id]["messages"])

    def append_turn(self, user_id, conversation_id, user_msg, asst_msg, trace):
        if self.get_conversation(user_id, conversation_id) is None:
            return False
        self.append_calls += 1
        self.clock += 1
        row = self.rows[conversation_id]
        row["messages"].append({"role": "user", "content": user_msg, "trace": None})
        row["messages"].append(
            {"role": "assistant", "content": asst_msg, "trace": list(trace or [])}
        )
        row["updated"] = self.clock
        return True

    def save_state(self, user_id, conversation_id, profile, facts):
        if self.get_conversation(user_id, conversation_id) is None:
            return False
        row = self.rows[conversation_id]
        row["profile"] = dict(profile or {})
        row["facts"] = dict(facts or {})
        return True


def fake_run_turn(history, user_message, profile, trace_log, session_facts=None):
    trace_log.append(
        {
            "tool": "fake_tool",
            "arguments": {"echo": user_message},
            "result": {"reasons": ["fake reason"]},
        }
    )
    profile["location"] = "Rangpur"
    if session_facts is not None:
        session_facts["weather"] = {"summary": {"total_rainfall_mm": 6.0}}
    reply = f"echo: {user_message}"
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return reply


@pytest.fixture
def fake_store(monkeypatch):
    # Empty string (not delenv): app.py's load_dotenv() would re-inject a
    # developer's real DATABASE_URL from .env, but it never overrides an
    # existing env var, and blank counts as unconfigured.
    monkeypatch.setenv("DATABASE_URL", "")
    db.close_pool()
    store = FakeConvStore()
    for name in (
        "list_conversations",
        "create_conversation",
        "get_conversation",
        "load_messages",
        "append_turn",
        "save_state",
    ):
        monkeypatch.setattr(convs, name, getattr(store, name))
    import agent.orchestrator as orch

    monkeypatch.setattr(orch, "run_turn", fake_run_turn)
    yield store
    db.close_pool()


def _boot(user):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["auth_user"] = dict(user)
    return at.run()


def test_first_message_creates_and_saves_conversation(fake_store):
    at = _boot(ALICE)
    assert not at.exception
    at = at.chat_input[0].set_value("I have 2 acres in Rangpur").run()
    assert not at.exception

    assert fake_store.append_calls == 1, "turn must be saved exactly once"
    listed = fake_store.list_conversations(ALICE["id"])
    assert len(listed) == 1
    assert listed[0]["title"].startswith("I have 2 acres")

    saved = fake_store.rows[listed[0]["id"]]
    roles = [m["role"] for m in saved["messages"]]
    assert roles == ["user", "assistant"]
    assert saved["messages"][1]["trace"], "assistant turn must carry its trace"
    assert saved["profile"].get("location") == "Rangpur"
    assert saved["facts"].get("weather"), "agent facts must be persisted"


def test_boot_restores_most_recent_conversation(fake_store):
    at = _boot(ALICE)
    at = at.chat_input[0].set_value("plan my boro season").run()

    # Fresh browser session (new AppTest): transcript + memory must return.
    at2 = _boot(ALICE)
    assert not at2.exception
    texts = [
        " ".join(md.value for md in m.markdown)
        for m in at2.chat_message
        if m.markdown
    ]
    assert any("plan my boro season" in t for t in texts)
    assert any("echo: plan my boro season" in t for t in texts)
    assert at2.session_state["farmer_profile"].get("location") == "Rangpur"
    assert at2.session_state["session_facts"].get("weather")
    assert at2.session_state["trace_log"], "trace must be restored"
    turn_traces = at2.session_state["turn_traces"]
    assert len(turn_traces) == 1, "one assistant turn -> one trace slice"
    assert turn_traces[0][0]["tool"] == "fake_tool"


def test_users_cannot_see_each_others_chats(fake_store):
    at = _boot(ALICE)
    at.chat_input[0].set_value("alice's secret farm plan").run()

    at_bob = _boot(BOB)
    assert not at_bob.exception
    conv_buttons = [b for b in at_bob.button if (b.key or "").startswith("conv_")]
    assert conv_buttons == [], "bob must not see alice's conversations"
    texts = [
        " ".join(md.value for md in m.markdown)
        for m in at_bob.chat_message
        if m.markdown
    ]
    assert not any("alice's secret" in t for t in texts)
    # And the fake store confirms scoping directly.
    assert fake_store.list_conversations(BOB["id"]) == []


def test_new_chat_starts_clean_without_touching_old_chats(fake_store):
    at = _boot(ALICE)
    at = at.chat_input[0].set_value("first conversation").run()
    before = {cid: len(r["messages"]) for cid, r in fake_store.rows.items()}

    new_chat = [b for b in at.button if b.key == "new_chat_btn"]
    at = new_chat[0].click().run()
    assert not at.exception
    assert at.session_state["conversation_history"] == []
    assert at.session_state["active_conversation_id"] is None
    after = {cid: len(r["messages"]) for cid, r in fake_store.rows.items()}
    assert after == before, "new chat must not modify stored conversations"

    # Second conversation persists independently.
    at = at.chat_input[0].set_value("second conversation").run()
    assert len(fake_store.list_conversations(ALICE["id"])) == 2


def test_switching_back_to_old_conversation_restores_it(fake_store):
    at = _boot(ALICE)
    at = at.chat_input[0].set_value("first conversation").run()
    first_id = fake_store.list_conversations(ALICE["id"])[0]["id"]

    new_chat = [b for b in at.button if b.key == "new_chat_btn"]
    at = new_chat[0].click().run()
    at = at.chat_input[0].set_value("second conversation").run()

    old_btn = [b for b in at.button if b.key == f"conv_{first_id}"]
    assert old_btn and not old_btn[0].disabled
    at = old_btn[0].click().run()
    assert not at.exception
    texts = [
        " ".join(md.value for md in m.markdown)
        for m in at.chat_message
        if m.markdown
    ]
    assert any("first conversation" in t for t in texts)
    assert not any("second conversation" in t for t in texts)
    assert at.session_state["active_conversation_id"] == first_id
