"""Offline tests for memory/rate_limit.py plus an AppTest proving a blocked
user cannot trigger the agent.

Live counting/blocking against real Neon is exercised by a separate
verification script (see the Phase 7 report); here the database-dependent
paths run in degraded mode and the app-level enforcement runs against a
patched quota.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

from memory import conversations as convs, db, rate_limit

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.delenv("RATE_LIMIT_PER_USER_PER_DAY", raising=False)
    monkeypatch.delenv("RATE_LIMIT_GLOBAL_PER_DAY", raising=False)
    db.close_pool()
    yield
    db.close_pool()


def test_defaults_and_config_parsing(monkeypatch):
    assert rate_limit.get_limits() == {"per_user": 20, "global": 200}
    monkeypatch.setenv("RATE_LIMIT_PER_USER_PER_DAY", "5")
    monkeypatch.setenv("RATE_LIMIT_GLOBAL_PER_DAY", "50")
    assert rate_limit.get_limits() == {"per_user": 5, "global": 50}
    monkeypatch.setenv("RATE_LIMIT_PER_USER_PER_DAY", "not-a-number")
    assert rate_limit.get_limits()["per_user"] == 20  # safe default


def test_guest_and_no_database_fail_open_untracked():
    snapshot = rate_limit.check(None)
    assert snapshot["allowed"] is True
    assert snapshot["tracked"] is False

    snapshot = rate_limit.check(42)  # DB configured? no -> untracked
    assert snapshot["allowed"] is True
    assert snapshot["tracked"] is False

    assert rate_limit.record(None) is False
    assert rate_limit.record(42) is False  # no DB: nothing recorded


BLOCKED_QUOTA = {
    "allowed": False,
    "tracked": True,
    "used": 20,
    "limit": 20,
    "remaining": 0,
    "retry_at": None,
    "blocked_by": "user",
    "global_used": 20,
    "global_limit": 200,
}

OPEN_QUOTA = dict(BLOCKED_QUOTA, allowed=True, used=0, remaining=20, blocked_by=None)


def test_blocked_user_cannot_trigger_the_agent(monkeypatch):
    calls = {"run_turn": 0, "record": 0}

    import agent.orchestrator as orch

    def forbidden_run_turn(*args, **kwargs):
        calls["run_turn"] += 1
        raise AssertionError("agent must not run when blocked")

    monkeypatch.setattr(orch, "run_turn", forbidden_run_turn)
    monkeypatch.setattr(rate_limit, "check", lambda uid: dict(BLOCKED_QUOTA))
    monkeypatch.setattr(
        rate_limit, "record", lambda uid: calls.__setitem__("record", calls["record"] + 1)
    )
    monkeypatch.setattr(convs, "list_conversations", lambda uid: [])

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["auth_user"] = {"id": 1, "username": "alice"}
    at = at.run()
    at = at.chat_input[0].set_value("plan my season").run()

    assert not at.exception
    assert calls["run_turn"] == 0, "expensive workflow ran despite the block"
    assert calls["record"] == 0, "blocked attempts must not be counted"
    assert at.warning, "the block must be explained to the user"
    warning_text = " ".join(w.value for w in at.warning)
    assert "20/20" in warning_text
    # History/chat UI stays accessible: input still present, no crash.
    assert at.chat_input


def test_open_quota_runs_agent_and_counts_once(monkeypatch):
    calls = {"record": []}

    import agent.orchestrator as orch

    def fake_run_turn(history, user_message, profile, trace_log, session_facts=None):
        trace_log.append(
            {"tool": "fake_tool", "arguments": {}, "result": {"reasons": ["r"]}}
        )
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(orch, "run_turn", fake_run_turn)
    monkeypatch.setattr(rate_limit, "check", lambda uid: dict(OPEN_QUOTA))
    monkeypatch.setattr(rate_limit, "record", lambda uid: calls["record"].append(uid))
    monkeypatch.setattr(convs, "list_conversations", lambda uid: [])
    monkeypatch.setattr(convs, "create_conversation", lambda uid, m=None: None)

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["auth_user"] = {"id": 7, "username": "bob"}
    at = at.run()
    at = at.chat_input[0].set_value("hello").run()

    assert not at.exception
    assert calls["record"] == [7], "exactly one usage record per successful turn"


def test_failed_turn_without_llm_activity_is_not_counted(monkeypatch):
    calls = {"record": []}

    import agent.orchestrator as orch

    def crashing_run_turn(*args, **kwargs):
        raise RuntimeError("LLM unreachable before any call completed")

    monkeypatch.setattr(orch, "run_turn", crashing_run_turn)
    monkeypatch.setattr(rate_limit, "check", lambda uid: dict(OPEN_QUOTA))
    monkeypatch.setattr(rate_limit, "record", lambda uid: calls["record"].append(uid))
    monkeypatch.setattr(convs, "list_conversations", lambda uid: [])
    monkeypatch.setattr(convs, "create_conversation", lambda uid, m=None: None)

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["auth_user"] = {"id": 7, "username": "bob"}
    at = at.run()
    at = at.chat_input[0].set_value("hello").run()

    assert not at.exception
    assert calls["record"] == [], "a turn that never reached the LLM is free"
