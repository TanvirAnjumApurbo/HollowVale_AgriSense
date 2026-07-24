"""Conversation, profile, trace, and durable facts for the current session.

The store is backed by ``st.session_state``.  Mapping-style access is used
deliberately: Streamlit's session-state proxy supports it, and tests can
replace ``st.session_state`` with an ordinary dictionary.

The functions in this module are the persistence seam.  A later SQLite or
Postgres-backed implementation can keep the public API while changing where
the values are stored.
"""

import streamlit as st


SESSION_FACT_KEYS = (
    "weather",
    "ranking",
    "chosen_crop",
    "projection",
    "calendar",
)
_SESSION_FACTS_STATE_KEY = "session_facts"


def _empty_session_facts():
    """Create a fresh fact record; never share a mutable default."""
    return {key: None for key in SESSION_FACT_KEYS}


def _ensure_session_facts(state):
    """Return the fact dictionary, migrating a partial older record in place."""
    facts = state.get(_SESSION_FACTS_STATE_KEY)
    if not isinstance(facts, dict):
        facts = _empty_session_facts()
        state[_SESSION_FACTS_STATE_KEY] = facts
    else:
        for key in SESSION_FACT_KEYS:
            facts.setdefault(key, None)
    return facts


def init_session():
    state = st.session_state
    if "conversation_history" not in state:
        state["conversation_history"] = []
    if "farmer_profile" not in state:
        state["farmer_profile"] = {}
    if "trace_log" not in state:
        state["trace_log"] = []
    _ensure_session_facts(state)


def get_conversation_history():
    init_session()
    return st.session_state["conversation_history"]


def get_farmer_profile():
    init_session()
    return st.session_state["farmer_profile"]


def get_trace_log():
    init_session()
    return st.session_state["trace_log"]


def get_session_facts():
    """Return the live mutable fact dictionary stored for this session."""
    init_session()
    return st.session_state[_SESSION_FACTS_STATE_KEY]


def get_session_fact(key):
    """Return one known fact, raising ``KeyError`` for an unsupported key."""
    if key not in SESSION_FACT_KEYS:
        raise KeyError(
            f"Unknown session fact '{key}'. "
            f"Supported facts: {', '.join(SESSION_FACT_KEYS)}"
        )
    return get_session_facts()[key]


def set_session_fact(key, value):
    """Store one fact and return the live fact dictionary."""
    if key not in SESSION_FACT_KEYS:
        raise KeyError(
            f"Unknown session fact '{key}'. "
            f"Supported facts: {', '.join(SESSION_FACT_KEYS)}"
        )
    facts = get_session_facts()
    facts[key] = value
    return facts


def update_session_facts(values=None, **updates):
    """Update several facts atomically and return the live fact dictionary.

    ``values`` may be any mapping accepted by ``dict``; keyword updates are
    applied after it.  All keys are validated before the stored record changes.
    """
    changes = {}
    if values is not None:
        changes.update(dict(values))
    changes.update(updates)

    unknown = [key for key in changes if key not in SESSION_FACT_KEYS]
    if unknown:
        supported = ", ".join(SESSION_FACT_KEYS)
        raise KeyError(
            f"Unknown session fact(s): {', '.join(map(str, unknown))}. "
            f"Supported facts: {supported}"
        )

    facts = get_session_facts()
    facts.update(changes)
    return facts


def reset_session_facts():
    """Reset only durable tool facts, preserving chat/profile/trace state."""
    st.session_state[_SESSION_FACTS_STATE_KEY] = _empty_session_facts()


def reset_session():
    state = st.session_state
    state["conversation_history"] = []
    state["farmer_profile"] = {}
    state["trace_log"] = []
    state[_SESSION_FACTS_STATE_KEY] = _empty_session_facts()
