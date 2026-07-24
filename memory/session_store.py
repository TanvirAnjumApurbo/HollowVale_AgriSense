"""Conversation/profile state for the current session.

Backed by st.session_state for Tier 0. The function signatures here are
the seam for Tier 1 persistent memory: swap the bodies to read/write
SQLite (or Supabase Postgres) keyed by farmer phone/id, and nothing in
app.py or agent/orchestrator.py needs to change.
"""

import streamlit as st


def init_session():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "farmer_profile" not in st.session_state:
        st.session_state.farmer_profile = {}
    if "trace_log" not in st.session_state:
        st.session_state.trace_log = []


def get_conversation_history():
    return st.session_state.conversation_history


def get_farmer_profile():
    return st.session_state.farmer_profile


def get_trace_log():
    return st.session_state.trace_log


def reset_session():
    st.session_state.conversation_history = []
    st.session_state.farmer_profile = {}
    st.session_state.trace_log = []
