"""AgriSense AI -- Streamlit entrypoint.

Chat UI on the left driven by agent.orchestrator.run_turn(); a sidebar
trace panel renders every tool call made this session (name, arguments,
raw returned values) so a judge can confirm each number in a plan came
from a real tool call, not the model's imagination.
"""

import streamlit as st

from agent.orchestrator import run_turn
from agent.prompts import missing_fields
from memory.session_store import (
    init_session,
    get_conversation_history,
    get_farmer_profile,
    get_trace_log,
    reset_session,
)

st.set_page_config(page_title="AgriSense AI", page_icon="🌾", layout="wide")
init_session()

with st.sidebar:
    st.header("🔍 Agent trace")
    st.caption("Every tool call this agent has made, with real parameters and raw returned values.")

    profile = get_farmer_profile()
    st.subheader("Farmer profile (known so far)")
    if profile:
        for k, v in profile.items():
            st.write(f"**{k}**: {v}")
    else:
        st.write("_(nothing known yet)_")
    still_missing = missing_fields(profile)
    if still_missing:
        st.caption("Still missing: " + ", ".join(still_missing))

    st.divider()
    st.subheader(f"Tool calls ({len(get_trace_log())})")
    for i, entry in enumerate(reversed(get_trace_log())):
        with st.expander(f"{len(get_trace_log()) - i}. {entry['tool']}", expanded=False):
            st.markdown("**Arguments:**")
            st.json(entry["arguments"])
            st.markdown("**Raw result:**")
            st.json(entry["result"])

    st.divider()
    if st.button("Reset conversation"):
        reset_session()
        st.rerun()

st.title("🌾 AgriSense AI")
st.caption("From a vague opening message to a grounded, explained, costed season plan.")

for msg in get_conversation_history():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not get_conversation_history():
    with st.chat_message("assistant"):
        st.markdown(
            "Hi, I'm AgriSense AI. Tell me a bit about your farm -- where it is, "
            "how big it is, and what you're hoping to plant -- and I'll help you "
            "build a costed, weather-aware season plan."
        )

user_input = st.chat_input("Tell me about your farm...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = run_turn(
                    get_conversation_history(),
                    user_input,
                    get_farmer_profile(),
                    get_trace_log(),
                )
            except Exception as e:
                reply = f"Something went wrong on my end: `{e}`. Please try again or rephrase."
        st.markdown(reply)

    st.rerun()
