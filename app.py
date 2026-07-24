"""AgriSense AI -- Streamlit entrypoint.

Chat UI on the left driven by agent.orchestrator.run_turn(); a sidebar
trace panel renders every tool call made this session (name, arguments,
raw returned values) so a judge can confirm each number in a plan came
from a real tool call, not the model's imagination.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Local dev convenience: load a local .env if present. On Streamlit Cloud
# there is no .env -- credentials come from the Secrets manager instead.
load_dotenv()

# Base URL of the bdapps CaaS sidecar (server.py). The Streamlit app cannot
# serve bdapps' inbound callbacks itself, so payments go through the sidecar.
BDAPPS_SIDECAR_URL = os.environ.get("BDAPPS_SIDECAR_URL", "http://localhost:8000").rstrip("/")

from agent.orchestrator import run_turn
from agent.prompts import missing_fields
from memory.session_store import (
    init_session,
    get_conversation_history,
    get_farmer_profile,
    get_session_facts,
    get_trace_log,
    reset_session,
)

st.set_page_config(page_title="AgriSense AI", page_icon="🌾", layout="wide")
init_session()

with st.sidebar:
    st.header("🔍 Agent trace")
    st.caption("Every tool call this agent has made, with real parameters and raw returned values.")
    st.caption("Weather data: [Open-Meteo](https://open-meteo.com/)")

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
    with st.expander("💳 Pay via bdapps (CaaS sandbox)", expanded=False):
        st.caption(
            "Charging-as-a-Service demo. Sends a charge to the bdapps sidecar "
            "(`server.py`); in simulate mode no real money or credentials are used."
        )
        proj = get_session_facts().get("projection") or {}
        fert = (proj.get("cost_breakdown_bdt") or {}).get("fertilizer")
        if fert:
            st.caption(f"Tip: this plan's fertilizer input cost is {fert:.0f} BDT.")
        pay_msisdn = st.text_input("Mobile number", value="01700000000", key="bdapps_msisdn")
        pay_amount = st.number_input(
            "Amount (BDT)", min_value=1.0, value=20.0, step=1.0, key="bdapps_amount"
        )
        pay_desc = st.text_input("For", value="AgriSense season advisory", key="bdapps_desc")
        if st.button("Charge via bdapps", key="bdapps_charge"):
            try:
                resp = requests.post(
                    f"{BDAPPS_SIDECAR_URL}/bdapps/checkout",
                    json={
                        "msisdn": pay_msisdn,
                        "amount": float(pay_amount),
                        "description": pay_desc,
                    },
                    timeout=20,
                )
                data = resp.json()
            except Exception as exc:
                st.error(
                    f"Could not reach the bdapps sidecar at {BDAPPS_SIDECAR_URL}. "
                    f"Is it running (`uvicorn server:app --port 8000`)? ({exc})"
                )
            else:
                if data.get("status") == "CHARGED":
                    st.success(
                        f"Charged {data['amount']} {data['currency']} "
                        f"({data['status_code']})"
                    )
                    if data.get("balance_after") is not None:
                        st.caption(f"Balance after: {data['balance_after']} {data['currency']}")
                    receipt = data.get("receipt_url", "")
                    if receipt.startswith("/"):
                        receipt = f"{BDAPPS_SIDECAR_URL}{receipt}"
                    st.markdown(f"[View receipt]({receipt}) · ref `{data['reference']}`")
                else:
                    st.error(f"{data.get('status')}: {data.get('status_detail')}")

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
    # run_turn appends the user + assistant messages to conversation history
    # on success. On failure it raises *before* appending, so we record the
    # user's message AND the error here -- otherwise the rerun below redraws
    # history from scratch and the failed turn (message + error) silently
    # disappears, making the app look like it ignored the user.
    with st.spinner("Thinking..."):
        try:
            run_turn(
                get_conversation_history(),
                user_input,
                get_farmer_profile(),
                get_trace_log(),
                get_session_facts(),
            )
        except Exception as e:
            hist = get_conversation_history()
            hist.append({"role": "user", "content": user_input})
            hist.append({
                "role": "assistant",
                "content": f"Something went wrong on my end: `{e}`. Please try again or rephrase.",
            })
            get_trace_log().append({
                "tool": "ERROR",
                "arguments": {"user_input": user_input},
                "result": {"exception": repr(e)},
            })

    # The history loop at the top of the script renders every turn (including
    # anything just appended) on this rerun -- so no manual st.chat_message
    # blocks here, which would otherwise double-draw each turn.
    st.rerun()
