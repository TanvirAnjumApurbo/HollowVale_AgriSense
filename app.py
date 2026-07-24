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
from memory import auth, db
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

# Best-effort Neon persistence bootstrap. ensure_schema() is idempotent,
# latches after the first success, and backs off after failures, so this
# per-rerun call is effectively free. A missing/unreachable database leaves
# the app fully functional in session-only mode (errors are redacted and go
# to stderr inside memory.db -- never the UI).
db.ensure_schema()


# --- Authentication gate -------------------------------------------------
# DB-backed sessions: the raw token travels only in the browser's URL query
# param; the database stores its SHA-256 digest with an expiry. Guest mode
# exists ONLY when no DATABASE_URL is configured at all (local dev/tests) --
# the deployed app always requires login.

def _establish_session(user):
    token = auth.create_session(user["id"])
    st.session_state["auth_user"] = user
    if token:
        st.session_state["auth_token"] = token
        st.query_params["session"] = token
    # Never inherit another user's in-memory chat on a shared browser tab.
    reset_session()
    st.rerun()


def _logout():
    auth.delete_session(st.session_state.get("auth_token"))
    for key in ("auth_user", "auth_token"):
        st.session_state.pop(key, None)
    try:
        del st.query_params["session"]
    except KeyError:
        pass
    reset_session()
    st.rerun()


def _resolve_user():
    """The authenticated user for this browser session, or None."""
    user = st.session_state.get("auth_user")
    if user:
        return user
    token = st.query_params.get("session")
    if token:
        user = auth.get_session_user(token)
        if user:
            st.session_state["auth_user"] = user
            st.session_state["auth_token"] = token
            return user
        try:
            del st.query_params["session"]
        except KeyError:
            pass
    return None


def _login_screen():
    st.title("🌾 AgriSense AI")
    st.caption("Sign in to build grounded, costed season plans for your farm.")

    if not db.is_configured():
        st.info(
            "No database is configured (`DATABASE_URL` is unset), so login and "
            "saved conversations are unavailable. You can continue as a guest "
            "for this browser session only."
        )
        if st.button("Continue as guest (session-only)"):
            st.session_state["auth_user"] = {"id": None, "username": "guest"}
            reset_session()
            st.rerun()
        st.stop()

    signin_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with signin_tab:
        with st.form("signin_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user = auth.authenticate(username, password)
            if user:
                _establish_session(user)
            elif not db.ping():
                st.error(
                    "The database is unreachable right now. "
                    "Please try again in a minute."
                )
            else:
                st.error("Incorrect username or password.")
    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account", type="primary")
        if created:
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                user, error = auth.create_user(new_username, new_password)
                if user:
                    _establish_session(user)
                else:
                    st.error(error)
    st.stop()


current_user = _resolve_user()
if current_user is None:
    _login_screen()

with st.sidebar:
    account_col, logout_col = st.columns([3, 1])
    account_col.markdown(f"👤 **{current_user['username']}**")
    if logout_col.button("Log out", key="logout_btn"):
        _logout()
    st.divider()

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
