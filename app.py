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
from memory import auth, conversations as convs, db
from memory.session_store import (
    init_session,
    get_conversation_history,
    get_farmer_profile,
    get_session_facts,
    get_trace_log,
    get_turn_traces,
    reset_session,
)

st.set_page_config(page_title="AgriSense AI", page_icon="🌾", layout="wide")
init_session()

# Layout polish the theme config (.streamlit/config.toml) cannot express.
# Only stable selectors are used: data-testid attributes and the st-key-*
# classes Streamlit derives from widget/container keys -- never generated
# class names.
st.markdown(
    """
    <style>
    /* Centered, readable conversation column (ChatGPT-style width) */
    [data-testid="stMainBlockContainer"] {
        max-width: 52rem;
        padding-top: 2.2rem;
        margin: 0 auto;
    }
    /* Chat bubbles: light card feel */
    [data-testid="stChatMessage"] {
        padding: 0.9rem 1.1rem;
        border-radius: 0.9rem;
    }
    /* Sidebar chat list: left-aligned single-line entries with ellipsis */
    .st-key-chat_list button {
        justify-content: flex-start !important;
    }
    .st-key-chat_list button p {
        max-width: 15rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Pin the account row to the sidebar bottom when there is room.
       If these selectors ever stop matching, the row simply sits after
       the other sections -- a graceful degradation. */
    [data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {
        min-height: calc(100vh - 6rem);
    }
    .st-key-user_section {
        margin-top: auto;
    }
    /* Login: centered constrained card */
    .st-key-login_card {
        max-width: 26rem;
        margin: 0 auto;
    }
    .st-key-login_card h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat avatars (branding): the agent is the crop, the farmer is the person.
AVATARS = {"assistant": "🌾", "user": "🧑‍🌾"}

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
    for key in ("auth_user", "auth_token", "active_conversation_id"):
        st.session_state.pop(key, None)
    for param in ("session", "c"):
        try:
            del st.query_params[param]
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
    login_card = st.container(key="login_card")
    with login_card:
        _login_card_body()
    st.stop()


def _login_card_body():
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


current_user = _resolve_user()
if current_user is None:
    _login_screen()

# Guests (only possible with no DATABASE_URL configured) get session-only
# chats; authenticated users get Neon-backed persistence.
_is_persistent = current_user["id"] is not None


# --- Conversation persistence -------------------------------------------

def _load_conversation(conversation_id):
    """Restore transcript + agent memory (profile, facts, trace) from Neon."""
    conv = convs.get_conversation(current_user["id"], conversation_id)
    if conv is None:
        return False
    rows = convs.load_messages(current_user["id"], conversation_id) or []
    history, flat_trace, turn_traces = [], [], []
    for row in rows:
        history.append({"role": row["role"], "content": row["content"]})
        if row["role"] == "assistant":
            turn_traces.append(row.get("trace") or [])
        if row.get("trace"):
            flat_trace.extend(row["trace"])
    st.session_state["turn_traces"] = turn_traces
    st.session_state["conversation_history"] = history
    st.session_state["farmer_profile"] = conv["farmer_profile"] or {}
    # Partial/legacy fact records are tolerated: session_store and the
    # orchestrator both re-fill missing fact keys with None on access.
    st.session_state["session_facts"] = conv["session_facts"] or {}
    st.session_state["trace_log"] = flat_trace
    st.session_state["active_conversation_id"] = conversation_id
    st.query_params["c"] = str(conversation_id)
    return True


def _start_new_chat():
    reset_session()
    st.session_state["active_conversation_id"] = None
    try:
        del st.query_params["c"]
    except KeyError:
        pass


# On the first script run of a browser session (fresh tab, refresh, or just
# after login) reopen the conversation named in the URL, else the most
# recent one. The key's presence marks that restore already happened, so a
# deliberate "New chat" (active id None) is never overridden.
if _is_persistent and "active_conversation_id" not in st.session_state:
    _restored = False
    _c_param = st.query_params.get("c")
    if _c_param and str(_c_param).isdigit():
        _restored = _load_conversation(int(_c_param))
    if not _restored:
        _recent = convs.list_conversations(current_user["id"])
        if _recent:
            _restored = _load_conversation(_recent[0]["id"])
    if not _restored:
        _start_new_chat()

with st.sidebar:
    st.markdown("## 🌾 AgriSense AI")
    st.caption("Agentic season planning for Bangladeshi farmers")

    if st.button(
        "➕ New chat", type="primary", use_container_width=True, key="new_chat_btn"
    ):
        _start_new_chat()
        st.rerun()

    with st.container(key="chat_list"):
        if _is_persistent:
            chat_rows = convs.list_conversations(current_user["id"])
            active_id = st.session_state.get("active_conversation_id")
            if chat_rows:
                st.caption("Recent chats")
            for row in chat_rows:
                is_active = row["id"] == active_id
                label = ("🟢 " if is_active else "") + (row["title"] or "New chat")
                if st.button(
                    label,
                    key=f"conv_{row['id']}",
                    type="tertiary",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    if _load_conversation(row["id"]):
                        st.rerun()
                    else:
                        st.toast("⚠️ Could not open that conversation.")
            if not chat_rows:
                st.caption("No saved chats yet — send a message to start one.")
        else:
            st.caption("Guest mode: chats are not saved.")

    st.divider()

    profile = get_farmer_profile()
    with st.expander("🧑‍🌾 Farmer profile", expanded=False):
        if profile:
            for k, v in profile.items():
                st.write(f"**{k}**: {v}")
        else:
            st.write("_(nothing known yet)_")
        still_missing = missing_fields(profile)
        if still_missing:
            st.caption("Still missing: " + ", ".join(still_missing))
        st.caption("Weather data: [Open-Meteo](https://open-meteo.com/)")

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

    with st.container(key="user_section"):
        st.divider()
        account_col, logout_col = st.columns([3, 2])
        account_col.markdown(f"👤 **{current_user['username']}**")
        if logout_col.button("Log out", key="logout_btn", use_container_width=True):
            _logout()

st.title("🌾 AgriSense AI")
st.caption("From a vague opening message to a grounded, explained, costed season plan.")


# --- Inline agent activity ----------------------------------------------
# The visible agent trace lives inside the conversation: while a turn runs,
# each tool call is streamed into an st.status block in the assistant's
# message; afterwards (and for reopened conversations) the same real
# entries render in a collapsed expander above the answer. Everything shown
# comes from actual tool execution -- never model-invented steps.

def _compact_args(arguments):
    text = ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items())
    return text if len(text) <= 120 else text[:117] + "…"


class _LiveTrace(list):
    """Trace sink passed to run_turn: every append renders the tool call
    live into the running status container as it happens."""

    def __init__(self, container):
        super().__init__()
        self._container = container

    def append(self, entry):
        super().append(entry)
        try:
            self._render(entry)
        except Exception:
            pass  # display problems must never break the agent turn

    def _render(self, entry):
        tool = entry.get("tool", "?")
        result = entry.get("result") or {}
        self._container.markdown(
            f"🔧 `{tool}`({_compact_args(entry.get('arguments'))})"
        )
        error = result.get("error") or result.get("exception")
        if error:
            self._container.markdown(f"❌ {db.redact(error)}")
        else:
            reasons = [
                r for r in (result.get("reasons") or []) if isinstance(r, str)
            ]
            if reasons:
                self._container.caption(reasons[0])


def _render_turn_trace(trace):
    """Collapsed, inspectable record of a finished turn's real tool calls."""
    calls = len(trace)
    failed = any(
        (entry.get("result") or {}).get("error")
        or (entry.get("result") or {}).get("exception")
        or entry.get("tool") == "ERROR"
        for entry in trace
    )
    label = f"🛠️ Agent activity · {calls} tool call{'s' if calls != 1 else ''}"
    if failed:
        label += " · ⚠️ errors"
    with st.expander(label, expanded=False):
        for i, entry in enumerate(trace, 1):
            st.markdown(f"**{i}. `{entry.get('tool', '?')}`**")
            st.caption("Arguments")
            st.json(entry.get("arguments") or {})
            st.caption("Result")
            st.json(entry.get("result") or {}, expanded=False)


_turn_traces = get_turn_traces()
_assistant_idx = 0
for msg in get_conversation_history():
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        if msg["role"] == "assistant":
            if _assistant_idx < len(_turn_traces) and _turn_traces[_assistant_idx]:
                _render_turn_trace(_turn_traces[_assistant_idx])
            _assistant_idx += 1
        st.markdown(msg["content"])

if not get_conversation_history():
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
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
    # Echo the user's message immediately, then run the agent inside a live
    # status block in the assistant's message: tool calls appear as they
    # execute (see _LiveTrace), and the block collapses when the turn ends.
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status = st.status("🤖 Agent working…", expanded=True)
        turn_trace = _LiveTrace(status)
        turn_failed = False
        try:
            run_turn(
                get_conversation_history(),
                user_input,
                get_farmer_profile(),
                turn_trace,
                get_session_facts(),
            )
        except Exception as e:
            turn_failed = True
            safe_error = db.redact(str(e))
            hist = get_conversation_history()
            hist.append({"role": "user", "content": user_input})
            hist.append({
                "role": "assistant",
                "content": f"Something went wrong on my end: `{safe_error}`. Please try again or rephrase.",
            })
            turn_trace.append({
                "tool": "ERROR",
                "arguments": {"user_input": user_input},
                "result": {"exception": safe_error},
            })
        calls = len(turn_trace)
        status.update(
            label=(
                f"🛠️ Agent activity · {calls} tool call{'s' if calls != 1 else ''}"
                + (" · ⚠️ errors" if turn_failed else "")
            ),
            state="error" if turn_failed else "complete",
            expanded=False,
        )

    # Record this turn's trace: flat log (backward compatibility) plus the
    # per-turn slice aligned with the assistant message just appended.
    get_trace_log().extend(turn_trace)
    get_turn_traces().append(list(turn_trace))

    # Persist the completed turn. This branch runs exactly once per
    # submission (chat_input only returns a value on the submitting rerun,
    # and we st.rerun() right after), so turns are never double-saved.
    # Failed turns are saved too -- they are part of the transcript.
    if _is_persistent:
        conv_id = st.session_state.get("active_conversation_id")
        if conv_id is None:
            conv_id = convs.create_conversation(current_user["id"], user_input)
            if conv_id is not None:
                st.session_state["active_conversation_id"] = conv_id
                st.query_params["c"] = str(conv_id)
        hist = get_conversation_history()
        assistant_reply = (
            hist[-1]["content"]
            if hist and hist[-1]["role"] == "assistant"
            else ""
        )
        saved = conv_id is not None and convs.append_turn(
            current_user["id"],
            conv_id,
            user_input,
            assistant_reply,
            list(turn_trace),
        )
        if saved:
            convs.save_state(
                current_user["id"],
                conv_id,
                get_farmer_profile(),
                get_session_facts(),
            )
        else:
            st.toast("⚠️ Could not save this turn to the database.")

    # The history loop at the top of the script renders every turn (including
    # anything just appended) on this rerun -- so no manual st.chat_message
    # blocks here, which would otherwise double-draw each turn.
    st.rerun()
