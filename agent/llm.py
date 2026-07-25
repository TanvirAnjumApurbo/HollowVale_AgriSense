"""Provider-agnostic chat() wrapper around the LLM.

Defaults to OpenAI gpt-4o-mini (real API credit confirmed for this
event). Swapping to Groq (free) or another OpenAI-compatible/Anthropic
provider only requires changing LLM_PROVIDER/LLM_MODEL and adding the
matching client branch below -- the orchestrator only ever calls chat().

LLM_REASONING_EFFORT (minimal|low|medium|high) trades latency for depth on
reasoning models; it is sent only to models that accept it (the GPT-5 family
and o-series) and is a no-op otherwise.
"""

import os
import re


def _get_secret(key, default=None):
    """Look up a secret from Streamlit secrets first, then env vars.

    st.secrets raises if no secrets.toml exists (even for a membership
    check), so this is wrapped defensively -- the app must still run from
    plain environment variables (e.g. a local .env, or a CI/cron context)
    with no secrets file present.
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


LLM_PROVIDER = _get_secret("LLM_PROVIDER", "openai")
LLM_MODEL = _get_secret("LLM_MODEL", "gpt-4o-mini")
# Speed<->depth lever for reasoning models (GPT-5 family, o-series):
# minimal | low | medium | high. Blank/unset => the provider default (medium
# for GPT-5). Withheld for non-reasoning models (gpt-4o*, gpt-4.1*), which
# reject the parameter, and ignored if set to an unrecognized value.
LLM_REASONING_EFFORT = (_get_secret("LLM_REASONING_EFFORT", "") or "").strip().lower()

_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}

_client = None


def _get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (env var or .streamlit/secrets.toml).")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_groq_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def _supports_reasoning_effort(model):
    """True for models that accept a ``reasoning_effort`` argument -- the GPT-5
    family (``gpt-5``, ``gpt-5-mini``, ...) and the o-series (``o1``/``o3``/
    ``o4``...). Classic chat models (``gpt-4o*``, ``gpt-4.1*``) reject it, so it
    must be withheld for them. Tolerates a provider prefix like ``openai/``."""
    name = (model or "").lower().split("/")[-1]
    return name.startswith("gpt-5") or bool(re.match(r"o\d", name))


def chat(messages, tools=None, tool_choice="auto"):
    """Send messages (+ optional tool schemas) to the configured LLM.

    Returns the raw response message object (OpenAI-format): has
    `.content` (str or None) and `.tool_calls` (list or None).
    """
    if LLM_PROVIDER == "groq":
        client = _get_groq_client()
        model = LLM_MODEL if LLM_MODEL != "gpt-4o-mini" else "llama-3.3-70b-versatile"
    else:
        client = _get_openai_client()
        model = LLM_MODEL

    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    # Send reasoning_effort only when configured AND the model supports it, so
    # this stays a no-op for gpt-4o-mini/gpt-4.1 and for the API's own default.
    if LLM_REASONING_EFFORT in _REASONING_EFFORTS and _supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
