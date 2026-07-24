"""Provider-agnostic chat() wrapper around the LLM.

Defaults to OpenAI gpt-4o-mini (real API credit confirmed for this
event). Swapping to Groq (free) or another OpenAI-compatible/Anthropic
provider only requires changing LLM_PROVIDER/LLM_MODEL and adding the
matching client branch below -- the orchestrator only ever calls chat().
"""

import os

try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}


def _get_secret(key, default=None):
    if key in _SECRETS:
        return _SECRETS[key]
    return os.environ.get(key, default)


LLM_PROVIDER = _get_secret("LLM_PROVIDER", "openai")
LLM_MODEL = _get_secret("LLM_MODEL", "gpt-4o-mini")

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

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
