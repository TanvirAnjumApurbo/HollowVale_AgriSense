"""App-level tests of the authentication gate using Streamlit's AppTest.

These run app.py for real (no browser, no LLM calls: the gate stops the
script before any agent code executes). With DATABASE_URL unset the app must
show the login screen with guest mode, never the chat; guest mode unlocks the
chat; logout locks it again.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    # Empty string (not delenv): app.py's load_dotenv() would re-inject a
    # developer's real DATABASE_URL from .env, but it never overrides an
    # existing env var, and blank counts as unconfigured.
    monkeypatch.setenv("DATABASE_URL", "")
    from memory import db

    db.close_pool()
    yield
    db.close_pool()


def _boot():
    return AppTest.from_file(APP, default_timeout=30).run()


def test_unauthenticated_users_never_see_the_chat():
    at = _boot()
    assert not at.exception
    assert not at.chat_input
    assert any("AgriSense" in t.value for t in at.title)


def test_guest_mode_only_offered_without_database_and_unlocks_chat():
    at = _boot()
    guest = [b for b in at.button if "guest" in b.label.lower()]
    assert guest, "guest entry must exist when DATABASE_URL is unset"

    at = guest[0].click().run()
    assert not at.exception
    assert at.chat_input, "chat should render after guest entry"
    sidebar_md = " ".join(m.value for m in at.sidebar.markdown)
    assert "guest" in sidebar_md.lower()


def test_logout_returns_to_login_screen():
    at = _boot()
    guest = [b for b in at.button if "guest" in b.label.lower()]
    at = guest[0].click().run()
    logout = [b for b in at.button if b.label == "Log out"]
    assert logout
    at = logout[0].click().run()
    assert not at.exception
    assert not at.chat_input


def test_sidebar_copy_is_farmer_friendly():
    at = _boot()
    guest = [b for b in at.button if "guest" in b.label.lower()]
    at = guest[0].click().run()
    at.session_state["farmer_profile"] = {
        "location": "Gazipur",
        "soil_type": "loam",
        "budget_bdt": 50000,
        "target_season": "Kharif-2",
        "farm_size_acres": 1,
        "water_availability": "high",
    }
    at = at.run()

    sidebar_copy = " ".join(element.value for element in at.sidebar.markdown)
    assert "Soil type" in sidebar_copy
    assert "50,000 BDT" in sidebar_copy
    assert "1 acre" in sidebar_copy
    assert "soil_type" not in sidebar_copy
    assert "Weather data" not in sidebar_copy
    assert "server.py" not in sidebar_copy
    assert "sidecar" not in sidebar_copy
    assert "simulate mode" not in sidebar_copy
