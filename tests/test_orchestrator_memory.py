"""Offline regression tests for durable facts and the hand-rolled agent loop.

The LLM and external services are scripted/mocked here.  These tests exercise
the real orchestration behavior: tool results enter the visible trace, durable
facts survive into later turns, and established weather is reused rather than
silently fetched again.
"""

from copy import deepcopy
from datetime import date
import json
import re
from types import SimpleNamespace

import pytest

from agent import orchestrator
from agent.prompts import build_facts_digest, build_system_prompt
from memory import session_store
from tools.agronomy import rank_crops
from tools.season_plan import build_season_calendar


PROFILE = {
    "location": "Rangpur",
    "farm_size_acres": 1,
    "soil_type": "clay loam",
    "water_availability": "high",
    "budget_bdt": 80_000,
    "target_season": "Rabi",
}

WEATHER = {
    "location": {
        "name": "Rangpur",
        "admin1": "Rangpur Division",
        "country": "Bangladesh",
    },
    "forecast": {
        "daily": {
            "time": ["2026-07-24", "2026-07-25", "2026-07-26"],
            "precipitation_sum": [2.0, 0.0, 4.0],
        },
        "summary": {
            "forecast_days": 3,
            "total_rainfall_mm": 6.0,
            "rainy_days": 2,
            "avg_max_temp_c": 30.0,
            "avg_min_temp_c": 23.0,
        },
    },
    "reasons": ["Open-Meteo returned three dated forecast days for Rangpur."],
}

RANKING = {
    "as_of_date": "2026-07-24",
    "season_used": "Rabi",
    "season_source": "stated",
    "area_acres": 1,
    "weights": {
        "soil_fit": 0.25,
        "season_fit": 0.25,
        "water_fit": 0.20,
        "temp_fit": 0.10,
        "profit_score": 0.20,
    },
    "ranked": [
        {
            "crop": "rice_boro",
            "label": "Boro Rice",
            "overall_score": 0.91,
            "components": {
                "soil_fit": 1.0,
                "season_fit": 1.0,
                "water_fit": 0.9,
                "temp_fit": 0.8,
                "profit_score": 0.75,
            },
            "economics": {
                "area_acres": 1,
                "total_cost_bdt": 24_865,
                "revenue_bdt": 67_200,
                "net_profit_bdt": 42_335,
                "roi_pct": 170.3,
            },
            "budget": {"affordable": True},
            "duration_days": 145,
            "reasons": ["Clay-loam soil and reliable irrigation fit Boro."],
        }
    ],
    "also_ranked": [],
    "reasons": ["Boro ranked first from the supplied farm and weather inputs."],
}

PROJECTION = {
    "crop": "rice_boro",
    "label": "Boro Rice",
    "area_acres": 1,
    "unit": "kg",
    "duration_days": 145,
    "cost_breakdown_bdt": {
        "seed": 990,
        "fertilizer": 7_275,
        "labour": 10_000,
        "irrigation": 5_000,
        "pesticide": 1_600,
    },
    "total_cost_bdt": 24_865,
    "expected_yield": 2_400,
    "price_per_unit_bdt": 28,
    "revenue_bdt": 67_200,
    "net_profit_bdt": 42_335,
    "roi_pct": 170.3,
    "break_even_yield_units": 888.04,
    "break_even_price_per_unit_bdt": 10.36,
    "reasons": ["Projection used the stored Boro inputs for one acre."],
}

CALENDAR = {
    "crop": "rice_boro",
    "label": "Boro Rice",
    "area_acres": 1,
    "sowing_date": "2026-07-01",
    "harvest_date": "2026-11-23",
    "duration_days": 145,
    "events": [
        {
            "date": "2026-07-01",
            "type": "seed",
            "action": "Procure Boro seed",
            "quantity": "18 kg",
            "cost_bdt": 990,
            "source": "data/crops.yaml",
        },
        {
            "date": "2026-11-23",
            "type": "harvest",
            "action": "Harvest Boro rice",
            "quantity": "2400 kg expected",
            "cost_bdt": 0,
            "source": "data/crops.yaml",
        },
    ],
    "total_event_cost_bdt": 24_865,
    "financial_total_cost_bdt": 24_865,
    "costs_reconciled": True,
    "weather_adjustments": 0,
    "warnings": [],
    "reasons": ["Every date came from the sowing date plus stored offsets."],
}


def _empty_facts():
    return {key: None for key in session_store.SESSION_FACT_KEYS}


def _tool_message(*calls):
    tool_calls = []
    for index, (name, arguments) in enumerate(calls, start=1):
        tool_calls.append(
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments),
                ),
            )
        )
    return SimpleNamespace(content=None, tool_calls=tool_calls)


def _final_message(text):
    return SimpleNamespace(content=text, tool_calls=[])


class ScriptedChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": deepcopy(messages), "tools": tools})
        assert self.responses, "scripted chat ran out of responses"
        return self.responses.pop(0)


def _scaled_projection(area_acres):
    result = deepcopy(PROJECTION)
    factor = float(area_acres)
    result["area_acres"] = area_acres
    for key, value in result["cost_breakdown_bdt"].items():
        result["cost_breakdown_bdt"][key] = round(value * factor, 2)
    for key in ("total_cost_bdt", "expected_yield", "revenue_bdt", "net_profit_bdt"):
        result[key] = round(PROJECTION[key] * factor, 2)
    result["reasons"] = [f"Projection used area_acres={area_acres}."]
    return result


def _scaled_calendar(area_acres, weather):
    result = deepcopy(CALENDAR)
    result["area_acres"] = area_acres
    result["total_event_cost_bdt"] = round(24_865 * float(area_acres), 2)
    result["financial_total_cost_bdt"] = result["total_event_cost_bdt"]
    result["weather_received"] = weather
    result["reasons"] = [f"Calendar used area_acres={area_acres}."]
    return result


def _install_offline_tool_mocks(monkeypatch):
    calls = {"weather": 0, "ranking": [], "projection": [], "calendar": []}

    def fake_weather(location):
        calls["weather"] += 1
        result = deepcopy(WEATHER)
        result["requested_location"] = location
        return result

    def fake_kb(query, crop_filter=None):
        return {
            "query": query,
            "crop_filter_applied": crop_filter,
            "results": [
                {
                    "text": "Grounded Boro guidance.",
                    "source_file": "rice_boro_bamis.md",
                    "source_title": "BAMIS Boro guidance",
                    "source_url": "https://example.test/boro",
                }
            ],
            "reasons": ["The requested Boro source was retrieved."],
        }

    def fake_ranking(profile, weather, top_n=5):
        calls["ranking"].append(
            {
                "profile": deepcopy(profile),
                "weather": deepcopy(weather),
                "top_n": top_n,
            }
        )
        result = deepcopy(RANKING)
        result["area_acres"] = profile["farm_size_acres"]
        result["ranked"][0]["economics"]["area_acres"] = profile[
            "farm_size_acres"
        ]
        return result

    def fake_projection(
        crop,
        area_acres,
        yield_adjustment_pct=0.0,
        price_override=None,
    ):
        calls["projection"].append(
            {
                "crop": crop,
                "area_acres": area_acres,
                "yield_adjustment_pct": yield_adjustment_pct,
                "price_override": price_override,
            }
        )
        return _scaled_projection(area_acres)

    def fake_calendar(crop_key, sowing_date, area_acres, weather=None):
        calls["calendar"].append(
            {
                "crop_key": crop_key,
                "sowing_date": sowing_date,
                "area_acres": area_acres,
                "weather": deepcopy(weather),
            }
        )
        return _scaled_calendar(area_acres, weather)

    monkeypatch.setattr(orchestrator, "get_weather_for_location", fake_weather)
    monkeypatch.setattr(orchestrator, "search_knowledge_base", fake_kb)
    monkeypatch.setattr(orchestrator, "rank_crops", fake_ranking)
    monkeypatch.setattr(
        orchestrator,
        "compute_financial_projection",
        fake_projection,
    )
    monkeypatch.setattr(orchestrator, "build_season_calendar", fake_calendar)
    return calls


def test_session_facts_initialize_migrate_and_reset(monkeypatch):
    state = {
        "session_facts": {"weather": {"cached": True}},
        "conversation_history": [{"role": "user", "content": "old"}],
        "farmer_profile": {"location": "Rangpur"},
        "trace_log": [{"tool": "old"}],
    }
    monkeypatch.setattr(
        session_store,
        "st",
        SimpleNamespace(session_state=state),
    )

    session_store.init_session()
    facts = session_store.get_session_facts()
    assert set(facts) == set(session_store.SESSION_FACT_KEYS)
    assert facts["weather"] == {"cached": True}
    assert all(facts[key] is None for key in session_store.SESSION_FACT_KEYS[1:])

    facts["ranking"] = {"ranked": ["rice_boro"]}
    session_store.init_session()
    assert session_store.get_session_facts() is facts
    assert facts["ranking"] == {"ranked": ["rice_boro"]}

    session_store.reset_session()
    assert session_store.get_conversation_history() == []
    assert session_store.get_farmer_profile() == {}
    assert session_store.get_trace_log() == []
    assert session_store.get_session_facts() == _empty_facts()


def test_prompt_uses_a_compact_digest_of_established_tool_facts():
    sentinel = "DEEP_PAYLOAD_MUST_NOT_ENTER_PROMPT"
    facts = {
        "weather": deepcopy(WEATHER),
        "ranking": deepcopy(RANKING),
        "chosen_crop": "rice_boro",
        "projection": deepcopy(PROJECTION),
        "calendar": deepcopy(CALENDAR),
    }
    facts["weather"]["forecast"]["daily"]["raw_debug"] = [sentinel] * 1_000
    facts["ranking"]["ranked"][0]["reasons"].append(sentinel)
    facts["projection"]["data_source_note"] = sentinel
    facts["calendar"]["source_refs"] = [sentinel]

    digest = build_facts_digest(facts)
    prompt = build_system_prompt(PROFILE, facts)

    assert "location=Rangpur" in digest
    assert "rice_boro overall_score=0.91" in digest
    assert "Chosen crop: rice_boro." in digest
    assert "total_cost_bdt=24865" in digest
    assert "sowing_date=2026-07-01" in digest
    assert "harvest_date=2026-11-23" in digest
    assert "Harvest Boro rice" in digest
    assert sentinel not in digest
    assert sentinel not in prompt
    assert len(digest) < 3_000
    assert len(prompt) < 12_000


def test_scripted_turn_persists_all_tool_facts_and_exposes_reasons(
    monkeypatch,
):
    mock_calls = _install_offline_tool_mocks(monkeypatch)
    registered_names = {
        schema["function"]["name"] for schema in orchestrator.TOOL_SCHEMAS
    }
    scripted = ScriptedChat(
        [
            _tool_message(
                (
                    "update_farmer_profile",
                    {**PROFILE, "chosen_crop": "boro"},
                ),
                ("get_weather", {"location": "Rangpur"}),
                (
                    "search_knowledge_base",
                    {"query": "Boro fertilizer timing", "crop_filter": "rice"},
                ),
                ("rank_crops", {"top_n": 3}),
                (
                    "compute_financial_projection",
                    {"crop": "rice_boro", "area_acres": 1},
                ),
                (
                    "build_season_calendar",
                    {
                        "crop_key": "rice_boro",
                        "sowing_date": "2026-07-01",
                        "area_acres": 1,
                    },
                ),
            ),
            _final_message("The stored plan is ready."),
        ]
    )
    monkeypatch.setattr(orchestrator, "chat", scripted)

    history = []
    profile = {}
    trace = []
    facts = _empty_facts()
    reply = orchestrator.run_turn(
        history,
        "Build my Boro plan.",
        profile,
        trace,
        facts,
    )

    assert reply == "The stored plan is ready."
    assert {entry["tool"] for entry in trace} == registered_names
    for entry in trace:
        reasons = entry["result"].get("reasons")
        assert isinstance(reasons, list) and reasons
        assert all(isinstance(reason, str) and reason.strip() for reason in reasons)

    assert profile == {**PROFILE, "chosen_crop": "rice_boro"}
    assert facts["weather"]["requested_location"] == "Rangpur"
    assert facts["ranking"]["ranked"][0]["crop"] == "rice_boro"
    assert facts["projection"]["area_acres"] == 1
    assert facts["calendar"]["sowing_date"] == "2026-07-01"
    assert facts["chosen_crop"] == "rice_boro"
    assert mock_calls["weather"] == 1
    assert mock_calls["ranking"][0]["weather"]["location"]["name"] == "Rangpur"
    assert mock_calls["calendar"][0]["weather"]["location"]["name"] == "Rangpur"
    assert history[-2:] == [
        {"role": "user", "content": "Build my Boro plan."},
        {"role": "assistant", "content": "The stored plan is ready."},
    ]


def test_area_follow_up_reuses_weather_without_network_refetch(monkeypatch):
    mock_calls = _install_offline_tool_mocks(monkeypatch)
    scripted = ScriptedChat(
        [
            _tool_message(("get_weather", {"location": "Rangpur"})),
            _final_message("Weather stored."),
            _tool_message(
                (
                    "update_farmer_profile",
                    {"farm_size_acres": 2},
                ),
                ("get_weather", {"location": "  rangpur  "}),
                ("rank_crops", {"top_n": 3}),
                (
                    "compute_financial_projection",
                    {"crop": "rice_boro", "area_acres": 2},
                ),
                (
                    "build_season_calendar",
                    {
                        "crop_key": "rice_boro",
                        "sowing_date": "2026-07-01",
                        "area_acres": 2,
                    },
                ),
            ),
            _final_message("Two-acre scenario ready."),
        ]
    )
    monkeypatch.setattr(orchestrator, "chat", scripted)

    history = []
    profile = deepcopy(PROFILE)
    trace = []
    facts = _empty_facts()

    orchestrator.run_turn(
        history,
        "Check the weather.",
        profile,
        trace,
        facts,
    )
    original_weather = deepcopy(facts["weather"])
    # Crop choice is an explicit farmer fact, not inferred from a projection.
    facts["chosen_crop"] = "rice_boro"

    reply = orchestrator.run_turn(
        history,
        "Make that two acres.",
        profile,
        trace,
        facts,
    )

    assert reply == "Two-acre scenario ready."
    assert mock_calls["weather"] == 1, "same-location weather must come from facts"
    assert profile["farm_size_acres"] == 2
    assert facts["weather"]["location"] == original_weather["location"]
    assert facts["projection"]["area_acres"] == 2
    assert facts["calendar"]["area_acres"] == 2
    assert facts["chosen_crop"] == "rice_boro"
    assert mock_calls["ranking"][-1]["weather"]["location"]["name"] == "Rangpur"
    assert mock_calls["calendar"][-1]["weather"]["location"]["name"] == "Rangpur"

    weather_entries = [
        entry for entry in trace if entry["tool"] == "get_weather"
    ]
    assert len(weather_entries) == 2
    assert weather_entries[-1]["result"]["cache_status"] == "reused_session_fact"
    assert any(
        "no weather api request was made" in reason.lower()
        for reason in weather_entries[-1]["result"]["reasons"]
    )


def test_rank_crops_returns_the_canonical_explainable_schema():
    result = rank_crops(
        PROFILE,
        WEATHER,
        today=date(2026, 7, 24),
        top_n=3,
    )

    assert set(result) >= {
        "as_of_date",
        "season_used",
        "season_source",
        "area_acres",
        "weights",
        "ranked",
        "also_ranked",
        "reasons",
    }
    assert set(result["weights"]) == {
        "soil_fit",
        "season_fit",
        "water_fit",
        "temp_fit",
        "profit_score",
    }
    assert len(result["ranked"]) == 3
    assert result["reasons"] and all(
        isinstance(reason, str) and reason.strip()
        for reason in result["reasons"]
    )

    for candidate in result["ranked"]:
        assert set(candidate) >= {
            "crop",
            "label",
            "overall_score",
            "components",
            "economics",
            "budget",
            "duration_days",
            "reasons",
        }
        assert set(candidate["components"]) == set(result["weights"])
        assert candidate["reasons"]


@pytest.mark.parametrize("invalid_area", [0, -1, float("nan"), "not-a-number"])
def test_rank_crops_returns_explainable_error_for_invalid_area(invalid_area):
    profile = {**PROFILE, "farm_size_acres": invalid_area}
    result = rank_crops(
        profile,
        WEATHER,
        today=date(2026, 7, 24),
        top_n=3,
    )

    assert result["error"]
    assert result["ranked"] == []
    assert result["also_ranked"] == []
    assert result["reasons"]
    assert "positive finite number" in result["reasons"][0]


def test_season_calendar_returns_the_canonical_explainable_schema():
    result = build_season_calendar("rice_boro", "2026-07-01", 1)

    assert set(result) >= {
        "crop",
        "label",
        "area_acres",
        "sowing_date",
        "harvest_date",
        "duration_days",
        "events",
        "total_event_cost_bdt",
        "financial_total_cost_bdt",
        "cost_breakdown_bdt",
        "costs_reconciled",
        "weather_adjustments",
        "warnings",
        "reasons",
    }
    assert result["costs_reconciled"] is True
    assert result["total_event_cost_bdt"] == result["financial_total_cost_bdt"]
    assert result["reasons"] and all(
        isinstance(reason, str) and reason.strip()
        for reason in result["reasons"]
    )
    assert [event["date"] for event in result["events"]] == sorted(
        event["date"] for event in result["events"]
    )
    assert {
        event["type"] for event in result["events"]
    } == {
        "stage",
        "seed",
        "labour",
        "fertilizer",
        "irrigation",
        "weed",
        "pest",
        "harvest",
    }
    assert all(
        {"date", "type", "action", "quantity", "cost_bdt", "source"}
        <= set(event)
        for event in result["events"]
    )


def test_full_acceptance_path_and_one_acre_follow_up(monkeypatch):
    """Test 4: vague intake -> sourced plan -> acreage-only recalculation."""
    mock_calls = _install_offline_tool_mocks(monkeypatch)
    first_plan_reply = (
        "For 2 acres, total cost is BDT 49730, expected yield is 4800 kg, "
        "and harvest is 2026-11-23."
    )
    follow_up_reply = (
        "At 1 acre, total cost is BDT 24865, expected yield is 2400 kg, "
        "and harvest remains 2026-11-23."
    )
    scripted = ScriptedChat(
        [
            _final_message(
                "Where is your farm, how large is it, and what soil do you have?"
            ),
            _tool_message(
                (
                    "update_farmer_profile",
                    {
                        "location": "Rangpur",
                        "farm_size_acres": 2,
                        "soil_type": "clay loam",
                    },
                )
            ),
            _final_message(
                "What irrigation, budget, and growing season are you planning?"
            ),
            _tool_message(
                (
                    "update_farmer_profile",
                    {
                        "water_availability": "high",
                        "budget_bdt": 80_000,
                        "target_season": "Rabi",
                    },
                ),
                ("get_weather", {"location": "Rangpur"}),
                ("rank_crops", {"top_n": 3}),
                (
                    "search_knowledge_base",
                    {
                        "query": "Boro rice fit and fertilizer timing",
                        "crop_filter": "rice",
                    },
                ),
                (
                    "search_knowledge_base",
                    {
                        "query": "Wheat fit and irrigation timing",
                        "crop_filter": "wheat",
                    },
                ),
            ),
            _final_message(
                "Boro, wheat, and potato are the sourced finalists. "
                "Which crop do you choose?"
            ),
            _tool_message(
                ("update_farmer_profile", {"chosen_crop": "boro"}),
                (
                    "compute_financial_projection",
                    {"crop": "rice_boro", "area_acres": 2},
                ),
                (
                    "build_season_calendar",
                    {
                        "crop_key": "rice_boro",
                        "sowing_date": "2026-07-01",
                        "area_acres": 2,
                    },
                ),
            ),
            _final_message(first_plan_reply),
            _tool_message(
                ("update_farmer_profile", {"farm_size_acres": 1}),
                (
                    "compute_financial_projection",
                    {"crop": "rice_boro", "area_acres": 1},
                ),
                (
                    "build_season_calendar",
                    {
                        "crop_key": "rice_boro",
                        "sowing_date": "2026-07-01",
                        "area_acres": 1,
                    },
                ),
            ),
            _final_message(follow_up_reply),
        ]
    )
    monkeypatch.setattr(orchestrator, "chat", scripted)

    history = []
    profile = {}
    trace = []
    facts = _empty_facts()

    vague_reply = orchestrator.run_turn(
        history,
        "Hello, can you help me plan my farm?",
        profile,
        trace,
        facts,
    )
    assert "where" in vague_reply.lower()
    assert profile == {}
    assert trace == []

    partial_reply = orchestrator.run_turn(
        history,
        "I am in Rangpur with 2 acres of clay-loam soil.",
        profile,
        trace,
        facts,
    )
    assert "irrigation" in partial_reply.lower()
    assert profile == {
        "location": "Rangpur",
        "farm_size_acres": 2,
        "soil_type": "clay loam",
    }

    orchestrator.run_turn(
        history,
        "I have high irrigation, an 80000 BDT budget, and want Rabi.",
        profile,
        trace,
        facts,
    )
    recommendation_tools = [entry["tool"] for entry in trace]
    assert recommendation_tools[-5:] == [
        "update_farmer_profile",
        "get_weather",
        "rank_crops",
        "search_knowledge_base",
        "search_knowledge_base",
    ]
    assert facts["weather"]["location"]["name"] == "Rangpur"
    assert facts["ranking"]["ranked"]

    trace_before_plan = len(trace)
    plan_reply = orchestrator.run_turn(
        history,
        "I choose Boro and will sow on 2026-07-01.",
        profile,
        trace,
        facts,
    )
    initial_plan_trace = trace[trace_before_plan:]
    assert [entry["tool"] for entry in initial_plan_trace] == [
        "update_farmer_profile",
        "compute_financial_projection",
        "build_season_calendar",
    ]
    assert plan_reply == first_plan_reply
    assert facts["chosen_crop"] == "rice_boro"
    assert facts["projection"]["area_acres"] == 2
    assert facts["calendar"]["area_acres"] == 2
    assert mock_calls["calendar"][-1]["weather"]["location"]["name"] == "Rangpur"

    token_pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|(?<![\w.-])\d+(?:\.\d+)?(?![\w.-])"
    )
    plan_trace_json = json.dumps(initial_plan_trace, sort_keys=True)
    numeric_and_date_tokens = token_pattern.findall(plan_reply)
    assert numeric_and_date_tokens
    assert all(token in plan_trace_json for token in numeric_and_date_tokens)

    first_projection = deepcopy(facts["projection"])
    first_calendar = deepcopy(facts["calendar"])
    trace_before_follow_up = len(trace)
    scenario_reply = orchestrator.run_turn(
        history,
        "What if I only have 1 acre?",
        profile,
        trace,
        facts,
    )
    follow_up_trace = trace[trace_before_follow_up:]

    assert scenario_reply == follow_up_reply
    assert [entry["tool"] for entry in follow_up_trace] == [
        "update_farmer_profile",
        "compute_financial_projection",
        "build_season_calendar",
    ]
    assert not {
        "get_weather",
        "rank_crops",
    }.intersection(entry["tool"] for entry in follow_up_trace)
    assert profile["farm_size_acres"] == 1
    assert facts["projection"]["area_acres"] == 1
    assert facts["calendar"]["area_acres"] == 1
    assert facts["projection"]["total_cost_bdt"] == pytest.approx(
        first_projection["total_cost_bdt"] / 2
    )
    assert facts["projection"]["expected_yield"] == pytest.approx(
        first_projection["expected_yield"] / 2
    )
    assert facts["calendar"]["total_event_cost_bdt"] == pytest.approx(
        first_calendar["total_event_cost_bdt"] / 2
    )

    follow_up_trace_json = json.dumps(follow_up_trace, sort_keys=True)
    follow_up_tokens = token_pattern.findall(scenario_reply)
    assert follow_up_tokens
    assert all(token in follow_up_trace_json for token in follow_up_tokens)

    assert mock_calls["weather"] == 1
    assert [entry["tool"] for entry in trace].count("get_weather") == 1
    assert [entry["tool"] for entry in trace].count("rank_crops") == 1


def test_explicit_weather_refresh_bypasses_same_location_cache(monkeypatch):
    mock_calls = _install_offline_tool_mocks(monkeypatch)
    cached = deepcopy(WEATHER)
    cached["requested_location"] = "Rangpur"
    facts = _empty_facts()
    facts["weather"] = cached
    facts["ranking"] = deepcopy(RANKING)
    facts["calendar"] = deepcopy(CALENDAR)
    context = {
        "farmer_profile": deepcopy(PROFILE),
        "last_weather": cached,
        "session_facts": facts,
    }

    reused = orchestrator._execute_tool(
        "get_weather",
        {"location": "Rangpur"},
        context,
    )
    assert reused["cache_status"] == "reused_session_fact"
    assert mock_calls["weather"] == 0

    refreshed = orchestrator._execute_tool(
        "get_weather",
        {"location": "Rangpur", "refresh": True},
        context,
    )
    assert refreshed.get("cache_status") != "reused_session_fact"
    assert refreshed["requested_location"] == "Rangpur"
    assert mock_calls["weather"] == 1
    orchestrator._persist_tool_facts("get_weather", refreshed, context)
    assert facts["ranking"] is None
    assert facts["calendar"] is None


def test_projection_calendar_scope_coherence_does_not_change_explicit_choice():
    facts = _empty_facts()
    facts.update(
        {
            "chosen_crop": "rice_boro",
            "projection": _scaled_projection(2),
            "calendar": _scaled_calendar(2, WEATHER),
        }
    )
    context = {"session_facts": facts, "last_weather": WEATHER}

    wheat_projection = _scaled_projection(1)
    wheat_projection["crop"] = "wheat"
    orchestrator._persist_tool_facts(
        "compute_financial_projection",
        wheat_projection,
        context,
    )

    assert facts["chosen_crop"] == "rice_boro"
    assert facts["projection"]["crop"] == "wheat"
    assert facts["calendar"] is None

    boro_calendar = _scaled_calendar(1, WEATHER)
    orchestrator._persist_tool_facts(
        "build_season_calendar",
        boro_calendar,
        context,
    )
    assert facts["chosen_crop"] == "rice_boro"
    assert facts["calendar"]["crop"] == "rice_boro"
    assert facts["projection"] is None


def test_location_change_invalidates_only_weather_dependent_facts():
    facts = {
        "weather": deepcopy(WEATHER),
        "ranking": deepcopy(RANKING),
        "chosen_crop": "rice_boro",
        "projection": deepcopy(PROJECTION),
        "calendar": deepcopy(CALENDAR),
    }
    profile = deepcopy(PROFILE)
    context = {
        "farmer_profile": profile,
        "last_weather": facts["weather"],
        "session_facts": facts,
    }

    result = orchestrator._execute_tool(
        "update_farmer_profile",
        {"location": "Dinajpur"},
        context,
    )

    assert result["invalidated_facts"] == ["calendar", "ranking", "weather"]
    assert profile["location"] == "Dinajpur"
    assert facts["weather"] is None
    assert facts["ranking"] is None
    assert facts["calendar"] is None
    assert facts["chosen_crop"] == "rice_boro"
    assert facts["projection"] == PROJECTION
    assert context["last_weather"] is None


def test_yield_adjusted_projection_clears_baseline_yield_calendar():
    facts = _empty_facts()
    facts["chosen_crop"] = "rice_boro"
    facts["calendar"] = deepcopy(CALENDAR)
    context = {"session_facts": facts, "last_weather": WEATHER}
    adjusted = deepcopy(PROJECTION)
    adjusted["yield_adjustment_pct_applied"] = -30
    adjusted["expected_yield"] = 1680

    orchestrator._persist_tool_facts(
        "compute_financial_projection",
        adjusted,
        context,
    )

    assert facts["projection"]["yield_adjustment_pct_applied"] == -30
    assert facts["calendar"] is None
    assert facts["chosen_crop"] == "rice_boro"
