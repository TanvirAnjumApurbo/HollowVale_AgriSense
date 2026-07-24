"""Hand-rolled tool-calling agent loop.

Deliberately not using a framework (LangChain/CrewAI/etc): at hackathon
scale, a ~150-line manual loop is fully transparent, which is exactly
what the "visible agent trace" requirement needs -- every tool call,
its parameters, and its raw return value are captured here and nothing
is hidden inside framework internals.
"""

from copy import deepcopy
from datetime import datetime
import json

from agent.llm import chat
from agent.prompts import build_system_prompt
from data.crop_loader import normalize_key
from memory.session_store import SESSION_FACT_KEYS, get_session_facts
from tools.weather import get_weather_for_location
from tools.knowledge_base import search_knowledge_base
from tools.financials import compute_financial_projection
from tools.agronomy import rank_crops
from tools.season_plan import build_season_calendar

MAX_TOOL_ITERATIONS = 12

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "update_farmer_profile",
            "description": "Record/update known facts about the farmer's profile as they are learned from the conversation. Call this as soon as any field is known, even one at a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Farm location, e.g. district/upazila/village in Bangladesh."},
                    "farm_size_acres": {"type": "number", "description": "Farm size in acres."},
                    "soil_type": {"type": "string", "description": "Soil type, e.g. sandy, sandy loam, loam, clay loam, clay."},
                    "water_availability": {"type": "string", "description": "Water availability, e.g. low, medium, high, or a description of irrigation access."},
                    "budget_bdt": {"type": "number", "description": "Season budget in BDT."},
                    "target_season": {"type": "string", "description": "Target growing season, e.g. Rabi, Kharif-1, Kharif-2, or a month range."},
                    "chosen_crop": {
                        "type": "string",
                        "description": "The crop explicitly chosen by the farmer. Record it as soon as they choose; do not infer a choice from the ranking.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a rainfall/temperature forecast for the farm location via Open-Meteo, with daily data, summary totals, and exact quotable reasons. Matching established weather is returned from session cache without a network request; use refresh=true only when the farmer explicitly asks for fresh weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name as given by the farmer."},
                    "refresh": {
                        "type": "boolean",
                        "description": "Set true only when the farmer explicitly asks for fresh weather; otherwise established same-location weather is reused.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Retrieve relevant agronomic passages (fertilizer doses, sowing windows, soil suitability, pest risk) from the grounded knowledge base. Returns source-grounded reasons that must be quoted for agronomic claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query, e.g. 'fertilizer dose for boro rice' or 'soil suitable for lentil'."},
                    "crop_filter": {"type": "string", "description": "Optional crop name to bias results toward, e.g. 'rice'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_crops",
            "description": (
                "Score and rank ALL candidate crops for THIS farm using the recorded farmer profile "
                "(soil, area, water availability, budget, target season) and the latest established/live "
                "weather forecast -- you do not pass these, the application supplies them. Returns, per crop, a 0-1 "
                "overall suitability score plus soil_fit/season_fit/water_fit/temp_fit/profit_score components, "
                "an economics block, a budget flag (with max affordable area), and a `reasons` list that already "
                "explains each score in plain language with the exact input values used. NARRATE those reasons; "
                "do not invent your own suitability judgement or numbers. It is season-aware: for today's date it "
                "knows which crops are actually plantable now (e.g. Aman rice in Kharif-2), so trust its ranking "
                "over generic crop knowledge. If no matching weather is established, call get_weather first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "How many top crops to return in detail. Default 5."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_financial_projection",
            "description": "Compute an exact costed financial projection (cost breakdown, revenue, net profit, ROI, break-even) for one crop and area, with exact quotable reasons. Always call it for a changed-area/price/yield scenario and never calculate financial numbers yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "Crop key: rice_boro, rice_aman, rice_aus, wheat, maize, potato, lentil, jute, mustard, onion, chili, tomato, or chickpea (aliases like rice/boro/aman/aus/masur/piyaj/morich/chola also accepted)."},
                    "area_acres": {"type": "number", "description": "Area in acres."},
                    "yield_adjustment_pct": {"type": "number", "description": "Optional % adjustment to expected yield, for scenario questions (e.g. -30 for a 30% rainfall/yield drop). Default 0."},
                    "price_override": {"type": "number", "description": "Optional override for the market price per unit, for scenario questions."},
                },
                "required": ["crop", "area_acres"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_season_calendar",
            "description": (
                "Build one chronological, dated and fully costed crop calendar from the exact "
                "sowing/transplanting date plus day offsets in crops.yaml. Returns stage, seed, "
                "labour, fertilizer, irrigation, pest and harvest events with sources; its event "
                "costs exactly reconcile with the financial projection and it returns exact "
                "quotable reasons. The latest established Open-Meteo forecast is injected "
                "automatically so rain-sensitive "
                "non-basal nitrogen applications exposed to >10 mm total rain within 48 hours "
                "can move to the first safe forecast day under 5 mm. Never invent or estimate "
                "sowing_date: get an explicit YYYY-MM-DD date from the farmer first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_key": {
                        "type": "string",
                        "description": "Crop key or accepted alias, e.g. rice_boro, boro, wheat, or potato.",
                    },
                    "sowing_date": {
                        "type": "string",
                        "description": "Farmer-confirmed sowing/transplanting date in YYYY-MM-DD format.",
                    },
                    "area_acres": {
                        "type": "number",
                        "description": "Area planted to this crop in acres.",
                    },
                },
                "required": ["crop_key", "sowing_date", "area_acres"],
            },
        },
    },
]


def _normalise_location(value):
    return " ".join(str(value or "").casefold().split())


def _weather_matches_location(weather, requested_location):
    """Return whether a cached forecast belongs to the requested place."""
    if not isinstance(weather, dict) or weather.get("error"):
        return False

    requested = _normalise_location(requested_location)
    if not requested:
        return False

    original_request = _normalise_location(weather.get("requested_location"))
    if original_request:
        return original_request == requested

    place = weather.get("location") or {}
    candidates = [
        _normalise_location(place.get("name")),
        _normalise_location(place.get("admin1")),
    ]
    return any(
        candidate
        and (
            candidate == requested
            or candidate in requested
            or requested in candidate
        )
        for candidate in candidates
    )


def _ensure_fact_shape(session_facts):
    facts = session_facts if isinstance(session_facts, dict) else {}
    for key in SESSION_FACT_KEYS:
        facts.setdefault(key, None)
    return facts


def _invalidate_facts_for_profile_update(updated_fields, context):
    """Drop only facts whose tool inputs actually changed."""
    profile = context["farmer_profile"]
    facts = context["session_facts"]
    changed = {
        key
        for key, value in updated_fields.items()
        if value is not None and profile.get(key) != value
    }
    invalidated = set()

    if "location" in changed:
        invalidated.update({"weather", "ranking", "calendar"})
        context["last_weather"] = None
    if "farm_size_acres" in changed:
        invalidated.update({"ranking", "projection", "calendar"})
    if changed.intersection(
        {"soil_type", "water_availability", "budget_bdt", "target_season"}
    ):
        invalidated.add("ranking")
    if "chosen_crop" in changed:
        canonical = normalize_key(updated_fields.get("chosen_crop"))
        previous_choice = facts.get("chosen_crop")
        new_choice = canonical or updated_fields.get("chosen_crop")
        facts["chosen_crop"] = new_choice
        updated_fields["chosen_crop"] = new_choice
        projection_crop = (
            (facts.get("projection") or {}).get("crop")
            if isinstance(facts.get("projection"), dict)
            else None
        )
        calendar_crop = (
            (facts.get("calendar") or {}).get("crop")
            if isinstance(facts.get("calendar"), dict)
            else None
        )
        if (
            (previous_choice and previous_choice != new_choice)
            or (projection_crop and projection_crop != new_choice)
            or (calendar_crop and calendar_crop != new_choice)
        ):
            invalidated.update({"projection", "calendar"})

    for key in invalidated:
        facts[key] = None
    return sorted(invalidated)


def _ensure_tool_reasons(name, result):
    """Guarantee the agent-facing result always has a top-level reasons list."""
    if not isinstance(result, dict):
        result = {"result": result}
    reasons = result.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
        result["reasons"] = reasons
    if not reasons:
        if result.get("error"):
            reasons.append(
                f"{name} returned error={result['error']}; it produced no recommendation."
            )
        else:
            reasons.append(
                f"{name} completed; use only the exact fields in this returned result."
            )
    return result


def _persist_tool_facts(name, result, context):
    """Persist successful decision facts for reuse on later turns."""
    if result.get("error"):
        return

    facts = context["session_facts"]
    if name == "get_weather":
        if result.get("cache_status") != "reused_session_fact":
            # A genuinely new forecast makes weather-dependent decisions stale.
            facts["ranking"] = None
            facts["calendar"] = None
        facts["weather"] = result
        context["last_weather"] = result
    elif name == "rank_crops":
        facts["ranking"] = result
    elif name == "compute_financial_projection":
        calendar = facts.get("calendar")
        if (
            isinstance(calendar, dict)
            and (
                calendar.get("crop") != result.get("crop")
                or calendar.get("area_acres") != result.get("area_acres")
                or result.get("yield_adjustment_pct_applied") not in (None, 0, 0.0)
            )
        ):
            facts["calendar"] = None
        facts["projection"] = result
    elif name == "build_season_calendar":
        projection = facts.get("projection")
        if (
            isinstance(projection, dict)
            and (
                projection.get("crop") != result.get("crop")
                or projection.get("area_acres") != result.get("area_acres")
                or projection.get("yield_adjustment_pct_applied")
                not in (None, 0, 0.0)
            )
        ):
            facts["projection"] = None
        facts["calendar"] = result


def _execute_tool(name, args, context):
    context["session_facts"] = _ensure_fact_shape(
        context.get("session_facts")
    )
    if name == "update_farmer_profile":
        updated = {key: value for key, value in args.items() if value is not None}
        invalidated = _invalidate_facts_for_profile_update(updated, context)
        context["farmer_profile"].update(updated)
        recorded = ", ".join(
            f"{key}={value}" for key, value in updated.items()
        ) or "(no fields)"
        reasons = [f"Recorded farmer profile fields exactly as supplied: {recorded}."]
        if invalidated:
            reasons.append(
                "Changed inputs invalidated stale session facts: "
                + ", ".join(invalidated)
                + "."
            )
        return {
            "updated_fields": updated,
            "invalidated_facts": invalidated,
            "reasons": reasons,
        }
    if name == "get_weather":
        cached = context.get("last_weather")
        if (
            not bool(args.get("refresh"))
            and _weather_matches_location(cached, args["location"])
        ):
            result = deepcopy(cached)
            result["cache_status"] = "reused_session_fact"
            reuse_reason = (
                "Reused the established Open-Meteo result for "
                f"requested_location={args['location']!r}; no weather API "
                "request was made."
            )
            result.setdefault("reasons", [])
            if reuse_reason not in result["reasons"]:
                result["reasons"].append(reuse_reason)
            return result

        result = get_weather_for_location(args["location"])
        if "error" not in result:
            result["requested_location"] = args["location"]
            result.setdefault(
                "fetched_at",
                datetime.now().astimezone().isoformat(timespec="minutes"),
            )
            fetch_reason = (
                "Open-Meteo retrieval metadata: "
                f"requested_location={result['requested_location']!r}, "
                f"fetched_at={result['fetched_at']}, "
                f"explicit_refresh={bool(args.get('refresh'))}."
            )
            result.setdefault("reasons", [])
            if fetch_reason not in result["reasons"]:
                result["reasons"].append(fetch_reason)
            context["last_weather"] = result
        return result
    if name == "search_knowledge_base":
        return search_knowledge_base(args["query"], crop_filter=args.get("crop_filter"))
    if name == "rank_crops":
        # Profile + forecast are injected from live state, not from LLM args.
        return rank_crops(
            context["farmer_profile"],
            context.get("last_weather"),
            top_n=int(args.get("top_n") or 5),
        )
    if name == "compute_financial_projection":
        return compute_financial_projection(
            args["crop"],
            args["area_acres"],
            yield_adjustment_pct=args.get("yield_adjustment_pct", 0.0),
            price_override=args.get("price_override"),
        )
    if name == "build_season_calendar":
        return build_season_calendar(
            args["crop_key"],
            args["sowing_date"],
            args["area_acres"],
            weather=context.get("last_weather"),
        )
    return {
        "error": f"Unknown tool '{name}'",
        "reasons": [f"No registered implementation exists for tool={name!r}."],
    }


def run_turn(
    conversation_history,
    user_message,
    farmer_profile,
    trace_log,
    session_facts=None,
):
    """Run one full agent turn: user message in, final assistant reply out.

    - conversation_history: list of prior {"role", "content"} dicts (no
      system prompt included -- it's rebuilt fresh each turn from the
      current farmer_profile).
    - farmer_profile: dict of known fields, mutated in place as
      update_farmer_profile calls come in.
    - trace_log: list appended in place with one entry per tool call --
      this is what the UI renders as the visible agent trace.
    - session_facts: optional live fact dictionary. When omitted, the
      Streamlit-backed session store is used. Successful weather, ranking,
      crop-choice, projection, and calendar results persist here.

    Returns the assistant's final text reply.
    """
    if session_facts is None:
        session_facts = get_session_facts()
    session_facts = _ensure_fact_shape(session_facts)

    messages = [{
        "role": "system",
        "content": build_system_prompt(farmer_profile, session_facts),
    }]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Per-turn tool context: the live profile (mutated by update_farmer_profile)
    # and the most recent established real forecast, so rank_crops and the
    # calendar use application state rather than LLM-supplied weather numbers.
    context = {
        "farmer_profile": farmer_profile,
        "last_weather": session_facts.get("weather"),
        "session_facts": session_facts,
    }

    for i in range(MAX_TOOL_ITERATIONS):
        # On the final pass, withhold the tools so the model is forced to
        # answer from everything gathered so far instead of requesting yet
        # another tool call and dead-ending on the iteration cap. A full
        # plan chains weather -> rank -> 2-3 KB lookups -> projection, so
        # the cap is reachable on a real request.
        last = (i == MAX_TOOL_ITERATIONS - 1)
        assistant_message = chat(messages, tools=None if last else TOOL_SCHEMAS)

        if not assistant_message.tool_calls:
            final_text = assistant_message.content or ""
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": final_text})
            return final_text

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_message.tool_calls
            ],
        })

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            result = _ensure_tool_reasons(
                name,
                _execute_tool(name, args, context),
            )
            _persist_tool_facts(name, result, context)

            trace_log.append({"tool": name, "arguments": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

        # Profile and durable facts may both change mid-loop. Refresh the
        # prompt so subsequent choices see current intake and established data.
        messages[0] = {
            "role": "system",
            "content": build_system_prompt(farmer_profile, session_facts),
        }

    fallback = "I'm having trouble completing that -- could you rephrase or simplify your question?"
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": fallback})
    return fallback
