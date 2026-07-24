"""Hand-rolled tool-calling agent loop.

Deliberately not using a framework (LangChain/CrewAI/etc): at hackathon
scale, a ~150-line manual loop is fully transparent, which is exactly
what the "visible agent trace" requirement needs -- every tool call,
its parameters, and its raw return value are captured here and nothing
is hidden inside framework internals.
"""

import json

from agent.llm import chat
from agent.prompts import build_system_prompt, missing_fields
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
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a real live rainfall/temperature forecast for the farm's location via Open-Meteo. Returns real geocoded location info plus the daily forecast and summary totals. Never invent these numbers yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name as given by the farmer."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Retrieve relevant agronomic passages (fertilizer doses, sowing windows, soil suitability, pest risk) from the grounded knowledge base. Use this instead of relying on your own memory for crop/fertilizer/season-plan facts.",
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
                "(soil, area, water availability, budget, target season) and the latest weather forecast "
                "you fetched -- you do not pass these, the agent supplies them. Returns, per crop, a 0-1 "
                "overall suitability score plus soil_fit/season_fit/water_fit/temp_fit/profit_score components, "
                "an economics block, a budget flag (with max affordable area), and a `reasons` list that already "
                "explains each score in plain language with the exact input values used. NARRATE those reasons; "
                "do not invent your own suitability judgement or numbers. It is season-aware: for today's date it "
                "knows which crops are actually plantable now (e.g. Aman rice in Kharif-2), so trust its ranking "
                "over generic crop knowledge. Call get_weather BEFORE this so the water/temp scores use real data."
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
            "description": "Compute an exact costed financial projection (cost breakdown, revenue, net profit, ROI, break-even) for one crop and area. Always use this for any financial numbers in a season plan -- never compute them yourself.",
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
                "costs exactly reconcile with the financial projection. The latest Open-Meteo "
                "forecast fetched in this same turn is injected automatically so rain-sensitive "
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


def _execute_tool(name, args, context):
    if name == "update_farmer_profile":
        return {"updated_fields": {k: v for k, v in args.items() if v is not None}}
    if name == "get_weather":
        result = get_weather_for_location(args["location"])
        # Cache the real forecast so rank_crops can score water/temp against it
        # instead of trusting numbers relayed (and possibly fudged) by the LLM.
        if "error" not in result:
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
    return {"error": f"Unknown tool '{name}'"}


def run_turn(conversation_history, user_message, farmer_profile, trace_log):
    """Run one full agent turn: user message in, final assistant reply out.

    - conversation_history: list of prior {"role", "content"} dicts (no
      system prompt included -- it's rebuilt fresh each turn from the
      current farmer_profile).
    - farmer_profile: dict of known fields, mutated in place as
      update_farmer_profile calls come in.
    - trace_log: list appended in place with one entry per tool call --
      this is what the UI renders as the visible agent trace.

    Returns the assistant's final text reply.
    """
    messages = [{"role": "system", "content": build_system_prompt(farmer_profile)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Per-turn tool context: the live profile (mutated by update_farmer_profile)
    # and the most recent real forecast, so rank_crops scores against injected
    # state rather than LLM-supplied arguments.
    context = {"farmer_profile": farmer_profile, "last_weather": None}

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

            result = _execute_tool(name, args, context)

            if name == "update_farmer_profile":
                for k, v in args.items():
                    if v is not None:
                        farmer_profile[k] = v

            trace_log.append({"tool": name, "arguments": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

        # profile changed mid-loop (e.g. update_farmer_profile) -- refresh
        # the system prompt so subsequent tool-choice reasoning in this
        # same turn sees the up-to-date missing-fields list.
        messages[0] = {"role": "system", "content": build_system_prompt(farmer_profile)}

    fallback = "I'm having trouble completing that -- could you rephrase or simplify your question?"
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": fallback})
    return fallback
