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
from tools.financials import compute_financial_projection, rank_candidate_crops

MAX_TOOL_ITERATIONS = 8

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
            "name": "rank_candidate_crops",
            "description": "Get a deterministic profit/ROI snapshot across candidate crops for the farm's area, to use as the financial basis for ranking crop candidates. Combine this with search_knowledge_base for suitability/risk grounding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area_acres": {"type": "number", "description": "Farm size in acres."},
                    "water_availability": {"type": "string", "description": "Farmer's water availability, used to flag irrigation risk."},
                },
                "required": ["area_acres"],
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
                    "crop": {"type": "string", "description": "Crop key, e.g. rice, wheat, maize, potato, lentil, jute."},
                    "area_acres": {"type": "number", "description": "Area in acres."},
                    "yield_adjustment_pct": {"type": "number", "description": "Optional % adjustment to expected yield, for scenario questions (e.g. -30 for a 30% rainfall/yield drop). Default 0."},
                    "price_override": {"type": "number", "description": "Optional override for the market price per unit, for scenario questions."},
                },
                "required": ["crop", "area_acres"],
            },
        },
    },
]


def _execute_tool(name, args):
    if name == "update_farmer_profile":
        return {"updated_fields": {k: v for k, v in args.items() if v is not None}}
    if name == "get_weather":
        return get_weather_for_location(args["location"])
    if name == "search_knowledge_base":
        return search_knowledge_base(args["query"], crop_filter=args.get("crop_filter"))
    if name == "rank_candidate_crops":
        return rank_candidate_crops(args["area_acres"], water_availability=args.get("water_availability"))
    if name == "compute_financial_projection":
        return compute_financial_projection(
            args["crop"],
            args["area_acres"],
            yield_adjustment_pct=args.get("yield_adjustment_pct", 0.0),
            price_override=args.get("price_override"),
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

    for _ in range(MAX_TOOL_ITERATIONS):
        assistant_message = chat(messages, tools=TOOL_SCHEMAS)

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

            result = _execute_tool(name, args)

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
