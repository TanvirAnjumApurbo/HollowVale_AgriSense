"""System prompt construction and required-field tracking for farmer intake."""

from datetime import date

from tools.agronomy import SEASON_DISPLAY, infer_season

# target_season is deliberately NOT here: the growing season is derived
# deterministically from today's date (infer_season), not guessed by the LLM.
# Asking the farmer led the model to mislabel it (e.g. Kharif-1 in late July),
# which flipped the ranking. It stays an optional update_farmer_profile field
# for a farmer who explicitly plans for a different future season.
REQUIRED_FIELDS = {
    "location": "farm location (district/upazila/village in Bangladesh)",
    "farm_size_acres": "farm size in acres",
    "soil_type": "soil type (e.g. sandy, sandy loam, loam, clay loam, clay)",
    "water_availability": "water availability (e.g. low, medium, high / irrigation access)",
    "budget_bdt": "budget for the season, in BDT",
}


def current_season_line(today=None):
    """A deterministic 'today's date + current cropping season' statement for
    the prompt, so the model never has to (mis)compute the season itself."""
    today = today or date.today()
    season = infer_season(today)
    return today.isoformat(), SEASON_DISPLAY.get(season, season)


def missing_fields(profile):
    profile = profile or {}
    return [field for field in REQUIRED_FIELDS if not profile.get(field)]


def _display(value):
    """Render a stored tool value without recalculating or reformatting it."""
    return "(not returned)" if value is None else str(value)


def _reasons_digest(label, result, limit=None):
    reasons = [
        reason
        for reason in (result.get("reasons") or [])
        if isinstance(reason, str) and reason.strip()
    ]
    if limit is not None:
        reasons = reasons[:limit]
    if not reasons:
        return ""
    return (
        f"\n{label} reasons (each bullet is a verbatim stored `reasons` entry):\n"
        + "\n".join(f"  - {reason}" for reason in reasons)
    )


def _weather_digest(weather):
    forecast = weather.get("forecast") or weather
    summary = forecast.get("summary") or weather.get("summary") or {}
    location = weather.get("location") or {}
    place = location.get("name") or weather.get("requested_location") or "(unknown location)"
    fetched_at = weather.get("fetched_at") or "(time not recorded)"
    summary_line = (
        f"Weather (Open-Meteo, fetched {fetched_at}): location={place}; "
        f"total_rainfall_mm={_display(summary.get('total_rainfall_mm'))}; "
        f"forecast_days={_display(summary.get('forecast_days'))}; "
        f"avg_max_temp_c={_display(summary.get('avg_max_temp_c'))}; "
        f"avg_min_temp_c={_display(summary.get('avg_min_temp_c'))}."
    )
    return summary_line + _reasons_digest("Weather", weather, limit=3)


def _ranking_digest(ranking):
    # The workflow requires three finalists; keeping the digest to those three
    # aligns the visible facts with the four retained reasons (context + 3).
    ranked = (ranking.get("ranked") or [])[:3]
    finalists = " | ".join(
        (
            f"{item.get('crop')} overall_score={_display(item.get('overall_score'))} "
            f"total_cost_bdt={_display((item.get('economics') or {}).get('total_cost_bdt'))} "
            f"net_profit_bdt={_display((item.get('economics') or {}).get('net_profit_bdt'))} "
            f"roi_pct={_display((item.get('economics') or {}).get('roi_pct'))}"
        )
        for item in ranked
    )
    summary_line = (
        "Ranking: "
        f"as_of_date={_display(ranking.get('as_of_date'))}; "
        f"area_acres={_display(ranking.get('area_acres'))}; "
        f"season_used={_display(ranking.get('season_used'))}; "
        f"finalists={finalists or '(none returned)'}."
    )
    # Per-finalist KB sources are attached to the ranking by the orchestrator
    # (deterministic grounding); keep a compact citation line so later turns
    # can still cite them without re-retrieving.
    citation_bits = []
    for entry in (ranking.get("knowledge_base") or [])[:3]:
        titles = sorted({
            source.get("source_title")
            for source in (entry.get("sources") or [])
            if source.get("source_title")
        })
        if titles:
            citation_bits.append(f"{entry.get('crop')}: {', '.join(titles)}")
    citation_line = (
        "\nRanking KB sources (cite these) — " + "; ".join(citation_bits) + "."
        if citation_bits
        else ""
    )
    # Context + the first three finalist reasons are enough for the required
    # three-candidate narration without reinjecting the whole raw ranking.
    return summary_line + citation_line + _reasons_digest("Ranking", ranking, limit=4)


def _projection_digest(projection):
    summary_line = (
        "Projection: "
        f"crop={_display(projection.get('crop'))}; "
        f"area_acres={_display(projection.get('area_acres'))}; "
        f"total_cost_bdt={_display(projection.get('total_cost_bdt'))}; "
        f"expected_yield={_display(projection.get('expected_yield'))}; "
        f"price_per_unit_bdt={_display(projection.get('price_per_unit_bdt'))}; "
        f"revenue_bdt={_display(projection.get('revenue_bdt'))}; "
        f"net_profit_bdt={_display(projection.get('net_profit_bdt'))}; "
        f"roi_pct={_display(projection.get('roi_pct'))}; "
        f"break_even_yield_units={_display(projection.get('break_even_yield_units'))}; "
        f"break_even_price_per_unit_bdt="
        f"{_display(projection.get('break_even_price_per_unit_bdt'))}."
    )
    return summary_line + _reasons_digest("Projection", projection)


def _calendar_digest(calendar):
    header = (
        "Calendar: "
        f"crop={_display(calendar.get('crop'))}; "
        f"area_acres={_display(calendar.get('area_acres'))}; "
        f"sowing_date={_display(calendar.get('sowing_date'))}; "
        f"harvest_date={_display(calendar.get('harvest_date'))}; "
        f"duration_days={_display(calendar.get('duration_days'))}; "
        f"total_event_cost_bdt={_display(calendar.get('total_event_cost_bdt'))}; "
        f"weather_adjustments={_display(calendar.get('weather_adjustments'))}."
    )
    reasons = calendar.get("reasons") or []
    has_event_reasons = any(
        isinstance(reason, str) and reason.startswith("Calendar event ")
        for reason in reasons
    )
    if has_event_reasons:
        # The calendar tool emits one exact reason per event. Reusing those
        # strings avoids a second paraphrased representation of the same plan.
        return header + _reasons_digest("Calendar", calendar)

    # Backward compatibility for an older/stub calendar that predates event
    # reasons: keep its exact event fields visible and include any reasons it
    # does have.
    event_lines = []
    for event in calendar.get("events") or []:
        line = (
            f"  - date={_display(event.get('date'))}; "
            f"type={_display(event.get('type'))}; "
            f"action={_display(event.get('action'))}; "
            f"quantity={_display(event.get('quantity'))}; "
            f"cost_bdt={_display(event.get('cost_bdt'))}"
        )
        if event.get("shifted_from"):
            line += f"; shifted_from={event['shifted_from']}"
        if event.get("adjustment_reason"):
            line += f"; adjustment_reason={event['adjustment_reason']}"
        event_lines.append(line + ".")
    event_block = (
        "\nCalendar events (stored tool output; use these exact dates):\n"
        + "\n".join(event_lines)
        if event_lines
        else ""
    )
    return header + event_block + _reasons_digest("Calendar", calendar)


def build_facts_digest(session_facts):
    """Build the compact, quotable tool-fact block injected each turn."""
    facts = session_facts or {}
    lines = []
    weather = facts.get("weather")
    ranking = facts.get("ranking")
    chosen_crop = facts.get("chosen_crop")
    projection = facts.get("projection")
    calendar = facts.get("calendar")

    if isinstance(weather, dict) and not weather.get("error"):
        lines.append(_weather_digest(weather))
    if isinstance(ranking, dict) and not ranking.get("error"):
        lines.append(_ranking_digest(ranking))
    if chosen_crop:
        lines.append(f"Chosen crop: {chosen_crop}.")
    if isinstance(projection, dict) and not projection.get("error"):
        lines.append(_projection_digest(projection))
    if isinstance(calendar, dict) and not calendar.get("error"):
        lines.append(_calendar_digest(calendar))
    return "\n".join(lines) if lines else "(none yet)"


def build_system_prompt(profile, session_facts=None):
    profile = profile or {}
    missing = missing_fields(profile)

    known_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items() if value) or "(none yet)"
    missing_lines = (
        "\n".join(f"- {field}: {REQUIRED_FIELDS[field]}" for field in missing)
        if missing
        else "(none -- all required fields are known)"
    )
    facts_digest = build_facts_digest(session_facts)
    today_iso, season_disp = current_season_line()

    return f"""You are AgriSense AI, an agronomic advisor agent for smallholder farmers in Bangladesh.

Your job is to take a farmer from a vague opening message to a grounded, explained, costed season plan, and to keep advising through the season. The LLM gathers inputs and narrates; tools decide rankings, dates, weather adjustments, and every number.

## Current growing season (derived from today's date -- never compute it yourself)
Today is {today_iso}. From that date, the current Bangladesh cropping season is **{season_disp}**. Use this as the growing season for recommendations unless the farmer explicitly asks to plan for a different future season or month range. Never infer the season from vague words like "now", "this season", or "current" -- it is already determined for you here, and `rank_crops` uses today's date directly.

## Required intake fields
Currently known about this farmer:
{known_lines}

Still missing:
{missing_lines}

Rules for intake:
- Call `update_farmer_profile` as soon as you learn any required field. If the farmer explicitly chooses a crop, record `chosen_crop` with that same tool.
- Ask only for missing fields. Never re-ask for a known value.
- Do not ask the farmer which season it is: it is fixed by today's date ({season_disp}, see above). Only record `target_season` if the farmer explicitly names a specific different season or month range to plan for.
- Do not recommend crops or build a projection/calendar until intake is complete, unless the farmer explicitly refuses a field and you clearly state an assumption.

## Established facts (do not re-derive; quote these)
{facts_digest}

Established-fact rules:
- These are successful tool outputs from this session and are already visible in the trace panel. For the same farm and same scenario, treat them as authoritative and do not fetch or recalculate them.
- If established weather is for the current location, reuse it. Do not call `get_weather` again for a summary or an acreage-only follow-up. You may call it without `refresh` when a question needs daily fields omitted from this compact digest; the application will return the full cached forecast without a network request. For an explicit farmer-requested refresh, call it with `refresh=true`.
- If a crop is already chosen, keep it for scenario follow-ups unless the farmer chooses another crop. Do not rerun ranking merely because acreage changed unless the farmer asks for a new recommendation.
- When acreage or crop changes, rerun both financial projection and calendar. When only the sowing date changes, rerun the calendar. When only yield adjustment or sale price changes, rerun the financial projection. Never quote a stored result whose inputs do not match the requested scenario.

## Required workflow
For a new recommendation, follow this chain:
1. Complete intake.
2. Call `get_weather` when no matching established weather exists. If established weather exists, call it only for cache-served daily detail or an explicit `refresh=true` request as described above.
3. Call `rank_crops`; it receives the profile and established/live weather from application state, not from model-supplied numbers. Its result includes a per-finalist `knowledge_base` block with the source(s) retrieved for that crop -- this grounding is done for you automatically.
4. Cite those sources: for each finalist you present, quote its `source_title` and `source_url` from the ranking's `knowledge_base` block. You may also call `search_knowledge_base` yourself for extra depth on a finalist or a specific agronomic question.
5. Present at least three ranked finalists using the exact tool reasons, each with a cited knowledge-base source, then let the farmer choose.
6. Record the explicit choice. Obtain an explicit sowing/transplanting date in YYYY-MM-DD form; never invent one.
7. Call `compute_financial_projection` and `build_season_calendar` for the chosen crop and requested acreage. The calendar receives established weather from application state.
8. Narrate only from returned `reasons`, fields, calendar events, and cited retrieved evidence.

## Structural explainability rules
- Every recommendation or agronomic/financial claim must quote at least one relevant entry verbatim from the `reasons` array of the tool result that produced it. Per-crop `ranked[*].reasons` entries also qualify.
- When you present or recommend a crop, cite at least one knowledge-base source for it -- the `source_title` and `source_url` from the ranking's `knowledge_base` block, or from a `search_knowledge_base` result. Do not state agronomic facts (fertilizer doses, sowing windows, pest risks) without a cited source.
- Do not paraphrase numbers or dates. Restate them exactly as returned by the producing tool or as shown in Established facts.
- If a tool returned no relevant reason for a claim, do not make that claim.
- Never perform arithmetic in the reply. Every numeric result must come from a tool.
- Only state exact plan dates returned by `build_season_calendar`.
- When a calendar event has `shifted_from` and `adjustment_reason`, quote the adjustment reason exactly.
- If a tool returns an error, explain the error and ask for the missing/corrected input instead of guessing.
- Keep responses practical and in plain language a farmer can act on.
"""
