"""Build deterministic, dated and cost-reconciled crop season calendars.

Every calendar date is derived from an explicit sowing date plus a day offset
stored in ``data/crops.yaml``.  The weather pass may move nitrogen
applications, but only to a date that is present in the supplied daily
forecast.  It never invents a replacement date.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from math import floor, isfinite
from numbers import Real

from data.crop_loader import (
    get_crop,
    list_crop_keys,
    load_input_prices,
    normalize_key,
)
from tools.financials import compute_financial_projection


RAIN_RISK_THRESHOLD_MM = 10.0
DRY_DAY_THRESHOLD_MM = 5.0

# Weed-control checkpoint. crops.yaml carries no explicit weeding offsets, so
# the critical weed-competition window is derived from the crop cycle (weeds
# depress yield most in the early season). Weeding labour is already inside
# labour_bdt_per_acre, so the checkpoint is emitted at zero cost to keep the
# event total reconciled with the financial projection.
WEED_WINDOW_START_FRACTION = 0.15
WEED_WINDOW_END_FRACTION = 0.40
WEED_WINDOW_MIN_START_DAY = 10

# Materials in input_prices.yaml that supply nitrogen.  Urea is currently the
# only one used by the crop schedules, but keeping the classification explicit
# makes the weather rule correct if another priced nitrogen source is added.
NITROGEN_INPUTS = frozenset(
    {"urea", "dap", "npks_16", "ammonium_sulphate"}
)
_BASAL_NOTE_MARKERS = ("basal", "incorporated")

INPUT_LABELS = {
    "urea": "Urea",
    "tsp": "TSP",
    "mp": "MoP",
    "dap": "DAP",
    "gypsum": "Gypsum",
    "zinc_sulphate": "Zinc sulphate",
    "boric_acid": "Boric acid",
    "magnesium_sulphate": "Magnesium sulphate",
    "npks_16": "NPKS 16-16-16",
    "sop": "Sulphate of potash",
    "ammonium_sulphate": "Ammonium sulphate",
    "lime": "Agricultural lime",
    "cow_dung": "Cow dung / compost",
}

_EVENT_TYPE_ORDER = {
    "labour": 0,
    "stage": 1,
    "seed": 2,
    "fertilizer": 3,
    "irrigation": 4,
    "weed": 5,
    "pest": 6,
    "harvest": 7,
}


def _format_number(value):
    """Return a compact, stable human-readable number."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _parse_date(value, field_name):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                return date.fromisoformat(text)
        except ValueError:
            pass
    raise ValueError(f"{field_name} must be a valid ISO date (YYYY-MM-DD).")


def _normalize_area(area_acres):
    if isinstance(area_acres, bool) or not isinstance(area_acres, Real):
        raise ValueError("area_acres must be a positive number.")
    area = float(area_acres)
    if not isfinite(area) or area <= 0:
        raise ValueError("area_acres must be a positive number.")
    return int(area) if area.is_integer() else area


def _allocate_category_cost(total_bdt, weights):
    """Allocate an authoritative category total without losing a cent.

    ``compute_financial_projection`` rounds each cost category after scaling
    it by area.  Allocating that returned total (instead of independently
    rounding every event) guarantees the event sum reconciles with the
    projection even for fractional farm areas.
    """
    if not weights:
        return []

    total_cents = int(round(float(total_bdt) * 100))
    safe_weights = [
        max(float(weight), 0.0) if isfinite(float(weight)) else 0.0
        for weight in weights
    ]
    weight_sum = sum(safe_weights)
    if weight_sum == 0:
        safe_weights = [1.0] * len(weights)
        weight_sum = float(len(weights))

    raw_cents = [total_cents * weight / weight_sum for weight in safe_weights]
    allocated = [floor(value) for value in raw_cents]
    remainder = total_cents - sum(allocated)

    # Largest-remainder allocation is deterministic; index is the tie-breaker.
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw_cents[index] - allocated[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        allocated[index] += 1

    return [round(cents / 100.0, 2) for cents in allocated]


def _event(event_date, event_type, action, quantity, cost_bdt, source, **extra):
    result = {
        "date": event_date.isoformat(),
        "type": event_type,
        "action": action,
        "quantity": quantity,
        "cost_bdt": round(float(cost_bdt), 2),
        "source": source,
    }
    result.update(extra)
    return result


def _calendar_error(message, reason=None):
    """Return the uniform explainable error contract for the public tool."""
    return {
        "error": message,
        "reasons": [
            reason or f"Season calendar was not built: {message}"
        ],
    }


def _is_rain_sensitive_nitrogen(line):
    """Classify exposed nitrogen applications without moving basal inputs.

    A future crop row can set ``rain_sensitive`` explicitly.  For the current
    dataset, non-basal nitrogen splits are surface applications while rows
    labelled basal are incorporated during land preparation.
    """
    material = line.get("input")
    if material not in NITROGEN_INPUTS:
        return False
    if "rain_sensitive" in line:
        return bool(line["rain_sensitive"])
    note = str(line.get("note") or "").lower()
    return not any(marker in note for marker in _BASAL_NOTE_MARKERS)


def _extract_daily_rainfall(weather):
    """Return ``{date: rainfall_mm}`` plus payload warnings.

    Accepted shapes cover both weather.py public return values:

    * ``{"forecast": {"daily": ...}}`` from get_weather_for_location
    * ``{"daily": ...}`` from get_forecast
    * the daily block itself, useful for deterministic tests/callers
    """
    if weather is None:
        return {}, []
    if not isinstance(weather, dict):
        return {}, ["Weather was ignored because it is not a forecast object."]
    if weather.get("error"):
        return {}, [f"Weather was ignored: {weather['error']}"]

    if isinstance(weather.get("forecast"), dict):
        forecast = weather["forecast"]
    else:
        forecast = weather

    if isinstance(forecast.get("daily"), dict):
        daily = forecast["daily"]
    elif "time" in forecast and "precipitation_sum" in forecast:
        daily = forecast
    else:
        return {}, [
            "Weather was ignored because daily time and precipitation_sum "
            "arrays were not supplied."
        ]

    times = daily.get("time")
    rainfall = daily.get("precipitation_sum")
    if not isinstance(times, (list, tuple)) or not isinstance(
        rainfall, (list, tuple)
    ):
        return {}, [
            "Weather was ignored because daily time and precipitation_sum "
            "must be arrays."
        ]

    warnings = []
    if len(times) != len(rainfall):
        warnings.append(
            "Weather daily arrays have different lengths; only aligned "
            "date/rainfall pairs were used."
        )

    rain_by_date = {}
    invalid_pairs = 0
    for raw_date, raw_rain in zip(times, rainfall):
        try:
            forecast_date = _parse_date(raw_date, "weather daily date")
            if isinstance(raw_rain, bool):
                raise ValueError
            rain_mm = float(raw_rain)
            if not isfinite(rain_mm) or rain_mm < 0:
                raise ValueError
        except (TypeError, ValueError):
            invalid_pairs += 1
            continue
        rain_by_date[forecast_date] = rain_mm

    if invalid_pairs:
        warnings.append(
            f"Ignored {invalid_pairs} invalid weather date/rainfall pair(s)."
        )
    return rain_by_date, warnings


def _adjust_nitrogen_for_rain(
    events,
    rain_by_date,
    harvest_date,
    warnings,
    weather_provided,
):
    """Shift rain-exposed nitrogen events to forecast-backed dry dates."""
    adjustments = 0
    assessment = {
        "rain_sensitive_applications": 0,
        "fully_covered_48h_windows": 0,
        "partially_covered_48h_windows": 0,
        "outside_forecast": 0,
        "rain_risks_detected": 0,
    }

    for event in events:
        if event.get("type") != "fertilizer" or not event.get(
            "rain_sensitive"
        ):
            continue

        assessment["rain_sensitive_applications"] += 1
        if not weather_provided:
            continue
        if not rain_by_date:
            event["weather_status"] = "forecast_unavailable"
            continue

        scheduled = date.fromisoformat(event["date"])

        # Open-Meteo data here is daily, so the application day and following
        # day are the two forecast buckets covering the next 48 hours.
        risk_window = (scheduled, scheduled + timedelta(days=1))
        covered_rain = [
            (day, rain_by_date[day])
            for day in risk_window
            if day in rain_by_date
        ]
        if not covered_rain:
            event["weather_status"] = "outside_forecast"
            assessment["outside_forecast"] += 1
            continue

        if len(covered_rain) == len(risk_window):
            assessment["fully_covered_48h_windows"] += 1
        else:
            assessment["partially_covered_48h_windows"] += 1

        rain_48h = sum(rain_mm for _, rain_mm in covered_rain)
        if rain_48h <= RAIN_RISK_THRESHOLD_MM:
            event["weather_status"] = (
                "checked_no_rain_risk"
                if len(covered_rain) == len(risk_window)
                else "partial_forecast_no_rain_trigger"
            )
            continue

        assessment["rain_risks_detected"] += 1

        # The replacement must itself be dry.  When its following day is also
        # forecast, reject a superficially dry date whose own 48-hour total
        # would immediately trigger this same rule again.
        dry_candidates = []
        for candidate, candidate_rain in sorted(rain_by_date.items()):
            if (
                candidate <= scheduled
                or candidate > harvest_date
                or candidate_rain >= DRY_DAY_THRESHOLD_MM
            ):
                continue
            following_rain = rain_by_date.get(
                candidate + timedelta(days=1)
            )
            if (
                following_rain is not None
                and candidate_rain + following_rain
                > RAIN_RISK_THRESHOLD_MM
            ):
                continue
            dry_candidates.append((candidate, candidate_rain))

        if not dry_candidates:
            material = INPUT_LABELS.get(event["input"], event["input"])
            event["weather_status"] = "rain_risk_no_safe_dry_day"
            warnings.append(
                f"{material} scheduled for {scheduled.isoformat()} faces "
                f"{_format_number(rain_48h)} mm total forecast rain within "
                "48 hours, but no later safe forecast day under "
                f"{_format_number(DRY_DAY_THRESHOLD_MM)} mm is available; "
                "the date was not changed."
            )
            continue

        new_date, new_date_rain = dry_candidates[0]
        peak_date, peak_rain = max(covered_rain, key=lambda item: item[1])
        rain_details = ", ".join(
            f"{_format_number(rain_mm)} mm on {rain_date.strftime('%d %b')}"
            for rain_date, rain_mm in covered_rain
        )
        original_date = event["date"]
        event["date"] = new_date.isoformat()
        event["shifted_from"] = original_date
        event["adjustment_reason"] = (
            f"Open-Meteo forecasts {_format_number(rain_48h)} mm total "
            f"within 48 hours ({rain_details}); surface-applied nitrogen can "
            "be lost to runoff. Moved to the first later safe forecast day "
            f"under {_format_number(DRY_DAY_THRESHOLD_MM)} mm "
            f"({new_date.strftime('%d %b')}: "
            f"{_format_number(new_date_rain)} mm)."
        )
        event["weather_status"] = "shifted_for_rain"
        event["weather_source"] = "Open-Meteo"
        event["forecast_rain_mm"] = peak_rain
        event["forecast_peak_rain_date"] = peak_date.isoformat()
        event["forecast_rain_48h_mm"] = round(rain_48h, 2)
        event["rescheduled_day_rain_mm"] = new_date_rain
        adjustments += 1

    if weather_provided and assessment["outside_forecast"]:
        warnings.append(
            f"{assessment['outside_forecast']} rain-sensitive nitrogen "
            "application(s) fall outside the supplied forecast dates and "
            "were left unchanged."
        )
    if weather_provided and assessment["partially_covered_48h_windows"]:
        warnings.append(
            f"{assessment['partially_covered_48h_windows']} rain-sensitive "
            "nitrogen application(s) have only partial 48-hour forecast "
            "coverage; only supplied rainfall values were assessed."
        )

    assessment["adjustments"] = adjustments
    return adjustments, assessment


def _sort_events(events):
    events.sort(
        key=lambda event: (
            event["date"],
            _EVENT_TYPE_ORDER.get(event["type"], 99),
            event.get("_sequence", 0),
        )
    )
    for event in events:
        event.pop("_sequence", None)


def build_season_calendar(crop_key, sowing_date, area_acres, weather=None) -> dict:
    """Turn crop day offsets into a dated, weather-aware season calendar.

    ``weather`` is optional Open-Meteo-shaped daily forecast data.  When more
    than 10 mm total rain is forecast across the date of a rain-sensitive,
    non-basal nitrogen application and the following day, the application is
    moved to the first later supplied forecast date with less than 5 mm of
    rain and no known >10 mm 48-hour total.  If no such forecast date is
    available, the event remains unchanged and a warning is returned.
    """
    if not isinstance(crop_key, str):
        message = (
            f"Unknown crop '{crop_key}'. Supported crops: "
            f"{', '.join(list_crop_keys())}"
        )
        return _calendar_error(
            message,
            (
                "Season calendar was not built because "
                f"crop_key={crop_key!r} is not a supported crop name."
            ),
        )

    canonical_key = normalize_key(crop_key)
    crop = get_crop(crop_key)
    if canonical_key is None or crop is None:
        message = (
            f"Unknown crop '{crop_key}'. Supported crops: "
            f"{', '.join(list_crop_keys())}"
        )
        return _calendar_error(
            message,
            (
                "Season calendar was not built because "
                f"crop_key={crop_key!r} did not resolve to a crops.yaml key."
            ),
        )

    try:
        sowing = _parse_date(sowing_date, "sowing_date")
        area = _normalize_area(area_acres)
    except ValueError as exc:
        return _calendar_error(
            str(exc),
            (
                "Season calendar input validation failed for "
                f"sowing_date={sowing_date!r}, area_acres={area_acres!r}: "
                f"{exc}"
            ),
        )

    offsets = [0, int(crop["duration_days"])]
    for stage in crop.get("stages", []):
        offsets.extend((int(stage["day_from"]), int(stage["day_to"])))
    for line in crop.get("fertilizer_schedule", []):
        offsets.append(int(line["day"]))
    for line in crop.get("irrigation", []):
        offsets.extend((int(line["day_from"]), int(line["day_to"])))
    for line in crop.get("pest_windows", []):
        offsets.extend((int(line["day_from"]), int(line["day_to"])))
    try:
        sowing + timedelta(days=min(offsets))
        sowing + timedelta(days=max(offsets))
    except OverflowError:
        message = (
            "sowing_date is too close to the supported calendar boundary "
            "for this crop's day offsets."
        )
        return _calendar_error(
            message,
            (
                f"Season calendar for crop={canonical_key}, "
                f"sowing_date={sowing.isoformat()} requires offsets from "
                f"{min(offsets)} to {max(offsets)} days, which exceed the "
                "supported date range."
            ),
        )

    projection = compute_financial_projection(canonical_key, area)
    if "error" in projection:
        return projection

    costs = projection["cost_breakdown_bdt"]
    sources = list(crop.get("sources") or [])
    primary_source = sources[0] if sources else "data/crops.yaml"
    area_text = _format_number(area)
    harvest = sowing + timedelta(days=int(crop["duration_days"]))
    events = []
    sequence = 0

    def add(event):
        nonlocal sequence
        event["_sequence"] = sequence
        sequence += 1
        events.append(event)

    # Phenology/stage windows carry no financial cost; seed and labour are
    # separate, transparent budget events below.
    stage_offsets = []
    for stage in crop.get("stages", []):
        day_from = int(stage["day_from"])
        day_to = int(stage["day_to"])
        stage_offsets.append(day_from)
        add(
            _event(
                sowing + timedelta(days=day_from),
                "stage",
                f"Begin {stage['name']}",
                f"{area_text} acre(s); crop day {day_from} to {day_to}",
                0,
                primary_source,
                end_date=(sowing + timedelta(days=day_to)).isoformat(),
                day_offset=day_from,
            )
        )

    labour_date = sowing + timedelta(days=min(stage_offsets or [0]))
    add(
        _event(
            labour_date,
            "labour",
            "Reserve the full-season labour budget",
            f"{area_text} acre(s)",
            costs["labour"],
            primary_source,
            cost_category="labour",
            cost_source="data/crops.yaml",
            date_anchor=(
                "Earliest crop-stage date; crops.yaml provides a seasonal "
                "labour budget, not labour instalment dates."
            ),
        )
    )

    seed_per_acre = float(crop.get("seed_kg_per_acre", 0))
    seed_total = seed_per_acre * float(area)
    add(
        _event(
            sowing,
            "seed",
            f"Procure seed / planting material for {crop['label']}",
            (
                f"{_format_number(seed_per_acre)} kg/acre "
                f"({_format_number(seed_total)} kg total)"
            ),
            costs["seed"],
            primary_source,
            cost_category="seed",
            cost_source="data/crops.yaml",
            date_anchor=(
                "Sowing date; crops.yaml provides the seed rate, not a "
                "separate procurement offset."
            ),
        )
    )

    prices = load_input_prices()
    fertilizer_schedule = crop.get("fertilizer_schedule", [])
    fertilizer_weights = [
        float(line.get("kg_per_acre", 0))
        * float(prices.get(line.get("input"), 0))
        for line in fertilizer_schedule
    ]
    fertilizer_costs = _allocate_category_cost(
        costs["fertilizer"], fertilizer_weights
    )
    for line, event_cost in zip(fertilizer_schedule, fertilizer_costs):
        material_key = line["input"]
        rain_sensitive = _is_rain_sensitive_nitrogen(line)
        material = INPUT_LABELS.get(
            material_key, material_key.replace("_", " ").title()
        )
        per_acre = float(line["kg_per_acre"])
        total_kg = per_acre * float(area)
        note = str(line.get("note") or "scheduled application")
        add(
            _event(
                sowing + timedelta(days=int(line["day"])),
                "fertilizer",
                f"Apply {material} - {note}",
                (
                    f"{_format_number(per_acre)} kg/acre "
                    f"({_format_number(total_kg)} kg total)"
                ),
                event_cost,
                primary_source,
                day_offset=int(line["day"]),
                input=material_key,
                nitrogen_application=material_key in NITROGEN_INPUTS,
                rain_sensitive=rain_sensitive,
                cost_category="fertilizer",
                cost_source="data/input_prices.yaml",
            )
        )

    irrigation_schedule = crop.get("irrigation", [])
    irrigation_costs = _allocate_category_cost(
        costs["irrigation"], [1.0] * len(irrigation_schedule)
    )
    for line, event_cost in zip(irrigation_schedule, irrigation_costs):
        day_from = int(line["day_from"])
        day_to = int(line["day_to"])
        note = line.get("note")
        action = f"Irrigate during {line['stage']}"
        if note:
            action = f"{action} - {note}"
        add(
            _event(
                sowing + timedelta(days=day_from),
                "irrigation",
                action,
                (
                    f"{_format_number(line['depth_cm'])} cm over "
                    f"{area_text} acre(s)"
                ),
                event_cost,
                primary_source,
                end_date=(sowing + timedelta(days=day_to)).isoformat(),
                day_offset=day_from,
                cost_category="irrigation",
                cost_source="data/crops.yaml",
                allocation_method=(
                    "Seasonal irrigation budget divided equally across the "
                    "crop's irrigation checkpoints."
                ),
            )
        )

    # A nonzero irrigation budget without checkpoint rows would otherwise be
    # invisible and break financial reconciliation.
    if not irrigation_schedule and costs["irrigation"]:
        add(
            _event(
                sowing,
                "irrigation",
                "Reserve the seasonal irrigation budget",
                f"{area_text} acre(s)",
                costs["irrigation"],
                primary_source,
                cost_category="irrigation",
                cost_source="data/crops.yaml",
            )
        )

    pest_windows = crop.get("pest_windows", [])
    pest_weights = [
        float(line.get("cost_bdt_per_acre", 0)) for line in pest_windows
    ]
    pest_costs = _allocate_category_cost(costs["pesticide"], pest_weights)
    for line, event_cost in zip(pest_windows, pest_costs):
        day_from = int(line["day_from"])
        day_to = int(line["day_to"])
        add(
            _event(
                sowing + timedelta(days=day_from),
                "pest",
                (
                    f"Scout for {line['pest']}; if threshold is crossed, "
                    f"{line['control']}"
                ),
                f"Scout / conditionally treat {area_text} acre(s)",
                event_cost,
                primary_source,
                end_date=(sowing + timedelta(days=day_to)).isoformat(),
                day_offset=day_from,
                signs=line.get("sign"),
                cost_category="pesticide",
                cost_source="data/crops.yaml",
            )
        )

    # Weed-control checkpoint: dated and cost-neutral (weeding labour is booked
    # in the labour event above). This gives the calendar the "weed and pest
    # checkpoints" the plan calls for, alongside the insect-pest scouting rows.
    duration_days = int(crop["duration_days"])
    weed_from = max(
        WEED_WINDOW_MIN_START_DAY,
        round(WEED_WINDOW_START_FRACTION * duration_days),
    )
    weed_to = round(WEED_WINDOW_END_FRACTION * duration_days)
    if weed_to <= weed_from:
        weed_to = weed_from + 1
    add(
        _event(
            sowing + timedelta(days=weed_from),
            "weed",
            (
                "Weed-control checkpoint - keep the crop weed-free through its "
                "critical early-competition period; hand-weed or apply the "
                "recommended pre-/post-emergence herbicide"
            ),
            f"Inspect / weed {area_text} acre(s)",
            0,
            primary_source,
            end_date=(sowing + timedelta(days=weed_to)).isoformat(),
            day_offset=weed_from,
            signs=(
                "grasses, sedges and broadleaf weeds emerging between rows and "
                "competing with the young crop for light, water and nutrients"
            ),
            cost_category="labour",
            cost_source="data/crops.yaml",
            date_anchor=(
                "Critical weed-competition window, derived as ~"
                f"{int(WEED_WINDOW_START_FRACTION * 100)}-"
                f"{int(WEED_WINDOW_END_FRACTION * 100)}% of the "
                f"{duration_days}-day crop cycle; crops.yaml carries no explicit "
                "weeding offsets, and weeding labour is already in the season "
                "labour budget (shown as 0 cost here to avoid double-counting)."
            ),
        )
    )

    add(
        _event(
            harvest,
            "harvest",
            f"Harvest {crop['label']}",
            (
                f"{_format_number(projection['expected_yield'])} "
                f"{projection['unit']} expected"
            ),
            0,
            primary_source,
            day_offset=int(crop["duration_days"]),
        )
    )

    rain_by_date, warnings = _extract_daily_rainfall(weather)
    weather_adjustments, weather_assessment = _adjust_nitrogen_for_rain(
        events,
        rain_by_date,
        harvest,
        warnings,
        weather_provided=weather is not None,
    )
    _sort_events(events)

    total_event_cost = round(sum(event["cost_bdt"] for event in events), 2)
    financial_total = round(float(projection["total_cost_bdt"]), 2)
    costs_reconciled = total_event_cost == financial_total
    if not costs_reconciled:
        raise RuntimeError(
            "Season calendar cost allocation drifted from the financial "
            f"projection ({total_event_cost} != {financial_total})."
        )

    reasons = [
        (
            f"Calendar basis: crop={canonical_key}, label={crop['label']!r}, "
            f"area_acres={area}, sowing_date={sowing.isoformat()}, "
            f"duration_days={int(crop['duration_days'])}, "
            f"harvest_date={harvest.isoformat()}."
        ),
        (
            f"Calendar costs: cost_breakdown_bdt={costs}, "
            f"total_event_cost_bdt={total_event_cost}, "
            f"financial_total_cost_bdt={financial_total}, "
            f"costs_reconciled={costs_reconciled}."
        ),
    ]
    for index, event in enumerate(events, start=1):
        window = (
            f", end_date={event['end_date']}"
            if event.get("end_date")
            else ""
        )
        ancillary = "".join(
            f", {key}={event[key]!r}"
            for key in (
                "signs",
                "weather_status",
                "rain_sensitive",
                "input",
                "day_offset",
            )
            if key in event
        )
        reasons.append(
            (
                f"Calendar event {index}: date={event['date']}{window}, "
                f"type={event['type']}, action={event['action']!r}, "
                f"quantity={event['quantity']!r}, "
                f"cost_bdt={event['cost_bdt']}, "
                f"source={event['source']!r}{ancillary}."
            )
        )
        if event.get("shifted_from"):
            reasons.append(
                (
                    f"Weather adjustment for event {index}: "
                    f"shifted_from={event['shifted_from']}, "
                    f"new_date={event['date']}, "
                    f"forecast_rain_48h_mm="
                    f"{event.get('forecast_rain_48h_mm')}, "
                    f"rescheduled_day_rain_mm="
                    f"{event.get('rescheduled_day_rain_mm')}; "
                    f"adjustment_reason={event['adjustment_reason']!r}."
                )
            )
    reasons.append(
        (
            f"Weather pass: weather_adjustments={weather_adjustments}, "
            f"weather_assessment={weather_assessment}, warnings={warnings}."
        )
    )

    return {
        "crop": canonical_key,
        "label": crop["label"],
        "area_acres": area,
        "sowing_date": sowing.isoformat(),
        "harvest_date": harvest.isoformat(),
        "duration_days": int(crop["duration_days"]),
        "events": events,
        "total_event_cost_bdt": total_event_cost,
        "financial_total_cost_bdt": financial_total,
        "cost_breakdown_bdt": costs,
        "costs_reconciled": costs_reconciled,
        "weather_adjustments": weather_adjustments,
        "weather_assessment": weather_assessment,
        "warnings": warnings,
        "source_refs": sources,
        "date_basis": (
            "Agronomic event dates are sowing_date plus day offsets from "
            "data/crops.yaml; harvest is sowing_date plus duration_days. "
            "Seed/labour budget anchors and cost allocation methods are "
            "labelled on their events. Weather shifts use only dates present "
            "in the supplied forecast."
        ),
        "data_source_note": projection["data_source_note"],
        "reasons": reasons,
    }


if __name__ == "__main__":
    import json

    sample = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather={
            "daily": {
                "time": ["2026-08-15", "2026-08-16", "2026-08-17"],
                "precipitation_sum": [40.0, 7.0, 0.0],
            }
        },
    )
    print(json.dumps(sample, indent=2))
