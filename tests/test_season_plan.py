"""Deterministic season-calendar and weather-adjustment checks.

The calendar is built entirely from day offsets in ``crops.yaml``.  These
tests therefore stay offline: synthetic Open-Meteo-shaped daily arrays exercise
the adjustment pass without making a network request.
"""

from copy import deepcopy
from datetime import date

import pytest

from data.crop_loader import get_crop, load_crops
from tools.financials import compute_financial_projection
from tools.season_plan import build_season_calendar


EVENT_KEYS = {"date", "type", "action", "quantity", "cost_bdt", "source"}
EXPECTED_BORO_TYPES = {
    "stage",
    "seed",
    "labour",
    "fertilizer",
    "irrigation",
    "weed",
    "pest",
    "harvest",
}


def _events_of_type(calendar, event_type):
    return [event for event in calendar["events"] if event["type"] == event_type]


def _second_urea_split(calendar):
    return next(
        event
        for event in _events_of_type(calendar, "fertilizer")
        if "2nd split" in event["action"].lower()
    )


def _forecast(rainfall, *, wrapped=False):
    daily = {
        "time": ["2026-08-15", "2026-08-16", "2026-08-17"],
        "precipitation_sum": rainfall,
    }
    forecast = {"daily": daily}
    return {"forecast": forecast} if wrapped else forecast


def _assert_chronological(calendar):
    dates = [event["date"] for event in calendar["events"]]
    assert dates == sorted(dates)


def test_boro_calendar_has_real_dates_complete_events_and_harvest():
    calendar = build_season_calendar("rice_boro", "2026-07-01", 1)

    assert "error" not in calendar
    assert calendar["crop"] == "rice_boro"
    assert calendar["sowing_date"] == "2026-07-01"
    assert calendar["harvest_date"] == "2026-11-23"
    assert calendar["duration_days"] == 145
    assert calendar["area_acres"] == 1
    assert set(calendar) >= {
        "crop",
        "sowing_date",
        "harvest_date",
        "duration_days",
        "area_acres",
        "events",
        "total_event_cost_bdt",
        "financial_total_cost_bdt",
        "costs_reconciled",
        "weather_adjustments",
        "warnings",
    }

    assert calendar["events"]
    assert all(EVENT_KEYS <= set(event) for event in calendar["events"])
    assert {event["type"] for event in calendar["events"]} == EXPECTED_BORO_TYPES
    _assert_chronological(calendar)

    land_prep = next(
        event
        for event in _events_of_type(calendar, "stage")
        if "land preparation" in event["action"].lower()
    )
    assert land_prep["date"] == "2026-06-19"  # sowing minus the YAML's 12 days

    harvest_events = _events_of_type(calendar, "harvest")
    assert len(harvest_events) == 1
    assert harvest_events[0]["date"] == calendar["harvest_date"]


def test_every_fertilizer_line_is_present_and_sourced():
    crop = get_crop("rice_boro")
    calendar = build_season_calendar("rice_boro", "2026-07-01", 1)
    fertilizer_events = _events_of_type(calendar, "fertilizer")

    assert len(fertilizer_events) == len(crop["fertilizer_schedule"])
    assert all(event["source"] for event in fertilizer_events)


@pytest.mark.parametrize("crop_key", sorted(load_crops()))
def test_every_crop_builds_a_chronological_reconciled_calendar(crop_key):
    calendar = build_season_calendar(crop_key, "2026-01-15", 1.3)
    projection = compute_financial_projection(crop_key, 1.3)

    assert "error" not in calendar
    assert calendar["costs_reconciled"] is True
    assert calendar["financial_total_cost_bdt"] == pytest.approx(
        projection["total_cost_bdt"], abs=0.01
    )
    assert all(EVENT_KEYS <= set(event) for event in calendar["events"])
    assert all(event["source"] for event in calendar["events"])
    crop = get_crop(crop_key)
    assert len(_events_of_type(calendar, "stage")) == len(crop["stages"])
    assert len(_events_of_type(calendar, "fertilizer")) == len(
        crop["fertilizer_schedule"]
    )
    assert len(_events_of_type(calendar, "irrigation")) == len(
        crop["irrigation"]
    )
    assert len(_events_of_type(calendar, "pest")) == len(crop["pest_windows"])
    assert all(
        event["end_date"] >= event["date"]
        for event in calendar["events"]
        if event["type"] in {"stage", "irrigation", "pest"}
    )
    _assert_chronological(calendar)


@pytest.mark.parametrize("wrapped", [False, True], ids=["direct-forecast", "weather-wrapper"])
def test_heavy_rain_shifts_nitrogen_to_first_dry_day(wrapped):
    weather = _forecast([40.0, 7.0, 0.0], wrapped=wrapped)
    original_weather = deepcopy(weather)

    calendar = build_season_calendar("rice_boro", "2026-07-01", 1, weather=weather)
    second_split = _second_urea_split(calendar)

    assert weather == original_weather, "calendar building must not mutate the forecast payload"
    assert second_split["date"] == "2026-08-17"
    assert second_split["shifted_from"] == "2026-08-15"
    assert second_split["adjustment_reason"]
    assert "open-meteo" in second_split["adjustment_reason"].lower()
    assert "40" in second_split["adjustment_reason"]
    assert calendar["weather_adjustments"] == 1
    _assert_chronological(calendar)


def test_rain_and_dry_thresholds_are_strict():
    at_rain_threshold = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather=_forecast([10.0, 0.0, 0.0]),
    )
    unchanged = _second_urea_split(at_rain_threshold)
    assert unchanged["date"] == "2026-08-15"
    assert "shifted_from" not in unchanged
    assert at_rain_threshold["weather_adjustments"] == 0

    at_dry_threshold = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather=_forecast([40.0, 5.0, 4.9]),
    )
    shifted = _second_urea_split(at_dry_threshold)
    assert shifted["date"] == "2026-08-17", "exactly 5 mm is not a day under 5 mm"


def test_rain_threshold_uses_cumulative_48_hour_total():
    calendar = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather=_forecast([6.0, 6.0, 0.0]),
    )
    shifted = _second_urea_split(calendar)

    assert shifted["date"] == "2026-08-17"
    assert shifted["forecast_rain_48h_mm"] == 12.0
    assert "12" in shifted["adjustment_reason"]


def test_dry_candidate_is_rejected_if_its_next_day_is_heavily_wet():
    calendar = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather=_forecast([40.0, 0.0, 40.0]),
    )
    second_split = _second_urea_split(calendar)

    assert second_split["date"] == "2026-08-15"
    assert second_split["weather_status"] == "rain_risk_no_safe_dry_day"
    assert calendar["weather_adjustments"] == 0


def test_incorporated_basal_nitrogen_is_not_weather_shifted():
    calendar = build_season_calendar(
        "wheat",
        "2026-08-15",
        1,
        weather=_forecast([40.0, 7.0, 0.0]),
    )
    basal_urea = next(
        event
        for event in _events_of_type(calendar, "fertilizer")
        if event["input"] == "urea" and "basal" in event["action"].lower()
    )

    assert basal_urea["date"] == "2026-08-15"
    assert basal_urea["rain_sensitive"] is False
    assert "shifted_from" not in basal_urea


def test_no_forecast_dry_day_does_not_invent_a_shift():
    calendar = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather=_forecast([40.0, 7.0, 5.0]),
    )
    second_split = _second_urea_split(calendar)

    assert second_split["date"] == "2026-08-15"
    assert "shifted_from" not in second_split
    assert calendar["weather_adjustments"] == 0
    assert calendar["warnings"]


def test_calendar_reports_nitrogen_events_outside_forecast_coverage():
    calendar = build_season_calendar(
        "rice_boro",
        "2026-07-01",
        1,
        weather={
            "daily": {
                "time": ["2026-07-01", "2026-07-02"],
                "precipitation_sum": [0.0, 0.0],
            }
        },
    )

    assert calendar["weather_assessment"]["outside_forecast"] == 3
    assert all(
        event["weather_status"] == "outside_forecast"
        for event in _events_of_type(calendar, "fertilizer")
        if event["rain_sensitive"]
    )
    assert any("outside" in warning.lower() for warning in calendar["warnings"])


@pytest.mark.parametrize("area_acres", [1, 2.5])
def test_event_costs_reconcile_by_financial_category(area_acres):
    calendar = build_season_calendar("rice_boro", "2026-07-01", area_acres)
    projection = compute_financial_projection("rice_boro", area_acres)

    event_total = round(sum(event["cost_bdt"] for event in calendar["events"]), 2)
    assert event_total == pytest.approx(projection["total_cost_bdt"], abs=0.01)
    assert calendar["total_event_cost_bdt"] == pytest.approx(event_total, abs=0.01)
    assert calendar["financial_total_cost_bdt"] == pytest.approx(
        projection["total_cost_bdt"], abs=0.01
    )
    assert calendar["costs_reconciled"] is True

    event_to_projection_category = {
        "seed": "seed",
        "labour": "labour",
        "fertilizer": "fertilizer",
        "irrigation": "irrigation",
        "pest": "pesticide",
    }
    for event_type, category in event_to_projection_category.items():
        actual = round(
            sum(event["cost_bdt"] for event in _events_of_type(calendar, event_type)),
            2,
        )
        assert actual == pytest.approx(
            projection["cost_breakdown_bdt"][category],
            abs=0.01,
        )

    assert all(
        event["cost_bdt"] == 0
        for event in calendar["events"]
        if event["type"] in {"stage", "harvest"}
    )


def test_alias_and_date_object_are_normalized():
    calendar = build_season_calendar("boro", date(2026, 7, 1), 1)

    assert "error" not in calendar
    assert calendar["crop"] == "rice_boro"
    assert calendar["sowing_date"] == "2026-07-01"
    assert calendar["harvest_date"] == "2026-11-23"


@pytest.mark.parametrize(
    "crop,sowing_date,area_acres",
    [
        ("not_a_crop", "2026-07-01", 1),
        (None, "2026-07-01", 1),
        ("rice_boro", "2026/07/01", 1),
        ("rice_boro", "20260701", 1),
        ("rice_boro", "2026-W27-3", 1),
        ("rice_boro", "9999-12-31", 1),
        ("rice_boro", "2026-07-01", 0),
        ("rice_boro", "2026-07-01", -1),
    ],
    ids=[
        "unknown-crop",
        "non-string-crop",
        "invalid-date-separators",
        "compact-iso-date",
        "iso-week-date",
        "date-overflow",
        "zero-area",
        "negative-area",
    ],
)
def test_invalid_inputs_return_an_error(crop, sowing_date, area_acres):
    result = build_season_calendar(crop, sowing_date, area_acres)

    assert result.get("error")
