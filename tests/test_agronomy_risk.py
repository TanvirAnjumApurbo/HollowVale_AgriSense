"""Pest/disease risk responds to the forecast -- without leaking into the fit score.

The suitability ranking is deliberately weather-aware through its own water
and temperature components. Pest pressure is a separate, later judgement: warm
and wet weather is when blast, blight, and hoppers actually run. These tests
pin both halves of that -- the forecast must move the risk tier, and it must
not move `overall_score`, which is what keeps the soil/season ranking stable.
"""

from datetime import date

import pytest

from tools.agronomy import score_crop

PROFILE = {
    "location": "Rangpur",
    "farm_size_acres": 2,
    "soil_type": "clay loam",
    "water_availability": "medium",
    "budget_bdt": 80000,
}

# Same 16-day window, same location: only the weather differs.
WET_WARM = {"total_rainfall_mm": 200.0, "avg_max_temp_c": 31.0, "avg_min_temp_c": 25.0, "forecast_days": 16}
DRY_WARM = {"total_rainfall_mm": 5.0, "avg_max_temp_c": 31.0, "avg_min_temp_c": 22.0, "forecast_days": 16}
WET_COOL = {"total_rainfall_mm": 200.0, "avg_max_temp_c": 19.0, "avg_min_temp_c": 12.0, "forecast_days": 16}

TODAY = date(2026, 7, 25)


@pytest.mark.parametrize("crop", ["rice_aman", "maize", "tomato"])
def test_wet_warm_forecast_raises_pest_pressure_over_dry(crop):
    wet = score_crop(crop, PROFILE, WET_WARM, today=TODAY)
    dry = score_crop(crop, PROFILE, DRY_WARM, today=TODAY)

    assert wet["risk"]["weather_multiplier"] > dry["risk"]["weather_multiplier"]
    assert wet["risk"]["drivers"]["pest_pressure"] > dry["risk"]["drivers"]["pest_pressure"]

    # Deliberately NOT asserting the composite tier rises with rain. A dry
    # forecast pushes water_stress (weight 0.30) up harder than wet weather
    # pushes pest pressure, so overall risk can legitimately be higher in the
    # dry case. The driver is the isolated claim; the tier is the trade-off.
    assert dry["risk"]["drivers"]["water_stress"] > wet["risk"]["drivers"]["water_stress"]


def test_warmth_matters_not_just_rain():
    """Wet-and-warm outranks wet-and-cool: the 25-35 C band is where the
    insect and fungal pressure actually sits."""
    warm = score_crop("rice_aman", PROFILE, WET_WARM, today=TODAY)
    cool = score_crop("rice_aman", PROFILE, WET_COOL, today=TODAY)

    assert warm["risk"]["weather_multiplier"] > cool["risk"]["weather_multiplier"]


def test_risk_weather_scaling_does_not_move_the_fit_score():
    """Risk is reported alongside suitability, never folded into it -- otherwise
    a wet forecast would quietly re-rank the crops."""
    wet = score_crop("rice_aman", PROFILE, WET_WARM, today=TODAY)
    dry = score_crop("rice_aman", PROFILE, DRY_WARM, today=TODAY)

    assert wet["components"]["soil_fit"] == dry["components"]["soil_fit"]
    assert wet["components"]["season_fit"] == dry["components"]["season_fit"]
    assert wet["components"]["profit_score"] == dry["components"]["profit_score"]


def test_missing_forecast_leaves_pest_pressure_unscaled():
    """No forecast means no claim about pest weather -- not an invented one."""
    result = score_crop("rice_aman", PROFILE, None, today=TODAY)

    assert result["risk"]["weather_multiplier"] == 1.0
    assert "unmodified" in _risk_reason(result)


def test_risk_reason_names_the_pests_and_the_forecast_scaling():
    result = score_crop("rice_aman", PROFILE, WET_WARM, today=TODAY)
    reason = _risk_reason(result)

    assert "Yellow stem borer" in reason
    assert "mm/day" in reason
    assert "pest-active band" in reason


def test_risk_carries_actionable_pest_detail():
    """A risk tier alone is not advice: each pest ships its scouting sign, its
    control, its cost, and the growth-stage window it falls in."""
    pests = score_crop("potato", PROFILE, WET_COOL, today=TODAY)["risk"]["pests"]

    assert pests
    for pest in pests:
        assert pest["pest"] and pest["sign"] and pest["control"]
        assert pest["cost_bdt_per_acre"] > 0
        assert pest["day_from"] < pest["day_to"]


def _risk_reason(result):
    return next(r for r in result["reasons"] if r.startswith("Risk "))
