"""Schema + internal-consistency checks for the machine-readable crop layer.

Covers Test 1's dataset gate: crops.yaml parses, every crop carries all the
keys the tools rely on, and the derived economics line up with the YAML
(fertilizer cost really is summed from the dose schedule, area scales cost
linearly). These are cheap and catch the classic ways the data layer breaks
when someone edits a dose, renames an input, or adds a crop.
"""

import pytest

from data.crop_loader import (
    cost_breakdown_per_acre,
    fertilizer_cost_per_acre,
    load_crops,
    load_input_prices,
)
from tools.financials import compute_financial_projection

REQUIRED_KEYS = [
    "label", "seasons", "sowing_window", "duration_days", "soil_suitability",
    "water_requirement_mm", "temp_range_c", "yield_per_acre_kg", "price_per_kg_bdt",
    "seed_kg_per_acre", "seed_price_per_kg_bdt", "labour_bdt_per_acre",
    "irrigation_bdt_per_acre", "fertilizer_schedule", "irrigation", "pest_windows",
    "stages", "sources",
]
SOIL_CLASSES = ["clay", "clay_loam", "loam", "sandy_loam", "sandy"]
VALID_SEASONS = {"rabi", "kharif_1", "kharif_2"}
EXPECTED_CROPS = {
    "rice_boro", "rice_aman", "rice_aus", "wheat", "maize", "potato", "lentil",
    "jute", "mustard", "onion", "chili", "tomato", "chickpea",
}

CROPS = load_crops()
PRICES = load_input_prices()


def test_yaml_parses_and_has_expected_crops():
    assert isinstance(CROPS, dict)
    assert set(CROPS) == EXPECTED_CROPS, f"crop set drifted: {set(CROPS) ^ EXPECTED_CROPS}"


@pytest.mark.parametrize("key", sorted(EXPECTED_CROPS))
def test_crop_has_all_required_keys(key):
    rec = CROPS[key]
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    assert not missing, f"{key} missing keys: {missing}"
    # nested shapes the scorer indexes into
    assert set(rec["soil_suitability"]) == set(SOIL_CLASSES), f"{key} soil classes off"
    assert set(rec["sowing_window"]) >= {"start", "end"}
    assert set(rec["temp_range_c"]) >= {"min", "max"}
    assert rec["seasons"] and set(rec["seasons"]) <= VALID_SEASONS, f"{key} bad seasons"
    assert rec["fertilizer_schedule"], f"{key} has an empty fertilizer schedule"


@pytest.mark.parametrize("key", sorted(EXPECTED_CROPS))
def test_fertilizer_inputs_are_all_priced(key):
    """Every dose line must reference a priced input -- guards against a typo
    (e.g. 'map' for 'mp') silently zeroing part of the fertilizer cost."""
    for line in CROPS[key]["fertilizer_schedule"]:
        assert line["input"] in PRICES, f"{key}: unpriced fertilizer input '{line['input']}'"
        assert line.get("kg_per_acre", 0) > 0, f"{key}: non-positive dose in schedule"


def test_season_logic_anchors():
    # the two crops whose season is the whole point of the demo's date-awareness
    assert CROPS["rice_boro"]["seasons"] == ["rabi"]
    assert CROPS["rice_aman"]["seasons"] == ["kharif_2"]


@pytest.mark.parametrize("key", sorted(EXPECTED_CROPS))
def test_projection_cost_is_schedule_derived(key):
    """The financial projection's fertilizer line equals the schedule sum, and
    total cost scales linearly with area."""
    rec = CROPS[key]
    proj1 = compute_financial_projection(key, 1)
    proj2 = compute_financial_projection(key, 2)

    assert proj1["cost_breakdown_bdt"]["fertilizer"] == fertilizer_cost_per_acre(rec)
    assert proj2["total_cost_bdt"] == pytest.approx(proj1["total_cost_bdt"] * 2)

    per_acre = cost_breakdown_per_acre(rec)
    assert proj1["total_cost_bdt"] == pytest.approx(round(sum(per_acre.values()), 2))
    assert proj1["roi_pct"] is not None
