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
from tools.financials import (
    COW_DUNG_NPK_PCT,
    NUTRIENT_CONTENT_PCT,
    ORGANIC_N_SUBSTITUTION_PCT,
    compute_financial_projection,
    organic_alternative,
)

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


@pytest.mark.parametrize("key", sorted(EXPECTED_CROPS))
def test_organic_alternative_is_dose_derived_and_scales(key):
    """The manure substitution is derived from the crop's own nitrogen dose,
    replaces only the safe share of it, and scales with area."""
    one = organic_alternative(key, 1)
    two = organic_alternative(key, 2)
    assert one["available"] is True

    # Nitrogen comes from the schedule, not a constant.
    rec = CROPS[key]
    scheduled_n = sum(
        line["kg_per_acre"] * NUTRIENT_CONTENT_PCT[line["input"]]["n"] / 100.0
        for line in rec["fertilizer_schedule"]
        if line["input"] in NUTRIENT_CONTENT_PCT
        and "n" in NUTRIENT_CONTENT_PCT[line["input"]]
    )
    assert one["chemical_n_kg_per_acre"] == pytest.approx(scheduled_n, abs=0.01)

    # Only the safe share is substituted -- never the whole dose.
    assert one["substituted_n_kg_per_acre"] < one["chemical_n_kg_per_acre"]
    assert one["substituted_n_kg_per_acre"] == pytest.approx(
        scheduled_n * ORGANIC_N_SUBSTITUTION_PCT / 100.0, abs=0.01
    )

    # The manure supplies exactly the substituted nitrogen at its N content.
    assert one["cow_dung_kg_per_acre"] * COW_DUNG_NPK_PCT["n"] / 100.0 == pytest.approx(
        one["substituted_n_kg_per_acre"], abs=0.05
    )
    assert two["cow_dung_kg_total"] == pytest.approx(one["cow_dung_kg_total"] * 2)

    # Farm-supplied is a saving (urea not bought); purchased is dearer. Saying
    # both is the point -- quoting only one would mislead.
    assert one["net_cost_change_bdt_if_farm_supplied"] <= 0
    assert (
        one["net_cost_change_bdt_if_purchased"]
        > one["net_cost_change_bdt_if_farm_supplied"]
    )
    assert one["reasons"]


def test_organic_alternative_rejects_bad_input():
    assert organic_alternative("moon_rice", 1).get("error")
    assert organic_alternative("rice_boro", 0).get("error")
