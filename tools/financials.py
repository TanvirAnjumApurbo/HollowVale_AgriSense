"""Deterministic cost/yield/ROI/break-even calculator.

This is intentionally plain Python arithmetic, not something the LLM is
asked to compute -- the agent calls these functions as a tool and reports
the returned numbers verbatim. That keeps the math inspectable and
guarantees changing one input (area, yield, price) changes every
downstream number consistently.

As of the dataset pass, per-acre cost/yield/price figures are NOT hardcoded
here: they are read from data/crops.yaml + data/input_prices.yaml via
data.crop_loader, so there is one source of truth shared with the agronomy
scoring engine, and fertilizer cost is DERIVED from each crop's dose
schedule (sum of kg/acre * input price) rather than a flat guess. The
figures remain ESTIMATES for a hackathon demo (general Bangladesh
agronomic/market ranges, not a live market feed) and are labelled as such.
"""

from math import isfinite
from numbers import Real

from data.crop_loader import (
    cost_breakdown_per_acre,
    get_crop,
    list_crop_keys,
    load_input_prices,
    normalize_key,
)

DATA_SOURCE_NOTE = (
    "Cost, yield and price figures are ESTIMATED for a hackathon demo, read "
    "from data/crops.yaml + data/input_prices.yaml (general Bangladesh "
    "agronomic/market ranges, not a live market feed). Fertilizer cost is "
    "derived from the crop's dose schedule, so it is inspectable against the "
    "dose table -- see README for what is real vs estimated."
)


def list_supported_crops():
    return list_crop_keys()


# Nutrient content of each priced fertilizer, % by weight. Kept here (not in
# input_prices.yaml, which is a price list) because it is chemistry, not a
# market figure: urea is 46% N wherever it is sold.
NUTRIENT_CONTENT_PCT = {
    "urea": {"n": 46.0},
    "dap": {"n": 18.0, "p2o5": 46.0},
    "ammonium_sulphate": {"n": 21.0},
    "tsp": {"p2o5": 46.0},
    "mp": {"k2o": 60.0},
    "sop": {"k2o": 50.0},
}

# Well-decomposed cow dung / farmyard manure, % by fresh weight. Low and
# variable by nature; these are the conventional planning figures used in
# Bangladeshi extension guidance.
COW_DUNG_NPK_PCT = {"n": 0.5, "p2o5": 0.25, "k2o": 0.5}

# Share of chemical nitrogen it is safe to replace with manure in one season.
# Organic N mineralises slowly, so a full swap starves the crop at the split
# top-dressings; extension guidance caps the substitution rather than
# eliminating chemical N.
ORGANIC_N_SUBSTITUTION_PCT = 25.0


def organic_alternative(crop, area_acres=1.0):
    """Cost and quantity of part-substituting the chemical dose with manure.

    Converts the crop's scheduled nitrogen into the cow dung / compost weight
    that supplies the substitutable share of it, prices both sides, and
    reports the difference. Returns the same shape as the other tools -- every
    number with a `reasons` entry naming the input it came from -- so the
    agent can offer the organic option without doing arithmetic itself.
    """
    rec = get_crop(crop)
    if rec is None:
        message = (
            f"Unknown crop '{crop}'. Supported: {', '.join(list_crop_keys())}"
        )
        return {
            "error": message,
            "reasons": [f"No organic alternative was computed: {message}"],
        }
    try:
        area = float(area_acres)
    except (TypeError, ValueError):
        area = 0.0
    if not isfinite(area) or area <= 0:
        message = f"area_acres must be a positive number, got {area_acres!r}."
        return {
            "error": message,
            "reasons": [f"No organic alternative was computed: {message}"],
        }

    prices = load_input_prices()
    schedule = rec.get("fertilizer_schedule", [])

    # Nitrogen actually scheduled, by material.
    n_by_input = {}
    for line in schedule:
        material = line.get("input")
        content = NUTRIENT_CONTENT_PCT.get(material, {}).get("n")
        if not content:
            continue
        kg = float(line.get("kg_per_acre", 0) or 0)
        n_by_input[material] = n_by_input.get(material, 0.0) + kg * content / 100.0
    chemical_n = round(sum(n_by_input.values()), 2)

    manure_price = prices.get("cow_dung")
    if chemical_n <= 0 or manure_price is None:
        note = (
            "the crop's schedule carries no nitrogen-bearing priced input"
            if chemical_n <= 0
            else "cow_dung has no price in data/input_prices.yaml"
        )
        return {
            "crop": normalize_key(crop),
            "label": rec["label"],
            "area_acres": round(area, 3),
            "available": False,
            "reasons": [
                f"No organic substitution was costed for {rec['label']}: {note}."
            ],
        }

    substituted_n = round(chemical_n * ORGANIC_N_SUBSTITUTION_PCT / 100.0, 2)
    manure_kg_per_acre = round(substituted_n / (COW_DUNG_NPK_PCT["n"] / 100.0), 1)
    manure_kg_total = round(manure_kg_per_acre * area, 1)
    manure_cost = round(manure_kg_total * manure_price, 2)

    # Urea is the split-applied N source, so that is what a manure basal dose
    # displaces. Price the displaced weight at the same list price.
    urea_price = prices.get("urea")
    urea_n = NUTRIENT_CONTENT_PCT["urea"]["n"] / 100.0
    urea_saved_kg_per_acre = round(substituted_n / urea_n, 1) if "urea" in n_by_input else 0.0
    urea_saved_kg_total = round(urea_saved_kg_per_acre * area, 1)
    urea_cost_avoided = round(
        urea_saved_kg_total * urea_price, 2
    ) if urea_price is not None else 0.0
    # Two honest cash cases. Cow dung is normally farm-supplied (crops.yaml
    # excludes it from the cost breakdown for exactly this reason), in which
    # case the only cash movement is the urea no longer bought. Priced at the
    # market rate it is dearer than the urea it replaces -- say both rather
    # than quietly picking the flattering one.
    net_change_purchased = round(manure_cost - urea_cost_avoided, 2)
    net_change_farm_supplied = round(-urea_cost_avoided, 2)
    tonnes_per_ha = round(manure_kg_per_acre * 2.471 / 1000.0, 2)

    # The manure also supplies P and K, which is part of why farmers use it.
    p2o5_supplied = round(manure_kg_total * COW_DUNG_NPK_PCT["p2o5"] / 100.0, 2)
    k2o_supplied = round(manure_kg_total * COW_DUNG_NPK_PCT["k2o"] / 100.0, 2)

    reasons = [
        (
            f"Organic option basis: {rec['label']} scheduled nitrogen is "
            f"{chemical_n} kg N/acre, derived from data/crops.yaml doses at "
            f"{', '.join(f'{k} {NUTRIENT_CONTENT_PCT[k]['n']}% N' for k in sorted(n_by_input))}. "
            f"Substituting the safe share ({ORGANIC_N_SUBSTITUTION_PCT:.0f}%) "
            f"needs {substituted_n} kg N/acre from manure."
        ),
        (
            f"Cow dung / compost quantity: {manure_kg_per_acre} kg/acre "
            f"({manure_kg_total} kg over {round(area, 3)} acres, about "
            f"{tonnes_per_ha} t/ha), at {COW_DUNG_NPK_PCT['n']}% N by weight."
        ),
        (
            f"Urea displaced: {urea_saved_kg_per_acre} kg/acre "
            f"({urea_saved_kg_total} kg total) at 46% N, worth BDT "
            f"{urea_cost_avoided:,.2f} at BDT {urea_price}/kg."
        ),
        (
            f"Cash effect if the manure is farm-supplied (the usual case, and "
            f"why data/crops.yaml leaves it out of the cost breakdown): BDT "
            f"{net_change_farm_supplied:,.2f} -- you simply buy "
            f"{urea_saved_kg_total} kg less urea, and pay only cartage and "
            f"spreading labour. If you buy the manure at the BDT "
            f"{manure_price}/kg list price in data/input_prices.yaml, the "
            f"{manure_kg_total} kg costs BDT {manure_cost:,.2f} and the switch "
            f"is BDT {net_change_purchased:,.2f} dearer than the all-chemical "
            f"dose -- worth it for the soil, not as a way to cut cash cost."
        ),
        (
            f"The same manure also supplies about {p2o5_supplied} kg P2O5 and "
            f"{k2o_supplied} kg K2O, plus organic matter that improves water "
            "holding on light soils -- the usual reason for applying it "
            "beyond the nitrogen alone."
        ),
        (
            "Application: spread and incorporate the manure at land "
            "preparation, at least 15 days before sowing/transplanting, "
            "because organic nitrogen mineralises slowly. Keep the remaining "
            f"{round(chemical_n - substituted_n, 2)} kg N/acre as the "
            "scheduled chemical top-dressing splits -- do not substitute the "
            "whole dose."
        ),
    ]

    return {
        "crop": normalize_key(crop),
        "label": rec["label"],
        "area_acres": round(area, 3),
        "available": True,
        "chemical_n_kg_per_acre": chemical_n,
        "substitution_pct": ORGANIC_N_SUBSTITUTION_PCT,
        "substituted_n_kg_per_acre": substituted_n,
        "cow_dung_kg_per_acre": manure_kg_per_acre,
        "cow_dung_kg_total": manure_kg_total,
        "cow_dung_tonnes_per_ha": tonnes_per_ha,
        "cow_dung_cost_bdt_if_purchased": manure_cost,
        "urea_displaced_kg_per_acre": urea_saved_kg_per_acre,
        "urea_displaced_kg_total": urea_saved_kg_total,
        "urea_cost_avoided_bdt": urea_cost_avoided,
        "net_cost_change_bdt_if_farm_supplied": net_change_farm_supplied,
        "net_cost_change_bdt_if_purchased": net_change_purchased,
        "p2o5_supplied_kg": p2o5_supplied,
        "k2o_supplied_kg": k2o_supplied,
        "data_source_note": DATA_SOURCE_NOTE,
        "reasons": reasons,
    }


def _water_need_label(mm):
    if mm is None:
        return "unknown"
    if mm >= 700:
        return "high"
    if mm >= 350:
        return "medium"
    return "low"


def compute_financial_projection(crop, area_acres, yield_adjustment_pct=0.0, price_override=None):
    """Compute a full costed projection for one crop over a given area.

    yield_adjustment_pct lets scenario questions ("what if rainfall drops
    30%?") scale expected yield up/down without touching the base table.
    price_override lets scenario questions swap in a different sale price.
    """
    rec = get_crop(crop) if isinstance(crop, str) else None
    if rec is None:
        message = (
            f"Unknown crop '{crop}'. Supported crops: "
            f"{', '.join(list_supported_crops())}"
        )
        return {
            "error": message,
            "reasons": [
                f"Financial projection was not computed: {message}"
            ],
        }
    if (
        isinstance(area_acres, bool)
        or not isinstance(area_acres, Real)
        or not isfinite(float(area_acres))
        or area_acres <= 0
    ):
        message = "area_acres must be a positive number."
        return {
            "error": message,
            "reasons": [
                (
                    "Financial projection was not computed because "
                    f"area_acres={area_acres!r}; {message}"
                )
            ],
        }

    crop_key = normalize_key(crop)
    per_acre_costs = cost_breakdown_per_acre(rec)
    cost_breakdown = {k: round(v * area_acres, 2) for k, v in per_acre_costs.items()}
    total_cost = round(sum(cost_breakdown.values()), 2)

    base_yield = rec["yield_per_acre_kg"] * area_acres
    expected_yield = round(base_yield * (1 + yield_adjustment_pct / 100.0), 2)

    price = price_override if price_override is not None else rec["price_per_kg_bdt"]
    revenue = round(expected_yield * price, 2)
    net_profit = round(revenue - total_cost, 2)
    roi_pct = round((net_profit / total_cost) * 100, 1) if total_cost else None

    break_even_yield_units = round(total_cost / price, 2) if price else None
    break_even_price_per_unit = round(total_cost / expected_yield, 2) if expected_yield else None

    reasons = [
        (
            f"Cost projection for crop={crop_key}, area_acres={area_acres}: "
            f"seed={cost_breakdown['seed']} BDT, "
            f"fertilizer={cost_breakdown['fertilizer']} BDT, "
            f"labour={cost_breakdown['labour']} BDT, "
            f"irrigation={cost_breakdown['irrigation']} BDT, "
            f"pesticide={cost_breakdown['pesticide']} BDT; "
            f"total_cost_bdt={total_cost}."
        ),
        (
            f"Yield and revenue for crop={crop_key}: "
            f"expected_yield={expected_yield} kg after "
            f"yield_adjustment_pct_applied={yield_adjustment_pct}; "
            f"price_per_unit_bdt={price}; revenue_bdt={revenue}."
        ),
        (
            f"Profitability for crop={crop_key}: "
            f"net_profit_bdt={net_profit}, roi_pct={roi_pct}, "
            f"break_even_yield_units={break_even_yield_units} kg, "
            "break_even_price_per_unit_bdt="
            f"{break_even_price_per_unit}."
        ),
        (
            f"Planning context for crop={crop_key}: "
            f"duration_days={rec['duration_days']}, "
            f"water_need={_water_need_label(rec.get('water_requirement_mm'))}. "
            f"{DATA_SOURCE_NOTE}"
        ),
    ]

    return {
        "crop": crop_key,
        "label": rec["label"],
        "area_acres": area_acres,
        "unit": "kg",
        "duration_days": rec["duration_days"],
        "water_need": _water_need_label(rec.get("water_requirement_mm")),
        "cost_breakdown_bdt": cost_breakdown,
        "total_cost_bdt": total_cost,
        "expected_yield": expected_yield,
        "price_per_unit_bdt": price,
        "revenue_bdt": revenue,
        "net_profit_bdt": net_profit,
        "roi_pct": roi_pct,
        "break_even_yield_units": break_even_yield_units,
        "break_even_price_per_unit_bdt": break_even_price_per_unit,
        "yield_adjustment_pct_applied": yield_adjustment_pct,
        "data_source_note": DATA_SOURCE_NOTE,
        "reasons": reasons,
    }


if __name__ == "__main__":
    import json

    proj = compute_financial_projection("rice_boro", 2)
    print(json.dumps(proj, indent=2))

    proj_half_area = compute_financial_projection("rice_boro", 1)
    assert proj_half_area["total_cost_bdt"] == round(proj["total_cost_bdt"] / 2, 2), "cost should scale linearly with area"

    # alias handling: "rice"/"boro" resolve to rice_boro
    assert compute_financial_projection("rice", 1)["crop"] == "rice_boro"
    assert compute_financial_projection("boro", 1)["crop"] == "rice_boro"
    assert compute_financial_projection("aman", 1)["crop"] == "rice_aman"

    drought = compute_financial_projection("rice_boro", 2, yield_adjustment_pct=-30)
    assert drought["expected_yield"] < proj["expected_yield"], "yield_adjustment_pct should reduce yield"
    assert drought["net_profit_bdt"] < proj["net_profit_bdt"], "lower yield should reduce profit"

    # fertilizer cost must be the schedule-derived figure, not a flat number
    fert = proj["cost_breakdown_bdt"]["fertilizer"]
    assert fert == round(7275.0 * 2, 2), f"rice_boro fertilizer should be schedule-derived, got {fert}"

    print("\nAll sanity checks passed.")
