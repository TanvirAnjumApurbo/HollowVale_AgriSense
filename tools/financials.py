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
