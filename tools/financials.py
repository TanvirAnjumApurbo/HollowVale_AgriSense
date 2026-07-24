"""Deterministic cost/yield/ROI/break-even calculator.

This is intentionally plain Python arithmetic, not something the LLM is
asked to compute -- the agent calls these functions as a tool and reports
the returned numbers verbatim. That keeps the math inspectable and
guarantees changing one input (area, yield, price) changes every
downstream number consistently.

Per-acre cost/yield/price figures below are ESTIMATED for common
Bangladesh crops based on general agronomic/market ranges, not pulled
from a live market feed. This is called out explicitly wherever this
data is surfaced, and must be stated as mock/estimated in the README.
"""

CROP_DATA = {
    "rice": {
        "label": "Rice (Boro, HYV)",
        "unit": "kg",
        "yield_per_acre": 2400,
        "market_price_per_unit": 28,
        "cost_per_acre": {
            "seed": 1500,
            "fertilizer": 5000,
            "labor": 10000,
            "irrigation": 4000,
            "pesticide": 2000,
        },
        "duration_days": 145,
        "water_need": "high",
    },
    "wheat": {
        "label": "Wheat",
        "unit": "kg",
        "yield_per_acre": 1400,
        "market_price_per_unit": 35,
        "cost_per_acre": {
            "seed": 1600,
            "fertilizer": 3500,
            "labor": 6000,
            "irrigation": 1500,
            "pesticide": 1000,
        },
        "duration_days": 110,
        "water_need": "low",
    },
    "maize": {
        "label": "Maize",
        "unit": "kg",
        "yield_per_acre": 3500,
        "market_price_per_unit": 24,
        "cost_per_acre": {
            "seed": 2500,
            "fertilizer": 6000,
            "labor": 7000,
            "irrigation": 2500,
            "pesticide": 1500,
        },
        "duration_days": 115,
        "water_need": "medium",
    },
    "potato": {
        "label": "Potato",
        "unit": "kg",
        "yield_per_acre": 8000,
        "market_price_per_unit": 15,
        "cost_per_acre": {
            "seed": 12000,
            "fertilizer": 7000,
            "labor": 12000,
            "irrigation": 3000,
            "pesticide": 3000,
        },
        "duration_days": 95,
        "water_need": "medium",
    },
    "lentil": {
        "label": "Lentil (Masur)",
        "unit": "kg",
        "yield_per_acre": 500,
        "market_price_per_unit": 90,
        "cost_per_acre": {
            "seed": 1200,
            "fertilizer": 1500,
            "labor": 4000,
            "irrigation": 500,
            "pesticide": 800,
        },
        "duration_days": 100,
        "water_need": "low",
    },
    "jute": {
        "label": "Jute",
        "unit": "kg",
        "yield_per_acre": 900,
        "market_price_per_unit": 60,
        "cost_per_acre": {
            "seed": 800,
            "fertilizer": 3000,
            "labor": 8000,
            "irrigation": 1000,
            "pesticide": 1200,
        },
        "duration_days": 120,
        "water_need": "medium",
    },
}

DATA_SOURCE_NOTE = (
    "Cost, yield, and price figures are ESTIMATED for a hackathon demo "
    "based on general Bangladesh agronomic/market ranges. They are not "
    "pulled from a live market feed -- see README for what is real vs mock."
)


def list_supported_crops():
    return list(CROP_DATA.keys())


def compute_financial_projection(crop, area_acres, yield_adjustment_pct=0.0, price_override=None):
    """Compute a full costed projection for one crop over a given area.

    yield_adjustment_pct lets scenario questions ("what if rainfall drops
    30%?") scale expected yield up/down without touching the base table.
    price_override lets scenario questions swap in a different sale price.
    """
    crop_key = crop.strip().lower()
    if crop_key not in CROP_DATA:
        return {"error": f"Unknown crop '{crop}'. Supported crops: {', '.join(list_supported_crops())}"}
    if area_acres is None or area_acres <= 0:
        return {"error": "area_acres must be a positive number."}

    data = CROP_DATA[crop_key]
    per_acre_costs = data["cost_per_acre"]
    cost_breakdown = {k: round(v * area_acres, 2) for k, v in per_acre_costs.items()}
    total_cost = round(sum(cost_breakdown.values()), 2)

    base_yield = data["yield_per_acre"] * area_acres
    expected_yield = round(base_yield * (1 + yield_adjustment_pct / 100.0), 2)

    price = price_override if price_override is not None else data["market_price_per_unit"]
    revenue = round(expected_yield * price, 2)
    net_profit = round(revenue - total_cost, 2)
    roi_pct = round((net_profit / total_cost) * 100, 1) if total_cost else None

    break_even_yield_units = round(total_cost / price, 2) if price else None
    break_even_price_per_unit = round(total_cost / expected_yield, 2) if expected_yield else None

    return {
        "crop": crop_key,
        "label": data["label"],
        "area_acres": area_acres,
        "unit": data["unit"],
        "duration_days": data["duration_days"],
        "water_need": data["water_need"],
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
    }


def rank_candidate_crops(area_acres, crops=None, water_availability=None):
    """Quick profit/risk snapshot across candidate crops for ranking.

    water_availability, if given (e.g. "low"/"medium"/"high"), is used only
    to flag a simple risk note -- crop suitability against soil/season/
    weather should still be grounded via the knowledge base, this just
    supplies the deterministic profit/water-need side of the ranking.
    """
    candidates = crops or list_supported_crops()
    results = []
    for crop in candidates:
        projection = compute_financial_projection(crop, area_acres)
        if "error" in projection:
            continue
        risk_note = None
        if water_availability and projection["water_need"] == "high" and water_availability.lower() in ("low", "limited"):
            risk_note = "High water need vs. limited water availability -- irrigation risk."
        results.append({
            "crop": projection["crop"],
            "label": projection["label"],
            "water_need": projection["water_need"],
            "duration_days": projection["duration_days"],
            "net_profit_bdt": projection["net_profit_bdt"],
            "roi_pct": projection["roi_pct"],
            "risk_note": risk_note,
        })
    results.sort(key=lambda r: r["net_profit_bdt"], reverse=True)
    return results


if __name__ == "__main__":
    import json

    proj = compute_financial_projection("rice", 2)
    print(json.dumps(proj, indent=2))

    proj_half_area = compute_financial_projection("rice", 1)
    assert proj_half_area["total_cost_bdt"] == round(proj["total_cost_bdt"] / 2, 2), "cost should scale linearly with area"

    drought = compute_financial_projection("rice", 2, yield_adjustment_pct=-30)
    assert drought["expected_yield"] < proj["expected_yield"], "yield_adjustment_pct should reduce yield"
    assert drought["net_profit_bdt"] < proj["net_profit_bdt"], "lower yield should reduce profit"

    print("\nranked crops for 2 acres, low water availability:")
    print(json.dumps(rank_candidate_crops(2, water_availability="low"), indent=2))

    print("\nAll sanity checks passed.")
