"""Load the machine-readable crop layer (crops.yaml + input_prices.yaml)
and derive per-acre costs from it.

This is the single source of truth for crop agronomy and economics: both
tools/financials.py (costed projection) and tools/agronomy.py (suitability
scoring) read crops through here, so a change to a dose or a price flows
consistently into every downstream number. Nothing here is computed by the
LLM -- it's plain, inspectable Python arithmetic over the YAML.

The headline design point (see crops.yaml header): fertilizer cost is
SUMMED FROM THE SCHEDULE -- sum(kg_per_acre * input_price) -- not a flat
per-crop guess, so the fertilizer number is auditable against the dose table.
"""

import os

import yaml

_DIR = os.path.dirname(__file__)
CROPS_PATH = os.path.join(_DIR, "crops.yaml")
PRICES_PATH = os.path.join(_DIR, "input_prices.yaml")

_crops = None
_prices = None

# Common ways a crop gets named (by the LLM, the user, or the KB layer) mapped
# to the canonical crops.yaml key. Keeps the tool tolerant of "rice"/"boro"/
# "paddy" all meaning the flagship Boro crop, while still distinguishing Aman.
ALIASES = {
    "rice": "rice_boro",
    "boro": "rice_boro",
    "boro_rice": "rice_boro",
    "paddy": "rice_boro",
    "aman": "rice_aman",
    "aman_rice": "rice_aman",
    "t_aman": "rice_aman",
    "transplanted_aman": "rice_aman",
    "masur": "lentil",
    "lentil_masur": "lentil",
    "sarisha": "mustard",
    "rai": "mustard",
    "mustard_sarisha": "mustard",
    "aus": "rice_aus",
    "aus_rice": "rice_aus",
    "piyaj": "onion",
    "peyaj": "onion",
    "pyaj": "onion",
    "morich": "chili",
    "chilli": "chili",
    "green_chili": "chili",
    "tomatoes": "tomato",
    "chola": "chickpea",
    "chhola": "chickpea",
    "chana": "chickpea",
    "chick_pea": "chickpea",
    "gram": "chickpea",
    "bengal_gram": "chickpea",
}


def load_crops():
    global _crops
    if _crops is None:
        with open(CROPS_PATH, "r", encoding="utf-8") as f:
            _crops = yaml.safe_load(f)
    return _crops


def load_input_prices():
    global _prices
    if _prices is None:
        with open(PRICES_PATH, "r", encoding="utf-8") as f:
            _prices = yaml.safe_load(f)
    return _prices


def list_crop_keys():
    return list(load_crops().keys())


def normalize_key(crop):
    """Map a free-form crop name to a canonical crops.yaml key, or None."""
    if not crop:
        return None
    key = crop.strip().lower().replace(" ", "_").replace("-", "_")
    crops = load_crops()
    if key in crops:
        return key
    if key in ALIASES and ALIASES[key] in crops:
        return ALIASES[key]
    return None


def get_crop(crop):
    """Return the full crop record for a name/alias, or None if unknown."""
    key = normalize_key(crop)
    return load_crops().get(key) if key else None


def fertilizer_cost_per_acre(crop_record):
    """Derive fertilizer cost from the dose schedule and input prices.

    sum(kg_per_acre * price) over every line in fertilizer_schedule. An
    input with no price in input_prices.yaml contributes 0 and is ignored
    (so unpriced/organic inputs don't silently inflate the number).
    """
    prices = load_input_prices()
    total = 0.0
    for line in crop_record.get("fertilizer_schedule", []):
        price = prices.get(line.get("input"))
        if price is None:
            continue
        total += line.get("kg_per_acre", 0) * price
    return round(total, 2)


def pesticide_cost_per_acre(crop_record):
    """Budgeted plant-protection cost = sum of pest_windows cost_bdt_per_acre."""
    return round(sum(w.get("cost_bdt_per_acre", 0) for w in crop_record.get("pest_windows", [])), 2)


def cost_breakdown_per_acre(crop_record):
    """Full per-acre cost breakdown, every line derived from the YAML."""
    seed = round(crop_record.get("seed_kg_per_acre", 0) * crop_record.get("seed_price_per_kg_bdt", 0), 2)
    return {
        "seed": seed,
        "fertilizer": fertilizer_cost_per_acre(crop_record),
        "labour": round(crop_record.get("labour_bdt_per_acre", 0), 2),
        "irrigation": round(crop_record.get("irrigation_bdt_per_acre", 0), 2),
        "pesticide": pesticide_cost_per_acre(crop_record),
    }


if __name__ == "__main__":
    import json

    crops = load_crops()
    print(f"Loaded {len(crops)} crops: {', '.join(crops)}")
    for key, rec in crops.items():
        bd = cost_breakdown_per_acre(rec)
        print(f"\n{key} ({rec['label']})")
        print(f"  seasons={rec['seasons']} sowing={rec['sowing_window']} duration={rec['duration_days']}d")
        print(f"  cost/acre: {json.dumps(bd)}  total={round(sum(bd.values()),2)}")
