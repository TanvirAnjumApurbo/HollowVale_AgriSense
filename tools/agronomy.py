"""Deterministic multi-factor crop suitability + economics scoring.

This replaces "sort candidate crops by profit" with real agronomic
reasoning, and -- crucially -- it emits that reasoning AS DATA. Every crop
gets component scores (soil / season / water / temp / profit) in 0-1, an
overall weighted score, a budget flag, and a `reasons` list where each
string names the exact input value it used. The LLM's job is then to
NARRATE these reasons, not to invent the judgement: the ranking, the
numbers and the explanations are all computed here in plain Python.

Weights (overall = weighted sum; budget is a flag, not weighted):
    soil_fit    0.25
    season_fit  0.25
    water_fit   0.20
    temp_fit    0.10
    profit_score 0.20
"""

from datetime import date
from math import isfinite

from data.crop_loader import cost_breakdown_per_acre, get_crop, list_crop_keys
from tools.financials import compute_financial_projection

WEIGHTS = {"soil_fit": 0.25, "season_fit": 0.25, "water_fit": 0.20, "temp_fit": 0.10, "profit_score": 0.20}

# Profit is scored by mapping ROI onto 0-1 against a strong-ROI benchmark
# (ROI >= this % scores a full 1.0), NOT by min-max across the candidates.
# That is deliberate: min-max lets a single low-cost outlier crop (a pulse
# like lentil often clears 400%+ ROI) peg itself at 1.0 and crush everyone
# else near 0, which makes the 0.20 profit weight swamp the 0.25 soil weight
# so soil/season can never change the ranking. Bounding keeps profit an
# input to the decision, not the decider -- the agronomic fit stays in charge.
PROFIT_REF_PCT = 250.0

SEASON_DISPLAY = {"rabi": "Rabi", "kharif_1": "Kharif-1", "kharif_2": "Kharif-2"}

SOIL_SYNONYMS = {
    "clay": "clay", "clayey": "clay", "heavy": "clay",
    "clay_loam": "clay_loam", "clayloam": "clay_loam",
    "loam": "loam", "loamy": "loam", "silt": "loam", "silty": "loam", "silt_loam": "loam", "medium": "loam",
    "sandy_loam": "sandy_loam", "sandyloam": "sandy_loam",
    "sandy": "sandy", "sand": "sandy", "light": "sandy",
}

IRRIGATION_ACCESS = {"low": 0.2, "limited": 0.2, "medium": 0.6, "moderate": 0.6, "high": 1.0, "good": 1.0, "reliable": 1.0}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _num(x):
    """Coerce a possibly-stringy profile value to float, or None."""
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _profile_area(profile):
    """Return a positive acreage or an explainable validation error."""
    raw = (profile or {}).get("farm_size_acres")
    if raw is None or raw == "":
        return 1.0, None
    area = _num(raw)
    if area is None or not isfinite(area) or area <= 0:
        return None, (
            "farm_size_acres must be a positive finite number; "
            f"received {raw!r}."
        )
    return area, None


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------

def _normalize_soil(raw):
    if not raw:
        return None, None
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return SOIL_SYNONYMS.get(key, key if key in ("clay", "clay_loam", "loam", "sandy_loam", "sandy") else None), raw


def _normalize_season(raw):
    if not raw:
        return None
    t = str(raw).strip().lower()
    if "rabi" in t or "winter" in t or "dry" in t:
        return "rabi"
    if "kharif" in t and ("2" in t or "ii" in t):
        return "kharif_2"
    if "kharif" in t and ("1" in t or "i" in t):
        return "kharif_1"
    if "aman" in t or "monsoon" in t:
        return "kharif_2"
    if "aus" in t or "pre-monsoon" in t or "premonsoon" in t:
        return "kharif_1"
    if "kharif" in t:
        return "kharif_2"  # bare "kharif" -> main (Aman) season
    return None


def infer_season(today):
    """Bangladesh cropping calendar: which season a date falls in."""
    md = (today.month, today.day)
    if (3, 16) <= md <= (7, 15):
        return "kharif_1"
    if (7, 16) <= md <= (11, 15):
        return "kharif_2"
    return "rabi"  # mid-Nov .. mid-Mar (wraps year end)


def _irrigation_access(profile):
    raw = (profile or {}).get("water_availability")
    if not raw:
        return 0.5, "unstated"
    t = str(raw).strip().lower()
    for token, val in IRRIGATION_ACCESS.items():
        if token in t:
            return val, t
    return 0.5, t


def _extract_weather(weather_summary):
    """Accept a full get_weather result, a {'forecast': {...}} block, or a
    bare summary dict, and pull out the fields scoring needs."""
    ws = weather_summary or {}
    loc = ws.get("location") or {}
    fc = ws.get("forecast") or ws
    summ = fc.get("summary") or ws.get("summary") or ws
    return {
        "total_rainfall_mm": summ.get("total_rainfall_mm"),
        "avg_max_temp_c": summ.get("avg_max_temp_c"),
        "avg_min_temp_c": summ.get("avg_min_temp_c"),
        "forecast_days": summ.get("forecast_days") or 16,
        "location_name": loc.get("name") or ws.get("location_name") or "the farm",
    }


# ---------------------------------------------------------------------------
# season-window date math
# ---------------------------------------------------------------------------

def _mmdd(s):
    m, d = s.split("-")
    return int(m), int(d)


def window_timing(today, window):
    """Where `today` sits relative to a sowing window (MM-DD start/end).

    Returns in_window plus the distance to the nearest window edge, so the
    scorer can tell "open now" from "opens in N days" from "closed N days ago".
    """
    sm, sd = _mmdd(window["start"])
    em, ed = _mmdd(window["end"])
    start = date(today.year, sm, sd)
    end = date(today.year, em, ed)
    if start <= today <= end:
        return {"in_window": True, "days_until_open": 0, "days_since_close": 0}
    upcoming = min(d for d in (date(today.year + k, sm, sd) for k in (-1, 0, 1)) if d >= today)
    recent = max(d for d in (date(today.year + k, em, ed) for k in (-1, 0, 1)) if d <= today)
    return {
        "in_window": False,
        "days_until_open": (upcoming - today).days,
        "days_since_close": (today - recent).days,
    }


def _fmt_window(window):
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sm, sd = _mmdd(window["start"])
    em, ed = _mmdd(window["end"])
    return f"{sd} {months[sm]}–{ed} {months[em]}"


# ---------------------------------------------------------------------------
# component scorers -> (score, reason)
# ---------------------------------------------------------------------------

def _score_soil(rec, profile):
    soil_key, raw = _normalize_soil((profile or {}).get("soil_type"))
    if soil_key is None:
        if raw:
            return 0.5, f"Soil fit 0.50 — soil type '{raw}' isn't in {rec['label']}'s suitability table; using a neutral 0.50."
        return 0.5, f"Soil fit 0.50 — no soil type on file yet; using a neutral 0.50 (ask the farmer to firm this up)."
    score = float(rec["soil_suitability"].get(soil_key, 0.5))
    disp = soil_key.replace("_", " ")
    return score, f"Soil fit {score:.2f} — farm soil is {disp}; {rec['label']} rates {disp} at {score:.2f} on its soil-suitability table (crops.yaml)."


def _score_season(rec, profile, today):
    explicit = _normalize_season((profile or {}).get("target_season"))
    target = explicit or infer_season(today)
    target_disp = SEASON_DISPLAY.get(target, target)
    source = "stated" if explicit else "inferred from today's date"
    in_season = target in rec["seasons"]
    t = window_timing(today, rec["sowing_window"])
    win = _fmt_window(rec["sowing_window"])
    crop_seasons = "/".join(SEASON_DISPLAY.get(s, s) for s in rec["seasons"])
    tdisp = today.isoformat()

    if in_season and t["in_window"]:
        s = 1.0
        r = f"Season fit {s:.2f} — target season {target_disp} ({source}); {rec['label']}'s planting window {win} is open now (today {tdisp})."
    elif in_season and t["days_until_open"] > 0:
        d = t["days_until_open"]
        s = _clamp(0.9 - d / 150.0, 0.45, 0.9)
        r = f"Season fit {s:.2f} — target season {target_disp} ({source}); {rec['label']}'s window {win} opens in {d} days (today {tdisp}) — plantable when the season comes."
    elif in_season:
        d = t["days_since_close"]
        s = 0.3
        r = f"Season fit {s:.2f} — target season {target_disp} ({source}), but {rec['label']}'s window {win} closed {d} days ago (today {tdisp})."
    elif t["in_window"]:
        s = 0.25
        r = f"Season fit {s:.2f} — {rec['label']} is a {crop_seasons} crop and its window {win} is open now, but the stated target season is {target_disp} ({source})."
    elif t["days_until_open"] > 0:
        d = t["days_until_open"]
        s = _clamp(0.25 - d / 900.0, 0.1, 0.25)
        r = f"Season fit {s:.2f} — target season {target_disp} ({source}), but {rec['label']} is a {crop_seasons} crop; its window {win} is {d} days away (today {tdisp})."
    else:
        d = t["days_since_close"]
        s = 0.1
        r = f"Season fit {s:.2f} — target season {target_disp} ({source}), but {rec['label']} is a {crop_seasons} crop; its window {win} closed {d} days ago."
    return round(s, 2), r


def _score_water(rec, profile, wx):
    irr, irr_label = _irrigation_access(profile)
    req = rec["water_requirement_mm"]
    rain = wx["total_rainfall_mm"]
    fdays = wx["forecast_days"]
    if rain is None:
        s = _clamp(irr * (1.2 - req / 1500.0), 0.0, 1.0)
        r = f"Water fit {s:.2f} — {rec['label']} needs ~{req} mm/season; no live forecast, so judged on stated irrigation access ({irr_label})."
        return round(s, 2), r
    window_need = req * fdays / rec["duration_days"]
    rain_cover = rain / window_need if window_need > 0 else 1.0
    s = 1.0 if rain_cover >= 1 else _clamp(rain_cover + irr * (1 - rain_cover), 0.0, 1.0)
    r = (f"Water fit {s:.2f} — {rec['label']} needs ~{req} mm/season; forecast shows {rain} mm over the next "
         f"{fdays} days at {wx['location_name']} and irrigation access is {irr_label} "
         f"(pro-rated need for this window ~{round(window_need)} mm).")
    return round(s, 2), r


def _score_temp(rec, wx):
    tmax = wx["avg_max_temp_c"]
    tmin = wx["avg_min_temp_c"]
    lo, hi = rec["temp_range_c"]["min"], rec["temp_range_c"]["max"]
    if tmax is None or tmin is None:
        return 0.7, f"Temp fit 0.70 — no forecast temperatures available; using a neutral estimate against {rec['label']}'s band {lo}–{hi}°C."
    penalty = 0.0
    if tmax > hi:
        penalty += (tmax - hi) / 15.0
    if tmin < lo:
        penalty += (lo - tmin) / 15.0
    s = _clamp(1.0 - penalty, 0.0, 1.0)
    r = f"Temp fit {s:.2f} — forecast avg {tmax}°C max / {tmin}°C min vs {rec['label']}'s comfortable band {lo}–{hi}°C."
    return round(s, 2), r


def _score_profit(crop_key, area):
    proj = compute_financial_projection(crop_key, area)
    roi = proj.get("roi_pct")
    net = proj.get("net_profit_bdt")
    cost = proj.get("total_cost_bdt")
    s = _clamp((roi or 0) / PROFIT_REF_PCT)
    r = (f"Profit {s:.2f} — projected ROI {roi}% (net BDT {net:,.0f} on BDT {cost:,.0f} cost over "
         f"{area} acre{'s' if area != 1 else ''}), scored against a {PROFIT_REF_PCT:.0f}% strong-ROI benchmark.")
    return round(s, 2), r, proj


def _budget_flag(rec, profile, proj, area):
    per_acre = round(sum(cost_breakdown_per_acre(rec).values()), 2)
    cost_for_area = proj["total_cost_bdt"]
    budget = _num((profile or {}).get("budget_bdt"))
    info = {
        "stated_budget_bdt": budget,
        "cost_for_area_bdt": cost_for_area,
        "cost_per_acre_bdt": per_acre,
        "area_acres": area,
        "affordable": None,
        "max_affordable_area_acres": None,
        "over_by_bdt": None,
    }
    if budget is None:
        r = f"Budget — no season budget stated; {area} acre{'s' if area != 1 else ''} would cost BDT {cost_for_area:,.0f} (BDT {per_acre:,.0f}/acre)."
        return info, r
    max_area = round(budget / per_acre, 2) if per_acre else None
    info["max_affordable_area_acres"] = max_area
    if cost_for_area <= budget:
        info["affordable"] = True
        r = (f"Budget — {area} acre{'s' if area != 1 else ''} costs BDT {cost_for_area:,.0f} against a stated budget of "
             f"BDT {budget:,.0f}; within budget. Max affordable area at this cost is {max_area} acres.")
    else:
        info["affordable"] = False
        over = round(cost_for_area - budget, 2)
        info["over_by_bdt"] = over
        r = (f"Budget — {area} acre{'s' if area != 1 else ''} costs BDT {cost_for_area:,.0f} against a stated budget of "
             f"BDT {budget:,.0f}; over by BDT {over:,.0f}. Max affordable area at this cost is {max_area} acres.")
    return info, r


# ---------------------------------------------------------------------------
# risk scoring -> (risk info, reason)
# ---------------------------------------------------------------------------

# Risk is a weighted blend of adverse signals (higher = riskier), tiered to
# Low/Medium/High. It deliberately weighs two things suitability does NOT --
# pest/disease pressure and affordability -- so a high-scoring top pick can
# still read as Medium risk when it is pest-prone or over budget, which is the
# downside a farmer actually needs flagged.
RISK_WEIGHTS = {
    "water_stress": 0.30,
    "pest_pressure": 0.30,
    "off_window_timing": 0.20,
    "temperature_stress": 0.10,
    "affordability": 0.10,
}
RISK_FACTOR_LABELS = {
    "water_stress": "water stress",
    "pest_pressure": "pest/disease pressure",
    "off_window_timing": "off-window timing",
    "temperature_stress": "temperature stress",
    "affordability": "budget/affordability",
}


def _risk_tier(score):
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    return "High"


def _pest_weather_pressure(wx):
    """How much the forecast raises or lowers pest/disease pressure.

    Warm-and-wet is the condition every extension advisory flags: standing
    humidity plus 25-35 C is when blast, blight, and hopper populations run.
    A dry forecast in the same window suppresses them. Returned as a bounded
    multiplier so the forecast modulates the crop's tracked pest load without
    ever inventing pressure for a crop with no tracked pests.
    """
    rain = wx.get("total_rainfall_mm")
    tmax = wx.get("avg_max_temp_c")
    days = wx.get("forecast_days") or 16
    if rain is None or not isfinite(float(rain)):
        return 1.0, "no forecast rainfall available, so pest pressure is unmodified"

    rain_per_day = float(rain) / max(float(days), 1.0)
    warm = tmax is not None and isfinite(float(tmax)) and 25.0 <= float(tmax) <= 35.0

    if rain_per_day >= 8.0:
        band = "wet"
        multiplier = 1.30 if warm else 1.15
    elif rain_per_day >= 3.0:
        band = "moist"
        multiplier = 1.15 if warm else 1.05
    elif rain_per_day >= 1.0:
        band = "average"
        multiplier = 1.0
    else:
        band = "dry"
        multiplier = 0.85

    temp_note = (
        f"{float(tmax):.1f} C average max is in the 25-35 C pest-active band"
        if warm
        else (
            f"{float(tmax):.1f} C average max is outside the 25-35 C pest-active band"
            if tmax is not None and isfinite(float(tmax))
            else "no temperature available"
        )
    )
    note = (
        f"forecast is {band} ({float(rain):.1f} mm over {int(days)} days = "
        f"{rain_per_day:.1f} mm/day) and {temp_note}, so pest/disease pressure "
        f"is scaled x{multiplier:.2f}"
    )
    return multiplier, note


def _score_risk(rec, components, budget_info, wx=None):
    """Derive a Low/Medium/High risk level from adverse agronomic signals.

    Reuses the hazard side of the fit components (water/timing/temperature
    shortfalls) and adds pest/disease pressure and affordability (does the
    season cost clear the stated budget). Pest pressure starts from the count
    of the crop's tracked pest windows and is then scaled by the live forecast
    -- warm-and-wet weather raises it, a dry forecast lowers it -- so the tier
    responds to conditions, not just to the crop. Every driver is a 0-1 value
    named in the returned reason, so the tier is inspectable rather than a
    black box.
    """
    pests = rec.get("pest_windows") or []
    pest_count = len(pests)
    weather_multiplier, weather_note = _pest_weather_pressure(wx or {})
    drivers = {
        "water_stress": round(_clamp(1.0 - components["water_fit"]), 2),
        "pest_pressure": round(_clamp((pest_count / 4.0) * weather_multiplier), 2),
        "off_window_timing": round(_clamp(1.0 - components["season_fit"]), 2),
        "temperature_stress": round(_clamp(1.0 - components["temp_fit"]), 2),
        "affordability": 1.0 if budget_info.get("affordable") is False else 0.0,
    }
    score = round(sum(drivers[k] * RISK_WEIGHTS[k] for k in RISK_WEIGHTS), 2)
    level = _risk_tier(score)
    contributions = sorted(
        ((k, drivers[k] * RISK_WEIGHTS[k]) for k in RISK_WEIGHTS),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_factors = [
        RISK_FACTOR_LABELS[k] for k, contribution in contributions if contribution > 0
    ][:2]

    pest_names = ", ".join(p["pest"] for p in pests) if pests else "none tracked"
    afford = budget_info.get("affordable")
    budget_note = (
        "over the stated budget" if afford is False
        else "within the stated budget" if afford is True
        else "no budget stated"
    )
    headline = (
        f"driven by {' and '.join(top_factors)}"
        if top_factors
        else "no adverse factors above zero"
    )
    reason = (
        f"Risk {level} ({score:.2f}/1.0) — {headline}. Factors: "
        f"water stress {drivers['water_stress']:.2f}, "
        f"pest/disease pressure {drivers['pest_pressure']:.2f} "
        f"({pest_count} tracked pest window(s): {pest_names}; {weather_note}), "
        f"off-window timing {drivers['off_window_timing']:.2f}, "
        f"temperature stress {drivers['temperature_stress']:.2f}, "
        f"budget {drivers['affordability']:.2f} ({budget_note}). "
        f"Weights {RISK_WEIGHTS}; tiers <0.25 Low, <0.50 Medium, else High."
    )
    return {
        "level": level,
        "score": score,
        "drivers": drivers,
        "top_factors": top_factors,
        "weather_multiplier": round(weather_multiplier, 2),
        "pests": [
            {
                "pest": p.get("pest"),
                "day_from": p.get("day_from"),
                "day_to": p.get("day_to"),
                "sign": p.get("sign"),
                "control": p.get("control"),
                "cost_bdt_per_acre": p.get("cost_bdt_per_acre"),
            }
            for p in pests
        ],
    }, reason


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def score_crop(crop_key, profile, weather_summary, today=None):
    """Deterministic suitability + economics for one crop.

    Returns component scores 0-1, an overall weighted score, a budget flag,
    an economics block, and a `reasons` list where every entry names the
    exact input value it used. Fully self-contained -- rank_crops just calls
    this for each crop and sorts, so a single crop scores identically whether
    asked about on its own or inside a ranking.
    """
    try:
        rec = get_crop(crop_key)
    except (AttributeError, TypeError):
        rec = None
    if rec is None:
        message = (
            f"Unknown crop '{crop_key}'. Supported: "
            f"{', '.join(list_crop_keys())}"
        )
        return {
            "error": message,
            "reasons": [
                f"Crop suitability was not scored: {message}"
            ],
        }
    today = today or date.today()
    profile = profile or {}
    wx = _extract_weather(weather_summary)
    area, area_error = _profile_area(profile)
    if area_error:
        return {
            "error": area_error,
            "reasons": [
                f"Crop suitability was not scored: {area_error}"
            ],
        }

    soil_s, soil_r = _score_soil(rec, profile)
    season_s, season_r = _score_season(rec, profile, today)
    water_s, water_r = _score_water(rec, profile, wx)
    temp_s, temp_r = _score_temp(rec, wx)
    profit_s, profit_r, proj = _score_profit(crop_key, area)
    budget_info, budget_r = _budget_flag(rec, profile, proj, area)

    components = {
        "soil_fit": soil_s,
        "season_fit": season_s,
        "water_fit": water_s,
        "temp_fit": temp_s,
        "profit_score": profit_s,
    }
    overall = round(sum(components[k] * WEIGHTS[k] for k in WEIGHTS), 3)
    risk_info, risk_r = _score_risk(rec, components, budget_info, wx)

    return {
        "crop": proj["crop"],
        "label": rec["label"],
        "overall_score": overall,
        "water_need": proj.get("water_need"),
        "components": components,
        "economics": {
            "area_acres": area,
            "total_cost_bdt": proj["total_cost_bdt"],
            "revenue_bdt": proj["revenue_bdt"],
            "net_profit_bdt": proj["net_profit_bdt"],
            "roi_pct": proj["roi_pct"],
        },
        "budget": budget_info,
        "risk": risk_info,
        "duration_days": rec["duration_days"],
        "reasons": [soil_r, season_r, water_r, temp_r, profit_r, budget_r, risk_r],
    }


def rank_crops(profile, weather_summary, today=None, top_n=5):
    """Score every crop for this farm + forecast and return the best ones.

    Scores all crops with score_crop and sorts by overall score. Returns the
    top_n crops, each with the full component breakdown and quotable `reasons`,
    plus the also-ranked tail so the trace shows the whole field.
    """
    today = today or date.today()
    profile = profile or {}
    area, area_error = _profile_area(profile)
    target = _normalize_season(profile.get("target_season")) or infer_season(today)
    season_used = SEASON_DISPLAY.get(target, target)
    season_source = (
        "stated"
        if _normalize_season(profile.get("target_season"))
        else "inferred from date"
    )
    if area_error:
        return {
            "error": area_error,
            "as_of_date": today.isoformat(),
            "season_used": season_used,
            "season_source": season_source,
            "area_acres": profile.get("farm_size_acres"),
            "weights": WEIGHTS,
            "ranked": [],
            "also_ranked": [],
            "reasons": [f"Crop ranking was not computed: {area_error}"],
        }

    scored = [score_crop(k, profile, weather_summary, today=today) for k in list_crop_keys()]
    scored.sort(key=lambda c: c["overall_score"], reverse=True)
    ranked = scored[:top_n]
    also_ranked = [
        {"crop": c["crop"], "overall_score": c["overall_score"]}
        for c in scored[top_n:]
    ]
    reasons = [
        (
            f"Ranking context: as_of_date={today.isoformat()}, "
            f"season_used={season_used}, season_source={season_source}, "
            f"area_acres={area}; weights={WEIGHTS}."
        )
    ]
    for index, candidate in enumerate(ranked, start=1):
        components = candidate["components"]
        economics = candidate["economics"]
        reasons.append(
            (
                f"Rank {index}: crop={candidate['crop']}, "
                f"label={candidate['label']!r}, "
                f"overall_score={candidate['overall_score']}; "
                f"water_need={candidate.get('water_need')}, "
                f"risk_level={candidate['risk']['level']}, "
                f"risk_score={candidate['risk']['score']}; "
                f"soil_fit={components['soil_fit']}, "
                f"season_fit={components['season_fit']}, "
                f"water_fit={components['water_fit']}, "
                f"temp_fit={components['temp_fit']}, "
                f"profit_score={components['profit_score']}; "
                f"total_cost_bdt={economics['total_cost_bdt']}, "
                f"revenue_bdt={economics['revenue_bdt']}, "
                f"net_profit_bdt={economics['net_profit_bdt']}, "
                f"roi_pct={economics['roi_pct']}. "
                f"Computed rationale: {' | '.join(candidate['reasons'])}"
            )
        )
    if also_ranked:
        reasons.append(
            "Remaining ranking: "
            + "; ".join(
                f"crop={candidate['crop']}, "
                f"overall_score={candidate['overall_score']}"
                for candidate in also_ranked
            )
            + "."
        )

    return {
        "as_of_date": today.isoformat(),
        "season_used": season_used,
        "season_source": season_source,
        "area_acres": area,
        "weights": WEIGHTS,
        "ranked": ranked,
        "also_ranked": also_ranked,
        "reasons": reasons,
    }


if __name__ == "__main__":
    # Test 2 self-checks -- run: python -m tools.agronomy
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # em-dashes etc. in reasons -> Windows-console safe
    except Exception:
        pass
    KHARIF_WX = {"location": {"name": "Rangpur"},
                 "summary": {"total_rainfall_mm": 180.0, "avg_max_temp_c": 33.0, "avg_min_temp_c": 26.5, "forecast_days": 16}}
    DRY_WX = {"location": {"name": "Rangpur"},
              "summary": {"total_rainfall_mm": 12.0, "avg_max_temp_c": 27.0, "avg_min_temp_c": 14.0, "forecast_days": 16}}
    MILD_WX = {"location": {"name": "Rangpur"},
               "summary": {"total_rainfall_mm": 20.0, "avg_max_temp_c": 27.0, "avg_min_temp_c": 18.0, "forecast_days": 16}}
    TODAY = date(2026, 7, 24)  # Kharif-2

    def top(profile, wx, today=TODAY, **kw):
        return rank_crops(profile, wx, today=today, **kw)["ranked"][0]["crop"]

    # 1) SOIL flips the ranking. Hold season/water/temp steady (Rabi, reliable
    #    irrigation, mild forecast) so soil is the only thing moving.
    soil_farm = {"farm_size_acres": 2.0, "water_availability": "high", "budget_bdt": 100000, "target_season": "Rabi"}
    clay_top = top({**soil_farm, "soil_type": "clay"}, MILD_WX)
    sandy_top = top({**soil_farm, "soil_type": "sandy"}, MILD_WX)
    print(f"[soil]   clay -> {clay_top} | sandy -> {sandy_top}")
    assert clay_top != sandy_top, "flipping soil type must change the top crop"

    # 2) SEASON flips the ranking (same farm, Kharif-2 vs Rabi).
    farm = {"farm_size_acres": 2.0, "soil_type": "clay loam", "water_availability": "medium", "budget_bdt": 80000}
    kharif_top = top({**farm, "target_season": "Kharif-2"}, KHARIF_WX)
    rabi_top = top({**farm, "target_season": "Rabi"}, DRY_WX)
    print(f"[season] Kharif-2 -> {kharif_top} | Rabi -> {rabi_top}")
    assert kharif_top != rabi_top, "changing target season must change the top crop"
    assert kharif_top == "rice_aman", f"Kharif-2 today should surface Aman rice, got {kharif_top}"

    # 3) Halving the budget changes a budget flag AND the recommended area.
    #    Potato is the capital-heavy crop, so it's the clearest demonstrator.
    pot_rich = score_crop("potato", {**farm, "budget_bdt": 120000, "target_season": "Rabi"}, DRY_WX, today=TODAY)["budget"]
    pot_poor = score_crop("potato", {**farm, "budget_bdt": 15000, "target_season": "Rabi"}, DRY_WX, today=TODAY)["budget"]
    print(f"[budget] potato affordable @120k={pot_rich['affordable']} maxArea={pot_rich['max_affordable_area_acres']} | "
          f"@15k affordable={pot_poor['affordable']} maxArea={pot_poor['max_affordable_area_acres']}")
    assert pot_rich["affordable"] is True and pot_poor["affordable"] is False, "halving budget must flip the affordable flag"
    assert pot_rich["max_affordable_area_acres"] != pot_poor["max_affordable_area_acres"], "max affordable area must change with budget"

    # 4) Every returned crop carries >=4 non-empty reasons.
    res = rank_crops({**farm, "target_season": "Rabi"}, DRY_WX, today=TODAY)
    for c in res["ranked"]:
        assert len([r for r in c["reasons"] if r and r.strip()]) >= 4, f"{c['crop']} has too few reasons"
    print(f"[reasons] all {len(res['ranked'])} ranked crops carry >=4 reasons")

    # 5) Every ranked crop carries a risk tier and a water-need label.
    for c in res["ranked"]:
        assert c["risk"]["level"] in {"Low", "Medium", "High"}, f"{c['crop']} missing a risk level"
        assert c.get("water_need") in {"low", "medium", "high", "unknown"}, f"{c['crop']} missing a water-need label"
    tiers = ", ".join(f"{c['crop']}={c['risk']['level']}" for c in res["ranked"])
    print(f"[risk]    every ranked crop has a risk tier + water-need label ({tiers})")

    print("\nExample top pick today (Kharif-2, clay loam, 2 acres):")
    winner = rank_crops(farm, KHARIF_WX, today=TODAY)["ranked"][0]
    print(f"  {winner['crop']}  overall={winner['overall_score']}  components={winner['components']}")
    for r in winner["reasons"]:
        print(f"   - {r}")

    print("\nAll agronomy self-checks passed.")
