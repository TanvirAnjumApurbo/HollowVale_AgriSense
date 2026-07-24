"""Live weather grounding via Open-Meteo (free, no API key required).

Two calls: geocode the farmer's typed location to lat/lon, then pull a
daily forecast (rainfall, temperature) for that point. Every value returned
here is real and unmodified from the API response -- callers must not
invent or adjust these numbers.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "et0_fao_evapotranspiration",
]


def geocode_location(location_name, country_code="BD"):
    """Resolve a place name to lat/lon using Open-Meteo's geocoding API.

    Returns a dict with name, country, latitude, longitude, or None if
    no match was found.
    """
    params = {"name": location_name, "count": 5, "language": "en", "format": "json"}
    resp = requests.get(GEOCODE_URL, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return None

    for r in results:
        if r.get("country_code") == country_code:
            return _format_geocode_result(r)
    return _format_geocode_result(results[0])


def _format_geocode_result(r):
    return {
        "name": r.get("name"),
        "admin1": r.get("admin1"),
        "country": r.get("country"),
        "country_code": r.get("country_code"),
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
    }


def get_forecast(latitude, longitude, forecast_days=16):
    """Fetch a daily rainfall/temperature forecast for a lat/lon point.

    Returns the raw Open-Meteo daily block plus a few derived totals
    (total forecast rainfall, average max temp) so the agent can quote
    real figures without doing its own arithmetic on the raw arrays.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "forecast_days": forecast_days,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    precip = daily.get("precipitation_sum", []) or []
    tmax = daily.get("temperature_2m_max", []) or []
    tmin = daily.get("temperature_2m_min", []) or []

    return {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "daily": daily,
        "summary": {
            "forecast_days": len(daily.get("time", []) or []),
            "total_rainfall_mm": round(sum(precip), 1) if precip else None,
            "rainy_days": sum(1 for p in precip if p and p > 1.0),
            "avg_max_temp_c": round(sum(tmax) / len(tmax), 1) if tmax else None,
            "avg_min_temp_c": round(sum(tmin) / len(tmin), 1) if tmin else None,
        },
    }


def get_weather_for_location(location_name, country_code="BD", forecast_days=16):
    """Convenience wrapper: geocode then forecast in one call.

    This is the function exposed to the agent as a tool.
    """
    place = geocode_location(location_name, country_code=country_code)
    if place is None:
        return {"error": f"Could not find location '{location_name}'. Ask the farmer for a nearby district or upazila name."}

    forecast = get_forecast(place["latitude"], place["longitude"], forecast_days=forecast_days)
    return {"location": place, "forecast": forecast}


if __name__ == "__main__":
    import json

    for loc in ["Rangpur", "Comilla", "Jessore"]:
        result = get_weather_for_location(loc)
        print(f"--- {loc} ---")
        print(json.dumps(result.get("location"), indent=2))
        print(json.dumps(result.get("forecast", {}).get("summary"), indent=2))
