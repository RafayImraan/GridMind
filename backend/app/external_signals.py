from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import ExternalSignal, RiskAssessmentRequest


CITY_COORDS: dict[str, tuple[float, float]] = {
    "metrovale": (41.8781, -87.6298),  # Chicago proxy
    "bayhaven": (37.7749, -122.4194),  # San Francisco proxy
    "northbridge": (42.3601, -71.0589),  # Boston proxy
    "lakehurst": (40.7128, -74.0060),  # New York proxy
    "riverton": (39.9526, -75.1652),  # Philadelphia proxy
    "sunport": (33.4484, -112.0740),  # Phoenix proxy
}


def _resolve_coordinates(city: str) -> tuple[float, float] | None:
    key = city.strip().lower()
    return CITY_COORDS.get(key)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_live_weather_signal(payload: RiskAssessmentRequest) -> ExternalSignal:
    coords = _resolve_coordinates(payload.city)
    if coords is None:
        return ExternalSignal(
            source="open-meteo.com",
            signal_type="live_weather_crosscheck",
            status="unsupported_city",
            details={"city": payload.city, "reason": "no_coordinate_mapping"},
            influence_note="Live external signal unavailable for this city mapping.",
        )

    latitude, longitude = coords
    params = urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "current": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "UTC",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    request = Request(url, headers={"User-Agent": "GridMindAI/1.0"})

    try:
        with urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return ExternalSignal(
            source="open-meteo.com",
            signal_type="live_weather_crosscheck",
            status="fetch_failed",
            details={"city": payload.city, "error": str(exc)},
            influence_note="Live external signal fetch failed; assessment stayed deterministic + ML only.",
        )

    current = data.get("current", {}) if isinstance(data, dict) else {}
    ext_temp = _safe_float(current.get("temperature_2m"))
    ext_precip = _safe_float(current.get("precipitation"))
    ext_wind = _safe_float(current.get("wind_speed_10m"))

    input_temp = _safe_float(payload.weather.temperature_c)
    input_rain = _safe_float(payload.weather.rainfall_mm)

    temp_delta = None if ext_temp is None or input_temp is None else round(ext_temp - input_temp, 2)
    rain_delta = None if ext_precip is None or input_rain is None else round(ext_precip - input_rain, 2)

    anomaly_flag = bool(
        (temp_delta is not None and abs(temp_delta) >= 4.0)
        or (rain_delta is not None and rain_delta >= 10.0)
        or (ext_wind is not None and ext_wind >= 35.0)
    )
    note = (
        "External weather anomaly detected; elevated readiness recommended."
        if anomaly_flag
        else "External weather signal aligned with submitted telemetry profile."
    )

    return ExternalSignal(
        source="open-meteo.com",
        signal_type="live_weather_crosscheck",
        status="ok",
        retrieved_at=datetime.now(timezone.utc),
        details={
            "city": payload.city,
            "mapped_latitude": latitude,
            "mapped_longitude": longitude,
            "external_temperature_c": ext_temp,
            "external_precipitation_mm": ext_precip,
            "external_wind_speed_kmh": ext_wind,
            "temperature_delta_c": temp_delta,
            "rainfall_delta_mm": rain_delta,
            "weather_anomaly_flag": anomaly_flag,
        },
        influence_note=note,
    )
