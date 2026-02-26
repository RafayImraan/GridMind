from __future__ import annotations

import random
from datetime import datetime, timezone


CITY_PROFILES = [
    ("Metrovale", 11200),
    ("Riverton", 8400),
    ("Northbridge", 9600),
    ("Sunport", 13200),
    ("Crestfield", 7200),
    ("Lakehurst", 10100),
    ("Bayhaven", 14500),
    ("Ironwood", 8800),
    ("Stonegate", 11700),
    ("Hillford", 6700),
]


def _base_scenario(city: str, density: float, now: datetime) -> dict:
    return {
        "city": city,
        "timestamp": now.isoformat(),
        "weather": {
            "temperature_c": round(random.uniform(20, 35), 1),
            "humidity_percent": round(random.uniform(35, 75), 1),
            "heatwave_alert": False,
            "rainfall_mm": round(random.uniform(0, 20), 1),
        },
        "power_grid": {
            "current_load_percent": round(random.uniform(55, 82), 1),
            "peak_load_percent": round(random.uniform(70, 90), 1),
            "historical_failure_rate_percent": round(random.uniform(4, 15), 1),
            "transformer_age_years": round(random.uniform(8, 24), 1),
        },
        "water_system": {
            "pressure_variance_percent": round(random.uniform(10, 30), 1),
            "pipeline_age_years": round(random.uniform(10, 35), 1),
            "recent_repairs_last_30_days": random.randint(0, 4),
        },
        "traffic_system": {
            "congestion_index": round(random.uniform(45, 78), 1),
            "signal_failure_reports_last_7_days": random.randint(1, 6),
        },
        "population_density_per_km2": round(density, 1),
    }


def apply_heatwave_stress(scenario: dict) -> dict:
    scenario["weather"]["temperature_c"] = round(random.uniform(38, 46), 1)
    scenario["weather"]["humidity_percent"] = round(random.uniform(25, 60), 1)
    scenario["weather"]["heatwave_alert"] = True
    scenario["weather"]["rainfall_mm"] = round(random.uniform(0, 8), 1)
    scenario["power_grid"]["current_load_percent"] = round(random.uniform(86, 97), 1)
    scenario["power_grid"]["peak_load_percent"] = round(random.uniform(92, 100), 1)
    scenario["traffic_system"]["congestion_index"] = round(random.uniform(70, 90), 1)
    return scenario


def apply_high_density_stress(scenario: dict) -> dict:
    scenario["population_density_per_km2"] = round(random.uniform(11500, 18000), 1)
    scenario["traffic_system"]["congestion_index"] = round(random.uniform(78, 96), 1)
    scenario["traffic_system"]["signal_failure_reports_last_7_days"] = random.randint(6, 16)
    scenario["water_system"]["pressure_variance_percent"] = round(random.uniform(22, 45), 1)
    return scenario


def apply_failure_event_stress(scenario: dict) -> dict:
    scenario["power_grid"]["historical_failure_rate_percent"] = round(random.uniform(15, 28), 1)
    scenario["power_grid"]["transformer_age_years"] = round(random.uniform(21, 40), 1)
    scenario["water_system"]["recent_repairs_last_30_days"] = random.randint(4, 10)
    scenario["weather"]["rainfall_mm"] = round(random.uniform(35, 90), 1)
    scenario["water_system"]["pressure_variance_percent"] = round(random.uniform(30, 60), 1)
    return scenario


def generate_scenario(scenario_type: str, city: str, density: float, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    base = _base_scenario(city=city, density=density, now=now)
    if scenario_type == "heatwave":
        return apply_heatwave_stress(base)
    if scenario_type == "high_density":
        return apply_high_density_stress(base)
    if scenario_type == "failure_event":
        return apply_failure_event_stress(base)
    if scenario_type == "compound":
        stressed = apply_heatwave_stress(base)
        stressed = apply_high_density_stress(stressed)
        stressed = apply_failure_event_stress(stressed)
        return stressed
    return base


def generate_batch(count: int = 10, seed: int = 42) -> list[dict]:
    random.seed(seed)
    scenario_types = [
        "baseline",
        "heatwave",
        "high_density",
        "failure_event",
        "compound",
    ]
    now = datetime.now(timezone.utc)

    scenarios: list[dict] = []
    for index in range(count):
        city, density = CITY_PROFILES[index % len(CITY_PROFILES)]
        scenario_type = scenario_types[index % len(scenario_types)]
        scenario = generate_scenario(
            scenario_type=scenario_type,
            city=city,
            density=density,
            now=now,
        )
        scenario["scenario_type"] = scenario_type
        scenarios.append(scenario)
    return scenarios

