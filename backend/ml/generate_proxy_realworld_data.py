from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CityProfile:
    city: str
    population_density_per_km2: float
    transformer_age_years: float
    pipeline_age_years: float
    base_grid_load: float
    base_congestion: float


CITY_PROFILES = [
    CityProfile("Metrovale", 13800, 27.0, 32.0, 68.0, 64.0),
    CityProfile("Bayhaven", 17100, 30.0, 35.0, 72.0, 70.0),
    CityProfile("Riverton", 7200, 14.0, 19.0, 56.0, 45.0),
    CityProfile("Northbridge", 9800, 20.0, 24.0, 62.0, 52.0),
    CityProfile("Sunport", 12500, 24.0, 29.0, 66.0, 60.0),
    CityProfile("Lakehurst", 10600, 22.0, 27.0, 61.0, 55.0),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float) -> float:
    x = _clamp(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _choose_profiles(cities: str | None) -> list[CityProfile]:
    if not cities:
        return CITY_PROFILES
    requested = {part.strip() for part in cities.split(",") if part.strip()}
    chosen = [profile for profile in CITY_PROFILES if profile.city in requested]
    if not chosen:
        raise ValueError(f"none of requested cities found: {sorted(requested)}")
    return chosen


def _storm_state_transition(current: bool, rng: np.random.Generator) -> bool:
    if current:
        return bool(rng.uniform() < 0.82)
    return bool(rng.uniform() < 0.04)


def _heatwave_state_transition(current: bool, month: int, rng: np.random.Generator) -> bool:
    if current:
        return bool(rng.uniform() < 0.96)
    summer = month in {5, 6, 7, 8, 9}
    trigger = 0.008 if summer else 0.001
    return bool(rng.uniform() < trigger)


def generate_proxy_data(
    start_date: datetime,
    days: int,
    seed: int = 42,
    cities: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    profiles = _choose_profiles(cities)
    hours = days * 24

    telemetry_rows: list[dict] = []
    incident_rows: list[dict] = []

    for profile in profiles:
        storm_state = False
        heatwave_state = False
        rolling_repairs = 0
        rolling_signal_failures = 0

        for step in range(hours):
            ts = start_date + timedelta(hours=step)
            hour = ts.hour
            weekday = ts.weekday()
            day_of_year = ts.timetuple().tm_yday
            month = ts.month

            storm_state = _storm_state_transition(storm_state, rng)
            heatwave_state = _heatwave_state_transition(heatwave_state, month, rng)

            annual_temp = 22.0 + 9.5 * np.sin((2.0 * np.pi * (day_of_year - 80)) / 365.0)
            diurnal_temp = 5.8 * np.sin((2.0 * np.pi * (hour - 14)) / 24.0)
            temp_noise = rng.normal(0.0, 1.2)
            temperature_c = annual_temp + diurnal_temp + temp_noise + (8.0 if heatwave_state else 0.0)
            temperature_c = _clamp(float(temperature_c), -5.0, 48.0)

            humidity_base = 66.0 - (temperature_c - 25.0) * 0.8
            humidity_percent = _clamp(float(humidity_base + rng.normal(0.0, 6.0)), 18.0, 98.0)

            if storm_state:
                rainfall_mm = float(_clamp(rng.gamma(shape=2.2, scale=5.4), 2.0, 85.0))
            else:
                drizzle = rng.uniform(0.0, 2.4) if rng.uniform() < 0.15 else 0.0
                rainfall_mm = float(_clamp(drizzle, 0.0, 6.0))

            commute_peak = (
                np.exp(-((hour - 8) ** 2) / 7.5) * 12.0
                + np.exp(-((hour - 18) ** 2) / 8.5) * 15.0
            )
            weekend_factor = -8.0 if weekday >= 5 else 0.0

            grid_temp_pressure = max(0.0, temperature_c - 30.0) * 1.75
            current_load_percent = (
                profile.base_grid_load
                + grid_temp_pressure
                + (7.0 if heatwave_state else 0.0)
                + commute_peak * 0.35
                + weekend_factor * 0.2
                + rng.normal(0.0, 3.2)
            )
            current_load_percent = _clamp(float(current_load_percent), 32.0, 100.0)
            peak_load_percent = _clamp(
                float(current_load_percent + rng.uniform(4.0, 10.5) + max(0.0, grid_temp_pressure * 0.12)),
                38.0,
                100.0,
            )

            historical_failure_rate_percent = (
                4.5
                + max(0.0, profile.transformer_age_years - 16.0) * 0.26
                + max(0.0, current_load_percent - 78.0) * 0.08
                + rng.normal(0.0, 0.8)
            )
            historical_failure_rate_percent = _clamp(float(historical_failure_rate_percent), 1.0, 35.0)

            pressure_variance_percent = (
                10.0
                + max(0.0, profile.pipeline_age_years - 18.0) * 0.55
                + rainfall_mm * 0.32
                + rolling_repairs * 0.75
                + rng.normal(0.0, 2.1)
            )
            pressure_variance_percent = _clamp(float(pressure_variance_percent), 4.0, 78.0)

            repair_intensity = (
                0.15
                + max(0.0, profile.pipeline_age_years - 20.0) * 0.015
                + max(0.0, pressure_variance_percent - 24.0) * 0.012
            )
            new_repairs = int(rng.poisson(max(0.05, repair_intensity)))
            rolling_repairs = int(_clamp(rolling_repairs * 0.93 + new_repairs, 0.0, 50.0))
            recent_repairs_last_30_days = rolling_repairs

            congestion_index = (
                profile.base_congestion
                + commute_peak
                + weekend_factor
                + rainfall_mm * 0.35
                + (5.0 if heatwave_state else 0.0)
                + max(0.0, profile.population_density_per_km2 - 10000.0) * 0.0011
                + rng.normal(0.0, 4.0)
            )
            congestion_index = _clamp(float(congestion_index), 18.0, 100.0)

            signal_lambda = (
                0.5
                + max(0.0, congestion_index - 68.0) * 0.06
                + rainfall_mm * 0.03
                + max(0.0, profile.population_density_per_km2 - 9000.0) * 0.00012
            )
            new_signal_failures = int(rng.poisson(max(0.1, signal_lambda)))
            rolling_signal_failures = int(
                _clamp(rolling_signal_failures * 0.84 + new_signal_failures, 0.0, 100.0)
            )
            signal_failure_reports_last_7_days = rolling_signal_failures

            telemetry = {
                "timestamp": ts.isoformat(),
                "city": profile.city,
                "temperature_c": round(temperature_c, 2),
                "humidity_percent": round(humidity_percent, 2),
                "heatwave_alert": bool(heatwave_state),
                "rainfall_mm": round(rainfall_mm, 2),
                "current_load_percent": round(current_load_percent, 2),
                "peak_load_percent": round(peak_load_percent, 2),
                "historical_failure_rate_percent": round(historical_failure_rate_percent, 2),
                "transformer_age_years": round(profile.transformer_age_years, 2),
                "pressure_variance_percent": round(pressure_variance_percent, 2),
                "pipeline_age_years": round(profile.pipeline_age_years, 2),
                "recent_repairs_last_30_days": int(recent_repairs_last_30_days),
                "congestion_index": round(congestion_index, 2),
                "signal_failure_reports_last_7_days": int(signal_failure_reports_last_7_days),
                "population_density_per_km2": round(profile.population_density_per_km2, 2),
            }
            telemetry_rows.append(telemetry)

            power_prob = _sigmoid(
                -6.1
                + 0.052 * (current_load_percent - 70.0)
                + 0.038 * (peak_load_percent - 85.0)
                + 0.028 * (profile.transformer_age_years - 20.0)
                + (0.65 if heatwave_state else 0.0)
                + 0.018 * (historical_failure_rate_percent - 10.0)
            )
            transformer_prob = _sigmoid(
                -6.2
                + 0.056 * (peak_load_percent - 84.0)
                + 0.031 * (profile.transformer_age_years - 19.0)
                + (0.45 if heatwave_state else 0.0)
            )
            water_prob = _sigmoid(
                -6.4
                + 0.055 * (pressure_variance_percent - 22.0)
                + 0.030 * (profile.pipeline_age_years - 22.0)
                + 0.020 * rainfall_mm
                + 0.012 * recent_repairs_last_30_days
            )
            traffic_prob = _sigmoid(
                -5.9
                + 0.046 * (congestion_index - 64.0)
                + 0.070 * (signal_failure_reports_last_7_days - 4.0)
                + 0.00004 * (profile.population_density_per_km2 - 9000.0)
                + (0.35 if rainfall_mm > 25.0 else 0.0)
            )

            power_event = rng.uniform() < power_prob
            transformer_event = rng.uniform() < transformer_prob
            water_event = rng.uniform() < water_prob
            traffic_event = rng.uniform() < traffic_prob

            if power_event:
                incident_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "city": profile.city,
                        "system": "power",
                        "severity": int(_clamp(np.round(2 + power_prob * 4 + rng.normal(0, 0.7)), 1, 5)),
                    }
                )
            if transformer_event:
                incident_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "city": profile.city,
                        "system": "transformer",
                        "severity": int(
                            _clamp(np.round(2 + transformer_prob * 4 + rng.normal(0, 0.7)), 1, 5)
                        ),
                    }
                )
            if water_event:
                incident_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "city": profile.city,
                        "system": "water",
                        "severity": int(_clamp(np.round(2 + water_prob * 4 + rng.normal(0, 0.7)), 1, 5)),
                    }
                )
            if traffic_event:
                incident_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "city": profile.city,
                        "system": "traffic",
                        "severity": int(_clamp(np.round(2 + traffic_prob * 4 + rng.normal(0, 0.7)), 1, 5)),
                    }
                )

            event_count = int(power_event) + int(transformer_event) + int(water_event) + int(traffic_event)
            if event_count >= 2 and rng.uniform() < 0.28:
                incident_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "city": profile.city,
                        "system": "citywide",
                        "severity": int(_clamp(2 + event_count + rng.integers(0, 2), 1, 5)),
                    }
                )

    telemetry_df = pd.DataFrame(telemetry_rows).sort_values(["city", "timestamp"]).reset_index(drop=True)
    incidents_df = pd.DataFrame(incident_rows).sort_values(["city", "timestamp"]).reset_index(drop=True)
    return telemetry_df, incidents_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate proxy real-world telemetry and incident CSV files for GridMind."
    )
    parser.add_argument("--start-date", default="2024-01-01T00:00:00Z", help="ISO start datetime in UTC.")
    parser.add_argument("--days", type=int, default=365, help="Number of days to generate per city.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cities",
        default="",
        help="Comma-separated subset of cities. Leave empty for all built-in profiles.",
    )
    parser.add_argument(
        "--telemetry-output",
        default=str(Path("models") / "proxy_realworld_telemetry.csv"),
        help="Output path for telemetry CSV.",
    )
    parser.add_argument(
        "--incidents-output",
        default=str(Path("models") / "proxy_realworld_incidents.csv"),
        help="Output path for incidents CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.to_datetime(args.start_date, utc=True).to_pydatetime()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    telemetry_df, incidents_df = generate_proxy_data(
        start_date=start,
        days=args.days,
        seed=args.seed,
        cities=args.cities or None,
    )

    telemetry_path = Path(args.telemetry_output).expanduser().resolve()
    incidents_path = Path(args.incidents_output).expanduser().resolve()
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    incidents_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_df.to_csv(telemetry_path, index=False)
    incidents_df.to_csv(incidents_path, index=False)

    print(f"telemetry_rows={len(telemetry_df)} path={telemetry_path}")
    print(f"incident_rows={len(incidents_df)} path={incidents_path}")


if __name__ == "__main__":
    main()

