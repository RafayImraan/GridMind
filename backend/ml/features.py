from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TIME_COLUMN = "timestamp"

RAW_INPUT_COLUMNS = [
    "city",
    "temperature_c",
    "humidity_percent",
    "heatwave_alert",
    "rainfall_mm",
    "current_load_percent",
    "peak_load_percent",
    "historical_failure_rate_percent",
    "transformer_age_years",
    "pressure_variance_percent",
    "pipeline_age_years",
    "recent_repairs_last_30_days",
    "congestion_index",
    "signal_failure_reports_last_7_days",
    "population_density_per_km2",
]

TARGET_COLUMNS = [
    "overall_failure_72h_label",
    "power_failure_72h_label",
    "transformer_overload_72h_label",
    "water_failure_72h_label",
    "traffic_failure_72h_label",
]

MODEL_CATEGORICAL_FEATURES = ["city"]
MODEL_NUMERIC_FEATURES = [
    "temperature_c",
    "humidity_percent",
    "rainfall_mm",
    "current_load_percent",
    "peak_load_percent",
    "historical_failure_rate_percent",
    "transformer_age_years",
    "pressure_variance_percent",
    "pipeline_age_years",
    "recent_repairs_last_30_days",
    "congestion_index",
    "signal_failure_reports_last_7_days",
    "population_density_per_km2",
    "heatwave_alert_int",
    "load_margin_percent",
    "load_ratio",
    "heat_load_interaction",
    "transformer_stress_index",
    "water_stress_index",
    "repair_pressure_interaction",
    "congestion_signal_interaction",
    "density_congestion_interaction",
    "aging_density_intersection",
    "humidity_temperature_interaction",
    "rain_pressure_interaction",
    "high_load_flag",
    "peak_load_flag",
    "high_pressure_flag",
    "high_density_flag",
]
MODEL_FEATURE_COLUMNS = MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES


def flatten_payload_record(record: dict[str, Any]) -> dict[str, Any]:
    if "weather" in record and "power_grid" in record:
        weather = record.get("weather", {})
        power = record.get("power_grid", {})
        water = record.get("water_system", {})
        traffic = record.get("traffic_system", {})
        return {
            "city": record.get("city"),
            "timestamp": record.get("timestamp"),
            "temperature_c": weather.get("temperature_c"),
            "humidity_percent": weather.get("humidity_percent"),
            "heatwave_alert": weather.get("heatwave_alert"),
            "rainfall_mm": weather.get("rainfall_mm"),
            "current_load_percent": power.get("current_load_percent"),
            "peak_load_percent": power.get("peak_load_percent"),
            "historical_failure_rate_percent": power.get("historical_failure_rate_percent"),
            "transformer_age_years": power.get("transformer_age_years"),
            "pressure_variance_percent": water.get("pressure_variance_percent"),
            "pipeline_age_years": water.get("pipeline_age_years"),
            "recent_repairs_last_30_days": water.get("recent_repairs_last_30_days"),
            "congestion_index": traffic.get("congestion_index"),
            "signal_failure_reports_last_7_days": traffic.get("signal_failure_reports_last_7_days"),
            "population_density_per_km2": record.get("population_density_per_km2"),
        }
    return dict(record)


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    flat = [flatten_payload_record(record) for record in records]
    return pd.DataFrame(flat)


def payload_to_dataframe(payload: Any) -> pd.DataFrame:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise ValueError("payload must be dict-like")
    return records_to_dataframe([payload])


def validate_labeled_dataset_columns(df: pd.DataFrame, time_column: str = TIME_COLUMN) -> None:
    required = [time_column, *RAW_INPUT_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required labeled dataset columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["heatwave_alert_int"] = out["heatwave_alert"].fillna(False).astype(int)
    out["load_margin_percent"] = out["peak_load_percent"] - out["current_load_percent"]
    out["load_ratio"] = out["current_load_percent"] / (out["peak_load_percent"] + 1e-3)
    out["heat_load_interaction"] = out["heatwave_alert_int"] * out["current_load_percent"]
    out["transformer_stress_index"] = out["transformer_age_years"] * (out["peak_load_percent"] / 100.0)
    out["water_stress_index"] = out["pressure_variance_percent"] * (1.0 + (out["rainfall_mm"] / 100.0))
    out["repair_pressure_interaction"] = (
        out["recent_repairs_last_30_days"] * out["pressure_variance_percent"]
    )
    out["congestion_signal_interaction"] = (
        out["congestion_index"] * out["signal_failure_reports_last_7_days"]
    )
    out["density_congestion_interaction"] = (
        (out["population_density_per_km2"] / 10000.0) * out["congestion_index"]
    )
    out["aging_density_intersection"] = (
        np.maximum(out["transformer_age_years"], out["pipeline_age_years"])
        * out["population_density_per_km2"]
        / 10000.0
    )
    out["humidity_temperature_interaction"] = (
        out["humidity_percent"] * out["temperature_c"] / 100.0
    )
    out["rain_pressure_interaction"] = out["rainfall_mm"] * out["pressure_variance_percent"] / 100.0
    out["high_load_flag"] = (out["current_load_percent"] > 85).astype(int)
    out["peak_load_flag"] = (out["peak_load_percent"] > 90).astype(int)
    out["high_pressure_flag"] = (out["pressure_variance_percent"] > 30).astype(int)
    out["high_density_flag"] = (out["population_density_per_km2"] > 10000).astype(int)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    include_targets: bool = False,
    time_column: str | None = None,
) -> pd.DataFrame:
    if "city" not in df.columns:
        raise ValueError("input data must include city column")

    out = df.copy()
    numeric_inputs = [column for column in RAW_INPUT_COLUMNS if column not in {"city", "heatwave_alert"}]
    out = _coerce_numeric(out, numeric_inputs)
    out = _add_engineered_features(out)
    out["city"] = out["city"].fillna("UNKNOWN").astype(str)

    selected_columns = list(MODEL_FEATURE_COLUMNS)
    if include_targets:
        selected_columns += [column for column in TARGET_COLUMNS if column in out.columns]
    if time_column and time_column in out.columns:
        selected_columns += [time_column]

    return out[selected_columns]

