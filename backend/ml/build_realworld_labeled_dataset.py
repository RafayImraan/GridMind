from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import RAW_INPUT_COLUMNS, TARGET_COLUMNS, TIME_COLUMN


SYSTEM_MAP = {
    TARGET_COLUMNS[0]: {"overall", "any", "citywide", "power", "transformer", "water", "traffic"},
    TARGET_COLUMNS[1]: {"power", "power_grid", "grid"},
    TARGET_COLUMNS[2]: {"transformer", "transformer_overload"},
    TARGET_COLUMNS[3]: {"water", "water_pipeline", "pipeline"},
    TARGET_COLUMNS[4]: {"traffic", "traffic_infrastructure", "signals"},
}


def _normalize_system(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _series_to_epoch_ns(series: pd.Series) -> np.ndarray:
    # pandas may parse timezone-aware datetimes as microsecond resolution.
    # Force nanosecond resolution before integer conversion for horizon math.
    return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def _labels_for_city_window(
    telemetry_ts: np.ndarray,
    incident_ts: np.ndarray,
    horizon_hours: int,
) -> np.ndarray:
    if incident_ts.size == 0:
        return np.zeros(len(telemetry_ts), dtype=int)
    horizon_ns = np.int64(horizon_hours) * np.int64(3600) * np.int64(1_000_000_000)
    idx = np.searchsorted(incident_ts, telemetry_ts, side="left")
    valid = idx < incident_ts.size
    labels = np.zeros(len(telemetry_ts), dtype=int)
    if not np.any(valid):
        return labels

    next_incident = np.zeros(len(telemetry_ts), dtype=np.int64)
    next_incident[valid] = incident_ts[idx[valid]]
    delta = next_incident - telemetry_ts
    labels[valid] = (delta[valid] >= 0) & (delta[valid] <= horizon_ns)
    return labels.astype(int)


def build_labeled_dataset(
    telemetry_path: Path,
    incidents_path: Path,
    output_path: Path,
    horizon_hours: int = 72,
) -> Path:
    telemetry = pd.read_csv(telemetry_path)
    incidents = pd.read_csv(incidents_path)

    required_telemetry = [TIME_COLUMN, *RAW_INPUT_COLUMNS]
    missing_telemetry = [column for column in required_telemetry if column not in telemetry.columns]
    if missing_telemetry:
        raise ValueError(f"telemetry file missing columns: {missing_telemetry}")

    required_incidents = [TIME_COLUMN, "city", "system"]
    missing_incidents = [column for column in required_incidents if column not in incidents.columns]
    if missing_incidents:
        raise ValueError(f"incidents file missing columns: {missing_incidents}")

    telemetry[TIME_COLUMN] = pd.to_datetime(telemetry[TIME_COLUMN], utc=True, errors="coerce")
    incidents[TIME_COLUMN] = pd.to_datetime(incidents[TIME_COLUMN], utc=True, errors="coerce")
    if telemetry[TIME_COLUMN].isna().any() or incidents[TIME_COLUMN].isna().any():
        raise ValueError("failed to parse one or more timestamps in telemetry/incidents")

    telemetry = telemetry.sort_values(["city", TIME_COLUMN]).reset_index(drop=True)
    incidents = incidents.sort_values(["city", TIME_COLUMN]).reset_index(drop=True)
    incidents["system_norm"] = incidents["system"].map(_normalize_system)

    for target in TARGET_COLUMNS:
        telemetry[target] = 0

    for city, city_frame in telemetry.groupby("city", sort=False):
        telemetry_idx = city_frame.index.to_numpy()
        telemetry_ts = _series_to_epoch_ns(city_frame[TIME_COLUMN])

        city_incidents = incidents[incidents["city"] == city]
        all_incident_ts = _series_to_epoch_ns(city_incidents[TIME_COLUMN])
        telemetry.loc[telemetry_idx, TARGET_COLUMNS[0]] = _labels_for_city_window(
            telemetry_ts=telemetry_ts,
            incident_ts=all_incident_ts,
            horizon_hours=horizon_hours,
        )

        for target in TARGET_COLUMNS[1:]:
            systems = SYSTEM_MAP[target]
            filtered = city_incidents[city_incidents["system_norm"].isin(systems)]
            incident_ts = _series_to_epoch_ns(filtered[TIME_COLUMN])
            telemetry.loc[telemetry_idx, target] = _labels_for_city_window(
                telemetry_ts=telemetry_ts,
                incident_ts=incident_ts,
                horizon_hours=horizon_hours,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real-world labeled GridMind dataset from telemetry and incident logs."
    )
    parser.add_argument("--telemetry", required=True, help="Telemetry CSV path.")
    parser.add_argument("--incidents", required=True, help="Incident CSV path.")
    parser.add_argument("--output", required=True, help="Output labeled CSV path.")
    parser.add_argument("--horizon-hours", type=int, default=72, help="Forward label window in hours.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_labeled_dataset(
        telemetry_path=Path(args.telemetry).expanduser().resolve(),
        incidents_path=Path(args.incidents).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        horizon_hours=args.horizon_hours,
    )
    print(f"labeled dataset written to {path}")


if __name__ == "__main__":
    main()
