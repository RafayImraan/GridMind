from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.models import RiskAssessmentRequest
from app.risk_engine import run_risk_assessment
from app.simulation import CITY_PROFILES, generate_scenario
from ml.features import TARGET_COLUMNS, flatten_payload_record


def _sample_label(probability: float, rng: np.random.Generator) -> int:
    return int(rng.uniform(0.0, 1.0) < probability)


def _prob_from_score(score: float, base: float = 0.02, scale: float = 0.90) -> float:
    return float(max(0.01, min(0.99, base + scale * (score / 100.0))))


def generate_labeled_records(count: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    scenario_types = ["baseline", "heatwave", "high_density", "failure_event", "compound"]
    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for index in range(count):
        city, density = CITY_PROFILES[index % len(CITY_PROFILES)]
        scenario_type = scenario_types[index % len(scenario_types)]
        timestamp = start_time + timedelta(hours=index)
        scenario = generate_scenario(
            scenario_type=scenario_type,
            city=city,
            density=density,
            now=timestamp,
        )
        request = RiskAssessmentRequest.model_validate(scenario)
        assessment = run_risk_assessment(request)
        flattened = flatten_payload_record(scenario)

        power_prob = _prob_from_score(assessment.systems["power_grid"].risk_score, base=0.02, scale=0.92)
        transformer_prob = _prob_from_score(
            assessment.systems["transformer_overload"].risk_score,
            base=0.02,
            scale=0.92,
        )
        water_prob = _prob_from_score(assessment.systems["water_pipeline"].risk_score, base=0.02, scale=0.92)
        traffic_prob = _prob_from_score(
            assessment.systems["traffic_infrastructure"].risk_score,
            base=0.02,
            scale=0.92,
        )
        overall_prob = _prob_from_score(assessment.overall_risk.score, base=0.01, scale=0.94)

        flattened.update(
            {
                "scenario_type": scenario_type,
                TARGET_COLUMNS[0]: _sample_label(overall_prob, rng),
                TARGET_COLUMNS[1]: _sample_label(power_prob, rng),
                TARGET_COLUMNS[2]: _sample_label(transformer_prob, rng),
                TARGET_COLUMNS[3]: _sample_label(water_prob, rng),
                TARGET_COLUMNS[4]: _sample_label(traffic_prob, rng),
            }
        )
        rows.append(flattened)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo labeled dataset for GridMind ML pipeline testing.")
    parser.add_argument("--count", type=int, default=2000, help="Number of labeled rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=str(Path("models") / "demo_labeled_dataset.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    records = generate_labeled_records(count=args.count, seed=args.seed)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"wrote {len(records)} labeled rows to {output_path}")


if __name__ == "__main__":
    main()

