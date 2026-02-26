from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.models import RiskAssessmentRequest  # noqa: E402
from backend.app.risk_engine import run_risk_assessment  # noqa: E402
from backend.app.simulation import generate_batch  # noqa: E402


def generate_dataset(count: int, seed: int) -> list[dict]:
    scenarios = generate_batch(count=count, seed=seed)
    enriched: list[dict] = []
    for scenario in scenarios:
        request_model = RiskAssessmentRequest.model_validate(scenario)
        assessment = run_risk_assessment(request_model)
        enriched.append(
            {
                "input": scenario,
                "assessment": assessment.model_dump(by_alias=True),
            }
        )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic GridMind infrastructure scenarios.")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("outputs") / "city_scenarios.json"),
        help="Output JSON path relative to data_simulation directory.",
    )
    args = parser.parse_args()

    records = generate_dataset(count=args.count, seed=args.seed)
    output_path = Path(__file__).resolve().parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()

