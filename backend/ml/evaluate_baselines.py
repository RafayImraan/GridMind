from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml_runtime import ml_runtime
from app.models import (
    PowerGridData,
    RiskAssessmentRequest,
    TrafficSystemData,
    WaterSystemData,
    WeatherData,
)
from app.risk_engine import run_risk_assessment

from .features import TARGET_COLUMNS, TIME_COLUMN, validate_labeled_dataset_columns
from .numpy_modeling import classification_metrics


TARGET_TO_SYSTEM_KEY = {
    "power_failure_72h_label": "power_grid",
    "transformer_overload_72h_label": "transformer_overload",
    "water_failure_72h_label": "water_pipeline",
    "traffic_failure_72h_label": "traffic_infrastructure",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    return float(max(0.0, min(1.0, numeric)))


def _load_labeled_data(path: Path, time_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_labeled_dataset_columns(df, time_column=time_column)
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    if df[time_column].isna().any():
        raise ValueError("timestamp parsing failed for one or more rows")
    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def _split_holdout(df: pd.DataFrame, holdout_fraction: float) -> pd.DataFrame:
    holdout_size = max(1, int(len(df) * holdout_fraction))
    return df.iloc[-holdout_size:].copy()


def _downsample_even(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    indices = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return df.iloc[indices].copy()


def _build_payload_from_tuple(
    row: tuple[Any, ...],
    index_map: dict[str, int],
    time_column: str,
) -> RiskAssessmentRequest:
    timestamp = row[index_map[time_column]]
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()

    # model_construct skips Pydantic re-validation for speed during offline evaluation.
    return RiskAssessmentRequest.model_construct(
        city=str(row[index_map["city"]]),
        timestamp=timestamp,
        weather=WeatherData.model_construct(
            temperature_c=float(row[index_map["temperature_c"]]),
            humidity_percent=float(row[index_map["humidity_percent"]]),
            heatwave_alert=_to_bool(row[index_map["heatwave_alert"]]),
            rainfall_mm=float(row[index_map["rainfall_mm"]]),
        ),
        power_grid=PowerGridData.model_construct(
            current_load_percent=float(row[index_map["current_load_percent"]]),
            peak_load_percent=float(row[index_map["peak_load_percent"]]),
            historical_failure_rate_percent=float(row[index_map["historical_failure_rate_percent"]]),
            transformer_age_years=float(row[index_map["transformer_age_years"]]),
        ),
        water_system=WaterSystemData.model_construct(
            pressure_variance_percent=float(row[index_map["pressure_variance_percent"]]),
            pipeline_age_years=float(row[index_map["pipeline_age_years"]]),
            recent_repairs_last_30_days=int(row[index_map["recent_repairs_last_30_days"]]),
        ),
        traffic_system=TrafficSystemData.model_construct(
            congestion_index=float(row[index_map["congestion_index"]]),
            signal_failure_reports_last_7_days=int(row[index_map["signal_failure_reports_last_7_days"]]),
        ),
        population_density_per_km2=float(row[index_map["population_density_per_km2"]]),
    )


def _extract_rule_probabilities(assessment: Any) -> dict[str, float]:
    probs: dict[str, float] = {
        "overall_failure_72h_label": _clip01(float(assessment.overall_risk.score) / 100.0)
    }
    for target, system_key in TARGET_TO_SYSTEM_KEY.items():
        score = float(assessment.systems[system_key].risk_score)
        probs[target] = _clip01(score / 100.0)
    return probs


def _reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": idx,
                    "range_low": lo,
                    "range_high": hi,
                    "count": 0,
                    "predicted_mean": 0.0,
                    "observed_rate": 0.0,
                }
            )
            continue
        rows.append(
            {
                "bin": idx,
                "range_low": lo,
                "range_high": hi,
                "count": count,
                "predicted_mean": float(np.mean(y_prob[mask])),
                "observed_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def evaluate_baselines(
    data_path: Path,
    output_path: Path,
    holdout_fraction: float = 0.2,
    time_column: str = TIME_COLUMN,
    stride: int = 15,
    max_eval_rows: int = 1200,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    raw_df = _load_labeled_data(path=data_path, time_column=time_column)
    holdout_df = _split_holdout(raw_df, holdout_fraction=holdout_fraction)
    holdout_df = holdout_df.iloc[:: max(1, stride)].copy()
    holdout_df = _downsample_even(holdout_df, max_rows=max_eval_rows)
    eval_targets = list(targets) if targets else list(TARGET_COLUMNS)
    for target in eval_targets:
        if target not in TARGET_COLUMNS:
            raise ValueError(f"unsupported target: {target}")

    status = ml_runtime.status()
    if not status.get("enabled"):
        raise RuntimeError(f"ml_runtime_disabled:{status.get('last_error')}")

    y_true_map: dict[str, list[int]] = {target: [] for target in eval_targets}
    pred_map: dict[str, dict[str, list[float]]] = {
        "rule_only": {target: [] for target in eval_targets},
        "ml_only": {target: [] for target in eval_targets},
        "hybrid": {target: [] for target in eval_targets},
    }

    blend_weights: list[float] = []
    quality_scores: list[float] = []
    index_map = {column: idx for idx, column in enumerate(holdout_df.columns)}

    for i, row in enumerate(holdout_df.itertuples(index=False, name=None), start=1):
        payload = _build_payload_from_tuple(row=row, index_map=index_map, time_column=time_column)
        rule_assessment = run_risk_assessment(payload)
        rule_probs = _extract_rule_probabilities(rule_assessment)

        ml_prediction = ml_runtime.predict(payload)
        if ml_prediction is None:
            continue

        ml_probs = {target: _clip01(ml_prediction.probabilities.get(target, 0.0)) for target in eval_targets}
        blend = _clip01(ml_prediction.blend_weight)
        rule_weight = 1.0 - blend
        hybrid_probs = {
            target: _clip01((rule_weight * rule_probs[target]) + (blend * ml_probs[target]))
            for target in eval_targets
        }

        blend_weights.append(blend)
        if ml_prediction.quality_score is not None:
            quality_scores.append(float(ml_prediction.quality_score))

        for target in eval_targets:
            y_true = int(row[index_map[target]])
            y_true_map[target].append(y_true)
            pred_map["rule_only"][target].append(rule_probs[target])
            pred_map["ml_only"][target].append(ml_probs[target])
            pred_map["hybrid"][target].append(hybrid_probs[target])

        if i % 500 == 0:
            print(f"processed {i}/{len(holdout_df)} rows")

    target_metrics: dict[str, Any] = {}
    model_macro_pr_auc: dict[str, float] = {name: 0.0 for name in pred_map.keys()}
    prevalence_values: list[float] = []

    for target in eval_targets:
        y_true_np = np.asarray(y_true_map[target], dtype=int)
        prevalence = float(np.mean(y_true_np)) if len(y_true_np) else 0.0
        prevalence_values.append(prevalence)

        model_metrics: dict[str, Any] = {}
        for model_name in ["rule_only", "ml_only", "hybrid"]:
            y_prob_np = np.asarray(pred_map[model_name][target], dtype=float)
            metrics = classification_metrics(y_true=y_true_np, y_prob=y_prob_np, threshold=0.5)
            metrics["reliability_bins_10"] = _reliability_bins(y_true=y_true_np, y_prob=y_prob_np, bins=10)
            model_metrics[model_name] = metrics
            model_macro_pr_auc[model_name] += float(metrics.get("pr_auc", 0.0))

        target_metrics[target] = {
            "class_prevalence": prevalence,
            "random_baseline_pr_auc": prevalence,
            "models": model_metrics,
        }

    for model_name in model_macro_pr_auc:
        model_macro_pr_auc[model_name] = model_macro_pr_auc[model_name] / float(len(eval_targets))

    macro_random_baseline = float(sum(prevalence_values) / len(prevalence_values))
    macro_uplift = {
        model_name: float(model_macro_pr_auc[model_name] - macro_random_baseline)
        for model_name in model_macro_pr_auc
    }

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "rows_total": int(len(raw_df)),
        "rows_holdout_original": int(max(1, int(len(raw_df) * holdout_fraction))),
        "rows_holdout": int(len(y_true_map[eval_targets[0]])),
        "holdout_fraction": holdout_fraction,
        "sampling": {
            "stride": int(max(1, stride)),
            "max_eval_rows": int(max_eval_rows),
        },
        "targets_evaluated": eval_targets,
        "models_compared": ["rule_only", "ml_only", "hybrid"],
        "macro_random_pr_auc_baseline": macro_random_baseline,
        "macro_pr_auc_by_model": model_macro_pr_auc,
        "macro_uplift_vs_random_by_model": macro_uplift,
        "runtime_context": {
            "blend_weight_mean": float(np.mean(blend_weights)) if blend_weights else None,
            "ml_quality_score_mean": float(np.mean(quality_scores)) if quality_scores else None,
        },
        "targets": target_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rule-only vs ML-only vs hybrid baselines on chronological holdout."
    )
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument(
        "--output",
        default=str(Path("models") / "gridmind_baseline_comparison.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Chronological holdout fraction (default 0.2).",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=TIME_COLUMN,
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Evaluate every Nth row from holdout to reduce runtime (default 15).",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=1200,
        help="Upper bound on evaluated holdout rows after stride (0 disables cap).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset of targets to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_baselines(
        data_path=Path(args.data).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        holdout_fraction=args.holdout_fraction,
        time_column=args.time_column,
        stride=args.stride,
        max_eval_rows=args.max_eval_rows,
        targets=args.targets,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
