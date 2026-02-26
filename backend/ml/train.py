from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    build_feature_frame,
    validate_labeled_dataset_columns,
)
from .numpy_modeling import (
    NumpyPreprocessor,
    TrainedTarget,
    fit_logistic_regression,
    fit_target_model,
    macro_quality_score,
)


def _split_train_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_size = max(1, int(len(df) * holdout_fraction))
    split_index = len(df) - holdout_size
    if split_index < 100:
        raise ValueError("dataset too small for reliable training; require at least 100 train samples")
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def _load_dataset(path: Path, time_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_labeled_dataset_columns(df, time_column=time_column)
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    if df[time_column].isna().any():
        raise ValueError("timestamp parsing failed for one or more rows")
    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def _refit_model_on_full_data(
    trained: TrainedTarget,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> None:
    preprocessor = NumpyPreprocessor.from_training_frame(X_full)
    matrix = preprocessor.transform(X_full)
    model = fit_logistic_regression(
        X_matrix=matrix,
        y=y_full.astype(int).to_numpy(),
        learning_rate=float(trained.config["learning_rate"]),
        l2=float(trained.config["l2"]),
        epochs=int(trained.config["epochs"]),
    )
    trained.preprocessor = preprocessor
    trained.model = model


def train_supervised_models(
    data_path: Path,
    output_dir: Path,
    holdout_fraction: float = 0.2,
    time_column: str = TIME_COLUMN,
    random_state: int = 42,
) -> dict:
    df = _load_dataset(path=data_path, time_column=time_column)
    features = build_feature_frame(df, include_targets=True, time_column=time_column)
    train_df, holdout_df = _split_train_holdout(features, holdout_fraction=holdout_fraction)

    trained_targets: dict[str, TrainedTarget] = {}
    for target in TARGET_COLUMNS:
        y_train = train_df[target].astype(int)
        y_holdout = holdout_df[target].astype(int)
        if y_train.nunique() < 2 or y_holdout.nunique() < 2:
            raise ValueError(f"target {target} lacks class diversity in train/holdout split")

        trained = fit_target_model(
            X_train=train_df[MODEL_FEATURE_COLUMNS],
            y_train=y_train,
            X_holdout=holdout_df[MODEL_FEATURE_COLUMNS],
            y_holdout=y_holdout,
            target_name=target,
            random_state=random_state,
        )

        y_full = features[target].astype(int)
        _refit_model_on_full_data(
            trained=trained,
            X_full=features[MODEL_FEATURE_COLUMNS],
            y_full=y_full,
        )
        trained_targets[target] = trained

    quality = macro_quality_score(trained_targets)
    blend_weight = max(0.40, min(0.75, 0.45 + 0.40 * quality))

    bundle: dict = {
        "version": "1.0",
        "backend": "numpy_logistic",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "time_column": time_column,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "targets": {
            name: {
                "target_name": trained.target_name,
                "preprocessor": trained.preprocessor.to_dict(),
                "model": trained.model.to_dict(),
                "threshold": trained.threshold,
                "model_name": "numpy_logistic_regression",
                "train_cv_pr_auc": trained.train_cv_pr_auc,
                "train_cv_fold_scores": trained.train_cv_fold_scores,
                "holdout_metrics": trained.holdout_metrics,
                "positive_rate": trained.positive_rate,
                "support": trained.support,
                "config": trained.config,
            }
            for name, trained in trained_targets.items()
        },
        "global_metrics": {
            "macro_pr_auc_holdout": quality,
            "recommended_ml_blend_weight": blend_weight,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = output_dir / "gridmind_ml_bundle_numpy.json"
    report_path = output_dir / "gridmind_ml_training_report.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    report = {
        "bundle_path": str(bundle_path),
        "backend": "numpy_logistic",
        "trained_at": bundle["trained_at"],
        "dataset_rows": bundle["dataset_rows"],
        "global_metrics": bundle["global_metrics"],
        "targets": {
            target: {
                "model_name": bundle["targets"][target]["model_name"],
                "train_cv_pr_auc": bundle["targets"][target]["train_cv_pr_auc"],
                "holdout_metrics": bundle["targets"][target]["holdout_metrics"],
                "threshold": bundle["targets"][target]["threshold"],
                "support": bundle["targets"][target]["support"],
            }
            for target in TARGET_COLUMNS
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GridMind supervised failure-probability models.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("models")),
        help="Output directory for model bundle and training report.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Chronological holdout fraction for validation (default 0.2).",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=TIME_COLUMN,
        help="Timestamp column name in labeled dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_supervised_models(
        data_path=Path(args.data).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        holdout_fraction=args.holdout_fraction,
        time_column=args.time_column,
        random_state=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
