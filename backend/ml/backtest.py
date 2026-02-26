from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
    classification_metrics,
    fit_logistic_regression,
    fit_target_model,
)


def _load_dataset(path: Path, time_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_labeled_dataset_columns(df, time_column=time_column)
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    if df[time_column].isna().any():
        raise ValueError("timestamp parsing failed for one or more rows")
    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def _rolling_windows(
    sample_count: int,
    min_train_size: int,
    test_size: int,
    step_size: int,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    train_end = min_train_size
    while train_end + test_size <= sample_count:
        test_end = train_end + test_size
        windows.append((0, train_end, train_end, test_end))
        train_end += step_size
    return windows


def _aggregate_metric_values(values: list[float | None]) -> dict[str, float | None]:
    valid = [float(x) for x in values if isinstance(x, (float, int))]
    if not valid:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def _train_window_model(
    X_train_window: pd.DataFrame,
    y_train_window: pd.Series,
    target_name: str,
    seed: int,
):
    split_index = int(len(X_train_window) * 0.85)
    split_index = max(80, split_index)
    if split_index >= len(X_train_window):
        split_index = len(X_train_window) - 1

    X_subtrain = X_train_window.iloc[:split_index]
    y_subtrain = y_train_window.iloc[:split_index]
    X_calibration = X_train_window.iloc[split_index:]
    y_calibration = y_train_window.iloc[split_index:]
    if y_subtrain.nunique() < 2 or y_calibration.nunique() < 2:
        return None

    trained = fit_target_model(
        X_train=X_subtrain,
        y_train=y_subtrain,
        X_holdout=X_calibration,
        y_holdout=y_calibration,
        target_name=target_name,
        random_state=seed,
    )
    full_preprocessor = NumpyPreprocessor.from_training_frame(X_train_window)
    matrix = full_preprocessor.transform(X_train_window)
    full_model = fit_logistic_regression(
        X_matrix=matrix,
        y=y_train_window.astype(int).to_numpy(),
        learning_rate=float(trained.config["learning_rate"]),
        l2=float(trained.config["l2"]),
        epochs=int(trained.config["epochs"]),
    )
    trained.preprocessor = full_preprocessor
    trained.model = full_model
    return trained


def run_realworld_backtest(
    data_path: Path,
    output_path: Path,
    time_column: str = TIME_COLUMN,
    min_train_ratio: float = 0.5,
    test_ratio: float = 0.1,
    step_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    raw = _load_dataset(path=data_path, time_column=time_column)
    dataset = build_feature_frame(raw, include_targets=True, time_column=time_column)
    n = len(dataset)

    min_train_size = max(150, int(n * min_train_ratio))
    test_size = max(40, int(n * test_ratio))
    step_size = max(20, int(n * step_ratio))
    windows = _rolling_windows(
        sample_count=n,
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    if not windows:
        raise ValueError("not enough rows for rolling backtest windows")

    target_results: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        metric_history: dict[str, list[float | None]] = {
            "roc_auc": [],
            "pr_auc": [],
            "brier": [],
            "log_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "ece_10_bins": [],
        }
        window_counter = 0

        for _, train_end, test_start, test_end in windows:
            X_train_window = dataset.iloc[:train_end][MODEL_FEATURE_COLUMNS]
            y_train_window = dataset.iloc[:train_end][target].astype(int)
            X_test = dataset.iloc[test_start:test_end][MODEL_FEATURE_COLUMNS]
            y_test = dataset.iloc[test_start:test_end][target].astype(int)

            if y_train_window.nunique() < 2 or y_test.nunique() < 2:
                continue

            trained = _train_window_model(
                X_train_window=X_train_window,
                y_train_window=y_train_window,
                target_name=target,
                seed=seed,
            )
            if trained is None:
                continue

            prob = trained.predict_proba(X_test)
            metrics = classification_metrics(
                y_true=y_test.astype(int).to_numpy(),
                y_prob=prob,
                threshold=trained.threshold,
            )
            for metric_name in metric_history:
                metric_history[metric_name].append(metrics.get(metric_name))
            window_counter += 1

        target_results[target] = {
            "windows_evaluated": window_counter,
            "metrics": {
                metric_name: _aggregate_metric_values(values)
                for metric_name, values in metric_history.items()
            },
        }

    macro_pr_auc_values = [
        target_results[target]["metrics"]["pr_auc"]["mean"]
        for target in TARGET_COLUMNS
        if target_results[target]["metrics"]["pr_auc"]["mean"] is not None
    ]
    macro_pr_auc = float(np.mean(macro_pr_auc_values)) if macro_pr_auc_values else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "rows": n,
        "window_config": {
            "min_train_ratio": min_train_ratio,
            "test_ratio": test_ratio,
            "step_ratio": step_ratio,
            "windows_total": len(windows),
        },
        "macro_pr_auc_mean": macro_pr_auc,
        "targets": target_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-world chronological backtest for GridMind models.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument(
        "--output",
        default=str(Path("models") / "gridmind_realworld_backtest.json"),
        help="Output JSON report path.",
    )
    parser.add_argument("--time-column", default=TIME_COLUMN, help="Timestamp column name.")
    parser.add_argument("--min-train-ratio", type=float, default=0.5, help="Minimum train ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test window ratio.")
    parser.add_argument("--step-ratio", type=float, default=0.1, help="Walk-forward step ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_realworld_backtest(
        data_path=Path(args.data).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        time_column=args.time_column,
        min_train_ratio=args.min_train_ratio,
        test_ratio=args.test_ratio,
        step_ratio=args.step_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
