from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _expected_calibration_error(
    y_true: pd.Series,
    y_prob: np.ndarray,
    bins: int = 10,
) -> float:
    y_true_np = y_true.to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true_np)
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        mask = (y_prob >= lo) & (y_prob < hi if idx < bins - 1 else y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_prob = float(y_prob[mask].mean())
        bin_true = float(y_true_np[mask].mean())
        ece += abs(bin_prob - bin_true) * (count / total)
    return float(ece)


def optimal_threshold_by_f1(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    threshold = float(thresholds[best_index])
    return max(0.05, min(0.95, threshold))


def classification_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ece_10_bins": _expected_calibration_error(y_true, y_prob, bins=10),
        "positive_rate": float(y_true.mean()),
    }
    return metrics


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_pipe, MODEL_CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_estimators(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "logistic_balanced": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def _time_series_splits(
    sample_count: int,
    n_splits: int = 5,
    min_test_size: int = 40,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count < 2 * min_test_size:
        return []

    dynamic_splits = min(n_splits, max(2, sample_count // min_test_size))
    splitter = TimeSeriesSplit(n_splits=dynamic_splits)
    return list(splitter.split(np.arange(sample_count)))


def cross_validated_pr_auc(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
) -> tuple[float, list[float]]:
    fold_scores: list[float] = []
    for train_idx, valid_idx in _time_series_splits(sample_count=len(X)):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            continue

        model = clone(estimator)
        model.fit(X.iloc[train_idx], y_train)
        prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_scores.append(float(average_precision_score(y_valid, prob)))

    if not fold_scores:
        return -1.0, []
    return float(np.mean(fold_scores)), fold_scores


@dataclass
class TrainedTarget:
    target_name: str
    model_name: str
    pipeline: Pipeline
    threshold: float
    train_cv_pr_auc: float
    train_cv_fold_scores: list[float]
    holdout_metrics: dict[str, float | None]
    positive_rate: float
    support: int


def fit_target_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    target_name: str,
    random_state: int = 42,
) -> TrainedTarget:
    candidates = build_candidate_estimators(random_state=random_state)
    best_name = ""
    best_score = -1.0
    best_fold_scores: list[float] = []

    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        cv_score, fold_scores = cross_validated_pr_auc(X=X_train, y=y_train, estimator=pipeline)
        if cv_score > best_score:
            best_score = cv_score
            best_name = name
            best_fold_scores = fold_scores

    if not best_name:
        raise ValueError(f"no valid model could be trained for target={target_name}")

    best_pipeline = build_pipeline(candidates[best_name])
    best_pipeline.fit(X_train, y_train)
    holdout_prob = best_pipeline.predict_proba(X_holdout)[:, 1]
    threshold = optimal_threshold_by_f1(y_holdout, holdout_prob)
    holdout = classification_metrics(y_true=y_holdout, y_prob=holdout_prob, threshold=threshold)

    return TrainedTarget(
        target_name=target_name,
        model_name=best_name,
        pipeline=best_pipeline,
        threshold=threshold,
        train_cv_pr_auc=best_score,
        train_cv_fold_scores=best_fold_scores,
        holdout_metrics=holdout,
        positive_rate=float(y_train.mean()),
        support=int(len(y_train)),
    )


def macro_quality_score(targets: dict[str, TrainedTarget]) -> float:
    pr_aucs = [target.holdout_metrics.get("pr_auc") for target in targets.values()]
    valid = [float(x) for x in pr_aucs if isinstance(x, (float, int))]
    if not valid:
        return 0.0
    return float(np.mean(valid))

