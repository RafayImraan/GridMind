from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _safe_div(a: float, b: float) -> float:
    if abs(b) < EPS:
        return 0.0
    return float(a / b)


def _roc_auc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    pos_count = int(pos.sum())
    neg_count = int(neg.sum())
    if pos_count == 0 or neg_count == 0:
        return None

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def _average_precision_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def optimal_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(y_prob)
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    y_true = y_true.astype(int)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return float(max(0.05, min(0.95, best_threshold)))


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = _safe_div(tp + tn, len(y_true))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = float(
        -np.mean(
            y_true * np.log(np.clip(y_prob, EPS, 1.0 - EPS))
            + (1 - y_true) * np.log(np.clip(1.0 - y_prob, EPS, 1.0 - EPS))
        )
    )

    # 10-bin expected calibration error
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for idx in range(10):
        lo = bins[idx]
        hi = bins[idx + 1]
        mask = (y_prob >= lo) & (y_prob < hi if idx < 9 else y_prob <= hi)
        if not np.any(mask):
            continue
        prob_avg = float(np.mean(y_prob[mask]))
        true_avg = float(np.mean(y_true[mask]))
        ece += abs(prob_avg - true_avg) * float(np.mean(mask))

    return {
        "roc_auc": _roc_auc_binary(y_true, y_prob),
        "pr_auc": _average_precision_binary(y_true, y_prob),
        "brier": brier,
        "log_loss": logloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ece_10_bins": float(ece),
        "positive_rate": float(np.mean(y_true)),
    }


@dataclass
class NumpyPreprocessor:
    numeric_features: list[str]
    city_categories: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = X.copy()
        for feature in self.numeric_features:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")
            out[feature] = out[feature].fillna(self.medians[feature])
            std = self.stds[feature] if self.stds[feature] > 0 else 1.0
            out[feature] = (out[feature] - self.means[feature]) / std

        out["city"] = out["city"].fillna("UNKNOWN").astype(str)
        city_matrix = np.zeros((len(out), len(self.city_categories)), dtype=float)
        city_index = {name: idx for idx, name in enumerate(self.city_categories)}
        for row_idx, city in enumerate(out["city"]):
            if city in city_index:
                city_matrix[row_idx, city_index[city]] = 1.0

        numeric_matrix = out[self.numeric_features].to_numpy(dtype=float)
        return np.hstack([numeric_matrix, city_matrix])

    def to_dict(self) -> dict:
        return {
            "numeric_features": self.numeric_features,
            "city_categories": self.city_categories,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_training_frame(cls, X: pd.DataFrame) -> "NumpyPreprocessor":
        numeric_features = list(MODEL_NUMERIC_FEATURES)
        city_values = X["city"].fillna("UNKNOWN").astype(str)
        city_categories = sorted(city_values.unique().tolist())

        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for feature in numeric_features:
            series = pd.to_numeric(X[feature], errors="coerce")
            median = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(median)
            medians[feature] = median
            means[feature] = float(filled.mean())
            std_val = float(filled.std(ddof=0))
            stds[feature] = std_val if std_val > 1e-9 else 1.0

        return cls(
            numeric_features=numeric_features,
            city_categories=city_categories,
            medians=medians,
            means=means,
            stds=stds,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "NumpyPreprocessor":
        return cls(
            numeric_features=list(payload["numeric_features"]),
            city_categories=list(payload["city_categories"]),
            medians={k: float(v) for k, v in payload["medians"].items()},
            means={k: float(v) for k, v in payload["means"].items()},
            stds={k: float(v) for k, v in payload["stds"].items()},
        )


@dataclass
class NumpyLogisticModel:
    weights: np.ndarray
    bias: float
    l2: float
    learning_rate: float
    epochs: int

    def predict_proba(self, X_matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(X_matrix @ self.weights + self.bias)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NumpyLogisticModel":
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            l2=float(payload.get("l2", 0.001)),
            learning_rate=float(payload.get("learning_rate", 0.05)),
            epochs=int(payload.get("epochs", 400)),
        )


def fit_logistic_regression(
    X_matrix: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    l2: float,
    epochs: int,
) -> NumpyLogisticModel:
    y = y.astype(float)
    n, d = X_matrix.shape
    weights = np.zeros(d, dtype=float)
    bias = 0.0

    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    pos_weight = n / (2.0 * pos)
    neg_weight = n / (2.0 * neg)
    sample_weight = np.where(y == 1, pos_weight, neg_weight)

    for _ in range(epochs):
        probs = _sigmoid(X_matrix @ weights + bias)
        error = (probs - y) * sample_weight
        grad_w = (X_matrix.T @ error) / n + l2 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return NumpyLogisticModel(
        weights=weights,
        bias=bias,
        l2=l2,
        learning_rate=learning_rate,
        epochs=epochs,
    )


@dataclass
class TrainedTarget:
    target_name: str
    preprocessor: NumpyPreprocessor
    model: NumpyLogisticModel
    threshold: float
    train_cv_pr_auc: float
    train_cv_fold_scores: list[float]
    holdout_metrics: dict[str, float | None]
    positive_rate: float
    support: int
    config: dict[str, float]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(X)
        return self.model.predict_proba(matrix)

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "preprocessor": self.preprocessor.to_dict(),
            "model": self.model.to_dict(),
            "threshold": self.threshold,
            "train_cv_pr_auc": self.train_cv_pr_auc,
            "train_cv_fold_scores": self.train_cv_fold_scores,
            "holdout_metrics": self.holdout_metrics,
            "positive_rate": self.positive_rate,
            "support": self.support,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainedTarget":
        return cls(
            target_name=str(payload["target_name"]),
            preprocessor=NumpyPreprocessor.from_dict(payload["preprocessor"]),
            model=NumpyLogisticModel.from_dict(payload["model"]),
            threshold=float(payload["threshold"]),
            train_cv_pr_auc=float(payload["train_cv_pr_auc"]),
            train_cv_fold_scores=[float(x) for x in payload["train_cv_fold_scores"]],
            holdout_metrics=dict(payload["holdout_metrics"]),
            positive_rate=float(payload["positive_rate"]),
            support=int(payload["support"]),
            config=dict(payload["config"]),
        )


def _time_series_splits(sample_count: int, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count < 160:
        return []
    split_points = np.linspace(int(sample_count * 0.5), sample_count - 40, n_splits, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_end in split_points:
        test_end = min(sample_count, train_end + 40)
        if test_end <= train_end:
            continue
        train_idx = np.arange(0, train_end)
        valid_idx = np.arange(train_end, test_end)
        splits.append((train_idx, valid_idx))
    return splits


def _evaluate_config_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    learning_rate: float,
    l2: float,
    epochs: int,
) -> tuple[float, list[float]]:
    fold_scores: list[float] = []
    for train_idx, valid_idx in _time_series_splits(len(X)):
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        if np.unique(y_train).size < 2 or np.unique(y_valid).size < 2:
            continue
        pre = NumpyPreprocessor.from_training_frame(X.iloc[train_idx])
        X_train_matrix = pre.transform(X.iloc[train_idx])
        X_valid_matrix = pre.transform(X.iloc[valid_idx])
        model = fit_logistic_regression(
            X_matrix=X_train_matrix,
            y=y_train,
            learning_rate=learning_rate,
            l2=l2,
            epochs=epochs,
        )
        prob = model.predict_proba(X_valid_matrix)
        fold_scores.append(_average_precision_binary(y_valid, prob))
    if not fold_scores:
        return -1.0, []
    return float(np.mean(fold_scores)), fold_scores


def fit_target_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    target_name: str,
    random_state: int = 42,
) -> TrainedTarget:
    _ = random_state
    y_train_np = y_train.astype(int).to_numpy()
    y_holdout_np = y_holdout.astype(int).to_numpy()

    configs = [
        {"learning_rate": 0.03, "l2": 0.001, "epochs": 700},
        {"learning_rate": 0.05, "l2": 0.003, "epochs": 500},
        {"learning_rate": 0.08, "l2": 0.006, "epochs": 400},
    ]

    best_config: dict[str, float] | None = None
    best_cv = -1.0
    best_folds: list[float] = []
    for config in configs:
        cv_score, fold_scores = _evaluate_config_cv(
            X=X_train,
            y=y_train_np,
            learning_rate=float(config["learning_rate"]),
            l2=float(config["l2"]),
            epochs=int(config["epochs"]),
        )
        if cv_score > best_cv:
            best_cv = cv_score
            best_config = config
            best_folds = fold_scores

    if best_config is None:
        best_config = configs[0]
        best_cv = -1.0
        best_folds = []

    preprocessor = NumpyPreprocessor.from_training_frame(X_train)
    X_train_matrix = preprocessor.transform(X_train)
    model = fit_logistic_regression(
        X_matrix=X_train_matrix,
        y=y_train_np,
        learning_rate=float(best_config["learning_rate"]),
        l2=float(best_config["l2"]),
        epochs=int(best_config["epochs"]),
    )

    holdout_prob = model.predict_proba(preprocessor.transform(X_holdout))
    if np.unique(y_holdout_np).size < 2:
        threshold = 0.5
    else:
        threshold = optimal_threshold_by_f1(y_holdout_np, holdout_prob)
    holdout = classification_metrics(
        y_true=y_holdout_np,
        y_prob=holdout_prob,
        threshold=threshold,
    )

    return TrainedTarget(
        target_name=target_name,
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        train_cv_pr_auc=best_cv,
        train_cv_fold_scores=best_folds,
        holdout_metrics=holdout,
        positive_rate=float(y_train.mean()),
        support=int(len(y_train)),
        config={k: float(v) for k, v in best_config.items()},
    )


def macro_quality_score(targets: dict[str, TrainedTarget]) -> float:
    pr_aucs = [target.holdout_metrics.get("pr_auc") for target in targets.values()]
    valid = [float(x) for x in pr_aucs if isinstance(x, (float, int))]
    if not valid:
        return 0.0
    return float(np.mean(valid))
