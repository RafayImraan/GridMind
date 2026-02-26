from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class MLRuntimePrediction:
    probabilities: dict[str, float]
    blend_weight: float
    quality_score: float | None
    top_features: list[dict[str, Any]]
    metric_context: dict[str, float] | None


class GridMindMLRuntime:
    def __init__(self) -> None:
        self._lock = Lock()
        self._bundle: dict[str, Any] | None = None
        self._numpy_targets: dict[str, Any] = {}
        self._bundle_path: str | None = None
        self._last_error: str | None = None
        self._loaded = False

    @staticmethod
    def _configured_bundle_path() -> Path | None:
        configured = os.getenv("GRIDMIND_ML_BUNDLE_PATH", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return None

    @staticmethod
    def _candidate_bundle_paths() -> list[Path]:
        backend_root = Path(__file__).resolve().parents[1]
        return [
            backend_root / "models" / "gridmind_ml_bundle_numpy.json",
            backend_root / "models" / "gridmind_ml_bundle.joblib",
        ]

    def _load_numpy_bundle(self, path: Path) -> None:
        from ml.numpy_modeling import TrainedTarget

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("numpy bundle is not a dictionary")
        targets_payload = data.get("targets", {})
        if not isinstance(targets_payload, dict):
            raise ValueError("numpy bundle targets missing")

        parsed_targets: dict[str, Any] = {}
        for target_name, payload in targets_payload.items():
            parsed_targets[target_name] = TrainedTarget.from_dict(payload)

        self._bundle = data
        self._numpy_targets = parsed_targets
        self._last_error = None

    def _load_joblib_bundle(self, path: Path) -> None:
        import joblib

        loaded = joblib.load(path)
        self._bundle = loaded if isinstance(loaded, dict) else None
        self._numpy_targets = {}
        if not self._bundle:
            self._last_error = "invalid_bundle_format"
        else:
            self._last_error = None

    def _load_bundle(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._bundle = None
            self._numpy_targets = {}
            self._bundle_path = None

            configured = self._configured_bundle_path()
            candidates = [configured] if configured else self._candidate_bundle_paths()
            candidates = [path for path in candidates if path is not None]

            bundle_loaded = False
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    if path.suffix.lower() == ".json":
                        self._load_numpy_bundle(path)
                    else:
                        self._load_joblib_bundle(path)
                    bundle_loaded = self._bundle is not None
                    if bundle_loaded:
                        self._bundle_path = str(path)
                except ImportError:
                    self._last_error = "ml_dependency_missing"
                except Exception as exc:
                    self._last_error = f"bundle_load_failed:{exc}"
                if bundle_loaded:
                    break

            if not bundle_loaded and self._last_error is None:
                self._last_error = "bundle_not_found"
            self._loaded = True

    def _build_metric_context(self) -> dict[str, float] | None:
        if not self._bundle:
            return None

        global_metrics = self._bundle.get("global_metrics", {})
        macro_raw = global_metrics.get("macro_pr_auc_holdout")
        macro_pr_auc = float(macro_raw) if isinstance(macro_raw, (float, int)) else None

        positive_rates: list[float] = []
        targets_payload = self._bundle.get("targets", {})
        if isinstance(targets_payload, dict):
            for payload in targets_payload.values():
                if not isinstance(payload, dict):
                    continue
                holdout = payload.get("holdout_metrics", {})
                if not isinstance(holdout, dict):
                    continue
                positive_rate = holdout.get("positive_rate")
                if isinstance(positive_rate, (float, int)):
                    positive_rates.append(float(positive_rate))
        baseline_pr_auc = (
            float(sum(positive_rates) / len(positive_rates)) if positive_rates else None
        )

        if macro_pr_auc is None and baseline_pr_auc is None:
            return None

        context: dict[str, float] = {}
        if macro_pr_auc is not None:
            context["macro_pr_auc_holdout"] = macro_pr_auc
        if baseline_pr_auc is not None:
            context["random_baseline_pr_auc"] = baseline_pr_auc
        if macro_pr_auc is not None and baseline_pr_auc is not None:
            context["uplift_vs_baseline"] = macro_pr_auc - baseline_pr_auc
        return context

    def _compute_numpy_top_features(
        self,
        features: Any,
        target_name: str = "overall_failure_72h_label",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        trained = self._numpy_targets.get(target_name)
        if trained is None:
            return []

        try:
            matrix = trained.preprocessor.transform(features)
        except Exception:
            return []

        if matrix.shape[0] < 1:
            return []

        weights = trained.model.weights
        row = matrix[0]
        if len(weights) != len(row):
            return []

        feature_names = list(trained.preprocessor.numeric_features) + [
            f"city={city}" for city in trained.preprocessor.city_categories
        ]
        if len(feature_names) != len(weights):
            return []

        contributions = row * weights
        ranked = sorted(
            zip(feature_names, contributions),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        denom = sum(abs(float(value)) for _, value in ranked) or 1.0
        output: list[dict[str, Any]] = []
        for name, value in ranked[: max(1, top_k)]:
            signed = float(value)
            output.append(
                {
                    "feature": name,
                    "impact": round(abs(signed) / denom, 4),
                    "direction": "increases_risk" if signed >= 0 else "reduces_risk",
                    "logit_contribution": round(signed, 4),
                }
            )
        return output

    def status(self) -> dict[str, Any]:
        self._load_bundle()
        configured = self._configured_bundle_path()
        fallback_path = str(configured) if configured else str(self._candidate_bundle_paths()[0])
        return {
            "enabled": self._bundle is not None,
            "last_error": self._last_error,
            "bundle_path": self._bundle_path or fallback_path,
            "backend": self._bundle.get("backend", "unknown") if self._bundle else None,
            "targets": sorted(list(self._bundle.get("targets", {}).keys())) if self._bundle else [],
            "global_metrics": self._bundle.get("global_metrics", {}) if self._bundle else {},
        }

    def reload(self) -> dict[str, Any]:
        with self._lock:
            self._loaded = False
            self._bundle = None
            self._numpy_targets = {}
            self._bundle_path = None
            self._last_error = None
        return self.status()

    def predict(self, payload: Any) -> MLRuntimePrediction | None:
        self._load_bundle()
        if not self._bundle:
            return None

        try:
            from ml.features import build_feature_frame, payload_to_dataframe
        except Exception:
            self._last_error = "feature_module_load_failed"
            return None

        try:
            payload_df = payload_to_dataframe(payload)
            features = build_feature_frame(payload_df)
        except Exception as exc:
            self._last_error = f"feature_build_failed:{exc}"
            return None

        probabilities: dict[str, float] = {}
        top_features: list[dict[str, Any]] = []
        if self._bundle.get("backend") == "numpy_logistic":
            for target_name, trained in self._numpy_targets.items():
                try:
                    prob = float(trained.predict_proba(features)[0])
                except Exception:
                    continue
                probabilities[target_name] = max(0.0, min(1.0, prob))
            top_features = self._compute_numpy_top_features(features=features)
        else:
            for target_name, target_config in self._bundle.get("targets", {}).items():
                pipeline = target_config.get("pipeline")
                if pipeline is None:
                    continue
                try:
                    prob = float(pipeline.predict_proba(features)[0, 1])
                except Exception:
                    continue
                probabilities[target_name] = max(0.0, min(1.0, prob))

        if not probabilities:
            return None

        global_metrics = self._bundle.get("global_metrics", {})
        blend_weight = float(global_metrics.get("recommended_ml_blend_weight", 0.55))
        quality_score_raw = global_metrics.get("macro_pr_auc_holdout")
        quality_score = (
            float(quality_score_raw)
            if isinstance(quality_score_raw, (float, int))
            else None
        )
        metric_context = self._build_metric_context()

        return MLRuntimePrediction(
            probabilities=probabilities,
            blend_weight=blend_weight,
            quality_score=quality_score,
            top_features=top_features,
            metric_context=metric_context,
        )


ml_runtime = GridMindMLRuntime()
