from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from .ml_runtime import ml_runtime
from .models import (
    FeatureImpact,
    MLInsights,
    MetricContext,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
)
from .persistence import list_recent_assessments, save_assessment
from .risk_engine import apply_ml_overlay, run_risk_assessment
from .simulation import generate_batch


app = FastAPI(
    title="GridMind AI API",
    version="0.1.0",
    description="Predictive urban infrastructure risk intelligence platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "gridmind-ai", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/v1/risk/assess", response_model=RiskAssessmentResponse)
def assess_risk(payload: RiskAssessmentRequest, response: Response) -> Any:
    try:
        result = run_risk_assessment(payload)
        ml_prediction = ml_runtime.predict(payload)
        if ml_prediction is not None:
            result = apply_ml_overlay(
                assessment=result,
                ml_probabilities=ml_prediction.probabilities,
                ml_blend_weight=ml_prediction.blend_weight,
                ml_quality_score=ml_prediction.quality_score,
            )
            if ml_prediction.top_features or ml_prediction.metric_context:
                metric_context = (
                    MetricContext(**ml_prediction.metric_context)
                    if ml_prediction.metric_context
                    else None
                )
                result.ml_insights = MLInsights(
                    top_features=[FeatureImpact(**entry) for entry in ml_prediction.top_features],
                    metric_context=metric_context,
                )
        assessment_id = save_assessment(request_payload=payload, response_payload=result)
        response.headers["X-Assessment-Id"] = assessment_id
        return result.model_dump(by_alias=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid_payload: {exc}") from exc
    except Exception as exc:  # pragma: no cover - safeguard for hackathon runtime
        raise HTTPException(status_code=500, detail=f"assessment_failed: {exc}") from exc


@app.get("/api/v1/simulate/scenarios")
def simulate_scenarios(
    count: int = Query(default=10, ge=1, le=250),
    seed: int = Query(default=42, ge=0, le=100000),
) -> dict[str, Any]:
    try:
        scenarios = generate_batch(count=count, seed=seed)
        return {"count": len(scenarios), "seed": seed, "scenarios": scenarios}
    except Exception as exc:  # pragma: no cover - safeguard for hackathon runtime
        raise HTTPException(status_code=500, detail=f"simulation_failed: {exc}") from exc


@app.get("/api/v1/risk/history")
def risk_history(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    try:
        records = list_recent_assessments(limit=limit)
        return {"count": len(records), "items": records}
    except Exception as exc:  # pragma: no cover - safeguard for hackathon runtime
        raise HTTPException(status_code=500, detail=f"history_failed: {exc}") from exc


@app.get("/api/v1/ml/status")
def ml_status() -> dict[str, Any]:
    return ml_runtime.status()


@app.post("/api/v1/ml/reload")
def ml_reload() -> dict[str, Any]:
    return ml_runtime.reload()
