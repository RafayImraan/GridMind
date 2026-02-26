from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WeatherData(BaseModel):
    temperature_c: Optional[float] = Field(default=None, ge=-30, le=65)
    humidity_percent: Optional[float] = Field(default=None, ge=0, le=100)
    heatwave_alert: Optional[bool] = None
    rainfall_mm: Optional[float] = Field(default=None, ge=0, le=600)


class PowerGridData(BaseModel):
    current_load_percent: Optional[float] = Field(default=None, ge=0, le=100)
    peak_load_percent: Optional[float] = Field(default=None, ge=0, le=100)
    historical_failure_rate_percent: Optional[float] = Field(default=None, ge=0, le=100)
    transformer_age_years: Optional[float] = Field(default=None, ge=0, le=80)


class WaterSystemData(BaseModel):
    pressure_variance_percent: Optional[float] = Field(default=None, ge=0, le=100)
    pipeline_age_years: Optional[float] = Field(default=None, ge=0, le=100)
    recent_repairs_last_30_days: Optional[int] = Field(default=None, ge=0, le=50)


class TrafficSystemData(BaseModel):
    congestion_index: Optional[float] = Field(default=None, ge=0, le=100)
    signal_failure_reports_last_7_days: Optional[int] = Field(default=None, ge=0, le=100)


class RiskAssessmentRequest(BaseModel):
    city: str = Field(..., min_length=1)
    timestamp: datetime
    weather: WeatherData
    power_grid: PowerGridData
    water_system: WaterSystemData
    traffic_system: TrafficSystemData
    population_density_per_km2: Optional[float] = Field(default=None, ge=0, le=150000)


class SystemRisk(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str
    primary_drivers: list[str]
    compound_flags: list[str]


class OverallRisk(BaseModel):
    score: float = Field(..., ge=0, le=100)
    level: str
    hour_24_outlook: str = Field(alias="24_hour_outlook")
    hour_72_projection: str = Field(alias="72_hour_projection")


class FeatureImpact(BaseModel):
    feature: str
    impact: float = Field(..., ge=0)
    direction: str
    logit_contribution: float


class MetricContext(BaseModel):
    macro_pr_auc_holdout: Optional[float] = Field(default=None, ge=0, le=1)
    random_baseline_pr_auc: Optional[float] = Field(default=None, ge=0, le=1)
    uplift_vs_baseline: Optional[float] = None


class MLInsights(BaseModel):
    top_features: list[FeatureImpact]
    metric_context: Optional[MetricContext] = None


class RiskAssessmentResponse(BaseModel):
    city: str
    timestamp: datetime
    overall_risk: OverallRisk
    systems: dict[str, SystemRisk]
    cascading_failure_risks: list[str]
    priority_intervention_zones: list[str]
    executive_summary: str
    confidence_score: int = Field(..., ge=0, le=100)
    ml_insights: Optional[MLInsights] = None


def classify_risk(score: float) -> str:
    if score <= 35:
        return "LOW"
    if score <= 65:
        return "MODERATE"
    if score <= 85:
        return "HIGH"
    return "CRITICAL"
