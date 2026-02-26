export type RiskLevel = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";

export interface RiskRequest {
  city: string;
  timestamp: string;
  weather: {
    temperature_c: number;
    humidity_percent: number;
    heatwave_alert: boolean;
    rainfall_mm: number;
  };
  power_grid: {
    current_load_percent: number;
    peak_load_percent: number;
    historical_failure_rate_percent: number;
    transformer_age_years: number;
  };
  water_system: {
    pressure_variance_percent: number;
    pipeline_age_years: number;
    recent_repairs_last_30_days: number;
  };
  traffic_system: {
    congestion_index: number;
    signal_failure_reports_last_7_days: number;
  };
  population_density_per_km2: number;
}

export interface SystemRisk {
  risk_score: number;
  risk_level: RiskLevel;
  primary_drivers: string[];
  compound_flags: string[];
}

export interface FeatureImpact {
  feature: string;
  impact: number;
  direction: "increases_risk" | "reduces_risk";
  logit_contribution: number;
}

export interface MetricContext {
  macro_pr_auc_holdout?: number;
  random_baseline_pr_auc?: number;
  uplift_vs_baseline?: number;
}

export interface RiskResponse {
  city: string;
  timestamp: string;
  overall_risk: {
    score: number;
    level: RiskLevel;
    "24_hour_outlook": string;
    "72_hour_projection": string;
  };
  systems: {
    power_grid: SystemRisk;
    transformer_overload: SystemRisk;
    water_pipeline: SystemRisk;
    traffic_infrastructure: SystemRisk;
  };
  cascading_failure_risks: string[];
  priority_intervention_zones: string[];
  executive_summary: string;
  confidence_score: number;
  ml_insights?: {
    top_features: FeatureImpact[];
    metric_context?: MetricContext | null;
  };
}
