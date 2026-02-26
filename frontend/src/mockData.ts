import { RiskResponse } from "./types";

export const mockRiskResponse: RiskResponse = {
  city: "Metrovale",
  timestamp: "2026-02-25T18:00:00+00:00",
  overall_risk: {
    score: 83.6,
    level: "HIGH",
    "24_hour_outlook": "Pre-failure posture detected; activate standby crews for power grid, transformer overload within 24 hours.",
    "72_hour_projection": "Escalating trajectory with projected score 92.4 (CRITICAL) by 72 hours based on current stress indicators."
  },
  systems: {
    power_grid: {
      risk_score: 98.4,
      risk_level: "CRITICAL",
      primary_drivers: [
        "current_load_percent>85",
        "peak_load_percent>90",
        "transformer_age_years>20",
        "heatwave_alert=true",
        "historical_failure_rate_percent>15"
      ],
      compound_flags: [
        "multi_factor_amplification_15pct",
        "heatwave_plus_high_load_extra_amplification"
      ]
    },
    transformer_overload: {
      risk_score: 100,
      risk_level: "CRITICAL",
      primary_drivers: [
        "peak_load_percent>90",
        "current_load_percent>85",
        "transformer_age_years>20",
        "historical_failure_rate_percent>15",
        "heatwave_alert=true"
      ],
      compound_flags: ["overload_intersection_amplification_15pct", "aging_assets_under_peak_demand"]
    },
    water_pipeline: {
      risk_score: 76.7,
      risk_level: "HIGH",
      primary_drivers: [
        "pressure_variance_percent>30",
        "pipeline_age_years>25",
        "recent_repairs_last_30_days>3"
      ],
      compound_flags: ["water_stress_compound_amplification_15pct"]
    },
    traffic_infrastructure: {
      risk_score: 80.3,
      risk_level: "HIGH",
      primary_drivers: [
        "congestion_index>75",
        "signal_failure_reports_last_7_days>5",
        "population_density_per_km2>10000"
      ],
      compound_flags: ["traffic_compound_amplification_15pct", "cascading_adjustment_plus_11.5"]
    }
  },
  cascading_failure_risks: [
    "power_to_traffic_signal_dependency (impact +8.8 traffic risk)",
    "water_burst_to_traffic_disruption_in_dense_zones (impact +6.7 traffic risk)",
    "extreme_congestion_emergency_response_delay",
    "population_density_plus_aging_infrastructure_amplification"
  ],
  priority_intervention_zones: [
    "Substation clusters with sustained grid load above 85%",
    "Aging transformer corridors where age exceeds 20 years",
    "Legacy water mains with pressure variance above 30%",
    "High-congestion junctions with repeated signal failures",
    "Dense districts with overlapping aging power and water assets"
  ],
  executive_summary:
    "Metrovale is operating at HIGH infrastructure risk (83.6/100). Primary pressure is in transformer overload (100.0) followed by power grid (98.4). Cascading pathways currently include power-to-traffic dependency and water-to-traffic disruption. Prioritize staged load reduction, transformer watch crews, and junction signal resilience in the next operational window.",
  confidence_score: 91,
  ml_insights: {
    top_features: [
      {
        feature: "aging_density_intersection",
        impact: 0.26,
        direction: "increases_risk",
        logit_contribution: 0.482
      },
      {
        feature: "transformer_age_years",
        impact: 0.2,
        direction: "increases_risk",
        logit_contribution: 0.37
      },
      {
        feature: "current_load_percent",
        impact: 0.18,
        direction: "increases_risk",
        logit_contribution: 0.334
      },
      {
        feature: "temperature_c",
        impact: 0.14,
        direction: "increases_risk",
        logit_contribution: 0.265
      },
      {
        feature: "city=Metrovale",
        impact: 0.11,
        direction: "reduces_risk",
        logit_contribution: -0.204
      }
    ],
    metric_context: {
      macro_pr_auc_holdout: 0.357,
      random_baseline_pr_auc: 0.222,
      uplift_vs_baseline: 0.135
    }
  },
  impact_model: {
    estimated_cost_avoided_usd_72h: 1275600,
    estimated_outage_hours_avoided_72h: 13.2,
    estimated_response_time_gain_percent: 27.5,
    assumptions: [
      "Impact estimates are heuristic planning values, not audited financial statements.",
      "Service interruption cost per hour scales with density and critical infrastructure overlap.",
      "Estimated gains assume pre-emptive intervention is executed within 24-hour recommendations."
    ]
  },
  external_signal: {
    source: "open-meteo.com",
    signal_type: "live_weather_crosscheck",
    status: "ok",
    retrieved_at: "2026-02-26T12:05:00Z",
    details: {
      city: "Metrovale",
      external_temperature_c: 39.4,
      external_precipitation_mm: 1.6,
      external_wind_speed_kmh: 24.5,
      temperature_delta_c: -1.9,
      rainfall_delta_mm: -10.8,
      weather_anomaly_flag: false
    },
    influence_note: "External weather signal aligned with submitted telemetry profile."
  }
};
