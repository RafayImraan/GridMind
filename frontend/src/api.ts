import { RiskRequest, RiskResponse } from "./types";
import { mockRiskResponse } from "./mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const defaultRiskRequest: RiskRequest = {
  city: "Metrovale",
  timestamp: new Date().toISOString(),
  weather: {
    temperature_c: 41.3,
    humidity_percent: 48.0,
    heatwave_alert: true,
    rainfall_mm: 12.4
  },
  power_grid: {
    current_load_percent: 91.2,
    peak_load_percent: 97.5,
    historical_failure_rate_percent: 18.6,
    transformer_age_years: 27
  },
  water_system: {
    pressure_variance_percent: 34.2,
    pipeline_age_years: 31.5,
    recent_repairs_last_30_days: 5
  },
  traffic_system: {
    congestion_index: 82.1,
    signal_failure_reports_last_7_days: 9
  },
  population_density_per_km2: 13800
};

export async function fetchRiskAssessment(payload: RiskRequest): Promise<RiskResponse> {
  return fetchRiskAssessmentWithOptions(payload, { allowFallback: false });
}

export async function fetchRiskAssessmentWithOptions(
  payload: RiskRequest,
  options: { allowFallback: boolean }
): Promise<RiskResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/risk/assess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }
    return (await response.json()) as RiskResponse;
  } catch {
    if (!options.allowFallback) {
      throw new Error("assessment_request_failed");
    }
    return {
      ...mockRiskResponse,
      city: payload.city,
      timestamp: payload.timestamp
    };
  }
}
