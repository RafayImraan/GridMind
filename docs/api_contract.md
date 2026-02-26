# API Contract

## Request Schema (`POST /api/v1/risk/assess`)

```json
{
  "city": "string",
  "timestamp": "ISO-8601 datetime",
  "weather": {
    "temperature_c": 0,
    "humidity_percent": 0,
    "heatwave_alert": true,
    "rainfall_mm": 0
  },
  "power_grid": {
    "current_load_percent": 0,
    "peak_load_percent": 0,
    "historical_failure_rate_percent": 0,
    "transformer_age_years": 0
  },
  "water_system": {
    "pressure_variance_percent": 0,
    "pipeline_age_years": 0,
    "recent_repairs_last_30_days": 0
  },
  "traffic_system": {
    "congestion_index": 0,
    "signal_failure_reports_last_7_days": 0
  },
  "population_density_per_km2": 0
}
```

On success, response includes header:

- `X-Assessment-Id: asm_<id>`

## Response Schema

```json
{
  "city": "string",
  "timestamp": "ISO-8601 datetime",
  "overall_risk": {
    "score": 0,
    "level": "LOW|MODERATE|HIGH|CRITICAL",
    "24_hour_outlook": "string",
    "72_hour_projection": "string"
  },
  "systems": {
    "power_grid": {
      "risk_score": 0,
      "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
      "primary_drivers": ["string"],
      "compound_flags": ["string"]
    },
    "transformer_overload": {
      "risk_score": 0,
      "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
      "primary_drivers": ["string"],
      "compound_flags": ["string"]
    },
    "water_pipeline": {
      "risk_score": 0,
      "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
      "primary_drivers": ["string"],
      "compound_flags": ["string"]
    },
    "traffic_infrastructure": {
      "risk_score": 0,
      "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
      "primary_drivers": ["string"],
      "compound_flags": ["string"]
    }
  },
  "cascading_failure_risks": ["string"],
  "priority_intervention_zones": ["string"],
  "executive_summary": "string",
  "confidence_score": 0,
  "impact_model": {
    "estimated_cost_avoided_usd_72h": 0,
    "estimated_outage_hours_avoided_72h": 0,
    "estimated_response_time_gain_percent": 0,
    "assumptions": ["string"]
  },
  "external_signal": {
    "source": "open-meteo.com",
    "signal_type": "live_weather_crosscheck",
    "status": "ok|unsupported_city|fetch_failed",
    "retrieved_at": "ISO-8601 datetime",
    "details": {},
    "influence_note": "string"
  },
  "ml_insights": {
    "top_features": [
      {
        "feature": "aging_density_intersection",
        "impact": 0.21,
        "direction": "increases_risk",
        "logit_contribution": 0.33
      }
    ],
    "metric_context": {
      "macro_pr_auc_holdout": 0.36,
      "random_baseline_pr_auc": 0.22,
      "uplift_vs_baseline": 0.14
    }
  }
}
```

## Concrete Examples

- Request sample: `backend/example_request.json`
- Response sample: `backend/example_response.json`

## History Endpoint (`GET /api/v1/risk/history?limit=20`)

Returns most-recent saved assessments from local storage.

Response shape:

```json
{
  "count": 2,
  "items": [
    {
      "assessment_id": "asm_ab12cd34ef56",
      "saved_at": "ISO-8601 datetime",
      "request": {},
      "response": {}
    }
  ]
}
```

## ML Status Endpoint (`GET /api/v1/ml/status`)

```json
{
  "enabled": true,
  "last_error": null,
  "bundle_path": "string",
  "backend": "numpy_logistic",
  "targets": ["overall_failure_72h_label"],
  "global_metrics": {
    "macro_pr_auc_holdout": 0.81,
    "recommended_ml_blend_weight": 0.66
  }
}
```
