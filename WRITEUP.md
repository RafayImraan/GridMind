# GridMind AI Writeup

## 1) Project Summary

GridMind AI is a predictive urban infrastructure risk intelligence platform.  
It forecasts near-term (0-72h) failure risk across:

- Power Grid
- Transformer Overload
- Water Pipeline System
- Traffic Signal Infrastructure

The product is built for city operations teams that need early warning before failures cascade across systems.

## 2) Problem and Why It Matters

Municipal systems are highly interdependent. A high-load power failure can degrade traffic signal uptime, and water main breaks in dense corridors can amplify mobility and emergency response delays.

Most city response is reactive. GridMind shifts that workflow to proactive intervention by combining deterministic safety logic with machine-learned probabilities and clear operator-facing outputs.

## 3) What GridMind Delivers

For each assessment input, GridMind returns:

- Overall city risk score and level (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`)
- Per-system risk scores and primary drivers
- Compound risk flags and cascade risk statements
- 24-hour outlook and 72-hour projection text
- Priority intervention zones
- Executive summary
- Confidence score
- ML insights (`top_features`, metric context)
- Impact model (cost avoided, outage-hours reduced, response-time gain)
- External signal cross-check (live weather anomaly)

## 4) Architecture Overview

### Backend

- Framework: FastAPI
- Validation: Pydantic request/response models
- Services:
  - `risk_engine.py` for deterministic scoring and cascade logic
  - `ml_runtime.py` for loading/running trained bundle
  - `external_signals.py` for live weather cross-check
  - `persistence.py` for JSONL storage
  - `ai_reasoning.py` for executive narrative composition

### Frontend

- React + TypeScript + Vite + Tailwind
- Dashboard modules:
  - Executive header
  - Assessment input console + presets
  - Risk gauges
  - System risk matrix
  - Confidence + external signal + outlook
  - Explainability panel
  - District geo risk layer + heat map
  - Cascading paths
  - Projected impact block
  - Model evidence board
  - Executive summary + intervention zones

### Runtime Modes

- `Live API`: calls backend endpoints.
- `Offline fallback`: renders from local mock response for demo resiliency.

## 5) Input Contract and Validation

Main request model: `RiskAssessmentRequest`  
Location: `backend/app/models.py`

Validated fields include:

- Temperature (`-30` to `65`)
- Humidity (`0` to `100`)
- Rainfall (`0` to `600`)
- Grid load and peak (`0` to `100`)
- Historical failure rate (`0` to `100`)
- Transformer age (`0` to `80`)
- Water pressure variance (`0` to `100`)
- Pipeline age (`0` to `100`)
- Repairs in last 30 days (`0` to `50`)
- Congestion index (`0` to `100`)
- Signal failure reports (`0` to `100`)
- Population density (`0` to `150000`)

This range gating prevents invalid numeric payloads and stabilizes inference behavior.

## 6) Risk Engine Logic

The deterministic layer applies threshold-based and weighted scoring per subsystem, then models compound amplification and cascading interactions.

### Threshold examples

- Power stress increases when load/peak are high, heatwave alert is true, transformer age is high, or historical failure rate is elevated.
- Water stress increases with pressure variance, aging pipelines, high rainfall, and frequent recent repairs.
- Traffic stress increases with congestion, signal incident count, and high density.

### Compound amplification

- If 2 high-risk factors intersect within a system: multiplier around `1.10`
- If 3+ factors intersect: multiplier around `1.15`
- Special amplification for `heatwave_alert + high grid load`

### Cascading interactions

- Power instability raises traffic signal failure likelihood.
- Water failures in dense areas raise traffic disruption risk.
- Congestion intensifies emergency response delay.

### Overall risk aggregation

System scores are weighted into an overall city score, then amplified by cascade context, then clipped to `[0, 100]`.

### Core formulas

Subsystem score before amplification:

$$
S_i^{raw} = \mathrm{clip}_{0,100}\left(\sum_k w_{ik}x_k + b_i + \Delta_i\right)
$$

Subsystem score with compound amplification:

$$
S_i = \mathrm{clip}_{0,100}\left(S_i^{raw} \cdot A_i\right)
$$

Where:

- \( A_i = 1.00 \) for no compound intersection
- \( A_i \approx 1.10 \) for 2 concurrent high-risk factors
- \( A_i \approx 1.15 \) for 3 or more concurrent high-risk factors

Overall city risk:

$$
R_{overall}=\mathrm{clip}_{0,100}\left((0.33S_{power}+0.17S_{transformer}+0.25S_{water}+0.25S_{traffic})\cdot C\right)
$$

where \(C\) is the cascading multiplier driven by inter-system stress.

## 7) ML Layer and Hybrid Blending

ML backend: NumPy logistic regression (per target) with engineered infrastructure features and interaction terms.

### Target labels

- `overall_failure_72h_label`
- `power_failure_72h_label`
- `transformer_overload_72h_label`
- `water_failure_72h_label`
- `traffic_failure_72h_label`

### Artifacts

- Model bundle: `backend/models/gridmind_ml_bundle_numpy.json`
- Training report: `backend/models/gridmind_ml_training_report.json`
- Chronological backtest: `backend/models/gridmind_realworld_backtest.json`
- Baseline comparison: `backend/models/gridmind_baseline_comparison.json`

### Blend mechanism

The response combines rule and ML probabilities using learned blend weight:

- `recommended_ml_blend_weight = 0.5929935889167225`
- Holdout macro quality (`macro_pr_auc_holdout`) from bundle: `0.3574839722918062`

Hybrid is used for:

- deterministic safety anchoring
- better operational explainability
- graceful behavior under model drift or sparse classes

Hybrid probability blend:

$$
P_{hybrid}(y_t=1\mid x)=\alpha P_{ML}(y_t=1\mid x)+(1-\alpha)P_{rule}(y_t=1\mid x)
$$

with \(\alpha=0.5929935889167225\) from the trained bundle.

## 8) Model Evaluation Snapshot

### Holdout (from training report)

- Macro PR-AUC: `0.3575`
- Per-target PR-AUC:
  - overall: `0.7106`
  - power: `0.0741`
  - transformer: `0.1848`
  - water: `0.1496`
  - traffic: `0.6684`

### Chronological backtest

- Windows total: `5`
- Macro PR-AUC mean: `0.3831`
- Per-target PR-AUC mean:
  - overall: `0.7391`
  - power: `0.2080`
  - transformer: `0.1946`
  - water: `0.1421`
  - traffic: `0.6320`

### Baseline comparison (sampled holdout evaluation)

- Macro random PR-AUC baseline: `0.2180`
- Rule-only: `0.3693`
- ML-only: `0.3820`
- Hybrid: `0.3822`

All model families are above random baseline, with hybrid currently highest on macro PR-AUC.

## 9) Explainability and Calibration

Each assessment can expose:

- `ml_insights.top_features`:
  - feature name
  - impact share
  - direction (`increases_risk` / `reduces_risk`)
  - signed logit contribution
- `ml_insights.metric_context`:
  - macro PR-AUC holdout
  - random baseline PR-AUC
  - uplift vs baseline

Calibration and reliability outputs are included in model artifacts and rendered via evidence charts in the frontend (`frontend/public/evidence`).

## 10) External Signal Layer

GridMind fetches a live weather anomaly cross-check via `external_signals.py`.  
If anomaly is detected (`status=ok` + anomaly flag), the system applies a conservative uplift to overall risk and appends that context into executive summary.

This reduces blind spots where local telemetry may lag broader environmental stress.

## 11) Persistence and Auditability

Assessment results are persisted to:

- `backend/storage/assessments.jsonl`

Each successful assess request returns:

- `X-Assessment-Id` header

History retrieval endpoint:

- `GET /api/v1/risk/history?limit=20`

## 12) API Endpoints

- `GET /health`
- `POST /api/v1/risk/assess`
- `GET /api/v1/simulate/scenarios?count=10&seed=42`
- `GET /api/v1/risk/history?limit=20`
- `GET /api/v1/ml/status`
- `POST /api/v1/ml/reload`

## 13) How an Assessment Runs (End-to-End)

1. Frontend sends structured telemetry JSON to `/api/v1/risk/assess`.
2. Pydantic validates schema and numeric ranges.
3. Rule engine computes subsystem and overall risk with compound + cascade effects.
4. ML runtime predicts per-target probabilities from trained bundle.
5. Overlay blends rule and ML scores.
6. External signal cross-check applies conditional uplift.
7. Impact model computes operational planning estimates.
8. Final response is persisted and returned to dashboard.

## 14) How to Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Optional:

```bash
set VITE_API_BASE_URL=http://localhost:8000
```

### Verify runtime

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/ml/status
```

## 15) Data Status and Integrity

Current MVP datasets (`real_telemetry.csv`, `real_incidents.csv`, and derived labeled files) are structured simulation aligned to realistic infrastructure distributions for architecture and model validation.

The codebase is intentionally prepared for plug-in with municipal SCADA/IoT/incident feeds without changing API contracts.

## 16) Current Gaps and Next Steps

- Expand real labeled incident coverage across cities and longer time spans.
- Improve weaker classes (especially power/water) with richer external features.
- Add deeper geospatial map operations (district overlays + dependency paths).
- Add continuous monitoring, drift alerts, and scheduled recalibration for production governance.

## 17) Why This Project Is Practical

GridMind does not return only a score. It returns actionable operations output:

- what system is at risk,
- why risk is elevated,
- where intervention should happen first,
- what cascades are likely,
- and what impact can be avoided if teams act early.

That combination makes it useful for real control-room decision workflows.
