# GridMind AI Project Report (Rewritten)

## 1) Project Snapshot

- Project: GridMind AI
- Scope: Near-term (0-72h) predictive risk for urban critical infrastructure
- Systems modeled:
  - Power Grid
  - Transformer Overload
  - Water Pipeline
  - Traffic Signal Infrastructure
- Core delivery:
  - Hybrid rule engine + trained ML overlay
  - Cascading failure modeling
  - Confidence scoring
  - Assessment persistence
  - Enterprise dashboard with explainability

## 2) Problem Statement

City operations teams are typically reactive. They receive outages after failures occur, and cross-system dependencies (power to traffic, water to traffic) are under-modeled. GridMind AI shifts operations from reactive maintenance to predictive resilience by scoring system risk, cascade pathways, and 24h/72h escalation outlook from structured telemetry.

## 3) Solution Overview

GridMind combines deterministic engineering logic with probabilistic ML:

1. Rule-based scoring for transparent safety behavior.
2. Compound stress amplification when high-risk factors intersect.
3. Cascading interaction logic across systems.
4. ML probability overlay for calibration against labeled history.
5. Confidence score to represent data completeness and signal quality.
6. Interpretability output (`ml_insights.top_features`) per assessment.

## 4) System Architecture

Data flow:

1. Client submits telemetry JSON to `POST /api/v1/risk/assess`.
2. Pydantic validation checks schema and numeric ranges.
3. Risk engine computes system scores, compound flags, cascade impacts, 24h and 72h outlook.
4. ML runtime loads trained bundle and predicts per-target 72h probabilities.
5. Rule scores are blended with ML probabilities using learned blend weight.
6. Explainability block is generated with top feature impacts and PR-AUC context.
7. Result is persisted to `backend/storage/assessments.jsonl`.
8. Response is returned to dashboard and rendered in gauges, tables, cascades, summary, and explainability panel.

## 5) Backend and API

Framework: FastAPI

Endpoints:

- `GET /health`
- `POST /api/v1/risk/assess`
- `GET /api/v1/risk/history?limit=20`
- `GET /api/v1/simulate/scenarios?count=10&seed=42`
- `GET /api/v1/ml/status`
- `POST /api/v1/ml/reload`

Assessment persistence:

- Storage file: `backend/storage/assessments.jsonl`
- Header returned on successful assessment: `X-Assessment-Id`

## 6) Risk Model Logic (Hybrid)

System scoring structure:

- `score_system = clamp(0, 100, base_weighted + threshold_bonuses)`
- If 2+ high-risk factors intersect:
  - multiply by `1.10` (2 factors) or `1.15` (3+ factors)
- Special amplification:
  - heatwave + high load adds extra power amplification

Cascading logic examples:

- Power high risk increases traffic risk.
- Water high risk in high-density areas increases traffic disruption risk.
- Extreme congestion increases emergency response delay risk.
- Dense + aging infrastructure increases cascade multiplier.

Overall risk:

- `overall = clamp(0, 100, (0.33*power + 0.17*transformer + 0.25*water + 0.25*traffic) * cascade_multiplier)`

Projection:

- 24h outlook: operational posture text from current ranking and severity.
- 72h projection: trend delta from heatwave, load, rainfall, congestion, repair/age intersection.

Confidence:

- Starts high, then reduced for missing fields, weak history signal, and low trend observability.
- Adjusted by ML quality when ML overlay is active.

## 7) ML Pipeline and Training

Training artifacts:

- Model bundle: `backend/models/gridmind_ml_bundle_numpy.json`
- Holdout report: `backend/models/gridmind_ml_training_report.json`
- Chronological backtest: `backend/models/gridmind_realworld_backtest.json`

Dataset:

- Labeled rows: `52,560`
- Targets:
  - `overall_failure_72h_label`
  - `power_failure_72h_label`
  - `transformer_overload_72h_label`
  - `water_failure_72h_label`
  - `traffic_failure_72h_label`

Model backend:

- NumPy logistic regression per target with engineered infrastructure features
- Learned blend weight for online overlay: `0.5929935889167225`

## 8) Evaluation Results

### Holdout (Training Report)

- Macro PR-AUC: `0.3574839722918062`

Per-target PR-AUC:

- Overall failure: `0.7106`
- Power failure: `0.0741`
- Transformer overload: `0.1848`
- Water failure: `0.1496`
- Traffic failure: `0.6684`

### Chronological Backtest (Real-World Labeled CSV)

- Windows evaluated: `2`
- Macro PR-AUC mean: `0.3625957396357277`

Per-target PR-AUC mean:

- Overall failure: `0.7169`
- Power failure: `0.1616`
- Transformer overload: `0.1658`
- Water failure: `0.1463`
- Traffic failure: `0.6224`

### Metric Framing for Imbalance

- Random PR-AUC baseline approximates class prevalence.
- Runtime metric context now exposes:
  - `macro_pr_auc_holdout`
  - `random_baseline_pr_auc`
  - `uplift_vs_baseline`
- Current uplift vs baseline is positive (about `+0.136`), indicating non-random predictive signal under imbalanced classes.

## 9) Explainability and Trust

Each assessment can now include:

- `ml_insights.top_features`:
  - feature name
  - normalized impact share
  - direction (`increases_risk` / `reduces_risk`)
  - signed logit contribution
- `ml_insights.metric_context`:
  - macro PR-AUC
  - random baseline PR-AUC
  - uplift vs baseline

This is rendered in the dashboard “Model Explainability” panel for judge and operator transparency.

## 10) Frontend Product Layer

Frontend stack:

- React + TypeScript + Tailwind

Dashboard modules:

- Control-room header and overall posture
- Custom assessment input form (manual data entry + presets)
- Risk gauges (overall/system)
- System risk table (drivers + compound flags)
- Cascading failure panel
- Outlook panel (24h/72h)
- Executive summary + priority intervention zones
- Confidence indicator
- Model explainability panel

UI direction:

- Premium enterprise visual system with luxury dark-gold palette, layered gradients, and animated panel reveals.

## 11) Operational Status

Current implementation supports:

- Manual input assessment
- Saved assessment history
- ML-enabled inference when bundle is present
- Synthetic scenario generation for demos
- Real telemetry + incidents CSV path for training/backtesting

## 12) Known Gaps and Risks

- Data bottleneck remains the primary ceiling:
  - power and water incident labels are still sparse/noisy relative to overall and traffic
- Backtest window count is small (`2`) in current file; more temporal windows are needed for stronger stability claims
- Geospatial district map and calibrated reliability plot are not yet integrated into the dashboard

## 13) Next Execution Plan

1. Expand labeled history across multiple cities and longer timelines.
2. Increase backtest windows and publish confidence intervals per metric.
3. Add map-based district risk overlay and cascade arrows.
4. Add probability calibration chart (bin reliability curve + ECE visualization).
5. Integrate real-time ingestion connectors (SCADA/IoT/event bus) for live operations mode.

## 14) Positioning Statement

GridMind AI is a deployable municipal climate-resilience digital twin that predicts cross-system infrastructure failures before they cascade, combining deterministic safety logic, probabilistic ML calibration, interpretable outputs, and production-ready API persistence.
