# GridMind AI Project Report (Judge-Ready Revision)

## 1) Executive Snapshot

- Project: GridMind AI
- Objective: predict 0-72 hour urban infrastructure failure risk
- Systems covered: power grid, transformer overload, water pipeline, traffic infrastructure
- Core stack: deterministic risk engine + trained ML overlay + explainability + persistence + enterprise dashboard

## 2) Data Provenance and Authenticity

Current `real_telemetry.csv` and `real_incidents.csv` are structured simulation datasets aligned to realistic infrastructure distributions. They are used to validate architecture, risk logic, and model pipeline behavior.

GridMind is intentionally designed for direct replacement with municipal feeds:

- SCADA/utility telemetry
- IoT sensor streams
- incident ticket/event systems

This framing is explicit to avoid false claims of live municipal deployment.

## 3) Architecture and Runtime Flow

1. Client submits telemetry to `POST /api/v1/risk/assess`.
2. Pydantic validates nested fields and ranges.
3. Rule engine computes system scores, compound flags, cascading risks, and 24h/72h outlook.
4. ML runtime predicts per-target probabilities from trained bundle.
5. Rule and ML probabilities are blended into final risk scores.
6. Live external weather signal (`open-meteo`) cross-check is fetched and attached.
7. `ml_insights` returns top feature impacts and PR-AUC context.
8. Impact model returns projected cost avoided, outage-hours reduced, and response-time gain.
9. Request/response is persisted to `backend/storage/assessments.jsonl`.

## 4) Risk Engine Transparency

Scoring core:

- `score_system = clamp(0,100, base_weighted + threshold_bonuses)`
- compound multiplier:
  - `1.10` for 2 concurrent high-risk factors
  - `1.15` for 3+ concurrent high-risk factors
- cascading multipliers increase citywide risk under inter-system stress

Heuristic justification:

- Cascade multipliers are conservative engineering heuristics pending empirical calibration on production telemetry.

## 5) ML Pipeline Summary

Artifacts:

- `backend/models/gridmind_ml_bundle_numpy.json`
- `backend/models/gridmind_ml_training_report.json`
- `backend/models/gridmind_realworld_backtest.json`
- `backend/models/gridmind_baseline_comparison.json`
- `docs/assets/*.svg` evidence charts (benchmark, calibration, stability)

Modeling:

- NumPy logistic regression per target
- engineered interaction features for load, aging assets, density, rainfall, and congestion
- blend weight: `0.5929935889167225`

Blend-weight method:

- `recommended_ml_blend_weight` is derived from holdout macro quality and stored in bundle global metrics.

## 6) Validation Results

### Holdout Metrics

- Macro PR-AUC: `0.3575`
- Per-target PR-AUC:
  - overall `0.7106`
  - power `0.0741`
  - transformer `0.1848`
  - water `0.1496`
  - traffic `0.6684`

### Chronological Backtest (Updated)

Configuration:

- `min_train_ratio=0.5`, `test_ratio=0.1`, `step_ratio=0.1`
- windows total: `5`

Macro PR-AUC mean:

- `0.3831`

Per-target PR-AUC mean:

- overall `0.7391`
- power `0.2080`
- transformer `0.1946`
- water `0.1421`
- traffic `0.6320`

Temporal stability indicator:

- Mean per-target PR-AUC standard deviation across windows: `0.0621` (moderate temporal variance)

## 7) Baseline Benchmark (Rule vs ML vs Hybrid)

From `gridmind_baseline_comparison.json`:

- macro random PR-AUC baseline: `0.2180`
- macro PR-AUC:
  - rule-only `0.3693`
  - ML-only `0.3820`
  - hybrid `0.3822`
- macro uplift vs random:
  - rule-only `+0.1513`
  - ML-only `+0.1640`
  - hybrid `+0.1643`

Interpretation:

- All approaches are above random baseline.
- Hybrid currently yields the highest macro PR-AUC.
- Hybrid is retained not solely for marginal PR-AUC uplift, but for deterministic safety constraints, graceful degradation under model drift, and municipal explainability requirements.
- Power and water remain the hardest targets.

## 8) Class Imbalance Framing

Class prevalence (evaluation holdout):

- overall `0.4608`
- power `0.0542`
- transformer `0.1469`
- water `0.0970`
- traffic `0.3310`

Because failure targets are imbalanced, PR-AUC and uplift-vs-baseline are primary metrics; plain accuracy is secondary.

Power-target caveat:

- Power failure prediction remains the weakest class (holdout PR-AUC `0.0741`, backtest mean `0.2080`) due to extreme imbalance and weaker separability in current simulated telemetry; this class is prioritized for feature expansion and external telemetry enrichment.

## 9) Calibration and Reliability

Calibration is quantified:

- `ece_10_bins` is reported in training/backtest/baseline artifacts.
- reliability bin data (10 bins) is exported per target and model in `gridmind_baseline_comparison.json`.

Current risk:

- Power and water calibration error remains materially higher than overall and traffic classes.
- Baseline comparison mean ECE across targets:
  - rule-only `0.3290`
  - hybrid `0.2652`
  - ML-only `0.2253`

## 10) Explainability

Each assessment can include:

- `ml_insights.top_features`:
  - feature
  - impact share
  - direction
  - signed logit contribution
- `ml_insights.metric_context`:
  - macro PR-AUC
  - random baseline PR-AUC
  - uplift vs baseline

Dashboard includes a dedicated Model Explainability panel.

## 10.1) External Signal + Impact Model

Each assessment now also includes:

- `external_signal`:
  - source (`open-meteo.com`)
  - fetch status
  - weather anomaly flag and cross-check deltas
- `impact_model`:
  - estimated cost avoided over 72h
  - estimated outage-hours avoided over 72h
  - estimated emergency response-time gain
  - explicit assumptions list

## 11) Known Gaps (No Overclaim)

- Dataset is high-fidelity simulation, not live municipal telemetry.
- Power and water labels remain sparse/noisy vs overall and traffic.
- Geospatial dashboard district layer is now integrated with simulated zones; production GIS integration remains pending.
- Production deployment requires live connectors, governance, and continuous recalibration.
- Stress testing has been run on synthetic heatwave, high-density, and compound-failure scenarios; real-event stress validation remains pending external telemetry ingestion.

## 12) Geospatial Gap and Plan

Current output includes a simulated district geospatial layer in the dashboard. Remaining gap is production GIS integration.

Planned upgrade:

- district risk heatmap
- cascade arrows between dependent zones
- map-linked intervention prioritization

## 13) Runtime and Evaluation Efficiency

Baseline comparison script was optimized for hackathon speed:

- tuple-based iteration
- no per-row deep-copy overlay
- direct hybrid blend math
- fast default sampling (`stride=15`, `max_eval_rows=1200`)

Observed runtime reduced from ~20+ minutes to ~92 seconds on current dataset.

## 14) Positioning Statement

GridMind AI is a climate-resilience smart-city intelligence layer and digital-twin control surface that combines deterministic safety logic, probabilistic forecasting, calibration-aware confidence, and operator-grade explainability for pre-failure municipal action.

Production hardening plan includes model monitoring, drift detection, and scheduled recalibration cycles before municipal live operations.

## 15) Visual Evidence Artifacts

Generated judge-facing visuals:

- `docs/assets/baseline_macro_pr_auc.svg`
- `docs/assets/calibration_mean_ece.svg`
- `docs/assets/calibration_curve_overall_hybrid.svg`
- `docs/assets/temporal_pr_auc_stability.svg`

These provide direct visual evidence for baseline comparison, calibration behavior, and temporal stability.

Dashboard now renders these charts directly in the Model Evidence Board panel (not docs-only).

## 16) Risk-Control Matrix

Residual risk cannot be reduced to absolute zero in a hackathon MVP. Current controls are:

- Data authenticity risk:
  - Control: explicit simulation provenance statements + SCADA/IoT plug-in pathway.
- Model quality risk (power/water):
  - Control: PR-AUC baseline framing, per-target caveats, prioritized enrichment backlog.
- Calibration risk:
  - Control: ECE reporting + reliability-bin artifacts + recalibration plan.
- Operational drift risk:
  - Control: deterministic rules as fallback + planned drift monitoring and scheduled retraining.
- Deployment overclaim risk:
  - Control: explicit non-production declaration and staged hardening path.

## 17) Demo Reliability Plan

- Live URL path:
  - frontend + backend deploy endpoints
- Fallback path:
  - offline mode toggle in dashboard
- Backup path:
  - 90-second pre-recorded walkthrough

Reference:

- `docs/demo_readiness.md`
