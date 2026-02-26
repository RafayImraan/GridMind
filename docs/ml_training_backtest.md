# GridMind ML Training + Real-World Backtest

## 1) Objective

Train supervised probability models for:

- `overall_failure_72h_label`
- `power_failure_72h_label`
- `transformer_overload_72h_label`
- `water_failure_72h_label`
- `traffic_failure_72h_label`

Then validate with chronological backtesting for real-world predictive performance.

## 2) Install ML Dependencies

```bash
cd backend
pip install -r requirements-ml.txt
```

Optional advanced backend (sklearn/joblib):

```bash
cd backend
pip install -r requirements-ml-advanced.txt
```

## 3) Labeled Dataset Contract (CSV)

Required columns:

- `timestamp`
- `city`
- `temperature_c`
- `humidity_percent`
- `heatwave_alert`
- `rainfall_mm`
- `current_load_percent`
- `peak_load_percent`
- `historical_failure_rate_percent`
- `transformer_age_years`
- `pressure_variance_percent`
- `pipeline_age_years`
- `recent_repairs_last_30_days`
- `congestion_index`
- `signal_failure_reports_last_7_days`
- `population_density_per_km2`
- `overall_failure_72h_label`
- `power_failure_72h_label`
- `transformer_overload_72h_label`
- `water_failure_72h_label`
- `traffic_failure_72h_label`

Template header file:

- `backend/ml/realworld_labeled_template.csv`

Label rules (recommended):

- Label = `1` if failure event occurred within next 72h for the corresponding target.
- Label = `0` otherwise.
- Use strictly forward-looking labels to avoid leakage.

## 4) Build Labeled Dataset From Real Telemetry + Incidents

If you have separate telemetry and incident logs, build labels automatically:

Telemetry CSV must include:

- `timestamp`, `city`, and all infrastructure/environment feature columns listed above.

Incidents CSV must include:

- `timestamp`
- `city`
- `system` (accepted values: `power`, `transformer`, `water`, `traffic`, or aliases such as `power_grid`, `water_pipeline`, `traffic_infrastructure`)

Command:

```bash
cd backend
python -m ml.build_realworld_labeled_dataset --telemetry C:\path\to\telemetry.csv --incidents C:\path\to\incidents.csv --output C:\path\to\real_world_labeled.csv --horizon-hours 72
```

If you cannot obtain external city data, generate a high-fidelity proxy telemetry + incidents pair locally:

```bash
cd backend
python -m ml.generate_proxy_realworld_data --start-date 2024-01-01T00:00:00Z --days 365 --seed 42 --telemetry-output C:\Users\HomePC\Documents\gridmind\backend\models\proxy_realworld_telemetry.csv --incidents-output C:\Users\HomePC\Documents\gridmind\backend\models\proxy_realworld_incidents.csv
```

Then create labeled training data:

```bash
cd backend
python -m ml.build_realworld_labeled_dataset --telemetry C:\Users\HomePC\Documents\gridmind\backend\models\proxy_realworld_telemetry.csv --incidents C:\Users\HomePC\Documents\gridmind\backend\models\proxy_realworld_incidents.csv --output C:\Users\HomePC\Documents\gridmind\backend\models\proxy_realworld_labeled.csv --horizon-hours 72
```

## 5) Train Models

```bash
cd backend
python -m ml.train --data C:\path\to\real_world_labeled.csv --output-dir C:\Users\HomePC\Documents\gridmind\backend\models --holdout-fraction 0.2 --time-column timestamp --seed 42
```

Outputs:

- `backend/models/gridmind_ml_bundle_numpy.json`
- `backend/models/gridmind_ml_training_report.json`

## 6) Run Chronological Real-World Backtest

```bash
cd backend
python -m ml.backtest --data C:\path\to\real_world_labeled.csv --output C:\Users\HomePC\Documents\gridmind\backend\models\gridmind_realworld_backtest.json --time-column timestamp --min-train-ratio 0.5 --test-ratio 0.1 --step-ratio 0.1 --seed 42
```

Output:

- `backend/models/gridmind_realworld_backtest.json`

Primary metrics to optimize:

- `pr_auc` (class imbalance robust)
- `roc_auc`
- `brier` and `ece_10_bins` (probability calibration)
- `f1` (decision threshold performance)

## 7) Activate ML Inference in API

Default model path:

- `backend/models/gridmind_ml_bundle_numpy.json`

Optional override:

```bash
set GRIDMIND_ML_BUNDLE_PATH=C:\custom\path\gridmind_ml_bundle_numpy.json
```

Check status:

- `GET /api/v1/ml/status`

When a valid bundle is present, `/api/v1/risk/assess` automatically applies ML probability overlay on top of rule-based engine outputs.

## 8) Pipeline Test Dataset (Non-Real)

For dry-run testing only:

```bash
cd backend
python -m ml.generate_demo_labeled_data --count 2000 --seed 42 --output C:\Users\HomePC\Documents\gridmind\backend\models\demo_labeled_dataset.csv
```

Then train and backtest against that file before plugging in municipal/utility labeled data.

## 9) Metric Framing For Imbalanced Failure Targets

Use precision-recall framing first, not accuracy:

- Random PR-AUC baseline is approximately class prevalence.
- GridMind reports `macro_pr_auc_holdout` and also the prevalence-derived random baseline.
- Explain performance as `uplift_vs_baseline = macro_pr_auc_holdout - random_baseline_pr_auc`.

Example from current bundle:

- `macro_pr_auc_holdout ~= 0.357`
- `random_baseline_pr_auc ~= 0.222`
- `uplift_vs_baseline ~= +0.135`

This states predictive signal quality honestly under severe class imbalance and avoids misleading high-accuracy narratives.
