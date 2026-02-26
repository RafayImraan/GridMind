# GridMind AI

Predictive urban infrastructure risk intelligence platform for 0-72 hour failure forecasting across power, transformer, water, and traffic systems.

## Data Provenance

Current `real_telemetry.csv` and `real_incidents.csv` are structured simulation datasets aligned with real-world infrastructure distributions for architecture and modeling validation.

This repository is designed for direct plug-in with municipal SCADA / IoT telemetry and incident feeds without changing API contracts.

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### ML Training Stack

```bash
cd backend
pip install -r requirements-ml.txt
```

Optional advanced ML backend:

```bash
cd backend
pip install -r requirements-ml-advanced.txt
```

### Synthetic Data

```bash
cd data_simulation
python generate_scenarios.py --count 10 --seed 42 --output outputs/city_scenarios_10.json
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set API URL if needed:

```bash
set VITE_API_BASE_URL=http://localhost:8000
```

## API Endpoints

- `GET /health`
- `POST /api/v1/risk/assess`
- `GET /api/v1/simulate/scenarios?count=10&seed=42`
- `GET /api/v1/risk/history?limit=20`
- `GET /api/v1/ml/status`

## Assessment Persistence

Each risk assessment submitted to `POST /api/v1/risk/assess` is saved automatically to:

- `backend/storage/assessments.jsonl`

Each successful assessment response also returns:

- `X-Assessment-Id` header

You can read saved runs from:

- `GET /api/v1/risk/history?limit=20`

## Example Assets

- Backend request example: `backend/example_request.json`
- Backend response example: `backend/example_response.json`
- Generated 10-scenario dataset: `data_simulation/outputs/city_scenarios_10.json`
- ML training report: `backend/models/gridmind_ml_training_report.json`
- ML backtest report: `backend/models/gridmind_realworld_backtest.json`
- Baseline comparison report (rule vs ML vs hybrid): `backend/models/gridmind_baseline_comparison.json`
- Visual artifacts:
  - `docs/assets/baseline_macro_pr_auc.svg`
  - `docs/assets/calibration_mean_ece.svg`
  - `docs/assets/calibration_curve_overall_hybrid.svg`
  - `docs/assets/temporal_pr_auc_stability.svg`
- Proxy telemetry CSV: `backend/models/proxy_realworld_telemetry.csv`
- Proxy incidents CSV: `backend/models/proxy_realworld_incidents.csv`

## Documentation

- System architecture: `docs/architecture.md`
- Full rewritten project report: `docs/project_report.md`
- API contract and schemas: `docs/api_contract.md`
- Risk model formulas and weights: `docs/risk_model.md`
- ML training and backtest workflow: `docs/ml_training_backtest.md`
- Frontend dashboard design: `docs/frontend_dashboard_design.md`
- Demo script and judging strategy: `docs/demo_strategy.md`
- Demo readiness checklist: `docs/demo_readiness.md`
- Winning positioning and roadmap: `docs/winning_positioning.md`

## Artifact Generation

Generate visual evidence SVGs from current metrics:

```bash
cd backend
python -m ml.generate_visual_artifacts --baseline C:\Users\HomePC\Documents\gridmind\backend\models\gridmind_baseline_comparison.json --backtest C:\Users\HomePC\Documents\gridmind\backend\models\gridmind_realworld_backtest.json --output-dir C:\Users\HomePC\Documents\gridmind\docs\assets
```
