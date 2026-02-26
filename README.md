# GridMind AI

GridMind AI is a 0-72 hour urban infrastructure risk intelligence platform that predicts cascading failure risk across power, transformer, water, and traffic systems.

For a full technical and product explanation, see [`WRITEUP.md`](WRITEUP.md).

## Core Capabilities

- Hybrid risk engine: deterministic rules + ML probability overlay.
- Compound amplification and cascading failure modeling.
- 24-hour outlook and 72-hour projection.
- Explainability (`top_features`, baseline uplift context).
- Confidence scoring.
- External live weather anomaly cross-check.
- Assessment persistence with history API.
- Live API mode and offline fallback mode in dashboard.

## Quick Start

### 1) Backend API

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

Optional API base URL override:

```bash
set VITE_API_BASE_URL=http://localhost:8000
```

### 3) Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/ml/status
```

## API Endpoints

- `GET /health`
- `POST /api/v1/risk/assess`
- `GET /api/v1/risk/history?limit=20`
- `GET /api/v1/simulate/scenarios?count=10&seed=42`
- `GET /api/v1/ml/status`
- `POST /api/v1/ml/reload`

## Persistence

- Assessment storage file: `backend/storage/assessments.jsonl`
- Successful assessment responses include `X-Assessment-Id` response header.

## ML + Evaluation Commands

Install ML stack:

```bash
cd backend
pip install -r requirements-ml.txt
```

Train models from labeled dataset:

```bash
cd backend
python -m ml.train --data models/real_world_labeled.csv --output-dir models
```

Run chronological backtest:

```bash
cd backend
python -m ml.backtest --data models/real_world_labeled.csv --output models/gridmind_realworld_backtest.json
```

Run rule vs ML vs hybrid baseline comparison:

```bash
cd backend
python -m ml.evaluate_baselines --data models/real_world_labeled.csv --output models/gridmind_baseline_comparison.json
```

Generate dashboard evidence SVGs:

```bash
cd backend
python -m ml.generate_visual_artifacts --baseline models/gridmind_baseline_comparison.json --backtest models/gridmind_realworld_backtest.json --output-dir ..\frontend\public\evidence
```

## Example Files

- Request payload: `backend/example_request.json`
- Response payload: `backend/example_response.json`
- Trained bundle: `backend/models/gridmind_ml_bundle_numpy.json`
- Training report: `backend/models/gridmind_ml_training_report.json`
- Backtest report: `backend/models/gridmind_realworld_backtest.json`
- Baseline comparison report: `backend/models/gridmind_baseline_comparison.json`
- Evidence charts (frontend): `frontend/public/evidence/`

## Data Provenance

Current `real_telemetry.csv` and `real_incidents.csv` in this MVP are structured simulation datasets aligned to realistic infrastructure distributions, used to validate architecture and modeling behavior.

The system is intentionally designed for direct plug-in with municipal SCADA, IoT, and incident/event feeds without changing API contracts.
