# GridMind AI Architecture (MVP: 48-72h)

## 1) High-Level Architecture

```
[Synthetic / Real Data Sources]
         |
         v
[Data Ingestion API - FastAPI]
         |
         v
[Risk Engine]
  - Rule Thresholds
  - Weighted Scoring
  - Compound Amplification
  - Cascading Modeling
  - 24h / 72h Projection
  - Confidence Scoring
         |
         v
[AI Reasoning Layer]
  - Prompt Builder
  - External LLM Adapter (optional)
  - Deterministic Executive Summary Fallback
         |
         +----------------------------+
         |                            |
         v                            v
[REST JSON API]                [Dataset Generator]
         |                            |
         v                            v
[React + Tailwind Dashboard]   [Scenario Files for Demo]
```

## 2) Tech Stack Decisions

- Backend API: FastAPI
  - Justification: Fast iteration speed, OpenAPI support, strong validation via Pydantic.
- Modeling Layer: Pure Python deterministic engine
  - Justification: Explainable scoring for judges, low latency, no dependency on remote inference.
- AI Reasoning Layer: Prompt builder + optional external endpoint + deterministic fallback
  - Justification: Works offline during hackathon, still demonstrates extensible AI integration.
- Data Simulation: Python random scenario engine with deterministic seed
  - Justification: Reproducible demos and stress-test datasets.
- Frontend: React + TypeScript + Tailwind
  - Justification: Fast dashboard development, clean component modularity.
- Deployment: Render/Fly.io (API) + Vercel/Netlify (frontend)
  - Justification: minimal ops overhead in hackathon timebox.

## 3) Backend Services

- `POST /api/v1/risk/assess`
  - Validate payload, compute risks, return executive JSON.
- `GET /api/v1/simulate/scenarios?count=&seed=`
  - Generate deterministic synthetic scenarios for demo/testing.
- `GET /health`
  - Liveness endpoint.

## 4) AI Reasoning Layer

- File: `backend/app/ai_reasoning.py`
- Components:
  - `build_reasoning_prompt(...)`: structured prompt generation.
  - `_external_reasoning(...)`: optional HTTP call to external LLM endpoint when `GRIDMIND_REASONING_MODE=external`.
  - `generate_executive_summary(...)`: fallback deterministic executive summary when external reasoning is unavailable.

## 5) Data Simulation Layer

- File: `backend/app/simulation.py`
- Scenario modes:
  - `baseline`
  - `heatwave`
  - `high_density`
  - `failure_event`
  - `compound` (stacked stressors)
- CLI dataset builder:
  - File: `data_simulation/generate_scenarios.py`
  - Output: `data_simulation/outputs/city_scenarios_10.json`

## 6) Frontend Dashboard Modules

- Header: city snapshot and current posture.
- Gauges: overall + key subsystem risk.
- System risk matrix: scores, levels, primary drivers.
- Heat map: compressed system stress view.
- Cascading paths: dependency chain visibility.
- Outlook panel: 24h and 72h projections.
- Executive panel: command-level summary and intervention zones.
- Confidence indicator: trust level of model output.

## 7) Deployment Plan

1. Backend deploy:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Working dir: `backend`
2. Frontend deploy:
   - Build command: `npm install && npm run build`
   - Publish dir: `frontend/dist`
   - Env: `VITE_API_BASE_URL=<backend-url>`
3. Demo mode fallback:
   - Frontend includes mock fallback if API unreachable.

## 8) Folder Structure

```
gridmind/
  backend/
    app/
      __init__.py
      ai_reasoning.py
      main.py
      ml_runtime.py
      models.py
      persistence.py
      risk_engine.py
      simulation.py
    ml/
      __init__.py
      backtest.py
      build_realworld_labeled_dataset.py
      features.py
      generate_demo_labeled_data.py
      modeling.py
      numpy_modeling.py
      realworld_labeled_template.csv
      train.py
    models/
      demo_labeled_dataset.csv
      gridmind_ml_bundle_numpy.json
      gridmind_ml_training_report.json
      gridmind_realworld_backtest.json
    requirements-ml.txt
    requirements-ml-advanced.txt
    example_request.json
    example_response.json
    requirements.txt
  data_simulation/
    generate_scenarios.py
    outputs/
      city_scenarios_10.json
  frontend/
    package.json
    tailwind.config.js
    postcss.config.js
    tsconfig.json
    vite.config.ts
    index.html
    src/
      App.tsx
      api.ts
      index.css
      main.tsx
      mockData.ts
      types.ts
      components/
        CascadePanel.tsx
        ConfidenceIndicator.tsx
        DashboardHeader.tsx
        ExecutiveSummaryPanel.tsx
        OutlookPanel.tsx
        RiskGaugeCard.tsx
        RiskHeatMap.tsx
        SystemRiskTable.tsx
  docs/
    architecture.md
    api_contract.md
    ml_training_backtest.md
    frontend_dashboard_design.md
    risk_model.md
    demo_strategy.md
    winning_positioning.md
```

## 9) API Endpoint List

- `GET /health`
- `POST /api/v1/risk/assess`
- `GET /api/v1/simulate/scenarios?count=10&seed=42`
- `GET /api/v1/risk/history?limit=20`
- `GET /api/v1/ml/status`

## 10) Data Flow

1. Input payload arrives at `/api/v1/risk/assess`.
2. Pydantic models validate all nested fields and ranges.
3. Risk engine computes:
   - base weighted scores
   - threshold bonuses
   - compound amplification
   - cascading interactions
   - 24h/72h outlook
   - confidence score
4. AI reasoning layer generates executive summary (external or deterministic fallback).
5. Assessment request/response pair is persisted to `backend/storage/assessments.jsonl`.
6. Unified JSON response returns to frontend.
7. Frontend renders gauges, matrix, cascades, outlook, and intervention zones.
