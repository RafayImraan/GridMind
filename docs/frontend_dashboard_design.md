# GridMind Dashboard Design Spec

## 1) Component Breakdown

- `App`
  - Owns data fetch lifecycle and global dashboard state.
- `DashboardHeader`
  - City context, timestamp snapshot, overall risk badge.
- `RiskGaugeCard`
  - Semi-circular score gauge for overall/power/water/traffic.
- `SystemRiskTable`
  - Tabular subsystem drilldown with score/level/drivers.
- `RiskHeatMap`
  - 2x2 stress intensity blocks for quick scanning.
- `CascadePanel`
  - Ordered cascade pathway list.
- `OutlookPanel`
  - 24h operational posture + 72h projected trajectory.
- `ConfidenceIndicator`
  - Progress bar with confidence score.
- `ExecutiveSummaryPanel`
  - Narrative briefing + intervention zones.

## 2) State Structure

```ts
interface DashboardState {
  loading: boolean;
  error: string | null;
  data: RiskResponse | null;
}
```

`RiskResponse` mirrors backend response schema exactly.

## 3) UI Hierarchy

1. Header strip
2. KPI gauge row (4 cards)
3. Risk matrix + confidence/outlook right rail
4. Heat map + cascade panel
5. Executive summary + priority zones

## 4) Dashboard Layout Grid

- Root container: `max-w-7xl`
- Gauges: `grid gap-4 md:grid-cols-2 xl:grid-cols-4`
- Mid zone: `grid gap-4 xl:grid-cols-3`
  - Left `xl:col-span-2`: system matrix
  - Right: confidence + outlook stacked
- Lower zone: `grid gap-4 xl:grid-cols-2`
- Final zone: full-width executive panel

## 5) Tailwind Style System

- Typography:
  - Headings: `Space Grotesk`
  - Body: `IBM Plex Sans`
- Color direction:
  - Core: `gridmind-navy`, `gridmind-ink`, `gridmind-sky`
  - Risk accents: `mint`, `amber`, `coral`
- Panel styling:
  - Shared class `.panel`:
    - rounded corners
    - translucent white background
    - subtle blur and shadow
- Background:
  - radial gradient with light blue-green climate-resilience palette.

## 6) Responsive Behavior

- Mobile:
  - Single-column stacking for all modules.
  - Gauges become vertical cards.
- Tablet:
  - 2-column gauge row.
  - Matrix plus sidecards remain stacked if width constrained.
- Desktop:
  - Full 4-gauge row and split multi-panel grid.

## 7) Data Contract and Fallback

- Primary mode:
  - Fetch from `POST /api/v1/risk/assess`.
- Demo fallback:
  - If API fails, load deterministic `mockRiskResponse`.
- Env variable:
  - `VITE_API_BASE_URL` points frontend to deployed backend.

