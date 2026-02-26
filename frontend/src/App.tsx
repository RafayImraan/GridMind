import { useEffect, useMemo, useState } from "react";
import { defaultRiskRequest, fetchRiskAssessmentWithOptions } from "./api";
import { RiskRequest, RiskResponse } from "./types";
import { ConfidenceIndicator } from "./components/ConfidenceIndicator";
import { DashboardHeader } from "./components/DashboardHeader";
import { ExecutiveSummaryPanel } from "./components/ExecutiveSummaryPanel";
import { OutlookPanel } from "./components/OutlookPanel";
import { RiskGaugeCard } from "./components/RiskGaugeCard";
import { RiskHeatMap } from "./components/RiskHeatMap";
import { SystemRiskTable } from "./components/SystemRiskTable";
import { CascadePanel } from "./components/CascadePanel";
import { AssessmentInputForm } from "./components/AssessmentInputForm";
import { ModelInsightsPanel } from "./components/ModelInsightsPanel";

interface DashboardState {
  loading: boolean;
  submitting: boolean;
  error: string | null;
  data: RiskResponse | null;
}

export default function App() {
  const [state, setState] = useState<DashboardState>({
    loading: true,
    submitting: false,
    error: null,
    data: null
  });

  async function runAssessment(payload: RiskRequest, mode: "initial" | "manual" = "manual"): Promise<void> {
    if (mode === "initial") {
      setState((prev) => ({ ...prev, loading: true, error: null }));
    } else {
      setState((prev) => ({ ...prev, submitting: true, error: null }));
    }

    try {
      const data = await fetchRiskAssessmentWithOptions(payload, {
        allowFallback: mode === "initial"
      });
      setState({
        loading: false,
        submitting: false,
        error: null,
        data
      });
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        submitting: false,
        error: error instanceof Error ? error.message : "unknown_error"
      }));
    }
  }

  useEffect(() => {
    const firstPayload: RiskRequest = {
      ...defaultRiskRequest,
      timestamp: new Date().toISOString()
    };
    void runAssessment(firstPayload, "initial");
  }, []);

  const gauges = useMemo(() => {
    if (!state.data) return [];
    return [
      { label: "Overall Risk", score: state.data.overall_risk.score, level: state.data.overall_risk.level },
      {
        label: "Power Risk",
        score: state.data.systems.power_grid.risk_score,
        level: state.data.systems.power_grid.risk_level
      },
      {
        label: "Water Risk",
        score: state.data.systems.water_pipeline.risk_score,
        level: state.data.systems.water_pipeline.risk_level
      },
      {
        label: "Traffic Risk",
        score: state.data.systems.traffic_infrastructure.risk_score,
        level: state.data.systems.traffic_infrastructure.risk_level
      }
    ];
  }, [state.data]);

  if (state.loading) {
    return (
      <main className="dashboard-shell mx-auto max-w-7xl p-6">
        <div className="panel">
          <p className="font-heading text-lg text-gridmind-pearl">Loading GridMind AI risk telemetry...</p>
        </div>
      </main>
    );
  }

  if (!state.data) {
    return (
      <main className="dashboard-shell mx-auto max-w-7xl p-6">
        <div className="panel">
          <p className="font-heading text-lg text-gridmind-ruby">Dashboard load failed</p>
          <p className="mt-2 text-sm text-gridmind-haze">{state.error ?? "No data returned."}</p>
          <div className="mt-4">
            <AssessmentInputForm submitting={state.submitting} onSubmit={(payload) => runAssessment(payload)} />
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="dashboard-shell mx-auto max-w-7xl p-4 md:p-6">
      <DashboardHeader city={state.data.city} timestamp={state.data.timestamp} level={state.data.overall_risk.level} />
      <section className="mb-4 anim-fade-lift" style={{ animationDelay: "0.05s" }}>
        <AssessmentInputForm submitting={state.submitting} onSubmit={(payload) => runAssessment(payload)} />
      </section>

      {state.error && (
        <section className="mb-4 panel border border-red-300/30 bg-red-900/20">
          <p className="text-sm font-semibold text-red-200">
            Latest assessment request failed: {state.error}. No new record was saved.
          </p>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4 anim-fade-lift" style={{ animationDelay: "0.1s" }}>
        {gauges.map((gauge) => (
          <RiskGaugeCard key={gauge.label} label={gauge.label} score={gauge.score} level={gauge.level} />
        ))}
      </section>

      <section className="mt-4 grid gap-4 xl:grid-cols-3 anim-fade-lift" style={{ animationDelay: "0.15s" }}>
        <div className="xl:col-span-2">
          <SystemRiskTable systems={state.data.systems} />
        </div>
        <div className="space-y-4">
          <ConfidenceIndicator confidence={state.data.confidence_score} />
          <OutlookPanel
            outlook24={state.data.overall_risk["24_hour_outlook"]}
            projection72={state.data.overall_risk["72_hour_projection"]}
          />
          <ModelInsightsPanel
            topFeatures={state.data.ml_insights?.top_features ?? []}
            metricContext={state.data.ml_insights?.metric_context}
          />
        </div>
      </section>

      <section className="mt-4 grid gap-4 xl:grid-cols-2 anim-fade-lift" style={{ animationDelay: "0.2s" }}>
        <RiskHeatMap systems={state.data.systems} />
        <CascadePanel cascades={state.data.cascading_failure_risks} />
      </section>

      <section className="mt-4 anim-fade-lift" style={{ animationDelay: "0.25s" }}>
        <ExecutiveSummaryPanel summary={state.data.executive_summary} zones={state.data.priority_intervention_zones} />
      </section>
    </main>
  );
}
