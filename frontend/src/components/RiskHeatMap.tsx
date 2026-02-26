import { RiskResponse } from "../types";

interface RiskHeatMapProps {
  systems: RiskResponse["systems"];
}

const labels: Record<keyof RiskResponse["systems"], string> = {
  power_grid: "Power",
  transformer_overload: "Transformer",
  water_pipeline: "Water",
  traffic_infrastructure: "Traffic"
};

function scoreClass(score: number): string {
  if (score <= 35) return "border-emerald-300/40 bg-emerald-500/18 text-emerald-200";
  if (score <= 65) return "border-amber-300/40 bg-amber-500/18 text-amber-200";
  if (score <= 85) return "border-orange-300/40 bg-orange-500/20 text-orange-200";
  return "border-rose-300/40 bg-rose-500/20 text-rose-200";
}

export function RiskHeatMap({ systems }: RiskHeatMapProps) {
  const entries = Object.entries(systems) as [keyof RiskResponse["systems"], RiskResponse["systems"][keyof RiskResponse["systems"]]][];
  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">System Heat Map</h3>
      <div className="mt-4 grid grid-cols-2 gap-3">
        {entries.map(([key, value]) => (
          <div key={key} className={`rounded-xl border p-4 transition ${scoreClass(value.risk_score)}`}>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em]">{labels[key]}</p>
            <p className="mt-2 font-heading text-2xl">{value.risk_score.toFixed(1)}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
