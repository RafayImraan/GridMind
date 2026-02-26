import { RiskResponse } from "../types";

interface SystemRiskTableProps {
  systems: RiskResponse["systems"];
}

const labels: Record<keyof RiskResponse["systems"], string> = {
  power_grid: "Power Grid",
  transformer_overload: "Transformer Overload",
  water_pipeline: "Water Pipeline",
  traffic_infrastructure: "Traffic Infrastructure"
};

function levelClass(level: string): string {
  if (level === "LOW") return "border-emerald-300/30 bg-emerald-500/15 text-emerald-200";
  if (level === "MODERATE") return "border-amber-300/30 bg-amber-500/15 text-amber-200";
  if (level === "HIGH") return "border-orange-300/30 bg-orange-500/15 text-orange-200";
  return "border-rose-300/30 bg-rose-500/15 text-rose-200";
}

export function SystemRiskTable({ systems }: SystemRiskTableProps) {
  const rows = Object.entries(systems) as [keyof RiskResponse["systems"], RiskResponse["systems"][keyof RiskResponse["systems"]]][];

  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">System Risk Matrix</h3>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gridmind-gold/20 text-left text-gridmind-haze">
              <th className="pb-3">System</th>
              <th className="pb-3">Score</th>
              <th className="pb-3">Level</th>
              <th className="pb-3">Primary Drivers</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, value]) => (
              <tr key={key} className="border-b border-gridmind-gold/10 align-top transition hover:bg-gridmind-carbon/25">
                <td className="py-3 font-medium text-gridmind-pearl">{labels[key]}</td>
                <td className="py-3 text-gridmind-pearl">{value.risk_score.toFixed(1)}</td>
                <td className="py-3">
                  <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${levelClass(value.risk_level)}`}>
                    {value.risk_level}
                  </span>
                </td>
                <td className="py-3 text-gridmind-haze">{value.primary_drivers.slice(0, 3).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
