import { RiskResponse } from "../types";

interface ImpactModelPanelProps {
  impact?: RiskResponse["impact_model"];
}

function asMoney(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function ImpactModelPanel({ impact }: ImpactModelPanelProps) {
  if (!impact) return null;

  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">Projected Impact (72h)</h3>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-gridmind-gold/85">Cost Avoided</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">
            {asMoney(impact.estimated_cost_avoided_usd_72h)}
          </p>
        </div>
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-gridmind-gold/85">Outage Hours Reduced</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">
            {impact.estimated_outage_hours_avoided_72h.toFixed(1)} h
          </p>
        </div>
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-gridmind-gold/85">Response-Time Gain</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">
            {impact.estimated_response_time_gain_percent.toFixed(1)}%
          </p>
        </div>
      </div>
      <ul className="mt-3 space-y-1 text-xs text-gridmind-haze">
        {impact.assumptions.map((assumption) => (
          <li key={assumption}>- {assumption}</li>
        ))}
      </ul>
    </section>
  );
}
