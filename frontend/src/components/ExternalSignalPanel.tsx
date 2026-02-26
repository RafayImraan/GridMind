import { RiskResponse } from "../types";

interface ExternalSignalPanelProps {
  signal?: RiskResponse["external_signal"];
}

function statusClass(status: string): string {
  if (status === "ok") return "border-emerald-300/35 bg-emerald-500/18 text-emerald-200";
  if (status === "unsupported_city") return "border-amber-300/35 bg-amber-500/18 text-amber-100";
  return "border-rose-300/35 bg-rose-500/18 text-rose-100";
}

export function ExternalSignalPanel({ signal }: ExternalSignalPanelProps) {
  if (!signal) return null;

  const tempDelta = signal.details.temperature_delta_c;
  const anomaly = Boolean(signal.details.weather_anomaly_flag);

  return (
    <section className="panel">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-heading text-xl text-gridmind-pearl">Live External Signal</h3>
          <p className="mt-1 text-xs text-gridmind-haze">
            {signal.source} • {signal.signal_type}
          </p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] ${statusClass(signal.status)}`}>
          {signal.status}
        </span>
      </div>
      <p className="mt-3 text-sm text-gridmind-haze">{signal.influence_note ?? "No external note available."}</p>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2 text-xs text-gridmind-haze">
          <p className="uppercase tracking-[0.12em] text-gridmind-gold/85">Temp Delta</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">
            {typeof tempDelta === "number" ? `${tempDelta.toFixed(1)} C` : "n/a"}
          </p>
        </div>
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2 text-xs text-gridmind-haze">
          <p className="uppercase tracking-[0.12em] text-gridmind-gold/85">Wind (km/h)</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">
            {typeof signal.details.external_wind_speed_kmh === "number"
              ? `${signal.details.external_wind_speed_kmh.toFixed(1)}`
              : "n/a"}
          </p>
        </div>
        <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2 text-xs text-gridmind-haze">
          <p className="uppercase tracking-[0.12em] text-gridmind-gold/85">Anomaly</p>
          <p className="mt-1 text-sm font-semibold text-gridmind-pearl">{anomaly ? "Yes" : "No"}</p>
        </div>
      </div>
    </section>
  );
}
