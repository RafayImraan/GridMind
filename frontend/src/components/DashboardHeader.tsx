import { RiskLevel } from "../types";

interface DashboardHeaderProps {
  city: string;
  timestamp: string;
  level: RiskLevel;
}

function levelClass(level: RiskLevel): string {
  if (level === "LOW") return "border-emerald-300/40 bg-emerald-500/20 text-emerald-200";
  if (level === "MODERATE") return "border-amber-300/40 bg-amber-500/20 text-amber-200";
  if (level === "HIGH") return "border-orange-300/40 bg-orange-500/20 text-orange-200";
  return "border-rose-300/40 bg-rose-500/20 text-rose-200";
}

export function DashboardHeader({ city, timestamp, level }: DashboardHeaderProps) {
  return (
    <header className="mb-6 rounded-2xl border border-gridmind-gold/25 bg-gradient-to-r from-gridmind-midnight/90 via-gridmind-carbon/80 to-gridmind-midnight/90 px-5 py-5 shadow-panel md:px-7">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
      <div>
        <p className="font-heading text-[11px] uppercase tracking-[0.22em] text-gridmind-gold">
          GridMind AI Executive Command Layer
        </p>
        <h1 className="mt-1 font-heading text-3xl text-gridmind-pearl md:text-4xl">
          {city} Urban Infrastructure Risk Dashboard
        </h1>
        <p className="mt-2 text-sm text-gridmind-haze">Snapshot: {new Date(timestamp).toLocaleString()}</p>
      </div>
      <div className="flex items-center gap-2">
        <span className="gold-chip inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider">
          72 Hour Horizon
        </span>
        <span className={`inline-flex w-fit rounded-full border px-4 py-2 text-sm font-semibold ${levelClass(level)}`}>
          Overall {level}
        </span>
      </div>
      </div>
    </header>
  );
}
