import { RiskResponse } from "../types";

interface GeoRiskZoneMapProps {
  systems: RiskResponse["systems"];
  cascades: string[];
}

interface ZoneSpec {
  id: string;
  name: string;
  row: number;
  col: number;
  mix: {
    power: number;
    transformer: number;
    water: number;
    traffic: number;
  };
  bias: number;
  cascadeSensitive: boolean;
}

const ZONES: ZoneSpec[] = [
  {
    id: "north_core",
    name: "North Core",
    row: 0,
    col: 1,
    mix: { power: 0.28, transformer: 0.27, water: 0.18, traffic: 0.27 },
    bias: 2,
    cascadeSensitive: true
  },
  {
    id: "west_industrial",
    name: "West Industrial",
    row: 1,
    col: 0,
    mix: { power: 0.36, transformer: 0.32, water: 0.2, traffic: 0.12 },
    bias: 5,
    cascadeSensitive: false
  },
  {
    id: "central_exchange",
    name: "Central Exchange",
    row: 1,
    col: 1,
    mix: { power: 0.25, transformer: 0.18, water: 0.2, traffic: 0.37 },
    bias: 4,
    cascadeSensitive: true
  },
  {
    id: "east_residential",
    name: "East Residential",
    row: 1,
    col: 2,
    mix: { power: 0.2, transformer: 0.16, water: 0.34, traffic: 0.3 },
    bias: 1,
    cascadeSensitive: true
  },
  {
    id: "south_logistics",
    name: "South Logistics",
    row: 2,
    col: 1,
    mix: { power: 0.29, transformer: 0.21, water: 0.14, traffic: 0.36 },
    bias: 3,
    cascadeSensitive: true
  }
];

const FLOWS = [
  { from: "west_industrial", to: "central_exchange", label: "Grid -> Junction" },
  { from: "north_core", to: "central_exchange", label: "Signal Dependency" },
  { from: "east_residential", to: "south_logistics", label: "Water -> Traffic" }
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function zoneClass(score: number): string {
  if (score <= 35) return "border-emerald-300/35 bg-emerald-500/18 text-emerald-200";
  if (score <= 65) return "border-amber-300/35 bg-amber-500/18 text-amber-100";
  if (score <= 85) return "border-orange-300/40 bg-orange-500/20 text-orange-100";
  return "border-rose-300/45 bg-rose-600/25 text-rose-100";
}

function flowOpacity(cascadeCount: number, sensitive: boolean): number {
  const base = sensitive ? 0.5 : 0.35;
  return Math.min(0.95, base + cascadeCount * 0.07);
}

export function GeoRiskZoneMap({ systems, cascades }: GeoRiskZoneMapProps) {
  const power = systems.power_grid.risk_score;
  const transformer = systems.transformer_overload.risk_score;
  const water = systems.water_pipeline.risk_score;
  const traffic = systems.traffic_infrastructure.risk_score;

  const cascadeCount = cascades.length;
  const zoneScores = ZONES.map((zone) => {
    const weighted =
      zone.mix.power * power +
      zone.mix.transformer * transformer +
      zone.mix.water * water +
      zone.mix.traffic * traffic;
    const cascadeBoost = zone.cascadeSensitive ? Math.min(8, cascadeCount * 1.4) : Math.min(4, cascadeCount);
    const score = clamp01(weighted + zone.bias + cascadeBoost);
    return { ...zone, score };
  });

  const byId = Object.fromEntries(zoneScores.map((z) => [z.id, z])) as Record<string, (typeof zoneScores)[number]>;

  return (
    <section className="panel relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_10%,rgba(215,179,111,0.12),transparent_45%),radial-gradient(circle_at_90%_85%,rgba(111,169,215,0.12),transparent_45%)]" />
      <div className="relative">
        <div className="flex items-center justify-between">
          <h3 className="font-heading text-xl text-gridmind-pearl">District Risk Geo Layer</h3>
          <span className="rounded-full border border-gridmind-gold/30 bg-gridmind-carbon/70 px-3 py-1 text-[11px] uppercase tracking-[0.12em] text-gridmind-gold">
            Simulated Zones
          </span>
        </div>
        <p className="mt-2 text-xs text-gridmind-haze">
          Spatialized district projection using system risk composition and active cascade sensitivity.
        </p>

        <div className="mt-4 grid grid-cols-3 gap-3">
          {Array.from({ length: 9 }).map((_, idx) => {
            const row = Math.floor(idx / 3);
            const col = idx % 3;
            const zone = zoneScores.find((item) => item.row === row && item.col === col);
            if (!zone) {
              return (
                <div
                  key={`empty-${idx}`}
                  className="h-24 rounded-xl border border-gridmind-gold/10 bg-gridmind-midnight/30"
                />
              );
            }
            return (
              <div key={zone.id} className={`h-24 rounded-xl border p-3 ${zoneClass(zone.score)}`}>
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em]">{zone.name}</p>
                <p className="mt-2 font-heading text-2xl">{zone.score.toFixed(1)}</p>
              </div>
            );
          })}
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-3">
          {FLOWS.map((flow) => {
            const from = byId[flow.from];
            const to = byId[flow.to];
            const opacity = flowOpacity(cascadeCount, Boolean(from?.cascadeSensitive || to?.cascadeSensitive));
            return (
              <div
                key={`${flow.from}-${flow.to}`}
                className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2 text-xs text-gridmind-haze"
                style={{ boxShadow: `inset 0 0 0 1px rgba(215,179,111,${opacity * 0.35})` }}
              >
                <p className="font-semibold text-gridmind-pearl">
                  {from?.name ?? flow.from} {"->"} {to?.name ?? flow.to}
                </p>
                <p className="mt-1">{flow.label}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
