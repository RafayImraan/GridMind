import { RiskLevel } from "../types";

interface RiskGaugeCardProps {
  label: string;
  score: number;
  level: RiskLevel;
}

function levelColor(level: RiskLevel): string {
  if (level === "LOW") return "text-gridmind-emerald";
  if (level === "MODERATE") return "text-gridmind-amber";
  if (level === "HIGH") return "text-orange-300";
  return "text-gridmind-ruby";
}

function ringColor(level: RiskLevel): string {
  if (level === "LOW") return "#3ca67f";
  if (level === "MODERATE") return "#d59750";
  if (level === "HIGH") return "#f1a251";
  return "#d66f61";
}

export function RiskGaugeCard({ label, score, level }: RiskGaugeCardProps) {
  const normalized = Math.max(0, Math.min(100, score));
  const progressAngle = Math.round((normalized / 100) * 360);
  const color = ringColor(level);

  return (
    <article className="panel anim-fade-lift">
      <h3 className="font-heading text-xs uppercase tracking-[0.2em] text-gridmind-gold/80">{label}</h3>
      <div className="mt-4 grid place-items-center">
        <div
          className="relative grid h-36 w-36 place-items-center rounded-full transition"
          style={{
            background: `conic-gradient(${color} ${progressAngle}deg, rgba(143,165,196,0.16) ${progressAngle}deg)`
          }}
        >
          <div className="absolute h-28 w-28 rounded-full bg-gridmind-midnight/95" />
          <div className="relative text-center">
            <p className="font-heading text-3xl text-gridmind-pearl">{score.toFixed(1)}</p>
            <p className={`text-xs font-semibold uppercase tracking-wide ${levelColor(level)}`}>{level}</p>
          </div>
        </div>
      </div>
    </article>
  );
}
