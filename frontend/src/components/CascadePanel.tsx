import { ArrowRight } from "lucide-react";

interface CascadePanelProps {
  cascades: string[];
}

export function CascadePanel({ cascades }: CascadePanelProps) {
  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">Cascading Failure Paths</h3>
      <div className="mt-4 space-y-3">
        {cascades.map((cascade) => (
          <div
            key={cascade}
            className="flex items-center gap-3 rounded-xl border border-gridmind-gold/20 bg-gridmind-carbon/45 px-3 py-2 text-sm text-gridmind-haze"
          >
            <ArrowRight size={16} className="text-gridmind-gold" />
            <span>{cascade}</span>
          </div>
        ))}
        {cascades.length === 0 && <p className="text-sm text-gridmind-haze">No active cascading pathways detected.</p>}
      </div>
    </section>
  );
}
