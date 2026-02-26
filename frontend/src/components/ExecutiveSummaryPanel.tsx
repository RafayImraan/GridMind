interface ExecutiveSummaryPanelProps {
  summary: string;
  zones: string[];
}

export function ExecutiveSummaryPanel({ summary, zones }: ExecutiveSummaryPanelProps) {
  return (
    <section className="panel">
      <h3 className="font-heading text-2xl text-gridmind-pearl">Executive Summary</h3>
      <p className="mt-3 rounded-xl border border-gridmind-gold/20 bg-gridmind-carbon/45 p-4 text-sm leading-6 text-gridmind-haze">
        {summary}
      </p>
      <div className="mt-5">
        <h4 className="text-xs font-semibold uppercase tracking-[0.18em] text-gridmind-gold/85">
          Priority Intervention Zones
        </h4>
        <ul className="mt-2 space-y-2 text-sm text-gridmind-pearl">
          {zones.map((zone) => (
            <li key={zone} className="rounded-lg border border-gridmind-gold/15 bg-gridmind-midnight/60 px-3 py-2">
              {zone}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
