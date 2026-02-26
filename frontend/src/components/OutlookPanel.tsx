interface OutlookPanelProps {
  outlook24: string;
  projection72: string;
}

export function OutlookPanel({ outlook24, projection72 }: OutlookPanelProps) {
  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">72-Hour Risk Outlook</h3>
      <div className="mt-3 space-y-3 text-sm text-gridmind-haze">
        <div className="rounded-xl border border-gridmind-gold/20 bg-gridmind-carbon/45 px-3 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-gridmind-gold/80">24 Hour</p>
          <p className="mt-1">{outlook24}</p>
        </div>
        <div className="rounded-xl border border-gridmind-gold/20 bg-gridmind-carbon/45 px-3 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-gridmind-gold/80">72 Hour</p>
          <p className="mt-1">{projection72}</p>
        </div>
      </div>
    </section>
  );
}
