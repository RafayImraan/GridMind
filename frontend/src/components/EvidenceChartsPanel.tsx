const CHARTS = [
  {
    title: "Macro PR-AUC Benchmark",
    src: "/evidence/baseline_macro_pr_auc.svg",
    caption: "Random vs rule-only vs ML-only vs hybrid."
  },
  {
    title: "Calibration Error (Mean ECE)",
    src: "/evidence/calibration_mean_ece.svg",
    caption: "Lower is better; compares model families."
  },
  {
    title: "Overall Hybrid Calibration Curve",
    src: "/evidence/calibration_curve_overall_hybrid.svg",
    caption: "Predicted probability vs observed frequency."
  },
  {
    title: "Temporal Stability",
    src: "/evidence/temporal_pr_auc_stability.svg",
    caption: "PR-AUC standard deviation by target."
  }
];

export function EvidenceChartsPanel() {
  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">Model Evidence Board</h3>
      <p className="mt-2 text-xs text-gridmind-haze">
        Visual validation artifacts generated from training/backtest reports.
      </p>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {CHARTS.map((chart) => (
          <article key={chart.src} className="rounded-xl border border-gridmind-gold/18 bg-gridmind-midnight/55 p-2">
            <img
              src={chart.src}
              alt={chart.title}
              className="h-auto w-full rounded-lg border border-gridmind-gold/10 bg-gridmind-obsidian/70"
              loading="lazy"
            />
            <h4 className="mt-2 text-sm font-semibold text-gridmind-pearl">{chart.title}</h4>
            <p className="mt-1 text-xs text-gridmind-haze">{chart.caption}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
