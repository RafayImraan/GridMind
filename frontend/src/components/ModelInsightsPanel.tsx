import { FeatureImpact, MetricContext } from "../types";

interface ModelInsightsPanelProps {
  topFeatures: FeatureImpact[];
  metricContext?: MetricContext | null;
}

function asPercent(value: number | undefined): string | null {
  if (value === undefined || Number.isNaN(value)) return null;
  return `${(value * 100).toFixed(1)}%`;
}

function asSigned(value: number): string {
  const rounded = Math.abs(value).toFixed(3);
  return value >= 0 ? `+${rounded}` : `-${rounded}`;
}

export function ModelInsightsPanel({ topFeatures, metricContext }: ModelInsightsPanelProps) {
  if (topFeatures.length === 0 && !metricContext) {
    return null;
  }

  const macro = asPercent(metricContext?.macro_pr_auc_holdout);
  const baseline = asPercent(metricContext?.random_baseline_pr_auc);
  const uplift = asPercent(metricContext?.uplift_vs_baseline);

  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">Model Explainability</h3>

      {(macro || baseline || uplift) && (
        <div className="mt-3 grid gap-2 text-xs text-gridmind-haze sm:grid-cols-3">
          <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2">
            <p className="uppercase tracking-[0.14em] text-gridmind-gold/80">Macro PR-AUC</p>
            <p className="mt-1 text-sm font-semibold text-gridmind-pearl">{macro ?? "n/a"}</p>
          </div>
          <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2">
            <p className="uppercase tracking-[0.14em] text-gridmind-gold/80">Random Baseline</p>
            <p className="mt-1 text-sm font-semibold text-gridmind-pearl">{baseline ?? "n/a"}</p>
          </div>
          <div className="rounded-lg border border-gridmind-gold/20 bg-gridmind-carbon/55 p-2">
            <p className="uppercase tracking-[0.14em] text-gridmind-gold/80">Uplift</p>
            <p className="mt-1 text-sm font-semibold text-gridmind-pearl">{uplift ?? "n/a"}</p>
          </div>
        </div>
      )}

      <div className="mt-4 space-y-2">
        {topFeatures.map((item) => (
          <div key={`${item.feature}-${item.direction}`} className="rounded-lg border border-gridmind-gold/15 bg-gridmind-midnight/55 p-3">
            <div className="flex items-center justify-between gap-2 text-sm">
              <p className="font-semibold text-gridmind-pearl">{item.feature}</p>
              <p className="text-gridmind-haze">
                {item.direction === "increases_risk" ? "Risk Up" : "Risk Down"} ({asSigned(item.logit_contribution)})
              </p>
            </div>
            <div className="mt-2 h-2 rounded-full bg-gridmind-carbon/80">
              <div
                className={`h-2 rounded-full ${
                  item.direction === "increases_risk"
                    ? "bg-gradient-to-r from-gridmind-ruby/70 to-gridmind-ruby"
                    : "bg-gradient-to-r from-gridmind-emerald/70 to-gridmind-emerald"
                }`}
                style={{ width: `${Math.max(5, Math.min(100, item.impact * 100))}%` }}
              />
            </div>
            <p className="mt-1 text-[11px] uppercase tracking-[0.12em] text-gridmind-gold/75">
              Share of decision: {(item.impact * 100).toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
