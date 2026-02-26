interface ConfidenceIndicatorProps {
  confidence: number;
}

function confidenceColor(confidence: number): string {
  if (confidence >= 85) return "from-emerald-400 to-emerald-600";
  if (confidence >= 65) return "from-amber-300 to-amber-600";
  return "from-rose-300 to-rose-600";
}

export function ConfidenceIndicator({ confidence }: ConfidenceIndicatorProps) {
  return (
    <section className="panel">
      <h3 className="font-heading text-xl text-gridmind-pearl">Model Confidence</h3>
      <div className="mt-3 h-3 w-full rounded-full bg-gridmind-carbon/80">
        <div
          className={`h-3 rounded-full bg-gradient-to-r transition-all duration-700 ${confidenceColor(confidence)}`}
          style={{ width: `${Math.max(0, Math.min(100, confidence))}%` }}
        />
      </div>
      <p className="mt-2 text-sm font-semibold text-gridmind-pearl">{confidence}/100</p>
    </section>
  );
}
