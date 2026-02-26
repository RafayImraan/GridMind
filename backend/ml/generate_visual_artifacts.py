from __future__ import annotations

import argparse
import json
from pathlib import Path


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _chart_frame(title: str, width: int, height: int) -> str:
    return (
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0e1523"/>'
        f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" rx="16" '
        'fill="#162133" stroke="#d7b36f" stroke-opacity="0.35"/>'
        f'<text x="32" y="44" font-family="Segoe UI, Arial" font-size="18" fill="#f6f2e8">{title}</text>'
    )


def _bar_chart_svg(title: str, labels: list[str], values: list[float], colors: list[str]) -> str:
    width = 940
    height = 480
    plot_left = 120
    plot_bottom = 420
    plot_top = 90
    plot_height = plot_bottom - plot_top
    max_value = max(max(values), 1e-6)

    parts = [_svg_header(width, height), _chart_frame(title, width, height)]
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{width - 60}" y2="{plot_bottom}" '
        'stroke="#8fa5c4" stroke-opacity="0.5"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" '
        'stroke="#8fa5c4" stroke-opacity="0.5"/>'
    )

    n = len(labels)
    bar_width = 96
    gap = 58
    start_x = plot_left + 40
    for i, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        x = start_x + i * (bar_width + gap)
        h = (value / max_value) * (plot_height * 0.92)
        y = plot_bottom - h
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" rx="10" '
            f'fill="{color}" fill-opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{x + (bar_width / 2):.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            'font-family="Segoe UI, Arial" font-size="14" fill="#f6f2e8">'
            f"{value:.4f}</text>"
        )
        parts.append(
            f'<text x="{x + (bar_width / 2):.1f}" y="{plot_bottom + 22}" text-anchor="middle" '
            'font-family="Segoe UI, Arial" font-size="12" fill="#d7b36f">'
            f"{label}</text>"
        )

    parts.append(
        f'<text x="{width - 64}" y="{plot_top + 4}" text-anchor="end" font-family="Segoe UI, Arial" '
        f'font-size="12" fill="#8fa5c4">max={max_value:.4f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _stability_chart_svg(backtest: dict) -> str:
    targets = [
        "overall_failure_72h_label",
        "power_failure_72h_label",
        "transformer_overload_72h_label",
        "water_failure_72h_label",
        "traffic_failure_72h_label",
    ]
    labels = ["overall", "power", "transformer", "water", "traffic"]
    values = [float(backtest["targets"][target]["metrics"]["pr_auc"]["std"]) for target in targets]
    colors = ["#4f9deb", "#e67373", "#e09a4a", "#57a693", "#9c78d4"]
    return _bar_chart_svg("Temporal Stability (PR-AUC Std by Target)", labels, values, colors)


def _calibration_svg(baseline: dict) -> str:
    bins = baseline["targets"]["overall_failure_72h_label"]["models"]["hybrid"]["reliability_bins_10"]
    rows = [row for row in bins if row["count"] > 0]

    width = 940
    height = 480
    left = 120
    right = width - 70
    bottom = 410
    top = 92
    size_x = right - left
    size_y = bottom - top

    def x_map(v: float) -> float:
        return left + (v * size_x)

    def y_map(v: float) -> float:
        return bottom - (v * size_y)

    parts = [_svg_header(width, height), _chart_frame("Calibration Curve (Overall, Hybrid)", width, height)]
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#8fa5c4" stroke-opacity="0.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#8fa5c4" stroke-opacity="0.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#d7b36f" stroke-opacity="0.6" '
        'stroke-dasharray="5 5"/>'
    )

    points = []
    for row in rows:
        px = x_map(float(row["predicted_mean"]))
        py = y_map(float(row["observed_rate"]))
        points.append((px, py, int(row["count"])))
    if points:
        poly = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="#57a693" stroke-width="3"/>')
        for x, y, count in points:
            r = 4 + min(7, int(count / 60))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="#57a693" fill-opacity="0.78"/>')

    parts.append(
        f'<text x="{right}" y="{top - 12}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" '
        'fill="#d7b36f">Dashed line = perfect calibration</text>'
    )
    parts.append(
        f'<text x="{right}" y="{bottom + 28}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" '
        'fill="#8fa5c4">Predicted probability</text>'
    )
    parts.append(
        f'<text x="{left - 60}" y="{top + 6}" font-family="Segoe UI, Arial" font-size="12" '
        'fill="#8fa5c4">Observed rate</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def generate_artifacts(
    baseline_path: Path,
    backtest_path: Path,
    output_dir: Path,
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    backtest = json.loads(backtest_path.read_text(encoding="utf-8"))

    macro = baseline["macro_pr_auc_by_model"]
    baseline_random = float(baseline["macro_random_pr_auc_baseline"])
    pr_svg = _bar_chart_svg(
        "Macro PR-AUC Benchmark",
        ["random", "rule-only", "ml-only", "hybrid"],
        [baseline_random, float(macro["rule_only"]), float(macro["ml_only"]), float(macro["hybrid"])],
        ["#6d778c", "#d7b36f", "#57a693", "#4f9deb"],
    )
    _write(output_dir / "baseline_macro_pr_auc.svg", pr_svg)

    ece_vals = []
    for model_name in ["rule_only", "ml_only", "hybrid"]:
        per_target = [
            float(payload["models"][model_name]["ece_10_bins"])
            for payload in baseline["targets"].values()
        ]
        ece_vals.append(sum(per_target) / len(per_target))
    ece_svg = _bar_chart_svg(
        "Calibration Error (Mean ECE across Targets)",
        ["rule-only", "ml-only", "hybrid"],
        ece_vals,
        ["#d7b36f", "#57a693", "#4f9deb"],
    )
    _write(output_dir / "calibration_mean_ece.svg", ece_svg)

    stability_svg = _stability_chart_svg(backtest=backtest)
    _write(output_dir / "temporal_pr_auc_stability.svg", stability_svg)

    calibration_svg = _calibration_svg(baseline=baseline)
    _write(output_dir / "calibration_curve_overall_hybrid.svg", calibration_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG visual artifacts for GridMind reports.")
    parser.add_argument(
        "--baseline",
        default=str(Path("models") / "gridmind_baseline_comparison.json"),
        help="Path to baseline comparison JSON.",
    )
    parser.add_argument(
        "--backtest",
        default=str(Path("models") / "gridmind_realworld_backtest.json"),
        help="Path to backtest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("..") / "docs" / "assets"),
        help="Output directory for SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_artifacts(
        baseline_path=Path(args.baseline).expanduser().resolve(),
        backtest_path=Path(args.backtest).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print("visual_artifacts_generated")


if __name__ == "__main__":
    main()
