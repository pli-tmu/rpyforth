#!/usr/bin/env python3
"""Generate PDF graph reports from run_shootout benchmark logs."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from jitlog_analysis import (
    aggregate_benchmark_jitlogs,
    append_jitlog_visualization_pages,
    is_curve_benchmark,
    records_from_benchmark_results,
)

if TYPE_CHECKING:
    from run_shootout import BenchmarkResult, RunPlan


def is_stable_benchmark(name: str) -> bool:
    """Return True for single-run shootout benchmarks (not curve/ warmup runs)."""
    return not is_curve_benchmark(name)


def build_stable_elapsed_samples(
    results: List[BenchmarkResult],
) -> Dict[str, Dict[str, List[int]]]:
    """Return stable benchmark elapsed times grouped by name and config."""
    samples: Dict[str, Dict[str, List[int]]] = {}
    for result in results:
        if not is_stable_benchmark(result.name):
            continue
        if result.elapsed_samples:
            elapsed = result.elapsed_samples
        elif result.elapsed_usec is not None:
            elapsed = [result.elapsed_usec]
        else:
            continue
        samples.setdefault(result.name, {})[result.config] = elapsed
    return samples


def has_iteration_variance(results: List[BenchmarkResult]) -> bool:
    """Return True when at least one stable benchmark has multiple timed runs."""
    for by_config in build_stable_elapsed_samples(results).values():
        for elapsed in by_config.values():
            if len(elapsed) > 1:
                return True
    return False


def build_paired_stable_data(
    results: List[BenchmarkResult],
    plan: RunPlan,
) -> Tuple[List[str], List[int], List[int], List[float], List[float], List[float]]:
    """Return paired elapsed times, speedups, and per-config stddevs."""
    samples_by_name = build_stable_elapsed_samples(results)

    names: List[str] = []
    a_values: List[int] = []
    b_values: List[int] = []
    speedups: List[float] = []
    a_stdevs: List[float] = []
    b_stdevs: List[float] = []
    for name in sorted(samples_by_name):
        by_config = samples_by_name[name]
        a_samples = by_config.get("A")
        b_samples = by_config.get("B") if plan.compare else None
        if a_samples is None:
            continue
        if plan.compare and b_samples is None:
            continue

        a_elapsed = int(statistics.median(a_samples))
        if plan.compare:
            assert b_samples is not None
            b_elapsed = int(statistics.median(b_samples))
            if plan.speedup_a_over_b:
                if b_elapsed == 0:
                    continue
                speedup = a_elapsed / b_elapsed
            else:
                if a_elapsed == 0:
                    continue
                speedup = b_elapsed / a_elapsed
        else:
            b_elapsed = 0
            speedup = 0.0

        names.append(name)
        a_values.append(a_elapsed)
        b_values.append(b_elapsed)
        speedups.append(speedup)
        a_stdevs.append(statistics.stdev(a_samples) if len(a_samples) > 1 else 0.0)
        b_stdevs.append(
            statistics.stdev(b_samples) if b_samples and len(b_samples) > 1 else 0.0
        )
    return names, a_values, b_values, speedups, a_stdevs, b_stdevs


# Distinct colors for an arbitrary number of compared configurations.
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def config_colors(config_ids: List[str]) -> Dict[str, str]:
    """Map each configuration id to a stable color from the palette."""
    return {cid: _PALETTE[i % len(_PALETTE)] for i, cid in enumerate(config_ids)}


def build_multi_stable_data(
    results: List[BenchmarkResult],
    plan: RunPlan,
) -> Tuple[List[str], Dict[str, List[Optional[float]]],
           Dict[str, List[float]], List[str]]:
    """Return per-config median elapsed times and CI half-widths (us) per stable
    benchmark (only benchmarks with a reference sample are kept)."""
    from run_shootout import median_ci

    samples_by_name = build_stable_elapsed_samples(results)
    config_ids = [cfg_id for cfg_id, _, _ in plan.configs]
    names = sorted(
        name
        for name in samples_by_name
        if plan.reference_config in samples_by_name[name]
    )
    data: Dict[str, List[Optional[float]]] = {cid: [] for cid in config_ids}
    errs: Dict[str, List[float]] = {cid: [] for cid in config_ids}
    for name in names:
        by_config = samples_by_name[name]
        for cid in config_ids:
            samples = by_config.get(cid)
            if samples:
                med, ci_pct = median_ci(samples)
                data[cid].append(float(med))
                errs[cid].append(float(med) * ci_pct / 100.0)
            else:
                data[cid].append(None)
                errs[cid].append(0.0)
    return names, data, errs, config_ids


def _short(name: str) -> str:
    return name.replace("shootout/", "")


def draw_grouped_elapsed(ax, names, data, plan, colors, logx: bool = False) -> None:
    """Horizontal grouped bars: one bar per config, grouped by benchmark."""
    n_cfg = max(1, len(data))
    group = 0.8
    width = group / n_cfg
    y = range(len(names))
    for j, cid in enumerate(data):
        offsets = [i - group / 2 + width * (j + 0.5) for i in y]
        heights = [v if v is not None else 0 for v in data[cid]]
        ax.barh(offsets, heights, width, label=plan.label_for(cid), color=colors[cid])
    if logx:
        ax.set_xscale("log")
    ax.set_yticks(list(y))
    ax.set_yticklabels([_short(n) for n in names])
    ax.set_xlabel("Elapsed time (microseconds%s)" % (", log scale" if logx else ""))
    ax.set_title("Elapsed time per engine")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.5)


def draw_normalized(ax, names, data, plan, colors, errs=None) -> None:
    """Horizontal grouped bars normalized to the reference config (=1.0)."""
    ref = plan.reference_config
    ref_vals = data.get(ref, [None] * len(names))
    n_cfg = max(1, len(data))
    group = 0.8
    width = group / n_cfg
    y = range(len(names))
    for j, cid in enumerate(data):
        offsets = [i - group / 2 + width * (j + 0.5) for i in y]
        norm = [
            (v / r) if (v is not None and r) else 0
            for v, r in zip(data[cid], ref_vals)
        ]
        xerr = None
        if errs:
            xerr = [
                (errs[cid][i] / ref_vals[i]) if (ref_vals[i] and data[cid][i] is not None) else 0
                for i in y
            ]
        ax.barh(offsets, norm, width, label=plan.label_for(cid), color=colors[cid],
                xerr=xerr, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_short(n) for n in names])
    ax.set_xlabel(f"Elapsed relative to {plan.label_for(ref)} (lower = faster)")
    ax.set_title(f"Normalized to {plan.label_for(ref)}")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.5)


def generate_bar_chart(
    out_path: Path,
    results: List[BenchmarkResult],
    plan: RunPlan,
) -> None:
    """Write a single-image (PNG/PDF) bar chart comparing all configurations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    names, data, errs, config_ids = build_multi_stable_data(results, plan)
    if not names:
        raise RuntimeError("No stable benchmark results to plot")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = config_colors(config_ids)
    height = max(4.0, len(names) * 0.6)
    fig, (ax_norm, ax_abs) = plt.subplots(1, 2, figsize=(15, height))
    draw_normalized(ax_norm, names, data, plan, colors, errs=errs)
    draw_grouped_elapsed(ax_abs, names, data, plan, colors, logx=True)
    labels = ", ".join(plan.label_for(c) for c in config_ids)
    fig.suptitle(f"Shootout benchmarks: {labels}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def steady_state(times: List[int], frac: float = 0.5) -> Optional[float]:
    """Median of the converged tail (last `frac` of the curve)."""
    if not times:
        return None
    tail = times[int(len(times) * (1.0 - frac)):] or times
    return float(statistics.median(tail))


def build_curve_runs(
    results: List[BenchmarkResult],
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Return per-iteration curves (one list per timed run) grouped by curve
    benchmark and config."""
    curve_runs: Dict[str, Dict[str, List[List[int]]]] = {}
    for r in results:
        if not r.name.startswith("curve/"):
            continue
        runs = [c for c in (r.curve_runs or ([r.curve_times] if r.curve_times else [])) if c]
        if runs:
            curve_runs.setdefault(r.name, {})[r.config] = runs
    return curve_runs


def draw_curve(ax, program, runs_by_config, plan, colors, logy: bool = True) -> None:
    """Plot warm-up curves for one benchmark: per-config median over runs with a
    min-max variance band and a steady-state line."""
    for config in sorted(runs_by_config):
        runs = [c for c in runs_by_config[config] if c]
        if not runs:
            continue
        color = colors.get(config)
        m = min(len(c) for c in runs)
        cols = [[c[i] for c in runs] for i in range(m)]
        med = [statistics.median(col) for col in cols]
        iters = list(range(m))
        ax.plot(iters, med, marker="o", markersize=3, linewidth=1.2,
                color=color, label=plan.label_for(config))
        if len(runs) > 1:
            lo = [min(col) for col in cols]
            hi = [max(col) for col in cols]
            ax.fill_between(iters, lo, hi, color=color, alpha=0.18, linewidth=0)
        ss = steady_state(med)
        if ss:
            ax.axhline(ss, color=color, linestyle=":", linewidth=1, alpha=0.6)
    if logy:
        ax.set_yscale("log")
    ax.set_title(program.replace("curve/", ""))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time / iteration (us%s)" % (", log" if logy else ""))
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def generate_curve_chart(
    out_path: Path,
    results: List[BenchmarkResult],
    plan: RunPlan,
    logy: bool = True,
) -> None:
    """Write a single-image (PNG/PDF) grid of warm-up curves, one per benchmark."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    curve_runs = build_curve_runs(results)
    if not curve_runs:
        raise RuntimeError(
            "No warm-up-curve results to plot. Run the curve benchmarks first, "
            "e.g. --only curve/ (they live in shootout/curve/)."
        )

    programs = sorted(curve_runs)
    colors = config_colors([cfg_id for cfg_id, _, _ in plan.configs])
    cols = min(3, len(programs))
    rows = (len(programs) + cols - 1) // cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    flat = axes.flatten()
    for idx, program in enumerate(programs):
        draw_curve(flat[idx], program, curve_runs[program], plan, colors, logy=logy)
    for idx in range(len(programs), rows * cols):
        flat[idx].set_visible(False)
    fig.suptitle(
        "Warm-up curves: time per iteration (dotted = steady state)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def generate_pdf_report(
    pdf_path: Path,
    results: List[BenchmarkResult],
    plan: RunPlan,
    jitlog_mode: str,
) -> None:
    """Generate a multi-page PDF report with benchmark graphs.

    Requires matplotlib to be installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError(
            "PDF report generation requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    label_a = plan.label_for("A")
    label_b = plan.label_for("B")
    if plan.gforth_baseline:
        elapsed_title = f"Stable Performance: {label_b} vs {label_a} (baseline)"
        speedup_title = f"Stable Performance: Speedup vs {label_a} Baseline"
        speedup_xlabel = f"Speedup ({label_a} / {label_b}); >1 means {label_b} is faster"
    elif plan.compare:
        elapsed_title = "Stable Performance: Elapsed Time Comparison"
        speedup_title = "Stable Performance: Speedup Ratio"
        speedup_xlabel = f"Speedup ({label_b} / {label_a}); >1 means {label_a} is faster"
    else:
        elapsed_title = speedup_title = speedup_xlabel = ""

    curve_colors = {"A": "#3498db", "B": "#e74c3c"} if plan.gforth_baseline else {
        "A": "#1f77b4",
        "B": "#ff7f0e",
    }

    with PdfPages(str(pdf_path)) as pdf:
        show_variance = has_iteration_variance(results)

        if plan.multi:
            names, data, errs, config_ids = build_multi_stable_data(results, plan)
            if names:
                colors = config_colors(config_ids)
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
                draw_normalized(ax, names, data, plan, colors, errs=errs)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
                draw_grouped_elapsed(ax, names, data, plan, colors, logx=True)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        elif plan.compare:
            names, a_values, b_values, speedups, a_stdevs, b_stdevs = build_paired_stable_data(
                results, plan
            )
            if names:
                x = range(len(names))
                width = 0.35
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
                ax.barh(
                    [i - width / 2 for i in x],
                    a_values,
                    width,
                    label=label_a,
                    xerr=a_stdevs if show_variance else None,
                    capsize=3 if show_variance else 0,
                )
                ax.barh(
                    [i + width / 2 for i in x],
                    b_values,
                    width,
                    label=label_b,
                    xerr=b_stdevs if show_variance else None,
                    capsize=3 if show_variance else 0,
                )
                ax.set_yticks(list(x))
                ax.set_yticklabels(names)
                ax.set_xlabel("Elapsed time (microseconds)")
                title = elapsed_title
                if show_variance:
                    title += " (median +/- stdev)"
                ax.set_title(title)
                ax.legend()
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            if names:
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
                colors = ["green" if s > 1 else "red" for s in speedups]
                ax.barh(names, speedups, color=colors, alpha=0.7)
                ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel(speedup_xlabel)
                ax.set_title(speedup_title)
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        if show_variance:
            from matplotlib.patches import Patch

            samples_by_name = build_stable_elapsed_samples(results)
            benchmarks = [
                name
                for name in sorted(samples_by_name)
                if any(len(samples) > 1 for samples in samples_by_name[name].values())
            ]
            config_ids = [config_id for config_id, _, _ in plan.configs]
            bar_colors = config_colors(config_ids) if plan.multi else curve_colors

            fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.8), 6))
            positions: List[float] = []
            boxplot_data: List[List[int]] = []
            colors: List[str] = []
            x_ticks: List[float] = []
            x_labels: List[str] = []
            x_pos = 0.0

            for benchmark in benchmarks:
                group_start = x_pos
                for config_id in config_ids:
                    elapsed = samples_by_name.get(benchmark, {}).get(config_id, [])
                    if len(elapsed) < 2:
                        continue
                    boxplot_data.append(elapsed)
                    positions.append(x_pos)
                    colors.append(bar_colors.get(config_id, "#95a5a6"))
                    x_pos += 1.0
                if x_pos > group_start:
                    x_ticks.append((group_start + x_pos - 1.0) / 2.0)
                    x_labels.append(benchmark.replace("shootout/", ""))
                    x_pos += 0.5

            if boxplot_data:
                bp = ax.boxplot(
                    boxplot_data,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showmeans=True,
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                ax.set_ylabel("Elapsed time (microseconds)")
                ax.set_title("Run-to-run Variance Across Iterations")
                ax.grid(axis="y", linestyle="--", alpha=0.5)

                legend_elements = [
                    Patch(
                        facecolor=bar_colors.get(config_id, "#95a5a6"),
                        alpha=0.7,
                        label=plan.label_for(config_id),
                    )
                    for config_id in config_ids
                ]
                ax.legend(handles=legend_elements)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        curve_runs = build_curve_runs(results)

        if curve_runs:
            programs = sorted(curve_runs.keys())
            n_programs = len(programs)
            cols = min(3, n_programs)
            rows = (n_programs + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
            )
            colors = config_colors([cfg_id for cfg_id, _, _ in plan.configs])

            for idx, program in enumerate(programs):
                draw_curve(axes.flatten()[idx], program, curve_runs[program], plan, colors)

            for idx in range(n_programs, rows * cols):
                axes.flatten()[idx].set_visible(False)

            fig.suptitle("Warm-up curves: time per iteration (log scale)", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if jitlog_mode == "jit-summary" and plan.compare and not plan.skip_jit_analysis:
            grouped = aggregate_benchmark_jitlogs(results)
            if grouped:
                append_jitlog_visualization_pages(
                    pdf,
                    grouped,
                    records_from_benchmark_results(results),
                    label_a,
                    label_b,
                    chart_title_prefix="Virtualization Analysis",
                )


def load_results_from_logs(
    output_dir: Path,
    plan: RunPlan,
    only: Optional[str],
    iterations: int,
) -> List[BenchmarkResult]:
    """Load and aggregate benchmark results from saved log files."""
    from run_shootout import aggregate_iterations, discover_log_files, load_log

    log_files = discover_log_files(output_dir, plan.compare, only, iterations)
    if not log_files:
        raise RuntimeError(f"No log files found in {output_dir}")

    results: List[BenchmarkResult] = []
    for log_path in log_files:
        result = load_log(log_path)
        if result is not None:
            results.append(result)
    if not results:
        raise RuntimeError(f"No valid benchmark logs found in {output_dir}")
    return aggregate_iterations(results)


def main(argv: Optional[List[str]] = None) -> int:
    from run_shootout import COMPARE_PRESETS, build_run_plan

    parser = argparse.ArgumentParser(
        description="Generate PDF graph reports from run_shootout benchmark logs.",
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Output file. A .png/.svg/.jpg path writes a single bar chart; "
        "any other extension writes the multi-page PDF report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs"),
        help="Directory containing per-benchmark log files (default: ./logs)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        metavar="N",
        help="Expected iteration count when discovering log files (default: 1)",
    )
    parser.add_argument(
        "--only",
        metavar="PATTERN",
        default=None,
        help="Only plot benchmarks whose log path contains this substring",
    )
    parser.add_argument(
        "--compare",
        action="append",
        metavar="CMD",
        default=None,
        help=(
            "Executable or shell command used when logs were collected. "
            f"Presets: {', '.join(sorted(COMPARE_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--a-cmd",
        type=str,
        default=None,
        help="Shell command for configuration A",
    )
    parser.add_argument(
        "--b-cmd",
        type=str,
        default=None,
        help="Shell command for configuration B",
    )
    parser.add_argument(
        "--jitlog-mode",
        type=str,
        default="jit-summary",
        help="Jitlog category referenced in logs (default: jit-summary)",
    )
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Emit a warm-up curve chart (time per iteration) instead of the "
        "steady-state report; works with any image extension",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        print("Error: --iterations must be at least 1", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent
    plan = build_run_plan(args)
    output_dir = args.output if args.output.is_absolute() else repo_root / args.output
    pdf_path = args.pdf if args.pdf.is_absolute() else repo_root / args.pdf

    try:
        results = load_results_from_logs(output_dir, plan, args.only, args.iterations)
        if args.curve:
            generate_curve_chart(pdf_path, results, plan)
        elif pdf_path.suffix.lower() in (".png", ".svg", ".jpg", ".jpeg"):
            generate_bar_chart(pdf_path, results, plan)
        else:
            generate_pdf_report(pdf_path, results, plan, args.jitlog_mode)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Graph written to {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
