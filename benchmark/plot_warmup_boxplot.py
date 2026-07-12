#!/usr/bin/env python3
"""Generate a peak-speedup boxplot with one box per execution binary.

All shootout + appbench benchmarks are pooled. Each sample is
gforth-fast_warm / engine_warm (higher = faster than gforth-fast).
Warm-up curve figures stay separate (plot_logs.py).

The input is the warmup_curves.json produced by the benchmark log pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median


SHOOTOUT_BENCHMARKS = [
    "ack",
    "ary",
    "callheavy",
    "composite",
    "except",
    "fibo",
    "heap",
    "matrix",
    "methcall",
    "nestedloop",
    "random",
    "recurse",
    "sieve",
]

APPBENCH_BENCHMARKS = [
    "cd16sim",
    "brainless",
    "fcp",
    "benchgc",
    "coremark",
    "lexex",
]

# Display order for known engines (others append alphabetically).
ENGINE_ORDER = [
    "rpyforth-c-stkfrag",
    "rpyforth",
    "gforth-fast",
    "gforth",
    "vfxforth",
    "swiftforth",
]


def percentile(sorted_values, percent):
    """Return an inclusive percentile, matching common spreadsheet behavior."""
    if not sorted_values:
        raise ValueError("cannot compute percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percent
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def box_stats(values):
    values = sorted(values)
    return {
        "lower whisker": values[0],
        "lower quartile": percentile(values, 0.25),
        "median": median(values),
        "upper quartile": percentile(values, 0.75),
        "upper whisker": values[-1],
    }


def short_engine(name):
    if name.startswith("rpyforth"):
        return "rpyforth"
    return name


def collect_engine_speedups(data, benchmarks, baseline, engines):
    """Return {engine: [speedup, ...]} across benchmarks (peak / warm median)."""
    all_benchmarks = data["benchmarks"]
    by_engine = {engine: [] for engine in engines}
    skipped = []

    for benchmark in benchmarks:
        entry = all_benchmarks.get(benchmark)
        if entry is None or baseline not in entry:
            skipped.append(benchmark)
            continue
        base_time = float(entry[baseline]["warm_median_usec"])
        if base_time <= 0:
            skipped.append(benchmark)
            continue
        for engine in engines:
            if engine not in entry:
                continue
            target_time = float(entry[engine]["warm_median_usec"])
            if target_time <= 0:
                continue
            by_engine[engine].append(base_time / target_time)

    if skipped:
        print(
            "warning: skipped benchmarks: %s" % ", ".join(skipped),
            file=sys.stderr,
        )
    return {e: vals for e, vals in by_engine.items() if vals}


def bxp_entry(label, stats):
    return {
        "label": label,
        "whislo": stats["lower whisker"],
        "q1": stats["lower quartile"],
        "med": stats["median"],
        "q3": stats["upper quartile"],
        "whishi": stats["upper whisker"],
        "fliers": [],
    }


def render_pdf(stats_by_engine, output_path, xlabel):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, NullLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render this boxplot PDF"
        ) from exc

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(stats_by_engine.keys())
    box_data = [bxp_entry(label, stats_by_engine[label]) for label in rows]

    fig, ax = plt.subplots(figsize=(8.4, max(2.4, 0.55 * len(rows) + 1.2)))
    artists = ax.bxp(
        box_data,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
    )

    for box in artists["boxes"]:
        box.set(facecolor="#d8e9ff", edgecolor="#1f5f9f", linewidth=1.2)

    for median_line in artists["medians"]:
        median_line.set(color="#1f1f1f", linewidth=1.4)

    for whisker in artists["whiskers"]:
        whisker.set(color="#1f5f9f", linewidth=1.1)

    for cap in artists["caps"]:
        cap.set(color="#1f5f9f", linewidth=1.1)

    ax.set_xscale("log", base=10)
    ax.set_xlim(0.28, 4.0)
    ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=9, pad=4)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    ax.invert_yaxis()
    ax.axvline(1.0, color="#d75a5a", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", axis="both", color="#d8d8d8", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create a peak-speedup boxplot with one box per engine."
    )
    parser.add_argument(
        "input_json",
        help="Path to warmup_curves.json, e.g. logs/warmup/.../warmup_curves.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="warmup_boxplot.pdf",
        help="Output PDF path (default: warmup_boxplot.pdf)",
    )
    parser.add_argument(
        "--baseline",
        default="gforth-fast",
        help="Baseline implementation key in the JSON (default: gforth-fast)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    present = set()
    for entry in data["benchmarks"].values():
        present.update(entry.keys())

    engines = [e for e in ENGINE_ORDER if e in present]
    engines.extend(sorted(e for e in present if e not in ENGINE_ORDER))
    if args.baseline not in present:
        raise ValueError("baseline %r not found in JSON" % args.baseline)

    all_benchmarks = SHOOTOUT_BENCHMARKS + APPBENCH_BENCHMARKS
    by_engine = collect_engine_speedups(
        data, all_benchmarks, args.baseline, engines
    )
    stats_by_engine = {
        short_engine(engine): box_stats(values)
        for engine, values in by_engine.items()
    }

    xlabel = "peak speedup over %s (log scale, >1 = faster)" % args.baseline
    render_pdf(stats_by_engine, args.output, xlabel)

    for label, stats in stats_by_engine.items():
        formatted = ", ".join("%s=%.3f" % (key, value) for key, value in stats.items())
        print("%s: %s" % (label, formatted))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
