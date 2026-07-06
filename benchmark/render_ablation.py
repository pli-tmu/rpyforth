#!/usr/bin/env python3
"""Ablation visualization: three PDFs showing what each ladder step contributed.

Usage:
    .venv/bin/python benchmark/render_ablation.py logs/analysis/7038abb-dirty/results.json

Outputs:
    /tmp/ablation_waterfall.pdf
    /tmp/ablation_summary.pdf
    /tmp/ablation_vs_gforthfast.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


LADDER = [
    "rpyforth-c-naive",
    "rpyforth-c-prefix1",
    "rpyforth-c-fix1",
    "rpyforth-c-stkfrag",
]

STEP_LABELS = [
    "+stack cache & inlining",
    "+FIX1: return-address promotion split",
    "+FIX2: loop-limit promotion split",
]

STEP_COLORS = ["#4C72B0", "#55A868", "#C44E52"]
REGRESSION_HATCH = "///"

SHOOTOUT_BENCHES = [
    "ack", "ary", "callheavy", "composite", "except", "fibo",
    "heap", "matrix", "methcall", "nestedloop", "random", "recurse", "sieve",
]
APPBENCH_BENCHES = ["benchgc", "brainless", "cd16sim", "fcp", "lexex"]


def load_cells(path: str) -> Dict[Tuple[str, str, str], Optional[float]]:
    with open(path) as f:
        data = json.load(f)
    cells = data["cells"]
    lookup: Dict[Tuple[str, str, str], Optional[float]] = {}
    for c in cells:
        key = (c["config"], c["suite"], c["bench"])
        if c["status"] == "ok" and c["median"] is not None:
            lookup[key] = float(c["median"])
        else:
            lookup[key] = None
    return lookup


def speedup_vs_naive(
    lookup: Dict[Tuple[str, str, str], Optional[float]],
    suite: str,
    bench: str,
    config: str,
) -> Optional[float]:
    naive = lookup.get(("rpyforth-c-naive", suite, bench))
    val = lookup.get((config, suite, bench))
    if naive is None or val is None or val == 0:
        return None
    return naive / val


def step_multipliers(
    lookup: Dict[Tuple[str, str, str], Optional[float]],
    suite: str,
    bench: str,
) -> Tuple[Optional[float], List[Optional[float]]]:
    """Return (final_speedup_vs_naive, [step1_mult, step2_mult, step3_mult]).

    Each step_i_mult = time[i] / time[i+1] (>1 means improvement).
    Returns None for steps where data is missing.
    """
    times: List[Optional[float]] = [
        lookup.get((cfg, suite, bench)) for cfg in LADDER
    ]
    mults: List[Optional[float]] = []
    for i in range(3):
        t0, t1 = times[i], times[i + 1]
        if t0 is None or t1 is None or t1 == 0:
            mults.append(None)
        else:
            mults.append(t0 / t1)

    final = speedup_vs_naive(lookup, suite, bench, "rpyforth-c-stkfrag")
    return final, mults


def geomean(values: List[float]) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def fmt_mult(m: float) -> str:
    return f"\xd7{m:.2f}"


def render_waterfall(
    lookup: Dict[Tuple[str, str, str], Optional[float]],
    pdf_path: str,
) -> None:
    """Per-benchmark horizontal waterfall: rows sorted by final speedup.

    The x-axis is log-scale speedup vs naive.  Each benchmark row shows three
    stacked segments (one per ladder step).  Segment left edge = cumulative
    speedup after all prior steps; right edge = cumulative speedup after this
    step.  A step that regresses (mult < 1) is drawn in a hatched red overlay
    extending leftward from the current position.
    """

    suite_info = [
        ("shootout", SHOOTOUT_BENCHES, "Shootout (13 benchmarks)"),
        ("appbench", APPBENCH_BENCHES, "Appbench (5 benchmarks)"),
    ]

    n_shootout = len(SHOOTOUT_BENCHES)
    n_appbench = len(APPBENCH_BENCHES)
    fig_h = max(7, (n_shootout + n_appbench) * 0.48 + 2.5)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, fig_h),
        gridspec_kw={"width_ratios": [n_shootout, n_appbench]},
    )

    for ax_idx, (suite, benches, panel_title) in enumerate(suite_info):
        ax = axes[ax_idx]

        rows = []
        for bench in benches:
            final, mults = step_multipliers(lookup, suite, bench)
            rows.append((bench, final, mults))

        rows.sort(key=lambda r: (r[1] if r[1] is not None else 0.0))

        bench_names = [r[0] for r in rows]
        n_rows = len(bench_names)

        for yi, (bench, final, mults) in enumerate(rows):
            cum_log = 0.0

            for si, mult in enumerate(mults):
                if mult is None:
                    x_cur = math.exp(cum_log)
                    ax.text(
                        x_cur * 1.03, yi, "n/a",
                        va="center", ha="left", fontsize=6, color="gray",
                    )
                    continue

                is_regression = mult < 1.0
                color = STEP_COLORS[si]

                x_left = math.exp(cum_log)
                new_cum_log = cum_log + math.log(max(mult, 1e-9))
                x_right = math.exp(new_cum_log)

                if is_regression:
                    bar_left = x_right
                    bar_width = x_left - x_right
                    ec = "#8B0000"
                    fc = "#C44E52"
                    hatch = REGRESSION_HATCH
                    alpha = 0.65
                else:
                    bar_left = x_left
                    bar_width = x_right - x_left
                    ec = color
                    fc = color
                    hatch = ""
                    alpha = 0.85

                ax.barh(
                    yi, bar_width, left=bar_left,
                    height=0.72, color=fc, edgecolor=ec,
                    hatch=hatch, linewidth=0.5, alpha=alpha,
                )

                if abs(math.log(max(mult, 1e-9))) > math.log(1.05):
                    label_x = (bar_left + bar_left + bar_width) / 2
                    ax.text(
                        label_x, yi, fmt_mult(mult),
                        ha="center", va="center", fontsize=6,
                        color="white" if not is_regression else "#600000",
                        fontweight="bold",
                    )

                cum_log = new_cum_log

        ax.set_yticks(list(range(n_rows)))
        ax.set_yticklabels(bench_names, fontsize=8)
        ax.set_xscale("log")
        ax.axvline(x=1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
        ax.set_xlabel("Cumulative speedup vs. naive (log scale)", fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        all_finals = [r[1] for r in rows if r[1] is not None]
        x_max = max(all_finals) * 1.25 if all_finals else 4.0
        x_min = min(0.3, min(all_finals) * 0.7) if all_finals else 0.3
        ax.set_xlim(left=x_min, right=x_max)

    legend_patches = [
        mpatches.Patch(facecolor=STEP_COLORS[i], alpha=0.85, label=STEP_LABELS[i])
        for i in range(3)
    ]
    regression_patch = mpatches.Patch(
        facecolor="#C44E52", hatch=REGRESSION_HATCH, alpha=0.65,
        label="regression step (hatched, leftward)",
    )
    legend_patches.append(regression_patch)
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.suptitle(
        "Ablation waterfall: per-step multiplicative contribution to speedup vs. naive",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def render_summary(
    lookup: Dict[Tuple[str, str, str], Optional[float]],
    pdf_path: str,
) -> None:
    """Two grouped bars: shootout geomean and appbench geomean, cumulative after each step."""

    suite_info = [
        ("shootout", SHOOTOUT_BENCHES, "Shootout"),
        ("appbench", APPBENCH_BENCHES, "Appbench"),
    ]

    step_configs = LADDER[1:]
    n_steps = len(step_configs)
    suite_labels = [s[2] for s in suite_info]

    cumulative: Dict[str, List[float]] = {}
    for suite, benches, label in suite_info:
        cum_by_step = []
        for cfg in step_configs:
            vals = []
            for bench in benches:
                sp = speedup_vs_naive(lookup, suite, bench, cfg)
                if sp is not None and sp > 0:
                    vals.append(sp)
            cum_by_step.append(geomean(vals) if vals else 1.0)
        cumulative[label] = cum_by_step

    fig, ax = plt.subplots(figsize=(9, 5))

    n_suites = len(suite_labels)
    x = np.arange(n_suites)
    bar_total_width = 0.65
    bar_w = bar_total_width / n_steps
    offsets = np.linspace(
        -(bar_total_width - bar_w) / 2,
        (bar_total_width - bar_w) / 2,
        n_steps,
    )

    for si, (cfg, label, color) in enumerate(
        zip(step_configs, STEP_LABELS, STEP_COLORS)
    ):
        heights = [cumulative[sl][si] for sl in suite_labels]
        bars = ax.bar(
            x + offsets[si],
            heights,
            width=bar_w * 0.9,
            color=color,
            label=f"After {label}",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                color=STEP_COLORS[si],
                fontweight="bold",
            )

    for si in range(1, n_steps):
        for suite_idx, sl in enumerate(suite_labels):
            prev = cumulative[sl][si - 1]
            curr = cumulative[sl][si]
            delta = curr / prev
            x_pos = x[suite_idx] + offsets[si]
            y_pos = curr + 0.12
            sign = "+" if delta >= 1.0 else ""
            pct = (delta - 1.0) * 100
            ax.text(
                x_pos,
                y_pos,
                f"{sign}{pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="naive baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(suite_labels, fontsize=11)
    ax.set_ylabel("Geomean speedup vs. rpy-naive", fontsize=10)
    ax.set_title(
        "Ablation summary: cumulative geomean speedup per suite",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def render_vs_gforthfast(
    lookup: Dict[Tuple[str, str, str], Optional[float]],
    pdf_path: str,
) -> None:
    """Per-benchmark: stkfrag vs gforth-fast, with naive marker."""

    all_benches = []
    for bench in SHOOTOUT_BENCHES:
        all_benches.append(("shootout", bench))
    for bench in APPBENCH_BENCHES:
        all_benches.append(("appbench", bench))

    rows = []
    for suite, bench in all_benches:
        gf = lookup.get(("gforth-fast", suite, bench))
        stkfrag = lookup.get(("rpyforth-c-stkfrag", suite, bench))
        naive = lookup.get(("rpyforth-c-naive", suite, bench))
        if gf is None or stkfrag is None or gf == 0 or stkfrag == 0:
            continue
        ratio_stkfrag = gf / stkfrag
        ratio_naive = (gf / naive) if naive and naive > 0 else None
        rows.append((bench, suite, ratio_stkfrag, ratio_naive))

    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(12, 6))

    bench_labels = [f"{r[0]}" for r in rows]
    y_pos = list(range(len(rows)))

    win_color = "#4C72B0"
    loss_color = "#C44E52"

    for yi, (bench, suite, ratio_stkfrag, ratio_naive) in enumerate(rows):
        color = win_color if ratio_stkfrag >= 1.0 else loss_color
        ax.barh(
            yi,
            ratio_stkfrag,
            height=0.65,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        suite_marker = "S" if suite == "shootout" else "A"
        label_x = max(ratio_stkfrag + 0.02, 0.1)
        ax.text(
            label_x,
            yi,
            f"{ratio_stkfrag:.2f}x  [{suite_marker}]",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

        if ratio_naive is not None:
            ax.plot(
                ratio_naive,
                yi,
                marker="D",
                markersize=5,
                color="black",
                alpha=0.6,
                zorder=5,
            )

    ax.axvline(x=1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bench_labels, fontsize=8)
    ax.set_xlabel("Speedup of rpy-stkfrag vs. gforth-fast  (>1 = rpy wins)", fontsize=10)
    ax.set_title(
        "Final position: rpy-stkfrag vs gforth-fast\n(bar = stkfrag; diamond = rpy-naive; sorted by final)",
        fontsize=11,
        fontweight="bold",
    )

    win_patch = mpatches.Patch(facecolor=win_color, alpha=0.8, label="rpy-stkfrag wins")
    loss_patch = mpatches.Patch(facecolor=loss_color, alpha=0.8, label="rpy-stkfrag loses")
    naive_marker = plt.Line2D(
        [0], [0], marker="D", color="w", markerfacecolor="black",
        markersize=6, alpha=0.6, label="rpy-naive baseline",
    )
    ax.legend(handles=[win_patch, loss_patch, naive_marker], fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def load_warm_steady(path: str) -> Dict[Tuple[str, str], Optional[float]]:
    """Load warm_steady.json produced by run_warmup_curves.py --json.

    Returns a dict keyed by (engine_label, bench_name) -> warm_median_usec.
    """
    with open(path) as f:
        data = json.load(f)
    benchmarks = data.get("benchmarks", {})
    lookup: Dict[Tuple[str, str], Optional[float]] = {}
    for bench_name, eng_map in benchmarks.items():
        for label, info in eng_map.items():
            warm = info.get("warm_median_usec")
            lookup[(label, bench_name)] = float(warm) if warm is not None else None
    return lookup


def _suite_for(bench: str) -> str:
    if bench in SHOOTOUT_BENCHES:
        return "shootout"
    return "appbench"


def render_vs_gforthfast_warm(
    cold_lookup: Dict[Tuple[str, str, str], Optional[float]],
    warm_lookup: Dict[Tuple[str, str], Optional[float]],
    pdf_path: str,
) -> None:
    """vs-gforth-fast chart using WARM steady-state medians.

    Bar = warm ratio (gforth-fast_warm / flagship_warm).
    Naive warm marker = black diamond.
    Cold ratio shown as hollow diamond per bar.
    """
    FLAGSHIP_LABEL = "rpyforth-c-stkfrag"
    GFFAST_LABEL = "gforth-fast"
    NAIVE_LABEL = "rpyforth-naive"

    all_benches = []
    for bench in SHOOTOUT_BENCHES:
        all_benches.append(("shootout", bench))
    for bench in APPBENCH_BENCHES:
        all_benches.append(("appbench", bench))

    rows = []
    for suite, bench in all_benches:
        gf_warm = warm_lookup.get((GFFAST_LABEL, bench))
        flagship_warm = warm_lookup.get((FLAGSHIP_LABEL, bench))
        naive_warm = warm_lookup.get((NAIVE_LABEL, bench))

        # cold ratio from existing cold lookup
        gf_cold = cold_lookup.get(("gforth-fast", suite, bench))
        flagship_cold = cold_lookup.get(("rpyforth-c-stkfrag", suite, bench))

        if gf_warm is None or flagship_warm is None or flagship_warm == 0:
            continue

        warm_ratio = gf_warm / flagship_warm
        naive_warm_ratio = (gf_warm / naive_warm) if (naive_warm and naive_warm > 0) else None
        cold_ratio = (gf_cold / flagship_cold) if (gf_cold and flagship_cold and flagship_cold > 0) else None

        rows.append((bench, suite, warm_ratio, naive_warm_ratio, cold_ratio))

    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(13, 6.5))

    bench_labels = [r[0] for r in rows]
    y_pos = list(range(len(rows)))

    win_color = "#4C72B0"
    loss_color = "#C44E52"

    for yi, (bench, suite, warm_ratio, naive_warm_ratio, cold_ratio) in enumerate(rows):
        color = win_color if warm_ratio >= 1.0 else loss_color
        ax.barh(
            yi,
            warm_ratio,
            height=0.65,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        suite_marker = "S" if suite == "shootout" else "A"
        label_x = max(warm_ratio + 0.02, 0.1)
        ax.text(
            label_x,
            yi,
            f"{warm_ratio:.2f}x  [{suite_marker}]",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

        # Naive warm marker (black filled diamond)
        if naive_warm_ratio is not None:
            ax.plot(
                naive_warm_ratio,
                yi,
                marker="D",
                markersize=5,
                color="black",
                alpha=0.6,
                zorder=5,
            )

        # Cold ratio marker (hollow diamond, grey)
        if cold_ratio is not None:
            ax.plot(
                cold_ratio,
                yi,
                marker="D",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="#888888",
                markeredgewidth=1.0,
                alpha=0.75,
                zorder=4,
            )

    ax.axvline(x=1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bench_labels, fontsize=8)
    ax.set_xlabel("Speedup of rpy-stkfrag vs. gforth-fast  (>1 = rpy wins)", fontsize=10)
    ax.set_title(
        "Final position: rpy-stkfrag vs gforth-fast — warm steady-state\n"
        "(bar = warm ratio; filled ◆ = rpy-naive warm; hollow ◆ = cold ratio)",
        fontsize=11,
        fontweight="bold",
    )

    win_patch = mpatches.Patch(facecolor=win_color, alpha=0.8, label="rpy-stkfrag wins (warm)")
    loss_patch = mpatches.Patch(facecolor=loss_color, alpha=0.8, label="rpy-stkfrag loses (warm)")
    naive_marker = plt.Line2D(
        [0], [0], marker="D", color="w", markerfacecolor="black",
        markersize=6, alpha=0.6, label="rpy-naive warm baseline",
    )
    cold_marker = plt.Line2D(
        [0], [0], marker="D", color="w", markerfacecolor="none",
        markeredgecolor="#888888", markeredgewidth=1.0,
        markersize=6, alpha=0.75, label="cold ratio (hollow)",
    )
    ax.legend(handles=[win_patch, loss_patch, naive_marker, cold_marker],
              fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", help="cold ablation results.json")
    parser.add_argument("--steady-json", default=None,
                        help="warm_steady.json from run_warmup_curves.py --json; "
                             "renders warm vs-gforthfast chart to "
                             "/tmp/ablation_vs_gforthfast_warm.pdf")
    args = parser.parse_args()

    lookup = load_cells(args.results_json)

    render_waterfall(lookup, "/tmp/ablation_waterfall.pdf")
    render_summary(lookup, "/tmp/ablation_summary.pdf")
    render_vs_gforthfast(lookup, "/tmp/ablation_vs_gforthfast.pdf")

    if args.steady_json:
        warm_lookup = load_warm_steady(args.steady_json)
        render_vs_gforthfast_warm(
            lookup,
            warm_lookup,
            "/tmp/ablation_vs_gforthfast_warm.pdf",
        )


if __name__ == "__main__":
    main()
