#!/usr/bin/env python3
"""Render the RQ1-RQ4 verification results as publication-quality figures.

Reads an rq_results.json produced by run_rq.py (auto-discovering the newest one
under logs/rq/ unless --rq-json is given) and the steady_results.json it points
at (key `steady_json`, overridable with --steady-json). Emits a multi-page PDF
via matplotlib PdfPages; with --formats pdf,png it additionally writes one PNG
per figure alongside the PDF.

Pages (a page is skipped, not fatal, when its section is missing):
  1  RQ1 trace affinity : speedup-vs-gforth against bridge-exec fraction, plus a
                          |rho| bar panel over all RQ1 metrics.
  2  RQ2 gc contribution: GC pause fraction per program with the 20% flag line.
  3  RQ3 warmup economics: cumulative-time curves per program with break-even N*.
  4  RQ4 validity       : warm-tail drift, coverage matrix, survivorship geomeans.

Engine colors are shared with the other benchmark plots via plot_engines, so the
legend convention matches plot_shootout.py / plot_warmup_boxplot.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from plot_engines import engine_color, engine_display_name, sort_engines

REPO_ROOT = Path(__file__).resolve().parent.parent

GC_FRACTION_FLAG = 0.20
DRIFT_FLAG = 0.5  # percent per iteration

ENGINE_RPYFORTH = "rpyforth"
ENGINE_GFORTH_FAST = "gforth-fast"
ENGINE_VFXFORTH = "vfxforth"
ENGINE_SWIFTFORTH = "swiftforth"
OTHER_ENGINES = [ENGINE_GFORTH_FAST, ENGINE_VFXFORTH, ENGINE_SWIFTFORTH]

HIGHLIGHT_METRIC = "bridge_exec_fraction"

SUITE_APPBENCH = "appbench"
SUITE_SHOOTOUT = "shootout"
# Marker shape distinguishes the two suites; color/label stay the rpyforth
# convention shared with the other benchmark plots.
SUITE_MARKER = {SUITE_APPBENCH: "o", SUITE_SHOOTOUT: "^"}
SUITE_MARKER_DEFAULT = "o"


def find_newest_rq_json(logs_root):
    candidates = list(logs_root.rglob("rq_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_steady(path):
    """Return {program: {engine: result_dict}} from a steady_results.json."""
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    by_prog = {}
    for r in summary.get("results", []):
        by_prog.setdefault(r["program"], {})[r["engine"]] = r
    return summary, by_prog


def cumulative(times):
    out = []
    total = 0
    for t in times:
        total += t
        out.append(total)
    return out


def fmt(v, spec="%.3f"):
    return "n/a" if v is None else spec % v


# ===========================================================================
# page 1: RQ1 trace affinity
# ===========================================================================

def draw_rq1(fig, rq1):
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    import matplotlib.lines as mlines

    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    rows = rq1.get("rows", [])

    pts = [(r.get("bridge_exec_fraction"), r.get("speedup_vs_gforth"),
            r["program"], r.get("suite", SUITE_APPBENCH)) for r in rows]
    pts = [(x, y, n, s) for x, y, n, s in pts if x is not None and y is not None]

    suites_present = []
    for x, y, name, suite in pts:
        marker = SUITE_MARKER.get(suite, SUITE_MARKER_DEFAULT)
        ax.scatter(x, y, s=55, color=engine_color(ENGINE_RPYFORTH),
                   marker=marker, edgecolor="0.2", linewidth=0.6, zorder=3)
        if suite not in suites_present:
            suites_present.append(suite)

    _annotate_no_overlap(ax, pts)

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yscale("log", base=2)
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda v, _p: "%gx" % v))
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_xlabel("bridge-exec fraction (bridge runs / all compiled-code runs)")
    ax.set_ylabel("speedup vs gforth-fast (log2, 1.0x = parity)")

    rho = rq1.get("correlations", {}).get(HIGHLIGHT_METRIC)
    n_all = rq1.get("n", len(rows))
    sub = rq1.get("subgroup_correlations", {})
    title = "RQ1  trace affinity\ncombined rho = %s (n=%d)" % (
        fmt(rho, "%.3f"), n_all)
    for suite in (SUITE_APPBENCH, SUITE_SHOOTOUT):
        if suite in sub:
            title += "   %s rho = %s (n=%d)" % (
                suite, fmt(sub[suite]["correlations"].get(HIGHLIGHT_METRIC),
                          "%.3f"), sub[suite]["n"])
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if len(suites_present) > 1:
        handles = [
            mlines.Line2D([], [], color=engine_color(ENGINE_RPYFORTH),
                          marker=SUITE_MARKER.get(s, SUITE_MARKER_DEFAULT),
                          linestyle="none", markeredgecolor="0.2",
                          markersize=8, label=s)
            for s in suites_present]
        ax.legend(handles=handles, loc="lower left", fontsize=9,
                  title="suite", frameon=True)

    ax2 = fig.add_subplot(gs[0, 1])
    metrics = rq1.get("metrics", [])
    corr = rq1.get("correlations", {})
    labels = []
    mags = []
    colors = []
    for key, label in metrics:
        rho_m = corr.get(key)
        if rho_m is None:
            continue
        labels.append(label)
        mags.append(abs(rho_m))
        colors.append(engine_color(ENGINE_RPYFORTH)
                      if key == HIGHLIGHT_METRIC else "#7f7f7f")
    order = sorted(range(len(mags)), key=lambda i: mags[i])
    ypos = np.arange(len(order))
    ax2.barh(ypos, [mags[i] for i in order],
             color=[colors[i] for i in order], edgecolor="0.2", linewidth=0.5)
    for y, i in zip(ypos, order):
        ax2.text(mags[i] + 0.01, y, fmt(mags[i], "%.2f"),
                 va="center", fontsize=8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("|Spearman rho| vs speedup")
    ax2.set_title("Predictive strength by metric\n(highlight: bridge-exec)")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)

    fig.suptitle("RQ1: does JIT trace behaviour predict speedup-vs-gforth?",
                 fontsize=13)


def _annotate_no_overlap(ax, pts):
    """Place program labels near points, nudging alternate points to reduce
    visual overlap. Points are few (per-program), so a simple stagger suffices."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    for rank, i in enumerate(order):
        x, y, name = pts[i][0], pts[i][1], pts[i][2]
        dy = 8 if rank % 2 == 0 else -12
        ax.annotate(name, (x, y), fontsize=8,
                    xytext=(6, dy), textcoords="offset points",
                    ha="left", va="bottom" if dy > 0 else "top")


# ===========================================================================
# page 2: RQ2 gc contribution
# ===========================================================================

def draw_rq2(fig, rq2):
    import numpy as np

    ax = fig.add_subplot(1, 1, 1)
    if not rq2.get("available"):
        ax.axis("off")
        ax.text(0.5, 0.5,
                "RQ2 (GC vs JIT) unavailable\n\nNo PYPYLOG gc section produced "
                "output on this build.",
                ha="center", va="center", fontsize=13,
                transform=ax.transAxes)
        fig.suptitle("RQ2: GC pause contribution", fontsize=13)
        return

    rows = rq2.get("rows", [])
    names = [r["program"] for r in rows]
    fracs = [(r.get("gc_fraction") or 0.0) * 100.0 for r in rows]
    pauses_ms = [(r.get("gc_pause_seconds") or 0.0) * 1000.0 for r in rows]
    dominated = [bool(r.get("memory_management_dominated")) for r in rows]

    x = np.arange(len(names))
    colors = ["#d62728" if d else engine_color(ENGINE_RPYFORTH)
              for d in dominated]
    bars = ax.bar(x, fracs, color=colors, edgecolor="0.2", linewidth=0.5,
                  width=0.62)
    ax.axhline(GC_FRACTION_FLAG * 100.0, color="0.35", linestyle="--",
               linewidth=1.2,
               label="flag threshold (%d%%)" % int(GC_FRACTION_FLAG * 100))

    for rect, ms in zip(bars, pauses_ms):
        ax.text(rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(), "%.1f ms" % ms,
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("GC pause time as %% of wall time")
    ax.set_ylim(0, max(GC_FRACTION_FLAG * 100.0 * 1.15, max(fracs) * 1.35 + 1))
    ax.set_title("PYPYLOG section: %s" % rq2.get("section", "?"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.suptitle("RQ2: how much wall time is spent in GC per program?",
                 fontsize=13)


# ===========================================================================
# page 3: RQ3 warmup economics
# ===========================================================================

def draw_rq3(fig, rq3, by_prog):
    rows = [r for r in rq3.get("rows", []) if r.get("available")]
    rows = [r for r in rows
            if by_prog.get(r["program"], {}).get(ENGINE_RPYFORTH, {}).get("times")]
    if not rows:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, "RQ3: no per-iteration rpyforth data available",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
        fig.suptitle("RQ3: warmup economics", fontsize=13)
        return

    n = len(rows)
    cols = min(3, n)
    nrows = (n + cols - 1) // cols
    axes = fig.subplots(nrows, cols, squeeze=False)
    flat = [axes[i][j] for i in range(nrows) for j in range(cols)]

    engines = [ENGINE_RPYFORTH] + OTHER_ENGINES
    legend_handles = {}

    for idx, r in enumerate(rows):
        ax = flat[idx]
        prog = r["program"]
        eng_res = by_prog.get(prog, {})
        rpy_times = eng_res.get(ENGINE_RPYFORTH, {}).get("times", [])

        for engine in engines:
            res = eng_res.get(engine)
            if not res or not res.get("times"):
                continue
            times = res["times"]
            cum = [c / 1e6 for c in cumulative(times)]
            xs = list(range(1, len(cum) + 1))
            marker = "o" if len(xs) > 1 else "D"
            line, = ax.plot(xs, cum, marker=marker, markersize=3,
                            linewidth=1.4, color=engine_color(engine),
                            label=engine_display_name(engine))
            legend_handles.setdefault(engine, line)

        be = r.get("break_even", {})
        rpy_n = len(rpy_times)
        for engine in OTHER_ENGINES:
            d = be.get(engine, {})
            n_star = d.get("break_even")
            if n_star is not None and n_star <= rpy_n:
                ax.axvline(n_star, color=engine_color(engine),
                           linestyle=":", linewidth=1.2, alpha=0.8)
                ax.annotate("N*=%d" % n_star, (n_star, 0),
                            xytext=(2, 2), textcoords="offset points",
                            fontsize=7, color=engine_color(engine),
                            rotation=90, va="bottom", ha="left")

        wc = r.get("warmup_cost_usec")
        wc_s = (wc / 1e6) if wc is not None else None
        ax.set_title("%s  (warm-up cost %s s)" % (prog, fmt(wc_s, "%.2f")),
                     fontsize=10)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_ylabel("cumulative time (s)", fontsize=9)
        ax.set_xlim(left=0.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    for idx in range(n, len(flat)):
        flat[idx].set_visible(False)

    ordered = sort_engines(legend_handles.keys())
    handles = [legend_handles[e] for e in ordered if e in legend_handles]
    labels = [engine_display_name(e) for e in ordered if e in legend_handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("RQ3: cumulative time vs iteration; dotted vertical = "
                 "sustained break-even N* (absent = never)", fontsize=12)


# ===========================================================================
# page 4: RQ4 validity
# ===========================================================================

def draw_rq4(fig, rq4):
    import numpy as np

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.28)
    ax_drift = fig.add_subplot(gs[0, :])
    ax_cov = fig.add_subplot(gs[1, 0])
    ax_surv = fig.add_subplot(gs[1, 1])

    _draw_rq4_drift(ax_drift, rq4, np)
    _draw_rq4_coverage(ax_cov, rq4)
    _draw_rq4_survivorship(ax_surv, rq4, np)

    fig.suptitle("RQ4: methodology validity (drift, coverage, survivorship)",
                 fontsize=13)


def _draw_rq4_drift(ax, rq4, np):
    rows = [r for r in rq4.get("drift_rows", [])
            if r.get("drift_pct_per_iter") is not None]
    labels = ["%s / %s" % (r["program"], r["engine"]) for r in rows]
    vals = [r["drift_pct_per_iter"] for r in rows]
    colors = ["#d62728" if r.get("flag") else engine_color(r["engine"])
              for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.axvspan(-DRIFT_FLAG, DRIFT_FLAG, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(DRIFT_FLAG, color="0.4", linestyle=":", linewidth=1.0)
    ax.axvline(-DRIFT_FLAG, color="0.4", linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="0.2", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("warm-tail drift (%/iter); shaded band = within +/-0.5%% flag")
    ax.set_title("(a) Warm-tail drift per program+engine "
                 "(red = flagged)", fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def _draw_rq4_coverage(ax, rq4):
    from matplotlib.patches import Rectangle
    coverage = rq4.get("coverage", {})
    engines = rq4.get("coverage_engines", [])
    progs = sorted(coverage.keys())
    status_val = {"OK": 0, "TIMEOUT": 1, "NO-DATA": 2}
    status_color = {0: "#2ca02c", 1: "#ff7f0e", 2: "#7f7f7f"}

    grid = []
    for prog in progs:
        grid.append([status_val.get(coverage[prog].get(e), 2) for e in engines])

    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels([engine_display_name(e) for e in engines],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(progs)))
    ax.set_yticklabels(progs, fontsize=8)
    ax.set_xlim(-0.5, len(engines) - 0.5)
    ax.set_ylim(-0.5, len(progs) - 0.5)
    ax.invert_yaxis()

    for i, prog in enumerate(progs):
        for j, e in enumerate(engines):
            code = grid[i][j]
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                facecolor=status_color[code], edgecolor="white", linewidth=1.0))
            txt = coverage[prog].get(e, "NO-DATA")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white")
    ax.set_title("(b) Coverage matrix", fontsize=10)
    ax.tick_params(length=0)


def _draw_rq4_survivorship(ax, rq4, np):
    surv = rq4.get("survivorship", {})
    engines = sort_engines(surv.keys())
    engines = [e for e in engines if e in surv]
    if not engines:
        ax.axis("off")
        ax.text(0.5, 0.5, "no survivorship data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x = np.arange(len(engines))
    width = 0.38
    own = [surv[e].get("geomean_surviving") or 0.0 for e in engines]
    common = [surv[e].get("geomean_common_subset") or 0.0 for e in engines]
    ax.bar(x - width / 2, own, width, label="surviving subset",
           color=[engine_color(e) for e in engines], edgecolor="0.2",
           linewidth=0.4)
    ax.bar(x + width / 2, common, width, label="common subset",
           color=[engine_color(e) for e in engines], edgecolor="0.2",
           linewidth=0.4, alpha=0.5, hatch="//")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    for xi, v in zip(x - width / 2, own):
        ax.text(xi, v, fmt(v, "%.2f"), ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + width / 2, common):
        ax.text(xi, v, fmt(v, "%.2f"), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([engine_display_name(e) for e in engines],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("geomean speedup rpyforth-vs-engine", fontsize=9)
    ax.set_title("(c) Survivorship (solid = surviving, "
                 "hatched = common)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


# ===========================================================================
# driver
# ===========================================================================

PAGES = [
    ("rq1", "rq1_trace_affinity", draw_rq1),
    ("rq2", "rq2_gc", draw_rq2),
    ("rq3", "rq3_warmup", draw_rq3),
    ("rq4", "rq4_validity", draw_rq4),
]


def render(sections, by_prog, pdf_path, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    want_png = "png" in formats
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    n_pages = 0
    skipped = []
    with PdfPages(str(pdf_path)) as pdf:
        for key, stem, drawer in PAGES:
            section = sections.get(key)
            if not section:
                skipped.append(key)
                continue
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            if key == "rq3":
                drawer(fig, section, by_prog)
            else:
                drawer(fig, section)
            pdf.savefig(fig)
            n_pages += 1
            if want_png:
                png_path = pdf_path.parent / ("%s_%s.png"
                                              % (pdf_path.stem, stem))
                fig.savefig(str(png_path), dpi=150)
            plt.close(fig)

    return n_pages, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rq-json", default="",
                        help="rq_results.json (default: newest under logs/rq/)")
    parser.add_argument("--steady-json", default="",
                        help="steady_results.json (default: from rq json)")
    parser.add_argument("--output", default="",
                        help="output PDF (default: <rq-json dir>/rq_figures.pdf)")
    parser.add_argument("--formats", default="pdf",
                        help="comma-separated: pdf[,png] (pdf always emitted)")
    args = parser.parse_args(argv)

    if args.rq_json:
        rq_path = Path(args.rq_json)
        if not rq_path.is_absolute():
            rq_path = REPO_ROOT / rq_path
    else:
        rq_path = find_newest_rq_json(REPO_ROOT / "logs" / "rq")
        if rq_path is None:
            print("No rq_results.json found under logs/rq/; pass --rq-json",
                  file=sys.stderr)
            return 1
    if not rq_path.exists():
        print("rq json not found: %s" % rq_path, file=sys.stderr)
        return 1

    rq_data = json.loads(rq_path.read_text(encoding="utf-8"))
    sections = rq_data.get("sections", {})

    if args.steady_json:
        steady_path = Path(args.steady_json)
        if not steady_path.is_absolute():
            steady_path = REPO_ROOT / steady_path
    else:
        steady_path = Path(rq_data.get("steady_json", ""))

    by_prog = {}
    if steady_path and steady_path.exists():
        _summary, by_prog = load_steady(steady_path)
    else:
        print("warning: steady json not found (%s); RQ3 per-iteration curves "
              "will be limited to what's in the rq json" % steady_path,
              file=sys.stderr)

    formats = set(f.strip().lower() for f in args.formats.split(",") if f.strip())
    formats.add("pdf")

    if args.output:
        pdf_path = Path(args.output)
        if not pdf_path.is_absolute():
            pdf_path = REPO_ROOT / pdf_path
    else:
        pdf_path = rq_path.parent / "rq_figures.pdf"

    try:
        n_pages, skipped = render(sections, by_prog, pdf_path, formats)
    except ImportError:
        print("matplotlib is required to render figures", file=sys.stderr)
        return 1

    if n_pages == 0:
        print("No RQ sections present in %s; nothing rendered" % rq_path,
              file=sys.stderr)
        return 1

    print("rq json:    %s" % rq_path)
    print("steady json: %s" % steady_path)
    print("wrote %s (%d pages)" % (pdf_path, n_pages))
    if "png" in formats:
        print("also wrote per-figure PNGs alongside")
    if skipped:
        print("skipped missing sections: %s" % ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
