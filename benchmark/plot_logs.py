#!/usr/bin/env python3
"""Generate unified benchmark reports from runbench.sh and run_appbench.sh logs.

This script reads:
  * Appbench logs written by benchmark/run_appbench.sh
    (logs/<rev>/appbench/results.json and csv/*.csv)
  * Shootout logs written by benchmark/run_shootout.py
    (logs/<timestamp>/*.log, compatible with run_shootout.load_log)
  * Optional warmup_curves.json (paper-style warm-up dump)

and produces two complementary views (do not collapse curves into the boxplot):
  * <prefix>_warmup_curves.pdf : one subplot per benchmark (full warm-up curve)
  * <prefix>_boxplot.pdf       : overview — one box per engine across all benches
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

from plot_engines import (
    PRIMARY_ENGINES,
    engine_color,
    normalize_engine,
    sort_engines,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CurveData = Dict[str, Dict[str, List[int]]]


def load_appbench_csv(csv_dir: Path) -> CurveData:
    """Load per-iteration CSV files from appbench steady mode."""
    data: CurveData = {}
    if not csv_dir.is_dir():
        return data
    for path in sorted(csv_dir.glob("*.csv")):
        stem = path.stem
        # filename: <program>_<engine>.csv
        if "_" not in stem:
            continue
        engine = normalize_engine(stem.rsplit("_", 1)[1])
        program = stem.rsplit("_", 1)[0]
        times: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    try:
                        times.append(int(row[1]))
                    except ValueError:
                        pass
        if times:
            data.setdefault(program, {})[engine] = times
    return data


def load_appbench_json(json_path: Path) -> CurveData:
    """Load appbench results.json as a fallback / supplement."""
    data: CurveData = {}
    if not json_path.is_file():
        return data
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for entry in payload.get("results", []):
        program = entry.get("program")
        engine = normalize_engine(entry.get("engine", ""))
        times = entry.get("times")
        if program and engine and times:
            data.setdefault(program, {})[engine] = list(times)
    return data


def load_warmup_curves_json(json_path: Path) -> CurveData:
    """Load paper-style warmup_curves.json ({benchmarks: {name: {engine: {times_usec: []}}}})."""
    data: CurveData = {}
    if not json_path.is_file():
        return data
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    benches = payload.get("benchmarks", {})
    for program, by_engine in benches.items():
        if not isinstance(by_engine, dict):
            continue
        for engine, info in by_engine.items():
            times = None
            if isinstance(info, dict):
                times = info.get("times_usec") or info.get("times")
            elif isinstance(info, list):
                times = info
            if times:
                data.setdefault(program, {})[normalize_engine(engine)] = list(times)
    return data


def load_appbench(log_dir: Path) -> CurveData:
    """Load appbench logs from log_dir (either logs/<rev>/appbench or a dir
    containing appbench/ directly)."""
    candidates = [
        log_dir / "appbench",
        log_dir,
    ]
    for cand in candidates:
        data: CurveData = {}
        for json_name in ("steady_results.json", "results.json", "warmup_curves.json"):
            json_path = cand / json_name
            if not json_path.is_file():
                continue
            if json_name == "warmup_curves.json":
                data = merge_curves(data, load_warmup_curves_json(json_path))
            else:
                data = merge_curves(data, load_appbench_json(json_path))
        csv_dir = cand / "csv"
        if csv_dir.is_dir():
            data = merge_curves(data, load_appbench_csv(csv_dir))
        if data:
            return data
    return {}


def _load_config_labels(log_dir: Path) -> Dict[str, str]:
    """Map config id (A/B/...) -> engine label from results JSON if present."""
    for name in ("results.json", "shootout_results.json", "summary.json"):
        path = log_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        configs = payload.get("configs") or []
        out: Dict[str, str] = {}
        for entry in configs:
            cfg_id = str(entry.get("id", "")).upper()
            label = entry.get("label") or ""
            if not label and entry.get("cmd"):
                cmd = entry["cmd"]
                label = cmd[0] if isinstance(cmd, list) and cmd else str(cmd)
            if cfg_id and label:
                out[cfg_id] = normalize_engine(label)
        if out:
            return out
    return {}


def _engine_from_result(r, config_labels: Dict[str, str], n_configs: int = 0) -> str:
    """Resolve a canonical engine name for one loaded shootout result."""
    if getattr(r, "label", None):
        return normalize_engine(r.label)
    if getattr(r, "command", None):
        return normalize_engine(r.command.split()[0] if r.command.split() else r.command)
    cfg = str(getattr(r, "config", "") or "").upper()
    if cfg in config_labels:
        return config_labels[cfg]
    # Fallback for older logs that only recorded A/B/C/... without labels.
    # 4-way (current Makefile): rpyforth, gforth-fast, vfxforth, swiftforth
    # 5-way (legacy):          rpyforth, gforth-fast, gforth, vfxforth, swiftforth
    letter_map_4 = {
        "A": PRIMARY_ENGINES[0],
        "B": PRIMARY_ENGINES[1],
        "C": PRIMARY_ENGINES[2],
        "D": PRIMARY_ENGINES[3],
    }
    letter_map_5 = {
        "A": "rpyforth",
        "B": "gforth-fast",
        "C": "gforth",
        "D": "vfxforth",
        "E": "swiftforth",
    }
    letter_map = letter_map_5 if n_configs >= 5 else letter_map_4
    if cfg in letter_map:
        return letter_map[cfg]
    return normalize_engine(cfg)


def _median_curve(runs: List[List[int]]) -> List[int]:
    """Align runs by iteration index and take the per-iteration median."""
    usable = [list(run) for run in runs if run]
    if not usable:
        return []
    length = min(len(run) for run in usable)
    if length <= 0:
        return []
    return [
        int(statistics.median(run[i] for run in usable))
        for i in range(length)
    ]


def _program_name(name: str) -> str:
    if name.startswith("shootout/"):
        name = name[len("shootout/") :]
    if name.startswith("curve/"):
        name = name[len("curve/") :]
    if name.endswith(".fs"):
        name = name[: -len(".fs")]
    return name


def load_shootout(log_dir: Path) -> CurveData:
    """Load shootout curve logs using run_shootout.load_log.

    Only curve/ benchmarks contribute warm-up series. Multiple timed runs of the
    same (program, engine) are reduced to a per-iteration median curve.
    """
    sys.path.insert(0, str(REPO_ROOT / "benchmark"))
    try:
        from jitlog_analysis import is_curve_benchmark
        from run_shootout import load_log
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import run_shootout; is the repository root correct?"
        ) from exc

    config_labels = _load_config_labels(log_dir)
    runs_by_prog_engine: Dict[str, Dict[str, List[List[int]]]] = {}

    loaded = []
    for path in sorted(log_dir.glob("*.log")):
        r = load_log(path)
        if r is None or r.status not in ("ok", "warning"):
            continue
        if not is_curve_benchmark(r.name):
            continue
        loaded.append(r)

    n_configs = len({str(r.config).upper() for r in loaded})

    for r in loaded:
        program = _program_name(r.name)
        engine = _engine_from_result(r, config_labels, n_configs=n_configs)
        times = list(r.curve_times or [])
        if not times and r.curve_runs:
            # Prefer the first non-empty recorded run when curve_times is empty.
            for run in r.curve_runs:
                if run:
                    times = list(run)
                    break
        if not times:
            continue
        runs_by_prog_engine.setdefault(program, {}).setdefault(engine, []).append(times)

    data: CurveData = {}
    for program, by_engine in runs_by_prog_engine.items():
        for engine, runs in by_engine.items():
            med = _median_curve(runs)
            if med:
                data.setdefault(program, {})[engine] = med
    return data


def merge_curves(a: CurveData, b: CurveData) -> CurveData:
    """Merge two curve datasets, with b taking precedence on conflicts."""
    out: CurveData = {}
    for prog in set(a) | set(b):
        out[prog] = {}
        for engine in set(a.get(prog, {})) | set(b.get(prog, {})):
            if engine in b.get(prog, {}):
                out[prog][engine] = b[prog][engine]
            else:
                out[prog][engine] = a[prog][engine]
    return out


def steady_state_tail(times: List[int], frac: float = 0.5) -> Optional[float]:
    if not times:
        return None
    tail = times[int(len(times) * (1.0 - frac)) :] or times
    return float(statistics.median(tail))


SHOOTOUT_PROGRAMS = [
    "ack",
    "ary",
    "callheavy",
    "composite",
    "except",
    "fibo",
    "hash",
    "hash2",
    "heap",
    "hello",
    "lists",
    "matrix",
    "methcall",
    "moments",
    "nestedloop",
    "objinst",
    "random",
    "recurse",
    "reversefile",
    "sieve",
    "spellcheck",
    "strcat",
    "sumcol",
    "wc",
    "wordfreq",
]

APPBENCH_PROGRAMS = [
    "cd16sim",
    "brainless",
    "fcp",
    "benchgc",
    "coremark",
    "lexex",
]


def sort_programs(programs) -> List[str]:
    """Shootout first, then appbench, then any extras alphabetically."""
    preferred = SHOOTOUT_PROGRAMS + APPBENCH_PROGRAMS
    rank = {name: i for i, name in enumerate(preferred)}
    return sorted(programs, key=lambda p: (rank.get(p, 10_000), p))


def discover_appbench_dirs(log_dir: Path) -> List[Path]:
    """Find appbench steady/csv trees near a shootout log directory."""
    found: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        path = path.resolve()
        if path in seen or not path.exists():
            return
        seen.add(path)
        found.append(path)

    add(log_dir)
    add(log_dir / "appbench")

    # Nested layouts: logs/<ts>/<rev>/appbench/...
    if log_dir.is_dir():
        for steady in log_dir.rglob("steady_results.json"):
            add(steady.parent)
        for csv_dir in log_dir.rglob("csv"):
            if csv_dir.is_dir() and any(csv_dir.glob("*.csv")):
                add(csv_dir.parent)

    # Same revision name under other timestamp folders.
    rev = log_dir.name
    logs_root = REPO_ROOT / "logs"
    if logs_root.is_dir():
        for app_dir in logs_root.glob("*/%s/appbench" % rev):
            add(app_dir)
        for app_dir in logs_root.glob("%s/appbench" % rev):
            add(app_dir)

    return found


def discover_warmup_jsons(log_dir: Path) -> List[Path]:
    """Find warmup_curves.json under the given log directory tree."""
    found: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        path = path.resolve()
        if path in seen or not path.is_file():
            return
        seen.add(path)
        found.append(path)

    add(log_dir / "warmup_curves.json")
    if log_dir.is_dir():
        for path in log_dir.rglob("warmup_curves.json"):
            add(path)
    return found

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_warmup_curves(
    data: CurveData,
    out_path: Path,
    title: str = "Warm-up curves",
    logy: bool = True,
) -> None:
    """Write one PDF with every benchmark as a subplot (one line per engine)."""
    programs = sort_programs(data)
    if not programs:
        raise RuntimeError("No warm-up curve data to plot")

    cols = min(3, len(programs))
    rows = (len(programs) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False
    )

    for idx, program in enumerate(programs):
        ax = axes.flatten()[idx]
        by_engine = data[program]
        for engine in sort_engines(by_engine):
            times = by_engine.get(engine)
            if not times:
                continue
            ys = [t / 1000.0 for t in times]
            xs = list(range(1, len(ys) + 1))
            color = engine_color(engine)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                color=color,
                label=engine,
            )
            ss = steady_state_tail(times)
            if ss is not None:
                ax.axhline(
                    ss / 1000.0,
                    color=color,
                    linewidth=0.7,
                    linestyle=":",
                    alpha=0.6,
                )
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(left=0.5)
        ax.set_title(program)
        ax.set_xlabel("iteration")
        ax.set_ylabel("time / iteration (ms%s)" % (", log" if logy else ""))
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(programs), rows * cols):
        axes.flatten()[idx].set_visible(False)

    fig.suptitle(
        "%s\ndotted line = steady-state tail median" % title,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def plot_boxplot(
    data: CurveData,
    out_path: Path,
    title: str = "Cross-engine comparison",
    logy: bool = True,
    baseline: str = "gforth-fast",
) -> None:
    """Overview boxplot: one box per engine, one sample per benchmark.

    Warm-up dynamics stay in plot_warmup_curves (per-benchmark subplots).
    Here each sample is that benchmark's warm-tail median relative to the
    baseline engine's warm-tail median, so every program contributes equally.
    """
    programs = sort_programs(data)
    if not programs:
        raise RuntimeError("No data to plot")

    present = {e for by_engine in data.values() for e in by_engine}
    engines = [e for e in PRIMARY_ENGINES if e in present]
    engines.extend(sorted(e for e in present if e not in PRIMARY_ENGINES))
    if baseline not in present:
        for cand in PRIMARY_ENGINES:
            if cand in present:
                baseline = cand
                break
        else:
            baseline = engines[0]

    boxplot_data: List[List[float]] = []
    colors: List[str] = []
    labels: List[str] = []

    for engine in engines:
        samples: List[float] = []
        for program in programs:
            times = data[program].get(engine)
            base_times = data[program].get(baseline)
            if not times or not base_times:
                continue
            eng_med = steady_state_tail(times)
            base_med = steady_state_tail(base_times)
            if eng_med is None or base_med is None or base_med <= 0:
                continue
            samples.append(float(eng_med) / float(base_med))
        if not samples:
            continue
        boxplot_data.append(samples)
        colors.append(engine_color(engine))
        labels.append(engine)

    if not boxplot_data:
        raise RuntimeError("No runnable (program, engine) pairs to plot")

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(labels)), 5.5))
    positions = list(range(1, len(boxplot_data) + 1))
    bp = ax.boxplot(
        boxplot_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        sym="",
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if logy:
        ax.set_yscale("log")
    ax.axhline(1.0, color="#d75a5a", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_ylabel(
        "warm median / %s%s" % (baseline, (", log" if logy else ""))
    )
    ax.set_title(
        "%s (one box/engine; one sample/benchmark, n≤%d)"
        % (title, len(programs))
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    legend_elements = [
        Patch(facecolor=engine_color(e), alpha=0.7, label=e) for e in labels
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from runbench.sh / run_appbench.sh logs."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Log directory (e.g. logs/<timestamp> or logs/<rev>)",
    )
    parser.add_argument(
        "prefix",
        type=Path,
        nargs="?",
        default=Path("benchmark_report"),
        help="Output file prefix (default: benchmark_report)",
    )
    parser.add_argument(
        "--appbench",
        type=Path,
        default=None,
        help="Override appbench log directory",
    )
    parser.add_argument(
        "--shootout",
        type=Path,
        default=None,
        help="Override shootout log directory",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear y-axis instead of log",
    )
    args = parser.parse_args(argv)

    log_dir = args.log_dir
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir

    appbench_dir = args.appbench
    if appbench_dir is not None and not appbench_dir.is_absolute():
        appbench_dir = REPO_ROOT / appbench_dir

    shootout_dir = args.shootout
    if shootout_dir is not None and not shootout_dir.is_absolute():
        shootout_dir = REPO_ROOT / shootout_dir
    else:
        shootout_dir = log_dir

    shootout_data = load_shootout(shootout_dir)

    appbench_data: CurveData = {}
    if appbench_dir is not None:
        appbench_data = load_appbench(appbench_dir)
    else:
        for cand in discover_appbench_dirs(log_dir):
            appbench_data = merge_curves(appbench_data, load_appbench(cand))

    data = merge_curves(appbench_data, shootout_data)
    for warmup_json in discover_warmup_jsons(log_dir):
        data = merge_curves(data, load_warmup_curves_json(warmup_json))

    if not data:
        print("No benchmark data found in %s" % log_dir, file=sys.stderr)
        return 1

    prefix = args.prefix
    if not prefix.is_absolute():
        prefix = log_dir / prefix

    curve_path = prefix.parent / (prefix.name + "_warmup_curves.pdf")
    boxplot_path = prefix.parent / (prefix.name + "_boxplot.pdf")
    logy = not args.linear

    shootout_names = [p for p in sort_programs(data) if p in SHOOTOUT_PROGRAMS or p in shootout_data]
    appbench_names = [p for p in sort_programs(data) if p in APPBENCH_PROGRAMS]

    try:
        plot_warmup_curves(data, curve_path, logy=logy)
        print("Warm-up curves written to %s" % curve_path)
        print(
            "Programs (%d): shootout=%d appbench=%d -> %s"
            % (
                len(data),
                len(shootout_names),
                len(appbench_names),
                ", ".join(sort_programs(data)),
            )
        )
    except RuntimeError as exc:
        print("ERROR generating warm-up curves: %s" % exc, file=sys.stderr)
        return 1

    try:
        plot_boxplot(data, boxplot_path, logy=logy)
        print("Boxplot written to %s" % boxplot_path)
    except RuntimeError as exc:
        print("ERROR generating boxplot: %s" % exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
