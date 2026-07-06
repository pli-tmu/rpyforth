#!/usr/bin/env python3
"""All-benchmarks warm-up curve harness: shootout (13) + appbench (5).

Produces a single PDF (/tmp/warmup_all.pdf by default) with 18 subplots
showing per-iteration time from cold to plateau for every benchmark and all
three engines (rpyforth-c-stkfrag / gforth-fast / gforth).

Appbench specs and helpers are imported from run_appbench_steady.py.

Shootout strategy: each benchmark is a self-contained script that ends with
`bye`.  A driver redefines `bye` as a no-op and `(bye)` as the real exit,
then loops N times timing `s" <abs-path>/<bench>.fs" included` via UTIME.
Re-inclusion re-defines all words (gforth warns; warnings are suppressed) and
re-runs the workload identically each time.  Memory allocations accumulate
across iterations (ary.fs, heap.fs) but RSS growth is bounded (small allocs)
and timing is stable.

Misbehaving benchmarks are marked and excluded from the chart with a note.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# --- import helpers from run_appbench_steady ------------------------------------
# We import the shared primitives and the PROGRAMS list rather than duplicating.
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from run_appbench_steady import (
    PROGRAMS as APPBENCH_PROGRAMS,
    ENGINE_RPYFORTH,
    ENGINE_GFORTH_FAST,
    ENGINE_GFORTH,
    ENGINES,
    ENGINE_BINARY,
    GFORTH_SETUP,
    REPO_ROOT,
    build_driver as appbench_build_driver,
    parse_curve_output,
    steady_state_tail,
    fmt_usec,
)

# ---------------------------------------------------------------------------
# Shootout configuration
# ---------------------------------------------------------------------------

SHOOTOUT_DIR = REPO_ROOT / "shootout"

# All .fs files in shootout/ except the curve/ subdirectory
def _discover_shootout_benches():
    found = sorted(
        p for p in SHOOTOUT_DIR.glob("*.fs")
        if not p.name.startswith(".")
    )
    return found

SHOOTOUT_FILES = _discover_shootout_benches()

# Benchmarks that are known to misbehave with re-inclusion.
# Dictionary overflow or state carryover that causes degrading times or errors.
SHOOTOUT_EXCLUDED = {
    # composite.fs re-allocates big float arrays (heap-base) via `allocate` each
    # iteration -- each call leaks ~64 KB, and after ~1000 iterations on gforth
    # the C heap can OOM.  The per-iteration time also grows slightly as the
    # malloc arena grows.  Exclude to avoid confusion; the component benchmarks
    # (ack, fibo, sieve, etc.) all appear individually.
    # NOTE: actually tested 30 iters and it works fine; RSS growth is tiny.
    # Keep it INCLUDED.
}

# Per-engine gforth memory size for shootout (16M is enough for all of them)
SHOOTOUT_GFORTH_MEM = "64M"

# ---------------------------------------------------------------------------
# Shootout driver builder
# ---------------------------------------------------------------------------

SHOOTOUT_DRIVER_PREAMBLE = """\
: warnings 2drop ;
: (bye) bye ;
: bye ;
"""


def build_shootout_driver(bench_path, iterations):
    """Return Forth source for a driver that times `bench_path` `iterations` times.

    Strategy:
      - Redefine `bye` as a no-op so the script's trailing `bye` does not exit.
      - Define `(bye)` as the real exit word (gforth/rpyforth compatible).
      - Define `run-one-iter` to time a single `included` call.
      - Define `run-bench` as a DO loop calling `run-one-iter` and printing CSV.

    CSV format: `<i> ,<elapsed_usec>` -- matches parse_curve_output convention
    (parts[0].isdigit(), parts[1] parseable as int).

    A leading CR before the CSV line separates it from any output the benchmark
    itself printed on the same line; a trailing CR keeps the next iteration's
    output from appending to this data line.
    """
    abs_path = str(bench_path.resolve())
    lines = [SHOOTOUT_DRIVER_PREAMBLE.strip(), ""]
    lines.append(": run-one-iter ( -- usec )")
    lines.append("  utime 2>r")
    lines.append('  s" %s" included' % abs_path)
    lines.append("  utime 2r> d-")
    lines.append("  drop ;")
    lines.append("")
    lines.append(": run-bench ( n -- )")
    lines.append("  0 do")
    lines.append("    run-one-iter")
    lines.append('    cr I . ." ," . cr')
    lines.append("  loop ;")
    lines.append("")
    lines.append("%d run-bench" % iterations)
    lines.append("(bye)")
    return "\n".join(lines) + "\n"


def build_shootout_cmd(engine, driver_path):
    binary = ENGINE_BINARY[engine]
    if engine == ENGINE_RPYFORTH:
        return [str(binary), str(driver_path)]
    # gforth / gforth-fast: no appbench setup file needed for shootout -- plain
    # script with warnings suppressed in the driver itself.
    return [str(binary), "-m", SHOOTOUT_GFORTH_MEM, str(driver_path)]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DEFAULT_ITERATIONS = 30
LEXEX_ITERATIONS = 15
DEFAULT_TIMEOUT = 600
DEFAULT_PIN = 3
DEFAULT_PDF = "/tmp/warmup_all.pdf"


def run_engine(engine, driver_path, workdir, iterations, timeout, pin, rpy_env=None):
    """Run a single engine on a pre-built driver file. Returns a result dict."""
    binary = ENGINE_BINARY[engine]
    if not Path(binary).exists():
        return {"engine": engine, "times": [], "wall": 0, "rc": -1,
                "timed_out": False, "stderr": "binary not found", "skipped": True}

    cmd = build_shootout_cmd(engine, driver_path)
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    env = os.environ.copy()
    if engine == ENGINE_RPYFORTH and rpy_env:
        env.update(rpy_env)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = -1, True

    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")

    times = parse_curve_output(stdout)
    return {
        "engine": engine,
        "times": times,
        "wall": wall,
        "rc": rc,
        "timed_out": timed_out,
        "stderr": stderr,
        "cmd": cmd,
        "skipped": False,
    }


def run_appbench_engine(engine, spec, iterations, tmpdir, timeout, pin):
    """Wrap appbench run: build driver, write to tmpdir, run."""
    from run_appbench_steady import build_driver as appbench_build_driver, build_cmd as appbench_build_cmd

    driver_src = appbench_build_driver(spec, iterations)
    driver_path = Path(tmpdir) / ("%s_%s_driver.fs" % (spec.name, engine))
    driver_path.write_text(driver_src, encoding="utf-8")

    binary = ENGINE_BINARY[engine]
    if not Path(binary).exists():
        return {"engine": engine, "times": [], "wall": 0, "rc": -1,
                "timed_out": False, "stderr": "binary not found", "skipped": True}

    cmd = appbench_build_cmd(engine, driver_path, spec)
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    env = os.environ.copy()
    if engine == ENGINE_RPYFORTH and spec.rpy_env:
        env.update(spec.rpy_env)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
            cwd=str(spec.workdir), stdin=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = -1, True

    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")

    times = parse_curve_output(stdout)
    return {
        "engine": engine,
        "times": times,
        "wall": wall,
        "rc": rc,
        "timed_out": timed_out,
        "stderr": stderr,
        "cmd": cmd,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def convergence_iteration(times, threshold=0.10):
    """Return the first iteration index where the time enters a band within
    `threshold` fraction of the steady-state tail median and stays there.

    Returns None if the curve never converges or if there are fewer than 4 data
    points.
    """
    if len(times) < 4:
        return None
    warm = steady_state_tail(times)
    if warm is None or warm == 0:
        return None
    band_lo = warm * (1.0 - threshold)
    band_hi = warm * (1.0 + threshold)
    # Find the first index i such that all times[i:] are within the band.
    n = len(times)
    for start in range(n):
        if all(band_lo <= t <= band_hi for t in times[start:]):
            return start
    return None


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

COLORS = {
    ENGINE_RPYFORTH:   "#d62728",
    ENGINE_GFORTH_FAST: "#1f77b4",
    ENGINE_GFORTH:     "#2ca02c",
}

EXTRA_COLORS = [
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
]

ENGINE_LABELS = {
    ENGINE_RPYFORTH:    "rpyforth-c-stkfrag",
    ENGINE_GFORTH_FAST: "gforth-fast",
    ENGINE_GFORTH:      "gforth",
}


def make_all_chart(results, pdf_path, extra_engines=None):
    """Build an 18-subplot grid (4 cols) of warm-up curves for all benchmarks.

    `results` is an ordered list of (bench_name, eng_results_dict) where
    eng_results_dict maps engine key -> run_dict.
    `extra_engines` is an optional list of (engine_key, label) for additional
    rpyforth variants added via --extra-rpyforth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import math

    if extra_engines is None:
        extra_engines = []

    all_engine_keys = list(ENGINES) + [k for k, _label in extra_engines]

    color_map = dict(COLORS)
    for i, (k, _label) in enumerate(extra_engines):
        color_map[k] = EXTRA_COLORS[i % len(EXTRA_COLORS)]

    label_map = dict(ENGINE_LABELS)
    for k, label in extra_engines:
        label_map[k] = label

    n = len(results)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2),
                             squeeze=False)

    for idx, (bench_name, eng_res) in enumerate(results):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        all_times_ms = []
        for engine in all_engine_keys:
            r = eng_res.get(engine)
            if not r or not r["times"]:
                continue
            times_ms = [t / 1000.0 for t in r["times"]]
            all_times_ms.extend(times_ms)
            xs = list(range(len(times_ms)))
            ax.plot(xs, times_ms, marker=".", markersize=2.5, linewidth=1.0,
                    color=color_map[engine], label=label_map[engine])
            warm = steady_state_tail(r["times"])
            if warm is not None:
                ax.axhline(warm / 1000.0, color=color_map[engine], linewidth=0.7,
                           linestyle=":", alpha=0.7)

        # Use log y-axis if range > 10x
        if all_times_ms:
            lo = min(all_times_ms)
            hi = max(all_times_ms)
            if lo > 0 and hi / lo > 10.0:
                ax.set_yscale("log")

        ax.set_title(bench_name, fontsize=9, fontweight="bold")
        ax.set_xlabel("iter (0=cold)", fontsize=7)
        ax.set_ylabel("time (ms)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    # Shared legend (proxy artists)
    legend_handles = [
        mlines.Line2D([], [], color=color_map[e], linewidth=1.5, label=label_map[e])
        for e in all_engine_keys
    ]
    legend_handles.append(
        mlines.Line2D([], [], color="grey", linewidth=0.7, linestyle=":",
                      alpha=0.7, label="warm-tail median")
    )
    ncol_legend = min(4, len(legend_handles))
    fig.legend(handles=legend_handles, loc="lower center", ncol=ncol_legend,
               fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Warm-up curves: rpyforth (meta-tracing JIT) vs gforth-fast / gforth\n"
        "13 shootout kernels + 5 appbench programs | dotted = warm-tail median",
        fontsize=10, fontweight="bold")

    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(str(pdf_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_extra_shootout(extra_binary, extra_label, bench_path, iters, tmpdir,
                         timeout, pin):
    """Run an extra rpyforth binary on a shootout benchmark."""
    driver_src = build_shootout_driver(bench_path, iters)
    driver_path = Path(tmpdir) / ("%s_%s_driver.fs" % (bench_path.stem, extra_label))
    driver_path.write_text(driver_src, encoding="utf-8")

    if not Path(extra_binary).exists():
        return {"engine": extra_label, "times": [], "wall": 0, "rc": -1,
                "timed_out": False, "stderr": "binary not found", "skipped": True}

    cmd = [str(extra_binary), str(driver_path)]
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=os.environ.copy(), cwd=str(SHOOTOUT_DIR),
            stdin=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = -1, True

    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")

    times = parse_curve_output(stdout)
    return {
        "engine": extra_label,
        "times": times,
        "wall": wall,
        "rc": rc,
        "timed_out": timed_out,
        "stderr": stderr,
        "skipped": False,
    }


def _run_extra_appbench(extra_binary, extra_label, spec, iters, tmpdir, timeout, pin):
    """Run an extra rpyforth binary on an appbench benchmark."""
    from run_appbench_steady import build_driver as appbench_build_driver, build_cmd as appbench_build_cmd

    if not Path(extra_binary).exists():
        return {"engine": extra_label, "times": [], "wall": 0, "rc": -1,
                "timed_out": False, "stderr": "binary not found", "skipped": True}

    driver_src = appbench_build_driver(spec, iters)
    driver_path = Path(tmpdir) / ("%s_%s_driver.fs" % (spec.name, extra_label))
    driver_path.write_text(driver_src, encoding="utf-8")

    # build_cmd uses ENGINE_RPYFORTH key to select rpyforth-style command;
    # we override the binary path by building the cmd manually.
    cmd = [str(extra_binary), str(driver_path)]
    if spec.rpy_env:
        env = os.environ.copy()
        env.update(spec.rpy_env)
    else:
        env = os.environ.copy()
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=env, cwd=str(spec.workdir), stdin=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = -1, True

    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")

    times = parse_curve_output(stdout)
    return {
        "engine": extra_label,
        "times": times,
        "wall": wall,
        "rc": rc,
        "timed_out": timed_out,
        "stderr": stderr,
        "skipped": False,
    }


def _dump_json(results, iterations, json_path, extra_engines):
    """Dump warm data JSON.

    Schema:
    {
      "iterations": N,
      "benchmarks": {
        "<name>": {
          "<engine_label>": {
            "times_usec": [...],
            "cold_usec": <first>,
            "warm_median_usec": <median-of-last-50%>   # null if no data
          }
        }
      }
    }
    """
    all_engine_keys = list(ENGINES) + [k for k, _label in extra_engines]
    label_map = dict(ENGINE_LABELS)
    for k, label in extra_engines:
        label_map[k] = label

    benchmarks = {}
    for name, eng_res in results:
        bench_entry = {}
        for engine in all_engine_keys:
            r = eng_res.get(engine)
            label = label_map.get(engine, engine)
            if r is None or r.get("skipped") or r.get("timed_out") or not r["times"]:
                bench_entry[label] = {
                    "times_usec": [],
                    "cold_usec": None,
                    "warm_median_usec": None,
                }
            else:
                times = r["times"]
                warm = steady_state_tail(times)
                bench_entry[label] = {
                    "times_usec": times,
                    "cold_usec": times[0],
                    "warm_median_usec": warm,
                }
        benchmarks[name] = bench_entry

    payload = {"iterations": iterations, "benchmarks": benchmarks}
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Warm data JSON written to %s" % json_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="iterations per benchmark (default %d; lexex uses 15)"
                             % DEFAULT_ITERATIONS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-run timeout in seconds (default %d)" % DEFAULT_TIMEOUT)
    parser.add_argument("--pin", type=int, default=DEFAULT_PIN,
                        help="CPU core to pin via taskset -c (default %d)" % DEFAULT_PIN)
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF,
                        help="output PDF path (default %s)" % DEFAULT_PDF)
    parser.add_argument("--json", type=str, default=None, dest="json_path",
                        help="dump warm data JSON to this path")
    parser.add_argument("--extra-rpyforth", action="append", default=[],
                        metavar="PATH:LABEL", dest="extra_rpyforth",
                        help="additional rpyforth binary to run (repeatable); "
                             "format PATH:LABEL")
    parser.add_argument("--shootout-only", action="store_true",
                        help="run only shootout benchmarks (skip appbench)")
    parser.add_argument("--appbench-only", action="store_true",
                        help="run only appbench benchmarks (skip shootout)")
    args = parser.parse_args(argv)

    # Parse extra rpyforth entries: PATH:LABEL
    extra_engines = []
    for spec_str in args.extra_rpyforth:
        if ":" not in spec_str:
            print("ERROR: --extra-rpyforth must be PATH:LABEL, got: %s" % spec_str,
                  file=sys.stderr)
            return 1
        colon = spec_str.index(":")
        path_str = spec_str[:colon]
        label = spec_str[colon + 1:]
        # Resolve to absolute path so it works regardless of subprocess cwd.
        path_str = str(Path(path_str).resolve())
        extra_engines.append((path_str, label))

    # Print missing binaries early
    for engine in ENGINES:
        b = ENGINE_BINARY[engine]
        if not Path(b).exists():
            print("WARNING: engine binary missing: %s (%s)" % (engine, b),
                  file=sys.stderr)
    for path_str, label in extra_engines:
        if not Path(path_str).exists():
            print("WARNING: extra engine binary missing: %s (%s)" % (label, path_str),
                  file=sys.stderr)

    # Collect shootout benchmarks (excluding known-bad)
    shootout_benches = [
        p for p in SHOOTOUT_FILES
        if p.stem not in SHOOTOUT_EXCLUDED
    ]

    excluded_notes = {}
    for name in SHOOTOUT_EXCLUDED:
        excluded_notes[name] = SHOOTOUT_EXCLUDED[name] if isinstance(SHOOTOUT_EXCLUDED, dict) else "excluded"

    results = []  # list of (name, {engine_key: run_dict})

    with tempfile.TemporaryDirectory(prefix="warmup_all_") as tmpdir:

        # --- Shootout benchmarks ---
        if not args.appbench_only:
            for bench_path in shootout_benches:
                name = bench_path.stem
                eng_res = {}
                iters = args.iterations

                print("[shootout] %s  (%d iterations)" % (name, iters))

                for engine in ENGINES:
                    if not Path(ENGINE_BINARY[engine]).exists():
                        continue
                    driver_src = build_shootout_driver(bench_path, iters)
                    driver_path = Path(tmpdir) / ("%s_%s_driver.fs" % (name, engine))
                    driver_path.write_text(driver_src, encoding="utf-8")

                    print("  %-20s ... " % engine, end="", flush=True)
                    r = run_engine(engine, driver_path, SHOOTOUT_DIR, iters,
                                   args.timeout, args.pin)
                    warm = steady_state_tail(r["times"])
                    if r.get("skipped"):
                        print("SKIP (binary missing)")
                    elif r["timed_out"]:
                        print("TIMEOUT after %.1fs" % r["wall"])
                    elif not r["times"]:
                        print("NO CSV DATA (rc=%d)" % r["rc"])
                        if r["stderr"].strip():
                            print("    stderr: %s" % r["stderr"].strip()[:200])
                    else:
                        conv = convergence_iteration(r["times"])
                        print("%d iters, cold=%s warm=%s conv@%s (%.1fs)" % (
                            len(r["times"]),
                            fmt_usec(r["times"][0]),
                            fmt_usec(warm),
                            str(conv) if conv is not None else "n/a",
                            r["wall"],
                        ))
                    eng_res[engine] = r

                # Extra engines for shootout
                for path_str, label in extra_engines:
                    print("  %-20s ... " % label, end="", flush=True)
                    r = _run_extra_shootout(path_str, label, bench_path, iters,
                                            tmpdir, args.timeout, args.pin)
                    warm = steady_state_tail(r["times"])
                    if r.get("skipped"):
                        print("SKIP (binary missing)")
                    elif r["timed_out"]:
                        print("TIMEOUT after %.1fs" % r["wall"])
                    elif not r["times"]:
                        print("NO CSV DATA (rc=%d)" % r["rc"])
                        if r["stderr"].strip():
                            print("    stderr: %s" % r["stderr"].strip()[:200])
                    else:
                        conv = convergence_iteration(r["times"])
                        print("%d iters, cold=%s warm=%s conv@%s (%.1fs)" % (
                            len(r["times"]),
                            fmt_usec(r["times"][0]),
                            fmt_usec(warm),
                            str(conv) if conv is not None else "n/a",
                            r["wall"],
                        ))
                    eng_res[path_str] = r

                results.append((name, eng_res))

        # --- Appbench programs ---
        if not args.shootout_only:
            for spec in APPBENCH_PROGRAMS:
                iters = LEXEX_ITERATIONS if spec.name == "lexex" else args.iterations
                eng_res = {}

                print("[appbench] %s  (%d iterations)" % (spec.name, iters))

                for engine in ENGINES:
                    if not Path(ENGINE_BINARY[engine]).exists():
                        continue
                    print("  %-20s ... " % engine, end="", flush=True)
                    r = run_appbench_engine(engine, spec, iters, tmpdir,
                                            args.timeout, args.pin)
                    warm = steady_state_tail(r["times"])
                    if r.get("skipped"):
                        print("SKIP (binary missing)")
                    elif r["timed_out"]:
                        print("TIMEOUT after %.1fs" % r["wall"])
                    elif not r["times"]:
                        print("NO CSV DATA (rc=%d)" % r["rc"])
                        if r["stderr"].strip():
                            print("    stderr: %s" % r["stderr"].strip()[:200])
                    else:
                        conv = convergence_iteration(r["times"])
                        print("%d iters, cold=%s warm=%s conv@%s (%.1fs)" % (
                            len(r["times"]),
                            fmt_usec(r["times"][0]),
                            fmt_usec(warm),
                            str(conv) if conv is not None else "n/a",
                            r["wall"],
                        ))
                    eng_res[engine] = r

                # Extra engines for appbench
                for path_str, label in extra_engines:
                    print("  %-20s ... " % label, end="", flush=True)
                    r = _run_extra_appbench(path_str, label, spec, iters,
                                            tmpdir, args.timeout, args.pin)
                    warm = steady_state_tail(r["times"])
                    if r.get("skipped"):
                        print("SKIP (binary missing)")
                    elif r["timed_out"]:
                        print("TIMEOUT after %.1fs  [null]" % r["wall"])
                    elif not r["times"]:
                        print("NO CSV DATA (rc=%d)" % r["rc"])
                        if r["stderr"].strip():
                            print("    stderr: %s" % r["stderr"].strip()[:200])
                    else:
                        conv = convergence_iteration(r["times"])
                        print("%d iters, cold=%s warm=%s conv@%s (%.1fs)" % (
                            len(r["times"]),
                            fmt_usec(r["times"][0]),
                            fmt_usec(warm),
                            str(conv) if conv is not None else "n/a",
                            r["wall"],
                        ))
                    eng_res[path_str] = r

                results.append((spec.name, eng_res))

    # --- Convergence table (rpyforth) ---
    print("\n" + "=" * 72)
    print("CONVERGENCE TABLE  (rpyforth-c-stkfrag, within 10% of warm-tail)")
    print("=" * 72)
    print("%-14s  %8s  %12s  %12s  %8s" % (
        "benchmark", "conv@iter", "cold", "warm-tail", "speedup"))
    print("-" * 72)
    for name, eng_res in results:
        r = eng_res.get(ENGINE_RPYFORTH)
        if not r or not r["times"]:
            print("%-14s  %8s" % (name, "NO DATA"))
            continue
        times = r["times"]
        conv = convergence_iteration(times)
        warm = steady_state_tail(times)
        cold = times[0]
        speedup = ("%.2fx" % (cold / float(warm))) if warm and warm > 0 else "-"
        print("%-14s  %8s  %12s  %12s  %8s" % (
            name,
            str(conv) if conv is not None else "n/a",
            fmt_usec(cold),
            fmt_usec(warm),
            speedup,
        ))
    print("-" * 72)
    if excluded_notes:
        print("\nEXCLUDED BENCHMARKS:")
        for name, reason in excluded_notes.items():
            print("  %-14s  %s" % (name, reason))

    # --- JSON dump ---
    if args.json_path:
        _dump_json(results, args.iterations, args.json_path, extra_engines)

    # --- Chart ---
    pdf_path = Path(args.pdf)
    try:
        make_all_chart(results, pdf_path, extra_engines=extra_engines)
        print("\nWarm-up curve chart written to %s" % pdf_path)
    except Exception as exc:
        import traceback
        print("ERROR generating chart: %s" % exc, file=sys.stderr)
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
