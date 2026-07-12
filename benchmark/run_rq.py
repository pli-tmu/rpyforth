#!/usr/bin/env python3
"""Research-question verification harness for the appbench suite.

Four research questions, selected by subcommand (or `all`):

  rq1  trace affinity : does JIT trace behaviour predict speedup-vs-gforth?
  rq2  gc vs jit      : how much wall time is spent in GC per program?
  rq3  warmup economics: after how many iterations does rpyforth's cumulative
                          time overtake an AOT engine (break-even)?
  rq4  methodology     : warm-tail drift, coverage matrix, survivorship bias.

rq1 and rq2 run the rpyforth binary under PYPYLOG (jit-summary / gc) and must
have a working binary; rq3 and rq4 read only a steady_results.json produced by
run_appbench.py's steady mode and do no binary runs.

Every RQ joins against a steady_results.json (--steady-json, or the newest one
auto-discovered under logs/). The steady JSON schema is the one written by
run_appbench._save_steady_logs: a top-level dict with `iterations`, `engines`,
and `results` (a list of {program, engine, times[], warm_median_usec,
cold_usec, timed_out, ...}).

Outputs land under logs/rq/<git-rev>/ (mirroring run_appbench's log dir):
  rq_report.txt   human-readable, one section per RQ.
  rq_results.json all raw + derived numbers.
  rq_charts.pdf   multi-panel chart (only if matplotlib is importable).

Safety: never modifies appbench/appbench-1.4/. PYPYLOG runs reuse the same
driver builder as steady mode, so the workload is identical.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import run_appbench as ab
from run_appbench import (
    ENGINE_RPYFORTH,
    ENGINE_GFORTH_FAST,
    ENGINE_VFXFORTH,
    ENGINE_SWIFTFORTH,
    PROGRAMS,
    REPO_ROOT,
    build_driver,
    build_cmd,
    prepare_engine_workdir,
    with_workdir,
    git_revision,
    capture_environment,
)
from jitlog_analysis import parse_jit_summary_text

# AOT engines rpyforth is raced against for break-even (rq3).
AOT_ENGINES = [ENGINE_VFXFORTH, ENGINE_SWIFTFORTH]
# All non-rpyforth engines for the survivorship / coverage analysis (rq4).
OTHER_ENGINES = [ENGINE_GFORTH_FAST, ENGINE_VFXFORTH, ENGINE_SWIFTFORTH]

DEFAULT_ITERATIONS = 10
GC_FRACTION_FLAG = 0.20
DRIFT_FLAG = 0.5  # percent per iteration

# The `gc` section prints both minor and major collects with `time taken:` lines;
# gc-collect alone is empty on this build. Probed variants, in preference order.
GC_SECTIONS = ["gc-collect", "gc", "gc-minor"]

GC_TIME_RE = re.compile(r"time taken:\s*([0-9.]+)")

# jit-backend-counts (dumped at process exit) attributes an execution count to
# each assembled loop and bridge. Format probed on this build:
#   entry N:<count>              entry-point counters (excluded from the ratio)
#   TargetToken(<addr>):<count>  a compiled loop's run count
#   bridge <addr>:<count>        a compiled bridge's run count
BACKEND_LOOP_RE = re.compile(r"^TargetToken\([0-9]+\):(\d+)", re.MULTILINE)
BACKEND_BRIDGE_RE = re.compile(r"^bridge [0-9]+:(\d+)", re.MULTILINE)


def parse_backend_counts(text):
    """Return per-loop and per-bridge execution counts from a jit-backend-counts
    dump. Returns (loop_exec_total, bridge_exec_total, n_loops, n_bridges,
    bridge_exec_fraction). bridge_exec_fraction is None when no loop or bridge
    ran (nothing to attribute)."""
    loop_counts = [int(m) for m in BACKEND_LOOP_RE.findall(text)]
    bridge_counts = [int(m) for m in BACKEND_BRIDGE_RE.findall(text)]
    loop_total = sum(loop_counts)
    bridge_total = sum(bridge_counts)
    denom = loop_total + bridge_total
    frac = (bridge_total / float(denom)) if denom else None
    return loop_total, bridge_total, len(loop_counts), len(bridge_counts), frac


# ===========================================================================
# steady JSON helpers
# ===========================================================================

def find_newest_steady_json(logs_root):
    """Return the most recently modified steady_results.json under logs/."""
    candidates = list(logs_root.rglob("steady_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_steady(path):
    """Return {program: {engine: result_dict}} plus the raw summary."""
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    by_prog = {}
    for r in summary.get("results", []):
        by_prog.setdefault(r["program"], {})[r["engine"]] = r
    return summary, by_prog


def warm_median(result):
    """Warm-tail median for a steady result, recomputing if absent."""
    if result is None:
        return None
    wm = result.get("warm_median_usec")
    if wm is not None:
        return wm
    return ab.steady_state_tail(result.get("times", []))


def speedup_vs(by_prog, prog, ref_engine=ENGINE_GFORTH_FAST):
    """warm(ref) / warm(rpyforth): >1 means rpyforth is faster than ref."""
    engines = by_prog.get(prog, {})
    rpy = warm_median(engines.get(ENGINE_RPYFORTH))
    ref = warm_median(engines.get(ref_engine))
    if not rpy or not ref:
        return None
    return ref / float(rpy)


# ===========================================================================
# statistics (inline; scipy is not guaranteed to be installed)
# ===========================================================================

def _rankdata(values):
    """Average-rank of each value (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation of paired samples. None if < 3 pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    rx = _rankdata([p[0] for p in pairs])
    ry = _rankdata([p[1] for p in pairs])
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def linfit_slope(ys):
    """Least-squares slope of ys against index 0..n-1. None if < 2 points."""
    n = len(ys)
    if n < 2:
        return None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def geomean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ===========================================================================
# RQ1: trace affinity
# ===========================================================================

def _run_pypylog(spec, iterations, tmpdir, timeout, pin, section, log_path):
    """Run rpyforth on `spec` with PYPYLOG=<section>:<log_path>. Returns
    (returncode, wall, stderr, timed_out); log written to log_path."""
    run_spec = spec
    patched = prepare_engine_workdir(ENGINE_RPYFORTH, spec, tmpdir)
    if patched != Path(spec.workdir):
        run_spec = with_workdir(spec, patched)

    driver = build_driver(run_spec, iterations, ENGINE_RPYFORTH)
    driver_path = Path(tmpdir) / ("%s_rq_%s_driver.fs" % (spec.name, section))
    driver_path.write_text(driver, encoding="utf-8")

    cmd = build_cmd(ENGINE_RPYFORTH, driver_path, run_spec)
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    env = os.environ.copy()
    if run_spec.rpy_env:
        env.update(run_spec.rpy_env)
    env["PYPYLOG"] = "%s:%s" % (section, log_path)

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=str(run_spec.workdir), stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    timed_out = False
    try:
        _out, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        _out, stderr = proc.communicate()
        rc = -1
    wall = time.perf_counter() - t0
    return rc, wall, stderr or "", timed_out


def run_rq1(specs, by_prog, iterations, tmpdir, timeout, pin):
    """Per program: jit-summary metrics + derived ratios, joined with speedup."""
    rows = []
    for spec in specs:
        log_path = str(Path(tmpdir) / ("%s.jitsummary" % spec.name))
        print("  rq1 %-10s jit-summary ..." % spec.name, end="", flush=True)
        rc, wall, stderr, timed_out = _run_pypylog(
            spec, iterations, tmpdir, timeout, pin, "jit-summary", log_path)
        text = ""
        if Path(log_path).exists():
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        m = parse_jit_summary_text(text) if text else None

        loops = (m.loops if m else None) or 0
        bridges = (m.bridges if m else None) or 0
        aborts = (m.abort_total if m else None) or 0
        tracing = m.tracing_time_sec if m else None
        backend = m.backend_time_sec if m else None
        total = m.total_time_sec if m else None

        trace_frac = None
        if total and wall > 0:
            trace_frac = total / wall
        aborts_per_loop = (aborts / float(loops)) if loops else None
        bridges_per_loop = (bridges / float(loops)) if loops else None
        speedup = speedup_vs(by_prog, spec.name)

        counts_path = str(Path(tmpdir) / ("%s.backendcounts" % spec.name))
        print("  rq1 %-10s backend-counts ..." % spec.name,
              end="", flush=True)
        crc, cwall, cstderr, ctimed_out = _run_pypylog(
            spec, iterations, tmpdir, timeout, pin,
            "jit-backend-counts", counts_path)
        ctext = ""
        if Path(counts_path).exists():
            ctext = Path(counts_path).read_text(
                encoding="utf-8", errors="replace")
        (loop_exec, bridge_exec, n_loops_exec, n_bridges_exec,
         bridge_exec_fraction) = parse_backend_counts(ctext)
        have_counts = bool(ctext) and (loop_exec + bridge_exec) > 0

        rows.append({
            "program": spec.name,
            "loops": loops,
            "bridges": bridges,
            "aborts": aborts,
            "tracing_time_sec": tracing,
            "backend_time_sec": backend,
            "jit_total_time_sec": total,
            "wall_seconds": wall,
            "tracing_time_fraction": trace_frac,
            "aborts_per_loop": aborts_per_loop,
            "bridges_per_loop": bridges_per_loop,
            "loop_exec_count": loop_exec,
            "bridge_exec_count": bridge_exec,
            "n_loops_executed": n_loops_exec,
            "n_bridges_executed": n_bridges_exec,
            "bridge_exec_fraction": bridge_exec_fraction,
            "have_backend_counts": have_counts,
            "speedup_vs_gforth": speedup,
            "returncode": rc,
            "timed_out": timed_out,
            "have_summary": m is not None,
        })
        if m is None:
            print(" NO SUMMARY (rc=%d)" % rc)
        else:
            print(" loops=%d bridges=%d aborts=%d total=%.4fs"
                  " brdg-exec-frac=%s (%.1fs)" % (
                      loops, bridges, aborts, total or 0.0,
                      _fmt(bridge_exec_fraction, "%.3f"), wall))

    metrics = [
        ("tracing_time_fraction", "tracing-time fraction"),
        ("aborts_per_loop", "aborts per loop"),
        ("bridges_per_loop", "bridges per loop"),
        ("bridge_exec_fraction", "bridge exec fraction"),
        ("loops", "loop count"),
    ]
    correlations = {}
    speeds = [r["speedup_vs_gforth"] for r in rows]
    for key, _label in metrics:
        correlations[key] = spearman([r[key] for r in rows], speeds)

    return {"rows": rows, "metrics": metrics, "correlations": correlations}


# ===========================================================================
# RQ2: GC vs JIT contribution
# ===========================================================================

def _probe_gc_section(spec, tmpdir, timeout, pin):
    """Return the first GC PYPYLOG section that produces `time taken:` lines,
    or None if none do on this build."""
    for section in GC_SECTIONS:
        log_path = str(Path(tmpdir) / ("gcprobe_%s.log" % section))
        _run_pypylog(spec, 1, tmpdir, timeout, pin, section, log_path)
        if Path(log_path).exists():
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            if GC_TIME_RE.search(text):
                return section
    return None


def run_rq2(specs, iterations, tmpdir, timeout, pin):
    """Per program: sum GC pause time, report GC fraction of wall time."""
    section = None
    if specs:
        section = _probe_gc_section(specs[0], tmpdir, timeout, pin)
    if section is None:
        print("  rq2: no GC PYPYLOG section produced output on this build")
        return {"section": None, "rows": [], "available": False}

    print("  rq2: using PYPYLOG section '%s'" % section)
    rows = []
    for spec in specs:
        log_path = str(Path(tmpdir) / ("%s.gc.log" % spec.name))
        print("  rq2 %-10s gc ..." % spec.name, end="", flush=True)
        rc, wall, stderr, timed_out = _run_pypylog(
            spec, iterations, tmpdir, timeout, pin, section, log_path)
        gc_time = 0.0
        n_collects = 0
        if Path(log_path).exists():
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            for match in GC_TIME_RE.finditer(text):
                gc_time += float(match.group(1))
                n_collects += 1
        gc_frac = (gc_time / wall) if wall > 0 else None
        dominated = gc_frac is not None and gc_frac > GC_FRACTION_FLAG
        rows.append({
            "program": spec.name,
            "gc_pause_seconds": gc_time,
            "gc_collects": n_collects,
            "wall_seconds": wall,
            "gc_fraction": gc_frac,
            "memory_management_dominated": dominated,
            "returncode": rc,
            "timed_out": timed_out,
        })
        print(" gc=%.4fs/%d collects  frac=%.1f%% (%.1fs)%s" % (
            gc_time, n_collects, 100.0 * (gc_frac or 0.0), wall,
            "  [DOMINATED]" if dominated else ""))
    return {"section": section, "rows": rows, "available": True}


# ===========================================================================
# RQ3: warmup economics / break-even
# ===========================================================================

def _cumulative(times):
    out = []
    total = 0
    for t in times:
        total += t
        out.append(total)
    return out


def break_even_first(rpy_times, other_times):
    """First iteration N (1-based) where cumulative rpyforth <= cumulative
    other, comparing over the common prefix. None if never within range."""
    n = min(len(rpy_times), len(other_times))
    if n == 0:
        return None
    rpy_cum = _cumulative(rpy_times[:n])
    oth_cum = _cumulative(other_times[:n])
    for i in range(n):
        if rpy_cum[i] <= oth_cum[i]:
            return i + 1
    return None


def break_even_sustained(rpy_times, other_times):
    """Smallest N* (1-based) such that cumulative rpyforth <= cumulative other
    for ALL iterations in [N*, R] (sustained crossover). None if the condition
    never holds through the last iteration. Comparing over the common prefix.

    Found by scanning from the last iteration backwards: N* is one past the
    last iteration where rpyforth is still behind."""
    n = min(len(rpy_times), len(other_times))
    if n == 0:
        return None
    rpy_cum = _cumulative(rpy_times[:n])
    oth_cum = _cumulative(other_times[:n])
    if rpy_cum[n - 1] > oth_cum[n - 1]:
        return None
    n_star = 1
    for i in range(n - 1, -1, -1):
        if rpy_cum[i] > oth_cum[i]:
            n_star = i + 2
            break
    return n_star


def run_rq3(specs, by_prog, steady_iterations):
    rows = []
    for spec in specs:
        engines = by_prog.get(spec.name, {})
        rpy = engines.get(ENGINE_RPYFORTH)
        if not rpy or not rpy.get("times"):
            rows.append({"program": spec.name, "available": False})
            continue
        rpy_times = rpy["times"]
        rpy_warm = warm_median(rpy)
        warmup_cost = None
        if rpy_warm is not None:
            warmup_cost = sum(t - rpy_warm for t in rpy_times)

        against = {}
        for engine in AOT_ENGINES + [ENGINE_GFORTH_FAST]:
            other = engines.get(engine)
            if not other or not other.get("times") or other.get("timed_out"):
                against[engine] = {"break_even": None, "reason": "no-data"}
                continue
            n_star = break_even_sustained(rpy_times, other["times"])
            first = break_even_first(rpy_times, other["times"])
            n = min(len(rpy_times), len(other["times"]))
            against[engine] = {
                "break_even": n_star,
                "first_crossing": first,
                "reason": "ok" if n_star is not None else "never",
                "range": n,
            }
        rows.append({
            "program": spec.name,
            "available": True,
            "rpy_iterations": len(rpy_times),
            "rpy_warm_median_usec": rpy_warm,
            "warmup_cost_usec": warmup_cost,
            "break_even": against,
        })
    return {"rows": rows, "iterations": steady_iterations}


# ===========================================================================
# RQ4: methodology / validity
# ===========================================================================

def warm_tail_drift(times, frac=0.5):
    """Percent-per-iteration slope of the warm tail (last `frac`), normalised
    by the tail median. Positive = getting slower over iterations."""
    if not times:
        return None
    start = int(len(times) * (1.0 - frac))
    tail = times[start:]
    if len(tail) < 2:
        return None
    slope = linfit_slope([float(t) for t in tail])
    if slope is None:
        return None
    med = statistics.median(tail)
    if med == 0:
        return None
    return 100.0 * slope / med


def run_rq4(specs, by_prog, all_engines):
    engines = [ENGINE_RPYFORTH] + [e for e in OTHER_ENGINES if e in all_engines]

    drift_rows = []
    coverage = {}
    for spec in specs:
        eng_res = by_prog.get(spec.name, {})
        coverage[spec.name] = {}
        for engine in engines:
            r = eng_res.get(engine)
            if r is None:
                coverage[spec.name][engine] = "NO-DATA"
                status = "NO-DATA"
            elif r.get("timed_out"):
                coverage[spec.name][engine] = "TIMEOUT"
                status = "TIMEOUT"
            elif not r.get("times"):
                coverage[spec.name][engine] = "NO-DATA"
                status = "NO-DATA"
            else:
                coverage[spec.name][engine] = "OK"
                status = "OK"
            drift = warm_tail_drift(r.get("times", [])) if r else None
            drift_rows.append({
                "program": spec.name,
                "engine": engine,
                "status": status,
                "drift_pct_per_iter": drift,
                "flag": drift is not None and abs(drift) > DRIFT_FLAG,
            })

    # Survivorship: geomean speedup-vs-gforth-fast of each engine over (a) its
    # surviving programs and (b) the common subset all engines survive.
    survived = {}
    for engine in engines:
        survived[engine] = set(
            spec.name for spec in specs
            if coverage[spec.name].get(engine) == "OK")
    common = None
    for engine in engines:
        common = survived[engine] if common is None else (common & survived[engine])
    common = common or set()

    def engine_speedups(engine, prog_subset):
        vals = []
        for spec in specs:
            if spec.name not in prog_subset:
                continue
            sp = speedup_vs(by_prog, spec.name, ref_engine=engine)
            if sp is not None:
                vals.append(sp)
        return vals

    survivorship = {}
    for engine in engines:
        if engine == ENGINE_RPYFORTH:
            continue
        own = engine_speedups(engine, survived[engine])
        common_vals = engine_speedups(engine, common)
        survivorship[engine] = {
            "surviving_programs": sorted(survived[engine]),
            "geomean_surviving": geomean(own),
            "geomean_common_subset": geomean(common_vals),
        }

    return {
        "drift_rows": drift_rows,
        "coverage": coverage,
        "coverage_engines": engines,
        "common_subset": sorted(common),
        "survivorship": survivorship,
    }


# ===========================================================================
# report rendering
# ===========================================================================

def _fmt(v, spec="%.4f"):
    return "n/a" if v is None else spec % v


def render_report(sections, steady_path, revision):
    L = []
    L.append("=" * 80)
    L.append("RESEARCH-QUESTION VERIFICATION REPORT")
    L.append("=" * 80)
    L.append("commit: %s" % revision)
    L.append("steady_results.json: %s" % steady_path)
    L.append("")

    rq1 = sections.get("rq1")
    if rq1:
        L.append("-" * 80)
        L.append("RQ1  TRACE AFFINITY  (does trace behaviour predict speedup?)")
        L.append("-" * 80)
        L.append("%-10s %6s %8s %7s %9s %9s %9s %9s %9s" % (
            "program", "loops", "bridges", "aborts", "trace-fr",
            "abrt/lp", "brdg/lp", "brdg-exec", "speedup"))
        for r in rq1["rows"]:
            L.append("%-10s %6s %8s %7s %9s %9s %9s %9s %9s" % (
                r["program"], r["loops"], r["bridges"], r["aborts"],
                _fmt(r["tracing_time_fraction"], "%.3f"),
                _fmt(r["aborts_per_loop"], "%.2f"),
                _fmt(r["bridges_per_loop"], "%.2f"),
                _fmt(r.get("bridge_exec_fraction"), "%.3f"),
                _fmt(r["speedup_vs_gforth"], "%.2fx")))
        L.append("")
        L.append("Spearman rank correlation (metric vs speedup-vs-gforth):")
        for key, label in rq1["metrics"]:
            L.append("  %-24s rho = %s" % (
                label, _fmt(rq1["correlations"][key], "%.3f")))
        L.append("")

    rq2 = sections.get("rq2")
    if rq2:
        L.append("-" * 80)
        L.append("RQ2  GC vs JIT CONTRIBUTION")
        L.append("-" * 80)
        if not rq2.get("available"):
            L.append("gc sections unavailable in this build "
                     "(no PYPYLOG gc variant produced output).")
        else:
            L.append("PYPYLOG section: %s   (flag threshold: GC fraction > %d%%)"
                     % (rq2["section"], int(GC_FRACTION_FLAG * 100)))
            L.append("%-10s %12s %9s %10s %8s %s" % (
                "program", "gc-pause(s)", "collects", "wall(s)",
                "gc-frac", "flag"))
            for r in rq2["rows"]:
                L.append("%-10s %12s %9d %10s %8s %s" % (
                    r["program"], _fmt(r["gc_pause_seconds"], "%.4f"),
                    r["gc_collects"], _fmt(r["wall_seconds"], "%.2f"),
                    _fmt((r["gc_fraction"] or 0) * 100, "%.1f%%"),
                    "MEMORY-DOMINATED" if r["memory_management_dominated"] else ""))
        L.append("")

    rq3 = sections.get("rq3")
    if rq3:
        L.append("-" * 80)
        L.append("RQ3  WARMUP ECONOMICS / SUSTAINED BREAK-EVEN")
        L.append("-" * 80)
        L.append("N* = smallest iteration from which cumulative rpyforth stays "
                 "<= cumulative")
        L.append("     other engine for all remaining iterations (sustained "
                 "crossover)")
        L.append("%-10s %10s %14s %12s %12s %12s" % (
            "program", "rpy-iters", "warmup-cost(s)", "vs-gforth-f",
            "vs-vfx", "vs-swift"))
        for r in rq3["rows"]:
            if not r.get("available"):
                L.append("%-10s   (no rpyforth per-iteration data)" % r["program"])
                continue
            be = r["break_even"]

            def cell(engine):
                d = be.get(engine, {})
                if d.get("reason") == "no-data":
                    return "no-data"
                if d.get("break_even") is None:
                    return "never(%d)" % d.get("range", 0)
                return "N=%d" % d["break_even"]
            wc = r["warmup_cost_usec"]
            L.append("%-10s %10d %14s %12s %12s %12s" % (
                r["program"], r["rpy_iterations"],
                _fmt(wc / 1e6 if wc is not None else None, "%.3f"),
                cell(ENGINE_GFORTH_FAST), cell(ENGINE_VFXFORTH),
                cell(ENGINE_SWIFTFORTH)))
        L.append("")

    rq4 = sections.get("rq4")
    if rq4:
        L.append("-" * 80)
        L.append("RQ4  METHODOLOGY / VALIDITY")
        L.append("-" * 80)
        L.append("Warm-tail drift (%%/iter, last 50%%; flag |drift| > %.1f%%):"
                 % DRIFT_FLAG)
        L.append("%-10s %-12s %8s %9s %s" % (
            "program", "engine", "status", "drift", "flag"))
        for r in rq4["drift_rows"]:
            L.append("%-10s %-12s %8s %9s %s" % (
                r["program"], r["engine"], r["status"],
                _fmt(r["drift_pct_per_iter"], "%.3f"),
                "DRIFT" if r["flag"] else ""))
        L.append("")
        L.append("Coverage matrix (program x engine):")
        engs = rq4["coverage_engines"]
        L.append("%-10s %s" % ("program", " ".join("%-11s" % e for e in engs)))
        for prog in sorted(rq4["coverage"]):
            cells = rq4["coverage"][prog]
            L.append("%-10s %s" % (
                prog, " ".join("%-11s" % cells.get(e, "-") for e in engs)))
        L.append("")
        L.append("Survivorship bias (geomean speedup rpyforth-vs-engine):")
        L.append("  common subset (all engines survive): %s"
                 % (", ".join(rq4["common_subset"]) or "(empty)"))
        L.append("%-12s %18s %18s %s" % (
            "engine", "geomean-surviving", "geomean-common", "#surviving"))
        for engine in sorted(rq4["survivorship"]):
            s = rq4["survivorship"][engine]
            L.append("%-12s %18s %18s %d" % (
                engine, _fmt(s["geomean_surviving"], "%.3fx"),
                _fmt(s["geomean_common_subset"], "%.3fx"),
                len(s["surviving_programs"])))
        L.append("")

    L.append("=" * 80)
    return "\n".join(L) + "\n"


# ===========================================================================
# chart (optional)
# ===========================================================================

def make_charts(sections, by_prog, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if sections.get("rq1"):
        panels.append("rq1")
    if sections.get("rq3"):
        panels.append("rq3")
    if sections.get("rq4"):
        panels.append("rq4")
    if not panels:
        return False

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4.2 * len(panels)),
                             squeeze=False)
    row = 0

    if sections.get("rq1"):
        ax = axes[row][0]
        rq1 = sections["rq1"]
        xs = [r["tracing_time_fraction"] for r in rq1["rows"]]
        ys = [r["speedup_vs_gforth"] for r in rq1["rows"]]
        for r, x, y in zip(rq1["rows"], xs, ys):
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=40)
            ax.annotate(r["program"], (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        rho = rq1["correlations"].get("tracing_time_fraction")
        ax.set_xlabel("tracing-time fraction")
        ax.set_ylabel("speedup vs gforth-fast")
        ax.set_title("RQ1 trace affinity (Spearman rho = %s)"
                     % (_fmt(rho, "%.3f")))
        ax.grid(True, alpha=0.3)
        row += 1

    if sections.get("rq3"):
        ax = axes[row][0]
        for r in sections["rq3"]["rows"]:
            if not r.get("available"):
                continue
            prog = r["program"]
            rpy = by_prog.get(prog, {}).get(ENGINE_RPYFORTH)
            if not rpy or not rpy.get("times"):
                continue
            cum = _cumulative(rpy["times"])
            xs = list(range(1, len(cum) + 1))
            ax.plot(xs, [c / 1e6 for c in cum], marker="o", markersize=2,
                    linewidth=1, label=prog)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative rpyforth time (s)")
        ax.set_title("RQ3 cumulative-time curves")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        row += 1

    if sections.get("rq4"):
        ax = axes[row][0]
        rows = [r for r in sections["rq4"]["drift_rows"]
                if r["drift_pct_per_iter"] is not None]
        labels = ["%s/%s" % (r["program"], r["engine"]) for r in rows]
        vals = [r["drift_pct_per_iter"] for r in rows]
        colors = ["#d62728" if r["flag"] else "#1f77b4" for r in rows]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(DRIFT_FLAG, color="grey", linestyle=":", linewidth=0.8)
        ax.axvline(-DRIFT_FLAG, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("warm-tail drift (%/iter)")
        ax.set_title("RQ4 drift (red = flagged)")
        ax.grid(True, axis="x", alpha=0.3)
        row += 1

    fig.tight_layout()
    fig.savefig(str(pdf_path))
    plt.close(fig)
    return True


# ===========================================================================
# driver
# ===========================================================================

def select_programs(programs_arg):
    if not programs_arg:
        return list(PROGRAMS)
    want = set(p.strip() for p in programs_arg.split(","))
    selected = [p for p in PROGRAMS if p.name in want]
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rq", choices=["rq1", "rq2", "rq3", "rq4", "all"],
                        help="which research question(s) to verify")
    parser.add_argument("--programs", default="",
                        help="comma-separated program subset (default: all)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="iterations for PYPYLOG (rq1/rq2) runs")
    parser.add_argument("--pin", type=int, default=None,
                        help="taskset CPU core for PYPYLOG runs")
    parser.add_argument("--steady-json", default="",
                        help="steady_results.json (default: newest under logs/)")
    parser.add_argument("--timeout", type=int, default=ab.STEADY_DEFAULT_TIMEOUT,
                        help="per-run timeout seconds for PYPYLOG runs")
    parser.add_argument("--output", type=Path, default=Path("logs/rq"),
                        help="output base dir (default: logs/rq)")
    args = parser.parse_args(argv)

    logs_root = REPO_ROOT / "logs"
    if args.steady_json:
        steady_path = Path(args.steady_json)
        if not steady_path.is_absolute():
            steady_path = REPO_ROOT / steady_path
    else:
        steady_path = find_newest_steady_json(logs_root)
        if steady_path is None:
            print("No steady_results.json found under %s; pass --steady-json"
                  % logs_root, file=sys.stderr)
            return 1
    if not steady_path.exists():
        print("steady json not found: %s" % steady_path, file=sys.stderr)
        return 1

    summary, by_prog = load_steady(steady_path)
    steady_iterations = summary.get("iterations")
    all_engines = summary.get("engines", [])
    specs = select_programs(args.programs)
    if not specs:
        print("No matching programs for %r" % args.programs, file=sys.stderr)
        return 1

    revision = git_revision(REPO_ROOT)
    env_line = capture_environment()
    if args.pin is not None:
        env_line += " | pin core %d" % args.pin
    print(env_line + " | commit " + revision)
    print("steady json: %s (iterations=%s, engines=%s)" % (
        steady_path, steady_iterations, ",".join(all_engines)))

    out_base = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    log_dir = out_base / revision
    log_dir.mkdir(parents=True, exist_ok=True)

    want = {"rq1", "rq2", "rq3", "rq4"} if args.rq == "all" else {args.rq}
    sections = {}

    needs_binary = bool(want & {"rq1", "rq2"})
    if needs_binary and not Path(ab.ENGINE_BINARY[ENGINE_RPYFORTH]).exists():
        print("rpyforth binary missing: %s" % ab.ENGINE_BINARY[ENGINE_RPYFORTH],
              file=sys.stderr)
        return 1

    tmp = None
    if needs_binary:
        tmp = tempfile.TemporaryDirectory(prefix="rq_pypylog_")
    try:
        tmpdir = tmp.name if tmp else None
        if "rq1" in want:
            print("RQ1: tracing rpyforth under jit-summary ...")
            sections["rq1"] = run_rq1(specs, by_prog, args.iterations,
                                      tmpdir, args.timeout, args.pin)
        if "rq2" in want:
            print("RQ2: tracing rpyforth under gc PYPYLOG ...")
            sections["rq2"] = run_rq2(specs, args.iterations, tmpdir,
                                      args.timeout, args.pin)
        if "rq3" in want:
            print("RQ3: computing break-even from steady json ...")
            sections["rq3"] = run_rq3(specs, by_prog, steady_iterations)
        if "rq4" in want:
            print("RQ4: computing drift / coverage / survivorship ...")
            sections["rq4"] = run_rq4(specs, by_prog, all_engines)
    finally:
        if tmp:
            tmp.cleanup()

    report = render_report(sections, steady_path, revision)
    report_path = log_dir / "rq_report.txt"
    report_path.write_text(report, encoding="utf-8")

    results = {
        "commit": revision,
        "steady_json": str(steady_path),
        "steady_iterations": steady_iterations,
        "engines": all_engines,
        "programs": [s.name for s in specs],
        "sections": sections,
    }
    json_path = log_dir / "rq_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str),
                         encoding="utf-8")

    print("")
    print(report)
    print("report: %s" % report_path)
    print("json:   %s" % json_path)

    pdf_path = log_dir / "rq_charts.pdf"
    try:
        if make_charts(sections, by_prog, pdf_path):
            print("charts: %s" % pdf_path)
    except ImportError:
        print("matplotlib unavailable; skipping charts")
    except Exception as exc:
        print("chart generation failed: %s" % exc, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
