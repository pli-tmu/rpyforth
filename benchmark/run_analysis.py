#!/usr/bin/env python3
"""
Paper-grade comparative-analysis harness: rpyforth (RPython/PyPy meta-tracing JIT)
vs gforth, answering three causal questions with data rather than endpoint numbers.

  Q1  WHY did rpyforth get fast?          -> Axis 1: rpyforth ablation ladder
  Q2  WHY does it still lose where it loses? -> Axis 3: JIT internals vs outcomes
  Q3  WHERE does gforth-fast's speed come from? -> Axis 2: gforth decomposition ladder

The harness measures every (config x benchmark) cell, captures PYPYLOG jit-summary
for each rpyforth run, parses tracing/backend/loops/bridges/guards/aborts, and
correlates JIT internals with the win/loss vs gforth-fast. It produces a JSON dump,
a markdown report, and four PDF charts.

Design axes
-----------
Axis 1 (rpyforth ablation ladder, Q1):
  rpyforth-c-naive    HEAD src, virtualize build (metastack/stkfrag OFF)
  rpyforth-c-prefix1  da61000 stkfrag (metastack+inlining; pre-FIX1)
  rpyforth-c-fix1     3f8222b stkfrag (FIX1: tid-only promotion; pre-FIX2)
  rpyforth-c-stkfrag  HEAD stkfrag (FIX1+FIX2+benchgc words; flagship)
  Ladder deltas: (naive->stkfrag)=stack representation, (prefix1->fix1)=FIX1,
                 (fix1->stkfrag)=FIX2.

Axis 2 (gforth decomposition ladder, Q3):
  gforth-fast              full: dynamic replication + superinstructions
  gforth-fast --ss-number=0  replication only, no dynamically-formed superinsts
  gforth-fast --no-dynamic static primitives only
  gforth                   plain engine (debugging hooks)
  NOTE: gforth-fast --no-super HANGS on this build (never emits output); the
  equivalent "no superinstructions" point is --ss-number=0, used here instead.

Axis 3 (JIT internals, Q1/Q2): jit-summary parse per rpyforth (config x benchmark).

Usage:
  .venv/bin/python benchmark/run_analysis.py [--iterations 5] [--pin 6]
                    [--stages build,measure,report,chart] [--quiet-wait 3600]
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

# Import pure helpers from the existing harnesses (do not copy).
from run_shootout import (  # noqa: E402
    median_ci,
    discover_benchmarks,
    git_revision,
)
from run_appbench import (  # noqa: E402
    build_program_registry,
    build_gforth_cmd,
    build_rpyforth_cmd,
    ProgramSpec,
)

GFORTH_DIR = REPO_ROOT / "gforth-0.7.9"
GFORTH_FAST = GFORTH_DIR / "gforth-fast"
GFORTH = GFORTH_DIR / "gforth"

DEFAULT_ITERATIONS = 5
DEFAULT_PIN = 6
DEFAULT_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Config registries
# ---------------------------------------------------------------------------

# Axis 1: rpyforth ablation ladder. (id, exe_name, build_kind, source_commit)
# build_kind: "virt" (RPYFORTH_VIRTUALIZE=1) or "stkfrag" (RPYFORTH_STACK_FRAGMENT=1)
RPYFORTH_LADDER = [
    ("rpyforth-c-naive",   "rpyforth-c-naive",   "virt",    "7038abb"),
    ("rpyforth-c-prefix1", "rpyforth-c-prefix1", "stkfrag", "da61000"),
    ("rpyforth-c-fix1",    "rpyforth-c-fix1",    "stkfrag", "3f8222b"),
    ("rpyforth-c-stkfrag", "rpyforth-c-stkfrag", "stkfrag", "7038abb"),
]
RPYFORTH_FLAGSHIP = "rpyforth-c-stkfrag"

# Axis 2: gforth decomposition ladder. (id, binary, extra_flags)
GFORTH_LADDER = [
    ("gforth-fast",         GFORTH_FAST, []),
    ("gforth-fast-nosuper", GFORTH_FAST, ["--ss-number=0"]),
    ("gforth-fast-nodyn",   GFORTH_FAST, ["--no-dynamic"]),
    ("gforth",              GFORTH,      []),
]
GFORTH_REFERENCE = "gforth-fast"

SMOKE_BENCH = "sieve"
SMOKE_EXPECT = "Count: 1028"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One (config, benchmark) measurement."""
    config: str
    suite: str            # "shootout" | "appbench"
    bench: str
    metric: str           # "usec" (shootout self-timed) | "wall_s" (appbench cold)
    samples: List[float] = field(default_factory=list)
    median: Optional[float] = None
    ci_pct: float = 0.0
    status: str = "pending"     # ok | error | timeout | missing_binary
    detail: str = ""
    jit: Optional[Dict] = None  # parsed jit-summary (rpyforth only)


def sh(cmd: List[str], timeout=15, env=None, cwd=None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=cwd, stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", e.stderr or ""
    except FileNotFoundError as e:
        return -2, "", str(e)


# ---------------------------------------------------------------------------
# Stage 0: quiesce -- wait until no competing benchmark/translation processes
# ---------------------------------------------------------------------------

def competing_processes(my_pid: int) -> List[str]:
    """List other agents' translation (pypy/rpython) or benchmark (rpyforth-c/gforth)
    processes that would skew medians. Excludes this process tree."""
    rc, out, _ = sh(["ps", "-eo", "pid,ppid,comm,args"], timeout=10)
    if rc != 0:
        return []
    hits = []
    mypids = {str(my_pid), str(os.getppid())}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, comm, args = parts
        if pid in mypids or ppid in mypids:
            continue
        low = args.lower()
        is_translate = ("rpython" in low) or ("pypy" in low and "targetrpyforth" in low)
        is_bench = bool(re.search(r"(rpyforth-c[\w-]*|gforth(-fast)?)\b", args)) \
            and "run_analysis" not in low and "grep" not in comm
        # our own build script uses rpython -> exclude by build.log marker path
        if "/tmp/ana-" in args:
            continue
        if is_translate or is_bench:
            hits.append("%s %s: %s" % (pid, comm, args[:80]))
    return hits


def wait_for_quiet(my_pid: int, max_wait: int, poll: int = 20) -> bool:
    waited = 0
    while True:
        procs = competing_processes(my_pid)
        if not procs:
            return True
        if waited >= max_wait:
            print("WARNING: still %d competing processes after %ds; proceeding anyway"
                  % (len(procs), waited), file=sys.stderr)
            for p in procs[:5]:
                print("   busy: " + p, file=sys.stderr)
            return False
        print("[quiesce] %d competing process(es); waited %ds/%ds ..."
              % (len(procs), waited, max_wait), file=sys.stderr)
        time.sleep(poll)
        waited += poll


# ---------------------------------------------------------------------------
# Stage 1: build + validate Axis-1 binaries
# ---------------------------------------------------------------------------

def build_ladder(iterations_note=""):
    """Build the three ablation binaries in detached worktrees if not cached.
    Returns dict id->status. Uses /tmp/ana-<id> worktrees created out-of-band."""
    status = {}
    for cid, exe, kind, commit in RPYFORTH_LADDER:
        target = REPO_ROOT / exe
        if target.exists() and os.access(target, os.X_OK):
            status[cid] = "cached"
            continue
        wt = Path("/tmp") / ("ana-" + cid.replace("rpyforth-c-", ""))
        if not wt.exists():
            # create the worktree ourselves (worktree add is allowed)
            rc, _, err = sh(["git", "worktree", "add", str(wt), "--detach", commit],
                            timeout=120, cwd=str(REPO_ROOT))
            if rc != 0:
                status[cid] = "worktree_fail: " + err[:120]
                continue
        # symlink pypy toolchain from main repo
        for d in ("_pypy_binary", "pypy"):
            link = wt / d
            if not link.exists():
                try:
                    link.symlink_to(REPO_ROOT / d)
                except OSError:
                    pass
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["RPYFORTH_EXE_NAME"] = exe
        if kind == "stkfrag":
            env["RPYFORTH_STACK_FRAGMENT"] = "1"
        elif kind == "virt":
            env["RPYFORTH_VIRTUALIZE"] = "1"
        print("[build] %s (%s @ %s) ..." % (exe, kind, commit), file=sys.stderr)
        rc, _, err = sh(["./_pypy_binary/bin/python2", "./pypy/rpython/bin/rpython",
                         "-Ojit", "rpyforth/targetrpyforth.py"],
                        timeout=1200, env=env, cwd=str(wt))
        built = wt / exe
        if built.exists():
            sh(["cp", "-f", str(built), str(target)])
            status[cid] = "built"
        else:
            status[cid] = "build_fail: rc=%d %s" % (rc, err[-120:])
    return status


def validate_binaries(pin: Optional[int]) -> Dict[str, str]:
    """Run the sieve smoke benchmark on every config; expect 'Count: 1028'."""
    results = {}
    sieve = REPO_ROOT / "shootout" / (SMOKE_BENCH + ".fs")
    wrapper = ["taskset", "-c", str(pin)] if pin is not None else []
    for cid, exe, _, _ in RPYFORTH_LADDER:
        binp = REPO_ROOT / exe
        if not binp.exists():
            results[cid] = "MISSING"
            continue
        rc, out, err = sh(wrapper + [str(binp), str(sieve)], timeout=60)
        results[cid] = "PASS" if SMOKE_EXPECT in out else "FAIL(%s)" % (out + err)[:60]
    for cid, binp, flags in GFORTH_LADDER:
        rc, out, err = sh(wrapper + [str(binp)] + flags + [str(sieve)], timeout=60)
        results[cid] = "PASS" if SMOKE_EXPECT in out else "FAIL(rc=%d,%s)" % (rc, (out + err)[:50])
    return results


# ---------------------------------------------------------------------------
# Stage 2: measurement
# ---------------------------------------------------------------------------

_ELAPSED_RE = re.compile(r"Elapsed:\s*(\d+)\s*usec")


def run_shootout_once(binp: Path, flags: List[str], bench: Path, pin: Optional[int],
                      jitlog: Optional[Path]) -> Tuple[str, Optional[int], float, str]:
    """One shootout run. Returns (status, elapsed_usec_or_None, wall_s, output)."""
    wrapper = ["taskset", "-c", str(pin)] if pin is not None else []
    env = os.environ.copy()
    if jitlog is not None:
        env["PYPYLOG"] = "jit-summary:" + str(jitlog)
    cmd = wrapper + [str(binp)] + flags + [str(bench)]
    t0 = time.perf_counter()
    rc, out, err = sh(cmd, timeout=DEFAULT_TIMEOUT, env=env)
    wall = time.perf_counter() - t0
    if rc == -1:
        return "timeout", None, wall, "timeout"
    if rc != 0:
        return "error", None, wall, (out + err)[:120]
    m = _ELAPSED_RE.search(out)
    usec = int(m.group(1)) if m else None
    return "ok", usec, wall, out


def measure_shootout(configs_rpy, configs_gf, benches: List[Path], iterations: int,
                     pin: Optional[int], jitdir: Path) -> List[Cell]:
    cells = []
    for bench in benches:
        name = bench.stem
        # rpyforth configs (self-timed usec; capture jit-summary on last iter)
        for cid, exe, _, _ in configs_rpy:
            binp = REPO_ROOT / exe
            cell = Cell(config=cid, suite="shootout", bench=name, metric="usec")
            if not binp.exists():
                cell.status = "missing_binary"
                cells.append(cell)
                continue
            jitlog = None
            for it in range(iterations):
                if it == iterations - 1:
                    jitlog = jitdir / ("%s__%s.jitsum" % (cid, name))
                st, usec, wall, out = run_shootout_once(binp, [], bench, pin, jitlog)
                if st != "ok":
                    cell.status = st
                    cell.detail = out
                    break
                cell.samples.append(float(usec if usec is not None else wall * 1e6))
            else:
                cell.status = "ok"
            if jitlog and jitlog.exists():
                cell.jit = parse_jit_summary(jitlog)
            _finalize(cell)
            cells.append(cell)
        # gforth configs (self-timed usec; no jit)
        for cid, binp, flags in configs_gf:
            cell = Cell(config=cid, suite="shootout", bench=name, metric="usec")
            for it in range(iterations):
                st, usec, wall, out = run_shootout_once(binp, flags, bench, pin, None)
                if st != "ok":
                    cell.status = st
                    cell.detail = out
                    break
                cell.samples.append(float(usec if usec is not None else wall * 1e6))
            else:
                cell.status = "ok"
            _finalize(cell)
            cells.append(cell)
        print("[shootout] %s done" % name, file=sys.stderr)
    return cells


def measure_appbench(configs_rpy, configs_gf, specs: List[ProgramSpec],
                     iterations: int, pin: Optional[int], jitdir: Path,
                     tmpdir: Path) -> List[Cell]:
    """Cold appbench: fresh process per iteration, wall-clock metric."""
    cells = []
    wrapper = ["taskset", "-c", str(pin)] if pin is not None else []
    for spec in specs:
        # rpyforth configs
        for cid, exe, _, _ in configs_rpy:
            binp = REPO_ROOT / exe
            cell = Cell(config=cid, suite="appbench", bench=spec.name, metric="wall_s")
            if not binp.exists():
                cell.status = "missing_binary"
                cells.append(cell)
                continue
            cmd = build_rpyforth_cmd(binp, spec, tmpdir)
            jitlog = None
            for it in range(iterations):
                env = os.environ.copy()
                env.update(spec.rpy_env)
                if it == iterations - 1:
                    jitlog = jitdir / ("%s__%s.jitsum" % (cid, spec.name))
                    env["PYPYLOG"] = "jit-summary:" + str(jitlog)
                t0 = time.perf_counter()
                rc, out, err = sh(wrapper + cmd, timeout=DEFAULT_TIMEOUT, env=env,
                                  cwd=str(spec.workdir))
                wall = time.perf_counter() - t0
                if rc == -1:
                    cell.status = "timeout"
                    break
                if rc < 0 or rc > 0:
                    cell.status = "error"
                    cell.detail = (out + err)[:120]
                    break
                cell.samples.append(wall)
            else:
                cell.status = "ok"
            if jitlog and jitlog.exists():
                cell.jit = parse_jit_summary(jitlog)
            _finalize(cell)
            cells.append(cell)
        # gforth configs
        for cid, binp, flags in configs_gf:
            cell = Cell(config=cid, suite="appbench", bench=spec.name, metric="wall_s")
            base = build_gforth_cmd(binp, spec, tmpdir)
            # inject engine flags right after the binary
            cmd = [base[0]] + flags + base[1:]
            for it in range(iterations):
                t0 = time.perf_counter()
                rc, out, err = sh(wrapper + cmd, timeout=DEFAULT_TIMEOUT,
                                  cwd=str(spec.workdir))
                wall = time.perf_counter() - t0
                if rc == -1:
                    cell.status = "timeout"
                    break
                if rc != 0:
                    cell.status = "error"
                    cell.detail = (out + err)[:120]
                    break
                cell.samples.append(wall)
            else:
                cell.status = "ok"
            _finalize(cell)
            cells.append(cell)
        print("[appbench] %s done" % spec.name, file=sys.stderr)
    return cells


def _finalize(cell: Cell):
    if cell.samples:
        med, ci = median_ci(cell.samples)
        cell.median = med
        cell.ci_pct = ci
        if cell.status == "pending":
            cell.status = "ok"


# ---------------------------------------------------------------------------
# Stage 3: jit-summary parsing
# ---------------------------------------------------------------------------

# jit-summary lines look like:  Tracing:      123   0.045
#                               Backend:       12   0.008
#                               TOTAL:                0.500
#                               ops:  ...  recorded ops  ...
#                               guards: N
#                               opt ops: N
#                               opt guards: N
#                               forcings: N
#                               abort: N
#                               nvirtuals / nvholes ...
#                               Total # of loops: N
#                               Total # of bridges: N
_JIT_PATTERNS = {
    "tracing_time":   re.compile(r"^Tracing:\s+\d+\s+([\d.]+)", re.M),
    "backend_time":   re.compile(r"^Backend:\s+\d+\s+([\d.]+)", re.M),
    "tracing_no":     re.compile(r"^Tracing:\s+(\d+)", re.M),
    "backend_no":     re.compile(r"^Backend:\s+(\d+)", re.M),
    "total_time":     re.compile(r"^TOTAL:\s+([\d.]+)", re.M),
    "ops":            re.compile(r"^ops:\s+(\d+)", re.M),
    "recorded_ops":   re.compile(r"^recorded ops:\s+(\d+)", re.M),
    "guards":         re.compile(r"^guards:\s+(\d+)", re.M),
    "opt_ops":        re.compile(r"^opt ops:\s+(\d+)", re.M),
    "opt_guards":     re.compile(r"^opt guards:\s+(\d+)", re.M),
    "forcings":       re.compile(r"^forcings:\s+(\d+)", re.M),
    "loops":          re.compile(r"^Total # of loops:\s+(\d+)", re.M),
    "bridges":        re.compile(r"^Total # of bridges:\s+(\d+)", re.M),
    "loop_tokens":    re.compile(r"^Total # of loop tokens:\s+(\d+)", re.M),
}


def parse_jit_summary(path: Path) -> Dict:
    try:
        text = path.read_text()
    except OSError:
        return {}
    out = {}
    for key, pat in _JIT_PATTERNS.items():
        m = pat.search(text)
        if m:
            v = m.group(1)
            out[key] = float(v) if "." in v else int(v)
    # aborts: the summary emits several "abort: <reason>:\tN" lines; sum them.
    out["abort"] = sum(int(n) for n in re.findall(r"^abort:[^\t]*\t(\d+)", text, re.M))
    # derived
    tt = out.get("tracing_time", 0.0)
    bt = out.get("backend_time", 0.0)
    out["warmup_time"] = tt + bt
    loops = out.get("loops", 0)
    bridges = out.get("bridges", 0)
    out["bridges_per_loop"] = (float(bridges) / loops) if loops else 0.0
    out["raw_present"] = bool(out)
    return out


# ---------------------------------------------------------------------------
# Stage 4: counters (perf / cachegrind) auto-detect
# ---------------------------------------------------------------------------

def detect_counters() -> Dict[str, object]:
    """Auto-detect perf stat and valgrind/cachegrind usability. Structured so that
    enabling perf later needs no code change -- just flip 'perf_usable'."""
    info = {"perf_usable": False, "cachegrind_usable": False, "notes": []}
    rc, out, err = sh(["perf", "stat", "-e", "cycles,instructions,branches,branch-misses",
                       "--", "true"], timeout=15)
    combined = (out + err)
    if rc == 0 and "Performance counter stats" in combined and "not supported" not in combined \
            and "not permitted" not in combined and "<not counted>" not in combined:
        info["perf_usable"] = True
    else:
        try:
            para = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
        except OSError:
            para = "?"
        info["notes"].append("perf stat unusable (perf_event_paranoid=%s); rc=%d" % (para, rc))
    rc2, _, _ = sh(["valgrind", "--version"], timeout=10)
    if rc2 == 0:
        info["cachegrind_usable"] = True
    else:
        info["notes"].append("valgrind/cachegrind not installed")
    if not info["perf_usable"] and not info["cachegrind_usable"]:
        info["notes"].append("WARNING: no hardware counters available; counters stage SKIPPED")
    return info


def run_counters(configs_rpy, benches, pin, info) -> Optional[List[Dict]]:
    """Placeholder that runs only if perf/cachegrind become usable. Returns None
    when skipped. Structured so enabling perf later needs no other changes."""
    if not info.get("perf_usable") and not info.get("cachegrind_usable"):
        print("WARNING: counters stage skipped -- " + "; ".join(info["notes"]),
              file=sys.stderr)
        return None
    rows = []
    wrapper = ["taskset", "-c", str(pin)] if pin is not None else []
    for cid, exe, _, _ in configs_rpy:
        binp = REPO_ROOT / exe
        if not binp.exists():
            continue
        for bench in benches:
            cmd = ["perf", "stat", "-x", ",", "-e",
                   "cycles,instructions,branches,branch-misses", "--"] \
                + wrapper + [str(binp), str(bench)]
            rc, out, err = sh(cmd, timeout=DEFAULT_TIMEOUT)
            counters = {}
            for line in err.splitlines():
                parts = line.split(",")
                if len(parts) >= 3 and parts[0].replace(".", "").isdigit():
                    counters[parts[2]] = parts[0]
            rows.append({"config": cid, "bench": bench.stem, "counters": counters})
    return rows


# ---------------------------------------------------------------------------
# Stage 5: report + charts
# ---------------------------------------------------------------------------

def cells_index(cells: List[Cell]) -> Dict[Tuple[str, str, str], Cell]:
    return {(c.suite, c.bench, c.config): c for c in cells}


def _fmt(v, metric):
    if v is None:
        return "  N/A "
    if metric == "usec":
        return "%8.1f" % (v / 1000.0)  # ms
    return "%8.3f" % v  # seconds


def build_report(cells, smoke, build_status, counter_info, jit_note,
                 iterations, pin, rev) -> str:
    idx = cells_index(cells)
    rpy_ids = [c[0] for c in RPYFORTH_LADDER]
    gf_ids = [c[0] for c in GFORTH_LADDER]
    all_ids = rpy_ids + gf_ids
    suites = [("shootout", "usec", "ms"), ("appbench", "wall_s", "s")]
    L = []
    L.append("# rpyforth vs gforth -- comparative analysis (%s)" % rev)
    L.append("")
    L.append("iterations=%d (median-of-N, bootstrap 90%% CI), pin=core %s" % (iterations, pin))
    L.append("")

    # -- validation
    L.append("## Binary validation (sieve smoke, expect 'Count: 1028')")
    L.append("")
    for cid in all_ids:
        L.append("- %-22s %s" % (cid, smoke.get(cid, "?")))
    L.append("")
    L.append("Build status: " + ", ".join("%s=%s" % (k, v) for k, v in build_status.items()))
    L.append("")

    # -- master per-benchmark table
    for suite, metric, unit in suites:
        benches = sorted({c.bench for c in cells if c.suite == suite})
        L.append("## Master table -- %s (%s, lower is faster)" % (suite, unit))
        L.append("")
        hdr = "%-12s" % "bench" + "".join("%10s" % cid.replace("rpyforth-c-", "rpy-").replace("gforth-fast-", "gf-").replace("gforth-fast", "gf-full").replace("gforth", "gf") for cid in all_ids)
        L.append("```")
        L.append(hdr)
        for b in benches:
            row = "%-12s" % b
            for cid in all_ids:
                cell = idx.get((suite, b, cid))
                row += _fmt(cell.median if cell else None, metric)
            L.append(row)
        L.append("```")
        L.append("")

    # -- Axis 1 ladder with deltas (normalized to naive), shootout usec
    L.append("## Axis 1 -- rpyforth ablation ladder (shootout, speedup vs naive)")
    L.append("")
    L.append("Each cell = naive_time / config_time (>1 = faster than naive).")
    L.append("Ladder deltas: naive->stkfrag = stack representation; "
             "prefix1->fix1 = FIX1; fix1->stkfrag = FIX2.")
    L.append("```")
    L.append("%-12s%10s%10s%10s%10s" % ("bench", "naive", "prefix1", "fix1", "stkfrag"))
    sh_benches = sorted({c.bench for c in cells if c.suite == "shootout"})
    geomeans = {cid: [] for cid in rpy_ids}
    for b in sh_benches:
        base = idx.get(("shootout", b, "rpyforth-c-naive"))
        row = "%-12s" % b
        for cid in rpy_ids:
            cell = idx.get(("shootout", b, cid))
            if base and base.median and cell and cell.median:
                sp = base.median / cell.median
                geomeans[cid].append(sp)
                row += "%10.2f" % sp
            else:
                row += "%10s" % "N/A"
        L.append(row)
    row = "%-12s" % "GEOMEAN"
    for cid in rpy_ids:
        vals = geomeans[cid]
        row += "%10.2f" % (_geomean(vals) if vals else 0.0)
    L.append(row)
    L.append("```")
    L.append("")

    # -- Axis 2 ladder normalized to gforth-fast (shootout)
    L.append("## Axis 2 -- gforth decomposition ladder (shootout, time / gforth-fast)")
    L.append("")
    L.append("Each cell = config_time / gforth-fast_time (>1 = slower than full gforth-fast).")
    L.append("Configs: gf-full=dynamic+superinsts, gf-nosuper=--ss-number=0 (replication only), "
             "gf-nodyn=--no-dynamic (static prims), gf=plain engine.")
    L.append("```")
    L.append("%-12s%10s%10s%10s%10s" % ("bench", "gf-full", "gf-nosup", "gf-nodyn", "gf-plain"))
    for b in sh_benches:
        base = idx.get(("shootout", b, "gforth-fast"))
        row = "%-12s" % b
        for cid in gf_ids:
            cell = idx.get(("shootout", b, cid))
            if base and base.median and cell and cell.median:
                row += "%10.2f" % (cell.median / base.median)
            else:
                row += "%10s" % "N/A"
        L.append(row)
    L.append("```")
    L.append("")

    # -- Axis 3: jit metrics + speedup-vs-gforth-fast (flagship), sorted by speedup
    L.append("## Axis 3 -- JIT internals vs outcome (flagship rpyforth-c-stkfrag)")
    L.append("")
    L.append("speedup = gforth-fast_time / rpyforth_time (>1 = rpyforth wins). "
             "warmup_share = (tracing+backend)/wall.")
    L.append("```")
    L.append("%-12s%9s%8s%8s%8s%9s%8s%9s" %
             ("bench", "speedup", "loops", "bridges", "guards", "aborts", "b/loop", "warmup%"))
    corr_rows = []
    for suite in ("shootout", "appbench"):
        benches = sorted({c.bench for c in cells if c.suite == suite})
        for b in benches:
            fc = idx.get((suite, b, RPYFORTH_FLAGSHIP))
            gf = idx.get((suite, b, GFORTH_REFERENCE))
            if not fc or fc.median is None or not gf or gf.median is None:
                continue
            speedup = gf.median / fc.median
            jit = fc.jit or {}
            wall_s = fc.median / (1e6 if suite == "shootout" else 1.0)
            warmup = jit.get("warmup_time", 0.0)
            share = (warmup / wall_s * 100.0) if wall_s else 0.0
            corr_rows.append((suite, b, speedup, jit, share))
    corr_rows.sort(key=lambda r: r[2])
    for suite, b, speedup, jit, share in corr_rows:
        L.append("%-12s%9.2f%8d%8d%8d%9d%8.2f%9.1f" % (
            ("%s/%s" % (suite[:3], b))[:12], speedup,
            jit.get("loops", 0), jit.get("bridges", 0), jit.get("guards", 0),
            jit.get("abort", 0), jit.get("bridges_per_loop", 0.0), share))
    L.append("```")
    L.append("")

    # -- counters
    L.append("## Counters stage")
    L.append("")
    L.append("perf_usable=%s cachegrind_usable=%s" %
             (counter_info["perf_usable"], counter_info["cachegrind_usable"]))
    for n in counter_info["notes"]:
        L.append("- " + n)
    L.append("")
    L.append("_" + jit_note + "_")
    return "\n".join(L)


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return 0.0
    return statistics.geometric_mean(vals) if hasattr(statistics, "geometric_mean") \
        else (statistics.prod(vals) ** (1.0 / len(vals)))


def make_charts(cells, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("WARNING: matplotlib unavailable, charts skipped: %s" % e, file=sys.stderr)
        return []
    import math
    idx = cells_index(cells)
    rpy_ids = [c[0] for c in RPYFORTH_LADDER]
    gf_ids = [c[0] for c in GFORTH_LADDER]
    sh_benches = sorted({c.bench for c in cells if c.suite == "shootout"})
    app_benches = sorted({c.bench for c in cells if c.suite == "appbench"})
    paths = []

    # (a) gforth decomposition -- grouped bars, normalized to gforth-fast
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(sh_benches))
    w = 0.2
    labels = {"gforth-fast": "full", "gforth-fast-nosuper": "no-super(ss0)",
              "gforth-fast-nodyn": "no-dynamic", "gforth": "plain"}
    for i, cid in enumerate(gf_ids):
        vals = []
        for b in sh_benches:
            base = idx.get(("shootout", b, "gforth-fast"))
            cell = idx.get(("shootout", b, cid))
            vals.append((cell.median / base.median) if base and base.median and cell and cell.median else 0)
        ax.bar([xi + i * w for xi in x], vals, w, label=labels[cid])
    ax.set_xticks([xi + 1.5 * w for xi in x])
    ax.set_xticklabels(sh_benches, rotation=45, ha="right")
    ax.set_ylabel("time / gforth-fast (lower=faster)")
    ax.set_title("Axis 2: gforth decomposition (normalized to gforth-fast)")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.legend()
    fig.tight_layout()
    p = outdir / "chart_gforth_decomposition.pdf"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # (b) rpyforth ablation ladder -- grouped bars normalized to naive (speedup)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels2 = {"rpyforth-c-naive": "naive", "rpyforth-c-prefix1": "prefix1",
               "rpyforth-c-fix1": "fix1", "rpyforth-c-stkfrag": "stkfrag"}
    for i, cid in enumerate(rpy_ids):
        vals = []
        for b in sh_benches:
            base = idx.get(("shootout", b, "rpyforth-c-naive"))
            cell = idx.get(("shootout", b, cid))
            vals.append((base.median / cell.median) if base and base.median and cell and cell.median else 0)
        ax.bar([xi + i * w for xi in x], vals, w, label=labels2[cid])
    ax.set_xticks([xi + 1.5 * w for xi in x])
    ax.set_xticklabels(sh_benches, rotation=45, ha="right")
    ax.set_ylabel("speedup vs naive (higher=faster)")
    ax.set_title("Axis 1: rpyforth ablation ladder (normalized to naive)")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.legend()
    fig.tight_layout()
    p = outdir / "chart_rpyforth_ablation.pdf"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # (c) scatter bridges-per-loop (x, log) vs speedup-vs-gforth-fast (y)
    fig, ax = plt.subplots(figsize=(9, 7))
    for suite in ("shootout", "appbench"):
        benches = sorted({c.bench for c in cells if c.suite == suite})
        for b in benches:
            fc = idx.get((suite, b, RPYFORTH_FLAGSHIP))
            gf = idx.get((suite, b, GFORTH_REFERENCE))
            if not fc or fc.median is None or not gf or gf.median is None:
                continue
            bpl = (fc.jit or {}).get("bridges_per_loop", 0.0)
            speedup = gf.median / fc.median
            xv = max(bpl, 0.05)
            ax.scatter(xv, speedup, s=40,
                       c="C0" if suite == "shootout" else "C1")
            ax.annotate(b, (xv, speedup), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("bridges per loop (log)")
    ax.set_ylabel("speedup vs gforth-fast (>1 = rpyforth wins)")
    ax.set_title("Axis 3: bridges-per-loop vs speedup")
    fig.tight_layout()
    p = outdir / "chart_bridges_vs_speedup.pdf"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # (d) warm-up-share bar chart (appbench, flagship)
    fig, ax = plt.subplots(figsize=(9, 5))
    shares = []
    for b in app_benches:
        fc = idx.get(("appbench", b, RPYFORTH_FLAGSHIP))
        if not fc or fc.median is None:
            shares.append(0)
            continue
        warm = (fc.jit or {}).get("warmup_time", 0.0)
        shares.append(warm / fc.median * 100.0 if fc.median else 0)
    ax.bar(range(len(app_benches)), shares, color="C2")
    ax.set_xticks(range(len(app_benches)))
    ax.set_xticklabels(app_benches, rotation=30, ha="right")
    ax.set_ylabel("warmup share of wall time (%)")
    ax.set_title("Axis 3: JIT warm-up share (appbench cold, flagship)")
    fig.tight_layout()
    p = outdir / "chart_warmup_share.pdf"
    fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def cells_to_json(cells):
    return [{
        "config": c.config, "suite": c.suite, "bench": c.bench, "metric": c.metric,
        "median": c.median, "ci_pct": c.ci_pct, "status": c.status,
        "samples": c.samples, "detail": c.detail, "jit": c.jit,
    } for c in cells]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    ap.add_argument("--pin", type=int, default=DEFAULT_PIN)
    ap.add_argument("--stages", default="build,quiesce,measure,report,chart",
                    help="comma list: build,quiesce,measure,counters,report,chart")
    ap.add_argument("--quiet-wait", type=int, default=3600,
                    help="max seconds to wait for competing processes")
    ap.add_argument("--exclude", default="curve/")
    ap.add_argument("--merge-json", default=None, metavar="PATH",
                    help="merge cells from a prior results.json (e.g. warm "
                         "steady-state numbers) into this run's outputs instead "
                         "of re-measuring them; cells need config/suite/bench/"
                         "metric/median fields")
    args = ap.parse_args(argv)
    stages = set(args.stages.split(","))

    rev = git_revision(REPO_ROOT)
    outdir = REPO_ROOT / "logs" / "analysis" / rev
    outdir.mkdir(parents=True, exist_ok=True)
    jitdir = outdir / "jitsum"
    jitdir.mkdir(exist_ok=True)
    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True)

    print("=== analysis harness :: rev %s :: out %s ===" % (rev, outdir), file=sys.stderr)

    build_status = {}
    if "build" in stages:
        build_status = build_ladder()
        print("[build] " + str(build_status), file=sys.stderr)

    smoke = validate_binaries(args.pin)
    print("[smoke] " + str(smoke), file=sys.stderr)

    counter_info = detect_counters()
    print("[counters] " + str(counter_info), file=sys.stderr)

    if "quiesce" in stages:
        wait_for_quiet(os.getpid(), args.quiet_wait)

    cells = []
    if "measure" in stages:
        benches = discover_benchmarks(REPO_ROOT)
        benches = [b for b in benches if args.exclude not in str(b.relative_to(REPO_ROOT))]
        specs = build_program_registry()
        print("[measure] %d shootout benches, %d appbench programs, %d rpy + %d gf configs"
              % (len(benches), len(specs), len(RPYFORTH_LADDER), len(GFORTH_LADDER)),
              file=sys.stderr)
        cells += measure_shootout(RPYFORTH_LADDER, GFORTH_LADDER, benches,
                                  args.iterations, args.pin, jitdir)
        cells += measure_appbench(RPYFORTH_LADDER, GFORTH_LADDER, specs,
                                  args.iterations, args.pin, jitdir, tmpdir)

    # merge hook: fold cells from a prior run (e.g. the warm steady-state
    # harness JSON) into this run's tables/charts without re-measuring.
    if args.merge_json:
        try:
            prior = json.loads(Path(args.merge_json).read_text())
            merged = 0
            have = {(c.suite, c.bench, c.config) for c in cells}
            for pc in prior.get("cells", prior if isinstance(prior, list) else []):
                key = (pc.get("suite"), pc.get("bench"), pc.get("config"))
                if None in key or key in have:
                    continue
                cell = Cell(config=pc["config"], suite=pc["suite"], bench=pc["bench"],
                            metric=pc.get("metric", "wall_s"))
                cell.samples = pc.get("samples", [])
                cell.median = pc.get("median")
                cell.ci_pct = pc.get("ci_pct", 0.0)
                cell.status = pc.get("status", "ok")
                cell.jit = pc.get("jit")
                cells.append(cell)
                merged += 1
            print("[merge] folded %d cells from %s" % (merged, args.merge_json),
                  file=sys.stderr)
        except (OSError, ValueError, KeyError) as e:
            print("WARNING: --merge-json failed: %s" % e, file=sys.stderr)

    counters = None
    if "counters" in stages:
        counters = run_counters(RPYFORTH_LADDER,
                                discover_benchmarks(REPO_ROOT)[:3], args.pin, counter_info)

    jit_note = ("jit-summary captured per rpyforth (config x benchmark) via "
                "PYPYLOG=jit-summary on the final iteration.")

    # persist raw json
    results = {
        "rev": rev, "iterations": args.iterations, "pin": args.pin,
        "smoke": smoke, "build_status": build_status,
        "counter_info": counter_info, "counters": counters,
        "cells": cells_to_json(cells),
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print("[json] wrote " + str(outdir / "results.json"), file=sys.stderr)

    report = ""
    if "report" in stages:
        report = build_report(cells, smoke, build_status, counter_info, jit_note,
                              args.iterations, args.pin, rev)
        (outdir / "report.md").write_text(report)
        print("\n" + report + "\n")
        print("[report] wrote " + str(outdir / "report.md"), file=sys.stderr)

    if "chart" in stages and cells:
        paths = make_charts(cells, outdir)
        for p in paths:
            print("[chart] " + str(p), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
