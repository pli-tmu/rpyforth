#!/usr/bin/env python3
"""Appbench-1.4 benchmark harness.

Two modes, selected by subcommand:

  steady  (DEFAULT)  Warm STEADY-STATE + per-iteration warm-up curve.
  func               Cold functional + performance grid.

--- steady (default) --------------------------------------------------------

Single-shot appbench wall-clock is unfair to a meta-tracing JIT: a ~1-2s run
spends a large fraction on one-time JIT warm-up (tracing/compiling bridges).
gforth-fast is precompiled native code with zero warm-up, so a cold comparison
measures rpyforth's *compiler*, not its generated code quality.

This mode measures WARM STEADY-STATE performance instead. For each program
and each engine it builds a Forth driver that loads the program ONCE, then runs
the core workload word R times in a single process, timing EACH iteration with
UTIME ( -- d : microseconds, double-cell). It prints one CSV line per iteration
(`i, elapsed_usec`), records ALL iterations (cold ones included so the warm-up
curve is visible), and reports the steady-state as the median of the converged
tail (last ~50%), matching run_shootout.py's parse_curve_output /
steady_state_tail convention.

Deliverables:
  1. A steady-state comparison table (cold first-iter, warm-tail median,
     ratio gforth-fast/rpyforth; >1.0 means rpyforth wins warm).
  2. A warm-up curve PDF: per-iteration time from cold to plateau, all three
     engines on one axis per program, with the steady-state onset marked.

lexex is INCLUDED (see the lexex section below): run.fth is one-shot, but its
compute core (syntax-tree decoration + 1153-state FSM transition-table build +
output-array generation, i.e. everything `lexgen` does except the stt.fth file
write) is made repeatable by loading + parsing the input ONCE and re-running the
core per iteration with a dictionary rewind + followPos reset. Correctness is
preserved: a final saveAllTables + compare against ref.tt still passes.

Per-program core-word / count choices (see SteadySpec below):
  - cd16sim : unit = `150000 clear 150000 steps`  (clear resets state -> repeatable)
  - brainless: unit = benchmark3 x8               (self-contained movegen loops)
  - fcp     : unit = benchThink                    (position set up once in prelude)
  - lexex   : unit = lexcore + dict rewind         (re-do decorate + table build)

Safety: never modifies appbench/appbench-1.4/. Drivers are written to a tmpdir
and run with cwd set to the program dir. Subprocesses use a hard timeout so a
stuck engine is killed.

--- func --------------------------------------------------------------------

Functional + performance benchmark harness for the appbench-1.4 suite.

Programs covered: cd16sim, brainless, fcp, lexex, benchgc.

Each program is run under gforth (reference), gforth-fast, and rpyforth-c-stkfrag.
Functional status per (program, engine):
    PASS    - stdout matches gforth reference (after normalisation)
    PARTIAL - exit 0 but output differs
    FAIL    - crash / timeout / non-zero exit
"""

import argparse
import difflib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
APPBENCH_DIR = REPO_ROOT / "appbench" / "appbench-1.4"
GFORTH_DIR = REPO_ROOT / "gforth-0.7.9"
GFORTH_SETUP = APPBENCH_DIR / "setup" / "gforth.fs"

ENGINE_GFORTH = "gforth"
ENGINE_GFORTH_FAST = "gforth-fast"
ENGINE_RPYFORTH = "rpyforth"

REFERENCE_ENGINE = ENGINE_GFORTH_FAST

# Order matters for the steady-mode table / legend.
ENGINES = [ENGINE_RPYFORTH, ENGINE_GFORTH_FAST, ENGINE_GFORTH]

ENGINE_BINARY = {
    ENGINE_RPYFORTH: REPO_ROOT / "rpyforth-c-stkfrag",
    ENGINE_GFORTH_FAST: GFORTH_DIR / "gforth-fast",
    ENGINE_GFORTH: GFORTH_DIR / "gforth",
}

# steady mode defaults
STEADY_DEFAULT_ITERATIONS = 50
STEADY_DEFAULT_TIMEOUT = 600

# func mode defaults
FUNC_DEFAULT_TIMEOUT = 300
FUNC_DEFAULT_ITERATIONS = 3


# ===========================================================================
# steady mode (default): warm steady-state + warm-up curve
# ===========================================================================

class SteadySpec:
    """A repeatable appbench workload.

    prelude : code run ONCE before the timing loop (includes + one-time setup).
    unit    : the core workload word(s), timed each iteration. It MUST leave the
              stack balanced and be repeatable (idempotent across iterations).
    """

    def __init__(self, name, workdir, pre_include, include_file, setup, unit,
                 rpy_env=None, prelude=None, gforth_mem="16M"):
        self.name = name
        self.workdir = workdir
        # pre_include : words defined before the program is loaded (e.g. 3drop).
        self.pre_include = pre_include
        # include_file: the program's main source, resolved to an ABSOLUTE path
        #   at driver-build time so `include` works regardless of cwd / engine.
        self.include_file = include_file
        # setup : one-time state set up after loading, before the timing loop.
        self.setup = setup
        # unit  : the core workload word(s), timed each iteration.
        self.unit = unit
        # rpy_env : extra environment for the rpyforth engine only (benchgc needs
        #   a big ALLOCATE region for its GC memory block). gforth ignores it.
        self.rpy_env = rpy_env or {}
        # prelude : raw Forth that REPLACES the default `include <include_file>`
        #   load step. lexex needs it because it is a multi-file program whose
        #   libraries must load in a fixed order before its (truncated) input is
        #   parsed; a single `include` cannot express that. When set, include_file
        #   is ignored and this block is emitted verbatim as the one-time prelude.
        self.prelude = prelude
        # gforth_mem : dataspace size passed as `-m` to gforth/gforth-fast. lexex's
        #   per-iteration table build allots ~3.6 MB and needs a roomy dictionary.
        self.gforth_mem = gforth_mem


PROGRAMS = [
    SteadySpec(
        name="cd16sim",
        workdir=APPBENCH_DIR / "cd16sim",
        pre_include=": 3drop 2drop drop ;",
        include_file="bench.f",
        setup="",
        # clear resets the machine state so each 150000-step run is identical.
        unit="150000 clear 150000 steps",
    ),
    SteadySpec(
        name="brainless",
        workdir=APPBENCH_DIR / "brainless",
        pre_include="",
        include_file="benchmark.fs",
        # benchmark3 sets up its own positions and runs movegen loops -> fully
        # self-contained and repeatable. One call is ~4ms warm; x8 lands the
        # unit in the ~100-300ms window we want.
        setup=": steady-unit "
              "benchmark3 benchmark3 benchmark3 benchmark3 "
              "benchmark3 benchmark3 benchmark3 benchmark3 ;",
        unit="steady-unit",
    ),
    SteadySpec(
        name="fcp",
        workdir=APPBENCH_DIR / "fcp",
        pre_include="",
        include_file="fcp-1.31-64.f",
        # bench sets up a fixed position then does best-of-three benchThink.
        # We set the position up ONCE here and time a single benchThink per
        # iteration; benchThink re-runs the depth-5 search from the same
        # position each time (thinker resets its state), so it is repeatable.
        #
        # `' noop IS checkTime` neutralises fcp's DEFERred time/keyboard poll.
        # Without it, benchThink's ?thinkAbort fires QUIT when KEY? sees EOF on
        # the DEVNULL stdin (or a wall-time limit trips), truncating the timing
        # loop after a few iterations. Disabling the poll makes each search run
        # the full fixed depth deterministically -> a clean repeatable unit.
        setup="' noop IS checkTime\n"
              'S" setup 1rb2rk/p4ppp/1p1qp1n/3n2N/2pP4/2P3P/PPQ2PBP/R1B1R1K w" '
              "evaluate 5 sd",
        unit="benchThink drop",
    ),
    SteadySpec(
        name="benchgc",
        workdir=APPBENCH_DIR / "benchgc",
        # bench-gc5.fs starts with `cells dup constant limit`, so 64000 (the cell
        # count for limit=512000) must be on the stack before the include runs.
        pre_include="64000",
        include_file="bench-gc5.fs",
        # Loading bench-gc5.fs already runs one (cold) testgc and prints the GC
        # statistics; those non-CSV lines are ignored by the parser. testgc is a
        # self-contained, stack-balanced, repeatable unit: each call allocates
        # ~500 KB of live GC-managed nodes and the collector reclaims them, so the
        # heap stays bounded across iterations (verified: active-end stays ~514000
        # and RSS flat over 20+ runs). One testgc is ~270 ms warm -- in the
        # 100-300 ms window. The RNG seed carries over between calls, which only
        # varies the exact allocation sizes, not the balance or the work amount.
        setup="",
        unit="testgc drop",
        rpy_env={"RPYFORTH_ALLOC_MB": "256"},
    ),
]


# --- lexex: make the one-shot generator repeatable ------------------------------
#
# lexex's run.fth is one-shot: it loads ten library files, parses lexinput.fth
# (which builds a syntax tree from regex definitions and, on its LAST line, runs
# `syntaxTree lexgen` to decorate the tree, build the 1153-state FSM transition
# table, generate the output arrays and write stt.fth), then self-checks the
# file against ref.tt. `lexgen` cannot simply be looped: each call
#   * allots the whole transition table with `here to TransTable` + per-state
#     `allot` (~3.6 MB of dictionary per run -> unbounded growth), and
#   * mutates the syntax tree in place (calcFollowPos UNIONS into each leaf's
#     existing followPos set, so a naive re-run reads freed set pointers).
#
# The unit below re-does the COMPUTE CORE (createPositionSet + tree decoration +
# buildTransTable + loadLexTokens + buildLexArrays -- everything lexgen does
# EXCEPT the saveAllTables file write) and makes it repeatable with two resets:
#   1. `zero-followpos` walks the current leaf map and nulls every leaf's
#      followPos, so calcFollowPos re-allocates fresh sets instead of unioning
#      into pointers that the dictionary rewind just reclaimed;
#   2. `savedp @ here - allot` rewinds the dictionary pointer (portable negative
#      ALLOT; rpyforth has no writable `dp`) to a snapshot captured on the first
#      iteration, so the per-run allocations are reused and HERE stays flat.
# The snapshot is captured lazily inside `unit` (on the first call, when no later
# word definition sits above it) so rewinding never frees the running loop word.
#
# Verified on rpyforth, gforth and gforth-fast: after N looped units, one final
# saveAllTables + compare against ref.tt still prints "Output file is correct",
# and HERE is flat across iterations (bounded ~3.6 MB working set).
#
# lexinput.fth is used UNMODIFIED except for two edits made to a /tmp COPY (the
# real appbench tree is never touched): its last line `syntaxTree lexgen` is
# dropped (so loading it only PARSES, leaving the tree ready), and its
# `s" stt.fth" setOutputFile` is redirected to an absolute /tmp path so the final
# save never writes into appbench/appbench-1.4/.
LEXEX_DIR = APPBENCH_DIR / "lexex"
LEXEX_LIB_FILES = [
    "ansify.fth", "xmini_oof.fth", "sets.fth", "shellsort.fth", "syntaxtree.fth",
    "transitiontable.fth", "lexarrays.fth", "savetables.fth", "userinterface.fth",
    "anstokens.fth",
]
LEXEX_STT_OUT = "/tmp/rpyforth_lexex_stt_out.fth"


def _make_lexex_input_copy():
    """Write a /tmp copy of lexinput.fth that only parses (no lexgen) and whose
    output file is redirected out of the appbench tree. Returns its path."""
    src = (LEXEX_DIR / "lexinput.fth").read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    # Drop the trailing `syntaxTree lexgen` invocation (keep everything up to and
    # including `end-symbols syntaxTree`, which finishes building the tree).
    kept = []
    for ln in lines:
        if ln.strip() == "syntaxTree lexgen":
            break
        kept.append(ln)
    body = "\n".join(kept) + "\n"
    # Redirect the generated-file path so the final save+check never writes into
    # the protected appbench directory.
    body = body.replace('s" stt.fth" setOutputFile',
                        's" %s" setOutputFile' % LEXEX_STT_OUT)
    out = Path(tempfile.gettempdir()) / "rpyforth_lexex_lexinput_notrun.fth"
    out.write_text(body, encoding="utf-8")
    return out


def _lexex_prelude():
    """Build the one-time prelude: ordered library loads, tree parse, and the
    lexcore / reset / unit word definitions."""
    inc = ['s" %s" included' % str(LEXEX_DIR / f) for f in LEXEX_LIB_FILES]
    inc.append('s" %s" included' % str(_make_lexex_input_copy()))
    defs = [
        ": lexcore ( tree -- )",
        '   s" createPositionSet (PSC)" evaluate',
        "   dup updateSyntaxTree dup createLeafMap dup updateFollowPos",
        "   buildTransTable loadLexTokens buildLexArrays ;",
        "variable savedp 0 savedp !",
        ": zero-followpos ( -- )",
        "   maxPosition 1+ 1 ?do",
        "      i cells leaves + @ ?dup if 0 swap followPos ! then",
        "   loop ;",
        ": unit ( -- )",
        "   savedp @ if zero-followpos savedp @ here - allot",
        "   else here savedp ! then",
        "   syntaxTree lexcore ;",
    ]
    return "\n".join(inc + defs)


PROGRAMS.append(
    SteadySpec(
        name="lexex",
        workdir=LEXEX_DIR,
        pre_include="",
        # include_file is unused: the prelude does its own ordered multi-file load.
        include_file="",
        prelude=_lexex_prelude(),
        setup="",
        # unit re-runs the full compute core (decorate + FSM table build + output
        # arrays) with a dictionary rewind; ~4 s warm on rpyforth (buildTransTable
        # dominates at ~3 s of the 1153-state construction).
        unit="unit",
        # Per-iteration table build allots ~3.6 MB; gforth needs a roomy dict.
        gforth_mem="256M",
    )
)


def build_driver(spec, iterations):
    """Return Forth source for a driver that times `unit` `iterations` times.

    UTIME ( -- d ) is microseconds as a double cell. Elapsed for one iteration
    is far below 2^31 us (~35 min), so after `d-` we drop the high cell and
    print the low cell as a bare integer. A CR is emitted before each CSV line
    so the workload's own stdout cannot merge with our data line.
    """
    lines = []
    if spec.pre_include:
        lines.append(spec.pre_include)
    if spec.prelude is not None:
        # Multi-file program (lexex): the prelude does its own ordered loads.
        lines.append(spec.prelude)
    else:
        # Absolute include path: works no matter which dir the driver file lives
        # in.
        include_abs = str(spec.workdir / spec.include_file)
        lines.append("include " + include_abs)
    if spec.setup:
        lines.append(spec.setup)
    # Loop body: time the unit, print "i,elapsed_usec" on its own line.
    lines.append(": steady-run  ( n -- )")
    lines.append("  0 DO")
    lines.append("    utime 2>r")
    lines.append("    " + spec.unit)
    lines.append("    utime 2r> d-")
    lines.append("    drop            \\ elapsed low cell (usec)")
    # A CR both BEFORE and AFTER the CSV isolates the data line: the leading CR
    # detaches it from any stdout the unit printed, and the trailing CR keeps the
    # NEXT iteration's unit output (e.g. lexex's buildTransTable progress dots)
    # from being appended to this data line. The parser skips the blank lines.
    lines.append('    cr I . ." ," . cr \\ CSV: i , elapsed_usec')
    lines.append("  LOOP ;")
    lines.append("%d steady-run" % iterations)
    lines.append("cr bye")
    return "\n".join(lines) + "\n"


def build_cmd(engine, driver_path, spec):
    binary = ENGINE_BINARY[engine]
    if engine == ENGINE_RPYFORTH:
        return [str(binary), str(driver_path)]
    # gforth / gforth-fast: -m <mem> plus the appbench gforth setup shim, then
    # load our driver file.
    return [str(binary), "-m", spec.gforth_mem, str(GFORTH_SETUP),
            str(driver_path)]


def parse_curve_output(stdout):
    """Extract per-iteration timings from the driver's CSV output.

    Same convention as run_shootout.py: skip non-CSV lines; a data row has
    col0 as a digit and col1 an integer.
    """
    times = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("iteration"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                times.append(int(parts[1]))
            except ValueError:
                continue
    return times


def steady_state_tail(times, frac=0.5):
    """Median of the converged tail (last `frac`) of a per-iteration curve."""
    if not times:
        return None
    tail = times[int(len(times) * (1.0 - frac)):] or times
    return int(statistics.median(tail))


def steady_onset_index(times, frac=0.5):
    """Index where the steady-state tail begins (for marking the chart)."""
    if not times:
        return 0
    return int(len(times) * (1.0 - frac))


def run_engine(engine, spec, iterations, tmpdir, timeout, pin):
    driver = build_driver(spec, iterations)
    driver_path = Path(tmpdir) / ("%s_%s_driver.fs" % (spec.name, engine))
    driver_path.write_text(driver, encoding="utf-8")

    cmd = build_cmd(engine, driver_path, spec)
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd

    env = os.environ.copy()
    if engine == ENGINE_RPYFORTH and spec.rpy_env:
        env.update(spec.rpy_env)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(spec.workdir),
            stdin=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
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
    }


def fmt_usec(v):
    if v is None:
        return "  n/a"
    if v >= 1000:
        return "%.1f ms" % (v / 1000.0)
    return "%d us" % v


def print_table(results, iterations):
    """results: {prog_name: {engine: run_dict}}"""
    print("")
    print("=" * 92)
    print("STEADY-STATE COMPARISON  (R=%d iterations/program, warm tail = last 50%%)" % iterations)
    print("=" * 92)
    header = "%-10s %-13s %12s %14s %12s" % (
        "program", "engine", "cold[iter0]", "warm[tail-med]", "vs-ref",
    )
    print(header)
    print("-" * 92)
    for prog in results:
        eng_res = results[prog]
        ref = eng_res.get(REFERENCE_ENGINE)
        ref_warm = steady_state_tail(ref["times"]) if ref and ref["times"] else None
        first_prog_row = True
        for engine in ENGINES:
            r = eng_res.get(engine)
            if not r:
                continue
            times = r["times"]
            cold = times[0] if times else None
            warm = steady_state_tail(times)
            if r["timed_out"]:
                ratio = "TIMEOUT"
            elif not times:
                ratio = "NO-DATA"
            elif engine == REFERENCE_ENGINE:
                ratio = "1.00x (ref)"
            elif ref_warm and warm:
                # ratio = ref / this ; >1.0 means this engine is faster than ref.
                ratio = "%.2fx" % (ref_warm / float(warm))
            else:
                ratio = "  -"
            name_col = prog if first_prog_row else ""
            first_prog_row = False
            print("%-10s %-13s %12s %14s %12s" % (
                name_col, engine, fmt_usec(cold), fmt_usec(warm), ratio,
            ))
        print("-" * 92)
    print("Note: 'vs-ref' = gforth-fast_warm / rpyforth_warm.  >1.00x  => rpyforth WINS warm.")
    print("Note: lexex re-runs its compute core (decorate + FSM table build + output")
    print("      arrays) per iteration via a dictionary rewind; the file-writing")
    print("      saveAllTables is excluded and correctness is re-verified against ref.tt.")
    print("")


def make_chart(results, iterations, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    progs = list(results.keys())
    n = len(progs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.0 * n), squeeze=False)

    colors = {
        ENGINE_RPYFORTH: "#d62728",
        ENGINE_GFORTH_FAST: "#1f77b4",
        ENGINE_GFORTH: "#2ca02c",
    }

    for row, prog in enumerate(progs):
        ax = axes[row][0]
        eng_res = results[prog]
        for engine in ENGINES:
            r = eng_res.get(engine)
            if not r or not r["times"]:
                continue
            times_ms = [t / 1000.0 for t in r["times"]]
            xs = list(range(len(times_ms)))
            ax.plot(xs, times_ms, marker="o", markersize=2.5, linewidth=1.2,
                    color=colors[engine], label=engine)
            warm = steady_state_tail(r["times"])
            if warm is not None:
                ax.axhline(warm / 1000.0, color=colors[engine], linewidth=0.7,
                           linestyle=":", alpha=0.6)
        onset = steady_onset_index(
            eng_res.get(ENGINE_RPYFORTH, {}).get("times", []))
        if onset:
            ax.axvline(onset, color="grey", linestyle="--", linewidth=0.8,
                       alpha=0.7)
            ax.text(onset, ax.get_ylim()[1] * 0.95, " steady-state tail ->",
                    fontsize=7, color="grey", va="top")
        ax.set_title("%s : per-iteration warm-up curve" % prog)
        ax.set_xlabel("iteration (0 = cold)")
        ax.set_ylabel("iteration time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Warm-up curves: rpyforth (meta-tracing JIT) vs gforth-fast / gforth\n"
        "dotted line = warm-tail median; dashed vertical = steady-state onset",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(pdf_path))
    plt.close(fig)


def run_steady(args):
    selected = PROGRAMS
    if args.programs:
        want = set(p.strip() for p in args.programs.split(","))
        selected = [p for p in PROGRAMS if p.name in want]
        if not selected:
            print("No matching programs for %r" % args.programs, file=sys.stderr)
            return 1

    for engine in ENGINES:
        b = ENGINE_BINARY[engine]
        if not Path(b).exists():
            print("WARNING: engine binary missing: %s (%s)" % (engine, b),
                  file=sys.stderr)

    results = {}
    with tempfile.TemporaryDirectory(prefix="appbench_steady_") as tmpdir:
        for spec in selected:
            results[spec.name] = {}
            for engine in ENGINES:
                if not Path(ENGINE_BINARY[engine]).exists():
                    continue
                print("running %-10s on %-13s ..." % (spec.name, engine),
                      end="", flush=True)
                r = run_engine(engine, spec, args.iterations, tmpdir,
                               args.timeout, args.pin)
                warm = steady_state_tail(r["times"])
                if r["timed_out"]:
                    print(" TIMEOUT after %.1fs" % r["wall"])
                elif not r["times"]:
                    print(" NO CSV DATA (rc=%d)" % r["rc"])
                    if r["stderr"].strip():
                        print("    stderr: %s" % r["stderr"].strip()[:300])
                else:
                    print(" %d iters, cold=%s warm=%s (%.1fs)" % (
                        len(r["times"]), fmt_usec(r["times"][0]),
                        fmt_usec(warm), r["wall"]))
                results[spec.name][engine] = r

    print_table(results, args.iterations)

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = REPO_ROOT / pdf_path
    try:
        make_chart(results, args.iterations, pdf_path)
        print("Warm-up curve chart written to %s" % pdf_path)
    except Exception as exc:
        print("ERROR generating chart: %s" % exc, file=sys.stderr)
        return 1

    return 0


# ===========================================================================
# func mode: cold functional + performance grid
# ===========================================================================

@dataclass
class ProgramSpec:
    name: str
    workdir: Path
    prelude: str
    body: str
    supported_engines: List[str]
    rpy_jit_args: List[str] = field(default_factory=list)
    # Extra environment for the rpyforth engine only (gforth ignores it). benchgc
    # ALLOCATEs a large GC memory block, so it needs a big ALLOCATE region.
    rpy_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    program: str
    engine: str
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    wall_seconds: float = 0.0
    timed_out: bool = False
    error_message: str = ""
    elapsed_samples: List[float] = field(default_factory=list)


@dataclass
class FuncStatus:
    status: str
    diff_excerpt: str = ""
    first_error_line: str = ""
    differing_lines: int = 0


def git_revision(root: Path) -> str:
    def _git(cmd):
        return subprocess.check_output(
            ["git"] + cmd, cwd=str(root), stderr=subprocess.DEVNULL
        ).decode().strip()
    try:
        rev = _git(["rev-parse", "--short", "HEAD"])
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    try:
        if _git(["status", "--porcelain"]):
            rev += "-dirty"
    except (subprocess.CalledProcessError, OSError):
        pass
    return rev


def capture_environment() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        gov = "?"
    try:
        load1 = "%.2f" % os.getloadavg()[0]
    except (OSError, AttributeError):
        load1 = "?"
    return f"cpu: {cpu} | governor: {gov} | load1: {load1}"


def median_ci(
    samples: List[float],
    confidence: float = 0.90,
    resamples: int = 2000,
) -> Tuple[Optional[float], float]:
    if not samples:
        return (None, 0.0)
    med = statistics.median(samples)
    if len(samples) == 1 or med == 0:
        return (med, 0.0)
    rng = random.Random(20240624)
    n = len(samples)
    boot = sorted(
        statistics.median(samples[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    lo = boot[int((1.0 - confidence) / 2 * resamples)]
    hi = boot[min(resamples - 1, int((1.0 + confidence) / 2 * resamples))]
    return (med, 100.0 * (hi - lo) / 2.0 / med)


def build_program_registry() -> List[ProgramSpec]:
    appbench = APPBENCH_DIR

    cd16sim = ProgramSpec(
        name="cd16sim",
        workdir=appbench / "cd16sim",
        prelude=": 3drop 2drop drop ;",
        body="include bench.f\n1000000 benchmark\nbye",
        supported_engines=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
    )

    brainless = ProgramSpec(
        name="brainless",
        workdir=appbench / "brainless",
        prelude="",
        body="include benchmark.fs\nbenchmark\nbye",
        supported_engines=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
    )

    fcp = ProgramSpec(
        name="fcp",
        workdir=appbench / "fcp",
        prelude="",
        body="include fcp-1.31-64.f\nbench\nbye",
        supported_engines=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
    )

    lexex = ProgramSpec(
        name="lexex",
        workdir=appbench / "lexex",
        prelude="",
        body="include run.fth\nbye",
        supported_engines=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
    )

    # benchgc: a garbage-collector benchmark. bench-gc5.fs first does
    # `cells dup constant limit`, so `limit` (in cells) is pushed on the stack
    # BEFORE the include; 64000 cells -> limit=512000. gc.fs then ALLOCATEs one
    # ~120 MB memory block, so the rpyforth engine needs a large ALLOCATE region
    # (RPYFORTH_ALLOC_MB); gforth uses system malloc and ignores it.
    benchgc = ProgramSpec(
        name="benchgc",
        workdir=appbench / "benchgc",
        prelude="",
        body="64000 include bench-gc5.fs\nbye",
        supported_engines=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
        rpy_env={"RPYFORTH_ALLOC_MB": "256"},
    )

    return [cd16sim, brainless, fcp, lexex, benchgc]


def build_gforth_cmd(binary: Path, spec: ProgramSpec, tmpdir: Path) -> List[str]:
    forth_expr = ""
    if spec.prelude:
        forth_expr += spec.prelude + " "
    forth_expr += spec.body.replace("\n", " ")

    cmd = [
        str(binary),
        "-m", "16M",
        str(GFORTH_SETUP),
        "-e", forth_expr,
    ]
    return cmd


def build_rpyforth_cmd(binary: Path, spec: ProgramSpec, tmpdir: Path) -> List[str]:
    lines = []
    if spec.prelude:
        lines.append(spec.prelude)
    lines.append(spec.body)
    forth_expr = "\n".join(lines)

    wrapper_path = tmpdir / f"{spec.name}_rpy_wrapper.fs"
    wrapper_path.write_text(forth_expr, encoding="utf-8")

    cmd = [str(binary)] + list(spec.rpy_jit_args) + [str(wrapper_path)]
    return cmd


def run_once(
    cmd: List[str],
    workdir: Path,
    timeout: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float, bool]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

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
        return proc.returncode, proc.stdout, proc.stderr, wall, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        return -1, exc.stdout or "", exc.stderr or "", wall, True
    except FileNotFoundError as exc:
        return -2, "", f"binary not found: {exc.filename}", 0.0, False


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)


def normalise_output(text: str) -> List[str]:
    text = strip_ansi(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue
        if re.search(r'\b(seconds?|elapsed|ms,|Hz|nps\b)\b', line, re.IGNORECASE):
            continue
        if re.match(r'^(?:Gforth|Authors:|Copyright|License|Gforth comes|Type)', line):
            continue
        if re.match(r'^\*terminal\*:', line):
            continue
        if 'warning:' in line.lower() and 'redefined' in line.lower():
            continue
        if 'warning:' in line.lower() and 'original location' in line.lower():
            continue
        if 'warning:' in line.lower() and 'defined literal' in line.lower():
            continue
        if re.match(r'^(?:ok\s*)?$', line):
            continue
        if re.match(r'^Loading run\.fth', line):
            continue
        if re.match(r'^Time taken:', line):
            continue
        line = re.sub(r'\b\d+\.\d+\b', '<T>', line)
        lines.append(line)
    return lines


def compute_functional_status(
    ref_stdout: str,
    cand_stdout: str,
    cand_rc: int,
    cand_timed_out: bool,
    cand_stderr: str,
) -> FuncStatus:
    if cand_timed_out:
        first_err = cand_stderr.splitlines()[0] if cand_stderr else "timed out"
        return FuncStatus(status="FAIL", first_error_line=first_err)

    failure_markers = ["UNKNOWN:", "ABORT", "THROW:", "stack underflow", "Stack empty"]
    combined = cand_stdout + cand_stderr
    if cand_rc != 0 or any(m in combined for m in failure_markers):
        first_err = ""
        noise = re.compile(
            r'^(?:Gforth|Authors:|Copyright|License|Gforth comes|Type|\*terminal\*|warning:|\[|Loading |ok\s*$)'
        )
        for text in (cand_stderr, cand_stdout):
            for line in text.splitlines():
                stripped = strip_ansi(line).strip()
                if stripped and not noise.match(stripped):
                    first_err = stripped[:120]
                    break
            if first_err:
                break
        if not first_err:
            first_err = f"exit code {cand_rc}"
        return FuncStatus(status="FAIL", first_error_line=first_err)

    ref_lines = normalise_output(ref_stdout)
    cand_lines = normalise_output(cand_stdout)

    if ref_lines == cand_lines:
        return FuncStatus(status="PASS")

    diff = list(
        difflib.unified_diff(ref_lines, cand_lines, lineterm="", n=2)
    )
    differing = sum(
        1 for line in diff
        if line.startswith(("+", "-")) and not line.startswith(("---", "+++"))
    )
    excerpt = "\n".join(diff[:12])
    return FuncStatus(
        status="PARTIAL",
        diff_excerpt=excerpt,
        differing_lines=differing,
    )


def resolve_engines(overrides: Dict[str, str]) -> Dict[str, Path]:
    defaults = {
        ENGINE_GFORTH: GFORTH_DIR / "gforth",
        ENGINE_GFORTH_FAST: GFORTH_DIR / "gforth-fast",
        ENGINE_RPYFORTH: REPO_ROOT / "rpyforth-c-stkfrag",
    }
    result: Dict[str, Path] = {}
    for name, default in defaults.items():
        override = overrides.get(name)
        if override:
            p = Path(override)
            if not p.is_absolute():
                p = REPO_ROOT / p
            result[name] = p
        else:
            result[name] = default
    return result


def save_log(
    log_dir: Path,
    program: str,
    engine: str,
    iteration: int,
    total: int,
    returncode: int,
    stdout: str,
    stderr: str,
    wall: float,
    timed_out: bool,
    cmd: List[str],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_i{iteration:03d}" if total > 1 else ""
    path = log_dir / f"{program}_{engine}{suffix}.log"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# program: {program}\n")
        f.write(f"# engine: {engine}\n")
        f.write(f"# iteration: {iteration}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.write(f"# returncode: {returncode}\n")
        f.write(f"# wall_seconds: {wall:.6f}\n")
        if timed_out:
            f.write("# timed_out: true\n")
        f.write("# --- stdout ---\n")
        f.write(stdout)
        if stderr:
            f.write("\n# --- stderr ---\n")
            f.write(stderr)


def run_program(
    spec: ProgramSpec,
    engine_name: str,
    engine_path: Path,
    tmpdir: Path,
    log_dir: Path,
    iterations: int,
    timeout: int,
) -> RunResult:
    result = RunResult(program=spec.name, engine=engine_name)

    extra_env: Optional[Dict[str, str]] = None
    if engine_name == ENGINE_RPYFORTH:
        cmd = build_rpyforth_cmd(engine_path, spec, tmpdir)
        if spec.rpy_env:
            extra_env = dict(spec.rpy_env)
    else:
        cmd = build_gforth_cmd(engine_path, spec, tmpdir)

    for i in range(1, iterations + 1):
        rc, stdout, stderr, wall, timed_out = run_once(
            cmd, spec.workdir, timeout, extra_env
        )
        save_log(
            log_dir, spec.name, engine_name, i, iterations,
            rc, stdout, stderr, wall, timed_out, cmd,
        )
        if i == 1:
            result.returncode = rc
            result.stdout = stdout
            result.stderr = stderr
            result.wall_seconds = wall
            result.timed_out = timed_out
            if timed_out:
                result.error_message = f"timed out after {timeout}s"
            elif rc not in (0, -2):
                result.error_message = f"exit code {rc}"
        result.elapsed_samples.append(wall)

    return result


def print_status_table(
    func_statuses: Dict[Tuple[str, str], FuncStatus],
    programs: List[str],
    engines: List[str],
    timings: Dict[Tuple[str, str], Tuple[Optional[float], float]],
) -> None:
    col_w = 14
    header = f"{'Program':<12}" + "".join(f"{e:>{col_w}}" for e in engines)
    print("=" * (12 + col_w * len(engines)))
    print("Functional status (PASS / PARTIAL / FAIL)")
    print("=" * (12 + col_w * len(engines)))
    print(header)
    print("-" * (12 + col_w * len(engines)))

    for prog in programs:
        row = f"{prog:<12}"
        for eng in engines:
            key = (prog, eng)
            if key not in func_statuses:
                row += f"{'N/A':>{col_w}}"
            else:
                fs = func_statuses[key]
                row += f"{fs.status:>{col_w}}"
        print(row)
    print("=" * (12 + col_w * len(engines)))

    print()
    print("Wall-clock time in seconds (median of iterations, N/A = FAIL/not run)")
    print("-" * (12 + col_w * len(engines)))
    print(header)
    print("-" * (12 + col_w * len(engines)))
    for prog in programs:
        row = f"{prog:<12}"
        for eng in engines:
            key = (prog, eng)
            if key not in timings or timings[key][0] is None:
                row += f"{'N/A':>{col_w}}"
            else:
                med, ci = timings[key]
                cell = f"{med:.2f}s"
                if ci > 0:
                    cell += f" ±{ci:.0f}%"
                row += f"{cell:>{col_w}}"
        print(row)
    print("=" * (12 + col_w * len(engines)))


def print_diff_details(
    func_statuses: Dict[Tuple[str, str], FuncStatus],
    programs: List[str],
    engines: List[str],
) -> None:
    any_printed = False
    for prog in programs:
        for eng in engines:
            key = (prog, eng)
            if key not in func_statuses:
                continue
            fs = func_statuses[key]
            if fs.status == "PARTIAL" and fs.diff_excerpt:
                if not any_printed:
                    print()
                    print("PARTIAL diff excerpts (first 12 diff lines, ref=gforth-fast)")
                    print("=" * 70)
                    any_printed = True
                print(f"\n[{prog} / {eng}]  ({fs.differing_lines} differing lines)")
                for line in fs.diff_excerpt.splitlines()[:12]:
                    print("  " + line)
            elif fs.status == "FAIL" and fs.first_error_line:
                if not any_printed:
                    print()
                    print("FAIL details")
                    print("=" * 70)
                    any_printed = True
                print(f"\n[{prog} / {eng}]  first error: {fs.first_error_line}")


def generate_appbench_chart(
    out_path: Path,
    programs: List[str],
    engines: List[str],
    func_statuses: Dict[Tuple[str, str], FuncStatus],
    timings: Dict[Tuple[str, str], Tuple[Optional[float], float]],
    caption: Optional[str] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib import patches as mpatches
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    status_colors = {"PASS": "#2ca02c", "PARTIAL": "#ff7f0e", "FAIL": "#d62728", "N/A": "#aaaaaa"}

    fig = plt.figure(figsize=(14, 7), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35)

    ax_grid = fig.add_subplot(gs[0])
    n_prog = len(programs)
    n_eng = len(engines)
    grid = []
    for prog in programs:
        row = []
        for eng in engines:
            key = (prog, eng)
            s = func_statuses.get(key, FuncStatus(status="N/A")).status
            color = status_colors.get(s, "#aaaaaa")
            row.append(color)
        grid.append(row)

    for i, prog in enumerate(programs):
        for j, eng in enumerate(engines):
            color = grid[i][j]
            key = (prog, eng)
            s = func_statuses.get(key, FuncStatus(status="N/A")).status
            rect = mpatches.FancyBboxPatch(
                (j - 0.4, i - 0.4), 0.8, 0.8,
                boxstyle="round,pad=0.05",
                linewidth=0.5,
                edgecolor="white",
                facecolor=color,
            )
            ax_grid.add_patch(rect)
            ax_grid.text(j, i, s, ha="center", va="center", fontsize=7,
                         color="white" if s != "N/A" else "black", fontweight="bold")

    ax_grid.set_xlim(-0.6, n_eng - 0.4)
    ax_grid.set_ylim(-0.6, n_prog - 0.4)
    ax_grid.set_xticks(range(n_eng))
    ax_grid.set_xticklabels(engines, rotation=20, ha="right", fontsize=8)
    ax_grid.set_yticks(range(n_prog))
    ax_grid.set_yticklabels(programs, fontsize=9)
    ax_grid.set_title("Functional status", fontsize=10)
    ax_grid.set_aspect("equal")
    legend_patches = [
        mpatches.Patch(color=c, label=s)
        for s, c in status_colors.items() if s != "N/A"
    ]
    ax_grid.legend(handles=legend_patches, loc="upper right", fontsize=7)

    ax_bar = fig.add_subplot(gs[1])
    runnable_progs = []
    for prog in programs:
        has_any = any(
            timings.get((prog, eng), (None, 0))[0] is not None
            for eng in engines
        )
        if has_any:
            runnable_progs.append(prog)

    if runnable_progs:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        n_eng_colors = {eng: palette[i % len(palette)] for i, eng in enumerate(engines)}
        group = 0.8
        width = group / max(1, n_eng)
        y_pos = range(len(runnable_progs))

        for j, eng in enumerate(engines):
            offsets = [i - group / 2 + width * (j + 0.5) for i in y_pos]
            vals = []
            for prog in runnable_progs:
                med, _ = timings.get((prog, eng), (None, 0.0))
                vals.append((med * 1e6) if med is not None else 0)
            ax_bar.barh(
                offsets, vals, width,
                label=eng,
                color=n_eng_colors[eng],
                alpha=0.85,
            )

        ax_bar.set_xscale("log")
        ax_bar.set_yticks(list(y_pos))
        ax_bar.set_yticklabels(runnable_progs, fontsize=9)
        ax_bar.set_xlabel("Wall-clock time (microseconds, log scale)", fontsize=9)
        ax_bar.set_title("Runtime comparison (runnable subset)", fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax_bar.text(0.5, 0.5, "No runnable programs", ha="center", va="center",
                    transform=ax_bar.transAxes, fontsize=11, color="0.5")
        ax_bar.set_title("Runtime comparison", fontsize=10)

    fig.suptitle("Appbench-1.4 results", fontsize=13)
    if caption:
        fig.text(0.99, 0.005, caption, ha="right", va="bottom", fontsize=8, color="0.5")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def run_func(args) -> int:
    overrides: Dict[str, str] = {}
    if args.gforth:
        overrides[ENGINE_GFORTH] = args.gforth
    if getattr(args, "gforth_fast", None):
        overrides[ENGINE_GFORTH_FAST] = args.gforth_fast
    if args.rpyforth:
        overrides[ENGINE_RPYFORTH] = args.rpyforth

    engine_paths = resolve_engines(overrides)
    selected_engines = args.engines

    for eng in selected_engines:
        if eng not in engine_paths:
            print(f"Error: unknown engine '{eng}'", file=sys.stderr)
            return 1
        p = engine_paths[eng]
        if not p.exists():
            print(f"Warning: {eng} binary not found at {p}", file=sys.stderr)

    revision = git_revision(REPO_ROOT)
    env_line = capture_environment() + f" | commit {revision}"
    print(env_line)

    out_base = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    log_dir = out_base / revision / "appbench"
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = build_program_registry()
    if args.only:
        specs = [s for s in specs if s.name == args.only]
        if not specs:
            print(f"Error: no program named '{args.only}'", file=sys.stderr)
            return 1

    all_results: Dict[Tuple[str, str], RunResult] = {}
    func_statuses: Dict[Tuple[str, str], FuncStatus] = {}

    with tempfile.TemporaryDirectory(prefix="appbench_wrappers_") as _tmpdir:
        tmpdir = Path(_tmpdir)

        for spec in specs:
            print(f"\n--- {spec.name} ---", file=sys.stderr)
            for eng in selected_engines:
                ep = engine_paths.get(eng)
                if ep is None or not ep.exists():
                    print(f"  [{spec.name}/{eng}] skip (binary missing)", file=sys.stderr)
                    continue
                print(f"  [{spec.name}/{eng}] running {args.iterations}x ...", file=sys.stderr)
                result = run_program(
                    spec, eng, ep, tmpdir, log_dir,
                    args.iterations, args.timeout,
                )
                all_results[(spec.name, eng)] = result

        print(file=sys.stderr)

        ref_engine = REFERENCE_ENGINE
        for spec in specs:
            ref_key = (spec.name, ref_engine)
            ref = all_results.get(ref_key)
            ref_stdout = ref.stdout if ref else ""

            for eng in selected_engines:
                key = (spec.name, eng)
                if key not in all_results:
                    continue
                result = all_results[key]
                if eng == ref_engine and ref is not None:
                    if not ref.timed_out and ref.returncode == 0:
                        func_statuses[key] = FuncStatus(status="PASS")
                    else:
                        func_statuses[key] = compute_functional_status(
                            ref_stdout, result.stdout, result.returncode,
                            result.timed_out, result.stderr,
                        )
                else:
                    func_statuses[key] = compute_functional_status(
                        ref_stdout, result.stdout, result.returncode,
                        result.timed_out, result.stderr,
                    )

    programs = [s.name for s in specs]

    timings: Dict[Tuple[str, str], Tuple[Optional[float], float]] = {}
    for prog in programs:
        for eng in selected_engines:
            key = (prog, eng)
            result = all_results.get(key)
            if result is None:
                timings[key] = (None, 0.0)
                continue
            if result.timed_out or result.returncode not in (0,):
                timings[key] = (None, 0.0)
                continue
            if not result.elapsed_samples:
                timings[key] = (result.wall_seconds, 0.0)
                continue
            med, ci = median_ci(result.elapsed_samples)
            timings[key] = (med, ci)

    print_status_table(func_statuses, programs, selected_engines, timings)
    print_diff_details(func_statuses, programs, selected_engines)

    json_path = log_dir / "results.json"
    summary = {
        "revision": revision,
        "iterations": args.iterations,
        "timeout": args.timeout,
        "engines": selected_engines,
        "results": [
            {
                "program": prog,
                "engine": eng,
                "status": func_statuses.get((prog, eng), FuncStatus(status="N/A")).status,
                "diff_excerpt": func_statuses.get((prog, eng), FuncStatus(status="N/A")).diff_excerpt,
                "first_error_line": func_statuses.get((prog, eng), FuncStatus(status="N/A")).first_error_line,
                "differing_lines": func_statuses.get((prog, eng), FuncStatus(status="N/A")).differing_lines,
                "wall_median_s": timings.get((prog, eng), (None, 0))[0],
                "wall_ci_pct": timings.get((prog, eng), (None, 0))[1],
                "returncode": all_results.get((prog, eng), RunResult("", "")).returncode,
                "timed_out": all_results.get((prog, eng), RunResult("", "")).timed_out,
            }
            for prog in programs
            for eng in selected_engines
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults JSON written to {json_path}")

    if args.chart:
        chart_path = args.chart if args.chart.is_absolute() else REPO_ROOT / args.chart
        try:
            generate_appbench_chart(
                chart_path, programs, selected_engines,
                func_statuses, timings,
                caption=f"commit {revision}",
            )
            print(f"Chart written to {chart_path}")
        except RuntimeError as exc:
            print(f"Error generating chart: {exc}", file=sys.stderr)
            return 1

    return 0


# ===========================================================================
# CLI
# ===========================================================================

def _add_steady_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=STEADY_DEFAULT_ITERATIONS,
                        help="R: workload repetitions per process (default %d)"
                             % STEADY_DEFAULT_ITERATIONS)
    parser.add_argument("--timeout", type=int, default=STEADY_DEFAULT_TIMEOUT,
                        help="per-run timeout in seconds (default %d)" % STEADY_DEFAULT_TIMEOUT)
    parser.add_argument("--pin", type=int, default=None,
                        help="pin runs to this CPU core via taskset -c")
    parser.add_argument("--programs", type=str, default=None,
                        help="comma-separated subset of program names")
    parser.add_argument("--pdf", type=str,
                        default=str(REPO_ROOT / "appbench_steady_curves.pdf"),
                        help="output PDF for the warm-up curve chart")


def _add_func_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--iterations", type=int, default=FUNC_DEFAULT_ITERATIONS, metavar="N",
        help=f"Timed runs per (program, engine) pair (default: {FUNC_DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=FUNC_DEFAULT_TIMEOUT,
        help=f"Per-run timeout in seconds (default: {FUNC_DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--only", metavar="NAME", default=None,
        help="Run only the program matching this name",
    )
    parser.add_argument(
        "--chart", type=Path, default=None, metavar="PATH",
        help="Save a PDF/PNG status+timing chart to this path",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs"),
        help="Parent directory for per-run logs (default: logs/)",
    )
    parser.add_argument(
        "--engines", nargs="+", metavar="NAME",
        default=[ENGINE_GFORTH, ENGINE_GFORTH_FAST, ENGINE_RPYFORTH],
        help="Engines to benchmark (default: gforth gforth-fast rpyforth)",
    )
    parser.add_argument(
        "--gforth", metavar="PATH", default=None,
        help="Override path to gforth binary",
    )
    parser.add_argument(
        "--gforth-fast", metavar="PATH", default=None,
        help="Override path to gforth-fast binary",
    )
    parser.add_argument(
        "--rpyforth", metavar="PATH", default=None,
        help="Override path to rpyforth-c-stkfrag binary",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Appbench-1.4 harness: warm steady-state (default) or cold "
                    "functional + performance grid.",
    )
    sub = parser.add_subparsers(dest="mode")

    p_steady = sub.add_parser(
        "steady",
        help="warm steady-state + per-iteration warm-up curve (default)",
    )
    _add_steady_args(p_steady)

    p_func = sub.add_parser(
        "func",
        help="cold functional + performance grid",
    )
    _add_func_args(p_func)

    # Make `steady` the default when no subcommand is given: if the first
    # non-flag token is not a known subcommand, prepend `steady`.
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv[0] not in ("steady", "func", "-h", "--help"):
        argv = ["steady"] + argv

    args = parser.parse_args(argv)

    if args.mode == "func":
        return run_func(args)
    return run_steady(args)


if __name__ == "__main__":
    sys.exit(main())
