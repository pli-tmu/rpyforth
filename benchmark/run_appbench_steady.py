#!/usr/bin/env python3
"""Warm steady-state appbench harness for rpyforth vs gforth / gforth-fast.

Single-shot appbench wall-clock is unfair to a meta-tracing JIT: a ~1-2s run
spends a large fraction on one-time JIT warm-up (tracing/compiling bridges).
gforth-fast is precompiled native code with zero warm-up, so a cold comparison
measures rpyforth's *compiler*, not its generated code quality.

This harness measures WARM STEADY-STATE performance instead. For each program
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

Per-program core-word / count choices (see ProgramSpec below):
  - cd16sim : unit = `150000 clear 150000 steps`  (clear resets state -> repeatable)
  - brainless: unit = benchmark3 x8               (self-contained movegen loops)
  - fcp     : unit = benchThink                    (position set up once in prelude)
  - lexex   : unit = lexcore + dict rewind         (re-do decorate + table build)

Safety: never modifies appbench/appbench-1.4/. Drivers are written to a tmpdir
and run with cwd set to the program dir (like run_appbench.py). Subprocesses use
a hard timeout so a stuck engine is killed.
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
APPBENCH_DIR = REPO_ROOT / "appbench" / "appbench-1.4"
GFORTH_DIR = REPO_ROOT / "gforth-0.7.9"
GFORTH_SETUP = APPBENCH_DIR / "setup" / "gforth.fs"

ENGINE_RPYFORTH = "rpyforth"
ENGINE_GFORTH_FAST = "gforth-fast"
ENGINE_GFORTH = "gforth"

# Order matters for the table / legend.
ENGINES = [ENGINE_RPYFORTH, ENGINE_GFORTH_FAST, ENGINE_GFORTH]
REFERENCE_ENGINE = ENGINE_GFORTH_FAST

ENGINE_BINARY = {
    ENGINE_RPYFORTH: REPO_ROOT / "rpyforth-c-stkfrag",
    ENGINE_GFORTH_FAST: GFORTH_DIR / "gforth-fast",
    ENGINE_GFORTH: GFORTH_DIR / "gforth",
}

DEFAULT_ITERATIONS = 50
DEFAULT_TIMEOUT = 600


class ProgramSpec:
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
    ProgramSpec(
        name="cd16sim",
        workdir=APPBENCH_DIR / "cd16sim",
        pre_include=": 3drop 2drop drop ;",
        include_file="bench.f",
        setup="",
        # clear resets the machine state so each 150000-step run is identical.
        unit="150000 clear 150000 steps",
    ),
    ProgramSpec(
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
    ProgramSpec(
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
    ProgramSpec(
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
    ProgramSpec(
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="R: workload repetitions per process (default %d)"
                             % DEFAULT_ITERATIONS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-run timeout in seconds (default %d)" % DEFAULT_TIMEOUT)
    parser.add_argument("--pin", type=int, default=None,
                        help="pin runs to this CPU core via taskset -c")
    parser.add_argument("--programs", type=str, default=None,
                        help="comma-separated subset of program names")
    parser.add_argument("--pdf", type=str,
                        default=str(REPO_ROOT / "appbench_steady_curves.pdf"),
                        help="output PDF for the warm-up curve chart")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
