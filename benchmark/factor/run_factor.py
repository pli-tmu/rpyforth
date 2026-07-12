#!/usr/bin/env python3
"""Run rpyfactor paper / shootout benchmarks.

Usage:
    ./benchmark/factor/run_factor.py
    ./benchmark/factor/run_factor.py --suite shootout --iterations 5 --pin 3
    ./benchmark/factor/run_factor.py --engines ./rpyfactor-c,./rpyfactor-c-stkfrag
    ./benchmark/factor/run_factor.py --benchmarks sieve.factor,nrev.factor

Engines default to rpyfactor-c / rpyfactor-c-stkfrag / rpyfactor-c-stkfrag-vable /
./factor/factor (make setup-factor). Missing binaries are skipped with a warning.

Timing: rpyfactor engines that support `clock` use an in-process driver.
The real Factor binary is wrapped with USING/IN: and timed externally
(wall clock per iteration) because it has no `clock` word.

Results are persisted as JSON plus a text summary under
logs/factorbench/<git-rev>/.

Layout:
    benchmark/factor/shootout/         Forth shootout homologs
    benchmark/factor/axis-b/           Factor combinator / list / recursion suite
    benchmark/factor/phase-b/          mutable-array / larger Factor kernels
    shootout/*.fs                   Forth shootout originals
"""

from __future__ import annotations

import argparse
import datetime
import json
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


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTOR_DIR = REPO_ROOT / "benchmark" / "factor" / "shootout"
FACTOR_EXTRA = REPO_ROOT / "benchmark" / "factor" / "axis-b"
FACTOR_PHASEB = REPO_ROOT / "benchmark" / "factor" / "phase-b"
LOG_ROOT = REPO_ROOT / "logs" / "factorbench"

# Local Factor binary from `make setup-factor` (optimizing AOT compiler).
FACTOR_BIN = REPO_ROOT / "factor" / "factor"

DEFAULT_ENGINES = [
    "rpyfactor-c",
    "rpyfactor-c-stkfrag",
    "rpyfactor-c-stkfrag-vable",
    "factor/factor",
]
# Expected integer results, verified against the Forth originals
# (shootout/*.fs on ./rpyforth-c / gforth) or closed form.
EXPECTED: Dict[str, int] = {
    "ack.factor": 2045,
    "fibo.factor": 165580141,
    "sieve.factor": 1028,
    "nestedloop.factor": 729000000,
    "ary.factor": 1000,
    "recurse.factor": 18,
    "callheavy.factor": 16000000,
    "composite.factor": 895583921,
    "matrix.factor": 15205770,
    "methcall.factor": 1000001,
    "except.factor": 500000500000,
    "random.factor": 36792695,
    "heap.factor": 999707,
    # Axis B
    "nrev.factor": 1250025000,
    "sum.factor": 1250025000,
    "fact.factor": 479001600,
    "sumtree.factor": 860,
    "tak.factor": 3,
    "mapfold.factor": 1250075000,
    "step.factor": 1250025000,
    "filter.factor": 25000,
    "quotheavy.factor": 3200000,
    "fibrec.factor": 317811,
    "fib.factor": 102334155,
    # Phase B
    "nsieve.factor": 1028,
    "recursive.factor": 214,
}

# Per-iteration timeout (seconds); scaled by iteration count in clock mode.
TIMEOUT: Dict[str, int] = {
    "ack.factor": 60,
    "fibo.factor": 300,
    "sieve.factor": 120,
    "nestedloop.factor": 180,
    "ary.factor": 60,
    "recurse.factor": 120,
    "callheavy.factor": 120,
    "composite.factor": 600,
    "matrix.factor": 120,
    "methcall.factor": 240,
    "except.factor": 60,
    "random.factor": 180,
    "heap.factor": 60,
    "nrev.factor": 120,
    "sum.factor": 120,
    "mapfold.factor": 120,
    "step.factor": 120,
    "filter.factor": 120,
    "quotheavy.factor": 120,
    "tak.factor": 180,
    "fibrec.factor": 60,
    "fact.factor": 60,
    "sumtree.factor": 60,
    "fib.factor": 60,
    "nsieve.factor": 120,
    "recursive.factor": 120,
}

DEFAULT_TIMEOUT = 120

# Same problem, reduced size (immutable-list port is asymptotically more
# expensive than the Forth array code); result value still matches Forth.
REDUCED_BENCH = frozenset({
    "sieve.factor",     # filter-sieve; 100 reps instead of 10000
    "matrix.factor",    # list matmul; 300 reps instead of 3000
    "ary.factor",       # zip-add lists; length 1000 instead of 30096
    "composite.factor", # inherits the sieve/ary/heap reductions
})

# Same result value, but different machinery measured (see file comments).
ALTERED_BENCH = frozenset({
    "except.factor",    # no catch/throw in Factor: plain parity-count loop
    "heap.factor",      # no floats/arrays: LCG max instead of heapsort
})

# Axis A: Forth-homologous micros (paper claim 1).
AXIS_A = [
    "nestedloop.factor",
    "fibo.factor",
    "callheavy.factor",
    "random.factor",
]

# Axis B: Factor combinator / list / recursion pressure (paper claim 2).
AXIS_B = [
    "nrev.factor",
    "sum.factor",
    "mapfold.factor",
    "step.factor",
    "filter.factor",
    "quotheavy.factor",
    "fact.factor",
    "fibrec.factor",
    "tak.factor",
    "sumtree.factor",
]

PAPER_ORDER = AXIS_A + AXIS_B

SHOOTOUT_ORDER = [
    "ack.factor",
    "fibo.factor",
    "sieve.factor",
    "nestedloop.factor",
    "ary.factor",
    "recurse.factor",
    "callheavy.factor",
    "composite.factor",
    "matrix.factor",
    "methcall.factor",
    "except.factor",
    "random.factor",
    "heap.factor",
]

PHASE_B = [
    "nsieve.factor",
    "recursive.factor",
]

ALL_ORDER = SHOOTOUT_ORDER + AXIS_B + PHASE_B


@dataclass
class EngineSpec:
    name: str
    path: Path
    has_clock: bool = False


@dataclass
class RunOutcome:
    status: str = "ok"          # ok | error | timeout | overflow
    result: Optional[int] = None
    times: List[int] = field(default_factory=list)
    mode: str = "clock"          # clock | elapsed
    error: str = ""


def median_ci(samples: List[int], confidence: float = 0.90,
              resamples: int = 2000) -> Tuple[Optional[float], float]:
    if not samples:
        return (None, 0.0)
    med = statistics.median(samples)
    if len(samples) == 1 or med == 0:
        return (med, 0.0)
    rng = random.Random(20240709)
    n = len(samples)
    boot = sorted(
        statistics.median(samples[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    lo = boot[int((1.0 - confidence) / 2 * resamples)]
    hi = boot[min(resamples - 1, int((1.0 + confidence) / 2 * resamples))]
    return (med, 100.0 * (hi - lo) / 2.0 / med)


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


def resolve_bench(name: str) -> Path:
    if (FACTOR_DIR / name).is_file():
        return FACTOR_DIR / name
    if (FACTOR_EXTRA / name).is_file():
        return FACTOR_EXTRA / name
    if (FACTOR_PHASEB / name).is_file():
        return FACTOR_PHASEB / name
    raise RuntimeError("missing benchmark: %s" % name)


def discover_benchmarks(suite: str) -> List[Path]:
    order = {
        "shootout": SHOOTOUT_ORDER,
        "axis-a": AXIS_A,
        "axis-b": AXIS_B,
        "phase-b": PHASE_B,
        "paper": PAPER_ORDER,
        "all": ALL_ORDER,
    }.get(suite)
    if order is None:
        raise RuntimeError("unknown suite: %s" % suite)
    return [resolve_bench(n) for n in order]


def is_native_factor(engine: EngineSpec) -> bool:
    return engine.name == "factor" or engine.path.name == "factor"


def resolve_engines(spec: str) -> List[EngineSpec]:
    engines = []
    names = spec.split(",") if spec else DEFAULT_ENGINES
    for name in names:
        name = name.strip()
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = REPO_ROOT / name
        if not path.is_file():
            print("WARNING: engine %s not found at %s -- skipping"
                  % (name, path), file=sys.stderr)
            continue
        engines.append(EngineSpec(name=Path(name).name, path=path))
    return engines


def probe_clock(engine: EngineSpec, tmpdir: str) -> bool:
    """True if the engine supports the `clock` word ( -- usec )."""
    if is_native_factor(engine):
        return False
    probe = Path(tmpdir) / "clock_probe.factor"
    probe.write_text("clock 0 >=\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            [str(engine.path), str(probe)],
            capture_output=True, text=True, timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "Result: 1" in proc.stdout


_COMMENT_RE = re.compile(r"(?m)^[ \t]*!.*$")
_COLON_RE = re.compile(r":\s+\S+\b.*?;", re.DOTALL)

# Factor subset words our benches use that need real Factor vocab imports.
_FACTOR_PRELUDE = """\
USING: kernel math math.integers math.functions combinators
       sequences lists prettyprint io ;
IN: rpyfactor.bench
"""

# Benches that use rpyfactor-only list/array mixing; real Factor is skipped.
NATIVE_FACTOR_SKIP = frozenset({
    "nsieve.factor",   # pack3 via cons + mutable <array>
    "sieve.factor",     # immutable-list filter sieve
    "ary.factor",
    "matrix.factor",
    "composite.factor",
    "heap.factor",
    "except.factor",
    "nrev.factor",
    "sum.factor",
    "mapfold.factor",
    "step.factor",
    "filter.factor",
    "sumtree.factor",
    "recurse.factor",
    "tak.factor",
})


def wrap_for_native_factor(source: str) -> str:
    """Wrap a subset .factor so the real Factor binary can parse it."""
    text = _COMMENT_RE.sub("", source)
    text = re.sub(r"(\S+)\s+<array>", r"\1 0 <array>", text)
    text = text.replace("1+", "1 +")
    text = text.replace("1-", "1 -")
    # rpyfactor `/` is truncating integer division; Factor `/` yields ratios.
    text = re.sub(r"(?<![/=<>!])/(?![/=])", "/i", text)
    text = re.sub(r"(?<![A-Za-z0-9-])step(?![A-Za-z0-9-])", "each", text)
    text = re.sub(r"(?<![A-Za-z0-9-])empty\?(?![A-Za-z0-9-])", "empty?", text)

    def colon_fix(m):
        whole = m.group(0)
        m2 = re.match(r"(:\s+\S+)(\s*)(\([^)]*\))?(\s*)(.*)(;)\s*$",
                      whole, re.S)
        if not m2:
            return whole
        head, sp1, effect, sp2, body, term = m2.groups()
        name = head.split()[1]
        is_rec = re.search(
            r"(?<![A-Za-z0-9-])%s(?![A-Za-z0-9-])" % re.escape(name), body
        ) is not None
        if effect is None:
            effect = "( ..a -- ..b )"
            sp1 = " "
            sp2 = " "
        if is_rec:
            return "%s%s%s%s%s ; recursive" % (head, sp1, effect, sp2, body)
        return "%s%s%s%s%s%s" % (head, sp1, effect, sp2, body, term)

    text = re.sub(r":\s+\S+.*?;", colon_fix, text, flags=re.S)
    return _FACTOR_PRELUDE + "\n" + text + "\n.\n"


def build_driver(source: str, iterations: int) -> str:
    """Wrap a benchmark in a per-iteration timing loop using `clock`."""
    text = _COMMENT_RE.sub(" ", source)
    defines = _COLON_RE.findall(text)
    main = _COLON_RE.sub(" ", text).strip()
    if not main:
        raise RuntimeError("benchmark has no top-level code")
    parts = []
    parts.extend(defines)
    parts.append("")
    parts.append("0 0")
    parts.append("%d [" % iterations)
    parts.append("  drop")
    parts.append("  clock")
    parts.append(main)
    parts.append("  swap clock swap -")
    parts.append("  [ [ dup put ] dip ] dip")
    parts.append('  " ," putchars')
    parts.append("  dup put")
    parts.append('  "')
    parts.append('" putchars')
    parts.append("  drop")
    parts.append("  [ 1+ ] dip")
    parts.append("] times")
    parts.append("nip")
    return "\n".join(parts) + "\n"


def parse_curve_output(stdout: str) -> List[int]:
    """Per-iteration timings: lines "<i> ,<usec>" (run_ablation convention)."""
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


def _classify_failure(proc_stdout: str, proc_stderr: str, rc: int) -> Tuple[str, str]:
    err = (proc_stderr or proc_stdout or "exit %d" % rc).strip()
    if "StackOverflow" in err or "StackOverflow" in proc_stdout:
        return ("overflow", "stack overflow")
    if "call stack overflow" in err or "call stack overflow" in proc_stdout:
        return ("overflow", "call stack overflow")
    return ("error", err.splitlines()[-1][:160] if err else "exit %d" % rc)


def _run_proc(cmd: List[str], timeout: int):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
    )


def run_bench(engine: EngineSpec, bench: Path, iterations: int,
              timeout: int, pin: Optional[int], tmpdir: str) -> RunOutcome:
    wrapper = ["taskset", "-c", str(pin)] if pin is not None else []
    out = RunOutcome()

    if is_native_factor(engine):
        out.mode = "elapsed"
        if bench.name in NATIVE_FACTOR_SKIP:
            out.status = "skip"
            out.error = "native Factor skip (subset/host mismatch)"
            return out
        src = wrap_for_native_factor(bench.read_text(encoding="utf-8"))
        script = Path(tmpdir) / ("%s_native.factor" % bench.stem)
        script.write_text(src, encoding="utf-8")
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                proc = _run_proc(
                    wrapper + [str(engine.path), "-q", str(script)],
                    timeout,
                )
            except subprocess.TimeoutExpired:
                out.status = "timeout"
                out.error = "timeout (%ds)" % timeout
                return out
            wall_usec = int((time.perf_counter() - t0) * 1_000_000)
            if proc.returncode != 0:
                out.status, out.error = _classify_failure(
                    proc.stdout, proc.stderr, proc.returncode)
                return out
            # Last integer on a line is the printed TOS.
            nums = re.findall(r"(?m)^(-?\d+)\s*$", proc.stdout)
            if not nums:
                out.status = "error"
                out.error = "missing integer result from Factor"
                return out
            out.result = int(nums[-1])
            out.times.append(wall_usec)
        return out

    if engine.has_clock:
        out.mode = "clock"
        driver = build_driver(bench.read_text(encoding="utf-8"), iterations)
        driver_path = Path(tmpdir) / ("%s_%s_driver.factor"
                                      % (bench.stem, engine.name))
        driver_path.write_text(driver, encoding="utf-8")
        total_timeout = timeout * iterations + 60
        try:
            proc = _run_proc(wrapper + [str(engine.path), str(driver_path)],
                             total_timeout)
        except subprocess.TimeoutExpired:
            out.status = "timeout"
            out.error = "timeout (%ds)" % total_timeout
            return out
        if proc.returncode != 0:
            out.status, out.error = _classify_failure(
                proc.stdout, proc.stderr, proc.returncode)
            return out
        m = re.search(r"Result:\s*(-?\d+)", proc.stdout)
        if not m:
            out.status = "error"
            out.error = "missing Result in output"
            return out
        out.result = int(m.group(1))
        out.times = parse_curve_output(proc.stdout)
        if len(out.times) != iterations:
            out.status = "error"
            out.error = ("expected %d iteration timings, got %d"
                         % (iterations, len(out.times)))
        return out

    # Fallback: no `clock` word -- one process per iteration, whole-run
    # "Elapsed: <n> usec" (includes startup and JIT warmup every time).
    out.mode = "elapsed"
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            proc = _run_proc(wrapper + [str(engine.path), str(bench)], timeout)
        except subprocess.TimeoutExpired:
            out.status = "timeout"
            out.error = "timeout (%ds)" % timeout
            return out
        wall_usec = int((time.perf_counter() - t0) * 1_000_000)
        if proc.returncode != 0:
            out.status, out.error = _classify_failure(
                proc.stdout, proc.stderr, proc.returncode)
            return out
        m = re.search(r"Result:\s*(-?\d+)", proc.stdout)
        if not m:
            out.status = "error"
            out.error = "missing Result in output"
            return out
        out.result = int(m.group(1))
        e = re.search(r"Elapsed:\s*(\d+)\s*usec", proc.stdout)
        out.times.append(int(e.group(1)) if e else wall_usec)
    return out


def axis_of(name: str) -> str:
    if name in AXIS_A:
        return "A"
    if name in AXIS_B:
        return "B"
    return "-"


def tag_of(name: str) -> str:
    if name in REDUCED_BENCH:
        return "reduced"
    if name in ALTERED_BENCH:
        return "altered"
    return "full"


def fmt_usec(v) -> str:
    if v is None:
        return "-"
    return "%d" % int(v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rpyfactor benchmarks")
    parser.add_argument("--suite", default="paper",
                        choices=["paper", "axis-a", "axis-b", "phase-b",
                                 "shootout", "all"],
                        help="paper=Axis A+B (default); phase-b=mutable-array "
                             "kernels; shootout=13 homologs; all=shootout+"
                             "axis-b+phase-b")
    parser.add_argument("--benchmarks", default="",
                        help="comma-separated subset filter (e.g. sieve.factor,nrev.factor)")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-iteration timeout override (seconds)")
    parser.add_argument("--pin", type=int, default=None,
                        help="pin benchmark processes to this CPU core (taskset -c)")
    parser.add_argument("--engines", default="",
                        help="comma-separated engine binaries "
                             "(default: %s)" % ",".join(DEFAULT_ENGINES))
    parser.add_argument("--no-save", action="store_true",
                        help="do not write logs/factorbench/<rev>/")
    parser.add_argument("--list-stubs", action="store_true",
                        help="print benchmarks that are reduced or altered "
                             "relative to the Forth original")
    args = parser.parse_args()

    if args.list_stubs:
        for name in sorted(REDUCED_BENCH):
            print("%s (reduced)" % name)
        for name in sorted(ALTERED_BENCH):
            print("%s (altered)" % name)
        return 0

    benches = discover_benchmarks(args.suite)
    if args.benchmarks:
        keep = {n.strip() for n in args.benchmarks.split(",") if n.strip()}
        benches = [b for b in benches if b.name in keep]
        missing = keep - {b.name for b in benches}
        if missing:
            print("WARNING: not in suite '%s': %s"
                  % (args.suite, ", ".join(sorted(missing))), file=sys.stderr)

    engines = resolve_engines(args.engines)
    if not engines:
        print("ERROR: no engines available", file=sys.stderr)
        return 2

    rev = git_revision(REPO_ROOT)
    records = []
    all_ok = True

    with tempfile.TemporaryDirectory(prefix="factorbench_") as tmpdir:
        for eng in engines:
            eng.has_clock = probe_clock(eng, tmpdir)
            if not eng.has_clock:
                print("note: %s has no `clock` word; falling back to "
                      "whole-run Elapsed timing" % eng.name)

        for bench in benches:
            name = bench.name
            exp = EXPECTED.get(name)
            timeout = args.timeout or TIMEOUT.get(name, DEFAULT_TIMEOUT)
            for eng in engines:
                outcome = run_bench(eng, bench, args.iterations, timeout,
                                    args.pin, tmpdir)
                med, ci = median_ci(outcome.times)
                if outcome.status == "ok" and exp is not None \
                        and outcome.result != exp:
                    check = "FAIL got %s" % outcome.result
                    all_ok = False
                elif outcome.status == "ok":
                    check = "PASS"
                elif outcome.status == "skip":
                    check = "SKIP"
                else:
                    check = outcome.status.upper()
                    all_ok = False
                records.append({
                    "benchmark": name,
                    "engine": eng.name,
                    "axis": axis_of(name),
                    "tag": tag_of(name),
                    "status": outcome.status,
                    "check": check,
                    "result": outcome.result,
                    "expected": exp,
                    "mode": outcome.mode,
                    "times_usec": outcome.times,
                    "median_usec": int(med) if med is not None else None,
                    "ci_pct": round(ci, 2),
                    "error": outcome.error,
                })
                status_str = check if outcome.status == "ok" \
                    else "%s (%s)" % (check, outcome.error)
                print("%-16s %-24s %-10s median=%s usec  %s"
                      % (name, eng.name, outcome.mode,
                         fmt_usec(med), status_str))

    # Summary table: one row per benchmark, one median column per engine.
    eng_names = [e.name for e in engines]
    lines = []
    cpu = platform.processor() or platform.machine()
    lines.append("rpyfactor %s  rev=%s  iterations=%d  pin=%s  (%s)"
                 % (args.suite, rev, args.iterations, args.pin, cpu))
    lines.append("tags: reduced = smaller N than Forth original; "
                 "altered = same result, different machinery (see source)")
    hdr = ["benchmark", "axis", "tag", "check"] + \
          ["%s usec" % n for n in eng_names]
    lines.append(" | ".join(hdr))
    lines.append("-" * (40 + 18 * len(eng_names)))
    by_bench: Dict[str, Dict[str, dict]] = {}
    for r in records:
        by_bench.setdefault(r["benchmark"], {})[r["engine"]] = r
    for bench in benches:
        name = bench.name
        row = by_bench.get(name, {})
        checks = {r["check"] for r in row.values()}
        check = "PASS" if checks == {"PASS"} else \
            ";".join(sorted(c for c in checks if c != "PASS")) or "?"
        cells = [name, axis_of(name), tag_of(name), check]
        for en in eng_names:
            r = row.get(en)
            cells.append(fmt_usec(r["median_usec"]) if r else "-")
        lines.append(" | ".join(cells))
    summary = "\n".join(lines)
    print()
    print(summary)

    if not args.no_save:
        outdir = LOG_ROOT / rev
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        payload = {
            "revision": rev,
            "timestamp": stamp,
            "suite": args.suite,
            "iterations": args.iterations,
            "pin": args.pin,
            "cpu": cpu,
            "engines": [
                {"name": e.name, "path": str(e.path), "clock": e.has_clock}
                for e in engines
            ],
            "results": records,
        }
        json_path = outdir / ("results-%s-%s.json" % (args.suite, stamp))
        json_path.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
        txt_path = outdir / ("summary-%s-%s.txt" % (args.suite, stamp))
        txt_path.write_text(summary + "\n", encoding="utf-8")
        print()
        print("saved: %s" % json_path)
        print("saved: %s" % txt_path)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
