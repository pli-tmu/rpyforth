#!/usr/bin/env python3
"""
Functional + performance benchmark harness for the appbench-1.4 suite.

Programs covered: cd16sim, brainless, fcp, lexex.

Usage:
    .venv/bin/python benchmark/run_appbench.py [--chart appbench.pdf] [--iterations N]

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

DEFAULT_TIMEOUT = 300
DEFAULT_ITERATIONS = 3

ENGINE_GFORTH = "gforth"
ENGINE_GFORTH_FAST = "gforth-fast"
ENGINE_RPYFORTH = "rpyforth"

REFERENCE_ENGINE = ENGINE_GFORTH_FAST


@dataclass
class ProgramSpec:
    name: str
    workdir: Path
    prelude: str
    body: str
    supported_engines: List[str] = field(default_factory=list)


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

    return [cd16sim, brainless, fcp, lexex]


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

    # Large applications trace-thrash with the default bridge eagerness (the
    # gforth side gets its own tuning via -m 16M); this is a runtime knob,
    # not a code change.
    cmd = [str(binary), "--jit", "trace_eagerness=1000", str(wrapper_path)]
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

    if engine_name == ENGINE_RPYFORTH:
        cmd = build_rpyforth_cmd(engine_path, spec, tmpdir)
    else:
        cmd = build_gforth_cmd(engine_path, spec, tmpdir)

    for i in range(1, iterations + 1):
        rc, stdout, stderr, wall, timed_out = run_once(
            cmd, spec.workdir, timeout
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Functional + performance harness for appbench-1.4"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS, metavar="N",
        help=f"Timed runs per (program, engine) pair (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-run timeout in seconds (default: {DEFAULT_TIMEOUT})",
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
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
