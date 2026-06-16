#!/usr/bin/env python3
"""
Run all RPyForth shootout benchmarks and analyze their output.

Usage:
    ./run_shootout.py [--binary ./rpyforth-c] [--output logs/]

The script discovers benchmarks in shootout/ and shootout/curve/,
runs them, saves per-benchmark log files, and prints a summary.

A/B comparison modes:
    --ab                    compare ./rpyforth-c vs ./rpyforth-c --jit off
    --ab-virtualization     compare ./rpyforth-c vs ./rpyforth-c-novirt

Jitlog capture and analysis:
    --jitlog                capture a PYPYLOG jit-summary for each run
    --report <path>         write a combined text report
    --analyze-only          re-analyze existing logs without re-running
"""

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Per-benchmark command-line arguments (filename relative to repo root).
# Most shootout files hard-code their input size; only a few are wired to
# read an argument from the command line.
DEFAULT_ARGS: Dict[str, List[str]] = {
    "shootout/ack.fs": ["10"],
}

# How long we are willing to wait for a single benchmark run.
DEFAULT_TIMEOUT = 300


@dataclass
class BenchmarkResult:
    name: str
    path: Path
    args: List[str]
    config: str = "A"
    status: str = "pending"
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    wall_seconds: float = 0.0
    result_value: Optional[str] = None
    elapsed_usec: Optional[int] = None
    curve_times: List[int] = field(default_factory=list)
    error_message: str = ""
    jitlog_path: Optional[Path] = None


@dataclass
class JitlogMetrics:
    """Selected metrics from a PYPYLOG jit-summary file."""

    tracing_count: Optional[int] = None
    tracing_time_sec: Optional[float] = None
    backend_count: Optional[int] = None
    backend_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    ops: Optional[int] = None
    recorded_ops: Optional[int] = None
    guards: Optional[int] = None
    opt_ops: Optional[int] = None
    opt_guards: Optional[int] = None
    loops: Optional[int] = None
    bridges: Optional[int] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "tracing_count": self.tracing_count,
            "tracing_time_sec": self.tracing_time_sec,
            "backend_count": self.backend_count,
            "backend_time_sec": self.backend_time_sec,
            "total_time_sec": self.total_time_sec,
            "ops": self.ops,
            "recorded_ops": self.recorded_ops,
            "guards": self.guards,
            "opt_ops": self.opt_ops,
            "opt_guards": self.opt_guards,
            "loops": self.loops,
            "bridges": self.bridges,
        }


def discover_benchmarks(root: Path) -> List[Path]:
    """Return all .fs files under shootout/, sorted."""
    shootout_dir = root / "shootout"
    if not shootout_dir.is_dir():
        raise RuntimeError(f"shootout/ directory not found at {root}")
    return sorted(p for p in shootout_dir.rglob("*.fs") if p.is_file())


def run_benchmark(
    cmd_prefix: List[str],
    benchmark: Path,
    args: List[str],
    timeout: int,
    config: str,
    env: Optional[Dict[str, str]] = None,
) -> BenchmarkResult:
    """Execute one benchmark and capture its output."""
    rel_path = benchmark.relative_to(benchmark.parents[1])
    result = BenchmarkResult(
        name=str(rel_path),
        path=benchmark,
        args=list(args),
        config=config,
    )

    cmd = list(cmd_prefix) + [str(benchmark)] + args
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        result.status = "timeout"
        result.returncode = -1
        result.stdout = exc.stdout or ""
        result.stderr = exc.stderr or ""
        result.error_message = f"timed out after {timeout}s"
        result.wall_seconds = time.perf_counter() - start
        return result
    except FileNotFoundError as exc:
        result.status = "error"
        result.returncode = -1
        result.error_message = f"binary not found: {exc.filename}"
        return result

    result.wall_seconds = time.perf_counter() - start
    result.returncode = proc.returncode
    result.stdout = proc.stdout
    result.stderr = proc.stderr

    if proc.returncode != 0:
        result.status = "error"
        result.error_message = f"exit code {proc.returncode}"
        return result

    result.status = "ok"
    return result


def parse_standard_output(result: BenchmarkResult) -> None:
    """Extract result value and elapsed time from standard shootout output."""
    text = result.stdout

    # Look for "Elapsed: <number> usec"
    elapsed_match = re.search(r"Elapsed:\s*(\d+)\s*usec", text)
    if elapsed_match:
        result.elapsed_usec = int(elapsed_match.group(1))

    # Known result labels (e.g. "Ack: 8189", "Count: 1028").
    known_label_match = re.search(
        r"^\s*(Ack|Count|Fib|Result):\s*(\S+)", text, re.MULTILINE | re.IGNORECASE
    )
    if known_label_match:
        result.result_value = f"{known_label_match.group(1)}: {known_label_match.group(2)}"
        return

    # Other "Word: value" labels, but not the elapsed/time report lines.
    for match in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*(\S+)", text, re.MULTILINE):
        label = match.group(1)
        if label.lower() in ("elapsed", "time"):
            continue
        result.result_value = f"{label}: {match.group(2)}"
        return

    # Fallback: first line that looks like one or more unsigned integers.
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("elapsed"):
            continue
        numbers = re.findall(r"\d+", line)
        if numbers:
            result.result_value = " ".join(numbers)
            return


def parse_curve_output(result: BenchmarkResult) -> None:
    """Extract per-iteration timings from curve/ CSV output."""
    times: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("iteration"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                times.append(int(parts[1]))
            except ValueError:
                continue
    result.curve_times = times
    if times:
        result.elapsed_usec = sum(times)


def parse_jit_summary(path: Path) -> Optional[JitlogMetrics]:
    """Parse a PYPYLOG jit-summary file into structured metrics."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = JitlogMetrics()

    # Lines like "Tracing:      \t6\t0.001075"
    tracing_match = re.search(r"Tracing:\s*(\d+)\s+(\d+\.\d+)", text)
    if tracing_match:
        m.tracing_count = int(tracing_match.group(1))
        m.tracing_time_sec = float(tracing_match.group(2))

    backend_match = re.search(r"Backend:\s*(\d+)\s+(\d+\.\d+)", text)
    if backend_match:
        m.backend_count = int(backend_match.group(1))
        m.backend_time_sec = float(backend_match.group(2))

    total_match = re.search(r"TOTAL:\s*(\d+\.\d+)", text)
    if total_match:
        m.total_time_sec = float(total_match.group(1))

    # Single integer metrics.
    for attr, label in [
        ("ops", "ops:"),
        ("recorded_ops", "recorded ops:"),
        ("guards", "guards:"),
        ("opt_ops", "opt ops:"),
        ("opt_guards", "opt guards:"),
        ("loops", "Total # of loops:"),
        ("bridges", "Total # of bridges:"),
    ]:
        match = re.search(re.escape(label) + r"\s*(\d+)", text)
        if match:
            setattr(m, attr, int(match.group(1)))

    return m


def analyze_result(result: BenchmarkResult) -> None:
    """Parse benchmark output according to its category."""
    if result.status != "ok":
        return

    if result.name.startswith("curve/"):
        parse_curve_output(result)
    else:
        parse_standard_output(result)

    # Treat any stderr or obvious failure markers as a warning/error.
    if result.stderr and "error" not in result.status:
        # Some benchmarks may legitimately print to stderr, so just keep it.
        pass

    failure_markers = ["UNKNOWN:", "ABORT", "THROW:", "stack underflow"]
    combined = result.stdout + result.stderr
    if any(marker in combined for marker in failure_markers):
        result.status = "warning"
        result.error_message = "detected failure marker in output"


def save_log(result: BenchmarkResult, log_dir: Path, jitlog_path: Optional[Path] = None) -> Path:
    """Write the raw log for a benchmark run to disk."""
    result.jitlog_path = jitlog_path
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.name.replace("/", "_").replace("\\", "_")
    log_path = log_dir / f"{safe_name}_{result.config}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# benchmark: {result.name}\n")
        f.write(f"# config: {result.config}\n")
        f.write(f"# command args: {' '.join(result.args)}\n")
        f.write(f"# status: {result.status}\n")
        f.write(f"# return code: {result.returncode}\n")
        f.write(f"# wall seconds: {result.wall_seconds:.6f}\n")
        if jitlog_path:
            f.write(f"# jitlog: {jitlog_path}\n")
        if result.error_message:
            f.write(f"# error: {result.error_message}\n")
        f.write("# --- stdout ---\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n# --- stderr ---\n")
            f.write(result.stderr)
    return log_path


def load_log(log_path: Path) -> Optional[BenchmarkResult]:
    """Reconstruct a BenchmarkResult from a saved log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8")

    lines = text.splitlines()
    headers: Dict[str, str] = {}
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    section = None
    for line in lines:
        if line.startswith("# --- stdout ---"):
            section = "stdout"
            continue
        if line.startswith("# --- stderr ---"):
            section = "stderr"
            continue
        if section is None and line.startswith("# "):
            key_value = line[2:].strip()
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                headers[key.strip()] = value.strip()
            continue
        if section == "stdout":
            stdout_lines.append(line)
        elif section == "stderr":
            stderr_lines.append(line)

    name = headers.get("benchmark", log_path.stem)
    # Reconstruct the relative path from the safe name if possible.
    # Log files use underscores in place of slashes.
    name = name.replace("_curve_", "/curve/").replace("_shootout_", "/")

    result = BenchmarkResult(
        name=name,
        path=Path("shootout") / name,
        args=headers.get("command args", "").split(),
        config=headers.get("config", "A"),
        status=headers.get("status", "ok"),
        returncode=int(headers.get("return code", "0")),
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
        wall_seconds=float(headers.get("wall seconds", "0")),
        error_message=headers.get("error", ""),
    )
    jitlog = headers.get("jitlog")
    if jitlog:
        result.jitlog_path = Path(jitlog)
    analyze_result(result)
    return result


def format_summary(results: List[BenchmarkResult]) -> str:
    """Build a human-readable summary table."""
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"{'Benchmark':<25} {'Config':<8} {'Status':<10} {'Result / Metric':<30} {'Wall (s)':>12}")
    lines.append("-" * 90)

    for r in results:
        if r.status != "ok":
            metric = r.error_message or "-"
        elif r.name.startswith("curve/"):
            if r.curve_times:
                mean = statistics.mean(r.curve_times)
                median = statistics.median(r.curve_times)
                metric = f"{len(r.curve_times)} iters, mean={mean:.0f}us, median={median:.0f}us"
            else:
                metric = "no timings parsed"
        else:
            elapsed = f"{r.elapsed_usec}us" if r.elapsed_usec is not None else "-"
            value = r.result_value if r.result_value is not None else "-"
            metric = f"{value} ({elapsed})"

        lines.append(
            f"{r.name:<25} {r.config:<8} {r.status:<10} {metric:<30} {r.wall_seconds:>12.3f}"
        )

    lines.append("=" * 90)

    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warning")
    err = sum(1 for r in results if r.status in ("error", "timeout"))
    lines.append(f"Summary: {ok} ok, {warn} warnings, {err} errors/timeouts out of {len(results)} runs")
    return "\n".join(lines)


def format_ab_comparison(results: List[BenchmarkResult], config_names: Dict[str, str]) -> str:
    """Build an A/B comparison table from paired results."""
    # Group by benchmark name.
    groups: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        groups.setdefault(r.name, {})[r.config] = r

    lines: List[str] = []
    lines.append("=" * 100)
    header = (
        f"{'Benchmark':<25} "
        f"{config_names['A'] + ' elapsed':>16} "
        f"{config_names['B'] + ' elapsed':>16} "
        f"{'Speedup':>10} "
        f"{'Match':>8}"
    )
    lines.append(header)
    lines.append("-" * 100)

    mismatches = 0
    errors = 0
    for name in sorted(groups):
        group = groups[name]
        a = group.get("A")
        b = group.get("B")

        if a is None or b is None:
            errors += 1
            lines.append(f"{name:<25} {'missing config pair':>58}")
            continue

        a_status_ok = a.status in ("ok", "warning")
        b_status_ok = b.status in ("ok", "warning")
        if not (a_status_ok and b_status_ok):
            errors += 1
            a_err = a.error_message or a.status
            b_err = b.error_message or b.status
            lines.append(f"{name:<25} A:{a_err:>14} B:{b_err:>14}")
            continue

        a_elapsed = a.elapsed_usec if a.elapsed_usec is not None else 0
        b_elapsed = b.elapsed_usec if b.elapsed_usec is not None else 0

        if a_elapsed > 0:
            speedup = b_elapsed / a_elapsed
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "n/a"

        match = a.result_value == b.result_value
        if not match:
            mismatches += 1
        match_str = "yes" if match else "NO"

        lines.append(
            f"{name:<25} {a_elapsed:>16,}us {b_elapsed:>16,}us {speedup_str:>10} {match_str:>8}"
        )

    lines.append("=" * 100)
    lines.append(
        f"A/B summary: {len(groups)} pairs, {mismatches} result mismatches, {errors} errors"
    )
    if mismatches:
        lines.append("Warning: result values differ between configurations!")
    return "\n".join(lines)


def format_virtualization_analysis(
    results: List[BenchmarkResult], config_names: Dict[str, str]
) -> str:
    """Build a virtualization-focused report comparing JIT metrics from jit-summaries."""
    groups: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        groups.setdefault(r.name, {})[r.config] = r

    lines: List[str] = []
    lines.append("\n" + "=" * 110)
    lines.append("Virtualization Analysis (JIT metrics from jit-summary)")
    lines.append("=" * 110)

    header = (
        f"{'Benchmark':<25} "
        f"{'Metric':<20} "
        f"{config_names['A'][:18]:>18} "
        f"{config_names['B'][:18]:>18} "
        f"{'B/A':>10} "
        f"{'Diff':>12}"
    )
    lines.append(header)
    lines.append("-" * 110)

    metrics_to_compare = [
        ("tracing_time_sec", "Tracing time (s)", lambda x: f"{x:.6f}"),
        ("backend_time_sec", "Backend time (s)", lambda x: f"{x:.6f}"),
        ("total_time_sec", "JIT total time (s)", lambda x: f"{x:.6f}"),
        ("loops", "Loops", str),
        ("bridges", "Bridges", str),
        ("ops", "Ops", str),
        ("recorded_ops", "Recorded ops", str),
        ("guards", "Guards", str),
        ("opt_ops", "Opt ops", str),
        ("opt_guards", "Opt guards", str),
    ]

    for name in sorted(groups):
        group = groups[name]
        a = group.get("A")
        b = group.get("B")
        if a is None or b is None:
            continue
        a_metrics = parse_jit_summary(a.jitlog_path) if a.jitlog_path else None
        b_metrics = parse_jit_summary(b.jitlog_path) if b.jitlog_path else None
        if a_metrics is None and b_metrics is None:
            continue

        first = True
        for attr, label, fmt in metrics_to_compare:
            a_val = getattr(a_metrics, attr) if a_metrics else None
            b_val = getattr(b_metrics, attr) if b_metrics else None
            if a_val is None and b_val is None:
                continue

            a_str = fmt(a_val) if a_val is not None else "-"
            b_str = fmt(b_val) if b_val is not None else "-"

            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)) and a_val != 0:
                ratio = b_val / a_val
                ratio_str = f"{ratio:.2f}x"
                diff = b_val - a_val
                diff_str = f"{diff:+.2f}" if isinstance(diff, float) else f"{diff:+d}"
            else:
                ratio_str = "-"
                diff_str = "-"

            bench_col = name if first else ""
            lines.append(
                f"{bench_col:<25} {label:<20} {a_str:>18} {b_str:>18} {ratio_str:>10} {diff_str:>12}"
            )
            first = False

    lines.append("=" * 110)
    lines.append(
        "Interpretation: B/A > 1 means the non-virtualized build spent more time / produced more JIT artifacts."
    )
    return "\n".join(lines)


def save_analysis_report(
    report_path: Path,
    results: List[BenchmarkResult],
    config_names: Dict[str, str],
    jitlog_mode: str,
) -> None:
    """Write a full text analysis report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    sections.append(format_summary(results))
    sections.append("")
    sections.append(format_ab_comparison(results, config_names))
    if jitlog_mode == "jit-summary":
        sections.append(format_virtualization_analysis(results, config_names))

    report_path.write_text("\n".join(sections), encoding="utf-8")


def parse_cmd(text: str) -> List[str]:
    """Split a shell-style command string into a list."""
    return shlex.split(text)


def make_env_with_jitlog(
    base_env: Dict[str, str],
    jitlog_dir: Path,
    benchmark_name: str,
    config: str,
    mode: str = "jit-summary",
) -> Tuple[Dict[str, str], Path]:
    """Return environment dict and path for a PYPYLOG capture.

    mode is the PYPYLOG category (e.g. jit-summary, jit-log-opt, jit-backend).
    """
    jitlog_dir.mkdir(parents=True, exist_ok=True)
    safe_name = benchmark_name.replace("/", "_").replace("\\", "_")
    jitlog_path = jitlog_dir / f"{safe_name}_{config}.jitlog"
    env = dict(base_env)
    env["PYPYLOG"] = f"{mode}:{jitlog_path}"
    return env, jitlog_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RPyForth shootout benchmarks and analyze their output."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("rpyforth-c"),
        help="Path to the rpyforth compiled binary (default: ./rpyforth-c)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs"),
        help="Directory to save per-benchmark log files (default: ./logs)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write a machine-readable JSON summary to this file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for each benchmark run (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--only",
        metavar="PATTERN",
        default=None,
        help="Only run benchmarks whose path contains this substring",
    )
    parser.add_argument(
        "--ab",
        action="store_true",
        help="Run an A/B comparison: config A vs config B",
    )
    parser.add_argument(
        "--ab-virtualization",
        action="store_true",
        help="Compare with vs without interpreter virtualization (A: ./rpyforth-c, B: ./rpyforth-c-novirt)",
    )
    parser.add_argument(
        "--a-cmd",
        type=str,
        default=None,
        help="Shell command for configuration A (default: ./rpyforth-c)",
    )
    parser.add_argument(
        "--b-cmd",
        type=str,
        default=None,
        help='Shell command for configuration B (default depends on --ab mode)',
    )
    parser.add_argument(
        "--jitlog",
        action="store_true",
        help="Capture a PYPYLOG jit-log-opt trace for every run",
    )
    parser.add_argument(
        "--jitlog-dir",
        type=Path,
        default=Path("logs/jitlog"),
        help="Directory for jitlog files (default: ./logs/jitlog)",
    )
    parser.add_argument(
        "--jitlog-mode",
        type=str,
        default="jit-summary",
        help="PYPYLOG category to capture (default: jit-summary)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a combined text report (summary + A/B + virtualization analysis) to this file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip running benchmarks; analyze existing logs in --output and jitlogs in --jitlog-dir",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent

    # Resolve A/B commands (used for report labels).
    ab_mode = (
        args.ab
        or args.ab_virtualization
        or args.a_cmd is not None
        or args.b_cmd is not None
    )
    if args.ab_virtualization:
        a_cmd_text = args.a_cmd or "./rpyforth-c"
        b_cmd_text = args.b_cmd or "./rpyforth-c-novirt"
    else:
        a_cmd_text = args.a_cmd or "./rpyforth-c"
        b_cmd_text = args.b_cmd or "./rpyforth-c --jit off"

    a_cmd = parse_cmd(a_cmd_text)
    b_cmd = parse_cmd(b_cmd_text)

    configs: List[Tuple[str, List[str]]] = [("A", a_cmd)]
    if ab_mode:
        configs.append(("B", b_cmd))

    output_dir = args.output if args.output.is_absolute() else repo_root / args.output
    jitlog_dir = args.jitlog_dir if args.jitlog_dir.is_absolute() else repo_root / args.jitlog_dir

    if args.analyze_only:
        # Load previously saved logs and reconstruct results.
        if ab_mode:
            log_files = sorted(output_dir.glob("*_A.log")) + sorted(output_dir.glob("*_B.log"))
        else:
            log_files = sorted(p for p in output_dir.glob("*.log") if not p.name.endswith(("_A.log", "_B.log")))
        if args.only:
            log_files = [p for p in log_files if args.only in p.name]
        if not log_files:
            print("No log files found to analyze.", file=sys.stderr)
            return 1
        results: List[BenchmarkResult] = []
        for log_path in log_files:
            result = load_log(log_path)
            if result is not None:
                results.append(result)
    else:
        # Validate binaries exist.
        for config_name, cmd in configs:
            binary_path = Path(cmd[0])
            if not binary_path.is_absolute():
                binary_path = repo_root / binary_path
            if not binary_path.exists():
                print(f"Error: binary for config {config_name} not found: {binary_path}", file=sys.stderr)
                return 1

        benchmarks = discover_benchmarks(repo_root)
        if args.only:
            benchmarks = [b for b in benchmarks if args.only in str(b.relative_to(repo_root))]

        if not benchmarks:
            print("No benchmarks found.", file=sys.stderr)
            return 1

        base_env = os.environ.copy()
        results = []
        for benchmark in benchmarks:
            rel = str(benchmark.relative_to(repo_root))
            bench_args = DEFAULT_ARGS.get(rel, [])

            for config_name, cmd_prefix in configs:
                env = base_env
                jitlog_path: Optional[Path] = None
                if args.jitlog:
                    env, jitlog_path = make_env_with_jitlog(
                        base_env, jitlog_dir, rel, config_name, args.jitlog_mode
                    )

                result = run_benchmark(cmd_prefix, benchmark, bench_args, args.timeout, config_name, env=env)
                analyze_result(result)
                save_log(result, output_dir, jitlog_path=jitlog_path)
                results.append(result)

    print(format_summary(results))

    config_names = {"A": a_cmd_text, "B": b_cmd_text}
    if ab_mode:
        print()
        print(format_ab_comparison(results, config_names))

    if args.jitlog_mode == "jit-summary" and any(r.jitlog_path for r in results):
        print(format_virtualization_analysis(results, config_names))

    if args.report:
        report_path = args.report if args.report.is_absolute() else repo_root / args.report
        save_analysis_report(report_path, results, config_names, args.jitlog_mode)
        print(f"\nAnalysis report written to {report_path}")

    if args.json:
        summary = {
            "a_cmd": a_cmd_text,
            "b_cmd": b_cmd_text if ab_mode else None,
            "ab_mode": ab_mode,
            "jitlog": args.jitlog,
            "jitlog_mode": args.jitlog_mode,
            "benchmarks": [
                {
                    "name": r.name,
                    "config": r.config,
                    "status": r.status,
                    "returncode": r.returncode,
                    "wall_seconds": r.wall_seconds,
                    "result_value": r.result_value,
                    "elapsed_usec": r.elapsed_usec,
                    "curve_times": r.curve_times,
                    "error_message": r.error_message,
                    "jitlog_path": str(r.jitlog_path) if r.jitlog_path else None,
                }
                for r in results
            ],
        }
        json_path = args.json if args.json.is_absolute() else repo_root / args.json
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON summary written to {json_path}")

    return 0 if all(r.status in ("ok", "warning") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
