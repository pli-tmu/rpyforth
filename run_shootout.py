#!/usr/bin/env python3
"""
Run all RPyForth shootout benchmarks and analyze their output.

Usage:
    ./run_shootout.py [--output logs/]

Comparison presets (--compare):
    jit      ./rpyforth-c vs ./rpyforth-c --jit off
    virt     ./rpyforth-c vs ./rpyforth-c-novirt
    gforth   gforth vs ./rpyforth-c

Override either side with --a-cmd / --b-cmd. Omit --compare for a single run.
Use --iterations N to repeat each run and report median elapsed times.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jitlog_analysis import (
    aggregate_benchmark_jitlogs,
    append_jitlog_visualization_pages,
    format_jitlog_comparison,
    is_curve_benchmark,
    records_from_benchmark_results,
)


# Per-benchmark command-line arguments (filename relative to repo root).
# Most shootout files hard-code their input size; only a few are wired to
# read an argument from the command line.
DEFAULT_ARGS: Dict[str, List[str]] = {}

# How long we are willing to wait for a single benchmark run.
DEFAULT_TIMEOUT = 300

# Named A/B comparison presets: (a_cmd, b_cmd, speedup = a/b when True else b/a).
COMPARE_PRESETS: Dict[str, Tuple[str, str, bool]] = {
    "jit": ("./rpyforth-c", "./rpyforth-c --jit off", False),
    "virt": ("./rpyforth-c", "./rpyforth-c-novirt", False),
    "gforth": ("gforth", "./rpyforth-c", True),
}

COMPARE_TITLES: Dict[str, str] = {
    "jit": "JIT Comparison",
    "virt": "Virtualization Comparison",
    "gforth": "Gforth Baseline Comparison",
}


@dataclass(frozen=True)
class RunPlan:
    """Resolved commands and reporting options for one invocation."""

    preset: Optional[str]
    configs: Tuple[Tuple[str, List[str], str], ...]  # (id, cmd, label)
    speedup_a_over_b: bool
    skip_jit_analysis: bool
    skip_jitlog_for: frozenset[str]

    @property
    def compare(self) -> bool:
        return len(self.configs) > 1

    def label_for(self, config_id: str) -> str:
        for cfg_id, _, label in self.configs:
            if cfg_id == config_id:
                return label
        return config_id


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
    iteration: int = 1
    run_count: int = 1
    elapsed_samples: List[int] = field(default_factory=list)


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


def log_path_for(
    result: BenchmarkResult, log_dir: Path, iteration: int, total_iterations: int
) -> Path:
    """Return the log file path for a benchmark run."""
    safe_name = result.name.replace("/", "_").replace("\\", "_")
    if total_iterations > 1:
        return log_dir / f"{safe_name}_{result.config}_i{iteration:03d}.log"
    return log_dir / f"{safe_name}_{result.config}.log"


def discover_log_files(
    output_dir: Path, compare: bool, only: Optional[str], iterations: int
) -> List[Path]:
    """Find saved log files for analyze-only mode."""
    if compare:
        numbered = sorted(output_dir.glob("*_A_i*.log")) + sorted(output_dir.glob("*_B_i*.log"))
        unnumbered = sorted(output_dir.glob("*_A.log")) + sorted(output_dir.glob("*_B.log"))
        if iterations > 1 or (numbered and not unnumbered):
            log_files = numbered
        else:
            log_files = unnumbered or numbered
    elif iterations > 1:
        log_files = sorted(output_dir.glob("*_A_i*.log"))
    else:
        numbered = sorted(output_dir.glob("*_A_i*.log"))
        unnumbered = sorted(
            p
            for p in output_dir.glob("*.log")
            if not p.name.endswith(("_A.log", "_B.log"))
            and "_i" not in p.stem
        )
        log_files = numbered if numbered and not unnumbered else unnumbered or numbered

    if only:
        log_files = [p for p in log_files if only in p.name]
    return sorted(set(log_files))


def aggregate_iterations(results: List[BenchmarkResult]) -> List[BenchmarkResult]:
    """Collapse repeated runs into one result per benchmark/config (median elapsed)."""
    groups: Dict[Tuple[str, str], List[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.name, result.config), []).append(result)

    aggregated: List[BenchmarkResult] = []
    for (name, config) in sorted(groups):
        runs = sorted(groups[(name, config)], key=lambda r: r.iteration)
        if len(runs) == 1:
            run = runs[0]
            if run.elapsed_usec is not None:
                run.elapsed_samples = [run.elapsed_usec]
            aggregated.append(run)
            continue

        ok_runs = [r for r in runs if r.status in ("ok", "warning")]
        failed_count = len(runs) - len(ok_runs)
        elapsed_samples = [r.elapsed_usec for r in ok_runs if r.elapsed_usec is not None]
        last_ok = ok_runs[-1] if ok_runs else runs[-1]

        merged = BenchmarkResult(
            name=name,
            path=runs[0].path,
            args=runs[0].args,
            config=config,
            status=last_ok.status,
            returncode=last_ok.returncode,
            stdout=last_ok.stdout,
            stderr=last_ok.stderr,
            wall_seconds=statistics.mean(r.wall_seconds for r in runs),
            result_value=last_ok.result_value,
            elapsed_usec=int(statistics.median(elapsed_samples)) if elapsed_samples else None,
            curve_times=last_ok.curve_times,
            error_message=last_ok.error_message,
            jitlog_path=last_ok.jitlog_path,
            iteration=0,
            run_count=len(runs),
            elapsed_samples=elapsed_samples,
        )
        if failed_count:
            merged.status = "warning" if ok_runs else "error"
            detail = f"{failed_count}/{len(runs)} runs failed"
            merged.error_message = (
                f"{merged.error_message}; {detail}" if merged.error_message else detail
            )
        aggregated.append(merged)
    return aggregated


def save_log(
    result: BenchmarkResult,
    log_dir: Path,
    jitlog_path: Optional[Path] = None,
    *,
    iteration: int = 1,
    total_iterations: int = 1,
) -> Path:
    """Write the raw log for a benchmark run to disk."""
    result.jitlog_path = jitlog_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(result, log_dir, iteration, total_iterations)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# benchmark: {result.name}\n")
        f.write(f"# config: {result.config}\n")
        f.write(f"# iteration: {iteration}\n")
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
        iteration=int(headers.get("iteration", "1")),
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
            if r.run_count > 1:
                metric = f"{metric}, median of {r.run_count}"

        lines.append(
            f"{r.name:<25} {r.config:<8} {r.status:<10} {metric:<30} {r.wall_seconds:>12.3f}"
        )

    lines.append("=" * 90)

    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warning")
    err = sum(1 for r in results if r.status in ("error", "timeout"))
    lines.append(f"Summary: {ok} ok, {warn} warnings, {err} errors/timeouts out of {len(results)} runs")
    return "\n".join(lines)


def parse_cmd(text: str) -> List[str]:
    """Split a shell-style command string into a list."""
    return shlex.split(text)


def short_label(cmd: List[str]) -> str:
    """Return a compact display label for a command prefix."""
    name = Path(cmd[0]).name
    if len(cmd) == 1:
        return name
    return f"{name} {' '.join(cmd[1:])}"


def resolve_command(cmd: List[str], repo_root: Path) -> Optional[Path]:
    """Return the executable path if a command prefix is runnable."""
    if not cmd:
        return None
    exe = Path(cmd[0])
    if exe.is_absolute() and exe.exists():
        return exe
    local = repo_root / exe
    if local.exists():
        return local
    found = shutil.which(cmd[0])
    if found:
        return Path(found)
    return None


def build_run_plan(args: argparse.Namespace) -> RunPlan:
    """Resolve preset, commands, and reporting behavior from CLI args."""
    preset = args.compare
    if preset and preset not in COMPARE_PRESETS:
        raise SystemExit(f"Unknown compare preset: {preset}")

    if preset:
        default_a, default_b, speedup_a_over_b = COMPARE_PRESETS[preset]
        a_text = args.a_cmd or default_a
        b_text = args.b_cmd or default_b
        skip_jit_analysis = preset == "gforth"
        skip_jitlog_for = frozenset({"A"}) if preset == "gforth" else frozenset()
    elif args.b_cmd:
        a_text = args.a_cmd or "./rpyforth-c"
        b_text = args.b_cmd
        speedup_a_over_b = False
        skip_jit_analysis = parse_cmd(a_text)[0] == "gforth"
        skip_jitlog_for = frozenset({"A"}) if skip_jit_analysis else frozenset()
        preset = None
    else:
        a_cmd = parse_cmd(args.a_cmd or "./rpyforth-c")
        return RunPlan(
            preset=None,
            configs=(("A", a_cmd, short_label(a_cmd)),),
            speedup_a_over_b=False,
            skip_jit_analysis=True,
            skip_jitlog_for=frozenset(),
        )

    a_cmd = parse_cmd(a_text)
    b_cmd = parse_cmd(b_text)
    return RunPlan(
        preset=preset,
        configs=(
            ("A", a_cmd, short_label(a_cmd)),
            ("B", b_cmd, short_label(b_cmd)),
        ),
        speedup_a_over_b=speedup_a_over_b,
        skip_jit_analysis=skip_jit_analysis,
        skip_jitlog_for=skip_jitlog_for,
    )


def format_comparison(results: List[BenchmarkResult], plan: RunPlan) -> str:
    """Build a comparison table from paired A/B results."""
    groups: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        groups.setdefault(r.name, {})[r.config] = r

    a_label = plan.label_for("A")
    b_label = plan.label_for("B")

    lines: List[str] = []
    lines.append("=" * 100)
    if plan.preset:
        lines.append(COMPARE_TITLES[plan.preset])
        lines.append("=" * 100)
    header = (
        f"{'Benchmark':<25} "
        f"{a_label + ' elapsed':>16} "
        f"{b_label + ' elapsed':>16} "
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

        if plan.speedup_a_over_b:
            speedup_str = f"{a_elapsed / b_elapsed:.2f}x" if b_elapsed > 0 else "n/a"
        else:
            speedup_str = f"{b_elapsed / a_elapsed:.2f}x" if a_elapsed > 0 else "n/a"

        match = a.result_value == b.result_value
        if not match:
            mismatches += 1
        match_str = "yes" if match else "NO"

        lines.append(
            f"{name:<25} {a_elapsed:>16,}us {b_elapsed:>16,}us {speedup_str:>10} {match_str:>8}"
        )

    lines.append("=" * 100)
    lines.append(
        f"Comparison summary: {len(groups)} pairs, {mismatches} result mismatches, {errors} errors"
    )
    if plan.speedup_a_over_b:
        lines.append(
            f"Interpretation: speedup = {a_label} / {b_label}; >1 means {b_label} is faster."
        )
    elif mismatches:
        lines.append("Warning: result values differ between configurations!")
    return "\n".join(lines)


def format_virtualization_analysis(results: List[BenchmarkResult], plan: RunPlan) -> str:
    """Build a virtualization-focused report comparing JIT metrics from jit-summaries."""
    grouped = aggregate_benchmark_jitlogs(results)
    if not grouped:
        return ""
    return "\n" + format_jitlog_comparison(
        grouped,
        plan.label_for("A"),
        plan.label_for("B"),
        title="Virtualization Analysis (JIT metrics from jit-summary)",
    )


def save_analysis_report(
    report_path: Path,
    results: List[BenchmarkResult],
    plan: RunPlan,
    jitlog_mode: str,
) -> None:
    """Write a full text analysis report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = [format_summary(results), ""]
    if plan.compare:
        sections.append(format_comparison(results, plan))
    if jitlog_mode == "jit-summary" and not plan.skip_jit_analysis:
        sections.append(format_virtualization_analysis(results, plan))

    report_path.write_text("\n".join(sections), encoding="utf-8")


def is_stable_benchmark(name: str) -> bool:
    """Return True for single-run shootout benchmarks (not curve/ warmup runs)."""
    return not is_curve_benchmark(name)


def build_stable_elapsed_samples(
    results: List[BenchmarkResult],
) -> Dict[str, Dict[str, List[int]]]:
    """Return stable benchmark elapsed times grouped by name and config."""
    samples: Dict[str, Dict[str, List[int]]] = {}
    for result in results:
        if not is_stable_benchmark(result.name):
            continue
        if result.elapsed_samples:
            elapsed = result.elapsed_samples
        elif result.elapsed_usec is not None:
            elapsed = [result.elapsed_usec]
        else:
            continue
        samples.setdefault(result.name, {})[result.config] = elapsed
    return samples


def has_iteration_variance(results: List[BenchmarkResult]) -> bool:
    """Return True when at least one stable benchmark has multiple timed runs."""
    for by_config in build_stable_elapsed_samples(results).values():
        for elapsed in by_config.values():
            if len(elapsed) > 1:
                return True
    return False


def build_paired_stable_data(
    results: List[BenchmarkResult],
    plan: RunPlan,
) -> Tuple[List[str], List[int], List[int], List[float], List[float], List[float]]:
    """Return paired elapsed times, speedups, and per-config stddevs."""
    samples_by_name = build_stable_elapsed_samples(results)

    names: List[str] = []
    a_values: List[int] = []
    b_values: List[int] = []
    speedups: List[float] = []
    a_stdevs: List[float] = []
    b_stdevs: List[float] = []
    for name in sorted(samples_by_name):
        by_config = samples_by_name[name]
        a_samples = by_config.get("A")
        b_samples = by_config.get("B") if plan.compare else None
        if a_samples is None:
            continue
        if plan.compare and b_samples is None:
            continue

        a_elapsed = int(statistics.median(a_samples))
        if plan.compare:
            assert b_samples is not None
            b_elapsed = int(statistics.median(b_samples))
            if plan.speedup_a_over_b:
                if b_elapsed == 0:
                    continue
                speedup = a_elapsed / b_elapsed
            else:
                if a_elapsed == 0:
                    continue
                speedup = b_elapsed / a_elapsed
        else:
            b_elapsed = 0
            speedup = 0.0

        names.append(name)
        a_values.append(a_elapsed)
        b_values.append(b_elapsed)
        speedups.append(speedup)
        a_stdevs.append(statistics.stdev(a_samples) if len(a_samples) > 1 else 0.0)
        b_stdevs.append(
            statistics.stdev(b_samples) if b_samples and len(b_samples) > 1 else 0.0
        )
    return names, a_values, b_values, speedups, a_stdevs, b_stdevs


def generate_pdf_report(
    pdf_path: Path,
    results: List[BenchmarkResult],
    plan: RunPlan,
    jitlog_mode: str,
) -> None:
    """Generate a multi-page PDF report with benchmark graphs.

    Requires matplotlib to be installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError(
            "PDF report generation requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    label_a = plan.label_for("A")
    label_b = plan.label_for("B")
    if plan.preset == "gforth":
        elapsed_title = f"Stable Performance: {label_b} vs {label_a} (baseline)"
        speedup_title = f"Stable Performance: Speedup vs {label_a} Baseline"
        speedup_xlabel = f"Speedup ({label_a} / {label_b}); >1 means {label_b} is faster"
    elif plan.compare:
        elapsed_title = "Stable Performance: Elapsed Time Comparison"
        speedup_title = "Stable Performance: Speedup Ratio"
        speedup_xlabel = f"Speedup ({label_b} / {label_a}); >1 means {label_a} is faster"
    else:
        elapsed_title = speedup_title = speedup_xlabel = ""

    curve_colors = {"A": "#3498db", "B": "#e74c3c"} if plan.preset == "gforth" else {
        "A": "#1f77b4",
        "B": "#ff7f0e",
    }

    with PdfPages(str(pdf_path)) as pdf:
        show_variance = has_iteration_variance(results)

        if plan.compare:
            names, a_values, b_values, speedups, a_stdevs, b_stdevs = build_paired_stable_data(
                results, plan
            )
            if names:
                x = range(len(names))
                width = 0.35
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
                ax.barh(
                    [i - width / 2 for i in x],
                    a_values,
                    width,
                    label=label_a,
                    xerr=a_stdevs if show_variance else None,
                    capsize=3 if show_variance else 0,
                )
                ax.barh(
                    [i + width / 2 for i in x],
                    b_values,
                    width,
                    label=label_b,
                    xerr=b_stdevs if show_variance else None,
                    capsize=3 if show_variance else 0,
                )
                ax.set_yticks(list(x))
                ax.set_yticklabels(names)
                ax.set_xlabel("Elapsed time (microseconds)")
                title = elapsed_title
                if show_variance:
                    title += " (median +/- stdev)"
                ax.set_title(title)
                ax.legend()
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            if names:
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
                colors = ["green" if s > 1 else "red" for s in speedups]
                ax.barh(names, speedups, color=colors, alpha=0.7)
                ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel(speedup_xlabel)
                ax.set_title(speedup_title)
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        if show_variance:
            from matplotlib.patches import Patch

            samples_by_name = build_stable_elapsed_samples(results)
            benchmarks = [
                name
                for name in sorted(samples_by_name)
                if any(len(samples) > 1 for samples in samples_by_name[name].values())
            ]
            config_ids = [config_id for config_id, _, _ in plan.configs]
            bar_colors = curve_colors

            fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.8), 6))
            positions: List[float] = []
            boxplot_data: List[List[int]] = []
            colors: List[str] = []
            x_ticks: List[float] = []
            x_labels: List[str] = []
            x_pos = 0.0

            for benchmark in benchmarks:
                group_start = x_pos
                for config_id in config_ids:
                    elapsed = samples_by_name.get(benchmark, {}).get(config_id, [])
                    if len(elapsed) < 2:
                        continue
                    boxplot_data.append(elapsed)
                    positions.append(x_pos)
                    colors.append(bar_colors.get(config_id, "#95a5a6"))
                    x_pos += 1.0
                if x_pos > group_start:
                    x_ticks.append((group_start + x_pos - 1.0) / 2.0)
                    x_labels.append(benchmark.replace("shootout/", ""))
                    x_pos += 0.5

            if boxplot_data:
                bp = ax.boxplot(
                    boxplot_data,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showmeans=True,
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                ax.set_ylabel("Elapsed time (microseconds)")
                ax.set_title("Run-to-run Variance Across Iterations")
                ax.grid(axis="y", linestyle="--", alpha=0.5)

                legend_elements = [
                    Patch(
                        facecolor=bar_colors.get(config_id, "#95a5a6"),
                        alpha=0.7,
                        label=plan.label_for(config_id),
                    )
                    for config_id in config_ids
                ]
                ax.legend(handles=legend_elements)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # --- Page: Combined curve iteration time plots (all programs) ---
        curve_data: Dict[str, Dict[str, List[int]]] = {}
        for r in results:
            if not r.name.startswith("curve/") or not r.curve_times:
                continue
            curve_data.setdefault(r.name, {})[r.config] = r.curve_times

        if curve_data:
            programs = sorted(curve_data.keys())
            n_programs = len(programs)
            cols = min(3, n_programs)
            rows = (n_programs + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
            )
            color_map = curve_colors

            for idx, program in enumerate(programs):
                ax = axes.flatten()[idx]
                for config in sorted(curve_data[program]):
                    times = curve_data[program][config]
                    iters = list(range(len(times)))
                    legend_label = plan.label_for(config)
                    ax.plot(
                        iters,
                        times,
                        marker="o",
                        linestyle="-",
                        linewidth=1,
                        markersize=3,
                        label=legend_label,
                        color=color_map.get(config),
                    )
                short_name = program.replace("curve/", "")
                ax.set_title(short_name)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Time (microseconds)")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.5)

            for idx in range(n_programs, rows * cols):
                axes.flatten()[idx].set_visible(False)

            fig.suptitle("Per-iteration Timings (all programs)", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Page: JIT metrics comparison ---
        if jitlog_mode == "jit-summary" and plan.compare and not plan.skip_jit_analysis:
            grouped = aggregate_benchmark_jitlogs(results)
            if grouped:
                append_jitlog_visualization_pages(
                    pdf,
                    grouped,
                    records_from_benchmark_results(results),
                    label_a,
                    label_b,
                    chart_title_prefix="Virtualization Analysis",
                )


def make_env_with_jitlog(
    base_env: Dict[str, str],
    jitlog_dir: Path,
    benchmark_name: str,
    config: str,
    mode: str = "jit-summary",
    iteration: int = 1,
    total_iterations: int = 1,
) -> Tuple[Dict[str, str], Path]:
    """Return environment dict and path for a PYPYLOG capture.

    mode is the PYPYLOG category (e.g. jit-summary, jit-log-opt, jit-backend).
    """
    jitlog_dir.mkdir(parents=True, exist_ok=True)
    safe_name = benchmark_name.replace("/", "_").replace("\\", "_")
    if total_iterations > 1:
        jitlog_path = jitlog_dir / f"{safe_name}_{config}_i{iteration:03d}.jitlog"
    else:
        jitlog_path = jitlog_dir / f"{safe_name}_{config}.jitlog"
    env = dict(base_env)
    env["PYPYLOG"] = f"{mode}:{jitlog_path}"
    return env, jitlog_path


class RunProgress:
    """Simple stderr progress reporter for benchmark runs."""

    def __init__(self, total: int, *, enabled: bool = True) -> None:
        self.total = total
        self.enabled = enabled and total > 0 and sys.stderr.isatty()
        self.current = 0
        self._width = 36

    def update(self, label: str) -> None:
        self.current += 1
        if not self.enabled:
            print(f"[{self.current}/{self.total}] {label}", file=sys.stderr)
            return
        ratio = self.current / self.total
        filled = int(self._width * ratio)
        bar = "#" * filled + "-" * (self._width - filled)
        sys.stderr.write(
            f"\r[{bar}] {self.current}/{self.total} {label[:50]:<50}"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled and self.total > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RPyForth shootout benchmarks and analyze their output."
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
        "--iterations",
        type=int,
        default=1,
        metavar="N",
        help="Run each benchmark/configuration N times; reports use median elapsed time (default: 1)",
    )
    parser.add_argument(
        "--only",
        metavar="PATTERN",
        default=None,
        help="Only run benchmarks whose path contains this substring",
    )
    parser.add_argument(
        "--compare",
        choices=sorted(COMPARE_PRESETS),
        default=None,
        help="Compare two configurations using a preset (jit, virt, gforth)",
    )
    parser.add_argument(
        "--a-cmd",
        type=str,
        default=None,
        help="Shell command for configuration A (default: ./rpyforth-c, or preset default)",
    )
    parser.add_argument(
        "--b-cmd",
        type=str,
        default=None,
        help="Shell command for configuration B (enables comparison when set without --compare)",
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
        "--pdf",
        type=Path,
        default=None,
        help="Generate a multi-page PDF graph report and save it to this file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip running benchmarks; analyze existing logs in --output and jitlogs in --jitlog-dir",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        print("Error: --iterations must be at least 1", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent
    plan = build_run_plan(args)

    output_dir = args.output if args.output.is_absolute() else repo_root / args.output
    jitlog_dir = args.jitlog_dir if args.jitlog_dir.is_absolute() else repo_root / args.jitlog_dir

    if args.analyze_only:
        log_files = discover_log_files(output_dir, plan.compare, args.only, args.iterations)
        if not log_files:
            print("No log files found to analyze.", file=sys.stderr)
            return 1
        results: List[BenchmarkResult] = []
        for log_path in log_files:
            result = load_log(log_path)
            if result is not None:
                results.append(result)
    else:
        for config_id, cmd, _label in plan.configs:
            if resolve_command(cmd, repo_root) is None:
                print(
                    f"Error: command for config {config_id} not found: {' '.join(cmd)}",
                    file=sys.stderr,
                )
                return 1

        benchmarks = discover_benchmarks(repo_root)
        if args.only:
            benchmarks = [b for b in benchmarks if args.only in str(b.relative_to(repo_root))]

        if not benchmarks:
            print("No benchmarks found.", file=sys.stderr)
            return 1

        base_env = os.environ.copy()
        results = []
        total_runs = len(benchmarks) * len(plan.configs) * args.iterations
        progress = RunProgress(total_runs)
        try:
            for benchmark in benchmarks:
                rel = str(benchmark.relative_to(repo_root))
                bench_args = DEFAULT_ARGS.get(rel, [])

                for config_id, cmd_prefix, _label in plan.configs:
                    for iteration in range(1, args.iterations + 1):
                        env = base_env
                        jitlog_path: Optional[Path] = None
                        if args.jitlog and config_id not in plan.skip_jitlog_for:
                            env, jitlog_path = make_env_with_jitlog(
                                base_env,
                                jitlog_dir,
                                rel,
                                config_id,
                                args.jitlog_mode,
                                iteration=iteration,
                                total_iterations=args.iterations,
                            )

                        result = run_benchmark(
                            cmd_prefix, benchmark, bench_args, args.timeout, config_id, env=env
                        )
                        result.iteration = iteration
                        analyze_result(result)
                        save_log(
                            result,
                            output_dir,
                            jitlog_path=jitlog_path,
                            iteration=iteration,
                            total_iterations=args.iterations,
                        )
                        results.append(result)

                        label = f"{rel} ({config_id}"
                        if args.iterations > 1:
                            label += f", iter {iteration}"
                        label += ")"
                        progress.update(label)
        finally:
            progress.finish()

    results = aggregate_iterations(results)

    print(format_summary(results))

    if plan.compare:
        print()
        print(format_comparison(results, plan))

    if (
        args.jitlog_mode == "jit-summary"
        and not plan.skip_jit_analysis
        and any(r.jitlog_path for r in results)
    ):
        print(format_virtualization_analysis(results, plan))

    if args.report:
        report_path = args.report if args.report.is_absolute() else repo_root / args.report
        save_analysis_report(report_path, results, plan, args.jitlog_mode)
        print(f"\nAnalysis report written to {report_path}")

    if args.pdf:
        pdf_path = args.pdf if args.pdf.is_absolute() else repo_root / args.pdf
        try:
            generate_pdf_report(pdf_path, results, plan, args.jitlog_mode)
            print(f"PDF graph report written to {pdf_path}")
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.json:
        summary = {
            "compare": plan.preset,
            "iterations": args.iterations,
            "configs": [
                {"id": config_id, "cmd": cmd, "label": label}
                for config_id, cmd, label in plan.configs
            ],
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
                    "run_count": r.run_count,
                    "elapsed_samples": r.elapsed_samples,
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
