#!/usr/bin/env python3
"""Runtime JIT-parameter grid sweep over shootout + appbench.

Runs every --jit config in a grid against the canonical shootout kernels and
the appbench programs (no rebuilds: the binary's built-in defaults are
overridden per run), reports per-suite geomeans normalized to the baseline
config, and writes JSON next to the other sweep logs.

Usage:
    .venv/bin/python benchmark/run_jit_sweep.py \
        --configs "retrace_limit=10;retrace_limit=20;retrace_limit=20,trace_eagerness=1000"

The FIRST config in the list is the baseline every other config is normalized
against. Per benchmark the configs run interleaved to control thermal drift.
Warm time = median of the last 50% of in-process iterations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_appbench import (  # noqa: E402
    PROGRAMS, build_driver, build_cmd, prepare_engine_workdir,
    ENGINE_RPYFORTH,
)
from run_ablation import build_shootout_driver  # noqa: E402
import run_rq  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SHOOTOUT_KERNELS = [
    "ack", "ary", "callheavy", "composite", "except", "fibo", "heap",
    "matrix", "methcall", "nestedloop", "random", "recurse", "sieve",
]

ITER_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$", re.M)


def warm_median(times):
    return statistics.median(times[len(times) // 2:])


def geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_cmd(cmd, cwd, env, pin, timeout):
    full = ["taskset", "-c", str(pin)] + cmd
    p = subprocess.run(full, capture_output=True, text=True, timeout=timeout,
                       cwd=str(cwd), env=env)
    times = [int(m.group(2)) for m in ITER_RE.finditer(p.stdout)]
    if p.returncode != 0 or not times:
        return None
    return warm_median(times)


def sweep_shootout(binary, configs, iterations, pin, timeout, tmpdir):
    rows = {}
    for kernel in SHOOTOUT_KERNELS:
        bench = REPO_ROOT / "shootout" / ("%s.fs" % kernel)
        driver = build_shootout_driver(bench, iterations)
        dp = Path(tmpdir) / ("%s_jsweep.fs" % kernel)
        dp.write_text(driver, encoding="utf-8")
        row = {}
        for cfg in configs:
            w = run_cmd([str(binary), "--jit", cfg, str(dp)],
                        REPO_ROOT / "shootout", os.environ.copy(),
                        pin, timeout)
            row[cfg] = w
            print("  shootout %-11s %-40s warm=%s" %
                  (kernel, cfg, ("%.0fus" % w) if w else "FAIL"), flush=True)
        rows[kernel] = row
    return rows


def sweep_appbench(binary, configs, iterations, lexex_iterations, pin,
                   timeout, tmpdir):
    rows = {}
    for spec in PROGRAMS:
        iters = lexex_iterations if spec.name == "lexex" else iterations
        patched = prepare_engine_workdir(ENGINE_RPYFORTH, spec, tmpdir)
        run_spec = (spec if patched == Path(spec.workdir)
                    else run_rq.with_workdir(spec, patched))
        driver = build_driver(run_spec, iters, ENGINE_RPYFORTH)
        dp = Path(tmpdir) / ("%s_jsweep.fs" % spec.name)
        dp.write_text(driver, encoding="utf-8")
        base_cmd = build_cmd(ENGINE_RPYFORTH, dp, run_spec)
        env = os.environ.copy()
        if run_spec.rpy_env:
            env.update(run_spec.rpy_env)
        row = {}
        for cfg in configs:
            cmd = [str(binary), "--jit", cfg] + base_cmd[1:]
            w = run_cmd(cmd, run_spec.workdir, env, pin, timeout)
            row[cfg] = w
            print("  appbench %-11s %-40s warm=%s" %
                  (spec.name, cfg, ("%.0fus" % w) if w else "FAIL"),
                  flush=True)
        rows[spec.name] = row
    return rows


def report(results, configs):
    base = configs[0]
    print()
    print("%-42s %12s %12s" % ("config (vs %s)" % base, "shootout", "appbench"))
    for cfg in configs:
        cells = []
        for suite in ("shootout", "appbench"):
            rows = results.get(suite, {})
            ratios = []
            for row in rows.values():
                a, b = row.get(base), row.get(cfg)
                if a and b:
                    ratios.append(b / a)
            g = geomean(ratios)
            cells.append(("%.3f" % g) if g else "n/a")
        print("%-42s %12s %12s" % (cfg, cells[0], cells[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path,
                    default=REPO_ROOT / "rpyforth-c-stkfrag")
    ap.add_argument("--configs", required=True,
                    help="semicolon-separated --jit configs; first = baseline")
    ap.add_argument("--pin", type=int, default=3)
    ap.add_argument("--shootout-iterations", type=int, default=20)
    ap.add_argument("--appbench-iterations", type=int, default=20)
    ap.add_argument("--lexex-iterations", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--suites", default="shootout,appbench")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(";") if c.strip()]
    rev = run_rq.git_revision(REPO_ROOT)
    out_dir = REPO_ROOT / "logs" / "jit-sweep" / rev
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "tmp")) as td:
        if "shootout" in args.suites:
            print("shootout sweep (R=%d) ..." % args.shootout_iterations)
            results["shootout"] = sweep_shootout(
                args.binary, configs, args.shootout_iterations,
                args.pin, args.timeout, td)
        if "appbench" in args.suites:
            print("appbench sweep (R=%d, lexex R=%d) ..." %
                  (args.appbench_iterations, args.lexex_iterations))
            results["appbench"] = sweep_appbench(
                args.binary, configs, args.appbench_iterations,
                args.lexex_iterations, args.pin, args.timeout, td)

    json_path = out_dir / "jit_sweep.json"
    json_path.write_text(json.dumps(
        {"configs": configs, "results": results}, indent=1),
        encoding="utf-8")
    print("wrote %s" % json_path)
    report(results, configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
