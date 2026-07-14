#!/usr/bin/env python3
"""NTOP sweep: measure shootout + appbench absolute times across the
NTOP in {0,2,4,8,16} scalar-top ablation binaries and render a PDF.

NTOP=0  rpyforth-c-stkfrag-frameonly   (frame-only ablation)
NTOP=2  rpyforth-c-stkfrag             (flagship)
NTOP=4/8/16  rpyforth-c-stkfrag-ntopN  (parametric sweep builds)

Per benchmark the binaries run interleaved (bench outer, binary inner) to
control thermal/frequency drift. Warm time = median of the last 50% of
in-process iterations. Results: logs/ntop-sweep/<git-rev>/ntop_sweep.json
+ ntop_sweep.pdf (absolute times, log scale).
"""
from __future__ import annotations

import argparse
import json
import math
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

NTOP_BINARIES = {
    0: REPO_ROOT / "rpyforth-c-stkfrag-frameonly",
    2: REPO_ROOT / "rpyforth-c-stkfrag",
    4: REPO_ROOT / "rpyforth-c-stkfrag-ntop4",
    8: REPO_ROOT / "rpyforth-c-stkfrag-ntop8",
    16: REPO_ROOT / "rpyforth-c-stkfrag-ntop16",
}

SHOOTOUT_KERNELS = [
    "ack", "ary", "callheavy", "composite", "except", "fibo", "heap",
    "matrix", "methcall", "nestedloop", "random", "recurse", "sieve",
]

ITER_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$", re.M)


def warm_median(times):
    return statistics.median(times[len(times) // 2:])


def run_driver(binary, driver_path, cwd, pin, timeout, env=None):
    cmd = ["taskset", "-c", str(pin), str(binary), str(driver_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(cwd), env=env)
    times = [int(m.group(2)) for m in ITER_RE.finditer(p.stdout)]
    return p.returncode, times


def measure_shootout(iterations, pin, timeout, tmpdir):
    rows = {}
    for kernel in SHOOTOUT_KERNELS:
        bench = REPO_ROOT / "shootout" / ("%s.fs" % kernel)
        driver = build_shootout_driver(bench, iterations)
        dp = Path(tmpdir) / ("%s_sweep.fs" % kernel)
        dp.write_text(driver, encoding="utf-8")
        row = {}
        for ntop, binary in sorted(NTOP_BINARIES.items()):
            rc, times = run_driver(binary, dp, REPO_ROOT / "shootout",
                                   pin, timeout)
            if rc != 0 or not times:
                print("  shootout %-11s NTOP=%-2d FAILED rc=%d" %
                      (kernel, ntop, rc), flush=True)
                row[str(ntop)] = None
                continue
            row[str(ntop)] = {"cold_usec": times[0],
                              "warm_median_usec": warm_median(times),
                              "n": len(times)}
            print("  shootout %-11s NTOP=%-2d warm=%.0fus" %
                  (kernel, ntop, warm_median(times)), flush=True)
        rows[kernel] = row
    return rows


def measure_appbench(iterations, lexex_iterations, pin, timeout, tmpdir):
    import os
    rows = {}
    for spec in PROGRAMS:
        iters = lexex_iterations if spec.name == "lexex" else iterations
        patched = prepare_engine_workdir(ENGINE_RPYFORTH, spec, tmpdir)
        run_spec = (spec if patched == Path(spec.workdir)
                    else run_rq.with_workdir(spec, patched))
        driver = build_driver(run_spec, iters, ENGINE_RPYFORTH)
        dp = Path(tmpdir) / ("%s_sweep_drv.fs" % spec.name)
        dp.write_text(driver, encoding="utf-8")
        base_cmd = build_cmd(ENGINE_RPYFORTH, dp, run_spec)
        env = os.environ.copy()
        if run_spec.rpy_env:
            env.update(run_spec.rpy_env)
        row = {}
        for ntop, binary in sorted(NTOP_BINARIES.items()):
            cmd = (["taskset", "-c", str(pin), str(binary)] + base_cmd[1:])
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=run_spec.workdir, env=env)
            times = [int(m.group(2)) for m in ITER_RE.finditer(p.stdout)]
            if p.returncode != 0 or not times:
                print("  appbench %-11s NTOP=%-2d FAILED rc=%d" %
                      (spec.name, ntop, p.returncode), flush=True)
                row[str(ntop)] = None
                continue
            row[str(ntop)] = {"cold_usec": times[0],
                              "warm_median_usec": warm_median(times),
                              "n": len(times)}
            print("  appbench %-11s NTOP=%-2d warm=%.0fus" %
                  (spec.name, ntop, warm_median(times)), flush=True)
        rows[spec.name] = row
    return rows


def plot(results, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ntops = [0, 2, 4, 8, 16]
    colors = plt.get_cmap("viridis")([0.05, 0.28, 0.5, 0.72, 0.92])
    with PdfPages(pdf_path) as pdf:
        for suite, unit_div, unit in (("shootout", 1.0, "µs"),
                                      ("appbench", 1000.0, "ms")):
            data = results[suite]
            names = [n for n in data if any(data[n].values())]
            fig, ax = plt.subplots(figsize=(max(8, 0.9 * (len(names) + 1) * 1.6), 5),
                                   constrained_layout=True)
            width = 0.16
            xs = range(len(names) + 1)
            for j, ntop in enumerate(ntops):
                vals = []
                for name in names:
                    cell = data[name].get(str(ntop))
                    vals.append(cell["warm_median_usec"] / unit_div
                                if cell else float("nan"))
                ok = [v for v in vals if v == v]
                gm = (math.exp(sum(math.log(v) for v in ok) / len(ok))
                      if ok else float("nan"))
                ax.bar([x + (j - 2) * width for x in xs], vals + [gm], width,
                       label="NTOP=%d" % ntop, color=colors[j])
            ax.axvline(len(names) - 0.5, color="gray", linewidth=0.8,
                       linestyle=":")
            ax.set_yscale("log")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(names + ["geomean"], rotation=30, ha="right")
            ax.set_ylabel("warm median per iteration [%s, log]" % unit)
            ax.set_title("%s: absolute steady-state time by NTOP "
                         "(lower is better)" % suite)
            ax.grid(axis="y", alpha=0.3, which="both")
            ax.legend(ncol=5, fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    print("wrote %s" % pdf_path)


def geomean_vs2(rows):
    out = {}
    for ntop in (0, 2, 4, 8, 16):
        ratios = []
        for row in rows.values():
            a, b = row.get("2"), row.get(str(ntop))
            if a and b:
                ratios.append(b["warm_median_usec"] / a["warm_median_usec"])
        out[ntop] = (math.exp(sum(math.log(r) for r in ratios) / len(ratios))
                     if ratios else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=3)
    ap.add_argument("--shootout-iterations", type=int, default=30)
    ap.add_argument("--appbench-iterations", type=int, default=50)
    ap.add_argument("--lexex-iterations", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--suites", default="shootout,appbench")
    args = ap.parse_args()

    for ntop, b in NTOP_BINARIES.items():
        if not b.exists():
            print("missing binary for NTOP=%d: %s" % (ntop, b),
                  file=sys.stderr)
            return 1

    rev = run_rq.git_revision(REPO_ROOT)
    out_dir = REPO_ROOT / "logs" / "ntop-sweep" / rev
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "tmp")) as td:
        if "shootout" in args.suites:
            print("shootout sweep (R=%d) ..." % args.shootout_iterations)
            results["shootout"] = measure_shootout(
                args.shootout_iterations, args.pin, args.timeout, td)
        if "appbench" in args.suites:
            print("appbench sweep (R=%d, lexex R=%d) ..." %
                  (args.appbench_iterations, args.lexex_iterations))
            results["appbench"] = measure_appbench(
                args.appbench_iterations, args.lexex_iterations,
                args.pin, args.timeout, td)

    json_path = out_dir / "ntop_sweep.json"
    json_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote %s" % json_path)

    for suite in results:
        print("%s geomean vs NTOP=2:" % suite)
        for ntop, g in sorted(geomean_vs2(results[suite]).items()):
            print("  NTOP=%-2d %s" % (ntop, "%.3f" % g if g else "n/a"))

    plot(results, out_dir / "ntop_sweep.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
