#!/usr/bin/env python3
"""Resource-footprint harness: memory / GC / hardware counters per engine.

Companion to docs/PERF_ANALYSIS_PLAN.md. Reuses the steady-state program
registry and driver builder from run_appbench.py so every stage runs the
exact same workload as the timing harness.

Stages (--stages, comma-separated, default: rss,gc,counters):
  rss       max resident set size per engine x program (/usr/bin/time -v).
  gc        rpyforth-only: RPython GC log (PYPYLOG=gc:<file>); reports
            minor/major collection counts and the GC share of run ticks.
  counters  perf stat (instructions, IPC, branch/cache misses). Skipped
            gracefully with reason=BLOCKED when perf_event_paranoid > 2
            or perf(1) is absent.

Results land in logs/perf/<label>/results.json plus a report.md table.
Serialize with other benchmark runs; use --pin on a quiet core.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_appbench import (  # noqa: E402
    ENGINES,
    ENGINE_BINARY,
    ENGINE_RPYFORTH,
    PROGRAMS,
    REPO_ROOT,
    build_cmd,
    build_driver,
    parse_curve_output,
    steady_state_tail,
)

TIME_BIN = "/usr/bin/time"
PERF_EVENTS = ("task-clock,instructions,cycles,branches,branch-misses,"
               "cache-references,cache-misses")

# PYPYLOG section markers: {gc-minor ... gc-minor} etc. The number before
# the tag is a TSC-ish tick counter; deltas are comparable within one run.
PYPYLOG_OPEN = re.compile(r"^\[([0-9a-f]+)\] \{(gc-[a-z-]+)$")
PYPYLOG_CLOSE = re.compile(r"^\[([0-9a-f]+)\] (gc-[a-z-]+)\}$")


def pin_prefix(pin):
    return ["taskset", "-c", str(pin)] if pin is not None else []


def perf_available():
    """Return (ok, reason). BLOCKED reasons match ANALYSIS_GFORTH section 4."""
    try:
        subprocess.run(["perf", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False, "BLOCKED: perf(1) not installed"
    try:
        paranoid = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text())
    except (OSError, ValueError):
        paranoid = None
    if paranoid is not None and paranoid > 2 and os.geteuid() != 0:
        return False, ("BLOCKED: perf_event_paranoid=%d; unlock with "
                       "`sudo sysctl kernel.perf_event_paranoid=1`" % paranoid)
    return True, "ok"


def run_engine(engine, spec, iterations, pin, timeout, extra_env=None,
               wrapper=None):
    """Run one steady driver; return (rc, stdout_text, wrapper_file_text)."""
    driver = build_driver(spec, iterations)
    env = dict(os.environ)
    if engine == ENGINE_RPYFORTH:
        env.update(spec.rpy_env)
    if extra_env:
        env.update(extra_env)
    with tempfile.TemporaryDirectory(prefix="rpyperf-") as td:
        driver_path = Path(td) / ("%s-%s.fs" % (spec.name, engine))
        driver_path.write_text(driver)
        cmd = build_cmd(engine, driver_path, spec)
        aux_path = Path(td) / "aux.txt"
        if wrapper == "time":
            cmd = [TIME_BIN, "-v", "-o", str(aux_path)] + cmd
        elif wrapper == "perf":
            cmd = ["perf", "stat", "-x", ",", "-o", str(aux_path),
                   "-e", PERF_EVENTS] + cmd
        cmd = pin_prefix(pin) + cmd
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, env=env,
                                  cwd=str(spec.workdir))
        except subprocess.TimeoutExpired:
            return -1, "", ""
        aux = aux_path.read_text() if aux_path.exists() else ""
        return proc.returncode, proc.stdout, aux


def parse_time_v(text):
    out = {}
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if m:
        out["max_rss_kb"] = int(m.group(1))
    m = re.search(r"Elapsed \(wall clock\) time.*: ([0-9:.]+)", text)
    if m:
        out["wall"] = m.group(1)
    for key, pat in (("major_page_faults", r"Major .*page faults.*: (\d+)"),
                     ("minor_page_faults", r"Minor .*page faults.*: (\d+)"),
                     ("voluntary_ctx", r"Voluntary context switches: (\d+)"),
                     ("involuntary_ctx", r"Involuntary context switches: (\d+)")):
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def parse_perf_csv(text):
    """perf stat -x, rows: value,unit,event,... Derive IPC and miss rates."""
    vals = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0] not in ("", "<not supported>",
                                                "<not counted>"):
            try:
                vals[parts[2]] = float(parts[0])
            except ValueError:
                continue
    out = {"raw": vals}
    if vals.get("cycles"):
        out["ipc"] = vals.get("instructions", 0.0) / vals["cycles"]
    if vals.get("branches"):
        out["branch_miss_rate"] = vals.get("branch-misses", 0.0) / vals["branches"]
    if vals.get("cache-references"):
        out["cache_miss_rate"] = (vals.get("cache-misses", 0.0) /
                                  vals["cache-references"])
    return out


def parse_pypylog_gc(path):
    """Count GC sections and sum their tick durations from a PYPYLOG file."""
    counts = {}
    ticks = {}
    stack = []
    first = last = None
    try:
        fh = open(path)
    except OSError:
        return None
    with fh:
        for line in fh:
            m = PYPYLOG_OPEN.match(line.strip())
            if m:
                t = int(m.group(1), 16)
                first = t if first is None else first
                stack.append((m.group(2), t))
                continue
            m = PYPYLOG_CLOSE.match(line.strip())
            if m and stack and stack[-1][0] == m.group(2):
                tag, t0 = stack.pop()
                t = int(m.group(1), 16)
                last = t
                counts[tag] = counts.get(tag, 0) + 1
                # Sections nest (gc-minor-walkroots inside gc-minor); count
                # ticks only for top-level sections so shares do not double.
                if not stack:
                    ticks[tag] = ticks.get(tag, 0) + (t - t0)
    total_span = (last - first) if (first is not None and last is not None) else 0
    return {"counts": counts, "ticks": ticks, "log_span_ticks": total_span}


def stage_rss(specs, engines, iterations, pin, timeout, results):
    for spec in specs:
        for engine in engines:
            rc, stdout, aux = run_engine(engine, spec, iterations, pin,
                                         timeout, wrapper="time")
            times = parse_curve_output(stdout)
            entry = {"rc": rc, "iterations_seen": len(times)}
            entry.update(parse_time_v(aux))
            if times:
                entry["warm_median_usec"] = steady_state_tail(times)
            results.setdefault(spec.name, {}).setdefault(engine, {})["rss"] = entry
            print("rss  %-10s %-14s rc=%d max_rss=%s KB"
                  % (spec.name, engine, rc, entry.get("max_rss_kb", "?")))


def stage_gc(specs, engines, iterations, pin, timeout, results):
    if ENGINE_RPYFORTH not in engines:
        return
    for spec in specs:
        with tempfile.NamedTemporaryFile(prefix="pypylog-", suffix=".log",
                                         delete=False) as tf:
            log_path = tf.name
        rc, stdout, _ = run_engine(ENGINE_RPYFORTH, spec, iterations, pin,
                                   timeout,
                                   extra_env={"PYPYLOG": "gc:%s" % log_path})
        gc = parse_pypylog_gc(log_path) or {}
        gc["rc"] = rc
        os.unlink(log_path)
        results.setdefault(spec.name, {}).setdefault(
            ENGINE_RPYFORTH, {})["gc"] = gc
        share = ""
        if gc.get("log_span_ticks"):
            gc_ticks = sum(gc["ticks"].values())
            share = " gc_share_of_span=%.1f%%" % (
                100.0 * gc_ticks / gc["log_span_ticks"])
        print("gc   %-10s rpyforth       rc=%d %s%s"
              % (spec.name, rc, dict(gc.get("counts", {})), share))


def stage_counters(specs, engines, iterations, pin, timeout, results):
    ok, reason = perf_available()
    if not ok:
        print("counters: %s" % reason)
        results["_counters_status"] = reason
        return
    for spec in specs:
        for engine in engines:
            rc, stdout, aux = run_engine(engine, spec, iterations, pin,
                                         timeout, wrapper="perf")
            entry = parse_perf_csv(aux)
            entry["rc"] = rc
            results.setdefault(spec.name, {}).setdefault(
                engine, {})["counters"] = entry
            print("perf %-10s %-14s rc=%d ipc=%.2f branch_miss=%.2f%%"
                  % (spec.name, engine, rc, entry.get("ipc", 0.0),
                     100.0 * entry.get("branch_miss_rate", 0.0)))


def write_report(out_dir, results, args):
    lines = ["# Resource footprint (%s)" % args.label, "",
             "iterations=%d pin=%s stages=%s" % (args.iterations, args.pin,
                                                 args.stages), "",
             "| program | engine | max RSS (MB) | GC minor/major | GC share | IPC | branch miss |",
             "|---|---|---|---|---|---|---|"]
    for prog in sorted(k for k in results if not k.startswith("_")):
        for engine in results[prog]:
            r = results[prog][engine]
            rss = r.get("rss", {}).get("max_rss_kb")
            rss = "%.1f" % (rss / 1024.0) if rss else "-"
            gc = r.get("gc", {})
            gcc = gc.get("counts", {})
            gcs = "%d/%d" % (gcc.get("gc-minor", 0),
                             gcc.get("gc-collect-done", 0)) if gcc else "-"
            share = "-"
            if gc.get("log_span_ticks"):
                share = "%.1f%%" % (100.0 * sum(gc["ticks"].values()) /
                                    gc["log_span_ticks"])
            cnt = r.get("counters", {})
            ipc = "%.2f" % cnt["ipc"] if "ipc" in cnt else "-"
            bm = ("%.2f%%" % (100.0 * cnt["branch_miss_rate"])
                  if "branch_miss_rate" in cnt else "-")
            lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                         % (prog, engine, rss, gcs, share, ipc, bm))
    if "_counters_status" in results:
        lines += ["", "counters stage: %s" % results["_counters_status"]]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--programs", help="comma-separated subset of %s"
                    % ",".join(s.name for s in PROGRAMS))
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--pin", type=int, default=None, help="CPU core to pin")
    ap.add_argument("--stages", default="rss,gc,counters")
    ap.add_argument("--label", default=None, help="output dir name under logs/perf")
    args = ap.parse_args()

    specs = list(PROGRAMS)
    if args.programs:
        wanted = set(args.programs.split(","))
        unknown = wanted - set(s.name for s in specs)
        if unknown:
            ap.error("unknown programs: %s" % ",".join(sorted(unknown)))
        specs = [s for s in specs if s.name in wanted]
    engines = args.engines.split(",")
    for e in engines:
        if e not in ENGINE_BINARY:
            ap.error("unknown engine %s" % e)
    if not Path(TIME_BIN).exists() and "rss" in args.stages:
        ap.error("%s not found (apt install time)" % TIME_BIN)

    args.label = args.label or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "logs" / "perf" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    stages = args.stages.split(",")
    if "rss" in stages:
        stage_rss(specs, engines, args.iterations, args.pin, args.timeout,
                  results)
    if "gc" in stages:
        stage_gc(specs, engines, args.iterations, args.pin, args.timeout,
                 results)
    if "counters" in stages:
        stage_counters(specs, engines, args.iterations, args.pin,
                       args.timeout, results)

    (out_dir / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=1, sort_keys=True))
    write_report(out_dir, results, args)
    print("\nwrote %s" % (out_dir / "report.md"))


if __name__ == "__main__":
    main()
