#!/usr/bin/env python3
import os
import re
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = "logs"

def parse_elapsed_time(filepath):
    """Extract elapsed time in microseconds from a log file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Match both "usec" and "microseconds" formats
            match = re.search(r'Elapsed:\s*(\d+)\s*(?:usec|microseconds)', content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def collect_benchmark_data():
    """Collect all benchmark data from log files."""
    data = defaultdict(lambda: {"gforth": [], "rpyforth": []})

    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))

    for filepath in log_files:
        filename = os.path.basename(filepath)
        # Pattern: benchmark.fs_interpreter_run.log
        match = re.match(r'(.+\.fs)_(gforth|rpyforth\.sh)_(\d+)\.log', filename)
        if match:
            benchmark = match.group(1)
            interpreter = "gforth" if match.group(2) == "gforth" else "rpyforth"
            run_num = int(match.group(3))

            elapsed = parse_elapsed_time(filepath)
            if elapsed is not None:
                data[benchmark][interpreter].append(elapsed)

    return data

def compute_statistics(times):
    """Compute mean, std, min, max from a list of times."""
    if not times:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    arr = np.array(times)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "median": np.median(arr)
    }

def plot_bar_comparison(data, output_file="benchmark_comparison.pdf"):
    """Create a bar chart comparing gforth and rpyforth execution times."""
    benchmarks = sorted(data.keys())

    gforth_means = []
    rpyforth_means = []
    gforth_stds = []
    rpyforth_stds = []

    for bench in benchmarks:
        gforth_stats = compute_statistics(data[bench]["gforth"])
        rpyforth_stats = compute_statistics(data[bench]["rpyforth"])

        gforth_means.append(gforth_stats["mean"] / 1000)  # Convert to ms
        rpyforth_means.append(rpyforth_stats["mean"] / 1000)
        gforth_stds.append(gforth_stats["std"] / 1000)
        rpyforth_stds.append(rpyforth_stats["std"] / 1000)

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, gforth_means, width, label='gforth',
                   yerr=gforth_stds, capsize=3, color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, rpyforth_means, width, label='rpyforth-c',
                   yerr=rpyforth_stds, capsize=3, color='coral', alpha=0.8)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Benchmark Comparison: gforth vs rpyforth-c')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('.fs', '') for b in benchmarks], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved bar chart to {output_file}")

def plot_speedup(data, output_file="benchmark_speedup.pdf"):
    """Create a speedup chart (gforth time / rpyforth time)."""
    benchmarks = sorted(data.keys())

    speedups = []
    for bench in benchmarks:
        gforth_stats = compute_statistics(data[bench]["gforth"])
        rpyforth_stats = compute_statistics(data[bench]["rpyforth"])

        if rpyforth_stats["mean"] > 0:
            speedup = gforth_stats["mean"] / rpyforth_stats["mean"]
        else:
            speedup = 0
        speedups.append(speedup)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax.bar([b.replace('.fs', '') for b in benchmarks], speedups, color=colors, alpha=0.8)

    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal performance')
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Speedup (gforth time / rpyforth time)')
    ax.set_title('rpyforth-c Speedup over gforth\n(>1 means rpyforth is faster)')
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([b.replace('.fs', '') for b in benchmarks], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved speedup chart to {output_file}")

def plot_boxplot(data, output_file="benchmark_boxplot.pdf"):
    """Create box plots for each benchmark comparing both interpreters."""
    benchmarks = sorted(data.keys())
    n_benchmarks = len(benchmarks)

    fig, axes = plt.subplots(2, (n_benchmarks + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]

        gforth_times = [t / 1000 for t in data[bench]["gforth"]]  # Convert to ms
        rpyforth_times = [t / 1000 for t in data[bench]["rpyforth"]]

        bp = ax.boxplot([gforth_times, rpyforth_times],
                        tick_labels=['gforth', 'rpyforth-c'],
                        patch_artist=True)

        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')

        ax.set_title(bench.replace('.fs', ''))
        ax.set_ylabel('Time (ms)')
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(len(benchmarks), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Benchmark Distribution: gforth vs rpyforth-c', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved box plot to {output_file}")

def print_statistics_table(data):
    """Print a formatted statistics table."""
    print("\n" + "="*80)
    print("BENCHMARK STATISTICS (times in milliseconds)")
    print("="*80)

    header = f"{'Benchmark':<15} {'Interpreter':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}"
    print(header)
    print("-"*80)

    for bench in sorted(data.keys()):
        bench_name = bench.replace('.fs', '')
        for interp in ["gforth", "rpyforth"]:
            stats = compute_statistics(data[bench][interp])
            interp_name = "rpyforth-c" if interp == "rpyforth" else interp
            print(f"{bench_name:<15} {interp_name:<12} {stats['mean']/1000:>10.2f} {stats['std']/1000:>10.2f} {stats['min']/1000:>10.2f} {stats['max']/1000:>10.2f} {stats['median']/1000:>10.2f}")
        print()

    print("="*80)
    print("\nSPEEDUP SUMMARY (gforth time / rpyforth time)")
    print("-"*40)

    for bench in sorted(data.keys()):
        bench_name = bench.replace('.fs', '')
        gforth_stats = compute_statistics(data[bench]["gforth"])
        rpyforth_stats = compute_statistics(data[bench]["rpyforth"])

        if rpyforth_stats["mean"] > 0:
            speedup = gforth_stats["mean"] / rpyforth_stats["mean"]
            faster = "rpyforth" if speedup > 1 else "gforth"
            print(f"{bench_name:<15}: {speedup:.2f}x ({faster} is faster)")

    print("="*80)

def main():
    print("Collecting benchmark data from logs/...")
    data = collect_benchmark_data()

    if not data:
        print("No benchmark data found in logs/ directory")
        return

    print(f"Found {len(data)} benchmarks")

    # Print statistics table
    print_statistics_table(data)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_bar_comparison(data)
    plot_speedup(data)
    plot_boxplot(data)

    print("\nDone!")

if __name__ == "__main__":
    main()
