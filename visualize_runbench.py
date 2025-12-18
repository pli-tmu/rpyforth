import matplotlib
matplotlib.use('Agg')

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')

FILENAME_PATTERN = re.compile(r"^(.*?)_(.*?)_(warmup|run)_(\d+)\.log$")
OUTPUT_IMG = "benchmark_boxplot.pdf"
OUTPUT_CSV = "benchmark_stats.csv"

def parse_time_from_content(content):
    matches = re.findall(r"Elapsed:\s+(\d+)\s+usec", content)
    if matches:
        return float(matches[-1]) / 1_000_000.0
    return None

def load_logs(log_dir):
    data = []
    if not os.path.exists(log_dir):
        print(f"Error: Directory '{log_dir}' not found.")
        sys.exit(1)

    print(f"Loading logs from: {log_dir}")

    for filename in os.listdir(log_dir):
        match = FILENAME_PATTERN.match(filename)
        if not match: continue

        bm_name, cmd_name, run_type, run_id = match.groups()
        if run_type == "warmup": continue # Warmupを除外

        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            exec_time = parse_time_from_content(f.read())
            if exec_time is not None:
                data.append({
                    "benchmark": bm_name,
                    "command": cmd_name,
                    "time": exec_time
                })
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="Path to log directory")
    args = parser.parse_args()

    df = load_logs(args.log_dir)
    if df.empty:
        print("No valid data found.")
        return

    baseline_cmd = "gforth"

    baseline_means = df[df['command'] == baseline_cmd].groupby('benchmark')['time'].mean()

    if baseline_means.empty:
        print(f"Error: Baseline command '{baseline_cmd}' not found in logs.")
        return

    def calc_speedup(row):
        if row['benchmark'] in baseline_means:
            base_time = baseline_means[row['benchmark']]
            return base_time / row['time']
        return None

    df['speedup'] = df.apply(calc_speedup, axis=1)

    stats = df.groupby(["benchmark", "command"])[["time", "speedup"]].agg(["mean", "std", "min", "max"])
    print("\n--- Summary Statistics ---")
    print(stats)
    stats.to_csv(OUTPUT_CSV)
    print(f"\nStats saved to {OUTPUT_CSV}")

    benchmarks = df['benchmark'].unique()
    commands = df['command'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []

    for bm in sorted(benchmarks):
        for cmd in sorted(commands):
            subset = df[(df['benchmark'] == bm) & (df['command'] == cmd)]['speedup']
            boxplot_data.append(subset.values)
            labels.append(f"{bm}\n({cmd})")

    ax.boxplot(boxplot_data, labels=labels, showmeans=True)

    ax.set_title("Relative Performance Distribution (Speedup)\nHigher is Better (Baseline: GForth Mean = 1.0)", fontsize=14)
    ax.set_ylabel("Speedup Factor (x times gforth)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.axhline(y=1.0, color='r', linestyle='-', linewidth=1, label="Baseline (GForth)")
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(OUTPUT_IMG)
    print(f"Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
