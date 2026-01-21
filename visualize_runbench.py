#!/usr/bin/env python3
"""
Benchmark Visualization Tool for rpyforth

Generates:
1. Speedup boxplots comparing gforth vs rpyforth
2. Warmup curve analysis showing JIT compilation effects
3. Memory usage comparison charts
"""

import matplotlib
matplotlib.use('Agg')

import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statistics import geometric_mean as gmean
import sys

plt.style.use('ggplot')


def calc_geometric_mean(values):
    """Calculate geometric mean of positive values."""
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    return gmean(values)

# File patterns
FILENAME_PATTERN = re.compile(r"^(.*?)_(.*?)_(warmup|run)_(\d+)\.log$")
CURVE_PATTERN = re.compile(r"^(.*?)_(.*?)_curve\.log$")
MEM_PATTERN = re.compile(r"^(.*?)_(.*?)_(warmup|run|curve)_?(\d*)\.mem$")

# Output files
OUTPUT_DIR = "benchmark_results"
OUTPUT_BOXPLOT = "speedup_boxplot.pdf"
OUTPUT_PAPER_SPEEDUP = "speedup_paper.pdf"
OUTPUT_CURVE = "warmup_curves.pdf"
OUTPUT_MEMORY = "memory_comparison.pdf"
OUTPUT_COMBINED = "benchmark_report.pdf"
OUTPUT_CSV = "benchmark_stats.csv"


def parse_time_from_content(content):
    """Parse execution time from 'Elapsed: N usec' format."""
    matches = re.findall(r"Elapsed:\s+(\d+)\s+usec", content)
    if matches:
        return float(matches[-1]) / 1_000_000.0  # Convert to seconds
    return None


def parse_curve_data(content):
    """Parse curve benchmark output (CSV format: Iteration,Time(usec))."""
    lines = content.strip().split('\n')
    iterations = []
    times = []

    for line in lines:
        # Skip header
        if 'Iteration' in line or not line.strip():
            continue
        # Parse "N , time" format (Forth outputs with spaces around comma)
        match = re.match(r'\s*(\d+)\s*[,\s]+(-?\d+)', line)
        if match:
            iterations.append(int(match.group(1)))
            times.append(float(match.group(2)) / 1_000_000.0)  # Convert to seconds

    return iterations, times


def parse_memory_file(filepath):
    """Parse memory metrics from /usr/bin/time -v output."""
    if not os.path.exists(filepath):
        return None

    metrics = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Maximum resident set size (kbytes)
    match = re.search(r'Maximum resident set size.*?:\s*(\d+)', content)
    if match:
        metrics['peak_memory_kb'] = int(match.group(1))

    # Minor page faults
    match = re.search(r'Minor.*?page faults.*?:\s*(\d+)', content)
    if match:
        metrics['minor_faults'] = int(match.group(1))

    # Major page faults
    match = re.search(r'Major.*?page faults.*?:\s*(\d+)', content)
    if match:
        metrics['major_faults'] = int(match.group(1))

    # Voluntary context switches
    match = re.search(r'Voluntary context switches.*?:\s*(\d+)', content)
    if match:
        metrics['voluntary_ctx'] = int(match.group(1))

    return metrics if metrics else None


def load_standard_logs(log_dir):
    """Load standard benchmark logs (warmup and run phases)."""
    data = []

    for filename in os.listdir(log_dir):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue

        bm_name, cmd_name, run_type, run_id = match.groups()

        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            exec_time = parse_time_from_content(f.read())
            if exec_time is not None:
                data.append({
                    "benchmark": bm_name.replace('.fs', ''),
                    "command": cmd_name,
                    "phase": run_type,
                    "run_id": int(run_id),
                    "time": exec_time
                })

        # Load corresponding memory file
        mem_file = filepath.replace('.log', '.mem')
        mem_metrics = parse_memory_file(mem_file)
        if mem_metrics and data:
            data[-1].update(mem_metrics)

    return pd.DataFrame(data)


def load_curve_logs(log_dir):
    """Load curve benchmark logs for warmup analysis."""
    curve_data = {}

    for filename in os.listdir(log_dir):
        match = CURVE_PATTERN.match(filename)
        if not match:
            continue

        bm_name, cmd_name = match.groups()
        bm_name = bm_name.replace('.fs', '')

        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            iterations, times = parse_curve_data(f.read())
            if iterations and times:
                key = (bm_name, cmd_name)
                curve_data[key] = {'iterations': iterations, 'times': times}

    return curve_data


def load_memory_data(log_dir):
    """Load all memory metrics from .mem files."""
    data = []

    for filename in os.listdir(log_dir):
        match = MEM_PATTERN.match(filename)
        if not match:
            continue

        bm_name, cmd_name, phase, run_id = match.groups()
        bm_name = bm_name.replace('.fs', '')

        filepath = os.path.join(log_dir, filename)
        mem_metrics = parse_memory_file(filepath)
        if mem_metrics:
            entry = {
                "benchmark": bm_name,
                "command": cmd_name,
                "phase": phase,
            }
            entry.update(mem_metrics)
            data.append(entry)

    return pd.DataFrame(data)


def create_speedup_boxplot(df, output_path):
    """Create boxplot comparing speedup between gforth and rpyforth."""
    # Filter to measurement runs only
    df_runs = df[df['phase'] == 'run'].copy()

    if df_runs.empty:
        print("No run phase data found for boxplot.")
        return

    baseline_cmd = "gforth"
    baseline_means = df_runs[df_runs['command'] == baseline_cmd].groupby('benchmark')['time'].mean()

    if baseline_means.empty:
        print(f"Warning: Baseline command '{baseline_cmd}' not found.")
        return

    def calc_speedup(row):
        if row['benchmark'] in baseline_means.index:
            base_time = baseline_means[row['benchmark']]
            return base_time / row['time'] if row['time'] > 0 else None
        return None

    df_runs['speedup'] = df_runs.apply(calc_speedup, axis=1)

    benchmarks = sorted(df_runs['benchmark'].unique())
    commands = sorted(df_runs['command'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for grouped boxplot
    positions = []
    boxplot_data = []
    colors = []
    color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

    x_pos = 0
    x_ticks = []
    x_labels = []

    for bm in benchmarks:
        group_start = x_pos
        for cmd in commands:
            subset = df_runs[(df_runs['benchmark'] == bm) & (df_runs['command'] == cmd)]['speedup'].dropna()
            if not subset.empty:
                boxplot_data.append(subset.values)
                positions.append(x_pos)
                colors.append(color_map.get(cmd, '#95a5a6'))
                x_pos += 1
        x_ticks.append((group_start + x_pos - 1) / 2)
        x_labels.append(bm)
        x_pos += 0.5  # Gap between benchmark groups

    # Create boxplots
    bp = ax.boxplot(boxplot_data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Styling
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (gforth = 1.0)')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel('Speedup Factor (higher is better)', fontsize=12)
    ax.set_title('Performance Comparison: rpyforth vs gforth\n(Speedup = gforth_time / measured_time)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Calculate and display geometric mean for each command
    geomean_text = []
    for cmd in commands:
        cmd_speedups = df_runs[df_runs['command'] == cmd]['speedup'].dropna().values
        if len(cmd_speedups) > 0:
            gmean = calc_geometric_mean(cmd_speedups)
            if gmean is not None:
                geomean_text.append(f"{cmd}: {gmean:.2f}x")

    if geomean_text:
        geomean_str = "Geomean: " + ", ".join(geomean_text)
        ax.text(0.02, 0.98, geomean_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['gforth'], alpha=0.7, label='gforth'),
        Patch(facecolor=color_map['rpyforth'], alpha=0.7, label='rpyforth'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Speedup boxplot saved to {output_path}")


def create_speedup_barchart_for_paper(df, output_path):
    """Create a clean bar chart showing rpyforth speedup vs gforth for paper publication."""
    # Filter to measurement runs only
    df_runs = df[df['phase'] == 'run'].copy()

    if df_runs.empty:
        print("No run phase data found for paper barchart.")
        return

    baseline_cmd = "gforth"
    target_cmd = "rpyforth"

    # Calculate mean times for each benchmark and command
    mean_times = df_runs.groupby(['benchmark', 'command'])['time'].mean().unstack()

    if baseline_cmd not in mean_times.columns or target_cmd not in mean_times.columns:
        print(f"Warning: Need both '{baseline_cmd}' and '{target_cmd}' data for paper chart.")
        return

    # Calculate speedup (gforth_time / rpyforth_time)
    speedups = mean_times[baseline_cmd] / mean_times[target_cmd]
    speedups = speedups.dropna().sort_values(ascending=False)

    if speedups.empty:
        print("No speedup data available for paper chart.")
        return

    # Calculate standard deviation for error bars
    std_times = df_runs.groupby(['benchmark', 'command'])['time'].std().unstack()
    speedup_errors = []
    for bm in speedups.index:
        # Error propagation for ratio: relative_error = sqrt((da/a)^2 + (db/b)^2)
        gforth_mean = mean_times.loc[bm, baseline_cmd]
        rpyforth_mean = mean_times.loc[bm, target_cmd]
        gforth_std = std_times.loc[bm, baseline_cmd] if bm in std_times.index else 0
        rpyforth_std = std_times.loc[bm, target_cmd] if bm in std_times.index else 0

        if pd.isna(gforth_std):
            gforth_std = 0
        if pd.isna(rpyforth_std):
            rpyforth_std = 0

        rel_err_gforth = gforth_std / gforth_mean if gforth_mean > 0 else 0
        rel_err_rpyforth = rpyforth_std / rpyforth_mean if rpyforth_mean > 0 else 0
        rel_err_total = np.sqrt(rel_err_gforth**2 + rel_err_rpyforth**2)
        speedup_errors.append(speedups[bm] * rel_err_total)

    # Calculate geometric mean
    gmean_speedup = calc_geometric_mean(speedups.values)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prepare data with geomean
    benchmarks = list(speedups.index) + ['Geomean']
    values = list(speedups.values) + [gmean_speedup]
    errors = speedup_errors + [0]  # No error bar for geomean

    x = np.arange(len(benchmarks))

    # Colors: blue for individual benchmarks, red for geomean
    colors = ['#3498db'] * len(speedups) + ['#e74c3c']

    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add error bars for individual benchmarks only
    ax.errorbar(x[:-1], values[:-1], yerr=errors[:-1], fmt='none', color='black', capsize=3)

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add baseline line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='gforth baseline')

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Speedup (higher is better)', fontsize=11)
    ax.set_xlabel('Benchmark', fontsize=11)
    ax.set_title('rpyforth Speedup over gforth', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Paper speedup barchart saved to {output_path}")


def create_warmup_curves(curve_data, output_path):
    """Create warmup curve visualization showing JIT effects over iterations."""
    if not curve_data:
        print("No curve data found for warmup visualization.")
        return

    # Group by benchmark
    benchmarks = sorted(set(bm for bm, _ in curve_data.keys()))
    n_benchmarks = len(benchmarks)

    if n_benchmarks == 0:
        return

    # Create subplots
    cols = min(3, n_benchmarks)
    rows = (n_benchmarks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_benchmarks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_benchmarks > 1 else [axes]

    color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

    for idx, bm in enumerate(benchmarks):
        ax = axes[idx]

        for (benchmark, cmd), data in curve_data.items():
            if benchmark != bm:
                continue

            iterations = data['iterations']
            times = data['times']

            color = color_map.get(cmd, '#95a5a6')
            ax.plot(iterations, times, 'o-', color=color, label=cmd,
                    markersize=4, linewidth=1.5, alpha=0.8)

            # Add trend line for rpyforth to show warmup effect
            if cmd == 'rpyforth' and len(times) > 5:
                # Calculate moving average
                window = min(5, len(times) // 3)
                if window > 1:
                    moving_avg = np.convolve(times, np.ones(window)/window, mode='valid')
                    ax.plot(iterations[window-1:], moving_avg, '--', color=color,
                            alpha=0.5, linewidth=2, label=f'{cmd} (trend)')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{bm}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('JIT Warmup Curves\n(Lower time after warmup indicates JIT optimization)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Warmup curves saved to {output_path}")


def create_memory_comparison(mem_df, output_path):
    """Create memory usage comparison chart."""
    if mem_df.empty or 'peak_memory_kb' not in mem_df.columns:
        print("No memory data found for visualization.")
        return

    # Filter to run phase only and aggregate
    df_runs = mem_df[mem_df['phase'] == 'run'].copy() if 'phase' in mem_df.columns else mem_df.copy()

    if df_runs.empty:
        df_runs = mem_df.copy()

    # Aggregate by benchmark and command
    agg_data = df_runs.groupby(['benchmark', 'command']).agg({
        'peak_memory_kb': ['mean', 'std']
    }).reset_index()
    agg_data.columns = ['benchmark', 'command', 'memory_mean', 'memory_std']

    benchmarks = sorted(agg_data['benchmark'].unique())
    commands = sorted(agg_data['command'].unique())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart for peak memory
    x = np.arange(len(benchmarks))
    width = 0.35
    color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

    for i, cmd in enumerate(commands):
        cmd_data = agg_data[agg_data['command'] == cmd]
        heights = []
        errors = []
        for bm in benchmarks:
            row = cmd_data[cmd_data['benchmark'] == bm]
            if not row.empty:
                heights.append(row['memory_mean'].values[0] / 1024)  # Convert to MB
                errors.append(row['memory_std'].values[0] / 1024 if not np.isnan(row['memory_std'].values[0]) else 0)
            else:
                heights.append(0)
                errors.append(0)

        color = color_map.get(cmd, '#95a5a6')
        ax1.bar(x + i * width, heights, width, label=cmd, color=color, alpha=0.7, yerr=errors, capsize=3)

    ax1.set_xlabel('Benchmark')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Peak Memory Usage Comparison')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Memory ratio chart (rpyforth / gforth)
    ratios = []
    labels = []
    for bm in benchmarks:
        gforth_data = agg_data[(agg_data['benchmark'] == bm) & (agg_data['command'] == 'gforth')]
        rpyforth_data = agg_data[(agg_data['benchmark'] == bm) & (agg_data['command'] == 'rpyforth')]

        if not gforth_data.empty and not rpyforth_data.empty:
            ratio = rpyforth_data['memory_mean'].values[0] / gforth_data['memory_mean'].values[0]
            ratios.append(ratio)
            labels.append(bm)

    if ratios:
        colors = ['#e74c3c' if r > 1 else '#2ecc71' for r in ratios]
        ax2.barh(labels, ratios, color=colors, alpha=0.7)
        ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Memory Ratio (rpyforth / gforth)')
        ax2.set_title('Memory Usage Ratio\n(< 1.0 = rpyforth uses less memory)')
        ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Memory comparison saved to {output_path}")


def create_combined_report(df, curve_data, mem_df, output_path):
    """Create a combined multi-page PDF report."""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: Speedup boxplot
        if not df.empty:
            df_runs = df[df['phase'] == 'run'].copy()
            if not df_runs.empty:
                baseline_cmd = "gforth"
                baseline_means = df_runs[df_runs['command'] == baseline_cmd].groupby('benchmark')['time'].mean()

                if not baseline_means.empty:
                    def calc_speedup(row):
                        if row['benchmark'] in baseline_means.index:
                            base_time = baseline_means[row['benchmark']]
                            return base_time / row['time'] if row['time'] > 0 else None
                        return None

                    df_runs['speedup'] = df_runs.apply(calc_speedup, axis=1)

                    fig, ax = plt.subplots(figsize=(14, 7))
                    benchmarks = sorted(df_runs['benchmark'].unique())
                    commands = sorted(df_runs['command'].unique())

                    positions = []
                    boxplot_data = []
                    colors = []
                    color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

                    x_pos = 0
                    x_ticks = []
                    x_labels = []

                    for bm in benchmarks:
                        group_start = x_pos
                        for cmd in commands:
                            subset = df_runs[(df_runs['benchmark'] == bm) & (df_runs['command'] == cmd)]['speedup'].dropna()
                            if not subset.empty:
                                boxplot_data.append(subset.values)
                                positions.append(x_pos)
                                colors.append(color_map.get(cmd, '#95a5a6'))
                                x_pos += 1
                        x_ticks.append((group_start + x_pos - 1) / 2)
                        x_labels.append(bm)
                        x_pos += 0.5

                    if boxplot_data:
                        bp = ax.boxplot(boxplot_data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)

                        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(x_labels, fontsize=11)
                        ax.set_ylabel('Speedup Factor')
                        ax.set_title('Performance Comparison: rpyforth vs gforth')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                        # Calculate and display geometric mean
                        geomean_text = []
                        for cmd in commands:
                            cmd_speedups = df_runs[df_runs['command'] == cmd]['speedup'].dropna().values
                            if len(cmd_speedups) > 0:
                                gm = calc_geometric_mean(cmd_speedups)
                                if gm is not None:
                                    geomean_text.append(f"{cmd}: {gm:.2f}x")

                        if geomean_text:
                            geomean_str = "Geomean: " + ", ".join(geomean_text)
                            ax.text(0.02, 0.98, geomean_str, transform=ax.transAxes, fontsize=10,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor=color_map['gforth'], alpha=0.7, label='gforth'),
                            Patch(facecolor=color_map['rpyforth'], alpha=0.7, label='rpyforth'),
                        ]
                        ax.legend(handles=legend_elements)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()

        # Page 2: Warmup curves
        if curve_data:
            benchmarks = sorted(set(bm for bm, _ in curve_data.keys()))
            n_benchmarks = len(benchmarks)

            if n_benchmarks > 0:
                cols = min(3, n_benchmarks)
                rows = (n_benchmarks + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

                if n_benchmarks == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

                for idx, bm in enumerate(benchmarks):
                    ax = axes[idx]
                    for (benchmark, cmd), data in curve_data.items():
                        if benchmark != bm:
                            continue
                        color = color_map.get(cmd, '#95a5a6')
                        ax.plot(data['iterations'], data['times'], 'o-', color=color,
                                label=cmd, markersize=4, linewidth=1.5, alpha=0.8)

                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Time (seconds)')
                    ax.set_title(f'{bm}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                for idx in range(n_benchmarks, len(axes)):
                    axes[idx].set_visible(False)

                fig.suptitle('JIT Warmup Curves', fontsize=14)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

        # Page 3: Memory comparison
        if not mem_df.empty and 'peak_memory_kb' in mem_df.columns:
            df_runs = mem_df[mem_df['phase'] == 'run'].copy() if 'phase' in mem_df.columns else mem_df.copy()
            if df_runs.empty:
                df_runs = mem_df.copy()

            agg_data = df_runs.groupby(['benchmark', 'command']).agg({
                'peak_memory_kb': ['mean', 'std']
            }).reset_index()
            agg_data.columns = ['benchmark', 'command', 'memory_mean', 'memory_std']

            if not agg_data.empty:
                fig, ax = plt.subplots(figsize=(12, 6))

                benchmarks = sorted(agg_data['benchmark'].unique())
                commands = sorted(agg_data['command'].unique())

                x = np.arange(len(benchmarks))
                width = 0.35
                color_map = {'gforth': '#3498db', 'rpyforth': '#e74c3c'}

                for i, cmd in enumerate(commands):
                    cmd_data = agg_data[agg_data['command'] == cmd]
                    heights = []
                    for bm in benchmarks:
                        row = cmd_data[cmd_data['benchmark'] == bm]
                        heights.append(row['memory_mean'].values[0] / 1024 if not row.empty else 0)

                    color = color_map.get(cmd, '#95a5a6')
                    ax.bar(x + i * width, heights, width, label=cmd, color=color, alpha=0.7)

                ax.set_xlabel('Benchmark')
                ax.set_ylabel('Peak Memory (MB)')
                ax.set_title('Peak Memory Usage Comparison')
                ax.set_xticks(x + width / 2)
                ax.set_xticklabels(benchmarks, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    print(f"Combined report saved to {output_path}")


def save_statistics(df, output_path):
    """Save statistical summary to CSV."""
    if df.empty:
        return

    df_runs = df[df['phase'] == 'run'].copy()
    if df_runs.empty:
        return

    baseline_cmd = "gforth"
    baseline_means = df_runs[df_runs['command'] == baseline_cmd].groupby('benchmark')['time'].mean()

    def calc_speedup(row):
        if row['benchmark'] in baseline_means.index:
            base_time = baseline_means[row['benchmark']]
            return base_time / row['time'] if row['time'] > 0 else None
        return None

    df_runs['speedup'] = df_runs.apply(calc_speedup, axis=1)

    # Include memory if available
    agg_cols = ['time', 'speedup']
    if 'peak_memory_kb' in df_runs.columns:
        agg_cols.append('peak_memory_kb')

    stats = df_runs.groupby(["benchmark", "command"])[agg_cols].agg(["mean", "std", "min", "max"])

    print("\n--- Summary Statistics ---")
    print(stats.to_string())
    stats.to_csv(output_path)
    print(f"\nStats saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rpyforth benchmark results")
    parser.add_argument("log_dir", type=str, help="Path to log directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for generated files")
    parser.add_argument("--no-combined", action="store_true",
                        help="Skip generating combined PDF report")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Directory '{args.log_dir}' not found.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading logs from: {args.log_dir}")

    # Load all data
    df = load_standard_logs(args.log_dir)
    curve_data = load_curve_logs(args.log_dir)
    mem_df = load_memory_data(args.log_dir)

    print(f"Loaded {len(df)} standard benchmark entries")
    print(f"Loaded {len(curve_data)} curve benchmarks")
    print(f"Loaded {len(mem_df)} memory entries")

    if df.empty and not curve_data and mem_df.empty:
        print("No valid data found.")
        return

    # Generate individual visualizations
    if not df.empty:
        create_speedup_boxplot(df, os.path.join(args.output_dir, OUTPUT_BOXPLOT))
        create_speedup_barchart_for_paper(df, os.path.join(args.output_dir, OUTPUT_PAPER_SPEEDUP))
        save_statistics(df, os.path.join(args.output_dir, OUTPUT_CSV))

    if curve_data:
        create_warmup_curves(curve_data, os.path.join(args.output_dir, OUTPUT_CURVE))

    if not mem_df.empty:
        create_memory_comparison(mem_df, os.path.join(args.output_dir, OUTPUT_MEMORY))

    # Generate combined report
    if not args.no_combined:
        create_combined_report(df, curve_data, mem_df,
                               os.path.join(args.output_dir, OUTPUT_COMBINED))

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
