#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BENCHMARKS=(fibo.fs ack.fs nestedloop.fs sieve.fs heap.fs ary.fs)
CURVE_BENCHMARKS=(fibo.fs ack.fs nestedloop.fs sieve.fs heap.fs ary.fs)
COMMANDS=(gforth ./rpyforth.sh ./vfxforth.sh ./swiftforth.sh)
WARMUP_RUNS=${WARMUP_RUNS:-5}
MEASURE_RUNS=${MEASURE_RUNS:-100}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-60}

# Parse command line arguments
MODE="standard"  # standard, curve, or all
ENABLE_MEMORY=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --curve)
            MODE="curve"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --no-memory)
            ENABLE_MEMORY=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --curve      Run curve benchmarks only (for warmup analysis)"
            echo "  --all        Run both standard and curve benchmarks"
            echo "  --no-memory  Disable memory profiling"
            echo "  -h, --help   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

trap 'echo -e "\n\n[Aborted by user] Exiting..."; exit 1' INT

# Create a unique directory for this batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Benchmark Batch: $TIMESTAMP"
echo "Mode: $MODE"
echo "Memory profiling: $ENABLE_MEMORY"
echo "Logs directory: $LOG_DIR"
echo "---------------------------------------------------"

# Resolve the benchmark file path for a given target.
# VFXForth and SwiftForth use Forth-specific adaptations under shootout/<forth>/.
resolve_bench_file() {
    local cmd_name="$1"
    local bm="$2"
    case "$cmd_name" in
        run-vfxforth|vfxforth)
            if [ -f "shootout/vfxforth/${bm}" ]; then
                echo "shootout/vfxforth/${bm}"
            else
                echo "shootout/${bm}"
            fi
            ;;
        run-swiftforth|swiftforth)
            if [ -f "shootout/swiftforth/${bm}" ]; then
                echo "shootout/swiftforth/${bm}"
            else
                echo "shootout/${bm}"
            fi
            ;;
        *)
            echo "shootout/${bm}"
            ;;
    esac
}

# Function to run a single benchmark with optional memory profiling
run_benchmark() {
    local cmd="$1"
    local bench_file="$2"
    local log_file="$3"
    local mem_log_file="$4"

    if [ "$ENABLE_MEMORY" = true ]; then
        # Use /usr/bin/time -v to capture memory metrics; timeout prevents hangs.
        timeout "${BENCH_TIMEOUT}" /usr/bin/time -v ${cmd} "${bench_file}" > "${log_file}" 2> "${mem_log_file}.tmp"
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "TIMEOUT after ${BENCH_TIMEOUT}s" >> "${log_file}"
        fi
        # Extract memory info and append to mem_log_file, keep stderr output
        grep -E "(Maximum resident set size|Minor|Major|Voluntary|Involuntary|Page size|Elapsed)" "${mem_log_file}.tmp" > "${mem_log_file}" 2>/dev/null
        rm -f "${mem_log_file}.tmp"
    else
        timeout "${BENCH_TIMEOUT}" ${cmd} "${bench_file}" > "${log_file}" 2>&1
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "TIMEOUT after ${BENCH_TIMEOUT}s" >> "${log_file}"
        fi
    fi
}

# Standard benchmark mode
run_standard_benchmarks() {
    for bm in "${BENCHMARKS[@]}"; do
        for cmd in "${COMMANDS[@]}"; do
            cmd_name=$(basename "${cmd}" .sh)
            bench_file=$(resolve_bench_file "${cmd_name}" "${bm}")

            echo "[Target: ${cmd_name} | Bench: ${bm}]"

            # 1. Warmup Phase (JIT Training)
            echo "  - Warming up (${WARMUP_RUNS} runs)..."
            for i in $(seq 1 $WARMUP_RUNS); do
                run_benchmark "${cmd}" "${bench_file}" \
                    "${LOG_DIR}/${bm}_${cmd_name}_warmup_${i}.log" \
                    "${LOG_DIR}/${bm}_${cmd_name}_warmup_${i}.mem"
            done

            # 2. Measurement Phase (Steady State)
            echo "  - Measuring (${MEASURE_RUNS} runs)..."
            for i in $(seq 1 $MEASURE_RUNS); do
                run_benchmark "${cmd}" "${bench_file}" \
                    "${LOG_DIR}/${bm}_${cmd_name}_run_${i}.log" \
                    "${LOG_DIR}/${bm}_${cmd_name}_run_${i}.mem"
            done
        done
    done
}

# Curve benchmark mode for warmup analysis
run_curve_benchmarks() {
    echo "Running curve benchmarks for warmup analysis..."
    for bm in "${CURVE_BENCHMARKS[@]}"; do
        for cmd in "${COMMANDS[@]}"; do
            cmd_name=$(basename "${cmd}" .sh)
            curve_file=$(resolve_bench_file "${cmd_name}" "curve/${bm}")

            if [ ! -f "${curve_file}" ]; then
                echo "  [Skip] Curve benchmark not found: ${curve_file}"
                continue
            fi

            echo "[Curve: ${cmd_name} | Bench: ${bm}]"

            # Run curve benchmark (single run, outputs per-iteration times)
            run_benchmark "${cmd}" "${curve_file}" \
                "${LOG_DIR}/${bm}_${cmd_name}_curve.log" \
                "${LOG_DIR}/${bm}_${cmd_name}_curve.mem"
        done
    done
}

# Execute based on mode
case $MODE in
    standard)
        run_standard_benchmarks
        ;;
    curve)
        run_curve_benchmarks
        ;;
    all)
        run_standard_benchmarks
        echo ""
        run_curve_benchmarks
        ;;
esac

echo "Done. All logs saved to ${LOG_DIR}"
