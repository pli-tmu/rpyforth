#!/bin/bash
BENCHMARKS=(fibo.fs ack.fs nestedloop.fs sieve.fs heap.fs ary.fs)
COMMANDS=(gforth ./rpyforth.sh)
WARMUP_RUNS=5
MEASURE_RUNS=10

trap 'echo -e "\n\n[Aborted by user] Exiting..."; exit 1' INT

# Create a unique directory for this batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Benchmark Batch: $TIMESTAMP"
echo "Logs directory: $LOG_DIR"
echo "---------------------------------------------------"

for bm in "${BENCHMARKS[@]}"; do
    for cmd in "${COMMANDS[@]}"; do
        # Clean command name for filename (e.g., ./rpyforth.sh -> rpyforth)
        cmd_name=$(basename "${cmd}" .sh)

        echo "[Target: ${cmd_name} | Bench: ${bm}]"

        # 1. Warmup Phase (JIT Training)
        echo "  - Warming up (${WARMUP_RUNS} runs)..."
        for i in $(seq 1 $WARMUP_RUNS); do
            ${cmd} "shootout/${bm}" > "${LOG_DIR}/${bm}_${cmd_name}_warmup_${i}.log" 2>&1
        done

        # 2. Measurement Phase (Steady State)
        echo "  - Measuring (${MEASURE_RUNS} runs)..."
        for i in $(seq 1 $MEASURE_RUNS); do
            ${cmd} "shootout/${bm}" > "${LOG_DIR}/${bm}_${cmd_name}_run_${i}.log" 2>&1
        done
    done
done

echo "Done. All logs saved to ${LOG_DIR}"
