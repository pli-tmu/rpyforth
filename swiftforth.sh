#!/bin/bash
# Run a benchmark file on the locally installed SwiftForth.
# Preserves the caller's working directory so that relative includes resolve
# correctly (e.g. for appbench programs that load local data files).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_FILE="$1"

# Resolve the benchmark file against the caller's cwd for relative paths,
# otherwise use it as-is (absolute paths are already fine).
if [ -f "$BENCH_FILE" ]; then
    BENCH_FILE="$(cd "$(dirname "$BENCH_FILE")" && pwd)/$(basename "$BENCH_FILE")"
fi

# Use the Forth-specific benchmark adaptation when available.
SF_BENCH_FILE="${BENCH_FILE/${REPO_ROOT}\/shootout\//${REPO_ROOT}/shootout/swiftforth/}"
if [ ! -f "$SF_BENCH_FILE" ]; then
    SF_BENCH_FILE="$BENCH_FILE"
fi

# Append wall-clock elapsed time in microseconds for cross-engine comparison.
start_ns=$(date +%s%N)
"${REPO_ROOT}/swiftforth/SwiftForth/bin/linux/sf64" "include ${SF_BENCH_FILE}"
end_ns=$(date +%s%N)
elapsed_us=$(( (end_ns - start_ns) / 1000 ))
echo "Elapsed: ${elapsed_us} usec"
