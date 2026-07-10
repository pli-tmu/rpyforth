#!/bin/bash
# Run a benchmark file on the locally installed VFXForth.
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
VFX_BENCH_FILE="${BENCH_FILE/${REPO_ROOT}\/shootout\//${REPO_ROOT}/shootout/vfxforth/}"
if [ ! -f "$VFX_BENCH_FILE" ]; then
    VFX_BENCH_FILE="$BENCH_FILE"
fi

# Run VFXForth and strip the startup banner / include announcements so that
# downstream parsers see only the benchmark output. Also append wall-clock
# elapsed time in microseconds for cross-engine comparison.
start_ns=$(date +%s%N)
VFX_BATCH_MODE=1 "${REPO_ROOT}/vfxforth/_install/bin/vfxforth" "include ${VFX_BENCH_FILE}" \
    | sed -e '/^VFX Forth /d' \
          -e '/^© /d' \
          -e '/^[[:space:]]*$/d' \
          -e 's/^Including .*//' \
          -e '/ is redefined/d' \
          -e '/ contains a reference to /d' \
          -e '/^DATA STACK$/d' \
          -e '/^empty stack$/d'
end_ns=$(date +%s%N)
elapsed_us=$(( (end_ns - start_ns) / 1000 ))
echo "Elapsed: ${elapsed_us} usec"
