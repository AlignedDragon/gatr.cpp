#!/usr/bin/env bash
set -euo pipefail

if ! command -v perf >/dev/null 2>&1; then
  echo "perf is not installed. Install it first, then rerun this script." >&2
  exit 1
fi

TARGET="${1:-equi_geometric_attention}"
N="${2:-1}"
REPEATS="${3:-5}"

EVENTS="${EVENTS:-cycles,instructions,branches,branch-misses,cache-references,cache-misses,page-faults,task-clock}"

.venv/bin/python -c "import torch; print(torch.__version__)" >/dev/null

perf stat -r "${REPEATS}" -e "${EVENTS}" \
  .venv/bin/python benchmarks/benchmark_repo.py \
  --target "${TARGET}" \
  --n "${N}" \
  --warmup 5 \
  --repeats 1 \
  --inner-iters 100 \
  --threads 1
