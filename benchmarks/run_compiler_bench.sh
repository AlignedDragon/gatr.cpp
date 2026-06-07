#!/usr/bin/env bash
# Compiler comparison: build the extension with a given compiler and record the
# per-function + end-to-end timings (all versions) under LABEL into a shared JSON.
#
#   bash benchmarks/run_compiler_bench.sh gcc   g++     gcc      # run now
#   bash benchmarks/run_compiler_bench.sh clang clang++ clang    # after installing clang
#
# Re-running a label overwrites just that label, so the two series accumulate in
# benchmarks/results/compilers.json and plot_compiler_compare.py compares them.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

LABEL="${1:?usage: run_compiler_bench.sh LABEL CXX CC}"
CXX_BIN="${2:?need a C++ compiler, e.g. g++ or clang++}"
CC_BIN="${3:?need a C compiler, e.g. gcc or clang}"
JSON=benchmarks/results/compilers.json

if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
    echo "error: compiler '$CXX_BIN' not found in PATH" >&2
    exit 1
fi

echo "=== building with CXX=$CXX_BIN CC=$CC_BIN (label=$LABEL) ==="
rm -rf build/ src/ezgatr/opt/_opt_ops*.so
CXX="$CXX_BIN" CC="$CC_BIN" python setup.py build_ext --inplace > "/tmp/compiler_build_${LABEL}.log" 2>&1
python benchmarks/bench_variant.py --label "$LABEL" --n 2 --json-out "$JSON" --append

echo "done -> $JSON  (label=$LABEL)"
