#!/usr/bin/env bash
# Build the extension for the *native* ISA and benchmark the v4 AVX-512 hand-SIMD
# geometric_product / equi_join against v2 (autovectorized) and v3 (AVX2 hand-SIMD).
#
# Run this ON the Tiger Lake i7-1165G7 (where -march=native enables AVX-512):
#     bash benchmarks/run_v4_bench.sh
#
# On a non-AVX-512 host it still runs, but v4 falls back to the AVX2 v3 kernel
# (v4 == v3), so the AVX-512 win is not exercised.
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the project's python (the one that has torch). Adjust if your env differs.
PY="${EZGATR_PYTHON:-python}"

echo "=== host CPU / AVX-512 capability ==="
grep -m1 "model name" /proc/cpuinfo || true
if grep -q avx512f /proc/cpuinfo; then echo "AVX-512: YES (v4 will use the native-512 path)";
else echo "AVX-512: NO  (v4 falls back to AVX2 v3 -- run on the i7-1165G7 for the real comparison)"; fi
echo

echo "=== [1/2] rebuild extension with -march=native ==="
rm -rf build/ src/ezgatr/opt/_opt_ops*.so
"$PY" scripts/build_ext.py > /tmp/v4_build.log 2>&1 || { echo "BUILD FAILED -- see /tmp/v4_build.log"; tail -20 /tmp/v4_build.log; exit 1; }
echo "build ok"
echo

echo "=== [2/2] correctness + benchmark (single thread) ==="
"$PY" benchmarks/bench_v4.py "$@"
