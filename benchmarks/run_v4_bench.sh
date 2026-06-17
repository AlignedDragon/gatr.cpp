#!/usr/bin/env bash
# Build the extension for the native ISA and benchmark the v4 register-resident
# SoA geometric_product / equi_join against v2 (autovectorized) and v3 (SoA via
# stack buffers). v4 is AVX2 + FMA (the autovectorizer never uses AVX-512 for
# these ops), so the win shows up on any AVX2 host, including the Tiger Lake
# i7-1165G7 and the dev box.
#
#     bash benchmarks/run_v4_bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${EZGATR_PYTHON:-python}"

echo "=== host CPU ==="
grep -m1 "model name" /proc/cpuinfo || true
echo

echo "=== [1/2] rebuild extension with -march=native ==="
rm -rf build/ src/ezgatr/opt/_opt_ops*.so
"$PY" scripts/build_ext.py > /tmp/v4_build.log 2>&1 || { echo "BUILD FAILED -- see /tmp/v4_build.log"; tail -20 /tmp/v4_build.log; exit 1; }
echo "build ok"
echo

echo "=== [2/2] correctness + benchmark (single thread) ==="
"$PY" benchmarks/bench_v4.py "$@"
