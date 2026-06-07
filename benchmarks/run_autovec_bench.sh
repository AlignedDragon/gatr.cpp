#!/usr/bin/env bash
# Autovectorization comparison (single compiler = GCC):
#   v2-scalar   : built with -fno-tree-vectorize (compiler autovec OFF)
#   v2-autovec  : built with default -O3 (compiler autovec ON)
#   v3          : hand-written AVX2 intrinsics (unaffected by the flag)
#
# A clean rebuild is forced between variants because toggling the env flag does
# not change source mtimes, so build_ext would otherwise skip recompilation.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

JSON=benchmarks/results/autovec.json
rm -f "$JSON"

echo "=== [1/2] build with autovectorization DISABLED (-fno-tree-vectorize) ==="
rm -rf build/ src/ezgatr/opt/_opt_ops*.so
EZGATR_NO_VECTORIZE=1 python setup.py build_ext --inplace > /tmp/autovec_build_novec.log 2>&1
python benchmarks/bench_variant.py --label novec --n 2 --json-out "$JSON" --append

echo "=== [2/2] build with autovectorization ENABLED (default -O3) ==="
rm -rf build/ src/ezgatr/opt/_opt_ops*.so
python setup.py build_ext --inplace > /tmp/autovec_build_autovec.log 2>&1
python benchmarks/bench_variant.py --label autovec --n 2 --json-out "$JSON" --append

echo "done -> $JSON  (the extension is left in the default autovec build)"
