#!/usr/bin/env bash
# Resume from where run_all_benchmarks.sh failed (bench_repo_new_dims loop).
# Steps 1, 2, gen_legacy_plot_data already completed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FINAL=benchmarks/results/FINAL_PLOTS

echo "=== Turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown) (must be 1) ==="
date

# Explicit non-attention targets (avoids the v2_x targets build_target can't handle)
NON_ATTN_TARGETS=(
  --target geometric_product_v0 --target geometric_product_v1
  --target geometric_product_v2 --target geometric_product_v3
  --target equi_join_v0 --target equi_join_v1
  --target equi_join_v2 --target equi_join_v3
  --target equi_linear_ver_0 --target equi_linear_ver_1
  --target equi_linear_ver_2 --target equi_linear_ver_3
  --target equi_rms_norm_ver_0 --target equi_rms_norm_ver_1
  --target equi_rms_norm_ver_2 --target equi_rms_norm_ver_3
  --target scaler_gated_gelu_ver_0 --target scaler_gated_gelu_ver_1
  --target scaler_gated_gelu_ver_2 --target scaler_gated_gelu_ver_3
)
ATTN_ALL=(
  --target equi_geometric_attention_ver_0 --target equi_geometric_attention_ver_1
  --target equi_geometric_attention_ver_2 --target equi_geometric_attention_ver_3
)
ATTN_V3=(
  --target equi_geometric_attention_ver_3
)

# ── bench_repo_new_dims n=1..9 (explicit targets, attention caps)  (~8 min) ───
echo "[3b] bench_repo_new_dims n=1..9 (explicit targets)"
for n in $(seq 1 9); do
  if [ "$n" -le 5 ]; then
    ATTN_ARGS=("${ATTN_ALL[@]}")
  else
    ATTN_ARGS=("${ATTN_V3[@]}")
  fi
  $PY benchmarks/benchmark_repo.py \
    --n "$n" --threads 1 --warmup 1 --repeats 3 --inner-iters 1 \
    "${NON_ATTN_TARGETS[@]}" "${ATTN_ARGS[@]}" \
    --json-out benchmarks/results/bench_repo_new_dims/bench_repo_n${n}.json
  echo "  bench_repo n=${n} done"
done
echo "DONE bench_repo_new_dims"; date

# ── 4. End-to-end scale-M  (~5 min) ───────────────────────────────────────────
echo "[4] bench_endtoend_scaleM"
$PY benchmarks/bench_endtoend_scaleM.py \
  --json-out ${FINAL}/endtoend_scaleM.json
echo "DONE bench_endtoend_scaleM"; date

# ── 5. End-to-end scale-T  (~8 min) ───────────────────────────────────────────
echo "[5] bench_endtoend_scaleT"
$PY benchmarks/bench_endtoend_scaleT.py \
  --json-out ${FINAL}/endtoend_scaleT.json
echo "DONE bench_endtoend_scaleT"; date

# ── 6. Point-ops scale-M  (~3 min) ────────────────────────────────────────────
echo "[6] bench_pointops_scaleM"
$PY benchmarks/bench_pointops_scaleM.py \
  --json-out ${FINAL}/pointops_scaleM.json
echo "DONE bench_pointops_scaleM"; date

# ── 7. Full per-function run  (~20 min, Python timings for roofline_vs_python) ─
echo "[7] run_full_bench (per-function + attention)"
$PY benchmarks/run_full_bench.py \
  --skip-endtoend \
  --out-dir benchmarks/results/run_20260615
echo "DONE run_full_bench"; date

# ── 8. Autovectorization comparison  (~10 min) ────────────────────────────────
echo "[8] autovec comparison"
echo "  [8a] building novec extension..."
EZGATR_NO_VECTORIZE=1 $PY setup.py build_ext --inplace 2>&1 | grep -E "Compiling|copying|error:" || true
$PY benchmarks/bench_variant.py \
  --label novec --n 3 \
  --json-out benchmarks/results/autovec.json
echo "  [8a] novec done"
echo "  [8b] building normal GCC extension..."
$PY setup.py build_ext --inplace 2>&1 | grep -E "Compiling|copying|error:" || true
$PY benchmarks/bench_variant.py \
  --label autovec --n 3 \
  --json-out benchmarks/results/autovec.json --append
echo "DONE autovec comparison"; date

# ── 9. Compiler comparison  (~10 min) ─────────────────────────────────────────
echo "[9] compiler comparison"
$PY benchmarks/bench_variant.py \
  --label gcc --n 3 \
  --json-out benchmarks/results/compilers.json
echo "  [9a] gcc done"
echo "  [9b] building clang extension..."
CXX=clang++ CC=clang $PY setup.py build_ext --inplace 2>&1 | grep -E "Compiling|copying|error:" || true
$PY benchmarks/bench_variant.py \
  --label clang --n 3 \
  --json-out benchmarks/results/compilers.json --append
echo "  [9b] clang done"
echo "  restoring GCC extension..."
$PY setup.py build_ext --inplace 2>&1 | grep -E "Compiling|copying|error:" || true
echo "DONE compiler comparison"; date

# ── Re-run 1-thread sweep with extended v0/v1 caps  (~11 min) ─────────────────
echo "[re-run 1] version_sweep_1thread with v0/v1 up to n=5"
$PY benchmarks/run_project_version_sweep.py \
  --n 1 --n 2 --n 3 --n 4 --n 5 --n 6 --n 7 --n 8 --n 9 \
  --threads 1 \
  --json-out benchmarks/results/version_sweep_1thread.json
echo "DONE re-run version_sweep_1thread"; date

# ── Generate all plots ─────────────────────────────────────────────────────────
echo ""
echo "=== GENERATING PLOTS ==="

$PY benchmarks/plot_version_sweep_1thread.py
$PY benchmarks/plot_version_sweep_br48.py
$PY benchmarks/plot_runtime.py
$PY benchmarks/plot_roofline.py

$PY benchmarks/plot_project_versions_roofline.py \
  $(for n in $(seq 1 9); do echo "--runtime-json benchmarks/results/bench_repo_new_dims/bench_repo_n${n}.json"; done) \
  --peak-gflops 89.6 --peak-bandwidth-gbs 45.0 \
  --out-prefix ${FINAL}/roofline_project_versions

$PY benchmarks/plot_final_best_roofline.py \
  $(for n in $(seq 1 9); do echo "--runtime-json benchmarks/results/bench_repo_new_dims/bench_repo_n${n}.json"; done) \
  --peak-gflops 89.6 --peak-bandwidth-gbs 45.0 \
  --out-prefix ${FINAL}/roofline_final_best

$PY benchmarks/plot_roofline_vs_python.py

$PY benchmarks/plot_endtoend_scaleM.py ${FINAL}/endtoend_scaleM.json ${FINAL}
$PY benchmarks/plot_endtoend_scaleT.py ${FINAL}/endtoend_scaleT.json ${FINAL}
$PY benchmarks/plot_pointops_scaleM.py ${FINAL}/pointops_scaleM.json ${FINAL}

$PY benchmarks/plot_autovec.py benchmarks/results/autovec.json ${FINAL}
$PY benchmarks/plot_compiler_compare.py benchmarks/results/compilers.json ${FINAL}

$PY benchmarks/plot_attention_diagrams.py

echo ""
echo "=== ALL DONE ==="
date
echo "PNG files in FINAL_PLOTS: $(ls ${FINAL}/*.png 2>/dev/null | wc -l)"
