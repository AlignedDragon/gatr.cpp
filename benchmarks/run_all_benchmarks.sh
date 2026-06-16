#!/usr/bin/env bash
# Full benchmark re-run with turbo boost OFF.
# Includes autovectorization and compiler comparison from kiko's bench_variant workflow.
# Estimated total: ~65-75 min.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FINAL=benchmarks/results/FINAL_PLOTS

echo "=== Turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown) (must be 1) ==="
echo "=== CPUs: $(nproc) ==="
date
echo ""

# ── 1. Version sweep — 1 thread  (~4 min) ─────────────────────────────────────
echo "[1/9] version_sweep_1thread  (n=1..9, threads=1)"
$PY benchmarks/run_project_version_sweep.py \
  --n 1 --n 2 --n 3 --n 4 --n 5 --n 6 --n 7 --n 8 --n 9 \
  --threads 1 \
  --json-out benchmarks/results/version_sweep_1thread.json
echo "DONE version_sweep_1thread"; date

# ── 2. Version sweep — 8 threads  (~2 min) ────────────────────────────────────
echo "[2/9] version_sweep_br48  (n=1..9, threads=8)"
$PY benchmarks/run_project_version_sweep.py \
  --n 1 --n 2 --n 3 --n 4 --n 5 --n 6 --n 7 --n 8 --n 9 \
  --threads 8 \
  --json-out benchmarks/results/version_sweep_br48.json
echo "DONE version_sweep_br48"; date

# ── 3. Legacy per-n bench_repo data (for runtime / roofline plots)  (~4 min) ──
echo "[3/9] gen_legacy_plot_data  (n=1..12)"
$PY benchmarks/gen_legacy_plot_data.py
# Also regenerate bench_repo_new_dims per-n files (explicit targets — build_target
# can't handle v2_x variants returned by get_target_names/--target all)
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
for n in $(seq 1 9); do
  if [ "$n" -le 5 ]; then
    ATTN_ARGS=(--target equi_geometric_attention_ver_0 --target equi_geometric_attention_ver_1 --target equi_geometric_attention_ver_2 --target equi_geometric_attention_ver_3)
  else
    ATTN_ARGS=(--target equi_geometric_attention_ver_3)
  fi
  $PY benchmarks/benchmark_repo.py \
    --n "$n" --threads 1 --warmup 1 --repeats 3 --inner-iters 1 \
    "${NON_ATTN_TARGETS[@]}" "${ATTN_ARGS[@]}" \
    --json-out benchmarks/results/bench_repo_new_dims/bench_repo_n${n}.json
  echo "  bench_repo n=${n} done"
done
echo "DONE gen_legacy + bench_repo_new_dims n=1..9"; date

# ── 4. End-to-end scale-M  (~5 min) ───────────────────────────────────────────
echo "[4/9] bench_endtoend_scaleM"
$PY benchmarks/bench_endtoend_scaleM.py \
  --json-out ${FINAL}/endtoend_scaleM.json
echo "DONE bench_endtoend_scaleM"; date

# ── 5. End-to-end scale-T  (~8 min) ───────────────────────────────────────────
echo "[5/9] bench_endtoend_scaleT"
$PY benchmarks/bench_endtoend_scaleT.py \
  --json-out ${FINAL}/endtoend_scaleT.json
echo "DONE bench_endtoend_scaleT"; date

# ── 6. Point-ops scale-M  (~3 min) ────────────────────────────────────────────
echo "[6/9] bench_pointops_scaleM"
$PY benchmarks/bench_pointops_scaleM.py \
  --json-out ${FINAL}/pointops_scaleM.json
echo "DONE bench_pointops_scaleM"; date

# ── 7. Full per-function run  (~15 min, skip endtoend — those are done above) ─
echo "[7/9] run_full_bench (per-function + attention, Python timings)"
$PY benchmarks/run_full_bench.py \
  --skip-endtoend \
  --out-dir benchmarks/results/run_20260615
echo "DONE run_full_bench"; date

# ── 8. Autovectorization comparison  (~10 min including two builds) ────────────
echo "[8/9] autovec comparison (novec build → autovec build)"
# 8a. Build WITHOUT auto-vectorization (v2 kernels get scalar codegen)
echo "  [8a] building novec extension..."
EZGATR_NO_VECTORIZE=1 $PY setup.py build_ext --inplace 2>&1 | grep -E "ninja|Compiling|copying|error" || true
$PY benchmarks/bench_variant.py \
  --label novec --n 3 \
  --json-out benchmarks/results/autovec.json
echo "  [8a] novec done"
# 8b. Normal GCC build (auto-vec ON) — also used as gcc label for compiler compare
echo "  [8b] building normal GCC extension..."
$PY setup.py build_ext --inplace 2>&1 | grep -E "ninja|Compiling|copying|error" || true
$PY benchmarks/bench_variant.py \
  --label autovec --n 3 \
  --json-out benchmarks/results/autovec.json --append
echo "  [8b] autovec label done"
echo "DONE autovec comparison"; date

# ── 9. Compiler comparison  (~10 min including clang build) ───────────────────
echo "[9/9] compiler comparison (gcc vs clang)"
# 9a. GCC (already built in step 8b) — record as gcc label
$PY benchmarks/bench_variant.py \
  --label gcc --n 3 \
  --json-out benchmarks/results/compilers.json
echo "  [9a] gcc label done"
# 9b. Clang build
echo "  [9b] building clang extension..."
CXX=clang++ CC=clang $PY setup.py build_ext --inplace 2>&1 | grep -E "ninja|Compiling|copying|error" || true
$PY benchmarks/bench_variant.py \
  --label clang --n 3 \
  --json-out benchmarks/results/compilers.json --append
echo "  [9b] clang label done"
# Restore normal GCC build
echo "  restoring GCC extension..."
$PY setup.py build_ext --inplace 2>&1 | grep -E "ninja|Compiling|copying|error" || true
echo "DONE compiler comparison"; date

# ── Generate all plots ─────────────────────────────────────────────────────────
echo ""
echo "=== GENERATING PLOTS ==="

# Sweep plots (new, from kiko's branch)
$PY benchmarks/plot_version_sweep_1thread.py
$PY benchmarks/plot_version_sweep_br48.py

# Runtime + speedup bars
$PY benchmarks/plot_runtime.py

# Roofline: all versions (PAPI-anchored)
$PY benchmarks/plot_roofline.py


# Final best config roofline
$PY benchmarks/plot_final_best_roofline.py \
  $(for n in $(seq 1 9); do echo "--runtime-json benchmarks/results/bench_repo_new_dims/bench_repo_n${n}.json"; done) \
  --peak-gflops 89.6 --peak-bandwidth-gbs 45.0 \
  --out-prefix ${FINAL}/roofline_final_best

# Roofline vs Python
$PY benchmarks/plot_roofline_vs_python.py

# End-to-end scaling
$PY benchmarks/plot_endtoend_scaleM.py ${FINAL}/endtoend_scaleM.json ${FINAL}
$PY benchmarks/plot_endtoend_scaleT.py ${FINAL}/endtoend_scaleT.json ${FINAL}

# Point-ops per-M
$PY benchmarks/plot_pointops_scaleM.py ${FINAL}/pointops_scaleM.json ${FINAL}

# Autovec comparison (new)
$PY benchmarks/plot_autovec.py benchmarks/results/autovec.json ${FINAL}

# Compiler comparison (new)
$PY benchmarks/plot_compiler_compare.py benchmarks/results/compilers.json ${FINAL}

# Attention diagrams (static, no data needed)
$PY benchmarks/plot_attention_diagrams.py

# ── Re-run step 1 with extended v0/v1 caps (n=5) ─────────────────────────────
# Step 1 ran before the cap was raised; redo it now so 1-thread sweep matches.
echo "[re-run 1] version_sweep_1thread with v0/v1 extended to n=5"
$PY benchmarks/run_project_version_sweep.py \
  --n 1 --n 2 --n 3 --n 4 --n 5 --n 6 --n 7 --n 8 --n 9 \
  --threads 1 \
  --json-out benchmarks/results/version_sweep_1thread.json
echo "DONE re-run version_sweep_1thread"; date

echo ""
echo "=== ALL DONE ==="
date
echo "Results in ${FINAL}"
ls ${FINAL}/*.png 2>/dev/null | wc -l && echo "PNG files in FINAL_PLOTS"
