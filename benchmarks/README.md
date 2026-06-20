## Benchmarks

Lets keep any benchmarking stuff here.

### Reproducing our results

All numbers are single-thread with Turbo Boost **off** (to pin the clock), on the
Tiger Lake i7-1165G7. Build, pin, then run:

```bash
.venv/bin/python setup.py build_ext --inplace
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # turbo off

# 1. Full data + plots (~70 min)
bash benchmarks/run_all_benchmarks.sh

# 2. Roofline counters (needs PAPI), n=3 and n=9 — inner-iters auto-size per kernel
.venv/bin/python benchmarks/run_papi_roofline.py --n 3 --threads 1 \
  $(printf -- '--target %s ' geometric_product_v{0,1,2,3} equi_join_v{0,1,2,3} \
    equi_linear_ver_{0,1,2,3} equi_rms_norm_ver_{0,1,2,3} \
    scaler_gated_gelu_ver_{0,1,2,3} equi_geometric_attention_ver_{0,1,2,3}) \
  --json-out benchmarks/results/papi_1thread_n3.json   # repeat with --n 9 for n9.json

# 3. Paper figures (reads the JSON above)
.venv/bin/python benchmarks/plot_paper_figures.py      # -> FINAL_PLOTS/paper_plots/
```

### 1. Runtime measurements

Run one target:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target equi_geometric_attention --n 1
```

Run all current targets:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target all --n 1
```

Some useful options:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target equi_geometric_attention --n 1 --warmup 5 --repeats 20 --inner-iters 100 --threads 1
.venv/bin/python benchmarks/benchmark_repo.py --target mv_only_gatr_model --n 1 --json-out benchmarks/results/model_medium_cpu.json
```

A few notes:

- `inner-iters` repeats the workload inside one timed region so the measurement is less noisy.
- `threads=1` is the default because it is easier to compare runs.
- The full model benchmark uses `MVOnlyGATrModel`.

### 2. perf stat counters

After `perf` is installed on the machine:

```bash
bash benchmarks/run_perf_stat.sh equi_geometric_attention 1 5
```

That runs `perf stat` with a default event set:

- `cycles`
- `instructions`
- `branches`
- `branch-misses`
- `cache-references`
- `cache-misses`
- `page-faults`
- `task-clock`

You can override the event list if needed:

```bash
EVENTS="cycles,instructions,LLC-load-misses" bash benchmarks/run_perf_stat.sh geometric_product 1 5
```

### 3. PAPI roofline measurements

The PAPI runner measures any `benchmark_repo.py` target with hardware counters
and writes JSON that can be plotted as a roofline chart. It expects CPU
execution and the Python PAPI bindings on the benchmark machine:

```bash
module load papi  # if your machine uses environment modules
.venv/bin/python -m pip install pypapi matplotlib
```

Run a small set of final targets:

```bash
.venv/bin/python benchmarks/run_papi_roofline.py \
  --target geometric_product_v2 \
  --target equi_join_v2 \
  --target equi_linear_ver_3 \
  --target equi_rms_norm_ver_3 \
  --target equi_geometric_attention_ver_3 \
  --n 1 \
  --threads 1 \
  --estimate-missing \
  --json-out benchmarks/results/roofline/papi_project_n1.json
```

`inner-iters` now auto-sizes per kernel so each timed region runs
~`--target-region-ms` (default 100); defaults are `--warmup 3 --repeats 20`.
Pass `--inner-iters N` to override.

Plot the roofline after filling in the machine peak values:

```bash
.venv/bin/python benchmarks/plot_papi_roofline.py \
  --json benchmarks/results/roofline/papi_project_n1.json \
  --peak-gflops 80 \
  --peak-bandwidth-gbs 25 \
  --out benchmarks/results/roofline/papi_project_n1.svg
```

Notes:

- The default PAPI events are `PAPI_TOT_CYC`, `PAPI_TOT_INS`,
  `PAPI_FP_OPS`, and `PAPI_L3_TCM`.
- Some CPUs do not expose `PAPI_FP_OPS`. Override events with repeated
  `--event`, for example `--event PAPI_SP_OPS --event PAPI_DP_OPS`.
- `PAPI_L3_TCM * 64` is used as a rough memory-traffic proxy. It is useful for
  comparing implementations, but it is not a perfect DRAM-byte measurement.
- `--estimate-missing` fills missing FLOP/byte values from simple analytical
  cost models so the plot can still be generated while we wait for final
  implementations and stable machine counters.
