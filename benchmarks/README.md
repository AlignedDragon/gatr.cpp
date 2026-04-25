## Benchmarks

Lets keep any benchmarking stuff here.

### 1. Runtime measurements

Run one target:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target equi_geometric_attention --preset small
```

Run all current targets:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target all --preset small
```

Some useful options:

```bash
.venv/bin/python benchmarks/benchmark_repo.py --target equi_geometric_attention --preset small --warmup 5 --repeats 20 --inner-iters 100 --threads 1
.venv/bin/python benchmarks/benchmark_repo.py --target mv_only_gatr_model --preset medium --json-out benchmarks/results/model_medium_cpu.json
```

A few notes:

- `inner-iters` repeats the workload inside one timed region so the measurement is less noisy.
- `threads=1` is the default because it is easier to compare runs.
- The full model benchmark uses `MVOnlyGATrModel`.

### 2. perf stat counters

After `perf` is installed on the machine:

```bash
bash benchmarks/run_perf_stat.sh equi_geometric_attention small 5
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
EVENTS="cycles,instructions,LLC-load-misses" bash benchmarks/run_perf_stat.sh geometric_product tiny 5
```