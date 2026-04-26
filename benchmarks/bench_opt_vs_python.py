"""A/B microbenchmark: ezgatr.nn.functional vs ezgatr.opt for the two primitives.

Mirrors the medium preset shape (batch=4, tokens=256, channels=32, blades=16)
used by benchmarks/benchmark_repo.py. Run pinned to one thread for stable numbers:
    python benchmarks/bench_opt_vs_python.py
"""
import statistics
import time

import torch

from ezgatr.nn.functional import equi_join as join_py
from ezgatr.nn.functional import geometric_product as gp_py
from ezgatr.opt import equi_join as join_cpp
from ezgatr.opt import geometric_product as gp_cpp


torch.set_num_threads(1)
torch.manual_seed(0)
shape = (4, 256, 32, 16)
x = torch.randn(*shape)
y = torch.randn(*shape)
ref = torch.randn(*shape)


def bench(fn, *args, warmup=5, repeats=20, inner=20):
    for _ in range(warmup):
        for _ in range(inner):
            fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        times.append((time.perf_counter() - t0) * 1e3 / inner)
    return statistics.fmean(times), statistics.median(times), min(times)


fmt = "{:32s} mean={:7.3f} ms  p50={:7.3f} ms  min={:7.3f} ms"
for name, fn, args in [
    ("geometric_product (py einsum)", gp_py, (x, y)),
    ("geometric_product (cpp opt)",   gp_cpp, (x, y)),
    ("equi_join no-ref (py einsum)",  join_py, (x, y)),
    ("equi_join no-ref (cpp opt)",    join_cpp, (x, y)),
    ("equi_join with-ref (py einsum)", join_py, (x, y, ref)),
    ("equi_join with-ref (cpp opt)",   join_cpp, (x, y, ref)),
]:
    print(fmt.format(name, *bench(fn, *args)))
