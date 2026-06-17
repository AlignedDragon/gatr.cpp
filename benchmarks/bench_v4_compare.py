"""Benchmark v2 (autovec), v3 (explicit SIMD), and v4 (AVX-512 SoA) for
geometric_product and equi_join across n=1..12.

Saves one JSON per n to benchmarks/results/bench_v4/bench_v4_n{n}.json.
Format matches bench_repo_new_dims so the same plot scripts can read it.

Usage:
    python benchmarks/bench_v4_compare.py [--ns 1 2 3 ...]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ezgatr.opt import (
    geometric_product_v2, geometric_product_v3, geometric_product_v4,
    equi_join_v2, equi_join_v3, equi_join_v4,
)

torch.set_num_threads(1)
OUT = Path("benchmarks/results/bench_v4")
OUT.mkdir(parents=True, exist_ok=True)

WARMUP_S   = 0.3
MEASURE_S  = 1.2
MIN_REPS   = 5
MAX_REPS   = 5000


def best_ms(fn) -> tuple[float, float, float, float]:
    """Return (mean, p50, min, max) in ms over a timed window."""
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    # warmup
    while time.perf_counter() - t0 < WARMUP_S:
        fn()
    reps = max(MIN_REPS, min(MAX_REPS, int(MEASURE_S / single) if single > 0 else MAX_REPS))
    times = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1e3)
    return (statistics.mean(times), statistics.median(times), min(times), max(times))


def make_inputs(n: int):
    B, T, C = 2 * n, 128 * n, 8
    mv  = torch.randn(B, T, C, 16)
    mv2 = torch.randn(B, T, C, 16)
    ref = torch.randn(B, T, C, 16)  # join reference: full multivector (scalar blade used internally)
    return mv, mv2, ref


def bench_n(n: int) -> dict:
    mv, mv2, ref = make_inputs(n)
    cfg = {"batch": 2*n, "tokens": 128*n, "channels": 8, "heads": 4*n}

    targets = [
        ("geometric_product_v2", lambda: geometric_product_v2(mv, mv2)),
        ("geometric_product_v3", lambda: geometric_product_v3(mv, mv2)),
        ("geometric_product_v4", lambda: geometric_product_v4(mv, mv2)),
        ("equi_join_v2",         lambda: equi_join_v2(mv, mv2, ref)),
        ("equi_join_v3",         lambda: equi_join_v3(mv, mv2, ref)),
        ("equi_join_v4",         lambda: equi_join_v4(mv, mv2, ref)),
    ]

    results = []
    for name, fn in targets:
        mean, p50, mn, mx = best_ms(fn)
        results.append({
            "target": name, "n": n, "device": "cpu",
            "warmup": WARMUP_S, "repeats": MIN_REPS, "inner_iters": 1,
            "mean_ms": mean, "p50_ms": p50, "min_ms": mn, "max_ms": mx,
        })
        print(f"  n={n:2d}  {name:30s}  min={mn:8.3f} ms  mean={mean:8.3f} ms")

    return {"n": n, "cfg": cfg, "device": "cpu", "threads": 1,
            "warmup": WARMUP_S, "repeats": MIN_REPS, "inner_iters": 1,
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", type=int, default=list(range(1, 13)))
    args = ap.parse_args()

    for n in args.ns:
        print(f"\n── n={n} ──────────────────────────────────────────")
        payload = bench_n(n)
        out = OUT / f"bench_v4_n{n}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
