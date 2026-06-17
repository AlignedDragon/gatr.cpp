"""Benchmark v2 kernels compiled WITHOUT auto-vectorization.

Build first with:
    EZGATR_NO_VECTORIZE=1 python setup.py build_ext --inplace

Then run:
    python benchmarks/bench_novector_v2.py

Saves to benchmarks/results/bench_novector/bench_novector_n{n}.json.
Format matches bench_repo_new_dims.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ezgatr.opt import (
    geometric_product_v2,
    equi_join_v2,
    equi_linear_ver_2,
    equi_rms_norm_ver_2,
    scaler_gated_gelu_ver_2,
    equi_geometric_attention_ver_2,
)

torch.set_num_threads(1)
OUT = Path("benchmarks/results/bench_novector")
OUT.mkdir(parents=True, exist_ok=True)

WARMUP_S  = 0.3
MEASURE_S = 1.2
MIN_REPS  = 5
MAX_REPS  = 5000


def best_ms(fn):
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    while time.perf_counter() - t0 < WARMUP_S:
        fn()
    reps = max(MIN_REPS, min(MAX_REPS, int(MEASURE_S / single) if single > 0 else MAX_REPS))
    times = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1e3)
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def make_inputs(n: int):
    B, T, C, H = 2*n, 128*n, 8, 4*n
    mv  = torch.randn(B, T, C, 16)
    mv2 = torch.randn(B, T, C, 16)
    ref = torch.randn(B, T, C, 16)  # join reference: full multivector
    w   = torch.randn(C, C, 9)
    b   = torch.randn(C)
    q   = torch.randn(B, H, T, C, 16)
    k   = torch.randn(B, H, T, C, 16)
    v   = torch.randn(B, H, T, C, 16)
    return mv, mv2, ref, w, b, q, k, v


def bench_n(n: int) -> dict:
    mv, mv2, ref, w, b, q, k, v = make_inputs(n)
    cfg = {"batch": 2*n, "tokens": 128*n, "channels": 8, "heads": 4*n}

    targets = [
        ("geometric_product_v2",           lambda: geometric_product_v2(mv, mv2)),
        ("equi_join_v2",                   lambda: equi_join_v2(mv, mv2, ref)),
        ("equi_linear_ver_2",              lambda: equi_linear_ver_2(mv, w, b, False)),
        ("equi_rms_norm_ver_2",            lambda: equi_rms_norm_ver_2(mv, None, None)),
        ("scaler_gated_gelu_ver_2",        lambda: scaler_gated_gelu_ver_2(mv, "tanh")),
        ("equi_geometric_attention_ver_2", lambda: equi_geometric_attention_ver_2(
            q, k, v, kinds={"ipa": None, "daa": None}, is_causal=False)),
    ]

    results = []
    for name, fn in targets:
        mean, p50, mn, mx = best_ms(fn)
        results.append({
            "target": name, "n": n, "device": "cpu",
            "warmup": WARMUP_S, "repeats": MIN_REPS, "inner_iters": 1,
            "mean_ms": mean, "p50_ms": p50, "min_ms": mn, "max_ms": mx,
        })
        print(f"  n={n:2d}  {name:40s}  min={mn:8.3f} ms  mean={mean:8.3f} ms")

    return {"n": n, "cfg": cfg, "device": "cpu", "threads": 1,
            "warmup": WARMUP_S, "repeats": MIN_REPS, "inner_iters": 1,
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", type=int, default=list(range(1, 13)))
    args = ap.parse_args()

    for n in args.ns:
        print(f"\n── n={n} (no-vectorize v2) ────────────────────────")
        payload = bench_n(n)
        out = OUT / f"bench_novector_n{n}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
