"""Standalone single-core benchmark for the point-wise ops:
geometric_product, equi_join, equi_linear, equi_rms_norm, scaler_gated_gelu.

Unlike benchmark_repo.py, this does NOT allocate the attention q/k/v tensors
(which scale as O(n^3) via heads=128n and are the reason the n>=6 runs OOM).
These ops only touch (M, channels, 16) activations (+ small weights), so the
footprint is just the activation working set.

Scaling axis: M = number of multivectors (the flattened batch x tokens leading
dim). channels is held FIXED at a realistic value so weights stay L1-resident
and the only thing sweeping through the cache hierarchy is the activation data.
This isolates streaming throughput and gives a clean L1 -> L2 -> L3 -> DRAM curve.

Usage:
    python benchmarks/bench_pointops_scaleM.py                       # all five ops
    python benchmarks/bench_pointops_scaleM.py --func equi_linear
    python benchmarks/bench_pointops_scaleM.py --json-out benchmarks/results/pointops_scaleM.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ezgatr.opt import (
    equi_linear_ver_0, equi_linear_ver_1, equi_linear_ver_2, equi_linear_ver_3,
    equi_rms_norm_ver_0, equi_rms_norm_ver_1, equi_rms_norm_ver_2, equi_rms_norm_ver_3,
    geometric_product_v0, geometric_product_v1, geometric_product_v2, geometric_product_v3,
    equi_join_v0, equi_join_v1, equi_join_v2, equi_join_v3,
    scaler_gated_gelu_ver_0, scaler_gated_gelu_ver_1, scaler_gated_gelu_ver_2, scaler_gated_gelu_ver_3,
    geometric_bilinear_v3_1,
)

# Tiger Lake i7-1165G7 cache sizes (per-core), for labelling the working set.
L1D_BYTES = 48 * 1024
L2_BYTES = 1280 * 1024
L3_BYTES = 12 * 1024 * 1024

LINEAR_VERSIONS = [
    ("v0", equi_linear_ver_0),
    ("v1", equi_linear_ver_1),
    ("v2", equi_linear_ver_2),
    ("v3", equi_linear_ver_3),
]
RMS_VERSIONS = [
    ("v0", equi_rms_norm_ver_0),
    ("v1", equi_rms_norm_ver_1),
    ("v2", equi_rms_norm_ver_2),
    ("v3", equi_rms_norm_ver_3),
]
GP_VERSIONS = [
    ("v0", geometric_product_v0),
    ("v1", geometric_product_v1),
    ("v2", geometric_product_v2),
    ("v3", geometric_product_v3),
]
JOIN_VERSIONS = [
    ("v0", equi_join_v0),
    ("v1", equi_join_v1),
    ("v2", equi_join_v2),
    ("v3", equi_join_v3),
]
GELU_VERSIONS = [
    ("v0", scaler_gated_gelu_ver_0),
    ("v1", scaler_gated_gelu_ver_1),
    ("v2", scaler_gated_gelu_ver_2),
    ("v3", scaler_gated_gelu_ver_3),
]


def _bilinear_sep(p, ref):
    """Separate-op baseline: split + gp_v3 + join_v3 + torch.cat (what v3 does)."""
    inter = p.shape[-2] // 4
    lg, rg, lj, rj = torch.split(p, inter, dim=-2)
    return torch.cat([geometric_product_v3(lg, rg), equi_join_v3(lj, rj, ref)], dim=-2)


BILINEAR_VERSIONS = [
    ("sep", _bilinear_sep),
    ("v3_1", geometric_bilinear_v3_1),
]

# Default M sweep: 1 KB/multivector-block at channels=16, so this spans
# L1 (48 KB) -> L2 (1.25 MB) -> L3 (12 MB) -> DRAM (256 MB).
DEFAULT_M = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]

# Don't run the unvectorized baselines past this M (they are ~100x slower and
# would dominate wall-clock for no extra insight); they get marked "skipped".
SLOW_VERSION_MAX_M = {"v0": 16384, "v1": 16384}


# Algorithmic FLOP counts per call, matching roofline_costs.estimate_target_cost
# (parameterised by M instead of n) so GFLOP/s here lines up with the roofline.
def linear_flops(M, channels):
    # out_ch * 16 * in_ch * 9 * 2  (mul+add), summed over M multivectors.
    return M * channels * 16 * channels * 9 * 2


def rms_flops(M, channels):
    # 64 flops per multivector (two 16-wide square+accumulate + scale passes).
    return M * channels * 64


def gp_flops(M, channels):       # geometric product (sparse PGA), 384 flops/mv
    return M * channels * 384


def join_flops(M, channels):     # equivariant join (sparse), 384 flops/mv
    return M * channels * 384


def gelu_flops(M, channels):     # scaler-gated gelu, ~128 flops/mv
    return M * channels * 128


def bilinear_flops(M, channels):  # gp (384) + join (384) per inter-channel mv
    return M * channels * 768


def cache_level(footprint_bytes: int) -> str:
    if footprint_bytes <= L1D_BYTES:
        return "L1"
    if footprint_bytes <= L2_BYTES:
        return "L2"
    if footprint_bytes <= L3_BYTES:
        return "L3"
    return "DRAM"


def time_call(fn, *, warmup_budget=0.05, measure_budget=0.2, min_reps=5, max_reps=500):
    """Min-time over adaptive reps. Warmup until ~warmup_budget s, then measure
    until ~measure_budget s elapsed (clamped to [min_reps, max_reps])."""
    # Warmup + single-call cost estimate.
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    while time.perf_counter() - t0 < warmup_budget:
        fn()

    reps = int(measure_budget / single) if single > 0 else max_reps
    reps = max(min_reps, min(max_reps, reps))

    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6, reps  # microseconds, reps used


def make_linear_inputs(M, channels):
    x = torch.randn(M, channels, 16)
    w = torch.randn(channels, channels, 9)
    b = torch.randn(channels)
    return lambda fn: (lambda: fn(x, w, b))


def make_rms_inputs(M, channels):
    x = torch.randn(M, channels, 16)
    w = torch.randn(channels)
    return lambda fn: (lambda: fn(x, w))


def make_gp_inputs(M, channels):
    x = torch.randn(M, channels, 16)
    y = torch.randn(M, channels, 16)
    return lambda fn: (lambda: fn(x, y))


def make_join_inputs(M, channels):
    x = torch.randn(M, channels, 16)
    y = torch.randn(M, channels, 16)
    ref = torch.randn(M, channels, 16)
    return lambda fn: (lambda: fn(x, y, ref))


def make_gelu_inputs(M, channels):
    x = torch.randn(M, channels, 16)
    return lambda fn: (lambda: fn(x, "tanh"))


def make_bilinear_inputs(M, channels):
    # channels = inter; the proj_bil output p has 4*inter channels.
    p = torch.randn(M, 4 * channels, 16)
    ref = torch.randn(M, 1, 16)
    return lambda fn: (lambda: fn(p, ref))


def run_op(op_name, versions, make_inputs, flops_fn, m_values, channels):
    print(f"\n=== {op_name}  (channels={channels}, 1 thread) ===")
    header = f"{'M':>8} {'footprint':>11} {'lvl':>4}  " + "  ".join(f"{v:>10}" for v, _ in versions) + "   v3/v2  v3/v0"
    print(header)
    print("-" * len(header))

    rows = []
    for M in m_values:
        footprint = 2 * M * channels * 16 * 4  # read x + write out
        lvl = cache_level(M * channels * 16 * 4)  # streamed input footprint
        binder = make_inputs(M, channels)
        flops = flops_fn(M, channels)

        times, gflops = {}, {}
        for vname, fn in versions:
            if M > SLOW_VERSION_MAX_M.get(vname, float("inf")):
                times[vname] = None
                gflops[vname] = None
                continue
            us, _ = time_call(binder(fn))
            times[vname] = us
            gflops[vname] = flops / (us * 1e3)  # flops / (us*1e-6) / 1e9

        def fmt(v):
            return f"{times[v]:10.2f}" if times[v] is not None else f"{'skip':>10}"

        s32 = (times["v2"] / times["v3"]) if times.get("v2") and times.get("v3") else float("nan")
        s30 = (times["v0"] / times["v3"]) if times.get("v0") and times.get("v3") else float("nan")
        fp_str = f"{footprint/1024:.0f} KB" if footprint < 1 << 20 else f"{footprint/(1<<20):.1f} MB"
        print(f"{M:>8} {fp_str:>11} {lvl:>4}  " + "  ".join(fmt(v) for v, _ in versions)
              + f"   {s32:5.2f}x  " + (f"{s30:5.1f}x" if s30 == s30 else f"{'—':>6}"))
        rows.append({"M": M, "channels": channels, "footprint_bytes": footprint,
                     "cache_level": lvl, "flops_per_call": flops,
                     "times_us": times, "gflops": gflops,
                     "speedup_v3_over_v2": s32, "speedup_v3_over_v0": s30})
    return rows


# All point-wise op families: name -> (versions, input maker, flop fn).
OPS = {
    "geometric_product": (GP_VERSIONS, make_gp_inputs, gp_flops),
    "equi_join":         (JOIN_VERSIONS, make_join_inputs, join_flops),
    "equi_linear":       (LINEAR_VERSIONS, make_linear_inputs, linear_flops),
    "equi_rms_norm":     (RMS_VERSIONS, make_rms_inputs, rms_flops),
    "scaler_gated_gelu": (GELU_VERSIONS, make_gelu_inputs, gelu_flops),
    "geometric_bilinear": (BILINEAR_VERSIONS, make_bilinear_inputs, bilinear_flops),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", choices=["all", *OPS], default="all")
    ap.add_argument("--channels", type=int, default=16,
                    help="fixed channel count (16 keeps the v3 weight buffer L1-resident on Tiger Lake)")
    ap.add_argument("--m-values", type=int, nargs="*", default=DEFAULT_M)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    out = {"channels": args.channels, "num_threads": 1,
           "cache_bytes": {"L1D": L1D_BYTES, "L2": L2_BYTES, "L3": L3_BYTES}, "ops": {}}
    selected = list(OPS) if args.func == "all" else [args.func]
    for name in selected:
        versions, make_inputs, flops_fn = OPS[name]
        out["ops"][name] = run_op(name, versions, make_inputs, flops_fn,
                                  args.m_values, args.channels)

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
