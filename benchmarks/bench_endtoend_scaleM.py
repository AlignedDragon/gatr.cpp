"""Standalone single-core end-to-end benchmark for the MV-only GATr model,
sweeping the problem-size parameter n with the standard formula B=2n, H=4n,
T=128n, C=8 (same formula used everywhere in this benchmark suite).

Compares the reference MVOnlyGATrModel against ASL-optimized ver_0/1/2/3.
ver_3 is the fully optimized chain. Weights are copied from the reference into
every variant so the run doubles as a correctness check (max abs diff printed).

Usage:
    python benchmarks/bench_endtoend_scaleM.py
    python benchmarks/bench_endtoend_scaleM.py --n-values 1 2 3 4 5 --json-out results/endtoend_scaleM.json
"""
import argparse
import json
import gc
import time
from pathlib import Path

import torch

from ezgatr.nets.mv_only_gatr import (
    MVOnlyGATrConfig,
    MVOnlyGATrModel,
    MVOnlyGATrModelASL_ver_0,
    MVOnlyGATrModelASL_ver_1,
    MVOnlyGATrModelASL_ver_2,
    MVOnlyGATrModelASL_ver_3,
)

LAYERS = 2
L1D_BYTES = 48 * 1024
L2_BYTES = 1280 * 1024
L3_BYTES = 12 * 1024 * 1024

MODEL_VERSIONS = [
    ("ref", MVOnlyGATrModel),
    ("v0",  MVOnlyGATrModelASL_ver_0),
    ("v1",  MVOnlyGATrModelASL_ver_1),
    ("v2",  MVOnlyGATrModelASL_ver_2),
    ("v3",  MVOnlyGATrModelASL_ver_3),
]

# v0/v1 OOM / too slow past n=3 (T=384); v2 past n=5 (T=640); ref uses Python attention so same limit
SLOW_VERSION_MAX_N = {"ref": 3, "v0": 3, "v1": 3, "v2": 5}

DEFAULT_N = list(range(1, 10))  # n=1..9 (attention v3 measured to n=9)


def make_config(n: int) -> MVOnlyGATrConfig:
    B, T, H, C = 2 * n, 128 * n, 4 * n, 8
    return MVOnlyGATrConfig(
        num_layers=LAYERS, size_context=T, size_channels_in=1, size_channels_out=1,
        size_channels_hidden=C, size_channels_intermediate=C,
        attn_num_heads=H, attn_is_causal=False,
    ), B, T, H, C


def cache_level(footprint_bytes: int) -> str:
    if footprint_bytes <= L1D_BYTES:   return "L1"
    if footprint_bytes <= L2_BYTES:    return "L2"
    if footprint_bytes <= L3_BYTES:    return "L3"
    return "DRAM"


def time_call(fn, *, warmup_budget=0.3, measure_budget=1.0, min_reps=5, max_reps=500):
    t0 = time.perf_counter(); fn(); single = time.perf_counter() - t0
    while time.perf_counter() - t0 < warmup_budget: fn()
    reps = max(min_reps, min(max_reps, int(measure_budget / single) if single > 0 else max_reps))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e6  # microseconds


def run(n_values):
    print(f"\n=== MV-only GATr end-to-end  (B=2n H=4n T=128n C=8, layers={LAYERS}, 1 thread) ===")
    header = (f"{'n':>4} {'B':>4} {'T':>5} {'H':>5}  "
              + "  ".join(f"{v:>10}" for v, _ in MODEL_VERSIONS) + "   v3/ref  v3/v0   maxdiff")
    print(header)
    print("-" * len(header))

    rows = []
    for n in n_values:
        cfg, B, T, H, C = make_config(n)
        x = torch.randn(B, T, 1, 16)
        footprint = B * T * C * 16 * 4
        lvl = cache_level(footprint)

        ref = MVOnlyGATrModel(cfg).eval()
        ref_sd = ref.state_dict()
        times, max_diff = {}, 0.0
        with torch.inference_mode():
            y_ref = ref(x).clone()
            for vname, cls in MODEL_VERSIONS:
                if n > SLOW_VERSION_MAX_N.get(vname, float("inf")):
                    times[vname] = None
                    continue
                m = ref if vname == "ref" else cls(cfg).eval()
                if vname != "ref":
                    m.load_state_dict(ref_sd)
                    max_diff = max(max_diff, (m(x) - y_ref).abs().max().item())
                times[vname] = time_call(lambda m=m: m(x))
                if vname != "ref":
                    del m; gc.collect()

        def fmt(v):
            return f"{times[v]:10.2f}" if times[v] is not None else f"{'skip':>10}"

        s_ref = times["ref"] / times["v3"] if times.get("ref") and times.get("v3") else float("nan")
        s_v0  = times["v0"]  / times["v3"] if times.get("v0")  and times.get("v3") else float("nan")
        fp_str = f"{footprint/1024:.0f} KB" if footprint < 1<<20 else f"{footprint/(1<<20):.1f} MB"
        print(f"{n:>4} {B:>4} {T:>5} {H:>5}  " + "  ".join(fmt(v) for v, _ in MODEL_VERSIONS)
              + f"   {s_ref:6.2f}x {s_v0:6.2f}x  {max_diff:.1e}")
        rows.append({"n": n, "B": B, "T": T, "heads": H, "channels": C, "layers": LAYERS,
                     "footprint_bytes": footprint, "cache_level": lvl, "times_us": times,
                     "speedup_v3_over_ref": s_ref, "speedup_v3_over_v0": s_v0,
                     "max_abs_diff_vs_ref": max_diff})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="*", default=DEFAULT_N)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    rows = run(args.n_values)
    out = {"dim_formula": "B=2n,H=4n,T=128n,C=8", "layers": LAYERS, "num_threads": 1,
           "cache_bytes": {"L1D": L1D_BYTES, "L2": L2_BYTES, "L3": L3_BYTES}, "rows": rows}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
