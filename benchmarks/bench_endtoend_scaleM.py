"""Standalone single-core end-to-end benchmark for the MV-only GATr model,
scaling the BATCH dimension B (the "M" axis) with sequence length T fixed.

Companion to bench_endtoend_scaleT.py (which fixes B and sweeps T). Here the
per-sequence structure — attention's O(T^2), channels, heads, layers — is held
constant and only the number of independent sequences B grows. That scales the
activation working set linearly, mirroring bench_pointops_scaleM.py's M axis at
the full-model level: it shows model throughput as the activations sweep the
cache hierarchy, isolated from the O(T^2) attention growth that the scaleT bench
exercises.

Compares the reference MVOnlyGATrModel against ASL-optimized ver_0/1/2/3.
ver_3 is the fully optimized chain (equi_linear v3, rms_norm v3, gelu v3,
fused flash-SDPA attention v3). Weights are copied from the reference into every
variant, so the run doubles as a correctness check (max abs diff printed).

Usage:
    python benchmarks/bench_endtoend_scaleM.py
    python benchmarks/bench_endtoend_scaleM.py --b-values 1 8 64 --T 64 --json-out benchmarks/results/endtoend_scaleM.json
"""
import argparse
import json
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

# Fixed, realistic per-sequence shape. Only B (batch) sweeps; T is held fixed.
T_FIXED = 64
HEADS = 4
CHANNELS = 16
LAYERS = 2

# Tiger Lake cache sizes (per-core) for labelling the activation working set.
L1D_BYTES = 48 * 1024
L2_BYTES = 1280 * 1024
L3_BYTES = 12 * 1024 * 1024

MODEL_VERSIONS = [
    ("ref", MVOnlyGATrModel),
    ("v0", MVOnlyGATrModelASL_ver_0),
    ("v1", MVOnlyGATrModelASL_ver_1),
    ("v2", MVOnlyGATrModelASL_ver_2),
    ("v3", MVOnlyGATrModelASL_ver_3),
]

DEFAULT_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# v0/v1 are ~15-50x slower; don't run them past this B (marked "skip").
SLOW_VERSION_MAX_B = {"v0": 64, "v1": 128}


def make_config(T):
    return MVOnlyGATrConfig(
        num_layers=LAYERS, size_context=T, size_channels_in=1, size_channels_out=1,
        size_channels_hidden=CHANNELS, size_channels_intermediate=CHANNELS,
        attn_num_heads=HEADS, attn_is_causal=False,
    )


def cache_level(footprint_bytes: int) -> str:
    if footprint_bytes <= L1D_BYTES:
        return "L1"
    if footprint_bytes <= L2_BYTES:
        return "L2"
    if footprint_bytes <= L3_BYTES:
        return "L3"
    return "DRAM"


def time_call(fn, *, warmup_budget=0.4, measure_budget=1.2, min_reps=15, max_reps=2000):
    # Generous warmup + many reps: the fast versions (v2/v3) run in only a few ms
    # at small B, so a cold first window (page faults, freq ramp, a transient
    # background process) would otherwise inflate the min. Warm thoroughly, then
    # take the min over enough reps that at least one lands uncontended.
    import time
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
    return best * 1e6  # microseconds


def run(b_values, T):
    print(f"\n=== MV-only GATr end-to-end, scale-M  (T={T}, heads={HEADS}, "
          f"channels={CHANNELS}, layers={LAYERS}, 1 thread) ===")
    header = (f"{'B':>6} {'hidden-act':>11} {'lvl':>4}  "
              + "  ".join(f"{v:>10}" for v, _ in MODEL_VERSIONS) + "   v3/ref  v3/v0   maxdiff")
    print(header)
    print("-" * len(header))

    import gc
    cfg = make_config(T)
    rows = []
    for B in b_values:
        ref = MVOnlyGATrModel(cfg).eval()
        ref_sd = ref.state_dict()
        x = torch.randn(B, T, 1, 16)
        # Hidden-activation footprint per layer: B * T * channels * 16 floats.
        footprint = B * T * CHANNELS * 16 * 4
        lvl = cache_level(footprint)

        # Time each version with only ONE model resident at a time. Keeping all
        # five resident perturbs allocator/cache placement and inflates whichever
        # fast version (v2/v3) lands on a bad address — a harness artifact, not a
        # model property. y_ref is captured once for the correctness check.
        times, max_diff = {}, 0.0
        with torch.inference_mode():
            y_ref = ref(x).clone()
            for vname, cls in MODEL_VERSIONS:
                if B > SLOW_VERSION_MAX_B.get(vname, float("inf")):
                    times[vname] = None
                    continue
                m = ref if vname == "ref" else cls(cfg).eval()
                if vname != "ref":
                    m.load_state_dict(ref_sd)
                    max_diff = max(max_diff, (m(x) - y_ref).abs().max().item())
                times[vname] = time_call(lambda m=m: m(x))
                if vname != "ref":
                    del m
                    gc.collect()

        def fmt(v):
            return f"{times[v]:10.2f}" if times[v] is not None else f"{'skip':>10}"

        s_ref = times["ref"] / times["v3"] if times.get("ref") and times.get("v3") else float("nan")
        s_v0 = times["v0"] / times["v3"] if times.get("v0") and times.get("v3") else float("nan")
        fp_str = f"{footprint/1024:.0f} KB" if footprint < 1 << 20 else f"{footprint/(1<<20):.1f} MB"
        s_ref_str = f"{s_ref:6.2f}x" if s_ref == s_ref else f"{'—':>6}"
        s_v0_str = f"{s_v0:6.2f}x" if s_v0 == s_v0 else f"{'—':>6}"
        print(f"{B:>6} {fp_str:>11} {lvl:>4}  " + "  ".join(fmt(v) for v, _ in MODEL_VERSIONS)
              + f"   {s_ref_str} {s_v0_str}  {max_diff:.1e}")
        rows.append({"B": B, "T": T, "heads": HEADS, "channels": CHANNELS, "layers": LAYERS,
                     "footprint_bytes": footprint, "cache_level": lvl, "times_us": times,
                     "speedup_v3_over_ref": s_ref, "speedup_v3_over_v0": s_v0,
                     "max_abs_diff_vs_ref": max_diff})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-values", type=int, nargs="*", default=DEFAULT_B)
    ap.add_argument("--T", type=int, default=T_FIXED)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    rows = run(args.b_values, args.T)
    out = {"T": args.T, "heads": HEADS, "channels": CHANNELS, "layers": LAYERS,
           "num_threads": 1, "cache_bytes": {"L1D": L1D_BYTES, "L2": L2_BYTES, "L3": L3_BYTES},
           "rows": rows}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
