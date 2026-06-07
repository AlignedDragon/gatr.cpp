"""Standalone single-core end-to-end benchmark for the MV-only GATr model.

Companion to bench_pointops_scaleM.py. Where that isolates individual kernels,
this times the full model forward pass: embedding -> N x (attention + bilinear +
MLP) -> head. It compares the reference MVOnlyGATrModel against the ASL-optimized
variants ver_0/1/2/3. ver_3 is the fully optimized chain: equi_linear v3,
rms_norm v3, gelu v3, and the fused flash-SDPA attention v3.

Scaling axis: sequence length T (size_context). Attention is O(T^2), so this is
the canonical transformer axis and the one that exercises the fused flash-SDPA
attention most. batch / heads / channels / layers are held FIXED at realistic
values so memory stays controlled (the original benchmark_repo coupled
heads=128n, which is what made n>=6 OOM).

Weights are copied from the reference into every variant, so the run doubles as
a correctness check (max abs diff printed) on top of the timing.

Usage:
    python benchmarks/bench_endtoend_scaleT.py
    python benchmarks/bench_endtoend_scaleT.py --t-values 16 64 256 --json-out benchmarks/results/endtoend.json
"""
import argparse
import json
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

# Fixed, realistic model shape. Only T (sequence length) sweeps.
BATCH = 8
HEADS = 4
CHANNELS = 16
LAYERS = 2

MODEL_VERSIONS = [
    ("ref", MVOnlyGATrModel),
    ("v0", MVOnlyGATrModelASL_ver_0),
    ("v1", MVOnlyGATrModelASL_ver_1),
    ("v2", MVOnlyGATrModelASL_ver_2),
    ("v3", MVOnlyGATrModelASL_ver_3),
]

DEFAULT_T = [16, 32, 64, 128, 256, 512]


def make_config(T):
    return MVOnlyGATrConfig(
        num_layers=LAYERS,
        size_context=T,
        size_channels_in=1,
        size_channels_out=1,
        size_channels_hidden=CHANNELS,
        size_channels_intermediate=CHANNELS,
        attn_num_heads=HEADS,
        attn_is_causal=False,
    )


def time_call(fn, *, warmup_budget=0.4, measure_budget=1.2, min_reps=15, max_reps=2000):
    """Min-time over adaptive reps. Generous warmup so a cold first window (page
    faults, freq ramp, transient load) doesn't inflate the short fast-version
    runtimes; min over many reps so at least one lands uncontended."""
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


def run(t_values):
    print(f"\n=== MV-only GATr end-to-end  (batch={BATCH}, heads={HEADS}, "
          f"channels={CHANNELS}, layers={LAYERS}, 1 thread) ===")
    header = f"{'T':>6}  " + "  ".join(f"{v:>10}" for v, _ in MODEL_VERSIONS) + "   v3/ref  v3/v0   maxdiff"
    print(header)
    print("-" * len(header))

    import gc
    rows = []
    for T in t_values:
        cfg = make_config(T)
        ref = MVOnlyGATrModel(cfg).eval()
        ref_sd = ref.state_dict()
        x = torch.randn(BATCH, T, 1, 16)

        # Time each version with only ONE model resident at a time. Keeping all
        # five resident perturbs allocator/cache placement and inflates whichever
        # fast version (v2/v3) lands on a bad address — a harness artifact, not a
        # model property. y_ref is captured once for the correctness check.
        times, max_diff = {}, 0.0
        with torch.inference_mode():
            y_ref = ref(x).clone()
            for vname, cls in MODEL_VERSIONS:
                m = ref if vname == "ref" else cls(cfg).eval()
                if vname != "ref":
                    m.load_state_dict(ref_sd)  # share weights -> fair timing + correctness
                    max_diff = max(max_diff, (m(x) - y_ref).abs().max().item())
                times[vname] = time_call(lambda m=m: m(x))
                if vname != "ref":
                    del m
                    gc.collect()

        s_ref = times["ref"] / times["v3"]
        s_v0 = times["v0"] / times["v3"]
        print(f"{T:>6}  " + "  ".join(f"{times[v]:10.2f}" for v, _ in MODEL_VERSIONS)
              + f"   {s_ref:6.2f}x {s_v0:6.2f}x  {max_diff:.1e}")
        rows.append({"T": T, "batch": BATCH, "heads": HEADS, "channels": CHANNELS,
                     "layers": LAYERS, "times_us": times,
                     "speedup_v3_over_ref": s_ref, "speedup_v3_over_v0": s_v0,
                     "speedup_v2_over_ref": times["ref"] / times["v2"],
                     "max_abs_diff_vs_ref": max_diff})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-values", type=int, nargs="*", default=DEFAULT_T)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    rows = run(args.t_values)
    out = {"batch": BATCH, "heads": HEADS, "channels": CHANNELS, "layers": LAYERS,
           "num_threads": 1, "rows": rows}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
