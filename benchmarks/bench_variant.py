"""Measure per-function ops (v0-v3) + end-to-end model (ref, v0-v3) for the
*currently built* _opt_ops extension, tagging the results with a build --label.

Both the compiler-comparison and autovectorization benchmarks work the same way:
build the extension a particular way (different CXX, or with/without
-fno-tree-vectorize), then run this script with a label to append that build's
timings to a shared JSON. The plot scripts then compare labels.

    # autovectorization example
    EZGATR_NO_VECTORIZE=1 python setup.py build_ext --inplace
    python benchmarks/bench_variant.py --label v2_scalar  --json-out results/autovec.json
    python setup.py build_ext --inplace
    python benchmarks/bench_variant.py --label autovec    --json-out results/autovec.json --append

    # compiler example
    CXX=clang++ CC=clang python setup.py build_ext --inplace
    python benchmarks/bench_variant.py --label clang --json-out results/compilers.json --append
"""
import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_repo as br  # noqa: E402
from ezgatr.nets.mv_only_gatr import (  # noqa: E402
    MVOnlyGATrConfig, MVOnlyGATrModel,
    MVOnlyGATrModelASL_ver_0, MVOnlyGATrModelASL_ver_1,
    MVOnlyGATrModelASL_ver_2, MVOnlyGATrModelASL_ver_3,
    MVOnlyGATrModelASL_ver_3_1, MVOnlyGATrModelASL_ver_3_2,
)

torch.set_num_threads(1)
device = torch.device("cpu")

# Per-function families: display name -> {version: benchmark_repo target}.
FAMILIES = {
    "geometric_product": {"v0": "geometric_product_v0", "v1": "geometric_product_v1",
                          "v2": "geometric_product_v2", "v3": "geometric_product_v3"},
    "equi_join": {"v0": "equi_join_v0", "v1": "equi_join_v1",
                  "v2": "equi_join_v2", "v3": "equi_join_v3"},
    "equi_linear": {"v0": "equi_linear_ver_0", "v1": "equi_linear_ver_1",
                    "v2": "equi_linear_ver_2", "v3": "equi_linear_ver_3"},
    "equi_rms_norm": {"v0": "equi_rms_norm_ver_0", "v1": "equi_rms_norm_ver_1",
                      "v2": "equi_rms_norm_ver_2", "v3": "equi_rms_norm_ver_3"},
    "scaler_gated_gelu": {"v0": "scaler_gated_gelu_ver_0", "v1": "scaler_gated_gelu_ver_1",
                          "v2": "scaler_gated_gelu_ver_2", "v3": "scaler_gated_gelu_ver_3"},
    "equi_geometric_attention": {"v0": "equi_geometric_attention_ver_0",
                                 "v1": "equi_geometric_attention_ver_1",
                                 "v2": "equi_geometric_attention_ver_2",
                                 "v3": "equi_geometric_attention_ver_3",
                                 "v3_2": "equi_geometric_attention_ver_3_2"},
}
MODEL_VERSIONS = {
    "ref": MVOnlyGATrModel, "v0": MVOnlyGATrModelASL_ver_0, "v1": MVOnlyGATrModelASL_ver_1,
    "v2": MVOnlyGATrModelASL_ver_2, "v3": MVOnlyGATrModelASL_ver_3,
    "v3_1": MVOnlyGATrModelASL_ver_3_1, "v3_2": MVOnlyGATrModelASL_ver_3_2,
}

# Cache benchmark_repo inputs per n (avoids re-allocating attention tensors per target).
_orig_make_inputs = br.make_inputs
_cache: dict = {}
def _cached(dev, n):
    if n not in _cache:
        _cache[n] = _orig_make_inputs(dev, n)
    return _cache[n]
br.make_inputs = _cached


def time_call(fn, warmup_budget=0.1, measure_budget=0.4, min_reps=3, max_reps=200):
    t0 = time.perf_counter(); fn(); single = time.perf_counter() - t0
    while time.perf_counter() - t0 < warmup_budget:
        fn()
    reps = max(min_reps, min(max_reps, int(measure_budget / single) if single > 0 else max_reps))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def measure_ops(n):
    out = {}
    for fam, vers in FAMILIES.items():
        attn = "attention" in fam
        w, r, i = (1, 3, 1) if attn else (2, 6, 6)
        out[fam] = {v: br.measure_target(tgt, device=device, n=n,
                                         warmup=w, repeats=r, inner_iters=i)["min_ms"]
                    for v, tgt in vers.items()}
    return out


def measure_model(B, T):
    cfg = MVOnlyGATrConfig(num_layers=2, size_context=T, size_channels_in=1, size_channels_out=1,
                           size_channels_hidden=16, size_channels_intermediate=16,
                           attn_num_heads=4, attn_is_causal=False)
    import gc
    ref = MVOnlyGATrModel(cfg).eval()
    sd = ref.state_dict()
    x = torch.randn(B, T, 1, 16)
    out = {}
    with torch.inference_mode():
        # Only one model resident at a time — keeping all of them resident
        # perturbs allocator/cache placement and inflates the fast versions.
        for name, cls in MODEL_VERSIONS.items():
            m = ref if name == "ref" else cls(cfg).eval()
            if name != "ref":
                m.load_state_dict(sd)
            out[name] = time_call(lambda m=m: m(x))
            if name != "ref":
                del m
                gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="build-variant label (e.g. v2_scalar, autovec, gcc, clang)")
    ap.add_argument("--n", type=int, default=2, help="problem size for the per-function ops")
    ap.add_argument("--model-batch", type=int, default=8)
    ap.add_argument("--model-T", type=int, default=64)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--append", action="store_true", help="merge into an existing json instead of overwriting")
    args = ap.parse_args()

    print(f"[{args.label}] measuring ops (n={args.n}) ...", flush=True)
    ops = measure_ops(args.n)
    print(f"[{args.label}] measuring end-to-end model (B={args.model_batch}, T={args.model_T}) ...", flush=True)
    model = measure_model(args.model_batch, args.model_T)

    p = Path(args.json_out)
    payload = {}
    if args.append and p.exists():
        payload = json.loads(p.read_text())
    payload.setdefault("labels", {})
    payload["labels"][args.label] = {
        "n": args.n, "model_batch": args.model_batch, "model_T": args.model_T,
        "ops_ms": ops, "model_ms": model,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"[{args.label}] wrote {p}", flush=True)
    for fam, vers in ops.items():
        print(f"  {fam:28s} " + "  ".join(f"{v}={vers[v]:.3f}" for v in ("v0", "v1", "v2", "v3")))
    print("  model " + "  ".join(f"{k}={model[k]:.1f}ms" for k in ("ref", "v0", "v1", "v2", "v3")))


if __name__ == "__main__":
    main()
