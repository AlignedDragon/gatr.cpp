"""Full benchmark suite with new dimension formula B=2n, H=4n, T=128n, C=8.

This avoids the O(n^3) memory blowup of the old formula (heads=128n was the killer).
Point-wise ops run up to n=12; attention versions each run to their practical max:
  v0/v1 : n=1..4  (un-optimised, gets slow fast)
  v2    : n=1..5
  v3    : n=1..7
  v3_1  : n=1..9  (flash / hoisted-K variant)
  Python : n=1..4

Usage
-----
    python benchmarks/run_full_bench.py                      # full run, ~25 min
    python benchmarks/run_full_bench.py --skip-endtoend      # per-function only, ~10 min
    python benchmarks/run_full_bench.py --out-dir /tmp/bench
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gc

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Imports ────────────────────────────────────────────────────────────────────
from ezgatr.opt import (
    geometric_product_v0, geometric_product_v1, geometric_product_v2, geometric_product_v3,
    equi_join_v0, equi_join_v1, equi_join_v2, equi_join_v3,
    equi_linear_ver_0, equi_linear_ver_1, equi_linear_ver_2, equi_linear_ver_3,
    equi_rms_norm_ver_0, equi_rms_norm_ver_1, equi_rms_norm_ver_2, equi_rms_norm_ver_3,
    scaler_gated_gelu_ver_0, scaler_gated_gelu_ver_1, scaler_gated_gelu_ver_2, scaler_gated_gelu_ver_3,
    equi_geometric_attention_ver_0, equi_geometric_attention_ver_1,
    equi_geometric_attention_ver_2, equi_geometric_attention_ver_3,
    equi_geometric_attention_ver_3_1,
)
from ezgatr.nn.functional import (
    geometric_product as gp_py,
    equi_join as join_py,
    equi_linear as linear_py,
    equi_rms_norm as rms_py,
    scaler_gated_gelu as gelu_py,
    equi_geometric_attention as attn_py,
)
from ezgatr.nets.mv_only_gatr import (
    MVOnlyGATrConfig, MVOnlyGATrModel,
    MVOnlyGATrModelASL_ver_0, MVOnlyGATrModelASL_ver_1,
    MVOnlyGATrModelASL_ver_2, MVOnlyGATrModelASL_ver_3,
    MVOnlyGATrModelASL_ver_3,
)

MODEL_CLASSES = {
    "ref":  MVOnlyGATrModel,
    "v0":   MVOnlyGATrModelASL_ver_0,
    "v1":   MVOnlyGATrModelASL_ver_1,
    "v2":   MVOnlyGATrModelASL_ver_2,
    "v3":   MVOnlyGATrModelASL_ver_3,
    "v3_1": MVOnlyGATrModelASL_ver_3,
}

# v0/v1 are very slow at large T — skip them above these sequence lengths
MODEL_SLOW_MAX_T = {"v0": 128, "v1": 256}

KINDS = {"ipa": None, "daa": None}
C = 8  # fixed channels — lets heads/T/B scale without O(n^3) blowup

# ── Dimension formula ─────────────────────────────────────────────────────────
def dims(n: int) -> tuple[int, int, int]:
    """Return (B, H, T) for problem size n."""
    return 2 * n, 4 * n, 128 * n


# ── Timing helpers ─────────────────────────────────────────────────────────────
def bench_once(fn, warmup: int = 1, reps: int = 3) -> float:
    """Return min ms over reps calls (after warmup)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    return best * 1000.0


# ── Input factories ───────────────────────────────────────────────────────────
def make_mv(n: int) -> tuple:
    B, _, T = dims(n)
    return torch.randn(B, T, C, 16), torch.randn(B, T, C, 16)

def make_linear_inputs(n: int) -> tuple:
    mv, _ = make_mv(n)
    return mv, torch.randn(C, C, 9), torch.randn(C)

def make_rms_inputs(n: int) -> tuple:
    mv, _ = make_mv(n)
    return mv, torch.ones(C)

def make_attn(n: int) -> tuple:
    B, H, T = dims(n)
    return (torch.randn(B, H, T, C, 16),
            torch.randn(B, H, T, C, 16),
            torch.randn(B, H, T, C, 16))


# ── Point-wise op families ────────────────────────────────────────────────────
# Each entry: (label, fn, input_factory, extra_kwargs)
POINTWISE_FAMILIES = {
    "geometric_product": [
        ("py",  gp_py,              lambda n: make_mv(n),           {}),
        ("v0",  geometric_product_v0, lambda n: make_mv(n),         {}),
        ("v1",  geometric_product_v1, lambda n: make_mv(n),         {}),
        ("v2",  geometric_product_v2, lambda n: make_mv(n),         {}),
        ("v3",  geometric_product_v3, lambda n: make_mv(n),         {}),
    ],
    "equi_join": [
        ("py",  join_py,     lambda n: make_mv(n), {}),
        ("v0",  equi_join_v0, lambda n: make_mv(n), {}),
        ("v1",  equi_join_v1, lambda n: make_mv(n), {}),
        ("v2",  equi_join_v2, lambda n: make_mv(n), {}),
        ("v3",  equi_join_v3, lambda n: make_mv(n), {}),
    ],
    "equi_linear": [
        ("py",  linear_py,        lambda n: make_linear_inputs(n), {}),
        ("v0",  equi_linear_ver_0, lambda n: make_linear_inputs(n), {}),
        ("v1",  equi_linear_ver_1, lambda n: make_linear_inputs(n), {}),
        ("v2",  equi_linear_ver_2, lambda n: make_linear_inputs(n), {}),
        ("v3",  equi_linear_ver_3, lambda n: make_linear_inputs(n), {}),
    ],
    "equi_rms_norm": [
        ("py",  rms_py,           lambda n: make_rms_inputs(n), {}),
        ("v0",  equi_rms_norm_ver_0, lambda n: make_rms_inputs(n), {}),
        ("v1",  equi_rms_norm_ver_1, lambda n: make_rms_inputs(n), {}),
        ("v2",  equi_rms_norm_ver_2, lambda n: make_rms_inputs(n), {}),
        ("v3",  equi_rms_norm_ver_3, lambda n: make_rms_inputs(n), {}),
    ],
    "scaler_gated_gelu": [
        ("py",  gelu_py,              lambda n: (make_mv(n)[0],), {}),
        ("v0",  scaler_gated_gelu_ver_0, lambda n: (make_mv(n)[0],), {}),
        ("v1",  scaler_gated_gelu_ver_1, lambda n: (make_mv(n)[0],), {}),
        ("v2",  scaler_gated_gelu_ver_2, lambda n: (make_mv(n)[0],), {}),
        ("v3",  scaler_gated_gelu_ver_3, lambda n: (make_mv(n)[0],), {}),
    ],
}

# Attention versions with their max practical n.
# Caps are chosen so each version finishes in ~20s total:
#   v0/v1 n=4 → ~12s/call × 3 reps = 36s — too slow, cap at n=3
#   v2    n=5 → ~6.7s/call × 3 reps = 20s — OK
#   v3    n=7 → ~4s/call   × 3 reps = 12s — OK
#   v3_1  n=9 → ~5s/call   × 3 reps = 15s — OK
ATTENTION_VERSIONS = [
    ("py",   attn_py,                          range(1, 5),  {"kinds": KINDS, "is_causal": False}),
    ("v0",   equi_geometric_attention_ver_0,   range(1, 6),  {"kinds": KINDS, "is_causal": False}),
    ("v1",   equi_geometric_attention_ver_1,   range(1, 6),  {"kinds": KINDS, "is_causal": False}),
    ("v2",   equi_geometric_attention_ver_2,   range(1, 6),  {"kinds": KINDS, "is_causal": False}),
    ("v3",   equi_geometric_attention_ver_3,   range(1, 8),  {"kinds": KINDS, "is_causal": False}),
    ("v3_1", equi_geometric_attention_ver_3_1, range(1, 10), {"kinds": KINDS, "is_causal": False}),
]


# ── Benchmark runners ─────────────────────────────────────────────────────────
def run_pointwise(n_values: list[int]) -> dict:
    results: dict[str, dict] = {}
    total_fams = len(POINTWISE_FAMILIES)
    for fi, (fam, versions) in enumerate(POINTWISE_FAMILIES.items(), 1):
        print(f"  [{fi}/{total_fams}] {fam}")
        results[fam] = {"n_values": n_values, "versions": {}}
        for ver_label, fn, inp_factory, kwargs in versions:
            timings = []
            for n in n_values:
                inp = inp_factory(n)
                call = lambda fn=fn, inp=inp, kw=kwargs: fn(*inp, **kw)
                # Python refs are slower, fewer reps
                w, r = (1, 3) if ver_label == "py" else (2, 8)
                try:
                    ms = bench_once(call, warmup=w, reps=r)
                    timings.append(ms)
                    print(f"      {ver_label} n={n}: {ms:.2f}ms")
                except Exception as e:
                    timings.append(None)
                    print(f"      {ver_label} n={n}: FAILED ({e})")
            results[fam]["versions"][ver_label] = timings
    return results


def run_attention(n_per_version: dict | None = None) -> dict:
    results: dict[str, dict] = {}
    for ver_label, fn, n_range, kwargs in ATTENTION_VERSIONS:
        ns = list(n_per_version.get(ver_label, n_range) if n_per_version else n_range)
        timings = {}
        print(f"  attn {ver_label} — n={ns[0]}..{ns[-1]}")
        for n in ns:
            B, H, T = dims(n)
            q, k, v = make_attn(n)
            call = lambda q=q,k=k,v=v: fn(q, k, v, **kwargs)
            # slower versions get fewer reps
            slow = ver_label in ("v0", "v1", "py")
            w, r = (1, 2) if slow else (1, 3)
            try:
                ms = bench_once(call, warmup=w, reps=r)
                timings[n] = ms
                print(f"      n={n} (B={B},H={H},T={T}): {ms:.0f}ms")
            except Exception as e:
                timings[n] = None
                print(f"      n={n}: FAILED ({e})")
        results[ver_label] = timings
    return results


def speedup_table(pw_results: dict, attn_results: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("SPEEDUP SUMMARY  (Python ref → best C++ version,  min latency)")
    lines.append("=" * 72)

    # Point-wise: compare py vs v3 at each n
    for fam, data in pw_results.items():
        ns = data["n_values"]
        py_ms  = data["versions"].get("py",  [None]*len(ns))
        v3_ms  = data["versions"].get("v3",  [None]*len(ns))
        lines.append(f"\n{fam}")
        header = f"  {'n':>4}  {'B':>4} {'T':>5}  {'py (ms)':>10}  {'v3 (ms)':>10}  {'speedup':>8}"
        lines.append(header)
        for i, n in enumerate(ns):
            B, _, T = dims(n)
            py = py_ms[i]; v3 = v3_ms[i]
            if py is None or v3 is None:
                continue
            sp = py / v3 if v3 > 0 else float("inf")
            lines.append(f"  {n:>4}  {B:>4} {T:>5}  {py:>10.2f}  {v3:>10.2f}  {sp:>7.1f}x")

    # Attention: compare py vs v3_1 at common n values
    lines.append(f"\nequi_geometric_attention")
    py_t   = attn_results.get("py", {})
    v3_1_t = attn_results.get("v3_1", {})
    v3_t   = attn_results.get("v3", {})
    v0_t   = attn_results.get("v0", {})
    header = f"  {'n':>4}  {'B':>4} {'H':>4} {'T':>5}  {'py':>9}  {'v0':>9}  {'v3':>9}  {'v3_1':>9}  {'v3_1/py':>9}"
    lines.append(header)
    all_ns = sorted(set(list(py_t) + list(v3_1_t) + list(v3_t) + list(v0_t)))
    for n in all_ns:
        B, H, T = dims(n)
        py   = py_t.get(n)
        v0   = v0_t.get(n)
        v3   = v3_t.get(n)
        v31  = v3_1_t.get(n)
        def fmt(x): return f"{x:>9.0f}" if x is not None else f"{'—':>9}"
        sp = f"{py/v31:>8.1f}x" if (py is not None and v31 is not None and v31>0) else f"{'—':>9}"
        lines.append(f"  {n:>4}  {B:>4} {H:>4} {T:>5}  {fmt(py)}  {fmt(v0)}  {fmt(v3)}  {fmt(v31)}  {sp}")

    return "\n".join(lines) + "\n"


# ── Full-model sweep ──────────────────────────────────────────────────────────
def run_full_model(t_values: list[int], batch: int = 8,
                   channels: int = 16, heads: int = 4, layers: int = 2) -> dict:
    """Benchmark the complete model (ref + v0–v3_1) across sequence lengths.

    Weights are copied from ref into every version so this also acts as a
    correctness check. Only one model is resident at a time to avoid polluting
    the allocator / cache for fast versions.
    """
    results: dict[str, dict] = {v: {} for v in MODEL_CLASSES}

    cfg = MVOnlyGATrConfig(
        num_layers=layers,
        size_context=max(t_values),   # set to max; we re-build per T below
        size_channels_in=1,
        size_channels_out=1,
        size_channels_hidden=channels,
        size_channels_intermediate=channels,
        attn_num_heads=heads,
        attn_is_causal=False,
    )

    for T in t_values:
        print(f"  T={T}  (B={batch}, C={channels}, H={heads})")
        # Build ref at this T, share weights
        tcfg = MVOnlyGATrConfig(
            num_layers=layers, size_context=T,
            size_channels_in=1, size_channels_out=1,
            size_channels_hidden=channels,
            size_channels_intermediate=channels,
            attn_num_heads=heads, attn_is_causal=False,
        )
        ref_model = MVOnlyGATrModel(tcfg).eval()
        sd = ref_model.state_dict()
        x = torch.randn(batch, T, 1, 16)

        for ver, cls in MODEL_CLASSES.items():
            if T > MODEL_SLOW_MAX_T.get(ver, float("inf")):
                results[ver][T] = None
                print(f"      {ver}: skip (T>{MODEL_SLOW_MAX_T[ver]})")
                continue
            m = ref_model if ver == "ref" else cls(tcfg).eval()
            if ver != "ref":
                m.load_state_dict(sd, strict=False)
            with torch.inference_mode():
                # warmup
                m(x); m(x)
                t0 = time.perf_counter()
                for _ in range(5):
                    m(x)
                ms = (time.perf_counter() - t0) / 5 * 1000
            results[ver][T] = ms
            print(f"      {ver}: {ms:.1f}ms")
            if ver != "ref":
                del m; gc.collect()
        del ref_model; gc.collect()

    return results


def model_speedup_table(model_results: dict) -> str:
    versions = list(MODEL_CLASSES.keys())
    t_values = sorted({T for d in model_results.values() for T in d})
    lines = ["=" * 72,
             "FULL MODEL SPEEDUP  (ref vs v0–v3_1, min over 5 forward passes)",
             "=" * 72,
             "  %5s  " % "T" + "  ".join(f"{v:>8}" for v in versions) + "  ref/v3_1 speedup"]
    for T in t_values:
        row = "  %5d  " % T
        ref_ms  = model_results["ref"].get(T)
        v31_ms  = model_results["v3_1"].get(T)
        row += "  ".join(
            f"{'skip':>8}" if model_results[v].get(T) is None
            else f"{model_results[v][T]:>8.1f}"
            for v in versions
        )
        if ref_ms and v31_ms:
            row += f"    {ref_ms/v31_ms:>6.1f}x"
        lines.append(row)
    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max-pointwise", type=int, default=12)
    ap.add_argument("--skip-endtoend", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir: Path = args.out_dir or (ROOT / "benchmarks" / "results" / f"run_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {out_dir}")
    print(f"Dims: B=2n, H=4n, T=128n, C={C} fixed")
    print(f"Threads: {args.threads}")

    n_values_pw = list(range(1, args.n_max_pointwise + 1))

    # ── Point-wise ops ────────────────────────────────────────────────────────
    t0_total = time.time()
    print(f"\n{'='*60}")
    print(f"POINT-WISE OPS  n=1..{args.n_max_pointwise}")
    print(f"{'='*60}")
    pw_results = run_pointwise(n_values_pw)

    # ── Attention ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ATTENTION  (per-version n ranges)")
    print(f"{'='*60}")
    attn_results = run_attention()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        "meta": {
            "date": stamp, "threads": args.threads,
            "C": C, "dim_formula": "B=2n, H=4n, T=128n, C=8",
        },
        "pointwise": pw_results,
        "attention": {v: {str(k): ms for k, ms in d.items()} for v, d in attn_results.items()},
    }
    (out_dir / "per_function.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_dir / 'per_function.json'}")

    # ── Speedup table ──────────────────────────────────────────────────────────
    table = speedup_table(pw_results, attn_results)
    print("\n" + table)
    (out_dir / "speedup_table.txt").write_text(table)
    print(f"Saved: {out_dir / 'speedup_table.txt'}")

    # ── Full-model sweep (all versions incl. v3_1) ────────────────────────────
    print(f"\n{'='*60}")
    print("FULL PROJECT MODEL  (ref → v3_1, sweep T)")
    print(f"{'='*60}")
    model_t_values = [16, 32, 64, 128, 256, 512]
    model_results = run_full_model(model_t_values)
    payload["full_model"] = {v: {str(T): ms for T, ms in d.items()} for v, d in model_results.items()}
    (out_dir / "per_function.json").write_text(json.dumps(payload, indent=2))  # re-save with model data

    mtable = model_speedup_table(model_results)
    print("\n" + mtable)
    (out_dir / "model_speedup_table.txt").write_text(mtable)
    print(f"Saved: {out_dir / 'model_speedup_table.txt'}")

    # ── End-to-end benchmarks ─────────────────────────────────────────────────
    venv_py = str(ROOT / ".venv" / "bin" / "python")
    if not args.skip_endtoend:
        for script, label, out_name in [
            ("bench_endtoend_scaleM.py", "END-TO-END scale-M (sweep batch, fixed T=64)", "endtoend_scaleM.json"),
            ("bench_endtoend_scaleT.py", "END-TO-END scale-T (sweep T, fixed B=8)",     "endtoend_scaleT.json"),
            ("bench_pointops_scaleM.py", "POINT-OPS scale-M  (activation cache sweep)", "pointops_scaleM.json"),
        ]:
            print(f"\n{'='*60}\n{label}\n{'='*60}")
            out_path = out_dir / out_name
            cmd = [venv_py, str(ROOT / "benchmarks" / script), "--json-out", str(out_path)]
            print("$ " + " ".join(cmd))
            subprocess.run(cmd)

    elapsed = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min — results in {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
