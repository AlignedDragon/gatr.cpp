"""Attention-only sweep over large sequence length T.

Geometric attention is the O(T^2) part of the model, so its behaviour at large T
is what decides the end-to-end speedup once sequences get long. This isolates the
``equi_geometric_attention`` op and sweeps T (batch/heads/channels fixed) to show
how the speedup moves with T — both vs the Python reference and across the v0..v3
optimization tiers.

Every version is validated against the Python reference (max-abs-err) before its
time is trusted. Pins to one core; single-threaded.

    python benchmarks/bench_attention_bigT.py --T 128 256 512 1024 2048 4096 \
        --json-out benchmarks/results/attention_bigT.json \
        --plot-out benchmarks/results/plots/attention_bigT.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from ezgatr.nn.functional import equi_geometric_attention as attn_py
from ezgatr.nn.functional.attention_cpp import (
    equi_geometric_attention_cpp_ver_0 as v0,
    equi_geometric_attention_cpp_ver_1 as v1,
    equi_geometric_attention_cpp_ver_2 as v2,
    equi_geometric_attention_cpp_ver_3 as v3,
)

KINDS = {"ipa": None, "daa": None}
VERSIONS = {"v0": v0, "v1": v1, "v2": v2, "v3": v3}
VC = {"v0": "#B0BEC5", "v1": "#FF9800", "v2": "#2196F3", "v3": "#4CAF50"}


def pin(core: int) -> None:
    if core >= 0 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})


def time_call(fn, warmup: int, repeats: int, inner: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            for _ in range(inner):
                fn()
        xs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(inner):
                fn()
            xs.append((time.perf_counter() - t0) * 1e3 / inner)
    return statistics.median(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner-iters", type=int, default=2)
    ap.add_argument("--pin-core", type=int, default=0)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--plot-out", type=Path)
    args = ap.parse_args()

    torch.set_num_threads(1)
    pin(args.pin_core)
    B, H, C = args.batch, args.heads, args.channels

    rows = []
    print(f"attention sweep  B={B} H={H} C={C} causal={args.causal}  "
          f"(threads=1, core {args.pin_core})")
    hdr = f"{'T':>6} | {'py ms':>10} " + " ".join(f"{v+' ms':>10}" for v in VERSIONS)
    hdr += " | " + " ".join(f"{'py/'+v:>7}" for v in VERSIONS) + f" {'v0/v3':>7} {'maxerr':>9}"
    print(hdr)

    for T in args.T:
        torch.manual_seed(0)
        q = torch.randn(B, H, T, C, 16)
        k = torch.randn(B, H, T, C, 16)
        v = torch.randn(B, H, T, C, 16)

        ref, _ = attn_py(q, k, v, kinds=KINDS, is_causal=args.causal)
        err = 0.0
        times = {}
        times["py"] = time_call(
            lambda: attn_py(q, k, v, kinds=KINDS, is_causal=args.causal),
            args.warmup, args.repeats, args.inner_iters)
        for name, fn in VERSIONS.items():
            out, _ = fn(q, k, v, kinds=KINDS, is_causal=args.causal)
            err = max(err, (out - ref).abs().max().item())
            times[name] = time_call(
                lambda f=fn: f(q, k, v, kinds=KINDS, is_causal=args.causal),
                args.warmup, args.repeats, args.inner_iters)

        sp_py = {v: times["py"] / times[v] for v in VERSIONS}
        v0v3 = times["v0"] / times["v3"]
        rows.append({"T": T, "times_ms": times, "speedup_vs_py": sp_py,
                     "v0_over_v3": v0v3, "max_abs_err": err})
        line = f"{T:>6} | {times['py']:>10.3f} " + " ".join(f"{times[v]:>10.4f}" for v in VERSIONS)
        line += " | " + " ".join(f"{sp_py[v]:>6.1f}x" for v in VERSIONS)
        line += f" {v0v3:>6.1f}x {err:>9.1e}"
        print(line)

    payload = {"batch": B, "heads": H, "channels": C, "causal": args.causal,
               "threads": 1, "results": rows}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ts = [r["T"] for r in rows]
        fig, (axp, axv) = plt.subplots(1, 2, figsize=(13, 5.0))
        # left: speedup vs Python, line per version
        for ver in VERSIONS:
            axp.plot(Ts, [r["speedup_vs_py"][ver] for r in rows], "o-", lw=2.2, ms=7,
                     color=VC[ver], label=ver)
        axp.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
        axp.set_xscale("log", base=2); axp.set_yscale("log")
        axp.set_xticks(Ts); axp.set_xticklabels(Ts)
        axp.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}×" if y >= 2 else f"{y:.1f}×"))
        axp.set_xlabel("sequence length T"); axp.set_ylabel("speedup vs Python (log)")
        axp.set_title("Attention speedup vs Python")
        axp.legend(title="version"); axp.grid(True, which="both", alpha=0.25)
        # right: v0 -> v3 speedup
        axv.plot(Ts, [r["v0_over_v3"] for r in rows], "o-", lw=2.4, ms=8, color="#4CAF50")
        axv.set_xscale("log", base=2)
        axv.set_xticks(Ts); axv.set_xticklabels(Ts)
        axv.set_xlabel("sequence length T"); axv.set_ylabel("v0 → v3 speedup")
        axv.set_title("Attention v0 → v3 (custom flash kernel)")
        axv.grid(True, which="both", alpha=0.25)
        cz = "causal" if args.causal else "full"
        fig.suptitle(f"Geometric attention vs T  (B={B}, H={H}, C={C}, {cz})  |  "
                     f"Ryzen 9 3900X, 1 thread", fontsize=11)
        fig.tight_layout()
        args.plot_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.plot_out}")


if __name__ == "__main__":
    main()
