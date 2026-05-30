"""Plot runtime + roofline charts from benchmark_repo.py JSON output.

Usage:
    python benchmarks/plot_results.py \\
        --json benchmarks/results/runtime.json \\
        --peak-gflops 80 \\
        --peak-bw-gbs 25 \\
        --op gp \\
        --out benchmarks/results/

The JSON is the output of `benchmark_repo.py --json-out`.
The script picks the variants for the chosen op (gp or join) and emits two PNGs:
    runtime_<op>.png   - bar chart of mean_ms per variant
    roofline_<op>.png  - log-log roofline with achieved (AI, GFLOP/s) per variant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRESETS = {
    "tiny":   {"batch": 2, "tokens": 32,  "channels": 8},
    "small":  {"batch": 4, "tokens": 128, "channels": 16},
    "medium": {"batch": 4, "tokens": 256, "channels": 32},
}


# (flops_per_mv, bytes_per_mv_fp32). bytes assume 16+16 loads + 16 stores
# of the same dtype; for fp64 we double at runtime.
VARIANT_COSTS = {
    "gp": {
        "geometric_product":            (8192, 192),   # python einsum (dense math)
        "geometric_product_v0":      (8192, 192),   # cpp triple loop
        "geometric_product_v1":  (384,  192),   # 192 FMAs = 384 FLOPs
        "geometric_product_v2":   (384,  192),   # same FLOP count, unrolled
    },
    "join": {
        # Join non-zero entries: a typical 3D-PGA join has the same sparsity
        # pattern as the geometric product (192 non-zeros).
        "equi_join":                    (8192, 196),   # python einsum + ref read
        "equi_join_v0":              (8192, 196),
        "equi_join_v1":          (384,  196),
        "equi_join_v2":           (384,  196),
    },
}


VARIANT_ORDER = {
    "gp": [
        "geometric_product",
        "geometric_product_v0",
        "geometric_product_v1",
        "geometric_product_v2",
    ],
    "join": [
        "equi_join",
        "equi_join_v0",
        "equi_join_v1",
        "equi_join_v2",
    ],
}


VARIANT_LABEL = {
    "geometric_product":           "python-einsum",
    "geometric_product_v0":     "cpp-dense",
    "geometric_product_v1": "cpp-sparse-rt",
    "geometric_product_v2":  "cpp-unrolled",
    "equi_join":                   "python-einsum",
    "equi_join_v0":             "cpp-dense",
    "equi_join_v1":         "cpp-sparse-rt",
    "equi_join_v2":          "cpp-unrolled",
}


def n_multivectors(preset: str) -> int:
    cfg = PRESETS[preset]
    return cfg["batch"] * cfg["tokens"] * cfg["channels"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--peak-gflops", type=float, required=True,
                   help="Peak scalar FLOP/s of the target machine, in GFLOP/s.")
    p.add_argument("--peak-bw-gbs", type=float, required=True,
                   help="Peak DRAM bandwidth, in GB/s.")
    p.add_argument("--op", choices=["gp", "join"], required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for the PNGs.")
    p.add_argument("--dtype-bytes", type=int, default=4,
                   help="Element size used in the bench (4 for fp32, 8 for fp64).")
    return p.parse_args()


def select_rows(payload: dict, op: str) -> list[dict]:
    order = VARIANT_ORDER[op]
    by_target = {row["target"]: row for row in payload["results"]}
    return [by_target[t] for t in order if t in by_target]


def plot_runtime(rows: list[dict], op: str, out: Path) -> None:
    labels = [VARIANT_LABEL[r["target"]] for r in rows]
    means  = [r["mean_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, color=["#888", "#c44", "#cc7733", "#3a3"])
    ax.set_ylabel("mean runtime per call (ms)")
    ax.set_title(f"{op}: runtime per variant (lower is better)")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roofline(
    rows: list[dict],
    op: str,
    preset: str,
    peak_gflops: float,
    peak_bw_gbs: float,
    dtype_bytes: int,
    out: Path,
) -> None:
    N = n_multivectors(preset)
    points = []
    for r in rows:
        flops_per_mv, bytes_per_mv_fp32 = VARIANT_COSTS[op][r["target"]]
        bytes_per_mv = bytes_per_mv_fp32 * (dtype_bytes / 4.0)
        ai = flops_per_mv / bytes_per_mv
        gflops = (N * flops_per_mv) / (r["mean_ms"] * 1e-3) / 1e9
        points.append((VARIANT_LABEL[r["target"]], ai, gflops))

    ai_grid = np.logspace(-1, 3, 200)
    mem_ceiling = peak_bw_gbs * ai_grid
    compute_ceiling = np.full_like(ai_grid, peak_gflops)
    roof = np.minimum(mem_ceiling, compute_ceiling)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(ai_grid, roof, "-", color="#333", lw=1.5, label="roofline")
    ax.loglog(ai_grid, mem_ceiling, "--", color="#888", lw=0.8, alpha=0.6)
    ax.loglog(ai_grid, compute_ceiling, "--", color="#888", lw=0.8, alpha=0.6)

    colors = ["#888", "#c44", "#cc7733", "#3a3"]
    for (label, ai, gf), c in zip(points, colors):
        ax.plot(ai, gf, "o", color=c, ms=8, label=f"{label}: {gf:.2f} GF/s @ AI={ai:.2f}")

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"{op}: roofline (peak {peak_gflops:.0f} GFLOP/s, "
                 f"BW {peak_bw_gbs:.0f} GB/s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json.read_text())
    preset = payload["preset"]
    rows = select_rows(payload, args.op)
    if not rows:
        raise SystemExit(f"No matching {args.op} variants found in {args.json}")

    args.out.mkdir(parents=True, exist_ok=True)
    runtime_path  = args.out / f"runtime_{args.op}.png"
    roofline_path = args.out / f"roofline_{args.op}.png"

    plot_runtime(rows, args.op, runtime_path)
    plot_roofline(rows, args.op, preset,
                  args.peak_gflops, args.peak_bw_gbs, args.dtype_bytes,
                  roofline_path)
    print(f"wrote {runtime_path}")
    print(f"wrote {roofline_path}")


if __name__ == "__main__":
    main()
