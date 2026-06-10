"""Plot the equi_linear / equi_rms_norm M-scaling sweep.

Consumes the JSON emitted by bench_pointops_scaleM.py --json-out and draws, per
op, two panels:
  left  — achieved GFLOP/s vs working-set footprint (the cache-hierarchy curve)
  right — v3 speedup over v2 and v0 vs footprint
with vertical bands marking the L1/L2/L3 capacity boundaries. A roofline is NOT
used here: these ops have M-independent operational intensity, so every M-point
would collapse onto one vertical line. Perf-vs-size is the informative view.

Usage:
    python benchmarks/plot_pointops_scaleM.py benchmarks/results/pointops.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/pointops_scaleM.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())
channels = data["channels"]
cache = data["cache_bytes"]

VERSION_STYLE = {
    "v0": ("#B0BEC5", "v0 (baseline)"),
    "v1": ("#FF9800", "v1 (math)"),
    "v2": ("#2196F3", "v2 (pre-SIMD)"),
    "v3": ("#4CAF50", "v3 (AVX2)"),
}


def add_cache_bands(ax, rows):
    """Mark the M at which the working-set footprint crosses each cache level."""
    per_M = rows[0]["footprint_bytes"] / rows[0]["M"]
    Ms = [r["M"] for r in rows]
    for name, key in [("L1", "L1D"), ("L2", "L2"), ("L3", "L3")]:
        m_edge = cache[key] / per_M
        if Ms[0] <= m_edge <= Ms[-1]:
            ax.axvline(m_edge, color="k", ls=":", lw=0.8, alpha=0.4)
            ax.text(m_edge, ax.get_ylim()[1] * 0.97, f" {name}→", fontsize=8,
                    ha="left", va="top", alpha=0.55)


def plot_op(op_name, rows):
    M = [r["M"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: GFLOP/s vs M.
    ax = axes[0]
    for v, (color, label) in VERSION_STYLE.items():
        xs = [r["M"] for r in rows if r["gflops"].get(v) is not None]
        ys = [r["gflops"][v] for r in rows if r["gflops"].get(v) is not None]
        if xs:
            ax.plot(xs, ys, "o-", color=color, label=label, lw=1.8, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of multivectors M (log₂)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"{op_name}: performance  [channels={channels}, 1 thread]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    add_cache_bands(ax, rows)

    # Right: v3 speedup vs M.
    ax = axes[1]
    s32 = [r["speedup_v3_over_v2"] for r in rows]
    s30 = [r["speedup_v3_over_v0"] for r in rows]
    ax.plot(M, s32, "s-", color="#2196F3", label="v3 / v2", lw=1.8, ms=5)
    valid30 = [(m, s) for m, s in zip(M, s30) if s == s]
    if valid30:
        ax.plot([m for m, _ in valid30], [s for _, s in valid30],
                "^-", color="#B0BEC5", label="v3 / v0", lw=1.8, ms=5)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of multivectors M (log₂)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"{op_name}: v3 speedup")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    add_cache_bands(ax, rows)

    fig.suptitle(f"{op_name}  —  M-scaling at fixed channels={channels} (single-core)", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"pointops_scaleM_{op_name}.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


for op_name, rows in data["ops"].items():
    plot_op(op_name, rows)
