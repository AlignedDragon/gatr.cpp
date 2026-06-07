"""Plot the MV-only GATr end-to-end batch (M) sweep, pointops-style.

Mirrors plot_pointops_scaleM.py but for the whole model:
  left  — achieved GFLOP/s vs hidden-activation footprint (the cache-hierarchy
          curve: rises, plateaus while the working set is cache-resident, drops
          once it spills to DRAM)
  right — v3 speedup over the torch reference and over v0 vs footprint
with L1/L2/L3 capacity bands.

Model FLOPs are an analytical estimate summed over the forward pass using the
same per-op formulas as roofline_costs / bench_pointops_scaleM (so GFLOP/s here
is directly comparable to the per-function plots). FLOPs scale linearly with B,
so the curve shape is exact; the absolute scale is the usual analytical estimate.

    python benchmarks/plot_endtoend_scaleM.py benchmarks/results/endtoend_scaleM.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/endtoend_scaleM.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())
rows = data["rows"]
cache = data["cache_bytes"]
T, C, H, L = data["T"], data["channels"], data["heads"], data["layers"]
Ci = C  # size_channels_intermediate == size_channels_hidden in the bench config

VERSION_STYLE = {
    "ref": ("#9C27B0", "ref (torch)"),
    "v0": ("#B0BEC5", "v0 (baseline)"),
    "v1": ("#FF9800", "v1 (math)"),
    "v2": ("#2196F3", "v2 (pre-SIMD)"),
    "v3": ("#4CAF50", "v3 (full AVX2)"),
}


# --- analytical model FLOPs (same per-op formulas as the per-function plots) ---
def _eqlin(M, i, o):  return M * i * o * 16 * 9 * 2     # equi_linear
def _rms(M, c):       return M * c * 64                 # equi_rms_norm
def _gp(M, c):        return M * c * 384                # geometric_product (sparse)
def _join(M, c):      return M * c * 384                # equi_join (sparse)
def _gelu(M, c):      return M * c * 128                # scaler_gated_gelu
def _attn(B, h, t, c): return 56 * B * h * t * t * c + 496 * B * h * t * c


def model_flops(B):
    M = B * T
    embed = _eqlin(M, 1, C)
    head = _eqlin(M, C, 1)
    attention = _rms(M, C) + _eqlin(M, C, C * H * 3) + _attn(B, H, T, C) + _eqlin(M, C * H, C)
    mlp = (_rms(M, C) + _eqlin(M, C, Ci * 4) + _gp(M, Ci) + _join(M, Ci)
           + _eqlin(M, Ci * 2, C) + _gelu(M, C) + _eqlin(M, C, C))
    return embed + head + L * (attention + mlp)


def add_cache_bands(ax):
    # Mark the batch B at which the activation footprint crosses each cache level.
    per_B = rows[0]["footprint_bytes"] / rows[0]["B"]
    Bs = [r["B"] for r in rows]
    for name, key in [("L1", "L1D"), ("L2", "L2"), ("L3", "L3")]:
        b_edge = cache[key] / per_B
        if Bs[0] <= b_edge <= Bs[-1]:
            ax.axvline(b_edge, color="k", ls=":", lw=0.8, alpha=0.4)
            ax.text(b_edge, ax.get_ylim()[1], f" {name}→", fontsize=8, ha="left", va="top", alpha=0.55)


gflops = {v: [model_flops(r["B"]) / (r["times_us"][v] * 1e3) for r in rows if r["times_us"].get(v) is not None]
          for v in VERSION_STYLE}
B_v = {v: [r["B"] for r in rows if r["times_us"].get(v) is not None] for v in VERSION_STYLE}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: GFLOP/s vs batch size.
ax = axes[0]
for v, (color, label) in VERSION_STYLE.items():
    if B_v[v]:
        ax.plot(B_v[v], gflops[v], "o-", color=color, label=label, lw=1.8, ms=5)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size B (log₂)")
ax.set_ylabel("GFLOP/s")
ax.set_title(f"End-to-end performance  [T={T}, heads={H}, channels={C}, layers={L}, 1 thread]")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
add_cache_bands(ax)

# Right: v3 speedup vs batch size.
ax = axes[1]
s_ref = [(r["B"], r["speedup_v3_over_ref"]) for r in rows if r["speedup_v3_over_ref"] == r["speedup_v3_over_ref"]]
s_v0 = [(r["B"], r["speedup_v3_over_v0"]) for r in rows if r["speedup_v3_over_v0"] == r["speedup_v3_over_v0"]]
ax.plot([b for b, _ in s_ref], [s for _, s in s_ref], "s-", color="#9C27B0", label="v3 / ref", lw=1.8, ms=5)
ax.plot([b for b, _ in s_v0], [s for _, s in s_v0], "^-", color="#B0BEC5", label="v3 / v0", lw=1.8, ms=5)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size B (log₂)")
ax.set_ylabel("Speedup (×)")
ax.set_title("End-to-end: v3 speedup")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
add_cache_bands(ax)

fig.suptitle(f"MV-only GATr — end-to-end performance, batch (M) scaling at fixed T={T} (single-core)", fontsize=12)
fig.tight_layout()
out = out_dir / "endtoend_scaleM.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")
