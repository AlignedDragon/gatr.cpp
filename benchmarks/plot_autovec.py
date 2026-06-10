"""Plot the autovectorization comparison from bench_variant.py output.

Reads benchmarks/results/autovec.json (labels "novec" and "autovec") and draws,
per function and for the end-to-end model, the speedup over the scalar baseline:
  v2-scalar (1x)  vs  v2-autovec (compiler auto-vectorization)  vs  v3 (hand AVX2)

Shows how much GCC's auto-vectorizer recovers and how much headroom the
hand-written intrinsics still add on top.

    python benchmarks/plot_autovec.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/autovec.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())["labels"]
novec, autovec = data["novec"], data["autovec"]

FAM_LABEL = {
    "geometric_product": "Geom.\nProduct", "equi_join": "Equi.\nJoin",
    "equi_linear": "Equi.\nLinear", "equi_rms_norm": "RMS\nNorm",
    "scaler_gated_gelu": "Gated\nGELU", "equi_geometric_attention": "Attention",
}
BARS = [
    ("v2-scalar",  "#B0BEC5"),   # baseline (autovec off)
    ("v2-autovec", "#2196F3"),   # compiler auto-vectorization
    ("v3-handSIMD", "#4CAF50"),  # explicit AVX2 intrinsics
]


def speedups_ops(fam):
    """Speedup over the scalar-v2 baseline for one function family."""
    s = novec["ops_ms"][fam]["v2"]            # scalar baseline
    a = autovec["ops_ms"][fam]["v2"]          # autovectorized v2
    h = autovec["ops_ms"][fam]["v3"]          # hand-SIMD v3
    return [1.0, s / a, s / h]


def speedups_model():
    s = novec["model_ms"]["v2"]
    a = autovec["model_ms"]["v2"]
    h = autovec["model_ms"]["v3"]
    return [1.0, s / a, s / h]


fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [3, 1]})

# Left: per-function speedup over scalar v2.
ax = axes[0]
fams = list(FAM_LABEL)
bw, gap = 0.25, 0.5
xticks, xlabels = [], []
x = 0
for fam in fams:
    sp = speedups_ops(fam)
    for vi, (label, color) in enumerate(BARS):
        bx = x + vi * bw
        ax.bar(bx, sp[vi], bw, color=color, edgecolor="white", lw=0.5, zorder=3)
        ax.text(bx, sp[vi] * 1.03, f"{sp[vi]:.1f}×", ha="center", va="bottom", fontsize=7.5)
    xticks.append(x + bw)
    xlabels.append(FAM_LABEL[fam])
    x += 3 * bw + gap
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=9)
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_yscale("log")
ax.set_ylabel("Speedup over scalar v2 (×, log)", fontsize=11)
ax.set_title(f"Per-function: auto-vectorization vs hand-SIMD  [n={autovec['n']}, GCC, 1 thread]", fontsize=11)
ax.grid(axis="y", alpha=0.25, which="both", zorder=0)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in BARS],
          [b for b, _ in BARS], fontsize=9, title="Build")

# Right: end-to-end speedup over scalar v2.
ax = axes[1]
sp = speedups_model()
for vi, (label, color) in enumerate(BARS):
    ax.bar(vi, sp[vi], 0.7, color=color, edgecolor="white", lw=0.5, zorder=3)
    ax.text(vi, sp[vi] * 1.03, f"{sp[vi]:.1f}×", ha="center", va="bottom", fontsize=9)
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_xticks(range(len(BARS))); ax.set_xticklabels([b for b, _ in BARS], rotation=20, fontsize=8)
ax.set_ylabel("Speedup over scalar v2 (×)", fontsize=11)
ax.set_title(f"End-to-end model\n[B={autovec['model_batch']}, T={autovec['model_T']}]", fontsize=11)
ax.grid(axis="y", alpha=0.25, zorder=0)

fig.suptitle("Auto-vectorization vs hand-written SIMD — GCC -O3, single-core "
             "(v2 -fno-tree-vectorize vs v2 -O3 vs v3 AVX2)", fontsize=12)
fig.tight_layout()
out = out_dir / "autovec_compare.png"
fig.savefig(out, dpi=140)
print(f"saved: {out}")
