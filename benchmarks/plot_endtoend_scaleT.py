"""Plot the MV-only GATr end-to-end sequence-length sweep.

Consumes JSON from bench_endtoend_scaleT.py --json-out and draws two panels:
  left  — forward-pass runtime vs sequence length T (log-log; attention is O(T^2))
  right — v2 speedup over the reference and over v0 vs T

Usage:
    python benchmarks/plot_endtoend_scaleT.py benchmarks/results/endtoend_scaleT.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/endtoend_scaleT.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())
rows = data["rows"]
T = [r["T"] for r in rows]

VERSION_STYLE = {
    "ref": ("#9C27B0", "ref (torch)"),
    "v0": ("#B0BEC5", "v0 (baseline)"),
    "v1": ("#FF9800", "v1 (math)"),
    "v2": ("#2196F3", "v2 (pre-SIMD)"),
    "v3": ("#4CAF50", "v3 (full AVX2)"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))


def annotate_pts(ax, xs, ys, color, dy=6):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}×" if y >= 10 else f"{y:.1f}×", (x, y),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=6.5, color=color, fontweight="bold")


# Left: speedup over v0 vs T (all versions), linear y, values labelled at dots.
ax = axes[0]
for v, (color, label) in VERSION_STYLE.items():
    sp = [r["times_us"]["v0"] / r["times_us"][v] for r in rows]
    ax.plot(T, sp, "o-", color=color, label=label, lw=1.8, ms=5)
    if v != "v0":  # v0 is the flat 1x baseline; don't clutter it
        annotate_pts(ax, T, sp, color)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xlabel("Sequence length T (log₂)")
ax.set_ylabel("Speedup over v0 (×)")
ax.set_title(f"End-to-end speedup over v0  [batch={data['batch']}, heads={data['heads']}, "
             f"channels={data['channels']}, layers={data['layers']}]")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# Right: v3 speedup vs T, values labelled at dots.
ax = axes[1]
s_ref = [r["speedup_v3_over_ref"] for r in rows]
s_v0 = [r["speedup_v3_over_v0"] for r in rows]
ax.plot(T, s_ref, "s-", color="#9C27B0", label="v3 / ref", lw=1.8, ms=5)
ax.plot(T, s_v0, "^-", color="#B0BEC5", label="v3 / v0", lw=1.8, ms=5)
annotate_pts(ax, T, s_ref, "#9C27B0")
annotate_pts(ax, T, s_v0, "#B0BEC5")
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xlabel("Sequence length T (log₂)")
ax.set_ylabel("Speedup (×)")
ax.set_title("End-to-end: v3 speedup vs T")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("MV-only GATr — end-to-end forward, sequence-length scaling (single-core)", fontsize=12)
fig.tight_layout()
out = out_dir / "endtoend_scaleT.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")
