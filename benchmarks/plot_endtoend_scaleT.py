"""Plot the MV-only GATr end-to-end n-sweep (B=2n, H=4n, T=128n, C=8).

Shows runtime vs n (sequence length focus: T=128n) and per-version speedups.

Usage:
    python benchmarks/plot_endtoend_scaleT.py benchmarks/results/FINAL_PLOTS/endtoend_scaleT.json
    python benchmarks/plot_endtoend_scaleT.py <json> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/FINAL_PLOTS/endtoend_scaleT.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/FINAL_PLOTS")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())
rows = data["rows"]
layers = data.get("layers", 2)

VERSION_STYLE = {
    "ref": ("#9C27B0", "ref (torch)"),
    "v0":  ("#B0BEC5", "v0 (baseline)"),
    "v1":  ("#FF9800", "v1 (math)"),
    "v2":  ("#2196F3", "v2 (pre-SIMD)"),
    "v3":  ("#4CAF50", "v3 (full AVX2)"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))


def annotate_pts(ax, xs, ys, color, dy=6):
    for x, y in zip(xs, ys):
        if y == y:  # skip NaN
            ax.annotate(f"{y:.0f}×" if y >= 10 else f"{y:.1f}×", (x, y),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=6.5, color=color, fontweight="bold")


# Left: runtime (ms) vs T=128n for each version
ax = axes[0]
for v, (color, label) in VERSION_STYLE.items():
    xs = [r["T"] for r in rows if r["times_us"].get(v) is not None]
    ys = [r["times_us"][v] / 1e3 for r in rows if r["times_us"].get(v) is not None]
    if xs:
        ax.plot(xs, ys, "o-", color=color, label=label, lw=1.8, ms=5)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Sequence length T=128n  (log₂)")
ax.set_ylabel("Runtime (ms, log)")
ax.set_title(f"End-to-end runtime  [B=2n, H=4n, T=128n, C=8, L={layers}]")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# Right: speedup over v0 and ref vs T
ax = axes[1]
s_ref_xs = [r["T"] for r in rows if r.get("speedup_v3_over_ref") == r.get("speedup_v3_over_ref")]
s_ref_ys = [r["speedup_v3_over_ref"] for r in rows if r.get("speedup_v3_over_ref") == r.get("speedup_v3_over_ref")]
s_v0_xs  = [r["T"] for r in rows if r.get("speedup_v3_over_v0") == r.get("speedup_v3_over_v0")]
s_v0_ys  = [r["speedup_v3_over_v0"] for r in rows if r.get("speedup_v3_over_v0") == r.get("speedup_v3_over_v0")]

# filter NaN
s_ref = [(x, y) for x, y in zip(s_ref_xs, s_ref_ys) if y == y]
s_v0  = [(x, y) for x, y in zip(s_v0_xs,  s_v0_ys)  if y == y]

if s_ref:
    ax.plot([x for x, _ in s_ref], [y for _, y in s_ref], "s-", color="#9C27B0", label="v3 / ref", lw=1.8, ms=5)
    annotate_pts(ax, [x for x, _ in s_ref], [y for _, y in s_ref], "#9C27B0")
if s_v0:
    ax.plot([x for x, _ in s_v0], [y for _, y in s_v0], "^-", color="#B0BEC5", label="v3 / v0", lw=1.8, ms=5)
    annotate_pts(ax, [x for x, _ in s_v0], [y for _, y in s_v0], "#B0BEC5", dy=-12)

ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xlabel("Sequence length T=128n  (log₂)")
ax.set_ylabel("Speedup (×)")
ax.set_title("End-to-end: v3 speedup vs sequence length")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("MV-only GATr — end-to-end forward, n-scaling (B=2n, H=4n, T=128n, C=8, single-core)", fontsize=12)
fig.tight_layout()
out = out_dir / "endtoend_scaleT.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")
