"""Plot attention v0-v3 speedup from benchmark_repo.py --json-out."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/attn_n1_main.json")
out_dir   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data    = json.loads(json_path.read_text())
results = {r["target"]: r for r in data["results"]}

versions = ["equi_geometric_attention_ver_0",
            "equi_geometric_attention_ver_1",
            "equi_geometric_attention_ver_2",
            "equi_geometric_attention_ver_3"]
labels  = ["v0\n(baseline)", "v1\n(cache-opt)", "v2\n(explicit DAA)", "v3\n(flash SDPA)"]
colors  = ["#B0BEC5", "#FF9800", "#2196F3", "#4CAF50"]

t0 = results[versions[0]]["min_ms"]
mins    = [results[v]["min_ms"]  for v in versions]
speedups = [t0 / t for t in mins]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: absolute runtime
ax = axes[0]
bars = ax.bar(labels, mins, color=colors, edgecolor="white", linewidth=0.8)
for b, val in zip(bars, mins):
    ax.text(b.get_x() + b.get_width()/2, val + 1, f"{val:.1f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Runtime (ms) — lower is better", fontsize=11)
ax.set_title(f"Attention: runtime  [n=1, T=16, 1-thread]", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(mins) * 1.18)

# Right: speedup over v0
ax = axes[1]
bars = ax.bar(labels, speedups, color=colors, edgecolor="white", linewidth=0.8)
for b, val in zip(bars, speedups):
    ax.text(b.get_x() + b.get_width()/2, val + 0.05, f"{val:.2f}×",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.4)
ax.set_ylabel("Speedup over v0 — higher is better", fontsize=11)
ax.set_title(f"Attention: speedup over v0  [n=1, T=16, 1-thread]", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(speedups) * 1.18)

fig.suptitle(f"equi_geometric_attention  (n=1, single-core)  —  v3 speedup: {speedups[-1]:.2f}×", fontsize=12)
fig.tight_layout()
out = out_dir / "attn_speedup_main.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")
