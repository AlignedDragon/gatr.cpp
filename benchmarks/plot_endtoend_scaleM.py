"""Plot the MV-only GATr end-to-end n-sweep (B=2n, H=4n, T=128n, C=8).

Shows GFLOP/s and speedups vs B=2n (batch/model-size focus).
Uses analytical FLOPs per row (since B, T, H, C all scale with n).

    python benchmarks/plot_endtoend_scaleM.py benchmarks/results/FINAL_PLOTS/endtoend_scaleM.json
    python benchmarks/plot_endtoend_scaleM.py <json> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/FINAL_PLOTS/endtoend_scaleM.json")
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
    "v3":  ("#2E7D32", "v3 (SIMD + AVX-512 attn)"),
}


def _eqlin(M, i, o):      return M * i * o * 16 * 9 * 2
def _rms(M, c):            return M * c * 64
def _gp(M, c):             return M * c * 384
def _join(M, c):           return M * c * 384
def _gelu(M, c):           return M * c * 128
def _attn(B, h, t, c):    return 56 * B * h * t * t * c + 496 * B * h * t * c


def model_flops(B, T, H, C, L):
    M = B * T
    embed   = _eqlin(M, 1, C)
    head    = _eqlin(M, C, 1)
    attn    = _rms(M, C) + _eqlin(M, C, C * H * 3) + _attn(B, H, T, C) + _eqlin(M, C * H, C)
    mlp     = (_rms(M, C) + _eqlin(M, C, C * 4) + _gp(M, C) + _join(M, C)
               + _eqlin(M, C * 2, C) + _gelu(M, C) + _eqlin(M, C, C))
    return embed + head + L * (attn + mlp)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: GFLOP/s vs B=2n for each version
ax = axes[0]
for v, (color, label) in VERSION_STYLE.items():
    xs, ys = [], []
    for r in rows:
        t = r["times_us"].get(v)
        if t is None or t != t:
            continue
        B, T, H, C, L = r["B"], r["T"], r["heads"], r["channels"], r.get("layers", layers)
        flops = model_flops(B, T, H, C, L)
        xs.append(B)
        ys.append(flops / (t * 1e3))  # t in µs → ns → GFLOP/s = flops/(ns*1e9) = flops/(µs*1e3*1e9)
    if xs:
        ax.plot(xs, ys, "o-", color=color, label=label, lw=1.8, ms=5)

ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size B=2n  (log₂)")
ax.set_ylabel("GFLOP/s")
ax.set_title(f"End-to-end performance  [B=2n, H=4n, T=128n, C=8, L={layers}]")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# Right: v3 speedup vs B
ax = axes[1]
s_ref = [(r["B"], r["speedup_v3_over_ref"]) for r in rows
         if r.get("speedup_v3_over_ref") == r.get("speedup_v3_over_ref") and r.get("speedup_v3_over_ref") is not None]
s_v0  = [(r["B"], r["speedup_v3_over_v0"])  for r in rows
         if r.get("speedup_v3_over_v0")  == r.get("speedup_v3_over_v0")  and r.get("speedup_v3_over_v0")  is not None]

def annotate(ax, pts, color, dy=6):
    for x, y in pts:
        if y == y:
            ax.annotate(f"{y:.0f}×" if y >= 10 else f"{y:.1f}×", (x, y),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=6.5, color=color, fontweight="bold")

if s_ref:
    ax.plot([x for x, _ in s_ref], [y for _, y in s_ref], "s-", color="#9C27B0", label="v3 / ref", lw=1.8, ms=5)
    annotate(ax, s_ref, "#9C27B0")
if s_v0:
    ax.plot([x for x, _ in s_v0], [y for _, y in s_v0], "^-", color="#B0BEC5", label="v3 / v0", lw=1.8, ms=5)
    annotate(ax, s_v0, "#B0BEC5", dy=-12)

ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size B=2n  (log₂)")
ax.set_ylabel("Speedup (×)")
ax.set_title("End-to-end: v3 speedup vs batch size")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("MV-only GATr — end-to-end forward, n-scaling (B=2n, H=4n, T=128n, C=8, single-core)", fontsize=12)
fig.tight_layout()
out = out_dir / "endtoend_scaleM.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")
