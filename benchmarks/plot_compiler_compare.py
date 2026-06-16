"""Plot the compiler comparison from bench_variant.py output.

Reads benchmarks/results/compilers.json (one label per compiler, e.g. "gcc",
"clang") and draws, reusing the per-function bar style:
  left  — per-function speedup over v0, all versions; compiler = solid/hatched
  right — end-to-end model runtime per version, grouped by compiler

Works with a single compiler present (just GCC) and automatically adds the
second series once a clang run is appended.

    python benchmarks/plot_compiler_compare.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/compilers.json")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())["labels"]
compilers = list(data)                       # e.g. ["gcc"] or ["gcc", "clang"]
HATCHES = ["", "//", "xx", ".."]             # one per compiler
# Always use GCC v0 as the common speedup baseline so cross-compiler
# comparisons are fair. Falls back to first compiler if gcc is absent.
_baseline_comp = "gcc" if "gcc" in data else compilers[0]

FAM_LABEL = {
    "geometric_product": "Geom.\nProduct", "equi_join": "Equi.\nJoin",
    "equi_linear": "Equi.\nLinear", "equi_rms_norm": "RMS\nNorm",
    "scaler_gated_gelu": "Gated\nGELU", "equi_geometric_attention": "Attention",
}
VC = {"v0": "#B0BEC5", "v1": "#FF9800", "v2": "#2196F3", "v3": "#2E7D32"}
VERS = ["v0", "v1", "v2", "v3"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [3, 1.3]})

# Left: per-function ABSOLUTE runtime (ms, log) per version; compiler = hatch.
# Absolute time is the honest cross-compiler metric — speedup-over-own-v0 would
# be distorted whenever the two compilers' v0 baselines differ.
ax = axes[0]
bw = 0.18
fam_gap = 0.5
comp_gap = 0.15
x = 0
xticks, xlabels = [], []
for fam in FAM_LABEL:
    baseline_v0 = data[_baseline_comp]["ops_ms"][fam]["v0"]
    for ci, comp in enumerate(compilers):
        ops = data[comp]["ops_ms"][fam]
        for vi, ver in enumerate(VERS):
            bx = x + ci * (4 * bw + comp_gap) + vi * bw
            ax.bar(bx, ops[ver], bw, color=VC[ver], edgecolor="white", lw=0.5,
                   hatch=HATCHES[ci % len(HATCHES)], zorder=3)
            # speedup over GCC v0 (common baseline) so cross-compiler comparison is fair
            sp = baseline_v0 / ops[ver] if ops[ver] else 1.0
            ax.annotate(f"{sp:.0f}×" if sp >= 10 else f"{sp:.1f}×", (bx, ops[ver]),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", rotation=90, fontsize=5.5, color="#222", fontweight="bold")
        mid = x + ci * (4 * bw + comp_gap) + 1.5 * bw
        xticks.append(mid)
        xlabels.append(f"{FAM_LABEL[fam]}\n{comp}")
    x += len(compilers) * (4 * bw + comp_gap) + fam_gap
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("Runtime (ms, log)", fontsize=11)
ax.set_title(f"Per-function runtime by compiler  [n={data[compilers[0]]['n']}, 1 thread]  |  speedup over GCC v0", fontsize=11)
ax.grid(axis="y", alpha=0.25, which="both", zorder=0)
ver_leg = [mpatches.Patch(color=VC[v], label=v) for v in VERS]
comp_leg = [mpatches.Patch(fc="white", ec="gray", hatch=HATCHES[i % len(HATCHES)], label=c)
            for i, c in enumerate(compilers)]
l1 = ax.legend(handles=ver_leg, loc="upper left", fontsize=9, title="Version", ncol=2)
ax.add_artist(l1)
ax.legend(handles=comp_leg, loc="upper center", fontsize=9, title="Compiler")

# Right: end-to-end speedup over GCC v0 (taller bar = faster = better).
# Using speedup instead of absolute runtime makes the GCC vs Clang winner
# obvious: GCC v3 bar (82×) is clearly taller than Clang v3 (75×).
ax = axes[1]
mvers = ["ref", "v0", "v1", "v2", "v3"]
bw = 0.35
baseline_model_v0 = data[_baseline_comp]["model_ms"]["v0"]
comp_colors = {"gcc": "#1565C0", "clang": "#E65100"}  # blue=gcc, orange=clang
for ci, comp in enumerate(compilers):
    m = data[comp]["model_ms"]
    xs = [i + ci * bw for i in range(len(mvers))]
    sps = [baseline_model_v0 / m[v] if m[v] else 0.0 for v in mvers]
    color = comp_colors.get(comp, "#555")
    ax.bar(xs, sps, bw, label=comp, hatch=HATCHES[ci % len(HATCHES)],
           edgecolor="white", lw=0.5, color=color, alpha=0.85, zorder=3)
    for xi, sp in zip(xs, sps):
        if sp >= 0.5:
            ax.annotate(f"{sp:.0f}×" if sp >= 10 else f"{sp:.1f}×", (xi, sp),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", rotation=90, fontsize=7, color="#111", fontweight="bold")
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}×" if y >= 2 else f"{y:.1f}×"))
ax.set_xticks([i + bw * (len(compilers) - 1) / 2 for i in range(len(mvers))])
ax.set_xticklabels(mvers, fontsize=9)
ax.set_ylabel("Speedup over GCC v0  (taller = faster)", fontsize=10)
ax.set_title(f"End-to-end model  |  speedup over GCC v0\n[B={data[compilers[0]]['model_batch']}, "
             f"T={data[compilers[0]]['model_T']}]", fontsize=10)
ax.grid(axis="y", alpha=0.25, which="both", zorder=0)
if len(compilers) > 1:
    ax.legend(fontsize=9, title="Compiler")

fig.suptitle("Compiler comparison — per-function & end-to-end, all versions (single-core)", fontsize=12)
fig.tight_layout()
out = out_dir / "compiler_compare.png"
fig.savefig(out, dpi=140)
print(f"saved: {out}")
