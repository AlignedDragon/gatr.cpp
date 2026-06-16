"""
Three corrected roofline plots that avoid the W-shift comparison problem.

  Plot 1 — roofline_v3_only.png
      v3 kernels only; n=3 filled, n=9 hollow.
      No cross-version comparison, so W is fixed for each kernel.
      Shows how kernels transition from compute-bound (n=3) to memory-bound (n=9).

  Plot 2 — roofline_v{0,1,2,3}_kernels.png  (4 plots)
      One plot per version; all kernels for that version; n=3 + n=9.
      Within a single version W is fixed per kernel — a valid hardware-utilisation snapshot.

  Plot 3 — roofline_stable_w.png
      Only the three kernels where W is stable across versions (<10% change):
      rms_norm, gelu, attention.  All four versions shown — cross-version comparison valid.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

OUT = Path("benchmarks/results/FINAL_PLOTS/roofline_corrected")
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE = PEAK_GFLOPS / PEAK_BW_GBS


def load_papi(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {
        r["target"]: {
            "ai": r.get("arithmetic_intensity", 0),
            "gf": r.get("gflops", 0),
            "fl": r.get("flops_per_call", 0),
            "by": r.get("memory_bytes_per_call", 0),
        }
        for r in d["results"]
    }


papi3 = load_papi("benchmarks/results/papi_1thread_n3.json")
papi9 = load_papi("benchmarks/results/papi_1thread_n9.json")

FUNC_COLORS = {
    "geometric_product":        "#2196F3",
    "equi_join":                "#4CAF50",
    "equi_linear":              "#FF9800",
    "equi_rms_norm":            "#9C27B0",
    "scaler_gated_gelu":        "#F44336",
    "equi_geometric_attention": "#795548",
}
FUNC_LABELS = {
    "geometric_product":        "Geom. Product",
    "equi_join":                "Equi. Join",
    "equi_linear":              "Equi. Linear",
    "equi_rms_norm":            "RMS Norm",
    "scaler_gated_gelu":        "Gated GELU",
    "equi_geometric_attention": "Attention",
}

# All version targets per function (index = version: 0→v0 … 3→v3)
ALL_TARGETS = {
    "geometric_product":        ["geometric_product_v0",              "geometric_product_v1",
                                 "geometric_product_v2",              "geometric_product_v3"],
    "equi_join":                ["equi_join_v0",                      "equi_join_v1",
                                 "equi_join_v2",                      "equi_join_v3"],
    "equi_linear":              ["equi_linear_ver_0",                  "equi_linear_ver_1",
                                 "equi_linear_ver_2",                  "equi_linear_ver_3"],
    "equi_rms_norm":            ["equi_rms_norm_ver_0",               "equi_rms_norm_ver_1",
                                 "equi_rms_norm_ver_2",               "equi_rms_norm_ver_3"],
    "scaler_gated_gelu":        ["scaler_gated_gelu_ver_0",           "scaler_gated_gelu_ver_1",
                                 "scaler_gated_gelu_ver_2",           "scaler_gated_gelu_ver_3"],
    "equi_geometric_attention": ["equi_geometric_attention_ver_0",    "equi_geometric_attention_ver_1",
                                 "equi_geometric_attention_ver_2",    "equi_geometric_attention_ver_3"],
}

VER_IDX    = {"v0": 0, "v1": 1, "v2": 2, "v3": 3}
VER_MARKERS = {"v0": "o", "v1": "s", "v2": "^", "v3": "D"}
VER_SIZES   = {"v0": 55,  "v1": 70,  "v2": 90,  "v3": 115}
VER_ALPHAS  = {"v0": 0.45,"v1": 0.65,"v2": 0.85,"v3": 1.0}
VER_LABELS  = {
    "v0": "v0 — Baseline",
    "v1": "v1 — Math",
    "v2": "v2 — Scalar mem.",
    "v3": "v3 — SIMD + AVX-512 attn",
}


def ver_target(func: str, ver: str) -> str:
    return ALL_TARGETS[func][VER_IDX[ver]]


def draw_roofline(ax):
    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, "k-", lw=2.5, zorder=5)
    ax.axvline(RIDGE, color="k", ls="--", lw=1, alpha=0.35)
    ax.text(0.013, PEAK_BW_GBS * 0.013 * 1.3, f"Mem BW: {PEAK_BW_GBS} GB/s",
            fontsize=8.5, rotation=36, va="bottom")
    ax.text(3500, PEAK_GFLOPS * 1.08, f"Peak: {PEAK_GFLOPS} GFLOPS", fontsize=8.5, ha="right")
    ax.text(RIDGE * 1.1, PEAK_GFLOPS * 0.48, f"Ridge\n{RIDGE:.1f} F/B", fontsize=7.5, color="gray")


def finalise(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, 8000); ax.set_ylim(0.01, PEAK_GFLOPS * 2)
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=12)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.2, ls="--")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — v3 only, all kernels, n=3 (filled) and n=9 (hollow)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
draw_roofline(ax)

for func in ALL_TARGETS:
    tgt   = ver_target(func, "v3")
    color = FUNC_COLORS[func]
    label = FUNC_LABELS[func]

    p3 = papi3.get(tgt)
    p9 = papi9.get(tgt)

    if p3:
        ax.scatter(p3["ai"], p3["gf"], s=115, marker="D", color=color,
                   facecolors=color, edgecolors="white", linewidths=0.8, zorder=10)
        ax.annotate(f"{label}\nn=3", (p3["ai"], p3["gf"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=color, fontweight="bold")
    if p9:
        ax.scatter(p9["ai"], p9["gf"], s=90, marker="D", color=color,
                   facecolors="white", edgecolors=color, linewidths=2.0, zorder=10)
        ax.annotate(f"{label}\nn=9", (p9["ai"], p9["gf"]),
                    textcoords="offset points", xytext=(7, -13),
                    fontsize=7.5, color=color, alpha=0.8)
    if p3 and p9:
        ax.plot([p3["ai"], p9["ai"]], [p3["gf"], p9["gf"]],
                "-", color=color, lw=1.0, alpha=0.35, zorder=6)

func_leg = [mpatches.Patch(color=FUNC_COLORS[f], label=FUNC_LABELS[f]) for f in ALL_TARGETS]
n_leg    = [mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=8,
                           mfc="gray",  mew=0.8, label="n=3 (filled)"),
            mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=8,
                           mfc="white", mew=2.0, label="n=9 (hollow)")]
ax.legend(handles=func_leg + n_leg, fontsize=9, loc="lower right",
          framealpha=0.9, ncol=2, title="Kernel / problem size")

finalise(ax,
    "Roofline — v3 only, all kernels  |  PAPI-measured AI  |  Tiger Lake i7-1165G7\n"
    "No cross-version W comparison.  Lines show cache-pressure transition n=3 → n=9.")
fig.tight_layout()
fig.savefig(OUT / "roofline_v3_only.png", dpi=150, bbox_inches="tight")
print("Saved roofline_v3_only.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — one plot per version (v0…v3), all kernels, n=3 filled + n=9 hollow
# ─────────────────────────────────────────────────────────────────────────────
for ver in ["v0", "v1", "v2", "v3"]:
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_roofline(ax)

    for func in ALL_TARGETS:
        tgt   = ver_target(func, ver)
        color = FUNC_COLORS[func]
        label = FUNC_LABELS[func]

        p3 = papi3.get(tgt)
        p9 = papi9.get(tgt)

        if p3:
            ax.scatter(p3["ai"], p3["gf"], s=115, marker="D", color=color,
                       facecolors=color, edgecolors="white", linewidths=0.8, zorder=10)
            ax.annotate(f"{label}\nn=3", (p3["ai"], p3["gf"]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, color=color, fontweight="bold")
        if p9:
            ax.scatter(p9["ai"], p9["gf"], s=90, marker="D", color=color,
                       facecolors="white", edgecolors=color, linewidths=2.0, zorder=10)
            ax.annotate(f"{label}\nn=9", (p9["ai"], p9["gf"]),
                        textcoords="offset points", xytext=(7, -13),
                        fontsize=7.5, color=color, alpha=0.8)
        if p3 and p9:
            ax.plot([p3["ai"], p9["ai"]], [p3["gf"], p9["gf"]],
                    "-", color=color, lw=1.0, alpha=0.35, zorder=6)

    func_leg = [mpatches.Patch(color=FUNC_COLORS[f], label=FUNC_LABELS[f]) for f in ALL_TARGETS]
    n_leg    = [mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=8,
                               mfc="gray",  mew=0.8, label="n=3 (filled)"),
                mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=8,
                               mfc="white", mew=2.0, label="n=9 (hollow)")]
    ax.legend(handles=func_leg + n_leg, fontsize=9, loc="lower right",
              framealpha=0.9, ncol=2, title="Kernel / problem size")

    finalise(ax,
        f"Roofline — {VER_LABELS[ver]}, all kernels  |  PAPI-measured AI  |  Tiger Lake i7-1165G7\n"
        f"W is fixed per kernel within this version — hardware utilisation snapshot.")
    fig.tight_layout()
    out_name = f"roofline_{ver}_kernels.png"
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    print(f"Saved {out_name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — stable-W kernels only (rms_norm, gelu, attention), all versions
# ─────────────────────────────────────────────────────────────────────────────
STABLE_W = ["equi_rms_norm", "scaler_gated_gelu", "equi_geometric_attention"]

fig, ax = plt.subplots(figsize=(11, 7))
draw_roofline(ax)

for func in STABLE_W:
    color = FUNC_COLORS[func]
    label = FUNC_LABELS[func]
    pts   = []

    for ver in ["v0", "v1", "v2", "v3"]:
        tgt = ver_target(func, ver)
        p   = papi3.get(tgt)
        if not p:
            continue
        ai, gf = p["ai"], p["gf"]
        ax.scatter(ai, gf,
                   s=VER_SIZES[ver], marker=VER_MARKERS[ver], color=color,
                   facecolors=color, edgecolors="white",
                   linewidths=0.8, alpha=VER_ALPHAS[ver], zorder=10)
        pts.append((ai, gf, ver))

    # Connect v0→v1→v2→v3 for this function
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "-", color=color, lw=1.2, alpha=0.45, zorder=6)

    # Label at v3 (last point)
    if pts:
        ai_last, gf_last, _ = pts[-1]
        ax.annotate(label, (ai_last, gf_last),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=9, color=color, fontweight="bold")

func_leg = [mpatches.Patch(color=FUNC_COLORS[f], label=FUNC_LABELS[f]) for f in STABLE_W]
ver_leg  = [
    mlines.Line2D([], [], color="gray", marker=VER_MARKERS[v], ls="None",
                  ms=8, alpha=VER_ALPHAS[v], label=VER_LABELS[v])
    for v in ["v0", "v1", "v2", "v3"]
]
l1 = ax.legend(handles=func_leg, fontsize=9, loc="lower right",
               framealpha=0.9, title="Function")
ax.add_artist(l1)
ax.legend(handles=ver_leg, fontsize=9, loc="upper left",
          framealpha=0.9, title="Version")

finalise(ax,
    "Roofline — stable-W kernels only: RMS Norm, Gated GELU, Attention  |  n=3  |  PAPI-measured AI\n"
    "W changes <10% across versions → cross-version comparison is valid  |  Tiger Lake i7-1165G7")
fig.tight_layout()
fig.savefig(OUT / "roofline_stable_w.png", dpi=150, bbox_inches="tight")
print("Saved roofline_stable_w.png")
plt.close()
