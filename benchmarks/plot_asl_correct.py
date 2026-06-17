"""
ASL-guideline-correct performance plots.

ETH ASL rule (from course report guidelines):
  "Do not put two performance lines into the same plot if the operations
   count changed significantly (that's apples and oranges). In that case,
   first perform the optimizations that reduce op count and report the
   runtime gain in a plot. Then continue to optimize the best version and
   show performance plots."

This project has two kernel classes:

  Class A — W-shift: geometric_product, equi_join, equi_linear
      v0→v1 reduces FLOPs by 10-47×. Cannot compare v0 and v1+ in
      the same GFLOP/s plot.
      Plot 1: runtime (ms) vs n, v0-v3 — documents total gain including
              the op-count reduction step.
      Plot 2: GFLOP/s vs n, v1-v3 only — hardware-efficiency comparison
              after W is stable.
      Plot 3: Roofline at n=3, v1-v3 only — AI positioning vs limits.

  Class B — Stable W: equi_rms_norm, scaler_gated_gelu, attention
      FLOPs stable within ±10% across all versions.
      Plot 4: GFLOP/s vs n, v0-v3 — full optimization story, valid.
      Plot 5: Roofline at n=3, v0-v3 — full story on hardware limits.

Outputs saved to benchmarks/results/FINAL_PLOTS/asl_correct/
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

OUT = Path("benchmarks/results/FINAL_PLOTS/asl_correct")
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE = PEAK_GFLOPS / PEAK_BW_GBS
NS = list(range(1, 13))   # n = 1 .. 12

# ── Load PAPI data ────────────────────────────────────────────────────────────
def _papi(path):
    p = Path(path)
    return ({r["target"]: r for r in json.loads(p.read_text())["results"]}
            if p.exists() else {})

papi3 = _papi("benchmarks/results/papi_1thread_n3.json")
papi9 = _papi("benchmarks/results/papi_1thread_n9.json")

# ── Load bench_repo timing n=1..12 ────────────────────────────────────────────
def _timing():
    d = {}
    for n in NS:
        f = Path(f"benchmarks/results/bench_repo_new_dims/bench_repo_n{n}.json")
        if f.exists():
            d[n] = {r["target"]: r["min_ms"] for r in json.loads(f.read_text())["results"]}
    return d

timing = _timing()   # timing[n][target] = min_ms

# ── FLOPs scaling exponents (derived from PAPI n=3 and n=9) ──────────────────
# All non-attention: FLOPs ∝ n^2.00 (verified from PAPI)
# Attention:         FLOPs ∝ n^4.00 (verified from PAPI)
FL_EXP = {
    "geometric_product":        2.0,
    "equi_join":                2.0,
    "equi_linear":              2.0,
    "equi_rms_norm":            2.0,
    "scaler_gated_gelu":        2.0,
    "equi_geometric_attention": 4.0,
}

# ── Kernel metadata ───────────────────────────────────────────────────────────
FUNC_COLOR = {
    "geometric_product":        "#2196F3",
    "equi_join":                "#4CAF50",
    "equi_linear":              "#FF9800",
    "equi_rms_norm":            "#9C27B0",
    "scaler_gated_gelu":        "#F44336",
    "equi_geometric_attention": "#795548",
}
FUNC_LABEL = {
    "geometric_product":        "Geom. Product",
    "equi_join":                "Equi. Join",
    "equi_linear":              "Equi. Linear",
    "equi_rms_norm":            "RMS Norm",
    "scaler_gated_gelu":        "Gated GELU",
    "equi_geometric_attention": "Attention",
}
# Canonical PAPI reference target for each function (v3 W = canonical W)
PAPI_REF = {
    "geometric_product":        "geometric_product_v3",
    "equi_join":                "equi_join_v3",
    "equi_linear":              "equi_linear_ver_3",
    "equi_rms_norm":            "equi_rms_norm_ver_3",
    "scaler_gated_gelu":        "scaler_gated_gelu_ver_3",
    "equi_geometric_attention": "equi_geometric_attention_ver_3",
}
# bench_repo target names per version
TARGETS = {
    "geometric_product": {
        "v0": "geometric_product_v0", "v1": "geometric_product_v1",
        "v2": "geometric_product_v2", "v3": "geometric_product_v3",
    },
    "equi_join": {
        "v0": "equi_join_v0", "v1": "equi_join_v1",
        "v2": "equi_join_v2", "v3": "equi_join_v3",
    },
    "equi_linear": {
        "v0": "equi_linear_ver_0", "v1": "equi_linear_ver_1",
        "v2": "equi_linear_ver_2", "v3": "equi_linear_ver_3",
    },
    "equi_rms_norm": {
        "v0": "equi_rms_norm_ver_0", "v1": "equi_rms_norm_ver_1",
        "v2": "equi_rms_norm_ver_2", "v3": "equi_rms_norm_ver_3",
    },
    "scaler_gated_gelu": {
        "v0": "scaler_gated_gelu_ver_0", "v1": "scaler_gated_gelu_ver_1",
        "v2": "scaler_gated_gelu_ver_2", "v3": "scaler_gated_gelu_ver_3",
    },
    "equi_geometric_attention": {
        "v0": "equi_geometric_attention_ver_0", "v1": "equi_geometric_attention_ver_1",
        "v2": "equi_geometric_attention_ver_2", "v3": "equi_geometric_attention_ver_3_1",
    },
}
VER_STYLE = {
    "v0": dict(marker="o",  ms=7,   lw=1.3, ls="--",  alpha=0.55, label="v0 — Baseline C++"),
    "v1": dict(marker="s",  ms=7.5, lw=1.4, ls="-.",  alpha=0.70, label="v1 — Math opt."),
    "v2": dict(marker="^",  ms=8,   lw=1.6, ls=":",   alpha=0.85, label="v2 — Scalar/mem opt. (auto-vec)"),
    "v3": dict(marker="D",  ms=9,   lw=2.0, ls="-",   alpha=1.00, label="v3 — Explicit SIMD"),
}


def gflops_vs_n(func, ver):
    """Return (ns, gflops_list) using PAPI n=3 flops scaled by FL_EXP."""
    ref = papi3.get(PAPI_REF[func])
    if not ref:
        return [], []
    fl3 = ref["flops_per_call"]
    k   = FL_EXP[func]
    tgt = TARGETS[func][ver]
    xs, ys = [], []
    for n in NS:
        ms = timing.get(n, {}).get(tgt)
        if ms is None:
            continue
        fl_n = fl3 * (n / 3) ** k
        xs.append(n)
        ys.append(fl_n / (ms / 1000) / 1e9)
    return xs, ys


def ms_vs_n(func, ver):
    tgt = TARGETS[func][ver]
    xs, ys = [], []
    for n in NS:
        ms = timing.get(n, {}).get(tgt)
        if ms is not None:
            xs.append(n); ys.append(ms)
    return xs, ys


def draw_roofline(ax):
    ai = np.logspace(-2, 4, 500)
    ax.plot(ai, np.minimum(PEAK_BW_GBS * ai, PEAK_GFLOPS), "k-", lw=2.5, zorder=5)
    ax.axvline(RIDGE, color="k", ls="--", lw=1, alpha=0.35)
    ax.text(0.013, PEAK_BW_GBS * 0.013 * 1.3, f"Mem BW: {PEAK_BW_GBS} GB/s",
            fontsize=8.5, rotation=36, va="bottom")
    ax.text(3500, PEAK_GFLOPS * 1.08, f"Peak: {PEAK_GFLOPS} GFLOPS", fontsize=8.5, ha="right")
    ax.text(RIDGE * 1.1, PEAK_GFLOPS * 0.48, f"Ridge\n{RIDGE:.1f} F/B", fontsize=7.5, color="gray")


def finalise_roofline(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, 8000); ax.set_ylim(0.01, PEAK_GFLOPS * 2)
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=12)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.2, ls="--")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS A — W-SHIFT KERNELS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_A = ["geometric_product", "equi_join", "equi_linear"]
CLASS_B = ["equi_rms_norm", "scaler_gated_gelu", "equi_geometric_attention"]

# ── Plot 1: Runtime (ms) vs n, all versions, Class A ─────────────────────────
# Per ASL: "first perform the optimizations that reduce op count and report
# the runtime gain in a plot"
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, func in zip(axes, CLASS_A):
    color = FUNC_COLOR[func]
    for ver in ["v0", "v1", "v2", "v3"]:
        sty = VER_STYLE[ver]
        xs, ys = ms_vs_n(func, ver)
        if xs:
            ax.semilogy(xs, ys, color=color, marker=sty["marker"], ms=sty["ms"],
                        lw=sty["lw"], ls=sty["ls"], alpha=sty["alpha"],
                        label=sty["label"])

    # Annotate speedup at n=6
    ms_v0_n6 = timing.get(6, {}).get(TARGETS[func]["v0"])
    ms_v3_n6 = timing.get(6, {}).get(TARGETS[func]["v3"])
    if ms_v0_n6 and ms_v3_n6:
        su = ms_v0_n6 / ms_v3_n6
        ax.text(0.05, 0.05, f"v3 speedup\n(n=6): {su:.0f}×",
                transform=ax.transAxes, fontsize=9, color=color,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Problem size n", fontsize=11)
    ax.set_ylabel("Runtime [ms]", fontsize=11)
    ax.set_title(FUNC_LABEL[func], fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle(
    "Class A — W-shift kernels: Runtime vs problem size  |  All versions v0–v3\n"
    "Note: v0→v1 reduces FLOPs by 10–47× (algebraic elimination of zero-padded PGA elements).\n"
    "Because W changes, this runtime plot documents the total gain without implying a fair GFLOP/s comparison.",
    fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "asl_class_a_runtime.png", dpi=150, bbox_inches="tight")
print("Saved asl_class_a_runtime.png")
plt.close()


# ── Plot 2: GFLOP/s vs n, v1-v3 only, Class A ────────────────────────────────
# Per ASL: "Then continue to optimize the best version and show performance plots."
# v1 is the first algebraically-correct version. W is stable within v1-v3.
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, func in zip(axes, CLASS_A):
    color = FUNC_COLOR[func]
    for ver in ["v1", "v2", "v3"]:
        sty = VER_STYLE[ver]
        xs, ys = gflops_vs_n(func, ver)
        if xs:
            ax.plot(xs, ys, color=color, marker=sty["marker"], ms=sty["ms"],
                    lw=sty["lw"], ls=sty["ls"], alpha=sty["alpha"],
                    label=sty["label"])

    # Peak line
    ax.axhline(PEAK_GFLOPS, color="k", ls="--", lw=1.2, alpha=0.5,
               label=f"Peak ({PEAK_GFLOPS} GFLOPS)")

    # Best % of peak at n=3
    r3 = papi3.get(PAPI_REF[func])
    if r3:
        pct = r3["gflops"] / PEAK_GFLOPS * 100
        ax.text(0.97, 0.97, f"v3 best: {r3['gflops']:.1f} GF/s\n({pct:.0f}% of peak, n=3)",
                transform=ax.transAxes, fontsize=8.5, color=color,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Problem size n", fontsize=11)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=11)
    ax.set_title(FUNC_LABEL[func], fontsize=11, fontweight="bold")
    ax.set_ylim(0, PEAK_GFLOPS * 1.15)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle(
    "Class A — W-shift kernels: Performance [GFLOP/s] vs problem size  |  v1–v3 only  (apples-to-apples)\n"
    "FLOPs from PAPI n=3 reference scaled by n². v0 excluded: its W differs by 10–47× from v1+.\n"
    "Tiger Lake i7-1165G7  |  1 thread  |  B=2n H=4n T=128n C=8",
    fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "asl_class_a_gflops.png", dpi=150, bbox_inches="tight")
print("Saved asl_class_a_gflops.png")
plt.close()


# ── Plot 3: Roofline, v1-v3 only, Class A, n=3 + n=9 ────────────────────────
VER_ROOFLINE_A = {
    "v1": dict(marker="s", s=70,  alpha=0.70),
    "v2": dict(marker="^", s=90,  alpha=0.85),
    "v3": dict(marker="D", s=120, alpha=1.00),
}
PAPI_TARGETS = {  # per-function, per-version PAPI target names
    "geometric_product": {
        "v1": "geometric_product_v1", "v2": "geometric_product_v2", "v3": "geometric_product_v3",
    },
    "equi_join": {
        "v1": "equi_join_v1", "v2": "equi_join_v2", "v3": "equi_join_v3",
    },
    "equi_linear": {
        "v1": "equi_linear_ver_1", "v2": "equi_linear_ver_2", "v3": "equi_linear_ver_3",
    },
    "equi_rms_norm": {
        "v0": "equi_rms_norm_ver_0", "v1": "equi_rms_norm_ver_1",
        "v2": "equi_rms_norm_ver_2", "v3": "equi_rms_norm_ver_3",
    },
    "scaler_gated_gelu": {
        "v0": "scaler_gated_gelu_ver_0", "v1": "scaler_gated_gelu_ver_1",
        "v2": "scaler_gated_gelu_ver_2", "v3": "scaler_gated_gelu_ver_3",
    },
    "equi_geometric_attention": {
        "v0": "equi_geometric_attention_ver_0", "v1": "equi_geometric_attention_ver_1",
        "v2": "equi_geometric_attention_ver_2", "v3": "equi_geometric_attention_ver_3",
    },
}

fig, ax = plt.subplots(figsize=(12, 7.5))
draw_roofline(ax)

for func in CLASS_A:
    color = FUNC_COLOR[func]
    label = FUNC_LABEL[func]
    pts_n3, pts_n9 = [], []
    for ver, vsty in VER_ROOFLINE_A.items():
        tgt = PAPI_TARGETS[func].get(ver)
        if not tgt: continue
        p3 = papi3.get(tgt); p9 = papi9.get(tgt)
        if p3:
            ax.scatter(p3["arithmetic_intensity"], p3["gflops"],
                       s=vsty["s"], marker=vsty["marker"], color=color,
                       facecolors=color, edgecolors="white", lw=0.8,
                       alpha=vsty["alpha"], zorder=10)
            pts_n3.append((p3["arithmetic_intensity"], p3["gflops"], ver))
        if p9:
            ax.scatter(p9["arithmetic_intensity"], p9["gflops"],
                       s=vsty["s"] * 0.7, marker=vsty["marker"], color=color,
                       facecolors="white", edgecolors=color, lw=2.0,
                       alpha=vsty["alpha"] * 0.8, zorder=10)
            pts_n9.append((p9["arithmetic_intensity"], p9["gflops"], ver))

    # Connect v1→v2→v3 at n=3
    if len(pts_n3) > 1:
        ax.plot([p[0] for p in pts_n3], [p[1] for p in pts_n3],
                "-", color=color, lw=1.0, alpha=0.4, zorder=6)

    # Label v3
    p3_v3 = papi3.get(PAPI_TARGETS[func].get("v3", ""))
    if p3_v3:
        pct = p3_v3["gflops"] / PEAK_GFLOPS * 100
        ax.annotate(f"{label}\nv3: {p3_v3['gflops']:.0f} GF/s ({pct:.0f}%)",
                    (p3_v3["arithmetic_intensity"], p3_v3["gflops"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color, fontweight="bold")

# Legend
func_leg = [mpatches.Patch(color=FUNC_COLOR[f], label=FUNC_LABEL[f]) for f in CLASS_A]
ver_leg = [
    mlines.Line2D([], [], color="gray", marker=VER_ROOFLINE_A[v]["marker"], ls="None",
                  ms=8, alpha=VER_ROOFLINE_A[v]["alpha"], label=VER_STYLE[v]["label"])
    for v in ["v1", "v2", "v3"]
]
n_leg = [
    mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=8, mfc="gray", label="n=3 (filled)"),
    mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=7, mfc="white", mew=2, label="n=9 (hollow)"),
]
l1 = ax.legend(handles=func_leg, fontsize=9, loc="lower right", framealpha=0.9, title="Kernel")
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, fontsize=9, loc="upper left", framealpha=0.9, title="Version / size")

finalise_roofline(ax,
    "Class A — Roofline: geometric_product, equi_join, equi_linear  |  v1–v3 only  |  PAPI-measured AI\n"
    "v0 excluded (W differs 10–47×). v1 = first algebraically-correct version. "
    "Lines connect v1→v2→v3.  Tiger Lake i7-1165G7, 1 thread.")
fig.tight_layout()
fig.savefig(OUT / "asl_class_a_roofline.png", dpi=150, bbox_inches="tight")
print("Saved asl_class_a_roofline.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLASS B — STABLE-W KERNELS
# ─────────────────────────────────────────────────────────────────────────────

# ── Plot 4: GFLOP/s vs n, v0-v3, Class B ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, func in zip(axes, CLASS_B):
    color = FUNC_COLOR[func]
    for ver in ["v0", "v1", "v2", "v3"]:
        sty = VER_STYLE[ver]
        xs, ys = gflops_vs_n(func, ver)
        if xs:
            ax.plot(xs, ys, color=color, marker=sty["marker"], ms=sty["ms"],
                    lw=sty["lw"], ls=sty["ls"], alpha=sty["alpha"],
                    label=sty["label"])

    ax.axhline(PEAK_GFLOPS, color="k", ls="--", lw=1.2, alpha=0.5,
               label=f"Peak ({PEAK_GFLOPS} GFLOPS)")

    r3 = papi3.get(PAPI_REF[func])
    if r3:
        pct = r3["gflops"] / PEAK_GFLOPS * 100
        ax.text(0.05, 0.05, f"v3 best: {r3['gflops']:.1f} GF/s\n({pct:.0f}% of peak, n=3)",
                transform=ax.transAxes, fontsize=8.5, color=color,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Problem size n", fontsize=11)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=11)
    ax.set_title(FUNC_LABEL[func], fontsize=11, fontweight="bold")
    ax.set_ylim(0, PEAK_GFLOPS * 1.15)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle(
    "Class B — Stable-W kernels: Performance [GFLOP/s] vs problem size  |  All versions v0–v3\n"
    "FLOPs stable within ±10% across versions → cross-version GFLOP/s comparison is valid.\n"
    "Tiger Lake i7-1165G7  |  1 thread  |  B=2n H=4n T=128n C=8",
    fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "asl_class_b_gflops.png", dpi=150, bbox_inches="tight")
print("Saved asl_class_b_gflops.png")
plt.close()


# ── Plot 5: Roofline, v0-v3, Class B, n=3 + n=9 ─────────────────────────────
VER_ROOFLINE_B = {
    "v0": dict(marker="o", s=55,  alpha=0.45),
    "v1": dict(marker="s", s=70,  alpha=0.70),
    "v2": dict(marker="^", s=90,  alpha=0.85),
    "v3": dict(marker="D", s=120, alpha=1.00),
}

fig, ax = plt.subplots(figsize=(12, 7.5))
draw_roofline(ax)

for func in CLASS_B:
    color = FUNC_COLOR[func]
    label = FUNC_LABEL[func]
    pts_n3 = []
    for ver, vsty in VER_ROOFLINE_B.items():
        tgt = PAPI_TARGETS[func].get(ver)
        if not tgt: continue
        p3 = papi3.get(tgt); p9 = papi9.get(tgt)
        if p3:
            ax.scatter(p3["arithmetic_intensity"], p3["gflops"],
                       s=vsty["s"], marker=vsty["marker"], color=color,
                       facecolors=color, edgecolors="white", lw=0.8,
                       alpha=vsty["alpha"], zorder=10)
            pts_n3.append((p3["arithmetic_intensity"], p3["gflops"], ver))
        if p9:
            ax.scatter(p9["arithmetic_intensity"], p9["gflops"],
                       s=vsty["s"] * 0.7, marker=vsty["marker"], color=color,
                       facecolors="white", edgecolors=color, lw=2.0,
                       alpha=vsty["alpha"] * 0.8, zorder=10)

    if len(pts_n3) > 1:
        ax.plot([p[0] for p in pts_n3], [p[1] for p in pts_n3],
                "-", color=color, lw=1.2, alpha=0.45, zorder=6)

    p3_v3 = papi3.get(PAPI_TARGETS[func].get("v3", ""))
    if p3_v3:
        pct = p3_v3["gflops"] / PEAK_GFLOPS * 100
        ax.annotate(f"{label}\nv3: {p3_v3['gflops']:.0f} GF/s ({pct:.0f}%)",
                    (p3_v3["arithmetic_intensity"], p3_v3["gflops"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color, fontweight="bold")

func_leg = [mpatches.Patch(color=FUNC_COLOR[f], label=FUNC_LABEL[f]) for f in CLASS_B]
ver_leg = [
    mlines.Line2D([], [], color="gray", marker=VER_ROOFLINE_B[v]["marker"], ls="None",
                  ms=8, alpha=VER_ROOFLINE_B[v]["alpha"], label=VER_STYLE[v]["label"])
    for v in ["v0", "v1", "v2", "v3"]
]
n_leg = [
    mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=8, mfc="gray", label="n=3 (filled)"),
    mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=7, mfc="white", mew=2, label="n=9 (hollow)"),
]
l1 = ax.legend(handles=func_leg, fontsize=9, loc="lower right", framealpha=0.9, title="Kernel")
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, fontsize=9, loc="upper left", framealpha=0.9, title="Version / size")

finalise_roofline(ax,
    "Class B — Roofline: RMS Norm, Gated GELU, Attention  |  v0–v3  |  PAPI-measured AI\n"
    "W stable within ±10% across all versions → cross-version comparison is valid. "
    "Lines connect v0→v1→v2→v3.  Tiger Lake i7-1165G7, 1 thread.")
fig.tight_layout()
fig.savefig(OUT / "asl_class_b_roofline.png", dpi=150, bbox_inches="tight")
print("Saved asl_class_b_roofline.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY: v3-only roofline — all 6 kernels — final hardware utilisation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.5))
draw_roofline(ax)

ALL_FUNCS = CLASS_A + CLASS_B
for func in ALL_FUNCS:
    color = FUNC_COLOR[func]
    label = FUNC_LABEL[func]
    tgt3 = PAPI_TARGETS[func]["v3"]
    tgt9 = PAPI_TARGETS.get(func, {}).get("v3")
    p3 = papi3.get(tgt3); p9 = papi9.get(tgt3)
    if p3:
        ax.scatter(p3["arithmetic_intensity"], p3["gflops"],
                   s=130, marker="D", color=color,
                   facecolors=color, edgecolors="white", lw=0.8, zorder=10)
        pct = p3["gflops"] / PEAK_GFLOPS * 100
        ax.annotate(f"{label}\n{p3['gflops']:.0f} GF/s ({pct:.0f}%)",
                    (p3["arithmetic_intensity"], p3["gflops"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=color, fontweight="bold")
    if p9:
        ax.scatter(p9["arithmetic_intensity"], p9["gflops"],
                   s=90, marker="D", color=color,
                   facecolors="white", edgecolors=color, lw=2.0,
                   alpha=0.8, zorder=10)
    if p3 and p9:
        ax.plot([p3["arithmetic_intensity"], p9["arithmetic_intensity"]],
                [p3["gflops"], p9["gflops"]],
                "-", color=color, lw=0.9, alpha=0.35, zorder=6)

func_leg = [mpatches.Patch(color=FUNC_COLOR[f], label=FUNC_LABEL[f]) for f in ALL_FUNCS]
n_leg = [
    mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=9,
                  mfc="gray", label="n=3 (filled)"),
    mlines.Line2D([], [], color="gray", marker="D", ls="None", ms=8,
                  mfc="white", mew=2, label="n=9 (hollow)"),
]
ax.legend(handles=func_leg + n_leg, fontsize=9, loc="lower right",
          framealpha=0.9, ncol=2, title="Kernel / size")

finalise_roofline(ax,
    "Summary — Best version (v3) of each kernel  |  PAPI-measured AI  |  Tiger Lake i7-1165G7\n"
    "One consistent W per kernel. Lines show cache-pressure transition n=3→n=9. "
    "% of peak shown next to each kernel.")
fig.tight_layout()
fig.savefig(OUT / "asl_summary_v3.png", dpi=150, bbox_inches="tight")
print("Saved asl_summary_v3.png")
plt.close()

print("\nAll 6 plots written to", OUT)
