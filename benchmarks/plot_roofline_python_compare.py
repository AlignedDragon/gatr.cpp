"""
Python-vs-C++ corrected roofline plots (in FINAL_PLOTS/roofline_corrected/).

Convention: Python is placed at the same arithmetic intensity as v3 C++ (same
canonical work W = v3 FLOPs/bytes).  GF/s = v3_flops / py_timing.
This shows "given the same mathematical work, how fast is Python vs C++?"
placing both at the same AI so the vertical gap is pure implementation speed.

  Plot 1 — roofline_py_vs_v3.png
      All 6 kernels.  Python (★) and v3 (◆) at n=3; v3 also at n=9 (hollow).
      Arrow Python → v3 shows per-kernel speedup.

  Plot 2 — roofline_py_stable_w_versions.png
      Stable-W kernels only (rms_norm, gelu, attention).
      Python (★) plus every C++ version v0→v3 at n=3.
      Full optimisation path from Python through to best C++.
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
    return {r["target"]: r
            for r in json.loads(p.read_text())["results"]}


papi3 = load_papi("benchmarks/results/papi_1thread_n3.json")
papi9 = load_papi("benchmarks/results/papi_1thread_n9.json")

pf_path = Path("benchmarks/results/run_20260615/per_function.json")
if not pf_path.exists():
    raise SystemExit("per_function.json not found — cannot draw Python comparison")
pf = json.loads(pf_path.read_text())

N3_IDX = 2  # index in n_values=[1,2,3,...] for n=3

# Python timings at n=3 (ms)
py_ms = {
    "geometric_product":        pf["pointwise"]["geometric_product"]["versions"]["py"][N3_IDX],
    "equi_join":                pf["pointwise"]["equi_join"]["versions"]["py"][N3_IDX],
    "equi_linear":              pf["pointwise"]["equi_linear"]["versions"]["py"][N3_IDX],
    "equi_rms_norm":            pf["pointwise"]["equi_rms_norm"]["versions"]["py"][N3_IDX],
    "scaler_gated_gelu":        pf["pointwise"]["scaler_gated_gelu"]["versions"]["py"][N3_IDX],
    "equi_geometric_attention": float(pf["attention"]["py"]["3"]),
}

# v3 PAPI targets used as FLOPs/bytes reference for Python
V3_PAPI_REF = {
    "geometric_product":        "geometric_product_v3",
    "equi_join":                "equi_join_v3",
    "equi_linear":              "equi_linear_ver_3",
    "equi_rms_norm":            "equi_rms_norm_ver_3",
    "scaler_gated_gelu":        "scaler_gated_gelu_ver_3",
    "equi_geometric_attention": "equi_geometric_attention_ver_3",
}

# All version targets per function (index = version: 0=v0…3=v3)
ALL_TARGETS = {
    "geometric_product":        ["geometric_product_v0",             "geometric_product_v1",
                                 "geometric_product_v2",             "geometric_product_v3"],
    "equi_join":                ["equi_join_v0",                     "equi_join_v1",
                                 "equi_join_v2",                     "equi_join_v3"],
    "equi_linear":              ["equi_linear_ver_0",                "equi_linear_ver_1",
                                 "equi_linear_ver_2",                "equi_linear_ver_3"],
    "equi_rms_norm":            ["equi_rms_norm_ver_0",              "equi_rms_norm_ver_1",
                                 "equi_rms_norm_ver_2",              "equi_rms_norm_ver_3"],
    "scaler_gated_gelu":        ["scaler_gated_gelu_ver_0",          "scaler_gated_gelu_ver_1",
                                 "scaler_gated_gelu_ver_2",          "scaler_gated_gelu_ver_3"],
    "equi_geometric_attention": ["equi_geometric_attention_ver_0",   "equi_geometric_attention_ver_1",
                                 "equi_geometric_attention_ver_2",   "equi_geometric_attention_ver_3"],
}

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
VER_MARKERS = {"v0": "o", "v1": "s", "v2": "^", "v3": "D"}
VER_SIZES   = {"v0": 55,  "v1": 70,  "v2": 90,  "v3": 115}
VER_ALPHAS  = {"v0": 0.45,"v1": 0.65,"v2": 0.85,"v3": 1.0}
VER_LABELS  = {
    "v0": "v0 — Baseline C++",
    "v1": "v1 — Math",
    "v2": "v2 — Scalar mem.",
    "v3": "v3 — SIMD + AVX-512 attn",
}


def py_point(func: str) -> tuple[float, float] | None:
    """Return (AI, GF/s) for Python at n=3 using v3 FLOPs/bytes reference."""
    ref = papi3.get(V3_PAPI_REF[func])
    t   = py_ms.get(func)
    if not ref or not t:
        return None
    ai = ref["arithmetic_intensity"]
    gf = ref["flops_per_call"] / (t / 1000) / 1e9
    return ai, gf


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
# PLOT 1 — Python ★ vs v3 ◆, all kernels, n=3 (+ v3 n=9 hollow)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.5))
draw_roofline(ax)

for func in ALL_TARGETS:
    color  = FUNC_COLORS[func]
    label  = FUNC_LABELS[func]
    v3_tgt = ALL_TARGETS[func][3]

    # v3 at n=3 (filled diamond)
    p3 = papi3.get(v3_tgt)
    if p3:
        ai3, gf3 = p3["arithmetic_intensity"], p3["gflops"]
        ax.scatter(ai3, gf3, s=120, marker="D", color=color,
                   facecolors=color, edgecolors="white", linewidths=0.8, zorder=10)
        ax.annotate(f"{label}\nv3 n=3", (ai3, gf3),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=color, fontweight="bold")

    # v3 at n=9 (hollow diamond)
    p9 = papi9.get(v3_tgt)
    if p9:
        ai9, gf9 = p9["arithmetic_intensity"], p9["gflops"]
        ax.scatter(ai9, gf9, s=95, marker="D", color=color,
                   facecolors="white", edgecolors=color, linewidths=2.0, zorder=10)
        ax.annotate(f"{label}\nv3 n=9", (ai9, gf9),
                    textcoords="offset points", xytext=(7, -13),
                    fontsize=7.5, color=color, alpha=0.75)

    # Python at n=3 (★)
    pp = py_point(func)
    if pp:
        ai_py, gf_py = pp
        ax.scatter(ai_py, gf_py, s=200, marker="*", color=color,
                   facecolors=color, edgecolors="white", linewidths=0.8,
                   alpha=0.85, zorder=11)
        ax.annotate(f"{label}\nPython", (ai_py, gf_py),
                    textcoords="offset points", xytext=(-68, 2),
                    fontsize=7.5, color=color, alpha=0.9)

    # Arrow Python → v3 (same AI since same FLOPs ref, so arrow is vertical)
    if p3 and pp:
        ax.annotate("", xy=(ai3, gf3), xytext=(ai_py, gf_py),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.3, alpha=0.55))

# Legend
func_leg = [mpatches.Patch(color=FUNC_COLORS[f], label=FUNC_LABELS[f]) for f in ALL_TARGETS]
impl_leg = [
    mlines.Line2D([], [], color="gray", marker="*",  ls="None", ms=12, label="Python (PyTorch)  n=3"),
    mlines.Line2D([], [], color="gray", marker="D",  ls="None", ms=9,  mfc="gray",  label="v3 C++  n=3 (filled)"),
    mlines.Line2D([], [], color="gray", marker="D",  ls="None", ms=9,  mfc="white", mew=2, label="v3 C++  n=9 (hollow)"),
]
l1 = ax.legend(handles=func_leg, fontsize=9, loc="lower right", framealpha=0.9,
               title="Kernel", ncol=2)
ax.add_artist(l1)
ax.legend(handles=impl_leg, fontsize=9, loc="upper left", framealpha=0.9)

finalise(ax,
    "Roofline — Python (PyTorch) vs C++ v3  |  n=3  |  PAPI-measured AI  |  Tiger Lake i7-1165G7\n"
    "Python placed at v3 AI (same canonical W).  Arrows show vertical speedup gain.")
fig.tight_layout()
fig.savefig(OUT / "roofline_py_vs_v3.png", dpi=150, bbox_inches="tight")
print("Saved roofline_py_vs_v3.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Python + v0→v3, stable-W kernels only (rms_norm, gelu, attention)
# ─────────────────────────────────────────────────────────────────────────────
STABLE_W = ["equi_rms_norm", "scaler_gated_gelu", "equi_geometric_attention"]

fig, ax = plt.subplots(figsize=(11, 7))
draw_roofline(ax)

for func in STABLE_W:
    color = FUNC_COLORS[func]
    label = FUNC_LABELS[func]

    # Python ★
    pp = py_point(func)
    if pp:
        ax.scatter(pp[0], pp[1], s=220, marker="*", color=color,
                   facecolors=color, edgecolors="white", linewidths=0.8,
                   alpha=0.85, zorder=11)

    # C++ v0→v3 (sized by version)
    cpp_pts = []
    for vi, ver in enumerate(["v0", "v1", "v2", "v3"]):
        tgt = ALL_TARGETS[func][vi]
        p   = papi3.get(tgt)
        if not p:
            continue
        ai, gf = p["arithmetic_intensity"], p["gflops"]
        ax.scatter(ai, gf, s=VER_SIZES[ver], marker=VER_MARKERS[ver], color=color,
                   facecolors=color, edgecolors="white",
                   linewidths=0.8, alpha=VER_ALPHAS[ver], zorder=10)
        cpp_pts.append((ai, gf))

    # Connect Python → v0 → v1 → v2 → v3
    all_pts = ([pp] if pp else []) + cpp_pts
    if len(all_pts) > 1:
        ax.plot([p[0] for p in all_pts], [p[1] for p in all_pts],
                "-", color=color, lw=1.2, alpha=0.45, zorder=6)

    # Label at v3
    if cpp_pts:
        ai_v3, gf_v3 = cpp_pts[-1]
        ax.annotate(label, (ai_v3, gf_v3),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=9, color=color, fontweight="bold")

# Legend
func_leg = [mpatches.Patch(color=FUNC_COLORS[f], label=FUNC_LABELS[f]) for f in STABLE_W]
impl_leg = [
    mlines.Line2D([], [], color="gray", marker="*", ls="None", ms=12,
                  alpha=0.85, label="Python (PyTorch)"),
] + [
    mlines.Line2D([], [], color="gray", marker=VER_MARKERS[v], ls="None",
                  ms=8, alpha=VER_ALPHAS[v], label=VER_LABELS[v])
    for v in ["v0", "v1", "v2", "v3"]
]
l1 = ax.legend(handles=func_leg, fontsize=9, loc="lower right",
               framealpha=0.9, title="Function")
ax.add_artist(l1)
ax.legend(handles=impl_leg, fontsize=9, loc="upper left",
          framealpha=0.9, title="Implementation")

finalise(ax,
    "Roofline — Python → v0 → v1 → v2 → v3 optimisation path  |  stable-W kernels  |  n=3\n"
    "W changes <10% across all versions (including Python) → comparison valid  |  Tiger Lake i7-1165G7")
fig.tight_layout()
fig.savefig(OUT / "roofline_py_stable_w_versions.png", dpi=150, bbox_inches="tight")
print("Saved roofline_py_stable_w_versions.png")
plt.close()
