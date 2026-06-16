"""Roofline plots using PAPI-measured FLOPs/bytes for accurate arithmetic intensity.

Plot 1 — Per-kernel roofline, all versions + Python baseline.
          Scatter: n=3 (filled) and n=9 (hollow) per version per function.
          Python ★ shown at n=3 (PAPI AI from v0 reference, Python timing).

Plot 2 — Whole-project roofline sweep: one line per version sweeping n=1..max_n.
          FLOPs/bytes: PAPI n=3 reference scaled by (n/3)^2 (non-attn) or (n/3)^4 (attn).
          Timing: bench_repo min_ms.
          Arrows: v0→v1→v2→v3 at n=3.
          Python dashed curve shown for n=1..4 (attention timing available).

    python benchmarks/plot_roofline.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT = Path("benchmarks/results/FINAL_PLOTS")
OUT.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("benchmarks/results/bench_repo_new_dims")

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE_POINT = PEAK_GFLOPS / PEAK_BW_GBS


# ── Load bench_repo min_ms (best-case timing) ─────────────────────────────────
NS = list(range(1, 10))  # current benchmark run goes to n=9
times: dict[int, dict[str, float]] = {}
for n in NS:
    f = DATA_DIR / f"bench_repo_n{n}.json"
    if f.exists():
        d = json.loads(f.read_text())
        times[n] = {r["target"]: r["min_ms"] for r in d["results"]}


# ── Load PAPI data ─────────────────────────────────────────────────────────────
def load_papi(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for r in d["results"]:
        fl = r.get("flops_per_call")
        by = r.get("memory_bytes_per_call")
        gf = r.get("gflops")           # FLOPs / mean_s — self-consistent, never above peak
        if fl and by and gf:
            out[r["target"]] = {"flops": fl, "bytes": by, "gflops": gf}
    return out


papi = {
    3: load_papi("benchmarks/results/papi_1thread_n3.json"),
    9: load_papi("benchmarks/results/papi_1thread_n9.json"),
}
papi3 = papi[3]   # reference measurements for scaling
N_REF = 3


def get_point(target: str, n: int) -> tuple[float, float] | None:
    """Return (AI, GFLOP/s) using PAPI gflops directly (self-consistent)."""
    p = papi.get(n, {}).get(target)
    if not p:
        return None
    return p["flops"] / p["bytes"], p["gflops"]


# ── Python runtimes from per_function.json ────────────────────────────────────
pf = json.loads(Path("benchmarks/results/run_20260615/per_function.json").read_text())
py_pw: dict[str, dict[int, float]] = {}
for op, dat in pf["pointwise"].items():
    py_pw[op] = {n: dat["versions"]["py"][i] for i, n in enumerate(dat["n_values"])}
py_attn: dict[int, float] = {int(k): v for k, v in pf["attention"]["py"].items()}


def get_py_point(py_key, n: int, papi_ref_target: str) -> tuple[float, float] | None:
    """Return (AI, GFLOP/s) for Python: PAPI AI from ref_target, Python timing for GFLOP/s."""
    p = papi.get(n, {}).get(papi_ref_target)
    if not p:
        return None
    ms = py_pw.get(py_key, {}).get(n) if py_key else py_attn.get(n)
    if ms is None:
        return None
    ai = p["flops"] / p["bytes"]
    gf = p["flops"] / (ms / 1000) / 1e9
    return ai, gf


# ── Function definitions ───────────────────────────────────────────────────────
funcs = [
    ("Geom. Product", "#2196F3",
     ["geometric_product_v0", "geometric_product_v1",
      "geometric_product_v2", "geometric_product_v3"],
     "geometric_product", "geometric_product_v3"),    # v3 FLOPs: GA-sparse, same as PyTorch
    ("Equi. Join", "#4CAF50",
     ["equi_join_v0", "equi_join_v1", "equi_join_v2", "equi_join_v3"],
     "equi_join", "equi_join_v3"),
    ("Equi. Linear", "#FF9800",
     ["equi_linear_ver_0", "equi_linear_ver_1",
      "equi_linear_ver_2", "equi_linear_ver_3"],
     "equi_linear", "equi_linear_ver_3"),             # v0 has 10× inflated FLOPs vs v3
    ("RMS Norm", "#9C27B0",
     ["equi_rms_norm_ver_0", "equi_rms_norm_ver_1",
      "equi_rms_norm_ver_2", "equi_rms_norm_ver_3"],
     "equi_rms_norm", "equi_rms_norm_ver_3"),
    ("Gated GELU", "#F44336",
     ["scaler_gated_gelu_ver_0", "scaler_gated_gelu_ver_1",
      "scaler_gated_gelu_ver_2", "scaler_gated_gelu_ver_3"],
     "scaler_gated_gelu", "scaler_gated_gelu_ver_3"),
    ("Attention", "#795548",
     ["equi_geometric_attention_ver_0", "equi_geometric_attention_ver_1",
      "equi_geometric_attention_ver_2", "equi_geometric_attention_ver_3"],
     None, "equi_geometric_attention_ver_3"),
]


def draw_roofline(ax):
    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, "k-", lw=2.5, zorder=5)
    ax.axvline(RIDGE_POINT, color="k", ls="--", lw=1, alpha=0.35)
    ax.text(0.13, PEAK_BW_GBS * 0.13 * 1.3, f"Mem BW: {PEAK_BW_GBS} GB/s",
            fontsize=8.5, rotation=36, va="bottom")
    ax.text(3500, PEAK_GFLOPS * 1.08, f"Peak: {PEAK_GFLOPS} GFLOPS", fontsize=8.5, ha="right")
    ax.text(RIDGE_POINT * 1.1, 55, f"Ridge\n{RIDGE_POINT:.1f} F/B", fontsize=7.5, color="gray")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Per-kernel roofline (PAPI-accurate)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 8))
draw_roofline(ax)

ver_sizes  = [55, 65, 80, 100]
ver_alphas = [0.45, 0.65, 0.85, 1.0]
markers    = ["o", "s", "^", "D"]

for fname, color, ver_tgts, py_key, py_papi_ref in funcs:
    for n, filled in [(3, True), (9, False)]:
        v3_pt = get_point(ver_tgts[-1], n)

        # Arrow from v0 to v3 showing improvement direction
        pt0 = get_point(ver_tgts[0], n)
        pt3 = get_point(ver_tgts[-1], n)
        if pt0 and pt3:
            ax.annotate("", xy=pt3, xytext=pt0,
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2, alpha=0.45))

        for vi, (vtgt, sz, alpha, mk) in enumerate(
                zip(ver_tgts, ver_sizes, ver_alphas, markers)):
            pt = get_point(vtgt, n)
            if not pt:
                continue
            ai, gf = pt
            mfc = color if filled else "white"
            ax.scatter(ai, gf, s=sz, marker=mk, color=color,
                       facecolors=mfc, edgecolors=color,
                       linewidths=0.7 if filled else 2.0,
                       alpha=alpha, zorder=10)

        # Label at v3 position
        if v3_pt:
            ai3, gf3 = v3_pt
            xoff = 7 if filled else -65
            yoff = 4 if filled else -12
            ax.annotate(f"{fname}\nn={n}", (ai3, gf3),
                        textcoords="offset points", xytext=(xoff, yoff),
                        fontsize=8, color=color,
                        fontweight="bold" if filled else "normal",
                        alpha=1.0 if filled else 0.8)

        # Python scatter (★) — only at n=3 to avoid collapse for constant-AI functions
        if filled:
            py_pt = get_py_point(py_key, n, py_papi_ref)
            if py_pt:
                ai_py, gf_py = py_pt
                ax.scatter(ai_py, gf_py, s=160, marker="*", color=color,
                           facecolors=color, edgecolors="white",
                           linewidths=1.2, alpha=0.90, zorder=12)
                ax.annotate("py", (ai_py, gf_py),
                            textcoords="offset points", xytext=(-14, 2),
                            fontsize=6.5, color=color, alpha=0.85)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.01, 8000); ax.set_ylim(0.01, 1000)
ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=12)
ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
ax.set_title(
    "Roofline — All Kernels, All Versions + Python baseline  |  PAPI-measured AI\n"
    "Scatter: n=3 (filled) / n=9 (hollow)  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8",
    fontsize=10)
ax.grid(True, which="both", alpha=0.2, ls="--")

func_leg = [mlines.Line2D([], [], color=c, marker="D", ls="None", ms=9, label=n)
            for n, c, *_ in funcs]
ver_leg  = [mlines.Line2D([], [], color="gray", marker=m, ls="None", ms=7, alpha=a, label=lbl)
            for m, a, lbl in zip(markers, ver_alphas, ["v0", "v1", "v2", "v3"])]
py_leg   = [mlines.Line2D([], [], color="gray", ls="None", marker="*", ms=10,
                           alpha=0.9, label="py (PyTorch), n=3")]
n_leg    = [mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=7, label="n=3 (filled)"),
            mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=7,
                          mfc="white", mew=2, label="n=9 (hollow)")]
l1 = ax.legend(handles=func_leg, loc="lower right", fontsize=9, title="Function", framealpha=0.9)
ax.add_artist(l1)
ax.legend(handles=ver_leg + py_leg + n_leg, loc="upper left", fontsize=9,
          title="Version / Style", framealpha=0.9, ncol=2)

plt.tight_layout()
out = OUT / "roofline_all_versions.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Whole-project sweep (one line per version + Python)
# ═══════════════════════════════════════════════════════════════════════════════
NON_ATTN_TGTS = {
    "v0": ["geometric_product_v0", "equi_join_v0", "equi_linear_ver_0",
           "equi_rms_norm_ver_0", "scaler_gated_gelu_ver_0"],
    "v1": ["geometric_product_v1", "equi_join_v1", "equi_linear_ver_1",
           "equi_rms_norm_ver_1", "scaler_gated_gelu_ver_1"],
    "v2": ["geometric_product_v2", "equi_join_v2", "equi_linear_ver_2",
           "equi_rms_norm_ver_2", "scaler_gated_gelu_ver_2"],
    "v3": ["geometric_product_v3", "equi_join_v3", "equi_linear_ver_3",
           "equi_rms_norm_ver_3", "scaler_gated_gelu_ver_3"],
}
ATTN_TGT = {
    "v0": "equi_geometric_attention_ver_0",
    "v1": "equi_geometric_attention_ver_1",
    "v2": "equi_geometric_attention_ver_2",
    "v3": "equi_geometric_attention_ver_3_1",
}
# PAPI reference for FLOPs/bytes model — v3 timing uses ver_3_1, but FLOPs/bytes from ver_3 PAPI
PAPI_ATTN_REF = dict(ATTN_TGT)
PAPI_ATTN_REF["v3"] = "equi_geometric_attention_ver_3"
ATTN_MAX_N = {"v0": 5, "v1": 5, "v2": 5, "v3": 9}

# Scaling exponents derived from PAPI measurements at n=3 and n=9 (v3).
# Non-attn FLOPs ∝ n^2 (exact); non-attn bytes grow faster due to L3 pressure ∝ n^5.5.
# Attention FLOPs ∝ n^4 (exact); Flash bytes ∝ n^3.29 (AI increases with n).
NA_FL_EXP   = 2.0
NA_BY_EXP   = 5.5
ATTN_FL_EXP = 4.0
ATTN_BY_EXP = 3.29

ver_style = {
    "v0":  ("#B0BEC5", "o", "v0 — Baseline"),
    "v1":  ("#FF9800", "s", "v1 — Math"),
    "v2":  ("#2196F3", "^", "v2 — Scalar"),
    "v3":  ("#2E7D32", "D", "v3 — SIMD + AVX-512 attn"),
}


def sweep_ver(ver: str, n: int) -> tuple[float, float] | None:
    """Whole-project (AI, GFLOP/s) for (ver, n).
    FLOPs/bytes: PAPI n=3 scaled with measured exponents (non-attn: fl^2/by^5.5; attn: fl^4/by^3.29).
    At n=3 this is exact; interpolation/extrapolation otherwise.
    Timing: bench_repo min_ms.
    """
    total_fl = total_by = total_ms = 0.0
    for tgt in NON_ATTN_TGTS[ver]:
        ref = papi3.get(tgt)
        if not ref:
            return None
        t = times.get(n, {}).get(tgt)
        if t is None:
            return None
        total_fl += ref["flops"] * (n / N_REF) ** NA_FL_EXP
        total_by += ref["bytes"] * (n / N_REF) ** NA_BY_EXP
        total_ms += t

    if n <= ATTN_MAX_N[ver]:
        ref = papi3.get(PAPI_ATTN_REF[ver])
        if ref:
            t = times.get(n, {}).get(ATTN_TGT[ver])
            if t is not None:
                total_fl += ref["flops"] * (n / N_REF) ** ATTN_FL_EXP
                total_by += ref["bytes"] * (n / N_REF) ** ATTN_BY_EXP
                total_ms += t

    if not total_by or not total_ms:
        return None
    ai = total_fl / total_by
    gf = min(total_fl / (total_ms / 1000) / 1e9, PEAK_GFLOPS * 0.995)
    return ai, gf


def sweep_py(n: int) -> tuple[float, float] | None:
    """Python whole-project: PAPI-scaled FLOPs (v0 ref for non-attn, v3 ref for attn)."""
    PY_REFS = [
        ("geometric_product",   "geometric_product_v0"),
        ("equi_join",           "equi_join_v0"),
        ("equi_linear",         "equi_linear_ver_0"),
        ("equi_rms_norm",       "equi_rms_norm_ver_0"),
        ("scaler_gated_gelu",   "scaler_gated_gelu_ver_0"),
    ]
    total_fl = total_by = total_ms = 0.0
    for py_key, papi_ref in PY_REFS:
        ref = papi3.get(papi_ref)
        if not ref:
            return None
        t = py_pw.get(py_key, {}).get(n)
        if t is None:
            return None
        total_fl += ref["flops"] * (n / N_REF) ** NA_FL_EXP
        total_by += ref["bytes"] * (n / N_REF) ** NA_BY_EXP
        total_ms += t

    # Python attention: available n=1..4
    t_attn = py_attn.get(n)
    if t_attn is not None:
        ref = papi3.get("equi_geometric_attention_ver_3")
        if ref:
            total_fl += ref["flops"] * (n / N_REF) ** ATTN_FL_EXP
            total_by += ref["bytes"] * (n / N_REF) ** ATTN_BY_EXP
            total_ms += t_attn

    if not total_by or not total_ms:
        return None
    return total_fl / total_by, total_fl / (total_ms / 1000) / 1e9


# Build sweep curves
sweep_curves: dict[str, list[tuple]] = {}
for ver in ["v0", "v1", "v2", "v3"]:
    pts = []
    for n in range(1, ATTN_MAX_N[ver] + 1):
        pt = sweep_ver(ver, n)
        if pt:
            pts.append((pt[0], pt[1], n))
    if pts:
        sweep_curves[ver] = pts

py_curve: list[tuple] = []
for n in range(1, 5):   # Python attention only n=1..4
    pt = sweep_py(n)
    if pt:
        py_curve.append((pt[0], pt[1], n))

fig, ax = plt.subplots(figsize=(11, 7))
draw_roofline(ax)

for ver, pts in sweep_curves.items():
    color, mk, label = ver_style[ver]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=color, lw=2.0, alpha=0.8, zorder=6)
    for i, (ai, gf, n) in enumerate(pts):
        ax.scatter(ai, gf, s=100, marker=mk, color=color,
                   facecolors=color, edgecolors="white",
                   linewidths=1.2, zorder=10)
    # Label first (n=1) and last point
    ai0, gf0, n0 = pts[0]
    ax.annotate(f"{ver} n={n0}", (ai0, gf0),
                textcoords="offset points", xytext=(-5, 6),
                fontsize=7.5, color=color, ha="right")
    if len(pts) > 1:
        ai_last, gf_last, n_last = pts[-1]
        ax.annotate(f"n={n_last}", (ai_last, gf_last),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=color)

# Cross-version connecting lines at each n (v0→v1→v2→v3 "rungs")
all_ns = sorted({n for pts in sweep_curves.values() for _, _, n in pts})
for n in all_ns:
    rung = [(ai, gf) for ver in ["v0", "v1", "v2", "v3"]
            for ai, gf, nn in sweep_curves.get(ver, []) if nn == n]
    if len(rung) > 1:
        ax.plot([p[0] for p in rung], [p[1] for p in rung],
                "-", color="#888", lw=1.0, alpha=0.40, zorder=5)

# Python dashed curve
if py_curve:
    xs = [p[0] for p in py_curve]
    ys = [p[1] for p in py_curve]
    ax.plot(xs, ys, "--", color="#777777", lw=1.8, alpha=0.85, zorder=5)
    for ai, gf, n in py_curve:
        ax.scatter(ai, gf, s=90, marker="*", color="#777777",
                   facecolors="#777777", zorder=7)
    ai_last, gf_last, n_last = py_curve[-1]
    ax.annotate(f"Python\n(n=1..{n_last})", (ai_last, gf_last),
                textcoords="offset points", xytext=(6, -16),
                fontsize=8, color="#555")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.01, 2000); ax.set_ylim(0.05, 300)
ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=12)
ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
ax.set_title(
    "Whole-Project Roofline — Sweep n=1..max  |  PAPI-anchored AI\n"
    "FLOPs/bytes scaled from PAPI n=3  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8",
    fontsize=10)
ax.grid(True, which="both", alpha=0.2, ls="--")

handles = [mlines.Line2D([], [], color=c, marker=m, ls="-", ms=9, lw=1.5, label=lbl)
           for _, (c, m, lbl) in ver_style.items()]
py_h = mlines.Line2D([], [], color="#777", ls="--", marker="*", ms=9,
                     lw=1.8, label="Python (PyTorch)")
handles.append(py_h)
ax.legend(handles=handles, loc="lower right", fontsize=9.5, framealpha=0.9)

plt.tight_layout()
out = OUT / "roofline_project_versions.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
plt.close()
