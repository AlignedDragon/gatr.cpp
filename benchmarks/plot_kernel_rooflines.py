"""Roofline plots for equi_join, attention, and end-to-end model.

Produces in FINAL_PLOTS/kernel_rooflines/:
  attention_py_vs_v0.png      Python vs v0 on roofline (n=3)
  attention_all_versions.png  all v0..v3 on roofline (n=3)
  join_py_vs_v0.png
  join_all_versions.png
  endtoend_roofline.png       full model GFLOP/s vs n on roofline

Usage:
    python benchmarks/plot_kernel_rooflines.py
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Hardware (Tiger Lake i7-1165G7, 1 thread) ────────────────────────────────
PEAK_GFLOPS  = 89.6
PEAK_BW_GBS  = 45.0
RIDGE        = PEAK_GFLOPS / PEAK_BW_GBS   # ~1.99 FLOP/byte

# ── Style ─────────────────────────────────────────────────────────────────────
VC = {
    "py":  ("#607D8B", "*", "Python"),
    "v0":  ("#B0BEC5", "o", "v0"),
    "v1":  ("#FF9800", "s", "v1"),
    "v2":  ("#2196F3", "^", "v2"),
    "v3":  ("#4CAF50", "D", "v3"),
    "ref": ("#9E9E9E", "P", "ref (Python)"),
}
MS   = 11   # marker size
LW   = 2.0  # line width

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})

OUT = Path("FINAL_PLOTS/kernel_rooflines")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
PAPI3  = {r["target"]: r
          for r in json.loads(Path("benchmarks/results/papi_1thread_n3.json")
                              .read_text())["results"]}

REPO   = {}   # REPO[target][n] = row
for n in range(1, 13):
    f = Path(f"benchmarks/results/bench_repo_new_dims/bench_repo_n{n}.json")
    for r in json.loads(f.read_text())["results"]:
        REPO.setdefault(r["target"], {})[n] = r

PF = json.loads(Path("benchmarks/results/run_20260615/per_function.json").read_text())
EE = json.loads(Path("benchmarks/results/endtoend/endtoend_scaleM.json").read_text())


# ── Helpers ───────────────────────────────────────────────────────────────────
def roofline_ceiling(ai_arr):
    return np.minimum(PEAK_GFLOPS, PEAK_BW_GBS * np.asarray(ai_arr))


def draw_roofline(ax):
    ai_lo, ai_hi = 1e-1, 1e4
    ai = np.logspace(math.log10(ai_lo), math.log10(ai_hi), 500)
    perf = roofline_ceiling(ai)
    ax.plot(ai, perf, "k-", lw=2, label="Roofline ceiling")
    ax.axvline(RIDGE, color="k", lw=0.9, ls=":", alpha=0.5)
    ax.text(RIDGE * 1.15, PEAK_GFLOPS * 0.52, f"Ridge\n{RIDGE:.1f} F/B",
            fontsize=10, va="center", color="gray")
    ax.axhline(PEAK_GFLOPS, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.text(ai_lo * 1.4, PEAK_GFLOPS * 1.06,
            f"Peak {PEAK_GFLOPS} GFLOP/s", fontsize=10, color="gray")


def point_from_papi(target, timing_ms=None):
    """Return (AI, GFLOP/s) using PAPI FLOPs/bytes and timing.
    Always uses PAPI min_s unless timing_ms explicitly given (Python baseline only)."""
    p = PAPI3[target]
    ai    = p["arithmetic_intensity"]
    flops = p["flops_per_call"]
    t_s   = (timing_ms / 1000.0) if timing_ms is not None else p["min_s"]
    return ai, flops / t_s / 1e9


def repo_min_ms(target, n=3):
    return REPO[target][n]["min_ms"]


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


# ── Generic roofline scatter plot ─────────────────────────────────────────────
def make_roofline_fig(title, subtitle):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    draw_roofline(ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=11)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e-1, 1e4); ax.set_ylim(5e-3, 200)
    return fig, ax


def add_point(ax, ai, gflops, key, label=None):
    col, mk, lbl = VC[key]
    lbl = label or lbl
    ax.scatter(ai, gflops, color=col, marker=mk, s=MS**2, zorder=5,
               edgecolors="k", linewidths=0.5, label=f"{lbl}  {gflops:.1f} GFLOP/s")
    ax.annotate(lbl, (ai, gflops), textcoords="offset points",
                xytext=(6, 3), fontsize=10, color=col, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# 1 & 2 — ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
ATTN_MAP = {
    "v0": "equi_geometric_attention_ver_0",
    "v1": "equi_geometric_attention_ver_1",
    "v2": "equi_geometric_attention_ver_2",
    "v3": "equi_geometric_attention_ver_3",
}
PY_ATTN_MS_N3 = float(PF["attention"]["py"]["3"])

print("Attention plots …")

# --- 1a. Python vs v0 ---------------------------------------------------------
fig, ax = make_roofline_fig(
    "Attention: Python vs v0",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, H=12, T=384, C=8)")

v0_ai, v0_gf = point_from_papi(ATTN_MAP["v0"])
py_gf = PAPI3[ATTN_MAP["v0"]]["flops_per_call"] / (PY_ATTN_MS_N3 / 1000) / 1e9
add_point(ax, v0_ai, py_gf, "py")
add_point(ax, v0_ai, v0_gf, "v0")

ax.legend(fontsize=8, loc="lower right")
save(fig, "attention_py_vs_v0.png")

# --- 1b. All versions ---------------------------------------------------------
fig, ax = make_roofline_fig(
    "Attention: all versions v0–v3",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, H=12, T=384, C=8)")

for ver, target in ATTN_MAP.items():
    ai, gf = point_from_papi(target)
    add_point(ax, ai, gf, ver)

ax.legend(fontsize=8, loc="lower right")
save(fig, "attention_all_versions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3 & 4 — EQUI_LINEAR
# ══════════════════════════════════════════════════════════════════════════════
LIN_MAP = {
    "v0": "equi_linear_ver_0",
    "v1": "equi_linear_ver_1",
    "v2": "equi_linear_ver_2",
    "v3": "equi_linear_ver_3",
}
PY_LIN_MS_N3 = float(PF["pointwise"]["equi_linear"]["versions"]["py"][2])

print("Linear plots …")

# --- 3a. Python vs v0 ---------------------------------------------------------
fig, ax = make_roofline_fig(
    "Equi-linear: Python vs v0",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, T=384, C=8)")

v0_ai, v0_gf = point_from_papi("equi_linear_ver_0")
py_gf = PAPI3["equi_linear_ver_0"]["flops_per_call"] / (PY_LIN_MS_N3 / 1000) / 1e9
add_point(ax, v0_ai, py_gf, "py")
add_point(ax, v0_ai, v0_gf, "v0")

ax.legend(fontsize=8, loc="lower right")
save(fig, "linear_py_vs_v0.png")

# --- 3b. All versions ---------------------------------------------------------
fig, ax = make_roofline_fig(
    "Equi-linear: all versions v0–v3",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, T=384, C=8)")

for ver, target in LIN_MAP.items():
    ai, gf = point_from_papi(target)
    add_point(ax, ai, gf, ver)

ax.legend(fontsize=8, loc="lower right")
save(fig, "linear_all_versions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — END-TO-END roofline
# ══════════════════════════════════════════════════════════════════════════════
print("End-to-end plot …")

# Estimate model FLOPs at n from PAPI n=3 data (2 layers, scale by n^exponent)
# non-attention kernels scale ~n^2 (B×T); attention scales ~n^4
# Kernel counts per layer (approximate GATr layer structure):
#   1× gp, 1× join, 3× linear (q/k/v proj + out), 1× rms_norm, 1× gelu, 1× attention
FLOP_EXP = {"non_attn": 2.0, "attn": 4.0}
REF_N = 3

def model_flops_n(ver_suffix, n):
    """Approximate total model FLOPs at n for the given version."""
    # non-attention kernels (scale n^2)
    non_attn = 0.0
    for tgt in [f"geometric_product_v{ver_suffix}",
                f"equi_join_v{ver_suffix}",
                f"equi_rms_norm_ver_{ver_suffix}",
                f"scaler_gated_gelu_ver_{ver_suffix}"]:
        if tgt in PAPI3:
            non_attn += PAPI3[tgt]["flops_per_call"]
    # linear: 3 calls per layer
    lin = PAPI3.get(f"equi_linear_ver_{ver_suffix}", {}).get("flops_per_call", 0)
    non_attn += 3 * lin
    # attention (scale n^4)
    attn_tgt = f"equi_geometric_attention_ver_{ver_suffix}"
    if attn_tgt not in PAPI3:
        attn_tgt = "equi_geometric_attention_ver_3"
    attn = PAPI3[attn_tgt]["flops_per_call"]

    layers = 2
    total = layers * (non_attn * (n / REF_N)**2 + attn * (n / REF_N)**4)
    return total

def model_bytes_n(ver_suffix, n):
    """Approximate total model bytes at n."""
    non_attn_b = 0.0
    for tgt in [f"geometric_product_v{ver_suffix}",
                f"equi_join_v{ver_suffix}",
                f"equi_rms_norm_ver_{ver_suffix}",
                f"scaler_gated_gelu_ver_{ver_suffix}"]:
        if tgt in PAPI3:
            non_attn_b += PAPI3[tgt]["memory_bytes_per_call"]
    lin_b = PAPI3.get(f"equi_linear_ver_{ver_suffix}", {}).get("memory_bytes_per_call", 0)
    non_attn_b += 3 * lin_b
    attn_tgt = f"equi_geometric_attention_ver_{ver_suffix}"
    if attn_tgt not in PAPI3:
        attn_tgt = "equi_geometric_attention_ver_3"
    attn_b = PAPI3[attn_tgt]["memory_bytes_per_call"]
    layers = 2
    return layers * (non_attn_b * (n / REF_N)**2 + attn_b * (n / REF_N)**3)

EE_VER = {"ref": "3", "v0": "0", "v1": "1", "v2": "2", "v3": "3"}

fig, ax = make_roofline_fig(
    "End-to-end GATr model (2 layers): all versions",
    "Tiger Lake i7-1165G7 | 1 thread | B=2n, H=4n, T=128n, C=8")

best_gf = {}  # ver -> (gf, n) at max n
for row in EE["rows"]:
    n = row["n"]
    for ver, times_us in row["times_us"].items():
        if times_us is None:
            continue
        vsuffix = EE_VER.get(ver, "3")
        flops = model_flops_n(vsuffix, n)
        byt   = model_bytes_n(vsuffix, n)
        ai    = flops / byt
        gf    = flops / (times_us / 1e6) / 1e9
        best_gf[ver] = (gf, n, ai)  # last n is max n since rows are ordered

plotted = {}
for row in EE["rows"]:
    n = row["n"]
    for ver, times_us in row["times_us"].items():
        if times_us is None:
            continue
        vsuffix = EE_VER.get(ver, "3")
        flops = model_flops_n(vsuffix, n)
        byt   = model_bytes_n(vsuffix, n)
        ai    = flops / byt
        gf    = flops / (times_us / 1e6) / 1e9
        key   = "ref" if ver == "ref" else ver
        col, mk, lbl = VC[key]
        if ver not in plotted:
            bgf, bn, _ = best_gf[ver]
            label = f"{lbl}  {bgf:.1f} GF/s (n={bn})"
        else:
            label = None
        ax.scatter(ai, gf, color=col, marker=mk, s=MS**2, zorder=5,
                   edgecolors="k", linewidths=0.5,
                   label=label)
        plotted[ver] = True

ax.legend(fontsize=8, loc="lower right")
save(fig, "endtoend_roofline.png")

print("\nDone. All plots saved to FINAL_PLOTS/kernel_rooflines/")
