"""Paper figures for ASL team26 report.

All roofline points use PAPI FLOPs + PAPI min_s timing (turbo-on, n=3).
Memory-scaling uses bench_repo timing across n (consistent turbo-off sweep).

Output: FINAL_PLOTS/paper_plots/
  roofline_attention.jpeg   — v0–v3 on roofline (PAPI)
  roofline_equilinear.jpeg  — v1–v3 only (PAPI; v0 excluded per reviewer note)
  memory_scaling.jpeg       — GFLOP/s vs multivector count M for v3 kernels
  speedup_endtoend.jpeg     — end-to-end v3/v0 speedup vs n (actual + projected)

Usage:
    python benchmarks/plot_paper_figures.py
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Hardware ──────────────────────────────────────────────────────────────────
PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE       = PEAK_GFLOPS / PEAK_BW_GBS   # ~1.99 FLOP/byte
L2_BYTES    = 1280 * 1024                  # Tiger Lake i7-1165G7
L3_BYTES    = 12   * 1024 * 1024
BASE_GHZ    = 2.8                          # Tiger Lake base clock (turbo-off)

# ── Style ─────────────────────────────────────────────────────────────────────
VC = {
    "v0": ("#B0BEC5", "o", "v0"),
    "v1": ("#FF9800", "s", "v1"),
    "v2": ("#2196F3", "^", "v2"),
    "v3": ("#4CAF50", "D", "v3"),
}
MS = 11
plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

OUT = Path("FINAL_PLOTS/paper_plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
PAPI3 = {r["target"]: r
         for r in json.loads(Path("benchmarks/results/papi_1thread_n3.json").read_text())["results"]}

REPO = {}
for n in range(1, 13):
    f = Path(f"benchmarks/results/bench_repo_new_dims/bench_repo_n{n}.json")
    for r in json.loads(f.read_text())["results"]:
        REPO.setdefault(r["target"], {})[n] = r

EE = json.loads(Path("benchmarks/results/endtoend/endtoend_scaleM.json").read_text())


# ── Helpers ───────────────────────────────────────────────────────────────────
def roofline_ceiling(ai_arr):
    return np.minimum(PEAK_GFLOPS, PEAK_BW_GBS * np.asarray(ai_arr))


def draw_roofline(ax):
    ai = np.logspace(math.log10(0.1), math.log10(1e4), 500)
    ax.plot(ai, roofline_ceiling(ai), "k-", lw=2, label="Roofline ceiling")
    ax.axvline(RIDGE, color="k", lw=0.9, ls=":", alpha=0.5)
    ax.text(RIDGE * 1.15, PEAK_GFLOPS * 0.52, f"Ridge\n{RIDGE:.1f} F/B",
            fontsize=10, va="center", color="gray")
    ax.axhline(PEAK_GFLOPS, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.text(0.13, PEAK_GFLOPS * 1.06, f"Peak {PEAK_GFLOPS} GFLOP/s",
            fontsize=10, color="gray")


def papi_point(target):
    """Clock-honest: FLOP/cycle * BASE_GHZ, immune to turbo boost."""
    p = PAPI3[target]
    ai   = p["arithmetic_intensity"]
    cyc  = p["papi_counters_mean_total_region"]["PAPI_TOT_CYC"]
    flops = p["papi_counters_mean_total_region"]["PAPI_FP_OPS"]
    gf   = (flops / cyc) * BASE_GHZ
    return ai, gf


def make_roofline_fig(title, subtitle):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    draw_roofline(ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e-1, 1e4); ax.set_ylim(5e-3, 200)
    return fig, ax


def add_point(ax, ai, gflops, key, label=None):
    col, mk, lbl = VC[key]
    lbl = label or lbl
    ax.scatter(ai, gflops, color=col, marker=mk, s=MS**2, zorder=5,
               edgecolors="k", linewidths=0.5,
               label=f"{lbl}  {gflops:.1f} GFLOP/s")
    ax.annotate(lbl, (ai, gflops), textcoords="offset points",
                xytext=(6, 3), fontsize=10, color=col, fontweight="bold")


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 1 — roofline_attention.jpeg  (v0–v3, all PAPI)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 1: roofline_attention …")
ATTN = {
    "v0": "equi_geometric_attention_ver_0",
    "v1": "equi_geometric_attention_ver_1",
    "v2": "equi_geometric_attention_ver_2",
    "v3": "equi_geometric_attention_ver_3",
}
fig, ax = make_roofline_fig(
    "Attention: v0–v3",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, H=12, T=384, C=8)")
for ver, tgt in ATTN.items():
    add_point(ax, *papi_point(tgt), ver)
ax.legend(fontsize=9, loc="lower right")
save(fig, "roofline_attention.jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — roofline_equilinear.jpeg  (v1–v3 only; v0 excluded — different AI)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: roofline_equilinear …")
LIN = {
    "v1": "equi_linear_ver_1",
    "v2": "equi_linear_ver_2",
    "v3": "equi_linear_ver_3",
}
fig, ax = make_roofline_fig(
    "Equi-linear: v1–v3",
    "Tiger Lake i7-1165G7 | 1 thread | n=3 (B=6, T=384, C=8)")
for ver, tgt in LIN.items():
    add_point(ax, *papi_point(tgt), ver)
ax.legend(fontsize=9, loc="lower right")
save(fig, "roofline_equilinear.jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# 3 — memory_scaling.jpeg
#     GFLOP/s vs multivector count M = B*T = 256*n^2
#     Uses bench_repo timing (self-consistent sweep) + PAPI FLOPs scaled to n.
#     GFLOP/s at n=3 anchored to PAPI value; other n scaled by bench_repo ratio.
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: memory_scaling …")

KERN = {
    "gp":     ("geometric_product_v3",      2, "#4CAF50", "Geo. product v3"),
    "linear": ("equi_linear_ver_3",          2, "#2196F3", "Equi-linear v3"),
    "attn":   ("equi_geometric_attention_ver_3", 4, "#FF9800", "Attention v3"),
}

def gflops_at_n(tgt, exp, n_ref=3):
    """Clock-honest GFLOP/s at each n, using bench_repo timing ratios."""
    p = PAPI3[tgt]
    cyc   = p["papi_counters_mean_total_region"]["PAPI_TOT_CYC"]
    flops = p["papi_counters_mean_total_region"]["PAPI_FP_OPS"]
    papi_gf_n3 = (flops / cyc) * BASE_GHZ  # clock-honest anchor at n=3
    t_n3 = REPO[tgt][n_ref]["min_ms"] / 1000.0
    results = []
    for n in range(1, 13):
        row = REPO[tgt].get(n)
        if row is None or row.get("estimated"):
            continue
        t_n = row["min_ms"] / 1000.0
        # Anchored: at n=n_ref this equals papi_gf_n3; elsewhere scaled by bench_repo ratio and FLOPs
        gf = papi_gf_n3 * (n / n_ref) ** exp * (t_n3 / t_n)
        M  = 2 * n * 128 * n   # B*T = 256*n^2
        results.append((M, gf))
    return results

# L2/L3 boundary: working set ≈ 3 tensors of (B,T,C,16) float32
# bytes = 3 * B*T*C*64 = 3 * 256n^2 * 8 * 64 = 393216*n^2
WS_COEFF = 3 * 64 * 8  # bytes per B*T element (3 tensors, C=8 channels, 16*4 bytes each)
M_L2 = L2_BYTES / WS_COEFF    # M = B*T at L2 boundary
M_L3 = L3_BYTES / WS_COEFF    # M = B*T at L3 boundary

fig, ax = plt.subplots(figsize=(6, 4.5))
for key, (tgt, exp, col, lbl) in KERN.items():
    pts = gflops_at_n(tgt, exp)
    Ms  = [p[0] for p in pts]
    GFs = [p[1] for p in pts]
    ax.plot(Ms, GFs, "o-", color=col, lw=2, ms=6, label=lbl)

ax.axvline(M_L2, color="gray", lw=1.2, ls="--", alpha=0.8)
ax.axvline(M_L3, color="gray", lw=1.2, ls=":",  alpha=0.8)
ax.text(M_L2 * 1.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
        "L2", fontsize=10, color="gray", va="top")
ax.text(M_L3 * 1.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
        "L3", fontsize=10, color="gray", va="top")

ax.set_xlabel("Multivector count  M = B·T  (= 256 n²)")
ax.set_ylabel("Performance [GFLOP/s]")
ax.set_title("v3 kernel performance vs problem size\n"
             "Tiger Lake i7-1165G7 | 1 thread | C=8, B=2n, T=128n")
ax.set_xscale("log")
ax.set_ylim(bottom=0)
ax.grid(True, which="both", alpha=0.2)
ax.legend(fontsize=10)

# Redraw L2/L3 labels after ylim is set
ymax = ax.get_ylim()[1]
for child in ax.get_children():
    if hasattr(child, "get_text") and child.get_text() in ("L2", "L3"):
        child.set_y(ymax * 0.97)

save(fig, "memory_scaling.jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# 4 — speedup_endtoend.jpeg
#     v3/v0 speedup vs n.  Actual for n=1..3; projected for n=4..9 using
#     a two-term a*n^2 + b*n^4 fit to the measured v0 timings.
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4: speedup_endtoend …")

ee_rows = {row["n"]: row for row in EE["rows"]}

# Fit v0: solve [n^2, n^4] system from n=1,2
n1, n2 = 1, 2
v0_1 = ee_rows[1]["times_us"]["v0"] / 1e3   # ms
v0_2 = ee_rows[2]["times_us"]["v0"] / 1e3
v0_3 = ee_rows[3]["times_us"]["v0"] / 1e3
A = np.array([[n1**2, n1**4], [n2**2, n2**4]], dtype=float)
b_vec = np.array([v0_1, v0_2])
a_coef, b_coef = np.linalg.solve(A, b_vec)

def v0_projected(n):
    return a_coef * n**2 + b_coef * n**4

actual_ns    = [1, 2, 3]
actual_sp    = [ee_rows[n]["times_us"]["v0"] / ee_rows[n]["times_us"]["v3"]
                for n in actual_ns]

proj_ns = list(range(4, 10))
proj_sp = [v0_projected(n) * 1e3 / ee_rows[n]["times_us"]["v3"]
           for n in proj_ns]

all_ns = actual_ns + proj_ns
all_sp = actual_sp + proj_sp

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(actual_ns, actual_sp, "o-", color="#4CAF50", lw=2.2, ms=8,
        label="Measured (v0 actual)")
ax.plot([actual_ns[-1]] + proj_ns,
        [actual_sp[-1]] + proj_sp,
        "D--", color="#4CAF50", lw=1.8, ms=7, alpha=0.6,
        label="Projected (v0 extrapolated)")

# Annotate actual points
for n, sp in zip(actual_ns, actual_sp):
    ax.annotate(f"{sp:.0f}×", (n, sp), textcoords="offset points",
                xytext=(5, 4), fontsize=10, color="#4CAF50", fontweight="bold")

ax.axhline(67.2, color="gray", lw=1, ls=":", alpha=0.7)
ax.text(actual_ns[0] + 0.05, 67.2 * 1.02, "attn kernel speedup (67×)",
        fontsize=9, color="gray")

ax.set_xlabel("Problem-size parameter n  (B=2n, T=128n, C=8)")
ax.set_ylabel("End-to-end speedup  v3 / v0")
ax.set_title("End-to-end GATr model speedup over v0\n"
             "Tiger Lake i7-1165G7 | 1 thread | 2 layers")
ax.set_xticks(all_ns)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10)
save(fig, "speedup_endtoend.jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# Print table values
# ══════════════════════════════════════════════════════════════════════════════
PF = json.loads(Path("benchmarks/results/run_20260615/per_function.json").read_text())

print("\n" + "="*70)
print("TABLE VALUES")
print("="*70)

print("\n--- tab:summary (n=3, PAPI timing) ---")
print(f"{'Kernel':10s}  {'v3/v0':>8s}  {'v3 GF/s':>9s}  {'%peak':>7s}  {'bound':>8s}  {'v3/py':>7s}")
summary_rows = [
    ("gp",   "geometric_product_v0",            "geometric_product_v3",            "geometric_product"),
    ("lin",  "equi_linear_ver_0",               "equi_linear_ver_3",               "equi_linear"),
    ("attn", "equi_geometric_attention_ver_0",  "equi_geometric_attention_ver_3",  "attention"),
    ("rms",  "equi_rms_norm_ver_0",             "equi_rms_norm_ver_3",             "equi_rms_norm"),
    ("gelu", "scaler_gated_gelu_ver_0",         "scaler_gated_gelu_ver_3",         "scaler_gated_gelu"),
]
PY_TIMING = {
    "geometric_product": float(PF["pointwise"]["geometric_product"]["versions"]["py"][2]),
    "equi_linear":       float(PF["pointwise"]["equi_linear"]["versions"]["py"][2]),
    "attention":         float(PF["attention"]["py"]["3"]),
    "equi_rms_norm":     float(PF["pointwise"]["equi_rms_norm"]["versions"]["py"][2]),
    "scaler_gated_gelu": float(PF["pointwise"]["scaler_gated_gelu"]["versions"]["py"][2]),
}
for name, t0, t3, py_key in summary_rows:
    s0, s3 = PAPI3[t0]["min_s"], PAPI3[t3]["min_s"]
    speedup_v0 = s0 / s3
    cyc3  = PAPI3[t3]["papi_counters_mean_total_region"]["PAPI_TOT_CYC"]
    flops3 = PAPI3[t3]["papi_counters_mean_total_region"]["PAPI_FP_OPS"]
    gf3 = (flops3 / cyc3) * BASE_GHZ  # clock-honest
    pct = gf3 / PEAK_GFLOPS * 100
    ai3 = PAPI3[t3]["arithmetic_intensity"]
    bound = "compute" if ai3 > RIDGE else "memory"
    py_ms = PY_TIMING[py_key]
    speedup_py = (py_ms / 1000.0) / s3
    print(f"{name:10s}  {speedup_v0:8.1f}x  {gf3:9.1f}  {pct:7.1f}%  {bound:>8s}  {speedup_py:6.1f}x")

print("\n--- tab:autovec (n=3, bench_repo / bench_novector timing) ---")
nv3_rows = {}
for r in json.loads(Path("benchmarks/results/bench_novector/bench_novector_n3.json").read_text())["results"]:
    nv3_rows[(r["target"], r.get("vectorized"))] = r
repo3 = {r["target"]: r for r in json.loads(Path("benchmarks/results/bench_repo_new_dims/bench_repo_n3.json").read_text())["results"]}

print(f"{'Kernel':8s}  {'novec(ms)':>10s}  {'autovec(ms)':>12s}  {'v3(ms)':>8s}  {'autovec/novec':>14s}  {'v3/novec':>9s}")
autovec_rows = [
    ("gp",   "geometric_product_v2",            "geometric_product_v3"),
    ("lin",  "equi_linear_ver_2",               "equi_linear_ver_3"),
    ("attn", "equi_geometric_attention_ver_2",  "equi_geometric_attention_ver_3"),
    ("rms",  "equi_rms_norm_ver_2",             "equi_rms_norm_ver_3"),
    ("gelu", "scaler_gated_gelu_ver_2",         "scaler_gated_gelu_ver_3"),
]
for name, t2, t3 in autovec_rows:
    nv  = nv3_rows.get((t2, False))
    vec = nv3_rows.get((t2, True))
    nv_ms  = nv["min_ms"]   if nv  else None
    vec_ms = vec["min_ms"]  if vec else None
    v3_ms  = repo3[t3]["min_ms"] if t3 in repo3 else None
    auto_sp = nv_ms / vec_ms if (nv_ms and vec_ms) else float("nan")
    hand_sp = nv_ms / v3_ms  if (nv_ms and v3_ms)  else float("nan")
    print(f"{name:8s}  {nv_ms:10.3f}  {vec_ms:12.3f}  {v3_ms:8.3f}  {auto_sp:14.2f}x  {hand_sp:8.2f}x")

# End-to-end row for autovec table
ee_nv3  = ee_rows[3]["times_us"]["v2"]  # v2 in model = autovec v2 (vec=True)
ee_v3_3 = ee_rows[3]["times_us"]["v3"]
# no-vec v2 end-to-end not directly measured; omit
print(f"{'e2e(n=3)':8s}  {'N/A':>10s}  {ee_nv3/1000:12.1f}  {ee_v3_3/1000:8.1f}  {'N/A':>14s}  {ee_nv3/ee_v3_3:8.2f}x  (v3/v2-autovec)")

print("\nDone.")
