"""Roofline comparison: Python baseline vs C++ versions v0-v3.

Uses PAPI-measured FLOPs and bytes for accurate arithmetic intensity.
AI  = PAPI flops_per_call / PAPI memory_bytes_per_call
GF/s = PAPI flops_per_call / bench_repo min_ms  (best-case timing)

Scatter at n=3 (filled) and n=9 (hollow) per version per function.
Python points use PAPI flops/bytes from the closest C++ version
(v0 for pointwise ops; v3 for attention which uses Flash SDPA).

    python benchmarks/plot_roofline_vs_python.py
    python benchmarks/plot_roofline_vs_python.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/FINAL_PLOTS")
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0

# ── Load Python timings ───────────────────────────────────────────────────────
pf = json.loads(Path("benchmarks/results/run_20260615/per_function.json").read_text())
py_times: dict[str, dict[int, float]] = {}
for op, dat in pf["pointwise"].items():
    py_times[op] = {n: dat["versions"]["py"][i] for i, n in enumerate(dat["n_values"])}
attn_py = {int(k): float(v) for k, v in pf["attention"]["py"].items()}

# ── Load C++ bench_repo min timings ──────────────────────────────────────────
NS = list(range(1, 13))
cpp_times: dict[int, dict[str, float]] = {}
for n in NS:
    f = Path(f"benchmarks/results/bench_repo_new_dims/bench_repo_n{n}.json")
    if f.exists():
        d = json.loads(f.read_text())
        cpp_times[n] = {r["target"]: r["min_ms"] for r in d["results"]}

# ── Load PAPI data ────────────────────────────────────────────────────────────
def load_papi(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for r in d["results"]:
        fl = r.get("flops_per_call")
        by = r.get("memory_bytes_per_call")
        gf = r.get("gflops")
        if fl and by and gf:
            out[r["target"]] = {"flops": fl, "bytes": by, "gflops": gf}
    return out

papi = {
    3: load_papi("benchmarks/results/papi_1thread_n3.json"),
    9: load_papi("benchmarks/results/papi_1thread_n9.json"),
}

SCATTER_NS = [(3, True), (9, False)]   # (n, filled)

# ── Function definitions ──────────────────────────────────────────────────────
# (display_name, color, py_key, py_papi_ref, [(ver_label, cpp_target)])
# py_papi_ref: which C++ target's PAPI FLOPs to use as the Python AI reference
FUNCS = [
    ("Geom. Product",  "#2196F3", "geometric_product",    "geometric_product_v3",
     [("v0", "geometric_product_v0"), ("v1", "geometric_product_v1"),
      ("v2", "geometric_product_v2"), ("v3", "geometric_product_v3")]),
    ("Equi. Join",     "#4CAF50", "equi_join",             "equi_join_v3",
     [("v0", "equi_join_v0"), ("v1", "equi_join_v1"),
      ("v2", "equi_join_v2"), ("v3", "equi_join_v3")]),
    ("Equi. Linear",   "#FF9800", "equi_linear",           "equi_linear_ver_3",
     [("v0", "equi_linear_ver_0"), ("v1", "equi_linear_ver_1"),
      ("v2", "equi_linear_ver_2"), ("v3", "equi_linear_ver_3")]),
    ("RMS Norm",       "#9C27B0", "equi_rms_norm",         "equi_rms_norm_ver_3",
     [("v0", "equi_rms_norm_ver_0"), ("v1", "equi_rms_norm_ver_1"),
      ("v2", "equi_rms_norm_ver_2"), ("v3", "equi_rms_norm_ver_3")]),
    ("Gated GELU",     "#F44336", "scaler_gated_gelu",     "scaler_gated_gelu_ver_3",
     [("v0", "scaler_gated_gelu_ver_0"), ("v1", "scaler_gated_gelu_ver_1"),
      ("v2", "scaler_gated_gelu_ver_2"), ("v3", "scaler_gated_gelu_ver_3")]),
    ("Attention",      "#795548", None,                    "equi_geometric_attention_ver_3",
     [("v0", "equi_geometric_attention_ver_0"), ("v1", "equi_geometric_attention_ver_1"),
      ("v2", "equi_geometric_attention_ver_2"), ("v3", "equi_geometric_attention_ver_3")]),
]

VER_STYLE = {
    "py": ("None", 2.2, 0.80, "*",  9, "py (PyTorch)"),
    "v0": ("None", 1.5, 0.50, "o",  7, "v0 — Baseline"),
    "v1": ("None", 1.6, 0.65, "s",  7, "v1 — Math"),
    "v2": ("None", 1.8, 0.80, "^",  7, "v2 — Scalar"),
    "v3": ("None", 2.5, 1.00, "D",  8, "v3 — SIMD"),
}


def draw_roofline(ax):
    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ax.axvline(PEAK_GFLOPS / PEAK_BW_GBS, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.13, PEAK_BW_GBS * 0.13 * 1.3, f'Mem BW: {PEAK_BW_GBS} GB/s',
            fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, PEAK_GFLOPS * 1.08, f'Peak: {PEAK_GFLOPS} GFLOPS', fontsize=8.5, ha='right')


fig, ax = plt.subplots(figsize=(13, 8))
draw_roofline(ax)

for fname, color, py_key, py_papi_ref, cpp_versions in FUNCS:
    for n, filled in SCATTER_NS:
        papi_n = papi.get(n, {})
        mfc_solid  = color
        mfc_hollow = "white"

        # Python scatter (★)
        p_ref = papi_n.get(py_papi_ref)
        if p_ref:
            py_ms = py_times.get(py_key, {}).get(n) if py_key else attn_py.get(n)
            if py_ms is not None:
                ai_py = p_ref["flops"] / p_ref["bytes"]
                gf_py = p_ref["flops"] / (py_ms / 1000) / 1e9
                _, _, alpha, mk, ms, _ = VER_STYLE["py"]
                ax.scatter(ai_py, gf_py, s=ms**2, marker=mk, color=color,
                           facecolors=mfc_solid if filled else mfc_hollow,
                           edgecolors=color, linewidths=1.5, alpha=alpha, zorder=8)

        # C++ scatter per version — use PAPI gflops directly (self-consistent)
        for vname, tgt in cpp_versions:
            p = papi_n.get(tgt)
            if not p:
                continue
            ai = p["flops"] / p["bytes"]
            gf = p["gflops"]
            _, _, alpha, mk, ms_sz, _ = VER_STYLE[vname]
            ax.scatter(ai, gf, s=ms_sz**2, marker=mk, color=color,
                       facecolors=mfc_solid if filled else mfc_hollow,
                       edgecolors=color,
                       linewidths=0.7 if filled else 2.0,
                       alpha=alpha, zorder=9)

        # Label v3 position at n=3
        if filled:
            p_v3 = papi_n.get(cpp_versions[-1][1])
            if p_v3:
                ai3 = p_v3["flops"] / p_v3["bytes"]
                gf3 = p_v3["gflops"]
                ax.annotate(fname, (ai3, gf3),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7.5, color=color, fontweight='bold')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.01, 8000); ax.set_ylim(0.01, PEAK_GFLOPS * 2)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOP/s]', fontsize=12)
ax.set_title(
    'Roofline: Python (PyTorch) vs C++ versions v0–v3  |  PAPI-measured AI\n'
    'Scatter: n=3 (filled) / n=9 (hollow)  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8',
    fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')

func_leg = [mlines.Line2D([], [], color=c, marker='D', ls='None', ms=7, label=n)
            for n, c, *_ in FUNCS]
ver_leg  = [mlines.Line2D([], [], color='gray', ls='None', marker=mk, ms=ms,
                           alpha=alpha, label=label)
            for ver, (_, _, alpha, mk, ms, label) in VER_STYLE.items()]
n_leg    = [mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, label='n=3 (filled)'),
            mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7,
                          mfc='white', mew=2, label='n=9 (hollow)')]
l1 = ax.legend(handles=func_leg, loc='lower right', fontsize=8.5, title='Function', framealpha=0.9)
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, loc='upper left', fontsize=8.5, title='Version', framealpha=0.9)

fig.tight_layout()
out = OUT / "roofline_vs_python.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out}")
