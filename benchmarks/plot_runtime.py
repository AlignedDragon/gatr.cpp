"""Runtime / speedup bar charts using bench_repo_new_dims data.

  Plot 1 — Per-function speedup bars at n=3 (solid) and n=5 (hatched).
  Plot 2 — Total runtime + speedup scaling curves, n=1..9.

Both plots use the new dimension formula B=2n, H=4n, T=128n, C=8.

Output: benchmarks/results/FINAL_PLOTS/speedup_per_function.png
         benchmarks/results/FINAL_PLOTS/runtime_scaling.png

    python benchmarks/plot_runtime.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path("benchmarks/results/FINAL_PLOTS")
OUT.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("benchmarks/results/bench_repo_new_dims")

# ── Load runtimes ─────────────────────────────────────────────────────────────
NS = list(range(1, 13))
times: dict[int, dict[str, float]] = {}
for n in NS:
    f = DATA_DIR / f"bench_repo_n{n}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    times[n] = {r["target"]: r["min_ms"] for r in d["results"]}

ver_funcs = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0', 'equi_rms_norm_ver_0',
           'scaler_gated_gelu_ver_0', 'equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1', 'equi_rms_norm_ver_1',
           'scaler_gated_gelu_ver_1', 'equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2', 'equi_rms_norm_ver_2',
           'scaler_gated_gelu_ver_2', 'equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3', 'equi_rms_norm_ver_3',
           'scaler_gated_gelu_ver_3', 'equi_geometric_attention_ver_3_1'],
}

vc  = {'v0': '#B0BEC5', 'v1': '#FF9800', 'v2': '#2196F3', 'v3': '#2E7D32'}
vl  = {'v0': 'v0 — Baseline', 'v1': 'v1 — Math', 'v2': 'v2 — Scalar mem.', 'v3': 'v3 — SIMD + AVX-512 attn'}

func_info = [
    ('Geom.\nProduct',  ['geometric_product_v0', 'geometric_product_v1',
                         'geometric_product_v2', 'geometric_product_v3']),
    ('Equi.\nJoin',     ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('Equi.\nLinear',   ['equi_linear_ver_0', 'equi_linear_ver_1',
                         'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('RMS\nNorm',       ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1',
                         'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('Gated\nGELU',    ['scaler_gated_gelu_ver_0', 'scaler_gated_gelu_ver_1',
                         'scaler_gated_gelu_ver_2', 'scaler_gated_gelu_ver_3']),
    ('Attention',       ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1',
                         'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3_1']),
]

# ── PLOT 1: speedup bars n=3 (solid) and n=5 (hatched) ───────────────────────
# n=3: all versions have attention; n=5: v2 and v3 have attention (v0/v1 skip)
BAR_NS = [3, 5]

fig, ax = plt.subplots(figsize=(14, 5.5))
bw, gap = 0.17, 0.55
x = 0
xticks, xlabels = [], []

for fname, funcs in func_info:
    for ni, (n_val, hatch, nalpha) in enumerate([(BAR_NS[0], '', 1.0), (BAR_NS[1], '//', 0.85)]):
        t0 = times.get(n_val, {}).get(funcs[0])
        if not t0:
            continue
        for vi, (ver, color) in enumerate(vc.items()):
            tv = times.get(n_val, {}).get(funcs[vi])
            if not tv:
                continue
            sp = t0 / tv
            bx = x + ni * (4 * bw + 0.1) + vi * bw
            ax.bar(bx, sp, bw, color=color, edgecolor='white', lw=0.5,
                   alpha=nalpha, hatch=hatch, zorder=3)
            if sp >= 1.5:
                lbl = f'{sp:.0f}×' if sp >= 10 else f'{sp:.1f}×'
                ax.text(bx + bw / 2, sp * 1.15, lbl, ha='center', va='bottom',
                        rotation=90, fontsize=6, color='#222', fontweight='bold')
        mid = x + ni * (4 * bw + 0.1) + 1.5 * bw
        xticks.append(mid)
        xlabels.append(f'{fname}\nn={n_val}')
    x += 2 * (4 * bw + 0.1) + gap

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=8.5)
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_yscale('log')
ax.set_ylim(0.5, 5000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}×' if y >= 2 else f'{y:.1f}×'))
ax.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax.set_title(f'Per-Function Speedup — n={BAR_NS[0]} (solid) and n={BAR_NS[1]} (hatched)\n'
             'Threads=1  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8', fontsize=11)
ax.grid(axis='y', alpha=0.25, which='both', zorder=0)

vp  = [mpatches.Patch(color=c, label=vl[v]) for v, c in vc.items()]
np_ = [mpatches.Patch(fc='white', ec='gray', label=f'n={BAR_NS[0]} (solid)'),
       mpatches.Patch(fc='white', ec='gray', hatch='//', label=f'n={BAR_NS[1]} (hatched)')]
l1 = ax.legend(handles=vp, loc='upper left', fontsize=9, ncol=2, framealpha=0.9, title='Version')
ax.add_artist(l1)
ax.legend(handles=np_, loc='upper center', fontsize=9, framealpha=0.9, title='Problem size')

plt.tight_layout()
out1 = OUT / "speedup_per_function.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"saved: {out1}")
plt.close()

# ── PLOT 2: runtime and speedup scaling n=1..9 (whole project) ───────────────
# "Whole project" = sum of all 6 kernels; skip ops with missing data at a given n
SCALE_NS = list(range(1, 10))

# Fixed non-attention kernel set — same 5 kernels for all versions at all n.
# Attention speedup is shown separately in the per-function speedup chart.
NON_ATTN_GROUPS = [
    ('geometric_product', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3']),
    ('equi_join',         ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('equi_linear',       ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('equi_rms_norm',     ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('scaler_gated_gelu', ['scaler_gated_gelu_ver_0', 'scaler_gated_gelu_ver_1', 'scaler_gated_gelu_ver_2', 'scaler_gated_gelu_ver_3']),
]
VER_IDX = {'v0': 0, 'v1': 1, 'v2': 2, 'v3': 3}


def total_for(n, ver):
    idx = VER_IDX[ver]
    vals = [times.get(n, {}).get(funcs[idx]) for _, funcs in NON_ATTN_GROUPS]
    present = [v for v in vals if v is not None]
    return sum(present) if len(present) == len(NON_ATTN_GROUPS) else None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ver, color in vc.items():
    ns_v  = [n for n in SCALE_NS if total_for(n, ver) is not None]
    rts_v = [total_for(n, ver) for n in ns_v]
    ax1.plot(ns_v, rts_v, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if rts_v:
        ax1.text(ns_v[-1] + 0.1, rts_v[-1], f'{rts_v[-1]:.1f}ms', fontsize=7.5, color=color, va='center')

ax1.set_xlabel('n  (B=2n, T=128n, H=4n, C=8)', fontsize=11)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms, log)', fontsize=12)
ax1.set_title('Total Runtime vs Problem Size\n(5 non-attention kernels; attention speedup shown separately)', fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')

t0_v0 = {n: total_for(n, 'v0') for n in SCALE_NS}
for ver, color in vc.items():
    ns_v = [n for n in SCALE_NS if total_for(n, ver) is not None and t0_v0.get(n) is not None]
    sp_v = [t0_v0[n] / total_for(n, ver) for n in ns_v]
    ax2.plot(ns_v, sp_v, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if sp_v:
        ax2.text(ns_v[-1] + 0.1, sp_v[-1], f'{sp_v[-1]:.1f}×', fontsize=7.5, color=color, va='center')

ax2.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.set_xlabel('n  (B=2n, T=128n, H=4n, C=8)', fontsize=11)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}×' if y >= 2 else f'{y:.1f}×'))
ax2.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax2.set_title('Total Speedup over Baseline vs Problem Size', fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(alpha=0.3, which='both')

plt.suptitle('EzGATr Runtime Scaling  |  Threads=1  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8',
             fontsize=11)
plt.tight_layout()
out2 = OUT / "runtime_scaling.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"saved: {out2}")
plt.close()

# ── PLOT 3: runtime + speedup WITH attention included ─────────────────────────
# Per-version attention caps: v0/v1 up to n=3, v2 up to n=5, v3 up to n=9
ATTN_MAX_N = {'v0': 3, 'v1': 3, 'v2': 5, 'v3': 9}

ALL_GROUPS = NON_ATTN_GROUPS + [
    ('attention', ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1',
                   'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3_1']),
]


def total_with_attn(n, ver):
    if n > ATTN_MAX_N[ver]:
        return None
    idx = VER_IDX[ver]
    vals = [times.get(n, {}).get(funcs[idx]) for _, funcs in ALL_GROUPS]
    present = [v for v in vals if v is not None]
    return sum(present) if len(present) == len(ALL_GROUPS) else None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ver, color in vc.items():
    ns_v  = [n for n in SCALE_NS if total_with_attn(n, ver) is not None]
    rts_v = [total_with_attn(n, ver) for n in ns_v]
    lbl   = f'{vl[ver]} (n≤{ATTN_MAX_N[ver]})'
    ax1.plot(ns_v, rts_v, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=lbl)
    if rts_v:
        ax1.text(ns_v[-1] + 0.1, rts_v[-1], f'{rts_v[-1]:.1f}ms',
                 fontsize=7.5, color=color, va='center')

ax1.set_xlabel('n  (B=2n, T=128n, H=4n, C=8)', fontsize=11)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms, log)', fontsize=12)
ax1.set_title('Total Runtime vs Problem Size\n(6 kernels incl. attention; each version to its attention cap)',
              fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')

# Speedup: compare all versions against v0 at n=1..3 (where v0 has attention)
t0_v0_attn = {n: total_with_attn(n, 'v0') for n in SCALE_NS}
for ver, color in vc.items():
    ns_v = [n for n in SCALE_NS
            if total_with_attn(n, ver) is not None and t0_v0_attn.get(n) is not None]
    sp_v = [t0_v0_attn[n] / total_with_attn(n, ver) for n in ns_v]
    ax2.plot(ns_v, sp_v, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if sp_v:
        ax2.text(ns_v[-1] + 0.1, sp_v[-1], f'{sp_v[-1]:.1f}×',
                 fontsize=7.5, color=color, va='center')

ax2.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.set_xlabel('n  (B=2n, T=128n, H=4n, C=8)', fontsize=11)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}×' if y >= 2 else f'{y:.1f}×'))
ax2.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax2.set_title('Speedup over v0 (with attention)\nComparable range n=1..3 where v0 has attention data',
              fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(alpha=0.3, which='both')

plt.suptitle('EzGATr Runtime Scaling (incl. Attention)  |  Threads=1  |  Tiger Lake i7-1165G7  |  B=2n H=4n T=128n C=8',
             fontsize=11)
plt.tight_layout()
out3 = OUT / "runtime_scaling_with_attention.png"
plt.savefig(out3, dpi=150, bbox_inches='tight')
print(f"saved: {out3}")
plt.close()
