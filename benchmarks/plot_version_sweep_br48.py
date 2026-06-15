import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, 'benchmarks')
from roofline_costs import estimate_target_cost

OUT = Path('benchmarks/results/FINAL_PLOTS')
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE_POINT = PEAK_GFLOPS / PEAK_BW_GBS

# Runtime: per-target min_ms from the project version sweep (B=64, T=16n, C=4n, heads=32n).
sweep = json.load(open('benchmarks/results/version_sweep_br48.json'))
times = {}            # times[n][target] = min_ms
NS = []
for agg in sweep['results']:
    n = agg['n']
    times.setdefault(n, {})
    NS.append(n)
    for t in agg['targets']:
        times[n][t['target']] = t['min_ms']
NS = sorted(set(NS))

# PAPI flops/bytes per target (for the measured per-kernel roofline).
papi_flops, papi_bytes = {}, {}
for n in [3, 4]:
    fn = f'benchmarks/results/version_sweep_br48_papi_n{n}.json'
    if not Path(fn).exists():
        continue
    d = json.load(open(fn))
    for r in d['results']:
        if r.get('flops_per_call') and r.get('memory_bytes_per_call'):
            papi_flops[(r['target'], n)] = r['flops_per_call']
            papi_bytes[(r['target'], n)] = r['memory_bytes_per_call']

vc = {'v0': '#B0BEC5', 'v1': '#FF9800', 'v2': '#2196F3', 'v3': '#4CAF50'}
vl = {'v0': 'v0 - Baseline', 'v1': 'v1 - Math', 'v2': 'v2 - Scalar mem.', 'v3': 'v3 - SIMD'}

ver_funcs = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0', 'equi_rms_norm_ver_0', 'equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1', 'equi_rms_norm_ver_1', 'equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2', 'equi_rms_norm_ver_2', 'equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3', 'equi_rms_norm_ver_3', 'equi_geometric_attention_ver_3'],
}
tot = {n: {v: sum(times[n].get(f, 0) for f in fs) for v, fs in ver_funcs.items()} for n in NS}

func_info = [
    ('Geom.\nProduct', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3']),
    ('Equi.\nJoin',    ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('Equi.\nLinear',  ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('RMS\nNorm',      ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('Attention',      ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1', 'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3']),
]

# Two n values for the bar chart: a mid size and the largest measured.
N_LO = 4 if 4 in NS else NS[len(NS) // 2]
N_HI = NS[-1]

# PLOT 1: speedup bars
fig, ax = plt.subplots(figsize=(15, 6.4))
bw, gap = 0.23, 0.80
x = 0
xticks, xlabels = [], []
for fname, funcs in func_info:
    for ni, (n_val, hatch, nalpha) in enumerate([(N_LO, '', 1.0), (N_HI, '//', 0.85)]):
        for vi, (ver, color) in enumerate(vc.items()):
            t0 = times[n_val].get(funcs[0], 1)
            tv = times[n_val].get(funcs[vi], 1)
            sp = t0 / tv if tv else 1
            bx = x + ni * (4 * bw + 0.14) + vi * bw
            ax.bar(bx, sp, bw, color=color, edgecolor='white', lw=0.6,
                   alpha=nalpha, hatch=hatch, zorder=3)
            if sp > 1.05:
                lbl = f'{sp:.1f}x' if sp < 10 else f'{sp:.0f}x'
                ax.text(bx + bw / 2, sp * 1.06, lbl, ha='center', va='bottom',
                        fontsize=7.5, color='#111', fontweight='bold', rotation=90,
                        zorder=12, clip_on=False,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.82))
        mid = x + ni * (4 * bw + 0.14) + 1.5 * bw
        xticks.append(mid)
        xlabels.append(f'{fname}\nn={n_val}')
    x += 2 * (4 * bw + 0.14) + gap

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=9)
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_yscale('log')
ax.set_ylim(0.5, 30000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x' if y >= 2 else f'{y:.1f}x'))
ax.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax.set_title(f'Per-function speedup - n={N_LO} (solid) and n={N_HI} (hatched)\nThreads=8 | Tiger Lake i7-1165G7', fontsize=12)
ax.grid(axis='y', alpha=0.25, which='both', zorder=0)
ax.margins(x=0.025)
vp = [mpatches.Patch(color=c, label=vl[v]) for v, c in vc.items()]
np_ = [mpatches.Patch(fc='white', ec='gray', label=f'n={N_LO} (solid)'),
       mpatches.Patch(fc='white', ec='gray', hatch='//', label=f'n={N_HI} (hatched)')]
l1 = ax.legend(handles=vp, loc='upper right', fontsize=9, ncol=2, framealpha=0.95,
               title='Version', bbox_to_anchor=(0.995, 0.99))
ax.add_artist(l1)
ax.legend(handles=np_, loc='upper right', fontsize=9, framealpha=0.95,
          title='Problem size', bbox_to_anchor=(0.995, 0.74))
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_speedup.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_speedup.png')
plt.close()

# PLOT 2: runtime scaling + total speedup, n=1..max
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xl = [f'n={n}\n(T={16*n},C={4*n})' for n in NS]
xr = range(len(NS))

for ver, color in vc.items():
    rts = [tot[n][ver] for n in NS]
    ax1.plot(xr, rts, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    ax1.text(len(NS) - 0.92, rts[-1], f'{rts[-1]/1000:.1f}s', fontsize=7.5, color=color, va='center')
ax1.set_xticks(xr)
ax1.set_xticklabels(xl, fontsize=9)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms)', fontsize=12)
ax1.set_title('Total Project Runtime vs Problem Size', fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')
ax1.set_xlim(-0.3, len(NS) - 0.2)

for ver, color in vc.items():
    sp = [tot[n]['v0'] / tot[n][ver] for n in NS]
    ax2.plot(xr, sp, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    ax2.text(len(NS) - 0.92, sp[-1], f'{sp[-1]:.1f}x', fontsize=7.5, color=color, va='center')
ax2.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.set_xticks(xr)
ax2.set_xticklabels(xl, fontsize=9)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x' if y >= 2 else f'{y:.1f}x'))
ax2.set_ylabel('Speedup over v0', fontsize=12)
ax2.set_title('Total Speedup over Baseline vs Problem Size', fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(alpha=0.3, which='both')
ax2.set_xlim(-0.3, len(NS) - 0.2)
plt.suptitle('EzGATr Runtime Scaling  |  Threads=8  |  Tiger Lake i7-1165G7', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_runtime.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_runtime.png')
plt.close()


def draw_roofline(ax):
    ai_range = np.logspace(-1, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ax.axvline(RIDGE_POINT, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.13, PEAK_BW_GBS * 0.13 * 1.3, f'Mem BW: {PEAK_BW_GBS} GB/s', fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, PEAK_GFLOPS * 1.08, f'Peak: {PEAK_GFLOPS} GFLOPS', fontsize=8.5, ha='right')
    ax.text(RIDGE_POINT * 1.1, 55, f'Ridge\n{RIDGE_POINT:.1f} F/B', fontsize=7.5, color='gray')


# PLOT 3: measured per-kernel roofline (PAPI flops/bytes, sweep min_ms), n=3 filled / n=4 hollow
funcs = [
    ('geometric_product', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3'], 'Geom. Product', '#2196F3'),
    ('equi_join', ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3'], 'Equi. Join', '#4CAF50'),
    ('equi_linear', ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3'], 'Equi. Linear', '#FF9800'),
    ('equi_rms_norm', ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3'], 'RMS Norm', '#9C27B0'),
    ('attention', ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1', 'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3'], 'Attention', '#795548'),
]
ver_sizes = [55, 65, 80, 100]
ver_alphas = [0.45, 0.65, 0.85, 1.0]

fig, ax = plt.subplots(figsize=(12, 7))
draw_roofline(ax)
for _, ver_tgts, fname, color in funcs:
    for n, filled in [(3, True), (4, False)]:
        # measured AI from the v3 PAPI point (all versions share the same algorithm AI)
        prev_gf = None
        for vi, (vtgt, sz, alpha) in enumerate(zip(ver_tgts, ver_sizes, ver_alphas)):
            fl = papi_flops.get((vtgt, n))
            by = papi_bytes.get((vtgt, n))
            ms = times.get(n, {}).get(vtgt)
            if not fl or not by or not ms:
                continue
            ai = fl / by
            gf = fl / (ms / 1000) / 1e9
            mfc = color if filled else 'white'
            marker = ['o', 's', '^', 'D'][vi]
            ax.scatter(ai, gf, s=sz, marker=marker, color=color, facecolors=mfc,
                       edgecolors=color, linewidths=0.7 if filled else 2.0, alpha=alpha, zorder=10)
            prev_gf = gf
        # label v3
        fl3 = papi_flops.get((ver_tgts[-1], n))
        by3 = papi_bytes.get((ver_tgts[-1], n))
        ms3 = times.get(n, {}).get(ver_tgts[-1])
        if fl3 and by3 and ms3:
            ai3 = fl3 / by3
            gf3 = fl3 / (ms3 / 1000) / 1e9
            xoff = 7 if n == 3 else -65
            yoff = 4 if n == 3 else -12
            ax.annotate(f'{fname}\nn={n}', (ai3, gf3), textcoords='offset points',
                        xytext=(xoff, yoff), fontsize=8, color=color,
                        fontweight='bold' if n == 3 else 'normal', alpha=1.0 if n == 3 else 0.8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 8000); ax.set_ylim(0.2, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title('Roofline - all kernels, all versions (v0-v3), measured with PAPI\nFilled markers = n=3 | Hollow markers = n=4 | Tiger Lake i7-1165G7', fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')
func_leg = [mlines.Line2D([], [], color=c, marker='D', ls='None', ms=9, label=n) for _, _, n, c in funcs]
ver_leg = [mlines.Line2D([], [], color='gray', marker=m, ls='None', ms=7, alpha=a, label=l)
           for m, a, l in zip(['o', 's', '^', 'D'], ver_alphas, ['v0', 'v1', 'v2', 'v3'])]
n_leg = [mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, label='n=3 (filled)'),
         mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, mfc='white', mew=2, label='n=4 (hollow)')]
l1 = ax.legend(handles=func_leg, loc='lower right', fontsize=9, title='Function', framealpha=0.9, ncol=1)
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, loc='upper left', fontsize=9, title='Version / Size', framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_roofline_kernels.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_roofline_kernels.png')
plt.close()

# PLOT 4: whole-project roofline, n=1..max (cost-model flops/bytes + sweep total_ms)
ver_style = {
    'v0': ('#B0BEC5', 'o', 'v0 - Baseline C++'),
    'v1': ('#FF9800', 's', 'v1 - Math optimizations'),
    'v2': ('#2196F3', '^', 'v2 - Scalar memory'),
    'v3': ('#4CAF50', 'D', 'v3 - SIMD + FMA + parallel'),
}
points = {}
for n in NS:
    for ver, fl_list in ver_funcs.items():
        total_ms = sum(times[n].get(f, 0) for f in fl_list)
        total_fl = sum((estimate_target_cost(f, n) or {}).get('estimated_flops_per_call', 0) for f in fl_list)
        total_by = sum((estimate_target_cost(f, n) or {}).get('estimated_bytes_per_call', 0) for f in fl_list)
        ai = total_fl / total_by if total_by else 0
        gf = total_fl / (total_ms / 1000) / 1e9 if total_ms else 0
        points.setdefault(ver, []).append((ai, gf, n))

fig, ax = plt.subplots(figsize=(10, 6))
draw_roofline(ax)
for ver, pts in points.items():
    color, marker, label = ver_style[ver]
    ais = [p[0] for p in pts]; gfs = [p[1] for p in pts]
    ax.plot(ais, gfs, '-', color=color, lw=1.3, alpha=0.5, zorder=6)
    ax.scatter(ais, gfs, color=color, marker=marker, s=110, zorder=10, edgecolors='white', lw=0.8, label=label)
    for ai, gf, n in pts:
        if ver == 'v3' or (ver == 'v0' and n == NS[-1]):
            xoff = 6 if ver == 'v3' else -40
            yoff = 4 if ver == 'v3' else -12
            ax.annotate(f'n={n}', (ai, gf), textcoords='offset points', xytext=(xoff, yoff),
                        fontsize=7.5, color='#2E7D32' if ver == 'v3' else '#555')
for i in range(len(NS)):
    v0ai, v0gf, _ = points['v0'][i]
    v3ai, v3gf, _ = points['v3'][i]
    ax.annotate('', xy=(v3ai, v3gf), xytext=(v0ai, v0gf),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.1, alpha=0.55))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 5000); ax.set_ylim(0.8, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title(f'Whole-project roofline - all versions, n=1..{NS[-1]}\nTiger Lake i7-1165G7 | 5 kernels combined | Arrows: v0->v3', fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_roofline_project.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_roofline_project.png')
plt.close()
print('done')
