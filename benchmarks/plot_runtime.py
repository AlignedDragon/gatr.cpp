import json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, 'benchmarks')

# ── Load timings ──────────────────────────────────────────────────────────────
old = json.load(open('benchmarks/results/project_versions/project_versions_n1_n3_runtime.json'))
times = {}
for entry in old['results']:
    n = entry['n']
    times.setdefault(n, {})
    for t in entry['targets']:
        times[n][t['target']] = t['min_ms']
for n in [1, 2, 3]:
    d = json.load(open(f'benchmarks/results/run_2026_06_02/attention_n{n}.json'))
    for r in d['results']:
        times[n][r['target']] = r['min_ms']
# n=4: non-attention + attention
d4 = json.load(open('benchmarks/results/run_2026_06_02/non_attention_n4.json'))
times[4] = {r['target']: r['min_ms'] for r in d4['results']}
d4a = json.load(open('benchmarks/results/run_2026_06_02/attention_n4.json'))
for r in d4a['results']:
    times[4][r['target']] = r['min_ms']

ver_funcs = {
    'v0': ['geometric_product_v0','equi_join_v0','equi_linear_ver_0','equi_rms_norm_ver_0','scaler_gated_gelu_ver_0','equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1','equi_join_v1','equi_linear_ver_1','equi_rms_norm_ver_1','scaler_gated_gelu_ver_1','equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2','equi_join_v2','equi_linear_ver_2','equi_rms_norm_ver_2','scaler_gated_gelu_ver_2','equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3','equi_join_v3','equi_linear_ver_3','equi_rms_norm_ver_3','scaler_gated_gelu_ver_2','equi_geometric_attention_ver_3'],
}

tot = {n: {v: sum(times[n].get(f,0) for f in fs) for v,fs in ver_funcs.items()}
       for n in [1,2,3,4]}

vc = {'v0':'#B0BEC5','v1':'#FF9800','v2':'#2196F3','v3':'#4CAF50'}
vl = {'v0':'v0 — Baseline','v1':'v1 — Math','v2':'v2 — Scalar mem.','v3':'v3 — SIMD'}

# ── PLOT 1: speedup bars n=3 and n=4 (all functions) ─────────────────────────
func_info = [
    ('Geom.\nProduct', ['geometric_product_v0','geometric_product_v1','geometric_product_v2','geometric_product_v3']),
    ('Equi.\nJoin',    ['equi_join_v0','equi_join_v1','equi_join_v2','equi_join_v3']),
    ('Equi.\nLinear',  ['equi_linear_ver_0','equi_linear_ver_1','equi_linear_ver_2','equi_linear_ver_3']),
    ('RMS\nNorm',      ['equi_rms_norm_ver_0','equi_rms_norm_ver_1','equi_rms_norm_ver_2','equi_rms_norm_ver_3']),
    ('Gated\nGELU',   ['scaler_gated_gelu_ver_0','scaler_gated_gelu_ver_1','scaler_gated_gelu_ver_2','scaler_gated_gelu_ver_2']),
    ('Attention',      ['equi_geometric_attention_ver_0','equi_geometric_attention_ver_1','equi_geometric_attention_ver_2','equi_geometric_attention_ver_3']),
]

fig, ax = plt.subplots(figsize=(14, 5.5))
bw, gap = 0.17, 0.55
x = 0
xticks, xlabels = [], []

for fname, funcs in func_info:
    for ni, (n_val, hatch, nalpha) in enumerate([(3,'',1.0),(4,'//',0.85)]):
        for vi, (ver, color) in enumerate(vc.items()):
            t0 = times[n_val].get(funcs[0], 1)
            tv = times[n_val].get(funcs[vi], 1)
            sp = t0/tv if tv else 1
            bx = x + ni*(4*bw+0.1) + vi*bw
            ax.bar(bx, sp, bw, color=color, edgecolor='white', lw=0.5,
                   alpha=nalpha, hatch=hatch, zorder=3)
            if sp >= 5:
                ax.text(bx+bw/2, sp*1.2, f'{sp:.0f}×', ha='center',
                        va='bottom', fontsize=6.5, color='#222', fontweight='bold')
        mid = x + ni*(4*bw+0.1) + 1.5*bw
        xticks.append(mid)
        xlabels.append(f'{fname}\nn={n_val}')
    x += 2*(4*bw+0.1) + gap

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=8.5)
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_yscale('log')
ax.set_ylim(0.5, 5000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.0f}×' if y>=2 else f'{y:.1f}×'))
ax.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax.set_title('Per-Function Speedup — n=3 (solid) and n=4 (hatched)\nThreads=1  |  Tiger Lake i7-1165G7', fontsize=11)
ax.grid(axis='y', alpha=0.25, which='both', zorder=0)

vp = [mpatches.Patch(color=c, label=vl[v]) for v,c in vc.items()]
np_ = [mpatches.Patch(fc='white', ec='gray', label='n=3 (solid)'),
       mpatches.Patch(fc='white', ec='gray', hatch='//', label='n=4 (hatched)')]
l1 = ax.legend(handles=vp, loc='upper left', fontsize=9, ncol=2, framealpha=0.9, title='Version')
ax.add_artist(l1)
ax.legend(handles=np_, loc='upper center', fontsize=9, framealpha=0.9, title='Problem size')
plt.tight_layout()
plt.savefig('benchmarks/results/run_2026_06_02/speedup_per_function.png', dpi=150, bbox_inches='tight')
print('Saved speedup_per_function.png')
plt.close()

# ── PLOT 2: runtime scaling n=1..4 full project ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ns = [1, 2, 3, 4]
xl = ['n=1\n(T=16,C=4)', 'n=2\n(T=32,C=8)', 'n=3\n(T=48,C=12)', 'n=4\n(T=64,C=16)']

for ver, color in vc.items():
    rts = [tot[n][ver] for n in ns]
    ax1.plot(range(4), rts, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    ax1.text(3.08, rts[-1], f'{rts[-1]/1000:.1f}s', fontsize=7.5, color=color, va='center')

ax1.set_xticks(range(4))
ax1.set_xticklabels(xl, fontsize=9)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms)', fontsize=12)
ax1.set_title('Total Project Runtime vs Problem Size', fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')
ax1.set_xlim(-0.3, 3.8)

for ver, color in vc.items():
    sp = [tot[n]['v0']/tot[n][ver] for n in ns]
    ax2.plot(range(4), sp, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    ax2.text(3.08, sp[-1], f'{sp[-1]:.1f}×', fontsize=7.5, color=color, va='center')

ax2.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.set_xticks(range(4))
ax2.set_xticklabels(xl, fontsize=9)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.0f}×' if y>=2 else f'{y:.1f}×'))
ax2.set_ylabel('Speedup over v0', fontsize=12)
ax2.set_title('Total Speedup over Baseline vs Problem Size', fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(alpha=0.3, which='both')
ax2.set_xlim(-0.3, 3.8)

plt.suptitle('EzGATr Runtime Scaling  |  Threads=1  |  Tiger Lake i7-1165G7', fontsize=11)
plt.tight_layout()
plt.savefig('benchmarks/results/run_2026_06_02/runtime_scaling.png', dpi=150, bbox_inches='tight')
print('Saved runtime_scaling.png')
