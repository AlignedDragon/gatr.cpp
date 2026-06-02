import json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, 'benchmarks')
from roofline_costs import estimate_target_cost

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE_POINT = PEAK_GFLOPS / PEAK_BW_GBS

# ── Load runtimes ─────────────────────────────────────────────────────────────
old = json.load(open('benchmarks/results/project_versions/project_versions_n1_n3_runtime.json'))
times = {}
for entry in old['results']:
    n = entry['n']
    times.setdefault(n, {})
    for t in entry['targets']:
        times[n][t['target']] = t['min_ms']
for n in [1,2,3]:
    d = json.load(open(f'benchmarks/results/run_2026_06_02/attention_n{n}.json'))
    for r in d['results']:
        times[n][r['target']] = r['min_ms']
d4 = json.load(open('benchmarks/results/run_2026_06_02/non_attention_n4.json'))
times[4] = {r['target']: r['min_ms'] for r in d4['results']}
d4a = json.load(open('benchmarks/results/run_2026_06_02/attention_n4.json'))
for r in d4a['results']:
    times[4][r['target']] = r['min_ms']

# ── Load PAPI FLOPs/bytes ─────────────────────────────────────────────────────
papi_flops, papi_bytes = {}, {}
for n in [1,2,3,4]:
    try:
        d = json.load(open(f'benchmarks/results/run_2026_06_02/papi_v3_n{n}.json'))
        for r in d['results']:
            if r.get('flops_per_call') and r.get('memory_bytes_per_call'):
                papi_flops[(r['target'],n)] = r['flops_per_call']
                papi_bytes[(r['target'],n)]  = r['memory_bytes_per_call']
    except: pass
for fn in ['benchmarks/results/roofline/papi_attention_n3.json',
           'benchmarks/results/roofline/papi_attention_n4.json']:
    d = json.load(open(fn))
    n = d['n']
    for r in d['results']:
        if r.get('flops_per_call') and r.get('memory_bytes_per_call'):
            papi_flops[(r['target'],n)] = r['flops_per_call']
            papi_bytes[(r['target'],n)]  = r['memory_bytes_per_call']

def draw_roofline(ax):
    ai_range = np.logspace(-1, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ax.axvline(RIDGE_POINT, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.13, PEAK_BW_GBS*0.13*1.3, f'Mem BW: {PEAK_BW_GBS} GB/s', fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, PEAK_GFLOPS*1.08, f'Peak: {PEAK_GFLOPS} GFLOPS', fontsize=8.5, ha='right')
    ax.text(RIDGE_POINT*1.1, 55, f'Ridge\n{RIDGE_POINT:.1f} F/B', fontsize=7.5, color='gray')

funcs = [
    ('geometric_product_v3',
     ['geometric_product_v0','geometric_product_v1','geometric_product_v2','geometric_product_v3'],
     'Geom. Product', '#2196F3'),
    ('equi_join_v3',
     ['equi_join_v0','equi_join_v1','equi_join_v2','equi_join_v3'],
     'Equi. Join', '#4CAF50'),
    ('equi_linear_ver_3',
     ['equi_linear_ver_0','equi_linear_ver_1','equi_linear_ver_2','equi_linear_ver_3'],
     'Equi. Linear', '#FF9800'),
    ('equi_rms_norm_ver_3',
     ['equi_rms_norm_ver_0','equi_rms_norm_ver_1','equi_rms_norm_ver_2','equi_rms_norm_ver_3'],
     'RMS Norm', '#9C27B0'),
    ('scaler_gated_gelu_ver_2',
     ['scaler_gated_gelu_ver_0','scaler_gated_gelu_ver_1','scaler_gated_gelu_ver_2','scaler_gated_gelu_ver_2'],
     'Gated GELU', '#F44336'),
    ('equi_geometric_attention_ver_3',
     ['equi_geometric_attention_ver_0','equi_geometric_attention_ver_1',
      'equi_geometric_attention_ver_2','equi_geometric_attention_ver_3'],
     'Attention', '#795548'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Per-kernel roofline, clean layout
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
draw_roofline(ax)

ver_sizes   = [55, 65, 80, 100]
ver_alphas  = [0.45, 0.65, 0.85, 1.0]

for papi_tgt, ver_tgts, fname, color in funcs:
    for n, filled in [(3, True), (4, False)]:
        fl = papi_flops.get((papi_tgt, n))
        by = papi_bytes.get((papi_tgt, n))
        if not fl or not by:
            continue
        ai = fl / by
        prev_gf = None
        for vi, (vtgt, sz, alpha) in enumerate(zip(ver_tgts, ver_sizes, ver_alphas)):
            ms = times[n].get(vtgt)
            if not ms:
                continue
            gf = fl / (ms/1000) / 1e9
            mfc = color if filled else 'white'
            marker = ['o','s','^','D'][vi]
            ax.scatter(ai, gf, s=sz, marker=marker, color=color,
                       facecolors=mfc, edgecolors=color,
                       linewidths=0.7 if filled else 2.0,
                       alpha=alpha, zorder=10)
            if prev_gf is not None and abs(gf - prev_gf)/max(gf,prev_gf) > 0.05:
                ax.annotate('', xy=(ai, gf), xytext=(ai, prev_gf),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.0, alpha=0.4))
            prev_gf = gf
        # Label only v3 point per (function, n)
        gf_v3 = fl / (times[n].get(ver_tgts[-1], 1)/1000) / 1e9
        xoff = 7 if n == 3 else -65
        yoff = 4 if n == 3 else -12
        style = 'bold' if n==3 else 'normal'
        ax.annotate(f'{fname}\nn={n}', (ai, gf_v3),
                    textcoords='offset points', xytext=(xoff, yoff),
                    fontsize=8, color=color, fontweight=style, alpha=1.0 if n==3 else 0.8)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 8000); ax.set_ylim(0.2, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title('Roofline — All Kernels, All Versions (v0→v3)\nFilled markers = n=3  |  Hollow markers = n=4  |  Tiger Lake i7-1165G7', fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')

func_leg = [mlines.Line2D([],[],color=c,marker='D',ls='None',ms=9,label=n) for _,_,n,c in funcs]
ver_leg  = [mlines.Line2D([],[],color='gray',marker=m,ls='None',ms=7,alpha=a,label=l)
            for m,a,l in zip(['o','s','^','D'],ver_alphas,['v0','v1','v2','v3'])]
n_leg    = [mlines.Line2D([],[],color='gray',marker='o',ls='None',ms=7,label='n=3 (filled)'),
            mlines.Line2D([],[],color='gray',marker='o',ls='None',ms=7,mfc='white',mew=2,label='n=4 (hollow)')]
l1 = ax.legend(handles=func_leg, loc='lower right', fontsize=9, title='Function', framealpha=0.9, ncol=1)
ax.add_artist(l1)
l2 = ax.legend(handles=ver_leg+n_leg, loc='upper left', fontsize=9, title='Version / Size', framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig('benchmarks/results/run_2026_06_02/roofline_all_versions.png', dpi=150, bbox_inches='tight')
print('Saved roofline_all_versions.png')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Whole-project roofline v0-v3, n=1..4
# ═══════════════════════════════════════════════════════════════════════════════
ver_funcs = {
    'v0': ['geometric_product_v0','equi_join_v0','equi_linear_ver_0','equi_rms_norm_ver_0','scaler_gated_gelu_ver_0','equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1','equi_join_v1','equi_linear_ver_1','equi_rms_norm_ver_1','scaler_gated_gelu_ver_1','equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2','equi_join_v2','equi_linear_ver_2','equi_rms_norm_ver_2','scaler_gated_gelu_ver_2','equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3','equi_join_v3','equi_linear_ver_3','equi_rms_norm_ver_3','scaler_gated_gelu_ver_2','equi_geometric_attention_ver_3'],
}
ver_style = {
    'v0': ('#B0BEC5','o','v0 — Baseline C++'),
    'v1': ('#FF9800','s','v1 — Math optimizations'),
    'v2': ('#2196F3','^','v2 — Scalar memory'),
    'v3': ('#4CAF50','D','v3 — SIMD + FMA + parallel'),
}

points = {}
for n in [1,2,3,4]:
    for ver, fl_list in ver_funcs.items():
        total_ms = sum(times[n].get(f,0) for f in fl_list)
        total_fl = sum((estimate_target_cost(f,n) or {}).get('estimated_flops_per_call',0) for f in fl_list)
        total_by = sum((estimate_target_cost(f,n) or {}).get('estimated_bytes_per_call',0) for f in fl_list)
        ai = total_fl/total_by if total_by else 0
        gf = total_fl/(total_ms/1000)/1e9 if total_ms else 0
        points.setdefault(ver,[]).append((ai, gf, n))

fig, ax = plt.subplots(figsize=(10, 6))
draw_roofline(ax)

for ver, pts in points.items():
    color, marker, label = ver_style[ver]
    ais = [p[0] for p in pts]
    gfs = [p[1] for p in pts]
    ns  = [p[2] for p in pts]
    ax.plot(ais, gfs, '-', color=color, lw=1.3, alpha=0.5, zorder=6)
    ax.scatter(ais, gfs, color=color, marker=marker, s=110, zorder=10,
               edgecolors='white', lw=0.8, label=label)
    for ai, gf, n in pts:
        if ver == 'v3' or (ver == 'v0' and n == 4):
            xoff = 6 if ver == 'v3' else -40
            yoff = 4 if ver == 'v3' else -12
            ax.annotate(f'n={n}', (ai, gf), textcoords='offset points',
                        xytext=(xoff, yoff), fontsize=7.5,
                        color='#2E7D32' if ver=='v3' else '#555')

# Draw v0→v3 arrows for each n
for n_idx in range(4):
    v0ai, v0gf, _ = points['v0'][n_idx]
    v3ai, v3gf, _ = points['v3'][n_idx]
    ax.annotate('', xy=(v3ai, v3gf), xytext=(v0ai, v0gf),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.1, alpha=0.55))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 5000); ax.set_ylim(0.8, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title('Whole-Project Roofline — All Versions, n=1..4\nTiger Lake i7-1165G7  |  6 kernels combined  |  Arrows: v0→v3', fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

plt.tight_layout()
plt.savefig('benchmarks/results/run_2026_06_02/roofline_project_versions.png', dpi=150, bbox_inches='tight')
print('Saved roofline_project_versions.png')
