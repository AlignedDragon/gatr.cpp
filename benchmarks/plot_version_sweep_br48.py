import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

OUT = Path('benchmarks/results/FINAL_PLOTS')
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS    = 89.6   # single-core, base 2.8 GHz
PEAK_GFLOPS_MT = 4 * PEAK_GFLOPS  # 4 physical cores
PEAK_BW_GBS    = 45.0
RIDGE_POINT    = PEAK_GFLOPS    / PEAK_BW_GBS
RIDGE_MT       = PEAK_GFLOPS_MT / PEAK_BW_GBS

# Runtime: per-target min_ms from the project version sweep (B=2n, H=4n, T=128n, C=8).
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

# Python (PyTorch 1-thread) timings for comparison
_pf_path = Path('benchmarks/results/run_20260615/per_function.json')
py_times: dict[str, dict[int, float]] = {}
attn_py_ms: dict[int, float] = {}
if _pf_path.exists():
    _pf = json.load(open(_pf_path))
    for op, dat in _pf['pointwise'].items():
        py_times[op] = {n: dat['versions']['py'][i] for i, n in enumerate(dat['n_values'])}
    attn_py_ms = {int(k): float(v) for k, v in _pf['attention']['py'].items()}

# PAPI per-kernel roofline data (new dims B=2n H=4n T=128n C=8; 8-thread; current code).
# n=3 → filled markers, n=9 → hollow markers.
papi_flops, papi_bytes, papi_gflops_krnl, papi_ai_krnl = {}, {}, {}, {}
for n, fn in [(3, 'benchmarks/results/version_sweep_br48_papi_n3.json'),
              (9, 'benchmarks/results/version_sweep_br48_papi_n9.json')]:
    if not Path(fn).exists():
        continue
    d = json.load(open(fn))
    for r in d['results']:
        fl = r.get('flops_per_call'); by = r.get('memory_bytes_per_call')
        gf = r.get('gflops');         ai = r.get('arithmetic_intensity')
        if fl and by:
            papi_flops[(r['target'], n)] = fl
            papi_bytes[(r['target'], n)] = by
        if gf and ai:
            papi_gflops_krnl[(r['target'], n)] = gf
            papi_ai_krnl[(r['target'], n)]     = ai

vc = {'v0': '#B0BEC5', 'v1': '#FF9800', 'v2': '#2196F3', 'v3': '#2E7D32'}
vl = {'v0': 'v0 - Baseline', 'v1': 'v1 - Math', 'v2': 'v2 - Scalar mem.', 'v3': 'v3 - SIMD + AVX-512 attn'}

ver_funcs = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0', 'equi_rms_norm_ver_0', 'equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1', 'equi_rms_norm_ver_1', 'equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2', 'equi_rms_norm_ver_2', 'equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3', 'equi_rms_norm_ver_3', 'equi_geometric_attention_ver_3_1'],
}
# Only sum functions that ALL base versions (v0-v3) have data for at each n.
# At n>=6, v0/v1/v2 attention is missing → exclude attention from all versions' totals at those n.
def fair_total(n: int, ver: str) -> float:
    funcs = ver_funcs[ver]
    total = 0.0
    for fi, f in enumerate(funcs):
        if all(times[n].get(ver_funcs[v][fi], 0) > 0 for v in ['v0', 'v1', 'v2', 'v3']):
            total += times[n].get(f, 0)
    return total

tot = {n: {v: fair_total(n, v) for v in ['v0','v1','v2','v3']} for n in NS}
# Track which n values have attention included (for subtitle note)
attn_included_ns = [n for n in NS if all(times[n].get(ver_funcs[v][4], 0) > 0 for v in ['v0','v1','v2','v3'])]
attn_excluded_ns = [n for n in NS if n not in attn_included_ns]

func_info = [
    ('Geom.\nProduct', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3']),
    ('Equi.\nJoin',    ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('Equi.\nLinear',  ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('RMS\nNorm',      ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('Attention',      ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1', 'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3_1']),
]

N_LO = 4 if 4 in NS else NS[len(NS) // 2]
N_HI = NS[-1]  # always n=9 for all functions

# For attention at n>=6, v0/v1/v2 timings are missing.
# Extrapolate them using n^4 scaling from the last measured n.
def _attn_estimate(target, n):
    t = times[n].get(target, 0)
    if t > 0:
        return t, False
    for nn in reversed([m for m in NS if m < n]):
        t_ref = times[nn].get(target, 0)
        if t_ref > 0:
            return t_ref * (n / nn) ** 4.0, True
    return 0, False

def get_timing(funcs, vi, n_val, is_attn):
    """Return (t0, tv, estimated) for speedup computation."""
    t0_raw = times[n_val].get(funcs[0], 0)
    tv_raw = times[n_val].get(funcs[vi], 0)
    if is_attn and (not t0_raw or not tv_raw):
        t0, e0 = _attn_estimate(funcs[0], n_val)
        tv, ev = _attn_estimate(funcs[vi], n_val)
        return t0, tv, (e0 or ev)
    return t0_raw, tv_raw, False

# PLOT 1: speedup bars — n=N_LO (solid) and n=N_HI (hatched) for all functions.
# For attention at n=9, v0/v1/v2 are extrapolated (n^4); v3 is measured.
fig, ax = plt.subplots(figsize=(15, 6.4))
bw, gap = 0.23, 0.80
x = 0
xticks, xlabels = [], []
for fname, funcs in func_info:
    is_attn = 'Attention' in fname
    for ni, (n_val, hatch, nalpha) in enumerate([(N_LO, '', 1.0), (N_HI, '//', 0.85)]):
        has_estimated = False
        for vi, (ver, color) in enumerate(vc.items()):
            t0, tv, estimated = get_timing(funcs, vi, n_val, is_attn)
            has_estimated = has_estimated or estimated
            bx = x + ni * (4 * bw + 0.14) + vi * bw
            if not t0 or not tv:
                ax.bar(bx, 1.0, bw, color='#e0e0e0', edgecolor='white', lw=0.6, zorder=3)
                continue
            sp = t0 / tv
            ec = 'black' if estimated else 'white'
            ax.bar(bx, sp, bw, color=color, edgecolor=ec, lw=0.8 if estimated else 0.6,
                   alpha=nalpha, hatch=hatch, zorder=3)
            if sp > 1.05:
                lbl = f'{sp:.1f}x' if sp < 10 else f'{sp:.0f}x'
                ax.text(bx + bw / 2, sp * 1.06, lbl, ha='center', va='bottom',
                        fontsize=7.5, color='#111', fontweight='bold', rotation=90,
                        zorder=12, clip_on=False,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.82))
        mid = x + ni * (4 * bw + 0.14) + 1.5 * bw
        xticks.append(mid)
        est_note = '\n(v0 est.)' if has_estimated and is_attn else ''
        xlabels.append(f'{fname}\nn={n_val}{est_note}')
    x += 2 * (4 * bw + 0.14) + gap

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=9)
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_yscale('log')
ax.set_ylim(0.5, 30000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x' if y >= 2 else f'{y:.1f}x'))
ax.set_ylabel('Speedup over v0  (log scale)', fontsize=12)
ax.set_title(f'Per-function speedup — n={N_LO} (solid) and n={N_HI} (hatched)\n'
             f'Threads=8, B=2n H=4n T=128n C=8 | Tiger Lake i7-1165G7 | * attention n=9: v0/v1/v2 extrapolated (n⁴)', fontsize=11)
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
# Solid lines = n where all 4 versions have attention; dashed = non-attention only
ns_full    = [n for n in NS if n in attn_included_ns]   # n=1..5, attention included
ns_noattn  = [n for n in NS if n in attn_excluded_ns]   # n=6..9, non-attention only
# x indices
xi_full   = [NS.index(n) for n in ns_full]
xi_noattn = [NS.index(n) for n in ns_noattn]
split_xi  = xi_full[-1]  # index of last full-comparison n (n=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xl = [f'n={n}\n(T={128*n})' for n in NS]
xr = range(len(NS))

for ver, color in vc.items():
    rts = [tot[n][ver] for n in NS]
    ax1.plot(xi_full, [rts[i] for i in xi_full], 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if xi_noattn:
        dash_xi = [split_xi] + xi_noattn
        ax1.plot(dash_xi, [rts[i] for i in dash_xi], 's--', color=color, lw=1.5, ms=4, zorder=3, alpha=0.7)
    ax1.text(len(NS) - 0.92, rts[-1], f'{rts[-1]:.0f}ms', fontsize=7.5, color=color, va='center')
ax1.set_xticks(xr)
ax1.set_xticklabels(xl, fontsize=9)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms)', fontsize=12)
ax1.set_title('Total Project Runtime vs Problem Size\n(solid = incl. attention | dashed = non-attn only)', fontsize=10)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')
ax1.set_xlim(-0.3, len(NS) - 0.2)
# mark the methodology boundary
if xi_noattn:
    ax1.axvspan(split_xi + 0.5, len(NS) - 0.3, alpha=0.05, color='gray', zorder=0)
    ax1.axvline(split_xi + 0.5, color='gray', ls=':', lw=1.5, alpha=0.5)
    ax1.text(split_xi + 0.6, ax1.get_ylim()[0] * 3, 'no attn\n(v0/v1/v2\ntoo slow)',
             fontsize=7, color='gray', va='bottom')

for ver, color in vc.items():
    sp = [tot[n]['v0'] / tot[n][ver] for n in NS]
    ax2.plot(xi_full, [sp[i] for i in xi_full], 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if xi_noattn:
        dash_xi = [split_xi] + xi_noattn
        ax2.plot(dash_xi, [sp[i] for i in dash_xi], 's--', color=color, lw=1.5, ms=4, zorder=3, alpha=0.7)
    ax2.text(len(NS) - 0.92, sp[-1], f'{sp[-1]:.0f}x', fontsize=7.5, color=color, va='center')
ax2.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.set_xticks(xr)
ax2.set_xticklabels(xl, fontsize=9)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x' if y >= 2 else f'{y:.1f}x'))
ax2.set_ylabel('Speedup over v0', fontsize=12)
ax2.set_title('Total Speedup over Baseline vs Problem Size\n(solid = incl. attention | dashed = non-attn only)', fontsize=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(alpha=0.3, which='both')
ax2.set_xlim(-0.3, len(NS) - 0.2)
if xi_noattn:
    ax2.axvspan(split_xi + 0.5, len(NS) - 0.3, alpha=0.05, color='gray', zorder=0)
    ax2.axvline(split_xi + 0.5, color='gray', ls=':', lw=1.5, alpha=0.5)
plt.suptitle('EzGATr Runtime Scaling  |  Threads=8, B=2n H=4n T=128n C=8  |  Tiger Lake i7-1165G7', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_runtime.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_runtime.png')
plt.close()


def draw_roofline(ax, peak=PEAK_GFLOPS, bw=PEAK_BW_GBS, label_suffix=''):
    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(bw * ai_range, peak)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ridge = peak / bw
    ax.axvline(ridge, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.013, bw * 0.013 * 1.3, f'Mem BW: {bw} GB/s', fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, peak * 1.08, f'Peak: {peak:.0f} GFLOPS{label_suffix}', fontsize=8.5, ha='right')
    ax.text(ridge * 1.1, peak * 0.45, f'Ridge\n{ridge:.1f} F/B', fontsize=7.5, color='gray')


# PLOT 3: per-kernel roofline using self-consistent PAPI gflops/AI from old-dim 8T measurements.
# IMPORTANT: PAPI files use old dim formula (B=64 T=16n); gflops and AI are self-consistent
# within each file — we use them directly, NOT mixed with new-dim sweep timings.
funcs = [
    ('geometric_product', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3'], 'Geom. Product', '#2196F3'),
    ('equi_join', ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3'], 'Equi. Join', '#4CAF50'),
    ('equi_linear', ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3'], 'Equi. Linear', '#FF9800'),
    ('equi_rms_norm', ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3'], 'RMS Norm', '#9C27B0'),
    ('attention', ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1', 'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3'], 'Attention', '#795548'),
]
_PY_KEY_MAP = {
    'geometric_product': ('geometric_product', None),
    'equi_join':         ('equi_join',         None),
    'equi_linear':       ('equi_linear',        None),
    'equi_rms_norm':     ('equi_rms_norm',      None),
    'attention':         (None,                 True),  # use attn_py_ms
}
ver_sizes = [55, 65, 80, 100]
ver_alphas = [0.45, 0.65, 0.85, 1.0]

fig, ax = plt.subplots(figsize=(12, 7))
draw_roofline(ax)
for fname_key, ver_tgts, fname, color in funcs:
    for n, filled in [(3, True), (9, False)]:
        mfc_f = color if filled else 'white'

        # Arrow from v0 to v3 showing improvement direction
        ai0 = papi_ai_krnl.get((ver_tgts[0], n)); gf0 = papi_gflops_krnl.get((ver_tgts[0], n))
        ai3 = papi_ai_krnl.get((ver_tgts[-1], n)); gf3 = papi_gflops_krnl.get((ver_tgts[-1], n))
        if ai0 and gf0 and ai3 and gf3:
            ax.annotate('', xy=(ai3, gf3), xytext=(ai0, gf0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=0.45))

        for vi, (vtgt, sz, alpha) in enumerate(zip(ver_tgts, ver_sizes, ver_alphas)):
            ai = papi_ai_krnl.get((vtgt, n))
            gf = papi_gflops_krnl.get((vtgt, n))
            if ai is None or gf is None:
                continue
            marker = ['o', 's', '^', 'D'][vi]
            ax.scatter(ai, gf, s=sz, marker=marker, color=color, facecolors=mfc_f,
                       edgecolors=color, linewidths=0.7 if filled else 2.0, alpha=alpha, zorder=10)
        # Label v3 at n=3 and n=4
        ai3 = papi_ai_krnl.get((ver_tgts[-1], n))
        gf3 = papi_gflops_krnl.get((ver_tgts[-1], n))
        if ai3 and gf3:
            xoff = 7 if n == 3 else -65
            yoff = 4 if n == 3 else -12
            ax.annotate(f'{fname}\nn={n}', (ai3, gf3), textcoords='offset points',
                        xytext=(xoff, yoff), fontsize=8, color=color,
                        fontweight='bold' if n == 3 else 'normal', alpha=1.0 if n == 3 else 0.8)
        # Python (1-thread) scatter using v3 AI as reference
        py_key, is_attn = _PY_KEY_MAP[fname_key]
        ref_tgt = ver_tgts[-1]  # use v3 PAPI AI as reference for Python
        ai_ref = papi_ai_krnl.get((ref_tgt, n))
        fl_ref = papi_flops.get((ref_tgt, n))
        if ai_ref and fl_ref and n == 3:  # show Python only at n=3
            py_ms = py_times.get(py_key, {}).get(n) if py_key else attn_py_ms.get(n)
            if py_ms:
                gf_py = fl_ref / (py_ms / 1000) / 1e9
                mfc_py = color if filled else 'white'
                ax.scatter(ai_ref, gf_py, s=130, marker='*', color=color,
                           facecolors=mfc_py, edgecolors=color, linewidths=1.2,
                           alpha=0.9, zorder=11)
ax.set_xscale('log'); ax.set_yscale('log')
all_gf = [v for v in papi_gflops_krnl.values()]
ax.set_xlim(0.01, 8000)
ax.set_ylim(0.01, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title('Roofline - all kernels, all versions (v0-v3) | PAPI (8T, B=2n H=4n T=128n C=8)\n'
             'Filled = n=3 | Hollow = n=9 | ★ Python (1T) at n=3 | 1-core peak 89.6 GFLOPS | Tiger Lake i7-1165G7', fontsize=10)
ax.grid(True, which='both', alpha=0.2, ls='--')
func_leg = [mlines.Line2D([], [], color=c, marker='D', ls='None', ms=9, label=n) for _, _, n, c in funcs]
ver_leg = [mlines.Line2D([], [], color='gray', marker=m, ls='None', ms=7, alpha=a, label=l)
           for m, a, l in zip(['o', 's', '^', 'D'], ver_alphas, ['v0', 'v1', 'v2', 'v3'])]
n_leg = [mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, label='n=3 (filled)'),
         mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, mfc='white', mew=2, label='n=9 (hollow)'),
         mlines.Line2D([], [], color='gray', marker='*', ls='None', ms=10, label='★ Python (1T) n=3')]
l1 = ax.legend(handles=func_leg, loc='lower right', fontsize=9, title='Function', framealpha=0.9, ncol=1)
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, loc='upper left', fontsize=9, title='Version / Size', framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_roofline_kernels.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_roofline_kernels.png')
plt.close()

# PLOT 4: whole-project roofline — PAPI-anchored FLOPs/bytes + br48 sweep wall-time.
# Cost models gave physically impossible GFLOPS; PAPI measurements anchor the FLOPs correctly.
# Scaling exponents empirically measured from PAPI n=3→n=4 on B=2n H=4n T=128n C=8.
_papi3_br: dict[str, dict] = {}
_papi3_br_path = Path('benchmarks/results/version_sweep_br48_papi_n3.json')
if _papi3_br_path.exists():
    for _r in json.loads(_papi3_br_path.read_text())['results']:
        _fl = _r.get('flops_per_call'); _by = _r.get('memory_bytes_per_call')
        if _fl and _by:
            _papi3_br[_r['target']] = {'flops': _fl, 'bytes': _by}

# Attention completely dominates project FLOPs (≫100× non-attention at any n≥1).
# Use measured exponents: FL∝n^4.0, BY∝n^3.0 for attention; n^2.5/4.5 for non-attn.
_N_REF = 3
_ATTN_FL_EXP, _ATTN_BY_EXP = 4.0, 3.0
_NA_FL_EXP,   _NA_BY_EXP   = 2.5, 4.5
# PEAK_GFLOPS_MT / PEAK_BW_GBS_MT / RIDGE_MT already defined at top of file

_VER_MAX_N = {'v0': 5, 'v1': 5, 'v2': 5, 'v3': 9}
_NON_ATTN = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0', 'equi_rms_norm_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1', 'equi_rms_norm_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2', 'equi_rms_norm_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3', 'equi_rms_norm_ver_3'],
}
_ATTN = {v: f'equi_geometric_attention_ver_{i}' for i, v in enumerate(['v0', 'v1', 'v2', 'v3'])}
_ATTN['v3'] = 'equi_geometric_attention_ver_3_1'
# PAPI FLOPs/bytes reference — ver_3_1 not in PAPI, use ver_3 model (same complexity)
_PAPI_ATTN_REF = dict(_ATTN)
_PAPI_ATTN_REF['v3'] = 'equi_geometric_attention_ver_3'

_PY_OP_MAP_BR = {
    'geometric_product_v0': 'geometric_product',
    'equi_join_v0':         'equi_join',
    'equi_linear_ver_0':    'equi_linear',
    'equi_rms_norm_ver_0':  'equi_rms_norm',
}


def _proj_br(ver: str, n: int):
    if not _papi3_br:
        return None
    total_fl = total_by = total_ms = 0.0
    for tgt in _NON_ATTN[ver]:
        ref = _papi3_br.get(tgt)
        t = times.get(n, {}).get(tgt)
        if not ref or t is None:
            return None
        total_fl += ref['flops'] * (n / _N_REF) ** _NA_FL_EXP
        total_by += ref['bytes'] * (n / _N_REF) ** _NA_BY_EXP
        total_ms += t
    ref_a = _papi3_br.get(_PAPI_ATTN_REF[ver])
    t_a   = times.get(n, {}).get(_ATTN[ver])
    if not ref_a or t_a is None:
        return None
    total_fl += ref_a['flops'] * (n / _N_REF) ** _ATTN_FL_EXP
    total_by += ref_a['bytes'] * (n / _N_REF) ** _ATTN_BY_EXP
    total_ms += t_a
    if not total_by or not total_ms:
        return None
    return total_fl / total_by, total_fl / (total_ms / 1000) / 1e9


def _proj_py_br(n: int):
    """Python project roofline point using v0 PAPI FLOPs and Python timings."""
    if not _papi3_br or not py_times:
        return None
    total_fl = total_by = total_ms = 0.0
    for tgt, py_op in _PY_OP_MAP_BR.items():
        ref = _papi3_br.get(tgt)
        py_ms = py_times.get(py_op, {}).get(n)
        if not ref or py_ms is None:
            return None
        total_fl += ref['flops'] * (n / _N_REF) ** _NA_FL_EXP
        total_by += ref['bytes'] * (n / _N_REF) ** _NA_BY_EXP
        total_ms += py_ms
    ref_a = _papi3_br.get(_ATTN['v0'])
    py_ms_a = attn_py_ms.get(n)
    if not ref_a or py_ms_a is None:
        return None
    total_fl += ref_a['flops'] * (n / _N_REF) ** _ATTN_FL_EXP
    total_by += ref_a['bytes'] * (n / _N_REF) ** _ATTN_BY_EXP
    total_ms += py_ms_a
    if not total_by or not total_ms:
        return None
    return total_fl / total_by, total_fl / (total_ms / 1000) / 1e9


ver_style = {
    'v0': ('#B0BEC5', 'o', 'v0 - Baseline C++'),
    'v1': ('#FF9800', 's', 'v1 - Math optimizations'),
    'v2': ('#2196F3', '^', 'v2 - Scalar memory'),
    'v3': ('#2E7D32', 'D', 'v3 - SIMD + AVX-512 attn'),
}
br_points: dict[str, list] = {}
for _ver in ['v0', 'v1', 'v2', 'v3']:
    for _n in [nn for nn in NS if nn <= _VER_MAX_N[_ver]]:
        _pt = _proj_br(_ver, _n)
        if _pt:
            br_points.setdefault(_ver, []).append((_pt[0], _pt[1], _n))

fig, ax = plt.subplots(figsize=(10, 6))
draw_roofline(ax)
for _ver, pts in br_points.items():
    color, marker, label = ver_style[_ver]
    ais = [p[0] for p in pts]; gfs = [p[1] for p in pts]
    ax.plot(ais, gfs, '-', color=color, lw=1.3, alpha=0.5, zorder=6)
    ax.scatter(ais, gfs, color=color, marker=marker, s=110, zorder=10, edgecolors='white', lw=0.8, label=label)
    for ai, gf, n in pts:
        if _ver == 'v3' or n in (pts[0][2], pts[-1][2]):
            ax.annotate(f'n={n}', (ai, gf), textcoords='offset points',
                        xytext=(6, 4), fontsize=7.5,
                        color='#2E7D32' if _ver == 'v3' else '#555')
# Cross-version connecting lines at each n (v0→v1→v2→v3 "rungs")
all_ns_br = sorted({n for pts in br_points.values() for _, _, n in pts})
for _n in all_ns_br:
    rung = [(ai, gf) for ver in ['v0', 'v1', 'v2', 'v3']
            for ai, gf, nn in br_points.get(ver, []) if nn == _n]
    if len(rung) > 1:
        ax.plot([p[0] for p in rung], [p[1] for p in rung],
                '-', color='#888', lw=1.0, alpha=0.40, zorder=5)
# Python (1-thread) project reference points at n=1..4
py_br_pts = []
for _n in [1, 2, 3, 4]:
    _pt = _proj_py_br(_n)
    if _pt:
        py_br_pts.append((_pt[0], _pt[1], _n))
if py_br_pts:
    _py_ais = [p[0] for p in py_br_pts]; _py_gfs = [p[1] for p in py_br_pts]
    ax.plot(_py_ais, _py_gfs, '-', color='#9E9E9E', lw=1.2, alpha=0.5, zorder=6)
    ax.scatter(_py_ais, _py_gfs, color='#757575', marker='*', s=160, zorder=11,
               edgecolors='#444', linewidths=0.8, label='Python (1T)')
    for _ai, _gf, _n in py_br_pts:
        ax.annotate(f'n={_n}', (_ai, _gf), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='#555')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.01, 8000)
ax.set_ylim(0.05, 200)
ax.set_xlabel('Arithmetic Intensity [estimated FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [estimated GFLOPS]', fontsize=12)
_max_v012 = max(_VER_MAX_N[v] for v in ['v0', 'v1', 'v2'])
ax.set_title(
    f'Whole-project roofline by optimization version\n'
    f'B=2n H=4n T=128n C=8  |  Tiger Lake i7-1165G7  |  Arrows: v0→v3',
    fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / 'sweep_br48_roofline_project.png', dpi=150, bbox_inches='tight')
print('Saved sweep_br48_roofline_project.png')
plt.close()
print('done')
