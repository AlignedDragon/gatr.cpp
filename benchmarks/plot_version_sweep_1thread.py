import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

OUT = Path('benchmarks/results/FINAL_PLOTS')
OUT.mkdir(parents=True, exist_ok=True)

PEAK_GFLOPS = 89.6
PEAK_BW_GBS = 45.0
RIDGE_POINT = PEAK_GFLOPS / PEAK_BW_GBS

# Runtime: per-target min_ms from the project version sweep (B=2n, H=4n, T=128n, C=8).
sweep = json.load(open('benchmarks/results/version_sweep_1thread.json'))
times = {}            # times[n][target] = min_ms
NS = []
for agg in sweep['results']:
    n = agg['n']
    times.setdefault(n, {})
    NS.append(n)
    for t in agg['targets']:
        times[n][t['target']] = t['min_ms']
NS = sorted(set(NS))

# PAPI per-kernel data from current-code new-dims files (June 15).
# n=3 → filled markers; n=9 → hollow markers.
papi_flops, papi_bytes, papi_gflops_krnl, papi_ai_krnl = {}, {}, {}, {}
papi3: dict[str, dict] = {}   # also serves as project-roofline reference
for n, fn in [(3, 'benchmarks/results/papi_1thread_n3.json'),
              (9, 'benchmarks/results/papi_1thread_n9.json')]:
    if not Path(fn).exists():
        continue
    d = json.load(open(fn))
    for r in d['results']:
        fl = r.get('flops_per_call'); by = r.get('memory_bytes_per_call')
        gf = r.get('gflops');         ai = r.get('arithmetic_intensity')
        if fl and by and gf:
            papi_flops[(r['target'], n)]       = fl
            papi_bytes[(r['target'], n)]       = by
            papi_gflops_krnl[(r['target'], n)] = gf
            if ai:
                papi_ai_krnl[(r['target'], n)] = ai
            if n == 3:
                papi3[r['target']] = {'flops': fl, 'bytes': by, 'gflops': gf}
N_REF = 3

# Python (PyTorch 1-thread) timings for comparison
_pf_path = Path('benchmarks/results/run_20260615/per_function.json')
py_times: dict[str, dict[int, float]] = {}
attn_py_ms: dict[int, float] = {}
if _pf_path.exists():
    _pf = json.load(open(_pf_path))
    for op, dat in _pf['pointwise'].items():
        py_times[op] = {n: dat['versions']['py'][i] for i, n in enumerate(dat['n_values'])}
    attn_py_ms = {int(k): float(v) for k, v in _pf['attention']['py'].items()}
# Exponents measured from PAPI n=3 vs n=9 on new dims.
NA_FL_EXP   = 2.0
NA_BY_EXP   = 5.5
ATTN_FL_EXP = 4.0
ATTN_BY_EXP = 3.29

vc = {'v0': '#B0BEC5', 'v1': '#FF9800', 'v2': '#2196F3', 'v3': '#2E7D32'}
vl = {'v0': 'v0 - Baseline', 'v1': 'v1 - Math', 'v2': 'v2 - Scalar mem.', 'v3': 'v3 - SIMD + AVX-512 attn'}

ver_funcs = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0', 'equi_rms_norm_ver_0', 'equi_geometric_attention_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1', 'equi_rms_norm_ver_1', 'equi_geometric_attention_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2', 'equi_rms_norm_ver_2', 'equi_geometric_attention_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3', 'equi_rms_norm_ver_3', 'equi_geometric_attention_ver_3_1'],
}
NON_ATTN_GROUPS_SW = [
    ('geometric_product', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3']),
    ('equi_join',         ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('equi_linear',       ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('equi_rms_norm',     ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('scaler_gated_gelu', ['scaler_gated_gelu_ver_0', 'scaler_gated_gelu_ver_1', 'scaler_gated_gelu_ver_2', 'scaler_gated_gelu_ver_3']),
]
VER_IDX_SW = {'v0': 0, 'v1': 1, 'v2': 2, 'v3': 3}


def total_sw(n, ver):
    idx = VER_IDX_SW[ver]
    vals = [times.get(n, {}).get(funcs[idx]) for _, funcs in NON_ATTN_GROUPS_SW]
    present = [v for v in vals if v is not None]
    return sum(present) if len(present) == len(NON_ATTN_GROUPS_SW) else None


tot = {n: {v: total_sw(n, v) for v in ver_funcs} for n in NS}


def common_groups_sweep(n):
    return [(gname, funcs) for gname, funcs in NON_ATTN_GROUPS_SW]

func_info = [
    ('Geom.\nProduct', ['geometric_product_v0', 'geometric_product_v1', 'geometric_product_v2', 'geometric_product_v3']),
    ('Equi.\nJoin',    ['equi_join_v0', 'equi_join_v1', 'equi_join_v2', 'equi_join_v3']),
    ('Equi.\nLinear',  ['equi_linear_ver_0', 'equi_linear_ver_1', 'equi_linear_ver_2', 'equi_linear_ver_3']),
    ('RMS\nNorm',      ['equi_rms_norm_ver_0', 'equi_rms_norm_ver_1', 'equi_rms_norm_ver_2', 'equi_rms_norm_ver_3']),
    ('Attention',      ['equi_geometric_attention_ver_0', 'equi_geometric_attention_ver_1', 'equi_geometric_attention_ver_2', 'equi_geometric_attention_ver_3_1']),
]

# Two n values for the bar chart: a mid size and the largest measured.
N_LO = 4 if 4 in NS else NS[len(NS) // 2]
N_HI = NS[-1]
# Attention v0/v1 capped at n=3; use n=2 and n=3 for attention bars
ATTN_V0_MAX = max(n for n in NS if times.get(n, {}).get('equi_geometric_attention_ver_0'))
ATTN_N_LO = max(n for n in NS if n <= ATTN_V0_MAX - 1) if ATTN_V0_MAX > min(NS) else ATTN_V0_MAX
ATTN_N_HI = ATTN_V0_MAX

# Per-function (n_lo, n_hi) — attention uses its own range where all versions have data
func_n_vals = {fname: (N_LO, N_HI) for fname, _ in func_info}
func_n_vals['Attention'] = (ATTN_N_LO, ATTN_N_HI)

# PLOT 1: speedup bars
fig, ax = plt.subplots(figsize=(15, 6.4))
bw, gap = 0.23, 0.80
x = 0
xticks, xlabels = [], []
for fname, funcs in func_info:
    n_lo_f, n_hi_f = func_n_vals[fname]
    for ni, (n_val, hatch, nalpha) in enumerate([(n_lo_f, '', 1.0), (n_hi_f, '//', 0.85)]):
        for vi, (ver, color) in enumerate(vc.items()):
            t0 = times[n_val].get(funcs[0])
            tv = times[n_val].get(funcs[vi])
            if not t0 or not tv:
                continue
            sp = t0 / tv
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
ax.set_title(f'Per-function speedup - n={N_LO} (solid) and n={N_HI} (hatched) | Attention: n={ATTN_N_LO}/{ATTN_N_HI}\nThreads=1 | Tiger Lake i7-1165G7', fontsize=12)
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
plt.savefig(OUT / 'sweep_1thread_speedup.png', dpi=150, bbox_inches='tight')
print('Saved sweep_1thread_speedup.png')
plt.close()

# PLOT 2: runtime scaling + total speedup, n=1..max
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xl = [f'n={n}\n(T={128*n})' for n in NS]
xr = range(len(NS))

for ver, color in vc.items():
    ns_v  = [n for n in NS if tot[n][ver] is not None]
    rts_v = [tot[n][ver] for n in ns_v]
    xr_v  = [NS.index(n) for n in ns_v]
    ax1.plot(xr_v, rts_v, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if rts_v:
        ax1.text(xr_v[-1] + 0.1, rts_v[-1], f'{rts_v[-1]:.1f}ms', fontsize=7.5, color=color, va='center')
ax1.set_xticks(xr)
ax1.set_xticklabels(xl, fontsize=9)
ax1.set_yscale('log')
ax1.set_ylabel('Total runtime (ms)', fontsize=12)
ax1.set_title('Total Runtime vs Problem Size\n(5 non-attention kernels; attention speedup shown separately)', fontsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3, which='both')
ax1.set_xlim(-0.3, len(NS) - 0.2)

for ver, color in vc.items():
    ns_v = [n for n in NS if tot[n][ver] is not None and tot[n]['v0'] is not None]
    sp = [tot[n]['v0'] / tot[n][ver] for n in ns_v]
    xr_v = [NS.index(n) for n in ns_v]
    ax2.plot(xr_v, sp, 'o-', color=color, lw=2.2, ms=7, zorder=4, label=vl[ver])
    if sp:
        ax2.text(xr_v[-1] + 0.1, sp[-1], f'{sp[-1]:.1f}x', fontsize=7.5, color=color, va='center')
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
plt.suptitle('EzGATr Runtime Scaling  |  Threads=1  |  Tiger Lake i7-1165G7', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'sweep_1thread_runtime.png', dpi=150, bbox_inches='tight')
print('Saved sweep_1thread_runtime.png')
plt.close()


def draw_roofline(ax):
    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(PEAK_BW_GBS * ai_range, PEAK_GFLOPS)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ax.axvline(RIDGE_POINT, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.013, PEAK_BW_GBS * 0.013 * 1.3, f'Mem BW: {PEAK_BW_GBS} GB/s', fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, PEAK_GFLOPS * 1.08, f'Peak: {PEAK_GFLOPS} GFLOPS', fontsize=8.5, ha='right')
    ax.text(RIDGE_POINT * 1.1, 55, f'Ridge\n{RIDGE_POINT:.1f} F/B', fontsize=7.5, color='gray')


# PLOT 3: measured per-kernel roofline (PAPI flops/bytes, sweep min_ms), n=4 filled / n=6 hollow
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
for fname_key, ver_tgts, fname, color in funcs:
    for n, filled in [(3, True), (9, False)]:
        # Arrow from v0 to v3 showing improvement direction
        fl0 = papi_flops.get((ver_tgts[0], n)); by0 = papi_bytes.get((ver_tgts[0], n)); gf0 = papi_gflops_krnl.get((ver_tgts[0], n))
        fl3 = papi_flops.get((ver_tgts[-1], n)); by3 = papi_bytes.get((ver_tgts[-1], n)); gf3 = papi_gflops_krnl.get((ver_tgts[-1], n))
        if fl0 and by0 and gf0 and fl3 and by3 and gf3:
            ax.annotate('', xy=(fl3/by3, gf3), xytext=(fl0/by0, gf0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=0.45))

        for vi, (vtgt, sz, alpha) in enumerate(zip(ver_tgts, ver_sizes, ver_alphas)):
            fl = papi_flops.get((vtgt, n))
            by = papi_bytes.get((vtgt, n))
            gf = papi_gflops_krnl.get((vtgt, n))   # PAPI self-consistent, never above peak
            if not fl or not by or not gf:
                continue
            ai = fl / by
            mfc = color if filled else 'white'
            marker = ['o', 's', '^', 'D'][vi]
            ax.scatter(ai, gf, s=sz, marker=marker, color=color, facecolors=mfc,
                       edgecolors=color, linewidths=0.7 if filled else 2.0, alpha=alpha, zorder=10)
        # label v3
        fl3 = papi_flops.get((ver_tgts[-1], n))
        by3 = papi_bytes.get((ver_tgts[-1], n))
        gf3 = papi_gflops_krnl.get((ver_tgts[-1], n))
        if fl3 and by3 and gf3:
            ai3 = papi_ai_krnl.get((ver_tgts[-1], n), fl3 / by3)
            xoff = 7 if n == 3 else -65
            yoff = 4 if n == 3 else -12
            ax.annotate(f'{fname}\nn={n}', (ai3, gf3), textcoords='offset points',
                        xytext=(xoff, yoff), fontsize=8, color=color,
                        fontweight='bold' if n == 3 else 'normal', alpha=1.0 if n == 3 else 0.8)
        # Python star marker at n=3 using v3 PAPI AI and Python timing
        if n == 3:
            fl_ref = papi_flops.get((ver_tgts[-1], n))
            by_ref = papi_bytes.get((ver_tgts[-1], n))
            if fl_ref and by_ref:
                ai_ref = fl_ref / by_ref
                py_key = fname_key if fname_key != 'attention' else None
                py_ms = py_times.get(py_key, {}).get(n) if py_key else attn_py_ms.get(n)
                if py_ms:
                    gf_py = fl_ref / (py_ms / 1000) / 1e9
                    ax.scatter(ai_ref, gf_py, s=130, marker='*', color=color,
                               facecolors=color, edgecolors=color, linewidths=1.2,
                               alpha=0.9, zorder=11)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.01, 8000); ax.set_ylim(0.01, 200)
ax.set_xlabel('Arithmetic Intensity [FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [GFLOPS]', fontsize=12)
ax.set_title('Roofline - all kernels, all versions (v0-v3), measured with PAPI\n'
             'Filled = n=3 | Hollow = n=9 | ★ Python (1T) at n=3 | Tiger Lake i7-1165G7', fontsize=11)
ax.grid(True, which='both', alpha=0.2, ls='--')
func_leg = [mlines.Line2D([], [], color=c, marker='D', ls='None', ms=9, label=n) for _, _, n, c in funcs]
ver_leg = [mlines.Line2D([], [], color='gray', marker=m, ls='None', ms=7, alpha=a, label=l)
           for m, a, l in zip(['o', 's', '^', 'D'], ver_alphas, ['v0', 'v1', 'v2', 'v3'])]
n_leg = [mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, label='n=4 (filled)'),
         mlines.Line2D([], [], color='gray', marker='o', ls='None', ms=7, mfc='white', mew=2, label='n=9 (hollow)'),
         mlines.Line2D([], [], color='gray', marker='*', ls='None', ms=10, label='★ Python (1T) n=3')]
l1 = ax.legend(handles=func_leg, loc='lower right', fontsize=9, title='Function', framealpha=0.9, ncol=1)
ax.add_artist(l1)
ax.legend(handles=ver_leg + n_leg, loc='upper left', fontsize=9, title='Version / Size', framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(OUT / 'sweep_1thread_roofline_kernels.png', dpi=150, bbox_inches='tight')
print('Saved sweep_1thread_roofline_kernels.png')
plt.close()

# PLOT 4: whole-project roofline sweep (PAPI-anchored, same approach as roofline_project_versions.png).
# FLOPs/bytes: scaled from PAPI n=3 reference using measured exponents.
# Timing: version_sweep_1thread.json min_ms (new dims: B=2n H=4n T=128n C=8).
VER_MAX_N_ATTN = {'v0': 5, 'v1': 5, 'v2': 5, 'v3': 9}
NON_ATTN_REFS = {
    'v0': ['geometric_product_v0', 'equi_join_v0', 'equi_linear_ver_0',
           'equi_rms_norm_ver_0', 'scaler_gated_gelu_ver_0'],
    'v1': ['geometric_product_v1', 'equi_join_v1', 'equi_linear_ver_1',
           'equi_rms_norm_ver_1', 'scaler_gated_gelu_ver_1'],
    'v2': ['geometric_product_v2', 'equi_join_v2', 'equi_linear_ver_2',
           'equi_rms_norm_ver_2', 'scaler_gated_gelu_ver_2'],
    'v3': ['geometric_product_v3', 'equi_join_v3', 'equi_linear_ver_3',
           'equi_rms_norm_ver_3', 'scaler_gated_gelu_ver_3'],
}
ATTN_REFS = {
    'v0': 'equi_geometric_attention_ver_0',
    'v1': 'equi_geometric_attention_ver_1',
    'v2': 'equi_geometric_attention_ver_2',
    'v3': 'equi_geometric_attention_ver_3_1',
}
# PAPI FLOPs/bytes reference — v3 timing uses ver_3_1, but FLOPs/bytes come from ver_3 PAPI data
PAPI_ATTN_REFS = dict(ATTN_REFS)
PAPI_ATTN_REFS['v3'] = 'equi_geometric_attention_ver_3'
ver_style = {
    'v0': ('#B0BEC5', 'o', 'v0 - Baseline C++'),
    'v1': ('#FF9800', 's', 'v1 - Math optimizations'),
    'v2': ('#2196F3', '^', 'v2 - Scalar memory'),
    'v3': ('#2E7D32', 'D', 'v3 - SIMD + AVX-512 attn'),
}


def sweep_proj(ver: str, n: int) -> tuple[float, float] | None:
    total_fl = total_by = total_ms = 0.0
    for tgt in NON_ATTN_REFS[ver]:
        ref = papi3.get(tgt)
        if not ref:
            return None
        t = times.get(n, {}).get(tgt)
        if t is None:
            return None
        total_fl += ref['flops'] * (n / N_REF) ** NA_FL_EXP
        total_by += ref['bytes'] * (n / N_REF) ** NA_BY_EXP
        total_ms += t
    if n <= VER_MAX_N_ATTN[ver]:
        ref = papi3.get(PAPI_ATTN_REFS[ver])
        if ref:
            t = times.get(n, {}).get(ATTN_REFS[ver])
            if t is not None:
                total_fl += ref['flops'] * (n / N_REF) ** ATTN_FL_EXP
                total_by += ref['bytes'] * (n / N_REF) ** ATTN_BY_EXP
                total_ms += t
    if not total_by or not total_ms:
        return None
    ai = total_fl / total_by
    gf = min(total_fl / (total_ms / 1000) / 1e9, PEAK_GFLOPS * 0.995)
    return ai, gf


_PY_OP_MAP_1T = {
    'geometric_product_v0': 'geometric_product',
    'equi_join_v0':         'equi_join',
    'equi_linear_ver_0':    'equi_linear',
    'equi_rms_norm_ver_0':  'equi_rms_norm',
    'scaler_gated_gelu_ver_0': 'scaler_gated_gelu',
}


def sweep_proj_py(n: int):
    """Python project roofline point: papi3 FLOPs (new dims) + Python timing (new dims)."""
    if not papi3 or not py_times:
        return None
    total_fl = total_by = total_ms = 0.0
    for tgt, py_op in _PY_OP_MAP_1T.items():
        ref = papi3.get(tgt)
        py_ms = py_times.get(py_op, {}).get(n)
        if not ref or py_ms is None:
            return None
        total_fl += ref['flops'] * (n / N_REF) ** NA_FL_EXP
        total_by += ref['bytes'] * (n / N_REF) ** NA_BY_EXP
        total_ms += py_ms
    ref_a = papi3.get(ATTN_REFS['v0'])
    py_ms_a = attn_py_ms.get(n)
    if ref_a and py_ms_a is not None:
        total_fl += ref_a['flops'] * (n / N_REF) ** ATTN_FL_EXP
        total_by += ref_a['bytes'] * (n / N_REF) ** ATTN_BY_EXP
        total_ms += py_ms_a
    if not total_by or not total_ms:
        return None
    return total_fl / total_by, total_fl / (total_ms / 1000) / 1e9


points: dict[str, list] = {}
for ver in ['v0', 'v1', 'v2', 'v3']:
    for n in [nn for nn in NS if nn <= VER_MAX_N_ATTN[ver]]:
        pt = sweep_proj(ver, n)
        if pt:
            points.setdefault(ver, []).append((pt[0], pt[1], n))

fig, ax = plt.subplots(figsize=(10, 6))
draw_roofline(ax)
for ver, pts in points.items():
    color, marker, label = ver_style[ver]
    ais = [p[0] for p in pts]; gfs = [p[1] for p in pts]
    ax.plot(ais, gfs, '-', color=color, lw=1.3, alpha=0.5, zorder=6)
    ax.scatter(ais, gfs, color=color, marker=marker, s=110, zorder=10, edgecolors='white', lw=0.8, label=label)
    for ai, gf, n in pts:
        if n == max(p[2] for p in pts):
            xoff = 6; yoff = 4
            ax.annotate(f'n={n}', (ai, gf), textcoords='offset points', xytext=(xoff, yoff),
                        fontsize=7.5, color=color)
# Cross-version connecting lines at each n (v0→v1→v2→v3 "rungs")
all_ns_proj = sorted({n for pts in points.values() for _, _, n in pts})
for n in all_ns_proj:
    rung = [(ai, gf) for ver in ['v0', 'v1', 'v2', 'v3']
            for ai, gf, nn in points.get(ver, []) if nn == n]
    if len(rung) > 1:
        ax.plot([p[0] for p in rung], [p[1] for p in rung],
                '-', color='#888', lw=1.0, alpha=0.40, zorder=5)
# Python (1-thread) project reference at n=1..4
py_pts_1t = []
for _n in [1, 2, 3, 4]:
    _pt = sweep_proj_py(_n)
    if _pt:
        py_pts_1t.append((_pt[0], _pt[1], _n))
if py_pts_1t:
    _py_ais = [p[0] for p in py_pts_1t]; _py_gfs = [p[1] for p in py_pts_1t]
    ax.plot(_py_ais, _py_gfs, '-', color='#9E9E9E', lw=1.2, alpha=0.5, zorder=6)
    ax.scatter(_py_ais, _py_gfs, color='#757575', marker='*', s=160, zorder=11,
               edgecolors='#444', linewidths=0.8, label='Python (1T)')
    for _ai, _gf, _n in py_pts_1t:
        ax.annotate(f'n={_n}', (_ai, _gf), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='#555')
ax.set_xscale('log'); ax.set_yscale('log')
_p4_ais = [p[0] for pts in points.values() for p in pts] + [p[0] for p in py_pts_1t]
_p4_gfs = [p[1] for pts in points.values() for p in pts] + [p[1] for p in py_pts_1t]
ax.set_xlim(0.01, 5000); ax.set_ylim(0.05, PEAK_GFLOPS * 1.6)
ax.set_xlabel('Arithmetic Intensity [estimated FLOP/byte]', fontsize=12)
ax.set_ylabel('Performance [estimated GFLOPS]', fontsize=12)
ax.set_title('Whole-project roofline — v0:n=1..5, v1:n=1..5, v2:n=1..5, v3:n=1..9\n'
             'Tiger Lake i7-1165G7 | 6 kernels | Arrows: v0→v3 | PAPI-anchored AI', fontsize=10)
ax.grid(True, which='both', alpha=0.2, ls='--')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / 'sweep_1thread_roofline_project.png', dpi=150, bbox_inches='tight')
print('Saved sweep_1thread_roofline_project.png')
plt.close()
print('done')
