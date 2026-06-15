"""Attention optimization diagrams — clean visualizations for report figures.

No outer titles or captions; only diagram content. Run:
    python benchmarks/plot_attention_diagrams.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("benchmarks/results/attention_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
BG         = "#faf6f1"
OUTER      = "#ede5dc"
MID        = "#e0d3c5"
INNER      = "#d3c5b5"
RED_BOX    = "#c2504a"
GREEN_BOX  = "#4a9c6e"
BLUE_BOX   = "#4a72b0"
ORANGE_BOX = "#c47a35"
TEAL_BOX   = "#d6edd9"

TEXT       = "#1a1008"
TEXT_DIM   = "#5c4830"
FONT       = "DejaVu Sans"
FS_HEAD    = 11      # panel header
FS_LABEL   = 9.5    # section labels inside boxes
FS_BODY    = 8.5    # body text inside boxes
FS_SMALL   = 7.5    # minor labels

# ── helpers ──────────────────────────────────────────────────────────────────
def setup(fig, axes=None):
    fig.patch.set_facecolor(BG)
    for ax in (axes if axes else []):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

def box(ax, x, y, w, h, fill, text="", fs=FS_BODY, tc=TEXT, bold=False,
        r=0.015, lw=0.8, ec="#a08060", ty=None, zorder=2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=ec, linewidth=lw, zorder=zorder))
    if text:
        ax.text(x + w/2, ty if ty is not None else y + h/2, text,
                ha="center", va="center", fontsize=fs, color=tc,
                fontfamily=FONT, fontweight="bold" if bold else "normal",
                zorder=zorder+1, multialignment="center")

def txt(ax, x, y, text, fs=FS_BODY, ha="center", color=TEXT, bold=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            fontfamily=FONT, fontweight="bold" if bold else "normal",
            multialignment="center")

def arrow(ax, x0, y0, x1, y1, color=TEXT_DIM, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════════════════
# 1.  v0 vs v1 — math optimisations
# ═══════════════════════════════════════════════════════════════════════════
def draw_v0_v1():
    fig, (L, R) = plt.subplots(1, 2, figsize=(11, 5.8))
    setup(fig, [L, R])

    for ax, label in ((L, "v0 — Python-equivalent C++"), (R, "v1 — Explicit formula + cached constants")):
        txt(ax, 0.5, 0.96, label, fs=FS_HEAD, bold=True)

    # ── LEFT v0 ──────────────────────────────────────
    box(L, 0.04, 0.05, 0.92, 0.84, OUTER)
    txt(L, 0.5, 0.855, "for each token  (batch × T)", fs=FS_SMALL, color=TEXT_DIM)

    # IPA
    box(L, 0.10, 0.52, 0.80, 0.30, MID)
    txt(L, 0.5, 0.795, "IPA path", fs=FS_LABEL, bold=True)
    box(L, 0.14, 0.545, 0.72, 0.17, RED_BOX,
        "index_select(q, −1, sel)   ← 7-of-16 gather\n"
        "index_select(k, −1, sel)",
        fs=FS_BODY, tc="white")

    # DAA
    box(L, 0.10, 0.15, 0.80, 0.34, MID)
    txt(L, 0.5, 0.455, "DAA path", fs=FS_LABEL, bold=True)
    box(L, 0.14, 0.33, 0.72, 0.09, INNER,
        "normalize tri-vector blades {11, 12, 13, 14}", fs=FS_BODY)
    box(L, 0.14, 0.165, 0.72, 0.14, RED_BOX,
        "einsum(\"ijk,...i,...j→...k\", basis, norm_q, norm_q)\n"
        "einsum(\"ijk,...i,...j→...k\", basis, norm_k, norm_k)\n"
        "← Python dispatch + temporary tensors each call",
        fs=FS_BODY, tc="white")

    box(L, 0.10, 0.06, 0.80, 0.075, INNER,
        "cat([q_ipa, q_daa], dim=−1)  →  scaled_dot_product_attention",
        fs=FS_BODY)

    # ── RIGHT v1 ─────────────────────────────────────
    # cached-once callout
    box(R, 0.46, 0.865, 0.50, 0.058, TEAL_BOX,
        "cached once: basis tensors, IPA selector", fs=FS_SMALL, ec="#4a9c6e", lw=1.2)
    arrow(R, 0.71, 0.865, 0.71, 0.83, color="#4a9c6e")

    box(R, 0.04, 0.05, 0.92, 0.84, OUTER)
    txt(R, 0.5, 0.855, "for each token  (batch × T)", fs=FS_SMALL, color=TEXT_DIM)

    # IPA
    box(R, 0.10, 0.52, 0.80, 0.30, MID)
    txt(R, 0.5, 0.795, "IPA path", fs=FS_LABEL, bold=True)
    box(R, 0.14, 0.545, 0.72, 0.17, GREEN_BOX,
        "3 contiguous slices  →  cat\n"
        "blades {0},  {2,3,4},  {8,9,10}  — no gather",
        fs=FS_BODY, tc="white")

    # DAA
    box(R, 0.10, 0.15, 0.80, 0.34, MID)
    txt(R, 0.5, 0.455, "DAA path", fs=FS_LABEL, bold=True)
    box(R, 0.14, 0.33, 0.72, 0.09, INNER,
        "normalize tri-vector blades {11, 12, 13, 14}", fs=FS_BODY)
    box(R, 0.14, 0.165, 0.72, 0.14, GREEN_BOX,
        "explicit scalar formula — no einsum, no basis tensor\n"
        "q[0]=qn0²+qn1²+qn2²   q[1]=qn3²\n"
        "q[2]=qn0·qn3   q[3]=qn1·qn3   q[4]=qn2·qn3",
        fs=FS_BODY, tc="white")

    box(R, 0.10, 0.06, 0.80, 0.075, INNER,
        "cat([q_ipa, q_daa], dim=−1)  →  scaled_dot_product_attention",
        fs=FS_BODY)

    plt.tight_layout(pad=0.4)
    p = OUT / "v0_vs_v1.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  v1 vs v2 — two-pass → single interleaved pass + AoS→SoA
# ═══════════════════════════════════════════════════════════════════════════
def draw_v1_v2():
    fig, (L, R) = plt.subplots(1, 2, figsize=(11, 6.2))
    setup(fig, [L, R])

    for ax, label in ((L, "v1 — Two separate passes"), (R, "v2 — Single interleaved pass  (scalar)")):
        txt(ax, 0.5, 0.965, label, fs=FS_HEAD, bold=True)

    # ── LEFT v1 ──────────────────────────────────────
    box(L, 0.04, 0.05, 0.92, 0.88, OUTER)
    txt(L, 0.5, 0.898, "for block in  (batch × T)", fs=FS_SMALL, color=TEXT_DIM)

    # Pass 1
    box(L, 0.09, 0.65, 0.82, 0.22, MID)
    txt(L, 0.5, 0.845, "Pass 1 — IPA", fs=FS_LABEL, bold=True)
    box(L, 0.13, 0.665, 0.74, 0.14, RED_BOX,
        "read input  →  IPA slice  →  write Q_ipa, K_ipa\n"
        "output AoS: [block, C, 7]  ←  strided stores",
        fs=FS_BODY, tc="white")

    # Pass 2
    box(L, 0.09, 0.40, 0.82, 0.22, MID)
    txt(L, 0.5, 0.595, "Pass 2 — DAA", fs=FS_LABEL, bold=True)
    box(L, 0.13, 0.415, 0.74, 0.14, RED_BOX,
        "read input AGAIN  →  normalize  →  write Q_daa, K_daa\n"
        "output AoS: [block, C, 5]  ←  strided stores",
        fs=FS_BODY, tc="white")

    box(L, 0.09, 0.265, 0.82, 0.10, INNER,
        "torch::cat([Q_ipa, Q_daa], dim=−1)  ←  extra memory copy  (12×C values)",
        fs=FS_BODY)

    box(L, 0.09, 0.06, 0.82, 0.185, INNER)
    txt(L, 0.5, 0.155, "at::scaled_dot_product_attention", fs=FS_LABEL, bold=True)
    txt(L, 0.5, 0.095, "(Q_flat, K_flat, V_flat)", fs=FS_BODY, color=TEXT_DIM)

    # ── RIGHT v2 ─────────────────────────────────────
    box(R, 0.04, 0.855, 0.92, 0.075, TEAL_BOX,
        "allocate Q_out, K_out  [batch×T, 12, C]  (SoA) — once per call",
        fs=FS_BODY, ec="#4a9c6e", lw=1.2)

    box(R, 0.04, 0.05, 0.92, 0.795, OUTER)
    txt(R, 0.5, 0.815, "for block in  (batch × T)", fs=FS_SMALL, color=TEXT_DIM)

    box(R, 0.09, 0.47, 0.82, 0.32, MID)
    txt(R, 0.5, 0.755, "for c in 0 .. channels  (single pass)", fs=FS_LABEL, bold=True)

    box(R, 0.13, 0.635, 0.74, 0.095, GREEN_BOX,
        "load q[c, 0..15],  k[c, 0..15]   (AoS input, stride-16)",
        fs=FS_BODY, tc="white")
    box(R, 0.13, 0.535, 0.74, 0.09, GREEN_BOX,
        "IPA slots 0–6:  store Q_out[slot, c],  K_out[slot, c]   (SoA, stride-1 ✓)",
        fs=FS_BODY, tc="white")
    box(R, 0.13, 0.480, 0.74, 0.045, GREEN_BOX,
        "DAA slots 7–11:  normalize  →  store Q_out[slot, c],  K_out[slot, c]",
        fs=FS_BODY, tc="white")

    box(R, 0.09, 0.335, 0.82, 0.115, TEAL_BOX,
        "no cat — IPA (7) + DAA (5) slots already contiguous\n"
        "view Q_out as [batch×T, 12·C] at zero cost",
        fs=FS_BODY, ec="#4a9c6e", lw=1.2)

    box(R, 0.09, 0.06, 0.82, 0.255, INNER)
    txt(R, 0.5, 0.185, "at::scaled_dot_product_attention", fs=FS_LABEL, bold=True)
    txt(R, 0.5, 0.115, "(Q_out, K_out, V_flat)", fs=FS_BODY, color=TEXT_DIM)

    plt.tight_layout(pad=0.4)
    p = OUT / "v1_vs_v2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  v3 — AVX2 SIMD kernel
# ═══════════════════════════════════════════════════════════════════════════
def draw_v3_simd():
    fig, ax = plt.subplots(figsize=(12, 9.5))
    setup(fig, [ax])
    txt(ax, 0.5, 0.975, "v3 — AVX2 SIMD  (8 channels per iteration)", fs=FS_HEAD, bold=True)

    # ① Load  ─────────────────────────────────────────────────────────────
    box(ax, 0.03, 0.815, 0.94, 0.135, OUTER)
    txt(ax, 0.5, 0.934, "① Load — stride-16 gather across 8 channels", fs=FS_LABEL, bold=True)
    for i in range(8):
        x = 0.055 + i * 0.109
        box(ax, x, 0.825, 0.093, 0.095, "#dce8f5",
            f"ch {i}\nq[c+{i}, bl]\nk[c+{i}, bl]",
            fs=FS_SMALL, ec="#6090c0")
    txt(ax, 0.5, 0.818,
        "_mm256_set_ps(q[7·16+bl], …, q[0·16+bl])  —  8 scalar loads, pipelined by OoO engine",
        fs=FS_SMALL, color=TEXT_DIM)

    # ② IPA  and  ③ DAA normalize  (side by side)  ───────────────────────
    box(ax, 0.03, 0.610, 0.455, 0.19, MID)
    txt(ax, 0.255, 0.785, "② IPA  slots {0, 2, 3, 4, 8, 9, 10}", fs=FS_LABEL, bold=True)
    box(ax, 0.055, 0.623, 0.400, 0.14, BLUE_BOX, "", tc="white")
    txt(ax, 0.255, 0.718, "_mm256_mul_ps(lq(bl), ipa_w_v)  →  _mm256_storeu_ps(q_out + slot·C + c)", fs=FS_BODY, color="white")
    txt(ax, 0.255, 0.676, "7 slots × 2 (Q, K)  =  14 unit-stride stores", fs=FS_BODY, color="white")

    box(ax, 0.515, 0.610, 0.455, 0.19, MID)
    txt(ax, 0.742, 0.785, "③ DAA normalize  blades {11–14}", fs=FS_LABEL, bold=True)
    box(ax, 0.540, 0.623, 0.400, 0.14, ORANGE_BOX, "", tc="white")
    txt(ax, 0.740, 0.725, "denom = fmadd(q14, q14, eps)    ← FMA", fs=FS_BODY, color="white")
    txt(ax, 0.740, 0.693, "rcp   = _mm256_rcp_ps(denom)   ← ~4 cy  (div: 11–14 cy)", fs=FS_BODY, color="white")
    txt(ax, 0.740, 0.661, "rcp   = rcp × (2 − denom·rcp)  ← Newton-Raphson", fs=FS_BODY, color="white")

    # ④ DAA products  ──────────────────────────────────────────────────────
    box(ax, 0.03, 0.405, 0.94, 0.19, MID)
    txt(ax, 0.5, 0.578, "④ DAA products → 5 query slots + 5 key slots", fs=FS_LABEL, bold=True)
    labels = ["Q[0]\nqn0²+qn1²+qn2²", "Q[1]\nqn3²", "Q[2]\nqn0·qn3", "Q[3]\nqn1·qn3", "Q[4]\nqn2·qn3"]
    for i, lbl in enumerate(labels):
        x = 0.047 + i * 0.187
        box(ax, x, 0.420, 0.165, 0.11, GREEN_BOX, lbl, fs=FS_SMALL, tc="white")
    txt(ax, 0.5, 0.413,
        "fmadd / mul  →  _mm256_storeu_ps(q_out + slot·C + c, v)   (unit-stride, same for K)",
        fs=FS_SMALL, color=TEXT_DIM)

    # ⑤ Scalar tail  ───────────────────────────────────────────────────────
    box(ax, 0.03, 0.305, 0.94, 0.083, INNER,
        "⑤ Scalar tail — remaining channels (C mod 8) handled with identical scalar logic",
        fs=FS_BODY)

    # ⑥ SDPA  ──────────────────────────────────────────────────────────────
    box(ax, 0.03, 0.120, 0.94, 0.165, TEAL_BOX,
        "⑥ at::scaled_dot_product_attention(Q_soa, K_soa, V_flat)\n"
        "identical across v0–v3 — the entire speedup of v3 is in the Q/K assembly above",
        fs=FS_BODY, ec="#4a9c6e", lw=1.5)

    plt.tight_layout(pad=0.4)
    p = OUT / "v3_simd.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Pipeline speedup overview
# ═══════════════════════════════════════════════════════════════════════════
def draw_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    setup(fig, [ax])

    versions   = ["Python\n(ref)", "v0", "v1", "v2", "v3"]
    times_ms   = [192.7, 167.1, 140.9, 57.3, 58.2]
    colors     = ["#9c5c9c", "#b05050", "#b07030", "#3060b0", "#2a7a50"]
    v_desc     = [
        "Python einsum\n(reference)",
        "C++ port\nindex_select + einsum",
        "explicit formula\ncached constants",
        "single-pass\nAoS→SoA scalar",
        "AVX2 SIMD\nFMA + rcp NR",
    ]

    xs   = np.linspace(0.09, 0.91, len(versions))
    bw   = 0.11
    ref  = times_ms[0]
    maxt = max(times_ms)
    bar_base = 0.22

    for i, (ver, t, color, desc) in enumerate(zip(versions, times_ms, colors, v_desc)):
        x = xs[i]
        bh = 0.18 + 0.52 * (maxt / t) / (maxt / min(times_ms))

        # bar
        box(ax, x - bw/2, bar_base, bw, bh, color, ec="white", lw=1.0, r=0.008)

        # time label in bar
        ax.text(x, bar_base + bh/2, f"{t:.1f} ms",
                ha="center", va="center", fontsize=9.5, color="white",
                fontfamily=FONT, fontweight="bold")

        # speedup above bar
        if i > 0:
            sp = ref / t
            ax.text(x, bar_base + bh + 0.065, f"{sp:.1f}×",
                    ha="center", va="bottom", fontsize=11, color=color,
                    fontfamily=FONT, fontweight="bold")
            ax.text(x, bar_base + bh + 0.025, "vs ref",
                    ha="center", va="bottom", fontsize=7.5, color=color,
                    fontfamily=FONT)

        # version name below bar
        ax.text(x, bar_base - 0.04, ver,
                ha="center", va="top", fontsize=10, color=TEXT,
                fontfamily=FONT, fontweight="bold", multialignment="center")

        # description
        ax.text(x, bar_base - 0.13, desc,
                ha="center", va="top", fontsize=7.5, color=TEXT_DIM,
                fontfamily=FONT, multialignment="center")

    # arrows between bars
    for i in range(len(versions) - 1):
        x0, x1 = xs[i] + bw/2 + 0.005, xs[i+1] - bw/2 - 0.005
        ax.annotate("", xy=(x1, bar_base + 0.06), xytext=(x0, bar_base + 0.06),
                    arrowprops=dict(arrowstyle="->", color="#bbb0a0", lw=1.3))

    plt.tight_layout(pad=0.3)
    p = OUT / "pipeline_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {p}")


# ── run ──────────────────────────────────────────────────────────────────────
draw_v0_v1()
draw_v1_v2()
draw_v3_simd()
draw_pipeline_overview()
print("done →", OUT)
