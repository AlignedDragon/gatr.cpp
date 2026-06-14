"""Generate src/ezgatr/_csrc/basis_data.h and the unrolled per-blade kernels.

Run by hand whenever the .pt files change (effectively never — they're algebra
constants). The output headers are committed to git so the C++ build does not
depend on this script being available at compile time.

Generates:
  - src/ezgatr/_csrc/basis_data.h     (dense GP_BASIS / OP_BASIS / DUAL_*)
  - src/ezgatr/_csrc/kernels/gp_unrolled.inc  (16 templated inlines: gp_blade_NN<T>)
  - src/ezgatr/_csrc/kernels/join_unrolled.inc (16 templated inlines: join_blade_NN<T>)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BASIS_DIR = REPO / "src/ezgatr/nn/functional/basis"
OUT = REPO / "src/ezgatr/_csrc/basis_data.h"
OUT_GP_INC = REPO / "src/ezgatr/_csrc/kernels/gp_unrolled.inc"
OUT_JOIN_INC = REPO / "src/ezgatr/_csrc/kernels/join_unrolled.inc"
OUT_GP_ILP2_INC = REPO / "src/ezgatr/_csrc/kernels/gp_block_ilp2.inc"
OUT_GP_ILP4_INC = REPO / "src/ezgatr/_csrc/kernels/gp_block_ilp4.inc"
OUT_JOIN_ILP2_INC = REPO / "src/ezgatr/_csrc/kernels/join_block_ilp2.inc"
OUT_JOIN_ILP4_INC = REPO / "src/ezgatr/_csrc/kernels/join_block_ilp4.inc"
OUT_GP_ACC2_INC = REPO / "src/ezgatr/_csrc/kernels/gp_unrolled_acc2.inc"
OUT_GP_ACC4_INC = REPO / "src/ezgatr/_csrc/kernels/gp_unrolled_acc4.inc"
OUT_JOIN_ACC2_INC = REPO / "src/ezgatr/_csrc/kernels/join_unrolled_acc2.inc"
OUT_JOIN_ACC4_INC = REPO / "src/ezgatr/_csrc/kernels/join_unrolled_acc4.inc"
OUT_GP_UNROLL2_INC = REPO / "src/ezgatr/_csrc/kernels/gp_loop_unroll2.inc"
OUT_GP_UNROLL4_INC = REPO / "src/ezgatr/_csrc/kernels/gp_loop_unroll4.inc"
OUT_JOIN_UNROLL2_INC = REPO / "src/ezgatr/_csrc/kernels/join_loop_unroll2.inc"
OUT_JOIN_UNROLL4_INC = REPO / "src/ezgatr/_csrc/kernels/join_loop_unroll4.inc"
OUT_GP_SOA_AVX_INC = REPO / "src/ezgatr/_csrc/kernels/gp_soa_avx.inc"
OUT_JOIN_SOA_AVX_INC = REPO / "src/ezgatr/_csrc/kernels/join_soa_avx.inc"
OUT_GP_SOA_AVX512_INC = REPO / "src/ezgatr/_csrc/kernels/gp_soa_avx512.inc"
OUT_JOIN_SOA_AVX512_INC = REPO / "src/ezgatr/_csrc/kernels/join_soa_avx512.inc"

DUAL_PERM = list(range(15, -1, -1))
DUAL_SIGN = [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1]
INNER_SELECTOR = [0, 2, 3, 4, 8, 9, 10, 14]


def load_dense_int8(path: Path) -> torch.Tensor:
    t = torch.load(path, weights_only=True)
    if t.is_sparse:
        t = t.to_dense()
    if not torch.all((t == -1) | (t == 0) | (t == 1)):
        raise SystemExit(f"{path}: values outside {{-1,0,1}} — refusing to pack as int8")
    if tuple(t.shape) != (16, 16, 16):
        raise SystemExit(f"{path}: expected shape (16,16,16), got {tuple(t.shape)}")
    return t.to(torch.int8)


def fmt_3d(name: str, t: torch.Tensor) -> str:
    lines = [f"inline constexpr int8_t {name}[16][16][16] = {{"]
    for i in range(16):
        lines.append(f"    {{  // i={i}")
        for j in range(16):
            row = ", ".join(f"{int(v):2d}" for v in t[i, j].tolist())
            lines.append(f"        {{{row}}},")
        lines.append("    },")
    lines.append("};")
    return "\n".join(lines)


def fmt_1d(name: str, ctype: str, vals: list[int]) -> str:
    body = ", ".join(str(v) for v in vals)
    return f"inline constexpr {ctype} {name}[16] = {{{body}}};"


def compute_join_kernel(op: torch.Tensor) -> torch.Tensor:
    """Mirror ezgatr.nn.functional.dual._compute_efficient_join_kernel in int64.

    Build the (16,16,16) join kernel via dual ∘ outer ∘ dual. Verifies the
    result is a pure {-1, 0, 1} sign-flip table.
    """
    perm = torch.tensor(DUAL_PERM, dtype=torch.long)
    sign = torch.tensor(DUAL_SIGN, dtype=torch.int64)
    op64 = op.to(torch.int64)

    def dual(v: torch.Tensor) -> torch.Tensor:
        return sign * v[perm]

    kernel = torch.zeros((16, 16, 16), dtype=torch.int64)
    for i in range(16):
        for j in range(16):
            x = torch.zeros(16, dtype=torch.int64)
            y = torch.zeros(16, dtype=torch.int64)
            x[i] = 1
            y[j] = 1
            outer = torch.einsum("abc,b,c->a", op64, dual(x), dual(y))
            kernel[:, i, j] = dual(outer)

    if not torch.all((kernel == -1) | (kernel == 0) | (kernel == 1)):
        raise SystemExit("join kernel: values outside {-1,0,1} — refusing to emit")
    return kernel.to(torch.int8)


def fmt_unrolled(name_prefix: str, kernel: torch.Tensor) -> tuple[str, int]:
    """Emit 16 templated inline functions, one per output blade.

    Each function returns the signed sum-of-products that matches
        einsum("ijk, j, k -> i", kernel, x, y)
    for the given output index i. Returns (source, total_nonzero_count).
    """
    lines: list[str] = []
    total_nnz = 0
    for i in range(16):
        triples: list[tuple[int, int, int]] = []
        for j in range(16):
            for k in range(16):
                v = int(kernel[i, j, k].item())
                if v != 0:
                    triples.append((j, k, v))
        triples.sort()
        total_nnz += len(triples)

        lines.append("template <typename T>")
        lines.append(
            f"static inline T {name_prefix}_blade_{i:02d}"
            "(const T* __restrict__ x, const T* __restrict__ y) {"
        )
        if not triples:
            lines.append("    (void)x; (void)y;")
            lines.append("    return T(0);")
        else:
            terms: list[str] = []
            for idx, (j, k, s) in enumerate(triples):
                sign = "+" if s > 0 else "-"
                if idx == 0:
                    terms.append(f"x[{j}]*y[{k}]" if s > 0 else f"-x[{j}]*y[{k}]")
                else:
                    terms.append(f" {sign} x[{j}]*y[{k}]")
            lines.append(f"    return {''.join(terms)};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines), total_nnz


def fmt_unrolled_acc(name_prefix: str, kernel: torch.Tensor, K: int) -> tuple[str, int]:
    """Emit 16 per-blade inlines, each splitting its sum into K accumulators.

    Each function returns the same signed sum-of-products as
        einsum("ijk, j, k -> i", kernel, x, y)
    but split into K parallel partial sums so the compiler can keep K
    independent FP-add chains in flight (breaking the serial FMA dependency
    within a single blade). Across-blade ILP is left to the compiler when
    these are inlined into the per-batch loop.

    Layout per blade i:
      - sort sparse triples (j, k, sign)
      - bucket term t into slot (t % K); each bucket emits as a single
        flat expression `±x[j]*y[k] ± x[j']*y[k'] ± ...`
      - return is `a0 + a1 + ... + a_{K-1}`
    """
    lines: list[str] = []
    total_nnz = 0
    for i in range(16):
        triples: list[tuple[int, int, int]] = []
        for j in range(16):
            for k in range(16):
                v = int(kernel[i, j, k].item())
                if v != 0:
                    triples.append((j, k, v))
        triples.sort()
        total_nnz += len(triples)

        lines.append("template <typename T>")
        lines.append(
            f"static inline T {name_prefix}_blade_{i:02d}_acc{K}"
            "(const T* __restrict__ x, const T* __restrict__ y) {"
        )
        if not triples:
            lines.append("    (void)x; (void)y;")
            lines.append("    return T(0);")
        else:
            buckets: list[list[tuple[int, int, int]]] = [[] for _ in range(K)]
            for t_idx, t in enumerate(triples):
                buckets[t_idx % K].append(t)
            buckets = [b for b in buckets if b]

            slot_names: list[str] = []
            for s_idx, bucket in enumerate(buckets):
                name = f"a{s_idx}"
                slot_names.append(name)
                terms: list[str] = []
                for idx, (j, k, sgn) in enumerate(bucket):
                    if idx == 0:
                        terms.append(f"x[{j}]*y[{k}]" if sgn > 0 else f"-x[{j}]*y[{k}]")
                    else:
                        op = "+" if sgn > 0 else "-"
                        terms.append(f" {op} x[{j}]*y[{k}]")
                lines.append(f"    T {name} = {''.join(terms)};")
            lines.append(f"    return {' + '.join(slot_names)};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines), total_nnz


def fmt_block_ilp(name_prefix: str, kernel: torch.Tensor, K: int) -> tuple[str, int]:
    """Emit one big inline block kernel with K accumulators per output blade.

    Layout per blade i:
      - Collect sparse triples (j, k, sign) and assign term t to sub-acc t % K.
      - Declare T a_i_0 = 0, ..., a_i_{K-1} = 0 (skipped if blade has 0 terms).
      - Emit `a_i_s += x[j]*y[k];` (or `-=` for negative sign).
      - At the end, emit `o[i] = a_i_0 + a_i_1 + ... + a_i_{K-1};`.

    Term emission order: round-robin across blade index, so the compiler sees
    independent FMAs from many blades interleaved (cross-blade ILP).

    Degenerate cases:
      - Zero terms: emit `o[i] = T(0);` directly.
      - n < K terms: collapse — only allocate sub-accs that actually receive
        a term.
    """
    blades: list[list[tuple[int, int, int]]] = []
    for i in range(16):
        triples: list[tuple[int, int, int]] = []
        for j in range(16):
            for k in range(16):
                v = int(kernel[i, j, k].item())
                if v != 0:
                    triples.append((j, k, v))
        triples.sort()
        blades.append(triples)
    total_nnz = sum(len(b) for b in blades)

    sub_acc_count: list[int] = []
    for triples in blades:
        slots_used = set()
        for t_idx in range(len(triples)):
            slots_used.add(t_idx % K)
        sub_acc_count.append(len(slots_used))

    lines: list[str] = []
    lines.append("template <typename T>")
    lines.append(
        f"static inline void {name_prefix}_block_ilp{K}("
        "const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ o) {"
    )

    decl_chunks: list[str] = []
    for i, triples in enumerate(blades):
        n = sub_acc_count[i]
        if n == 0:
            continue
        names = ", ".join(f"a{i:02d}_{s} = T(0)" for s in range(n))
        decl_chunks.append(f"    T {names};")
    if decl_chunks:
        lines.extend(decl_chunks)

    max_terms = max((len(b) for b in blades), default=0)
    for t_idx in range(max_terms):
        for i, triples in enumerate(blades):
            if t_idx >= len(triples):
                continue
            j, k, s = triples[t_idx]
            slot = t_idx % K
            op = "+=" if s > 0 else "-=";
            lines.append(f"    a{i:02d}_{slot} {op} x[{j}]*y[{k}];")

    for i, triples in enumerate(blades):
        n = sub_acc_count[i]
        if n == 0:
            lines.append(f"    o[{i}] = T(0);")
        else:
            expr = " + ".join(f"a{i:02d}_{s}" for s in range(n))
            lines.append(f"    o[{i}] = {expr};")

    lines.append("    (void)x; (void)y;")
    lines.append("}")
    return "\n".join(lines), total_nnz


def _blade_triples(kernel: torch.Tensor) -> tuple[list[list[tuple[int, int, int]]], int]:
    """Return per-blade sorted sparse triples (j, k, sign) and total nnz."""
    blades: list[list[tuple[int, int, int]]] = []
    total_nnz = 0
    for i in range(16):
        triples: list[tuple[int, int, int]] = []
        for j in range(16):
            for k in range(16):
                v = int(kernel[i, j, k].item())
                if v != 0:
                    triples.append((j, k, v))
        triples.sort()
        blades.append(triples)
        total_nnz += len(triples)
    return blades, total_nnz


def fmt_loop_unroll(name_prefix: str, kernel: torch.Tensor, K: int,
                    is_join: bool) -> tuple[str, int]:
    """Emit a kernel that unrolls the multivector loop by K lanes.

    Each iteration processes K consecutive multivectors. Per (output blade,
    lane) the sum-of-products is split into 2 accumulators (round-robin over
    terms) so the serial FP-add chain is broken — i.e. "unroll the loop, then
    accumulate". This exposes up to 2*K independent FMA chains to the
    out-of-order engine while keeping only ~2*K live accumulators (one output
    blade is finished before the next begins), so register pressure stays low.

    The `N % K` tail reuses the scalar `<prefix>_blade_NN<T>` inlines.
    """
    blades, total_nnz = _blade_triples(kernel)

    L: list[str] = []
    L.append("template <typename T" + (", bool HasRef" if is_join else "") + ">")
    sig = (f"static inline void {name_prefix}_loop_unroll{K}("
           "const T* __restrict__ X, const T* __restrict__ Y, ")
    if is_join:
        sig += "const T* __restrict__ R, "
    sig += "T* __restrict__ O, int64_t N) {"
    L.append(sig)
    L.append("    int64_t n = 0;")
    L.append(f"    for (; n + {K} <= N; n += {K}) {{")
    for lane in range(K):
        L.append(f"        const T* x{lane} = X + 16 * (n + {lane});")
        L.append(f"        const T* y{lane} = Y + 16 * (n + {lane});")
        L.append(f"        T* o{lane} = O + 16 * (n + {lane});")

    for i, triples in enumerate(blades):
        L.append(f"        {{  // blade {i:02d}")
        if not triples:
            for lane in range(K):
                L.append(f"            o{lane}[{i}] = T(0);")
            L.append("        }")
            continue
        nslots = min(2, len(triples))
        for lane in range(K):
            slot_terms: list[list[str]] = [[] for _ in range(nslots)]
            for t_idx, (j, k, s) in enumerate(triples):
                slot = t_idx % nslots
                term = f"x{lane}[{j}]*y{lane}[{k}]"
                if not slot_terms[slot]:
                    slot_terms[slot].append(term if s > 0 else f"-{term}")
                else:
                    slot_terms[slot].append((" + " if s > 0 else " - ") + term)
            names = []
            for sl in range(nslots):
                nm = f"a{lane}_{sl}"
                names.append(nm)
                L.append(f"            T {nm} = {''.join(slot_terms[sl])};")
            L.append(f"            o{lane}[{i}] = {' + '.join(names)};")
        L.append("        }")

    if is_join:
        L.append("        if constexpr (HasRef) {")
        for lane in range(K):
            L.append(f"            const T s{lane} = R[16 * (n + {lane}) + 14];")
        for lane in range(K):
            L.append(f"            for (int i = 0; i < 16; ++i) o{lane}[i] *= s{lane};")
        L.append("        }")
    L.append("    }")

    L.append("    for (; n < N; ++n) {")
    L.append("        const T* x = X + 16 * n;")
    L.append("        const T* y = Y + 16 * n;")
    L.append("        T* o = O + 16 * n;")
    for i in range(16):
        L.append(f"        o[{i}] = {name_prefix}_blade_{i:02d}<T>(x, y);")
    if is_join:
        L.append("        if constexpr (HasRef) {")
        L.append("            const T s = R[16 * n + 14];")
        L.append("            for (int i = 0; i < 16; ++i) o[i] *= s;")
        L.append("        }")
    L.append("    }")
    L.append("}")
    return "\n".join(L), total_nnz


def fmt_soa_avx(name_prefix: str, kernel: torch.Tensor) -> tuple[str, int]:
    """Emit a float-specialized AVX2 SoA block kernel.

    Operates on transposed Struct-of-Arrays tiles: xb[j] / yb[k] each hold the
    8 lanes (8 consecutive multivectors) of component j / k. Each output blade
    is a sequence of packed FMAs over the 8 lanes (2 accumulators to break the
    add chain) — no gather, no reduction. The caller transposes AoS->SoA, calls
    this, then transposes back.
    """
    blades, total_nnz = _blade_triples(kernel)

    L: list[str] = []
    L.append(f"static inline void {name_prefix}_soa_block_f32(")
    L.append("        const float xb[16][8], const float yb[16][8], float ob[16][8]) {")
    for i, triples in enumerate(blades):
        if not triples:
            L.append(f"    _mm256_store_ps(ob[{i}], _mm256_setzero_ps());")
            continue
        nslots = min(2, len(triples))
        L.append(f"    {{  // blade {i:02d}")
        L.append("        __m256 a0 = _mm256_setzero_ps();")
        if nslots == 2:
            L.append("        __m256 a1 = _mm256_setzero_ps();")
        for t_idx, (j, k, s) in enumerate(triples):
            slot = t_idx % nslots
            fn = "_mm256_fmadd_ps" if s > 0 else "_mm256_fnmadd_ps"
            L.append(
                f"        a{slot} = {fn}("
                f"_mm256_load_ps(xb[{j}]), _mm256_load_ps(yb[{k}]), a{slot});"
            )
        if nslots == 2:
            L.append(f"        _mm256_store_ps(ob[{i}], _mm256_add_ps(a0, a1));")
        else:
            L.append(f"        _mm256_store_ps(ob[{i}], a0);")
        L.append("    }")
    L.append("}")
    return "\n".join(L), total_nnz


def fmt_soa_avx512(name_prefix: str, kernel: torch.Tensor) -> tuple[str, int]:
    """Emit a float-specialized AVX-512 SoA block kernel (16 lanes).

    Identical structure to fmt_soa_avx but over a 16-wide transposed tile:
    xb[j] / yb[k] each hold the 16 lanes (16 consecutive multivectors) of
    component j / k, so each output blade is a sequence of packed __m512 FMAs
    (2 accumulators to break the add chain) — no gather, no reduction. Doubles
    the lanes-per-FMA versus the AVX2 block; the caller transposes AoS<->SoA
    around it. fp32 + AVX-512F only.
    """
    blades, total_nnz = _blade_triples(kernel)

    L: list[str] = []
    L.append(f"static inline void {name_prefix}_soa_block_f32_avx512(")
    L.append("        const float xb[16][16], const float yb[16][16], float ob[16][16]) {")
    for i, triples in enumerate(blades):
        if not triples:
            L.append(f"    _mm512_store_ps(ob[{i}], _mm512_setzero_ps());")
            continue
        nslots = min(2, len(triples))
        L.append(f"    {{  // blade {i:02d}")
        L.append("        __m512 a0 = _mm512_setzero_ps();")
        if nslots == 2:
            L.append("        __m512 a1 = _mm512_setzero_ps();")
        for t_idx, (j, k, s) in enumerate(triples):
            slot = t_idx % nslots
            fn = "_mm512_fmadd_ps" if s > 0 else "_mm512_fnmadd_ps"
            L.append(
                f"        a{slot} = {fn}("
                f"_mm512_load_ps(xb[{j}]), _mm512_load_ps(yb[{k}]), a{slot});"
            )
        if nslots == 2:
            L.append(f"        _mm512_store_ps(ob[{i}], _mm512_add_ps(a0, a1));")
        else:
            L.append(f"        _mm512_store_ps(ob[{i}], a0);")
        L.append("    }")
    L.append("}")
    return "\n".join(L), total_nnz


def write_loop_unroll(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                      K: int, is_join: bool, sources_note: str) -> int:
    body, nnz = fmt_loop_unroll(name_prefix, kernel, K, is_join)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        f"// Multivector loop unrolled by K={K} lanes; 2 accumulators per "
        "(blade, lane).\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace,\n"
        f"// after {name_prefix}_unrolled.inc (uses the scalar blade inlines for the tail).\n\n"
        f"{body}\n"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, K={K}, nnz={nnz})")
    return nnz


def write_soa_avx(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                  sources_note: str) -> int:
    body, nnz = fmt_soa_avx(name_prefix, kernel)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        "// AVX2 (float) SoA block: packed 8-lane FMAs over a transposed tile,\n"
        "// 2 accumulators per blade. Requires <immintrin.h>, AVX2 + FMA.\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace.\n\n"
        "#if defined(__AVX2__) && defined(__FMA__)\n"
        f"{body}\n"
        "#endif  // __AVX2__ && __FMA__\n"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, nnz={nnz})")
    return nnz


def write_soa_avx512(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                     sources_note: str) -> int:
    body, nnz = fmt_soa_avx512(name_prefix, kernel)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        "// AVX-512 (float) SoA block: packed 16-lane FMAs over a transposed tile,\n"
        "// 2 accumulators per blade. Requires <immintrin.h>, AVX-512F.\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace.\n\n"
        "#if defined(__AVX512F__)\n"
        f"{body}\n"
        "#endif  // __AVX512F__\n"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, nnz={nnz})")
    return nnz


def write_unrolled(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                   sources_note: str) -> int:
    body, nnz = fmt_unrolled(name_prefix, kernel)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace.\n\n"
        f"{body}"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, nnz={nnz})")
    return nnz


def write_unrolled_acc(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                       K: int, sources_note: str) -> int:
    body, nnz = fmt_unrolled_acc(name_prefix, kernel, K)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        f"// Per-blade unrolled with K={K} parallel accumulators "
        "(breaks intra-blade FP-add dependency chain).\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace.\n\n"
        f"{body}"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, K={K}, nnz={nnz})")
    return nnz


def write_block_ilp(out_path: Path, name_prefix: str, kernel: torch.Tensor,
                    K: int, sources_note: str) -> int:
    body, nnz = fmt_block_ilp(name_prefix, kernel, K)
    text = (
        "// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.\n"
        f"// {sources_note}\n"
        f"// Single-block kernel with K={K} accumulators per output blade.\n"
        f"// Total non-zero terms across all 16 output blades: {nnz}.\n"
        "// Designed to be #include'd inside the ezgatr::opt namespace.\n\n"
        f"{body}\n"
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(REPO)} "
          f"({out_path.stat().st_size} bytes, K={K}, nnz={nnz})")
    return nnz


def main() -> None:
    gp_path = BASIS_DIR / "geometric_product.pt"
    op_path = BASIS_DIR / "outer_product.pt"

    gp = load_dense_int8(gp_path)
    op = load_dense_int8(op_path)

    gp_sha = hashlib.sha256(gp_path.read_bytes()).hexdigest()
    op_sha = hashlib.sha256(op_path.read_bytes()).hexdigest()

    header = f"""// AUTO-GENERATED by scripts/gen_basis_header.py — do not edit.
// Source files (sha256):
//   {gp_path.name}: {gp_sha}
//   {op_path.name}: {op_sha}
#pragma once
#include <cstdint>

namespace ezgatr {{ namespace opt {{

{fmt_3d("GP_BASIS", gp)}

{fmt_3d("OP_BASIS", op)}

{fmt_1d("DUAL_PERM", "int64_t", DUAL_PERM)}
{fmt_1d("DUAL_SIGN", "int8_t", DUAL_SIGN)}
{fmt_1d("INNER_SELECTOR", "int64_t", INNER_SELECTOR)}

}}}}  // namespace ezgatr::opt
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header)
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size} bytes)")

    gp_nnz = write_unrolled(
        OUT_GP_INC, "gp", gp,
        sources_note=f"Source: {gp_path.name} (sha256 {gp_sha[:16]}…)",
    )
    if gp_nnz != 192:
        raise SystemExit(
            f"GP_BASIS non-zero count is {gp_nnz}, expected 192. "
            "Algebra has changed — review before continuing."
        )

    gp_note = f"Source: {gp_path.name} (sha256 {gp_sha[:16]}…)"
    write_block_ilp(OUT_GP_ILP2_INC, "gp", gp, K=2, sources_note=gp_note)
    write_block_ilp(OUT_GP_ILP4_INC, "gp", gp, K=4, sources_note=gp_note)
    write_unrolled_acc(OUT_GP_ACC2_INC, "gp", gp, K=2, sources_note=gp_note)
    write_unrolled_acc(OUT_GP_ACC4_INC, "gp", gp, K=4, sources_note=gp_note)
    write_loop_unroll(OUT_GP_UNROLL2_INC, "gp", gp, K=2, is_join=False, sources_note=gp_note)
    write_loop_unroll(OUT_GP_UNROLL4_INC, "gp", gp, K=4, is_join=False, sources_note=gp_note)
    write_soa_avx(OUT_GP_SOA_AVX_INC, "gp", gp, sources_note=gp_note)
    write_soa_avx512(OUT_GP_SOA_AVX512_INC, "gp", gp, sources_note=gp_note)

    join = compute_join_kernel(op)
    write_unrolled(
        OUT_JOIN_INC, "join", join,
        sources_note=(
            f"Source: derived at codegen time as dual(outer(dual(x), dual(y))) "
            f"from {op_path.name} (sha256 {op_sha[:16]}…)."
        ),
    )
    join_note = (
        f"Source: derived at codegen time as dual(outer(dual(x), dual(y))) "
        f"from {op_path.name} (sha256 {op_sha[:16]}…)."
    )
    write_block_ilp(OUT_JOIN_ILP2_INC, "join", join, K=2, sources_note=join_note)
    write_block_ilp(OUT_JOIN_ILP4_INC, "join", join, K=4, sources_note=join_note)
    write_unrolled_acc(OUT_JOIN_ACC2_INC, "join", join, K=2, sources_note=join_note)
    write_unrolled_acc(OUT_JOIN_ACC4_INC, "join", join, K=4, sources_note=join_note)
    write_loop_unroll(OUT_JOIN_UNROLL2_INC, "join", join, K=2, is_join=True, sources_note=join_note)
    write_loop_unroll(OUT_JOIN_UNROLL4_INC, "join", join, K=4, is_join=True, sources_note=join_note)
    write_soa_avx(OUT_JOIN_SOA_AVX_INC, "join", join, sources_note=join_note)
    write_soa_avx512(OUT_JOIN_SOA_AVX512_INC, "join", join, sources_note=join_note)


if __name__ == "__main__":
    main()
