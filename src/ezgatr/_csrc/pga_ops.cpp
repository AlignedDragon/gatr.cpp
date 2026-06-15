#include "pga_ops.h"
#include "basis_data.h"

#include <ATen/Dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

// ver_3 uses explicit AVX2/FMA intrinsics on x86. The include and the whole
// vectorized path are compiled only when the target ISA provides them (e.g.
// -march=native on an Intel host); everything else falls back to the scalar
// ver_2 kernel, so this still compiles on ARM / pre-AVX2 builds.
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define EZGATR_HAVE_AVX2 1
#else
#define EZGATR_HAVE_AVX2 0
#endif

namespace ezgatr { namespace opt {

namespace {

#include "kernels/gp_unrolled.inc"
#include "kernels/join_unrolled.inc"
#include "kernels/gp_block_ilp2.inc"
#include "kernels/gp_block_ilp4.inc"
#include "kernels/join_block_ilp2.inc"
#include "kernels/join_block_ilp4.inc"
#include "kernels/gp_unrolled_acc2.inc"
#include "kernels/gp_unrolled_acc4.inc"
#include "kernels/join_unrolled_acc2.inc"
#include "kernels/join_unrolled_acc4.inc"
#include "kernels/gp_loop_unroll2.inc"
#include "kernels/gp_loop_unroll4.inc"
#include "kernels/join_loop_unroll2.inc"
#include "kernels/join_loop_unroll4.inc"
#include "kernels/gp_soa_avx.inc"
#include "kernels/join_soa_avx.inc"
#include "kernels/gp_soa_avx512.inc"
#include "kernels/join_soa_avx512.inc"

using BasisTable = int8_t[16][16][16];

struct SparseEntry {
    uint8_t i;
    uint8_t j;
    uint8_t k;
    int8_t  sign;
};

static const BasisTable& get_join_basis_int8() {
    static BasisTable table = {};
    static std::once_flag flag;
    std::call_once(flag, []{
        auto kernel = ezgatr::opt::compute_join_kernel(c10::Device(c10::kCPU), c10::kDouble);
        auto acc = kernel.accessor<double, 3>();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k)
                    table[i][j][k] = static_cast<int8_t>(std::lround(acc[i][j][k]));
    });
    return table;
}

static const std::vector<SparseEntry>& get_gp_sparse_entries() {
    static std::vector<SparseEntry> v;
    static std::once_flag flag;
    std::call_once(flag, []{
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k)
                    if (GP_BASIS[i][j][k] != 0)
                        v.push_back({(uint8_t)i, (uint8_t)j, (uint8_t)k, GP_BASIS[i][j][k]});
    });
    return v;
}

static const std::vector<SparseEntry>& get_join_sparse_entries() {
    static std::vector<SparseEntry> v;
    static std::once_flag flag;
    std::call_once(flag, []{
        const auto& B = get_join_basis_int8();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k)
                    if (B[i][j][k] != 0)
                        v.push_back({(uint8_t)i, (uint8_t)j, (uint8_t)k, B[i][j][k]});
    });
    return v;
}

template <typename T>
static void gp_kernel_v0(const T* __restrict__ X,
                                 const T* __restrict__ Y,
                                 T* __restrict__ O,
                                 int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        for (int i = 0; i < 16; ++i) {
            T acc = T(0);
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    acc += T(GP_BASIS[i][j][k]) * x[j] * y[k];
                }
            }
            o[i] = acc;
        }
    }
}

template <typename T>
static void gp_kernel_v1(const T* __restrict__ X,
                                     const T* __restrict__ Y,
                                     T* __restrict__ O,
                                     int64_t N) {
    const auto& entries = get_gp_sparse_entries();
    const SparseEntry* e = entries.data();
    const size_t M = entries.size();
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        for (int i = 0; i < 16; ++i) o[i] = T(0);
        for (size_t m = 0; m < M; ++m) {
            o[e[m].i] += T(e[m].sign) * x[e[m].j] * y[e[m].k];
        }
    }
}

template <typename T, bool HasRef>
static void join_kernel_v0(const T* __restrict__ X,
                                   const T* __restrict__ Y,
                                   const T* __restrict__ R,
                                   T* __restrict__ O,
                                   int64_t N) {
    const auto& B = get_join_basis_int8();
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        const T s = HasRef ? R[n] : T(1);
        for (int i = 0; i < 16; ++i) {
            T acc = T(0);
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    acc += T(B[i][j][k]) * x[j] * y[k];
                }
            }
            o[i] = HasRef ? s * acc : acc;
        }
    }
}

template <typename T, bool HasRef>
static void join_kernel_v1(const T* __restrict__ X,
                                       const T* __restrict__ Y,
                                       const T* __restrict__ R,
                                       T* __restrict__ O,
                                       int64_t N) {
    const auto& entries = get_join_sparse_entries();
    const SparseEntry* e = entries.data();
    const size_t M = entries.size();
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        for (int i = 0; i < 16; ++i) o[i] = T(0);
        for (size_t m = 0; m < M; ++m) {
            o[e[m].i] += T(e[m].sign) * x[e[m].j] * y[e[m].k];
        }
        if constexpr (HasRef) {
            const T s = R[n];
            for (int i = 0; i < 16; ++i) o[i] *= s;
        }
    }
}

template <typename T>
static void gp_kernel_v2(const T* __restrict__ X,
                           const T* __restrict__ Y,
                           T* __restrict__ O,
                           int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        o[0]  = gp_blade_00<T>(x, y);
        o[1]  = gp_blade_01<T>(x, y);
        o[2]  = gp_blade_02<T>(x, y);
        o[3]  = gp_blade_03<T>(x, y);
        o[4]  = gp_blade_04<T>(x, y);
        o[5]  = gp_blade_05<T>(x, y);
        o[6]  = gp_blade_06<T>(x, y);
        o[7]  = gp_blade_07<T>(x, y);
        o[8]  = gp_blade_08<T>(x, y);
        o[9]  = gp_blade_09<T>(x, y);
        o[10] = gp_blade_10<T>(x, y);
        o[11] = gp_blade_11<T>(x, y);
        o[12] = gp_blade_12<T>(x, y);
        o[13] = gp_blade_13<T>(x, y);
        o[14] = gp_blade_14<T>(x, y);
        o[15] = gp_blade_15<T>(x, y);
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2(const T* __restrict__ X,
                             const T* __restrict__ Y,
                             const T* __restrict__ R,
                             T* __restrict__ O,
                             int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        if constexpr (HasRef) {
            const T s = R[n];
            o[0]  = s * join_blade_00<T>(x, y);
            o[1]  = s * join_blade_01<T>(x, y);
            o[2]  = s * join_blade_02<T>(x, y);
            o[3]  = s * join_blade_03<T>(x, y);
            o[4]  = s * join_blade_04<T>(x, y);
            o[5]  = s * join_blade_05<T>(x, y);
            o[6]  = s * join_blade_06<T>(x, y);
            o[7]  = s * join_blade_07<T>(x, y);
            o[8]  = s * join_blade_08<T>(x, y);
            o[9]  = s * join_blade_09<T>(x, y);
            o[10] = s * join_blade_10<T>(x, y);
            o[11] = s * join_blade_11<T>(x, y);
            o[12] = s * join_blade_12<T>(x, y);
            o[13] = s * join_blade_13<T>(x, y);
            o[14] = s * join_blade_14<T>(x, y);
            o[15] = s * join_blade_15<T>(x, y);
        } else {
            (void)R;
            o[0]  = join_blade_00<T>(x, y);
            o[1]  = join_blade_01<T>(x, y);
            o[2]  = join_blade_02<T>(x, y);
            o[3]  = join_blade_03<T>(x, y);
            o[4]  = join_blade_04<T>(x, y);
            o[5]  = join_blade_05<T>(x, y);
            o[6]  = join_blade_06<T>(x, y);
            o[7]  = join_blade_07<T>(x, y);
            o[8]  = join_blade_08<T>(x, y);
            o[9]  = join_blade_09<T>(x, y);
            o[10] = join_blade_10<T>(x, y);
            o[11] = join_blade_11<T>(x, y);
            o[12] = join_blade_12<T>(x, y);
            o[13] = join_blade_13<T>(x, y);
            o[14] = join_blade_14<T>(x, y);
            o[15] = join_blade_15<T>(x, y);
        }
    }
}

template <typename T>
static void gp_kernel_v2_1(const T* __restrict__ X,
                                const T* __restrict__ Y,
                                T* __restrict__ O,
                                int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        o[0]  = gp_blade_00_acc2<T>(x, y);
        o[1]  = gp_blade_01_acc2<T>(x, y);
        o[2]  = gp_blade_02_acc2<T>(x, y);
        o[3]  = gp_blade_03_acc2<T>(x, y);
        o[4]  = gp_blade_04_acc2<T>(x, y);
        o[5]  = gp_blade_05_acc2<T>(x, y);
        o[6]  = gp_blade_06_acc2<T>(x, y);
        o[7]  = gp_blade_07_acc2<T>(x, y);
        o[8]  = gp_blade_08_acc2<T>(x, y);
        o[9]  = gp_blade_09_acc2<T>(x, y);
        o[10] = gp_blade_10_acc2<T>(x, y);
        o[11] = gp_blade_11_acc2<T>(x, y);
        o[12] = gp_blade_12_acc2<T>(x, y);
        o[13] = gp_blade_13_acc2<T>(x, y);
        o[14] = gp_blade_14_acc2<T>(x, y);
        o[15] = gp_blade_15_acc2<T>(x, y);
    }
}

template <typename T>
static void gp_kernel_v2_2(const T* __restrict__ X,
                                const T* __restrict__ Y,
                                T* __restrict__ O,
                                int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        o[0]  = gp_blade_00_acc4<T>(x, y);
        o[1]  = gp_blade_01_acc4<T>(x, y);
        o[2]  = gp_blade_02_acc4<T>(x, y);
        o[3]  = gp_blade_03_acc4<T>(x, y);
        o[4]  = gp_blade_04_acc4<T>(x, y);
        o[5]  = gp_blade_05_acc4<T>(x, y);
        o[6]  = gp_blade_06_acc4<T>(x, y);
        o[7]  = gp_blade_07_acc4<T>(x, y);
        o[8]  = gp_blade_08_acc4<T>(x, y);
        o[9]  = gp_blade_09_acc4<T>(x, y);
        o[10] = gp_blade_10_acc4<T>(x, y);
        o[11] = gp_blade_11_acc4<T>(x, y);
        o[12] = gp_blade_12_acc4<T>(x, y);
        o[13] = gp_blade_13_acc4<T>(x, y);
        o[14] = gp_blade_14_acc4<T>(x, y);
        o[15] = gp_blade_15_acc4<T>(x, y);
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2_1(const T* __restrict__ X,
                                  const T* __restrict__ Y,
                                  const T* __restrict__ R,
                                  T* __restrict__ O,
                                  int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        if constexpr (HasRef) {
            const T s = R[n];
            o[0]  = s * join_blade_00_acc2<T>(x, y);
            o[1]  = s * join_blade_01_acc2<T>(x, y);
            o[2]  = s * join_blade_02_acc2<T>(x, y);
            o[3]  = s * join_blade_03_acc2<T>(x, y);
            o[4]  = s * join_blade_04_acc2<T>(x, y);
            o[5]  = s * join_blade_05_acc2<T>(x, y);
            o[6]  = s * join_blade_06_acc2<T>(x, y);
            o[7]  = s * join_blade_07_acc2<T>(x, y);
            o[8]  = s * join_blade_08_acc2<T>(x, y);
            o[9]  = s * join_blade_09_acc2<T>(x, y);
            o[10] = s * join_blade_10_acc2<T>(x, y);
            o[11] = s * join_blade_11_acc2<T>(x, y);
            o[12] = s * join_blade_12_acc2<T>(x, y);
            o[13] = s * join_blade_13_acc2<T>(x, y);
            o[14] = s * join_blade_14_acc2<T>(x, y);
            o[15] = s * join_blade_15_acc2<T>(x, y);
        } else {
            (void)R;
            o[0]  = join_blade_00_acc2<T>(x, y);
            o[1]  = join_blade_01_acc2<T>(x, y);
            o[2]  = join_blade_02_acc2<T>(x, y);
            o[3]  = join_blade_03_acc2<T>(x, y);
            o[4]  = join_blade_04_acc2<T>(x, y);
            o[5]  = join_blade_05_acc2<T>(x, y);
            o[6]  = join_blade_06_acc2<T>(x, y);
            o[7]  = join_blade_07_acc2<T>(x, y);
            o[8]  = join_blade_08_acc2<T>(x, y);
            o[9]  = join_blade_09_acc2<T>(x, y);
            o[10] = join_blade_10_acc2<T>(x, y);
            o[11] = join_blade_11_acc2<T>(x, y);
            o[12] = join_blade_12_acc2<T>(x, y);
            o[13] = join_blade_13_acc2<T>(x, y);
            o[14] = join_blade_14_acc2<T>(x, y);
            o[15] = join_blade_15_acc2<T>(x, y);
        }
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2_2(const T* __restrict__ X,
                                  const T* __restrict__ Y,
                                  const T* __restrict__ R,
                                  T* __restrict__ O,
                                  int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        if constexpr (HasRef) {
            const T s = R[n];
            o[0]  = s * join_blade_00_acc4<T>(x, y);
            o[1]  = s * join_blade_01_acc4<T>(x, y);
            o[2]  = s * join_blade_02_acc4<T>(x, y);
            o[3]  = s * join_blade_03_acc4<T>(x, y);
            o[4]  = s * join_blade_04_acc4<T>(x, y);
            o[5]  = s * join_blade_05_acc4<T>(x, y);
            o[6]  = s * join_blade_06_acc4<T>(x, y);
            o[7]  = s * join_blade_07_acc4<T>(x, y);
            o[8]  = s * join_blade_08_acc4<T>(x, y);
            o[9]  = s * join_blade_09_acc4<T>(x, y);
            o[10] = s * join_blade_10_acc4<T>(x, y);
            o[11] = s * join_blade_11_acc4<T>(x, y);
            o[12] = s * join_blade_12_acc4<T>(x, y);
            o[13] = s * join_blade_13_acc4<T>(x, y);
            o[14] = s * join_blade_14_acc4<T>(x, y);
            o[15] = s * join_blade_15_acc4<T>(x, y);
        } else {
            (void)R;
            o[0]  = join_blade_00_acc4<T>(x, y);
            o[1]  = join_blade_01_acc4<T>(x, y);
            o[2]  = join_blade_02_acc4<T>(x, y);
            o[3]  = join_blade_03_acc4<T>(x, y);
            o[4]  = join_blade_04_acc4<T>(x, y);
            o[5]  = join_blade_05_acc4<T>(x, y);
            o[6]  = join_blade_06_acc4<T>(x, y);
            o[7]  = join_blade_07_acc4<T>(x, y);
            o[8]  = join_blade_08_acc4<T>(x, y);
            o[9]  = join_blade_09_acc4<T>(x, y);
            o[10] = join_blade_10_acc4<T>(x, y);
            o[11] = join_blade_11_acc4<T>(x, y);
            o[12] = join_blade_12_acc4<T>(x, y);
            o[13] = join_blade_13_acc4<T>(x, y);
            o[14] = join_blade_14_acc4<T>(x, y);
            o[15] = join_blade_15_acc4<T>(x, y);
        }
    }
}

template <typename T>
static void gp_kernel_v2_3(const T* __restrict__ X,
                                const T* __restrict__ Y,
                                T* __restrict__ O,
                                int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        gp_block_ilp2<T>(X + 16 * n, Y + 16 * n, O + 16 * n);
    }
}

template <typename T>
static void gp_kernel_v2_4(const T* __restrict__ X,
                                const T* __restrict__ Y,
                                T* __restrict__ O,
                                int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        gp_block_ilp4<T>(X + 16 * n, Y + 16 * n, O + 16 * n);
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2_3(const T* __restrict__ X,
                                  const T* __restrict__ Y,
                                  const T* __restrict__ R,
                                  T* __restrict__ O,
                                  int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        T* o = O + 16 * n;
        join_block_ilp2<T>(X + 16 * n, Y + 16 * n, o);
        if constexpr (HasRef) {
            const T s = R[n];
            for (int i = 0; i < 16; ++i) o[i] *= s;
        } else {
            (void)R;
        }
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2_4(const T* __restrict__ X,
                                  const T* __restrict__ Y,
                                  const T* __restrict__ R,
                                  T* __restrict__ O,
                                  int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        T* o = O + 16 * n;
        join_block_ilp4<T>(X + 16 * n, Y + 16 * n, o);
        if constexpr (HasRef) {
            const T s = R[n];
            for (int i = 0; i < 16; ++i) o[i] *= s;
        } else {
            (void)R;
        }
    }
}

// ---------------------------------------------------------------------------
// v2_5 / v2_6 — multivector loop unrolled by K=2 / K=4 (non-vectorized).
// The work is in the generated gp_/join_loop_unroll{K} kernels; these are thin
// wrappers so they slot into the run_*_variant dispatch like the other ops.
// ---------------------------------------------------------------------------

template <typename T>
static void gp_kernel_v2_5(const T* __restrict__ X, const T* __restrict__ Y,
                           T* __restrict__ O, int64_t N) {
    gp_loop_unroll2<T>(X, Y, O, N);
}

template <typename T>
static void gp_kernel_v2_6(const T* __restrict__ X, const T* __restrict__ Y,
                           T* __restrict__ O, int64_t N) {
    gp_loop_unroll4<T>(X, Y, O, N);
}

template <typename T, bool HasRef>
static void join_kernel_v2_5(const T* __restrict__ X, const T* __restrict__ Y,
                             const T* __restrict__ R, T* __restrict__ O, int64_t N) {
    join_loop_unroll2<T, HasRef>(X, Y, R, O, N);
}

template <typename T, bool HasRef>
static void join_kernel_v2_6(const T* __restrict__ X, const T* __restrict__ Y,
                             const T* __restrict__ R, T* __restrict__ O, int64_t N) {
    join_loop_unroll4<T, HasRef>(X, Y, R, O, N);
}

// ---------------------------------------------------------------------------
// v2_7 — cache blocking (GEMM-style tiling), non-vectorized.
// The batch is processed in tiles of GP_TILE multivectors sized to stay hot in
// L1d (GP_TILE=64 -> X+Y+O ~= 24 KB at fp64), with software prefetch of the
// next tile's X/Y. NOTE: this op has no cross-output operand reuse (each
// multivector's X/Y is read exactly once), so arithmetic intensity is fixed;
// the only thing blocking can buy here is locality/prefetch, not reuse. Kept as
// a standalone variant to measure that contribution; cache blocking on top of
// the vectorized v3 is a separate, later step.
// ---------------------------------------------------------------------------

constexpr int64_t kPgaTile = 64;

template <typename T>
static void gp_kernel_v2_7(const T* __restrict__ X, const T* __restrict__ Y,
                           T* __restrict__ O, int64_t N) {
    for (int64_t base = 0; base < N; base += kPgaTile) {
        const int64_t end = std::min(base + kPgaTile, N);
#if defined(__AVX2__) && defined(__FMA__)
        if (end < N) {
            _mm_prefetch(reinterpret_cast<const char*>(X + 16 * end), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(Y + 16 * end), _MM_HINT_T0);
        }
#endif
        for (int64_t n = base; n < end; ++n) {
            gp_block_ilp2<T>(X + 16 * n, Y + 16 * n, O + 16 * n);
        }
    }
}

template <typename T, bool HasRef>
static void join_kernel_v2_7(const T* __restrict__ X, const T* __restrict__ Y,
                             const T* __restrict__ R, T* __restrict__ O, int64_t N) {
    for (int64_t base = 0; base < N; base += kPgaTile) {
        const int64_t end = std::min(base + kPgaTile, N);
#if defined(__AVX2__) && defined(__FMA__)
        if (end < N) {
            _mm_prefetch(reinterpret_cast<const char*>(X + 16 * end), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(Y + 16 * end), _MM_HINT_T0);
        }
#endif
        for (int64_t n = base; n < end; ++n) {
            T* o = O + 16 * n;
            join_block_ilp2<T>(X + 16 * n, Y + 16 * n, o);
            if constexpr (HasRef) {
                const T s = R[n];
                for (int i = 0; i < 16; ++i) o[i] *= s;
            } else {
                (void)R;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3 - vectorization across multivectors via SoA transpose + AVX2 (fp32).
// Batch-size-agnostic: flattens to N multivectors, processes 8 at a time
// (one __m256 lane each) with a scalar remainder for N % 8. The 8-wide block
// is an internal SIMD detail, independent of the caller's dynamic batch shape.
// This variant is intentionally fp32 + AVX2/FMA only.
// ---------------------------------------------------------------------------

#if defined(__AVX2__) && defined(__FMA__)
static inline void transpose8x8_ps(__m256& r0, __m256& r1, __m256& r2, __m256& r3,
                                   __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    r0 = _mm256_permute2f128_ps(u0, u4, 0x20);
    r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    r2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    r4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    r6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    r7 = _mm256_permute2f128_ps(u3, u7, 0x31);
}

// AoS -> SoA: 8 consecutive multivectors (rows of 16 contiguous floats) become
// sb[16][8], where sb[j] holds component j across the 8 lanes consumed by one
// __m256 in the generated FMA block.
static inline void soa_transpose_in_f32(const float* __restrict__ S, int64_t n,
                                        float sb[16][8]) {
    const float* r0p = S + 16 * (n + 0);
    const float* r1p = S + 16 * (n + 1);
    const float* r2p = S + 16 * (n + 2);
    const float* r3p = S + 16 * (n + 3);
    const float* r4p = S + 16 * (n + 4);
    const float* r5p = S + 16 * (n + 5);
    const float* r6p = S + 16 * (n + 6);
    const float* r7p = S + 16 * (n + 7);
    for (int b = 0; b < 2; ++b) {
        __m256 r0 = _mm256_loadu_ps(r0p + 8 * b);
        __m256 r1 = _mm256_loadu_ps(r1p + 8 * b);
        __m256 r2 = _mm256_loadu_ps(r2p + 8 * b);
        __m256 r3 = _mm256_loadu_ps(r3p + 8 * b);
        __m256 r4 = _mm256_loadu_ps(r4p + 8 * b);
        __m256 r5 = _mm256_loadu_ps(r5p + 8 * b);
        __m256 r6 = _mm256_loadu_ps(r6p + 8 * b);
        __m256 r7 = _mm256_loadu_ps(r7p + 8 * b);
        transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
        _mm256_store_ps(sb[8 * b + 0], r0);
        _mm256_store_ps(sb[8 * b + 1], r1);
        _mm256_store_ps(sb[8 * b + 2], r2);
        _mm256_store_ps(sb[8 * b + 3], r3);
        _mm256_store_ps(sb[8 * b + 4], r4);
        _mm256_store_ps(sb[8 * b + 5], r5);
        _mm256_store_ps(sb[8 * b + 6], r6);
        _mm256_store_ps(sb[8 * b + 7], r7);
    }
}

// SoA -> AoS: ob[16][8] (component-major) back to 8 contiguous output rows.
// When Scale is true, each lane's row is multiplied by its reference scalar s[l]
// (the equi_join reference factor) before storing.
template <bool Scale>
static inline void soa_transpose_out_f32(const float ob[16][8], int64_t n,
                                         float* __restrict__ O,
                                         const float s[8]) {
    float* r0p = O + 16 * (n + 0);
    float* r1p = O + 16 * (n + 1);
    float* r2p = O + 16 * (n + 2);
    float* r3p = O + 16 * (n + 3);
    float* r4p = O + 16 * (n + 4);
    float* r5p = O + 16 * (n + 5);
    float* r6p = O + 16 * (n + 6);
    float* r7p = O + 16 * (n + 7);
    for (int b = 0; b < 2; ++b) {
        __m256 r0 = _mm256_load_ps(ob[8 * b + 0]);
        __m256 r1 = _mm256_load_ps(ob[8 * b + 1]);
        __m256 r2 = _mm256_load_ps(ob[8 * b + 2]);
        __m256 r3 = _mm256_load_ps(ob[8 * b + 3]);
        __m256 r4 = _mm256_load_ps(ob[8 * b + 4]);
        __m256 r5 = _mm256_load_ps(ob[8 * b + 5]);
        __m256 r6 = _mm256_load_ps(ob[8 * b + 6]);
        __m256 r7 = _mm256_load_ps(ob[8 * b + 7]);
        transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
        if constexpr (Scale) {
            r0 = _mm256_mul_ps(r0, _mm256_set1_ps(s[0]));
            r1 = _mm256_mul_ps(r1, _mm256_set1_ps(s[1]));
            r2 = _mm256_mul_ps(r2, _mm256_set1_ps(s[2]));
            r3 = _mm256_mul_ps(r3, _mm256_set1_ps(s[3]));
            r4 = _mm256_mul_ps(r4, _mm256_set1_ps(s[4]));
            r5 = _mm256_mul_ps(r5, _mm256_set1_ps(s[5]));
            r6 = _mm256_mul_ps(r6, _mm256_set1_ps(s[6]));
            r7 = _mm256_mul_ps(r7, _mm256_set1_ps(s[7]));
        }
        _mm256_storeu_ps(r0p + 8 * b, r0);
        _mm256_storeu_ps(r1p + 8 * b, r1);
        _mm256_storeu_ps(r2p + 8 * b, r2);
        _mm256_storeu_ps(r3p + 8 * b, r3);
        _mm256_storeu_ps(r4p + 8 * b, r4);
        _mm256_storeu_ps(r5p + 8 * b, r5);
        _mm256_storeu_ps(r6p + 8 * b, r6);
        _mm256_storeu_ps(r7p + 8 * b, r7);
    }
}
#endif  // __AVX2__ && __FMA__

// ---------------------------------------------------------------------------
// v3 (AVX-512) - same SoA scheme widened to 16 lanes per __m512. The 16-wide
// AoS<->SoA transpose is built from two of the proven 8x8 AVX2 transposes
// (lanes 0..7 -> tile columns 0..7, lanes 8..15 -> columns 8..15), so only the
// FMA block widens; the shuffle network is the already-tested one. Requires
// AVX-512F (and the AVX2/FMA helpers above, which Tiger Lake et al. also have).
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
// AoS -> SoA for one octet: 8 consecutive multivectors at rows n+col..n+col+7
// fill tile columns col..col+7 of sb[16][16] (sb[j] = component j, 16 lanes).
static inline void soa_in_octet_f32(const float* __restrict__ S, int64_t n,
                                    int col, float sb[16][16]) {
    const float* r0p = S + 16 * (n + col + 0);
    const float* r1p = S + 16 * (n + col + 1);
    const float* r2p = S + 16 * (n + col + 2);
    const float* r3p = S + 16 * (n + col + 3);
    const float* r4p = S + 16 * (n + col + 4);
    const float* r5p = S + 16 * (n + col + 5);
    const float* r6p = S + 16 * (n + col + 6);
    const float* r7p = S + 16 * (n + col + 7);
    for (int b = 0; b < 2; ++b) {
        __m256 r0 = _mm256_loadu_ps(r0p + 8 * b);
        __m256 r1 = _mm256_loadu_ps(r1p + 8 * b);
        __m256 r2 = _mm256_loadu_ps(r2p + 8 * b);
        __m256 r3 = _mm256_loadu_ps(r3p + 8 * b);
        __m256 r4 = _mm256_loadu_ps(r4p + 8 * b);
        __m256 r5 = _mm256_loadu_ps(r5p + 8 * b);
        __m256 r6 = _mm256_loadu_ps(r6p + 8 * b);
        __m256 r7 = _mm256_loadu_ps(r7p + 8 * b);
        transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
        _mm256_storeu_ps(&sb[8 * b + 0][col], r0);
        _mm256_storeu_ps(&sb[8 * b + 1][col], r1);
        _mm256_storeu_ps(&sb[8 * b + 2][col], r2);
        _mm256_storeu_ps(&sb[8 * b + 3][col], r3);
        _mm256_storeu_ps(&sb[8 * b + 4][col], r4);
        _mm256_storeu_ps(&sb[8 * b + 5][col], r5);
        _mm256_storeu_ps(&sb[8 * b + 6][col], r6);
        _mm256_storeu_ps(&sb[8 * b + 7][col], r7);
    }
}

// SoA -> AoS for one octet: tile columns col..col+7 of ob[16][16] become the 8
// output rows n+col..n+col+7. When Scale, lane l's row is multiplied by s[col+l].
template <bool Scale>
static inline void soa_out_octet_f32(const float ob[16][16], int64_t n, int col,
                                     float* __restrict__ O, const float s[16]) {
    for (int b = 0; b < 2; ++b) {
        __m256 r0 = _mm256_loadu_ps(&ob[8 * b + 0][col]);
        __m256 r1 = _mm256_loadu_ps(&ob[8 * b + 1][col]);
        __m256 r2 = _mm256_loadu_ps(&ob[8 * b + 2][col]);
        __m256 r3 = _mm256_loadu_ps(&ob[8 * b + 3][col]);
        __m256 r4 = _mm256_loadu_ps(&ob[8 * b + 4][col]);
        __m256 r5 = _mm256_loadu_ps(&ob[8 * b + 5][col]);
        __m256 r6 = _mm256_loadu_ps(&ob[8 * b + 6][col]);
        __m256 r7 = _mm256_loadu_ps(&ob[8 * b + 7][col]);
        transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
        if constexpr (Scale) {
            r0 = _mm256_mul_ps(r0, _mm256_set1_ps(s[col + 0]));
            r1 = _mm256_mul_ps(r1, _mm256_set1_ps(s[col + 1]));
            r2 = _mm256_mul_ps(r2, _mm256_set1_ps(s[col + 2]));
            r3 = _mm256_mul_ps(r3, _mm256_set1_ps(s[col + 3]));
            r4 = _mm256_mul_ps(r4, _mm256_set1_ps(s[col + 4]));
            r5 = _mm256_mul_ps(r5, _mm256_set1_ps(s[col + 5]));
            r6 = _mm256_mul_ps(r6, _mm256_set1_ps(s[col + 6]));
            r7 = _mm256_mul_ps(r7, _mm256_set1_ps(s[col + 7]));
        }
        _mm256_storeu_ps(O + 16 * (n + col + 0) + 8 * b, r0);
        _mm256_storeu_ps(O + 16 * (n + col + 1) + 8 * b, r1);
        _mm256_storeu_ps(O + 16 * (n + col + 2) + 8 * b, r2);
        _mm256_storeu_ps(O + 16 * (n + col + 3) + 8 * b, r3);
        _mm256_storeu_ps(O + 16 * (n + col + 4) + 8 * b, r4);
        _mm256_storeu_ps(O + 16 * (n + col + 5) + 8 * b, r5);
        _mm256_storeu_ps(O + 16 * (n + col + 6) + 8 * b, r6);
        _mm256_storeu_ps(O + 16 * (n + col + 7) + 8 * b, r7);
    }
}
#endif  // __AVX512F__ && __AVX2__ && __FMA__

template <typename T>
static void gp_kernel_v3(const T* __restrict__ X, const T* __restrict__ Y,
                         T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, float>) {
        int64_t n = 0;
        // AVX-512 SoA (16-MV blocks) is intentionally skipped for gp: the
        // 16×16 transpose overhead + ZMM frequency throttle on Tiger Lake
        // makes it slower than the AVX2 8-MV path across all tested sizes.
        for (; n + 8 <= N; n += 8) {
            alignas(32) float xb[16][8], yb[16][8], ob[16][8];
            soa_transpose_in_f32(X, n, xb);
            soa_transpose_in_f32(Y, n, yb);
            gp_soa_block_f32(xb, yb, ob);
            soa_transpose_out_f32<false>(ob, n, O, nullptr);
        }
        for (; n < N; ++n) {
            gp_block_ilp2<float>(X + 16 * n, Y + 16 * n, O + 16 * n);
        }
        return;
    }
#endif
    (void)X; (void)Y; (void)O; (void)N;
    // v3 is a float32 + AVX2/FMA-only kernel: no scalar/fp64 fallback by design.
    TORCH_CHECK(false, "geometric_product_v3: requires float32 inputs and an "
                       "AVX2/FMA build; got an unsupported dtype or target.");
}

template <typename T, bool HasRef>
static void join_kernel_v3(const T* __restrict__ X, const T* __restrict__ Y,
                           const T* __restrict__ R, T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (!HasRef) (void)R;
    if constexpr (std::is_same_v<T, float>) {
        int64_t n = 0;
        // Same rationale as gp: AVX-512 16-MV SoA is slower than AVX2 8-MV
        // on shuffle-heavy, bandwidth-bound operations on Tiger Lake.
        for (; n + 8 <= N; n += 8) {
            alignas(32) float xb[16][8], yb[16][8], ob[16][8];
            soa_transpose_in_f32(X, n, xb);
            soa_transpose_in_f32(Y, n, yb);
            join_soa_block_f32(xb, yb, ob);
            if constexpr (HasRef) {
                const float s[8] = {R[n + 0], R[n + 1],
                                    R[n + 2], R[n + 3],
                                    R[n + 4], R[n + 5],
                                    R[n + 6], R[n + 7]};
                soa_transpose_out_f32<true>(ob, n, O, s);
            } else {
                soa_transpose_out_f32<false>(ob, n, O, nullptr);
            }
        }
        for (; n < N; ++n) {
            T* o = O + 16 * n;
            join_block_ilp2<float>(X + 16 * n, Y + 16 * n, o);
            if constexpr (HasRef) {
                const float s = R[n];
                for (int i = 0; i < 16; ++i) o[i] *= s;
            }
        }
        return;
    }
#endif
    (void)X; (void)Y; (void)R; (void)O; (void)N;
    // v3 is a float32 + AVX2/FMA-only kernel: no scalar/fp64 fallback by design.
    TORCH_CHECK(false, "equi_join_v3: requires float32 inputs and an "
                       "AVX2/FMA build; got an unsupported dtype or target.");
}

using CacheKey = std::tuple<c10::DeviceType, c10::DeviceIndex, c10::ScalarType>;

CacheKey make_key(c10::Device device, c10::ScalarType dtype) {
    return std::make_tuple(device.type(), device.index(), dtype);
}

std::mutex g_cache_mu;
std::map<CacheKey, torch::Tensor> g_gp_basis;
std::map<CacheKey, torch::Tensor> g_op_basis;
std::map<CacheKey, torch::Tensor> g_join_kernel;
std::map<CacheKey, torch::Tensor> g_dual_sign;
std::map<c10::DeviceType, torch::Tensor> g_dual_perm;  // perm is Long, dtype-independent

torch::Tensor build_basis_3d(const int8_t (*src)[16][16],
                             c10::Device device, c10::ScalarType dtype) {
    auto cpu_int8 = torch::from_blob(
        const_cast<int8_t*>(reinterpret_cast<const int8_t*>(src)),
        {16, 16, 16},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
    return cpu_int8.clone().to(device, dtype);
}

enum class BasisKind { GP, OP };

torch::Tensor load_basis(BasisKind kind,
                         c10::Device device,
                         c10::ScalarType dtype) {
    auto& cache = (kind == BasisKind::GP) ? g_gp_basis : g_op_basis;
    auto key = make_key(device, dtype);

    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const int8_t (*src)[16][16] =
        (kind == BasisKind::GP) ? GP_BASIS : OP_BASIS;
    auto t = build_basis_3d(src, device, dtype);
    cache.emplace(key, t);
    return t;
}

std::pair<torch::Tensor, torch::Tensor>
load_dual_constants(c10::Device device, c10::ScalarType dtype) {
    std::lock_guard<std::mutex> lock(g_cache_mu);

    auto perm_it = g_dual_perm.find(device.type());
    if (perm_it == g_dual_perm.end()) {
        auto cpu_perm = torch::from_blob(
            const_cast<int64_t*>(DUAL_PERM), {16},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        perm_it = g_dual_perm.emplace(device.type(), cpu_perm.clone().to(device)).first;
    }

    auto sign_key = make_key(device, dtype);
    auto sign_it = g_dual_sign.find(sign_key);
    if (sign_it == g_dual_sign.end()) {
        auto cpu_sign = torch::from_blob(
            const_cast<int8_t*>(DUAL_SIGN), {16},
            torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
        sign_it = g_dual_sign.emplace(sign_key, cpu_sign.clone().to(device, dtype)).first;
    }

    return {perm_it->second, sign_it->second};
}

void check_multivector(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() >= 1, name, ": expected at least 1 dim, got ", t.dim());
    TORCH_CHECK(t.size(-1) == 16, name, ": last dim must be 16, got ", t.size(-1));
    TORCH_CHECK(t.is_floating_point(), name, ": expected floating-point dtype, got ", t.dtype());
}

}  // namespace

torch::Tensor load_gp_basis(c10::Device device, c10::ScalarType dtype) {
    return load_basis(BasisKind::GP, device, dtype);
}

torch::Tensor load_op_basis(c10::Device device, c10::ScalarType dtype) {
    return load_basis(BasisKind::OP, device, dtype);
}

torch::Tensor equi_dual(const torch::Tensor& x) {
    check_multivector(x, "equi_dual: x");
    auto [perm, sign] = load_dual_constants(x.device(), x.scalar_type());
    return sign * torch::index_select(x, -1, perm);
}

torch::Tensor outer_product(const torch::Tensor& x, const torch::Tensor& y) {
    check_multivector(x, "outer_product: x");
    check_multivector(y, "outer_product: y");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "outer_product: x and y must share dtype");
    auto basis = load_op_basis(x.device(), x.scalar_type());
    return torch::einsum("ijk, ...j, ...k -> ...i", {basis, x, y});
}

torch::Tensor compute_join_kernel(c10::Device device, c10::ScalarType dtype) {
    auto key = make_key(device, dtype);
    {
        std::lock_guard<std::mutex> lock(g_cache_mu);
        auto it = g_join_kernel.find(key);
        if (it != g_join_kernel.end()) return it->second;
    }

    auto opts = torch::TensorOptions().dtype(dtype).device(device);
    auto kernel = torch::zeros({16, 16, 16}, opts);

    for (int64_t i = 0; i < 16; ++i) {
        for (int64_t j = 0; j < 16; ++j) {
            auto x = torch::zeros({16}, opts);
            auto y = torch::zeros({16}, opts);
            x[i] = 1.0;
            y[j] = 1.0;
            auto col = equi_dual(outer_product(equi_dual(x), equi_dual(y)));
            kernel.select(1, i).select(1, j).copy_(col);
        }
    }

    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto [it, inserted] = g_join_kernel.emplace(key, kernel);
    return it->second;
}

torch::Tensor geometric_product(const torch::Tensor& x, const torch::Tensor& y) {
    check_multivector(x, "geometric_product: x");
    check_multivector(y, "geometric_product: y");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "geometric_product: x and y must share dtype");
    TORCH_CHECK(x.device() == y.device(),
                "geometric_product: x and y must share device");
    TORCH_CHECK(x.device().is_cpu(),
                "geometric_product: CPU-only kernel; got device ", x.device());

    auto bcast = at::broadcast_tensors({x, y});
    auto xc = bcast[0].contiguous();
    auto yc = bcast[1].contiguous();
    auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "geometric_product_cpu", [&]{
        gp_kernel_v2<scalar_t>(xc.data_ptr<scalar_t>(),
                                 yc.data_ptr<scalar_t>(),
                                 out.data_ptr<scalar_t>(),
                                 N);
    });
    return out;
}

torch::Tensor equi_join(const torch::Tensor& x,
                        const torch::Tensor& y,
                        const c10::optional<torch::Tensor>& reference) {
    check_multivector(x, "equi_join: x");
    check_multivector(y, "equi_join: y");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "equi_join: x and y must share dtype");
    TORCH_CHECK(x.device() == y.device(),
                "equi_join: x and y must share device");
    TORCH_CHECK(x.device().is_cpu(),
                "equi_join: CPU-only kernel; got device ", x.device());
    if (reference.has_value()) {
        check_multivector(*reference, "equi_join: reference");
        TORCH_CHECK(reference->scalar_type() == x.scalar_type(),
                    "equi_join: reference must share dtype with x");
        TORCH_CHECK(reference->device() == x.device(),
                    "equi_join: reference must share device with x");
    }

    auto bcast = at::broadcast_tensors({x, y});
    auto xc = bcast[0].contiguous();
    auto yc = bcast[1].contiguous();
    auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;

    // Blade-14 scalars only (see run_join_variant).
    torch::Tensor ref14;
    if (reference.has_value()) {
        ref14 = reference->expand_as(xc).select(-1, 14).contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "equi_join_cpu", [&]{
        if (reference.has_value()) {
            join_kernel_v2<scalar_t, true>(
                xc.data_ptr<scalar_t>(),
                yc.data_ptr<scalar_t>(),
                ref14.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                N);
        } else {
            join_kernel_v2<scalar_t, false>(
                xc.data_ptr<scalar_t>(),
                yc.data_ptr<scalar_t>(),
                nullptr,
                out.data_ptr<scalar_t>(),
                N);
        }
    });
    return out;
}

namespace {

template <typename Kernel>
torch::Tensor run_gp_variant(const torch::Tensor& x, const torch::Tensor& y,
                             const char* name, Kernel kernel) {
    check_multivector(x, name);
    check_multivector(y, name);
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                name, ": x and y must share dtype");
    TORCH_CHECK(x.device() == y.device(),
                name, ": x and y must share device");
    TORCH_CHECK(x.device().is_cpu(),
                name, ": CPU-only kernel; got device ", x.device());

    auto bcast = at::broadcast_tensors({x, y});
    auto xc = bcast[0].contiguous();
    auto yc = bcast[1].contiguous();
    auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "gp_variant_cpu", [&]{
        kernel(xc.data_ptr<scalar_t>(),
               yc.data_ptr<scalar_t>(),
               out.data_ptr<scalar_t>(),
               N);
    });
    return out;
}

template <typename Kernel>
torch::Tensor run_join_variant(const torch::Tensor& x, const torch::Tensor& y,
                               const c10::optional<torch::Tensor>& reference,
                               const char* name, Kernel kernel) {
    check_multivector(x, name);
    check_multivector(y, name);
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                name, ": x and y must share dtype");
    TORCH_CHECK(x.device() == y.device(),
                name, ": x and y must share device");
    TORCH_CHECK(x.device().is_cpu(),
                name, ": CPU-only kernel; got device ", x.device());
    if (reference.has_value()) {
        check_multivector(*reference, name);
        TORCH_CHECK(reference->scalar_type() == x.scalar_type(),
                    name, ": reference must share dtype with x");
        TORCH_CHECK(reference->device() == x.device(),
                    name, ": reference must share device with x");
    }

    auto bcast = at::broadcast_tensors({x, y});
    auto xc = bcast[0].contiguous();
    auto yc = bcast[1].contiguous();
    auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;

    // The kernels only consume blade 14 of the reference; extracting those
    // scalars from the broadcast view copies N floats instead of materializing
    // the full N x 16 expansion (16x less traffic; for the model's [B,1,1,16]
    // reference the read side is just B distinct values).
    torch::Tensor ref14;
    if (reference.has_value()) {
        ref14 = reference->expand_as(xc).select(-1, 14).contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "join_variant_cpu", [&]{
        const scalar_t* R = reference.has_value() ? ref14.data_ptr<scalar_t>() : nullptr;
        kernel(xc.data_ptr<scalar_t>(),
               yc.data_ptr<scalar_t>(),
               R,
               out.data_ptr<scalar_t>(),
               N,
               reference.has_value());
    });
    return out;
}

}  // namespace

torch::Tensor geometric_product_v0(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v0",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v0<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v1(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v1",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v1<T>(X, Y, O, N);
        });
}

torch::Tensor equi_join_v0(const torch::Tensor& x,
                              const torch::Tensor& y,
                              const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v0",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v0<T, true>(X, Y, R, O, N);
            else         join_kernel_v0<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v1(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v1",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v1<T, true>(X, Y, R, O, N);
            else         join_kernel_v1<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor geometric_product_v2(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2<T>(X, Y, O, N);
        });
}

torch::Tensor equi_join_v2(const torch::Tensor& x,
                           const torch::Tensor& y,
                           const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2<T, true>(X, Y, R, O, N);
            else         join_kernel_v2<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor geometric_product_v2_3(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_3",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_3<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v2_4(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_4",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_4<T>(X, Y, O, N);
        });
}

torch::Tensor equi_join_v2_3(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_3",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_3<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_3<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v2_4(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_4",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_4<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_4<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor geometric_product_v2_1(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_1",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_1<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v2_2(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_2",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_2<T>(X, Y, O, N);
        });
}

torch::Tensor equi_join_v2_1(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_1",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_1<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_1<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v2_2(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_2",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_2<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_2<T, false>(X, Y, nullptr, O, N);
        });
}

static constexpr double EQUI_LINEAR_INV_NORMS[9] = {
    1.0, 0.5, 0.40824829046386302, 0.5, 1.0,
    1.0, 0.5773502691896258, 0.5773502691896258, 1.0,
};

// Sparse structure of the Pin(3,0,1)-equivariant linear basis: each entry is
// {weight_index, dst_blade, src_blade}, i.e. out[dst] += w[wk] * x[src].
// 24 nonzero couplings (16 diagonal + 8 off-diagonal e_0-couplings): the
// single source of truth shared by every version.
struct EquiLinearCoupling { int wk, dst, src; };
static constexpr EquiLinearCoupling EQUI_LINEAR_COUPLINGS[24] = {
    {0, 0, 0},
    {1, 1, 1},  {1, 2, 2},   {1, 3, 3},   {1, 4, 4},
    {2, 5, 5},  {2, 6, 6},   {2, 7, 7},   {2, 8, 8}, {2, 9, 9}, {2, 10, 10},
    {3, 11, 11},{3, 12, 12}, {3, 13, 13}, {3, 14, 14},
    {4, 15, 15},
    {5, 1, 0},
    {6, 5, 2},  {6, 6, 3},   {6, 7, 4},
    {7, 11, 8}, {7, 12, 9},  {7, 13, 10},
    {8, 15, 14},
};

namespace {

// Shared preamble: validate, collapse batch dims, return reshaped views.
struct EquiLinearShapes {
    int64_t out_ch, in_ch, batch;
    torch::Tensor xf, wf;
    std::vector<int64_t> out_shape;
};

EquiLinearShapes equi_linear_prepare(const torch::Tensor& x,
                                     const torch::Tensor& weight) {
    TORCH_CHECK(x.dim() >= 2,        "equi_linear: x needs at least 2 dims");
    TORCH_CHECK(x.size(-1) == 16,    "equi_linear: x last dim must be 16");
    TORCH_CHECK(weight.dim() == 3,   "equi_linear: weight must be 3-D (out, in, 9)");
    TORCH_CHECK(weight.size(2) == 9, "equi_linear: weight last dim must be 9");
    TORCH_CHECK(x.size(-2) == weight.size(1), "equi_linear: in_channels mismatch");

    EquiLinearShapes s;
    s.out_ch = weight.size(0);
    s.in_ch  = weight.size(1);
    s.batch  = x.numel() / (s.in_ch * 16);
    s.xf = x.reshape({s.batch, s.in_ch, 16}).contiguous();
    s.wf = weight.contiguous();
    s.out_shape = x.sizes().vec();
    s.out_shape[s.out_shape.size() - 2] = s.out_ch;
    return s;
}

torch::Tensor equi_linear_finalize(torch::Tensor out,
                                   const c10::optional<torch::Tensor>& bias,
                                   const std::vector<int64_t>& out_shape) {
    if (bias.has_value()) {
        out.select(-1, 0).add_(bias.value());
    }
    return out.reshape(out_shape);
}

}  // namespace

// ---------------------------------------------------------------------------
// ver_0 - NAIVE DENSE BASELINE.
// Hand-written C++ loops, no library calls. For each (o, i) channel pair:
//   1. Build the full dense 16×16 matrix M by contracting weight[o,i,w]
//      with the equivariant basis[w,d,s] over all 9 weights and all 16×16
//      (d,s) entries — including the zeros.
//   2. For every batch element, do a full dense 16×16 matvec M × x[b,i].
// This is the literal C++ translation of einsum "oiw,wds,...is->...od".
// ---------------------------------------------------------------------------
torch::Tensor equi_linear_v0(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    auto s = equi_linear_prepare(x, weight);
    auto out = torch::zeros({s.batch, s.out_ch, 16}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear_v0", [&] {
        using T = scalar_t;

        // Build the 9×16×16 equivariant basis from the coupling table.
        T basis[9][16][16] = {};
        for (const auto& c : EQUI_LINEAR_COUPLINGS)
            basis[c.wk][c.dst][c.src] = normalize_basis
                ? T(EQUI_LINEAR_INV_NORMS[c.wk]) : T(1);

        auto xa = s.xf.accessor<T, 3>();
        auto wa = s.wf.accessor<T, 3>();
        auto oa = out.accessor<T, 3>();

        for (int64_t o = 0; o < s.out_ch; ++o) {
            for (int64_t i = 0; i < s.in_ch; ++i) {
                // Step 1: dense 16×16 matrix M[d][s] = sum_w weight[o,i,w] * basis[w,d,s]
                T M[16][16] = {};
                for (int w = 0; w < 9; ++w)
                    for (int d = 0; d < 16; ++d)
                        for (int sc = 0; sc < 16; ++sc)
                            M[d][sc] += wa[o][i][w] * basis[w][d][sc];

                // Step 2: dense 16×16 matvec for every batch element
                for (int64_t b = 0; b < s.batch; ++b)
                    for (int d = 0; d < 16; ++d)
                        for (int sc = 0; sc < 16; ++sc)
                            oa[b][o][d] += M[d][sc] * xa[b][i][sc];
            }
        }
    });

    return equi_linear_finalize(out, bias, s.out_shape);
}

// ---------------------------------------------------------------------------
// ver_1 - MATH OPTIMIZATION (current production kernel).
// Exploits basis sparsity: 9 distinct weight scalars, 24 nonzero couplings,
// hand-unrolled. ~24 MAC per (B,o,i), ~10x fewer FLOPs than ver_0.
// NOTE: normalization is (intentionally, for now) recomputed inside the batch
// loop; this is the redundancy ver_2 will remove.
// ---------------------------------------------------------------------------
torch::Tensor equi_linear_v1(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    auto s = equi_linear_prepare(x, weight);
    auto out = torch::zeros({s.batch, s.out_ch, 16}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear_v1", [&] {
        using T = scalar_t;
        const T inv_norms[9] = {
            T(EQUI_LINEAR_INV_NORMS[0]), T(EQUI_LINEAR_INV_NORMS[1]),
            T(EQUI_LINEAR_INV_NORMS[2]), T(EQUI_LINEAR_INV_NORMS[3]),
            T(EQUI_LINEAR_INV_NORMS[4]), T(EQUI_LINEAR_INV_NORMS[5]),
            T(EQUI_LINEAR_INV_NORMS[6]), T(EQUI_LINEAR_INV_NORMS[7]),
            T(EQUI_LINEAR_INV_NORMS[8]),
        };

        auto xa = s.xf.accessor<T, 3>();
        auto wa = s.wf.accessor<T, 3>();
        auto oa = out.accessor<T, 3>();

        for (int64_t b = 0; b < s.batch; ++b) {
            for (int64_t o = 0; o < s.out_ch; ++o) {
                for (int64_t i = 0; i < s.in_ch; ++i) {
                    T w0 = wa[o][i][0], w1 = wa[o][i][1], w2 = wa[o][i][2];
                    T w3 = wa[o][i][3], w4 = wa[o][i][4], w5 = wa[o][i][5];
                    T w6 = wa[o][i][6], w7 = wa[o][i][7], w8 = wa[o][i][8];

                    if (normalize_basis) {
                        w0 *= inv_norms[0]; w1 *= inv_norms[1]; w2 *= inv_norms[2];
                        w3 *= inv_norms[3]; w4 *= inv_norms[4]; w5 *= inv_norms[5];
                        w6 *= inv_norms[6]; w7 *= inv_norms[7]; w8 *= inv_norms[8];
                    }

                    oa[b][o][0]  += w0 * xa[b][i][0];

                    oa[b][o][1]  += w1 * xa[b][i][1];
                    oa[b][o][2]  += w1 * xa[b][i][2];
                    oa[b][o][3]  += w1 * xa[b][i][3];
                    oa[b][o][4]  += w1 * xa[b][i][4];

                    oa[b][o][5]  += w2 * xa[b][i][5];
                    oa[b][o][6]  += w2 * xa[b][i][6];
                    oa[b][o][7]  += w2 * xa[b][i][7];
                    oa[b][o][8]  += w2 * xa[b][i][8];
                    oa[b][o][9]  += w2 * xa[b][i][9];
                    oa[b][o][10] += w2 * xa[b][i][10];

                    oa[b][o][11] += w3 * xa[b][i][11];
                    oa[b][o][12] += w3 * xa[b][i][12];
                    oa[b][o][13] += w3 * xa[b][i][13];
                    oa[b][o][14] += w3 * xa[b][i][14];

                    oa[b][o][15] += w4 * xa[b][i][15];

                    oa[b][o][1]  += w5 * xa[b][i][0];

                    oa[b][o][5]  += w6 * xa[b][i][2];
                    oa[b][o][6]  += w6 * xa[b][i][3];
                    oa[b][o][7]  += w6 * xa[b][i][4];

                    oa[b][o][11] += w7 * xa[b][i][8];
                    oa[b][o][12] += w7 * xa[b][i][9];
                    oa[b][o][13] += w7 * xa[b][i][10];

                    oa[b][o][15] += w8 * xa[b][i][14];
                }
            }
        }
    });

    return equi_linear_finalize(out, bias, s.out_shape);
}

// ---------------------------------------------------------------------------
// ver_2 - SCALAR OPTIMIZATION UP TO (NOT INCLUDING) SIMD.
// Same FLOPs as v1 (24 couplings/iter); the win is memory & redundancy:
//   O1. Hoist basis normalization: precompute packed normalized weights once
//       per (o,i) (9*N muls) instead of per (b,o,i) (9*P muls).
//   O2. Register-resident output accumulator T acc[16]; accumulate over the
//       whole `i` sweep, store to `out` ONCE per (b,o), eliminating the
//       accessor read-modify-write traffic that pins v1 to memory.
//   O3. Raw `__restrict__` base pointers (data_ptr<T>() + index math) instead
//       of TensorAccessor, removing aliasing-induced reloads and creating the
//       precondition for the compiler to autovectorize.
//   O4. Grade-block structure: the 16 diagonal terms expressed as four
//       fixed-trip loops (lengths 1, 4, 6, 4, 1) the compiler can vectorize
//       under -O3 -march=native; 8 off-diagonal couplings stay scalar.
//   O5. torch::empty (no zero-init pass): output is fully overwritten by
//       the accumulator store.
// ---------------------------------------------------------------------------
namespace {
// Process-wide cache of v2's normalized weights (each of the 9 coefficients
// scaled by its inverse norm). Mirrors the v3 pack cache so that v2 also
// prepares the weights once per tensor rather than on every call, keeping the
// v2->v3 comparison a measure of the kernel and SIMD, not of weight prep.
struct ElNormCache {
    torch::Tensor src;         // strong ref pins the storage behind the key
    uint32_t version;
    torch::Tensor normalized;  // wf * inv_norms, same shape and dtype as wf
};
std::mutex g_el_norm_mu;
std::map<const void*, ElNormCache> g_el_norm_cache;

torch::Tensor el_get_normalized_weights(const torch::Tensor& weight,
                                        const torch::Tensor& wf) {
    // Only cache when wf shares storage with the caller's weight (contiguous);
    // otherwise wf is a fresh temp whose pointer would dangle. Inference
    // tensors carry no version counter and are immutable, so version 0 is safe.
    const bool cacheable = weight.is_contiguous();
    const void* key = wf.data_ptr();
    auto* impl = wf.unsafeGetTensorImpl();
    const uint32_t version =
        impl->is_inference() ? 0 : impl->version_counter().current_version();
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_norm_mu);
        auto it = g_el_norm_cache.find(key);
        if (it != g_el_norm_cache.end() && it->second.version == version &&
            it->second.normalized.scalar_type() == wf.scalar_type()) {
            return it->second.normalized;
        }
    }
    const double inv_d[9] = {
        EQUI_LINEAR_INV_NORMS[0], EQUI_LINEAR_INV_NORMS[1], EQUI_LINEAR_INV_NORMS[2],
        EQUI_LINEAR_INV_NORMS[3], EQUI_LINEAR_INV_NORMS[4], EQUI_LINEAR_INV_NORMS[5],
        EQUI_LINEAR_INV_NORMS[6], EQUI_LINEAR_INV_NORMS[7], EQUI_LINEAR_INV_NORMS[8]};
    auto inv = torch::tensor(c10::ArrayRef<double>(inv_d, 9), wf.options());
    auto nw = (wf * inv).contiguous();
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_norm_mu);
        if (g_el_norm_cache.size() > 64) g_el_norm_cache.clear();
        g_el_norm_cache[key] = ElNormCache{wf, version, nw};
    }
    return nw;
}
}  // namespace

torch::Tensor equi_linear_v2(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    auto s = equi_linear_prepare(x, weight);
    auto out = torch::empty({s.batch, s.out_ch, 16}, x.options());  // O5

    const bool has_bias = bias.has_value();
    torch::Tensor bias_t;
    if (has_bias) bias_t = bias.value().contiguous();

    // O1: normalized weights (9 per (o,i)), cached process-wide so inference
    // normalizes once per weight tensor instead of on every call. Layout:
    // wn[(o*in_ch + i)*9 + k], identical to the previous per-call packing.
    torch::Tensor wn_t =
        normalize_basis ? el_get_normalized_weights(weight, s.wf) : s.wf;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear_v2", [&] {
        using T = scalar_t;
        const T* __restrict__ wn = wn_t.data_ptr<T>();

        // O3: raw restricted base pointers.
        const T* __restrict__ xp_base = s.xf.data_ptr<T>();
        T*       __restrict__ op_base = out.data_ptr<T>();
        const T* __restrict__ bp = has_bias ? bias_t.data_ptr<T>() : nullptr;
        const int64_t in_ch  = s.in_ch;
        const int64_t out_ch = s.out_ch;

        for (int64_t b = 0; b < s.batch; ++b) {
            const T* __restrict__ xb = xp_base + b * in_ch * 16;
            for (int64_t o = 0; o < out_ch; ++o) {
                // O2: register-resident acc. (B) Bias folded into init:
                // blade 0 gets bias[o] (or 0), other blades start at 0.
                T acc[16];
                acc[0] = bp ? bp[o] : T(0);
                for (int k = 1; k < 16; ++k) acc[k] = T(0);

                for (int64_t i = 0; i < in_ch; ++i) {
                    const T* __restrict__ w  = wn + (o * in_ch + i) * 9;
                    const T* __restrict__ xi = xb + i * 16;
                    const T w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
                    const T w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7];
                    const T w8 = w[8];

                    // O4: grade-block diagonal: short fixed-trip loops,
                    // contiguous in xi[] and acc[], cleanly autovectorizable.
                    acc[0]  += w0 * xi[0];
                    for (int k = 1;  k < 5;  ++k) acc[k] += w1 * xi[k];
                    for (int k = 5;  k < 11; ++k) acc[k] += w2 * xi[k];
                    for (int k = 11; k < 15; ++k) acc[k] += w3 * xi[k];
                    acc[15] += w4 * xi[15];

                    // Sparse e_0-coupling (off-diagonal).
                    acc[1]  += w5 * xi[0];
                    acc[5]  += w6 * xi[2];
                    acc[6]  += w6 * xi[3];
                    acc[7]  += w6 * xi[4];
                    acc[11] += w7 * xi[8];
                    acc[12] += w7 * xi[9];
                    acc[13] += w7 * xi[10];
                    acc[15] += w8 * xi[14];
                }

                // O2/O5: single contiguous store per (b, o).
                T* __restrict__ op = op_base + (b * out_ch + o) * 16;
                for (int k = 0; k < 16; ++k) op[k] = acc[k];
            }
        }
    });

    // (B) Bias folded into kernel; finalize without re-adding it.
    return equi_linear_finalize(out, c10::nullopt, s.out_shape);
}

// ---------------------------------------------------------------------------
// ver_3 - EXPLICIT AVX2/FMA SIMD VECTORIZATION (float32 only).
//
// Each 16-element multivector splits into a lo half (blades 0..7) and a hi
// half (blades 8..15), one __m256 each, and the two halves are processed in
// separate passes. Per (o,i) channel pair and half, the couplings reduce to
// two load+FMA pairs: a packed diagonal weight vector on x, and a sparse
// off-diagonal weight vector on a permutation of x (zero lanes multiply
// away). Splitting the halves frees enough registers for a 4-output-channel
// x 2-batch micro-kernel per half: 8 accumulators + x0/x1 + their permutes
// = 12 of 16 YMM, nothing spills, each x load feeds 8 FMAs and each weight
// load feeds 2 FMAs (16 FMAs : 10 loads per inner iteration; the previous
// fused-halves kernel needed 20 registers and the compiler re-loaded x and
// shared weights from memory, ~26 load uops per 16 FMAs).
//
// Weights are packed once per distinct weight tensor into a process-wide
// cache keyed by data pointer + version counter: inference reuses the same
// weights every forward, so the pack runs once, not per call.
//
// Bias is folded into the lane-0 accumulator init of the lo half, and
// outputs too large to be cache-resident are written with non-temporal
// stores, skipping the read-for-ownership of the fresh destination pages.
//
// Falls back to scalar ver_2 for float64 or non-AVX2 builds.
// ---------------------------------------------------------------------------
#if EZGATR_HAVE_AVX2
namespace {

// Packed split-half weights for one weight tensor. Layout per (o,i) pair and
// half: [diag(8) | off(8)], lo halves in row 0, hi halves in row 1.
struct ElPacked {
    torch::Tensor src;     // strong ref pins the storage so the key stays valid
    uint32_t version;
    bool normalized;
    torch::Tensor packed;  // float32 [2][n_oi*16]
};

std::mutex g_el_pack_mu;
std::map<const void*, ElPacked> g_el_pack_cache;

void el_pack_split_f32(const float* __restrict__ wsrc, int64_t n_oi,
                       bool normalize, float* __restrict__ WL,
                       float* __restrict__ WH) {
    float inv[9];
    for (int k = 0; k < 9; ++k)
        inv[k] = normalize ? float(EQUI_LINEAR_INV_NORMS[k]) : 1.0f;
    for (int64_t p = 0; p < n_oi; ++p) {
        const float* __restrict__ ws = wsrc + p * 9;
        const float w0=ws[0]*inv[0], w1=ws[1]*inv[1], w2=ws[2]*inv[2];
        const float w3=ws[3]*inv[3], w4=ws[4]*inv[4], w5=ws[5]*inv[5];
        const float w6=ws[6]*inv[6], w7=ws[7]*inv[7], w8=ws[8]*inv[8];
        float* __restrict__ dl = WL + p * 16;
        float* __restrict__ dh = WH + p * 16;
        // lo diag {w0, w1,w1,w1,w1, w2,w2,w2}; lo off blade1<-w5, blades5,6,7<-w6
        dl[0]=w0; dl[1]=w1; dl[2]=w1;  dl[3]=w1;  dl[4]=w1;  dl[5]=w2;  dl[6]=w2;  dl[7]=w2;
        dl[8]=0;  dl[9]=w5; dl[10]=0;  dl[11]=0;  dl[12]=0;  dl[13]=w6; dl[14]=w6; dl[15]=w6;
        // hi diag {w2,w2,w2, w3,w3,w3,w3, w4}; hi off blades11,12,13<-w7, blade15<-w8
        dh[0]=w2; dh[1]=w2; dh[2]=w2;  dh[3]=w3;  dh[4]=w3;  dh[5]=w3;  dh[6]=w3;  dh[7]=w4;
        dh[8]=0;  dh[9]=0;  dh[10]=0;  dh[11]=w7; dh[12]=w7; dh[13]=w7; dh[14]=0;  dh[15]=w8;
    }
}

[[maybe_unused]] torch::Tensor el_get_packed_f32(const torch::Tensor& weight,
                                const torch::Tensor& wf,
                                int64_t n_oi, bool normalize) {
    // wf is a fresh temp when weight is non-contiguous; only cache stable ptrs.
    const bool cacheable = weight.is_contiguous();
    const void* key = wf.data_ptr();
    // Inference tensors (torch.inference_mode) carry no version counter — by
    // design they are treated as immutable, so a fixed version is safe. The
    // strong `src` ref in the cache entry pins the storage, so a data_ptr hit
    // can never be a recycled allocation.
    auto* impl = wf.unsafeGetTensorImpl();
    const uint32_t version =
        impl->is_inference() ? 0 : impl->version_counter().current_version();

    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_pack_mu);
        auto it = g_el_pack_cache.find(key);
        if (it != g_el_pack_cache.end() &&
            it->second.version == version &&
            it->second.normalized == normalize &&
            it->second.packed.size(1) == n_oi * 16) {
            return it->second.packed;
        }
    }

    auto packed = torch::empty({2, n_oi * 16}, wf.options());
    el_pack_split_f32(wf.data_ptr<float>(), n_oi, normalize,
                      packed.data_ptr<float>(),
                      packed.data_ptr<float>() + n_oi * 16);

    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_pack_mu);
        if (g_el_pack_cache.size() > 64) g_el_pack_cache.clear();
        g_el_pack_cache[key] = ElPacked{wf, version, normalize, packed};
    }
    return packed;
}

template <bool NT>
static inline void el_store(float* p, __m256 v) {
    if constexpr (NT) _mm256_stream_ps(p, v);
    else              _mm256_storeu_ps(p, v);
}

// One half (xoff=0: lo, xoff=8: hi) of 4 output channels x 2 batch rows.
// bias4 carries the lane-0 bias for the lo half, nullptr otherwise.
// NOTE (2026-06-13): a 3ch x 2batch variant with separate diag/off
// accumulators (12 single-FMA chains, breaking the 2x5-cycle dependency per
// accumulator) measures +13% with plain stores, but LOSES under the
// non-temporal stores used for large outputs (WC-buffer behavior favors the
// 4-line group pattern) and under the 1-channel tails at out_ch=32. This
// 4x2 form is the verified optimum for the production store configuration.
template <bool NT>
static inline void el_half_4x2(
    const float* __restrict__ xb0, const float* __restrict__ xb1,
    const float* __restrict__ w0, const float* __restrict__ w1,
    const float* __restrict__ w2, const float* __restrict__ w3,
    float* __restrict__ ob0, float* __restrict__ ob1,
    int64_t in_ch, __m256i idx, int64_t xoff, const float* bias4)
{
    const __m256 z = _mm256_setzero_ps();
    __m256 a0b0=z, a1b0=z, a2b0=z, a3b0=z;
    __m256 a0b1=z, a1b1=z, a2b1=z, a3b1=z;
    if (bias4) {
        a0b0 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[0]), 1); a0b1 = a0b0;
        a1b0 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[1]), 1); a1b1 = a1b0;
        a2b0 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[2]), 1); a2b1 = a2b0;
        a3b0 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[3]), 1); a3b1 = a3b0;
    }
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m256 x0  = _mm256_loadu_ps(xb0 + i * 16 + xoff);
        const __m256 x1  = _mm256_loadu_ps(xb1 + i * 16 + xoff);
        const __m256 xp0 = _mm256_permutevar8x32_ps(x0, idx);
        const __m256 xp1 = _mm256_permutevar8x32_ps(x1, idx);
        { const __m256 wd = _mm256_loadu_ps(w0 + i*16);
          a0b0 = _mm256_fmadd_ps(wd, x0, a0b0); a0b1 = _mm256_fmadd_ps(wd, x1, a0b1); }
        { const __m256 wd = _mm256_loadu_ps(w1 + i*16);
          a1b0 = _mm256_fmadd_ps(wd, x0, a1b0); a1b1 = _mm256_fmadd_ps(wd, x1, a1b1); }
        { const __m256 wd = _mm256_loadu_ps(w2 + i*16);
          a2b0 = _mm256_fmadd_ps(wd, x0, a2b0); a2b1 = _mm256_fmadd_ps(wd, x1, a2b1); }
        { const __m256 wd = _mm256_loadu_ps(w3 + i*16);
          a3b0 = _mm256_fmadd_ps(wd, x0, a3b0); a3b1 = _mm256_fmadd_ps(wd, x1, a3b1); }
        { const __m256 wo = _mm256_loadu_ps(w0 + i*16 + 8);
          a0b0 = _mm256_fmadd_ps(wo, xp0, a0b0); a0b1 = _mm256_fmadd_ps(wo, xp1, a0b1); }
        { const __m256 wo = _mm256_loadu_ps(w1 + i*16 + 8);
          a1b0 = _mm256_fmadd_ps(wo, xp0, a1b0); a1b1 = _mm256_fmadd_ps(wo, xp1, a1b1); }
        { const __m256 wo = _mm256_loadu_ps(w2 + i*16 + 8);
          a2b0 = _mm256_fmadd_ps(wo, xp0, a2b0); a2b1 = _mm256_fmadd_ps(wo, xp1, a2b1); }
        { const __m256 wo = _mm256_loadu_ps(w3 + i*16 + 8);
          a3b0 = _mm256_fmadd_ps(wo, xp0, a3b0); a3b1 = _mm256_fmadd_ps(wo, xp1, a3b1); }
    }
    el_store<NT>(ob0 + 0*16 + xoff, a0b0); el_store<NT>(ob0 + 1*16 + xoff, a1b0);
    el_store<NT>(ob0 + 2*16 + xoff, a2b0); el_store<NT>(ob0 + 3*16 + xoff, a3b0);
    el_store<NT>(ob1 + 0*16 + xoff, a0b1); el_store<NT>(ob1 + 1*16 + xoff, a1b1);
    el_store<NT>(ob1 + 2*16 + xoff, a2b1); el_store<NT>(ob1 + 3*16 + xoff, a3b1);
}

// One half of a single output channel x 2 batch rows (channel tail).
template <bool NT>
static inline void el_half_1x2(
    const float* __restrict__ xb0, const float* __restrict__ xb1,
    const float* __restrict__ w0,
    float* __restrict__ ob0, float* __restrict__ ob1,
    int64_t in_ch, __m256i idx, int64_t xoff, const float* bias1)
{
    const __m256 z = _mm256_setzero_ps();
    __m256 ab0 = z, ab1 = z;
    if (bias1) {
        ab0 = _mm256_blend_ps(z, _mm256_set1_ps(*bias1), 1); ab1 = ab0;
    }
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m256 x0  = _mm256_loadu_ps(xb0 + i * 16 + xoff);
        const __m256 x1  = _mm256_loadu_ps(xb1 + i * 16 + xoff);
        const __m256 wd  = _mm256_loadu_ps(w0 + i*16);
        const __m256 wo  = _mm256_loadu_ps(w0 + i*16 + 8);
        ab0 = _mm256_fmadd_ps(wd, x0, ab0);
        ab1 = _mm256_fmadd_ps(wd, x1, ab1);
        ab0 = _mm256_fmadd_ps(wo, _mm256_permutevar8x32_ps(x0, idx), ab0);
        ab1 = _mm256_fmadd_ps(wo, _mm256_permutevar8x32_ps(x1, idx), ab1);
    }
    el_store<NT>(ob0 + xoff, ab0);
    el_store<NT>(ob1 + xoff, ab1);
}

// One half of 4 output channels x 1 batch row (odd-batch tail).
template <bool NT>
static inline void el_half_4x1(
    const float* __restrict__ xb0,
    const float* __restrict__ w0, const float* __restrict__ w1,
    const float* __restrict__ w2, const float* __restrict__ w3,
    float* __restrict__ ob0,
    int64_t in_ch, __m256i idx, int64_t xoff, const float* bias4)
{
    const __m256 z = _mm256_setzero_ps();
    __m256 a0=z, a1=z, a2=z, a3=z;
    if (bias4) {
        a0 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[0]), 1);
        a1 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[1]), 1);
        a2 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[2]), 1);
        a3 = _mm256_blend_ps(z, _mm256_set1_ps(bias4[3]), 1);
    }
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m256 x0  = _mm256_loadu_ps(xb0 + i * 16 + xoff);
        const __m256 xp0 = _mm256_permutevar8x32_ps(x0, idx);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i*16), x0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i*16), x0, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i*16), x0, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i*16), x0, a3);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i*16 + 8), xp0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i*16 + 8), xp0, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i*16 + 8), xp0, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i*16 + 8), xp0, a3);
    }
    el_store<NT>(ob0 + 0*16 + xoff, a0); el_store<NT>(ob0 + 1*16 + xoff, a1);
    el_store<NT>(ob0 + 2*16 + xoff, a2); el_store<NT>(ob0 + 3*16 + xoff, a3);
}

// One half of a single output channel x 1 batch row.
template <bool NT>
static inline void el_half_1x1(
    const float* __restrict__ xb0, const float* __restrict__ w0,
    float* __restrict__ ob0,
    int64_t in_ch, __m256i idx, int64_t xoff, const float* bias1)
{
    const __m256 z = _mm256_setzero_ps();
    __m256 a0 = bias1 ? _mm256_blend_ps(z, _mm256_set1_ps(*bias1), 1) : z;
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m256 x0 = _mm256_loadu_ps(xb0 + i * 16 + xoff);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i*16), x0, a0);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i*16 + 8),
                             _mm256_permutevar8x32_ps(x0, idx), a0);
    }
    el_store<NT>(ob0 + xoff, a0);
}

template <bool NT>
static void el_v3_kernel_f32(const float* __restrict__ X,
                             const float* __restrict__ WL,
                             const float* __restrict__ WH,
                             const float* __restrict__ bp,
                             float* __restrict__ O,
                             int64_t batch, int64_t in_ch, int64_t out_ch)
{
    // Off-diagonal gather indices. idx[j] is the source lane for output lane
    // j; lanes whose packed off-weight is 0 use idx=0 as a don't-care.
    //   lo: blade1<-x[0], blades5,6,7<-x[2,3,4]
    const __m256i idx_lo = _mm256_setr_epi32(0, 0, 0, 0, 0, 2, 3, 4);
    //   hi: blades11,12,13<-x[8,9,10], blade15<-x[14] (lane-relative 0,1,2,6)
    const __m256i idx_hi = _mm256_setr_epi32(0, 0, 0, 0, 1, 2, 0, 6);

    const int64_t n_quad = out_ch / 4;
    int64_t b = 0;
    for (; b + 1 < batch; b += 2) {
        const float* xb0 = X + b       * in_ch * 16;
        const float* xb1 = X + (b + 1) * in_ch * 16;
        float* ob0 = O + b       * out_ch * 16;
        float* ob1 = O + (b + 1) * out_ch * 16;
        for (int64_t g = 0; g < n_quad; ++g) {
            const int64_t o0 = g * 4;
            el_half_4x2<NT>(xb0, xb1,
                            WL + (o0 + 0) * in_ch * 16, WL + (o0 + 1) * in_ch * 16,
                            WL + (o0 + 2) * in_ch * 16, WL + (o0 + 3) * in_ch * 16,
                            ob0 + o0 * 16, ob1 + o0 * 16,
                            in_ch, idx_lo, 0, bp ? bp + o0 : nullptr);
            el_half_4x2<NT>(xb0, xb1,
                            WH + (o0 + 0) * in_ch * 16, WH + (o0 + 1) * in_ch * 16,
                            WH + (o0 + 2) * in_ch * 16, WH + (o0 + 3) * in_ch * 16,
                            ob0 + o0 * 16, ob1 + o0 * 16,
                            in_ch, idx_hi, 8, nullptr);
        }
        for (int64_t o = n_quad * 4; o < out_ch; ++o) {
            el_half_1x2<NT>(xb0, xb1, WL + o * in_ch * 16,
                            ob0 + o * 16, ob1 + o * 16,
                            in_ch, idx_lo, 0, bp ? bp + o : nullptr);
            el_half_1x2<NT>(xb0, xb1, WH + o * in_ch * 16,
                            ob0 + o * 16, ob1 + o * 16,
                            in_ch, idx_hi, 8, nullptr);
        }
    }
    for (; b < batch; ++b) {
        const float* xb0 = X + b * in_ch * 16;
        float* ob0 = O + b * out_ch * 16;
        for (int64_t g = 0; g < n_quad; ++g) {
            const int64_t o0 = g * 4;
            el_half_4x1<NT>(xb0,
                            WL + (o0 + 0) * in_ch * 16, WL + (o0 + 1) * in_ch * 16,
                            WL + (o0 + 2) * in_ch * 16, WL + (o0 + 3) * in_ch * 16,
                            ob0 + o0 * 16, in_ch, idx_lo, 0,
                            bp ? bp + o0 : nullptr);
            el_half_4x1<NT>(xb0,
                            WH + (o0 + 0) * in_ch * 16, WH + (o0 + 1) * in_ch * 16,
                            WH + (o0 + 2) * in_ch * 16, WH + (o0 + 3) * in_ch * 16,
                            ob0 + o0 * 16, in_ch, idx_hi, 8, nullptr);
        }
        for (int64_t o = n_quad * 4; o < out_ch; ++o) {
            el_half_1x1<NT>(xb0, WL + o * in_ch * 16, ob0 + o * 16,
                            in_ch, idx_lo, 0, bp ? bp + o : nullptr);
            el_half_1x1<NT>(xb0, WH + o * in_ch * 16, ob0 + o * 16,
                            in_ch, idx_hi, 8, nullptr);
        }
    }
    if constexpr (NT) _mm_sfence();
}

// ---------------------------------------------------------------------------
// AVX-512 equi_linear kernel — fuses the lo/hi half-passes into a single
// 16-wide pass using __m512 + _mm512_permutexvar_ps.
// Weight layout (32 floats per (o,i) pair):
//   W512[p*32+0..15]  = combined diagonal:   lo_diag[8] | hi_diag[8]
//   W512[p*32+16..31] = combined off-diag:   lo_off[8]  | hi_off[8]
// Combined permutation for the off-diagonal swap (absolute lane indices):
//   lo: {0,0,0,0,0,2,3,4}  (blade1←x0; blades5,6,7←x2,x3,x4)
//   hi: {8,8,8,8,9,10,8,14} (blades11,12,13←x8,x9,x10; blade15←x14)
// ---------------------------------------------------------------------------
#if defined(__AVX512F__)

struct ElPacked512 {
    torch::Tensor src;
    uint32_t version;
    bool normalized;
    torch::Tensor packed;   // float32 [n_oi * 32]
};

std::mutex g_el_pack512_mu;
std::map<const void*, ElPacked512> g_el_pack512_cache;

static void el_pack_avx512_f32(const float* __restrict__ wsrc, int64_t n_oi,
                                bool normalize, float* __restrict__ W512) {
    float inv[9];
    for (int k = 0; k < 9; ++k)
        inv[k] = normalize ? float(EQUI_LINEAR_INV_NORMS[k]) : 1.0f;
    for (int64_t p = 0; p < n_oi; ++p) {
        const float* ws = wsrc + p * 9;
        const float w0=ws[0]*inv[0], w1=ws[1]*inv[1], w2=ws[2]*inv[2];
        const float w3=ws[3]*inv[3], w4=ws[4]*inv[4], w5=ws[5]*inv[5];
        const float w6=ws[6]*inv[6], w7=ws[7]*inv[7], w8=ws[8]*inv[8];
        float* wd = W512 + p * 32;
        float* wo = wd + 16;
        // combined diagonal: lo_diag[8] | hi_diag[8]
        wd[0]=w0; wd[1]=w1;  wd[2]=w1;  wd[3]=w1;  wd[4]=w1;  wd[5]=w2;  wd[6]=w2;  wd[7]=w2;
        wd[8]=w2; wd[9]=w2;  wd[10]=w2; wd[11]=w3; wd[12]=w3; wd[13]=w3; wd[14]=w3; wd[15]=w4;
        // combined off-diagonal: lo_off[8] | hi_off[8]
        wo[0]=0;  wo[1]=w5;  wo[2]=0;   wo[3]=0;   wo[4]=0;   wo[5]=w6;  wo[6]=w6;  wo[7]=w6;
        wo[8]=0;  wo[9]=0;   wo[10]=0;  wo[11]=w7; wo[12]=w7; wo[13]=w7; wo[14]=0;  wo[15]=w8;
    }
}

static torch::Tensor el_get_packed_avx512_f32(const torch::Tensor& weight,
                                               const torch::Tensor& wf,
                                               int64_t n_oi, bool normalize) {
    const bool cacheable = weight.is_contiguous();
    const void* key = wf.data_ptr();
    auto* impl = wf.unsafeGetTensorImpl();
    const uint32_t version =
        impl->is_inference() ? 0 : impl->version_counter().current_version();
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_pack512_mu);
        auto it = g_el_pack512_cache.find(key);
        if (it != g_el_pack512_cache.end() &&
            it->second.version == version &&
            it->second.normalized == normalize &&
            it->second.packed.size(0) == n_oi * 32)
            return it->second.packed;
    }
    auto packed = torch::empty({n_oi * 32}, wf.options());
    el_pack_avx512_f32(wf.data_ptr<float>(), n_oi, normalize,
                       packed.data_ptr<float>());
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_el_pack512_mu);
        if (g_el_pack512_cache.size() > 64) g_el_pack512_cache.clear();
        g_el_pack512_cache[key] = ElPacked512{wf, version, normalize, packed};
    }
    return packed;
}

// Store helper: NT path requires 64-byte alignment (caller must ensure).
template <bool NT>
static inline void el_store_512(float* p, __m512 v) {
    if constexpr (NT) _mm512_stream_ps(p, v);
    else              _mm512_storeu_ps(p, v);
}

// 4 output channels × 2 batch rows, full 16-float multivector at once.
// W0..W3 point to the start of each output channel's packed weights (32f/pair).
template <bool NT>
static inline void el_half_4x2_avx512(
    const float* __restrict__ xb0, const float* __restrict__ xb1,
    const float* __restrict__ w0,  const float* __restrict__ w1,
    const float* __restrict__ w2,  const float* __restrict__ w3,
    float* __restrict__ ob0, float* __restrict__ ob1,
    int64_t in_ch, __m512i idx, const float* bias4)
{
    const __m512 z = _mm512_setzero_ps();
    __m512 a0b0=z, a1b0=z, a2b0=z, a3b0=z;
    __m512 a0b1=z, a1b1=z, a2b1=z, a3b1=z;
    if (bias4) {
        a0b0 = _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[0])); a0b1 = a0b0;
        a1b0 = _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[1])); a1b1 = a1b0;
        a2b0 = _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[2])); a2b1 = a2b0;
        a3b0 = _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[3])); a3b1 = a3b0;
    }
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m512 x0  = _mm512_loadu_ps(xb0 + i * 16);
        const __m512 x1  = _mm512_loadu_ps(xb1 + i * 16);
        const __m512 xp0 = _mm512_permutexvar_ps(idx, x0);
        const __m512 xp1 = _mm512_permutexvar_ps(idx, x1);
        { const __m512 wd = _mm512_loadu_ps(w0 + i*32);
          const __m512 wo = _mm512_loadu_ps(w0 + i*32 + 16);
          a0b0 = _mm512_fmadd_ps(wd, x0,  a0b0); a0b1 = _mm512_fmadd_ps(wd, x1,  a0b1);
          a0b0 = _mm512_fmadd_ps(wo, xp0, a0b0); a0b1 = _mm512_fmadd_ps(wo, xp1, a0b1); }
        { const __m512 wd = _mm512_loadu_ps(w1 + i*32);
          const __m512 wo = _mm512_loadu_ps(w1 + i*32 + 16);
          a1b0 = _mm512_fmadd_ps(wd, x0,  a1b0); a1b1 = _mm512_fmadd_ps(wd, x1,  a1b1);
          a1b0 = _mm512_fmadd_ps(wo, xp0, a1b0); a1b1 = _mm512_fmadd_ps(wo, xp1, a1b1); }
        { const __m512 wd = _mm512_loadu_ps(w2 + i*32);
          const __m512 wo = _mm512_loadu_ps(w2 + i*32 + 16);
          a2b0 = _mm512_fmadd_ps(wd, x0,  a2b0); a2b1 = _mm512_fmadd_ps(wd, x1,  a2b1);
          a2b0 = _mm512_fmadd_ps(wo, xp0, a2b0); a2b1 = _mm512_fmadd_ps(wo, xp1, a2b1); }
        { const __m512 wd = _mm512_loadu_ps(w3 + i*32);
          const __m512 wo = _mm512_loadu_ps(w3 + i*32 + 16);
          a3b0 = _mm512_fmadd_ps(wd, x0,  a3b0); a3b1 = _mm512_fmadd_ps(wd, x1,  a3b1);
          a3b0 = _mm512_fmadd_ps(wo, xp0, a3b0); a3b1 = _mm512_fmadd_ps(wo, xp1, a3b1); }
    }
    el_store_512<NT>(ob0 + 0*16, a0b0); el_store_512<NT>(ob0 + 1*16, a1b0);
    el_store_512<NT>(ob0 + 2*16, a2b0); el_store_512<NT>(ob0 + 3*16, a3b0);
    el_store_512<NT>(ob1 + 0*16, a0b1); el_store_512<NT>(ob1 + 1*16, a1b1);
    el_store_512<NT>(ob1 + 2*16, a2b1); el_store_512<NT>(ob1 + 3*16, a3b1);
}

template <bool NT>
static inline void el_half_1x2_avx512(
    const float* __restrict__ xb0, const float* __restrict__ xb1,
    const float* __restrict__ w0,
    float* __restrict__ ob0, float* __restrict__ ob1,
    int64_t in_ch, __m512i idx, const float* bias1)
{
    const __m512 z = _mm512_setzero_ps();
    __m512 ab0 = bias1 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(*bias1)) : z;
    __m512 ab1 = ab0;
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m512 x0  = _mm512_loadu_ps(xb0 + i * 16);
        const __m512 x1  = _mm512_loadu_ps(xb1 + i * 16);
        const __m512 wd  = _mm512_loadu_ps(w0 + i*32);
        const __m512 wo  = _mm512_loadu_ps(w0 + i*32 + 16);
        ab0 = _mm512_fmadd_ps(wd, x0, ab0);
        ab1 = _mm512_fmadd_ps(wd, x1, ab1);
        ab0 = _mm512_fmadd_ps(wo, _mm512_permutexvar_ps(idx, x0), ab0);
        ab1 = _mm512_fmadd_ps(wo, _mm512_permutexvar_ps(idx, x1), ab1);
    }
    el_store_512<NT>(ob0, ab0);
    el_store_512<NT>(ob1, ab1);
}

template <bool NT>
static inline void el_half_4x1_avx512(
    const float* __restrict__ xb0,
    const float* __restrict__ w0, const float* __restrict__ w1,
    const float* __restrict__ w2, const float* __restrict__ w3,
    float* __restrict__ ob0,
    int64_t in_ch, __m512i idx, const float* bias4)
{
    const __m512 z = _mm512_setzero_ps();
    __m512 a0 = bias4 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[0])) : z;
    __m512 a1 = bias4 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[1])) : z;
    __m512 a2 = bias4 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[2])) : z;
    __m512 a3 = bias4 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(bias4[3])) : z;
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m512 x0  = _mm512_loadu_ps(xb0 + i * 16);
        const __m512 xp0 = _mm512_permutexvar_ps(idx, x0);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i*32), x0, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i*32), x0, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i*32), x0, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i*32), x0, a3);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i*32 + 16), xp0, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i*32 + 16), xp0, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i*32 + 16), xp0, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i*32 + 16), xp0, a3);
    }
    el_store_512<NT>(ob0 + 0*16, a0); el_store_512<NT>(ob0 + 1*16, a1);
    el_store_512<NT>(ob0 + 2*16, a2); el_store_512<NT>(ob0 + 3*16, a3);
}

template <bool NT>
static inline void el_half_1x1_avx512(
    const float* __restrict__ xb0, const float* __restrict__ w0,
    float* __restrict__ ob0,
    int64_t in_ch, __m512i idx, const float* bias1)
{
    const __m512 z = _mm512_setzero_ps();
    __m512 a0 = bias1 ? _mm512_maskz_broadcastss_ps(0x0001u, _mm_set_ss(*bias1)) : z;
    for (int64_t i = 0; i < in_ch; ++i) {
        const __m512 x0 = _mm512_loadu_ps(xb0 + i * 16);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i*32), x0, a0);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i*32 + 16),
                             _mm512_permutexvar_ps(idx, x0), a0);
    }
    el_store_512<NT>(ob0, a0);
}

template <bool NT>
static void el_v3_kernel_f32_avx512(const float* __restrict__ X,
                                     const float* __restrict__ W512,
                                     const float* __restrict__ bp,
                                     float* __restrict__ O,
                                     int64_t batch, int64_t in_ch, int64_t out_ch)
{
    // Combined off-diagonal permutation: lo {0,0,0,0,0,2,3,4} | hi {8,8,8,8,9,10,8,14}
    const __m512i idx = _mm512_setr_epi32(0,0,0,0,0,2,3,4, 8,8,8,8,9,10,8,14);

    const int64_t n_quad = out_ch / 4;
    int64_t b = 0;
    for (; b + 1 < batch; b += 2) {
        const float* xb0 = X + b       * in_ch * 16;
        const float* xb1 = X + (b + 1) * in_ch * 16;
        float* ob0 = O + b       * out_ch * 16;
        float* ob1 = O + (b + 1) * out_ch * 16;
        for (int64_t g = 0; g < n_quad; ++g) {
            const int64_t o0 = g * 4;
            el_half_4x2_avx512<NT>(xb0, xb1,
                W512 + (o0 + 0) * in_ch * 32,
                W512 + (o0 + 1) * in_ch * 32,
                W512 + (o0 + 2) * in_ch * 32,
                W512 + (o0 + 3) * in_ch * 32,
                ob0 + o0 * 16, ob1 + o0 * 16,
                in_ch, idx, bp ? bp + o0 : nullptr);
        }
        for (int64_t o = n_quad * 4; o < out_ch; ++o) {
            el_half_1x2_avx512<NT>(xb0, xb1, W512 + o * in_ch * 32,
                ob0 + o * 16, ob1 + o * 16,
                in_ch, idx, bp ? bp + o : nullptr);
        }
    }
    for (; b < batch; ++b) {
        const float* xb0 = X + b * in_ch * 16;
        float* ob0 = O + b * out_ch * 16;
        for (int64_t g = 0; g < n_quad; ++g) {
            const int64_t o0 = g * 4;
            el_half_4x1_avx512<NT>(xb0,
                W512 + (o0 + 0) * in_ch * 32,
                W512 + (o0 + 1) * in_ch * 32,
                W512 + (o0 + 2) * in_ch * 32,
                W512 + (o0 + 3) * in_ch * 32,
                ob0 + o0 * 16, in_ch, idx, bp ? bp + o0 : nullptr);
        }
        for (int64_t o = n_quad * 4; o < out_ch; ++o) {
            el_half_1x1_avx512<NT>(xb0, W512 + o * in_ch * 32, ob0 + o * 16,
                in_ch, idx, bp ? bp + o : nullptr);
        }
    }
    if constexpr (NT) _mm_sfence();
}

#endif  // __AVX512F__

}  // namespace
#endif  // EZGATR_HAVE_AVX2
torch::Tensor equi_linear_v3(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
#if EZGATR_HAVE_AVX2
    if (x.scalar_type() == torch::kFloat) {
        auto s = equi_linear_prepare(x, weight);
        auto out = torch::empty({s.batch, s.out_ch, 16}, x.options());

        const float* __restrict__ bp = nullptr;
        torch::Tensor bias_t;
        if (bias.has_value()) {
            bias_t = bias.value().contiguous();
            bp = bias_t.data_ptr<float>();
        }

        const int64_t in_ch  = s.in_ch;
        const int64_t out_ch = s.out_ch;
        const int64_t batch  = s.batch;
        const int64_t n_oi   = out_ch * in_ch;

        const float* __restrict__ xp_base = s.xf.data_ptr<float>();
        float*       __restrict__ op_base = out.data_ptr<float>();

        const int64_t out_bytes =
            batch * out_ch * 16 * static_cast<int64_t>(sizeof(float));

#if defined(__AVX512F__)
        {
            auto packed512 = el_get_packed_avx512_f32(weight, s.wf, n_oi, normalize_basis);
            const float* __restrict__ W512 = packed512.data_ptr<float>();
            // NT stores require 64-byte alignment.
            const bool use_nt =
                out_bytes >= (16 << 20) &&
                reinterpret_cast<uintptr_t>(op_base) % 64 == 0;
            if (use_nt)
                el_v3_kernel_f32_avx512<true>(xp_base, W512, bp, op_base,
                                               batch, in_ch, out_ch);
            else
                el_v3_kernel_f32_avx512<false>(xp_base, W512, bp, op_base,
                                                batch, in_ch, out_ch);
        }
#else
        {
            auto packed = el_get_packed_f32(weight, s.wf, n_oi, normalize_basis);
            const float* __restrict__ WL = packed.data_ptr<float>();
            const float* __restrict__ WH = WL + n_oi * 16;
            // Non-temporal stores only pay off when the output cannot stay
            // cache-resident (16 MB = one CCX worth of L3).
            const bool use_nt =
                out_bytes >= (16 << 20) &&
                reinterpret_cast<uintptr_t>(op_base) % 32 == 0;
            if (use_nt)
                el_v3_kernel_f32<true>(xp_base, WL, WH, bp, op_base,
                                       batch, in_ch, out_ch);
            else
                el_v3_kernel_f32<false>(xp_base, WL, WH, bp, op_base,
                                        batch, in_ch, out_ch);
        }
#endif  // __AVX512F__

        // Bias was folded into the accumulator init; finalize without it.
        return equi_linear_finalize(out, c10::nullopt, s.out_shape);
    }
#endif
    // float64 / no-AVX2 build: scalar pre-SIMD kernel (numerically identical).
    return equi_linear_v2(x, weight, bias, normalize_basis);
}

// Dispatcher. version: 0=no-opt, 1=math, 2=pre-SIMD, 3=SIMD.
torch::Tensor equi_linear(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const c10::optional<torch::Tensor>& bias,
                          bool normalize_basis,
                          int64_t version) {
    switch (version) {
        case 0: return equi_linear_v0(x, weight, bias, normalize_basis);
        case 1: return equi_linear_v1(x, weight, bias, normalize_basis);
        case 2: return equi_linear_v2(x, weight, bias, normalize_basis);
        case 3: return equi_linear_v3(x, weight, bias, normalize_basis);
        default:
            TORCH_CHECK(false, "equi_linear: unknown version ", version,
                        " (expected 0, 1, 2, or 3)");
    }
}

torch::Tensor geometric_product_v2_5(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_5",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_5<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v2_6(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_6",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_6<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v2_7(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v2_7",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v2_7<T>(X, Y, O, N);
        });
}

torch::Tensor geometric_product_v3(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v3",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v3<T>(X, Y, O, N);
        });
}

torch::Tensor equi_join_v2_5(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_5",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_5<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_5<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v2_6(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_6",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_6<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_6<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v2_7(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v2_7",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v2_7<T, true>(X, Y, R, O, N);
            else         join_kernel_v2_7<T, false>(X, Y, nullptr, O, N);
        });
}

torch::Tensor equi_join_v3(const torch::Tensor& x,
                           const torch::Tensor& y,
                           const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v3",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v3<T, true>(X, Y, R, O, N);
            else         join_kernel_v3<T, false>(X, Y, nullptr, O, N);
        });
}

// Fused geometric bilinear (v3_1). Takes the proj_bil output p of shape
// (..., 4*inter, 16) and computes cat([gp(lg,rg), join(lj,rj,ref)], dim=-2) in a
// single kernel, writing both halves directly into one (..., 2*inter, 16) output.
// This eliminates the torch.cat that the separate-op path pays (a full read-2x /
// write-2x pass over the bilinear intermediate) plus the second launch and two
// temporaries. The per-output-row layout keeps gp's and join's outputs contiguous
// within a row, so the existing gp_kernel_v3 / join_kernel_v3 are reused verbatim.
// join_kernel_v3 only consumes blade 14 of the reference as packed [inter] scalars,
// so we select(-1, 14) instead of materializing the full (inter, 16) view.
torch::Tensor geometric_bilinear_v3_1(const torch::Tensor& p,
                                      const c10::optional<torch::Tensor>& reference) {
    const char* name = "geometric_bilinear_v3_1";
    check_multivector(p, name);
    TORCH_CHECK(p.device().is_cpu(), name, ": CPU-only kernel; got device ", p.device());
    TORCH_CHECK(p.scalar_type() == torch::kFloat, name,
                ": requires float32 inputs (fp32/AVX2-only kernel).");
    const int64_t four_inter = p.size(-2);
    TORCH_CHECK(four_inter % 4 == 0, name,
                ": channel dim (", four_inter, ") must be divisible by 4");
    const int64_t inter = four_inter / 4;

    const bool has_ref = reference.has_value();
    if (has_ref) {
        check_multivector(*reference, name);
        TORCH_CHECK(reference->scalar_type() == p.scalar_type(),
                    name, ": reference must share dtype with p");
        TORCH_CHECK(reference->device() == p.device(),
                    name, ": reference must share device with p");
    }

    auto pc = p.contiguous();
    const int64_t M = pc.numel() / (four_inter * 16);

    // Output: same shape as p but with channel dim 2*inter.
    auto out_shape = pc.sizes().vec();
    out_shape[out_shape.size() - 2] = 2 * inter;
    auto out = torch::empty(out_shape, pc.options());

    // join_kernel_v3 only reads blade 14 of the reference; extract those scalars
    // (one per join input row) instead of materializing the full (inter, 16) view.
    torch::Tensor ref14;
    if (has_ref) {
        auto join_shape = pc.sizes().vec();
        join_shape[join_shape.size() - 2] = inter;
        ref14 = reference->expand(join_shape).select(-1, 14).contiguous();  // (..., inter)
    }

    const float* __restrict__ P = pc.data_ptr<float>();
    float*       __restrict__ O = out.data_ptr<float>();
    const float* __restrict__ R = has_ref ? ref14.data_ptr<float>() : nullptr;

    for (int64_t m = 0; m < M; ++m) {
        const float* prow = P + m * 4 * inter * 16;
        float*       orow = O + m * 2 * inter * 16;
        gp_kernel_v3<float>(prow, prow + inter * 16, orow, inter);
        if (has_ref) {
            join_kernel_v3<float, true>(prow + 2 * inter * 16, prow + 3 * inter * 16,
                                        R + m * inter, orow + inter * 16, inter);
        } else {
            join_kernel_v3<float, false>(prow + 2 * inter * 16, prow + 3 * inter * 16,
                                         nullptr, orow + inter * 16, inter);
        }
    }
    return out;
}

}}  // namespace ezgatr::opt
