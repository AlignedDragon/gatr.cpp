#include "pga_ops.h"
#include "basis_data.h"

#include <ATen/Dispatch.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace ezgatr { namespace opt {

namespace {

#include "gp_unrolled.inc"
#include "join_unrolled.inc"
#include "gp_block_ilp2.inc"
#include "gp_block_ilp4.inc"
#include "join_block_ilp2.inc"
#include "join_block_ilp4.inc"
#include "gp_unrolled_acc2.inc"
#include "gp_unrolled_acc4.inc"
#include "join_unrolled_acc2.inc"
#include "join_unrolled_acc4.inc"
#include "gp_loop_unroll2.inc"
#include "gp_loop_unroll4.inc"
#include "join_loop_unroll2.inc"
#include "join_loop_unroll4.inc"
#include "gp_soa_avx.inc"
#include "join_soa_avx.inc"
#include "gp_soa_auto.inc"
#include "join_soa_auto.inc"

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
        const T s = HasRef ? R[16 * n + 14] : T(1);
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
            const T s = R[16 * n + 14];
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
            const T s = R[16 * n + 14];
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
            const T s = R[16 * n + 14];
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
            const T s = R[16 * n + 14];
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
            const T s = R[16 * n + 14];
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
            const T s = R[16 * n + 14];
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
                const T s = R[16 * n + 14];
                for (int i = 0; i < 16; ++i) o[i] *= s;
            } else {
                (void)R;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3 — vectorization across multivectors via SoA transpose + AVX2 (fp64).
// Batch-size-agnostic: flattens to N multivectors, processes 4 at a time
// (one __m256d lane each) with a scalar remainder for N % 4. The 4-wide block
// is an internal SIMD detail, independent of the caller's (dynamic) batch
// shape. Falls back to the scalar v2 kernel for non-double dtypes or when the
// build target lacks AVX2/FMA.
// ---------------------------------------------------------------------------

#if defined(__AVX2__) && defined(__FMA__)
// AoS -> SoA: 4 consecutive multivectors (rows of 16 contiguous doubles) become
// sb[16][4], where sb[j] holds component j across the 4 lanes. Done as four 4x4
// __m256d transposes (vs a 128-element scalar copy) so the transpose tax stays
// small relative to the per-tile FMA work.
static inline void soa_transpose_in_f64(const double* __restrict__ S, int64_t n,
                                        double sb[16][4]) {
    const double* r0 = S + 16 * (n + 0);
    const double* r1 = S + 16 * (n + 1);
    const double* r2 = S + 16 * (n + 2);
    const double* r3 = S + 16 * (n + 3);
    for (int b = 0; b < 4; ++b) {
        __m256d v0 = _mm256_loadu_pd(r0 + 4 * b);
        __m256d v1 = _mm256_loadu_pd(r1 + 4 * b);
        __m256d v2 = _mm256_loadu_pd(r2 + 4 * b);
        __m256d v3 = _mm256_loadu_pd(r3 + 4 * b);
        __m256d t0 = _mm256_unpacklo_pd(v0, v1);
        __m256d t1 = _mm256_unpackhi_pd(v0, v1);
        __m256d t2 = _mm256_unpacklo_pd(v2, v3);
        __m256d t3 = _mm256_unpackhi_pd(v2, v3);
        _mm256_storeu_pd(sb[4 * b + 0], _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_storeu_pd(sb[4 * b + 1], _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_storeu_pd(sb[4 * b + 2], _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_storeu_pd(sb[4 * b + 3], _mm256_permute2f128_pd(t1, t3, 0x31));
    }
}

// SoA -> AoS: ob[16][4] (component-major) back to 4 contiguous output rows.
// When Scale is true, each lane's row is multiplied by its reference scalar s[l]
// (the equi_join reference factor) before storing.
template <bool Scale>
static inline void soa_transpose_out_f64(const double ob[16][4], int64_t n,
                                         double* __restrict__ O,
                                         const double s[4]) {
    __m256d sv0, sv1, sv2, sv3;
    if constexpr (Scale) {
        sv0 = _mm256_set1_pd(s[0]); sv1 = _mm256_set1_pd(s[1]);
        sv2 = _mm256_set1_pd(s[2]); sv3 = _mm256_set1_pd(s[3]);
    }
    double* r0 = O + 16 * (n + 0);
    double* r1 = O + 16 * (n + 1);
    double* r2 = O + 16 * (n + 2);
    double* r3 = O + 16 * (n + 3);
    for (int b = 0; b < 4; ++b) {
        __m256d c0 = _mm256_loadu_pd(ob[4 * b + 0]);
        __m256d c1 = _mm256_loadu_pd(ob[4 * b + 1]);
        __m256d c2 = _mm256_loadu_pd(ob[4 * b + 2]);
        __m256d c3 = _mm256_loadu_pd(ob[4 * b + 3]);
        __m256d t0 = _mm256_unpacklo_pd(c0, c1);
        __m256d t1 = _mm256_unpackhi_pd(c0, c1);
        __m256d t2 = _mm256_unpacklo_pd(c2, c3);
        __m256d t3 = _mm256_unpackhi_pd(c2, c3);
        __m256d o0 = _mm256_permute2f128_pd(t0, t2, 0x20);
        __m256d o1 = _mm256_permute2f128_pd(t1, t3, 0x20);
        __m256d o2 = _mm256_permute2f128_pd(t0, t2, 0x31);
        __m256d o3 = _mm256_permute2f128_pd(t1, t3, 0x31);
        if constexpr (Scale) {
            o0 = _mm256_mul_pd(o0, sv0); o1 = _mm256_mul_pd(o1, sv1);
            o2 = _mm256_mul_pd(o2, sv2); o3 = _mm256_mul_pd(o3, sv3);
        }
        _mm256_storeu_pd(r0 + 4 * b, o0);
        _mm256_storeu_pd(r1 + 4 * b, o1);
        _mm256_storeu_pd(r2 + 4 * b, o2);
        _mm256_storeu_pd(r3 + 4 * b, o3);
    }
}
#endif  // __AVX2__ && __FMA__

template <typename T>
static void gp_kernel_v3(const T* __restrict__ X, const T* __restrict__ Y,
                         T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, double>) {
        int64_t n = 0;
        for (; n + 4 <= N; n += 4) {
            alignas(32) double xb[16][4], yb[16][4], ob[16][4];
            soa_transpose_in_f64(X, n, xb);
            soa_transpose_in_f64(Y, n, yb);
            gp_soa_block_f64(xb, yb, ob);
            soa_transpose_out_f64<false>(ob, n, O, nullptr);
        }
        for (; n < N; ++n) {
            gp_block_ilp2<double>(X + 16 * n, Y + 16 * n, O + 16 * n);
        }
        return;
    }
#endif
    // v3 is a float64 + AVX2/FMA-only kernel: no scalar/fp32 fallback by design.
    TORCH_CHECK(false, "geometric_product_v3: requires float64 inputs and an "
                       "AVX2/FMA build; got an unsupported dtype or target.");
}

template <typename T, bool HasRef>
static void join_kernel_v3(const T* __restrict__ X, const T* __restrict__ Y,
                           const T* __restrict__ R, T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, double>) {
        int64_t n = 0;
        for (; n + 4 <= N; n += 4) {
            alignas(32) double xb[16][4], yb[16][4], ob[16][4];
            soa_transpose_in_f64(X, n, xb);
            soa_transpose_in_f64(Y, n, yb);
            join_soa_block_f64(xb, yb, ob);
            if constexpr (HasRef) {
                const double s[4] = {R[16 * (n + 0) + 14], R[16 * (n + 1) + 14],
                                     R[16 * (n + 2) + 14], R[16 * (n + 3) + 14]};
                soa_transpose_out_f64<true>(ob, n, O, s);
            } else {
                soa_transpose_out_f64<false>(ob, n, O, nullptr);
            }
        }
        for (; n < N; ++n) {
            T* o = O + 16 * n;
            join_block_ilp2<double>(X + 16 * n, Y + 16 * n, o);
            if constexpr (HasRef) {
                const double s = R[16 * n + 14];
                for (int i = 0; i < 16; ++i) o[i] *= s;
            }
        }
        return;
    }
#endif
    (void)X; (void)Y; (void)R; (void)O; (void)N;
    // v3 is a float64 + AVX2/FMA-only kernel: no scalar/fp32 fallback by design.
    TORCH_CHECK(false, "equi_join_v3: requires float64 inputs and an "
                       "AVX2/FMA build; got an unsupported dtype or target.");
}

// ---------------------------------------------------------------------------
// v3_1 — same SoA-across-multivectors idea as v3, but written as portable
// plain-C++ loops (no intrinsics) and left to the compiler's autovectorizer.
// Dtype-generic; block width BW chosen per dtype. Used to compare compiler
// auto-vectorization against the hand-written AVX2 intrinsics in v3.
// ---------------------------------------------------------------------------

template <typename T>
struct soa_block_width { static constexpr int value = 4; };  // fp64: one __m256d
template <>
struct soa_block_width<float> { static constexpr int value = 8; };  // fp32: one __m256

template <typename T>
static void gp_kernel_v3_1(const T* __restrict__ X, const T* __restrict__ Y,
                           T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, double>) {
        // Vectorized transpose (as in v3) + auto-vectorized compute block, to
        // isolate how well the compiler vectorizes the FMA work on its own.
        int64_t n = 0;
        for (; n + 4 <= N; n += 4) {
            alignas(32) double xb[16][4], yb[16][4], ob[16][4];
            soa_transpose_in_f64(X, n, xb);
            soa_transpose_in_f64(Y, n, yb);
            gp_soa_block_auto<double, 4>(xb, yb, ob);
            soa_transpose_out_f64<false>(ob, n, O, nullptr);
        }
        for (; n < N; ++n) {
            gp_block_ilp2<double>(X + 16 * n, Y + 16 * n, O + 16 * n);
        }
        return;
    }
#endif
    constexpr int BW = soa_block_width<T>::value;
    int64_t n = 0;
    for (; n + BW <= N; n += BW) {
        T xb[16][BW], yb[16][BW], ob[16][BW];
        for (int l = 0; l < BW; ++l) {
            const T* x = X + 16 * (n + l);
            const T* y = Y + 16 * (n + l);
            for (int j = 0; j < 16; ++j) { xb[j][l] = x[j]; yb[j][l] = y[j]; }
        }
        gp_soa_block_auto<T, BW>(xb, yb, ob);
        for (int l = 0; l < BW; ++l) {
            T* o = O + 16 * (n + l);
            for (int i = 0; i < 16; ++i) o[i] = ob[i][l];
        }
    }
    for (; n < N; ++n) {
        gp_block_ilp2<T>(X + 16 * n, Y + 16 * n, O + 16 * n);
    }
}

template <typename T, bool HasRef>
static void join_kernel_v3_1(const T* __restrict__ X, const T* __restrict__ Y,
                             const T* __restrict__ R, T* __restrict__ O, int64_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, double>) {
        int64_t n = 0;
        for (; n + 4 <= N; n += 4) {
            alignas(32) double xb[16][4], yb[16][4], ob[16][4];
            soa_transpose_in_f64(X, n, xb);
            soa_transpose_in_f64(Y, n, yb);
            join_soa_block_auto<double, 4>(xb, yb, ob);
            if constexpr (HasRef) {
                const double s[4] = {R[16 * (n + 0) + 14], R[16 * (n + 1) + 14],
                                     R[16 * (n + 2) + 14], R[16 * (n + 3) + 14]};
                soa_transpose_out_f64<true>(ob, n, O, s);
            } else {
                soa_transpose_out_f64<false>(ob, n, O, nullptr);
            }
        }
        for (; n < N; ++n) {
            T* o = O + 16 * n;
            join_block_ilp2<double>(X + 16 * n, Y + 16 * n, o);
            if constexpr (HasRef) {
                const double s = R[16 * n + 14];
                for (int i = 0; i < 16; ++i) o[i] *= s;
            }
        }
        return;
    }
#endif
    constexpr int BW = soa_block_width<T>::value;
    int64_t n = 0;
    for (; n + BW <= N; n += BW) {
        T xb[16][BW], yb[16][BW], ob[16][BW];
        for (int l = 0; l < BW; ++l) {
            const T* x = X + 16 * (n + l);
            const T* y = Y + 16 * (n + l);
            for (int j = 0; j < 16; ++j) { xb[j][l] = x[j]; yb[j][l] = y[j]; }
        }
        join_soa_block_auto<T, BW>(xb, yb, ob);
        for (int l = 0; l < BW; ++l) {
            T* o = O + 16 * (n + l);
            if constexpr (HasRef) {
                const T s = R[16 * (n + l) + 14];
                for (int i = 0; i < 16; ++i) o[i] = ob[i][l] * s;
            } else {
                for (int i = 0; i < 16; ++i) o[i] = ob[i][l];
            }
        }
    }
    for (; n < N; ++n) {
        T* o = O + 16 * n;
        join_block_ilp2<T>(X + 16 * n, Y + 16 * n, o);
        if constexpr (HasRef) {
            const T s = R[16 * n + 14];
            for (int i = 0; i < 16; ++i) o[i] *= s;
        } else {
            (void)R;
        }
    }
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

    torch::Tensor refc;
    if (reference.has_value()) {
        refc = reference->expand_as(xc).contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "equi_join_cpu", [&]{
        if (reference.has_value()) {
            join_kernel_v2<scalar_t, true>(
                xc.data_ptr<scalar_t>(),
                yc.data_ptr<scalar_t>(),
                refc.data_ptr<scalar_t>(),
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

    torch::Tensor refc;
    if (reference.has_value()) {
        refc = reference->expand_as(xc).contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "join_variant_cpu", [&]{
        const scalar_t* R = reference.has_value() ? refc.data_ptr<scalar_t>() : nullptr;
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

torch::Tensor geometric_product_v3_1(const torch::Tensor& x, const torch::Tensor& y) {
    return run_gp_variant(x, y, "geometric_product_v3_1",
        [](auto X, auto Y, auto O, int64_t N){
            using T = std::remove_pointer_t<decltype(O)>;
            gp_kernel_v3_1<T>(X, Y, O, N);
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

torch::Tensor equi_join_v3_1(const torch::Tensor& x,
                             const torch::Tensor& y,
                             const c10::optional<torch::Tensor>& reference) {
    return run_join_variant(x, y, reference, "equi_join_v3_1",
        [](auto X, auto Y, auto R, auto O, int64_t N, bool has_ref){
            using T = std::remove_pointer_t<decltype(O)>;
            if (has_ref) join_kernel_v3_1<T, true>(X, Y, R, O, N);
            else         join_kernel_v3_1<T, false>(X, Y, nullptr, O, N);
        });
}

}}  // namespace ezgatr::opt
