#include "pga_ops.h"
#include "basis_data.h"

#include <ATen/Dispatch.h>

#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

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
// ver_0 - NO OPTIMIZATION.
// Honest dense baseline: rebuild the full weighted 16x16 basis matrix per
// (out, in) channel pair and run a dense 16x16 matvec. Mirrors the arithmetic
// of the Python einsum "oiw, wds, ...is -> ...od" (~256 MAC per (B,o,i),
// plus the 16x16 matrix assembly).
// ---------------------------------------------------------------------------
torch::Tensor equi_linear_v0(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    auto s = equi_linear_prepare(x, weight);
    auto out = torch::zeros({s.batch, s.out_ch, 16}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear_v0", [&] {
        using T = scalar_t;
        T inv[9];
        for (int k = 0; k < 9; ++k)
            inv[k] = normalize_basis ? T(EQUI_LINEAR_INV_NORMS[k]) : T(1);

        auto xa = s.xf.accessor<T, 3>();
        auto wa = s.wf.accessor<T, 3>();
        auto oa = out.accessor<T, 3>();

        for (int64_t o = 0; o < s.out_ch; ++o) {
            for (int64_t i = 0; i < s.in_ch; ++i) {
                // Assemble the dense 16x16 weighted basis matrix M.
                T M[16][16];
                for (int d = 0; d < 16; ++d)
                    for (int sc = 0; sc < 16; ++sc) M[d][sc] = T(0);
                for (const auto& c : EQUI_LINEAR_COUPLINGS)
                    M[c.dst][c.src] += wa[o][i][c.wk] * inv[c.wk];

                for (int64_t b = 0; b < s.batch; ++b) {
                    for (int d = 0; d < 16; ++d) {
                        T acc = T(0);
                        for (int sc = 0; sc < 16; ++sc)
                            acc += M[d][sc] * xa[b][i][sc];   // dense matvec
                        oa[b][o][d] += acc;
                    }
                }
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
torch::Tensor equi_linear_v2(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    auto s = equi_linear_prepare(x, weight);
    auto out = torch::empty({s.batch, s.out_ch, 16}, x.options());  // O5

    const bool has_bias = bias.has_value();
    torch::Tensor bias_t;
    if (has_bias) bias_t = bias.value().contiguous();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear_v2", [&] {
        using T = scalar_t;
        const T inv_norms[9] = {
            T(EQUI_LINEAR_INV_NORMS[0]), T(EQUI_LINEAR_INV_NORMS[1]),
            T(EQUI_LINEAR_INV_NORMS[2]), T(EQUI_LINEAR_INV_NORMS[3]),
            T(EQUI_LINEAR_INV_NORMS[4]), T(EQUI_LINEAR_INV_NORMS[5]),
            T(EQUI_LINEAR_INV_NORMS[6]), T(EQUI_LINEAR_INV_NORMS[7]),
            T(EQUI_LINEAR_INV_NORMS[8]),
        };

        // (A) Thread-local scratch for packed normalized weights. Keeps
        // capacity across calls so we don't pay the allocator on every
        // forward pass when a model calls equi_linear many times.
        static thread_local std::vector<uint8_t> tls_wn_buf;
        const int64_t n_oi = s.out_ch * s.in_ch;
        const size_t need_bytes = static_cast<size_t>(n_oi) * 9 * sizeof(T);
        if (tls_wn_buf.size() < need_bytes) tls_wn_buf.resize(need_bytes);
        T* __restrict__ wn = reinterpret_cast<T*>(tls_wn_buf.data());

        // O1: pack normalized weights once. Layout: wn[o*Ci*9 + i*9 + k].
        const T* __restrict__ wsrc = s.wf.data_ptr<T>();
        if (normalize_basis) {
            for (int64_t p = 0; p < n_oi; ++p) {
                const T* __restrict__ sptr = wsrc + p * 9;
                T*       __restrict__ dptr = wn   + p * 9;
                for (int k = 0; k < 9; ++k) dptr[k] = sptr[k] * inv_norms[k];
            }
        } else {
            std::memcpy(wn, wsrc, need_bytes);
        }

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
// ver_3 - EXPLICIT SIMD VECTORIZATION.  *** PLACEHOLDER ***
// Planned: AVX2/FMA intrinsics over the grade blocks built in ver_2, plus
// blocking over the batch dimension. Falls back to ver_1 until implemented.
// ---------------------------------------------------------------------------
torch::Tensor equi_linear_v3(const torch::Tensor& x,
                             const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis) {
    // TODO(ver_3): implement SIMD kernel.
    return equi_linear_v1(x, weight, bias, normalize_basis);
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

}}  // namespace ezgatr::opt
