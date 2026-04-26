#include "pga_ops.h"
#include "basis_data.h"

#include <ATen/Dispatch.h>

#include <map>
#include <mutex>
#include <tuple>

namespace ezgatr { namespace opt {

namespace {

#include "gp_unrolled.inc"
#include "join_unrolled.inc"

template <typename T>
static void gp_kernel_impl(const T* __restrict__ X,
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
static void join_kernel_impl(const T* __restrict__ X,
                             const T* __restrict__ Y,
                             const T* __restrict__ R,
                             T* __restrict__ O,
                             int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;
        T*       o = O + 16 * n;
        const T s = HasRef ? R[16 * n + 14] : T(1);
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
        gp_kernel_impl<scalar_t>(xc.data_ptr<scalar_t>(),
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
            join_kernel_impl<scalar_t, true>(
                xc.data_ptr<scalar_t>(),
                yc.data_ptr<scalar_t>(),
                refc.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                N);
        } else {
            join_kernel_impl<scalar_t, false>(
                xc.data_ptr<scalar_t>(),
                yc.data_ptr<scalar_t>(),
                nullptr,
                out.data_ptr<scalar_t>(),
                N);
        }
    });
    return out;
}

}}  // namespace ezgatr::opt
