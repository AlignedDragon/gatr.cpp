#include "pga_ops.h"
#include "basis_data.h"

#include <map>
#include <mutex>
#include <tuple>

namespace ezgatr { namespace opt {

namespace {

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
std::map<c10::DeviceType, torch::Tensor> g_inner_selector;

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

torch::Tensor load_inner_selector(c10::Device device) {
    std::lock_guard<std::mutex> lock(g_cache_mu);

    auto it = g_inner_selector.find(device.type());
    if (it == g_inner_selector.end()) {
        auto cpu_inner_selector = torch::from_blob(
            const_cast<int64_t*>(INNER_SELECTOR), {8},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        it = g_inner_selector.emplace(device.type(), cpu_inner_selector.clone().to(device)).first;
    }

    return it->second;
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
    auto basis = load_gp_basis(x.device(), x.scalar_type());
    return torch::einsum("ijk, ...j, ...k -> ...i", {basis, x, y});
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

    auto kernel = compute_join_kernel(x.device(), x.scalar_type());
    auto ret = torch::einsum("ijk, ...j, ...k -> ...i", {kernel, x, y});

    if (reference.has_value()) {
        check_multivector(*reference, "equi_join: reference");
        ret = ret * reference->narrow(-1, 14, 1);
    }
    return ret;
}

torch::Tensor inner_product(const torch::Tensor& x,
                            const torch::Tensor& y){
    check_multivector(x, "inner_product x");
    check_multivector(y, "inner_product y");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "inner_product: x and y must share dtype");
    TORCH_CHECK(x.device() == y.device(), 
                "inner_product: x and y must share device");
    auto selector = load_inner_selector(x.device());
    auto xsel = torch::index_select(x,-1,selector);
    auto ysel = torch::index_select(y,-1,selector);
    return torch::einsum("...i, ...i -> ...",{xsel,ysel}).unsqueeze(-1);
}

torch::Tensor equi_rms_norm(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt){
    check_multivector(x, "equi_rms_norm x");
    double eps = eps_opt.has_value()
        ? *eps_opt
        : 1e-7;

    auto ip = inner_product(x,x);
    auto normip = ip.mean(/*dim=*/-2, /*keepdim=*/true);
    auto result = x/torch::sqrt(torch::clamp(normip,eps));
    if (weight.has_value()){
        result = result * weight->view({-1,1});
    }
    return result;
}

torch::Tensor scaler_gated_gelu(const torch::Tensor& x,
                                const std::string& approximate){
    check_multivector(x, "scalar_gated_gelu: x");
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "scalar_gated_gelu: approximate must be 'none' or 'tanh' string",
        approximate
    );

    auto reduction = x.narrow(-1, 0, 1);
    auto gates = torch::nn::functional::gelu(
        reduction,
        torch::nn::functional::GELUFuncOptions().approximate(approximate)
    );
    return x * gates;
}


}}  // namespace ezgatr::opt
