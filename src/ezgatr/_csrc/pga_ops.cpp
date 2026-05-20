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


static constexpr double EQUI_LINEAR_INV_NORMS[9] = {
    1.0, 0.5, 0.40824829046386302, 0.5, 1.0,
    1.0, 0.5773502691896258, 0.5773502691896258, 1.0,
};

torch::Tensor equi_linear(const torch::Tensor& x,
                           const torch::Tensor& weight,
                           const c10::optional<torch::Tensor>& bias,
                           bool normalize_basis) {
    TORCH_CHECK(x.dim() >= 2,        "equi_linear: x needs at least 2 dims");
    TORCH_CHECK(x.size(-1) == 16,    "equi_linear: x last dim must be 16");
    TORCH_CHECK(weight.dim() == 3,   "equi_linear: weight must be 3-D (out, in, 9)");
    TORCH_CHECK(weight.size(2) == 9, "equi_linear: weight last dim must be 9");
    TORCH_CHECK(x.size(-2) == weight.size(1), "equi_linear: in_channels mismatch");

    const int64_t out_ch = weight.size(0);
    const int64_t in_ch  = weight.size(1);
    const int64_t batch  = x.numel() / (in_ch * 16);

    auto xf  = x.reshape({batch, in_ch, 16}).contiguous();
    auto wf  = weight.contiguous();
    auto out = torch::zeros({batch, out_ch, 16}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "equi_linear", [&] {
        using T = scalar_t;
        const T inv_norms[9] = {
            T(EQUI_LINEAR_INV_NORMS[0]), T(EQUI_LINEAR_INV_NORMS[1]),
            T(EQUI_LINEAR_INV_NORMS[2]), T(EQUI_LINEAR_INV_NORMS[3]),
            T(EQUI_LINEAR_INV_NORMS[4]), T(EQUI_LINEAR_INV_NORMS[5]),
            T(EQUI_LINEAR_INV_NORMS[6]), T(EQUI_LINEAR_INV_NORMS[7]),
            T(EQUI_LINEAR_INV_NORMS[8]),
        };

        auto xa = xf.accessor<T, 3>();   
        auto wa = wf.accessor<T, 3>();   
        auto oa = out.accessor<T, 3>();  

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t o = 0; o < out_ch; ++o) {
                for (int64_t i = 0; i < in_ch; ++i) {
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

    if (bias.has_value()) {
        out.select(-1, 0).add_(bias.value());
    }

    auto out_shape = x.sizes().vec();
    out_shape[out_shape.size() - 2] = out_ch;
    return out.reshape(out_shape);
}

}}  // namespace ezgatr::opt
