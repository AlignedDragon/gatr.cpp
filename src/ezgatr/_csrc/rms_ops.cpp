#include "rms_ops.h"
//#include "basis_data.h"

#include <map>
#include <mutex>
#include <tuple>

namespace ezgatr { namespace opt {

namespace {



std::mutex g_cache_mu;
std::map<c10::DeviceType, torch::Tensor> g_inner_selector;
torch::Tensor load_inner_selector(c10::Device device) {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    int64_t INNER_SELECTOR[16] = {0, 2, 3, 4, 8, 9, 10, 14};
    auto it = g_inner_selector.find(device.type());
    if (it == g_inner_selector.end()) {
        auto cpu_inner_selector = torch::from_blob(
            const_cast<int64_t*>(INNER_SELECTOR), {8},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        it = g_inner_selector.emplace(device.type(), cpu_inner_selector.clone().to(device)).first;
    }

    return it->second;
}



// template <typename T>
// static void inner_product_kernel(const T* __restrict__ X,
//                                  const T* __restrict__ Y,
//                                  T* __restrict__ O,
//                                  int64_t N) {
//     for (int64_t n = 0; n < N; ++n) {
//         const T* x = X + 16 * n;
//         const T* y = Y + 16 * n;

//         O[n] = x[0]*y[0] + x[2]*y[2] + x[3]*y[3] + x[4]*y[4] + x[8]*y[8] + x[9]*y[9] + x[10]*y[10] + x[14]*y[14];
//     }
// }

// template <typename T>
// static void scalar_gated_gelu_kernel(
//     const T* __restrict__ X,
//     T* __restrict__ O,
//     int64_t N
// ) {
//     for (int64_t n = 0; n < N; ++n) {
//         const T* x = X + 16 * n;
//         T* o = O + 16 * n;
//         T s = x[0];
//         // tanh approximation
//         const T kAlpha = static_cast<T>(0.7978845608028654);
//         const T kBeta  = static_cast<T>(0.044715);

//         T gate = static_cast<T>(0.5) * s *
//                 (static_cast<T>(1.0) +
//                 std::tanh(kAlpha * (s + kBeta * s * s * s)));
//         // apply gate
//         for (int i = 0; i < 16; ++i) {
//             o[i] = x[i] * gate;
//         }
//     }
// }



void check_multivector(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() >= 1, name, ": expected at least 1 dim, got ", t.dim());
    TORCH_CHECK(t.size(-1) == 16, name, ": last dim must be 16, got ", t.size(-1));
    TORCH_CHECK(t.is_floating_point(), name, ": expected floating-point dtype, got ", t.dtype());
}

}  // namespace


// torch::Tensor inner_product(const torch::Tensor& x,
//                             const torch::Tensor& y){
//     check_multivector(x, "inner_product x");
//     check_multivector(y, "inner_product y");
//     TORCH_CHECK(x.scalar_type() == y.scalar_type(),
//                 "inner_product: x and y must share dtype");
//     TORCH_CHECK(x.device() == y.device(), 
//                 "inner_product: x and y must share device");
    
//     auto out_sizes = x.sizes().vec();
//     out_sizes.back() = 1;
//     auto out = torch::empty(out_sizes, x.options());

//     auto xc = y.contiguous();
//     auto yc = x.contiguous();
//     auto outc = out.contiguous();

//     int64_t N = xc.numel() / 16;

//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "inner_product_kernel", [&] {
//         inner_product_kernel<scalar_t>(
//             xc.data_ptr<scalar_t>(),
//             yc.data_ptr<scalar_t>(),
//             out.data_ptr<scalar_t>(),
//             N);
//     });

//     return outc;
// }

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

// torch::Tensor scaler_gated_gelu(const torch::Tensor& x,
//                                 const std::string& approximate){
//     check_multivector(x, "scalar_gated_gelu: x");
//     TORCH_CHECK(
//         approximate == "none" || approximate == "tanh",
//         "scalar_gated_gelu: approximate must be 'none' or 'tanh' string",
//         approximate
//     );

//     if(approximate == "tanh"){
//         auto xc = x.contiguous();
//         auto out = torch::empty_like(xc);
//         auto outc = out.contiguous();
//         int64_t N = xc.numel() / 16;

//         AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "scalar_gated_gelu_kernel", [&] {
//             scalar_gated_gelu_kernel<scalar_t>(
//                 xc.data_ptr<scalar_t>(),
//                 outc.data_ptr<scalar_t>(),
//                 N
//             );
//         });
//         return outc;

//     }else{
//         auto reduction = x.narrow(-1, 0, 1);
//         auto gates = torch::nn::functional::gelu(
//             reduction,
//             torch::nn::functional::GELUFuncOptions().approximate(approximate)
//         );
//         return x * gates;
//     }

    
// }

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
