#include "rms_ops.h"
#include "basis_data.h"

#include <map>
#include <mutex>
#include <tuple>

namespace ezgatr { namespace opt {

namespace {



// template <typename T>
// static inline void pack_inner(const T* __restrict__ X,
//                               T* __restrict__ O,
//                               int64_t N) {
//     for (int64_t n = 0; n < N; ++n) {
//         const T* x = X + (n << 4);   // 16-dim input
//         T* o = O + (n << 3);         // 8-dim output

//         o[0] = x[0];
//         o[1] = x[2];
//         o[2] = x[3];
//         o[3] = x[4];
//         o[4] = x[8];
//         o[5] = x[9];
//         o[6] = x[10];
//         o[7] = x[14];
//     }
// }


// torch::Tensor pack_inner_features(const torch::Tensor& x) {
//     TORCH_CHECK(x.size(-1) == 16, "last dim must be 16");

//     auto xc = x.contiguous();

//     auto out_sizes = x.sizes().vec();
//     out_sizes.back() = 8;

//     auto out = torch::empty(out_sizes, x.options());

//     int64_t N = xc.numel() / 16;

//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "pack_inner", [&] {
//         pack_inner<scalar_t>(
//             xc.data_ptr<scalar_t>(),
//             out.data_ptr<scalar_t>(),
//             N
//         );
//     });

//     return out;
// }


//mögliches packing für simd
// torch::Tensor pack_inner_features(const torch::Tensor& x) {
//     TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

//     constexpr int64_t IDX[8] = {0, 2, 3, 4, 8, 9, 10, 14};

//     auto sizes = x.sizes().vec();
//     TORCH_CHECK(sizes.back() == 16, "last dim must be 16");

//     sizes.back() = 8;

//     auto out = torch::empty(sizes, x.options());

//     const auto groups = x.numel() / (16 * sizes[sizes.size() - 2]);

//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pack_inner_features", [&] {
//         const scalar_t* X = x.data_ptr<scalar_t>();
//         scalar_t* O = out.data_ptr<scalar_t>();

//         const int64_t M = x.size(-2);

//         for (int64_t g = 0; g < groups; ++g) {
//             const scalar_t* xg = X + g * M * 16;
//             scalar_t* og = O + g * M * 8;

//             for (int64_t m = 0; m < M; ++m) {
//                 const scalar_t* xm = xg + m * 16;
//                 scalar_t* om = og + m * 8;

//                 om[0] = xm[0];
//                 om[1] = xm[2];
//                 om[2] = xm[3];
//                 om[3] = xm[4];
//                 om[4] = xm[8];
//                 om[5] = xm[9];
//                 om[6] = xm[10];
//                 om[7] = xm[14];
//             }
//         }
//     });

//     return out;
// }



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



template <typename T>
static void inner_product_kernel(const T* __restrict__ X,
                                 const T* __restrict__ Y,
                                 T* __restrict__ O,
                                 int64_t N) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + 16 * n;
        const T* y = Y + 16 * n;

        O[n] = x[0]*y[0] + x[2]*y[2] + x[3]*y[3] + x[4]*y[4] + x[8]*y[8] + x[9]*y[9] + x[10]*y[10] + x[14]*y[14];
    }
}

template <typename T>
static void scalar_gated_gelu_kernel(
    const T* __restrict__ X,
    T* __restrict__ O,
    int64_t N
) {
    for (int64_t n = 0; n < N; ++n) {
        const T* x = X + (n << 4);
        T* o = O + (n << 4);
        T s = x[0];
        // tanh approximation
        const T kAlpha = static_cast<T>(0.7978845608028654);
        const T kBeta  = static_cast<T>(0.044715);

        T gate = static_cast<T>(0.5) * s *
                (static_cast<T>(1.0) +
                std::tanh(kAlpha * (s + kBeta * s * s * s)));
        // apply gate
        for (int i = 0; i < 16; ++i) {
            o[i] = x[i] * gate;
        }
    }
}



void check_multivector(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() >= 1, name, ": expected at least 1 dim, got ", t.dim());
    TORCH_CHECK(t.size(-1) == 16, name, ": last dim must be 16, got ", t.size(-1));
    TORCH_CHECK(t.is_floating_point(), name, ": expected floating-point dtype, got ", t.dtype());
}


template <typename scalar_t, bool HasWeight>
void rms_norm_kernel(
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ O,
    const scalar_t* __restrict__ W,
    int64_t groups,
    int64_t M,
    double eps
) {
    const scalar_t eps_t = static_cast<scalar_t>(eps);

    for (int64_t g = 0; g < groups; ++g) {

        const scalar_t* group_in  = X + g * (M << 4);
        scalar_t* group_out       = O + g * (M << 4);

        scalar_t acc = 0;

        // ---- RMS accumulation ----
        // for (int64_t m = 0; m < M; ++m) {
        //     const scalar_t* x = group_in + (m << 4);

        //     acc +=
        //         x[0]  * x[0]  +
        //         x[2]  * x[2]  +
        //         x[3]  * x[3]  +
        //         x[4]  * x[4]  +
        //         x[8]  * x[8]  +
        //         x[9]  * x[9]  +
        //         x[10] * x[10] +
        //         x[14] * x[14];
        // }

            scalar_t acc0 = 0;
            scalar_t acc1 = 0;
            scalar_t acc2 = 0;
            scalar_t acc3 = 0;
            int64_t m = 0;

            for (; m + 3 < M; m+=4) {

                const scalar_t* x0 = group_in + (m << 4);
                const scalar_t* x1 = x0 + 16;
                const scalar_t* x2 = x1 + 16;
                const scalar_t* x3 = x2 + 16;
                

                acc0 +=
                    x0[0]*x0[0] +
                    x0[2]*x0[2] +
                    x0[3]*x0[3] +
                    x0[4]*x0[4] +
                    x0[8]*x0[8] +
                    x0[9]*x0[9] +
                    x0[10]*x0[10] +
                    x0[14]*x0[14];

                acc1 +=
                    x1[0]*x1[0] +
                    x1[2]*x1[2] +
                    x1[3]*x1[3] +
                    x1[4]*x1[4] +
                    x1[8]*x1[8] +
                    x1[9]*x1[9] +
                    x1[10]*x1[10] +
                    x1[14]*x1[14];

                acc2 +=
                    x2[0]*x2[0] +
                    x2[2]*x2[2] +
                    x2[3]*x2[3] +
                    x2[4]*x2[4] +
                    x2[8]*x2[8] +
                    x2[9]*x2[9] +
                    x2[10]*x2[10] +
                    x2[14]*x2[14];

                acc3 +=
                    x3[0]*x3[0] +
                    x3[2]*x3[2] +
                    x3[3]*x3[3] +
                    x3[4]*x3[4] +
                    x3[8]*x3[8] +
                    x3[9]*x3[9] +
                    x3[10]*x3[10] +
                    x3[14]*x3[14];
        }

        scalar_t acc = acc0 + acc1 + acc2 + acc3;

        for (; m < M; ++m) {
            const scalar_t* x = group_in + (m << 4);

            acc +=
                x[0]*x[0] +
                x[2]*x[2] +
                x[3]*x[3] +
                x[4]*x[4] +
                x[8]*x[8] +
                x[9]*x[9] +
                x[10]*x[10] +
                x[14]*x[14];
        }



        acc /= static_cast<scalar_t>(M);

        scalar_t scale = scalar_t(1) / std::sqrt(std::max(acc, eps_t));
        //scalar_t scale = std::rsqrt(acc + eps_t);
        //scalar_t scale = scalar_t(1) / std::sqrt(acc + eps_t);

        // ---- write + optional weight ----
        for (int64_t m = 0; m < M; ++m) {

            const scalar_t* x = group_in + (m << 4);
            scalar_t* o       = group_out + (m << 4);

            scalar_t scalew = scale;

            if constexpr (HasWeight) {
                scalew *= W[m];
            }

            o[0]  = x[0]  * scalew;
            o[1]  = x[1]  * scalew;
            o[2]  = x[2]  * scalew;
            o[3]  = x[3]  * scalew;
            o[4]  = x[4]  * scalew;
            o[5]  = x[5]  * scalew;
            o[6]  = x[6]  * scalew;
            o[7]  = x[7]  * scalew;
            o[8]  = x[8]  * scalew;
            o[9]  = x[9]  * scalew;
            o[10] = x[10] * scalew;
            o[11] = x[11] * scalew;
            o[12] = x[12] * scalew;
            o[13] = x[13] * scalew;
            o[14] = x[14] * scalew;
            o[15] = x[15] * scalew;
        }
    }
}




}  // namespace


 torch::Tensor inner_product_ver_0(const torch::Tensor& x,
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




torch::Tensor inner_product_ver_1(const torch::Tensor& x,
                            const torch::Tensor& y){
    check_multivector(x, "inner_product x");
    check_multivector(y, "inner_product y");

    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "inner_product: x and y must share dtype");

    TORCH_CHECK(x.device() == y.device(),
                "inner_product: x and y must share device");

    auto out_sizes = x.sizes().vec();
    out_sizes.back() = 1;

    auto xc = x.contiguous();
    auto yc = y.contiguous();
    auto out = torch::empty(out_sizes, x.options());
    //auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;
    //int64_t N = x.size(-2);
    AT_DISPATCH_FLOATING_TYPES(
        xc.scalar_type(),
        "inner_product_kernel",
        [&] {

            const scalar_t* __restrict__ X =
                xc.data_ptr<scalar_t>();

            const scalar_t* __restrict__ Y =
                yc.data_ptr<scalar_t>();

            scalar_t* __restrict__ O =
                out.data_ptr<scalar_t>();

            for (int64_t n = 0; n < N; ++n) {

                const scalar_t* x = X + (n << 4);
                const scalar_t* y = Y + (n << 4);
                  
                O[n] =
                    x[0]  * y[0]  +
                    x[2]  * y[2]  +
                    x[3]  * y[3]  +
                    x[4]  * y[4]  +
                    x[8]  * y[8]  +
                    x[9]  * y[9]  +
                    x[10] * y[10] +
                    x[14] * y[14];
            }
        });

    return out;
}


torch::Tensor inner_product_ver_2(const torch::Tensor& x,
                            const torch::Tensor& y){
    check_multivector(x, "inner_product x");
    check_multivector(y, "inner_product y");

    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "inner_product: x and y must share dtype");

    TORCH_CHECK(x.device() == y.device(),
                "inner_product: x and y must share device");

    auto out_sizes = x.sizes().vec();
    out_sizes.back() = 1;

    auto xc = x.contiguous();
    auto yc = y.contiguous();
    auto out = torch::empty(out_sizes, x.options());
    //auto out = torch::empty_like(xc);
    int64_t N = xc.numel() / 16;
    //int64_t N = x.size(-2);
    AT_DISPATCH_FLOATING_TYPES(
        xc.scalar_type(),
        "inner_product_kernel",
        [&] {

            const scalar_t* __restrict__ X =
                xc.data_ptr<scalar_t>();

            const scalar_t* __restrict__ Y =
                yc.data_ptr<scalar_t>();

            scalar_t* __restrict__ O =
                out.data_ptr<scalar_t>();

            for (int64_t n = 0; n < N; ++n) {

                const scalar_t* x = X + (n << 4);
                const scalar_t* y = Y + (n << 4);
                // const scalar_t o1 = x[0]  * y[0];
                // const scalar_t o2 = x[2]  * y[2];  
                // const scalar_t o3 = x[3]  * y[3];
                // const scalar_t o4 = x[4]  * y[4]; 

                // const scalar_t o8 = x[8]  * y[8];
                // const scalar_t o9 = x[9]  * y[9]; 
                // const scalar_t o10 = x[10]  * y[10];
                // const scalar_t o14 = x[14]  * y[14]; 

                // O[n] = o1 + o2 + o3 + o4 + o8 + o9 + o10 + o14;    

                O[n] =
                    x[0]  * y[0]  +
                    x[2]  * y[2]  +
                    x[3]  * y[3]  +
                    x[4]  * y[4]  +
                    x[8]  * y[8]  +
                    x[9]  * y[9]  +
                    x[10] * y[10] +
                    x[14] * y[14];
            }
        });

    return out;
}


// ---------------------------------------------------------------------------
// ver_0 - NO OPTIMIZATION.
// c++ implemnetation of the base version
// ---------------------------------------------------------------------------
torch::Tensor equi_rms_norm_ver_0(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt){
    check_multivector(x, "equi_rms_norm x");
    double eps = eps_opt.has_value()
        ? *eps_opt
        : 1e-7;

    auto ip = inner_product_ver_0(x,x);
    auto normip = ip.mean(/*dim=*/-2, /*keepdim=*/true);
    auto result = x/torch::sqrt(torch::clamp(normip,eps));
    if (weight.has_value()){
        result = result * weight->view({-1,1});
    }
    return result;
}


// ---------------------------------------------------------------------------
// ver_1 - Optimized C++ implementation.
// Performance improvements are mainly achieved through the optimized
// inner_product kernel using direct raw pointer access, contiguous memory
// layout, fixed 16-element block processing, pointer arithmetic, manual
// loop unrolling, hardcoded selectors and __restrict__ alias optimization. These changes
// reduce PyTorch overhead, improve cache locality, enable better SIMD
// vectorization, and minimize temporary memory operations.
// ---------------------------------------------------------------------------
torch::Tensor equi_rms_norm_ver_1(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt){
    check_multivector(x, "equi_rms_norm x");
    double eps = eps_opt.has_value()
        ? *eps_opt
        : 1e-7;

    auto ip = inner_product_ver_1(x,x);
    auto normip = ip.mean(/*dim=*/-2, /*keepdim=*/true);
    auto result = x/torch::sqrt(torch::clamp(normip,eps));
    if (weight.has_value()){
        result = result * weight->view({-1,1});
    }
    return result;
}



// ---------------------------------------------------------------------------
// ver_2 - Fully fused C++ implementation.
//
// Compared to ver_1, the RMS normalization pipeline is completely fused
// into a single kernel. Instead of separate passes for inner product,
// mean reduction, clamp, sqrt, normalization, and optional weight scaling,
// all computations are performed in one streaming loop.
//
// This removes intermediate tensors and eliminates multiple full memory
// traversals over the same data. The result is reduced memory bandwidth
// usage, fewer kernel launches, and improved cache efficiency.
//
// Additional optimizations include compile-time
// specialization via templates, multiple independent accumulators for
// better instruction-level parallelism
// ---------------------------------------------------------------------------
torch::Tensor equi_rms_norm_ver_2(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<double>& eps_opt
) {
    check_multivector(x, "equi_rms_norm x");

    double eps = eps_opt.has_value() ? *eps_opt : 1e-7;

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto out = torch::empty_like(xc);

    int64_t M = x.size(-2);

    int64_t groups = 1;
    for (int64_t i = 0; i < x.dim() - 2; ++i)
        groups *= x.size(i);

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel", [&] {
        const scalar_t* X = xc.data_ptr<scalar_t>();
        scalar_t* O       = out.data_ptr<scalar_t>();

        if (weight.has_value()) {
            auto wc = weight->contiguous();
            const scalar_t* W = wc.data_ptr<scalar_t>();

            rms_norm_kernel<scalar_t, true>(
                X, O, W, groups, M, eps
            );
        } else {
            rms_norm_kernel<scalar_t, false>(
                X, O, nullptr, groups, M, eps
            );
        }
    });

    return out;
}


// ---------------------------------------------------------------------------
// ver_3 - 
//create new tensor of x and pack it [16] -> [8]. tile loops and do intrinsics
//8 vectors for each index one and then multiply
// ---------------------------------------------------------------------------
torch::Tensor equi_rms_norm_ver_3(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<double>& eps_opt
) {
    check_multivector(x, "equi_rms_norm x");

    double eps = eps_opt.has_value() ? *eps_opt : 1e-7;

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto out = torch::empty_like(xc);

    int64_t M = x.size(-2);

    int64_t groups = 1;
    for (int64_t i = 0; i < x.dim() - 2; ++i)
        groups *= x.size(i);

    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel", [&] {
        const scalar_t* X = xc.data_ptr<scalar_t>();
        scalar_t* O       = out.data_ptr<scalar_t>();

        if (weight.has_value()) {
            auto wc = weight->contiguous();
            const scalar_t* W = wc.data_ptr<scalar_t>();

            rms_norm_kernel<scalar_t, true>(
                X, O, W, groups, M, eps
            );
        } else {
            rms_norm_kernel<scalar_t, false>(
                X, O, nullptr, groups, M, eps
            );
        }
    });

    return out;
}



torch::Tensor scaler_gated_gelu_ver_0(const torch::Tensor& x,
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



torch::Tensor scaler_gated_gelu_ver_1(const torch::Tensor& x,
                                const std::string& approximate){
    check_multivector(x, "scalar_gated_gelu: x");
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "scalar_gated_gelu: approximate must be 'none' or 'tanh' string",
        approximate
    );

    if(approximate == "tanh"){
        auto xc = x.contiguous();
        auto out = torch::empty_like(xc);
        auto outc = out.contiguous();
        int64_t N = xc.numel() / 16;
        //int64_t N = x.size(-2);
        AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "scalar_gated_gelu_kernel", [&] {
            scalar_gated_gelu_kernel<scalar_t>(
                xc.data_ptr<scalar_t>(),
                outc.data_ptr<scalar_t>(),
                N
            );
        });
        return outc;

    }else{
        auto reduction = x.narrow(-1, 0, 1);
        auto gates = torch::nn::functional::gelu(
            reduction,
            torch::nn::functional::GELUFuncOptions().approximate(approximate)
        );
        return x * gates;
    }

    
}



torch::Tensor scaler_gated_gelu_ver_2(
    const torch::Tensor& x,
    const std::string& approximate
){
    check_multivector(x, "scalar_gated_gelu: x");
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "scalar_gated_gelu: approximate must be 'none' or 'tanh' string",
        approximate
    );

    if(approximate == "tanh"){
        auto xc = x.contiguous();
        auto out = torch::empty_like(xc);
        auto outc = out.contiguous();
        int64_t N = xc.numel() / 16;
        //int64_t N = x.size(-2);

        AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "scalar_gated_gelu_kernel", [&] {
            scalar_gated_gelu_kernel<scalar_t>(
                xc.data_ptr<scalar_t>(),
                outc.data_ptr<scalar_t>(),
                N
            );
        });
        return outc;

    }else{
        auto reduction = x.narrow(-1, 0, 1);
        auto gates = torch::nn::functional::gelu(
            reduction,
            torch::nn::functional::GELUFuncOptions().approximate(approximate)
        );
        return x * gates;
    }
}




}}  // namespace ezgatr::opt
