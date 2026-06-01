#include "rms_ops.h"
#include "basis_data.h"

#include <map>
#include <mutex>
#include <tuple>





#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <type_traits>






namespace ezgatr { namespace opt {

namespace {




//Packing!! for simd
static constexpr int IDX[8] = {0, 2, 3, 4, 8, 9, 10, 14};

template <typename scalar_t>
void pack_groups_16_to_8_soa(
    const scalar_t* __restrict__ X,   // [groups, M, 16]
    scalar_t* __restrict__ P,         // [groups, 8, M]
    int64_t groups,
    int64_t M
) {
    for (int64_t g = 0; g < groups; ++g) {

        const scalar_t* xg = X + g * (M << 4);
        scalar_t* pg       = P + g * (8 * M);

        for (int64_t m = 0; m < M; ++m) {

            const scalar_t* x = xg + (m << 4);

            pg[0 * M + m] = x[IDX[0]];
            pg[1 * M + m] = x[IDX[1]];
            pg[2 * M + m] = x[IDX[2]];
            pg[3 * M + m] = x[IDX[3]];
            pg[4 * M + m] = x[IDX[4]];
            pg[5 * M + m] = x[IDX[5]];
            pg[6 * M + m] = x[IDX[6]];
            pg[7 * M + m] = x[IDX[7]];
        }
    }
}



//  template <typename T>
//  static inline void pack_inner(const T* __restrict__ X,
//                                T* __restrict__ O,
//                               int64_t N) {
//      for (int64_t n = 0; n+7 < N; n=n+8) {
//          const T* x0 = X + (n << 4);   // 16-dim input
//         const T* x1 = x0 + 16;   // 16-dim input
//         const T* x2 = x1 + 16;   // 16-dim input
//         const T* x3 = x1 + 16;   // 16-dim input
//         const T* x4 = x1 + 16;   // 16-dim input
//         const T* x5 = x1 + 16;   // 16-dim input
//         const T* x6 = x1 + 16;   // 16-dim input
//         const T* x7 = x1 + 16;   // 16-dim input
//         T* o0 = O + (n << 3);         // 8-dim output
//         T* o1 = o0 + 8;   // 16-dim input
//         T* o2 = o1 + 8;   // 16-dim input
//         T* o3 = o2 + 8;   // 16-dim input
//         T* o4 = o3 + 8;   // 16-dim input
//         T* o5 = o4 + 8;   // 16-dim input
//         T* o6 = o5 + 8;   // 16-dim input
//         T* o7 = o6 + 8;   // 16-dim input

//         o0[2] = x2[0];
//         o1[2] = x2[2];
//         o2[2] = x2[3];
//         o3[2] = x2[4];
//         o4[2] = x2[8];
//         o5[2] = x2[9];
//         o6[2] = x2[10];
//         o7[2] = x2[14];

//         o0[3] = x3[0];
//         o1[3] = x3[2];
//         o2[3] = x3[3];
//         o3[3] = x3[4];
//         o4[3] = x3[8];
//         o5[3] = x3[9];
//         o6[3] = x3[10];
//         o7[3] = x3[14];

//         o0[4] = x4[0];
//         o1[4] = x4[2];
//         o2[4] = x4[3];
//         o3[4] = x4[4];
//         o4[4] = x4[8];
//         o5[4] = x4[9];
//         o6[4] = x4[10];
//         o7[4] = x4[14];

//         o0[5] = x5[0];
//         o1[5] = x5[2];
//         o2[5] = x5[3];
//         o3[5] = x5[4];
//         o4[5] = x5[8];
//         o5[5] = x5[9];
//         o6[5] = x5[10];
//         o7[5] = x5[14];

//         o0[6] = x6[0];
//         o1[6] = x6[2];
//         o2[6] = x6[3];
//         o3[6] = x6[4];
//         o4[6] = x6[8];
//         o5[6] = x6[9];
//         o6[6] = x6[10];
//         o7[6] = x6[14];

//         o0[7] = x7[0];
//         o1[7] = x7[2];
//         o2[7] = x7[3];
//         o3[7] = x7[4];
//         o4[7] = x7[8];
//         o5[7] = x7[9];
//         o6[7] = x7[10];
//         o7[7] = x7[14];
//     }
//  }


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

        //scalar_t acc = 0;

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
            // scalar_t acc4 = 0;
            // scalar_t acc5 = 0;
            // scalar_t acc6 = 0;
            // scalar_t acc7 = 0;
            int64_t m = 0;

            for (; m + 3 < M; m+=4) {
                const scalar_t* x0 = group_in + (m << 4);
                const scalar_t* x1 = x0 + 16;
                const scalar_t* x2 = x1 + 16;
                const scalar_t* x3 = x2 + 16;
                // const scalar_t* x4 = x3 + 16;
                // const scalar_t* x5 = x4 + 16;
                // const scalar_t* x6 = x5 + 16;
                // const scalar_t* x7 = x6 + 16;
                

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

                // acc0 +=
                //     x1[0]*x1[0] + x2[0]*x2[0] + x3[0]*x3[0] + x4[0]*x4[0] +
                //     x5[0]*x5[0] + x6[0]*x6[0] + x7[0]*x7[0];

                // acc1 +=
                //     x1[2]*x1[2] + x2[2]*x2[2] + x3[2]*x3[2] + x4[2]*x4[2] +
                //     x5[2]*x5[2] + x6[2]*x6[2] + x7[2]*x7[2];

                // acc2 +=
                //     x1[3]*x1[3] + x2[3]*x2[3] + x3[3]*x3[3] + x4[3]*x4[3] +
                //     x5[3]*x5[3] + x6[3]*x6[3] + x7[3]*x7[3];

                // acc3 +=
                //     x1[4]*x1[4] + x2[4]*x2[4] + x3[4]*x3[4] + x4[4]*x4[4] +
                //     x5[4]*x5[4] + x6[4]*x6[4] + x7[4]*x7[4];

                // acc4 +=
                //     x1[8]*x1[8] + x2[8]*x2[8] + x3[8]*x3[8] + x4[8]*x4[8] +
                //     x5[8]*x5[8] + x6[8]*x6[8] + x7[8]*x7[8];

                // acc5 +=
                //     x1[9]*x1[9] + x2[9]*x2[9] + x3[9]*x3[9] + x4[9]*x4[9] +
                //     x5[9]*x5[9] + x6[9]*x6[9] + x7[9]*x7[9];

                // acc6 +=
                //     x1[10]*x1[10] + x2[10]*x2[10] + x3[10]*x3[10] + x4[10]*x4[10] +
                //     x5[10]*x5[10] + x6[10]*x6[10] + x7[10]*x7[10];

                // acc7 +=
                //     x1[14]*x1[14] + x2[14]*x2[14] + x3[14]*x3[14] + x4[14]*x4[14] +
                //     x5[14]*x5[14] + x6[14]*x6[14]; //+ x7[14]*x7[14];
        }

        scalar_t acc = acc0 + acc1 + acc2 + acc3 ;//+ acc4 +acc5 + acc6 + acc7;

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



// template <typename scalar_t, bool HasWeight>
// void rms_norm_kernel_intrins(
//     const scalar_t* __restrict__ X,
//     scalar_t* __restrict__ O,
//     const scalar_t* __restrict__ W,
//     int64_t groups,
//     int64_t M,
//     double eps
// ) {
//     const scalar_t eps_t = static_cast<scalar_t>(eps);

//     if constexpr (std::is_same_v<scalar_t, float>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const scalar_t* group_in  = X + g * (M << 4);
//             scalar_t* group_out       = O + g * (M << 4);


//             __m256 acc  = _mm256_setzero_ps();

//             int64_t m = 0;

//             for (; m + 7 < M; m += 8) {

//                 const scalar_t* x0 = group_in + (m << 4);
//                 const scalar_t* x1 = x0 + 16;
//                 const scalar_t* x2 = x1 + 16;
//                 const scalar_t* x3 = x2 + 16;
//                 const scalar_t* x4 = x3 + 16;
//                 const scalar_t* x5 = x4 + 16;
//                 const scalar_t* x6 = x5 + 16;
//                 const scalar_t* x7 = x6 + 16;

//                 // load selected scalars (gather-light, no real gather)
//                 __m256 v0 = _mm256_set_ps(
//                     x7[0], x6[0], x5[0], x4[0],
//                     x3[0], x2[0], x1[0], x0[0]);

//                 __m256 v2 = _mm256_set_ps(
//                     x7[2], x6[2], x5[2], x4[2],
//                     x3[2], x2[2], x1[2], x0[2]);

//                 __m256 v3 = _mm256_set_ps(
//                     x7[3], x6[3], x5[3], x4[3],
//                     x3[3], x2[3], x1[3], x0[3]);

//                 __m256 v4 = _mm256_set_ps(
//                     x7[4], x6[4], x5[4], x4[4],
//                     x3[4], x2[4], x1[4], x0[4]);

//                 __m256 v8 = _mm256_set_ps(
//                     x7[8], x6[8], x5[8], x4[8],
//                     x3[8], x2[8], x1[8], x0[8]);

//                 __m256 v9 = _mm256_set_ps(
//                     x7[9], x6[9], x5[9], x4[9],
//                     x3[9], x2[9], x1[9], x0[9]);

//                 __m256 v10 = _mm256_set_ps(
//                     x7[10], x6[10], x5[10], x4[10],
//                     x3[10], x2[10], x1[10], x0[10]);

//                 __m256 v14 = _mm256_set_ps(
//                     x7[14], x6[14], x5[14], x4[14],
//                     x3[14], x2[14], x1[14], x0[14]);

//                 // accumulate squares
//                 acc  = _mm256_fmadd_ps(v0,  v0,  acc);
//                 acc  = _mm256_fmadd_ps(v2,  v2,  acc);
//                 acc  = _mm256_fmadd_ps(v3,  v3,  acc);
//                 acc  = _mm256_fmadd_ps(v4,  v4,  acc);
//                 acc  = _mm256_fmadd_ps(v8,  v8,  acc);
//                 acc  = _mm256_fmadd_ps(v9,  v9,  acc);
//                 acc = _mm256_fmadd_ps(v10, v10, acc);
//                 acc = _mm256_fmadd_ps(v14, v14, acc);
//             }
            
//             __m128 lo = _mm256_castps256_ps128(acc);
//             __m128 hi = _mm256_extractf128_ps(acc, 1);

//             __m128 sum = _mm_add_ps(lo, hi);
//             sum = _mm_hadd_ps(sum, sum);
//             sum = _mm_hadd_ps(sum, sum);

//             scalar_t accs = _mm_cvtss_f32(sum);

//             for (; m < M; ++m) {
//                 const scalar_t* x = group_in + (m << 4);

//                 accs +=
//                     x[0]*x[0] +
//                     x[2]*x[2] +
//                     x[3]*x[3] +
//                     x[4]*x[4] +
//                     x[8]*x[8] +
//                     x[9]*x[9] +
//                     x[10]*x[10] +
//                     x[14]*x[14];
//             }



//             accs /= static_cast<scalar_t>(M);

//             scalar_t scale = scalar_t(1) / std::sqrt(std::max(accs, eps_t));
//             //scalar_t scale = std::rsqrt(acc + eps_t);
//             //scalar_t scale = scalar_t(1) / std::sqrt(acc + eps_t);

//             // ---- write + optional weight ----
//             for (int64_t m = 0; m < M; ++m) {

//                 const scalar_t* x = group_in + (m << 4);
//                 scalar_t* o       = group_out + (m << 4);

//                 scalar_t scalew = scale;

//                 if constexpr (HasWeight) {
//                     scalew *= W[m];
//                 }

//                 o[0]  = x[0]  * scalew;
//                 o[1]  = x[1]  * scalew;
//                 o[2]  = x[2]  * scalew;
//                 o[3]  = x[3]  * scalew;
//                 o[4]  = x[4]  * scalew;
//                 o[5]  = x[5]  * scalew;
//                 o[6]  = x[6]  * scalew;
//                 o[7]  = x[7]  * scalew;
//                 o[8]  = x[8]  * scalew;
//                 o[9]  = x[9]  * scalew;
//                 o[10] = x[10] * scalew;
//                 o[11] = x[11] * scalew;
//                 o[12] = x[12] * scalew;
//                 o[13] = x[13] * scalew;
//                 o[14] = x[14] * scalew;
//                 o[15] = x[15] * scalew;
//             }

                // for (int64_t m = 0; m < M; ++m) {
                //     const float* x = group_in + (m << 4);
                //     float* o       = group_out + (m << 4);

                //     float scalew = scale;

                //     if constexpr (HasWeight) {
                //         scalew *= W[m];
                //     }

                //     __m256 vscale = _mm256_set1_ps(scalew);

                //     __m256 v0 = _mm256_loadu_ps(x + 0);
                //     __m256 v1 = _mm256_loadu_ps(x + 8);

                //     v0 = _mm256_mul_ps(v0, vscale);
                //     v1 = _mm256_mul_ps(v1, vscale);

                //     _mm256_storeu_ps(o + 0, v0);
                //     _mm256_storeu_ps(o + 8, v1);
                // }
//         }
//     }else if constexpr (std::is_same_v<scalar_t, double>) {
//         for (int64_t g = 0; g < groups; ++g) {

//             const scalar_t* group_in  = X + g * (M << 4);
//             scalar_t* group_out       = O + g * (M << 4);

//             __m256d acc = _mm256_setzero_pd();

//             int64_t m = 0;

//             // AVX2 double: 4 lanes → process 4 "blocks"
//             for (; m + 3 < M; m += 4) {

//                 const scalar_t* x0 = group_in + (m << 4);
//                 const scalar_t* x1 = x0 + 16;
//                 const scalar_t* x2 = x1 + 16;
//                 const scalar_t* x3 = x2 + 16;

//                 // ---- load selected indices (double version) ----

//                 __m256d v0 = _mm256_set_pd(
//                     x3[0], x2[0], x1[0], x0[0]);

//                 __m256d v2 = _mm256_set_pd(
//                     x3[2], x2[2], x1[2], x0[2]);

//                 __m256d v3 = _mm256_set_pd(
//                     x3[3], x2[3], x1[3], x0[3]);

//                 __m256d v4 = _mm256_set_pd(
//                     x3[4], x2[4], x1[4], x0[4]);

//                 __m256d v8 = _mm256_set_pd(
//                     x3[8], x2[8], x1[8], x0[8]);

//                 __m256d v9 = _mm256_set_pd(
//                     x3[9], x2[9], x1[9], x0[9]);

//                 __m256d v10 = _mm256_set_pd(
//                     x3[10], x2[10], x1[10], x0[10]);

//                 __m256d v14 = _mm256_set_pd(
//                     x3[14], x2[14], x1[14], x0[14]);

//                 // ---- accumulate squares ----
//                 acc = _mm256_fmadd_pd(v0,  v0,  acc);
//                 acc = _mm256_fmadd_pd(v2,  v2,  acc);
//                 acc = _mm256_fmadd_pd(v3,  v3,  acc);
//                 acc = _mm256_fmadd_pd(v4,  v4,  acc);
//                 acc = _mm256_fmadd_pd(v8,  v8,  acc);
//                 acc = _mm256_fmadd_pd(v9,  v9,  acc);
//                 acc = _mm256_fmadd_pd(v10, v10, acc);
//                 acc = _mm256_fmadd_pd(v14, v14, acc);
//             }

//             // ---- horizontal sum (double) ----
//             __m128d lo = _mm256_castpd256_pd128(acc);
//             __m128d hi = _mm256_extractf128_pd(acc, 1);

//             __m128d sum = _mm_add_pd(lo, hi);

//             // reduce 2 doubles → scalar
//             scalar_t accs =
//                 _mm_cvtsd_f64(sum) +
//                 _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));

//             // ---- scalar tail ----
//             for (; m < M; ++m) {
//                 const scalar_t* x = group_in + (m << 4);

//                 accs +=
//                     x[0]*x[0] +
//                     x[2]*x[2] +
//                     x[3]*x[3] +
//                     x[4]*x[4] +
//                     x[8]*x[8] +
//                     x[9]*x[9] +
//                     x[10]*x[10] +
//                     x[14]*x[14];
//             }

//             accs /= static_cast<scalar_t>(M);

//             scalar_t scale = scalar_t(1) / std::sqrt(std::max(accs, eps_t));

//             // ---- write back ----
//             for (int64_t m = 0; m < M; ++m) {

//                 const scalar_t* x = group_in + (m << 4);
//                 scalar_t* o       = group_out + (m << 4);

//                 scalar_t scalew = scale;

//                 if constexpr (HasWeight) {
//                     scalew *= W[m];
//                 }

//                 o[0]  = x[0]  * scalew;
//                 o[1]  = x[1]  * scalew;
//                 o[2]  = x[2]  * scalew;
//                 o[3]  = x[3]  * scalew;
//                 o[4]  = x[4]  * scalew;
//                 o[5]  = x[5]  * scalew;
//                 o[6]  = x[6]  * scalew;
//                 o[7]  = x[7]  * scalew;
//                 o[8]  = x[8]  * scalew;
//                 o[9]  = x[9]  * scalew;
//                 o[10] = x[10] * scalew;
//                 o[11] = x[11] * scalew;
//                 o[12] = x[12] * scalew;
//                 o[13] = x[13] * scalew;
//                 o[14] = x[14] * scalew;
//                 o[15] = x[15] * scalew;
//             }
//         }

//     }
   
// }




// #include <immintrin.h>
// #include <cmath>
// #include <algorithm>
// #include <type_traits>

// template <typename scalar_t, bool HasWeight>
// void rms_norm_kernel_intrins(
//     const scalar_t* __restrict__ X,        // [groups][M][16]
//     const scalar_t* __restrict__ X_sel,    // [groups][M][8]
//     scalar_t* __restrict__ O,              // [groups][M][16]
//     const scalar_t* __restrict__ W,        // [M]
//     int64_t groups,
//     int64_t M,
//     double eps
// ) {
//     const scalar_t eps_t = static_cast<scalar_t>(eps);

//     if constexpr (std::is_same_v<scalar_t, float>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const float* group_x   = X + g * (M << 4);
//             const float* group_sel = X_sel + g * (M << 3);
//             float* group_o         = O + g * (M << 4);

//             __m256 acc = _mm256_setzero_ps();

//             int64_t m = 0;

//             // ---------------------------
//             // vectorized RMS over X_sel
//             // ---------------------------
//             for (; m + 7 < M; m += 8) {

//                 const float* s0 = group_sel + (m << 3);
//                 const float* s1 = s0 + 8;
//                 const float* s2 = s1 + 8;
//                 const float* s3 = s2 + 8;
//                 const float* s4 = s3 + 8;
//                 const float* s5 = s4 + 8;
//                 const float* s6 = s5 + 8;
//                 const float* s7 = s6 + 8;

//                 __m256 v0 = _mm256_loadu_ps(s0);
//                 __m256 v1 = _mm256_loadu_ps(s1);
//                 __m256 v2 = _mm256_loadu_ps(s2);
//                 __m256 v3 = _mm256_loadu_ps(s3);
//                 __m256 v4 = _mm256_loadu_ps(s4);
//                 __m256 v5 = _mm256_loadu_ps(s5);
//                 __m256 v6 = _mm256_loadu_ps(s6);
//                 __m256 v7 = _mm256_loadu_ps(s7);

//                 acc = _mm256_fmadd_ps(v0, v0, acc);
//                 acc = _mm256_fmadd_ps(v1, v1, acc);
//                 acc = _mm256_fmadd_ps(v2, v2, acc);
//                 acc = _mm256_fmadd_ps(v3, v3, acc);
//                 acc = _mm256_fmadd_ps(v4, v4, acc);
//                 acc = _mm256_fmadd_ps(v5, v5, acc);
//                 acc = _mm256_fmadd_ps(v6, v6, acc);
//                 acc = _mm256_fmadd_ps(v7, v7, acc);
//             }

//             // ---------------------------
//             // horizontal reduce
//             // ---------------------------
//             __m128 lo = _mm256_castps256_ps128(acc);
//             __m128 hi = _mm256_extractf128_ps(acc, 1);

//             __m128 sum = _mm_add_ps(lo, hi);
//             sum = _mm_hadd_ps(sum, sum);
//             sum = _mm_hadd_ps(sum, sum);

//             float accs = _mm_cvtss_f32(sum);

//             // tail
//             for (; m < M; ++m) {
//                 const float* s = group_sel + (m << 3);

//                 accs +=
//                     s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3] +
//                     s[4]*s[4] + s[5]*s[5] + s[6]*s[6] + s[7]*s[7];
//             }

//             accs /= static_cast<float>(M);

//             float scale = 1.0f / std::sqrt(std::max(accs, eps_t));

//             // ---------------------------
//             // writeback using original X
//             // ---------------------------
//             for (int64_t i = 0; i < M; ++i) {

//                 const float* x = group_x + (i << 4);
//                 float* o       = group_o + (i << 4);

//                 float scalew = scale;

//                 if constexpr (HasWeight) {
//                     scalew *= W[i];
//                 }

//                 __m256 vscale = _mm256_set1_ps(scalew);

//                 __m256 v0 = _mm256_loadu_ps(x + 0);
//                 __m256 v1 = _mm256_loadu_ps(x + 8);

//                 v0 = _mm256_mul_ps(v0, vscale);
//                 v1 = _mm256_mul_ps(v1, vscale);

//                 _mm256_storeu_ps(o + 0, v0);
//                 _mm256_storeu_ps(o + 8, v1);
//             }
//         }

//     } else if constexpr (std::is_same_v<scalar_t, double>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const double* group_x   = X + g * (M << 4);
//             const double* group_sel = X_sel + g * (M << 3);
//             double* group_o         = O + g * (M << 4);

//             __m256d acc = _mm256_setzero_pd();

//             int64_t m = 0;

//             for (; m + 3 < M; m += 4) {

//                 const double* s0 = group_sel + (m << 3);
//                 const double* s1 = s0 + 8;
//                 const double* s2 = s1 + 8;
//                 const double* s3 = s2 + 8;

//                 __m256d v0 = _mm256_loadu_pd(s0);
//                 __m256d v1 = _mm256_loadu_pd(s1);
//                 __m256d v2 = _mm256_loadu_pd(s2);
//                 __m256d v3 = _mm256_loadu_pd(s3);

//                 acc = _mm256_fmadd_pd(v0, v0, acc);
//                 acc = _mm256_fmadd_pd(v1, v1, acc);
//                 acc = _mm256_fmadd_pd(v2, v2, acc);
//                 acc = _mm256_fmadd_pd(v3, v3, acc);
//             }

//             __m128d lo = _mm256_castpd256_pd128(acc);
//             __m128d hi = _mm256_extractf128_pd(acc, 1);

//             __m128d sum = _mm_add_pd(lo, hi);

//             double accs =
//                 _mm_cvtsd_f64(sum) +
//                 _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));

//             for (; m < M; ++m) {
//                 const double* s = group_sel + (m << 3);

//                 accs +=
//                     s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3] +
//                     s[4]*s[4] + s[5]*s[5] + s[6]*s[6] + s[7]*s[7];
//             }

//             accs /= static_cast<double>(M);

//             double scale = 1.0 / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {

//                 const double* x = group_x + (i << 4);
//                 double* o       = group_o + (i << 4);

//                 double scalew = scale;

//                 if constexpr (HasWeight) {
//                     scalew *= W[i];
//                 }

//                 __m256d vscale = _mm256_set1_pd(scalew);

//                 __m256d v0 = _mm256_loadu_pd(x + 0);
//                 __m256d v1 = _mm256_loadu_pd(x + 4);

//                 v0 = _mm256_mul_pd(v0, vscale);
//                 v1 = _mm256_mul_pd(v1, vscale);

//                 _mm256_storeu_pd(o + 0, v0);
//                 _mm256_storeu_pd(o + 4, v1);
//             }
//         }
//     }
// }













template <typename scalar_t, bool HasWeight>
void rms_norm_kernel_intrins(
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ O,
    const scalar_t* __restrict__ W,
    int64_t groups,
    int64_t M,
    double eps
) {
    const scalar_t eps_t = static_cast<scalar_t>(eps);

    if constexpr (std::is_same_v<scalar_t, float>) {

        for (int64_t g = 0; g < groups; ++g) {

            const scalar_t* group_in  = X + g * (M << 4);
            scalar_t* group_out       = O + g * (M << 4);

            float32x4_t acc = vdupq_n_f32(0.0f);

            int64_t m = 0;

            // ---- SIMD accumulation (4 lanes = 4 m values) ----
            for (; m + 3 < M; m += 4) {

                const float* x0 = group_in + ((m + 0) << 4);
                const float* x1 = group_in + ((m + 1) << 4);
                const float* x2 = group_in + ((m + 2) << 4);
                const float* x3 = group_in + ((m + 3) << 4);

                float32x4_t v0  = { x0[0],  x1[0],  x2[0],  x3[0] };
                float32x4_t v2  = { x0[2],  x1[2],  x2[2],  x3[2] };
                float32x4_t v3  = { x0[3],  x1[3],  x2[3],  x3[3] };
                float32x4_t v4  = { x0[4],  x1[4],  x2[4],  x3[4] };

                float32x4_t v8  = { x0[8],  x1[8],  x2[8],  x3[8] };
                float32x4_t v9  = { x0[9],  x1[9],  x2[9],  x3[9] };
                float32x4_t v10 = { x0[10], x1[10], x2[10], x3[10] };
                float32x4_t v14 = { x0[14], x1[14], x2[14], x3[14] };

                acc = vfmaq_f32(acc, v0,  v0);
                acc = vfmaq_f32(acc, v2,  v2);
                acc = vfmaq_f32(acc, v3,  v3);
                acc = vfmaq_f32(acc, v4,  v4);
                acc = vfmaq_f32(acc, v8,  v8);
                acc = vfmaq_f32(acc, v9,  v9);
                acc = vfmaq_f32(acc, v10, v10);
                acc = vfmaq_f32(acc, v14, v14);
            }

            // ---- horizontal reduction ----
            float accs = vaddvq_f32(acc);

            // ---- scalar tail ----
            for (; m < M; ++m) {
                const float* x = group_in + (m << 4);

                accs +=
                    x[0]*x[0] +
                    x[2]*x[2] +
                    x[3]*x[3] +
                    x[4]*x[4] +
                    x[8]*x[8] +
                    x[9]*x[9] +
                    x[10]*x[10] +
                    x[14]*x[14];
            }

            accs /= static_cast<float>(M);

            float scale = 1.0f / std::sqrt(std::max(accs, eps_t));

            // ---- write ----
            for (int64_t m = 0; m < M; ++m) {

                const float* x = group_in + (m << 4);
                float* o       = group_out + (m << 4);

                float scalew = scale;

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

    } else if constexpr (std::is_same_v<scalar_t, double>) {

        for (int64_t g = 0; g < groups; ++g) {

            const scalar_t* group_in  = X + g * (M << 4);
            scalar_t* group_out       = O + g * (M << 4);

            float64x2_t acc = vdupq_n_f64(0.0);

            int64_t m = 0;

            for (; m + 1 < M; m += 2) {

                const double* x0 = group_in + ((m + 0) << 4);
                const double* x1 = group_in + ((m + 1) << 4);

                float64x2_t v0  = { x0[0],  x1[0] };
                float64x2_t v2  = { x0[2],  x1[2] };
                float64x2_t v3  = { x0[3],  x1[3] };
                float64x2_t v4  = { x0[4],  x1[4] };

                float64x2_t v8  = { x0[8],  x1[8] };
                float64x2_t v9  = { x0[9],  x1[9] };
                float64x2_t v10 = { x0[10], x1[10] };
                float64x2_t v14 = { x0[14], x1[14] };

                acc = vfmaq_f64(acc, v0,  v0);
                acc = vfmaq_f64(acc, v2,  v2);
                acc = vfmaq_f64(acc, v3,  v3);
                acc = vfmaq_f64(acc, v4,  v4);
                acc = vfmaq_f64(acc, v8,  v8);
                acc = vfmaq_f64(acc, v9,  v9);
                acc = vfmaq_f64(acc, v10, v10);
                acc = vfmaq_f64(acc, v14, v14);
            }

            double accs = vaddvq_f64(acc);

            for (; m < M; ++m) {
                const double* x = group_in + (m << 4);

                accs +=
                    x[0]*x[0] +
                    x[2]*x[2] +
                    x[3]*x[3] +
                    x[4]*x[4] +
                    x[8]*x[8] +
                    x[9]*x[9] +
                    x[10]*x[10] +
                    x[14]*x[14];
            }

            accs /= static_cast<double>(M);

            double scale = 1.0 / std::sqrt(std::max(accs, eps_t));

            for (int64_t m = 0; m < M; ++m) {

                const double* x = group_in + (m << 4);
                double* o       = group_out + (m << 4);

                double scalew = scale;

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


    bool use_intrins =
        (xc.scalar_type() == torch::kFloat32 ||
         xc.scalar_type() == torch::kFloat64);

    if (!use_intrins) {
        //FALLBACK PATH
        AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel", [&] {
            if (weight.has_value()) {
                rms_norm_kernel<scalar_t, true>(
                    xc.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    weight->data_ptr<scalar_t>(),
                    groups, M, eps
                );
            } else {
                rms_norm_kernel<scalar_t, false>(
                    xc.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    nullptr,
                    groups, M, eps
                );
            }
        });
        return out;
    }

    //FAST PATH (intrinsics)
    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel_intrins", [&] {
        if (weight.has_value()) {
            rms_norm_kernel_intrins<scalar_t, true>(
                xc.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                weight->data_ptr<scalar_t>(),
                groups, M, eps
            );
        } else {
            rms_norm_kernel_intrins<scalar_t, false>(
                xc.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                nullptr,
                groups, M, eps
            );
        }
    });

    return out;
}

// torch::Tensor equi_rms_norm_ver_3(
//     const torch::Tensor& x,
//     const c10::optional<torch::Tensor>& weight,
//     const c10::optional<double>& eps_opt
// ) {
//     check_multivector(x, "equi_rms_norm x");

//     double eps = eps_opt.has_value() ? *eps_opt : 1e-7;

//     auto xc = x.is_contiguous() ? x : x.contiguous();
//     auto out = torch::empty_like(xc);

//     int64_t M = x.size(-2);

//     int64_t groups = 1;
//     for (int64_t i = 0; i < x.dim() - 2; ++i)
//         groups *= x.size(i);

//     bool use_intrins =
//         (xc.scalar_type() == torch::kFloat32 ||
//          xc.scalar_type() == torch::kFloat64);

//     // -----------------------------
//     // BUILD PACKED X_SEL (NEW)
//     // -----------------------------
//     torch::Tensor x_sel = torch::empty({groups, 8, M}, xc.options());

//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "pack_groups_16_to_8_soa", [&] {
//         pack_groups_16_to_8_soa<scalar_t>(
//             xc.data_ptr<scalar_t>(),
//             x_sel.data_ptr<scalar_t>(),
//             groups,
//             M
//         );
//     });

//     // -----------------------------
//     // FALLBACK PATH
//     // -----------------------------
//     if (!use_intrins) {

//         AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel", [&] {
//             if (weight.has_value()) {
//                 rms_norm_kernel<scalar_t, true>(
//                     xc.data_ptr<scalar_t>(),
//                     out.data_ptr<scalar_t>(),
//                     weight->data_ptr<scalar_t>(),
//                     groups, M, eps
//                 );
//             } else {
//                 rms_norm_kernel<scalar_t, false>(
//                     xc.data_ptr<scalar_t>(),
//                     out.data_ptr<scalar_t>(),
//                     nullptr,
//                     groups, M, eps
//                 );
//             }
//         });

//         return out;
//     }

//     // -----------------------------
//     // FAST PATH (AVX2)
//     // -----------------------------
//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel_intrins", [&] {
//         if (weight.has_value()) {
//             rms_norm_kernel_intrins<scalar_t, true>(
//                 xc.data_ptr<scalar_t>(),
//                 x_sel.data_ptr<scalar_t>(),   // <-- NEW
//                 out.data_ptr<scalar_t>(),
//                 weight->data_ptr<scalar_t>(),
//                 groups,
//                 M,
//                 eps
//             );
//         } else {
//             rms_norm_kernel_intrins<scalar_t, false>(
//                 xc.data_ptr<scalar_t>(),
//                 x_sel.data_ptr<scalar_t>(),   // <-- NEW
//                 out.data_ptr<scalar_t>(),
//                 nullptr,
//                 groups,
//                 M,
//                 eps
//             );
//         }
//     });

//     return out;
// }



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
