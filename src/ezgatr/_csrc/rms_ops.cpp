#include "rms_ops.h"
#include "basis_data.h"

#include <map>
#include <mutex>
#include <tuple>





#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
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

#if defined(__AVX2__) && defined(__FMA__)
// Vectorized expf (Cephes minimax, ~1 ulp for float32). Standard range reduction
// e^x = 2^k * e^r with k = round(x*log2(e)) and r = x - k*ln2.
static inline __m256 exp256_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 C1 = _mm256_set1_ps(0.693359375f);      // ln2 split, hi part
    const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f);   // ln2 split, lo part
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 half = _mm256_set1_ps(0.5f);

    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, LOG2EF, half));
    // r = x - fx*ln2  (two-step to preserve precision)
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, C1));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, C2));

    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    // 2^fx via direct exponent-field construction.
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_slli_epi32(_mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}

// tanh(x) = 1 - 2/(e^{2x}+1). Accurate to ~1e-6 in float32 over the gelu range.
static inline __m256 tanh256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 two = _mm256_set1_ps(2.f);
    __m256 e = exp256_ps(_mm256_mul_ps(x, two));
    return _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e, one)));
}

// AVX2 scaler-gated GELU (tanh approx, float32). Computes 8 gates per iteration
// with a vectorized tanh, then broadcasts each gate across its 16-lane
// multivector. The expensive transcendental is amortized 8-wide instead of the
// per-multivector scalar std::tanh call in the v0-v2 kernel.
static void gelu_gate_kernel_avx2(const float* __restrict__ X,
                                  float* __restrict__ O, int64_t N) {
    const __m256 half   = _mm256_set1_ps(0.5f);
    const __m256 one    = _mm256_set1_ps(1.f);
    const __m256 kAlpha = _mm256_set1_ps(0.7978845608028654f);
    const __m256 kBeta  = _mm256_set1_ps(0.044715f);
    // Scalar blade of 8 consecutive multivectors: float offsets 0,16,...,112.
    const __m256i sidx = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

    int64_t n = 0;
    for (; n + 8 <= N; n += 8) {
        const float* base = X + (n << 4);
        const __m256 s  = _mm256_i32gather_ps(base, sidx, 4);
        const __m256 s2 = _mm256_mul_ps(s, s);
        // arg = alpha * s * (1 + beta*s^2)
        const __m256 arg = _mm256_mul_ps(_mm256_mul_ps(kAlpha, s),
                                         _mm256_fmadd_ps(kBeta, s2, one));
        const __m256 gate = _mm256_mul_ps(_mm256_mul_ps(half, s),
                                          _mm256_add_ps(one, tanh256_ps(arg)));
        alignas(32) float g[8];
        _mm256_store_ps(g, gate);
        for (int j = 0; j < 8; ++j) {
            const float* xi = base + (j << 4);
            float* oi = O + ((n + j) << 4);
            const __m256 gj = _mm256_set1_ps(g[j]);
            _mm256_storeu_ps(oi,     _mm256_mul_ps(_mm256_loadu_ps(xi),     gj));
            _mm256_storeu_ps(oi + 8, _mm256_mul_ps(_mm256_loadu_ps(xi + 8), gj));
        }
    }
    // Scalar tail (< 8 multivectors).
    for (; n < N; ++n) {
        const float* xi = X + (n << 4);
        float* oi = O + (n << 4);
        const float sc = xi[0];
        const float gate = 0.5f * sc *
            (1.f + std::tanh(0.7978845608028654f * (sc + 0.044715f * sc * sc * sc)));
        for (int i = 0; i < 16; ++i) oi[i] = xi[i] * gate;
    }
}
#endif  // __AVX2__ && __FMA__


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
        // Active blades: 0,2,3,4 (lo 8) and 8,9,10,14 (hi 8).
        // Load all 16 floats per row with two vmovups, apply mask to zero
        // inactive lanes, then fmadd — avoids the expensive _mm256_set_ps
        // scatter-gather (8 scalar extractions per vector) and works for
        // any M including M=4 where the old 8-row loop never fired.
        const __m256 mask_lo = _mm256_setr_ps(1.f,0.f,1.f,1.f,1.f,0.f,0.f,0.f);
        const __m256 mask_hi = _mm256_setr_ps(1.f,1.f,1.f,0.f,0.f,0.f,1.f,0.f);

        for (int64_t g = 0; g < groups; ++g) {
            const float* group_in  = X + g * (M << 4);
            float*       group_out = O + g * (M << 4);

            // Pass 1: masked reduction with 4 independent accumulators to
            // hide the 4-cycle FMA latency on Tiger Lake.
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int64_t m = 0;
            for (; m + 3 < M; m += 4) {
                const float* x0 = group_in + (m    ) * 16;
                const float* x1 = group_in + (m + 1) * 16;
                const float* x2 = group_in + (m + 2) * 16;
                const float* x3 = group_in + (m + 3) * 16;
                __m256 lo0 = _mm256_loadu_ps(x0), hi0 = _mm256_loadu_ps(x0 + 8);
                __m256 lo1 = _mm256_loadu_ps(x1), hi1 = _mm256_loadu_ps(x1 + 8);
                __m256 lo2 = _mm256_loadu_ps(x2), hi2 = _mm256_loadu_ps(x2 + 8);
                __m256 lo3 = _mm256_loadu_ps(x3), hi3 = _mm256_loadu_ps(x3 + 8);
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(mask_lo, lo0), lo0, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(mask_hi, hi0), hi0, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_mul_ps(mask_lo, lo1), lo1, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_mul_ps(mask_hi, hi1), hi1, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_mul_ps(mask_lo, lo2), lo2, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_mul_ps(mask_hi, hi2), hi2, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_mul_ps(mask_lo, lo3), lo3, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_mul_ps(mask_hi, hi3), hi3, acc3);
            }
            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                 _mm256_add_ps(acc2, acc3));
            for (; m < M; ++m) {
                const float* x = group_in + (m << 4);
                __m256 lo = _mm256_loadu_ps(x), hi = _mm256_loadu_ps(x + 8);
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(mask_lo, lo), lo, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(mask_hi, hi), hi, acc0);
            }
            __m128 lo128 = _mm256_castps256_ps128(acc0);
            __m128 hi128 = _mm256_extractf128_ps(acc0, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float accs = _mm_cvtss_f32(sum128);

            accs /= static_cast<float>(M);
            accs = std::max(accs, eps_t);
            // rsqrt + one Newton-Raphson step gives ~23-bit accuracy,
            // saving ~8 cycles vs sqrtss+divss per group.
            const float half = 0.5f * accs;
            __m128 vr = _mm_rsqrt_ss(_mm_set_ss(accs));
            // r = r * (1.5 - 0.5*accs*r*r)
            vr = _mm_mul_ss(vr,
                 _mm_sub_ss(_mm_set_ss(1.5f),
                 _mm_mul_ss(_mm_set_ss(half),
                 _mm_mul_ss(vr, vr))));
            float scale = _mm_cvtss_f32(vr);

            // Pass 2: scale write-back with optional per-channel weight.
            for (int64_t m = 0; m < M; ++m) {
                const float* x = group_in + (m << 4);
                float*       o = group_out + (m << 4);
                float scalew   = scale;
                if constexpr (HasWeight) scalew *= W[m];
                __m256 vs = _mm256_set1_ps(scalew);
                _mm256_storeu_ps(o,     _mm256_mul_ps(_mm256_loadu_ps(x),     vs));
                _mm256_storeu_ps(o + 8, _mm256_mul_ps(_mm256_loadu_ps(x + 8), vs));
            }
        }
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        // Same mask-based approach for double. Active blades: 0,2,3,4 in lo
        // quad and 8,9,10,14 (relative positions 0,1,2,6) in hi quad.
        const __m256d mask_lo = _mm256_setr_pd(1.,0.,1.,1.);  // blades 0,2,3,4 → positions 0,1,2,3 cover 0,1,2,3: active at 0,2,3 with blade 4 split
        // Note: each __m256d holds 4 doubles, so 16-element MV needs 4 vectors.
        // We use 4 independent mask vectors covering positions 0-3, 4-7, 8-11, 12-15.
        const __m256d msk0 = _mm256_setr_pd(1.,0.,1.,1.);   // blades 0,2,3
        const __m256d msk1 = _mm256_setr_pd(1.,0.,0.,0.);   // blade  4
        const __m256d msk2 = _mm256_setr_pd(1.,1.,1.,0.);   // blades 8,9,10
        const __m256d msk3 = _mm256_setr_pd(0.,0.,1.,0.);   // blade  14

        for (int64_t g = 0; g < groups; ++g) {
            const double* group_in  = X + g * (M << 4);
            double*       group_out = O + g * (M << 4);

            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            int64_t m = 0;
            for (; m + 1 < M; m += 2) {
                const double* x0 = group_in + (m    ) * 16;
                const double* x1 = group_in + (m + 1) * 16;
                __m256d a0 = _mm256_loadu_pd(x0),    b0 = _mm256_loadu_pd(x0 + 4);
                __m256d c0 = _mm256_loadu_pd(x0 + 8),d0 = _mm256_loadu_pd(x0 + 12);
                __m256d a1 = _mm256_loadu_pd(x1),    b1 = _mm256_loadu_pd(x1 + 4);
                __m256d c1 = _mm256_loadu_pd(x1 + 8),d1 = _mm256_loadu_pd(x1 + 12);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk0, a0), a0, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk1, b0), b0, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk2, c0), c0, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk3, d0), d0, acc0);
                acc1 = _mm256_fmadd_pd(_mm256_mul_pd(msk0, a1), a1, acc1);
                acc1 = _mm256_fmadd_pd(_mm256_mul_pd(msk1, b1), b1, acc1);
                acc1 = _mm256_fmadd_pd(_mm256_mul_pd(msk2, c1), c1, acc1);
                acc1 = _mm256_fmadd_pd(_mm256_mul_pd(msk3, d1), d1, acc1);
            }
            acc0 = _mm256_add_pd(acc0, acc1);
            for (; m < M; ++m) {
                const double* x = group_in + (m << 4);
                __m256d a = _mm256_loadu_pd(x),    b = _mm256_loadu_pd(x + 4);
                __m256d c = _mm256_loadu_pd(x + 8),d = _mm256_loadu_pd(x + 12);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk0, a), a, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk1, b), b, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk2, c), c, acc0);
                acc0 = _mm256_fmadd_pd(_mm256_mul_pd(msk3, d), d, acc0);
            }
            __m128d lo128 = _mm256_castpd256_pd128(acc0);
            __m128d hi128 = _mm256_extractf128_pd(acc0, 1);
            __m128d sum128 = _mm_add_pd(lo128, hi128);
            double accs = _mm_cvtsd_f64(sum128) +
                          _mm_cvtsd_f64(_mm_unpackhi_pd(sum128, sum128));

            accs /= static_cast<double>(M);
            double scale = 1.0 / std::sqrt(std::max(accs, (double)eps_t));

            for (int64_t i = 0; i < M; ++i) {
                const double* x = group_in + (i << 4);
                double*       o = group_out + (i << 4);
                double scalew   = scale;
                if constexpr (HasWeight) scalew *= W[i];
                __m256d vs = _mm256_set1_pd(scalew);
                // All 16 elements must be written (bug fix: old code only wrote 8).
                _mm256_storeu_pd(o,      _mm256_mul_pd(_mm256_loadu_pd(x),      vs));
                _mm256_storeu_pd(o + 4,  _mm256_mul_pd(_mm256_loadu_pd(x + 4),  vs));
                _mm256_storeu_pd(o + 8,  _mm256_mul_pd(_mm256_loadu_pd(x + 8),  vs));
                _mm256_storeu_pd(o + 12, _mm256_mul_pd(_mm256_loadu_pd(x + 12), vs));
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
//Tested two simd version. one were we use the mamory layout from version 2 
//and use gather intrinsics. also tested a packed version, where we change the
//memory layout such that we can load the vectors directly. 
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






// ---------------------------------------------------------------------------
// Packed rmsnormver3
// ---------------------------------------------------------------------------
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

//     // pack [groups][M][16] -> [groups][8][M]
//     auto x_sel = torch::empty({groups, 8, M}, xc.options());

//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "pack_groups_16_to_8_soa", [&] {
//         pack_groups_16_to_8_soa<scalar_t>(
//             xc.data_ptr<scalar_t>(),
//             x_sel.data_ptr<scalar_t>(),
//             groups, M
//         );
//     });

//     AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rms_norm_kernel_packed", [&] {
//         if (weight.has_value()) {
//             rms_norm_kernel_packed<scalar_t, true>(
//                 xc.data_ptr<scalar_t>(),
//                 x_sel.data_ptr<scalar_t>(),
//                 out.data_ptr<scalar_t>(),
//                 weight->contiguous().data_ptr<scalar_t>(),
//                 groups, M, eps
//             );
//         } else {
//             rms_norm_kernel_packed<scalar_t, false>(
//                 xc.data_ptr<scalar_t>(),
//                 x_sel.data_ptr<scalar_t>(),
//                 out.data_ptr<scalar_t>(),
//                 nullptr,
//                 groups, M, eps
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



// torch::Tensor scaler_gated_gelu_ver_1(const torch::Tensor& x,
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
//         //int64_t N = x.size(-2);
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



// torch::Tensor scaler_gated_gelu_ver_2(
//     const torch::Tensor& x,
//     const std::string& approximate
// ){
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
//         //int64_t N = x.size(-2);

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



// ver_3 — AVX2-vectorized scaler-gated GELU (float32 + "tanh" fast path).
// Falls back to ver_2 for float64, the "none"/erf variant, or non-AVX2 builds.
torch::Tensor scaler_gated_gelu_ver_3(
    const torch::Tensor& x,
    const std::string& approximate
){
    check_multivector(x, "scalar_gated_gelu: x");
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "scalar_gated_gelu: approximate must be 'none' or 'tanh' string",
        approximate
    );

#if defined(__AVX2__) && defined(__FMA__)
    if (approximate == "tanh" && x.scalar_type() == torch::kFloat) {
        auto xc = x.contiguous();
        auto out = torch::empty_like(xc);
        const int64_t N = xc.numel() / 16;
        gelu_gate_kernel_avx2(xc.data_ptr<float>(), out.data_ptr<float>(), N);
        return out;
    }
#endif
    // float64 / "none" / non-AVX2: numerically identical scalar path.
    return scaler_gated_gelu_ver_2(x, approximate);
}




}}  // namespace ezgatr::opt



//DUMP of older versions

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

//             const scalar_t* group_in = X + g * (M << 4);
//             scalar_t*       group_out = O + g * (M << 4);

//             // 4 independent accumulator chains → ILP
//             __m256 acc0 = _mm256_setzero_ps();
//             __m256 acc1 = _mm256_setzero_ps();
//             __m256 acc2 = _mm256_setzero_ps();
//             __m256 acc3 = _mm256_setzero_ps();

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

//                 // Distribute FMAs across 4 chains — no RAW dependency between them
//                 acc0 = _mm256_fmadd_ps(v0,  v0,  acc0);
//                 acc1 = _mm256_fmadd_ps(v2,  v2,  acc1);
//                 acc2 = _mm256_fmadd_ps(v3,  v3,  acc2);
//                 acc3 = _mm256_fmadd_ps(v4,  v4,  acc3);

//                 acc0 = _mm256_fmadd_ps(v8,  v8,  acc0);
//                 acc1 = _mm256_fmadd_ps(v9,  v9,  acc1);
//                 acc2 = _mm256_fmadd_ps(v10, v10, acc2);
//                 acc3 = _mm256_fmadd_ps(v14, v14, acc3);
//             }

//             // Merge the 4 chains
//             __m256 acc = _mm256_add_ps(
//                 _mm256_add_ps(acc0, acc1),
//                 _mm256_add_ps(acc2, acc3));

//             __m128 lo  = _mm256_castps256_ps128(acc);
//             __m128 hi  = _mm256_extractf128_ps(acc, 1);
//             __m128 sum = _mm_add_ps(lo, hi);
//             sum = _mm_hadd_ps(sum, sum);
//             sum = _mm_hadd_ps(sum, sum);

//             scalar_t accs = _mm_cvtss_f32(sum);

//             for (; m < M; ++m) {
//                 const scalar_t* x = group_in + (m << 4);
//                 accs +=
//                     x[0]*x[0]  + x[2]*x[2]  +
//                     x[3]*x[3]  + x[4]*x[4]  +
//                     x[8]*x[8]  + x[9]*x[9]  +
//                     x[10]*x[10] + x[14]*x[14];
//             }

//             accs /= static_cast<scalar_t>(M);
//             scalar_t scale = scalar_t(1) / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const float* x = group_in  + (i << 4);
//                 float*       o = group_out + (i << 4);

//                 float scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 __m256 vscale = _mm256_set1_ps(scalew);
//                 _mm256_storeu_ps(o + 0, _mm256_mul_ps(_mm256_loadu_ps(x + 0), vscale));
//                 _mm256_storeu_ps(o + 8, _mm256_mul_ps(_mm256_loadu_ps(x + 8), vscale));
//             }
//         }

//     } else if constexpr (std::is_same_v<scalar_t, double>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const scalar_t* group_in  = X + g * (M << 4);
//             scalar_t*       group_out = O + g * (M << 4);

//             __m256d acc0 = _mm256_setzero_pd();
//             __m256d acc1 = _mm256_setzero_pd();
//             __m256d acc2 = _mm256_setzero_pd();
//             __m256d acc3 = _mm256_setzero_pd();

//             int64_t m = 0;

//             // Double: 4 lanes → unroll by 16 to keep 4 accumulators busy
//             for (; m + 15 < M; m += 16) {

//                 // acc0: blocks 0–3
//                 const scalar_t* x0  = group_in + (m      << 4);
//                 const scalar_t* x1  = x0  + 16;
//                 const scalar_t* x2  = x1  + 16;
//                 const scalar_t* x3  = x2  + 16;
//                 // acc1: blocks 4–7
//                 const scalar_t* x4  = x3  + 16;
//                 const scalar_t* x5  = x4  + 16;
//                 const scalar_t* x6  = x5  + 16;
//                 const scalar_t* x7  = x6  + 16;
//                 // acc2: blocks 8–11
//                 const scalar_t* x8  = x7  + 16;
//                 const scalar_t* x9  = x8  + 16;
//                 const scalar_t* x10 = x9  + 16;
//                 const scalar_t* x11 = x10 + 16;
//                 // acc3: blocks 12–15
//                 const scalar_t* x12 = x11 + 16;
//                 const scalar_t* x13 = x12 + 16;
//                 const scalar_t* x14 = x13 + 16;
//                 const scalar_t* x15 = x14 + 16;

//                 // --- acc0: rows 0–3 ---
//                 #define ACCPD4(acc, xa, xb, xc, xd, idx) do { \
//                     __m256d _v = _mm256_set_pd((xd)[idx], (xc)[idx], (xb)[idx], (xa)[idx]); \
//                     acc = _mm256_fmadd_pd(_v, _v, acc); \
//                 } while(0)

//                 ACCPD4(acc0, x0, x1, x2, x3,  0);
//                 ACCPD4(acc0, x0, x1, x2, x3,  2);
//                 ACCPD4(acc0, x0, x1, x2, x3,  3);
//                 ACCPD4(acc0, x0, x1, x2, x3,  4);
//                 ACCPD4(acc0, x0, x1, x2, x3,  8);
//                 ACCPD4(acc0, x0, x1, x2, x3,  9);
//                 ACCPD4(acc0, x0, x1, x2, x3, 10);
//                 ACCPD4(acc0, x0, x1, x2, x3, 14);

//                 // --- acc1: rows 4–7 ---
//                 ACCPD4(acc1, x4, x5, x6, x7,  0);
//                 ACCPD4(acc1, x4, x5, x6, x7,  2);
//                 ACCPD4(acc1, x4, x5, x6, x7,  3);
//                 ACCPD4(acc1, x4, x5, x6, x7,  4);
//                 ACCPD4(acc1, x4, x5, x6, x7,  8);
//                 ACCPD4(acc1, x4, x5, x6, x7,  9);
//                 ACCPD4(acc1, x4, x5, x6, x7, 10);
//                 ACCPD4(acc1, x4, x5, x6, x7, 14);

//                 // --- acc2: rows 8–11 ---
//                 ACCPD4(acc2, x8, x9, x10, x11,  0);
//                 ACCPD4(acc2, x8, x9, x10, x11,  2);
//                 ACCPD4(acc2, x8, x9, x10, x11,  3);
//                 ACCPD4(acc2, x8, x9, x10, x11,  4);
//                 ACCPD4(acc2, x8, x9, x10, x11,  8);
//                 ACCPD4(acc2, x8, x9, x10, x11,  9);
//                 ACCPD4(acc2, x8, x9, x10, x11, 10);
//                 ACCPD4(acc2, x8, x9, x10, x11, 14);

//                 // --- acc3: rows 12–15 ---
//                 ACCPD4(acc3, x12, x13, x14, x15,  0);
//                 ACCPD4(acc3, x12, x13, x14, x15,  2);
//                 ACCPD4(acc3, x12, x13, x14, x15,  3);
//                 ACCPD4(acc3, x12, x13, x14, x15,  4);
//                 ACCPD4(acc3, x12, x13, x14, x15,  8);
//                 ACCPD4(acc3, x12, x13, x14, x15,  9);
//                 ACCPD4(acc3, x12, x13, x14, x15, 10);
//                 ACCPD4(acc3, x12, x13, x14, x15, 14);

//                 #undef ACCPD4
//             }

//             // Scalar fallback for any remaining blocks (original stride-4 loop)
//             for (; m + 3 < M; m += 4) {
//                 const scalar_t* x0 = group_in + (m << 4);
//                 const scalar_t* x1 = x0 + 16;
//                 const scalar_t* x2 = x1 + 16;
//                 const scalar_t* x3 = x2 + 16;

//                 #define ACCPD4F(acc, idx) do { \
//                     __m256d _v = _mm256_set_pd(x3[idx], x2[idx], x1[idx], x0[idx]); \
//                     acc = _mm256_fmadd_pd(_v, _v, acc); \
//                 } while(0)

//                 ACCPD4F(acc0,  0); ACCPD4F(acc1,  2);
//                 ACCPD4F(acc2,  3); ACCPD4F(acc3,  4);
//                 ACCPD4F(acc0,  8); ACCPD4F(acc1,  9);
//                 ACCPD4F(acc2, 10); ACCPD4F(acc3, 14);

//                 #undef ACCPD4F
//             }

//             // Merge
//             __m256d acc = _mm256_add_pd(
//                 _mm256_add_pd(acc0, acc1),
//                 _mm256_add_pd(acc2, acc3));

//             __m128d lo  = _mm256_castpd256_pd128(acc);
//             __m128d hi  = _mm256_extractf128_pd(acc, 1);
//             __m128d sum = _mm_add_pd(lo, hi);

//             scalar_t accs =
//                 _mm_cvtsd_f64(sum) +
//                 _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));

//             for (; m < M; ++m) {
//                 const scalar_t* x = group_in + (m << 4);
//                 accs +=
//                     x[0]*x[0]  + x[2]*x[2]  +
//                     x[3]*x[3]  + x[4]*x[4]  +
//                     x[8]*x[8]  + x[9]*x[9]  +
//                     x[10]*x[10] + x[14]*x[14];
//             }

//             accs /= static_cast<scalar_t>(M);
//             scalar_t scale = scalar_t(1) / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const double* x = group_in  + (i << 4);
//                 double*       o = group_out + (i << 4);

//                 double scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 __m256d vscale = _mm256_set1_pd(scalew);
//                 _mm256_storeu_pd(o + 0, _mm256_mul_pd(_mm256_loadu_pd(x + 0), vscale));
//                 _mm256_storeu_pd(o + 4, _mm256_mul_pd(_mm256_loadu_pd(x + 4), vscale));
//                 _mm256_storeu_pd(o + 8, _mm256_mul_pd(_mm256_loadu_pd(x + 8), vscale));
//                 _mm256_storeu_pd(o + 12, _mm256_mul_pd(_mm256_loadu_pd(x + 12), vscale));
//             }
//         }
//     }
// }








// //normal intrins intel
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
//             // for (int64_t m = 0; m < M; ++m) {

//             //     const scalar_t* x = group_in + (m << 4);
//             //     scalar_t* o       = group_out + (m << 4);

//             //     scalar_t scalew = scale;

//             //     if constexpr (HasWeight) {
//             //         scalew *= W[m];
//             //     }

//             //     o[0]  = x[0]  * scalew;
//             //     o[1]  = x[1]  * scalew;
//             //     o[2]  = x[2]  * scalew;
//             //     o[3]  = x[3]  * scalew;
//             //     o[4]  = x[4]  * scalew;
//             //     o[5]  = x[5]  * scalew;
//             //     o[6]  = x[6]  * scalew;
//             //     o[7]  = x[7]  * scalew;
//             //     o[8]  = x[8]  * scalew;
//             //     o[9]  = x[9]  * scalew;
//             //     o[10] = x[10] * scalew;
//             //     o[11] = x[11] * scalew;
//             //     o[12] = x[12] * scalew;
//             //     o[13] = x[13] * scalew;
//             //     o[14] = x[14] * scalew;
//             //     o[15] = x[15] * scalew;
//             // }

//             for (int64_t m = 0; m < M; ++m) {
//                 const float* x = group_in + (m << 4);
//                 float* o       = group_out + (m << 4);

//                 float scalew = scale;

//                 if constexpr (HasWeight) {
//                     scalew *= W[m];
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
//             // for (int64_t m = 0; m < M; ++m) {

//             //     const scalar_t* x = group_in + (m << 4);
//             //     scalar_t* o       = group_out + (m << 4);

//             //     scalar_t scalew = scale;

//             //     if constexpr (HasWeight) {
//             //         scalew *= W[m];
//             //     }

//             //     o[0]  = x[0]  * scalew;
//             //     o[1]  = x[1]  * scalew;
//             //     o[2]  = x[2]  * scalew;
//             //     o[3]  = x[3]  * scalew;
//             //     o[4]  = x[4]  * scalew;
//             //     o[5]  = x[5]  * scalew;
//             //     o[6]  = x[6]  * scalew;
//             //     o[7]  = x[7]  * scalew;
//             //     o[8]  = x[8]  * scalew;
//             //     o[9]  = x[9]  * scalew;
//             //     o[10] = x[10] * scalew;
//             //     o[11] = x[11] * scalew;
//             //     o[12] = x[12] * scalew;
//             //     o[13] = x[13] * scalew;
//             //     o[14] = x[14] * scalew;
//             //     o[15] = x[15] * scalew;
//             // }

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




// ---------------------------------------------------------------------------
// Packed SOA AVX2 kernel
// ---------------------------------------------------------------------------
// template <typename scalar_t, bool HasWeight>
// void rms_norm_kernel_packed(
//     const scalar_t* __restrict__ X,
//     const scalar_t* __restrict__ X_sel,
//     scalar_t* __restrict__ O,
//     const scalar_t* __restrict__ W,
//     int64_t groups,
//     int64_t M,
//     double eps
// ) {
//     const scalar_t eps_t = static_cast<scalar_t>(eps);

//     if constexpr (std::is_same_v<scalar_t, float>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const float* gx   = X     + g * (M << 4);
//             const float* gsel = X_sel + g * (8 * M);
//             float*       go   = O     + g * (M << 4);

//             const float* s0 = gsel + 0 * M;
//             const float* s1 = gsel + 1 * M;
//             const float* s2 = gsel + 2 * M;
//             const float* s3 = gsel + 3 * M;
//             const float* s4 = gsel + 4 * M;
//             const float* s5 = gsel + 5 * M;
//             const float* s6 = gsel + 6 * M;
//             const float* s7 = gsel + 7 * M;

//             __m256 acc = _mm256_setzero_ps();
//             int64_t m = 0;

//             for (; m + 7 < M; m += 8) {
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s0 + m), _mm256_loadu_ps(s0 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s1 + m), _mm256_loadu_ps(s1 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s2 + m), _mm256_loadu_ps(s2 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s3 + m), _mm256_loadu_ps(s3 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s4 + m), _mm256_loadu_ps(s4 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s5 + m), _mm256_loadu_ps(s5 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s6 + m), _mm256_loadu_ps(s6 + m), acc);
//                 acc = _mm256_fmadd_ps(_mm256_loadu_ps(s7 + m), _mm256_loadu_ps(s7 + m), acc);
//             }

//             // horizontal reduce
//             __m128 lo  = _mm256_castps256_ps128(acc);
//             __m128 hi  = _mm256_extractf128_ps(acc, 1);
//             __m128 sum = _mm_add_ps(lo, hi);
//             sum = _mm_hadd_ps(sum, sum);
//             sum = _mm_hadd_ps(sum, sum);
//             float accs = _mm_cvtss_f32(sum);

//             for (; m < M; ++m) {
//                 accs += s0[m]*s0[m] + s1[m]*s1[m] + s2[m]*s2[m] + s3[m]*s3[m]
//                       + s4[m]*s4[m] + s5[m]*s5[m] + s6[m]*s6[m] + s7[m]*s7[m];
//             }

//             accs /= static_cast<float>(M);
//             float scale = 1.0f / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const float* x = gx + (i << 4);
//                 float*       o = go + (i << 4);

//                 float scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 __m256 vs = _mm256_set1_ps(scalew);
//                 _mm256_storeu_ps(o + 0, _mm256_mul_ps(_mm256_loadu_ps(x + 0), vs));
//                 _mm256_storeu_ps(o + 8, _mm256_mul_ps(_mm256_loadu_ps(x + 8), vs));
//             }
//         }

//     } else if constexpr (std::is_same_v<scalar_t, double>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const double* gx   = X     + g * (M << 4);
//             const double* gsel = X_sel + g * (8 * M);
//             double*       go   = O     + g * (M << 4);

//             const double* s0 = gsel + 0 * M;
//             const double* s1 = gsel + 1 * M;
//             const double* s2 = gsel + 2 * M;
//             const double* s3 = gsel + 3 * M;
//             const double* s4 = gsel + 4 * M;
//             const double* s5 = gsel + 5 * M;
//             const double* s6 = gsel + 6 * M;
//             const double* s7 = gsel + 7 * M;

//             __m256d acc = _mm256_setzero_pd();
//             int64_t m = 0;

//             for (; m + 3 < M; m += 4) {
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s0 + m), _mm256_loadu_pd(s0 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s1 + m), _mm256_loadu_pd(s1 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s2 + m), _mm256_loadu_pd(s2 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s3 + m), _mm256_loadu_pd(s3 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s4 + m), _mm256_loadu_pd(s4 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s5 + m), _mm256_loadu_pd(s5 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s6 + m), _mm256_loadu_pd(s6 + m), acc);
//                 acc = _mm256_fmadd_pd(_mm256_loadu_pd(s7 + m), _mm256_loadu_pd(s7 + m), acc);
//             }

//             __m128d lo  = _mm256_castpd256_pd128(acc);
//             __m128d hi  = _mm256_extractf128_pd(acc, 1);
//             __m128d sum = _mm_add_pd(lo, hi);
//             double accs = _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));

//             for (; m < M; ++m) {
//                 accs += s0[m]*s0[m] + s1[m]*s1[m] + s2[m]*s2[m] + s3[m]*s3[m]
//                       + s4[m]*s4[m] + s5[m]*s5[m] + s6[m]*s6[m] + s7[m]*s7[m];
//             }

//             accs /= static_cast<double>(M);
//             double scale = 1.0 / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const double* x = gx + (i << 4);
//                 double*       o = go + (i << 4);

//                 double scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 __m256d vs = _mm256_set1_pd(scalew);
//                 _mm256_storeu_pd(o + 0,  _mm256_mul_pd(_mm256_loadu_pd(x + 0),  vs));
//                 _mm256_storeu_pd(o + 4,  _mm256_mul_pd(_mm256_loadu_pd(x + 4),  vs));
//                 _mm256_storeu_pd(o + 8,  _mm256_mul_pd(_mm256_loadu_pd(x + 8),  vs));
//                 _mm256_storeu_pd(o + 12, _mm256_mul_pd(_mm256_loadu_pd(x + 12), vs));
//             }
//         }
//     }
// }







//AMD INTRINS
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

//             float32x4_t acc = vdupq_n_f32(0.0f);

//             int64_t m = 0;

//             // ---- SIMD accumulation (4 lanes = 4 m values) ----
//             for (; m + 3 < M; m += 4) {

//                 const float* x0 = group_in + ((m + 0) << 4);
//                 const float* x1 = group_in + ((m + 1) << 4);
//                 const float* x2 = group_in + ((m + 2) << 4);
//                 const float* x3 = group_in + ((m + 3) << 4);

//                 float32x4_t v0  = { x0[0],  x1[0],  x2[0],  x3[0] };
//                 float32x4_t v2  = { x0[2],  x1[2],  x2[2],  x3[2] };
//                 float32x4_t v3  = { x0[3],  x1[3],  x2[3],  x3[3] };
//                 float32x4_t v4  = { x0[4],  x1[4],  x2[4],  x3[4] };

//                 float32x4_t v8  = { x0[8],  x1[8],  x2[8],  x3[8] };
//                 float32x4_t v9  = { x0[9],  x1[9],  x2[9],  x3[9] };
//                 float32x4_t v10 = { x0[10], x1[10], x2[10], x3[10] };
//                 float32x4_t v14 = { x0[14], x1[14], x2[14], x3[14] };

//                 acc = vfmaq_f32(acc, v0,  v0);
//                 acc = vfmaq_f32(acc, v2,  v2);
//                 acc = vfmaq_f32(acc, v3,  v3);
//                 acc = vfmaq_f32(acc, v4,  v4);
//                 acc = vfmaq_f32(acc, v8,  v8);
//                 acc = vfmaq_f32(acc, v9,  v9);
//                 acc = vfmaq_f32(acc, v10, v10);
//                 acc = vfmaq_f32(acc, v14, v14);
//             }

//             // ---- horizontal reduction ----
//             float accs = vaddvq_f32(acc);

//             // ---- scalar tail ----
//             for (; m < M; ++m) {
//                 const float* x = group_in + (m << 4);

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

//             accs /= static_cast<float>(M);

//             float scale = 1.0f / std::sqrt(std::max(accs, eps_t));

//             // ---- write ----
//             for (int64_t m = 0; m < M; ++m) {

//                 const float* x = group_in + (m << 4);
//                 float* o       = group_out + (m << 4);

//                 float scalew = scale;

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
//             // for (int64_t i = 0; i < M; ++i) {

//             //     const float* x = group_in + (i << 4);
//             //     float* o       = group_out + (i << 4);

//             //     float scalew = scale;

//             //     if constexpr (HasWeight) {
//             //         scalew *= W[i];
//             //     }


//             //     float32x4_t vscale = vdupq_n_f32(scalew);

//             //     float32x4_t v0 = vld1q_f32(x);
//             //     float32x4_t v1 = vld1q_f32(x + 8);

//             //     v0 = vmulq_f32(v0, vscale);
//             //     v1 = vmulq_f32(v1, vscale);

//             //     vst1q_f32(o, v0);
//             //     vst1q_f32(o + 8, v1);
//             // }
//         }

//     } else if constexpr (std::is_same_v<scalar_t, double>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const scalar_t* group_in  = X + g * (M << 4);
//             scalar_t* group_out       = O + g * (M << 4);

//             float64x2_t acc = vdupq_n_f64(0.0);

//             int64_t m = 0;

//             for (; m + 1 < M; m += 2) {

//                 const double* x0 = group_in + ((m + 0) << 4);
//                 const double* x1 = group_in + ((m + 1) << 4);

//                 float64x2_t v0  = { x0[0],  x1[0] };
//                 float64x2_t v2  = { x0[2],  x1[2] };
//                 float64x2_t v3  = { x0[3],  x1[3] };
//                 float64x2_t v4  = { x0[4],  x1[4] };

//                 float64x2_t v8  = { x0[8],  x1[8] };
//                 float64x2_t v9  = { x0[9],  x1[9] };
//                 float64x2_t v10 = { x0[10], x1[10] };
//                 float64x2_t v14 = { x0[14], x1[14] };

//                 acc = vfmaq_f64(acc, v0,  v0);
//                 acc = vfmaq_f64(acc, v2,  v2);
//                 acc = vfmaq_f64(acc, v3,  v3);
//                 acc = vfmaq_f64(acc, v4,  v4);
//                 acc = vfmaq_f64(acc, v8,  v8);
//                 acc = vfmaq_f64(acc, v9,  v9);
//                 acc = vfmaq_f64(acc, v10, v10);
//                 acc = vfmaq_f64(acc, v14, v14);
//             }

//             double accs = vaddvq_f64(acc);

//             for (; m < M; ++m) {
//                 const double* x = group_in + (m << 4);

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

//             accs /= static_cast<double>(M);

//             double scale = 1.0 / std::sqrt(std::max(accs, eps_t));

//             for (int64_t m = 0; m < M; ++m) {

//                 const double* x = group_in + (m << 4);
//                 double* o       = group_out + (m << 4);

//                 double scalew = scale;

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
//             // for (int64_t i = 0; i < M; ++i) {

//             //     const double* x = group_in + (i << 4);
//             //     double* o       = group_out + (i << 4);

//             //     double scalew = scale;

//             //     if constexpr (HasWeight) {
//             //         scalew *= W[i];
//             //     }

//             //     float64x2_t vscale = vdupq_n_f64(scalew);

//             //     float64x2_t v0 = vld1q_f64(x);
//             //     float64x2_t v1 = vld1q_f64(x + 4);
//             //     float64x2_t v2 = vld1q_f64(x + 8);
//             //     float64x2_t v3 = vld1q_f64(x + 12);

//             //     v0 = vmulq_f64(v0, vscale);
//             //     v1 = vmulq_f64(v1, vscale);
//             //     v2 = vmulq_f64(v2, vscale);
//             //     v3 = vmulq_f64(v3, vscale);

//             //     vst1q_f64(o, v0);
//             //     vst1q_f64(o + 4, v1);
//             //     vst1q_f64(o + 8, v2);
//             //     vst1q_f64(o + 12, v3);
//             // }
//         }
//     }
// }



// ---------------------------------------------------------------------------
// Packed AMD kernel.
// ---------------------------------------------------------------------------
// template <typename scalar_t, bool HasWeight>
// void rms_norm_kernel_packed(
//     const scalar_t* __restrict__ X,
//     const scalar_t* __restrict__ X_sel,
//     scalar_t* __restrict__ O,
//     const scalar_t* __restrict__ W,
//     int64_t groups,
//     int64_t M,
//     double eps
// ) {
//     const scalar_t eps_t = static_cast<scalar_t>(eps);

//     if constexpr (std::is_same_v<scalar_t, float>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const float* gx   = X     + g * (M << 4);
//             const float* gsel = X_sel + g * (8 * M);
//             float*       go   = O     + g * (M << 4);

//             const float* s0 = gsel + 0 * M;
//             const float* s1 = gsel + 1 * M;
//             const float* s2 = gsel + 2 * M;
//             const float* s3 = gsel + 3 * M;
//             const float* s4 = gsel + 4 * M;
//             const float* s5 = gsel + 5 * M;
//             const float* s6 = gsel + 6 * M;
//             const float* s7 = gsel + 7 * M;

//             float32x4_t acc = vdupq_n_f32(0.0f);
//             int64_t m = 0;

//             for (; m + 3 < M; m += 4) {
//                 acc = vfmaq_f32(acc, vld1q_f32(s0 + m), vld1q_f32(s0 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s1 + m), vld1q_f32(s1 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s2 + m), vld1q_f32(s2 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s3 + m), vld1q_f32(s3 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s4 + m), vld1q_f32(s4 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s5 + m), vld1q_f32(s5 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s6 + m), vld1q_f32(s6 + m));
//                 acc = vfmaq_f32(acc, vld1q_f32(s7 + m), vld1q_f32(s7 + m));
//             }

//             float accs = vaddvq_f32(acc);

//             for (; m < M; ++m) {
//                 accs += s0[m]*s0[m] + s1[m]*s1[m] + s2[m]*s2[m] + s3[m]*s3[m]
//                       + s4[m]*s4[m] + s5[m]*s5[m] + s6[m]*s6[m] + s7[m]*s7[m];
//             }

//             accs /= static_cast<float>(M);
//             float scale = 1.0f / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const float* x = gx + (i << 4);
//                 float*       o = go + (i << 4);

//                 float scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 float32x4_t vs = vdupq_n_f32(scalew);
//                 vst1q_f32(o + 0,  vmulq_f32(vld1q_f32(x + 0),  vs));
//                 vst1q_f32(o + 4,  vmulq_f32(vld1q_f32(x + 4),  vs));
//                 vst1q_f32(o + 8,  vmulq_f32(vld1q_f32(x + 8),  vs));
//                 vst1q_f32(o + 12, vmulq_f32(vld1q_f32(x + 12), vs));
//             }
//         }

//     } else if constexpr (std::is_same_v<scalar_t, double>) {

//         for (int64_t g = 0; g < groups; ++g) {

//             const double* gx   = X     + g * (M << 4);
//             const double* gsel = X_sel + g * (8 * M);
//             double*       go   = O     + g * (M << 4);

//             const double* s0 = gsel + 0 * M;
//             const double* s1 = gsel + 1 * M;
//             const double* s2 = gsel + 2 * M;
//             const double* s3 = gsel + 3 * M;
//             const double* s4 = gsel + 4 * M;
//             const double* s5 = gsel + 5 * M;
//             const double* s6 = gsel + 6 * M;
//             const double* s7 = gsel + 7 * M;

//             float64x2_t acc = vdupq_n_f64(0.0);
//             int64_t m = 0;

//             for (; m + 1 < M; m += 2) {
//                 acc = vfmaq_f64(acc, vld1q_f64(s0 + m), vld1q_f64(s0 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s1 + m), vld1q_f64(s1 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s2 + m), vld1q_f64(s2 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s3 + m), vld1q_f64(s3 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s4 + m), vld1q_f64(s4 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s5 + m), vld1q_f64(s5 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s6 + m), vld1q_f64(s6 + m));
//                 acc = vfmaq_f64(acc, vld1q_f64(s7 + m), vld1q_f64(s7 + m));
//             }

//             double accs = vaddvq_f64(acc);

//             if (m < M) {
//                 accs += s0[m]*s0[m] + s1[m]*s1[m] + s2[m]*s2[m] + s3[m]*s3[m]
//                       + s4[m]*s4[m] + s5[m]*s5[m] + s6[m]*s6[m] + s7[m]*s7[m];
//             }

//             accs /= static_cast<double>(M);
//             double scale = 1.0 / std::sqrt(std::max(accs, eps_t));

//             for (int64_t i = 0; i < M; ++i) {
//                 const double* x = gx + (i << 4);
//                 double*       o = go + (i << 4);

//                 double scalew = scale;
//                 if constexpr (HasWeight) scalew *= W[i];

//                 float64x2_t vs = vdupq_n_f64(scalew);
//                 vst1q_f64(o + 0,  vmulq_f64(vld1q_f64(x + 0),  vs));
//                 vst1q_f64(o + 2,  vmulq_f64(vld1q_f64(x + 2),  vs));
//                 vst1q_f64(o + 4,  vmulq_f64(vld1q_f64(x + 4),  vs));
//                 vst1q_f64(o + 6,  vmulq_f64(vld1q_f64(x + 6),  vs));
//                 vst1q_f64(o + 8,  vmulq_f64(vld1q_f64(x + 8),  vs));
//                 vst1q_f64(o + 10, vmulq_f64(vld1q_f64(x + 10), vs));
//                 vst1q_f64(o + 12, vmulq_f64(vld1q_f64(x + 12), vs));
//                 vst1q_f64(o + 14, vmulq_f64(vld1q_f64(x + 14), vs));
//             }
//         }
//     }
// }
