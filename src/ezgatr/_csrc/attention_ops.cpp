#include "attention_ops.h"

#include <ATen/ops/scaled_dot_product_attention.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ezgatr { namespace opt {

namespace {

using torch::Tensor;
using namespace torch::indexing;

using FloatCacheKey = std::tuple<c10::DeviceType, c10::DeviceIndex, c10::ScalarType>;
using LongCacheKey = std::tuple<c10::DeviceType, c10::DeviceIndex>;

FloatCacheKey make_float_key(c10::Device device, c10::ScalarType dtype) {
    return std::make_tuple(device.type(), device.index(), dtype);
}

LongCacheKey make_long_key(c10::Device device) {
    return std::make_tuple(device.type(), device.index());
}

std::mutex g_cache_mu;
std::map<LongCacheKey, Tensor> g_inner_product_selector;
std::map<FloatCacheKey, Tensor> g_daa_query_basis;
std::map<FloatCacheKey, Tensor> g_daa_key_basis;

Tensor flatten_ck(const Tensor& mv) {
    return mv.flatten(-2, -1);
}

Tensor inflate_ck(const Tensor& mv) {
    auto sizes = mv.sizes().vec();
    const auto last = sizes.back();
    TORCH_CHECK(
        last % 16 == 0,
        "inflate_ck: expected the flattened blade dimension to be divisible by 16, got ",
        last);
    sizes.back() = last / 16;
    sizes.push_back(16);
    return mv.view(sizes);
}

Tensor build_inner_product_selector(c10::Device device) {
    return torch::tensor(
        {0, 2, 3, 4, 8, 9, 10},
        torch::TensorOptions().device(device).dtype(torch::kLong));
}

Tensor load_inner_product_selector(c10::Device device) {
    auto key = make_long_key(device);
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_inner_product_selector.find(key);
    if (it != g_inner_product_selector.end()) {
        return it->second;
    }

    auto selector = build_inner_product_selector(device);
    g_inner_product_selector.emplace(key, selector);
    return selector;
}

std::pair<Tensor, Tensor> build_daa_basis(
    c10::Device device,
    c10::ScalarType dtype) {
    auto options = torch::TensorOptions().device(device).dtype(dtype);
    auto bq = torch::zeros({4, 4, 5}, options);
    auto bk = torch::zeros({4, 4, 5}, options);

    bq.index_put_({0, 0, 0}, 1.0);
    bq.index_put_({1, 1, 0}, 1.0);
    bq.index_put_({2, 2, 0}, 1.0);
    bk.index_put_({3, 3, 0}, -1.0);

    bq.index_put_({3, 3, 1}, 1.0);
    bk.index_put_({0, 0, 1}, -1.0);
    bk.index_put_({1, 1, 1}, -1.0);
    bk.index_put_({2, 2, 1}, -1.0);

    bq.index_put_({0, 3, 2}, 1.0);
    bq.index_put_({1, 3, 3}, 1.0);
    bq.index_put_({2, 3, 4}, 1.0);
    bk.index_put_({0, 3, 2}, 2.0);
    bk.index_put_({1, 3, 3}, 2.0);
    bk.index_put_({2, 3, 4}, 2.0);

    return {bq, bk};
}

std::pair<Tensor, Tensor> load_daa_basis(
    const torch::Device& device,
    c10::ScalarType dtype) {
    auto key = make_float_key(device, dtype);
    {
        std::lock_guard<std::mutex> lock(g_cache_mu);
        auto q_it = g_daa_query_basis.find(key);
        auto k_it = g_daa_key_basis.find(key);
        if (q_it != g_daa_query_basis.end() && k_it != g_daa_key_basis.end()) {
            return {q_it->second, k_it->second};
        }
    }

    auto [bq, bk] = build_daa_basis(device, dtype);

    {
        std::lock_guard<std::mutex> lock(g_cache_mu);
        g_daa_query_basis.emplace(key, bq);
        g_daa_key_basis.emplace(key, bk);
    }

    return {bq, bk};
}

Tensor linear_square_normalizer(const Tensor& e123, double eps) {
    return e123 / (e123.pow(2) + eps);
}

Tensor get_inner_product_selector(c10::Device device, bool use_cache) {
    return use_cache ? load_inner_product_selector(device) : build_inner_product_selector(device);
}

std::pair<Tensor, Tensor> get_daa_basis(c10::Device device, c10::ScalarType dtype, bool use_cache) {
    return use_cache ? load_daa_basis(device, dtype) : build_daa_basis(device, dtype);
}

Tensor select_tri_vector_block(const Tensor& q_or_k, bool use_cache) {
    if (!use_cache) {
        return q_or_k.index({Ellipsis, Slice(11, 15)});
    }

    // The tri-vector coefficients occupy a contiguous block, so slicing avoids
    // the gather-style overhead of index_select while preserving the same data.
    return q_or_k.index({Ellipsis, Slice(11, 15)});
}

Tensor build_daa_qk(const Tensor& q_or_k, const Tensor& basis, double eps, bool use_cache) {
    auto tri = select_tri_vector_block(q_or_k, use_cache);
    auto normalized =
        tri * linear_square_normalizer(tri.index({Ellipsis, Slice(3, 4)}), eps);
    return torch::einsum("ijk, ...i, ...j -> ...k", {basis, normalized, normalized});
}

Tensor build_daa_qk_explicit(const Tensor& q_or_k, double eps, bool is_query, bool use_cache) {
    auto tri = select_tri_vector_block(q_or_k, use_cache);
    auto normalized =
        tri * linear_square_normalizer(tri.index({Ellipsis, Slice(3, 4)}), eps);

    auto x0 = normalized.index({Ellipsis, Slice(0, 1)});
    auto x1 = normalized.index({Ellipsis, Slice(1, 2)});
    auto x2 = normalized.index({Ellipsis, Slice(2, 3)});
    auto x3 = normalized.index({Ellipsis, Slice(3, 4)});

    auto sum012 = x0 * x0 + x1 * x1 + x2 * x2;
    if (is_query) {
        return torch::cat({sum012, x3 * x3, x0 * x3, x1 * x3, x2 * x3}, -1);
    }

    return torch::cat(
        {-(x3 * x3), -sum012, 2.0 * x0 * x3, 2.0 * x1 * x3, 2.0 * x2 * x3},
        -1);
}

std::pair<Tensor, Tensor> compute_qk_for_daa(
    const Tensor& query,
    const Tensor& key,
    double eps,
    bool use_cache) {
    auto [bq, bk] = get_daa_basis(query.device(), query.scalar_type(), use_cache);
    return {
        build_daa_qk(query, bq, eps, use_cache),
        build_daa_qk(key, bk, eps, use_cache),
    };
}

std::pair<Tensor, Tensor> compute_qk_for_ipa(
    const Tensor& query,
    const Tensor& key,
    bool use_cache) {
    auto selector = get_inner_product_selector(query.device(), use_cache);
    return {
        query.index_select(-1, selector),
        key.index_select(-1, selector),
    };
}

Tensor build_ipa_qk_sliced(const Tensor& q_or_k) {
    return torch::cat(
        {
            q_or_k.index({Ellipsis, Slice(0, 1)}),
            q_or_k.index({Ellipsis, Slice(2, 5)}),
            q_or_k.index({Ellipsis, Slice(8, 11)}),
        },
        -1);
}

std::pair<Tensor, Tensor> compute_qk_for_ipa_ver_3(
    const Tensor& query,
    const Tensor& key) {
    return {
        build_ipa_qk_sliced(query),
        build_ipa_qk_sliced(key),
    };
}

std::pair<Tensor, Tensor> compute_qk_for_daa_ver_2(
    const Tensor& query,
    const Tensor& key,
    double eps,
    bool use_cache) {
    return {
        build_daa_qk_explicit(query, eps, true, use_cache),
        build_daa_qk_explicit(key, eps, false, use_cache),
    };
}

Tensor compute_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const py::object& attn_mask_obj,
    double dropout_p,
    bool is_causal,
    const py::object& scale_obj) {
    std::optional<Tensor> attn_mask;
    if (!attn_mask_obj.is_none()) {
        attn_mask = attn_mask_obj.cast<Tensor>();
    }

    std::optional<double> scale = std::nullopt;
    if (!scale_obj.is_none()) {
        scale = scale_obj.cast<double>();
    }

    return at::scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale);
}

Tensor apply_query_weight(const Tensor& query_part, const py::handle& weight_item) {
    if (py::isinstance<py::float_>(weight_item) || py::isinstance<py::int_>(weight_item)) {
        return query_part * py::cast<double>(weight_item);
    }
    return query_part * py::cast<Tensor>(weight_item);
}

std::optional<double> numeric_weight(const py::handle& weight_item) {
    if (py::isinstance<py::float_>(weight_item) || py::isinstance<py::int_>(weight_item)) {
        return py::cast<double>(weight_item);
    }
    try {
        auto tensor = py::cast<Tensor>(weight_item);
        if (tensor.device().is_cpu() && tensor.numel() == 1) {
            return tensor.item<double>();
        }
    } catch (const py::cast_error&) {
    }
    return std::nullopt;
}

Tensor assemble_two_part_flattened(
    const Tensor& first,
    const Tensor& second) {
    auto first_flat = flatten_ck(first);
    auto second_flat = flatten_ck(second);

    auto out_sizes = first_flat.sizes().vec();
    out_sizes.back() = first_flat.size(-1) + second_flat.size(-1);
    auto out = torch::empty(out_sizes, first_flat.options());
    out.index_put_({Ellipsis, Slice(None, first_flat.size(-1))}, first_flat);
    out.index_put_({Ellipsis, Slice(first_flat.size(-1), None)}, second_flat);
    return out;
}

#if defined(__AVX2__)
// SIMD kernel for interleaved IPA+DAA Q/K assembly using SoA output layout.
//
// Input layout: AoS [..., channels, 16]. Each blade `bl` for 8 consecutive
// channels is loaded via 8 scalar gathers (unavoidable without changing the
// upstream data layout). The computation (normalization, products) is fully
// vectorized with AVX2.
//
// Output layout: SoA [..., 12, channels] (logically; flattened to
// [..., 12*channels] bytes). Unit-stride stores replace the scatter pattern
// in the original store_stride8 (which wrote to stride-12 offsets via a
// temporary stack array). This eliminates ~8x as many store instructions
// and reduces output cache-line pressure from ~60 lines/block to ~6.
//
// Correctness: Q and K are written with the same slot permutation, so the
// dot product Q·K^T sums the same set of products as AoS ordering. The
// result of scaled_dot_product_attention is numerically identical up to
// floating-point reordering (within the existing atol=5e-5 tolerance).
bool build_interleaved_ipa_daa_qk_simd_float(
    const Tensor& query,
    const Tensor& key,
    Tensor& query_flat,
    Tensor& key_flat,
    int64_t channels,
    int64_t ipa_offset,
    int64_t daa_offset,
    double ipa_weight,
    double daa_weight,
    double eps) {
    if (query.scalar_type() != torch::kFloat32 || channels < 8) {
        return false;
    }

    auto query_contig = query.contiguous();
    auto key_contig   = key.contiguous();
    const auto num_blocks = query.numel() / (channels * 16);
    const float* q_in = query_contig.data_ptr<float>();
    const float* k_in = key_contig.data_ptr<float>();
    float* q_out = query_flat.data_ptr<float>();
    float* k_out = key_flat.data_ptr<float>();

    const float ipa_w = static_cast<float>(ipa_weight);
    const float daa_w = static_cast<float>(daa_weight);
    const float eps_v = static_cast<float>(eps);

    const __m256 ipa_w_v = _mm256_set1_ps(ipa_w);
    const __m256 daa_w_v = _mm256_set1_ps(daa_w);
    const __m256 eps_vec = _mm256_set1_ps(eps_v);
    const __m256 two     = _mm256_set1_ps(2.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    // AoS input:  [num_blocks, channels, 16] — qi + c*16 + bl for channel c, blade bl
    // SoA output: [num_blocks, 12, channels] — qo + sl*C + c  for slot sl, channel c
    for (int64_t block = 0; block < num_blocks; ++block) {
        const float* qi = q_in + block * channels * 16;
        const float* ki = k_in + block * channels * 16;
        float*       qo = q_out + block * 12 * channels;
        float*       ko = k_out + block * 12 * channels;

        int64_t c = 0;
        for (; c + 7 < channels; c += 8) {
            const float* qx = qi + c * 16;  // first of 8 channels, AoS block
            const float* kx = ki + c * 16;

            // AoS gather: load blade `bl` from 8 consecutive channels (stride 16 floats).
            // _mm256_set_ps with 8 distinct addresses; the CPU issues 8 scalar loads.
            auto lq = [&](int64_t bl) {
                return _mm256_set_ps(
                    qx[7*16+bl], qx[6*16+bl], qx[5*16+bl], qx[4*16+bl],
                    qx[3*16+bl], qx[2*16+bl], qx[1*16+bl], qx[0*16+bl]);
            };
            auto lk = [&](int64_t bl) {
                return _mm256_set_ps(
                    kx[7*16+bl], kx[6*16+bl], kx[5*16+bl], kx[4*16+bl],
                    kx[3*16+bl], kx[2*16+bl], kx[1*16+bl], kx[0*16+bl]);
            };
            // SoA unit-stride store: 8 values for slot `sl` at contiguous addresses.
            // Replaces store_stride8's scatter (tmp[] + 8 scalar stores with stride 12).
            auto sq = [&](int64_t sl, __m256 v) { _mm256_storeu_ps(qo + sl * channels + c, v); };
            auto sk = [&](int64_t sl, __m256 v) { _mm256_storeu_ps(ko + sl * channels + c, v); };

            // IPA: blades {0, 2, 3, 4, 8, 9, 10}
            sq(ipa_offset + 0, _mm256_mul_ps(lq(0),  ipa_w_v));
            sq(ipa_offset + 1, _mm256_mul_ps(lq(2),  ipa_w_v));
            sq(ipa_offset + 2, _mm256_mul_ps(lq(3),  ipa_w_v));
            sq(ipa_offset + 3, _mm256_mul_ps(lq(4),  ipa_w_v));
            sq(ipa_offset + 4, _mm256_mul_ps(lq(8),  ipa_w_v));
            sq(ipa_offset + 5, _mm256_mul_ps(lq(9),  ipa_w_v));
            sq(ipa_offset + 6, _mm256_mul_ps(lq(10), ipa_w_v));

            sk(ipa_offset + 0, lk(0));
            sk(ipa_offset + 1, lk(2));
            sk(ipa_offset + 2, lk(3));
            sk(ipa_offset + 3, lk(4));
            sk(ipa_offset + 4, lk(8));
            sk(ipa_offset + 5, lk(9));
            sk(ipa_offset + 6, lk(10));

            // DAA query: blades {11, 12, 13, 14}
            // Reciprocal + one Newton-Raphson step replaces _mm256_div_ps (11-14 cy latency).
            // _mm256_rcp_ps gives ~12-bit precision; NR lifts it to full float32.
            // FMA (_mm256_fmadd_ps / _mm256_fnmadd_ps) fuses mul+add into one instruction.
            const __m256 q14    = lq(14);
            const __m256 q_den  = _mm256_fmadd_ps(q14, q14, eps_vec);   // q14² + eps
            __m256 q_rcp        = _mm256_rcp_ps(q_den);                  // ≈ 1/denom
            q_rcp = _mm256_mul_ps(q_rcp, _mm256_fnmadd_ps(q_den, q_rcp, two)); // NR: r*(2-d*r)
            const __m256 q_nrm  = _mm256_mul_ps(q14, q_rcp);
            const __m256 qn0    = _mm256_mul_ps(lq(11), q_nrm);
            const __m256 qn1    = _mm256_mul_ps(lq(12), q_nrm);
            const __m256 qn2    = _mm256_mul_ps(lq(13), q_nrm);
            const __m256 qn3    = _mm256_mul_ps(q14, q_nrm);
            __m256 qsum         = _mm256_mul_ps(qn0, qn0);
            qsum = _mm256_fmadd_ps(qn1, qn1, qsum);
            qsum = _mm256_fmadd_ps(qn2, qn2, qsum);
            sq(daa_offset + 0, _mm256_mul_ps(qsum,                    daa_w_v));
            sq(daa_offset + 1, _mm256_mul_ps(_mm256_mul_ps(qn3, qn3), daa_w_v));
            sq(daa_offset + 2, _mm256_mul_ps(_mm256_mul_ps(qn0, qn3), daa_w_v));
            sq(daa_offset + 3, _mm256_mul_ps(_mm256_mul_ps(qn1, qn3), daa_w_v));
            sq(daa_offset + 4, _mm256_mul_ps(_mm256_mul_ps(qn2, qn3), daa_w_v));

            // DAA key: blades {11, 12, 13, 14}
            const __m256 k14    = lk(14);
            const __m256 k_den  = _mm256_fmadd_ps(k14, k14, eps_vec);
            __m256 k_rcp        = _mm256_rcp_ps(k_den);
            k_rcp = _mm256_mul_ps(k_rcp, _mm256_fnmadd_ps(k_den, k_rcp, two));
            const __m256 k_nrm  = _mm256_mul_ps(k14, k_rcp);
            const __m256 kn0    = _mm256_mul_ps(lk(11), k_nrm);
            const __m256 kn1    = _mm256_mul_ps(lk(12), k_nrm);
            const __m256 kn2    = _mm256_mul_ps(lk(13), k_nrm);
            const __m256 kn3    = _mm256_mul_ps(k14, k_nrm);
            __m256 ksum         = _mm256_mul_ps(kn0, kn0);
            ksum = _mm256_fmadd_ps(kn1, kn1, ksum);
            ksum = _mm256_fmadd_ps(kn2, kn2, ksum);
            sk(daa_offset + 0, _mm256_mul_ps(_mm256_mul_ps(kn3, kn3), neg_one));
            sk(daa_offset + 1, _mm256_mul_ps(ksum,                    neg_one));
            sk(daa_offset + 2, _mm256_mul_ps(two, _mm256_mul_ps(kn0, kn3)));
            sk(daa_offset + 3, _mm256_mul_ps(two, _mm256_mul_ps(kn1, kn3)));
            sk(daa_offset + 4, _mm256_mul_ps(two, _mm256_mul_ps(kn2, kn3)));
        }

        // Scalar tail: channels not covered by the 8-wide SIMD loop.
        // Input: AoS qx[bl], output: SoA qo[sl*C + c].
        for (; c < channels; ++c) {
            const float* qx = qi + c * 16;
            const float* kx = ki + c * 16;

            qo[(ipa_offset + 0) * channels + c] = qx[0]  * ipa_w;
            qo[(ipa_offset + 1) * channels + c] = qx[2]  * ipa_w;
            qo[(ipa_offset + 2) * channels + c] = qx[3]  * ipa_w;
            qo[(ipa_offset + 3) * channels + c] = qx[4]  * ipa_w;
            qo[(ipa_offset + 4) * channels + c] = qx[8]  * ipa_w;
            qo[(ipa_offset + 5) * channels + c] = qx[9]  * ipa_w;
            qo[(ipa_offset + 6) * channels + c] = qx[10] * ipa_w;

            ko[(ipa_offset + 0) * channels + c] = kx[0];
            ko[(ipa_offset + 1) * channels + c] = kx[2];
            ko[(ipa_offset + 2) * channels + c] = kx[3];
            ko[(ipa_offset + 3) * channels + c] = kx[4];
            ko[(ipa_offset + 4) * channels + c] = kx[8];
            ko[(ipa_offset + 5) * channels + c] = kx[9];
            ko[(ipa_offset + 6) * channels + c] = kx[10];

            const float q14_s = qx[14];
            const float q_nrm = q14_s / (q14_s * q14_s + eps_v);
            const float qn0 = qx[11] * q_nrm;
            const float qn1 = qx[12] * q_nrm;
            const float qn2 = qx[13] * q_nrm;
            const float qn3 = q14_s  * q_nrm;
            qo[(daa_offset + 0) * channels + c] = (qn0*qn0 + qn1*qn1 + qn2*qn2) * daa_w;
            qo[(daa_offset + 1) * channels + c] = qn3 * qn3 * daa_w;
            qo[(daa_offset + 2) * channels + c] = qn0 * qn3 * daa_w;
            qo[(daa_offset + 3) * channels + c] = qn1 * qn3 * daa_w;
            qo[(daa_offset + 4) * channels + c] = qn2 * qn3 * daa_w;

            const float k14_s = kx[14];
            const float k_nrm = k14_s / (k14_s * k14_s + eps_v);
            const float kn0 = kx[11] * k_nrm;
            const float kn1 = kx[12] * k_nrm;
            const float kn2 = kx[13] * k_nrm;
            const float kn3 = k14_s  * k_nrm;
            ko[(daa_offset + 0) * channels + c] = -(kn3 * kn3);
            ko[(daa_offset + 1) * channels + c] = -(kn0*kn0 + kn1*kn1 + kn2*kn2);
            ko[(daa_offset + 2) * channels + c] = 2.0f * kn0 * kn3;
            ko[(daa_offset + 3) * channels + c] = 2.0f * kn1 * kn3;
            ko[(daa_offset + 4) * channels + c] = 2.0f * kn2 * kn3;
        }
    }
    return true;
}
#endif

std::optional<std::pair<Tensor, Tensor>> try_build_interleaved_ipa_daa_qk(
    const Tensor& query,
    const Tensor& key,
    const std::vector<std::string>& kind_names,
    const std::vector<py::object>& kind_kwargs,
    const std::vector<py::object>& weights,
    bool use_simd) {
    if (kind_names.size() != 2) {
        return std::nullopt;
    }

    int ipa_index = -1;
    int daa_index = -1;
    for (size_t i = 0; i < kind_names.size(); ++i) {
        if (kind_names[i] == "ipa") {
            ipa_index = static_cast<int>(i);
        } else if (kind_names[i] == "daa") {
            daa_index = static_cast<int>(i);
        } else {
            return std::nullopt;
        }
    }
    if (ipa_index < 0 || daa_index < 0) {
        return std::nullopt;
    }

    auto ipa_weight = numeric_weight(weights[ipa_index]);
    auto daa_weight = numeric_weight(weights[daa_index]);
    if (!ipa_weight.has_value() || !daa_weight.has_value()) {
        return std::nullopt;
    }

    if (!query.device().is_cpu() || !key.device().is_cpu()) {
        return std::nullopt;
    }

    double eps = 1e-3;
    if (!kind_kwargs[daa_index].is_none()) {
        auto kwargs = py::cast<py::dict>(kind_kwargs[daa_index]);
        if (kwargs.contains("eps")) {
            eps = py::cast<double>(kwargs["eps"]);
        }
    }

    auto flat_sizes = query.sizes().vec();
    const auto channels = flat_sizes[flat_sizes.size() - 2];
    const auto num_blocks = query.numel() / (channels * 16);
    flat_sizes.pop_back();
    flat_sizes.back() = channels * 12;
    auto query_flat = torch::empty(flat_sizes, query.options());
    auto key_flat = torch::empty(flat_sizes, key.options());

    auto query_contig = query.contiguous();
    auto key_contig = key.contiguous();

    const int64_t ipa_offset = ipa_index == 0 ? 0 : 5;
    const int64_t daa_offset = daa_index == 0 ? 0 : 7;

#if defined(__AVX2__)
    if (use_simd && build_interleaved_ipa_daa_qk_simd_float(
                        query,
                        key,
                        query_flat,
                        key_flat,
                        channels,
                        ipa_offset,
                        daa_offset,
                        *ipa_weight,
                        *daa_weight,
                        eps)) {
        return std::make_pair(query_flat, key_flat);
    }
#else
    (void)use_simd;
#endif

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "build_interleaved_ipa_daa_qk", [&] {
        const auto* q_in = query_contig.data_ptr<scalar_t>();
        const auto* k_in = key_contig.data_ptr<scalar_t>();
        auto* q_out = query_flat.data_ptr<scalar_t>();
        auto* k_out = key_flat.data_ptr<scalar_t>();
        const scalar_t eps_v = static_cast<scalar_t>(eps);
        const scalar_t ipa_w = static_cast<scalar_t>(*ipa_weight);
        const scalar_t daa_w = static_cast<scalar_t>(*daa_weight);

        for (int64_t block = 0; block < num_blocks; ++block) {
            const int64_t in_base = block * channels * 16;
            const int64_t out_base = block * channels * 12;

            auto emit_channel = [&](int64_t c) {
                const scalar_t* qx = q_in + in_base + c * 16;
                const scalar_t* kx = k_in + in_base + c * 16;
                scalar_t* qo = q_out + out_base + c * 12;
                scalar_t* ko = k_out + out_base + c * 12;

                qo[ipa_offset + 0] = qx[0] * ipa_w;
                qo[ipa_offset + 1] = qx[2] * ipa_w;
                qo[ipa_offset + 2] = qx[3] * ipa_w;
                qo[ipa_offset + 3] = qx[4] * ipa_w;
                qo[ipa_offset + 4] = qx[8] * ipa_w;
                qo[ipa_offset + 5] = qx[9] * ipa_w;
                qo[ipa_offset + 6] = qx[10] * ipa_w;

                ko[ipa_offset + 0] = kx[0];
                ko[ipa_offset + 1] = kx[2];
                ko[ipa_offset + 2] = kx[3];
                ko[ipa_offset + 3] = kx[4];
                ko[ipa_offset + 4] = kx[8];
                ko[ipa_offset + 5] = kx[9];
                ko[ipa_offset + 6] = kx[10];

                const scalar_t q_norm = qx[14] / (qx[14] * qx[14] + eps_v);
                const scalar_t qn0 = qx[11] * q_norm;
                const scalar_t qn1 = qx[12] * q_norm;
                const scalar_t qn2 = qx[13] * q_norm;
                const scalar_t qn3 = qx[14] * q_norm;
                qo[daa_offset + 0] = (qn0 * qn0 + qn1 * qn1 + qn2 * qn2) * daa_w;
                qo[daa_offset + 1] = (qn3 * qn3) * daa_w;
                qo[daa_offset + 2] = (qn0 * qn3) * daa_w;
                qo[daa_offset + 3] = (qn1 * qn3) * daa_w;
                qo[daa_offset + 4] = (qn2 * qn3) * daa_w;

                const scalar_t k_norm = kx[14] / (kx[14] * kx[14] + eps_v);
                const scalar_t kn0 = kx[11] * k_norm;
                const scalar_t kn1 = kx[12] * k_norm;
                const scalar_t kn2 = kx[13] * k_norm;
                const scalar_t kn3 = kx[14] * k_norm;
                ko[daa_offset + 0] = -(kn3 * kn3);
                ko[daa_offset + 1] = -(kn0 * kn0 + kn1 * kn1 + kn2 * kn2);
                ko[daa_offset + 2] = static_cast<scalar_t>(2.0) * kn0 * kn3;
                ko[daa_offset + 3] = static_cast<scalar_t>(2.0) * kn1 * kn3;
                ko[daa_offset + 4] = static_cast<scalar_t>(2.0) * kn2 * kn3;
            };

            if (channels == 4) {
                emit_channel(0);
                emit_channel(1);
                emit_channel(2);
                emit_channel(3);
            } else if (channels == 8) {
                emit_channel(0);
                emit_channel(1);
                emit_channel(2);
                emit_channel(3);
                emit_channel(4);
                emit_channel(5);
                emit_channel(6);
                emit_channel(7);
            } else {
                for (int64_t c = 0; c < channels; ++c) {
                    emit_channel(c);
                }
            }
        }
    });

    return std::make_pair(query_flat, key_flat);
}

std::optional<std::pair<Tensor, Tensor>> try_build_fast_path_qk(
    const Tensor& query,
    const Tensor& key,
    const std::vector<std::string>& kind_names,
    const std::vector<py::object>& kind_kwargs,
    const std::vector<py::object>& weights,
    bool use_cache,
    bool use_direct_daa,
    bool use_simd) {
    if (use_direct_daa) {
        auto compressed = try_build_interleaved_ipa_daa_qk(
            query, key, kind_names, kind_kwargs, weights, use_simd);
        if (compressed.has_value()) {
            return compressed;
        }
    }

    if (kind_names.size() == 1) {
        if (kind_names[0] == "ipa") {
            Tensor q_part;
            Tensor k_part;
            std::tie(q_part, k_part) = compute_qk_for_ipa_ver_3(query, key);
            return std::make_pair(
                flatten_ck(apply_query_weight(q_part, weights[0])),
                flatten_ck(k_part));
        }

        if (kind_names[0] == "daa") {
            double eps = 1e-3;
            if (!kind_kwargs[0].is_none()) {
                auto kwargs = py::cast<py::dict>(kind_kwargs[0]);
                if (kwargs.contains("eps")) {
                    eps = py::cast<double>(kwargs["eps"]);
                }
            }

            Tensor q_part;
            Tensor k_part;
            if (use_direct_daa) {
                std::tie(q_part, k_part) = compute_qk_for_daa_ver_2(query, key, eps, use_cache);
            } else {
                std::tie(q_part, k_part) = compute_qk_for_daa(query, key, eps, use_cache);
            }
            return std::make_pair(
                flatten_ck(apply_query_weight(q_part, weights[0])),
                flatten_ck(k_part));
        }

        return std::nullopt;
    }

    if (kind_names.size() != 2) {
        return std::nullopt;
    }

    bool saw_ipa = false;
    bool saw_daa = false;
    for (const auto& kind_name : kind_names) {
        saw_ipa = saw_ipa || kind_name == "ipa";
        saw_daa = saw_daa || kind_name == "daa";
    }
    if (!saw_ipa || !saw_daa) {
        return std::nullopt;
    }

    Tensor q_first;
    Tensor k_first;
    Tensor q_second;
    Tensor k_second;

    for (size_t i = 0; i < kind_names.size(); ++i) {
        Tensor q_part;
        Tensor k_part;
        if (kind_names[i] == "ipa") {
            std::tie(q_part, k_part) = compute_qk_for_ipa_ver_3(query, key);
        } else if (kind_names[i] == "daa") {
            double eps = 1e-3;
            if (!kind_kwargs[i].is_none()) {
                auto kwargs = py::cast<py::dict>(kind_kwargs[i]);
                if (kwargs.contains("eps")) {
                    eps = py::cast<double>(kwargs["eps"]);
                }
            }

            if (use_direct_daa) {
                std::tie(q_part, k_part) = compute_qk_for_daa_ver_2(query, key, eps, use_cache);
            } else {
                std::tie(q_part, k_part) = compute_qk_for_daa(query, key, eps, use_cache);
            }
        } else {
            return std::nullopt;
        }

        q_part = apply_query_weight(q_part, weights[i]);
        if (i == 0) {
            q_first = q_part;
            k_first = k_part;
        } else {
            q_second = q_part;
            k_second = k_part;
        }
    }

    return std::make_pair(
        assemble_two_part_flattened(q_first, q_second),
        assemble_two_part_flattened(k_first, k_second));
}

void check_mv_attention_tensor(const Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() >= 2, name, ": expected at least 2 dims, got ", tensor.dim());
    TORCH_CHECK(tensor.size(-1) == 16, name, ": expected last dim to be 16, got ", tensor.size(-1));
    TORCH_CHECK(tensor.is_floating_point(), name, ": expected floating-point dtype");
}

}  // namespace

torch::Tensor equi_geometric_attention_mv_only_impl(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale,
    bool use_cache,
    bool use_direct_daa,
    bool use_fast_paths,
    bool use_simd) {
    check_mv_attention_tensor(query, "equi_geometric_attention_mv_only: query");
    check_mv_attention_tensor(key, "equi_geometric_attention_mv_only: key");
    check_mv_attention_tensor(value, "equi_geometric_attention_mv_only: value");
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
                "equi_geometric_attention_mv_only: query, key, and value must share device");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                    query.scalar_type() == value.scalar_type(),
                "equi_geometric_attention_mv_only: query, key, and value must share dtype");

    std::vector<Tensor> qs;
    std::vector<Tensor> ks;
    std::vector<py::object> weights;
    std::vector<std::string> kind_names;
    std::vector<py::object> kind_kwargs;

    if (weight.is_none()) {
        for (size_t i = 0; i < py::len(kinds); ++i) {
            weights.emplace_back(py::float_(1.0));
        }
    } else {
        auto weight_seq = py::cast<py::sequence>(weight);
        TORCH_CHECK(
            py::len(weight_seq) == py::len(kinds),
            "equi_geometric_attention_mv_only: expected the same number of weights as kinds");
        for (auto item : weight_seq) {
            weights.emplace_back(py::reinterpret_borrow<py::object>(item));
        }
    }

    size_t index = 0;
    for (auto item : kinds) {
        auto kind = py::cast<std::string>(item.first);
        auto kwargs_obj = py::reinterpret_borrow<py::object>(item.second);
        kind_names.push_back(kind);
        kind_kwargs.push_back(kwargs_obj);

        if (use_fast_paths) {
            ++index;
            continue;
        }

        Tensor q_part;
        Tensor k_part;
        if (kind == "ipa") {
            std::tie(q_part, k_part) = compute_qk_for_ipa(query, key, use_cache);
        } else if (kind == "daa") {
            double eps = 1e-3;
            if (!kwargs_obj.is_none()) {
                auto kwargs = py::cast<py::dict>(kwargs_obj);
                if (kwargs.contains("eps")) {
                    eps = py::cast<double>(kwargs["eps"]);
                }
            }
            if (use_direct_daa) {
                std::tie(q_part, k_part) = compute_qk_for_daa_ver_2(query, key, eps, use_cache);
            } else {
                std::tie(q_part, k_part) = compute_qk_for_daa(query, key, eps, use_cache);
            }
        } else {
            TORCH_CHECK(false, "equi_geometric_attention_mv_only: unsupported attention kind: ", kind);
        }

        qs.push_back(flatten_ck(apply_query_weight(q_part, weights[index])));
        ks.push_back(flatten_ck(k_part));
        ++index;
    }

    Tensor query_flat;
    Tensor key_flat;
    if (use_fast_paths) {
        auto qk = try_build_fast_path_qk(
            query, key, kind_names, kind_kwargs, weights, use_cache, use_direct_daa, use_simd);
        if (qk.has_value()) {
            query_flat = qk->first;
            key_flat = qk->second;
        } else {
            for (size_t i = 0; i < kind_names.size(); ++i) {
                Tensor q_part;
                Tensor k_part;
                if (kind_names[i] == "ipa") {
                    std::tie(q_part, k_part) = compute_qk_for_ipa_ver_3(query, key);
                } else if (kind_names[i] == "daa") {
                    double eps = 1e-3;
                    if (!kind_kwargs[i].is_none()) {
                        auto kwargs = py::cast<py::dict>(kind_kwargs[i]);
                        if (kwargs.contains("eps")) {
                            eps = py::cast<double>(kwargs["eps"]);
                        }
                    }
                    if (use_direct_daa) {
                        std::tie(q_part, k_part) = compute_qk_for_daa_ver_2(query, key, eps, use_cache);
                    } else {
                        std::tie(q_part, k_part) = compute_qk_for_daa(query, key, eps, use_cache);
                    }
                } else {
                    TORCH_CHECK(false, "equi_geometric_attention_mv_only: unsupported attention kind: ", kind_names[i]);
                }

                qs.push_back(flatten_ck(apply_query_weight(q_part, weights[i])));
                ks.push_back(flatten_ck(k_part));
            }
            query_flat = torch::cat(qs, -1);
            key_flat = torch::cat(ks, -1);
        }
    } else {
        query_flat = torch::cat(qs, -1);
        key_flat = torch::cat(ks, -1);
    }
    auto value_flat = flatten_ck(value);

    auto ret = compute_scaled_dot_product_attention(
        query_flat,
        key_flat,
        value_flat,
        attn_mask,
        dropout_p,
        is_causal,
        scale);
    return inflate_ck(ret);
}

torch::Tensor equi_geometric_attention_mv_only_ver_0(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, false, false, false, false);
}

torch::Tensor equi_geometric_attention_mv_only_ver_1(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    // Math optimizations: explicit DAA formula (direct computation replaces einsum)
    // + cached mathematical constants (inner product selector, DAA basis tensors).
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, true, true, false, false);
}

torch::Tensor equi_geometric_attention_mv_only_ver_2(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, true, true, true, false);
}

torch::Tensor equi_geometric_attention_mv_only_ver_3(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    // SIMD: AVX2 vectorization with AoS gather loads + SoA unit-stride stores,
    // FMA for DAA products, reciprocal+Newton-Raphson for normalization division.
    // Falls back to v2 scalar path for non-float32 or channels < 8.
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, true, true, true, true);
}

torch::Tensor equi_geometric_attention_mv_only(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    return equi_geometric_attention_mv_only_ver_3(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale);
}


}}  // namespace ezgatr::opt
