#include "attention_ops.h"

#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/Parallel.h>
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
// AVX2 IPA+DAA Q/K assembly. Input is AoS [..., channels, 16]; we load each
// blade for 8 channels with scalar gathers and vectorize the math. Output is
// SoA [..., 12, channels] so the stores are unit-stride. Q and K use the same
// slot order, so Q.K^T is unchanged up to fp reordering.
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

    // in:  [num_blocks, channels, 16] at qi + c*16 + bl
    // out: [num_blocks, 12, channels] at qo + sl*C + c
    at::parallel_for(0, num_blocks, 0, [&](int64_t begin, int64_t end) {
    for (int64_t block = begin; block < end; ++block) {
        const float* qi = q_in + block * channels * 16;
        const float* ki = k_in + block * channels * 16;
        float*       qo = q_out + block * 12 * channels;
        float*       ko = k_out + block * 12 * channels;

        int64_t c = 0;
        for (; c + 7 < channels; c += 8) {
            const float* qx = qi + c * 16;
            const float* kx = ki + c * 16;

            // Gather blade bl across 8 channels. The values are a cache line
            // apart, so scalar loads beat vgatherdps here.
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

            // DAA query: blades {11, 12, 13, 14}.
            // rcp + one Newton step instead of div (rcp is ~12-bit, NR brings it to fp32).
            const __m256 q14    = lq(14);
            const __m256 q_den  = _mm256_fmadd_ps(q14, q14, eps_vec);   // q14*q14 + eps
            __m256 q_rcp        = _mm256_rcp_ps(q_den);
            q_rcp = _mm256_mul_ps(q_rcp, _mm256_fnmadd_ps(q_den, q_rcp, two)); // r*(2-d*r)
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
    }); // at::parallel_for
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

        at::parallel_for(0, num_blocks, 0, [&](int64_t begin, int64_t end) {
        for (int64_t block = begin; block < end; ++block) {
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
        }); // at::parallel_for
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

// Flash attention (FlashAttention-2 style): tile over query and key blocks and
// keep a running max/sum so we never materialize the full T x T score matrix.
// Per query tile, loop over key tiles updating O with the online-softmax
// correction, then normalize once at the end. Br=Bc=32 keeps the tiles in L2.

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(lo, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, sh));
}

static inline float hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_max_ps(lo, sh);
    sh = _mm_movehl_ps(lo, lo);
    return _mm_cvtss_f32(_mm_max_ss(lo, sh));
}

// Row max over s[0..n) (n <= kFlashBc). Used by the v3_2 online softmax instead
// of the scalar comparison loop. AVX2 (works on any build with __AVX2__).
static inline float rowmax_simd(const float* __restrict__ s, int64_t n) {
    int64_t c = 0;
    if (n >= 8) {
        __m256 mx = _mm256_loadu_ps(s);
        for (c = 8; c + 8 <= n; c += 8) mx = _mm256_max_ps(mx, _mm256_loadu_ps(s + c));
        float m = hmax256_ps(mx);
        for (; c < n; ++c) if (s[c] > m) m = s[c];
        return m;
    }
    float m = s[0];
    for (c = 1; c < n; ++c) if (s[c] > m) m = s[c];
    return m;
}

// 8-wide exp, Cephes minimax (~1e-6 rel error). Softmax only feeds x <= 0 and
// our tolerance is 5e-5, so this is plenty. Avoids scalar std::exp in the loop.
static inline __m256 exp256_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 C1 = _mm256_set1_ps(0.693359375f);
    const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256 fx = _mm256_fmadd_ps(x, LOG2EF, _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);

    __m256 z = _mm256_fnmadd_ps(fx, C1, x);   // x - fx*C1
    z = _mm256_fnmadd_ps(fx, C2, z);          // - fx*C2
    const __m256 z2 = _mm256_mul_ps(z, z);

    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, z2, z);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    // 2^fx by assembling the float exponent field.
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(emm0));
}
#endif

// Dot product over n floats. SIMD picks the AVX2 (v3) or scalar (v2) path.
template <bool SIMD>
static inline float flash_dot(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
#if defined(__AVX2__)
    if constexpr (SIMD) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 15 < n; k += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),     _mm256_loadu_ps(b + k),     acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
        }
        for (; k + 7 < n; k += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
        }
        float s = hsum256_ps(_mm256_add_ps(acc0, acc1));
        for (; k < n; ++k) s += a[k] * b[k];
        return s;
    }
#endif
    float s = 0.0f;
    for (int64_t k = 0; k < n; ++k) s += a[k] * b[k];
    return s;
}

// AXPY: out[v] += alpha * x[v] for dv elements.
template <bool SIMD>
static inline void flash_axpy(float* __restrict__ out, const float* __restrict__ x, float alpha, int64_t dv) {
#if defined(__AVX2__)
    if constexpr (SIMD) {
        __m256 a = _mm256_set1_ps(alpha);
        int64_t v = 0;
        for (; v + 7 < dv; v += 8) {
            _mm256_storeu_ps(out + v,
                _mm256_fmadd_ps(a, _mm256_loadu_ps(x + v), _mm256_loadu_ps(out + v)));
        }
        for (; v < dv; ++v) out[v] += alpha * x[v];
        return;
    }
#endif
    for (int64_t v = 0; v < dv; ++v) out[v] += alpha * x[v];
}

// Scale: out[v] *= alpha for dv elements.
template <bool SIMD>
static inline void flash_scale(float* __restrict__ out, float alpha, int64_t dv) {
#if defined(__AVX2__)
    if constexpr (SIMD) {
        __m256 a = _mm256_set1_ps(alpha);
        int64_t v = 0;
        for (; v + 7 < dv; v += 8) {
            _mm256_storeu_ps(out + v, _mm256_mul_ps(_mm256_loadu_ps(out + v), a));
        }
        for (; v < dv; ++v) out[v] *= alpha;
        return;
    }
#endif
    for (int64_t v = 0; v < dv; ++v) out[v] *= alpha;
}

static constexpr int64_t kFlashBr = 48;
static constexpr int64_t kFlashBc = 32;
// Toggle: direct (materialize T x T scores) vs flash (online softmax) SDPA core.
static constexpr bool kUseDirectSDPA = false;

#if defined(__AVX2__)
// Transpose a key tile [bc, d] into [d, kFlashBc] so the keys for a fixed k are
// contiguous. Done once per key tile, reused across all query rows.
static inline void pack_k_tile(const float* __restrict__ Kj, float* __restrict__ Kp,
                               int64_t bc, int64_t d) {
    for (int64_t c = 0; c < bc; ++c) {
        const float* krow = Kj + c * d;
        for (int64_t k = 0; k < d; ++k) Kp[k * kFlashBc + c] = krow[k];
    }
}

// S = scale * (Qi @ Kp). 4x8 register block: 4 query rows reuse each K vector
// load and accumulate over k with FMAs, so there's no horizontal sum per output.
static inline void qk_block_simd(const float* __restrict__ Qi,
                                 const float* __restrict__ Kp,
                                 float* __restrict__ S,
                                 int64_t br, int64_t bc, int64_t d, float scale,
                                 int64_t s_ld = kFlashBc) {
    const __m256 sv = _mm256_set1_ps(scale);
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        const float* q0 = Qi + (r0 + 0) * d;
        const float* q1 = Qi + (r0 + 1) * d;
        const float* q2 = Qi + (r0 + 2) * d;
        const float* q3 = Qi + (r0 + 3) * d;
        int64_t c0 = 0;
        for (; c0 + 16 <= bc; c0 += 16) {  // 4x16: 8 accumulators, each Q broadcast hits 2 K vectors
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            for (int64_t k = 0; k < d; ++k) {
                const __m256 kv0 = _mm256_loadu_ps(Kp + k * kFlashBc + c0);
                const __m256 kv1 = _mm256_loadu_ps(Kp + k * kFlashBc + c0 + 8);
                const __m256 w0 = _mm256_set1_ps(q0[k]);
                const __m256 w1 = _mm256_set1_ps(q1[k]);
                const __m256 w2 = _mm256_set1_ps(q2[k]);
                const __m256 w3 = _mm256_set1_ps(q3[k]);
                a0 = _mm256_fmadd_ps(w0, kv0, a0); b0 = _mm256_fmadd_ps(w0, kv1, b0);
                a1 = _mm256_fmadd_ps(w1, kv0, a1); b1 = _mm256_fmadd_ps(w1, kv1, b1);
                a2 = _mm256_fmadd_ps(w2, kv0, a2); b2 = _mm256_fmadd_ps(w2, kv1, b2);
                a3 = _mm256_fmadd_ps(w3, kv0, a3); b3 = _mm256_fmadd_ps(w3, kv1, b3);
            }
            _mm256_storeu_ps(S + (r0 + 0) * s_ld + c0,     _mm256_mul_ps(a0, sv));
            _mm256_storeu_ps(S + (r0 + 1) * s_ld + c0,     _mm256_mul_ps(a1, sv));
            _mm256_storeu_ps(S + (r0 + 2) * s_ld + c0,     _mm256_mul_ps(a2, sv));
            _mm256_storeu_ps(S + (r0 + 3) * s_ld + c0,     _mm256_mul_ps(a3, sv));
            _mm256_storeu_ps(S + (r0 + 0) * s_ld + c0 + 8, _mm256_mul_ps(b0, sv));
            _mm256_storeu_ps(S + (r0 + 1) * s_ld + c0 + 8, _mm256_mul_ps(b1, sv));
            _mm256_storeu_ps(S + (r0 + 2) * s_ld + c0 + 8, _mm256_mul_ps(b2, sv));
            _mm256_storeu_ps(S + (r0 + 3) * s_ld + c0 + 8, _mm256_mul_ps(b3, sv));
        }
        for (; c0 + 8 <= bc; c0 += 8) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int64_t k = 0; k < d; ++k) {
                const __m256 kv = _mm256_loadu_ps(Kp + k * kFlashBc + c0);
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(q0[k]), kv, a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(q1[k]), kv, a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(q2[k]), kv, a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(q3[k]), kv, a3);
            }
            _mm256_storeu_ps(S + (r0 + 0) * s_ld + c0, _mm256_mul_ps(a0, sv));
            _mm256_storeu_ps(S + (r0 + 1) * s_ld + c0, _mm256_mul_ps(a1, sv));
            _mm256_storeu_ps(S + (r0 + 2) * s_ld + c0, _mm256_mul_ps(a2, sv));
            _mm256_storeu_ps(S + (r0 + 3) * s_ld + c0, _mm256_mul_ps(a3, sv));
        }
        for (; c0 < bc; ++c0) {  // column remainder
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int64_t k = 0; k < d; ++k) {
                const float kk = Kp[k * kFlashBc + c0];
                s0 += q0[k] * kk; s1 += q1[k] * kk; s2 += q2[k] * kk; s3 += q3[k] * kk;
            }
            S[(r0 + 0) * s_ld + c0] = s0 * scale;
            S[(r0 + 1) * s_ld + c0] = s1 * scale;
            S[(r0 + 2) * s_ld + c0] = s2 * scale;
            S[(r0 + 3) * s_ld + c0] = s3 * scale;
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        const float* q0 = Qi + r0 * d;
        int64_t c0 = 0;
        for (; c0 + 8 <= bc; c0 += 8) {
            __m256 a0 = _mm256_setzero_ps();
            for (int64_t k = 0; k < d; ++k)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(q0[k]), _mm256_loadu_ps(Kp + k * kFlashBc + c0), a0);
            _mm256_storeu_ps(S + r0 * s_ld + c0, _mm256_mul_ps(a0, sv));
        }
        for (; c0 < bc; ++c0) {
            float s0 = 0;
            for (int64_t k = 0; k < d; ++k) s0 += q0[k] * Kp[k * kFlashBc + c0];
            S[r0 * s_ld + c0] = s0 * scale;
        }
    }
}

// O = diag(es) * O + P @ V, with P [br, kFlashBc] and V [bc, dv] row-major.
// 4x8 register block over (rows, dv): each V vector load is reused across 4
// query rows, and the es[] rescale is folded into the accumulator init.
static inline void pv_block_simd(float* __restrict__ O, const float* __restrict__ P,
                                 const float* __restrict__ V, const float* __restrict__ es,
                                 int64_t br, int64_t bc, int64_t dv,
                                 int64_t p_ld = kFlashBc) {
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        float* o0 = O + (r0 + 0) * dv;
        float* o1 = O + (r0 + 1) * dv;
        float* o2 = O + (r0 + 2) * dv;
        float* o3 = O + (r0 + 3) * dv;
        const float* p0 = P + (r0 + 0) * p_ld;
        const float* p1 = P + (r0 + 1) * p_ld;
        const float* p2 = P + (r0 + 2) * p_ld;
        const float* p3 = P + (r0 + 3) * p_ld;
        const __m256 e0 = _mm256_set1_ps(es[r0 + 0]);
        const __m256 e1 = _mm256_set1_ps(es[r0 + 1]);
        const __m256 e2 = _mm256_set1_ps(es[r0 + 2]);
        const __m256 e3 = _mm256_set1_ps(es[r0 + 3]);
        int64_t v = 0;
        for (; v + 8 <= dv; v += 8) {
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v), e0);
            __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(o1 + v), e1);
            __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(o2 + v), e2);
            __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(o3 + v), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m256 vv = _mm256_loadu_ps(V + c * dv + v);
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[c]), vv, a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[c]), vv, a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(p2[c]), vv, a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(p3[c]), vv, a3);
            }
            _mm256_storeu_ps(o0 + v, a0);
            _mm256_storeu_ps(o1 + v, a1);
            _mm256_storeu_ps(o2 + v, a2);
            _mm256_storeu_ps(o3 + v, a3);
        }
        for (; v < dv; ++v) {  // dv remainder
            float s0 = o0[v] * es[r0 + 0], s1 = o1[v] * es[r0 + 1];
            float s2 = o2[v] * es[r0 + 2], s3 = o3[v] * es[r0 + 3];
            for (int64_t c = 0; c < bc; ++c) {
                const float vv = V[c * dv + v];
                s0 += p0[c] * vv; s1 += p1[c] * vv; s2 += p2[c] * vv; s3 += p3[c] * vv;
            }
            o0[v] = s0; o1[v] = s1; o2[v] = s2; o3[v] = s3;
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        float* o0 = O + r0 * dv;
        const float* p0 = P + r0 * p_ld;
        const __m256 e0 = _mm256_set1_ps(es[r0]);
        int64_t v = 0;
        for (; v + 8 <= dv; v += 8) {
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v), e0);
            for (int64_t c = 0; c < bc; ++c)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[c]), _mm256_loadu_ps(V + c * dv + v), a0);
            _mm256_storeu_ps(o0 + v, a0);
        }
        for (; v < dv; ++v) {
            float s0 = o0[v] * es[r0];
            for (int64_t c = 0; c < bc; ++c) s0 += p0[c] * V[c * dv + v];
            o0[v] = s0;
        }
    }
}

// v3_3 P@V kernel: same math as pv_block_simd but NR=16 (16 value-columns per
// row block, 8 accumulators) so each p[r][c] broadcast feeds TWO FMAs. The NR=8
// pv_block_simd was load-port bound (1 V-load + 4 broadcasts : 4 FMA per step);
// NR=16 is 2 V-loads + 4 broadcasts : 8 FMA, i.e. FMA-bound. Per-element
// accumulation order is unchanged, so the result is bit-identical to pv_block_simd.
static inline void pv_block_simd_nr16(float* __restrict__ O, const float* __restrict__ P,
                                      const float* __restrict__ V, const float* __restrict__ es,
                                      int64_t br, int64_t bc, int64_t dv,
                                      int64_t p_ld = kFlashBc) {
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        float* o0 = O + (r0 + 0) * dv;
        float* o1 = O + (r0 + 1) * dv;
        float* o2 = O + (r0 + 2) * dv;
        float* o3 = O + (r0 + 3) * dv;
        const float* p0 = P + (r0 + 0) * p_ld;
        const float* p1 = P + (r0 + 1) * p_ld;
        const float* p2 = P + (r0 + 2) * p_ld;
        const float* p3 = P + (r0 + 3) * p_ld;
        const __m256 e0 = _mm256_set1_ps(es[r0 + 0]);
        const __m256 e1 = _mm256_set1_ps(es[r0 + 1]);
        const __m256 e2 = _mm256_set1_ps(es[r0 + 2]);
        const __m256 e3 = _mm256_set1_ps(es[r0 + 3]);
        int64_t v = 0;
        for (; v + 16 <= dv; v += 16) {  // 16 cols/row -> 8 accumulators (2 per row)
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v),     e0);
            __m256 a0b= _mm256_mul_ps(_mm256_loadu_ps(o0 + v + 8), e0);
            __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(o1 + v),     e1);
            __m256 a1b= _mm256_mul_ps(_mm256_loadu_ps(o1 + v + 8), e1);
            __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(o2 + v),     e2);
            __m256 a2b= _mm256_mul_ps(_mm256_loadu_ps(o2 + v + 8), e2);
            __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(o3 + v),     e3);
            __m256 a3b= _mm256_mul_ps(_mm256_loadu_ps(o3 + v + 8), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m256 vv  = _mm256_loadu_ps(V + c * dv + v);
                const __m256 vvb = _mm256_loadu_ps(V + c * dv + v + 8);
                const __m256 w0 = _mm256_set1_ps(p0[c]);
                const __m256 w1 = _mm256_set1_ps(p1[c]);
                const __m256 w2 = _mm256_set1_ps(p2[c]);
                const __m256 w3 = _mm256_set1_ps(p3[c]);
                a0 = _mm256_fmadd_ps(w0, vv, a0); a0b = _mm256_fmadd_ps(w0, vvb, a0b);
                a1 = _mm256_fmadd_ps(w1, vv, a1); a1b = _mm256_fmadd_ps(w1, vvb, a1b);
                a2 = _mm256_fmadd_ps(w2, vv, a2); a2b = _mm256_fmadd_ps(w2, vvb, a2b);
                a3 = _mm256_fmadd_ps(w3, vv, a3); a3b = _mm256_fmadd_ps(w3, vvb, a3b);
            }
            _mm256_storeu_ps(o0 + v, a0); _mm256_storeu_ps(o0 + v + 8, a0b);
            _mm256_storeu_ps(o1 + v, a1); _mm256_storeu_ps(o1 + v + 8, a1b);
            _mm256_storeu_ps(o2 + v, a2); _mm256_storeu_ps(o2 + v + 8, a2b);
            _mm256_storeu_ps(o3 + v, a3); _mm256_storeu_ps(o3 + v + 8, a3b);
        }
        for (; v + 8 <= dv; v += 8) {  // 8-wide remainder (dv not a multiple of 16)
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v), e0);
            __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(o1 + v), e1);
            __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(o2 + v), e2);
            __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(o3 + v), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m256 vv = _mm256_loadu_ps(V + c * dv + v);
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[c]), vv, a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[c]), vv, a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(p2[c]), vv, a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(p3[c]), vv, a3);
            }
            _mm256_storeu_ps(o0 + v, a0);
            _mm256_storeu_ps(o1 + v, a1);
            _mm256_storeu_ps(o2 + v, a2);
            _mm256_storeu_ps(o3 + v, a3);
        }
        for (; v < dv; ++v) {  // scalar remainder
            float s0 = o0[v] * es[r0 + 0], s1 = o1[v] * es[r0 + 1];
            float s2 = o2[v] * es[r0 + 2], s3 = o3[v] * es[r0 + 3];
            for (int64_t c = 0; c < bc; ++c) {
                const float vv = V[c * dv + v];
                s0 += p0[c] * vv; s1 += p1[c] * vv; s2 += p2[c] * vv; s3 += p3[c] * vv;
            }
            o0[v] = s0; o1[v] = s1; o2[v] = s2; o3[v] = s3;
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        float* o0 = O + r0 * dv;
        const float* p0 = P + r0 * p_ld;
        const __m256 e0 = _mm256_set1_ps(es[r0]);
        int64_t v = 0;
        for (; v + 16 <= dv; v += 16) {
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v),     e0);
            __m256 a0b= _mm256_mul_ps(_mm256_loadu_ps(o0 + v + 8), e0);
            for (int64_t c = 0; c < bc; ++c) {
                const __m256 w0 = _mm256_set1_ps(p0[c]);
                a0  = _mm256_fmadd_ps(w0, _mm256_loadu_ps(V + c * dv + v),     a0);
                a0b = _mm256_fmadd_ps(w0, _mm256_loadu_ps(V + c * dv + v + 8), a0b);
            }
            _mm256_storeu_ps(o0 + v, a0); _mm256_storeu_ps(o0 + v + 8, a0b);
        }
        for (; v + 8 <= dv; v += 8) {
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(o0 + v), e0);
            for (int64_t c = 0; c < bc; ++c)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[c]), _mm256_loadu_ps(V + c * dv + v), a0);
            _mm256_storeu_ps(o0 + v, a0);
        }
        for (; v < dv; ++v) {
            float s0 = o0[v] * es[r0];
            for (int64_t c = 0; c < bc; ++c) s0 += p0[c] * V[c * dv + v];
            o0[v] = s0;
        }
    }
}
#endif

#if defined(__AVX512F__)
// ---------------------------------------------------------------------------
// v3_2 AVX-512 micro-kernels: the AVX2 qk/pv blocks above at 16-wide (ZMM) with
// masks for the column / dv remainders (no scalar tail). Compiled only for
// AVX-512 builds (e.g. -march=tigerlake); AVX2 builds use qk_block_simd /
// pv_block_simd. On a core with one 512-bit FMA unit these stay FMA-bound, so
// the gain over AVX2 is from fewer load/broadcast/store uops and loop trips, not
// more FLOP/cycle; the ISA-independent wins (hoisted K-pack, vectorized row-max)
// are in flash_attn_head_f32_v32.

// S = scale * (Qi @ Kp). 4x32 ZMM register block (8 accumulators); masked 16
// tail for the last (partial) key tile.
static inline void qk_block_avx512(const float* __restrict__ Qi,
                                   const float* __restrict__ Kp,
                                   float* __restrict__ S,
                                   int64_t br, int64_t bc, int64_t d, float scale,
                                   int64_t s_ld = kFlashBc) {
    const __m512 sv = _mm512_set1_ps(scale);
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        const float* q0 = Qi + (r0 + 0) * d;
        const float* q1 = Qi + (r0 + 1) * d;
        const float* q2 = Qi + (r0 + 2) * d;
        const float* q3 = Qi + (r0 + 3) * d;
        int64_t c0 = 0;
        for (; c0 + 32 <= bc; c0 += 32) {  // 4x32: each Q broadcast hits 2 ZMM K vectors
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
            __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
            for (int64_t k = 0; k < d; ++k) {
                const __m512 kv0 = _mm512_loadu_ps(Kp + k * kFlashBc + c0);
                const __m512 kv1 = _mm512_loadu_ps(Kp + k * kFlashBc + c0 + 16);
                const __m512 w0 = _mm512_set1_ps(q0[k]);
                const __m512 w1 = _mm512_set1_ps(q1[k]);
                const __m512 w2 = _mm512_set1_ps(q2[k]);
                const __m512 w3 = _mm512_set1_ps(q3[k]);
                a0 = _mm512_fmadd_ps(w0, kv0, a0); b0 = _mm512_fmadd_ps(w0, kv1, b0);
                a1 = _mm512_fmadd_ps(w1, kv0, a1); b1 = _mm512_fmadd_ps(w1, kv1, b1);
                a2 = _mm512_fmadd_ps(w2, kv0, a2); b2 = _mm512_fmadd_ps(w2, kv1, b2);
                a3 = _mm512_fmadd_ps(w3, kv0, a3); b3 = _mm512_fmadd_ps(w3, kv1, b3);
            }
            _mm512_storeu_ps(S + (r0 + 0) * s_ld + c0,      _mm512_mul_ps(a0, sv));
            _mm512_storeu_ps(S + (r0 + 1) * s_ld + c0,      _mm512_mul_ps(a1, sv));
            _mm512_storeu_ps(S + (r0 + 2) * s_ld + c0,      _mm512_mul_ps(a2, sv));
            _mm512_storeu_ps(S + (r0 + 3) * s_ld + c0,      _mm512_mul_ps(a3, sv));
            _mm512_storeu_ps(S + (r0 + 0) * s_ld + c0 + 16, _mm512_mul_ps(b0, sv));
            _mm512_storeu_ps(S + (r0 + 1) * s_ld + c0 + 16, _mm512_mul_ps(b1, sv));
            _mm512_storeu_ps(S + (r0 + 2) * s_ld + c0 + 16, _mm512_mul_ps(b2, sv));
            _mm512_storeu_ps(S + (r0 + 3) * s_ld + c0 + 16, _mm512_mul_ps(b3, sv));
        }
        for (; c0 < bc; c0 += 16) {  // 16-wide, masked when fewer than 16 cols remain
            const int64_t rem = bc - c0;
            const __mmask16 mk = (rem >= 16) ? (__mmask16)0xFFFF
                                             : (__mmask16)((1u << rem) - 1u);
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            for (int64_t k = 0; k < d; ++k) {
                const __m512 kv = _mm512_maskz_loadu_ps(mk, Kp + k * kFlashBc + c0);
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(q0[k]), kv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(q1[k]), kv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(q2[k]), kv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(q3[k]), kv, a3);
            }
            _mm512_mask_storeu_ps(S + (r0 + 0) * s_ld + c0, mk, _mm512_mul_ps(a0, sv));
            _mm512_mask_storeu_ps(S + (r0 + 1) * s_ld + c0, mk, _mm512_mul_ps(a1, sv));
            _mm512_mask_storeu_ps(S + (r0 + 2) * s_ld + c0, mk, _mm512_mul_ps(a2, sv));
            _mm512_mask_storeu_ps(S + (r0 + 3) * s_ld + c0, mk, _mm512_mul_ps(a3, sv));
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        const float* q0 = Qi + r0 * d;
        for (int64_t c0 = 0; c0 < bc; c0 += 16) {
            const int64_t rem = bc - c0;
            const __mmask16 mk = (rem >= 16) ? (__mmask16)0xFFFF
                                             : (__mmask16)((1u << rem) - 1u);
            __m512 a0 = _mm512_setzero_ps();
            for (int64_t k = 0; k < d; ++k)
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(q0[k]),
                                     _mm512_maskz_loadu_ps(mk, Kp + k * kFlashBc + c0), a0);
            _mm512_mask_storeu_ps(S + r0 * s_ld + c0, mk, _mm512_mul_ps(a0, sv));
        }
    }
}

// O = diag(es) * O + P @ V. 4x16 ZMM register block over dv; masked 16 tail when
// dv is not a multiple of 16 (dv = 16C in this model, so the tail rarely fires).
static inline void pv_block_avx512(float* __restrict__ O, const float* __restrict__ P,
                                   const float* __restrict__ V, const float* __restrict__ es,
                                   int64_t br, int64_t bc, int64_t dv,
                                   int64_t p_ld = kFlashBc) {
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        float* o0 = O + (r0 + 0) * dv;
        float* o1 = O + (r0 + 1) * dv;
        float* o2 = O + (r0 + 2) * dv;
        float* o3 = O + (r0 + 3) * dv;
        const float* p0 = P + (r0 + 0) * p_ld;
        const float* p1 = P + (r0 + 1) * p_ld;
        const float* p2 = P + (r0 + 2) * p_ld;
        const float* p3 = P + (r0 + 3) * p_ld;
        const __m512 e0 = _mm512_set1_ps(es[r0 + 0]);
        const __m512 e1 = _mm512_set1_ps(es[r0 + 1]);
        const __m512 e2 = _mm512_set1_ps(es[r0 + 2]);
        const __m512 e3 = _mm512_set1_ps(es[r0 + 3]);
        int64_t v = 0;
        for (; v + 16 <= dv; v += 16) {
            __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(o0 + v), e0);
            __m512 a1 = _mm512_mul_ps(_mm512_loadu_ps(o1 + v), e1);
            __m512 a2 = _mm512_mul_ps(_mm512_loadu_ps(o2 + v), e2);
            __m512 a3 = _mm512_mul_ps(_mm512_loadu_ps(o3 + v), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m512 vv = _mm512_loadu_ps(V + c * dv + v);
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]), vv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(p1[c]), vv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(p2[c]), vv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(p3[c]), vv, a3);
            }
            _mm512_storeu_ps(o0 + v, a0);
            _mm512_storeu_ps(o1 + v, a1);
            _mm512_storeu_ps(o2 + v, a2);
            _mm512_storeu_ps(o3 + v, a3);
        }
        if (v < dv) {  // dv remainder
            const __mmask16 mk = (__mmask16)((1u << (dv - v)) - 1u);
            __m512 a0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o0 + v), e0);
            __m512 a1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o1 + v), e1);
            __m512 a2 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o2 + v), e2);
            __m512 a3 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o3 + v), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m512 vv = _mm512_maskz_loadu_ps(mk, V + c * dv + v);
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]), vv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(p1[c]), vv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(p2[c]), vv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(p3[c]), vv, a3);
            }
            _mm512_mask_storeu_ps(o0 + v, mk, a0);
            _mm512_mask_storeu_ps(o1 + v, mk, a1);
            _mm512_mask_storeu_ps(o2 + v, mk, a2);
            _mm512_mask_storeu_ps(o3 + v, mk, a3);
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        float* o0 = O + r0 * dv;
        const float* p0 = P + r0 * p_ld;
        const __m512 e0 = _mm512_set1_ps(es[r0]);
        int64_t v = 0;
        for (; v + 16 <= dv; v += 16) {
            __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(o0 + v), e0);
            for (int64_t c = 0; c < bc; ++c)
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]), _mm512_loadu_ps(V + c * dv + v), a0);
            _mm512_storeu_ps(o0 + v, a0);
        }
        if (v < dv) {
            const __mmask16 mk = (__mmask16)((1u << (dv - v)) - 1u);
            __m512 a0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o0 + v), e0);
            for (int64_t c = 0; c < bc; ++c)
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]),
                                     _mm512_maskz_loadu_ps(mk, V + c * dv + v), a0);
            _mm512_mask_storeu_ps(o0 + v, mk, a0);
        }
    }
}

// NR=32 AVX-512 P@V kernel. NOT wired into the dispatch (see flash_attn_head_f32_v32):
// the README target has a single 512-bit FMA unit, so NR=32 is no better than the
// NR=16 pv_block_avx512 there. Retained for a hypothetical dual-512-bit-FMA CPU,
// where the extra broadcast reuse would help. NR=32 (two ZMM per row, 8
// accumulators) so each P broadcast feeds two FMAs; the 16-wide masked loop
// handles dv not a multiple of 32. Bit-identical to pv_block_avx512.
static inline void pv_block_avx512_nr32(float* __restrict__ O, const float* __restrict__ P,
                                        const float* __restrict__ V, const float* __restrict__ es,
                                        int64_t br, int64_t bc, int64_t dv,
                                        int64_t p_ld = kFlashBc) {
    int64_t r0 = 0;
    for (; r0 + 4 <= br; r0 += 4) {
        float* o0 = O + (r0 + 0) * dv;
        float* o1 = O + (r0 + 1) * dv;
        float* o2 = O + (r0 + 2) * dv;
        float* o3 = O + (r0 + 3) * dv;
        const float* p0 = P + (r0 + 0) * p_ld;
        const float* p1 = P + (r0 + 1) * p_ld;
        const float* p2 = P + (r0 + 2) * p_ld;
        const float* p3 = P + (r0 + 3) * p_ld;
        const __m512 e0 = _mm512_set1_ps(es[r0 + 0]);
        const __m512 e1 = _mm512_set1_ps(es[r0 + 1]);
        const __m512 e2 = _mm512_set1_ps(es[r0 + 2]);
        const __m512 e3 = _mm512_set1_ps(es[r0 + 3]);
        int64_t v = 0;
        for (; v + 32 <= dv; v += 32) {
            __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(o0 + v), e0), a0b = _mm512_mul_ps(_mm512_loadu_ps(o0 + v + 16), e0);
            __m512 a1 = _mm512_mul_ps(_mm512_loadu_ps(o1 + v), e1), a1b = _mm512_mul_ps(_mm512_loadu_ps(o1 + v + 16), e1);
            __m512 a2 = _mm512_mul_ps(_mm512_loadu_ps(o2 + v), e2), a2b = _mm512_mul_ps(_mm512_loadu_ps(o2 + v + 16), e2);
            __m512 a3 = _mm512_mul_ps(_mm512_loadu_ps(o3 + v), e3), a3b = _mm512_mul_ps(_mm512_loadu_ps(o3 + v + 16), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m512 vv = _mm512_loadu_ps(V + c * dv + v), vvb = _mm512_loadu_ps(V + c * dv + v + 16);
                const __m512 w0 = _mm512_set1_ps(p0[c]); a0 = _mm512_fmadd_ps(w0, vv, a0); a0b = _mm512_fmadd_ps(w0, vvb, a0b);
                const __m512 w1 = _mm512_set1_ps(p1[c]); a1 = _mm512_fmadd_ps(w1, vv, a1); a1b = _mm512_fmadd_ps(w1, vvb, a1b);
                const __m512 w2 = _mm512_set1_ps(p2[c]); a2 = _mm512_fmadd_ps(w2, vv, a2); a2b = _mm512_fmadd_ps(w2, vvb, a2b);
                const __m512 w3 = _mm512_set1_ps(p3[c]); a3 = _mm512_fmadd_ps(w3, vv, a3); a3b = _mm512_fmadd_ps(w3, vvb, a3b);
            }
            _mm512_storeu_ps(o0 + v, a0); _mm512_storeu_ps(o0 + v + 16, a0b);
            _mm512_storeu_ps(o1 + v, a1); _mm512_storeu_ps(o1 + v + 16, a1b);
            _mm512_storeu_ps(o2 + v, a2); _mm512_storeu_ps(o2 + v + 16, a2b);
            _mm512_storeu_ps(o3 + v, a3); _mm512_storeu_ps(o3 + v + 16, a3b);
        }
        for (; v < dv; v += 16) {  // 16-wide masked tail for dv not a multiple of 32
            const int64_t rem = dv - v;
            const __mmask16 mk = (rem >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
            __m512 a0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o0 + v), e0);
            __m512 a1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o1 + v), e1);
            __m512 a2 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o2 + v), e2);
            __m512 a3 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o3 + v), e3);
            for (int64_t c = 0; c < bc; ++c) {
                const __m512 vv = _mm512_maskz_loadu_ps(mk, V + c * dv + v);
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]), vv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(p1[c]), vv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(p2[c]), vv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(p3[c]), vv, a3);
            }
            _mm512_mask_storeu_ps(o0 + v, mk, a0); _mm512_mask_storeu_ps(o1 + v, mk, a1);
            _mm512_mask_storeu_ps(o2 + v, mk, a2); _mm512_mask_storeu_ps(o3 + v, mk, a3);
        }
    }
    for (; r0 < br; ++r0) {  // row remainder (MR=1)
        float* o0 = O + r0 * dv;
        const float* p0 = P + r0 * p_ld;
        const __m512 e0 = _mm512_set1_ps(es[r0]);
        int64_t v = 0;
        for (; v + 32 <= dv; v += 32) {
            __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(o0 + v), e0), a0b = _mm512_mul_ps(_mm512_loadu_ps(o0 + v + 16), e0);
            for (int64_t c = 0; c < bc; ++c) {
                const __m512 w0 = _mm512_set1_ps(p0[c]);
                a0  = _mm512_fmadd_ps(w0, _mm512_loadu_ps(V + c * dv + v), a0);
                a0b = _mm512_fmadd_ps(w0, _mm512_loadu_ps(V + c * dv + v + 16), a0b);
            }
            _mm512_storeu_ps(o0 + v, a0); _mm512_storeu_ps(o0 + v + 16, a0b);
        }
        for (; v < dv; v += 16) {
            const int64_t rem = dv - v;
            const __mmask16 mk = (rem >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
            __m512 a0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, o0 + v), e0);
            for (int64_t c = 0; c < bc; ++c)
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[c]), _mm512_maskz_loadu_ps(mk, V + c * dv + v), a0);
            _mm512_mask_storeu_ps(o0 + v, mk, a0);
        }
    }
}
#endif  // __AVX512F__

// out[c] = exp(in[c] - m); returns sum(out). Vectorized exp for v3, std::exp for v2.
template <bool SIMD>
static inline float exp_row(const float* __restrict__ in, float* __restrict__ out,
                            int64_t n, float m) {
#if defined(__AVX2__)
    if constexpr (SIMD) {
        const __m256 mv = _mm256_set1_ps(m);
        __m256 acc = _mm256_setzero_ps();
        int64_t c = 0;
        for (; c + 8 <= n; c += 8) {
            const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(in + c), mv));
            _mm256_storeu_ps(out + c, e);
            acc = _mm256_add_ps(acc, e);
        }
        float s = hsum256_ps(acc);
        for (; c < n; ++c) { const float e = std::exp(in[c] - m); out[c] = e; s += e; }
        return s;
    }
#endif
    float s = 0.0f;
    for (int64_t c = 0; c < n; ++c) { const float e = std::exp(in[c] - m); out[c] = e; s += e; }
    return s;
}

// Per-head flash attention, float32. Q/K [T,d], V/O [T,dv], all contiguous.
// l, m, Kp are caller-supplied scratch. SIMD picks the v3 (AVX2) or v2 path.
template <bool SIMD>
static void flash_attn_head_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int64_t T, int64_t d, int64_t dv, float scale,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ Kp)
{
    (void)Kp;
    for (int64_t i = 0; i < T; ++i) { l[i] = 0.0f; m[i] = -1e30f; }
    std::memset(O, 0, static_cast<size_t>(T * dv) * sizeof(float));

    alignas(32) float S_block[kFlashBr * kFlashBc];
    alignas(32) float P_block[kFlashBr * kFlashBc];
    alignas(32) float es[kFlashBr];

    for (int64_t qi = 0; qi < T; qi += kFlashBr) {
        const int64_t br = std::min(kFlashBr, T - qi);
        const float* Qi = Q + qi * d;
        float*       Oi = O + qi * dv;
        float*       li = l + qi;
        float*       mi = m + qi;

        for (int64_t ki = 0; ki < T; ki += kFlashBc) {
            const int64_t bc = std::min(kFlashBc, T - ki);
            const float* Kj = K + ki * d;
            const float* Vj = V + ki * dv;

            // S = scale * Qi @ Kj^T  (br x bc)
#if defined(__AVX2__)
            if constexpr (SIMD) {
                pack_k_tile(Kj, Kp, bc, d);
                qk_block_simd(Qi, Kp, S_block, br, bc, d, scale);
            } else
#endif
            {
                for (int64_t r = 0; r < br; ++r) {
                    for (int64_t c = 0; c < bc; ++c) {
                        S_block[r * kFlashBc + c] = flash_dot<SIMD>(Qi + r * d, Kj + c * d, d) * scale;
                    }
                }
            }

            // Online softmax: fill P_block, record the per-row rescale es[r], update m/l.
            for (int64_t r = 0; r < br; ++r) {
                const float* Srow = S_block + r * kFlashBc;
                float* Prow = P_block + r * kFlashBc;

                float m_new = mi[r];
                for (int64_t c = 0; c < bc; ++c) {
                    if (Srow[c] > m_new) m_new = Srow[c];
                }
                es[r] = std::exp(mi[r] - m_new);
                li[r] = es[r] * li[r] + exp_row<SIMD>(Srow, Prow, bc, m_new);
                mi[r] = m_new;
            }

            // O = diag(es) * O + P_block @ Vj
#if defined(__AVX2__)
            if constexpr (SIMD) {
                pv_block_simd(Oi, P_block, Vj, es, br, bc, dv);
            } else
#endif
            {
                for (int64_t r = 0; r < br; ++r) {
                    float* Oi_row = Oi + r * dv;
                    flash_scale<SIMD>(Oi_row, es[r], dv);
                    const float* Prow = P_block + r * kFlashBc;
                    for (int64_t c = 0; c < bc; ++c) {
                        flash_axpy<SIMD>(Oi_row, Vj + c * dv, Prow[c], dv);
                    }
                }
            }
        }

        for (int64_t r = 0; r < br; ++r) {
            flash_scale<SIMD>(Oi + r * dv, 1.0f / li[r], dv);
        }
    }
}

// v3_2 per-head flash attention. Same online-softmax math as flash_attn_head_f32,
// with three changes over v3:
//   (1) K is packed (transposed) once per head into Kp [nKT, d, kFlashBc] and
//       reused across all query blocks (v3 re-packed each tile per query block);
//   (2) the per-row score max is computed with rowmax_simd instead of a scalar
//       comparison loop;
//   (3) the QK^T / P@V micro-kernels use AVX-512 (16-wide, masked tails) when the
//       build has __AVX512F__, else the AVX2 kernels.
// (1) and (2) are ISA-independent and apply on any AVX2 build. Kp must be sized
// nKT * d * kFlashBc floats by the caller (nKT = ceil(T / kFlashBc)).
// PV_NR16 selects the v3_3 P@V kernel (NR=16 AVX2 / NR=32 AVX-512) over the v3_2
// one; everything else is identical, so v3_3 output is bit-identical to v3_2.
template <bool SIMD, bool PV_NR16 = false>
static void flash_attn_head_f32_v32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int64_t T, int64_t d, int64_t dv, float scale,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ Kp)
{
    for (int64_t i = 0; i < T; ++i) { l[i] = 0.0f; m[i] = -1e30f; }
    std::memset(O, 0, static_cast<size_t>(T * dv) * sizeof(float));

    const int64_t nKT = (T + kFlashBc - 1) / kFlashBc;

    // (1) Hoisted K-pack: transpose every key tile once, reuse across query blocks.
#if defined(__AVX2__)
    if constexpr (SIMD) {
        for (int64_t ti = 0; ti < nKT; ++ti) {
            const int64_t ki = ti * kFlashBc;
            const int64_t bc = std::min(kFlashBc, T - ki);
            pack_k_tile(K + ki * d, Kp + ti * (d * kFlashBc), bc, d);
        }
    }
#endif

    alignas(64) float S_block[kFlashBr * kFlashBc];
    alignas(64) float P_block[kFlashBr * kFlashBc];
    alignas(64) float es[kFlashBr];

    for (int64_t qi = 0; qi < T; qi += kFlashBr) {
        const int64_t br = std::min(kFlashBr, T - qi);
        const float* Qi = Q + qi * d;
        float*       Oi = O + qi * dv;
        float*       li = l + qi;
        float*       mi = m + qi;

        for (int64_t ti = 0; ti < nKT; ++ti) {
            const int64_t ki = ti * kFlashBc;
            const int64_t bc = std::min(kFlashBc, T - ki);
            const float* Vj = V + ki * dv;

            // S = scale * Qi @ Kj^T  (br x bc)
#if defined(__AVX2__)
            if constexpr (SIMD) {
                const float* Kp_tile = Kp + ti * (d * kFlashBc);
#if defined(__AVX512F__)
                qk_block_avx512(Qi, Kp_tile, S_block, br, bc, d, scale);
#else
                qk_block_simd(Qi, Kp_tile, S_block, br, bc, d, scale);
#endif
            } else
#endif
            {
                const float* Kj = K + ki * d;
                for (int64_t r = 0; r < br; ++r) {
                    for (int64_t c = 0; c < bc; ++c) {
                        S_block[r * kFlashBc + c] = flash_dot<SIMD>(Qi + r * d, Kj + c * d, d) * scale;
                    }
                }
            }

            // (2) Online softmax with vectorized row max.
            for (int64_t r = 0; r < br; ++r) {
                const float* Srow = S_block + r * kFlashBc;
                float* Prow = P_block + r * kFlashBc;

                float m_new = mi[r];
#if defined(__AVX2__)
                if constexpr (SIMD) {
                    const float rmax = rowmax_simd(Srow, bc);
                    if (rmax > m_new) m_new = rmax;
                } else
#endif
                {
                    for (int64_t c = 0; c < bc; ++c) {
                        if (Srow[c] > m_new) m_new = Srow[c];
                    }
                }
                es[r] = std::exp(mi[r] - m_new);
                li[r] = es[r] * li[r] + exp_row<SIMD>(Srow, Prow, bc, m_new);
                mi[r] = m_new;
            }

            // O = diag(es) * O + P_block @ Vj
#if defined(__AVX2__)
            if constexpr (SIMD) {
#if defined(__AVX512F__)
                // AVX-512 intentionally ignores PV_NR16 and always uses the NR=16
                // kernel: the README target (i7-1165G7) has a single 512-bit FMA
                // unit, so P@V is already FMA-bound at NR=16 and NR=32 gives no
                // benefit (and at C=1 would degenerate to a masked path). Keeping
                // NR=16 makes the AVX-512 path identical to v3_2 so the NR-reblock
                // cannot regress on the target; the win is AVX2-only by design.
                pv_block_avx512(Oi, P_block, Vj, es, br, bc, dv);
#else
                if constexpr (PV_NR16) pv_block_simd_nr16(Oi, P_block, Vj, es, br, bc, dv);
                else                   pv_block_simd(Oi, P_block, Vj, es, br, bc, dv);
#endif
            } else
#endif
            {
                for (int64_t r = 0; r < br; ++r) {
                    float* Oi_row = Oi + r * dv;
                    flash_scale<SIMD>(Oi_row, es[r], dv);
                    const float* Prow = P_block + r * kFlashBc;
                    for (int64_t c = 0; c < bc; ++c) {
                        flash_axpy<SIMD>(Oi_row, Vj + c * dv, Prow[c], dv);
                    }
                }
            }
        }

        for (int64_t r = 0; r < br; ++r) {
            flash_scale<SIMD>(Oi + r * dv, 1.0f / li[r], dv);
        }
    }
}

// Direct (non-flash) per-head SDPA: build the full T x T scores in scratch (Sf,
// leading dim T), softmax each row, then O = P @ V. ones[] is a length-kFlashBr
// buffer of 1 for the P@V kernel's es=1 rescale.
template <bool SIMD>
static void direct_attn_head_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int64_t T, int64_t d, int64_t dv, float scale,
    float* __restrict__ Kp, float* __restrict__ Sf, const float* __restrict__ ones)
{
    (void)Kp; (void)ones;
    // Phase 1: Sf = scale * Q @ K^T  (row-major, leading dim T).
    for (int64_t qi = 0; qi < T; qi += kFlashBr) {
        const int64_t br = std::min(kFlashBr, T - qi);
        for (int64_t ki = 0; ki < T; ki += kFlashBc) {
            const int64_t bc = std::min(kFlashBc, T - ki);
#if defined(__AVX2__)
            if constexpr (SIMD) {
                pack_k_tile(K + ki * d, Kp, bc, d);
                qk_block_simd(Q + qi * d, Kp, Sf + qi * T + ki, br, bc, d, scale, /*s_ld=*/T);
            } else
#endif
            {
                for (int64_t r = 0; r < br; ++r)
                    for (int64_t c = 0; c < bc; ++c)
                        Sf[(qi + r) * T + ki + c] =
                            flash_dot<SIMD>(Q + (qi + r) * d, K + (ki + c) * d, d) * scale;
            }
        }
    }
    // Phase 2: softmax each full row.
    for (int64_t i = 0; i < T; ++i) {
        float* row = Sf + i * T;
        float m = -1e30f;
        for (int64_t j = 0; j < T; ++j) if (row[j] > m) m = row[j];
        const float s = exp_row<SIMD>(row, row, T, m);
        flash_scale<SIMD>(row, 1.0f / s, T);
    }
    // Phase 3: O = P @ V.
    std::memset(O, 0, static_cast<size_t>(T * dv) * sizeof(float));
    for (int64_t qi = 0; qi < T; qi += kFlashBr) {
        const int64_t br = std::min(kFlashBr, T - qi);
#if defined(__AVX2__)
        if constexpr (SIMD) {
            for (int64_t ki = 0; ki < T; ki += kFlashBc) {
                const int64_t bc = std::min(kFlashBc, T - ki);
                pv_block_simd(O + qi * dv, Sf + qi * T + ki, V + ki * dv, ones, br, bc, dv, /*p_ld=*/T);
            }
        } else
#endif
        {
            for (int64_t r = 0; r < br; ++r) {
                float* orow = O + (qi + r) * dv;
                const float* prow = Sf + (qi + r) * T;
                for (int64_t j = 0; j < T; ++j)
                    flash_axpy<SIMD>(orow, V + j * dv, prow[j], dv);
            }
        }
    }
}

// Naive per-head SDPA for v0/v1: per query row, score all keys, softmax, O = P@V.
// One score row at a time, plain scalar loops. The reference baseline.
static void naive_sdpa_head_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int64_t T, int64_t d, int64_t dv, float scale,
    float* __restrict__ s)
{
    for (int64_t i = 0; i < T; ++i) {
        const float* Qi = Q + i * d;

        // Scores for this query row, tracking the running max.
        float m = -1e30f;
        for (int64_t j = 0; j < T; ++j) {
            const float* Kj = K + j * d;
            float acc = 0.0f;
            for (int64_t k = 0; k < d; ++k) acc += Qi[k] * Kj[k];
            acc *= scale;
            s[j] = acc;
            if (acc > m) m = acc;
        }

        // Softmax: exponentiate (shifted by max) and accumulate the partition sum.
        float sum = 0.0f;
        for (int64_t j = 0; j < T; ++j) {
            s[j] = std::exp(s[j] - m);
            sum += s[j];
        }
        const float inv = 1.0f / sum;

        // Weighted sum of value rows.
        float* Oi = O + i * dv;
        for (int64_t v = 0; v < dv; ++v) Oi[v] = 0.0f;
        for (int64_t j = 0; j < T; ++j) {
            const float p = s[j] * inv;
            const float* Vj = V + j * dv;
            for (int64_t v = 0; v < dv; ++v) Oi[v] += p * Vj[v];
        }
    }
}

Tensor compute_naive_sdpa(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const py::object& attn_mask_obj,
    double dropout_p,
    bool is_causal,
    const py::object& scale_obj)
{
    if (query.scalar_type() != torch::kFloat32 ||
        query.dim() != 4 ||
        !query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous() ||
        !attn_mask_obj.is_none() || dropout_p > 0.0 || is_causal) {
        return compute_scaled_dot_product_attention(
            query, key, value, attn_mask_obj, dropout_p, is_causal, scale_obj);
    }

    const int64_t B  = query.size(0);
    const int64_t H  = query.size(1);
    const int64_t T  = query.size(2);
    const int64_t d  = query.size(3);
    const int64_t dv = value.size(3);

    const float scale = scale_obj.is_none()
        ? 1.0f / std::sqrt(static_cast<float>(d))
        : static_cast<float>(scale_obj.cast<double>());

    auto output = torch::empty({B, H, T, dv}, query.options());

    const float* Q_ptr = query.data_ptr<float>();
    const float* K_ptr = key.data_ptr<float>();
    const float* V_ptr = value.data_ptr<float>();
    float*       O_ptr = output.data_ptr<float>();

    at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
        std::vector<float> s_buf(T);
        for (int64_t bh = begin; bh < end; ++bh) {
            naive_sdpa_head_f32(
                Q_ptr + bh * T * d,
                K_ptr + bh * T * d,
                V_ptr + bh * T * dv,
                O_ptr + bh * T * dv,
                T, d, dv, scale,
                s_buf.data());
        }
    });

    return output;
}

// Flash attention over the whole tensor. Falls back to PyTorch SDPA for cases we
// don't handle (non-float32, masks, dropout, causal, non-4D). use_simd = v3 path.
Tensor compute_flash_attention_sdpa(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const py::object& attn_mask_obj,
    double dropout_p,
    bool is_causal,
    const py::object& scale_obj,
    bool use_simd)
{
    if (query.scalar_type() != torch::kFloat32 ||
        query.dim() != 4 ||
        !query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous() ||
        !attn_mask_obj.is_none() || dropout_p > 0.0 || is_causal) {
        return compute_scaled_dot_product_attention(
            query, key, value, attn_mask_obj, dropout_p, is_causal, scale_obj);
    }

    const int64_t B  = query.size(0);
    const int64_t H  = query.size(1);
    const int64_t T  = query.size(2);
    const int64_t d  = query.size(3);
    const int64_t dv = value.size(3);

    const float scale = scale_obj.is_none()
        ? 1.0f / std::sqrt(static_cast<float>(d))
        : static_cast<float>(scale_obj.cast<double>());

    // Each head's slice is fully written by flash_attn_head_f32 (it memsets O
    // before accumulating), so we can skip the redundant zero-fill here.
    auto output = torch::empty({B, H, T, dv}, query.options());

    const float* Q_ptr = query.data_ptr<float>();
    const float* K_ptr = key.data_ptr<float>();
    const float* V_ptr = value.data_ptr<float>();
    float*       O_ptr = output.data_ptr<float>();

    // Parallelize over (batch, head) pairs; each thread owns its l/m scratch and
    // (for the SIMD path) a transposed key-tile pack buffer of d * kFlashBc floats.
    at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
        std::vector<float> l_buf(T), m_buf(T);
        std::vector<float> kp_buf(use_simd ? static_cast<size_t>(d * kFlashBc) : 0);
        for (int64_t bh = begin; bh < end; ++bh) {
            if (use_simd) {
                flash_attn_head_f32<true>(
                    Q_ptr + bh * T * d,
                    K_ptr + bh * T * d,
                    V_ptr + bh * T * dv,
                    O_ptr + bh * T * dv,
                    T, d, dv, scale,
                    l_buf.data(), m_buf.data(), kp_buf.data());
            } else {
                flash_attn_head_f32<false>(
                    Q_ptr + bh * T * d,
                    K_ptr + bh * T * d,
                    V_ptr + bh * T * dv,
                    O_ptr + bh * T * dv,
                    T, d, dv, scale,
                    l_buf.data(), m_buf.data(), nullptr);
            }
        }
    });

    return output;
}

// Assemble one head's IPA+DAA flattened Q/K rows [T, 12C] from the raw [T,C,16]
// multivectors. Each channel's 12 outputs are contiguous (IPA in 0-6, DAA in
// 7-11). Vectorizing this was a wash (it's gather-bound on the strided blade
// reads), so the scalar form is kept for both v2 and v3.
static inline void assemble_ipa_daa_head(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ Qh, float* __restrict__ Kh,
    int64_t T, int64_t C, float ipa_w, float daa_w, float eps) {
    const int64_t d = 12 * C;
    for (int64_t t = 0; t < T; ++t) {
        const float* qrow = q + t * C * 16;
        const float* krow = k + t * C * 16;
        float* qo_row = Qh + t * d;
        float* ko_row = Kh + t * d;
        for (int64_t c = 0; c < C; ++c) {
            const float* qx = qrow + c * 16;
            const float* kx = krow + c * 16;
            float* qo = qo_row + c * 12;
            float* ko = ko_row + c * 12;

            qo[0] = qx[0] * ipa_w;  qo[1] = qx[2] * ipa_w;  qo[2] = qx[3] * ipa_w;
            qo[3] = qx[4] * ipa_w;  qo[4] = qx[8] * ipa_w;  qo[5] = qx[9] * ipa_w;
            qo[6] = qx[10] * ipa_w;
            ko[0] = kx[0];  ko[1] = kx[2];  ko[2] = kx[3];  ko[3] = kx[4];
            ko[4] = kx[8];  ko[5] = kx[9];  ko[6] = kx[10];

            const float qn = qx[14] / (qx[14] * qx[14] + eps);
            const float qn0 = qx[11] * qn, qn1 = qx[12] * qn, qn2 = qx[13] * qn, qn3 = qx[14] * qn;
            qo[7]  = (qn0 * qn0 + qn1 * qn1 + qn2 * qn2) * daa_w;
            qo[8]  = (qn3 * qn3) * daa_w;
            qo[9]  = (qn0 * qn3) * daa_w;
            qo[10] = (qn1 * qn3) * daa_w;
            qo[11] = (qn2 * qn3) * daa_w;

            const float kn = kx[14] / (kx[14] * kx[14] + eps);
            const float kn0 = kx[11] * kn, kn1 = kx[12] * kn, kn2 = kx[13] * kn, kn3 = kx[14] * kn;
            ko[7]  = -(kn3 * kn3);
            ko[8]  = -(kn0 * kn0 + kn1 * kn1 + kn2 * kn2);
            ko[9]  = 2.0f * kn0 * kn3;
            ko[10] = 2.0f * kn1 * kn3;
            ko[11] = 2.0f * kn2 * kn3;
        }
    }
}

// Fused IPA+DAA flash attention: assemble each head's Q/K into thread-local
// scratch that stays in cache, then run the flash SDPA on it directly. This
// avoids materializing the global query_flat/key_flat tensors (a full DRAM
// write + read-back), which is the dominant cost once the SDPA is tiled.
Tensor compute_fused_ipa_daa_flash_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    float ipa_w, float daa_w, double eps,
    const py::object& scale_obj, bool use_simd, bool use_v32 = false) {
    const int64_t B = query.size(0);
    const int64_t H = query.size(1);
    const int64_t T = query.size(2);
    const int64_t C = query.size(3);
    const int64_t d = 12 * C;
    const int64_t dv = 16 * C;
    // v3_2 packs all key tiles once per head, so its pack buffer holds nKT tiles
    // rather than one.
    const int64_t nKT = (T + kFlashBc - 1) / kFlashBc;

    const float scale = scale_obj.is_none()
        ? 1.0f / std::sqrt(static_cast<float>(d))
        : static_cast<float>(scale_obj.cast<double>());
    const float epsf = static_cast<float>(eps);

    auto output = torch::empty({B, H, T, C, 16}, query.options());
    const float* Q = query.data_ptr<float>();
    const float* K = key.data_ptr<float>();
    const float* V = value.data_ptr<float>();
    float*       O = output.data_ptr<float>();

    at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
        std::vector<float> Qh(static_cast<size_t>(T * d));
        std::vector<float> Kh(static_cast<size_t>(T * d));
        std::vector<float> l_buf(T), m_buf(T);
        std::vector<float> kp_buf(use_simd
            ? static_cast<size_t>((use_v32 ? nKT : 1) * d * kFlashBc) : 0);
        std::vector<float> sf_buf(kUseDirectSDPA ? static_cast<size_t>(T * T) : 0);
        std::vector<float> ones_buf(kUseDirectSDPA ? static_cast<size_t>(kFlashBr) : 0, 1.0f);
        for (int64_t bh = begin; bh < end; ++bh) {
            const float* q = Q + bh * T * C * 16;
            const float* k = K + bh * T * C * 16;
            const float* v = V + bh * T * dv;   // value flat [T, 16C] == [T, dv]
            float*       o = O + bh * T * dv;
            assemble_ipa_daa_head(q, k, Qh.data(), Kh.data(), T, C, ipa_w, daa_w, epsf);
            if (kUseDirectSDPA) {
                if (use_simd)
                    direct_attn_head_f32<true>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                               kp_buf.data(), sf_buf.data(), ones_buf.data());
                else
                    direct_attn_head_f32<false>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                                nullptr, sf_buf.data(), nullptr);
            } else if (use_v32) {
                // v3_1 optimized attention: hoisted K-pack + vectorized row-max +
                // NR-reblocked P@V (NR=16 on AVX2; AVX-512 stays NR=16 per option A).
                if (use_simd)
                    flash_attn_head_f32_v32<true, true>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                                        l_buf.data(), m_buf.data(), kp_buf.data());
                else
                    flash_attn_head_f32_v32<false, true>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                                         l_buf.data(), m_buf.data(), nullptr);
            } else if (use_simd) {
                flash_attn_head_f32<true>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                          l_buf.data(), m_buf.data(), kp_buf.data());
            } else {
                flash_attn_head_f32<false>(Qh.data(), Kh.data(), v, o, T, d, dv, scale,
                                           l_buf.data(), m_buf.data(), nullptr);
            }
        }
    });
    return output;
}

// Decide whether the fused IPA+DAA path applies (float32, contiguous 5D
// [B,H,T,C,16], exactly kinds {ipa, daa} with ipa first, scalar weights, no
// mask/dropout/causal). Returns the fused output if so.
std::optional<Tensor> try_fused_ipa_daa_flash(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::vector<std::string>& kind_names,
    const std::vector<py::object>& kind_kwargs,
    const std::vector<py::object>& weights,
    const py::object& attn_mask, double dropout_p, bool is_causal,
    const py::object& scale, bool use_simd, bool use_v32 = false) {
    if (kind_names.size() != 2) return std::nullopt;
    if (!(kind_names[0] == "ipa" && kind_names[1] == "daa")) return std::nullopt;
    if (query.scalar_type() != torch::kFloat32 ||
        key.scalar_type() != torch::kFloat32 ||
        value.scalar_type() != torch::kFloat32) return std::nullopt;
    if (query.dim() != 5 || key.dim() != 5 || value.dim() != 5) return std::nullopt;
    if (!query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous()) return std::nullopt;
    if (!query.device().is_cpu()) return std::nullopt;
    if (!attn_mask.is_none() || dropout_p > 0.0 || is_causal) return std::nullopt;

    auto ipa_w = numeric_weight(weights[0]);
    auto daa_w = numeric_weight(weights[1]);
    if (!ipa_w.has_value() || !daa_w.has_value()) return std::nullopt;

    double eps = 1e-3;
    if (!kind_kwargs[1].is_none()) {
        auto kwargs = py::cast<py::dict>(kind_kwargs[1]);
        if (kwargs.contains("eps")) eps = py::cast<double>(kwargs["eps"]);
    }

    return compute_fused_ipa_daa_flash_attention(
        query, key, value, static_cast<float>(*ipa_w), static_cast<float>(*daa_w),
        eps, scale, use_simd, use_v32);
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
    bool use_simd,
    int sdpa_mode = 0) {
    // sdpa_mode selects how the scaled-dot-product-attention core is computed:
    //   0 = naive scalar (v0/v1), 1 = flash tiled scalar (v2),
    //   2 = flash tiled AVX2/FMA (v3),
    //   3 = v3_1 optimized (hoisted K-pack + vectorized row-max + NR-reblocked
    //       P@V; AVX-512 kernels when the build targets a CPU with AVX-512, else
    //       the AVX2 kernels),
    //   other = PyTorch library SDPA.
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

    // Fused path for the flash SDPA modes (v2/v3): assemble Q/K per head into
    // cache-resident scratch and run the SDPA on it, skipping the global
    // query_flat/key_flat materialization.
    if (sdpa_mode == 1 || sdpa_mode == 2 || sdpa_mode == 3) {
        auto fused = try_fused_ipa_daa_flash(
            query, key, value, kind_names, kind_kwargs, weights,
            attn_mask, dropout_p, is_causal, scale,
            /*use_simd=*/sdpa_mode >= 2, /*use_v32=*/sdpa_mode == 3);
        if (fused.has_value()) {
            return *fused;
        }
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

    Tensor ret;
    switch (sdpa_mode) {
        case 0:  // naive scalar SDPA (v0/v1 baseline)
            ret = compute_naive_sdpa(
                query_flat, key_flat, value_flat,
                attn_mask, dropout_p, is_causal, scale);
            break;
        case 1:  // flash attention, scalar inner kernel (v2: tiling / memory layout)
            ret = compute_flash_attention_sdpa(
                query_flat, key_flat, value_flat,
                attn_mask, dropout_p, is_causal, scale, /*use_simd=*/false);
            break;
        case 2:  // flash attention, AVX2/FMA inner kernel (v3: SIMD)
        case 3:  // v3_1 optimized: if the fused path declined, fall back to v3 flash SDPA
            ret = compute_flash_attention_sdpa(
                query_flat, key_flat, value_flat,
                attn_mask, dropout_p, is_causal, scale, /*use_simd=*/true);
            break;
        default:  // PyTorch library SDPA (reference fallback)
            ret = compute_scaled_dot_product_attention(
                query_flat, key_flat, value_flat,
                attn_mask, dropout_p, is_causal, scale);
            break;
    }
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
    // Naive everything: einsum QK assembly + naive scalar SDPA core.
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale,
        /*use_cache=*/false, /*use_direct_daa=*/false, /*use_fast_paths=*/false,
        /*use_simd=*/false, /*sdpa_mode=*/0);
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
    // Math: explicit DAA formula instead of einsum, plus cached constants
    // (IPA selector, DAA basis). SDPA core is still the naive scalar one.
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale,
        /*use_cache=*/true, /*use_direct_daa=*/true, /*use_fast_paths=*/false,
        /*use_simd=*/false, /*sdpa_mode=*/0);
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
    // Scalar memory/compiler optimizations: compact single-pass Q/K assembly +
    // flash-attention SDPA core (tiling + online softmax, scalar inner loops).
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale,
        /*use_cache=*/true, /*use_direct_daa=*/true, /*use_fast_paths=*/true,
        /*use_simd=*/false, /*sdpa_mode=*/1);
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
    // SIMD: fused per-head Q/K assembly + AVX2 flash SDPA (packed-K register-
    // blocked QK^T micro-kernel, register-blocked P@V, vectorized exp). The
    // assembly stays in cache scratch so query_flat/key_flat never hit DRAM.
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale,
        /*use_cache=*/true, /*use_direct_daa=*/true, /*use_fast_paths=*/true,
        /*use_simd=*/true, /*sdpa_mode=*/2);
}

torch::Tensor equi_geometric_attention_mv_only_ver_3_1(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight,
    const py::object& attn_mask,
    double dropout_p,
    bool is_causal,
    const py::object& scale) {
    // v3_1 optimized attention: fused IPA+DAA assembly (as v3) with the optimized
    // flash SDPA — K packed once per head, vectorized row-max, NR-reblocked P@V,
    // and AVX-512 QK^T / P@V micro-kernels when the build has AVX-512 (else the
    // AVX2 kernels). Bit-identical to v3 up to fp reassociation.
    return equi_geometric_attention_mv_only_impl(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale,
        /*use_cache=*/true, /*use_direct_daa=*/true, /*use_fast_paths=*/true,
        /*use_simd=*/true, /*sdpa_mode=*/3);
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
