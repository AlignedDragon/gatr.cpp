#include "attention_ops.h"

#include <cmath>
#include <limits>
#include <map>
#include <mutex>
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
std::map<LongCacheKey, Tensor> g_tri_vector_selector;
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

Tensor build_tri_vector_selector(c10::Device device) {
    return torch::tensor(
        {11, 12, 13, 14},
        torch::TensorOptions().device(device).dtype(torch::kLong));
}

Tensor load_tri_vector_selector(c10::Device device) {
    auto key = make_long_key(device);
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_tri_vector_selector.find(key);
    if (it != g_tri_vector_selector.end()) {
        return it->second;
    }

    auto selector = build_tri_vector_selector(device);
    g_tri_vector_selector.emplace(key, selector);
    return selector;
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

Tensor get_tri_vector_selector(c10::Device device, bool use_cache) {
    return use_cache ? load_tri_vector_selector(device) : build_tri_vector_selector(device);
}

Tensor get_inner_product_selector(c10::Device device, bool use_cache) {
    return use_cache ? load_inner_product_selector(device) : build_inner_product_selector(device);
}

std::pair<Tensor, Tensor> get_daa_basis(c10::Device device, c10::ScalarType dtype, bool use_cache) {
    return use_cache ? load_daa_basis(device, dtype) : build_daa_basis(device, dtype);
}

Tensor build_daa_qk(const Tensor& q_or_k, const Tensor& basis, double eps, bool use_cache) {
    auto selector = get_tri_vector_selector(q_or_k.device(), use_cache);
    auto tri = q_or_k.index_select(-1, selector);
    auto normalized =
        tri * linear_square_normalizer(tri.index({Ellipsis, Slice(3, 4)}), eps);
    return torch::einsum("ijk, ...i, ...j -> ...k", {basis, normalized, normalized});
}

Tensor build_daa_qk_explicit(const Tensor& q_or_k, double eps, bool is_query, bool use_cache) {
    auto selector = get_tri_vector_selector(q_or_k.device(), use_cache);
    auto tri = q_or_k.index_select(-1, selector);
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

std::pair<Tensor, Tensor> compute_qk_for_daa_opt2(
    const Tensor& query,
    const Tensor& key,
    double eps,
    bool use_cache) {
    return {
        build_daa_qk_explicit(query, eps, true, use_cache),
        build_daa_qk_explicit(key, eps, false, use_cache),
    };
}

Tensor apply_attention_mask(Tensor scores, const py::object& attn_mask_obj) {
    if (attn_mask_obj.is_none()) {
        return scores;
    }

    auto attn_mask = attn_mask_obj.cast<Tensor>();
    if (attn_mask.scalar_type() == torch::kBool) {
        return scores.masked_fill(attn_mask.logical_not(), -std::numeric_limits<double>::infinity());
    }
    return scores + attn_mask;
}

Tensor apply_causal_mask(Tensor scores) {
    const auto q_tokens = scores.size(-2);
    const auto k_tokens = scores.size(-1);
    auto mask = torch::ones(
                    {q_tokens, k_tokens},
                    torch::TensorOptions().device(scores.device()).dtype(torch::kBool))
                    .triu(1);
    return scores.masked_fill(mask, -std::numeric_limits<double>::infinity());
}

Tensor compute_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const py::object& attn_mask_obj,
    double dropout_p,
    bool is_causal,
    const py::object& scale_obj) {
    double scale = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
    if (!scale_obj.is_none()) {
        scale = scale_obj.cast<double>();
    }

    auto scores = torch::matmul(query, key.transpose(-2, -1)) * scale;
    scores = apply_attention_mask(scores, attn_mask_obj);
    if (is_causal) {
        scores = apply_causal_mask(scores);
    }

    auto attention = torch::softmax(scores, -1);
    if (dropout_p > 0.0) {
        attention = torch::dropout(attention, dropout_p, true);
    }
    return torch::matmul(attention, value);
}

Tensor apply_query_weight(const Tensor& query_part, const py::handle& weight_item) {
    if (py::isinstance<py::float_>(weight_item) || py::isinstance<py::int_>(weight_item)) {
        return query_part * py::cast<double>(weight_item);
    }
    return query_part * py::cast<Tensor>(weight_item);
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
    bool use_direct_daa) {
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
                std::tie(q_part, k_part) = compute_qk_for_daa_opt2(query, key, eps, use_cache);
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

    auto query_flat = torch::cat(qs, -1);
    auto key_flat = torch::cat(ks, -1);
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

torch::Tensor equi_geometric_attention_mv_only_base(
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
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, false, false);
}

torch::Tensor equi_geometric_attention_mv_only_opt1(
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
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, true, false);
}

torch::Tensor equi_geometric_attention_mv_only_opt2(
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
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale, true, true);
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
    return equi_geometric_attention_mv_only_opt2(
        query, key, value, kinds, weight, attn_mask, dropout_p, is_causal, scale);
}

}}  // namespace ezgatr::opt
