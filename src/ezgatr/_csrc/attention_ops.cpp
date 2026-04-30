#include "attention_ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ezgatr { namespace opt {

namespace {

using torch::Tensor;
using namespace torch::indexing;

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

Tensor compute_tri_vector_selector(const torch::Device& device) {
    return torch::tensor(
        {11, 12, 13, 14},
        torch::TensorOptions().device(device).dtype(torch::kLong));
}

Tensor compute_inner_product_selector(const torch::Device& device) {
    return torch::tensor(
        {0, 2, 3, 4, 8, 9, 10},
        torch::TensorOptions().device(device).dtype(torch::kLong));
}

std::pair<Tensor, Tensor> compute_daa_basis(
    const torch::Device& device,
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

Tensor linear_square_normalizer(const Tensor& e123, double eps) {
    return e123 / (e123.pow(2) + eps);
}

Tensor build_daa_qk(const Tensor& q_or_k, const Tensor& basis, double eps) {
    auto selector = compute_tri_vector_selector(q_or_k.device());
    auto tri = q_or_k.index_select(-1, selector);
    auto normalized =
        tri * linear_square_normalizer(tri.index({Ellipsis, Slice(3, 4)}), eps);
    return torch::einsum("ijk, ...i, ...j -> ...k", {basis, normalized, normalized});
}

std::pair<Tensor, Tensor> compute_qk_for_daa(
    const Tensor& query,
    const Tensor& key,
    double eps) {
    auto [bq, bk] = compute_daa_basis(query.device(), query.scalar_type());
    return {
        build_daa_qk(query, bq, eps),
        build_daa_qk(key, bk, eps),
    };
}

std::pair<Tensor, Tensor> compute_qk_for_ipa(
    const Tensor& query,
    const Tensor& key) {
    auto selector = compute_inner_product_selector(query.device());
    return {
        query.index_select(-1, selector),
        key.index_select(-1, selector),
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
            std::tie(q_part, k_part) = compute_qk_for_ipa(query, key);
        } else if (kind == "daa") {
            double eps = 1e-3;
            if (!kwargs_obj.is_none()) {
                auto kwargs = py::cast<py::dict>(kwargs_obj);
                if (kwargs.contains("eps")) {
                    eps = py::cast<double>(kwargs["eps"]);
                }
            }
            std::tie(q_part, k_part) = compute_qk_for_daa(query, key, eps);
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

}}  // namespace ezgatr::opt
