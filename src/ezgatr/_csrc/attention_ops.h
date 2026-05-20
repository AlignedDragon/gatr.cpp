#pragma once

#include <torch/extension.h>

namespace py = pybind11;

namespace ezgatr { namespace opt {

torch::Tensor equi_geometric_attention_mv_only_ver_0(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight = py::none(),
    const py::object& attn_mask = py::none(),
    double dropout_p = 0.0,
    bool is_causal = false,
    const py::object& scale = py::none());

torch::Tensor equi_geometric_attention_mv_only_ver_1(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight = py::none(),
    const py::object& attn_mask = py::none(),
    double dropout_p = 0.0,
    bool is_causal = false,
    const py::object& scale = py::none());

torch::Tensor equi_geometric_attention_mv_only_ver_2(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight = py::none(),
    const py::object& attn_mask = py::none(),
    double dropout_p = 0.0,
    bool is_causal = false,
    const py::object& scale = py::none());

torch::Tensor equi_geometric_attention_mv_only_ver_3(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight = py::none(),
    const py::object& attn_mask = py::none(),
    double dropout_p = 0.0,
    bool is_causal = false,
    const py::object& scale = py::none());

torch::Tensor equi_geometric_attention_mv_only(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const py::dict& kinds,
    const py::object& weight = py::none(),
    const py::object& attn_mask = py::none(),
    double dropout_p = 0.0,
    bool is_causal = false,
    const py::object& scale = py::none());

}}  // namespace ezgatr::opt
