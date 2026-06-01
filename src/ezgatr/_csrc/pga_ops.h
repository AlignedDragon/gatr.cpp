#pragma once
#include <torch/torch.h>

namespace ezgatr { namespace opt {

torch::Tensor geometric_product(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v0(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v1(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_3(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_4(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_1(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_2(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_5(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_6(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v2_7(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_v3(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_join(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v0(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v1(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_3(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_4(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_1(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_2(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_5(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_6(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v2_7(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_v3(const torch::Tensor& x, const torch::Tensor& y, const c10::optional<torch::Tensor>& reference);

torch::Tensor load_gp_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor load_op_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor compute_join_kernel(c10::Device device, c10::ScalarType dtype);

torch::Tensor equi_dual(const torch::Tensor& x);
torch::Tensor outer_product(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_linear(const torch::Tensor& x,
                           const torch::Tensor& weight,
                           const c10::optional<torch::Tensor>& bias,
                           bool normalize_basis,
                           int64_t version);

torch::Tensor equi_linear_v0(const torch::Tensor& x, const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis);
torch::Tensor equi_linear_v1(const torch::Tensor& x, const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis);
torch::Tensor equi_linear_v2(const torch::Tensor& x, const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis);
torch::Tensor equi_linear_v3(const torch::Tensor& x, const torch::Tensor& weight,
                             const c10::optional<torch::Tensor>& bias,
                             bool normalize_basis);

}}  // namespace ezgatr::opt
