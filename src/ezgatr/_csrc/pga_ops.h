#pragma once
#include <torch/torch.h>

namespace ezgatr { namespace opt {

torch::Tensor geometric_product(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_dense(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor geometric_product_sparse_rt(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_join(const torch::Tensor& x,
                        const torch::Tensor& y,
                        const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_dense(const torch::Tensor& x,
                              const torch::Tensor& y,
                              const c10::optional<torch::Tensor>& reference);
torch::Tensor equi_join_sparse_rt(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const c10::optional<torch::Tensor>& reference);

torch::Tensor load_gp_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor load_op_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor compute_join_kernel(c10::Device device, c10::ScalarType dtype);

torch::Tensor equi_dual(const torch::Tensor& x);
torch::Tensor outer_product(const torch::Tensor& x, const torch::Tensor& y);

}}  // namespace ezgatr::opt
