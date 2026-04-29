#pragma once
#include <torch/torch.h>

namespace ezgatr { namespace opt {

torch::Tensor geometric_product(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor equi_join(const torch::Tensor& x,
                        const torch::Tensor& y,
                        const c10::optional<torch::Tensor>& reference);

torch::Tensor load_gp_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor load_op_basis(c10::Device device, c10::ScalarType dtype);
torch::Tensor compute_join_kernel(c10::Device device, c10::ScalarType dtype);

torch::Tensor equi_dual(const torch::Tensor& x);
torch::Tensor outer_product(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor inner_product(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_rms_norm(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);

torch::Tensor scaler_gated_gelu(const torch::Tensor& x,
                                const std::string& approximate);

}}  // namespace ezgatr::opt
