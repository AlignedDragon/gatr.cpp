#pragma once
#include <torch/torch.h>

namespace ezgatr { namespace opt {

torch::Tensor inner_product(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_rms_norm(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);

torch::Tensor scaler_gated_gelu(const torch::Tensor& x,
                                const std::string& approximate);

}}  // namespace ezgatr::opt
