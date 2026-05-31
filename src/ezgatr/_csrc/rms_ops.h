#pragma once
#include <torch/torch.h>

namespace ezgatr { namespace opt {

torch::Tensor inner_product_ver_0(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_rms_norm_ver_0(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);

torch::Tensor scaler_gated_gelu_ver_0(const torch::Tensor& x,
                                const std::string& approximate);

torch::Tensor inner_product_ver_1(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_rms_norm_ver_1(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);

torch::Tensor scaler_gated_gelu_ver_1(const torch::Tensor& x,
                                const std::string& approximate);

torch::Tensor inner_product_ver_2(const torch::Tensor& x, const torch::Tensor& y);

torch::Tensor equi_rms_norm_ver_2(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);

torch::Tensor scaler_gated_gelu_ver_2(const torch::Tensor& x,
                                const std::string& approximate);

torch::Tensor equi_rms_norm_ver_3(const torch::Tensor& x,
                            const c10::optional<torch::Tensor>& weight,
                            const c10::optional<double>& eps_opt);




}}  // namespace ezgatr::opt
