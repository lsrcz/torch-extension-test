#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> lltm_cuda_forward_with_check(torch::Tensor input,
                                                        torch::Tensor weights,
                                                        torch::Tensor bias,
                                                        torch::Tensor old_h,
                                                        torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward_with_check(
    torch::Tensor grad_h, torch::Tensor grad_cell, torch::Tensor new_cell,
    torch::Tensor input_gate, torch::Tensor output_gate,
    torch::Tensor candidate_cell, torch::Tensor X, torch::Tensor gate_weights,
    torch::Tensor weights);

std::vector<torch::Tensor> lltm_forward(torch::Tensor input,
                                        torch::Tensor weights,
                                        torch::Tensor bias, torch::Tensor old_h,
                                        torch::Tensor old_cell);

std::vector<torch::Tensor>
lltm_backward(torch::Tensor grad_h, torch::Tensor grad_cell,
              torch::Tensor new_cell, torch::Tensor input_gate,
              torch::Tensor output_gate, torch::Tensor candidate_cell,
              torch::Tensor X, torch::Tensor gate_weights,
              torch::Tensor weights);
