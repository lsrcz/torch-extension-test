#include "lltm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_cpu", &lltm_forward, "LLTM forward");
  m.def("backward_cpu", &lltm_backward, "LLTM backward");
  m.def("forward_cuda", &lltm_cuda_forward_with_check, "LLTM forward (CUDA)");
  m.def("backward_cuda", &lltm_cuda_backward_with_check,
        "LLTM backward (CUDA)");
}
