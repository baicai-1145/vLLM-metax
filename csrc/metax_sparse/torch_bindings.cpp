#include "core/registration.h"

#include <torch/all.h>

namespace metax_sparse {

void softmax_fp32_out(torch::Tensor const& input, torch::Tensor& out);
void gemm_bf16_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                        torch::Tensor& out);
void gemm_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                   torch::Tensor& out);

}  // namespace metax_sparse

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Caller-owned float32 row softmax.
  m.def("softmax_fp32_out(Tensor input, Tensor! out) -> ()");
  m.impl("softmax_fp32_out", torch::kCUDA,
         &metax_sparse::softmax_fp32_out);

  // In-place BF16 GEMM with FP32 accumulation/output: out = a @ b.T.
  m.def("gemm_bf16_fp32_out(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("gemm_bf16_fp32_out", torch::kCUDA,
         &metax_sparse::gemm_bf16_fp32_out);

  // In-place float32 GEMM: out = a @ b.T.
  m.def("gemm_fp32_out(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("gemm_fp32_out", torch::kCUDA, &metax_sparse::gemm_fp32_out);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
