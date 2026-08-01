#include "core/registration.h"

#include <torch/all.h>

namespace metax_sparse {

void softmax_fp32_out(torch::Tensor const& input, torch::Tensor& out);
void mhc_sinkhorn_fp32_out(torch::Tensor const& input, torch::Tensor& out,
                           double eps, int64_t repeat);
void mhc_sinkhorn_rms_norm_out(
    torch::Tensor const& input, torch::Tensor const& pre_norm,
    torch::Tensor const& weight, torch::Tensor& comb_out,
    torch::Tensor& norm_out, double sinkhorn_eps, double norm_eps,
    int64_t repeat);
void mhc_raw_fp32_out(torch::Tensor const& residual, torch::Tensor const& weight,
                      torch::Tensor& gemm_out, torch::Tensor& sqrsum_out);
void mhc_cast_sqrsum_out(torch::Tensor const& residual,
                         torch::Tensor& residual_fp32,
                         torch::Tensor& sqrsum_out);
void mhc_downstream_rms_out(
    torch::Tensor const& residual, torch::Tensor const& gemm_out,
    torch::Tensor const& sqrsum, torch::Tensor const& scale,
    torch::Tensor const& base, torch::Tensor const& norm_weight,
    torch::Tensor& post_out, torch::Tensor& comb_out,
    torch::Tensor& pre_norm_out, torch::Tensor& norm_out, double rms_eps,
    double pre_eps, double sinkhorn_eps, double post_mult, int64_t repeat);
void mhc_sigmoid_probe_out(torch::Tensor const& input, torch::Tensor& out);
void mhc_gemv_fp32_out(torch::Tensor const& input, torch::Tensor const& weight,
                       torch::Tensor& out);
void mhc_gemv_fp32_grouped_out(torch::Tensor const& input,
                               torch::Tensor const& weight,
                               torch::Tensor& out);
void gemv_bf16_serial_rows_out(torch::Tensor const& input,
                               torch::Tensor const& weight,
                               torch::Tensor& out);
void gemv_bf16_exact_grouped_rows_out(torch::Tensor const& input,
                                      torch::Tensor const& weight,
                                      torch::Tensor& out);
void gemv_bf16_exact_oproj_grouped_rows_out(torch::Tensor const& input,
                                            torch::Tensor const& weight,
                                            torch::Tensor& out);
void gemv_bf16_exact_oproj_row_list_out(
    c10::List<torch::Tensor> const& inputs, torch::Tensor const& weight,
    torch::Tensor& out);
void gemv_bf16_fp32_serial_rows_out(torch::Tensor const& input,
                                    torch::Tensor const& weight,
                                    torch::Tensor& out);
void gemm_bf16_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                        torch::Tensor& out);
void gemm_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                   torch::Tensor& out);
void gemm_fp32_strided_batched_out(torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor& out);

}  // namespace metax_sparse

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Caller-owned float32 row softmax.
  m.def("softmax_fp32_out(Tensor input, Tensor! out) -> ()");
  m.impl("softmax_fp32_out", torch::kCUDA,
         &metax_sparse::softmax_fp32_out);
  m.def("mhc_sinkhorn_fp32_out(Tensor input, Tensor! out, float eps, int repeat) -> ()");
  m.impl("mhc_sinkhorn_fp32_out", torch::kCUDA,
         &metax_sparse::mhc_sinkhorn_fp32_out);
  m.def("mhc_sinkhorn_rms_norm_out(Tensor input, Tensor pre_norm, Tensor weight, Tensor! comb_out, Tensor! norm_out, float sinkhorn_eps, float norm_eps, int repeat) -> ()");
  m.impl("mhc_sinkhorn_rms_norm_out", torch::kCUDA,
         &metax_sparse::mhc_sinkhorn_rms_norm_out);
  m.def("mhc_raw_fp32_out(Tensor residual, Tensor weight, Tensor! gemm_out, Tensor! sqrsum_out) -> ()");
  m.impl("mhc_raw_fp32_out", torch::kCUDA, &metax_sparse::mhc_raw_fp32_out);
  m.def("mhc_cast_sqrsum_out(Tensor residual, Tensor! residual_fp32, Tensor! sqrsum_out) -> ()");
  m.impl("mhc_cast_sqrsum_out", torch::kCUDA,
         &metax_sparse::mhc_cast_sqrsum_out);
  m.def("mhc_downstream_rms_out(Tensor residual, Tensor gemm_out, Tensor sqrsum, Tensor scale, Tensor base, Tensor norm_weight, Tensor! post_out, Tensor! comb_out, Tensor! pre_norm_out, Tensor! norm_out, float rms_eps, float pre_eps, float sinkhorn_eps, float post_mult, int repeat) -> ()");
  m.impl("mhc_downstream_rms_out", torch::kCUDA,
         &metax_sparse::mhc_downstream_rms_out);
  m.def("mhc_sigmoid_probe_out(Tensor input, Tensor! out) -> ()");
  m.impl("mhc_sigmoid_probe_out", torch::kCUDA,
         &metax_sparse::mhc_sigmoid_probe_out);
  m.def("mhc_gemv_fp32_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("mhc_gemv_fp32_out", torch::kCUDA,
         &metax_sparse::mhc_gemv_fp32_out);
  m.def("mhc_gemv_fp32_grouped_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("mhc_gemv_fp32_grouped_out", torch::kCUDA,
         &metax_sparse::mhc_gemv_fp32_grouped_out);
  m.def("gemv_bf16_serial_rows_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("gemv_bf16_serial_rows_out", torch::kCUDA,
         &metax_sparse::gemv_bf16_serial_rows_out);
  m.def("gemv_bf16_exact_grouped_rows_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("gemv_bf16_exact_grouped_rows_out", torch::kCUDA,
         &metax_sparse::gemv_bf16_exact_grouped_rows_out);
  m.def("gemv_bf16_exact_oproj_grouped_rows_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("gemv_bf16_exact_oproj_grouped_rows_out", torch::kCUDA,
         &metax_sparse::gemv_bf16_exact_oproj_grouped_rows_out);
  m.def("gemv_bf16_exact_oproj_row_list_out(Tensor[] inputs, Tensor weight, Tensor! out) -> ()");
  m.impl("gemv_bf16_exact_oproj_row_list_out", torch::kCUDA,
         &metax_sparse::gemv_bf16_exact_oproj_row_list_out);
  m.def("gemv_bf16_fp32_serial_rows_out(Tensor input, Tensor weight, Tensor! out) -> ()");
  m.impl("gemv_bf16_fp32_serial_rows_out", torch::kCUDA,
         &metax_sparse::gemv_bf16_fp32_serial_rows_out);
  // In-place BF16 GEMM with FP32 accumulation/output: out = a @ b.T.
  m.def("gemm_bf16_fp32_out(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("gemm_bf16_fp32_out", torch::kCUDA,
         &metax_sparse::gemm_bf16_fp32_out);
  // In-place float32 GEMM: out = a @ b.T.
  m.def("gemm_fp32_out(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("gemm_fp32_out", torch::kCUDA, &metax_sparse::gemm_fp32_out);
  m.def("gemm_fp32_strided_batched_out(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("gemm_fp32_strided_batched_out", torch::kCUDA,
         &metax_sparse::gemm_fp32_strided_batched_out);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
