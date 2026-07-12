// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// bf16 x bf16 -> fp32 router GEMM via cuBLAS.
// Uses CUBLAS_COMPUTE_32F so bf16 operands accumulate into fp32,
// matching TRT-LLM's cuBLAS fallback behaviour in dsv3RouterGemmOp.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

#include <limits>

// cuBLAS column-major math for row-major PyTorch tensors:
//   weight[N,K]_row  lda=K  -> cuBLAS sees (K,N) col-major; CUBLAS_OP_T ->
//   (N,K) input[M,K]_row   ldb=K  -> cuBLAS sees (K,M) col-major; CUBLAS_OP_N
//   -> (K,M) out[M,N]_row     ldc=N  -> cuBLAS sees (N,M) col-major (written as
//   output^T)
// cuBLAS: C(N,M) = weight(N,K) @ input(K,M)  =>  C^T = output[M,N]
// params: m=N, n=M, k=K, lda=K (weight), ldb=K (input), ldc=N (output)

torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: input must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: weight must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "router_gemm_bf16_fp32: input and weight must be 2-D");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "router_gemm_bf16_fp32: inner dimensions must match");

  int64_t const M = input.size(0);
  int64_t const N = weight.size(0);
  int64_t const K = input.size(1);

  auto out = torch::empty({M, N}, input.options().dtype(torch::kFloat32));

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;

  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, weight.data_ptr(),
      CUDA_R_16BF, static_cast<int>(K), input.data_ptr(), CUDA_R_16BF,
      static_cast<int>(K), &beta, out.data_ptr(), CUDA_R_32F,
      static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  return out;
}

void gemm_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                   torch::Tensor& out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
              "gemm_fp32_out: a, b, and out must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(),
              "gemm_fp32_out: a, b, and out must be on the same device");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                  b.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "gemm_fp32_out: a, b, and out must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "gemm_fp32_out: a, b, and out must be 2-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "gemm_fp32_out: a, b, and out must be contiguous");

  int64_t const M = a.size(0);
  int64_t const K = a.size(1);
  int64_t const N = b.size(0);
  TORCH_CHECK(b.size(1) == K,
              "gemm_fp32_out: a and b inner dimensions must match");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N,
              "gemm_fp32_out: out shape must be [a.size(0), b.size(0)]");
  TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max() &&
                  K <= std::numeric_limits<int>::max(),
              "gemm_fp32_out: dimensions must fit in a 32-bit integer");
  TORCH_CHECK(!a.is_alias_of(b) && !a.is_alias_of(out) &&
                  !b.is_alias_of(out),
              "gemm_fp32_out: a, b, and out must not alias");

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;
  // cuBLAS consumes row-major b and a as column-major transposed operands:
  // b[N,K] @ a[K,M] writes the transpose of out[M,N].
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, b.data_ptr(),
      CUDA_R_32F, static_cast<int>(K), a.data_ptr(), CUDA_R_32F,
      static_cast<int>(K), &beta, out.data_ptr(), CUDA_R_32F,
      static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
