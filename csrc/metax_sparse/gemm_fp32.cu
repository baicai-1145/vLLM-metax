// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>

#include "../cub_helpers.h"

#include <cfloat>
#include <limits>

namespace metax_sparse {

template <int TPB>
__launch_bounds__(TPB) __global__ void row_softmax_fp32_kernel(
    float const* input, float* out, int const cols) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float row_max;
  __shared__ float inverse_sum;

  int64_t const row_offset = static_cast<int64_t>(blockIdx.x) * cols;
  float thread_value = -FLT_MAX;
  for (int col = threadIdx.x; col < cols; col += TPB) {
    thread_value = max(thread_value, input[row_offset + col]);
  }

  float const max_value =
      BlockReduce(temp_storage).Reduce(thread_value, CubMaxOp());
  if (threadIdx.x == 0) {
    row_max = max_value;
  }
  __syncthreads();

  thread_value = 0.0f;
  for (int col = threadIdx.x; col < cols; col += TPB) {
    thread_value += expf(input[row_offset + col] - row_max);
  }

  float const sum = BlockReduce(temp_storage).Reduce(thread_value, CubAddOp());
  if (threadIdx.x == 0) {
    inverse_sum = 1.0f / sum;
  }
  __syncthreads();

  for (int col = threadIdx.x; col < cols; col += TPB) {
    out[row_offset + col] =
        expf(input[row_offset + col] - row_max) * inverse_sum;
  }
}

// This is the fixed-width persistent-softmax specialization used by
// MetaX Torch for 128 columns: two rows per warp and two warps per block.
__launch_bounds__(128) __global__ void row_softmax_fp32_128_kernel(
    float const* input, float* out, int const rows) {
  constexpr int WARP_SIZE = C10_WARP_SIZE;
  constexpr int WARP_BATCH = 2;
  constexpr int WARP_ITERATIONS = 2;

  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const first_row = (blockIdx.x * blockDim.y + warp) * WARP_BATCH;
  int const local_rows = rows - first_row;
  if (local_rows <= 0) {
    return;
  }

  float elements[WARP_BATCH][WARP_ITERATIONS];
  #pragma unroll
  for (int batch = 0; batch < WARP_BATCH; ++batch) {
    bool const valid = batch < local_rows;
    int64_t const row_offset =
        static_cast<int64_t>(first_row + batch) * 128;
    #pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int const column = lane + it * WARP_SIZE;
      elements[batch][it] = valid
          ? input[row_offset + column]
          : -std::numeric_limits<float>::infinity();
    }
  }

  float max_value[WARP_BATCH];
  #pragma unroll
  for (int batch = 0; batch < WARP_BATCH; ++batch) {
    max_value[batch] = elements[batch][0];
    #pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[batch] = max_value[batch] > elements[batch][it]
          ? max_value[batch]
          : elements[batch][it];
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float const peer = WARP_SHFL_XOR(max_value[batch], offset, WARP_SIZE);
      max_value[batch] = max_value[batch] > peer ? max_value[batch] : peer;
    }
  }

  float sum[WARP_BATCH] = {0.0f, 0.0f};
  #pragma unroll
  for (int batch = 0; batch < WARP_BATCH; ++batch) {
    #pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[batch][it] = __expf(elements[batch][it] - max_value[batch]);
      sum[batch] += elements[batch][it];
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum[batch] += WARP_SHFL_XOR(sum[batch], offset, WARP_SIZE);
    }
  }

  float reciprocal_sum[WARP_BATCH];
  #pragma unroll
  for (int batch = 0; batch < WARP_BATCH; ++batch) {
    reciprocal_sum[batch] = __builtin_mxc_rcpf(sum[batch]);
  }

  #pragma unroll
  for (int batch = 0; batch < WARP_BATCH; ++batch) {
    if (batch >= local_rows) {
      continue;
    }
    #pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int const column = lane + it * WARP_SIZE;
      int64_t const row_offset =
          static_cast<int64_t>(first_row + batch) * 128;
      out[row_offset + column] = elements[batch][it] * reciprocal_sum[batch];
    }
  }
}

void softmax_fp32_out(torch::Tensor const& input, torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "softmax_fp32_out: input and out must be CUDA tensors");
  TORCH_CHECK(input.device() == out.device(),
              "softmax_fp32_out: input and out must be on the same device");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "softmax_fp32_out: input and out must be float32");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2,
              "softmax_fp32_out: input and out must be 2-D");
  TORCH_CHECK(input.sizes() == out.sizes(),
              "softmax_fp32_out: out shape must match input shape");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "softmax_fp32_out: input and out must be contiguous");

  int64_t const rows = input.size(0);
  int64_t const cols = input.size(1);
  TORCH_CHECK(cols > 0 && cols <= 1024,
              "softmax_fp32_out: columns must be in [1, 1024]");
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(),
              "softmax_fp32_out: rows must fit in a 32-bit integer");
  // The row kernel loads each row completely before storing it, so exact
  // input/output aliasing is graph-safe and avoids a second workspace.

  if (rows == 0) {
    return;
  }

  c10::cuda::CUDAGuard const device_guard(input.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  if (cols == 128) {
    dispatch_softmax_forward<float, float, float, false, false>(
        out.data_ptr<float>(), input.data_ptr<float>(), static_cast<int>(cols),
        static_cast<int>(cols), static_cast<int>(rows));
  } else if (cols < 128) {
    row_softmax_fp32_kernel<128><<<static_cast<int>(rows), 128, 0, stream>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(cols));
  } else {
    row_softmax_fp32_kernel<256><<<static_cast<int>(rows), 256, 0, stream>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(cols));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_bf16_fp32_out(torch::Tensor const& a, torch::Tensor const& b,
                        torch::Tensor& out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
              "gemm_bf16_fp32_out: a, b, and out must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(),
              "gemm_bf16_fp32_out: a, b, and out must be on the same device");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
                  b.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kFloat32,
              "gemm_bf16_fp32_out: a and b must be bfloat16 and out must be "
              "float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "gemm_bf16_fp32_out: a, b, and out must be 2-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "gemm_bf16_fp32_out: a, b, and out must be contiguous");

  int64_t const M = a.size(0);
  int64_t const K = a.size(1);
  int64_t const N = b.size(0);
  TORCH_CHECK(b.size(1) == K,
              "gemm_bf16_fp32_out: a and b inner dimensions must match");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N,
              "gemm_bf16_fp32_out: out shape must be [a.size(0), b.size(0)]");
  TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max() &&
                  K <= std::numeric_limits<int>::max(),
              "gemm_bf16_fp32_out: dimensions must fit in a 32-bit integer");
  TORCH_CHECK(!a.is_alias_of(b) && !a.is_alias_of(out) &&
                  !b.is_alias_of(out),
              "gemm_bf16_fp32_out: a, b, and out must not alias");

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
      CUDA_R_16BF, static_cast<int>(K), a.data_ptr(), CUDA_R_16BF,
      static_cast<int>(K), &beta, out.data_ptr(), CUDA_R_32F,
      static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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

}  // namespace metax_sparse
