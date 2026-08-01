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
#include <cuda_bf16.h>

#include "../cub_helpers.h"

#include <cfloat>
#include <limits>

namespace metax_sparse {

constexpr int kMhcMaxBatchedTokens = 6;

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
  if (cols == 4 || cols == 128) {
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

__global__ void mhc_sinkhorn_fp32_kernel(float const* input, float* out,
                                         float eps, int repeat) {
  int const matrix = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  float values[16];
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    values[index] = __fadd_rn(input[matrix * 16 + index], eps);
  }
  if (repeat == 21 || repeat == 22) {
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      float const pair0 = __fadd_rn(values[col], values[8 + col]);
      float const pair1 = __fadd_rn(values[4 + col], values[12 + col]);
      float const sum = __fadd_rn(pair0, pair1);
      if (repeat == 21) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
          out[matrix * 16 + row * 4 + col] = sum;
        }
      } else {
        float const denominator = __fadd_rn(sum, eps);
        float const numerator = values[col];
        out[matrix * 16 + col] = __fdiv_rn(numerator, denominator);
        out[matrix * 16 + 4 + col] = __fdividef(numerator, denominator);
        out[matrix * 16 + 8 + col] =
            __builtin_mxc_rcpf(denominator) * numerator;
        out[matrix * 16 + 12 + col] = numerator / denominator;
      }
    }
    return;
  }
#pragma unroll
  for (int iteration = 0; iteration < repeat; ++iteration) {
    if (iteration > 0) {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        float const pair0 =
            __fadd_rn(values[row * 4], values[row * 4 + 1]);
        float const pair1 =
            __fadd_rn(values[row * 4 + 2], values[row * 4 + 3]);
        float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
        for (int col = 0; col < 4; ++col) {
          values[row * 4 + col] =
              __fdiv_rn(values[row * 4 + col], __fadd_rn(sum, eps));
        }
      }
      if (repeat == 23) {
#pragma unroll
        for (int index = 0; index < 16; ++index) {
          out[matrix * 16 + index] = values[index];
        }
        return;
      }
    }
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      float const pair0 = __fadd_rn(values[col], values[8 + col]);
      float const pair1 = __fadd_rn(values[4 + col], values[12 + col]);
      float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        values[row * 4 + col] =
            __fdiv_rn(values[row * 4 + col], __fadd_rn(sum, eps));
      }
    }
  }
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    out[matrix * 16 + index] = values[index];
  }
}

void mhc_sinkhorn_fp32_out(torch::Tensor const& input, torch::Tensor& out,
                           double eps, int64_t repeat) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "mhc_sinkhorn_fp32_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == out.device(),
              "mhc_sinkhorn_fp32_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "mhc_sinkhorn_fp32_out: tensors must be float32");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "mhc_sinkhorn_fp32_out: tensors must be contiguous");
  TORCH_CHECK(input.sizes() == out.sizes() && input.dim() == 3 &&
                  input.size(1) == 4 && input.size(2) == 4,
              "mhc_sinkhorn_fp32_out: expected matching [N,4,4] tensors");
  TORCH_CHECK(repeat >= 0 && repeat <= 23,
              "mhc_sinkhorn_fp32_out: repeat must be in [0,23]");
  if (input.size(0) == 0) {
    return;
  }
  c10::cuda::CUDAGuard const device_guard(input.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  mhc_sinkhorn_fp32_kernel<<<static_cast<int>(input.size(0)), 32, 0, stream>>>(
      input.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(eps),
      static_cast<int>(repeat));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mhc_sinkhorn_rms_norm_kernel(
    float const* input, c10::BFloat16 const* pre_norm,
    c10::BFloat16 const* weight, float* comb_out,
    c10::BFloat16* norm_out, float sinkhorn_eps, float norm_eps, int repeat) {
  __shared__ float reduction[512];
  __shared__ float inverse_rms;
  int const tid = threadIdx.x;

  if (tid == 0) {
    float values[16];
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = __fadd_rn(input[index], sinkhorn_eps);
    }
#pragma unroll
    for (int iteration = 0; iteration < 20; ++iteration) {
      if (iteration > 0) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
          float const pair0 =
              __fadd_rn(values[row * 4], values[row * 4 + 1]);
          float const pair1 =
              __fadd_rn(values[row * 4 + 2], values[row * 4 + 3]);
          float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
          for (int col = 0; col < 4; ++col) {
            values[row * 4 + col] = __fdiv_rn(
                values[row * 4 + col], __fadd_rn(sum, sinkhorn_eps));
          }
        }
      }
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        float const pair0 = __fadd_rn(values[col], values[8 + col]);
        float const pair1 = __fadd_rn(values[4 + col], values[12 + col]);
        float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
        for (int row = 0; row < 4; ++row) {
          values[row * 4 + col] = __fdiv_rn(
              values[row * 4 + col], __fadd_rn(sum, sinkhorn_eps));
        }
      }
    }
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      comb_out[index] = values[index];
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int vector = 0; vector < 2; ++vector) {
    int const base = (tid + vector * 512) * 4;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value = static_cast<float>(pre_norm[base + item]);
      sum += value * value;
    }
  }
  reduction[tid] = sum;
  for (int offset = 256; offset >= 64; offset >>= 1) {
    __syncthreads();
    if (tid < offset) {
      reduction[tid] += reduction[tid + offset];
    }
  }
  __syncthreads();
  if (tid < 64) {
    sum = reduction[tid];
    for (int offset = 1; offset < 64; offset <<= 1) {
      sum += WARP_SHFL_DOWN(sum, offset);
    }
    if (tid == 0) {
      inverse_rms = rsqrtf(sum * (1.0f / 4096.0f) + norm_eps);
    }
  }
  __syncthreads();
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    int const index = tid + item * 512;
    float const value = static_cast<float>(pre_norm[index]) * inverse_rms;
    c10::BFloat16 const normalized(value);
    norm_out[index] = c10::BFloat16(
        static_cast<float>(normalized) * static_cast<float>(weight[index]));
  }
}

void mhc_sinkhorn_rms_norm_out(
    torch::Tensor const& input, torch::Tensor const& pre_norm,
    torch::Tensor const& weight, torch::Tensor& comb_out,
    torch::Tensor& norm_out, double sinkhorn_eps, double norm_eps,
    int64_t repeat) {
  TORCH_CHECK(input.is_cuda() && pre_norm.is_cuda() && weight.is_cuda() &&
                  comb_out.is_cuda() && norm_out.is_cuda(),
              "mhc_sinkhorn_rms_norm_out: tensors must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                  comb_out.scalar_type() == torch::kFloat32 &&
                  pre_norm.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kBFloat16 &&
                  norm_out.scalar_type() == torch::kBFloat16,
              "mhc_sinkhorn_rms_norm_out: dtype mismatch");
  TORCH_CHECK(input.is_contiguous() && pre_norm.is_contiguous() &&
                  weight.is_contiguous() && comb_out.is_contiguous() &&
                  norm_out.is_contiguous(),
              "mhc_sinkhorn_rms_norm_out: tensors must be contiguous");
  TORCH_CHECK(input.numel() == 16 && comb_out.numel() == 16 &&
                  pre_norm.numel() == 4096 && weight.numel() == 4096 &&
                  norm_out.numel() == 4096,
              "mhc_sinkhorn_rms_norm_out: exact decode shape required");
  TORCH_CHECK(repeat == 20,
              "mhc_sinkhorn_rms_norm_out: repeat must be 20");
  c10::cuda::CUDAGuard const device_guard(input.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  mhc_sinkhorn_rms_norm_kernel<<<1, 512, 0, stream>>>(
      input.data_ptr<float>(), pre_norm.data_ptr<c10::BFloat16>(),
      weight.data_ptr<c10::BFloat16>(), comb_out.data_ptr<float>(),
      norm_out.data_ptr<c10::BFloat16>(), static_cast<float>(sinkhorn_eps),
      static_cast<float>(norm_eps), static_cast<int>(repeat));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mhc_raw_fp32_kernel(c10::BFloat16 const* residual,
                                    float const* weight, float* gemm_out,
                                    float* sqrsum_out) {
  if (threadIdx.x != 0) {
    return;
  }
  int const output = blockIdx.x;
  float sum = 0.0f;
  if (output < 24) {
    for (int index = 0; index < 16384; ++index) {
      float const product = __fmul_rn(
          static_cast<float>(residual[index]), weight[output * 16384 + index]);
      sum = __fadd_rn(sum, product);
    }
    gemm_out[output] = sum;
  } else {
    for (int index = 0; index < 16384; ++index) {
      float const value = static_cast<float>(residual[index]);
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sqrsum_out[0] = sum;
  }
}

void mhc_raw_fp32_out(torch::Tensor const& residual, torch::Tensor const& weight,
                      torch::Tensor& gemm_out, torch::Tensor& sqrsum_out) {
  TORCH_CHECK(residual.is_cuda() && weight.is_cuda() && gemm_out.is_cuda() &&
                  sqrsum_out.is_cuda(),
              "mhc_raw_fp32_out: tensors must be CUDA");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kFloat32 &&
                  gemm_out.scalar_type() == torch::kFloat32 &&
                  sqrsum_out.scalar_type() == torch::kFloat32,
              "mhc_raw_fp32_out: dtype mismatch");
  TORCH_CHECK(residual.is_contiguous() && weight.is_contiguous() &&
                  gemm_out.is_contiguous() && sqrsum_out.is_contiguous(),
              "mhc_raw_fp32_out: tensors must be contiguous");
  TORCH_CHECK(residual.numel() == 16384 && weight.numel() == 24 * 16384 &&
                  gemm_out.numel() == 24 && sqrsum_out.numel() == 1,
              "mhc_raw_fp32_out: exact decode shape required");
  c10::cuda::CUDAGuard const device_guard(residual.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  mhc_raw_fp32_kernel<<<25, 64, 0, stream>>>(
      residual.data_ptr<c10::BFloat16>(), weight.data_ptr<float>(),
      gemm_out.data_ptr<float>(), sqrsum_out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mhc_cast_sqrsum_kernel(c10::BFloat16 const* residual,
                                       float* residual_fp32,
                                       float* sqrsum_out) {
  __shared__ float reduction[512];
  int const tid = threadIdx.x;
  int64_t const row = static_cast<int64_t>(blockIdx.x);
  residual += row * 16384;
  residual_fp32 += row * 16384;
  sqrsum_out += row;
  float sum = 0.0f;
#pragma unroll
  for (int vector = 0; vector < 8; ++vector) {
    int const base = (tid + vector * 512) * 4;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value = static_cast<float>(residual[base + item]);
      residual_fp32[base + item] = value;
      sum += value * value;
    }
  }
  reduction[tid] = sum;
  for (int offset = 256; offset >= 64; offset >>= 1) {
    __syncthreads();
    if (tid < offset) {
      reduction[tid] += reduction[tid + offset];
    }
  }
  __syncthreads();
  if (tid < 64) {
    sum = reduction[tid];
    for (int offset = 1; offset < 64; offset <<= 1) {
      sum += WARP_SHFL_DOWN(sum, offset);
    }
    if (tid == 0) {
      sqrsum_out[0] = sum;
    }
  }
}

void mhc_cast_sqrsum_out(torch::Tensor const& residual,
                         torch::Tensor& residual_fp32,
                         torch::Tensor& sqrsum_out) {
  TORCH_CHECK(residual.is_cuda() && residual_fp32.is_cuda() &&
                  sqrsum_out.is_cuda(),
              "mhc_cast_sqrsum_out: tensors must be CUDA");
  TORCH_CHECK(residual.device() == residual_fp32.device() &&
                  residual.device() == sqrsum_out.device(),
              "mhc_cast_sqrsum_out: device mismatch");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16 &&
                  residual_fp32.scalar_type() == torch::kFloat32 &&
                  sqrsum_out.scalar_type() == torch::kFloat32,
              "mhc_cast_sqrsum_out: dtype mismatch");
  TORCH_CHECK(residual.is_contiguous() && residual_fp32.is_contiguous() &&
                  sqrsum_out.is_contiguous(),
              "mhc_cast_sqrsum_out: tensors must be contiguous");
  bool const batched_shape = residual.dim() == 3 && residual.size(1) == 4 &&
                              residual.size(2) == 4096 &&
                              residual_fp32.dim() == 2 &&
                              residual_fp32.size(0) == residual.size(0) &&
                              residual_fp32.size(1) == 16384 &&
                              sqrsum_out.dim() == 2 &&
                              sqrsum_out.size(0) == residual.size(0) &&
                              sqrsum_out.size(1) == 1;
  bool const legacy_shape = residual.dim() == 3 && residual.size(0) == 1 &&
                            residual.size(1) == 4 && residual.size(2) == 4096 &&
                            residual_fp32.dim() == 1 &&
                            residual_fp32.size(0) == 16384 &&
                            sqrsum_out.dim() == 1 && sqrsum_out.size(0) == 1;
  TORCH_CHECK((batched_shape || legacy_shape) && residual.numel() > 0 &&
                  residual.numel() / 16384 <= kMhcMaxBatchedTokens,
              "mhc_cast_sqrsum_out: expected contiguous BF16 [N,4,4096], "
              "FP32 [N,16384], and FP32 [N,1] with 1 <= N <= 5");
  c10::cuda::CUDAGuard const device_guard(residual.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int64_t const rows = residual.numel() / 16384;
  mhc_cast_sqrsum_kernel<<<static_cast<int>(rows), 512, 0, stream>>>(
      residual.data_ptr<c10::BFloat16>(), residual_fp32.data_ptr<float>(),
      sqrsum_out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float mhc_sigmoid(float value) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

__global__ void mhc_downstream_rms_kernel(
    c10::BFloat16 const* residual, float const* gemm_out,
    float const* sqrsum, float const* scale, float const* base,
    c10::BFloat16 const* norm_weight, float* post_out, float* comb_out,
    c10::BFloat16* pre_norm_out, c10::BFloat16* norm_out, float rms_eps,
    float pre_eps, float sinkhorn_eps, float post_mult, int repeat) {
  __shared__ float pre_mix[4];
  __shared__ float values[16];
  __shared__ float reduction[512];
  __shared__ float inverse_rms;
  int const tid = threadIdx.x;
  int64_t const row = static_cast<int64_t>(blockIdx.x);
  residual += row * 16384;
  gemm_out += row * 24;
  sqrsum += row;
  post_out += row * 4;
  comb_out += row * 16;
  pre_norm_out += row * 4096;
  norm_out += row * 4096;

  if (tid == 0) {
    float const raw_inverse_rms = rsqrtf(__fadd_rn(
        __fdiv_rn(sqrsum[0], 16384.0f), rms_eps));
    float mixes[24];
#pragma unroll
    for (int index = 0; index < 24; ++index) {
      mixes[index] = __fmul_rn(gemm_out[index], raw_inverse_rms);
    }
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      float const logit = __fadd_rn(
          __fmul_rn(mixes[index], scale[0]), base[index]);
      pre_mix[index] = __fadd_rn(mhc_sigmoid(logit), pre_eps);
      float const post_logit = __fadd_rn(
          __fmul_rn(mixes[4 + index], scale[1]), base[4 + index]);
      post_out[index] = __fmul_rn(mhc_sigmoid(post_logit), post_mult);
    }

#pragma unroll
    for (int row = 0; row < 4; ++row) {
      float logits[4];
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        int const index = row * 4 + col;
        logits[col] = __fadd_rn(
            __fmul_rn(mixes[8 + index], scale[2]), base[8 + index]);
      }
      float maximum = fmaxf(fmaxf(logits[0], logits[1]),
                            fmaxf(logits[2], logits[3]));
      float exponentials[4];
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        exponentials[col] = __expf(logits[col] - maximum);
      }
      float const pair0 = __fadd_rn(exponentials[0], exponentials[2]);
      float const pair1 = __fadd_rn(exponentials[1], exponentials[3]);
      float const reciprocal = __builtin_mxc_rcpf(__fadd_rn(pair0, pair1));
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        values[row * 4 + col] = __fadd_rn(
            __fmul_rn(exponentials[col], reciprocal), sinkhorn_eps);
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int iteration = 0; iteration < 20; ++iteration) {
    if (iteration > 0) {
      if (tid < 4) {
        int const row = tid;
        float const pair0 =
            __fadd_rn(values[row * 4], values[row * 4 + 1]);
        float const pair1 =
            __fadd_rn(values[row * 4 + 2], values[row * 4 + 3]);
        float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
        for (int col = 0; col < 4; ++col) {
          values[row * 4 + col] = __fdiv_rn(
              values[row * 4 + col], __fadd_rn(sum, sinkhorn_eps));
        }
      }
      __syncthreads();
    }
    if (tid < 4) {
      int const col = tid;
      float const pair0 = __fadd_rn(values[col], values[8 + col]);
      float const pair1 = __fadd_rn(values[4 + col], values[12 + col]);
      float const sum = __fadd_rn(pair0, pair1);
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        values[row * 4 + col] = __fdiv_rn(
            values[row * 4 + col], __fadd_rn(sum, sinkhorn_eps));
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      comb_out[index] = values[index];
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < 8; ++item) {
    int const index = tid + item * 512;
    float const term0 = __fmul_rn(
        static_cast<float>(residual[index]), pre_mix[0]);
    float const term1 = __fmul_rn(
        static_cast<float>(residual[4096 + index]), pre_mix[1]);
    float const term2 = __fmul_rn(
        static_cast<float>(residual[8192 + index]), pre_mix[2]);
    float const term3 = __fmul_rn(
        static_cast<float>(residual[12288 + index]), pre_mix[3]);
    float const value = __fadd_rn(
        __fadd_rn(term0, term2), __fadd_rn(term1, term3));
    pre_norm_out[index] = c10::BFloat16(value);
  }
  __syncthreads();

  float sum = 0.0f;
#pragma unroll
  for (int vector = 0; vector < 2; ++vector) {
    int const base_index = (tid + vector * 512) * 4;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value = static_cast<float>(pre_norm_out[base_index + item]);
      sum += value * value;
    }
  }
  reduction[tid] = sum;
  for (int offset = 256; offset >= 64; offset >>= 1) {
    __syncthreads();
    if (tid < offset) {
      reduction[tid] += reduction[tid + offset];
    }
  }
  __syncthreads();
  if (tid < 64) {
    sum = reduction[tid];
    for (int offset = 1; offset < 64; offset <<= 1) {
      sum += WARP_SHFL_DOWN(sum, offset);
    }
    if (tid == 0) {
      inverse_rms = rsqrtf(sum * (1.0f / 4096.0f) + rms_eps);
    }
  }
  __syncthreads();
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    int const index = tid + item * 512;
    float const value = static_cast<float>(pre_norm_out[index]) * inverse_rms;
    c10::BFloat16 const normalized(value);
    norm_out[index] = c10::BFloat16(
        static_cast<float>(normalized) * static_cast<float>(norm_weight[index]));
  }
}

void mhc_downstream_rms_out(
    torch::Tensor const& residual, torch::Tensor const& gemm_out,
    torch::Tensor const& sqrsum, torch::Tensor const& scale,
    torch::Tensor const& base, torch::Tensor const& norm_weight,
    torch::Tensor& post_out, torch::Tensor& comb_out,
    torch::Tensor& pre_norm_out, torch::Tensor& norm_out, double rms_eps,
    double pre_eps, double sinkhorn_eps, double post_mult, int64_t repeat) {
  TORCH_CHECK(residual.is_cuda() && gemm_out.is_cuda() && sqrsum.is_cuda() &&
                  scale.is_cuda() && base.is_cuda() && norm_weight.is_cuda() &&
                  post_out.is_cuda() && comb_out.is_cuda() &&
                  pre_norm_out.is_cuda() && norm_out.is_cuda(),
              "mhc_downstream_rms_out: tensors must be CUDA");
  TORCH_CHECK(
      residual.device() == gemm_out.device() &&
          residual.device() == sqrsum.device() &&
          residual.device() == scale.device() && residual.device() == base.device() &&
          residual.device() == norm_weight.device() &&
          residual.device() == post_out.device() &&
          residual.device() == comb_out.device() &&
          residual.device() == pre_norm_out.device() &&
          residual.device() == norm_out.device(),
      "mhc_downstream_rms_out: device mismatch");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16 &&
                  gemm_out.scalar_type() == torch::kFloat32 &&
                  sqrsum.scalar_type() == torch::kFloat32 &&
                  scale.scalar_type() == torch::kFloat32 &&
                  base.scalar_type() == torch::kFloat32 &&
                  norm_weight.scalar_type() == torch::kBFloat16 &&
                  post_out.scalar_type() == torch::kFloat32 &&
                  comb_out.scalar_type() == torch::kFloat32 &&
                  pre_norm_out.scalar_type() == torch::kBFloat16 &&
                  norm_out.scalar_type() == torch::kBFloat16,
              "mhc_downstream_rms_out: dtype mismatch");
  bool const batched_shape =
      residual.dim() == 3 && residual.size(1) == 4 &&
          residual.size(2) == 4096 && gemm_out.dim() == 3 &&
          gemm_out.size(0) == residual.size(0) && gemm_out.size(1) == 1 &&
          gemm_out.size(2) == 24 && sqrsum.dim() == 2 &&
          sqrsum.size(0) == residual.size(0) && sqrsum.size(1) == 1 &&
          scale.dim() == 1 && scale.size(0) == 3 && base.dim() == 1 &&
          base.size(0) == 24 && norm_weight.dim() == 1 &&
          norm_weight.size(0) == 4096 && post_out.dim() == 2 &&
          post_out.size(0) == residual.size(0) && post_out.size(1) == 4 &&
          comb_out.dim() == 3 && comb_out.size(0) == residual.size(0) &&
          comb_out.size(1) == 4 && comb_out.size(2) == 4 &&
          pre_norm_out.dim() == 2 &&
          pre_norm_out.size(0) == residual.size(0) &&
          pre_norm_out.size(1) == 4096 && norm_out.dim() == 2 &&
          norm_out.size(0) == residual.size(0) && norm_out.size(1) == 4096 &&
          residual.size(0) >= 1 && residual.size(0) <= kMhcMaxBatchedTokens;
  bool const legacy_shape =
      residual.dim() == 3 && residual.size(0) == 1 && residual.size(1) == 4 &&
      residual.size(2) == 4096 && gemm_out.dim() == 1 && gemm_out.size(0) == 24 &&
      sqrsum.dim() == 1 && sqrsum.size(0) == 1 && scale.dim() == 1 &&
      scale.size(0) == 3 && base.dim() == 1 && base.size(0) == 24 &&
      norm_weight.dim() == 1 && norm_weight.size(0) == 4096 &&
      post_out.dim() == 1 && post_out.size(0) == 4 && comb_out.dim() == 1 &&
      comb_out.size(0) == 16 && pre_norm_out.dim() == 1 &&
      pre_norm_out.size(0) == 4096 && norm_out.dim() == 1 &&
      norm_out.size(0) == 4096;
  TORCH_CHECK(
      (batched_shape || legacy_shape) &&
          residual.numel() / 16384 <= kMhcMaxBatchedTokens,
      "mhc_downstream_rms_out: expected contiguous decode tensors with "
      "1 <= N <= 6");
  TORCH_CHECK(repeat == 20,
              "mhc_downstream_rms_out: repeat must be 20");
  c10::cuda::CUDAGuard const device_guard(residual.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int64_t const rows = residual.numel() / 16384;
  mhc_downstream_rms_kernel<<<static_cast<int>(rows), 512, 0, stream>>>(
      residual.data_ptr<c10::BFloat16>(), gemm_out.data_ptr<float>(),
      sqrsum.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(),
      norm_weight.data_ptr<c10::BFloat16>(), post_out.data_ptr<float>(),
      comb_out.data_ptr<float>(), pre_norm_out.data_ptr<c10::BFloat16>(),
      norm_out.data_ptr<c10::BFloat16>(), static_cast<float>(rms_eps),
      static_cast<float>(pre_eps), static_cast<float>(sinkhorn_eps),
      static_cast<float>(post_mult), static_cast<int>(repeat));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mhc_sigmoid_probe_kernel(float const* input, float* out) {
  int const index = threadIdx.x;
  if (index >= 4) {
    return;
  }
  float const fast_denominator = __fadd_rn(1.0f, __expf(-input[index]));
  float const accurate_denominator = __fadd_rn(1.0f, expf(-input[index]));
  out[index] = __builtin_mxc_rcpf(fast_denominator);
  out[4 + index] = __fdiv_rn(1.0f, fast_denominator);
  out[8 + index] = __builtin_mxc_rcpf(accurate_denominator);
  out[12 + index] = __fdiv_rn(1.0f, accurate_denominator);
}

void mhc_sigmoid_probe_out(torch::Tensor const& input, torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() &&
                  input.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32 && input.numel() == 4 &&
                  out.numel() == 16 && input.is_contiguous() &&
                  out.is_contiguous(),
              "mhc_sigmoid_probe_out: expected CUDA FP32 [4] and [4,4]");
  c10::cuda::CUDAGuard const device_guard(input.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  mhc_sigmoid_probe_kernel<<<1, 64, 0, stream>>>(input.data_ptr<float>(),
                                                 out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mhc_gemv_fp32_out(torch::Tensor const& input, torch::Tensor const& weight,
                       torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "mhc_gemv_fp32_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == weight.device() && input.device() == out.device(),
              "mhc_gemv_fp32_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                  weight.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "mhc_gemv_fp32_out: tensors must be float32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "mhc_gemv_fp32_out: tensors must be contiguous");
  TORCH_CHECK(input.numel() == 16384 && weight.numel() == 24 * 16384 &&
                  out.numel() == 24,
              "mhc_gemv_fp32_out: exact decode shape required");
  auto out_view = out.view({1, 24});
  auto input_view = input.view({1, 16384});
  auto weight_t = weight.transpose(0, 1);
  at::mm_out(out_view, input_view, weight_t);
}

void mhc_gemv_fp32_grouped_out(torch::Tensor const& input,
                               torch::Tensor const& weight,
                               torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "mhc_gemv_fp32_grouped_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == weight.device() && input.device() == out.device(),
              "mhc_gemv_fp32_grouped_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                  weight.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "mhc_gemv_fp32_grouped_out: tensors must be float32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "mhc_gemv_fp32_grouped_out: tensors must be contiguous");
  int64_t const rows = input.dim() == 2 ? input.size(0) : -1;
  TORCH_CHECK(rows >= 2 && rows <= 6 && input.size(1) == 16384 &&
                  weight.dim() == 2 && weight.size(0) == 24 &&
                  weight.size(1) == 16384 && out.numel() == rows * 24,
              "mhc_gemv_fp32_grouped_out: expected input [N,16384], "
              "weight [24,16384], and contiguous out with 2 <= N <= 6");
  TORCH_CHECK(!input.is_alias_of(weight) && !input.is_alias_of(out) &&
                  !weight.is_alias_of(out),
              "mhc_gemv_fp32_grouped_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(input.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));
  float const alpha = 1.0f;
  float const beta = 0.0f;
  constexpr int k = 16384;
  constexpr int output_width = 24;
  TORCH_CUDABLAS_CHECK(cublasSgemvStridedBatched(
      handle, CUBLAS_OP_T, k, output_width, &alpha,
      weight.data_ptr<float>(), k, 0, input.data_ptr<float>(), 1, k, &beta,
      out.data_ptr<float>(), 1, output_width, static_cast<int>(rows)));
}

void gemv_bf16_serial_rows_out(torch::Tensor const& input,
                               torch::Tensor const& weight,
                               torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "gemv_bf16_serial_rows_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == weight.device() && input.device() == out.device(),
              "gemv_bf16_serial_rows_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "gemv_bf16_serial_rows_out: tensors must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && out.dim() == 2,
              "gemv_bf16_serial_rows_out: tensors must be 2-D");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "gemv_bf16_serial_rows_out: tensors must be contiguous");

  int64_t const rows = input.size(0);
  int64_t const k = input.size(1);
  int64_t const n = weight.size(0);
  TORCH_CHECK(rows >= 1 && rows <= 6 && weight.size(1) == k &&
                  out.size(0) == rows && out.size(1) == n,
              "gemv_bf16_serial_rows_out: expected input [B,K], weight "
              "[N,K], out [B,N], and 1 <= B <= 6");
  TORCH_CHECK(!input.is_alias_of(weight) && !input.is_alias_of(out) &&
                  !weight.is_alias_of(out),
              "gemv_bf16_serial_rows_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(input.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n), 1,
        static_cast<int>(k), &alpha, weight.data_ptr(),
        CUDA_R_16BF, static_cast<int>(k),
        input.data_ptr<c10::BFloat16>() + row * k, CUDA_R_16BF,
        static_cast<int>(k), &beta, out.data_ptr<c10::BFloat16>() + row * n,
        CUDA_R_16BF, static_cast<int>(n), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
  }
}

typedef __NATIVE_VECTOR__(2, float) GemvFloat2;

struct alignas(16) GemvBf16x8 {
  __nv_bfloat16 values[8];
};

__device__ __forceinline__ float gemv_read_lane(float value, int lane) {
  union {
    float f;
    unsigned int u;
  } bits{value};
  bits.u = __builtin_mxc_readlane(bits.u, lane);
  return bits.f;
}

template <int Delta>
__device__ __forceinline__ float gemv_mov_shfl_down(float value) {
  union {
    float f;
    int i;
  } bits{value};
  bits.i = __builtin_mxc_mov_shfl(bits.i, 0x100 + Delta, 0xf, 0xf, false);
  return bits.f;
}

__launch_bounds__(512) __global__ void gemv_bf16_exact_grouped_rows_kernel(
    __nv_bfloat16 const* input, __nv_bfloat16 const* weight,
    __nv_bfloat16* out, int rows, int n, int k) {
  int const row = blockIdx.y;
  int const lane = threadIdx.x & 63;
  int const wave = threadIdx.x >> 6;
  int const output = blockIdx.x * 16 + wave * 2;
  if (row >= rows || output >= n) {
    return;
  }

  GemvFloat2 accum0 = {0.0f, 0.0f};
  GemvFloat2 accum1 = {0.0f, 0.0f};
  int const vector_count = k / 8;
  for (int vector_index = vector_count - 64 + lane; vector_index >= lane;
       vector_index -= 64) {
    auto const input_values = *reinterpret_cast<GemvBf16x8 const*>(
        input + static_cast<int64_t>(row) * k + vector_index * 8);
    auto const weight0_values = *reinterpret_cast<GemvBf16x8 const*>(
        weight + static_cast<int64_t>(output) * k + vector_index * 8);
    auto const weight1_values = *reinterpret_cast<GemvBf16x8 const*>(
        weight + static_cast<int64_t>(output + 1) * k + vector_index * 8);
#pragma unroll
    for (int element = 0; element < 8; element += 2) {
      GemvFloat2 const input_pair = {
          __bfloat162float(input_values.values[element]),
          __bfloat162float(input_values.values[element + 1])};
      GemvFloat2 const weight0_pair = {
          __bfloat162float(weight0_values.values[element]),
          __bfloat162float(weight0_values.values[element + 1])};
      GemvFloat2 const weight1_pair = {
          __bfloat162float(weight1_values.values[element]),
          __bfloat162float(weight1_values.values[element + 1])};
      accum0 = __builtin_mxc_pk_fma_f32(weight0_pair, input_pair, accum0);
      accum1 = __builtin_mxc_pk_fma_f32(weight1_pair, input_pair, accum1);
    }
  }

  float sum0 = accum0[0] + accum0[1];
  float sum1 = accum1[0] + accum1[1];
  sum0 += gemv_mov_shfl_down<8>(sum0);
  sum0 += gemv_mov_shfl_down<4>(sum0);
  sum0 += gemv_mov_shfl_down<2>(sum0);
  sum0 += gemv_mov_shfl_down<1>(sum0);
  sum1 += gemv_mov_shfl_down<8>(sum1);
  sum1 += gemv_mov_shfl_down<4>(sum1);
  sum1 += gemv_mov_shfl_down<2>(sum1);
  sum1 += gemv_mov_shfl_down<1>(sum1);

  float const other0 =
      (gemv_read_lane(sum0, 48) + gemv_read_lane(sum0, 32)) +
      gemv_read_lane(sum0, 16);
  float const other1 =
      (gemv_read_lane(sum1, 48) + gemv_read_lane(sum1, 32)) +
      gemv_read_lane(sum1, 16);
  if (lane == 0) {
    int64_t const out_offset = static_cast<int64_t>(row) * n + output;
    out[out_offset] = __float2bfloat16(sum0 + other0);
    out[out_offset + 1] = __float2bfloat16(sum1 + other1);
  }
}

void gemv_bf16_exact_grouped_rows_out(torch::Tensor const& input,
                                       torch::Tensor const& weight,
                                       torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "gemv_bf16_exact_grouped_rows_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == weight.device() && input.device() == out.device(),
              "gemv_bf16_exact_grouped_rows_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "gemv_bf16_exact_grouped_rows_out: tensors must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && out.dim() == 2 &&
                  input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "gemv_bf16_exact_grouped_rows_out: contiguous 2-D tensors required");

  int64_t const rows = input.size(0);
  int64_t const k = input.size(1);
  int64_t const n = weight.size(0);
  bool const wq_b_shape = k == 1024 && n == 8192;
  bool const o_proj_shape = k == 2048 && n == 4096;
  TORCH_CHECK(rows >= 1 && rows <= 6 && weight.size(1) == k &&
                  out.size(0) == rows && out.size(1) == n &&
                  (wq_b_shape || o_proj_shape),
              "gemv_bf16_exact_grouped_rows_out: expected input [B,K], weight "
              "[N,K], out [B,N], 1 <= B <= 6, and one of K=1024,N=8192, "
              "or K=2048,N=4096");
  TORCH_CHECK(!input.is_alias_of(weight) && !input.is_alias_of(out) &&
                  !weight.is_alias_of(out),
              "gemv_bf16_exact_grouped_rows_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(input.device());
  if (o_proj_shape) {
    auto const weight_t = weight.transpose(0, 1);
    for (int64_t row = 0; row < rows; ++row) {
      auto input_row = input.narrow(0, row, 1);
      auto out_row = out.narrow(0, row, 1);
      at::mm_out(out_row, input_row, weight_t);
    }
    return;
  }

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  dim3 const grid(static_cast<unsigned>(n / 16),
                  static_cast<unsigned>(rows));
  gemv_bf16_exact_grouped_rows_kernel<<<grid, 512, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
      static_cast<int>(rows), static_cast<int>(n), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemv_bf16_exact_oproj_grouped_rows_out(torch::Tensor const& input,
                                            torch::Tensor const& weight,
                                            torch::Tensor& out) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && out.dim() == 2 &&
                  input.size(1) == 2048 && weight.size(0) == 4096 &&
                  weight.size(1) == 2048 && out.size(0) == input.size(0) &&
                  out.size(1) == 4096,
              "gemv_bf16_exact_oproj_grouped_rows_out: expected input "
              "[B,2048], weight [4096,2048], and out [B,4096]");
  gemv_bf16_exact_grouped_rows_out(input, weight, out);
}

void gemv_bf16_exact_oproj_row_list_out(c10::List<torch::Tensor> const& inputs,
                                        torch::Tensor const& weight,
                                        torch::Tensor& out) {
  int64_t const rows = inputs.size();
  TORCH_CHECK(weight.is_cuda() && out.is_cuda(),
              "gemv_bf16_exact_oproj_row_list_out: tensors must be CUDA");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "gemv_bf16_exact_oproj_row_list_out: tensors must be bfloat16");
  TORCH_CHECK(rows >= 1 && rows <= 6 && weight.dim() == 2 &&
                  weight.size(0) == 4096 && weight.size(1) == 2048 &&
                  weight.is_contiguous() && out.dim() == 2 &&
                  out.size(0) == rows && out.size(1) == 4096 &&
                  out.is_contiguous(),
              "gemv_bf16_exact_oproj_row_list_out: expected 1 <= rows <= 6, "
              "weight [4096,2048], and out [rows,4096]");
  TORCH_CHECK(!weight.is_alias_of(out),
              "gemv_bf16_exact_oproj_row_list_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(weight.device());
  auto const weight_t = weight.transpose(0, 1);
  for (int64_t row = 0; row < rows; ++row) {
    torch::Tensor const input = inputs.get(row);
    TORCH_CHECK(input.is_cuda() && input.device() == weight.device() &&
                    out.device() == weight.device(),
                "gemv_bf16_exact_oproj_row_list_out: device mismatch");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
                    input.size(0) == 1 && input.size(1) == 2048 &&
                    input.is_contiguous(),
                "gemv_bf16_exact_oproj_row_list_out: expected each input "
                "row to be contiguous BF16 [1,2048]");
    TORCH_CHECK(!input.is_alias_of(weight) && !input.is_alias_of(out),
                "gemv_bf16_exact_oproj_row_list_out: tensors must not alias");
    auto out_row = out.narrow(0, row, 1);
    at::mm_out(out_row, input, weight_t);
  }
}

void gemv_bf16_fp32_serial_rows_out(torch::Tensor const& input,
                                    torch::Tensor const& weight,
                                    torch::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "gemv_bf16_fp32_serial_rows_out: tensors must be CUDA");
  TORCH_CHECK(input.device() == weight.device() && input.device() == out.device(),
              "gemv_bf16_fp32_serial_rows_out: device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kFloat32,
              "gemv_bf16_fp32_serial_rows_out: input and weight must be "
              "bfloat16 and out must be float32");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && out.dim() == 2,
              "gemv_bf16_fp32_serial_rows_out: tensors must be 2-D");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "gemv_bf16_fp32_serial_rows_out: tensors must be contiguous");

  int64_t const rows = input.size(0);
  int64_t const k = input.size(1);
  int64_t const n = weight.size(0);
  TORCH_CHECK(rows >= 1 && rows <= 6 && weight.size(1) == k &&
                  out.size(0) == rows && out.size(1) == n,
              "gemv_bf16_fp32_serial_rows_out: expected input [B,K], weight "
              "[N,K], out [B,N], and 1 <= B <= 6");
  TORCH_CHECK(!input.is_alias_of(weight) && !input.is_alias_of(out) &&
                  !weight.is_alias_of(out),
              "gemv_bf16_fp32_serial_rows_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(input.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n), 1,
        static_cast<int>(k), &alpha, weight.data_ptr(),
        CUDA_R_16BF, static_cast<int>(k),
        input.data_ptr<c10::BFloat16>() + row * k, CUDA_R_16BF,
        static_cast<int>(k), &beta, out.data_ptr<float>() + row * n,
        CUDA_R_32F, static_cast<int>(n), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
  }
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

void gemm_fp32_strided_batched_out(torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor& out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
              "gemm_fp32_strided_batched_out: tensors must be CUDA");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(),
              "gemm_fp32_strided_batched_out: device mismatch");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                  b.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "gemm_fp32_strided_batched_out: tensors must be float32");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && out.dim() == 3,
              "gemm_fp32_strided_batched_out: tensors must be 3-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "gemm_fp32_strided_batched_out: tensors must be contiguous");

  int64_t const batch = a.size(0);
  int64_t const M = a.size(1);
  int64_t const K = a.size(2);
  int64_t const N = b.size(1);
  TORCH_CHECK(batch >= 2 && batch <= 6 && b.size(0) == batch &&
                  b.size(2) == K && out.size(0) == batch &&
                  out.size(1) == M && out.size(2) == N,
              "gemm_fp32_strided_batched_out: expected a [B,M,K], "
              "b [B,N,K], out [B,M,N], and 2 <= B <= 6");
  TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max() &&
                  K <= std::numeric_limits<int>::max(),
              "gemm_fp32_strided_batched_out: dimensions must fit int32");
  TORCH_CHECK(!a.is_alias_of(b) && !a.is_alias_of(out) &&
                  !b.is_alias_of(out),
              "gemm_fp32_strided_batched_out: tensors must not alias");

  c10::cuda::CUDAGuard const device_guard(a.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));
  float const alpha = 1.0f;
  float const beta = 0.0f;
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, b.data_ptr<float>(),
      static_cast<int>(K), N * K, a.data_ptr<float>(), static_cast<int>(K),
      M * K, &beta, out.data_ptr<float>(), static_cast<int>(N), M * N,
      static_cast<int>(batch)));
}

}  // namespace metax_sparse
