// SPDX-License-Identifier: Apache-2.0
// Fixed-shape Plan 03 O-projection probe. This file is intentionally not part
// of the production dispatch path.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>

namespace {

constexpr int kHeads = 16;
constexpr int kHeadDim = 512;
constexpr int kGroups = 2;
constexpr int kHeadsPerGroup = 8;
constexpr int kRopeStart = 448;
constexpr int kRopeDim = 64;
constexpr int kHalfRope = kRopeDim / 2;
constexpr int kRank = 1024;
constexpr int kD = kHeadsPerGroup * kHeadDim;
constexpr int kThreads = 64;
constexpr int kRankTile = 8;
constexpr int kWaveSize = 32;
constexpr int kOutputsPerWave = 4;
constexpr int kVectorWidth = 8;

__global__ void o_proj_probe_kernel(
    c10::BFloat16 const* __restrict__ o,
    int64_t const* __restrict__ positions,
    float const* __restrict__ cos_sin,
    c10::BFloat16 const* __restrict__ wo_a,
    c10::BFloat16* __restrict__ z) {
  int const group = blockIdx.y;
  int const rank_base = blockIdx.x * kRankTile;
  int const wave = threadIdx.x % 2;
  int const lane = threadIdx.x / 2;
  int const wave_rank_base = rank_base + wave * kOutputsPerWave;
  int64_t const pos = positions[0];
  float sums[kOutputsPerWave] = {0.0f};

  // The fixed one-token shape lets each block own one grouped output row.
  // Accumulation is FP32, matching the BF16 grouped GEMM contract.
  for (int base = lane * kVectorWidth; base < kD;
       base += kWaveSize * kVectorWidth) {
#pragma unroll
    for (int item = 0; item < kVectorWidth; ++item) {
      int const d = base + item;
      int const head_in_group = d / kHeadDim;
      int const dim = d - head_in_group * kHeadDim;
      int const head = group * kHeadsPerGroup + head_in_group;
      float value = static_cast<float>(o[head * kHeadDim + dim]);
      if (dim >= kRopeStart) {
        int const rope_local = dim - kRopeStart;
        int const partner_dim = dim ^ 1;
        float const partner =
            static_cast<float>(o[head * kHeadDim + partner_dim]);
        int const cache_col = rope_local >> 1;
        float const cos_value = cos_sin[pos * kRopeDim + cache_col];
        float const sin_value =
            cos_sin[pos * kRopeDim + kHalfRope + cache_col];
        value = (rope_local & 1) == 0
            ? value * cos_value + partner * sin_value
            : value * cos_value - partner * sin_value;
      }
      value = static_cast<float>(c10::BFloat16(value));
#pragma unroll
      for (int output = 0; output < kOutputsPerWave; ++output) {
        int const rank = wave_rank_base + output;
        sums[output] += value * static_cast<float>(
            wo_a[group * kRank * kD + rank * kD + d]);
      }
    }
  }

  __shared__ float partial[2][kOutputsPerWave][kWaveSize];
#pragma unroll
  for (int output = 0; output < kOutputsPerWave; ++output) {
    partial[wave][output][lane] = sums[output];
  }
  __syncthreads();
  for (int stride = kWaveSize / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
#pragma unroll
      for (int output = 0; output < kOutputsPerWave; ++output) {
        partial[wave][output][lane] +=
            partial[wave][output][lane + stride];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
#pragma unroll
    for (int output = 0; output < kOutputsPerWave; ++output) {
      z[group * kRank + wave_rank_base + output] =
          c10::BFloat16(partial[wave][output][0]);
    }
  }
}

void fused_bf16_out(torch::Tensor const& o, torch::Tensor const& positions,
                    torch::Tensor const& cos_sin,
                    torch::Tensor const& wo_a, torch::Tensor& z) {
  TORCH_CHECK(o.is_cuda() && positions.is_cuda() && cos_sin.is_cuda() &&
                  wo_a.is_cuda() && z.is_cuda(),
              "o_proj_probe: all tensors must be CUDA tensors");
  TORCH_CHECK(o.device() == positions.device() && o.device() == cos_sin.device() &&
                  o.device() == wo_a.device() && o.device() == z.device(),
              "o_proj_probe: all tensors must be on one device");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16 &&
                  wo_a.scalar_type() == torch::kBFloat16 &&
                  z.scalar_type() == torch::kBFloat16 &&
                  cos_sin.scalar_type() == torch::kFloat32 &&
                  positions.scalar_type() == torch::kInt64,
              "o_proj_probe: expected BF16 o/wo_a/z, FP32 cos_sin, and int64 positions");
  TORCH_CHECK(o.dim() == 3 && o.size(0) == 1 && o.size(1) == kHeads &&
                  o.size(2) == kHeadDim,
              "o_proj_probe: o must have shape [1, 16, 512]");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == 1,
              "o_proj_probe: positions must have shape [1]");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kRopeDim &&
                  cos_sin.size(0) > 0,
              "o_proj_probe: cos_sin must have shape [max_position, 64]");
  TORCH_CHECK(wo_a.dim() == 3 && wo_a.size(0) == kGroups &&
                  wo_a.size(1) == kRank && wo_a.size(2) == kD,
              "o_proj_probe: wo_a must have shape [2, 1024, 4096]");
  TORCH_CHECK(z.dim() == 3 && z.size(0) == 1 && z.size(1) == kGroups &&
                  z.size(2) == kRank,
              "o_proj_probe: z must have shape [1, 2, 1024]");
  TORCH_CHECK(o.is_contiguous() && positions.is_contiguous() &&
                  cos_sin.is_contiguous() && wo_a.is_contiguous() &&
                  z.is_contiguous(),
              "o_proj_probe: all tensors must be contiguous");
  // Bounds are part of the production position contract. Avoid a host read
  // here so the probe remains CUDA-graph capturable; the kernel consumes the
  // device position directly.

  c10::cuda::CUDAGuard const device_guard(o.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  dim3 const grid(kRank / kRankTile, kGroups, 1);
  o_proj_probe_kernel<<<grid, kThreads, 0, stream>>>(
      o.data_ptr<c10::BFloat16>(), positions.data_ptr<int64_t>(),
      cos_sin.data_ptr<float>(), wo_a.data_ptr<c10::BFloat16>(),
      z.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(metax_o_proj_probe, m) {
  m.def("fused_bf16_out(Tensor o, Tensor positions, Tensor cos_sin, "
        "Tensor wo_a, Tensor! z) -> ()");
  m.impl("fused_bf16_out", torch::kCUDA, &fused_bf16_out);
}
