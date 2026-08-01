# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused post-processing of `mhc_pre_norm_fn_ref`.

The reference implementation launches ~6 tiny elementwise/reduce kernels per
call right after the einsum:

    sqrsum = residual.square().sum(-1)                    # reduce
    denom  = (sqrsum / rms_group_size + eps).rsqrt()      # elementwise
    mixes  = (mixes * denom.unsqueeze(-1)).sum(-2)        # elementwise + reduce

This module replaces that with ONE triton kernel: for each residual row
(program id), compute the sum-of-squares over rms_group_size elements (a
scalar accumulator loop over BLOCK_R tiles), derive the rsqrt denominator, and
write `mixes * denom` for the N mix channels.

Production shapes (DeepSeek-V4-Flash, decode M=1):
    mixes        : [B, 1, 24]   (einsum "mbk,nbk->mbn" output, B = residual rows)
    residual_flat: [B, 16384]   (residual flattened to (B, hc_mult*hidden_size))
    out          : [B, 24]
    grid         : (B,)

Only the production layout is supported:
  - dtype: float32, CUDA, contiguous inputs
  - mixes.dim() == 3 and mixes.shape[1] == 1
  - rms_group_size % BLOCK_R == 0
Anything else raises ValueError and the dispatch wrapper in `torch.py` falls
back to the reference ops.
"""

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

# Tile width for the sum-of-squares reduce. 16384 % 4096 == 0, so the
# production residual row is reduced in 4 loop iterations.
BLOCK_R = 4096


@triton.jit
def _mhc_pre_norm_post_kernel(
    mixes_ptr,
    residual_ptr,
    out_ptr,
    mhc_norm_eps,
    N: tl.constexpr,
    RMS_GROUP_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per residual row.

    Computes, for row `pid`:
        s      = sum(residual[pid, :]**2)
        denom  = rsqrt(s / RMS_GROUP_SIZE + mhc_norm_eps)
        out[pid, n] = mixes[pid, 0, n] * denom      for n in [0, N)

    `s` is a scalar accumulator loop-carried across `tl.range` iterations
    (BLOCK_R-wide 1D tiles). This is safe on the MetaX backend: only 2D
    loop-carried tiles core-dump; scalar accumulators are supported.
    """
    pid = tl.program_id(0)
    # sum-of-squares over the residual row, in BLOCK_R-wide tiles.
    s = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, RMS_GROUP_SIZE, BLOCK_R):
        offs_r = off + tl.arange(0, BLOCK_R)
        x = tl.load(residual_ptr + pid * RMS_GROUP_SIZE + offs_r)
        s += tl.sum(x * x)
    denom = tl.rsqrt(s / RMS_GROUP_SIZE + mhc_norm_eps)
    # mixes row [1, N], padded to BLOCK_N (power of 2) with masks.
    offs_n = tl.arange(0, BLOCK_N)
    mixes = tl.load(mixes_ptr + pid * N + offs_n, mask=offs_n < N, other=0.0)
    out = mixes * denom
    tl.store(out_ptr + pid * N + offs_n, out, mask=offs_n < N)


def fused_mhc_pre_norm_post(
    mixes: torch.Tensor, residual_flat: torch.Tensor, mhc_norm_eps: float
) -> torch.Tensor:
    """Fused sqrsum + rsqrt-denom + mixes-scaling.

    Args:
        mixes: [B, 1, N] fp32 CUDA contiguous (einsum output).
        residual_flat: [B, RMS_GROUP_SIZE] fp32 CUDA contiguous.
        mhc_norm_eps: rms epsilon.
    Returns:
        [B, N] fp32 tensor with mixes * rsqrt(sqrsum / R + eps).
    """
    if mixes.dtype != torch.float32 or residual_flat.dtype != torch.float32:
        raise ValueError(
            "fused_mhc_pre_norm_post requires float32 inputs, "
            f"got mixes.dtype={mixes.dtype} residual.dtype={residual_flat.dtype}"
        )
    if not mixes.is_cuda or not residual_flat.is_cuda:
        raise ValueError(
            "fused_mhc_pre_norm_post requires CUDA inputs, "
            f"got mixes.device={mixes.device} residual.device={residual_flat.device}"
        )
    if not mixes.is_contiguous() or not residual_flat.is_contiguous():
        raise ValueError(
            "fused_mhc_pre_norm_post requires contiguous inputs, "
            f"mixes.is_contiguous={mixes.is_contiguous()} "
            f"residual.is_contiguous={residual_flat.is_contiguous()}"
        )
    if mixes.dim() != 3 or mixes.shape[1] != 1:
        raise ValueError(
            f"fused_mhc_pre_norm_post expects mixes [B, 1, N], got {mixes.shape}"
        )
    if residual_flat.dim() != 2:
        raise ValueError(
            "fused_mhc_pre_norm_post expects residual_flat [B, RMS], "
            f"got {residual_flat.shape}"
        )
    b = mixes.shape[0]
    n = mixes.shape[2]
    rms = residual_flat.shape[1]
    if residual_flat.shape[0] != b:
        raise ValueError(
            "fused_mhc_pre_norm_post row mismatch: "
            f"mixes.shape[0]={b} residual_flat.shape[0]={residual_flat.shape[0]}"
        )
    if rms % BLOCK_R != 0:
        raise ValueError(
            f"fused_mhc_pre_norm_post requires rms % BLOCK_R == 0, "
            f"got rms={rms} BLOCK_R={BLOCK_R}"
        )
    if n <= 0 or n > 4096:
        raise ValueError(
            f"fused_mhc_pre_norm_post unsupported N={n} (must be 1..4096)"
        )
    out = torch.empty(b, n, dtype=torch.float32, device=mixes.device)
    block_n = triton.next_power_of_2(n)
    _mhc_pre_norm_post_kernel[(b,)](
        mixes,
        residual_flat,
        out,
        mhc_norm_eps,
        N=n,
        RMS_GROUP_SIZE=rms,
        BLOCK_R=BLOCK_R,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def _fused_mhc_pre_norm_post_fake(
    mixes: torch.Tensor, residual_flat: torch.Tensor, mhc_norm_eps: float
) -> torch.Tensor:
    return torch.empty(
        mixes.shape[0], mixes.shape[2], dtype=mixes.dtype, device=mixes.device
    )


direct_register_custom_op(
    op_name="mx_mhc_pre_norm_post",
    op_func=fused_mhc_pre_norm_post,
    mutates_args=[],
    fake_impl=_fused_mhc_pre_norm_post_fake,
)
