# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Sinkhorn normalization kernel for MHC routing mixes.

Replaces the pure-PyTorch `sinkhorn_normalize_ref` (which launches ~39 tiny
2-4us kernels per call: 1 softmax + 1 initial col-normalize + 19*2 row/col
normalizes for hc_sinkhorn_iters=20) with two small triton kernels:

  1. `_sinkhorn_prologue_kernel`: softmax(-1) + add(eps) + col-normalize
     (sum over dim -2). Runs ONCE.
  2. `_sinkhorn_iter_kernel`: one row-normalize + one col-normalize, unrolled
     `ITERS` times (constexpr). Runs ceil(total_iters / ITERS) times.

The MetaX triton backend cannot unroll more than ~7 loop iterations in
`tl.static_range` (compile time grows super-exponentially; 19 iters hangs the
compiler). We therefore chunk the 19 sinkhorn iterations into multiple launches
of a small fixed unroll (ITERS_PER_CHUNK=4 by default), trading a handful of
extra launches for compilable kernels. With hc_sinkhorn_iters=20 (19 iters)
this is 1 prologue + ceil(19/4)=5 iter launches = 6 kernels instead of 39, all
compiled (<1s each).

Only the production layout is supported:
  - dtype: float32
  - last two dims: (MHC_MULT, MHC_MULT) == (4, 4), contiguous (row stride 4)
  - leading dims: dense (stride[i] == stride[i+1] * shape[i+1])
Anything else falls back to `sinkhorn_normalize_ref` in the dispatch wrapper.
"""

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

MHC_MULT = 4
# Unroll count per `_sinkhorn_iter_kernel` launch. Kept small so the MetaX
# triton compiler finishes quickly (4 iters compiles in ~0.5s; >=8 starts to
# slow sharply). Must divide common iteration counts reasonably; 4 gives
# ceil(19/4)=5 chunks for the default hc_sinkhorn_iters=20.
ITERS_PER_CHUNK = 4


@triton.jit
def _sinkhorn_prologue_kernel(
    x_ptr,
    out_ptr,
    n_matrices,
    mh_stride,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """softmax(-1) + eps + col-normalize (sum over dim -2). Matches the first
    three lines of `sinkhorn_normalize_ref`."""
    pid = tl.program_id(0)
    if pid >= n_matrices:
        return
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    ptrs = x_ptr + pid * mh_stride + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(ptrs)  # (M, N) fp32
    # softmax over dim -1
    row_max = tl.max(x, axis=1, keep_dims=True)
    e = tl.exp(x - row_max)
    x = e / tl.sum(e, axis=1, keep_dims=True)
    x = x + eps
    # x / (x.sum(-2, keepdim=True) + eps)
    col_sum = tl.sum(x, axis=0, keep_dims=True)
    x = x / (col_sum + eps)
    out_ptrs = out_ptr + pid * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, x)


@triton.jit
def _sinkhorn_iter_kernel(
    x_ptr,
    out_ptr,
    n_matrices,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    ITERS: tl.constexpr,
):
    """ITERS (constexpr, small) iterations of row-normalize then col-normalize.
    Matches the body of the `for _ in range(repeat - 1)` loop in
    `sinkhorn_normalize_ref`. Unrolled at trace time via `tl.static_range`.

    Note: this kernel operates on the CONTIGUOUS ping-pong buffers produced by
    `_sinkhorn_prologue_kernel` (each 4x4 matrix packed at stride M*N==16). It
    must NOT use the original input's mh_stride, which may be > 16 when the
    input is a strided view (e.g. comb_mix is a slice of a larger tensor with
    block stride 24). The prologue already densified the data."""
    pid = tl.program_id(0)
    if pid >= n_matrices:
        return
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    ptrs = x_ptr + pid * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(ptrs)
    for _ in tl.static_range(0, ITERS):
        # x / (x.sum(-1, keepdim=True) + eps)
        row_sum = tl.sum(x, axis=1, keep_dims=True)
        x = x / (row_sum + eps)
        # x / (x.sum(-2, keepdim=True) + eps)
        col_sum = tl.sum(x, axis=0, keep_dims=True)
        x = x / (col_sum + eps)
    out_ptrs = out_ptr + pid * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, x)


def _compute_layout(x: torch.Tensor):
    """Return (n_matrices, mh_stride) if x uses the supported layout, else None."""
    if x.dim() < 2 or x.shape[-2] != MHC_MULT or x.shape[-1] != MHC_MULT:
        return None
    if x.stride(-1) != 1 or x.stride(-2) != MHC_MULT:
        return None
    if x.dim() == 2:
        return 1, 1
    mh_stride = x.stride(-3)
    for i in range(x.dim() - 3):
        if x.stride(i) != x.stride(i + 1) * x.shape[i + 1]:
            return None
    return x.numel() // (MHC_MULT * MHC_MULT), mh_stride


def fused_sinkhorn(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Fused Sinkhorn normalization.

    Equivalent to `sinkhorn_normalize_ref(x, repeat, eps)`. Requires the
    layout validated by the dispatch wrapper; raises otherwise.
    """
    if x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError(
            "fused_sinkhorn requires a CUDA float32 tensor, "
            f"got dtype={x.dtype} device={x.device}"
        )
    layout = _compute_layout(x)
    if layout is None:
        raise ValueError(
            f"fused_sinkhorn unsupported layout: shape={x.shape} strides={x.stride()}"
        )
    n_matrices, mh_stride = layout

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # Prologue: softmax + eps + col-normalize.
    _sinkhorn_prologue_kernel[(n_matrices,)](
        x, out, n_matrices, mh_stride, eps, M=MHC_MULT, N=MHC_MULT, num_warps=1
    )
    # Iterations: (repeat - 1) row/col normalize pairs, chunked so each launch
    # unrolls only ITERS_PER_CHUNK iterations (MetaX triton cannot compile a
    # large static unroll). Ping-pong between out and a scratch buffer.
    remaining = repeat - 1
    if remaining > 0:
        scratch = torch.empty_like(out)
        src = out
        dst = scratch
        chunk = min(ITERS_PER_CHUNK, remaining)
        # First chunk compiles the ITERS=chunk specialization.
        _sinkhorn_iter_kernel[(n_matrices,)](
            src, dst, n_matrices, eps,
            M=MHC_MULT, N=MHC_MULT, ITERS=chunk, num_warps=1,
        )
        src, dst = dst, src
        remaining -= chunk
        while remaining > 0:
            chunk = min(ITERS_PER_CHUNK, remaining)
            _sinkhorn_iter_kernel[(n_matrices,)](
                src, dst, n_matrices, eps,
                M=MHC_MULT, N=MHC_MULT, ITERS=chunk, num_warps=1,
            )
            src, dst = dst, src
            remaining -= chunk
        out = src
    return out


def _fused_sinkhorn_fake(
    x: torch.Tensor, repeat: int, eps: float
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="mx_mhc_sinkhorn",
    op_func=fused_sinkhorn,
    mutates_args=[],
    fake_impl=_fused_sinkhorn_fake,
)
