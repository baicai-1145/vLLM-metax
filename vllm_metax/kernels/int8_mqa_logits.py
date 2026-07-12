# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + offsets, 0.0, mask=offsets < n_elements)


@triton.jit
def _int8_mqa_logits_head_grid_kernel(
    q,
    kv,
    scales,
    weights,
    output,
    seq_len,
    seq_len_kv,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvn,
    stride_kvd,
    stride_wm,
    stride_wh,
    stride_om,
    stride_on,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    head = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len_kv

    q_values = tl.load(
        q
        + offs_m[:, None] * stride_qm
        + head * stride_qh
        + offs_k[None, :] * stride_qd,
        mask=mask_m[:, None] & (offs_k[None, :] < head_dim),
        other=0,
    )
    kv_values = tl.load(
        kv + offs_n[:, None] * stride_kvn + offs_k[None, :] * stride_kvd,
        mask=mask_n[:, None] & (offs_k[None, :] < head_dim),
        other=0,
    )
    accumulator = tl.dot(q_values, tl.trans(kv_values)).to(tl.float32)
    scale = tl.load(scales + offs_n, mask=mask_n, other=0.0)
    weight = tl.load(
        weights + offs_m * stride_wm + head * stride_wh,
        mask=mask_m,
        other=0.0,
    )
    contribution = tl.maximum(accumulator * scale[None, :], 0.0) * weight[:, None]
    tl.atomic_add(
        output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        contribution,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _mask_logits_kernel(
    output,
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    seq_len,
    seq_len_kv,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    start = tl.load(cu_seq_len_k_start + row)
    end = tl.load(cu_seq_len_k_end + row)
    valid = (offs_n >= start) & (offs_n < end)
    tl.store(
        output + row * stride_om + offs_n * stride_on,
        float("-inf"),
        mask=(offs_n < seq_len_kv) & ~valid,
    )


def int8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute sparse-indexer logits without mcTriton's multi-dot loop."""
    kv_int8, kv_scales = kv
    q = q.view(torch.int8)
    kv_int8 = kv_int8.view(torch.int8)
    seq_len, num_heads, head_dim = q.shape
    seq_len_kv, kv_head_dim = kv_int8.shape
    if head_dim != kv_head_dim:
        raise ValueError(f"Q/K head dimensions differ: {head_dim} != {kv_head_dim}")
    if weights.shape != (seq_len, num_heads):
        raise ValueError(
            f"weights must have shape {(seq_len, num_heads)}, got {weights.shape}"
        )
    if kv_scales.numel() != seq_len_kv:
        raise ValueError(
            f"KV scales must have {seq_len_kv} elements, got {kv_scales.numel()}"
        )
    if head_dim > 128:
        raise ValueError(f"head_dim > 128 is not supported, got {head_dim}")

    output = torch.empty((seq_len, seq_len_kv), device=q.device, dtype=torch.float32)
    _zero_kernel[(triton.cdiv(output.numel(), 256),)](output, output.numel(), BLOCK=256)
    block_m = 64
    block_n = 64
    _int8_mqa_logits_head_grid_kernel[
        (triton.cdiv(seq_len, block_m), triton.cdiv(seq_len_kv, block_n), num_heads)
    ](
        q,
        kv_int8,
        kv_scales,
        weights,
        output,
        seq_len,
        seq_len_kv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_int8.stride(0),
        kv_int8.stride(1),
        weights.stride(0),
        weights.stride(1),
        output.stride(0),
        output.stride(1),
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
        num_stages=2,
        pipeline="cpasync",
        scenario="storeCoalesce",
    )
    if clean_logits:
        _mask_logits_kernel[(seq_len, triton.cdiv(seq_len_kv, block_n))](
            output,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            seq_len,
            seq_len_kv,
            output.stride(0),
            output.stride(1),
            BLOCK_N=block_n,
        )
    return output


__all__ = ["int8_mqa_logits"]
