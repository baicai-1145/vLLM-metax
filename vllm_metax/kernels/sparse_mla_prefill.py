# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Streaming sparse MLA prefill kernel.

The kernel deliberately keeps the sparse dimension in a loop.  In particular,
it never forms the ``[tokens, topk, head_dim]`` gather used by the old torch
reference implementation.
"""

import math

import torch
from vllm.triton_utils import tl, triton

_LOG2E = math.log2(math.e)


def _assert_topk_length_device(topk_length: torch.Tensor, topk: int) -> None:
    """Asynchronously reject lengths outside ``[0, topk]`` on-device.

    ``torch._assert_async`` lowers to an accelerator-side assertion and does
    not read the reduction result on the host, so this remains valid during
    CUDA graph capture.  A runtime without that primitive fails closed rather
    than silently accepting malformed metadata.
    """
    condition = torch.all((topk_length >= 0) & (topk_length <= topk))
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is None:
        raise RuntimeError(
            "sparse MLA prefill requires torch._assert_async for topk_length validation"
        )
    assert_async(condition)


@triton.jit
def _sparse_mla_prefill_kernel(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    output,
    max_logits,
    lse,
    tokens,
    kv_tokens,
    heads,
    head_dim,
    value_dim,
    topk,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kvt,
    stride_kvh,
    stride_kvd,
    stride_it,
    stride_ih,
    stride_ik,
    stride_ot,
    stride_oh,
    stride_od,
    stride_mt,
    stride_mh,
    stride_lt,
    stride_lh,
    stride_sh,
    LOG2E: tl.constexpr,
    USE_ATTN_SINK: tl.constexpr,
    USE_TOPK_LENGTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    row_mask = (token < tokens) & (head < heads)
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    q_ptrs = q + token * stride_qt + head * stride_qh + d_offsets * stride_qd
    q_values = tl.load(
        q_ptrs,
        mask=row_mask & (d_offsets < head_dim),
        other=0.0,
    ).to(tl.float32)

    # Scores and the running state use base-2 units, matching FlashMLA's
    # max_logits/lse contract (natural dot products are scaled by log2(e)).
    max_value = tl.full((), float("-inf"), dtype=tl.float32)
    norm = tl.zeros((), dtype=tl.float32)
    value_acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    if USE_TOPK_LENGTH:
        row_length = tl.load(topk_length + token, mask=row_mask, other=0)
    else:
        row_length = topk

    for k in tl.range(0, topk):
        index_ptr = (
            indices
            + token * stride_it
            + 0 * stride_ih
            + k * stride_ik
        )
        index = tl.load(index_ptr, mask=row_mask, other=-1).to(tl.int32)
        valid = (
            row_mask
            & (k < row_length)
            & (index >= 0)
            & (index < kv_tokens)
        )
        # Clamp only the address.  The validity predicate still masks all
        # negative and out-of-range metadata entries.
        safe_index = tl.maximum(index, 0)
        kv_ptrs = (
            kv
            + safe_index * stride_kvt
            + 0 * stride_kvh
            + d_offsets * stride_kvd
        )
        kv_values = tl.load(
            kv_ptrs,
            mask=valid & (d_offsets < head_dim),
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q_values * kv_values, axis=0) * sm_scale * LOG2E
        score = tl.where(valid, score, float("-inf"))

        new_max = tl.maximum(max_value, score)
        # Explicit validity predicates avoid inf-inf when a row starts with
        # padding or contains no valid entries at all.
        safe_new_max = tl.where(new_max > float("-inf"), new_max, 0.0)
        safe_old_max = tl.where(max_value > float("-inf"), max_value, 0.0)
        old_weight = tl.where(
            valid & (max_value > float("-inf")),
            tl.exp2(safe_old_max - safe_new_max),
            1.0,
        )
        weight = tl.where(
            valid & (score > float("-inf")),
            tl.exp2(score - safe_new_max),
            0.0,
        )
        norm = norm * old_weight + weight
        value_ptrs = (
            kv
            + safe_index * stride_kvt
            + 0 * stride_kvh
            + dv_offsets * stride_kvd
        )
        values = tl.load(
            value_ptrs,
            mask=valid & (dv_offsets < value_dim),
            other=0.0,
        ).to(tl.float32)
        value_acc = value_acc * old_weight + weight * values
        max_value = new_max

    # The sink is a virtual zero-valued logit. It contributes to the output
    # denominator, but deliberately does not change the KV-only max/lse
    # metadata returned to FlashMLA callers.
    nonempty = norm > 0.0
    # ``attn_sink`` points at q when sinks are disabled; the constexpr guard
    # below removes its contribution while keeping one stable kernel ABI.
    sink_value = tl.load(attn_sink + head * stride_sh, mask=row_mask, other=0.0)
    safe_max = tl.where(nonempty, max_value, 0.0)
    sink_weight = tl.where(
        USE_ATTN_SINK & nonempty & (sink_value > float("-inf")),
        tl.exp2(sink_value * LOG2E - safe_max),
        0.0,
    )
    total_norm = norm + sink_weight
    output_values = tl.where(total_norm > 0.0, value_acc / total_norm, 0.0)
    output_ptrs = (
        output
        + token * stride_ot
        + head * stride_oh
        + dv_offsets * stride_od
    )
    tl.store(
        output_ptrs,
        output_values.to(tl.bfloat16),
        mask=row_mask & (dv_offsets < value_dim),
    )
    max_ptr = max_logits + token * stride_mt + head * stride_mh
    lse_ptr = lse + token * stride_lt + head * stride_lh
    tl.store(max_ptr, tl.where(nonempty, max_value, float("-inf")), mask=row_mask)
    tl.store(
        lse_ptr,
        tl.where(nonempty, max_value + tl.log2(norm), float("-inf")),
        mask=row_mask,
    )


def sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native streaming sparse MLA prefill kernel.

    Validation is intentionally strict: callers must not silently fall back to
    a torch implementation when a production shape is unsupported.
    """
    if q.device.type != "cuda":
        raise RuntimeError(
            "native sparse MLA prefill requires a MetaX CUDA-compatible device"
        )
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("sparse MLA prefill requires BF16 q and kv")
    if indices.dtype != torch.int32:
        raise TypeError("sparse MLA prefill indices must be int32")
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("q, kv, and indices must be rank-3 tensors")
    tokens, heads, head_dim = q.shape
    kv_tokens, kv_heads, kv_dim = kv.shape
    if kv_heads != 1:
        raise ValueError(f"kv must have one head, got {kv_heads}")
    if indices.shape[0] != tokens or indices.shape[1] != 1:
        raise ValueError(
            f"indices must have shape [tokens, 1, topk], got {tuple(indices.shape)}"
        )
    topk = indices.shape[2]
    if topk <= 0:
        raise ValueError("sparse MLA prefill requires topk > 0")
    if head_dim != kv_dim:
        raise ValueError(f"q/kv head dimensions differ: {head_dim} != {kv_dim}")
    if head_dim > 1024:
        raise ValueError(f"head_dim > 1024 is not supported, got {head_dim}")
    if not 0 < d_v <= 512 or d_v > kv_dim:
        raise ValueError(f"d_v must satisfy 0 < d_v <= min(512, {kv_dim})")
    if q.device != kv.device or q.device != indices.device:
        raise ValueError("q, kv, and indices must be on the same device")
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32:
            raise TypeError("attn_sink must be float32")
        if attn_sink.ndim != 1 or attn_sink.shape[0] != heads:
            raise ValueError(
                f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}"
            )
        if attn_sink.device != q.device:
            raise ValueError("attn_sink must be on the same device as q")
    if topk_length is not None:
        if topk_length.dtype != torch.int32:
            raise TypeError("topk_length must be int32")
        if topk_length.numel() != tokens:
            raise ValueError(
                f"topk_length must contain {tokens} entries, got {topk_length.numel()}"
            )
        if topk_length.device != q.device:
            raise ValueError("topk_length must be on the same device as q")
        topk_length = topk_length.reshape(-1)
        _assert_topk_length_device(topk_length, topk)

    if out is None:
        out = torch.empty((tokens, heads, d_v), device=q.device, dtype=torch.bfloat16)
    elif (
        out.shape != (tokens, heads, d_v)
        or out.dtype != torch.bfloat16
        or out.device != q.device
    ):
        raise ValueError(
            "out must have shape [tokens, heads, d_v], BF16 dtype, and q's device"
        )
    max_logits = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    lse = torch.empty_like(max_logits)
    if tokens == 0 or heads == 0:
        return out, max_logits, lse

    # A power-of-two block is required for the vectorized loads; the masks
    # preserve support for the non-aligned 513/576-dimensional test cases.
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(d_v)
    _sparse_mla_prefill_kernel[(tokens, heads)](
        q,
        kv,
        indices,
        attn_sink if attn_sink is not None else q,
        topk_length if topk_length is not None else q,
        out,
        max_logits,
        lse,
        tokens,
        kv_tokens,
        heads,
        head_dim,
        d_v,
        topk,
        float(sm_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        max_logits.stride(0),
        max_logits.stride(1),
        lse.stride(0),
        lse.stride(1),
        attn_sink.stride(0) if attn_sink is not None else q.stride(1),
        LOG2E=_LOG2E,
        USE_ATTN_SINK=attn_sink is not None,
        USE_TOPK_LENGTH=topk_length is not None,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        num_warps=1,
        num_stages=2,
    )
    return out, max_logits, lse


__all__ = ["sparse_mla_prefill"]
