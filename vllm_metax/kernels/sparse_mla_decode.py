# SPDX-License-Identifier: Apache-2.0
"""Streaming dual-cache sparse MLA decode kernel.

The kernel consumes paged SWA and compressed caches directly.  It never forms
the ``[tokens, selected, head_dim]`` gather used by the old Torch oracle.
"""

import os

import torch
from vllm.triton_utils import tl, triton


# Workspaces are intentionally process-local and shape keyed.  They are
# allocated during warmup and reused by CUDA graph replay; callers can pass
# their own tensors through ``workspace`` when they need explicit lifetime
# control.
_COMPAT_WORKSPACES: dict[tuple, tuple[torch.Tensor, ...]] = {}
_LAST_COMPAT_WORKSPACE: tuple[torch.Tensor, ...] | None = None
_COMPAT_CUDAGRAPHS: dict[tuple, object] = {}
_COMPAT_CUDAGRAPH_POOL = None
_COMPAT_CUDAGRAPH_ENV = "VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH"


def _gemm_fp32_out_op():
    # Importing the extension registers its torch.library namespace. Keep this
    # lazy so source inspection on hosts without accelerator libraries works.
    try:
        import vllm_metax._metax_sparse_C  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "native sparse MLA decode requires the vllm_metax._metax_sparse_C extension"
        ) from exc
    op = getattr(getattr(torch.ops, "_metax_sparse_C", None), "gemm_fp32_out", None)
    if op is None:
        raise RuntimeError(
            "native sparse MLA decode requires _metax_sparse_C.gemm_fp32_out"
        )
    return op


def _softmax_fp32_out_op():
    try:
        import vllm_metax._metax_sparse_C  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("native sparse MLA requires the MetaX extension") from exc
    op = getattr(getattr(torch.ops, "_metax_sparse_C", None), "softmax_fp32_out", None)
    if op is None:
        raise RuntimeError("native sparse MLA requires softmax_fp32_out")
    return op



# Production decode intentionally matches the frozen Torch oracle until the
# physical metadata contract is corrected.  The alternate path remains
# available for validator/tool experiments via ``compatibility_mode=False``.
SPARSE_MLA_DECODE_MODE = "torch_compat"


@triton.jit
def _sparse_mla_prepare_compat_kernel(
    q,
    swa_cache,
    swa_indices,
    topk_indices,
    probs,
    values,
    tokens,
    heads,
    head_dim,
    d_v,
    swa_blocks,
    swa_block_size,
    swa_width,
    topk_width,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_swb,
    stride_swt,
    stride_swd,
    stride_sit,
    stride_sik,
    stride_tit,
    stride_tik,
    stride_pt,
    stride_ph,
    stride_pk,
    stride_vt,
    stride_vk,
    stride_vd,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    active = (token < tokens) & (head < heads)
    d = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_DV)
    q_ptr = q + token * stride_qt + head * stride_qh + d * stride_qd
    qv = tl.load(q_ptr, mask=active & (d < head_dim), other=0.0).to(tl.float32)
    max_score = tl.full((), float("-inf"), tl.float32)

    # The compatibility contract is deliberately physical and ignores all
    # lengths, tables, compressed-cache metadata, and attention sinks.
    for stream in tl.static_range(0, 2):
        width = topk_width if stream == 0 else swa_width
        index_base = topk_indices if stream == 0 else swa_indices
        index_stride_t = stride_tit if stream == 0 else stride_sit
        index_stride_k = stride_tik if stream == 0 else stride_sik
        for k in tl.range(0, width):
            index = tl.load(
                index_base + token * index_stride_t + k * index_stride_k,
                mask=active,
                other=-1,
            ).to(tl.int32)
            physical_block = index // swa_block_size
            block_offset = index % swa_block_size
            valid = active & (index >= 0) & (physical_block < swa_blocks)
            safe_block = tl.maximum(physical_block, 0)
            safe_offset = tl.maximum(block_offset, 0)
            key = tl.load(
                swa_cache
                + safe_block * stride_swb
                + safe_offset * stride_swt
                + d * stride_swd,
                mask=valid & (d < head_dim),
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(qv * key, axis=0) * sm_scale
            max_score = tl.maximum(max_score, tl.where(valid, score, float("-inf")))

    fixed_max = tl.where(max_score > float("-inf"), max_score, 0.0)
    norm = tl.zeros((), tl.float32)
    for stream in tl.static_range(0, 2):
        width = topk_width if stream == 0 else swa_width
        index_base = topk_indices if stream == 0 else swa_indices
        index_stride_t = stride_tit if stream == 0 else stride_sit
        index_stride_k = stride_tik if stream == 0 else stride_sik
        for k in tl.range(0, width):
            index = tl.load(
                index_base + token * index_stride_t + k * index_stride_k,
                mask=active,
                other=-1,
            ).to(tl.int32)
            physical_block = index // swa_block_size
            block_offset = index % swa_block_size
            valid = active & (index >= 0) & (physical_block < swa_blocks)
            safe_block = tl.maximum(physical_block, 0)
            safe_offset = tl.maximum(block_offset, 0)
            key = tl.load(
                swa_cache
                + safe_block * stride_swb
                + safe_offset * stride_swt
                + d * stride_swd,
                mask=valid & (d < head_dim),
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(qv * key, axis=0) * sm_scale
            weight = tl.where(valid, tl.exp(score - fixed_max), 0.0)
            norm += weight
            prob_k = k if stream == 0 else topk_width + k
            tl.store(
                probs
                + token * stride_pt
                + head * stride_ph
                + prob_k * stride_pk,
                weight,
                mask=active,
            )
            if head == 0:
                gathered = tl.load(
                    swa_cache
                    + safe_block * stride_swb
                    + safe_offset * stride_swt
                    + dv * stride_swd,
                    mask=valid & (dv < d_v),
                    other=0.0,
                ).to(tl.float32)
                value_k = k if stream == 0 else topk_width + k
                tl.store(
                    values
                    + token * stride_vt
                    + value_k * stride_vk
                    + dv * stride_vd,
                    gathered,
                    mask=(token < tokens) & (dv < d_v),
                )

    # Normalize after the complete stream has been visited, matching the
    # fixed-max compatibility softmax while keeping one FP32 probability row.
    for k in tl.range(0, topk_width + swa_width):
        p = tl.load(
            probs + token * stride_pt + head * stride_ph + k * stride_pk,
            mask=active,
            other=0.0,
        )
        tl.store(
            probs + token * stride_pt + head * stride_ph + k * stride_pk,
            tl.where(norm > 0.0, p / norm, 0.0),
            mask=active,
        )


@triton.jit
def _sparse_mla_values_transpose_kernel(values, transposed, tokens, k, d_v,
                                        stride_vt, stride_vk, stride_vd,
                                        stride_bt, stride_bd, stride_bk,
                                        BLOCK_K: tl.constexpr,
                                        BLOCK_DV: tl.constexpr):
    token = tl.program_id(0)
    kt = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    dv = tl.program_id(2) * BLOCK_DV + tl.arange(0, BLOCK_DV)
    x = tl.load(values + token * stride_vt + kt[:, None] * stride_vk + dv[None, :] * stride_vd,
                mask=(token < tokens) & (kt[:, None] < k) & (dv[None, :] < d_v), other=0.0)
    tl.store(transposed + token * stride_bt + dv[None, :] * stride_bd + kt[:, None] * stride_bk,
             x, mask=(token < tokens) & (kt[:, None] < k) & (dv[None, :] < d_v))


@triton.jit
def _sparse_mla_gather_values_kernel(cache, indices, values, tokens, width,
                                     value_offset, d_v, cache_blocks,
                                     cache_block_size, stride_cb, stride_ct,
                                     stride_cd, stride_it, stride_ik,
                                     stride_vt, stride_vk, stride_vd,
                                     BLOCK_DV: tl.constexpr):
    token = tl.program_id(0)
    selected = tl.program_id(1)
    dv = tl.arange(0, BLOCK_DV)
    active = (token < tokens) & (selected < width)
    index = tl.load(indices + token * stride_it + selected * stride_ik,
                    mask=active, other=-1).to(tl.int32)
    block = index // cache_block_size
    offset = index % cache_block_size
    valid = active & (index >= 0) & (block < cache_blocks)
    safe_block = tl.maximum(block, 0)
    safe_offset = tl.maximum(offset, 0)
    value = tl.load(cache + safe_block * stride_cb + safe_offset * stride_ct +
                    dv * stride_cd, mask=valid & (dv < d_v), other=0.0)
    tl.store(values + token * stride_vt + (value_offset + selected) * stride_vk +
             dv * stride_vd, value,
             mask=(token < tokens) & (selected < width) & (dv < d_v))


@triton.jit
def _sparse_mla_cast_kernel(src, dst, tokens, heads, d_v, stride_st, stride_sh, stride_sd,
                            stride_dt, stride_dh, stride_dd, BLOCK_DV: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dv = tl.arange(0, BLOCK_DV)
    mask = (token < tokens) & (head < heads) & (dv < d_v)
    tl.store(dst + token * stride_dt + head * stride_dh + dv * stride_dd,
             tl.load(src + token * stride_st + head * stride_sh + dv * stride_sd, mask=mask, other=0.0).to(tl.bfloat16),
             mask=mask)


@triton.jit
def _sparse_mla_cast_q_kernel(src, dst, tokens, heads, head_dim,
                              stride_st, stride_sh, stride_sd,
                              stride_dt, stride_dh, stride_dd,
                              BLOCK_D: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = (token < tokens) & (head < heads) & (d < head_dim)
    value = tl.load(src + token * stride_st + head * stride_sh + d * stride_sd,
                    mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + token * stride_dt + head * stride_dh + d * stride_dd,
             value, mask=mask)


@triton.jit
def _sparse_mla_scale_mask_kernel(logits, indices, tokens, heads, width,
                                  scale, stride_lt, stride_lh, stride_lk,
                                  stride_it, stride_ik,
                                  BLOCK_K: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    k = tl.arange(0, BLOCK_K)
    active = (token < tokens) & (head < heads) & (k < width)
    index = tl.load(indices + token * stride_it + k * stride_ik,
                    mask=(token < tokens) & (k < width), other=-1)
    value = tl.load(logits + token * stride_lt + head * stride_lh + k * stride_lk,
                    mask=active, other=0.0)
    value = tl.where(index >= 0, value * scale, float("-inf"))
    tl.store(logits + token * stride_lt + head * stride_lh + k * stride_lk,
             value, mask=active)



@triton.jit
def _sparse_mla_decode_kernel(
    q,
    swa_cache,
    compressed_cache,
    swa_indices,
    topk_indices,
    swa_lens,
    topk_lens,
    swa_table,
    compressed_table,
    token_to_req,
    out,
    tokens,
    heads,
    head_dim,
    d_v,
    swa_blocks,
    compressed_blocks,
    swa_block_size,
    compressed_block_size,
    swa_table_width,
    compressed_table_width,
    swa_table_rows,
    compressed_table_rows,
    swa_width,
    topk_width,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_swb,
    stride_swt,
    stride_swd,
    stride_cmb,
    stride_cmt,
    stride_cmd,
    stride_sit,
    stride_si1,
    stride_sik,
    stride_tit,
    stride_ti1,
    stride_tik,
    stride_sll,
    stride_tll,
    stride_sbt,
    stride_sbr,
    stride_cbt,
    stride_cbr,
    stride_req,
    stride_ot,
    stride_oh,
    stride_od,
    attn_sink,
    stride_sh,
    USE_SWA: tl.constexpr,
    USE_COMPRESSED: tl.constexpr,
    USE_SINK: tl.constexpr,
    USE_TOKEN_REQ: tl.constexpr,
    SWA_GLOBAL: tl.constexpr,
    COMPRESSED_GLOBAL: tl.constexpr,
    TORCH_COMPAT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    active = (token < tokens) & (head < heads)
    d = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_DV)
    q_ptr = q + token * stride_qt + head * stride_qh + d * stride_qd
    qv = tl.load(q_ptr, mask=active & (d < head_dim), other=0.0).to(tl.float32)
    if USE_TOKEN_REQ:
        req = tl.load(token_to_req + token * stride_req, mask=active, other=token)
    else:
        req = token

    max_score = tl.full((), float("-inf"), tl.float32)
    norm = tl.zeros((), tl.float32)
    value_acc = tl.zeros((BLOCK_DV,), tl.float32)

    # Compatibility mode follows ``_torch_sparse_decode``: top-k rows precede
    # SWA rows, both indices are physical slots in the SWA cache,
    # lengths/tables/compressed cache/sinks are ignored, and negative indices
    # are masked.  Use a fixed row max so the second-pass FP32 accumulation is
    # closer to Torch softmax than per-index online rescaling.
    if TORCH_COMPAT:
        # Pass 1: stream logits to compute one max across top-k and SWA.
        for stream in tl.static_range(0, 2):
            if stream == 0:
                width = topk_width
                index_ptr_base = topk_indices
                index_stride_t = stride_tit
                index_stride_k = stride_tik
            else:
                width = swa_width
                index_ptr_base = swa_indices
                index_stride_t = stride_sit
                index_stride_k = stride_sik
            for k in tl.range(0, width):
                idx_ptr = index_ptr_base + token * index_stride_t + k * index_stride_k
                physical_slot = tl.load(idx_ptr, mask=active, other=-1).to(tl.int32)
                selected = active & (physical_slot >= 0)
                physical_block = physical_slot // swa_block_size
                block_offset = physical_slot % swa_block_size
                mapped = selected & (physical_block < swa_blocks)
                safe_block = tl.maximum(physical_block, 0)
                safe_offset = tl.maximum(block_offset, 0)
                key_ptr = swa_cache + safe_block * stride_swb + safe_offset * stride_swt + d * stride_swd
                key = tl.load(key_ptr, mask=mapped & (d < head_dim), other=0.0).to(tl.float32)
                score = tl.sum(qv * key, axis=0) * sm_scale
                score = tl.where(mapped, score, float("-inf"))
                max_score = tl.maximum(max_score, score)

        # Pass 2: recompute logits and accumulate against the fixed row max.
        fixed_max = tl.where(max_score > float("-inf"), max_score, 0.0)
        for stream in tl.static_range(0, 2):
            if stream == 0:
                width = topk_width
                index_ptr_base = topk_indices
                index_stride_t = stride_tit
                index_stride_k = stride_tik
            else:
                width = swa_width
                index_ptr_base = swa_indices
                index_stride_t = stride_sit
                index_stride_k = stride_sik
            for k in tl.range(0, width):
                idx_ptr = index_ptr_base + token * index_stride_t + k * index_stride_k
                physical_slot = tl.load(idx_ptr, mask=active, other=-1).to(tl.int32)
                selected = active & (physical_slot >= 0)
                physical_block = physical_slot // swa_block_size
                block_offset = physical_slot % swa_block_size
                mapped = selected & (physical_block < swa_blocks)
                safe_block = tl.maximum(physical_block, 0)
                safe_offset = tl.maximum(block_offset, 0)
                key_ptr = swa_cache + safe_block * stride_swb + safe_offset * stride_swt + d * stride_swd
                key = tl.load(key_ptr, mask=mapped & (d < head_dim), other=0.0).to(tl.float32)
                score = tl.sum(qv * key, axis=0) * sm_scale
                weight = tl.where(mapped, tl.exp(score - fixed_max), 0.0)
                value_ptr = swa_cache + safe_block * stride_swb + safe_offset * stride_swt + dv * stride_swd
                values = tl.load(value_ptr, mask=mapped & (dv < d_v), other=0.0).to(tl.float32)
                norm += weight
                value_acc += weight * values
    else:
        # Validator/tool path retaining the metadata-driven physical/logical
        # dispatch used before the Torch compatibility mode was introduced.
        for stream in tl.static_range(0, 2):
            if stream == 0:
                width = swa_width
                block_size = swa_block_size
                block_count = swa_blocks
                index_ptr_base = swa_indices
                index_stride_t = stride_sit
                index_stride_k = stride_sik
                length_ptr = swa_lens
                length_stride = stride_sll
                table_ptr = swa_table
                table_stride_t = stride_sbt
                table_stride_r = stride_sbr
                table_width = swa_table_width
                table_rows = swa_table_rows
                cache_ptr = swa_cache
                cache_stride_b = stride_swb
                cache_stride_t = stride_swt
                cache_stride_d = stride_swd
                global_indices = SWA_GLOBAL
            else:
                width = topk_width if USE_COMPRESSED else 0
                block_size = compressed_block_size
                block_count = compressed_blocks
                index_ptr_base = topk_indices
                index_stride_t = stride_tit
                index_stride_k = stride_tik
                length_ptr = topk_lens
                length_stride = stride_tll
                table_ptr = compressed_table
                table_stride_t = stride_cbt
                table_stride_r = stride_cbr
                table_width = compressed_table_width
                table_rows = compressed_table_rows
                cache_ptr = compressed_cache
                cache_stride_b = stride_cmb
                cache_stride_t = stride_cmt
                cache_stride_d = stride_cmd
                global_indices = COMPRESSED_GLOBAL
            length = tl.load(length_ptr + token * length_stride, mask=active, other=0)

            for k in tl.range(0, width):
                idx_ptr = index_ptr_base + token * index_stride_t + k * index_stride_k
                logical_idx = tl.load(idx_ptr, mask=active, other=-1).to(tl.int32)
                selected = active & (k < length) & (logical_idx >= 0)
                if global_indices:
                    physical_slot = logical_idx
                    physical_block = physical_slot // block_size
                    block_offset = physical_slot % block_size
                    mapped = selected & (physical_block < block_count)
                else:
                    logical_block = logical_idx // block_size
                    block_offset = logical_idx % block_size
                    safe_logical_block = tl.maximum(logical_block, 0)
                    safe_req = tl.maximum(req, 0)
                    physical_block = tl.load(
                        table_ptr + safe_req * table_stride_t + safe_logical_block * table_stride_r,
                        mask=selected & (req >= 0) & (req < table_rows)
                        & (logical_block >= 0) & (logical_block < table_width),
                        other=-1,
                    ).to(tl.int32)
                    mapped = selected & (logical_block >= 0) & (physical_block >= 0) & (physical_block < block_count)
                    physical_slot = physical_block * block_size + block_offset
                safe_block = tl.maximum(physical_block, 0)
                safe_offset = tl.maximum(block_offset, 0)
                key_ptr = cache_ptr + safe_block * cache_stride_b + safe_offset * cache_stride_t + d * cache_stride_d
                key = tl.load(key_ptr, mask=mapped & (d < head_dim), other=0.0).to(tl.float32)
                score = tl.sum(qv * key, axis=0) * sm_scale
                valid = mapped
                score = tl.where(valid, score, float("-inf"))
                new_max = tl.maximum(max_score, score)
                safe_new = tl.where(new_max > float("-inf"), new_max, 0.0)
                safe_old = tl.where(max_score > float("-inf"), max_score, 0.0)
                old_weight = tl.where(valid & (max_score > float("-inf")), tl.exp(safe_old - safe_new), 1.0)
                weight = tl.where(valid, tl.exp(score - safe_new), 0.0)
                value_ptr = cache_ptr + safe_block * cache_stride_b + safe_offset * cache_stride_t + dv * cache_stride_d
                values = tl.load(value_ptr, mask=valid & (dv < d_v), other=0.0).to(tl.float32)
                norm = norm * old_weight + weight
                value_acc = value_acc * old_weight + weight * values
                max_score = new_max

    nonempty = norm > 0.0
    if USE_SINK:
        sink = tl.load(attn_sink + head * stride_sh, mask=active, other=0.0)
    else:
        sink = tl.zeros((), tl.float32)
    safe_max = tl.where(nonempty, max_score, 0.0)
    sink_weight = tl.where(USE_SINK & nonempty & (sink > float("-inf")), tl.exp(sink - safe_max), 0.0)
    total = norm + sink_weight
    result = tl.where(total > 0.0, value_acc / total, 0.0)
    out_ptr = out + token * stride_ot + head * stride_oh + dv * stride_od
    tl.store(out_ptr, result.to(tl.bfloat16), mask=active & (dv < d_v))


def _cache_layout(cache: torch.Tensor | None, name: str) -> tuple[torch.Tensor, int, int, int, int]:
    if cache is None:
        raise ValueError(f"{name} is required")
    if cache.ndim == 4:
        if cache.shape[2] != 1:
            raise ValueError(f"{name} must have singleton head dimension")
        return cache, cache.shape[0], cache.stride(0), cache.stride(1), cache.stride(3)
    if cache.ndim == 3:
        return cache, cache.shape[0], cache.stride(0), cache.stride(1), cache.stride(2)
    raise ValueError(f"{name} must be rank-3 or rank-4, got {cache.ndim}")


def _compat_workspace(
    q: torch.Tensor,
    tokens: int,
    heads: int,
    k: int,
    d_v: int,
    head_dim: int,
    workspace: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, ...]:
    if workspace is None:
        key = (q.device.type, q.device.index, tokens, heads, k, d_v, head_dim)
        workspace = _COMPAT_WORKSPACES.get(key)
        if workspace is None:
            workspace = (
                torch.empty((tokens, heads, k), device=q.device, dtype=torch.float32),
                torch.empty((tokens, k, d_v), device=q.device, dtype=torch.float32),
                torch.empty((tokens, d_v, k), device=q.device, dtype=torch.float32),
                torch.empty((tokens, heads, d_v), device=q.device, dtype=torch.float32),
                torch.empty((tokens, heads, head_dim), device=q.device, dtype=torch.float32),
            )
            _COMPAT_WORKSPACES[key] = workspace
    if len(workspace) != 5:
        raise ValueError("compatibility workspace must contain five tensors")
    probs, values, transposed, gemm_out, q_fp32 = workspace
    expected = ((tokens, heads, k), (tokens, k, d_v), (tokens, d_v, k),
                (tokens, heads, d_v), (tokens, heads, head_dim))
    for tensor, shape in zip(workspace, expected):
        if tensor.device != q.device or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError("compatibility workspace tensors have invalid device, dtype, or shape")
        if not tensor.is_contiguous():
            raise ValueError("compatibility workspace tensors must be contiguous")
    return probs, values, transposed, gemm_out, q_fp32


def _sparse_mla_decode_compat_eager(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, ...] | None,
) -> None:
    global _LAST_COMPAT_WORKSPACE
    tokens, heads, head_dim = q.shape
    swa_cache, swa_blocks, swb, swt, swd = _cache_layout(swa_cache, "swa_cache")
    topk = topk_indices if topk_indices is not None else swa_indices
    swa_width = swa_indices.shape[2]
    topk_width = topk_indices.shape[2] if topk_indices is not None else 0
    k = topk_width + swa_width
    probs, values, transposed, gemm_out, q_fp32 = _compat_workspace(
        q, tokens, heads, k, d_v, head_dim, workspace
    )
    _LAST_COMPAT_WORKSPACE = (probs, values, transposed, gemm_out, q_fp32)
    if topk_width:
        _sparse_mla_gather_values_kernel[(tokens, topk_width)](
            swa_cache, topk, values, tokens, topk_width, 0, d_v, swa_blocks,
            swa_cache.shape[1], swb, swt, swd, topk.stride(0), topk.stride(2),
            values.stride(0), values.stride(1), values.stride(2),
            BLOCK_DV=triton.next_power_of_2(d_v), num_warps=4, num_stages=2,
        )
    _sparse_mla_gather_values_kernel[(tokens, swa_width)](
        swa_cache, swa_indices, values, tokens, swa_width, topk_width, d_v,
        swa_blocks, swa_cache.shape[1], swb, swt, swd, swa_indices.stride(0),
        swa_indices.stride(2), values.stride(0), values.stride(1),
        values.stride(2), BLOCK_DV=triton.next_power_of_2(d_v), num_warps=4,
        num_stages=2,
    )
    _sparse_mla_values_transpose_kernel[
        (tokens, triton.cdiv(k, 32), triton.cdiv(d_v, 32))
    ](
        values, transposed, tokens, k, d_v, values.stride(0), values.stride(1),
        values.stride(2), transposed.stride(0), transposed.stride(1),
        transposed.stride(2), BLOCK_K=32, BLOCK_DV=32, num_warps=1,
    )
    _sparse_mla_cast_q_kernel[(tokens, heads)](
        q, q_fp32, tokens, heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        q_fp32.stride(0), q_fp32.stride(1), q_fp32.stride(2),
        BLOCK_D=triton.next_power_of_2(head_dim), num_warps=4, num_stages=2,
    )
    gemm_op = _gemm_fp32_out_op()
    for token in range(tokens):
        gemm_op(q_fp32[token].reshape(heads, head_dim), values[token], probs[token])
    _sparse_mla_scale_mask_kernel[(tokens, heads)](
        probs, topk, tokens, heads, k, float(sm_scale),
        probs.stride(0), probs.stride(1), probs.stride(2),
        topk.stride(0), topk.stride(2),
        BLOCK_K=triton.next_power_of_2(k), num_warps=4, num_stages=2,
    )
    _softmax_fp32_out_op()(probs.view(tokens * heads, k),
                           probs.view(tokens * heads, k))
    for token in range(tokens):
        gemm_op(probs[token].reshape(heads, k), transposed[token], gemm_out[token])
    _sparse_mla_cast_kernel[(tokens, heads)](
        gemm_out, out, tokens, heads, d_v, gemm_out.stride(0), gemm_out.stride(1),
        gemm_out.stride(2), out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DV=triton.next_power_of_2(d_v), num_warps=4,
    )


def _compat_graph_pool():
    global _COMPAT_CUDAGRAPH_POOL
    if _COMPAT_CUDAGRAPH_POOL is not None:
        return _COMPAT_CUDAGRAPH_POOL
    from vllm.platforms import current_platform

    _COMPAT_CUDAGRAPH_POOL = current_platform.graph_pool_handle()
    return _COMPAT_CUDAGRAPH_POOL


def _compat_cudagraph_enabled() -> bool:
    return os.environ.get(_COMPAT_CUDAGRAPH_ENV) == "1"


def _compat_tensor_key(tensor: torch.Tensor | None) -> tuple:
    if tensor is None:
        return (None,)
    device = tensor.device
    return (
        device.type,
        device.index,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.data_ptr(),
        tensor.storage_offset(),
    )


def _compat_cudagraph_key(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, ...],
    sm_scale: float,
    d_v: int,
) -> tuple:
    stream = torch.cuda.current_stream(q.device)
    stream_key = getattr(stream, "cuda_stream", id(stream))
    return (
        _compat_tensor_key(q),
        _compat_tensor_key(swa_cache),
        _compat_tensor_key(swa_indices),
        _compat_tensor_key(topk_indices),
        _compat_tensor_key(out),
        tuple(_compat_tensor_key(tensor) for tensor in workspace),
        float(sm_scale),
        int(d_v),
        stream_key,
    )


def sparse_mla_decode_compat_cudagraph_clear_cache() -> None:
    global _COMPAT_CUDAGRAPH_POOL
    if _COMPAT_CUDAGRAPHS:
        torch.cuda.synchronize()
    _COMPAT_CUDAGRAPHS.clear()
    _COMPAT_CUDAGRAPH_POOL = None


def _sparse_mla_decode_compat(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, ...] | None,
) -> None:
    if not _compat_cudagraph_enabled():
        _sparse_mla_decode_compat_eager(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            d_v=d_v,
            out=out,
            workspace=workspace,
        )
        return

    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        capture_active = bool(probe()) if probe is not None else True
    except Exception as exc:
        raise RuntimeError(
            "sparse MLA compatibility CUDA graph capture status is unavailable"
        ) from exc
    if capture_active:
        raise RuntimeError("sparse MLA compatibility CUDA graph capture cannot nest")

    tokens, heads, head_dim = q.shape
    swa_width = swa_indices.shape[2]
    topk_width = topk_indices.shape[2] if topk_indices is not None else 0
    resolved_workspace = _compat_workspace(
        q,
        tokens,
        heads,
        topk_width + swa_width,
        d_v,
        head_dim,
        workspace,
    )
    key = _compat_cudagraph_key(
        q,
        swa_cache,
        swa_indices,
        topk_indices,
        out,
        resolved_workspace,
        sm_scale,
        d_v,
    )
    graph = _COMPAT_CUDAGRAPHS.get(key)
    global _LAST_COMPAT_WORKSPACE
    _LAST_COMPAT_WORKSPACE = resolved_workspace
    if graph is not None:
        graph.replay()
        return

    _sparse_mla_decode_compat_eager(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        sm_scale=sm_scale,
        d_v=d_v,
        out=out,
        workspace=resolved_workspace,
    )
    stream = torch.cuda.current_stream(q.device)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=_compat_graph_pool()):
        _sparse_mla_decode_compat_eager(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            d_v=d_v,
            out=out,
            workspace=resolved_workspace,
        )
    _COMPAT_CUDAGRAPHS[key] = graph


def sparse_mla_decode_compat_workspace() -> tuple[torch.Tensor, ...] | None:
    return _LAST_COMPAT_WORKSPACE


def sparse_mla_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    swa_lens: torch.Tensor,
    topk_lens: torch.Tensor | None,
    swa_block_table: torch.Tensor,
    compressed_block_table: torch.Tensor | None,
    swa_block_size: int,
    compressed_block_size: int | None,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    token_to_req: torch.Tensor | None = None,
    swa_indices_are_global: bool = False,
    compressed_indices_are_global: bool = False,
    compatibility_mode: bool = True,
    workspace: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Run native sparse decode without a Torch fallback.

    ``compatibility_mode`` is a compile-time dispatch choice.  Its default is
    the frozen ``_torch_sparse_decode`` contract; set it to ``False`` only for
    physical metadata validator/tool experiments.
    """
    if q.device.type != "cuda":
        raise RuntimeError("native sparse MLA decode requires a MetaX CUDA-compatible device")
    if compatibility_mode:
        _gemm_fp32_out_op()
    if q.dtype != torch.bfloat16 or swa_cache.dtype != torch.bfloat16:
        raise TypeError("sparse MLA decode requires BF16 q and caches")
    if q.ndim != 3 or swa_indices.ndim != 3 or swa_indices.shape[1] != 1:
        raise ValueError("q and indices must have q=[tokens,heads,d], indices=[tokens,1,width]")
    if swa_indices.dtype != torch.int32 or (topk_indices is not None and topk_indices.dtype != torch.int32):
        raise TypeError("sparse MLA decode indices must be int32")
    tokens, heads, head_dim = q.shape
    if topk_indices is not None and (
        topk_indices.ndim != 3
        or topk_indices.shape[0] != tokens
        or topk_indices.shape[1] != 1
        or topk_indices.device != q.device
    ):
        raise ValueError("topk_indices must have shape [tokens, 1, width] on q's device")
    if attn_sink is not None and (
        attn_sink.dtype != torch.float32
        or attn_sink.shape != (heads,)
        or attn_sink.device != q.device
    ):
        raise ValueError("attn_sink must be float32 [heads] on q's device")
    if token_to_req is not None and (
        token_to_req.dtype != torch.int32
        or token_to_req.numel() != tokens
        or token_to_req.device != q.device
    ):
        raise ValueError("token_to_req must be int32 [tokens] on q's device")
    if swa_indices.shape[0] != tokens or swa_lens.numel() != tokens:
        raise ValueError("SWA metadata token dimensions do not match q")
    if swa_block_size <= 0 or d_v <= 0 or d_v > 512 or d_v > head_dim:
        raise ValueError("invalid block size or value dimension")
    if swa_cache.device != q.device or swa_indices.device != q.device or swa_lens.device != q.device:
        raise ValueError("all sparse decode inputs must share q's device")
    if swa_lens.dtype != torch.int32:
        raise TypeError("swa_lens must be int32")
    if out is None:
        out = torch.empty((tokens, heads, d_v), device=q.device, dtype=torch.bfloat16)
    elif out.shape != (tokens, heads, d_v) or out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must have shape [tokens, heads, d_v], BF16 dtype, and q's device")
    if compatibility_mode:
        _sparse_mla_decode_compat(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            d_v=d_v,
            out=out,
            workspace=workspace,
        )
        return out
    if compressed_cache is not None and not compatibility_mode:
        if compressed_cache.dtype != torch.bfloat16 or compressed_cache.device != q.device:
            raise ValueError("compressed_cache must be BF16 on q's device")
        if topk_indices is None or topk_lens is None or compressed_block_table is None or compressed_block_size is None:
            raise ValueError("compressed cache requires topk indices/lens/table/block size")
        if topk_indices.shape[0] != tokens or topk_indices.shape[1] != 1 or topk_lens.numel() != tokens:
            raise ValueError("compressed metadata token dimensions do not match q")
        if topk_lens.dtype != torch.int32:
            raise TypeError("topk_lens must be int32")
    elif topk_indices is not None and not compatibility_mode:
        raise ValueError("topk_indices require compressed_cache")
    if swa_block_table.ndim != 2 or swa_block_table.dtype != torch.int32 or swa_block_table.device != q.device:
        raise ValueError("swa_block_table must be int32 rank-2 on q's device")
    if token_to_req is None and swa_block_table.shape[0] < tokens:
        raise ValueError("per-token SWA block table must have one row per token")

    swa_cache, swa_blocks, swb, swt, swd = _cache_layout(swa_cache, "swa_cache")
    comp_cache, comp_blocks, cmb, cmt, cmd = _cache_layout(
        (compressed_cache if compressed_cache is not None and not compatibility_mode
         else swa_cache),
        "compressed_cache",
    )
    if swa_cache.shape[-1] < head_dim or comp_cache.shape[-1] < head_dim:
        raise ValueError("cache head dimension must cover q head dimension")
    if compressed_block_table is None:
        compressed_block_table = swa_block_table
    if compressed_block_size is None:
        compressed_block_size = 1
    use_token_req = token_to_req is not None
    token_req_tensor = token_to_req if token_to_req is not None else q
    kernel_swa_block_size = swa_cache.shape[1] if compatibility_mode else swa_block_size

    _sparse_mla_decode_kernel[(tokens, heads)](
        q, swa_cache, comp_cache, swa_indices, topk_indices if topk_indices is not None else swa_indices,
        swa_lens, topk_lens if topk_lens is not None else swa_lens,
        swa_block_table, compressed_block_table, token_req_tensor, out,
        tokens, heads, head_dim, d_v, swa_blocks, comp_blocks,
        kernel_swa_block_size,
        compressed_block_size, swa_block_table.shape[1], compressed_block_table.shape[1],
        swa_block_table.shape[0], compressed_block_table.shape[0],
        swa_indices.shape[2], topk_indices.shape[2] if topk_indices is not None else 0,
        float(sm_scale), q.stride(0), q.stride(1), q.stride(2), swb, swt, swd, cmb, cmt, cmd,
        swa_indices.stride(0), swa_indices.stride(1), swa_indices.stride(2),
        (topk_indices if topk_indices is not None else swa_indices).stride(0),
        (topk_indices if topk_indices is not None else swa_indices).stride(1),
        (topk_indices if topk_indices is not None else swa_indices).stride(2),
        swa_lens.stride(0), (topk_lens if topk_lens is not None else swa_lens).stride(0),
        swa_block_table.stride(0), swa_block_table.stride(1), compressed_block_table.stride(0), compressed_block_table.stride(1),
        token_req_tensor.stride(0), out.stride(0), out.stride(1), out.stride(2),
        attn_sink if attn_sink is not None else q, attn_sink.stride(0) if attn_sink is not None else q.stride(1),
        USE_SWA=True,
        USE_COMPRESSED=compressed_cache is not None and not compatibility_mode,
        USE_SINK=attn_sink is not None and not compatibility_mode,
        USE_TOKEN_REQ=use_token_req, SWA_GLOBAL=swa_indices_are_global,
        COMPRESSED_GLOBAL=compressed_indices_are_global,
        TORCH_COMPAT=compatibility_mode,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_DV=triton.next_power_of_2(d_v), num_warps=4, num_stages=2,
    )
    return out


__all__ = [
    "SPARSE_MLA_DECODE_MODE",
    "sparse_mla_decode",
    "sparse_mla_decode_compat_workspace",
    "sparse_mla_decode_compat_cudagraph_clear_cache",
]
