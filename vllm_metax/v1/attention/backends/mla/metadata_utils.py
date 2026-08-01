# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Shared metadata helpers for MetaX MLA backends."""

import torch

from vllm.triton_utils import tl, triton


def build_token_to_req_indices_out(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    out: torch.Tensor,
    *,
    max_query_len: int,
) -> torch.Tensor:
    """Fill ``out[:num_tokens]`` with the request index for each query token."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_reqs < 0:
        raise ValueError(f"num_reqs must be non-negative, got {num_reqs}")
    if out.numel() < num_tokens:
        raise ValueError(
            f"output buffer has {out.numel()} elements, needs at least {num_tokens}"
        )
    if num_tokens == 0:
        return out[:0]
    if not query_start_loc.is_cuda or not out.is_cuda:
        raise ValueError("query_start_loc and out must be device tensors")
    if query_start_loc.dtype != torch.int32 or out.dtype != torch.int32:
        raise ValueError("query_start_loc and out must be int32 tensors")
    if max_query_len <= 0:
        raise ValueError(f"max_query_len must be positive, got {max_query_len}")

    block_size = triton.next_power_of_2(int(max_query_len))
    _build_token_to_req_indices_kernel[(num_reqs,)](
        out,
        query_start_loc,
        num_reqs,
        BLOCK_SIZE=block_size,
    )
    return out[:num_tokens]


@triton.jit
def _build_token_to_req_indices_kernel(
    out_ptr,
    query_start_loc_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    start = tl.load(query_start_loc_ptr + req_idx, mask=req_idx < num_reqs)
    end = tl.load(query_start_loc_ptr + req_idx + 1, mask=req_idx < num_reqs)
    token_idx = start + offsets
    tl.store(
        out_ptr + token_idx,
        req_idx,
        mask=(req_idx < num_reqs) & (token_idx < end),
    )
