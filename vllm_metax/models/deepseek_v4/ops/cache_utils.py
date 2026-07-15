# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.triton_utils import tl, triton
from typing import Any

@triton.jit
def _gather_k_cache_kernel(
    out_ptr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    k_cache_ptr,
    k_cache_stride0: tl.constexpr,
    k_cache_stride1: tl.constexpr,
    k_cache_stride2: tl.constexpr,
    seq_lens_ptr,
    block_table_ptr,
    offset: tl.constexpr,
    gather_lens_ptr,
    # constexpr
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_worker_id = tl.program_id(1)
    dim_block_id = tl.program_id(2)

    num_token_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len

    start_pos = seq_len - gather_len

    dim_offsets = dim_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_size

    for i in range(token_worker_id, gather_len, num_token_workers):
        pos = start_pos + i

        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        # k_cache layout:
        # [num_blocks, cache_block_size, head_size]
        k_ptr = (
            k_cache_ptr
            + physical_block_idx.to(tl.int64) * k_cache_stride0
            + pos_in_block * k_cache_stride1
            + dim_offsets * k_cache_stride2
        )

        out_ptr_cur = (
            out_ptr
            + batch_idx * out_stride0
            + (offset + i) * out_stride1
            + dim_offsets
        )

        vals = tl.load(k_ptr, mask=dim_mask, other=0.0)
        tl.store(out_ptr_cur, vals, mask=dim_mask)


def gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_size]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs] or None
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    num_reqs = seq_lens.shape[0]
    head_size = k_cache.shape[2]


    if gather_lens is not None:
        assert gather_lens.is_cuda
        assert gather_lens.shape == seq_lens.shape

    NUM_TOKEN_WORKERS = 128
    BLOCK_D = triton.next_power_of_2(head_size)


    _gather_k_cache_kernel[(num_reqs, NUM_TOKEN_WORKERS, 1)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        head_size=head_size,
        BLOCK_D=BLOCK_D,
    )


def compute_global_topk_indices_and_lens_bounded(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map C4 local top-k indices, rejecting unmaterialized cache rows.

    The indexer buffer is sized for the maximum model length, while a request
    may have only a prefix of its compressed cache materialized.  Bounds are
    therefore derived per token from its request sequence length rather than
    from the block table capacity.
    """
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    _compute_global_topk_indices_and_lens_bounded_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _compute_global_topk_indices_and_lens_bounded_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    compress_ratio,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_is_valid = tl.load(is_valid_token_ptr + token_idx) != 0
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    compressed_len = seq_len // compress_ratio

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        is_valid = (local_idx >= 0) & (local_idx < compressed_len)

        # Keep invalid indices from forming an out-of-range block-table
        # pointer.  The masked load guarantees no block-table read occurs for
        # an invalid local index.
        safe_local_idx = tl.where(is_valid, local_idx, 0)
        block_indices = safe_local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & is_valid,
            other=0,
        )
        block_offsets = safe_local_idx % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(token_is_valid, count, 0))
