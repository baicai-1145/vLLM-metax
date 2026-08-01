# SPDX-License-Identifier: Apache-2.0
"""Focused metadata and kernel tests for DeepSeek-V4 DSpark SWA."""

import pytest
import torch

from vllm_metax.v1.attention.backends.mla import sparse_swa
from vllm_metax.v1.attention.backends.mla.metadata_utils import (
    build_token_to_req_indices_out,
)


def _buffer_builder(*, is_dspark: bool) -> sparse_swa.DeepseekSparseSWAMetadataBuilder:
    builder = object.__new__(sparse_swa.DeepseekSparseSWAMetadataBuilder)
    builder.is_dspark = is_dspark
    builder.window_size = 129
    builder.num_speculative_tokens = 3
    builder.noncausal_index_width = (
        (builder.window_size + builder.num_speculative_tokens + 127) // 128 * 128
        if is_dspark
        else 0
    )
    builder._max_tokens = 2
    builder.device = torch.device("cpu")
    builder.decode_swa_indices_noncausal = None
    return builder


def test_dspark_noncausal_buffer_is_aligned_and_graph_stable() -> None:
    builder = _buffer_builder(is_dspark=True)

    first = builder._get_noncausal_decode_indices()
    second = builder._get_noncausal_decode_indices()

    assert first is second
    assert first.shape == (2, 1, 256)
    assert first.dtype == torch.int32


def test_noncausal_swa_fails_closed_without_dspark() -> None:
    builder = _buffer_builder(is_dspark=False)

    with pytest.raises(AssertionError, match="only supported for the DSpark"):
        builder._get_noncausal_decode_indices()


def test_dspark_noncausal_query_length_fails_closed_above_allocated_width() -> None:
    sparse_swa._validate_dspark_query_lengths(
        torch.tensor([0, 5], dtype=torch.int32), max_query_tokens=5
    )

    with pytest.raises(ValueError, match="query length 6 exceeds"):
        sparse_swa._validate_dspark_query_lengths(
            torch.tensor([0, 6], dtype=torch.int32), max_query_tokens=5
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="request-id metadata kernel requires a CUDA-compatible device",
)
@pytest.mark.parametrize(
    "starts",
    ([0, 6], [0, 6, 12], [0, 1, 7], [0, 6, 7, 13]),
)
def test_token_to_req_indices_device_builder_matches_cpu_repeat(
    starts: list[int],
) -> None:
    device = torch.device("cuda")
    query_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)
    num_reqs = len(starts) - 1
    num_tokens = starts[-1]
    out = torch.full((num_tokens + 4,), -7, device=device, dtype=torch.int32)

    actual = build_token_to_req_indices_out(
        query_start_loc,
        num_reqs,
        num_tokens,
        out,
        max_query_len=max(b - a for a, b in zip(starts, starts[1:])),
    )
    torch.cuda.synchronize()

    expected = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.int32),
        torch.tensor(
            [b - a for a, b in zip(starts, starts[1:])],
            device=device,
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out[num_tokens:],
        torch.full((4,), -7, device=device, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSpark non-causal index kernel requires a CUDA-compatible device",
)
def test_dspark_noncausal_kernel_includes_future_query_tokens() -> None:
    device = torch.device("cuda")
    window_size = 8
    query_len = 4
    seq_len = 14
    block_size = 4
    index_width = 128

    indices = torch.zeros((query_len, 1, index_width), device=device, dtype=torch.int32)
    lens = torch.zeros(query_len, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    token_to_req = torch.zeros(query_len, device=device, dtype=torch.int32)
    valid = torch.ones(query_len, device=device, dtype=torch.bool)
    block_table = torch.arange(
        (seq_len + block_size - 1) // block_size,
        device=device,
        dtype=torch.int32,
    ).view(1, -1)

    sparse_swa._compute_dspark_noncausal_swa_indices_kernel[(query_len,)](
        indices,
        indices.stride(0),
        lens,
        window_size,
        index_width,
        query_start_loc,
        seq_lens,
        token_to_req,
        valid,
        block_table,
        block_table.stride(0),
        block_size,
        token_offset=0,
        TRITON_BLOCK_SIZE=1024,
    )
    torch.cuda.synchronize()

    # Context [2, 10) plus the complete query block [10, 14). The first query
    # therefore sees future positions 11, 12, and 13 as well as itself.
    expected = torch.arange(2, seq_len, device=device, dtype=torch.int32)
    torch.testing.assert_close(
        indices[:, 0, : expected.numel()], expected.expand(query_len, -1)
    )
    assert torch.all(indices[:, 0, expected.numel() :] == -1)
    assert torch.all(lens == expected.numel())
