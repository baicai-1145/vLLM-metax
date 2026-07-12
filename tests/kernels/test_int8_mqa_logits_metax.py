# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_metax.kernels.int8_mqa_logits import int8_mqa_logits


@pytest.mark.parametrize("seq_len,seq_len_kv", [(1, 64), (63, 79), (64, 64)])
def test_int8_mqa_logits_matches_torch(seq_len: int, seq_len_kv: int) -> None:
    torch.manual_seed(7)
    num_heads = 64
    head_dim = 128
    q = torch.randint(
        -8,
        8,
        (seq_len, num_heads, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    k = torch.randint(
        -8,
        8,
        (seq_len_kv, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    scales = torch.rand(seq_len_kv, device="cuda")
    weights = torch.randn(seq_len, num_heads, device="cuda")
    starts = torch.arange(seq_len, device="cuda", dtype=torch.int32).remainder(
        max(1, seq_len_kv // 4)
    )
    ends = torch.full(
        (seq_len,), seq_len_kv, device="cuda", dtype=torch.int32
    ) - starts

    actual = int8_mqa_logits(q, (k, scales), weights, starts, ends)
    expected = (
        torch.einsum("mhd,nd->mhn", q.float(), k.float())
        .mul(scales[None, None, :])
        .relu()
        .mul(weights[:, :, None])
        .sum(1)
    )
    indices = torch.arange(seq_len_kv, device="cuda")[None, :]
    expected.masked_fill_(
        (indices < starts[:, None]) | (indices >= ends[:, None]),
        float("-inf"),
    )

    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=0.1, rtol=1e-4)
