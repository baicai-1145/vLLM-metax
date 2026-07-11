# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_metax.models.deepseek_v4.flashmla import (
    MacaDeepseekV4FlashMLAAttention,
)
from vllm_metax.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
)


class _Fp32ConversionRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        self.input_numels: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func == torch.ops.aten._to_copy.default
            and kwargs.get("dtype") == torch.float32
        ):
            self.input_numels.append(args[0].numel())
        return func(*args, **kwargs)


def test_torch_sparse_decode_converts_only_selected_cache_rows() -> None:
    torch.manual_seed(0)
    batch, num_heads, head_dim = 1, 2, 8
    num_blocks, block_size = 4, 4
    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.bfloat16)

    # Slice a padded allocation to reproduce the non-contiguous cache stride.
    cache_storage = torch.randn(
        num_blocks,
        block_size,
        1,
        head_dim + 4,
        dtype=torch.bfloat16,
    )
    cache = cache_storage[..., :head_dim]
    indices = torch.tensor([[[0, 5, 9, -1]]], dtype=torch.int64)
    output = torch.empty(batch, num_heads, head_dim, dtype=torch.bfloat16)

    invalid = indices[:, 0] < 0
    gather_idx = indices[:, 0].masked_fill(invalid, 0)
    gathered = cache.reshape(-1, head_dim).float().index_select(
        0, gather_idx.reshape(-1)
    ).view(batch, -1, head_dim)
    scores = torch.matmul(q.squeeze(1).float(), gathered.transpose(1, 2))
    scores.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores * head_dim**-0.5, dim=-1)
    expected = torch.matmul(probs, gathered).to(torch.bfloat16)

    recorder = _Fp32ConversionRecorder()
    with recorder:
        MacaDeepseekV4FlashMLAAttention._torch_sparse_decode(
            q=q,
            swa_cache=cache,
            swa_indices=indices,
            topk_indices=None,
            output=output,
            scale=head_dim**-0.5,
        )

    selected_cache_numel = indices.numel() * head_dim
    assert max(recorder.input_numels) <= selected_cache_numel
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_swa_metadata_short_context_uses_host_max_seq_len() -> None:
    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_size=256,
        max_seq_len=128,
    )

    assert metadata.is_short_context(128)
    assert not metadata.is_short_context(127)
