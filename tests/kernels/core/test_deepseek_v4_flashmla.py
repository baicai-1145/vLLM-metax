# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest
from types import SimpleNamespace
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_metax.models.deepseek_v4.flashmla as flashmla
from vllm_metax.kernels.sparse_mla_decode import (
    _sparse_mla_scale_mask_kernel,
    sparse_mla_decode,
)
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


def test_torch_sparse_decode_uses_separate_compressed_and_swa_caches() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16)
    swa_cache = torch.tensor(
        [[[[10.0, 0.0]], [[20.0, 0.0]]]], dtype=torch.bfloat16
    )
    compressed_cache = torch.tensor(
        [[[[30.0, 0.0]], [[40.0, 0.0]]]], dtype=torch.bfloat16
    )
    topk_indices = torch.tensor([[[0]]], dtype=torch.int32)
    swa_indices = torch.tensor([[[1]]], dtype=torch.int32)
    output = torch.empty((1, 1, 2), dtype=torch.bfloat16)

    MacaDeepseekV4FlashMLAAttention._torch_sparse_decode(
        q=q,
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        output=output,
        scale=1.0,
    )

    # The first selected row is top-k and must come from compressed_cache;
    # changing the same physical row in swa_cache must not affect it.
    expected = torch.tensor([[[30.0, 0.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    polluted_swa = swa_cache.clone()
    polluted_swa[0, 0, 0].fill_(300.0)
    polluted = torch.empty_like(output)
    MacaDeepseekV4FlashMLAAttention._torch_sparse_decode(
        q=q,
        swa_cache=polluted_swa,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        output=polluted,
        scale=1.0,
    )
    torch.testing.assert_close(polluted, output, atol=0, rtol=0)


def test_torch_sparse_decode_requires_compressed_cache_for_topk() -> None:
    with pytest.raises(ValueError, match="compressed_cache"):
        MacaDeepseekV4FlashMLAAttention._torch_sparse_decode(
            q=torch.ones((1, 1, 1, 2), dtype=torch.bfloat16),
            swa_cache=torch.ones((1, 1, 1, 2), dtype=torch.bfloat16),
            swa_indices=torch.zeros((1, 1, 1), dtype=torch.int32),
            topk_indices=torch.zeros((1, 1, 1), dtype=torch.int32),
            output=torch.empty((1, 1, 2), dtype=torch.bfloat16),
            scale=1.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="native sparse MLA decode requires a MetaX CUDA-compatible device",
)
def test_sparse_mla_scale_mask_uses_swa_indices_after_topk_view() -> None:
    device = torch.device("cuda")
    # Keep positive sentinels after the narrow top-k view.  A combined-index
    # kernel that reads past topk_indices will treat these as valid SWA rows.
    topk_storage = torch.tensor(
        [[[7, 1, 101, 102, 103]]], device=device, dtype=torch.int32
    )
    topk_indices = topk_storage[..., :2]
    swa_indices = torch.tensor([[[0, -1, 2]]], device=device, dtype=torch.int32)
    logits = torch.ones((1, 1, 5), device=device, dtype=torch.float32)

    _sparse_mla_scale_mask_kernel[(1, 1)](
        logits,
        topk_indices,
        swa_indices,
        1,
        1,
        2,
        3,
        2.0,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(2),
        swa_indices.stride(0),
        swa_indices.stride(2),
        BLOCK_K=8,
        num_warps=1,
    )

    expected = torch.tensor(
        [[[2.0, 2.0, 2.0, float("-inf"), 2.0]]],
        device=device,
    )
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)


def test_sparse_mla_decode_backend_defaults_to_native(monkeypatch) -> None:
    monkeypatch.delenv(flashmla._SPARSE_MLA_DECODE_BACKEND_ENV, raising=False)
    assert flashmla._get_sparse_mla_decode_backend() == "native"


def test_sparse_mla_decode_backend_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_BACKEND_ENV, "fallback")
    with pytest.raises(ValueError, match="expected one of"):
        flashmla._get_sparse_mla_decode_backend()


def test_sparse_mla_decode_sync_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv(flashmla._SPARSE_MLA_DECODE_SYNC_ENV, raising=False)
    assert flashmla._get_sparse_mla_decode_sync() is False


def test_sparse_mla_decode_sync_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_SYNC_ENV, "true")
    with pytest.raises(ValueError, match="expected one of: 0, 1"):
        flashmla._get_sparse_mla_decode_sync()


def test_sparse_mla_decode_sync_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv(flashmla._SPARSE_MLA_DECODE_SYNC_ENV, raising=False)
    monkeypatch.setattr(
        flashmla.torch.cuda,
        "current_stream",
        lambda *_: pytest.fail("stream synchronization should be disabled"),
    )
    flashmla._maybe_sync_sparse_mla_decode(torch.empty(1))


def test_sparse_mla_decode_diff_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, raising=False)
    monkeypatch.delenv(
        flashmla._SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV, raising=False
    )
    assert flashmla._get_sparse_mla_decode_diff() is False
    assert flashmla._get_sparse_mla_decode_diff_max_calls() == 256


def test_sparse_mla_decode_diff_rejects_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "true")
    with pytest.raises(ValueError, match="expected one of: 0, 1"):
        flashmla._get_sparse_mla_decode_diff()
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "1")
    monkeypatch.setenv(
        flashmla._SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV, "invalid"
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        flashmla._get_sparse_mla_decode_diff_max_calls()


def test_sparse_mla_decode_diff_restores_native_output_and_logs_mismatch(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "1")
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV, "1")
    flashmla._sparse_mla_decode_diff_reset_state()
    attention = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    attention.scale = 0.5
    q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    cache = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    output = torch.ones((1, 2, 4), dtype=torch.bfloat16)

    def reference(**kwargs):
        kwargs["output"].zero_()

    monkeypatch.setattr(
        flashmla.MacaDeepseekV4FlashMLAAttention,
        "_torch_sparse_decode",
        staticmethod(reference),
    )
    with caplog.at_level("WARNING"):
        flashmla._maybe_diff_sparse_mla_decode(
            attention=attention,
            q=q,
            swa_cache=cache,
            swa_indices=indices,
            topk_indices=None,
            output=output,
            compress_ratio=1,
            mode="test",
        )

    assert torch.equal(output, torch.ones_like(output))
    assert any('"event": "first_mismatch"' in record.message for record in caplog.records)
    assert any('"event": "summary"' in record.message for record in caplog.records)


def test_sparse_mla_decode_diff_dump_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "1")
    monkeypatch.delenv(flashmla._SPARSE_MLA_DECODE_DIFF_DUMP_DIR_ENV, raising=False)
    flashmla._sparse_mla_decode_diff_reset_state()
    attention = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    attention.scale = 0.5
    q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    cache = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    output = torch.ones((1, 2, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(
        flashmla.MacaDeepseekV4FlashMLAAttention,
        "_torch_sparse_decode",
        staticmethod(lambda **kwargs: kwargs["output"].zero_()),
    )
    flashmla._maybe_diff_sparse_mla_decode(
        attention=attention,
        q=q,
        swa_cache=cache,
        swa_indices=indices,
        topk_indices=None,
        output=output,
        compress_ratio=1,
        mode="test",
    )
    assert not list(tmp_path.glob("*.pt"))


def test_sparse_mla_decode_diff_dump_contains_cpu_input_and_output_tensors(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "1")
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_DUMP_DIR_ENV, str(tmp_path))
    flashmla._sparse_mla_decode_diff_reset_state()
    attention = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    attention.scale = 0.5
    attention.prefix = "layer.0"
    q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    cache = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    topk = torch.tensor([[[1, 0]]], dtype=torch.int32)
    output = torch.ones((1, 2, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(
        flashmla.MacaDeepseekV4FlashMLAAttention,
        "_torch_sparse_decode",
        staticmethod(lambda **kwargs: kwargs["output"].zero_()),
    )
    flashmla._maybe_diff_sparse_mla_decode(
        attention=attention,
        q=q,
        swa_cache=cache,
        swa_indices=indices,
        topk_indices=topk,
        output=output,
        compress_ratio=4,
        mode="test",
        q_raw=q.squeeze(1),
        swa_cache_physical=cache.squeeze(-2),
        backend="native",
    )

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    assert payload["schema"] == "dsv4_sparse_mla_decode_diff"
    assert payload["rank"]
    assert payload["call"] == 0
    assert payload["layer_prefix"] == "layer.0"
    assert payload["backend"] == "native"
    assert payload["scale"] == 0.5
    assert payload["compress_ratio"] == 4
    for key in (
        "q_raw",
        "swa_cache_physical",
        "swa_indices",
        "topk_indices",
        "native_output",
        "reference_output",
    ):
        assert payload[key].device.type == "cpu"
        assert key in payload["shapes"]
        assert key in payload["strides"]


def test_sparse_mla_decode_diff_compares_past_log_bound_until_mismatch(monkeypatch):
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_ENV, "1")
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV, "1")
    flashmla._sparse_mla_decode_diff_reset_state()
    attention = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    attention.scale = 0.5
    q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    cache = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    calls = []

    def reference(**kwargs):
        calls.append(True)
        kwargs["output"].fill_(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(
        flashmla.MacaDeepseekV4FlashMLAAttention,
        "_torch_sparse_decode",
        staticmethod(reference),
    )
    for _ in range(2):
        output = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
        flashmla._maybe_diff_sparse_mla_decode(
            attention=attention,
            q=q,
            swa_cache=cache,
            swa_indices=indices,
            topk_indices=None,
            output=output,
            compress_ratio=1,
            mode="test",
        )
    assert len(calls) == 2
    assert flashmla._sparse_mla_decode_diff_comparison_count == 2


def test_sparse_mla_decode_sync_warns_once_and_synchronizes(monkeypatch, caplog) -> None:
    monkeypatch.setenv(flashmla._SPARSE_MLA_DECODE_SYNC_ENV, "1")
    monkeypatch.setattr(flashmla, "_sparse_mla_decode_sync_warning_emitted", False)
    synchronizations = []

    class _Stream:
        def synchronize(self):
            synchronizations.append(True)

    monkeypatch.setattr(
        flashmla.torch.cuda, "current_stream", lambda device: _Stream()
    )
    q = torch.empty(1)
    with caplog.at_level("WARNING"):
        flashmla._maybe_sync_sparse_mla_decode(q)
        flashmla._maybe_sync_sparse_mla_decode(q)

    assert len(synchronizations) == 2
    assert sum("DIAGNOSTIC_ONLY" in record.message for record in caplog.records) == 1


def test_torch_reference_decode_dispatch_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.setenv(
        flashmla._SPARSE_MLA_DECODE_BACKEND_ENV, "torch_reference"
    )
    monkeypatch.setattr(flashmla, "_torch_reference_warning_emitted", False)
    attention = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    attention.compress_ratio = 1
    attention.scale = 0.5
    attention.attn_sink = None
    attention.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    )
    metadata = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_indices=torch.tensor([[[0, 1]]]),
        decode_swa_lens=torch.tensor([2]),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=4,
        token_to_req_indices=torch.zeros(1, dtype=torch.int32),
    )
    calls = []

    def reference(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        flashmla.MacaDeepseekV4FlashMLAAttention,
        "_torch_sparse_decode",
        staticmethod(reference),
    )
    monkeypatch.setattr(
        flashmla,
        "sparse_mla_decode",
        lambda **_: pytest.fail("native backend selected"),
    )
    q = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    with caplog.at_level("WARNING"):
        attention._forward_decode(
            q=q,
            kv_cache=None,
            swa_metadata=metadata,
            attn_metadata=None,
            swa_only=True,
            output=output,
        )
        attention._forward_decode(
            q=q,
            kv_cache=None,
            swa_metadata=metadata,
            attn_metadata=None,
            swa_only=True,
            output=output,
        )

    assert len(calls) == 2
    assert sum("DIAGNOSTIC_ONLY" in record.message for record in caplog.records) == 1
    assert "backend=torch_reference" in caplog.records[0].message


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="native sparse MLA decode requires a MetaX CUDA-compatible device",
)
def test_native_sparse_decode_compat_matches_torch_fixed_softmax() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260712)
    tokens, heads, head_dim, value_dim = 1, 2, 64, 32
    num_blocks, block_size = 4, 4
    q = torch.randn(
        (tokens, heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_storage = torch.randn(
        (num_blocks, block_size, 1, head_dim + 8),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    swa_cache = cache_storage[..., :head_dim]
    compressed_cache = torch.randn(
        (num_blocks, block_size, 1, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    topk_indices = torch.tensor(
        [[[7, 1, -1, 12]]], device=device, dtype=torch.int32
    )
    swa_indices = torch.tensor(
        [[[0, 5, 9, -1, 14]]], device=device, dtype=torch.int32
    )
    zero_lens = torch.zeros(tokens, device=device, dtype=torch.int32)
    block_table = torch.zeros((tokens, num_blocks), device=device, dtype=torch.int32)
    attn_sink = torch.tensor([0.5, -0.25], device=device, dtype=torch.float32)
    expected = torch.empty(
        (tokens, heads, value_dim), device=device, dtype=torch.bfloat16
    )
    MacaDeepseekV4FlashMLAAttention._torch_sparse_decode(
        q=q.unsqueeze(1),
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        output=expected,
        scale=head_dim**-0.5,
    )

    actual = torch.empty_like(expected)
    returned = sparse_mla_decode(
        q=q,
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        swa_lens=zero_lens,
        topk_lens=zero_lens,
        swa_block_table=block_table,
        compressed_block_table=block_table,
        swa_block_size=block_size,
        compressed_block_size=block_size,
        sm_scale=head_dim**-0.5,
        d_v=value_dim,
        attn_sink=attn_sink,
        out=actual,
        compatibility_mode=True,
    )

    assert returned is actual
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)


def test_swa_metadata_short_context_uses_host_max_seq_len() -> None:
    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_size=256,
        max_seq_len=128,
    )

    assert metadata.is_short_context(128)
    assert not metadata.is_short_context(127)
