from types import SimpleNamespace

import pytest
import torch

import vllm_metax.models.deepseek_v4.attention as attention
import vllm_metax.models.deepseek_v4.flashmla as flashmla
import vllm_metax.models.deepseek_v4.layer_debug as layer_debug
import vllm_metax.models.deepseek_v4.model as model
from vllm_metax.models.deepseek_v4.collective_census import (
    collective_census_context,
)
from vllm_metax.models.deepseek_v4.ops import sparse_mla_debug


ENV = "VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING"


@pytest.mark.parametrize("rows", [1, 2, 5, 6])
def test_cached_row_indices_values_and_reuses_storage(rows):
    from vllm_metax.models.deepseek_v4.row_indices import (
        get_cached_row_indices,
    )

    first = get_cached_row_indices(rows, torch.device("cpu"))
    second = get_cached_row_indices(rows, torch.device("cpu"))

    torch.testing.assert_close(first, torch.arange(rows, dtype=torch.int64))
    assert first.dtype is torch.int64
    assert first.data_ptr() == second.data_ptr()


def test_cached_row_indices_keeps_dtype_in_cache_key():
    from vllm_metax.models.deepseek_v4.row_indices import (
        get_cached_row_indices,
    )

    int64_rows = get_cached_row_indices(3, torch.device("cpu"))
    int32_rows = get_cached_row_indices(3, torch.device("cpu"), torch.int32)

    assert int64_rows.dtype is torch.int64
    assert int32_rows.dtype is torch.int32
    assert int64_rows.data_ptr() != int32_rows.data_ptr()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or MACA device"
)
def test_cached_row_indices_reuses_accelerator_storage():
    from vllm_metax.models.deepseek_v4.row_indices import (
        get_cached_row_indices,
    )

    device = torch.device("cuda")
    first = get_cached_row_indices(6, device)
    second = get_cached_row_indices(6, device)

    torch.testing.assert_close(
        first, torch.arange(6, dtype=torch.int64, device=device)
    )
    assert first.data_ptr() == second.data_ptr()


def test_tokenwise_sparse_selectors_do_not_use_row_index_cache(monkeypatch):
    def fail_cached_indices(*args, **kwargs):
        raise AssertionError("sparse selector must use nonzero")

    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", raising=False)
    positions = torch.tensor([766, 767, 768])
    cases = [
        (
            attention,
            attention._target_tokenwise_wq_b_selected_indices,
            "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS",
        ),
        (
            attention,
            attention._target_tokenwise_qkv_selected_indices,
            "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS",
        ),
        (
            model,
            lambda value: model._tokenwise_ffn_selected_indices(
                value, torch.empty(value.shape[0], 2)
            ),
            "VLLM_METAX_DSV4_TOKENWISE_FFN_POSITIONS",
        ),
        (
            flashmla,
            flashmla._tokenwise_o_proj_selected_indices,
            "VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS",
        ),
    ]
    for module, selector, env_name in cases:
        monkeypatch.setenv(env_name, "767")
        monkeypatch.setattr(module, "get_cached_row_indices", fail_cached_indices)
        torch.testing.assert_close(selector(positions), torch.tensor([1]))


class _FakeLayer(attention.MacaDeepseekV4Attention):
    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def forward_mqa(self, q, kv, positions, output):
        raise NotImplementedError

    def _o_proj(self, o, positions):
        raise NotImplementedError


class _ShapeSensitiveWqB:
    def __init__(self):
        self.calls = []

    def __call__(self, qr):
        self.calls.append(qr.shape[0])
        # Model a native GEMM whose reduction path depends on M.
        return qr + qr.new_tensor(float(qr.shape[0]))


class _SharedBufferWqB:
    def __init__(self):
        self.buffer = torch.empty(1, 2)
        self.calls = 0

    def __call__(self, qr):
        self.calls += 1
        self.buffer.copy_(qr + self.calls)
        return self.buffer


class _ShapeSensitiveFusedQKV:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states.shape[0])
        return (hidden_states + hidden_states.new_tensor(float(hidden_states.shape[0])),)


class _AddNorm:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


def _layer(*, enabled: bool, chunk_size: int = 256):
    layer = object.__new__(_FakeLayer)
    layer._prefill_gemm_chunking_enabled = enabled
    layer._prefill_gemm_chunk_size = chunk_size
    return layer


def _rope_config(compress_ratios):
    return SimpleNamespace(
        num_hidden_layers=4,
        compress_ratios=compress_ratios,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 1024,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )


def _flashmla_layer(*, enabled: bool, chunk_size: int = 256):
    layer = object.__new__(flashmla.MacaDeepseekV4FlashMLAAttention)
    layer.layer_idx = 7
    layer._prefill_gemm_chunking_enabled = enabled
    layer._prefill_gemm_chunk_size = chunk_size
    layer.rotary_emb = type("Rotary", (), {"cos_sin_cache": torch.empty(1)})()
    layer.wo_a = object()
    layer.wo_b = object()
    layer.n_local_groups = 1
    layer.n_local_heads = 1
    layer.nope_head_dim = 2
    layer.rope_head_dim = 2
    layer.o_lora_rank = 3
    return layer


@pytest.mark.parametrize(
    "compress_ratios,layer_id,expected_ratio,expected_unscaled",
    [
        ([0, 4, 128, 4], 0, 1, False),
        ([0, 4, 128, 4], 1, 4, False),
        ([0, 4, 128, 4, 0], 4, 1, True),
        ([0, 4, 128, 4, 4], 4, 4, False),
        ([0, 4, 128, 4], 4, 1, False),
    ],
)
def test_resolve_layer_compress_ratio_honors_mtp_raw_zero(
    compress_ratios, layer_id, expected_ratio, expected_unscaled
):
    ratio, use_unscaled_rope = attention.resolve_layer_compress_ratio(
        _rope_config(compress_ratios), layer_id
    )

    assert ratio == expected_ratio
    assert use_unscaled_rope is expected_unscaled
    assert ratio >= 1


def test_build_deepseek_v4_rope_uses_unscaled_rope_without_mutating(monkeypatch):
    captured = {}

    def fake_get_rope(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "rope"

    monkeypatch.setattr(attention, "get_rope", fake_get_rope)
    config = _rope_config([0, 4, 128, 4, 0])
    original = dict(config.rope_parameters)

    result = attention.build_deepseek_v4_rope(
        config,
        head_dim=128,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
        use_unscaled_rope=True,
    )

    assert result == "rope"
    rope_parameters = captured["kwargs"]["rope_parameters"]
    assert rope_parameters["rope_type"] == "default"
    assert rope_parameters["rope_theta"] == config.rope_theta
    assert rope_parameters["is_deepseek_v4"] is True
    assert rope_parameters["rope_dim"] == 64
    assert captured["kwargs"]["dtype"] is torch.float32
    assert config.rope_parameters == original


def test_build_deepseek_v4_rope_handles_nested_rope_parameters(monkeypatch):
    captured = {}

    def fake_get_rope(*args, **kwargs):
        captured["kwargs"] = kwargs
        return "rope"

    monkeypatch.setattr(attention, "get_rope", fake_get_rope)
    config = SimpleNamespace(
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0},
            "compress": {
                "rope_type": "yarn",
                "rope_theta": 160000.0,
                "factor": 16,
                "original_max_position_embeddings": 1024,
            },
        },
    )

    attention.build_deepseek_v4_rope(
        config,
        head_dim=128,
        rope_head_dim=64,
        max_position_embeddings=4096,
        compress_ratio=4,
    )

    rope_parameters = captured["kwargs"]["rope_parameters"]
    assert rope_parameters["rope_type"] == "deepseek_yarn"
    assert rope_parameters["rope_theta"] == config.compress_rope_theta
    assert rope_parameters["rope_dim"] == 64


def test_attention_forward_uses_slot0_for_packed_hidden_states(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.padded_heads = 1
    layer.head_dim = 2
    layer.n_local_heads = 1
    layer.q_lora_rank = 1
    layer.q_norm = SimpleNamespace(weight=SimpleNamespace(data=torch.ones(1)))
    layer.kv_norm = SimpleNamespace(weight=SimpleNamespace(data=torch.ones(2)))
    layer.eps = 1e-6
    observed = {}

    def fake_gemm(hidden_states):
        observed["gemm_hidden_states"] = hidden_states.clone()
        return (
            torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            None,
            None,
            None,
        )

    def fake_rmsnorm(qr, kv, *args):
        return qr, kv

    def fake_attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
    ):
        observed["impl_hidden_states"] = hidden_states.clone()
        observed["qr"] = qr.clone()
        observed["kv"] = kv.clone()
        out.copy_(kv.view(2, 1, 2))

    def fake_o_proj(o, positions):
        observed["o_proj_positions"] = positions.clone()
        return o.reshape(2, 2)

    monkeypatch.setattr(attention, "fused_q_kv_rmsnorm", fake_rmsnorm)
    layer.attn_gemm_parallel_execute = fake_gemm
    layer.attention_impl = fake_attention_impl
    layer._o_proj = fake_o_proj
    hidden_states = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [101.0, 102.0, 103.0]],
            [[4.0, 5.0, 6.0], [104.0, 105.0, 106.0]],
        ]
    )
    positions = torch.tensor([7, 8], dtype=torch.int64)

    result = attention.MacaDeepseekV4Attention.forward(layer, positions, hidden_states)

    torch.testing.assert_close(observed["gemm_hidden_states"], hidden_states[:, 0, :])
    torch.testing.assert_close(observed["impl_hidden_states"], hidden_states[:, 0, :])
    torch.testing.assert_close(observed["qr"], torch.tensor([[10.0], [40.0]]))
    torch.testing.assert_close(
        observed["kv"], torch.tensor([[20.0, 30.0], [50.0, 60.0]])
    )
    torch.testing.assert_close(result, torch.tensor([[20.0, 30.0], [50.0, 60.0]]))
    torch.testing.assert_close(observed["o_proj_positions"], positions)


def test_prefill_o_proj_chunking_uses_block_sized_calls(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append(
            (
                o.shape[0],
                positions.shape[0],
                kwargs["layer_idx"],
                kwargs["chunk_index"],
            )
        )
        return o[:, 0, :]

    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=True)
    o = torch.arange(633 * 2 * 4).reshape(633, 2, 4)
    positions = torch.arange(633)

    result = layer._o_proj(o, positions)

    assert calls == [
        (256, 256, 7, 0),
        (256, 256, 7, 1),
        (121, 121, 7, 2),
    ]
    torch.testing.assert_close(result, o[:, 0, :])


def test_prefill_o_proj_chunking_disabled_or_small_delegates_once(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append(
            (
                o.shape[0],
                positions.shape[0],
                kwargs["layer_idx"],
                kwargs["chunk_index"],
            )
        )
        return o[:, 0, :]

    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    o = torch.zeros(4, 1, 2)
    positions = torch.arange(4)

    torch.testing.assert_close(
        _flashmla_layer(enabled=False)._o_proj(o, positions), o[:, 0, :]
    )
    torch.testing.assert_close(
        _flashmla_layer(enabled=True, chunk_size=4)._o_proj(o, positions),
        o[:, 0, :],
    )
    assert calls == [(4, 4, 7, 0), (4, 4, 7, 0)]


def test_tokenwise_o_proj_uses_one_complete_native_call_per_row(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return o[:, 0, :] + kwargs["chunk_index"]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3).reshape(2, 1, 3)
    positions = torch.tensor([646, 647])

    result = layer._o_proj(o, positions)

    assert calls == [(1, [646], 0), (1, [647], 1)]
    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0], o[1, 0] + 1]),
    )


def test_tokenwise_o_proj_all_rows_skips_selected_index_tensor(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return o[:, 0, :] + kwargs["chunk_index"]

    def selected_indices(_positions):
        raise AssertionError("all-row tokenwise O-proj must not allocate indices")

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS", raising=False)
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    monkeypatch.setattr(
        flashmla,
        "_tokenwise_o_proj_selected_indices",
        selected_indices,
    )
    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    positions = torch.tensor([646, 647])

    result = layer._o_proj(o, positions)

    assert calls == [(1, [646], 0), (1, [647], 1)]
    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0], o[1, 0] + 1]),
    )


def test_dspark_draft_keeps_batched_o_proj(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist()))
        return o[:, 0, :]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=False)
    layer.is_target_model = False
    o = torch.arange(2 * 1 * 3).reshape(2, 1, 3)

    layer._o_proj(o, torch.tensor([646, 647]))

    assert calls == [(2, [646, 647])]


def test_tokenwise_o_proj_materializes_each_shared_buffer_result(monkeypatch):
    buffer = torch.empty(1, 3)

    def native(o, positions, *args, **kwargs):
        buffer.copy_(o[:, 0, :] + kwargs["chunk_index"])
        return buffer

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)

    result = layer._o_proj(o, torch.tensor([646, 647]))

    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0], o[1, 0] + 1]),
    )


def test_k1_candidate_enables_all_row_tokenwise_o_proj(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return o[:, 0, :] + kwargs["chunk_index"]

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS", "659")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)

    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    positions = torch.tensor([658, 659])
    result = layer._o_proj(o, positions)

    assert calls == [(1, [658], 0), (1, [659], 1)]
    torch.testing.assert_close(result, torch.stack([o[0, 0], o[1, 0] + 1]))


def test_k1_native_o_proj_candidate_uses_batched_o_proj_unless_explicit(
    monkeypatch,
):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return o[:, 0, :] + o.shape[0] * 100 + kwargs["chunk_index"]

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE", "1")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    positions = torch.tensor([658, 659])
    result = layer._o_proj(o, positions)
    assert calls == [(2, [658, 659], 0)]
    torch.testing.assert_close(result, o[:, 0, :] + 200)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    calls.clear()
    result = layer._o_proj(o, positions)
    assert calls == [(1, [658], 0), (1, [659], 1)]
    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0] + 100, o[1, 0] + 101]),
    )


def test_tokenwise_o_proj_replaces_only_selected_positions(monkeypatch):
    calls = []

    def native(o, positions, *args, **kwargs):
        calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return o[:, 0, :] + o.shape[0] * 100 + kwargs["chunk_index"]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS", "659")
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj", native)
    layer = _flashmla_layer(enabled=False)
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    positions = torch.tensor([658, 659])

    result = layer._o_proj(o, positions)

    assert calls == [(2, [658, 659], 0), (1, [659], 1)]
    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0] + 200, o[1, 0] + 101]),
    )


def test_target_tokenwise_o_proj_can_coalesce_only_wo_b_reduction(monkeypatch):
    local_calls = []
    reduce_calls = []

    def row_inputs(o, positions, *args, **kwargs):
        local_calls.append((o.shape[0], positions.tolist(), kwargs["chunk_index"]))
        return [o[0:1, 0, :], o[1:2, 0, :] + 1]

    def coalesce(wo_b, row_inputs, *, group_rows):
        assert group_rows == len(row_inputs)
        reduce_calls.append((wo_b, [row.clone() for row in row_inputs]))
        return torch.cat(row_inputs, dim=0) + 100

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_COALESCE_TOKENWISE_O_PROJ_REDUCE", "1"
    )
    monkeypatch.setattr(flashmla, "deep_gemm_bf16_o_proj_row_inputs", row_inputs)
    monkeypatch.setattr(flashmla, "coalesce_wo_b_row_reductions", coalesce)
    layer = _flashmla_layer(enabled=False)
    layer.is_target_model = True
    o = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)

    with collective_census_context() as census:
        result = layer._o_proj(o, torch.tensor([646, 647]))

    assert local_calls == [(2, [646, 647], 0)]
    assert len(reduce_calls) == 1
    assert census == [
        {
            "layer_idx": layer.layer_idx,
            "projection": "target_o_proj",
            "rows": 2,
            "expects_reduce": True,
        }
    ]
    torch.testing.assert_close(
        result,
        torch.stack([o[0, 0] + 100, o[1, 0] + 101]),
    )


def test_tokenwise_sparse_decode_slices_all_token_aligned_inputs(monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(
            {
                name: None if kwargs[name] is None else kwargs[name].clone()
                for name in (
                    "q",
                    "swa_indices",
                    "topk_indices",
                    "swa_lens",
                    "topk_lens",
                    "token_to_req",
                )
            }
        )
        kwargs["out"].fill_(kwargs["q"][0, 0, 0])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE", "1")
    monkeypatch.setattr(flashmla, "sparse_mla_decode", native)
    out = torch.empty(2, 1, 3)
    token_aligned = {
        "q": torch.tensor([[[2.0, 0.0, 0.0]], [[7.0, 0.0, 0.0]]]),
        "swa_indices": torch.tensor([[[20]], [[70]]]),
        "topk_indices": torch.tensor([[[21]], [[71]]]),
        "swa_lens": torch.tensor([2, 7]),
        "topk_lens": torch.tensor([3, 8]),
        "token_to_req": torch.tensor([0, 0]),
        "out": out,
    }

    flashmla._run_sparse_mla_decode(
        **token_aligned,
        swa_cache=object(),
        compressed_cache=object(),
    )

    assert len(calls) == 2
    for index, expected in enumerate((2, 7)):
        assert calls[index]["q"].shape[0] == 1
        assert calls[index]["swa_indices"].item() == expected * 10
        assert calls[index]["topk_indices"].item() == expected * 10 + 1
        assert calls[index]["swa_lens"].item() == expected
        assert calls[index]["topk_lens"].item() == expected + 1
        assert calls[index]["token_to_req"].item() == 0
    torch.testing.assert_close(out, torch.tensor([[[2.0] * 3], [[7.0] * 3]]))


def test_tokenwise_sparse_decode_drops_empty_topk_stream(monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE", "1")
    monkeypatch.setattr(flashmla, "sparse_mla_decode", native)

    flashmla._run_sparse_mla_decode(
        q=torch.zeros(2, 1, 3),
        swa_cache=object(),
        compressed_cache=object(),
        swa_indices=torch.zeros(2, 1, 4, dtype=torch.int32),
        topk_indices=torch.full((2, 1, 8), -1, dtype=torch.int32),
        swa_lens=torch.tensor([4, 4], dtype=torch.int32),
        topk_lens=torch.tensor([0, 2], dtype=torch.int32),
        token_to_req=torch.tensor([0, 0], dtype=torch.int32),
        out=torch.empty(2, 1, 3),
        compressed_block_table=object(),
        compressed_block_size=2,
    )

    assert calls[0]["topk_indices"] is None
    assert calls[0]["topk_lens"] is None
    assert calls[0]["compressed_cache"] is None
    assert calls[0]["compressed_block_table"] is None
    assert calls[0]["compressed_block_size"] is None
    assert calls[1]["topk_indices"].shape == (1, 1, 8)
    assert calls[1]["topk_lens"].item() == 2


@pytest.mark.parametrize("topk_lens", ([2, 2], [0, 0]))
def test_grouped_sparse_decode_batches_uniform_topk_state(monkeypatch, topk_lens):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_GROUPED_ROWS", "1")
    monkeypatch.setattr(flashmla, "sparse_mla_decode", native)
    flashmla._run_sparse_mla_decode(
        q=torch.zeros(2, 1, 3),
        swa_cache=object(),
        compressed_cache=object(),
        swa_indices=torch.zeros(2, 1, 4, dtype=torch.int32),
        topk_indices=torch.ones(2, 1, 8, dtype=torch.int32),
        swa_lens=torch.tensor([4, 4], dtype=torch.int32),
        topk_lens=torch.tensor(topk_lens, dtype=torch.int32),
        token_to_req=torch.tensor([0, 0], dtype=torch.int32),
        out=torch.empty(2, 1, 3),
        compressed_block_table=object(),
        compressed_block_size=2,
    )

    assert len(calls) == 1
    assert calls[0]["q"].shape[0] == 2
    assert calls[0]["topk_indices"].shape == (2, 1, 8)
    assert calls[0]["compressed_cache"] is not None


def test_grouped_sparse_decode_keeps_mixed_topk_state_tokenwise(monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_GROUPED_ROWS", "1")
    monkeypatch.setattr(flashmla, "sparse_mla_decode", native)
    flashmla._run_sparse_mla_decode(
        q=torch.zeros(2, 1, 3),
        swa_cache=object(),
        compressed_cache=object(),
        swa_indices=torch.zeros(2, 1, 4, dtype=torch.int32),
        topk_indices=torch.ones(2, 1, 8, dtype=torch.int32),
        swa_lens=torch.tensor([4, 4], dtype=torch.int32),
        topk_lens=torch.tensor([0, 2], dtype=torch.int32),
        token_to_req=torch.tensor([0, 0], dtype=torch.int32),
        out=torch.empty(2, 1, 3),
        compressed_block_table=object(),
        compressed_block_size=2,
    )

    assert len(calls) == 1
    assert calls[0]["q"].shape[0] == 2
    assert calls[0]["topk_indices"].shape == (2, 1, 8)
    assert calls[0]["topk_lens"].tolist() == [0, 2]


def test_grouped_sparse_decode_requires_topk_lens(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_GROUPED_ROWS", "1")
    monkeypatch.setattr(flashmla, "sparse_mla_decode", lambda **_: None)

    with pytest.raises(ValueError, match="topk_indices require topk_lens"):
        flashmla._run_sparse_mla_decode(
            q=torch.zeros(2, 1, 3),
            swa_cache=object(),
            compressed_cache=object(),
            swa_indices=torch.zeros(2, 1, 4, dtype=torch.int32),
            topk_indices=torch.ones(2, 1, 8, dtype=torch.int32),
            swa_lens=torch.tensor([4, 4], dtype=torch.int32),
            topk_lens=None,
            token_to_req=torch.tensor([0, 0], dtype=torch.int32),
            out=torch.empty(2, 1, 3),
            compressed_block_table=object(),
            compressed_block_size=2,
        )


def test_short_context_rows_mask_compressed_topk_across_window_boundary():
    topk_indices = torch.tensor([[[11, 12]], [[21, 22]]], dtype=torch.int32)
    topk_lens = torch.tensor([2, 2], dtype=torch.int32)

    masked_indices, masked_lens = flashmla._mask_short_context_topk(
        torch.tensor([127, 128]),
        128,
        topk_indices,
        topk_lens,
    )

    assert torch.equal(masked_indices[0], torch.full((1, 2), -1, dtype=torch.int32))
    assert masked_lens.tolist() == [0, 2]
    torch.testing.assert_close(masked_indices[1], topk_indices[1])


def test_tokenwise_qnorm_rope_kv_insert_slices_rows_and_slots(monkeypatch):
    calls = []

    def native(q, kv, cache, slots, positions, cos_sin, eps, block_size):
        calls.append((q.shape[0], kv.shape[0], slots.tolist(), positions.tolist()))
        q.add_(positions[:, None, None])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_QKV_INSERT", "1")
    monkeypatch.setattr(attention, "_qnorm_rope_kv_insert_native", native)
    q = torch.zeros(2, 1, 3)
    kv = torch.zeros(2, 3)

    attention._run_qnorm_rope_kv_insert(
        q,
        kv,
        object(),
        torch.tensor([12, 27]),
        torch.tensor([646, 647]),
        object(),
        1e-6,
        64,
    )

    assert calls == [(1, 1, [12], [646]), (1, 1, [27], [647])]
    torch.testing.assert_close(
        q, torch.tensor([[[646.0] * 3], [[647.0] * 3]])
    )


def test_qkv_insert_capture_filters_position_and_cache_block(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("VLLM_METAX_DSV4_QKV_INSERT_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "3")
    monkeypatch.setenv("VLLM_METAX_DSV4_QKV_INSERT_CAPTURE_POSITIONS", "651")
    layer_debug.reset_layer_capture_state()
    positions = torch.tensor([650, 651])
    q = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    kv = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    cache = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    slot_mapping = torch.tensor([640, 651])

    capture = layer_debug.maybe_prepare_qkv_insert_capture(
        3,
        positions,
        q,
        kv,
        cache,
        slot_mapping,
        64,
    )
    assert capture is not None
    cache[10].add_(1000)
    path = capture.finish(cache)

    assert path == tmp_path / "rank0_layer3_call0_qkv_insert.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["positions"].tolist() == [651]
    assert payload["token_indices"].tolist() == [1]
    assert payload["slot_mapping"].tolist() == [651]
    assert payload["cache_block_indices"].tolist() == [10]
    assert payload["cache_slot_offsets"].tolist() == [11]
    torch.testing.assert_close(payload["kv"], kv[1:2])
    torch.testing.assert_close(payload["cache_before"], cache[10:11] - 1000)
    torch.testing.assert_close(payload["cache_after"], cache[10:11])


def test_sparse_mla_decode_capture_filters_position_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_LAYERS", "2")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_POSITIONS", "651")
    sparse_mla_debug.reset_sparse_mla_capture_state()

    q = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    swa_cache = torch.arange(80, dtype=torch.float32).reshape(5, 16)
    compressed_cache = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    swa_indices = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    topk_indices = torch.arange(18, dtype=torch.int32).reshape(3, 1, 6)
    swa_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
    topk_lens = torch.tensor([4, 5, 6], dtype=torch.int32)
    output = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + 100
    positions = torch.tensor([650, 651, 652], dtype=torch.int64)
    token_to_req = torch.tensor([0, 1, 1], dtype=torch.int32)

    sparse_mla_debug.maybe_capture_sparse_mla_decode(
        q=q,
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        swa_lens=swa_lens,
        topk_lens=topk_lens,
        sm_scale=0.125,
        d_v=4,
        attn_sink=None,
        output=output,
        positions=positions,
        token_to_req=token_to_req,
        swa_block_table=torch.tensor([[9, 10], [11, 12]], dtype=torch.int32),
        compressed_block_table=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        swa_block_size=64,
        compressed_block_size=16,
        compress_ratio=4,
        window_size=512,
        layer_idx=2,
    )

    [capture_path] = sorted(tmp_path.glob("rank*_call0.pt"))
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert payload["positions"].tolist() == [651]
    assert payload["token_indices"].tolist() == [1]
    assert payload["token_to_req"].tolist() == [1]
    torch.testing.assert_close(payload["q"], q[1:2])
    torch.testing.assert_close(payload["swa_indices"], swa_indices[1:2])
    torch.testing.assert_close(payload["topk_indices"], topk_indices[1:2])
    torch.testing.assert_close(payload["swa_lens"], swa_lens[1:2])
    torch.testing.assert_close(payload["topk_lens"], topk_lens[1:2])
    torch.testing.assert_close(payload["output"], output[1:2])


def test_tokenwise_wq_b_splits_indexer_q_quant(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS", "target")
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda left, right, *_args, **_kwargs: (left(), right()),
    )
    quant_calls = []

    def fake_quant(positions, q, _cache, weights, *_args):
        quant_calls.append(
            (positions.tolist(), tuple(q.shape), weights.flatten().tolist())
        )
        return q.to(torch.int8), weights

    monkeypatch.setattr(attention, "fused_indexer_q_rope_int8_quant", fake_quant)
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.prefix = "model.layers.2.attn.indexer"
    indexer.layer_idx = 2
    indexer.config = type(
        "Config", (), {"sliding_window": 512, "num_hidden_layers": 61}
    )()
    indexer.compressor = lambda *_args: torch.empty(2, 1)
    indexer.compressor.overlap = False
    indexer.wq_b = lambda qr: (qr + 1, None)
    indexer.n_head = 1
    indexer.head_dim = 1
    indexer.softmax_scale = 1.0
    indexer.ln_events = [None, None]
    indexer.aux_stream = None

    def fake_indexer_op(_hidden_states, q_quant, _k, weights):
        torch.testing.assert_close(q_quant, torch.tensor([[[2]], [[3]]], dtype=torch.int8))
        torch.testing.assert_close(weights, torch.tensor([[10.0], [20.0]]))
        return q_quant

    indexer.indexer_op = fake_indexer_op
    result = indexer.forward(
        hidden_states=torch.zeros(2, 1),
        qr=torch.tensor([[1.0], [2.0]]),
        compressed_kv_score=torch.zeros(2, 1),
        indexer_weights=torch.tensor([[10.0], [20.0]]),
        positions=torch.tensor([650, 651], dtype=torch.int64),
        rotary_emb=type("Rotary", (), {"cos_sin_cache": torch.empty(1)})(),
    )

    assert quant_calls == [
        ([650], (1, 1, 1), [10.0]),
        ([651], (1, 1, 1), [20.0]),
    ]
    torch.testing.assert_close(result, torch.tensor([[[2]], [[3]]], dtype=torch.int8))


def test_exact_grouped_indexer_wq_b_batches_projection_and_quant(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS", "target")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_INDEXER_WQ_B_ROWS", "1")
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda left, right, *_args, **_kwargs: (left(), right()),
    )

    grouped_calls = []

    def fake_grouped(qr, weight, output):
        grouped_calls.append((tuple(qr.shape), tuple(weight.shape), output.data_ptr()))
        output.copy_(qr[:, :1].expand_as(output))

    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_grouped_rows_out",
        fake_grouped,
        raising=False,
    )
    quant_calls = []

    def fake_quant(positions, q, _cache, weights, *_args):
        quant_calls.append(
            (positions.tolist(), tuple(q.shape), weights.flatten().tolist())
        )
        return q.to(torch.int8), weights

    monkeypatch.setattr(attention, "fused_indexer_q_rope_int8_quant", fake_quant)
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.prefix = "model.layers.2.attn.indexer"
    indexer.layer_idx = 2
    indexer.config = type(
        "Config", (), {"sliding_window": 512, "num_hidden_layers": 43}
    )()
    indexer.compressor = lambda *_args: torch.empty(2, 1)
    indexer.compressor.overlap = False
    indexer.q_lora_rank = 1024
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = type(
        "Linear",
        (),
        {"weight": torch.ones(8192, 1024, dtype=torch.bfloat16)},
    )()
    indexer.softmax_scale = 1.0
    indexer.ln_events = [None, None]
    indexer.aux_stream = None
    indexer.indexer_op = lambda _hidden, q, _k, _weights: q

    qr = torch.stack(
        [
            torch.full((1024,), 1.0, dtype=torch.bfloat16),
            torch.full((1024,), 2.0, dtype=torch.bfloat16),
        ]
    )
    kwargs = {
        "hidden_states": torch.zeros(2, 1),
        "qr": qr,
        "compressed_kv_score": torch.zeros(2, 1),
        "indexer_weights": torch.tensor(
            [[10.0] * 64, [20.0] * 64], dtype=torch.bfloat16
        ),
        "positions": torch.tensor([650, 651], dtype=torch.int64),
        "rotary_emb": type("Rotary", (), {"cos_sin_cache": torch.empty(1)})(),
    }
    first = indexer.forward(**kwargs)
    second = indexer.forward(**kwargs)

    assert len(grouped_calls) == 2
    assert grouped_calls[0][:2] == ((2, 1024), (8192, 1024))
    assert grouped_calls[0][2] == grouped_calls[1][2]
    assert quant_calls == [
        ([650, 651], (2, 64, 128), [10.0] * 64 + [20.0] * 64),
        ([650, 651], (2, 64, 128), [10.0] * 64 + [20.0] * 64),
    ]
    torch.testing.assert_close(first[:, 0, 0], torch.tensor([1, 2], dtype=torch.int8))
    torch.testing.assert_close(second, first)


def test_exact_grouped_indexer_wq_b_fails_closed_on_invalid_shape(monkeypatch):
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.q_lora_rank = 1024
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = type(
        "Linear",
        (),
        {"weight": torch.ones(8192, 1024, dtype=torch.bfloat16)},
    )()

    with pytest.raises(RuntimeError, match="contiguous BF16 input"):
        indexer._project_wq_b_exact_grouped_rows(
            torch.ones(2, 1023, dtype=torch.bfloat16)
        )


def test_exact_grouped_indexer_wq_b_falls_back_on_misaligned_weight_rows(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS", "target")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_INDEXER_WQ_B_ROWS", "1")
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.prefix = "model.layers.2.attn.indexer"
    indexer.layer_idx = 2
    indexer.config = type(
        "Config", (), {"sliding_window": 512, "num_hidden_layers": 43}
    )()
    indexer.compressor = lambda *_args: torch.empty(2, 1)
    indexer.compressor.overlap = False
    indexer.q_lora_rank = 1024
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = type(
        "Linear",
        (),
        {"weight": torch.ones(8192, 1024, dtype=torch.bfloat16)},
    )()
    indexer.softmax_scale = 1.0
    indexer.ln_events = [None, None]
    indexer.aux_stream = None
    project_calls = []
    quant_calls = []

    def project(qr_chunk):
        project_calls.append(tuple(qr_chunk.shape))
        return torch.ones(
            qr_chunk.shape[0], 8192, dtype=torch.bfloat16
        ), None

    def fake_quant(positions, q, _cache, weights, *_args):
        quant_calls.append((positions.tolist(), tuple(q.shape), tuple(weights.shape)))
        return q.to(torch.int8), weights

    indexer.wq_b = project
    indexer.indexer_op = lambda _hidden, q, _k, _weights: q
    monkeypatch.setattr(attention, "fused_indexer_q_rope_int8_quant", fake_quant)

    indexer.forward(
        hidden_states=torch.zeros(2, 1),
        qr=torch.ones(2, 1024, dtype=torch.bfloat16),
        compressed_kv_score=torch.zeros(2, 1),
        indexer_weights=torch.ones(1, 64),
        positions=torch.tensor([650, 651], dtype=torch.int64),
        rotary_emb=type("Rotary", (), {"cos_sin_cache": torch.empty(1)})(),
    )

    assert project_calls == [(1, 1024), (1, 1024)]
    assert quant_calls == [
        ([650], (1, 64, 128), (1, 64)),
        ([651], (1, 64, 128), (1, 64)),
    ]


def test_exact_grouped_indexer_wq_b_accepts_noncontiguous_metadata(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS", "target")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_INDEXER_WQ_B_ROWS", "1")
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda left, right, *_args, **_kwargs: (left(), right()),
    )
    grouped_calls = []

    def fake_grouped(qr, weight, output):
        grouped_calls.append((tuple(qr.shape), tuple(weight.shape)))
        output.fill_(1)

    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_grouped_rows_out",
        fake_grouped,
        raising=False,
    )
    quant_calls = []

    def fake_quant(positions, q, _cache, weights, *_args):
        quant_calls.append(
            (
                positions.is_contiguous(),
                weights.is_contiguous(),
                positions.tolist(),
                tuple(weights.shape),
            )
        )
        return q.to(torch.int8), weights

    monkeypatch.setattr(attention, "fused_indexer_q_rope_int8_quant", fake_quant)
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.prefix = "model.layers.2.attn.indexer"
    indexer.layer_idx = 2
    indexer.config = type(
        "Config", (), {"sliding_window": 512, "num_hidden_layers": 43}
    )()
    indexer.compressor = lambda *_args: torch.empty(2, 1)
    indexer.compressor.overlap = False
    indexer.q_lora_rank = 1024
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = type(
        "Linear",
        (),
        {"weight": torch.ones(8192, 1024, dtype=torch.bfloat16)},
    )()
    indexer.softmax_scale = 1.0
    indexer.ln_events = [None, None]
    indexer.aux_stream = None
    indexer.indexer_op = lambda _hidden, q, _k, _weights: q

    positions_base = torch.tensor([0, 650, 0, 651], dtype=torch.int64)
    weights_base = torch.ones(2, 128, dtype=torch.bfloat16)
    indexer.forward(
        hidden_states=torch.zeros(2, 1),
        qr=torch.ones(2, 1024, dtype=torch.bfloat16),
        compressed_kv_score=torch.zeros(2, 1),
        indexer_weights=weights_base[:, ::2],
        positions=positions_base[1::2],
        rotary_emb=type("Rotary", (), {"cos_sin_cache": torch.empty(1)})(),
    )

    assert grouped_calls == [((2, 1024), (8192, 1024))]
    assert quant_calls == [(True, True, [650, 651], (2, 64))]


def test_short_context_defers_initial_overlap_clear(monkeypatch):
    prefix = "model.layers.2.attn.indexer"
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type(
            "Context",
            (),
            {
                "attn_metadata": {
                    prefix.replace(".indexer", ".swa_cache"): type(
                        "Metadata",
                        (),
                        {"is_short_context": lambda _self, _window: True},
                    )()
                }
            },
        )(),
    )
    indexer = object.__new__(attention.MacaDeepseekV4Indexer)
    indexer.prefix = prefix
    indexer.config = type("Config", (), {"sliding_window": 128})()
    indexer.topk_indices_buffer = torch.zeros(2, 4, dtype=torch.int32)
    indexer._short_context_pending = False
    indexer.compressor = object()
    indexer.wq_b = lambda _qr: (_ for _ in ()).throw(
        AssertionError("short context must skip indexer Q projection")
    )
    indexer.indexer_op = lambda *_args: (_ for _ in ()).throw(
        AssertionError("short context must skip indexer logits")
    )

    hidden_states = torch.zeros(2, 1)
    compressed_kv_score = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    positions = torch.tensor([124, 125], dtype=torch.int64)
    rotary_emb = type("Rotary", (), {"cos_sin_cache": torch.empty(1)})()
    result = indexer.forward(
        hidden_states=hidden_states,
        qr=torch.zeros(2, 1),
        compressed_kv_score=compressed_kv_score,
        indexer_weights=torch.zeros(2, 1),
        positions=positions,
        rotary_emb=rotary_emb,
    )

    assert indexer._short_context_pending is True
    assert torch.equal(result, torch.full((2, 4), -1, dtype=torch.int32))


def test_tokenwise_wq_b_target_and_indexer_envs_are_independent(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", "1")
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B", raising=False)
    assert attention._target_tokenwise_wq_b_enabled()
    assert not attention._indexer_tokenwise_wq_b_enabled(2)
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B", "1")
    assert not attention._target_tokenwise_wq_b_enabled()
    assert attention._indexer_tokenwise_wq_b_enabled(2)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B_LAYERS", "2,4")
    assert attention._indexer_tokenwise_wq_b_enabled(2)
    assert not attention._indexer_tokenwise_wq_b_enabled(3)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", "1")
    assert attention._target_tokenwise_wq_b_enabled()
    assert attention._indexer_tokenwise_wq_b_enabled(4)


def test_tokenwise_indexer_weight_rows_target_scope(monkeypatch):
    env = "VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS"
    monkeypatch.setenv(env, "target")
    assert attention._indexer_tokenwise_weight_rows_enabled(60, 61)
    assert not attention._indexer_tokenwise_weight_rows_enabled(61, 61)

    monkeypatch.setenv(env, "invalid")
    with pytest.raises(ValueError, match="must be off or target"):
        attention._indexer_tokenwise_weight_rows_enabled(2, 61)


def test_k1_candidate_enables_target_wq_b_and_kv_prenorm(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")

    assert attention._target_tokenwise_wq_b_enabled()
    assert attention._target_tokenwise_kv_prenorm_enabled()


def test_k1_native_wq_b_candidate_uses_batched_wq_b_unless_explicit(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE", "1")
    assert not attention._target_tokenwise_wq_b_enabled()
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", "1")
    assert attention._target_tokenwise_wq_b_enabled()


def test_k1_native_wq_b_candidate_layer_selector(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_WQ_B", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS", "2,4")

    assert attention._target_tokenwise_wq_b_enabled()
    assert attention._target_tokenwise_wq_b_layer_enabled(1)
    assert not attention._target_tokenwise_wq_b_layer_enabled(2)
    assert not attention._target_tokenwise_wq_b_layer_enabled(4)
    assert attention._target_tokenwise_wq_b_layer_enabled(5)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", "1")
    assert attention._target_tokenwise_wq_b_layer_enabled(2)

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS", "all")
    assert not attention._target_tokenwise_wq_b_enabled()

    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS", "-1")
    with pytest.raises(ValueError, match="MTP_K1_NATIVE_WQ_B_LAYERS"):
        attention._target_tokenwise_wq_b_enabled()


def test_k1_native_kv_prenorm_candidate_uses_batched_unless_explicit(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_KV_PRENORM_CANDIDATE", "1")
    assert not attention._target_tokenwise_kv_prenorm_enabled()
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", "1")
    assert attention._target_tokenwise_kv_prenorm_enabled()


def test_k1_candidate_target_selectors_are_all_rows(monkeypatch):
    positions = torch.tensor([658, 659])
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", "9")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "999")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "9")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "999")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_CALLS", "9")

    assert attention._target_tokenwise_wq_b_layer_enabled(0)
    assert attention._target_tokenwise_wq_b_position_enabled(positions)
    torch.testing.assert_close(
        attention._target_tokenwise_wq_b_selected_indices(positions),
        torch.tensor([0, 1]),
    )
    assert attention._target_tokenwise_qkv_layer_enabled(0)
    assert attention._target_tokenwise_qkv_position_enabled(positions)
    assert attention._target_tokenwise_qkv_call_enabled(0)
    torch.testing.assert_close(
        attention._target_tokenwise_qkv_selected_indices(positions),
        torch.tensor([0, 1]),
    )


def test_k1_candidate_can_explicitly_respect_target_qkv_scope(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_MTP_K1_RESPECT_TARGET_QKV_SCOPE", "1"
    )
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "35")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_CALLS", "2")
    positions = torch.tensor([34, 35])

    assert attention._target_tokenwise_qkv_layer_enabled(0)
    assert not attention._target_tokenwise_qkv_layer_enabled(1)
    assert attention._target_tokenwise_qkv_position_enabled(positions)
    assert not attention._target_tokenwise_qkv_position_enabled(
        torch.tensor([34, 36])
    )
    assert attention._target_tokenwise_qkv_call_enabled(2)
    assert not attention._target_tokenwise_qkv_call_enabled(1)
    torch.testing.assert_close(
        attention._target_tokenwise_qkv_selected_indices(positions),
        torch.tensor([1]),
    )


def test_tokenwise_target_wq_b_layer_selector(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", raising=False)
    assert attention._target_tokenwise_wq_b_layer_enabled(7)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", "all")
    assert attention._target_tokenwise_wq_b_layer_enabled(7)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", " 0, 7, 42 ")
    assert attention._target_tokenwise_wq_b_layer_enabled(7)
    assert not attention._target_tokenwise_wq_b_layer_enabled(8)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", "-1")
    with pytest.raises(ValueError, match="TOKENWISE_TARGET_WQ_B_LAYERS"):
        attention._target_tokenwise_wq_b_layer_enabled(0)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", "bad")
    with pytest.raises(ValueError, match="TOKENWISE_TARGET_WQ_B_LAYERS"):
        attention._target_tokenwise_wq_b_layer_enabled(0)


def test_tokenwise_target_wq_b_position_selector(monkeypatch):
    positions = torch.tensor([766, 767])
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", raising=False
    )
    assert attention._target_tokenwise_wq_b_position_enabled(positions)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "all")
    assert attention._target_tokenwise_wq_b_position_enabled(positions)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "647,767")
    assert attention._target_tokenwise_wq_b_position_enabled(positions)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "647")
    assert not attention._target_tokenwise_wq_b_position_enabled(positions)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "bad")
    with pytest.raises(ValueError, match="TOKENWISE_TARGET_WQ_B_POSITIONS"):
        attention._target_tokenwise_wq_b_position_enabled(positions)


def test_tokenwise_target_qkv_selectors(monkeypatch):
    positions = torch.tensor([766, 767])
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV", raising=False)
    assert not attention._target_tokenwise_qkv_enabled()
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV", "1")
    assert attention._target_tokenwise_qkv_enabled()
    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", raising=False)
    assert attention._target_tokenwise_qkv_layer_enabled(7)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0,7")
    assert attention._target_tokenwise_qkv_layer_enabled(7)
    assert not attention._target_tokenwise_qkv_layer_enabled(8)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "767")
    assert attention._target_tokenwise_qkv_position_enabled(positions)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "765")
    assert not attention._target_tokenwise_qkv_position_enabled(positions)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "bad")
    with pytest.raises(ValueError, match="TOKENWISE_TARGET_QKV_LAYERS"):
        attention._target_tokenwise_qkv_layer_enabled(0)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "bad")
    with pytest.raises(ValueError, match="TOKENWISE_TARGET_QKV_POSITIONS"):
        attention._target_tokenwise_qkv_position_enabled(positions)


def test_prefill_gemm_chunking_disabled_or_small_delegates_once(monkeypatch):
    calls = []

    def upstream(self, hidden_states):
        calls.append(hidden_states.shape[0])
        return (hidden_states, None, None, None)

    monkeypatch.setattr(attention.DeepseekV4Attention,
                        "attn_gemm_parallel_execute", upstream)
    hidden_states = torch.arange(8).reshape(4, 2)

    assert _layer(enabled=False).attn_gemm_parallel_execute(hidden_states)[0] is hidden_states
    assert _layer(enabled=True, chunk_size=4).attn_gemm_parallel_execute(hidden_states)[0] is hidden_states
    assert calls == [4, 4]


def test_tokenwise_q_only_preserves_batched_kv_and_auxiliary_outputs(monkeypatch):
    upstream_calls = []
    tokenwise_calls = []
    batched_qr_kv = torch.arange(20).reshape(4, 5)
    auxiliary = torch.arange(4).reshape(4, 1)

    def upstream(self, hidden_states):
        upstream_calls.append(hidden_states.shape[0])
        return (batched_qr_kv, auxiliary, None, auxiliary + 10)

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        value = hidden_states[:, :1]
        return torch.cat(
            [value.repeat(1, 3) + 100, value.repeat(1, 2) + 900], dim=-1
        ), None

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_Q_ONLY", "1")
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    layer = _layer(enabled=False)
    layer.q_lora_rank = 3
    layer.fused_wqa_wkv = fused
    hidden_states = torch.arange(8).reshape(4, 2)

    result = layer.attn_gemm_parallel_execute(hidden_states)

    expected_q = hidden_states[:, :1].repeat(1, 3) + 100
    torch.testing.assert_close(result[0][:, :3], expected_q)
    torch.testing.assert_close(result[0][:, 3:], batched_qr_kv[:, 3:])
    assert result[1] is auxiliary
    assert result[2] is None
    torch.testing.assert_close(result[3], auxiliary + 10)
    assert upstream_calls == [4]
    assert tokenwise_calls == [1, 1, 1, 1]


def test_tokenwise_q_only_supports_six_row_dspark_verifier(monkeypatch):
    tokenwise_calls = []
    batched_qr_kv = torch.arange(30).reshape(6, 5)

    def upstream(self, hidden_states):
        return (batched_qr_kv, None, None, None)

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5), None

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_Q_ONLY", "1")
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    layer = _layer(enabled=False)
    layer.q_lora_rank = 3
    layer.fused_wqa_wkv = fused

    result = layer.attn_gemm_parallel_execute(torch.arange(12).reshape(6, 2))

    assert result[0].shape == (6, 5)
    assert tokenwise_calls == [1] * 6


def test_tokenwise_qkv_materializes_rows_and_preserves_auxiliary_outputs(
    monkeypatch,
):
    auxiliary = torch.arange(2).reshape(2, 1)
    shared_buffer = torch.empty(1, 5)
    tokenwise_calls = []

    def upstream(self, hidden_states):
        return (torch.full((2, 5), -1.0), auxiliary, None, auxiliary + 10)

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        shared_buffer.copy_(hidden_states[:, :1].repeat(1, 5))
        return shared_buffer, None

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_QKV", "1")
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    layer = _layer(enabled=False)
    layer.q_lora_rank = 3
    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])

    result = layer.attn_gemm_parallel_execute(hidden_states)

    torch.testing.assert_close(
        result[0], hidden_states[:, :1].repeat(1, 5)
    )
    assert result[1] is auxiliary
    assert result[2] is None
    torch.testing.assert_close(result[3], auxiliary + 10)
    assert tokenwise_calls == [1, 1]


def test_tokenwise_attn_gemm_aux_replaces_only_selected_components(monkeypatch):
    calls = []
    batched_qr_kv = torch.full((4, 2), -1.0)
    batched_kv_score = torch.full((4, 1), -2.0)
    batched_indexer_kv_score = torch.full((4, 1), -3.0)
    batched_indexer_weights = torch.full((4, 1), -4.0)

    def upstream(self, hidden_states):
        calls.append(hidden_states.shape[0])
        if hidden_states.shape[0] == 4:
            return (
                batched_qr_kv,
                batched_kv_score,
                batched_indexer_kv_score,
                batched_indexer_weights,
            )
        value = hidden_states[:, :1].float()
        return (
            value + 10,
            value + 100,
            value + 1000,
            value + 10000,
        )

    monkeypatch.setenv(
        "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX",
        "indexer_kv_score,indexer_weights",
    )
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    result = _layer(enabled=False).attn_gemm_parallel_execute(hidden_states)

    assert result[0] is batched_qr_kv
    assert result[1] is batched_kv_score
    torch.testing.assert_close(result[2], hidden_states[:, :1] + 1000)
    torch.testing.assert_close(result[3], hidden_states[:, :1] + 10000)
    assert calls == [4, 1, 1, 1, 1]


def test_tokenwise_attn_gemm_aux_preserves_selected_none_component(monkeypatch):
    calls = []
    batched_qr_kv = torch.full((2, 2), -1.0)
    batched_kv_score = torch.full((2, 1), -2.0)
    batched_indexer_weights = torch.full((2, 1), -4.0)

    def upstream(self, hidden_states):
        calls.append(hidden_states.shape[0])
        if hidden_states.shape[0] == 2:
            return (
                batched_qr_kv,
                batched_kv_score,
                None,
                batched_indexer_weights,
            )
        value = hidden_states[:, :1].float()
        return (value + 10, value + 100, None, value + 10000)

    monkeypatch.setenv(
        "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX",
        "kv_score,indexer_kv_score",
    )
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    result = _layer(enabled=False).attn_gemm_parallel_execute(hidden_states)

    assert result[0] is batched_qr_kv
    torch.testing.assert_close(result[1], hidden_states[:, :1] + 100)
    assert result[2] is None
    assert result[3] is batched_indexer_weights
    assert calls == [2, 1, 1]


def test_tokenwise_attn_gemm_aux_rejects_invalid_component(monkeypatch):
    def upstream(self, hidden_states):
        return (hidden_states, hidden_states[:, :1], None, hidden_states[:, :1])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX", "bad")
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )

    with pytest.raises(
        ValueError, match="VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX"
    ):
        _layer(enabled=False).attn_gemm_parallel_execute(
            torch.zeros(2, 2)
        )


def test_tokenwise_qkv_combines_with_tokenwise_aux(monkeypatch):
    upstream_calls = []
    tokenwise_qkv_calls = []
    batched_qr_kv = torch.full((2, 2), -1.0)
    batched_kv_score = torch.full((2, 1), -2.0)
    batched_indexer_kv_score = torch.full((2, 1), -3.0)
    batched_indexer_weights = torch.full((2, 1), -4.0)

    def upstream(self, hidden_states):
        upstream_calls.append(hidden_states.shape[0])
        if hidden_states.shape[0] == 2:
            return (
                batched_qr_kv,
                batched_kv_score,
                batched_indexer_kv_score,
                batched_indexer_weights,
            )
        value = hidden_states[:, :1].float()
        return (value + 10, value + 100, value + 1000, value + 10000)

    def fused(hidden_states):
        tokenwise_qkv_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 2) + 900, None

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_QKV", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX", "kv_score")
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        upstream,
    )
    layer = _layer(enabled=False)
    layer.fused_wqa_wkv = fused
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    result = layer.attn_gemm_parallel_execute(hidden_states)

    torch.testing.assert_close(result[0], hidden_states[:, :1].repeat(1, 2) + 900)
    torch.testing.assert_close(result[1], hidden_states[:, :1] + 100)
    assert result[2] is batched_indexer_kv_score
    assert result[3] is batched_indexer_weights
    assert upstream_calls == [2, 1, 1]
    assert tokenwise_qkv_calls == [1, 1]


def test_tokenwise_qkv_kv_score_uses_native_serial_attention_ops(monkeypatch):
    execute_calls = []
    qkv_native_calls = []
    kv_score_native_calls = []
    indexer_weight_calls = []
    indexer_kv_score_calls = []

    def fake_execute(
        main_fn,
        aux_fns,
        _start_event,
        _done_events,
        _streams,
        enable,
    ):
        execute_calls.append(enable)
        return main_fn(), [
            fn() if fn is not None else None
            for fn in aux_fns
        ]

    def qkv_native(input_, weight, output):
        qkv_native_calls.append((input_.shape, weight.shape, output.data_ptr()))
        output.copy_(input_[:, :1].expand_as(output))

    def kv_score_native(input_, weight, output):
        kv_score_native_calls.append(
            (input_.shape, weight.shape, output.dtype, output.data_ptr())
        )
        output.copy_(input_[:, :1].float().expand_as(output))

    def fake_mm(input_, weight_t, *, out_dtype=None):
        indexer_kv_score_calls.append((input_.shape, weight_t.shape, out_dtype))
        return input_[:, :1].float().expand(input_.shape[0], weight_t.shape[1])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_QKV", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX", "kv_score")
    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_ATTN_GEMM_ROWS", "1")
    monkeypatch.setattr(attention, "execute_in_parallel", fake_execute)
    monkeypatch.setattr(attention.torch, "mm", fake_mm)
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_serial_rows_out",
        qkv_native,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_fp32_serial_rows_out",
        kv_score_native,
        raising=False,
    )

    layer = _layer(enabled=False)
    layer.q_lora_rank = 3
    layer.head_dim = 2
    layer.aux_stream_list = None
    layer.ln_events = [None, None, None, None]
    layer.fused_wqa_wkv = SimpleNamespace(
        weight=torch.ones(5, 2, dtype=torch.bfloat16)
    )
    layer.compressor = SimpleNamespace(
        fused_wkv_wgate=SimpleNamespace(
            weight=torch.ones(4, 2, dtype=torch.bfloat16)
        )
    )

    def weights_proj(hidden_states):
        indexer_weight_calls.append(hidden_states.shape)
        return hidden_states[:, :1].float() + 100, None

    layer.indexer = SimpleNamespace(
        weights_proj=weights_proj,
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(
                weight=torch.ones(6, 2, dtype=torch.bfloat16)
            )
        ),
    )
    hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)

    result = layer.attn_gemm_parallel_execute(hidden_states)

    torch.testing.assert_close(result[0], hidden_states[:, :1].expand(4, 5))
    torch.testing.assert_close(
        result[1], hidden_states[:, :1].float().expand(4, 4)
    )
    torch.testing.assert_close(
        result[2], hidden_states[:, :1].float().expand(4, 6)
    )
    torch.testing.assert_close(result[3], hidden_states[:, :1].float() + 100)
    assert execute_calls == [False]
    assert len(qkv_native_calls) == 1
    assert len(kv_score_native_calls) == 1
    assert indexer_weight_calls == [torch.Size([4, 2])]
    assert indexer_kv_score_calls == [
        (torch.Size([4, 2]), torch.Size([2, 6]), torch.float32)
    ]


def test_target_tokenwise_qkv_matches_normalized_attention_boundary():
    layer = _layer(enabled=False)
    layer.q_lora_rank = 3
    layer.head_dim = 2
    layer.q_norm = _AddNorm(100)
    layer.kv_norm = _AddNorm(900)
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5), None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])

    qr, kv = layer._project_target_qkv_tokenwise(hidden_states)

    torch.testing.assert_close(qr, hidden_states[:, :1].repeat(1, 3) + 100)
    torch.testing.assert_close(kv, hidden_states[:, :1].repeat(1, 2) + 900)
    assert qr.is_contiguous()
    assert kv.is_contiguous()
    assert tokenwise_calls == [1, 1]


def test_target_prenorm_tokenwise_qkv_replaces_selected_small_batch(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.q_lora_rank = 3
    layer.head_dim = 2
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5) + 100, None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    batched_qr_kv = torch.full((2, 5), -1.0)
    positions = torch.tensor([766, 767])

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_PRENORM", raising=False)
    assert (
        layer._replace_target_fused_qkv_prenorm(
            hidden_states, positions, batched_qr_kv
        )
        is batched_qr_kv
    )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_PRENORM", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "all")

    result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )

    torch.testing.assert_close(result, hidden_states[:, :1].repeat(1, 5) + 100)
    assert result.is_contiguous()
    assert tokenwise_calls == [1, 1]


def test_k1_scoped_target_qkv_uses_graph_safe_position_mask(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.q_lora_rank = 3
    layer.head_dim = 2
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5) + 100, None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    batched_qr_kv = torch.full((2, 5), -1.0)
    positions = torch.tensor([34, 35])
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_MTP_K1_RESPECT_TARGET_QKV_SCOPE", "1"
    )
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_PRENORM", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "35")
    monkeypatch.setattr(
        attention.torch,
        "nonzero",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scoped target QKV must not call torch.nonzero")
        ),
    )

    result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )

    torch.testing.assert_close(result[0], batched_qr_kv[0])
    torch.testing.assert_close(result[1], torch.full((5,), 107.0))
    assert result.is_contiguous()
    assert tokenwise_calls == [1, 1]
    assert tokenwise_calls == [1, 1]


def test_target_prenorm_tokenwise_kv_preserves_batched_q(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.q_lora_rank = 3
    layer.head_dim = 2
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5) + 100, None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    batched_qr_kv = torch.full((2, 5), -1.0)
    positions = torch.tensor([766, 767])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "all")

    result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )

    torch.testing.assert_close(result[:, :3], batched_qr_kv[:, :3])
    torch.testing.assert_close(
        result[:, 3:], hidden_states[:, :1].repeat(1, 2) + 100
    )
    assert result.is_contiguous()
    assert tokenwise_calls == [1, 1]


def test_target_prenorm_tokenwise_kv_replaces_only_selected_positions(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.q_lora_rank = 3
    layer.head_dim = 2
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5) + 100, None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    batched_qr_kv = torch.full((2, 5), -1.0)
    positions = torch.tensor([766, 767])

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "767")

    result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )

    torch.testing.assert_close(result[0], batched_qr_kv[0])
    torch.testing.assert_close(result[1, :3], batched_qr_kv[1, :3])
    torch.testing.assert_close(
        result[1, 3:], hidden_states[1:2, :1].repeat(1, 2)[0] + 100
    )
    assert result.is_contiguous()
    assert tokenwise_calls == [1]


def test_target_prenorm_tokenwise_kv_can_select_call_and_position(monkeypatch):
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.q_lora_rank = 3
    layer.head_dim = 2
    tokenwise_calls = []

    def fused(hidden_states):
        tokenwise_calls.append(hidden_states.shape[0])
        return hidden_states[:, :1].repeat(1, 5) + 100, None

    layer.fused_wqa_wkv = fused
    hidden_states = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    positions = torch.tensor([673, 674])
    batched_qr_kv = torch.full((2, 5), -1.0)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS", "674")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_CALLS", "1")

    first_result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )
    second_result = layer._replace_target_fused_qkv_prenorm(
        hidden_states, positions, batched_qr_kv
    )

    assert first_result is batched_qr_kv
    torch.testing.assert_close(second_result[0], batched_qr_kv[0])
    torch.testing.assert_close(second_result[1, :3], batched_qr_kv[1, :3])
    torch.testing.assert_close(
        second_result[1, 3:], hidden_states[1:2, :1].repeat(1, 2)[0] + 100
    )
    assert second_result.is_contiguous()
    assert tokenwise_calls == [1]


def test_tokenwise_wq_b_uses_one_native_call_per_row():
    layer = _layer(enabled=False)
    layer.wq_b = _ShapeSensitiveWqB()
    qr = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    result = layer._project_wq_b_tokenwise(qr)

    assert layer.wq_b.calls == [1, 1, 1, 1]
    torch.testing.assert_close(result, qr + 1)


def test_tokenwise_wq_b_native_serial_rows_reuses_output(monkeypatch):
    class NativeWqB:
        def __init__(self):
            self.weight = torch.ones(3, 2, dtype=torch.bfloat16)
            self.calls = 0

        def __call__(self, _qr):
            self.calls += 1
            raise AssertionError("Python row path must not run")

    native_calls = []

    def native_op(qr, weight, output):
        native_calls.append((qr.shape, weight.shape, output.data_ptr()))
        output.copy_(qr[:, :1].expand_as(output))

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_WQ_B_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_serial_rows_out",
        native_op,
        raising=False,
    )
    layer = _layer(enabled=False)
    layer.wq_b = NativeWqB()
    qr = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)

    first = layer._project_wq_b_tokenwise(qr)
    second = layer._project_wq_b_tokenwise(qr)

    assert first.data_ptr() == second.data_ptr()
    assert layer.wq_b.calls == 0
    assert native_calls == [
        (torch.Size([6, 2]), torch.Size([3, 2]), first.data_ptr()),
        (torch.Size([6, 2]), torch.Size([3, 2]), first.data_ptr()),
    ]
    torch.testing.assert_close(second, qr[:, :1].expand(6, 3))


def test_tokenwise_wq_b_exact_grouped_rows_selects_grouped_op(monkeypatch):
    class NativeWqB:
        def __init__(self):
            self.weight = torch.ones(3, 2, dtype=torch.bfloat16)

        def __call__(self, _qr):
            raise AssertionError("Python row path must not run")

    grouped_calls = []

    def grouped_op(qr, weight, output):
        grouped_calls.append((qr.shape, weight.shape))
        output.copy_(qr[:, :1].expand_as(output))

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_WQ_B_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_WQ_B_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_grouped_rows_out",
        grouped_op,
        raising=False,
    )
    layer = _layer(enabled=False)
    layer.wq_b = NativeWqB()
    qr = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)

    result = layer._project_wq_b_tokenwise(qr)

    assert grouped_calls == [(torch.Size([6, 2]), torch.Size([3, 2]))]
    torch.testing.assert_close(result, qr[:, :1].expand(6, 3))


def test_tokenwise_wq_b_materializes_each_shared_buffer_result():
    layer = _layer(enabled=False)
    layer.wq_b = _SharedBufferWqB()
    qr = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    result = layer._project_wq_b_tokenwise(qr)

    torch.testing.assert_close(
        result,
        torch.tensor([[1.0, 2.0], [4.0, 5.0]]),
    )


def test_wq_b_shadow_compare_keeps_batched_q_live(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_WQ_B_SHADOW_COMPARE_DIR", str(tmp_path))
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    layer = _layer(enabled=False)
    layer.is_target_model = True
    layer.layer_idx = 0
    layer.indexer = None
    layer.compressor = None
    layer.n_local_heads = 1
    layer.head_dim = 1
    layer.wq_b = _ShapeSensitiveWqB()
    observed = {}

    def fused(q, _kv, _positions, _metadata):
        observed["q"] = q.detach().clone()
        return q

    layer._fused_qnorm_rope_kv_insert = fused
    layer.forward_mqa = lambda q, _kv, _positions, output: output.copy_(q)
    out = torch.empty(2, 1, 1)

    layer.attention_impl(
        torch.zeros(2, 1),
        torch.tensor([[1.0], [2.0]]),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([767, 768]),
        out,
    )

    assert layer.wq_b.calls == [1, 1, 2]
    torch.testing.assert_close(
        observed["q"], torch.tensor([[[3.0]], [[4.0]]])
    )
    torch.testing.assert_close(out, observed["q"])


def test_qkv_prenorm_shadow_compare_keeps_batched_qkv_live(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_QKV_PRENORM_SHADOW_COMPARE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_POSITIONS", "674")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attn_gemm_parallel_execute",
        lambda self, hidden_states: (
            self.fused_wqa_wkv(hidden_states)[0],
            None,
            None,
            None,
        ),
    )
    layer_debug.reset_layer_capture_state()
    layer = _layer(enabled=False)
    layer.layer_idx = 0
    layer.fused_wqa_wkv = _ShapeSensitiveFusedQKV()
    layer.q_lora_rank = 1
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = layer.attn_gemm_parallel_execute(hidden_states)
    positions = torch.tensor([673, 674])
    # Mirror MacaDeepseekV4Attention.forward's shadow hook without running the full
    # attention stack.
    if attention.qkv_prenorm_shadow_compare_selected(layer.layer_idx, positions):
        rowwise = torch.cat(
            [
                layer.fused_wqa_wkv(hidden_states[index : index + 1])[0].clone()
                for index in range(hidden_states.shape[0])
            ],
            dim=0,
        )
        layer_debug.maybe_capture_qkv_prenorm_shadow_compare(
            layer.layer_idx,
            positions,
            result[0],
            rowwise,
            layer.q_lora_rank,
        )
    assert layer.fused_wqa_wkv.calls == [2, 1, 1]
    torch.testing.assert_close(result[0], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    payload = torch.load(next(tmp_path.glob("*qkv_prenorm_shadow_compare.pt")))
    assert payload["summary"]["qr"]["num_diff"] == 2
    assert payload["summary"]["kv"]["num_diff"] == 2


def test_target_tokenwise_wq_b_replaces_only_selected_positions(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS", "659")
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": None})(),
    )
    layer = _layer(enabled=False)
    layer.is_target_model = True
    layer.layer_idx = 0
    layer.indexer = None
    layer.compressor = None
    layer.n_local_heads = 1
    layer.head_dim = 1
    layer.wq_b = _ShapeSensitiveWqB()
    observed = {}

    def fused(q, _kv, _positions, _metadata):
        observed["q"] = q.detach().clone()
        return q

    layer._fused_qnorm_rope_kv_insert = fused
    layer.forward_mqa = lambda q, _kv, _positions, output: output.copy_(q)
    out = torch.empty(2, 1, 1)

    layer.attention_impl(
        torch.zeros(2, 1),
        torch.tensor([[1.0], [2.0]]),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([658, 659]),
        out,
    )

    assert layer.wq_b.calls == [2, 1]


def test_dspark_draft_keeps_batched_wq_b(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B", "1")

    def upstream(self, _hidden_states, qr, *_args):
        self.wq_b(qr)

    monkeypatch.setattr(attention.DeepseekV4Attention, "attention_impl", upstream)
    layer = _layer(enabled=False)
    layer.is_target_model = False
    layer.layer_idx = 43
    layer.indexer = None
    layer.compressor = None
    layer.n_local_heads = 1
    layer.head_dim = 1
    layer.wq_b = _ShapeSensitiveWqB()

    layer.attention_impl(
        torch.zeros(2, 1),
        torch.tensor([[1.0], [2.0]]),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([658, 659]),
        torch.empty(2, 1, 1),
    )

    assert layer.wq_b.calls == [2]


def test_prefill_gemm_chunking_concatenates_all_components(monkeypatch):
    calls = []

    def upstream(self, hidden_states):
        calls.append(hidden_states.shape[0])
        return (
            hidden_states + 1,
            hidden_states[:, :1] + 2,
            None,
            hidden_states[:, 1:] + 3,
        )

    monkeypatch.setattr(attention.DeepseekV4Attention,
                        "attn_gemm_parallel_execute", upstream)
    hidden_states = torch.arange(633 * 2).reshape(633, 2)
    result = _layer(enabled=True).attn_gemm_parallel_execute(hidden_states)

    assert calls == [256, 256, 121]
    torch.testing.assert_close(result[0], hidden_states + 1)
    torch.testing.assert_close(result[1], hidden_states[:, :1] + 2)
    assert result[2] is None
    torch.testing.assert_close(result[3], hidden_states[:, 1:] + 3)


def test_prefill_gemm_chunking_rejects_inconsistent_optional_components(monkeypatch):
    calls = []

    def upstream(self, hidden_states):
        calls.append(hidden_states.shape[0])
        return (hidden_states, None, None, None) if len(calls) == 1 else (
            hidden_states, hidden_states, None, None
        )

    monkeypatch.setattr(attention.DeepseekV4Attention,
                        "attn_gemm_parallel_execute", upstream)
    with pytest.raises(RuntimeError, match="inconsistent None/non-None"):
        _layer(enabled=True).attn_gemm_parallel_execute(torch.zeros(300, 2))


def test_prefill_wq_b_chunking_aligns_prefix_overlap_shapes():
    layer = _layer(enabled=True, chunk_size=16)
    layer.wq_b = _ShapeSensitiveWqB()
    qr = torch.arange(633 * 2, dtype=torch.float32).reshape(633, 2)

    direct_full = layer.wq_b(qr)
    direct_overlap = layer.wq_b(qr[512:])
    assert not torch.equal(direct_full[512:], direct_overlap)

    layer.wq_b.calls.clear()
    chunked_full = layer._project_wq_b_prefill(qr)
    assert layer.wq_b.calls == [16] * 39 + [9]

    layer.wq_b.calls.clear()
    chunked_overlap = layer._project_wq_b_prefill(qr[512:])
    assert layer.wq_b.calls == [16] * 7 + [9]
    torch.testing.assert_close(chunked_full[512:], chunked_overlap)


@pytest.mark.parametrize(
    ("enabled", "tokens"), [(False, 633), (True, 16), (True, 1)]
)
def test_prefill_wq_b_chunking_disabled_small_or_decode_calls_once(enabled, tokens):
    layer = _layer(enabled=enabled, chunk_size=16)
    layer.wq_b = _ShapeSensitiveWqB()

    layer._project_wq_b_prefill(torch.zeros(tokens, 2))

    assert layer.wq_b.calls == [tokens]


def test_prefill_gemm_chunking_init_reads_env_and_block_size(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert attention.mx_envs.VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING is False
    monkeypatch.setenv(ENV, "1")
    assert attention.mx_envs.VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING is True
