import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm import ModelRegistry
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as UpstreamDSpark,
)
from vllm_metax.models import register_model
from vllm_metax.models.deepseek_v4 import dspark
from vllm_metax.models.deepseek_v4.model import DeepseekV4DecoderLayer


_DSPARK_STAGING = Path("/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-DSpark-staging")


def test_dspark_registry_resolves_to_metax_adapter():
    register_model()
    model_cls = ModelRegistry._try_load_model_cls("DSparkDraftModel")
    assert model_cls is dspark.DSparkDeepseekV4ForCausalLM
    assert issubclass(model_cls, UpstreamDSpark)


@pytest.mark.skipif(not _DSPARK_STAGING.is_dir(), reason="DSpark staging is absent")
def test_real_dspark_staging_has_three_bf16_draft_stages():
    from safetensors import safe_open

    config = json.loads((_DSPARK_STAGING / "config.json").read_text())
    index = json.loads((_DSPARK_STAGING / "model.safetensors.index.json").read_text())
    mtp_weights = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith("mtp.")
    }

    assert config["n_mtp_layers"] == 3
    assert config["dspark_target_layer_ids"] == [40, 41, 42]
    assert {name.split(".", 2)[1] for name in mtp_weights} == {"0", "1", "2"}

    dtypes = Counter()
    for shard in sorted(set(mtp_weights.values())):
        with safe_open(_DSPARK_STAGING / shard, framework="pt", device="cpu") as file:
            for name in file.keys():  # noqa: SIM118 - safe_open is not a Mapping
                if name.startswith("mtp."):
                    dtypes[str(file.get_slice(name).get_dtype())] += 1
    assert dtypes == {"BF16": 2349, "F32": 27}


def test_dspark_decoder_layer_factory_uses_metax_decoder(monkeypatch):
    class SentinelLayer(torch.nn.Module):
        def __init__(self, vllm_config, prefix):
            super().__init__()
            self.vllm_config = vllm_config
            self.prefix = prefix

    config = object()
    assert dspark.DSparkDeepseekV4Model.decoder_layer_cls is DeepseekV4DecoderLayer
    monkeypatch.setattr(
        dspark.DSparkDeepseekV4Model, "decoder_layer_cls", SentinelLayer
    )
    layers = dspark.DSparkDeepseekV4Model.build_decoder_layers(
        vllm_config=config,
        prefix="model",
        num_hidden_layers=17,
        num_dspark_layers=2,
    )
    assert [layer.prefix for layer in layers] == ["model.layers.17", "model.layers.18"]


def _fake_attention(cache_dtype=torch.bfloat16):
    cache = torch.empty((2, 4, 8), dtype=cache_dtype)
    return SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=4),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty((16, 8))),
        n_local_heads=2,
        head_dim=8,
        eps=1e-6,
    )


def test_context_insert_dispatches_bf16_native_op(monkeypatch):
    calls = []

    def native(*args):
        calls.append(args)

    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(fused_deepseek_v4_qnorm_rope_kv_rope_insert=native),
    )
    attn = _fake_attention()
    kv = torch.empty((3, 8), dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int64)
    slots = torch.tensor([0, 1, 4], dtype=torch.int64)

    dspark._insert_context_kv(attn, kv, positions, slots)

    assert len(calls) == 1
    (
        dummy_q,
        got_kv,
        cache,
        got_slots,
        got_positions,
        cos_sin,
        eps,
        block_size,
    ) = calls[0]
    assert dummy_q.shape == (3, 2, 8)
    assert dummy_q.dtype is torch.bfloat16
    assert got_kv is kv
    assert cache.shape == (2, 32)
    assert got_slots is slots
    assert got_positions is positions
    assert cos_sin is attn.rotary_emb.cos_sin_cache
    assert eps == attn.eps
    assert block_size == 4


def test_context_insert_accepts_packed_cache_block_stride(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            fused_deepseek_v4_qnorm_rope_kv_rope_insert=lambda *args: calls.append(args)
        ),
    )
    packed = torch.empty((2, 40), dtype=torch.bfloat16)
    attn = _fake_attention()
    attn.swa_cache_layer.kv_cache = packed[:, :32].view(2, 4, 8)
    assert not attn.swa_cache_layer.kv_cache.is_contiguous()

    dspark._insert_context_kv(
        attn,
        torch.empty((1, 8), dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    )

    assert calls[0][2].shape == (2, 32)
    assert calls[0][2].stride() == (40, 1)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float8_e4m3fn])
def test_context_insert_rejects_non_bf16_cache(dtype):
    with pytest.raises(ValueError, match="bfloat16"):
        dspark._insert_context_kv(
            _fake_attention(dtype),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )


def test_context_insert_rejects_missing_native_op(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    with pytest.raises(RuntimeError, match="native BF16"):
        dspark._insert_context_kv(
            _fake_attention(),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )


@pytest.mark.parametrize(
    ("positions", "slots", "match"),
    [
        (
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            "length",
        ),
        (torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64), "int64"),
        (torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int32), "int64"),
    ],
)
def test_context_insert_rejects_invalid_index_tensors(positions, slots, match):
    with pytest.raises(ValueError, match=match):
        dspark._insert_context_kv(
            _fake_attention(),
            torch.empty((1, 8), dtype=torch.bfloat16),
            positions,
            slots,
        )


def test_context_insert_rejects_noncontiguous_kv():
    kv = torch.empty((8, 2), dtype=torch.bfloat16).t()
    assert not kv.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        dspark._insert_context_kv(
            _fake_attention(),
            kv,
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
        )


def test_context_kv_preserves_one_slot_mapping_per_layer(monkeypatch):
    calls = []

    class FakeAttention:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.q_lora_rank = 0

        def fused_wqa_wkv(self, hidden):
            return hidden, None

        def kv_norm(self, value):
            return value

    model = dspark.DSparkDeepseekV4Model.__new__(dspark.DSparkDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.layers = [
        SimpleNamespace(attn=FakeAttention(0)),
        SimpleNamespace(attn=FakeAttention(1)),
    ]

    def fake_insert(attn, kv, positions, slots):
        calls.append((attn.layer_id, kv, positions, slots))

    monkeypatch.setattr(dspark, "_insert_context_kv", fake_insert)
    main_x = torch.empty((2, 8), dtype=torch.bfloat16)
    positions = torch.arange(2, dtype=torch.int64)
    slots = [torch.tensor([3, 5]), torch.tensor([7, 9])]

    model.precompute_and_store_context_kv(main_x, positions, slots)

    assert [(layer_id, mapping.tolist()) for layer_id, _, _, mapping in calls] == [
        (0, [3, 5]),
        (1, [7, 9]),
    ]
