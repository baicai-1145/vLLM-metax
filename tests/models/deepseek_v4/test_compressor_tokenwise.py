from types import SimpleNamespace

import torch

import vllm_metax.models.deepseek_v4.compressor as compressor_mod


def _compressor():
    compressor = object.__new__(compressor_mod.MacaDeepseekCompressor)
    compressor.coff = 1
    compressor.head_dim = 2
    compressor.ape = torch.empty(1)
    compressor.compress_ratio = 4
    compressor.rope_head_dim = 2
    compressor.overlap = False
    compressor.use_fp4_cache = False
    compressor.norm = SimpleNamespace(weight=torch.ones(2))
    compressor.rms_norm_eps = 1e-6
    compressor._quant_block = 64
    compressor._token_stride = 2
    compressor._scale_dim = 1
    compressor.state_cache = SimpleNamespace(
        prefix="layer.compressor.state",
        kv_cache=torch.zeros(4, 64, 4),
    )
    compressor.k_cache_prefix = "layer.attn"
    compressor._static_forward_context = {
        "layer.attn": SimpleNamespace(kv_cache=torch.zeros(4, 64, 2)),
    }
    return compressor


def _metadata():
    state_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([3, 5]),
        slot_mapping=torch.tensor([10, 11]),
        block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        block_size=64,
    )
    k_metadata = SimpleNamespace(slot_mapping=torch.tensor([20, 21]))
    return {
        "layer.compressor.state": state_metadata,
        "layer.attn": k_metadata,
    }


def test_tokenwise_compressor_slices_metadata(monkeypatch):
    calls = []
    save_calls = []

    def fake_save_partial_states(**kwargs):
        save_calls.append(
            {
                "kv": kwargs["kv"].clone(),
                "score": kwargs["score"].clone(),
                "positions": kwargs["positions"].clone(),
                "slot_mapping": kwargs["slot_mapping"].clone(),
            }
        )

    def fake_compress_norm_rope_store(**kwargs):
        calls.append(
            {
                "num_actual": kwargs["num_actual"],
                "token_to_req_indices": kwargs["token_to_req_indices"].clone(),
                "positions": kwargs["positions"].clone(),
                "slot_mapping": kwargs["slot_mapping"].clone(),
                "kv_slot_mapping": kwargs["k_cache_metadata"].slot_mapping.clone(),
            }
        )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", "1")
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(compressor_mod, "save_partial_states", fake_save_partial_states)
    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        fake_compress_norm_rope_store,
    )
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=_metadata()),
    )

    kv_score = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    positions = torch.tensor([658, 659])
    _compressor().forward(kv_score, positions, SimpleNamespace(cos_sin_cache=torch.empty(1)))

    assert [call["positions"].tolist() for call in save_calls] == [[658], [659]]
    assert [call["slot_mapping"].tolist() for call in save_calls] == [[10], [11]]
    assert [call["num_actual"] for call in calls] == [1, 1]
    assert [call["positions"].tolist() for call in calls] == [[658], [659]]
    assert [call["slot_mapping"].tolist() for call in calls] == [[10], [11]]
    assert [call["kv_slot_mapping"].tolist() for call in calls] == [[20], [21]]
    assert [call["token_to_req_indices"].tolist() for call in calls] == [[3], [5]]


def test_tokenwise_compressor_can_fuse_partial_state_save(monkeypatch):
    calls = []
    save_calls = []

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_COMPRESSOR_FUSED_SAVE_PARTIAL_STATES", "1"
    )
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(
        compressor_mod,
        "save_partial_states",
        lambda **kwargs: save_calls.append(kwargs),
    )

    def fake_compress_norm_rope_store(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        fake_compress_norm_rope_store,
    )
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=_metadata()),
    )

    kv_score = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    positions = torch.tensor([658, 659])
    compressor = _compressor()
    compressor.ape = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    compressor.forward(
        kv_score, positions, SimpleNamespace(cos_sin_cache=torch.empty(1))
    )

    assert save_calls == []
    assert len(calls) == 2
    assert all(call["fuse_save_partial_states"] for call in calls)
    assert torch.equal(calls[0]["kv"], kv_score[0:1, :2])
    assert torch.equal(calls[1]["score"], kv_score[1:2, 2:])
    assert all(call["ape"] is compressor.ape for call in calls)


def test_tokenwise_compressor_supports_six_row_dspark_verifier(monkeypatch):
    calls = []
    metadata = _metadata()
    metadata["layer.compressor.state"].slot_mapping = torch.arange(10, 16)
    metadata["layer.compressor.state"].token_to_req_indices = torch.arange(6)
    metadata["layer.attn"].slot_mapping = torch.arange(20, 26)

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", "1")
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(compressor_mod, "save_partial_states", lambda **kwargs: None)
    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        lambda **kwargs: calls.append(kwargs["num_actual"]),
    )
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )

    _compressor().forward(
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(6),
        SimpleNamespace(cos_sin_cache=torch.empty(1)),
    )

    assert calls == [1] * 6


def test_tokenwise_indexer_compressor_skips_short_context_rows(monkeypatch):
    saved_positions = []
    compressed_positions = []

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", "1")
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(
        compressor_mod,
        "save_partial_states",
        lambda **kwargs: saved_positions.extend(kwargs["positions"].tolist()),
    )
    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        lambda **kwargs: compressed_positions.extend(kwargs["positions"].tolist()),
    )
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=_metadata()),
    )
    compressor = _compressor()
    compressor._tokenwise_min_position = 128

    compressor.forward(
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.tensor([127, 128]),
        SimpleNamespace(cos_sin_cache=torch.empty(1)),
    )

    assert saved_positions == [128]
    assert compressed_positions == [128]


def test_compressor_default_uses_single_launch(monkeypatch):
    calls = []

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", raising=False)
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(compressor_mod, "save_partial_states", lambda **kwargs: None)
    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        lambda **kwargs: calls.append(
            (
                kwargs["num_actual"],
                kwargs["positions"].tolist(),
                kwargs["k_cache_metadata"].slot_mapping.tolist(),
            )
        ),
    )
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=_metadata()),
    )

    kv_score = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    _compressor().forward(
        kv_score,
        torch.tensor([658, 659]),
        SimpleNamespace(cos_sin_cache=torch.empty(1)),
    )

    assert calls == [(2, [658, 659], [20, 21])]


def test_compressor_capture_filters_physical_kv_slots(monkeypatch, tmp_path):
    compressor_mod._COMPRESSOR_CAPTURE_CALL_COUNT = 0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_LAYERS", "4")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_HEAD_DIMS", "2")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_SLOTS", "223,226")
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(compressor_mod, "save_partial_states", lambda **kwargs: None)

    def fake_compress_norm_rope_store(**kwargs):
        kv_cache = kwargs["kv_cache"]
        for position, slot in zip(
            kwargs["positions"].tolist(),
            kwargs["k_cache_metadata"].slot_mapping.tolist(),
        ):
            kv_cache[slot // kv_cache.shape[1], slot % kv_cache.shape[1]].fill_(
                float(position)
            )

    monkeypatch.setattr(
        compressor_mod,
        "compress_norm_rope_store_triton",
        fake_compress_norm_rope_store,
    )
    state_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0, 0]),
        slot_mapping=torch.tensor([70, 71, 72]),
        block_table=torch.arange(20, dtype=torch.int64).reshape(1, 20),
        block_size=64,
    )
    k_metadata = SimpleNamespace(slot_mapping=torch.tensor([222, 223, 226]))
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "layer.compressor.state": state_metadata,
                "model.layers.4.self_attn": k_metadata,
            }
        ),
    )
    compressor = _compressor()
    compressor.k_cache_prefix = "model.layers.4.self_attn"
    compressor._static_forward_context = {
        "model.layers.4.self_attn": SimpleNamespace(kv_cache=torch.zeros(4, 64, 2)),
    }
    compressor.state_cache.kv_cache = torch.arange(
        20 * 64 * 4, dtype=torch.float32
    ).reshape(20, 64, 4)
    kv_score = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    compressor.forward(
        kv_score,
        torch.tensor([890, 891, 903]),
        SimpleNamespace(cos_sin_cache=torch.empty(1)),
    )

    [capture_path] = sorted(tmp_path.glob("rank0_layer4_call0_compressor.pt"))
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert payload["positions"].tolist() == [891, 903]
    assert payload["token_indices"].tolist() == [1, 2]
    assert payload["slot_mapping"].tolist() == [71, 72]
    assert payload["kv_slot_mapping"].tolist() == [223, 226]
    assert payload["head_dim"] == 2
    assert payload["cache_block_indices"].tolist() == [3, 3]
    assert payload["cache_slot_offsets"].tolist() == [31, 34]
    assert payload["state_positions"].tolist() == [
        [888, 889, 890, 891],
        [900, 901, 902, 903],
    ]
    torch.testing.assert_close(payload["kv"], kv_score.split([2, 2], dim=-1)[0][1:])
    torch.testing.assert_close(payload["score"], kv_score.split([2, 2], dim=-1)[1][1:])
    torch.testing.assert_close(payload["kv_cache_rows_before"], torch.zeros(2, 2))
    torch.testing.assert_close(
        payload["kv_cache_rows_after"],
        torch.tensor([[891.0, 891.0], [903.0, 903.0]]),
    )


def test_compressor_state_only_capture_allows_invalid_kv_slot(monkeypatch, tmp_path):
    compressor_mod._COMPRESSOR_CAPTURE_CALL_COUNT = 0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_LAYERS", "4")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_HEAD_DIMS", "2")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_POSITIONS", "127")
    monkeypatch.setenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_STATE_ONLY", "1")
    monkeypatch.setattr(
        compressor_mod, "current_platform", SimpleNamespace(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(compressor_mod, "save_partial_states", lambda **kwargs: None)
    monkeypatch.setattr(
        compressor_mod, "compress_norm_rope_store_triton", lambda **kwargs: None
    )
    state_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0]),
        slot_mapping=torch.tensor([127]),
        block_table=torch.arange(4, dtype=torch.int64).reshape(1, 4),
        block_size=64,
    )
    k_metadata = SimpleNamespace(slot_mapping=torch.tensor([-1]))
    monkeypatch.setattr(
        compressor_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "layer.compressor.state": state_metadata,
                "model.layers.4.self_attn": k_metadata,
            }
        ),
    )
    compressor = _compressor()
    compressor.k_cache_prefix = "model.layers.4.self_attn"
    compressor._static_forward_context = {
        "model.layers.4.self_attn": SimpleNamespace(kv_cache=torch.zeros(4, 64, 2)),
    }
    compressor.state_cache.kv_cache = torch.arange(
        4 * 64 * 4, dtype=torch.float32
    ).reshape(4, 64, 4)

    compressor.forward(
        torch.arange(4, dtype=torch.float32).reshape(1, 4),
        torch.tensor([127]),
        SimpleNamespace(cos_sin_cache=torch.empty(1)),
    )

    [capture_path] = sorted(tmp_path.glob("rank0_layer4_call0_compressor.pt"))
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert payload["positions"].tolist() == [127]
    assert payload["kv_slot_mapping"].tolist() == [-1]
    assert payload["cache_block_indices"].numel() == 0
    assert payload["kv_cache_rows_before"].shape == (0, 2)
    assert payload["kv_cache_rows_after"].shape == (0, 2)


def test_capture_kv_cache_rows_reads_planar_int8_layout():
    cache = torch.zeros(2, 4, 132, dtype=torch.uint8)
    flat = cache.reshape(2, -1)
    values = torch.arange(128, dtype=torch.uint8)
    scale = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    cache[1, 2].fill_(255)
    flat[1, 2 * 128 : 3 * 128] = values
    flat[1, 4 * 128 + 2 * 4 : 4 * 128 + 3 * 4] = scale

    row = compressor_mod._capture_kv_cache_rows(
        cache,
        torch.tensor([1]),
        torch.tensor([2]),
        head_dim=128,
    )

    torch.testing.assert_close(row[0, :128], values)
    torch.testing.assert_close(row[0, 128:], scale)
