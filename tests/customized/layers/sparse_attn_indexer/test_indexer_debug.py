from pathlib import Path

import torch

from vllm_metax.customized.layers.sparse_attn_indexer import indexer_debug
from vllm_metax.customized.layers.sparse_attn_indexer import int8


def _set_capture_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(indexer_debug.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    indexer_debug.reset_call_counters()


def _context(monkeypatch, tmp_path: Path):
    _set_capture_env(monkeypatch, tmp_path)
    return indexer_debug.begin_capture(
        layer="model.layers.0.indexer",
        has_prefill=True,
        has_decode=False,
        num_tokens=2,
        num_decode_tokens=0,
        num_prefill_tokens=2,
        hidden_states=torch.ones(2, 4),
        q_quant=torch.ones(2, 8, dtype=torch.int8),
        weights=torch.ones(2, 8),
        topk_tokens=2,
        slot_mapping=torch.tensor([3, 4]),
    )


def test_decode_logits_clean_invalid_context_tail(monkeypatch):
    sentinel = object()
    calls = []

    def fake_logits(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(int8, "int8_paged_mqa_logits", fake_logits)
    args = tuple(object() for _ in range(6))

    result = int8._int8_paged_decode_logits(*args, max_model_len=1024)

    assert result is sentinel
    assert calls == [(args, {"max_model_len": 1024, "clean_logits": True})]


def test_tokenwise_decode_logits_preserves_batch_token_order(monkeypatch):
    calls = []

    def fake_metadata(context_lens, block_size, num_sms):
        assert block_size == 64
        assert num_sms == 12
        return context_lens.clone()

    def fake_logits(q, _cache, weights, context_lens, _table, schedule, *, max_model_len):
        calls.append((q.clone(), weights.clone(), context_lens.clone(), schedule.clone()))
        assert max_model_len == 1
        return context_lens.reshape(-1, 1).to(torch.float32)

    monkeypatch.setattr(int8, "get_paged_mqa_logits_metadata", fake_metadata)
    monkeypatch.setattr(int8, "_int8_paged_decode_logits", fake_logits)

    q = torch.arange(2 * 3 * 2, dtype=torch.int8).reshape(2, 3, 1, 2)
    weights = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(6, 4)
    seq_lens = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    result = int8._int8_paged_decode_logits_tokenwise(
        q,
        torch.empty(1, 64),
        weights,
        seq_lens,
        torch.zeros(2, 1, dtype=torch.int32),
        max_model_len=1,
        num_sms=12,
    )

    assert result.tolist() == [[11.0], [12.0], [13.0], [21.0], [22.0], [23.0]]
    assert len(calls) == 3
    for index, (row_q, row_weights, row_lens, schedule) in enumerate(calls):
        assert torch.equal(row_q, q[:, index : index + 1])
        assert torch.equal(row_weights, weights.reshape(2, 3, 4)[:, index])
        assert torch.equal(row_lens, seq_lens[:, index : index + 1])
        assert torch.equal(schedule, row_lens)


def test_prefill_logits_clean_outside_each_query_context(monkeypatch):
    sentinel = object()
    calls = []

    def fake_logits(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(int8, "int8_mqa_logits", fake_logits)
    args = tuple(object() for _ in range(5))

    result = int8._int8_prefill_logits(*args)

    assert result is sentinel
    assert calls == [(args, {"clean_logits": True})]


def test_topk_indices_mark_entries_beyond_context_as_invalid():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 2.0, 4.0]])
    output = torch.empty(2, 3, dtype=torch.int32)

    int8._fill_topk_indices_torch(
        logits,
        output,
        valid_counts=torch.tensor([2, 1]),
    )

    assert output.tolist() == [[0, 1, -1], [3, -1, -1]]


def test_capture_is_inert_when_env_is_off(monkeypatch, tmp_path):
    monkeypatch.delenv(indexer_debug.CAPTURE_DIR_ENV, raising=False)
    indexer_debug.reset_call_counters()
    context = indexer_debug.begin_capture(
        layer="layer0",
        has_prefill=False,
        has_decode=True,
        num_tokens=1,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        hidden_states=torch.ones(1, 2),
        q_quant=torch.ones(1, 2, dtype=torch.int8),
        weights=torch.ones(1, 2),
        topk_tokens=1,
        slot_mapping=torch.zeros(1, dtype=torch.int32),
    )
    assert context is None
    assert indexer_debug.call_count("layer0", "0") == 0
    assert not list(tmp_path.iterdir())


def test_filters_and_per_layer_counters_allow_duplicate_corpus_calls(monkeypatch, tmp_path):
    _set_capture_env(monkeypatch, tmp_path)
    monkeypatch.setenv(indexer_debug.CAPTURE_RANKS_ENV, "0")
    monkeypatch.setenv(indexer_debug.CAPTURE_LAYERS_ENV, "layers.0")
    monkeypatch.setenv(indexer_debug.CAPTURE_CALLS_ENV, "0,70")
    selected = []
    for _ in range(71):
        context = indexer_debug.begin_capture(
            layer="model.layers.0.indexer",
            has_prefill=False,
            has_decode=True,
            num_tokens=1,
            num_decode_tokens=1,
            num_prefill_tokens=0,
            hidden_states=torch.zeros(1, 2),
            q_quant=torch.zeros(1, 2, dtype=torch.int8),
            weights=torch.zeros(1, 2),
            topk_tokens=1,
            slot_mapping=torch.zeros(1, dtype=torch.int32),
        )
        if context is not None:
            selected.append(context.call)
    assert selected == [0, 70]
    assert indexer_debug.call_count("model.layers.0.indexer", "0") == 71


def test_capture_fails_closed_during_cuda_graph_capture(monkeypatch, tmp_path):
    _set_capture_env(monkeypatch, tmp_path)
    monkeypatch.setattr(indexer_debug, "_is_cuda_graph_capturing", lambda: True)
    assert _context(monkeypatch, tmp_path) is None
    assert indexer_debug.call_count("model.layers.0.indexer", "0") == 0
    assert not list(tmp_path.iterdir())


def test_prefill_payload_schema_and_cpu_serialization(monkeypatch, tmp_path):
    context = _context(monkeypatch, tmp_path)
    assert context is not None
    path = tmp_path / "capture.pt"
    context.save_prefill(
        path=path,
        q_slice=torch.ones(2, 8, dtype=torch.int8),
        k_quant=torch.ones(3, 8, dtype=torch.int8),
        k_scale=torch.ones(3, 1),
        weights_slice=torch.ones(2, 8),
        cu_seqlen_ks=torch.tensor([0, 2]),
        cu_seqlen_ke=torch.tensor([2, 3]),
        native_logits=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        native_topk=torch.tensor([[1, 0], [2, 1]], dtype=torch.int32),
        chunk_index=0,
        token_start=0,
        token_end=2,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["rank"] == 0
    assert payload["layer"] == "model.layers.0.indexer"
    assert payload["call"] == 0
    assert payload["branch"] == "prefill"
    assert payload["replay"]["q_slice"].device.type == "cpu"
    assert payload["native_logits"].shape == (2, 3)
    assert payload["last_token"]["q_quant"]["sha256"]


def test_save_cpu_payload_recursively_snapshots_tensors(tmp_path):
    path = tmp_path / "nested.pt"
    indexer_debug.save_cpu_payload(
        path,
        {"items": [torch.tensor([1, 2]), (torch.tensor([3]),)]},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["items"][0].device.type == "cpu"
    assert payload["items"][1][0].tolist() == [3]


def test_decode_capture_seq_lens_filter_skips_unselected_call(monkeypatch, tmp_path):
    context = _context(monkeypatch, tmp_path)
    assert context is not None
    monkeypatch.setenv(indexer_debug.CAPTURE_SEQ_LENS_ENV, "139")
    path = tmp_path / "decode.pt"

    context.save_decode(
        path=path,
        padded_q=torch.ones(1, 1, 8, dtype=torch.int8),
        kv_cache=torch.ones(1, 64, 132, dtype=torch.uint8),
        weights_slice=torch.ones(1, 8),
        seq_lens=torch.tensor([[138]], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        schedule_metadata=torch.zeros(1, dtype=torch.int32),
        decode_lens=torch.ones(1, dtype=torch.int32),
        native_logits=torch.ones(1, 138),
        native_topk=torch.zeros(1, 1, dtype=torch.int32),
        requires_padding=False,
    )

    assert not path.exists()
