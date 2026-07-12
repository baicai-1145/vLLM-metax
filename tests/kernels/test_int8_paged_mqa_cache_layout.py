import json

import torch

from vllm_metax.customized.layers.sparse_attn_indexer import int8 as int8_indexer


def _metadata(tensor: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], int, bool]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.is_contiguous(),
    )


def test_standard_cache_layout_passes_metadata_validator() -> None:
    cache = torch.empty((2, 4, 1, 16), dtype=torch.uint8)
    assert int8_indexer._indexer_cache_layout_is_standard(*_metadata(cache))


def test_padded_and_offset_cache_layouts_are_red_capable() -> None:
    padded = torch.empty((2, 5, 1, 16), dtype=torch.uint8)[:, :4]
    assert not int8_indexer._indexer_cache_layout_is_standard(*_metadata(padded))

    offset = torch.empty((3, 4), dtype=torch.uint8)[1:]
    assert offset.is_contiguous()
    assert not int8_indexer._indexer_cache_layout_is_standard(*_metadata(offset))


def test_cache_layout_log_is_disabled_without_env(monkeypatch, tmp_path) -> None:
    path = tmp_path / "layout.jsonl"
    monkeypatch.delenv("VLLM_METAX_DSV4_INDEXER_CACHE_LAYOUT_LOG", raising=False)
    int8_indexer._reset_indexer_cache_layout_log_state()
    int8_indexer._maybe_log_indexer_cache_layout(torch.empty((2, 4)))
    assert not path.exists()


def test_cache_layout_log_emits_one_json_record(monkeypatch, tmp_path) -> None:
    path = tmp_path / "nested" / "layout.jsonl"
    monkeypatch.setenv("VLLM_METAX_DSV4_INDEXER_CACHE_LAYOUT_LOG", str(path))
    monkeypatch.setenv("RANK", "7")
    int8_indexer._reset_indexer_cache_layout_log_state()

    cache = torch.empty((2, 4, 1, 16), dtype=torch.uint8)
    int8_indexer._maybe_log_indexer_cache_layout(cache)
    int8_indexer._maybe_log_indexer_cache_layout(cache[:, :3])

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0] == {
        "schema_version": 1,
        "rank": 7,
        "shape": [2, 4, 1, 16],
        "stride": [64, 16, 16, 1],
        "dtype": "torch.uint8",
        "storage_offset": 0,
        "is_contiguous": True,
        "is_standard_layout": True,
    }
