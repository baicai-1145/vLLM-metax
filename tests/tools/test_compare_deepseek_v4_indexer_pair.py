from pathlib import Path

import pytest
import torch

from tools.debug.compare_deepseek_v4_indexer_pair import compare_indexer_pair


def _payload(
    *,
    table: list[int],
    scores: list[float],
    topk: list[int],
    final_topk: list[int] | None = None,
    block_size: int = 4,
    branch: str = "decode",
    seq_len: int | None = None,
) -> dict[str, object]:
    seq_len = len(scores) if seq_len is None else seq_len
    width = 128
    logits = torch.full((1, width), float("-inf"), dtype=torch.float32)
    for logical, score in enumerate(scores[:seq_len]):
        logits[0, logical] = score
    return {
        "branch": branch,
        "replay": {
            "seq_lens": torch.tensor([seq_len], dtype=torch.int32),
            "block_table": torch.tensor([table], dtype=torch.int32),
            "kv_cache": torch.zeros((max(table) + 1, block_size, 1, 132), dtype=torch.uint8),
            "schedule_metadata": torch.tensor([0], dtype=torch.int32),
        },
        "native_logits": logits,
        "native_topk": torch.tensor([topk], dtype=torch.int32),
        "final_topk": torch.tensor([final_topk if final_topk is not None else topk], dtype=torch.int32),
        "metadata": {"block_size": block_size},
    }


def _save(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


def _save_and_return(path: Path, payload: dict[str, object]) -> Path:
    _save(path, payload)
    return path


def _set_planar_row(payload: dict[str, object], block: int, offset: int, value: int, scale: int = 0) -> None:
    replay = payload["replay"]
    assert isinstance(replay, dict)
    cache = replay["kv_cache"]
    assert isinstance(cache, torch.Tensor)
    flat = cache[block].view(torch.uint8).flatten()
    block_size = cache.shape[1]
    flat[offset * 128 : (offset + 1) * 128] = value
    flat[block_size * 128 + offset * 4 : block_size * 128 + offset * 4 + 4] = scale


def test_same_logical_logits_different_physical_tables(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    scores = list(range(40, 0, -1))
    _save(base, _payload(table=list(range(40, 50)), scores=scores, topk=[0, 1, 2], final_topk=[0, 1, 2], seq_len=40))
    _save(
        candidate,
        _payload(table=list(range(50, 60)), scores=scores, topk=[0, 1, 2], final_topk=[0, 1, 2], seq_len=40),
    )

    result = compare_indexer_pair(base, candidate)

    assert result["classification"] == "exact"
    assert result["logical_logits"]["exact"] is True
    assert result["topk"]["ordered_equal"] is True
    assert result["topk"]["set_equal"] is True


def test_equal_score_order_permutation_is_tie_order_only(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    scores = [4, 4, 1, 0]
    _save(base, _payload(table=[0], scores=scores, topk=[0, 1], final_topk=[0, 1]))
    _save(
        candidate,
        _payload(table=[0], scores=scores, topk=[1, 0], final_topk=[1, 0]),
    )

    result = compare_indexer_pair(base, candidate)

    assert result["classification"] == "tie_order_only"
    assert result["logical_logits"]["exact"] is True
    assert result["topk"]["ordered_equal"] is False
    assert result["topk"]["set_equal"] is True


def test_unequal_score_order_permutation_is_not_tie_order_only(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    scores = [4, 3, 1, 0]
    _save(base, _payload(table=[0], scores=scores, topk=[0, 1], final_topk=[0, 1]))
    _save(
        candidate,
        _payload(table=[0], scores=scores, topk=[1, 0], final_topk=[1, 0]),
    )

    result = compare_indexer_pair(base, candidate)

    assert result["classification"] == "logits_drift"


def test_hashes_available_indexer_inputs(tmp_path):
    base = _payload(table=[0], scores=[2, 1], topk=[0, 1])
    candidate = _payload(table=[0], scores=[2, 1], topk=[0, 1])
    for payload in (base, candidate):
        replay = payload["replay"]
        assert isinstance(replay, dict)
        replay.update(
            {
                "padded_q": torch.ones((1, 1, 1, 1)),
                "weights_slice": torch.ones((1, 1)),
                "schedule_metadata": torch.tensor([7], dtype=torch.int32),
                "decode_lens": torch.tensor([1], dtype=torch.int32),
            }
        )
        payload["q_quant"] = torch.ones((1, 1), dtype=torch.int8)
        payload["weights"] = torch.ones((1, 1))
    result = compare_indexer_pair(
        _save_and_return(tmp_path / "base.pt", base),
        _save_and_return(tmp_path / "candidate.pt", candidate),
    )

    assert result["input_hash_equal"]["replay.padded_q"] is True
    assert result["input_hash_equal"]["replay.weights_slice"] is True
    assert result["input_hash_equal"]["replay.schedule_metadata"] is True
    assert result["input_hash_equal"]["replay.decode_lens"] is True
    assert result["input_hash_equal"]["capture.q_quant"] is True
    assert result["input_hash_equal"]["capture.weights"] is True


def test_logical_kv_rows_ignore_physical_table_allocation(tmp_path):
    base = _payload(table=[0], scores=[2], topk=[0])
    candidate = _payload(table=[1], scores=[2], topk=[0])
    for payload in (base, candidate):
        replay = payload["replay"]
        assert isinstance(replay, dict)
        replay["kv_cache"] = torch.zeros((2, 4, 1, 132), dtype=torch.uint8)
    _set_planar_row(base, 0, 0, value=3, scale=7)
    _set_planar_row(candidate, 1, 0, value=3, scale=7)
    # Unselected bytes in candidate's physical row differ; planar gathering
    # must not mistake them for logical position zero's scale bytes.
    candidate["replay"]["kv_cache"][1, 1, 0, :4] = 99

    result = compare_indexer_pair(
        _save_and_return(tmp_path / "base.pt", base),
        _save_and_return(tmp_path / "candidate.pt", candidate),
    )

    assert result["logical_kv"]["exact"] is True
    assert result["logical_kv"]["total_bytes"] == 132
    assert result["logical_kv"]["differing_bytes"] == 0
    assert result["logical_kv"]["packed_132"]["value_bytes_exact"] is True
    assert result["logical_kv"]["packed_132"]["scale_bytes_exact"] is True
    assert result["input_equivalent"] is True


def test_logical_kv_difference_reports_first_byte(tmp_path):
    base = _payload(table=[0], scores=[2, 1], topk=[0, 1])
    candidate = _payload(table=[1], scores=[2, 1], topk=[0, 1])
    for payload in (base, candidate):
        replay = payload["replay"]
        assert isinstance(replay, dict)
        replay["kv_cache"] = torch.zeros((2, 4, 1, 132), dtype=torch.uint8)
    _set_planar_row(base, 0, 1, value=0)
    _set_planar_row(candidate, 1, 1, value=0)
    base["replay"]["kv_cache"][0].view(torch.uint8).flatten()[128 + 7] = 3
    candidate["replay"]["kv_cache"][1].view(torch.uint8).flatten()[128 + 7] = 4

    result = compare_indexer_pair(
        _save_and_return(tmp_path / "base.pt", base),
        _save_and_return(tmp_path / "candidate.pt", candidate),
    )

    assert result["logical_kv"]["exact"] is False
    assert result["logical_kv"]["differing_bytes"] == 1
    assert result["logical_kv"]["first_difference"] == [0, 1, 7]
    assert result["input_equivalent"] is False


def test_score_drift_and_set_change_are_distinguished(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _save(base, _payload(table=[0], scores=[4, 3, 1], topk=[0, 1]))
    _save(candidate, _payload(table=[0], scores=[4, 2, 3], topk=[0, 2]))

    result = compare_indexer_pair(base, candidate)

    assert result["classification"] == "topk_set_change"
    assert result["logical_logits"]["exact"] is False
    assert result["topk"]["set_equal"] is False


def test_invalid_length_or_layout_is_rejected(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _save(base, _payload(table=[0], scores=[1, 0], topk=[0]))
    bad = _payload(table=[0], scores=[1, 0], topk=[0], seq_len=3)
    _save(candidate, bad)

    with pytest.raises(ValueError, match="logical length"):
        compare_indexer_pair(base, candidate)
