from pathlib import Path

import torch

from tools.debug.diff_deepseek_v4_qkv_insert_rows import diff_qkv_insert_rows


def _capture(
    path: Path,
    *,
    call: int = 0,
    positions: list[int],
    token_indices: list[int],
    slot_mapping: list[int],
    q: torch.Tensor,
    kv: torch.Tensor,
    cache_before_rows: dict[int, torch.Tensor],
    cache_after_rows: dict[int, torch.Tensor],
    block_size: int = 4,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    head_dim = kv.shape[-1]
    blocks = max(slot // block_size for slot in slot_mapping) + 1
    width = block_size * head_dim
    before = torch.zeros(blocks, width)
    after = torch.zeros(blocks, width)
    block_indices = []
    offsets = []
    for slot in slot_mapping:
        block = slot // block_size
        offset = slot % block_size
        block_indices.append(block)
        offsets.append(offset)
        start = offset * head_dim
        end = start + head_dim
        before[block, start:end] = cache_before_rows[slot]
        after[block, start:end] = cache_after_rows[slot]
    torch.save(
        {
            "schema_version": 1,
            "rank": 0,
            "layer_idx": 0,
            "call": call,
            "stage": "qkv_insert",
            "token_indices": torch.tensor(token_indices, dtype=torch.int64),
            "positions": torch.tensor(positions, dtype=torch.int64),
            "q": q,
            "kv": kv,
            "slot_mapping": torch.tensor(slot_mapping, dtype=torch.int64),
            "block_size": block_size,
            "cache_block_indices": torch.tensor(block_indices, dtype=torch.int64),
            "cache_slot_offsets": torch.tensor(offsets, dtype=torch.int64),
            "cache_before": before.index_select(
                0, torch.tensor(block_indices, dtype=torch.int64)
            ),
            "cache_after": after.index_select(
                0, torch.tensor(block_indices, dtype=torch.int64)
            ),
        },
        path,
    )
    return path


def test_diff_qkv_insert_rows_compares_actual_cache_row_not_whole_block(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_qkv_insert.pt",
        positions=[658],
        token_indices=[0],
        slot_mapping=[6],
        q=torch.tensor([[[1.0, 2.0]]]),
        kv=torch.tensor([[3.0, 4.0]]),
        cache_before_rows={6: torch.tensor([0.0, 0.0])},
        cache_after_rows={6: torch.tensor([3.0, 4.0])},
    )
    _capture(
        candidate / "rank0_layer0_call0_qkv_insert.pt",
        positions=[657, 658],
        token_indices=[0, 1],
        slot_mapping=[4, 10],
        q=torch.tensor([[[9.0, 9.0]], [[1.0, 2.5]]]),
        kv=torch.tensor([[9.0, 9.0], [3.0, 4.0]]),
        cache_before_rows={
            4: torch.tensor([8.0, 8.0]),
            10: torch.tensor([0.0, 0.0]),
        },
        cache_after_rows={
            4: torch.tensor([8.0, 8.0]),
            10: torch.tensor([3.0, 4.0]),
        },
    )
    result = diff_qkv_insert_rows(
        base=base,
        candidate=candidate,
        position=658,
        layer=0,
    )
    assert result["summary"] == {
        "num_candidate_rows": 1,
        "exact_candidate_rows": 0,
        "first_different_tensor": "q",
    }
    tensors = result["comparisons"][0]["tensors"]
    assert tensors["slot_mapping"]["base_value"] == 6
    assert tensors["slot_mapping"]["candidate_value"] == 10
    assert tensors["q"]["first_diff_index"] == [0, 0, 1]
    assert tensors["kv"]["exact"]
    assert tensors["cache_after_row"]["exact"]
