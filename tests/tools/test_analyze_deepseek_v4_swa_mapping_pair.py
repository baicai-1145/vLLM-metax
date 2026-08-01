from pathlib import Path

import torch

from tools.debug.analyze_deepseek_v4_swa_mapping_pair import (
    analyze_swa_mapping_pair,
)


def _payload(
    *,
    position: int,
    indices: list[int],
    block_table: list[int],
    call: int = 0,
    token_index: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "decode",
        "decode_mode": "swa",
        "decode_backend": "native",
        "native_decode_mode": "torch_compat",
        "rank": 0,
        "call": call,
        "observed_call": 0,
        "positions": torch.tensor([position], dtype=torch.int64),
        "token_indices": torch.tensor([token_index], dtype=torch.int64),
        "token_to_req": torch.tensor([0], dtype=torch.int32),
        "q": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        "swa_cache": torch.zeros((260, 64, 1), dtype=torch.bfloat16),
        "compressed_cache": None,
        "swa_indices": torch.tensor([[indices]], dtype=torch.int32),
        "topk_indices": None,
        "swa_lens": torch.tensor([len(indices)], dtype=torch.int32),
        "topk_lens": None,
        "swa_block_table": torch.tensor([block_table], dtype=torch.int32),
        "compressed_block_table": None,
        "sm_scale": 1.0,
        "d_v": 1,
        "head_dim": 1,
        "q_heads": 1,
        "attn_sink": torch.zeros((1,), dtype=torch.float32),
        "input_meta": {},
        "cache_meta": {
            "swa_block_size": 64,
            "compressed_block_size": None,
            "compress_ratio": 1,
            "window_size": len(indices),
        },
        "caller_out": None,
        "output": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
    }


def _save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_analyze_swa_mapping_pair_explains_block_table_shift(tmp_path):
    base = tmp_path / "rank0_call0.pt"
    candidate = tmp_path / "rank0_call1.pt"
    # position 789, width 128, offset 0 maps logical position 662 and block 10.
    base_indices = [181 * 64 + 22 + i for i in range(128)]
    candidate_indices = [180 * 64 + 22 + i for i in range(128)]
    base_payload = _payload(
        position=789,
        indices=base_indices,
        block_table=[0] * 10 + [181, 254, 259],
    )
    candidate_payload = _payload(
        position=789,
        indices=candidate_indices,
        block_table=[0] * 10 + [180, 254, 259],
        call=1,
        token_index=1,
    )
    base_payload["swa_cache"][181, 22, 0] = 1
    candidate_payload["swa_cache"][180, 22, 0] = 2
    _save(base, base_payload)
    _save(candidate, candidate_payload)

    result = analyze_swa_mapping_pair(base, candidate)

    assert result["decision"] == "fail"
    assert result["num_swa_index_differences"] == 128
    assert result["num_block_table_differences"] == 1
    assert result["block_table_differences"] == [
        {"logical_block": 10, "base_block": 181, "candidate_block": 180}
    ]
    first = result["first_differences"][0]
    assert first["logical_position"] == 662
    assert first["logical_block"] == 10
    assert first["slot_delta"] == -64
    assert first["base"]["physical_block"] == 181
    assert first["candidate"]["physical_block"] == 180
    assert result["cache_row_content"]["num_rows_with_differences"] == 1
    assert result["cache_row_content"]["first_difference"]["logical_position"] == 662
