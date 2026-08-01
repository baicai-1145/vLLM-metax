from pathlib import Path

import pytest
import torch

from tools.debug.diff_deepseek_v4_layer_stages import diff_layer_stage_captures


def _capture(
    path: Path,
    *,
    rank: int = 0,
    layer: int = 0,
    call: int = 0,
    stage: str = "before_attention",
    positions: list[int],
    tensors: dict[str, torch.Tensor],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "rank": rank,
            "layer_idx": layer,
            "call": call,
            "stage": stage,
            "positions": torch.tensor(positions, dtype=torch.int64),
            "input_ids": torch.tensor([100 + position for position in positions]),
            "tensors": tensors,
        },
        path,
    )
    return path


def test_diff_layer_stage_captures_compares_position_rows(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_before_attention.pt",
        positions=[767],
        tensors={
            "hidden_states": torch.tensor([[1.0, 2.0]]),
            "residual": torch.tensor([[3.0, 4.0]]),
        },
    )
    _capture(
        candidate / "rank0_layer0_call0_before_attention.pt",
        positions=[766, 767],
        tensors={
            "hidden_states": torch.tensor([[9.0, 9.0], [1.0, 2.5]]),
            "residual": torch.tensor([[9.0, 9.0], [3.0, 4.0]]),
        },
    )
    _capture(
        candidate / "rank0_layer0_call1_before_attention.pt",
        call=1,
        positions=[767, 768],
        tensors={
            "hidden_states": torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
            "residual": torch.tensor([[3.5, 4.0], [9.0, 9.0]]),
        },
    )

    result = diff_layer_stage_captures(
        base=base,
        candidate=candidate,
        position=767,
        layer=0,
        stage="before_attention",
    )

    assert result["summary"] == {
        "num_candidate_rows": 2,
        "exact_candidate_rows": 0,
        "first_different_tensor": "hidden_states",
    }
    assert result["comparisons"][0]["tensors"]["hidden_states"]["num_diff"] == 1
    assert result["comparisons"][0]["tensors"]["hidden_states"]["max_abs"] == 0.5
    assert result["comparisons"][0]["tensors"]["hidden_states"]["first_diff_index"] == [
        0,
        1,
    ]
    assert result["comparisons"][0]["tensors"]["hidden_states"]["base_value"] == 2.0
    assert (
        result["comparisons"][0]["tensors"]["hidden_states"]["candidate_value"]
        == 2.5
    )
    assert result["comparisons"][0]["tensors"]["residual"]["num_diff"] == 0
    assert result["comparisons"][1]["tensors"]["hidden_states"]["num_diff"] == 0
    assert result["comparisons"][1]["tensors"]["residual"]["num_diff"] == 1


def test_diff_layer_stage_captures_requires_full_tensor_row(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_after_ffn.pt",
        stage="after_ffn",
        positions=[767],
        tensors={"hidden_states": torch.zeros(1, 2)},
    )
    _capture(
        candidate / "rank0_layer0_call0_after_ffn.pt",
        stage="after_ffn",
        positions=[768],
        tensors={"hidden_states": torch.zeros(1, 2)},
    )

    with pytest.raises(ValueError, match="no candidate rows"):
        diff_layer_stage_captures(
            base=base,
            candidate=candidate,
            position=767,
            layer=0,
            stage="after_ffn",
        )


def test_diff_layer_stage_captures_supports_attention_top_level_tensors(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_attention_inputs.pt",
        stage="attention_inputs",
        positions=[767],
        tensors={},
    )
    _capture(
        candidate / "rank0_layer0_call0_attention_inputs.pt",
        stage="attention_inputs",
        positions=[766, 767],
        tensors={},
    )
    torch.save(
        {
            "schema_version": 1,
            "rank": 0,
            "layer_idx": 0,
            "call": 0,
            "stage": "attention_inputs",
            "positions": torch.tensor([767], dtype=torch.int64),
            "qr": torch.tensor([[1.0, 2.0]]),
            "kv": torch.tensor([[3.0, 4.0]]),
        },
        base / "rank0_layer0_call0_attention_inputs.pt",
    )
    torch.save(
        {
            "schema_version": 1,
            "rank": 0,
            "layer_idx": 0,
            "call": 0,
            "stage": "attention_inputs",
            "positions": torch.tensor([766, 767], dtype=torch.int64),
            "qr": torch.tensor([[9.0, 9.0], [1.0, 2.5]]),
            "kv": torch.tensor([[9.0, 9.0], [3.0, 4.0]]),
        },
        candidate / "rank0_layer0_call0_attention_inputs.pt",
    )
    result = diff_layer_stage_captures(
        base=base,
        candidate=candidate,
        position=767,
        layer=0,
        stage="attention_inputs",
    )
    assert result["summary"]["first_different_tensor"] == "qr"
    assert result["comparisons"][0]["tensors"]["qr"]["num_diff"] == 1
    assert result["comparisons"][0]["tensors"]["kv"]["num_diff"] == 0
