from pathlib import Path

import pytest
import torch

from tools.debug.diff_deepseek_v4_q_stages import diff_q_stage_captures


def _capture(
    path: Path,
    *,
    rank: int = 0,
    layer: int = 0,
    call: int = 0,
    positions: list[int],
    raw_q: torch.Tensor,
    post_q: torch.Tensor,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "rank": rank,
            "layer_idx": layer,
            "call": call,
            "stage": "q_stages",
            "positions": torch.tensor(positions, dtype=torch.int64),
            "raw_q": raw_q,
            "post_q": post_q,
        },
        path,
    )
    return path


def test_diff_q_stage_captures_compares_selected_position(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_q_stages.pt",
        positions=[767],
        raw_q=torch.tensor([[[1.0, 2.0]]]),
        post_q=torch.tensor([[[3.0, 4.0]]]),
    )
    _capture(
        candidate / "rank0_layer0_call0_q_stages.pt",
        call=0,
        positions=[766, 767],
        raw_q=torch.tensor([[[9.0, 9.0]], [[1.0, 2.5]]]),
        post_q=torch.tensor([[[9.0, 9.0]], [[3.0, 4.0]]]),
    )
    _capture(
        candidate / "rank0_layer0_call1_q_stages.pt",
        call=1,
        positions=[767, 768],
        raw_q=torch.tensor([[[1.0, 2.0]], [[9.0, 9.0]]]),
        post_q=torch.tensor([[[3.5, 4.0]], [[9.0, 9.0]]]),
    )

    result = diff_q_stage_captures(
        base=base, candidate=candidate, position=767, layer=0
    )

    assert result["summary"] == {
        "num_candidate_rows": 2,
        "raw_q_exact_rows": 1,
        "post_q_exact_rows": 1,
    }
    assert result["comparisons"][0]["raw_q"]["num_diff"] == 1
    assert result["comparisons"][0]["raw_q"]["max_abs"] == 0.5
    assert result["comparisons"][0]["post_q"]["num_diff"] == 0
    assert result["comparisons"][1]["raw_q"]["num_diff"] == 0
    assert result["comparisons"][1]["post_q"]["num_diff"] == 1
    assert result["comparisons"][1]["post_q"]["max_abs"] == 0.5


def test_diff_q_stage_captures_requires_matching_candidate(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _capture(
        base / "rank0_layer0_call0_q_stages.pt",
        positions=[767],
        raw_q=torch.zeros(1, 1, 2),
        post_q=torch.zeros(1, 1, 2),
    )
    _capture(
        candidate / "rank0_layer0_call0_q_stages.pt",
        positions=[768],
        raw_q=torch.zeros(1, 1, 2),
        post_q=torch.zeros(1, 1, 2),
    )

    with pytest.raises(ValueError, match="no candidate rows"):
        diff_q_stage_captures(
            base=base, candidate=candidate, position=767, layer=0
        )
