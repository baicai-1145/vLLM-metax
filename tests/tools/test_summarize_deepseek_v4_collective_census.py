import json

import pytest

from tools.debug.summarize_deepseek_v4_collective_census import (
    read_jsonl,
    summarize,
)


def test_summary_is_stable_and_aggregates_counts():
    records = [
        {
            "layer_idx": 2,
            "projection": "wo_b",
            "rows": 1,
            "expects_reduce": True,
        },
        {
            "layer_idx": 0,
            "projection": "wq_b",
            "rows": 2,
            "expects_reduce": False,
        },
        {
            "layer_idx": 2,
            "projection": "ffn",
            "rows": 4,
            "expects_reduce": True,
        },
        {
            "layer_idx": 0,
            "projection": "wq_b",
            "rows": 1,
            "expects_reduce": False,
        },
    ]
    assert summarize(records) == [
        {
            "layer_idx": 0,
            "projection": "wq_b",
            "invocations": 2,
            "rows": 3,
            "expected_collectives": 0,
        },
        {
            "layer_idx": 2,
            "projection": "ffn",
            "invocations": 1,
            "rows": 4,
            "expected_collectives": 1,
        },
        {
            "layer_idx": 2,
            "projection": "wo_b",
            "invocations": 1,
            "rows": 1,
            "expected_collectives": 1,
        },
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"layer_idx": 0},
        [],
        {
            "layer_idx": -1,
            "projection": "x",
            "rows": 1,
            "expects_reduce": True,
        },
    ],
)
def test_summary_rejects_malformed_records(value):
    with pytest.raises(ValueError):
        summarize([value])


def test_read_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "layer_idx": 0,
                "projection": "x",
                "rows": 1,
                "expects_reduce": True,
            }
        )
        + "\nnot-json\n"
    )
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)
