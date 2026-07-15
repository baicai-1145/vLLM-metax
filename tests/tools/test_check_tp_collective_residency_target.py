"""Fast residency-target gate tests for analyzer collectives output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.debug.check_tp_collective_residency_target import check_residency_target


def _write_collectives(
    path: Path,
    *,
    ranks: tuple[int, ...] = (0, 1, 2, 3),
    calls: int = 435,
    residency_us: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for _ in range(calls):
            stream.write(
                json.dumps(
                    {
                        "ranks": [
                            {"rank": rank, "residency_us": residency_us} for rank in ranks
                        ]
                    }
                )
                + "\n"
            )
    return path


def test_passes_when_all_four_ranks_are_under_target(tmp_path: Path) -> None:
    result = check_residency_target([_write_collectives(tmp_path / "run1" / "collectives.jsonl")])

    assert result["ok"] is True
    assert result["runs"][0]["complete_four_ranks"] is True
    assert all(item["total_residency_ms"] < 5.0 for item in result["runs"][0]["ranks"])


def test_fails_and_lists_rank_value_over_target(tmp_path: Path) -> None:
    result = check_residency_target(
        [_write_collectives(tmp_path / "run1" / "collectives.jsonl", residency_us=20.0)]
    )

    assert result["ok"] is False
    assert result["failures"] == [
        {"run": "run1", "rank": rank, "total_residency_ms": 8.7, "threshold_ms": 5.0}
        for rank in range(4)
    ]


def test_fails_when_a_rank_is_missing(tmp_path: Path) -> None:
    result = check_residency_target(
        [_write_collectives(tmp_path / "run1" / "collectives.jsonl", ranks=(0, 1, 2))]
    )

    assert result["ok"] is False
    run = result["runs"][0]
    assert run["complete_four_ranks"] is False
    assert run["missing_ranks"] == [3]
    assert any(item["reason"] == "missing-rank" for item in run["diagnostics"])


def test_fails_when_call_count_is_not_435(tmp_path: Path) -> None:
    result = check_residency_target(
        [_write_collectives(tmp_path / "run1" / "collectives.jsonl", calls=434)]
    )

    assert result["ok"] is False
    assert any(item["reason"] == "unexpected-call-count" for item in result["runs"][0]["diagnostics"])


def test_cli_emits_json_and_nonzero_for_real_red_run() -> None:
    path = ".logs/plan35_mccl_rank_skew_20260714/run1/analysis/collectives.jsonl"
    completed = subprocess.run(
        [sys.executable, "tools/debug/check_tp_collective_residency_target.py", path],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    summary = json.loads(completed.stdout)
    assert summary["ok"] is False
    assert {item["rank"] for item in summary["failures"]} == {0, 1, 2}
    assert all(item["total_residency_ms"] > 5.0 for item in summary["failures"])
