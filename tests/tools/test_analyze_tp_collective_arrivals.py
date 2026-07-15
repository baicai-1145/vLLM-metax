"""Synthetic tests for the four-rank collective arrival analyzer."""

from __future__ import annotations

import json
import gzip
from pathlib import Path

from tools.debug.analyze_tp_collective_arrivals import (
    _operator_category,
    analyze_traces,
    map_dsv4_ordinal,
    write_outputs,
)


def _write_trace(
    directory: Path,
    rank: int,
    graph_start: float,
    *,
    collective_offset: float = 20,
    collective_duration: float = 10,
    preceding: bool = False,
    graph_id: str = "decode",
    nested: bool = False,
    extra_collective: bool = False,
    calibration_offset: float | None = None,
    calibration_marker: str | None = None,
) -> Path:
    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "execute_context_1",
            "ts": graph_start - 100,
            "dur": 500,
            "args": {"execute_context": 1},
        },
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": "graph_parent",
            "ts": graph_start,
            "dur": 100,
            "args": {"execute_context": 1, "graph_parent": graph_id},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "all_reduce",
            "ts": graph_start + collective_offset,
            "dur": collective_duration,
            "tid": 7,
            "args": {"execute_context": 1, "graph_parent": graph_id, "ordinal": 0},
        },
    ]
    if preceding:
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "MHC_route_op",
                "ts": graph_start,
                "dur": 5,
                "tid": 7,
                "args": {"stream": 7},
            }
        )
    if nested:
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce_kernel_child",
                "ts": graph_start + 22,
                "dur": 6,
                "tid": 7,
                "args": {"execute_context": 1, "graph_parent": graph_id, "ordinal": 0},
            }
        )
    if extra_collective:
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce_extra",
                "ts": graph_start + 50,
                "dur": 8,
                "tid": 7,
                "args": {"execute_context": 1, "graph_parent": graph_id, "ordinal": 1},
            }
        )
    path = directory / f"trace_rank{rank}.json"
    payload = {"traceEvents": events}
    if calibration_offset is not None or calibration_marker is not None:
        payload["metadata"] = {
            "clock_offset_us": calibration_offset,
            "calibration_marker": calibration_marker,
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_equal_graph_relative_arrivals(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]

    result = analyze_traces(paths)

    collective = result["collectives"][0]
    assert collective["arrival_skew_us"] == 0
    assert collective["relative_start_us"] == [20, 20, 20, 20]
    assert collective["late_rank_candidates"] == [0, 1, 2, 3]
    assert collective["late_rank_tie"] is True
    assert collective["late_rank_candidate"] is None
    assert result["summary"]["late_rank_tie_count"] == 1


def test_one_late_rank_is_reported(tmp_path: Path) -> None:
    paths = [
        _write_trace(tmp_path, rank, 1000 + rank * 1000, collective_offset=35 if rank == 3 else 20)
        for rank in range(4)
    ]

    collective = analyze_traces(paths)["collectives"][0]

    assert collective["arrival_skew_us"] == 15
    assert collective["late_rank_candidate"] == 3


def test_long_completion_tail_is_reported(tmp_path: Path) -> None:
    paths = [
        _write_trace(tmp_path, rank, 1000 + rank * 1000, collective_duration=40 if rank == 2 else 10)
        for rank in range(4)
    ]

    collective = analyze_traces(paths)["collectives"][0]

    assert collective["completion_tail"] == 40


def test_graph_gap_attributes_preceding_operator(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True) for rank in range(4)]

    member = analyze_traces(paths)["collectives"][0]["ranks"][0]

    assert member["pre_gap_us"] == 15
    assert member["preceding_operator_category"] == "MHC"


def test_constant_clock_offsets_are_removed_by_graph_anchor(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]

    result = analyze_traces(paths)
    collective = result["collectives"][0]

    assert collective["raw_start_us"] == [1020, 2020, 3020, 4020]
    assert collective["relative_start_us"] == [20, 20, 20, 20]
    assert result["summary"]["absolute_alignment"] == "unavailable"


def test_mismatched_graph_anchors_are_inconclusive(tmp_path: Path) -> None:
    paths = [
        _write_trace(tmp_path, rank, 1000 + rank * 1000, graph_id="other" if rank == 3 else "decode")
        for rank in range(4)
    ]

    result = analyze_traces(paths)

    assert result["summary"]["complete_four_rank_match"] is False
    assert any(item["reason"] == "mismatched-graph-anchor" for item in result["summary"]["unmatched_diagnostics"])


def test_nested_parent_child_collective_is_not_double_counted(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000, nested=True) for rank in range(4)]

    result = analyze_traces(paths)

    assert len(result["collectives"]) == 1
    assert result["collectives"][0]["nested_parent_child_collapsed"] is True


def test_missing_rank_is_rejected_with_diagnostic(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(3)]

    result = analyze_traces(paths)

    assert result["summary"]["complete_four_rank_match"] is False
    assert result["summary"]["missing_ranks"] == [3]


def test_mismatched_collective_counts_are_rejected(tmp_path: Path) -> None:
    paths = [
        _write_trace(tmp_path, rank, 1000 + rank * 1000, extra_collective=rank < 3)
        for rank in range(4)
    ]

    result = analyze_traces(paths)

    assert result["summary"]["complete_four_rank_match"] is False
    assert any(item["reason"] == "mismatched-collective-count" for item in result["summary"]["unmatched_diagnostics"])


def test_absolute_calibration_residual_over_two_us_is_rejected(tmp_path: Path) -> None:
    paths = [
        _write_trace(
            tmp_path,
            rank,
            1000 + rank * 1000,
            calibration_offset=float(rank * 3),
            calibration_marker="shared",
        )
        for rank in range(4)
    ]

    result = analyze_traces(paths)

    assert result["summary"]["absolute_alignment"] == "rejected-residual-over-2us"


def test_gzip_input_and_cli_outputs(tmp_path: Path) -> None:
    plain = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]
    compressed = []
    for path in plain:
        target = path.with_suffix(".json.gz")
        with path.open("rb") as source, gzip.open(target, "wb") as destination:
            destination.write(source.read())
        compressed.append(target)

    result = analyze_traces(compressed)
    write_outputs(result, tmp_path / "out")

    assert (tmp_path / "out" / "collectives.jsonl").is_file()
    assert (tmp_path / "out" / "summary.json").is_file()
    assert (tmp_path / "out" / "summary.csv").is_file()


def test_gpu_graph_anchor_wins_over_host_range(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "host_graph_parent",
                "ts": payload["traceEvents"][1]["ts"] - 50,
                "dur": 500,
                "args": {"execute_context": 1, "graph_parent": "decode"},
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

    collective = analyze_traces(paths)["collectives"][0]

    assert collective["relative_start_us"] == [20, 20, 20, 20]


def test_correlation_chain_selects_host_step_before_overlapping_gpu_parent(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        offset = rank * 1000
        events = [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "graph_parent_g0",
                "ts": offset + 1000,
                "dur": 101,
                "args": {"execute_context": 1, "graph_parent": "g0"},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "graph_parent_g1",
                "ts": offset + 1080,
                "dur": 150,
                "args": {"execute_context": 1, "graph_parent": "g1"},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_0",
                "ts": offset + 900,
                "dur": 160,
                "args": {"execute_context": 1, "graph_parent": "g0"},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_1",
                "ts": offset + 1060,
                "dur": 140,
                "args": {"execute_context": 1, "graph_parent": "g1"},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "mcGraphLaunch",
                "ts": offset + 1010,
                "dur": 5,
                "args": {"correlation": 100 + rank, "External id": "g0"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce_g0",
                "ts": offset + 1020,
                "dur": 10,
                "args": {"stream": 7, "correlation": 100 + rank, "ordinal": 0},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "mcGraphLaunch",
                "ts": offset + 1090,
                "dur": 5,
                "args": {"correlation": 200 + rank, "External id": "g1"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce_g1",
                "ts": offset + 1100,
                "dur": 10,
                "args": {"stream": 7, "correlation": 200 + rank, "ordinal": 0},
            },
        ]
        path = tmp_path / f"overlap_chain_rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=2, expected_collectives_per_step=1)

    assert result["summary"]["routing_valid"] is True
    assert [record["graph_parent"] for record in result["collectives"]] == ["g0", "g1"]
    assert result["collectives"][1]["relative_start_us"] == [20, 20, 20, 20]
    assert all(member["enclosing_graph"] == "graph_parent_g1" for member in result["collectives"][1]["ranks"])


def test_overlapping_gpu_parents_without_unique_host_chain_fail_closed(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        offset = rank * 1000
        events = [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "graph_parent_g0",
                "ts": offset + 1000,
                "dur": 101,
                "args": {"execute_context": 1, "graph_parent": "g0"},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "graph_parent_g1",
                "ts": offset + 1080,
                "dur": 150,
                "args": {"execute_context": 1, "graph_parent": "g1"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce",
                "ts": offset + 1100,
                "dur": 10,
                "args": {"stream": 7, "ordinal": 0},
            },
        ]
        path = tmp_path / f"ambiguous_parent_rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=2, expected_collectives_per_step=1)

    assert result["summary"]["routing_valid"] is False
    assert any(item["reason"] == "ambiguous-parent" for item in result["summary"]["unmatched_diagnostics"])


def test_real_mccl_kernel_schema_uses_enclosing_gpu_execute_parent(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        graph_start = 1000 + rank * 1000
        payload = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "execute_context_0(0)_generation_1(1)",
                    "ts": graph_start,
                    "dur": 100,
                    "args": {"External id": "step-0"},
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "mcclKernel_AllReduce_FC4_LL_Sum_mccl_bfloat16(fcinfo)",
                    "ts": graph_start + 20,
                    "dur": 10,
                    "args": {"stream": 7, "device": rank, "correlation": 100 + rank},
                },
            ]
        }
        path = tmp_path / f"real_rank{rank}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths)

    assert result["summary"]["complete_four_rank_match"] is True
    assert result["collectives"][0]["relative_start_us"] == [20, 20, 20, 20]


def test_top_level_distributed_info_rank_is_supported(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        path = _write_trace(tmp_path, rank, 1000 + rank * 1000)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["distributedInfo"] = {"rank": rank}
        target = tmp_path / f"worker_{rank}.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(target)

    result = analyze_traces(paths)

    assert result["ranks"] == [0, 1, 2, 3]


def test_expected_count_gate_is_configurable_without_changing_match_status(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)

    assert result["summary"]["complete_four_rank_match"] is True
    assert result["summary"]["routing_valid"] is True
    assert result["summary"]["expected"]["total_collectives"] == 1
    assert result["summary"]["per_rank_collective_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}


def test_summary_contains_auditable_metric_aggregates(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True) for rank in range(4)]

    summary = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)["summary"]

    assert summary["aggregates"]["arrival_skew_us"]["median"] == 0
    assert summary["aggregates"]["arrival_skew_us"]["p90"] == 0
    assert summary["aggregates"]["residency_us_by_rank"]["0"]["median"] == 10
    assert summary["aggregates"]["late_rank_distribution"] == {}
    assert summary["aggregates"]["preceding_category_distribution"] == {"MHC": 4}


def test_duplicate_rank_trace_fails_closed(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000) for rank in range(4)]
    duplicate = _write_trace(tmp_path, 0, 9000)

    result = analyze_traces(paths + [duplicate], expected_steps=1, expected_collectives_per_step=1)

    assert result["summary"]["routing_valid"] is False
    assert any(item["reason"] == "duplicate-rank-trace" for item in result["summary"]["unmatched_diagnostics"])


def test_five_step_external_ids_match_by_step_position(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        events = []
        for step in range(5):
            graph_start = 1000 + step * 200 + rank * 10000
            events.append(
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "execute_context_0(0)_generation_1(1)",
                    "ts": graph_start,
                    "dur": 100,
                    "args": {"External id": f"step-{step}"},
                }
            )
            for ordinal in range(2):
                events.append(
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "mcclKernel_AllReduce_FC4_LL_Sum_mccl_bfloat16(fcinfo)",
                        "ts": graph_start + 20 + ordinal * 20,
                        "dur": 10,
                        "args": {"stream": 7, "device": rank, "correlation": step * 10 + ordinal},
                    }
                )
        path = tmp_path / f"five_rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=5, expected_collectives_per_step=2)

    assert result["summary"]["routing_valid"] is True
    assert result["summary"]["collective_count"] == 10
    assert [record["step_index"] for record in result["collectives"]] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_correlation_chain_attributes_preceding_attention_host_range(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        graph_start = 1000 + rank * 1000
        events = [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "execute_context_0(0)_generation_1(1)",
                "ts": graph_start,
                "dur": 100,
                "args": {"External id": "step-0"},
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "attention.py::forward",
                "ts": graph_start,
                "dur": 25,
                "args": {"External id": "host-attn"},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "mcGraphLaunch",
                "ts": graph_start + 5,
                "dur": 5,
                "args": {"correlation": 42, "External id": "host-attn"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "opaque_kernel",
                "ts": graph_start + 10,
                "dur": 5,
                "args": {"stream": 7, "correlation": 42},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce",
                "ts": graph_start + 20,
                "dur": 10,
                "args": {"stream": 7, "ordinal": 0},
            },
        ]
        path = tmp_path / f"chain_rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)
    member = result["collectives"][0]["ranks"][0]

    assert member["preceding_category"] == "attention"
    assert member["preceding_host_chain"]["host"]["name"] == "attention.py::forward"


def test_opaque_b16gemvt_needs_host_attention_evidence() -> None:
    assert _operator_category("void b16gemvt_kernel<256>") == "other"


def test_all_unique_same_step_graph_ids_fail_closed(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000, graph_id=f"g{rank}") for rank in range(4)]

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)

    assert result["summary"]["routing_valid"] is False
    assert any(item["reason"] == "mismatched-graph-anchor" for item in result["summary"]["unmatched_diagnostics"])


def test_unlinked_or_partial_attention_host_range_is_not_attributed(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        graph_start = 1000 + rank * 1000
        events = [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "execute_context_0(0)_generation_1(1)",
                "ts": graph_start,
                "dur": 100,
                "args": {"External id": "step-0"},
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "attention.py::forward",
                "ts": graph_start + 5,
                "dur": 8,
                "args": {"External id": "unrelated"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "opaque_kernel",
                "ts": graph_start + 10,
                "dur": 10,
                "args": {"stream": 7, "correlation": 99},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "all_reduce",
                "ts": graph_start + 25,
                "dur": 10,
                "args": {"stream": 7, "ordinal": 0},
            },
        ]
        path = tmp_path / f"unlinked_rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)
    member = result["collectives"][0]["ranks"][0]

    assert member["preceding_category"] in {"other", "ambiguous"}


def test_attention_stage_annotation_overrides_host_chain(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        path = _write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_start = payload["traceEvents"][1]["ts"]
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "plan35.attention_o_proj",
                "ts": graph_start,
                "dur": 6,
                "args": {"External id": "stage-attention"},
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)
    member = result["collectives"][0]["ranks"][0]

    assert member["preceding_category"] == "attention"
    assert member["preceding_stage_annotation"]["name"] == "plan35.attention_o_proj"
    assert member["preceding_stage_family"] == "attention_o_proj"
    assert result["summary"]["aggregates"]["stage_annotation_distribution"] == {
        "plan35.attention_o_proj": 4
    }


def test_mlp_stage_annotation_preserves_prefix_without_moe_routing(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        path = _write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_start = payload["traceEvents"][1]["ts"]
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "plan35.mlp_down_proj:shared_experts",
                "ts": graph_start,
                "dur": 6,
                "args": {},
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    member = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)["collectives"][0]["ranks"][0]

    assert member["preceding_category"] == "other"
    assert member["preceding_stage_family"] == "mlp_down_proj"
    assert member["preceding_stage_prefix"] == "shared_experts"


def test_overlapping_stage_annotations_are_ambiguous(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        path = _write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_start = payload["traceEvents"][1]["ts"]
        payload["traceEvents"].extend(
            [
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "plan35.attention_o_proj",
                    "ts": graph_start,
                    "dur": 6,
                    "args": {},
                },
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "plan35.mlp_down_proj:ambiguous",
                    "ts": graph_start + 4,
                    "dur": 4,
                    "args": {},
                },
            ]
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    member = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)["collectives"][0]["ranks"][0]

    assert member["preceding_category"] == "other"
    assert member["preceding_stage_annotation"]["ambiguous"] is True
    assert member["preceding_stage_family"] is None


def test_user_annotation_stage_range_is_trusted(tmp_path: Path) -> None:
    paths = []
    for rank in range(4):
        path = _write_trace(tmp_path, rank, 1000 + rank * 1000, preceding=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_start = payload["traceEvents"][1]["ts"]
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "plan35.attention_o_proj",
                "ts": graph_start,
                "dur": 6,
                "args": {"External id": "stage-attention"},
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=1)
    member = result["collectives"][0]["ranks"][0]

    assert member["preceding_category"] == "attention"
    assert member["preceding_stage_annotation"]["cat"] == "user_annotation"
    assert result["summary"]["aggregates"]["stage_annotation_distribution"] == {
        "plan35.attention_o_proj": 4
    }


def test_dsv4_ordinal_stage_boundaries() -> None:
    assert map_dsv4_ordinal(0) == ("embedding", None)
    assert map_dsv4_ordinal(1) == ("attention", 0)
    assert map_dsv4_ordinal(2) == ("ffn", 0)
    assert map_dsv4_ordinal(85) == ("attention", 42)
    assert map_dsv4_ordinal(86) == ("ffn", 42)
    assert map_dsv4_ordinal(87) == ("unknown", None)


def test_inter_collective_span_math_and_stage_aggregates(tmp_path: Path) -> None:
    paths = [
        _write_trace(tmp_path, rank, 1000 + rank * 1000, extra_collective=True)
        for rank in range(4)
    ]

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=2)
    first, second = result["collectives"]

    assert first["semantic_stage"] == "embedding"
    assert first["inter_collective_span_us"] == [20, 20, 20, 20]
    assert first["span_skew_us"] == 0
    assert second["semantic_stage"] == "attention"
    assert second["layer"] == 0
    assert second["inter_collective_span_us"] == [20, 20, 20, 20]
    assert second["span_skew_us"] == 0
    assert result["summary"]["stage_aggregates"]["attention"]["span_skew_us"]["median"] == 0
    assert result["summary"]["stage_aggregates"]["attention"]["top_span_skew_ordinals"][0]["ordinal"] == 1


def test_completion_tail_interval_and_union_do_not_double_count_overlap(tmp_path: Path) -> None:
    paths = [_write_trace(tmp_path, rank, 1000 + rank * 1000, extra_collective=True) for rank in range(4)]

    result = analyze_traces(paths, expected_steps=1, expected_collectives_per_step=2)
    records = result["collectives"]

    assert all(len(record["completion_tail_interval_us"]) == 2 for record in records)
    assert result["summary"]["completion_tail_union_us"] <= sum(
        record["completion_tail_interval_us"][1] - record["completion_tail_interval_us"][0]
        for record in records
    )
    assert result["summary"]["active_device_window_us"] == 100
