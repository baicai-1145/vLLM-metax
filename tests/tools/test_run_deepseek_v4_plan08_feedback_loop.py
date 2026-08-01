import json
from pathlib import Path

import tools.debug.run_deepseek_v4_plan08_feedback_loop as feedback_loop
from tools.debug.run_deepseek_v4_plan08_feedback_loop import (
    build_verdict,
    choose_next_action,
    render_decision_markdown,
    sparse_diff_focus,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sparse_diff_focus_promotes_gathered_swa_rows(tmp_path):
    diff_path = tmp_path / "sparse_diff.json"
    _write_json(
        diff_path,
        {
            "decision": "fail",
            "first_difference": "gathered_swa_rows",
            "comparisons": [
                {
                    "key": "q",
                    "kind": "tensor",
                    "exact": False,
                    "num_diff": 5,
                    "max_abs": 0.001953125,
                },
                {
                    "key": "swa_indices",
                    "kind": "tensor",
                    "exact": False,
                    "num_diff": 106,
                    "max_abs": 64,
                    "first_diff_index": [0, 0],
                    "base_value": 11606,
                    "candidate_value": 11542,
                },
                {
                    "key": "gathered_swa_rows",
                    "kind": "tensor",
                    "exact": False,
                    "num_diff": 12,
                    "max_abs": 0.00048828125,
                    "first_diff_index": [18, 86],
                    "base_value": -0.055908203125,
                    "candidate_value": -0.055419921875,
                },
            ],
        },
    )

    result = sparse_diff_focus(diff_path)

    assert result == {
        "available": True,
        "path": str(diff_path),
        "decision": "fail",
        "first_difference": "gathered_swa_rows",
        "first_attention_input_difference": None,
        "gathered_swa_rows": {
            "kind": "tensor",
            "exact": False,
            "num_diff": 12,
            "max_abs": 0.00048828125,
            "first_diff_index": [18, 86],
            "base_value": -0.055908203125,
            "candidate_value": -0.055419921875,
        },
        "q": {
            "kind": "tensor",
            "exact": False,
            "num_diff": 5,
            "max_abs": 0.001953125,
            "first_diff_index": None,
            "base_value": None,
            "candidate_value": None,
        },
        "swa_indices": {
            "kind": "tensor",
            "exact": False,
            "num_diff": 106,
            "max_abs": 64,
            "first_diff_index": [0, 0],
            "base_value": 11606,
            "candidate_value": 11542,
        },
    }


def test_choose_next_action_prioritizes_swa_metadata_when_q_is_exact():
    decision, next_action = choose_next_action(
        token_decision="fail",
        sparse_decision="fail",
        sparse={
            "q": {"exact": True},
            "swa_indices": {"exact": False},
            "gathered_swa_rows": {"exact": False},
        },
    )

    assert decision == "fail"
    assert next_action == "audit_swa_metadata_slot_mapping"


def test_choose_next_action_moves_to_cache_rows_when_mapping_is_consistent():
    decision, next_action = choose_next_action(
        token_decision="fail",
        sparse_decision="fail",
        sparse={
            "q": {"exact": True},
            "swa_indices": {"exact": False},
            "swa_block_table": {"exact": False},
            "gathered_swa_rows": {"exact": False},
        },
        swa_mapping={
            "available": True,
            "decision": "fail",
            "first_differences": [
                {
                    "base": {"slot_matches_block_table": True},
                    "candidate": {"slot_matches_block_table": True},
                },
            ],
        },
    )

    assert decision == "fail"
    assert next_action == "compare_swa_cache_rows_by_logical_position"


def test_build_verdict_keeps_red_gate_and_next_action(tmp_path):
    root = tmp_path / "artifact"
    _write_json(
        root / "comparison.json",
        {
            "decision": "fail",
            "first_mismatch": 134,
            "oracle_window": [343],
            "candidate_window": [1527],
        },
    )
    _write_json(
        root / "debug_loop_summary.json",
        {
            "next_focus": "diff_mtp_capture_against_oracle_or_add_next_boundary_probe",
            "capture": {"mtp_record_count": 8},
        },
    )
    _write_json(
        root / "sparse_diff.json",
        {
            "decision": "pass",
            "comparisons": [
                {"key": "gathered_swa_rows", "exact": True},
            ],
        },
    )

    verdict = build_verdict(root, token_gate_exit_code=1, sparse_diff_path=root / "sparse_diff.json")

    assert verdict["decision"] == "fail"
    assert verdict["token_gate"]["first_mismatch"] == 134
    assert verdict["sparse_diff"]["gathered_swa_rows"]["exact"] is True
    assert verdict["next_action"] == "move_probe_boundary_past_sparse_decode_inputs"
    assert verdict["gates"] == {
        "k1_exact": "fail",
        "k4_correctness_unblocked": False,
        "performance_unblocked": False,
        "stage": "blocked_on_k1_exact",
    }
    assert verdict["feedback_loop"]["red_capable"] is True
    assert verdict["feedback_loop"]["rerun_command"] == [
        "python",
        "tools/debug/run_deepseek_v4_plan08_feedback_loop.py",
        "--output-root",
        str(root),
    ]


def test_build_verdict_includes_swa_mapping_analysis(tmp_path):
    root = tmp_path / "artifact"
    _write_json(root / "comparison.json", {"decision": "fail"})
    _write_json(
        root / "sparse_diff.json",
        {
            "decision": "fail",
            "comparisons": [
                {"key": "q", "kind": "tensor", "exact": True},
                {"key": "swa_indices", "kind": "tensor", "exact": False},
            ],
        },
    )
    _write_json(
        root / "swa_mapping_analysis.json",
        {
            "decision": "fail",
            "position": 789,
            "window_width": 128,
            "swa_block_size": 64,
            "num_swa_index_differences": 106,
            "num_block_table_differences": 2,
            "block_table_differences": [
                {"logical_block": 10, "base_block": 181, "candidate_block": 180},
            ],
            "first_differences": [
                {"logical_position": 662, "slot_delta": -64},
            ],
        },
    )

    verdict = build_verdict(
        root,
        token_gate_exit_code=1,
        sparse_diff_path=root / "sparse_diff.json",
        swa_mapping_analysis_path=root / "swa_mapping_analysis.json",
    )

    assert verdict["next_action"] == "audit_swa_metadata_slot_mapping"
    assert verdict["swa_mapping_analysis"]["position"] == 789
    assert verdict["swa_mapping_analysis"]["num_swa_index_differences"] == 106


def test_swa_mapping_analysis_error_does_not_block_verdict(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifact"
    base_root = tmp_path / "base"
    base = base_root / "sparse_capture/rank0_call2.pt"
    candidate = root / "sparse_capture/rank0_call2.pt"
    base.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    base.write_bytes(b"base")
    candidate.write_bytes(b"candidate")
    _write_json(root / "comparison.json", {"decision": "fail"})
    _write_json(root / "debug_loop_summary.json", {"next_focus": "continue"})

    def raise_shape_mismatch(base_path, candidate_path):
        raise ValueError("base table width 64 != candidate table width 16")

    monkeypatch.setattr(
        feedback_loop,
        "analyze_swa_mapping_pair",
        raise_shape_mismatch,
    )

    analysis_path = feedback_loop._write_swa_mapping_analysis(
        root,
        base_output_root=base_root,
        base_sparse_call="sparse_capture/rank0_call2.pt",
        candidate_sparse_call="sparse_capture/rank0_call2.pt",
    )
    verdict = build_verdict(
        root,
        token_gate_exit_code=1,
        sparse_diff_path=None,
        swa_mapping_analysis_path=analysis_path,
    )

    assert analysis_path is not None
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["decision"] == "error"
    assert analysis["error_type"] == "ValueError"
    assert verdict["decision"] == "fail"
    assert verdict["swa_mapping_analysis"]["decision"] == "error"
    assert "table width" in verdict["swa_mapping_analysis"]["error"]


def test_build_verdict_unblocks_k4_only_after_k1_pass(tmp_path):
    root = tmp_path / "artifact"
    _write_json(
        root / "comparison.json",
        {
            "decision": "pass",
            "oracle_token_count": 212,
            "candidate_token_count": 212,
        },
    )
    _write_json(root / "debug_loop_summary.json", {"next_focus": "advance"})

    verdict = build_verdict(root, token_gate_exit_code=0, sparse_diff_path=None)

    assert verdict["decision"] == "pass"
    assert verdict["gates"] == {
        "k1_exact": "pass",
        "k4_correctness_unblocked": True,
        "performance_unblocked": False,
        "stage": "ready_for_k4_correctness",
    }
    assert verdict["next_action"] == "advance_to_piecewise_k4_and_perf_gates"


def test_render_decision_markdown_reports_blocked_stage(tmp_path):
    root = tmp_path / "artifact"
    verdict = {
        "decision": "fail",
        "next_action": "compare_swa_cache_rows_by_logical_position",
        "output_root": str(root),
        "gates": {
            "k1_exact": "fail",
            "k4_correctness_unblocked": False,
            "performance_unblocked": False,
            "stage": "blocked_on_k1_exact",
        },
        "token_gate": {
            "decision": "fail",
            "first_mismatch": 156,
            "oracle_window": [34593],
            "candidate_window": [47593],
            "oracle_token_count": 212,
            "candidate_token_count": 220,
        },
        "sparse_diff": {"available": False},
        "swa_mapping_analysis": {"available": False},
        "feedback_loop": {
            "red_capable": True,
            "rerun_command": [
                "python",
                "tools/debug/run_deepseek_v4_plan08_feedback_loop.py",
                "--output-root",
                str(root),
            ],
        },
    }

    markdown = render_decision_markdown(verdict)

    assert "Stage: blocked_on_k1_exact" in markdown
    assert "k=4 correctness: blocked" in markdown
    assert "performance: blocked" in markdown
    assert "First mismatch: 156" in markdown
    assert "compare_swa_cache_rows_by_logical_position" in markdown
