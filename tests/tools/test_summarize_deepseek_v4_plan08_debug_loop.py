import json
from pathlib import Path

from tools.debug.summarize_deepseek_v4_plan08_debug_loop import (
    summarize_debug_loop,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_record(root: Path, payload: object) -> None:
    capture = root / "mtp_capture"
    capture.mkdir(parents=True, exist_ok=True)
    with (capture / "rank0.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def test_summarize_debug_loop_reports_gate_and_capture_counts(tmp_path):
    _write_json(
        tmp_path / "comparison.json",
        {
            "decision": "fail",
            "first_mismatch": 134,
            "oracle_window": [212],
            "candidate_window": [210],
        },
    )
    _write_json(tmp_path / "token_gate_command.json", {"plan08_debug_loop": True})
    _append_record(tmp_path, {"stage": "verify"})
    _append_record(tmp_path, {"stage": "commit"})
    compressor = tmp_path / "compressor_capture"
    compressor.mkdir()
    (compressor / "rank0_layer0_call0_compressor.pt").write_bytes(b"stub")
    result = summarize_debug_loop(tmp_path)
    assert result["token_gate"]["decision"] == "fail"
    assert result["token_gate"]["first_mismatch"] == 134
    assert result["capture"]["mtp_record_count"] == 2
    assert result["capture"]["compressor_file_count"] == 1
    assert result["capture"]["mtp_stage_counts"] == {"verify": 1, "commit": 1}
    assert result["next_focus"] == (
        "diff_mtp_capture_against_oracle_or_add_next_boundary_probe"
    )


def test_summarize_debug_loop_flags_rejected_non_padding_slot(tmp_path):
    _write_json(tmp_path / "comparison.json", {"decision": "fail"})
    _append_record(
        tmp_path,
        {
            "stage": "v1_proposer_first_pass",
            "is_rejected_token_mask": [False, True, False],
            "after_slot_mapping": [10, 99, 12],
            "positions_after": [100, 101, 102],
        },
    )
    result = summarize_debug_loop(tmp_path)
    assert result["state_flow_findings"]["first_rejected_non_padding_slot"] == {
        "record_index": 0,
        "token_index": 1,
        "position": 101,
        "slot": 99,
    }
    assert result["next_focus"] == "audit_rejected_token_slot_mapping_and_cache_writes"
