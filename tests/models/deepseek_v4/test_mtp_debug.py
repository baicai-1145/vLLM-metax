import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from vllm_metax.models.deepseek_v4.mtp_debug import (
    analyze_capture,
    maybe_capture_mtp_stage,
    maybe_capture_greedy_verifier,
    maybe_capture_greedy_verifier_batch,
    maybe_capture_v1_model_runner_metadata,
    maybe_capture_v1_proposer_first_pass,
    maybe_capture_v1_proposer_prepare_inputs_padded,
    maybe_capture_v1_sampler_metadata,
    maybe_capture_v1_greedy_verifier_batch,
)


def test_mtp_diff_cli_loads_from_outside_repo(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[3] / "tools/debug/diff_deepseek_v4_mtp.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_mtp_capture_is_opt_in(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", raising=False)

    assert maybe_capture_mtp_stage(
        "before_input_rms", 0, {"input_ids": torch.tensor([1, 2])}
    ) is None
    assert not list(tmp_path.iterdir())


def test_mtp_capture_schema_and_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    maybe_capture_mtp_stage(
        "after_logits",
        2,
        {
            "input_ids": torch.tensor([1, 2]),
            "positions": torch.tensor([8, 9]),
            "pre_hidden_states": torch.ones((2, 4), dtype=torch.float32),
            "post_hidden_states": torch.zeros((2, 4), dtype=torch.float32),
            "logits": torch.tensor([[1.0, 4.0, 2.0]]),
            "accepted_count": 0,
        },
    )

    records = [json.loads(line) for line in (tmp_path / "rank3.jsonl").read_text().splitlines()]
    record = records[0]
    assert record["rank"] == 3
    assert record["step"] == 2
    assert record["stage"] == "after_logits"
    assert record["tensor_meta"]["pre_hidden_states"]["shape"] == [2, 4]
    assert record["tensor_meta"]["pre_hidden_states"]["stride"] == [4, 1]
    assert record["input_ids"] == [1, 2]
    assert record["positions"] == [8, 9]
    assert record["accepted_count"] == 0
    assert record["pre_hidden_hash"]
    assert record["post_hidden_hash"]
    assert record["logits_top_k"][0][0][0] == 1


def test_mtp_capture_fails_closed_during_graph_capture(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert maybe_capture_mtp_stage("after_logits", 0, {"input_ids": torch.tensor([1])}) is None
    assert not list(tmp_path.iterdir())


def test_greedy_verifier_capture_k1_rejected(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    maybe_capture_greedy_verifier(
        request_idx=0,
        draft_ids=torch.tensor([11]),
        target_ids=torch.tensor([12]),
        accepted_count=0,
        committed_ids=torch.tensor([12]),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert [record["stage"] for record in records] == ["verify", "commit"]
    assert records[0]["request_idx"] == 0
    assert records[0]["draft_ids"] == [11]
    assert records[0]["target_ids"] == [12]
    assert records[0]["accepted_count"] == 0
    assert records[1]["committed_ids"] == [12]


def test_greedy_verifier_capture_k1_accepted(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    maybe_capture_greedy_verifier(
        request_idx=0,
        draft_ids=torch.tensor([11]),
        target_ids=torch.tensor([11]),
        accepted_count=1,
        committed_ids=torch.tensor([11, 12]),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert [record["stage"] for record in records] == ["verify", "commit"]
    assert records[0]["request_idx"] == 0
    assert records[0]["draft_ids"] == [11]
    assert records[0]["target_ids"] == [11]
    assert records[0]["accepted_count"] == 1
    assert records[1]["committed_ids"] == [11, 12]


def test_greedy_verifier_batch_groups_common_wrapper_tensors(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    target_logits = torch.eye(32)[[12, 22, 21, 31]]
    draft_sampled = torch.tensor([0, 11, 0, 21])
    cu_num_logits = torch.tensor([0, 2, 4], dtype=torch.int32)
    sampled = torch.tensor([[12, -1], [21, 22]])
    num_sampled = torch.tensor([1, 2], dtype=torch.int32)

    maybe_capture_greedy_verifier_batch(
        target_logits,
        draft_sampled,
        cu_num_logits,
        sampled,
        num_sampled,
        positions=torch.tensor([100, 101, 200, 201]),
        expanded_local_pos=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([4, 4, 9, 9], dtype=torch.int32),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "verify",
        "commit",
        "verify",
        "commit",
    ]
    assert [records[index]["accepted_count"] for index in (0, 2)] == [0, 1]
    assert records[0]["draft_ids"] == [11]
    assert records[2]["draft_ids"] == [21]
    assert records[0]["row_indices"] == [0, 1]
    assert records[0]["positions"] == [100, 101]
    assert records[0]["expanded_local_pos"] == [0, 1]
    assert records[0]["expanded_idx_mapping"] == [4, 4]
    assert records[0]["cu_num_logits"] == [0, 2, 4]
    assert records[1]["committed_ids"] == [12]
    assert records[3]["committed_ids"] == [21, 22]


def test_v1_greedy_verifier_batch_groups_cumulative_drafts(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    target_logits = torch.eye(32)[[12, 21]]
    draft_token_ids = torch.tensor([11, 21])
    cu_num_draft_tokens = torch.tensor([1, 2], dtype=torch.int32)
    sampled = torch.tensor([[12, -1], [21, 22]])

    maybe_capture_v1_greedy_verifier_batch(
        target_logits, draft_token_ids, cu_num_draft_tokens, sampled
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert [records[index]["accepted_count"] for index in (0, 2)] == [0, 1]
    assert records[1]["committed_ids"] == [12]
    assert records[3]["committed_ids"] == [21, 22]


def test_v1_sampler_metadata_capture_records_row_indices(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([11, 21], dtype=torch.int32),
        cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2, 4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 2], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([1, 3], dtype=torch.int64),
        logits_indices=torch.tensor([10, 11, 20, 21], dtype=torch.int64),
    )
    logits = torch.eye(32)[[12, 22, 21, 31]]

    maybe_capture_v1_sampler_metadata(metadata, logits)

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["stage"] == "v1_sampler_metadata"
    assert record["draft_ids"] == [11, 21]
    assert record["target_logits_indices"] == [0, 2]
    assert record["bonus_logits_indices"] == [1, 3]
    assert record["logits_indices"] == [10, 11, 20, 21]
    assert record["target_model_row_indices"] == [10, 20]
    assert record["bonus_model_row_indices"] == [11, 21]
    assert record["cu_num_logits"] == [2, 4]
    assert record["cu_num_draft_tokens"] == [1, 2]
    assert [row[0][0] for row in record["logits_top_k"]] == [12, 21]


def test_v1_model_runner_metadata_capture_records_positions(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([11, 21], dtype=torch.int32),
        cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2, 4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 2], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([1, 3], dtype=torch.int64),
        logits_indices=torch.tensor([10, 11, 20, 21], dtype=torch.int64),
    )
    input_ids = torch.arange(64, dtype=torch.int32) + 1000
    positions = torch.arange(64, dtype=torch.int64) + 600
    logits = torch.eye(32)[[12, 22, 21, 31]]

    maybe_capture_v1_model_runner_metadata(metadata, input_ids, positions, logits)

    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    record = records[0]
    assert record["stage"] == "v1_model_runner_metadata"
    assert record["input_ids"] == [1010, 1011, 1020, 1021]
    assert record["positions"] == [610, 611, 620, 621]
    assert record["target_input_ids"] == [1010, 1020]
    assert record["target_positions"] == [610, 620]
    assert record["bonus_input_ids"] == [1011, 1021]
    assert record["bonus_positions"] == [611, 621]


def test_v1_proposer_first_pass_capture_records_state_flow(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cad = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([102], dtype=torch.int32),
        slot_mapping=torch.tensor([10, -1, 12], dtype=torch.int64),
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
        num_actual_tokens=3,
        max_query_len=3,
        max_seq_len=102,
        num_reqs=1,
    )
    proposer = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=1,
        needs_extra_input_slots=True,
        extra_slots_per_request=1,
        net_num_new_slots_per_request=1,
        input_ids=torch.tensor([1000, 1001, 1002], dtype=torch.int32),
        positions=torch.tensor([100, 101, 102], dtype=torch.int64),
        is_rejected_token_mask=torch.tensor([False, True, False]),
        is_masked_token_mask=torch.tensor([False, False, True]),
    )

    maybe_capture_v1_proposer_first_pass(
        proposer,
        target_token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        next_token_ids=torch.tensor([4], dtype=torch.int32),
        target_positions=torch.tensor([100, 101, 102], dtype=torch.int64),
        token_indices_to_sample=torch.tensor([2], dtype=torch.int32),
        before_cad=cad,
        after_cad=cad,
        num_rejected_tokens=torch.tensor([1], dtype=torch.int32),
        num_tokens=3,
    )

    record = json.loads((tmp_path / "rank0.jsonl").read_text().splitlines()[0])
    assert record["stage"] == "v1_proposer_first_pass"
    assert record["method"] == "mtp"
    assert record["target_ids"] == [1, 2, 3]
    assert record["next_token_ids"] == [4]
    assert record["input_ids_after"] == [1000, 1001, 1002]
    assert record["positions_after"] == [100, 101, 102]
    assert record["is_rejected_token_mask"] == [False, True, False]
    assert record["after_slot_mapping"] == [10, -1, 12]


def test_v1_proposer_prepare_inputs_padded_capture_records_counts(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cad = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        slot_mapping=torch.tensor([8, 9], dtype=torch.int64),
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
    )

    maybe_capture_v1_proposer_prepare_inputs_padded(
        before_cad=cad,
        after_cad=cad,
        valid_sampled_tokens_count=torch.tensor([1], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1], dtype=torch.int32),
        num_rejected_tokens=torch.tensor([1], dtype=torch.int32),
    )

    record = json.loads((tmp_path / "rank0.jsonl").read_text().splitlines()[0])
    assert record["stage"] == "v1_proposer_prepare_inputs_padded"
    assert record["valid_sampled_tokens_count"] == [1]
    assert record["token_indices_to_sample"] == [1]
    assert record["num_rejected_tokens"] == [1]


def _write_stage(path: Path, stage: str, **fields: object) -> None:
    record = {"schema_version": 1, "rank": 0, "step": 0, "stage": stage, **fields}
    with (path / "rank0.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def test_mtp_diff_reports_propose_verify_commit_boundary(tmp_path: Path):
    _write_stage(tmp_path, "propose", draft_ids=[11], target_ids=[12])
    _write_stage(tmp_path, "verify", draft_ids=[11], target_ids=[12], accepted_count=0)
    _write_stage(tmp_path, "commit", committed_ids=[11])

    result = analyze_capture(tmp_path)

    assert result["first_invalid_boundary"] == "commit"
