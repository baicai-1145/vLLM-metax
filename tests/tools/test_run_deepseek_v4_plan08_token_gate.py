import argparse
import json
from pathlib import Path

import pytest

from tools.debug import run_deepseek_v4_plan08_token_gate as token_gate


def _summary(path: Path, *, question: str, prompt_hash: str, token_ids: list[int]):
    payload = {
        "status": "completed",
        "summary": {"total_output_tokens": len(token_ids)},
        "questions": [
            {
                "question": question,
                "prompt_hash": prompt_hash,
                "token_ids": token_ids,
                "finish_reason": "stop",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_candidate_to_oracle_reports_first_mismatch_window(tmp_path):
    oracle = {
        "question": "Melanie?",
        "prompt_hash": "p0",
        "token_ids": [10, 11, 12, 13, 14],
        "finish_reason": "stop",
    }
    candidate = _summary(
        tmp_path / "candidate.json",
        question="Melanie?",
        prompt_hash="p0",
        token_ids=[10, 11, 99, 13],
    )

    result = token_gate.compare_candidate_to_oracle(
        oracle, candidate, window_radius=1
    )

    assert result["decision"] == "fail"
    assert result["first_mismatch"] == 2
    assert result["oracle_window"] == [11, 12, 13]
    assert result["candidate_window"] == [11, 99, 13]
    assert result["oracle_token_count"] == 5
    assert result["candidate_token_count"] == 4


def test_compare_candidate_to_oracle_accepts_exact_match(tmp_path):
    oracle = {
        "question": "Melanie?",
        "prompt_hash": "p0",
        "token_ids": [10, 11],
        "finish_reason": "stop",
    }
    candidate = _summary(
        tmp_path / "candidate.json",
        question="Melanie?",
        prompt_hash="p0",
        token_ids=[10, 11],
    )

    result = token_gate.compare_candidate_to_oracle(oracle, candidate)

    assert result["decision"] == "pass"
    assert result["first_mismatch"] is None


def test_compare_candidate_to_oracle_accepts_requested_exact_prefix(tmp_path):
    oracle = {
        "question": "Melanie?",
        "prompt_hash": "p0",
        "token_ids": [10, 11, 12, 13, 14],
        "finish_reason": "stop",
    }
    candidate = _summary(
        tmp_path / "candidate.json",
        question="Melanie?",
        prompt_hash="p0",
        token_ids=[10, 11, 12],
    )
    payload = json.loads(candidate.read_text())
    payload["questions"][0]["finish_reason"] = "length"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    result = token_gate.compare_candidate_to_oracle(
        oracle,
        candidate,
        required_prefix_tokens=3,
    )

    assert result["decision"] == "pass"
    assert result["exact"] is True
    assert result["comparison_mode"] == "prefix"
    assert result["first_mismatch"] is None
    assert result["candidate_token_count"] == 3
    assert result["finish_reason_match"] is False


def test_compare_candidate_to_oracle_rejects_short_prefix(tmp_path):
    oracle = {
        "question": "Melanie?",
        "prompt_hash": "p0",
        "token_ids": [10, 11, 12, 13],
        "finish_reason": "stop",
    }
    candidate = _summary(
        tmp_path / "candidate.json",
        question="Melanie?",
        prompt_hash="p0",
        token_ids=[10, 11],
    )

    result = token_gate.compare_candidate_to_oracle(
        oracle,
        candidate,
        required_prefix_tokens=3,
    )

    assert result["decision"] == "fail"
    assert result["first_mismatch"] == 2


def test_write_single_question_fixture_uses_source_answer(tmp_path):
    oracle_question = {
        "question": "Melanie?",
        "expected_answer": "18",
    }
    source = tmp_path / "test.jsonl"
    source.write_text(
        json.dumps({"question": "Other", "answer": "#### 1"}) + "\n"
        + json.dumps({"question": "Melanie?", "answer": "work\n#### 18"}) + "\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "selected.jsonl"

    answer = token_gate.matching_test_answer(source, oracle_question)
    token_gate.write_single_question_fixture(fixture, oracle_question, answer)

    [row] = token_gate.load_jsonl(fixture)
    assert row == {"question": "Melanie?", "answer": "work\n#### 18"}


def test_matching_test_answer_falls_back_to_oracle_answer(tmp_path):
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps({"question": "Other", "answer": "#### 1"}) + "\n")

    assert token_gate.matching_test_answer(
        source,
        {"question": "Melanie?", "expected_answer": "18"},
    ) == "#### 18"


def test_build_evaluator_command_defaults_to_single_tp4_eager(tmp_path):
    args = argparse.Namespace(
        model="/model",
        train_data=tmp_path / "train.jsonl",
        num_shots=5,
        max_tokens=512,
        num_speculative_tokens=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        diagnostic_enforce_eager=True,
        cudagraph_mode="PIECEWISE",
    )

    command = token_gate.build_evaluator_command(
        args, tmp_path / "selected.jsonl", tmp_path / "summary.json"
    )

    assert "--test-data" in command
    assert str(tmp_path / "selected.jsonl") in command
    assert command[command.index("--num-questions") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert command[command.index("--num-speculative-tokens") + 1] == "1"
    assert "--diagnostic-enforce-eager" in command


def test_parse_env_overrides_rejects_malformed_values():
    assert token_gate.parse_env_overrides(["A=1", "B="]) == {"A": "1", "B": ""}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        token_gate.parse_env_overrides(["BAD"])


def test_default_exact_env_preserves_full_acceptance_accounting():
    assert "VLLM_METAX_MTP_DROP_UNVERIFIED_BONUS" not in token_gate.DEFAULT_EXACT_ENV


def test_plan08_debug_loop_env_sets_artifact_dirs(tmp_path):
    env = {}
    applied = token_gate.apply_plan08_debug_loop_env(env, tmp_path)
    assert applied["VLLM_METAX_DSV4_MTP_CAPTURE_DIR"] == str(
        tmp_path / "mtp_capture"
    )
    assert env["VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_POSITIONS"] == "767"
    assert env["VLLM_METAX_DSV4_QKV_INSERT_CAPTURE_DIR"] == str(
        tmp_path / "qkv_capture"
    )
    assert env["VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_DIR"] == str(
        tmp_path / "compressor_capture"
    )
    assert env["VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_POSITIONS"] == "767"


def test_plan08_debug_loop_env_preserves_explicit_values(tmp_path):
    env = {"VLLM_METAX_DSV4_MTP_CAPTURE_DIR": "/existing"}
    applied = token_gate.apply_plan08_debug_loop_env(env, tmp_path)
    assert applied["VLLM_METAX_DSV4_MTP_CAPTURE_DIR"] == "/existing"
    assert env["VLLM_METAX_DSV4_MTP_CAPTURE_DIR"] == "/existing"
