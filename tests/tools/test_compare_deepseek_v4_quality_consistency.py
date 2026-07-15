import hashlib
import json

import pytest

from tools.debug.compare_deepseek_v4_quality_consistency import (
    compare_quality_consistency,
)


def _artifact(
    path,
    questions,
    *,
    batch_size,
    max_tokens=512,
    status="completed",
):
    payload = {
        "schema_version": 1,
        "status": status,
        "manifest": {
            "model": "/model",
            "model_provenance": {
                "metadata_sha256": {"config.json": "c" * 64},
                "weight_files": [
                    {"name": "weights.safetensors", "size_bytes": 1, "sha256": "a" * 64}
                ],
                "weight_hash_complete": True,
            },
            "tensor_parallel_size": 4,
            "mtp": 0,
            "num_speculative_tokens": 0,
            "cudagraph_mode": "PIECEWISE",
            "enforce_eager": False,
            "num_shots": 5,
            "max_tokens": max_tokens,
            "seed": 42,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.0,
            "prefix_caching": True,
            "prompt_format": "gsm8k_completion",
            "sampling_params": {
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "min_tokens": 1,
                "seed": 42,
                "stop": ["\nQuestion:"],
            },
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [1, batch_size],
            },
            "completion_api": "LLM.generate",
            "profiler": False,
            "evaluator_source_sha256": "source-sha",
            "git_head": "git-head",
            "git_diff_sha256": "git-diff-sha",
            "environment": {
                "VLLM_METAX_DSV4_MHC_BACKEND": "tilelang",
                "VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS": "1",
                "VLLM_METAX_DSV4_MHC_EXACT_POST_MMA": "1",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
            },
            "batch_size": batch_size,
        },
        "data": {
            "train_sha256": "b" * 64,
            "test_sha256": hashlib.sha256(path.name.encode()).hexdigest(),
        },
        "questions": questions,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _question(prompt_hash, token_ids, *, finish_reason="stop", zero_tail=None):
    return {
        "question": f"question-{prompt_hash}",
        "prompt_hash": prompt_hash,
        "raw_output": "answer",
        "token_ids": token_ids,
        "token_zero_tail_start": zero_tail,
        "finish_reason": finish_reason,
        "expected_answer": "1",
        "predicted_answer": "1",
        "correct": True,
    }


def test_compare_quality_consistency_reports_exact_routes(tmp_path):
    q1 = _question("p1", [10, 11, 12])
    q2 = _question("p2", [20, 21])
    fresh_1 = _artifact(tmp_path / "fresh-1.json", [q1], batch_size=1)
    fresh_2 = _artifact(tmp_path / "fresh-2.json", [q2], batch_size=1)
    continuous = _artifact(
        tmp_path / "continuous.json", [q1, q2], batch_size=1
    )
    batched = _artifact(tmp_path / "batched.json", [q1, q2], batch_size=2)

    result = compare_quality_consistency(
        fresh_artifacts=[fresh_1, fresh_2],
        continuous_artifact=continuous,
        batched_artifact=batched,
    )

    assert result["decision"] == "pass"
    assert result["summary"] == {
        "num_prompts": 2,
        "num_exact_continuous": 2,
        "num_exact_batched": 2,
        "num_passed": 2,
    }
    assert all(row["passed"] for row in result["comparisons"])
    assert result["manifest_compatibility"]["compatible"] is True


def test_compare_quality_consistency_reports_first_diff_and_zero_tail(tmp_path):
    q1 = _question("p1", [10, 11, 12])
    divergent = _question("p1", [10, 99, 0, 0], zero_tail=2)
    fresh = _artifact(tmp_path / "fresh.json", [q1], batch_size=1)
    continuous = _artifact(tmp_path / "continuous.json", [q1], batch_size=1)
    batched = _artifact(tmp_path / "batched.json", [divergent], batch_size=2)

    result = compare_quality_consistency(
        fresh_artifacts=[fresh],
        continuous_artifact=continuous,
        batched_artifact=batched,
    )

    assert result["decision"] == "fail"
    row = result["comparisons"][0]
    assert row["batched"]["exact"] is False
    assert row["batched"]["first_diff"] == 1
    assert row["batched"]["token_zero_tail_start"] == 2
    assert row["passed"] is False


def test_compare_quality_consistency_rejects_manifest_mismatch(tmp_path):
    q1 = _question("p1", [10])
    fresh = _artifact(tmp_path / "fresh.json", [q1], batch_size=1)
    continuous = _artifact(tmp_path / "continuous.json", [q1], batch_size=1)
    batched = _artifact(
        tmp_path / "batched.json", [q1], batch_size=2, max_tokens=256
    )

    with pytest.raises(ValueError, match="max_tokens"):
        compare_quality_consistency(
            fresh_artifacts=[fresh],
            continuous_artifact=continuous,
            batched_artifact=batched,
        )


def test_compare_quality_consistency_rejects_missing_question_fields(tmp_path):
    malformed = _question("p1", [10])
    del malformed["finish_reason"]
    fresh = _artifact(tmp_path / "fresh.json", [malformed], batch_size=1)
    continuous = _artifact(
        tmp_path / "continuous.json", [_question("p1", [10])], batch_size=1
    )
    batched = _artifact(
        tmp_path / "batched.json", [_question("p1", [10])], batch_size=2
    )

    with pytest.raises(ValueError, match="finish_reason"):
        compare_quality_consistency(
            fresh_artifacts=[fresh],
            continuous_artifact=continuous,
            batched_artifact=batched,
        )


def test_compare_quality_consistency_fails_matching_length_terminations(tmp_path):
    truncated = _question("p1", [10, 11], finish_reason="length")
    fresh = _artifact(tmp_path / "fresh.json", [truncated], batch_size=1)
    continuous = _artifact(
        tmp_path / "continuous.json", [truncated], batch_size=1
    )
    batched = _artifact(tmp_path / "batched.json", [truncated], batch_size=2)

    result = compare_quality_consistency(
        fresh_artifacts=[fresh],
        continuous_artifact=continuous,
        batched_artifact=batched,
    )

    assert result["decision"] == "fail"
    assert result["comparisons"][0]["passed"] is False


def test_compare_quality_consistency_rejects_evaluator_provenance_mismatch(tmp_path):
    q1 = _question("p1", [10])
    fresh = _artifact(tmp_path / "fresh.json", [q1], batch_size=1)
    continuous = _artifact(tmp_path / "continuous.json", [q1], batch_size=1)
    batched = _artifact(tmp_path / "batched.json", [q1], batch_size=2)
    payload = json.loads(batched.read_text(encoding="utf-8"))
    payload["manifest"]["evaluator_source_sha256"] = "different-source"
    batched.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator_source_sha256"):
        compare_quality_consistency(
            fresh_artifacts=[fresh],
            continuous_artifact=continuous,
            batched_artifact=batched,
        )


def test_compare_quality_consistency_rejects_answer_label_mismatch(tmp_path):
    q1 = _question("p1", [10])
    changed_label = _question("p1", [10])
    changed_label["expected_answer"] = "2"
    fresh = _artifact(tmp_path / "fresh.json", [q1], batch_size=1)
    continuous = _artifact(tmp_path / "continuous.json", [q1], batch_size=1)
    batched = _artifact(tmp_path / "batched.json", [changed_label], batch_size=2)

    with pytest.raises(ValueError, match="expected_answer"):
        compare_quality_consistency(
            fresh_artifacts=[fresh],
            continuous_artifact=continuous,
            batched_artifact=batched,
        )
