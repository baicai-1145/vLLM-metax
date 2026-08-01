import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.debug.compare_deepseek_v4_mtp_consistency import compare_mtp_consistency


def test_mtp_comparator_cli_loads_from_outside_repo(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/debug/compare_deepseek_v4_mtp_consistency.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _artifact(path, *, mtp, token_ids, batch_size=1):
    question = {
        "question": "q0",
        "prompt_hash": "prompt-0",
        "raw_output": "answer",
        "token_ids": token_ids,
        "token_zero_tail_start": None,
        "finish_reason": "stop",
        "expected_answer": "1",
        "predicted_answer": "1",
        "correct": True,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "manifest": {
            "model": "/model",
            "model_provenance": {
                "metadata_sha256": {"config.json": "c" * 64},
                "weight_files": [{"name": "w.safetensors", "sha256": "a" * 64}],
                "weight_hash_complete": True,
            },
            "tensor_parallel_size": 4,
            "mtp": mtp,
            "num_speculative_tokens": mtp,
            "cudagraph_mode": "PIECEWISE",
            "enforce_eager": False,
            "num_shots": 5,
            "max_tokens": 16,
            "seed": 42,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.0,
            "prefix_caching": True,
            "prompt_format": "gsm8k_completion",
            "sampling_params": {"temperature": 0.0, "max_tokens": 16},
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [1, batch_size],
            },
            "completion_api": "LLM.generate",
            "profiler": False,
            "evaluator_source_sha256": "source-sha",
            "git_head": "git-head",
            "git_diff_sha256": "git-diff-sha",
            "environment": {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
            "batch_size": batch_size,
        },
        "data": {"train_sha256": "b" * 64, "test_sha256": hashlib.sha256(b"test").hexdigest()},
        "questions": [question],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_mtp_comparison_fails_at_first_divergent_token(tmp_path):
    base = _artifact(tmp_path / "base.json", mtp=0, token_ids=[10, 11, 12])
    candidate = _artifact(tmp_path / "k1.json", mtp=1, token_ids=[10, 99, 12])

    result = compare_mtp_consistency([base], candidate)

    assert result["decision"] == "fail"
    assert result["comparisons"][0]["first_diff"] == 1


def test_mtp_comparison_rejects_unrelated_manifest_difference(tmp_path):
    base = _artifact(tmp_path / "base.json", mtp=0, token_ids=[10])
    candidate = _artifact(tmp_path / "k1.json", mtp=1, token_ids=[10])
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["manifest"]["seed"] = 99
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    try:
        compare_mtp_consistency([base], candidate)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("expected unrelated manifest difference to fail")


def test_mtp_comparison_accepts_repeated_identical_base_runs(tmp_path):
    base = _artifact(tmp_path / "base.json", mtp=0, token_ids=[10, 11])
    base_repeat = _artifact(tmp_path / "base-repeat.json", mtp=0, token_ids=[10, 11])
    candidate = _artifact(tmp_path / "k1.json", mtp=1, token_ids=[10, 11])

    result = compare_mtp_consistency([base, base_repeat], candidate)

    assert result["decision"] == "pass"
    assert result["summary"]["num_prompts"] == 1


def test_mtp_comparison_reports_divergent_base_run_as_failure(tmp_path):
    base = _artifact(tmp_path / "base.json", mtp=0, token_ids=[10, 11])
    divergent_base = _artifact(
        tmp_path / "base-divergent.json", mtp=0, token_ids=[10, 99]
    )
    candidate = _artifact(tmp_path / "k1.json", mtp=1, token_ids=[10, 11])

    result = compare_mtp_consistency([base, divergent_base], candidate)

    assert result["decision"] == "fail"
    assert result["base_comparisons"][0]["comparisons"][0]["first_diff"] == 1
