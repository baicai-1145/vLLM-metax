from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import threading
from types import SimpleNamespace
import pytest

from tools.debug.evaluate_deepseek_v4_quality import (
    build_prompt,
    load_jsonl,
    sample_questions,
    sha256_file,
    parse_numeric_answer,
    evaluate_quality,
    build_arg_parser,
    main,
    _write_artifact_atomic,
)


def test_parse_numeric_answer_prefers_gsm8k_marker_and_supports_decimal_formats():
    assert parse_numeric_answer("work 10, then #### -1,234.50") == Decimal("-1234.50")
    assert parse_numeric_answer("The answer is 12.5") == Decimal("12.5")


def test_parse_numeric_answer_falls_back_to_last_number_and_marks_invalid():
    assert parse_numeric_answer("first 2, finally 3") == Decimal("3")
    assert parse_numeric_answer("No numeric answer") is None


def test_load_jsonl_and_sha256(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "# comment\n"
        + json.dumps({"question": "q1", "answer": "#### 1"})
        + "\n"
        + json.dumps({"question": "q2", "answer": "#### 2"})
        + "\n",
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert rows == [{"question": "q1", "answer": "#### 1"}, {"question": "q2", "answer": "#### 2"}]
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sample_questions_is_seed_deterministic_and_prompt_is_five_shot():
    train = [{"question": f"train-{i}", "answer": f"#### {i}"} for i in range(6)]
    test = [{"question": f"test-{i}", "answer": f"#### {i}"} for i in range(8)]
    assert sample_questions(test, 3, seed=7) == sample_questions(test, 3, seed=7)
    prompt = build_prompt(test[0]["question"], train, num_shots=5)
    assert prompt.count("Question:") == 6
    assert prompt.count("Answer:") == 6
    assert prompt.endswith("Question: test-0\nAnswer:")
    assert "Question: train-0\nAnswer: #### 0\n\n" in prompt
    assert "Answer:  ####" not in prompt


def test_evaluate_quality_constructs_once_generates_batches_and_writes_artifact(tmp_path):
    train = [{"question": f"train-{i}", "answer": f"#### {i}"} for i in range(5)]
    test = [
        {"question": "q0", "answer": "#### 1"},
        {"question": "q1", "answer": "#### 2"},
        {"question": "q2", "answer": "#### 3"},
    ]

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, params, use_tqdm=False):
            self.calls.append((list(prompts), params, use_tqdm))
            base = sum(len(call[0]) for call in self.calls[:-1])
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=f"#### {base + index + 1}",
                            token_ids=[10 + base + index],
                            finish_reason="stop",
                        )
                    ]
                )
                for index, _ in enumerate(prompts)
            ]

    instances = []

    def factory(**kwargs):
        instance = FakeLLM()
        instances.append((instance, kwargs))
        return instance

    artifact_path = tmp_path / "result.json"
    result = evaluate_quality(
        train_examples=train,
        test_examples=test,
        model="fake",
        artifact_path=artifact_path,
        num_questions=3,
        batch_size=2,
        seed=42,
        llm_factory=factory,
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    assert len(instances) == 1
    assert len(instances[0][0].calls) == 2
    assert all(call[2] is False for call in instances[0][0].calls)
    assert result["summary"]["num_questions"] == 3
    assert result["summary"]["num_correct"] == 3
    assert result["summary"]["num_length_terminated"] == 0
    assert result["status"] == "completed"
    assert len(result["questions"]) == 3
    assert result["questions"][0]["prompt_hash"]
    assert result["questions"][0]["token_zero_tail_start"] is None
    assert result["summary"]["num_token_zero_tails"] == 0
    assert result["schema_version"] == 1
    assert result["manifest"]["prompt_format"] == "gsm8k_completion"
    assert instances[0][0].calls[0][1]["stop"] == [
        "\nQuestion:",
        "Assistant:",
        "<|separator|>",
    ]
    assert result["manifest"]["sampling_params"] == {
        "temperature": 0.0,
        "max_tokens": 256,
        "min_tokens": 1,
        "seed": 42,
        "stop": ["\nQuestion:", "Assistant:", "<|separator|>"],
    }
    assert result["manifest"]["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2],
    }
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["manifest"]["tensor_parallel_size"] == 4


def test_evaluate_quality_passes_prefix_cache_setting_to_llm_and_manifest(tmp_path):
    llm_kwargs = {}
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_bytes = b'{"model_type":"test"}\n'
    tokenizer_bytes = b'{"tokenizer_class":"Test"}\n'
    (model_path / "config.json").write_bytes(config_bytes)
    (model_path / "tokenizer_config.json").write_bytes(tokenizer_bytes)
    (model_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="#### 1", token_ids=[7], finish_reason="stop")
                    ]
                )
                for _ in prompts
            ]

    def factory(**kwargs):
        llm_kwargs.update(kwargs)
        return FakeLLM()

    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model=str(model_path),
        artifact_path=tmp_path / "disabled-prefix-cache.json",
        enable_prefix_caching=False,
        llm_factory=factory,
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    assert llm_kwargs["enable_prefix_caching"] is False
    assert result["manifest"]["prefix_caching"] is False
    manifest = result["manifest"]
    assert len(manifest["evaluator_source_sha256"]) == 64
    assert manifest["git_diff_sha256"] is None or len(manifest["git_diff_sha256"]) == 64
    assert isinstance(manifest["git_status_porcelain"], str)
    provenance = manifest["model_provenance"]
    assert provenance["path"] == str(model_path)
    assert provenance["metadata_sha256"] == {
        "config.json": hashlib.sha256(config_bytes).hexdigest(),
        "tokenizer_config.json": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    assert provenance["weight_files"] == [
        {
            "name": "model-00001-of-00001.safetensors",
            "size_bytes": 7,
            "sha256": hashlib.sha256(b"weights").hexdigest(),
        }
    ]


def test_evaluate_quality_records_requested_sample_logprobs(tmp_path):
    observed_sampling = {}

    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            observed_sampling.update(params)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="#### 1",
                            token_ids=[10],
                            finish_reason="stop",
                            logprobs=[
                                {
                                    10: SimpleNamespace(
                                        logprob=-0.1, rank=1, decoded_token="one"
                                    ),
                                    11: SimpleNamespace(
                                        logprob=float("-inf"),
                                        rank=2,
                                        decoded_token="two",
                                    ),
                                }
                            ],
                        )
                    ]
                )
                for _ in prompts
            ]

    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model="fake",
        artifact_path=tmp_path / "logprobs.json",
        num_logprobs=2,
        logprob_token_ids=[10, 11],
        llm_factory=lambda **kwargs: FakeLLM(),
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert observed_sampling["logprobs"] == 2
    assert observed_sampling["logprob_token_ids"] == [10, 11]
    assert result["questions"][0]["sample_logprobs"] == [
        [
            {"token_id": 10, "logprob": -0.1, "rank": 1, "decoded_token": "one"},
            {
                "token_id": 11,
                "logprob": "-Infinity",
                "rank": 2,
                "decoded_token": "two",
            },
        ]
    ]
    artifact_text = (tmp_path / "logprobs.json").read_text(encoding="utf-8")
    json.loads(
        artifact_text,
        parse_constant=lambda value: pytest.fail(f"non-standard JSON: {value}"),
    )


def test_write_artifact_atomic_supports_concurrent_writers(
    monkeypatch, tmp_path
):
    target = tmp_path / "artifact.json"
    barrier = threading.Barrier(2)
    real_replace = os.replace

    def synchronized_replace(source, destination):
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(
        "tools.debug.evaluate_deepseek_v4_quality.os.replace",
        synchronized_replace,
    )
    payloads = [{"writer": 1}, {"writer": 2}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_write_artifact_atomic, target, payload)
            for payload in payloads
        ]
        for future in futures:
            future.result(timeout=5)

    assert json.loads(target.read_text(encoding="utf-8")) in payloads


def test_evaluate_quality_rejects_mismatched_requested_logprobs_before_model_load(
    tmp_path,
):
    with pytest.raises(ValueError, match="must equal"):
        evaluate_quality(
            train_examples=[{"question": "example", "answer": "#### 1"}],
            test_examples=[{"question": "q0", "answer": "#### 1"}],
            model="fake",
            artifact_path=tmp_path / "invalid-logprobs.json",
            num_logprobs=5,
            logprob_token_ids=[10, 11],
            llm_factory=lambda **kwargs: pytest.fail("model must not be loaded"),
            sampling_params_factory=lambda **kwargs: kwargs,
        )


def test_cli_has_fixed_acceptance_defaults_and_explicit_data_artifact_options():
    args = build_arg_parser().parse_args(
        ["--train-data", "train.jsonl", "--test-data", "test.jsonl"]
    )
    assert args.tensor_parallel_size == 4
    assert args.num_speculative_tokens == 0
    assert args.cudagraph_mode == "PIECEWISE"
    assert args.batch_size == 1
    assert args.num_questions == 20
    assert args.artifact == "deepseek_v4_quality.json"
    assert args.enable_prefix_caching is True
    disabled_args = build_arg_parser().parse_args(
        [
            "--train-data",
            "train.jsonl",
            "--test-data",
            "test.jsonl",
            "--disable-prefix-caching",
        ]
    )
    assert disabled_args.enable_prefix_caching is False


def test_evaluate_quality_records_mtp_speculative_config_and_capture_sizes(tmp_path):
    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="#### 1", token_ids=[7], finish_reason="stop")
                    ]
                )
                for _ in prompts
            ]

    kwargs = {}
    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model="fake",
        artifact_path=tmp_path / "mtp.json",
        num_speculative_tokens=4,
        llm_factory=lambda **values: (kwargs.update(values) or FakeLLM()),
        sampling_params_factory=lambda **values: values,
    )

    assert kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 4,
    }
    assert result["manifest"]["mtp"] == 4
    assert result["manifest"]["speculative_config"] == kwargs["speculative_config"]
    assert result["manifest"]["compilation_config"]["cudagraph_capture_sizes"] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_evaluate_quality_snapshots_speculative_manifest_before_llm_mutation(tmp_path):
    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="#### 1", token_ids=[7], finish_reason="stop"
                        )
                    ]
                )
                for _ in prompts
            ]

    def mutating_factory(**values):
        values["speculative_config"]["draft_model_config"] = object()
        return FakeLLM()

    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model="fake",
        artifact_path=tmp_path / "mtp-mutated.json",
        num_speculative_tokens=1,
        llm_factory=mutating_factory,
        sampling_params_factory=lambda **values: values,
    )

    assert result["manifest"]["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    json.dumps(result, allow_nan=False)


def test_evaluate_quality_allows_explicit_eager_diagnosis(tmp_path):
    class FakeLLM:
        def generate(self, prompts, params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="#### 1", token_ids=[7], finish_reason="stop")
                    ]
                )
                for _ in prompts
            ]

    kwargs = {}
    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model="fake",
        artifact_path=tmp_path / "eager.json",
        cudagraph_mode="NONE",
        diagnostic_enforce_eager=True,
        llm_factory=lambda **values: (kwargs.update(values) or FakeLLM()),
        sampling_params_factory=lambda **values: values,
    )

    assert kwargs["enforce_eager"] is True
    assert "compilation_config" not in kwargs
    assert result["manifest"]["enforce_eager"] is True
    assert result["manifest"]["cudagraph_mode"] == "NONE"
    assert result["manifest"]["compilation_config"] is None


def test_checkpoint_artifact_records_runtime_context_and_failed_batch(monkeypatch, tmp_path):
    for name, value in {
        "VLLM_METAX_DSV4_MHC_BACKEND": "fused",
        "VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS": "1",
        "VLLM_METAX_DSV4_MHC_EXACT_POST_MMA": "0",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "VLLM_METAX_USE_FP32_LOGITS": "1",
        "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND": "torch_reference",
        "VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING": "1",
    }.items():
        monkeypatch.setenv(name, value)

    class FailingLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompts, params, use_tqdm=False):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic decode failure")
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="#### 1", token_ids=[7], finish_reason="stop")
                    ]
                )
                for _ in prompts
            ]

    artifact_path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        evaluate_quality(
            train_examples=[{"question": "example", "answer": "#### 1"}],
            test_examples=[
                {"question": "q0", "answer": "#### 1"},
                {"question": "q1", "answer": "#### 2"},
            ],
            model="fake",
            artifact_path=artifact_path,
            batch_size=1,
            llm_factory=lambda **kwargs: FailingLLM(),
            sampling_params_factory=lambda **kwargs: kwargs,
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert len(artifact["questions"]) == 1
    assert artifact["error"]["type"] == "RuntimeError"
    manifest = artifact["manifest"]
    assert manifest["environment"]["VLLM_METAX_DSV4_MHC_BACKEND"] == "fused"
    assert manifest["environment"]["VLLM_METAX_USE_FP32_LOGITS"] == "1"
    assert (
        manifest["environment"]["VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND"]
        == "torch_reference"
    )
    assert manifest["environment"]["VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING"] == "1"
    assert "git_head" in manifest and "git_dirty" in manifest


def test_evaluate_quality_marks_repeated_token_zero_tail_as_runtime_blocked(tmp_path):
    class ZeroTailLLM:
        def generate(self, prompts, params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="The answer is 1",
                            token_ids=[7, 8, *([0] * 8)],
                            finish_reason="length",
                        )
                    ]
                )
                for _ in prompts
            ]

    result = evaluate_quality(
        train_examples=[{"question": "example", "answer": "#### 1"}],
        test_examples=[{"question": "q0", "answer": "#### 1"}],
        model="fake",
        artifact_path=tmp_path / "zero-tail.json",
        llm_factory=lambda **kwargs: ZeroTailLLM(),
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert result["status"] == "runtime_blocked"
    assert result["summary"]["num_token_zero_tails"] == 1
    assert result["summary"]["num_length_terminated"] == 1
    assert result["questions"][0]["token_zero_tail_start"] == 2


def test_main_returns_nonzero_for_runtime_blocker(monkeypatch, tmp_path, capsys):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    row = json.dumps({"question": "q", "answer": "#### 1"}) + "\n"
    train_path.write_text(row, encoding="utf-8")
    test_path.write_text(row, encoding="utf-8")
    monkeypatch.setattr(
        "tools.debug.evaluate_deepseek_v4_quality.evaluate_quality",
        lambda **kwargs: {
            "status": "runtime_blocked",
            "summary": {
                "accuracy": 0.0,
                "invalid_rate": 1.0,
                "num_questions": 1,
            },
        },
    )

    exit_code = main(
        [
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--artifact",
            str(tmp_path / "result.json"),
        ]
    )

    assert exit_code == 2
    assert "status=runtime_blocked" in capsys.readouterr().out


def test_data_hash_failure_writes_failed_artifact(tmp_path):
    artifact_path = tmp_path / "failed-hash.json"
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        evaluate_quality(
            train_examples=[{"question": "example", "answer": "#### 1"}],
            test_examples=[{"question": "q0", "answer": "#### 1"}],
            model="fake",
            artifact_path=artifact_path,
            train_data_path=missing_path,
            test_data_path=missing_path,
            llm_factory=lambda **kwargs: pytest.fail("LLM should not be constructed"),
            sampling_params_factory=lambda **kwargs: kwargs,
        )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert artifact["error"]["type"] == "FileNotFoundError"
