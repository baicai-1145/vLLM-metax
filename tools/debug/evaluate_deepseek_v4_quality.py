#!/usr/bin/env python3
"""Deterministic, offline GSM8K quality evaluation for DeepSeek-V4.

The evaluator deliberately uses the offline ``LLM.generate`` completion API.
It does not import vLLM until model execution is requested, so data and parser
tests can run without loading a model (or contacting a network service).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-comment JSON objects from a UTF-8 JSONL file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            if not isinstance(value.get("question"), str) or not isinstance(
                value.get("answer"), str
            ):
                raise ValueError(
                    f"JSONL row at {path}:{line_number} needs string question/answer"
                )
            rows.append(value)
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_questions(
    questions: Sequence[dict[str, Any]], num_questions: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic subset without mutating the input sequence."""
    if num_questions < 0:
        raise ValueError("num_questions must be non-negative")
    count = min(num_questions, len(questions))
    indices = sorted(random.Random(seed).sample(range(len(questions)), count))
    return [questions[index] for index in indices]


def _default_llm_factory(**kwargs: Any) -> Any:
    """Construct the one production LLM instance used for an evaluation."""
    from vllm import LLM

    return LLM(**kwargs)


def _default_sampling_params_factory(**kwargs: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**kwargs)


def _chunks(values: Sequence[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), size):
        yield start, list(values[start : start + size])


def _decimal_json(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _zero_tail_start(token_ids: Sequence[int], min_tail: int = 8) -> int | None:
    """Return the start of a suspicious repeated token-0 tail."""
    if len(token_ids) < min_tail:
        return None
    start = len(token_ids)
    while start > 0 and token_ids[start - 1] == 0:
        start -= 1
    return start if len(token_ids) - start >= min_tail else None


def _serialize_sample_logprobs(value: Any) -> list[Any] | None:
    if value is None:
        return None
    serialized = []
    for position in value:
        if position is None:
            serialized.append(None)
            continue
        entries = []
        for token_id, item in position.items():
            logprob = float(item.logprob)
            if math.isnan(logprob):
                serialized_logprob: float | str = "NaN"
            elif math.isinf(logprob):
                serialized_logprob = "Infinity" if logprob > 0 else "-Infinity"
            else:
                serialized_logprob = logprob
            entries.append(
                {
                    "token_id": int(token_id),
                    "logprob": serialized_logprob,
                    "rank": item.rank,
                    "decoded_token": item.decoded_token,
                }
            )
        entries.sort(
            key=lambda entry: (
                entry["rank"] is None,
                entry["rank"] if entry["rank"] is not None else 0,
                entry["token_id"],
            )
        )
        serialized.append(entries)
    return serialized


_RUNTIME_ENV_KEYS = (
    "VLLM_METAX_DSV4_MHC_BACKEND",
    "VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS",
    "VLLM_METAX_DSV4_MHC_EXACT_POST_MMA",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "VLLM_METAX_USE_FP32_LOGITS",
    "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND",
)
_MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
_WEIGHT_SUFFIXES = (".bin", ".gguf", ".pt", ".pth", ".safetensors")


def _model_provenance(model: str) -> dict[str, Any]:
    path = Path(model).expanduser()
    provenance: dict[str, Any] = {
        "path": model,
        "resolved_path": str(path.resolve(strict=False)),
        "metadata_sha256": {},
        "weight_files": [],
        "weight_hash_complete": False,
    }
    if not path.is_dir():
        return provenance
    metadata: dict[str, str] = {}
    for name in _MODEL_METADATA_FILES:
        metadata_path = path / name
        if metadata_path.is_file():
            try:
                metadata[name] = sha256_file(metadata_path)
            except OSError:
                continue
    weights: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in _WEIGHT_SUFFIXES
        )
    except OSError:
        candidates = []
    hash_errors = []
    for candidate in candidates:
        try:
            weights.append(
                {
                    "name": candidate.relative_to(path).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
        except OSError:
            hash_errors.append(candidate.relative_to(path).as_posix())
    provenance["metadata_sha256"] = metadata
    provenance["weight_files"] = weights
    provenance["weight_hash_complete"] = (
        bool(candidates) and len(weights) == len(candidates) and not hash_errors
    )
    provenance["weight_hash_errors"] = hash_errors
    return provenance


def _runtime_context() -> dict[str, Any]:
    """Capture environment and repository identity without making evaluation fail."""
    environment = {key: os.environ.get(key) for key in _RUNTIME_ENV_KEYS}
    repo_root = Path(__file__).resolve().parents[2]
    try:
        evaluator_source_sha256 = sha256_file(__file__)
    except OSError:
        evaluator_source_sha256 = None
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        git_head = head_result.stdout.strip() if head_result.returncode == 0 else None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        git_status_porcelain = (
            dirty_result.stdout if dirty_result.returncode == 0 else None
        )
        git_dirty = (
            git_status_porcelain != "" if git_status_porcelain is not None else None
        )
        diff_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        git_diff_sha256 = (
            hashlib.sha256(diff_result.stdout).hexdigest()
            if diff_result.returncode == 0
            else None
        )
    except (OSError, subprocess.SubprocessError):
        git_head = None
        git_dirty = None
        git_status_porcelain = None
        git_diff_sha256 = None
    return {
        "environment": environment,
        "evaluator_source_sha256": evaluator_source_sha256,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "git_status_porcelain": git_status_porcelain,
    }


def _write_artifact_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_quality(
    *,
    train_examples: Sequence[dict[str, Any]],
    test_examples: Sequence[dict[str, Any]],
    model: str,
    artifact_path: str | Path,
    num_questions: int = 20,
    num_shots: int = 5,
    batch_size: int = 1,
    max_tokens: int = 256,
    seed: int = 42,
    num_logprobs: int | None = None,
    logprob_token_ids: Sequence[int] | None = None,
    tensor_parallel_size: int = 4,
    num_speculative_tokens: int = 0,
    cudagraph_mode: str = "PIECEWISE",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    train_data_path: str | Path | None = None,
    test_data_path: str | Path | None = None,
    llm_factory: Any = _default_llm_factory,
    sampling_params_factory: Any = _default_sampling_params_factory,
) -> dict[str, Any]:
    """Run deterministic batched completion evaluation and write JSON evidence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if num_logprobs is not None and num_logprobs <= 0:
        raise ValueError("num_logprobs must be positive")
    if logprob_token_ids and num_logprobs is None:
        raise ValueError("logprob_token_ids require num_logprobs")
    if logprob_token_ids and num_logprobs != len(logprob_token_ids):
        raise ValueError("num_logprobs must equal len(logprob_token_ids)")
    if tensor_parallel_size != 4:
        raise ValueError("quality acceptance requires tensor_parallel_size=4")
    if num_speculative_tokens != 0:
        raise ValueError("quality acceptance requires MTP/speculative tokens=0")
    if cudagraph_mode != "PIECEWISE":
        raise ValueError("quality acceptance requires cudagraph_mode=PIECEWISE")

    selected = sample_questions(test_examples, num_questions, seed)
    prompts = [build_prompt(row["question"], train_examples, num_shots) for row in selected]
    expected = [parse_numeric_answer(row["answer"]) for row in selected]
    compilation_config = {
        "cudagraph_mode": cudagraph_mode,
        "cudagraph_capture_sizes": sorted({1, batch_size}),
    }
    sampling_config = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "min_tokens": 1,
        "seed": seed,
        "stop": ["\nQuestion:", "Assistant:", "<|separator|>"],
    }
    if num_logprobs is not None:
        sampling_config["logprobs"] = num_logprobs
        if logprob_token_ids:
            sampling_config["logprob_token_ids"] = [
                int(token_id) for token_id in logprob_token_ids
            ]
    llm_kwargs = {
        "model": model,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enforce_eager": False,
        "enable_prefix_caching": enable_prefix_caching,
        "compilation_config": compilation_config,
    }
    output_path = Path(artifact_path)
    runtime_context = _runtime_context()
    started = time.perf_counter()
    question_records: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "model": model,
        "model_provenance": _model_provenance(model),
        "tensor_parallel_size": tensor_parallel_size,
        "num_speculative_tokens": num_speculative_tokens,
        "mtp": 0,
        "cudagraph_mode": cudagraph_mode,
        "enforce_eager": False,
        "batch_size": batch_size,
        "num_questions": len(selected),
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "seed": seed,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "temperature": 0.0,
        "prefix_caching": enable_prefix_caching,
        "prompt_format": "gsm8k_completion",
        "completion_api": "LLM.generate",
        "sampling_params": sampling_config,
        "compilation_config": compilation_config,
        "generation_calls": (len(selected) + batch_size - 1) // batch_size,
        "profiler": False,
        **runtime_context,
    }
    data: dict[str, Any] = {
        "train_path": str(train_data_path) if train_data_path is not None else None,
        "test_path": str(test_data_path) if test_data_path is not None else None,
        "train_sha256": None,
        "test_sha256": None,
    }

    def make_artifact(status: str, error: BaseException | None = None) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        num_correct = sum(record["correct"] for record in question_records)
        num_invalid = sum(record["predicted_answer"] is None for record in question_records)
        total = len(question_records)
        artifact: dict[str, Any] = {
            "schema_version": 1,
            "created_at": created_at,
            "status": status,
            "manifest": {**manifest, "num_questions": total},
            "data": data,
            "questions": question_records,
            "summary": {
                "num_questions": total,
                "num_correct": num_correct,
                "num_invalid": num_invalid,
                "accuracy": num_correct / total if total else 0.0,
                "invalid_rate": num_invalid / total if total else 0.0,
                "elapsed_seconds": elapsed,
                "total_output_tokens": sum(
                    len(record["token_ids"]) for record in question_records
                ),
                "num_token_zero_tails": sum(
                    record["token_zero_tail_start"] is not None
                    for record in question_records
                ),
                "num_length_terminated": sum(
                    record["finish_reason"] == "length" for record in question_records
                ),
            },
        }
        if error is not None:
            artifact["error"] = {"type": type(error).__name__, "message": str(error)}
        return artifact

    try:
        data["train_sha256"] = (
            sha256_file(train_data_path) if train_data_path else None
        )
        data["test_sha256"] = sha256_file(test_data_path) if test_data_path else None
        llm = llm_factory(**llm_kwargs)
        sampling_params = sampling_params_factory(**sampling_config)
        for start, prompt_batch in _chunks(prompts, batch_size):
            outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
            if len(outputs) != len(prompt_batch):
                raise RuntimeError(
                    f"LLM.generate returned {len(outputs)} outputs for {len(prompt_batch)} prompts"
                )
            for offset, request_output in enumerate(outputs):
                candidates = getattr(request_output, "outputs", None)
                if not candidates:
                    raise RuntimeError("LLM.generate returned a request without an output")
                completion = candidates[0]
                raw_output = getattr(completion, "text", "") or ""
                if not isinstance(raw_output, str):
                    raw_output = str(raw_output)
                token_ids = [
                    int(token) for token in (getattr(completion, "token_ids", ()) or ())
                ]
                token_zero_tail_start = _zero_tail_start(token_ids)
                sample_logprobs = _serialize_sample_logprobs(
                    getattr(completion, "logprobs", None)
                )
                if num_logprobs is not None and sample_logprobs is None:
                    raise RuntimeError("requested sample logprobs were not returned")
                predicted = parse_numeric_answer(raw_output)
                expected_value = expected[start + offset]
                finish_reason = getattr(completion, "finish_reason", None)
                if finish_reason is not None and not isinstance(finish_reason, str):
                    finish_reason = str(finish_reason)
                question_records.append(
                    {
                        "index": start + offset,
                        "question": selected[start + offset]["question"],
                        "prompt_hash": hashlib.sha256(
                            prompt_batch[offset].encode("utf-8")
                        ).hexdigest(),
                        "raw_output": raw_output,
                        "token_ids": token_ids,
                        "token_zero_tail_start": token_zero_tail_start,
                        "sample_logprobs": sample_logprobs,
                        "finish_reason": finish_reason,
                        "expected_answer": _decimal_json(expected_value),
                        "predicted_answer": _decimal_json(predicted),
                        "correct": expected_value is not None
                        and predicted is not None
                        and expected_value == predicted,
                    }
                )
            _write_artifact_atomic(output_path, make_artifact("running"))
    except Exception as error:
        try:
            _write_artifact_atomic(output_path, make_artifact("failed", error))
        except OSError:
            pass
        raise
    result = make_artifact("completed")
    if result["summary"]["num_token_zero_tails"]:
        result["status"] = "runtime_blocked"
    _write_artifact_atomic(output_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP",
    )
    parser.add_argument("--train-data", required=True, help="OpenAI GSM8K train.jsonl")
    parser.add_argument("--test-data", required=True, help="OpenAI GSM8K test.jsonl")
    parser.add_argument("--artifact", default="deepseek_v4_quality.json")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--logprob-token-id", type=int, action="append")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="disable prefix caching for diagnosis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_examples = load_jsonl(args.train_data)
    test_examples = load_jsonl(args.test_data)
    result = evaluate_quality(
        train_examples=train_examples,
        test_examples=test_examples,
        model=args.model,
        artifact_path=args.artifact,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
        num_logprobs=args.logprobs,
        logprob_token_ids=args.logprob_token_id,
        tensor_parallel_size=args.tensor_parallel_size,
        num_speculative_tokens=args.num_speculative_tokens,
        cudagraph_mode=args.cudagraph_mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
    )
    summary = result["summary"]
    print(
        f"GSM8K status={result['status']} accuracy={summary['accuracy']:.4f} "
        f"invalid_rate={summary['invalid_rate']:.4f} "
        f"questions={summary['num_questions']} artifact={args.artifact}"
    )
    return 2 if result["status"] == "runtime_blocked" else 0


def build_prompt(
    question: str,
    train_examples: Sequence[dict[str, Any]],
    num_shots: int = 5,
    gen_prefix: str = "",
) -> str:
    """Build the plain completion prompt used by GSM8K's 5-shot protocol."""
    if num_shots < 0:
        raise ValueError("num_shots must be non-negative")
    examples = train_examples[:num_shots]
    separator = "" if gen_prefix.endswith(" ") else " "
    chunks = [
        f"Question: {example['question']}\n"
        f"Answer:{gen_prefix}{separator}{example['answer']}\n\n"
        for example in examples
    ]
    chunks.append(f"Question: {question}\nAnswer:{gen_prefix}")
    return "".join(chunks)


def parse_numeric_answer(text: str) -> Decimal | None:
    """Parse a GSM8K numeric answer, returning ``None`` for invalid output.

    GSM8K labels conventionally use ``####``.  When present, only the text
    after the final marker is considered; otherwise the final numeric token in
    the response is used.  Comma grouping is accepted and removed.
    """
    if not isinstance(text, str):
        return None
    marker = text.rsplit("####", 1)
    candidate = marker[1] if len(marker) == 2 else text
    matches = NUMBER_RE.findall(candidate.replace(",", ""))
    if not matches and len(marker) == 2:
        return None
    if not matches:
        matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
