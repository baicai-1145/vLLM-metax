"""Compare DeepSeek-V4 quality artifacts across scheduling modes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_MANIFEST_FIELDS = (
    "model",
    "model_provenance",
    "tensor_parallel_size",
    "mtp",
    "num_speculative_tokens",
    "cudagraph_mode",
    "enforce_eager",
    "num_shots",
    "max_tokens",
    "seed",
    "max_model_len",
    "gpu_memory_utilization",
    "temperature",
    "prefix_caching",
    "prompt_format",
    "sampling_params",
    "environment",
    "completion_api",
    "profiler",
    "evaluator_source_sha256",
    "git_head",
    "git_diff_sha256",
)

_QUESTION_FIELDS = (
    "question",
    "prompt_hash",
    "raw_output",
    "token_ids",
    "token_zero_tail_start",
    "finish_reason",
    "expected_answer",
    "predicted_answer",
    "correct",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifact(path: str | Path) -> tuple[Path, dict[str, Any]]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not a JSON object: {artifact_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported artifact schema_version: {artifact_path}")
    if not isinstance(payload.get("manifest"), dict):
        raise ValueError(f"artifact has no manifest object: {artifact_path}")
    if not isinstance(payload.get("data"), dict):
        raise ValueError(f"artifact has no data object: {artifact_path}")
    if not isinstance(payload.get("questions"), list):
        raise ValueError(f"artifact has no questions list: {artifact_path}")
    return artifact_path, payload


def _validate_model_provenance(path: Path, manifest: dict[str, Any]) -> None:
    provenance = manifest.get("model_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"artifact has no model_provenance: {path}")
    metadata = provenance.get("metadata_sha256")
    if not isinstance(metadata, dict) or not metadata or not all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value)
        for value in metadata.values()
    ):
        raise ValueError(f"artifact has invalid model metadata provenance: {path}")
    if provenance.get("weight_hash_complete") is not True:
        raise ValueError(f"artifact has incomplete weight provenance: {path}")
    weight_files = provenance.get("weight_files")
    if not isinstance(weight_files, list) or not weight_files:
        raise ValueError(f"artifact has no weight file provenance: {path}")
    for weight in weight_files:
        if not isinstance(weight, dict):
            raise ValueError(f"artifact has invalid weight provenance: {path}")
        sha256 = weight.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise ValueError(f"artifact weight has no SHA-256: {path}")


def _validate_data_provenance(path: Path, artifact: dict[str, Any]) -> None:
    data = artifact["data"]
    for field in ("train_sha256", "test_sha256"):
        value = data.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"artifact data has no valid {field}: {path}")


def _validate_graph_config(path: Path, manifest: dict[str, Any]) -> None:
    config = manifest.get("compilation_config")
    if not isinstance(config, dict):
        raise ValueError(f"artifact has no compilation_config: {path}")
    if config.get("cudagraph_mode") != manifest.get("cudagraph_mode"):
        raise ValueError(f"artifact cudagraph mode mismatch: {path}")
    capture_sizes = config.get("cudagraph_capture_sizes")
    batch_size = manifest.get("batch_size")
    if (
        not isinstance(capture_sizes, list)
        or not capture_sizes
        or not all(isinstance(size, int) and size > 0 for size in capture_sizes)
        or 1 not in capture_sizes
        or not isinstance(batch_size, int)
        or max(capture_sizes) < batch_size
    ):
        raise ValueError(f"artifact graph capture does not cover batch_size: {path}")


def _question_map(
    artifacts: Iterable[tuple[Path, dict[str, Any]]],
    route: str,
) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for path, artifact in artifacts:
        for question in artifact["questions"]:
            if not isinstance(question, dict):
                raise ValueError(f"{route} contains a non-object question: {path}")
            for field in _QUESTION_FIELDS:
                if field not in question:
                    raise ValueError(f"{route} question has no {field}: {path}")
            prompt_hash = question["prompt_hash"]
            if not isinstance(prompt_hash, str) or not prompt_hash:
                raise ValueError(f"{route} question has no prompt_hash: {path}")
            if not isinstance(question["question"], str):
                raise ValueError(f"{route} question text is invalid: {path}")
            if not isinstance(question["raw_output"], str):
                raise ValueError(f"{route} raw_output is invalid: {path}")
            if not isinstance(question["token_ids"], list) or not all(
                isinstance(token, int) for token in question["token_ids"]
            ):
                raise ValueError(f"{route} token_ids are invalid: {path}")
            if question["token_zero_tail_start"] is not None and not isinstance(
                question["token_zero_tail_start"], int
            ):
                raise ValueError(f"{route} token_zero_tail_start is invalid: {path}")
            if not isinstance(question["finish_reason"], str):
                raise ValueError(f"{route} finish_reason is invalid: {path}")
            if not isinstance(question["correct"], bool):
                raise ValueError(f"{route} correct is invalid: {path}")
            if prompt_hash in questions:
                raise ValueError(f"duplicate {route} prompt_hash: {prompt_hash}")
            questions[prompt_hash] = question
    return questions


def _first_diff(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _route_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_tokens = reference["token_ids"]
    candidate_tokens = candidate["token_ids"]
    first_diff = _first_diff(reference_tokens, candidate_tokens)
    return {
        "exact": first_diff is None,
        "first_diff": first_diff,
        "token_count": len(candidate_tokens),
        "token_zero_tail_start": candidate["token_zero_tail_start"],
        "finish_reason": candidate["finish_reason"],
        "predicted_answer": candidate["predicted_answer"],
        "expected_answer": candidate["expected_answer"],
        "correct": candidate["correct"],
    }


def _source_record(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "status": artifact.get("status"),
        "batch_size": artifact["manifest"].get("batch_size"),
        "num_questions": len(artifact["questions"]),
    }


def compare_quality_consistency(
    *,
    fresh_artifacts: Sequence[str | Path],
    continuous_artifact: str | Path,
    batched_artifact: str | Path,
) -> dict[str, Any]:
    """Compare fresh, continuous, and batched token sequences exactly."""
    if not fresh_artifacts:
        raise ValueError("at least one fresh artifact is required")

    fresh = [_load_artifact(path) for path in fresh_artifacts]
    continuous = _load_artifact(continuous_artifact)
    batched = _load_artifact(batched_artifact)
    all_artifacts = [*fresh, continuous, batched]
    reference_manifest = fresh[0][1]["manifest"]

    for path, artifact in all_artifacts:
        manifest = artifact["manifest"]
        for field in _MANIFEST_FIELDS:
            if field not in manifest:
                raise ValueError(f"artifact manifest has no {field}: {path}")
        _validate_model_provenance(path, manifest)
        _validate_graph_config(path, manifest)
        _validate_data_provenance(path, artifact)

    reference_train_sha256 = fresh[0][1]["data"].get("train_sha256")
    if not isinstance(reference_train_sha256, str) or not reference_train_sha256:
        raise ValueError(f"fresh artifact has no train_sha256: {fresh[0][0]}")

    for path, artifact in all_artifacts[1:]:
        manifest = artifact["manifest"]
        for field in _MANIFEST_FIELDS:
            if manifest.get(field) != reference_manifest.get(field):
                raise ValueError(
                    f"manifest mismatch for {field}: {fresh[0][0]} != {path}"
                )
        if artifact["data"].get("train_sha256") != reference_train_sha256:
            raise ValueError(
                f"data mismatch for train_sha256: {fresh[0][0]} != {path}"
            )

    if any(artifact["manifest"].get("batch_size") != 1 for _, artifact in fresh):
        raise ValueError("fresh artifacts must use batch_size=1")
    if continuous[1]["manifest"].get("batch_size") != 1:
        raise ValueError("continuous artifact must use batch_size=1")
    if batched[1]["manifest"].get("batch_size", 0) <= 1:
        raise ValueError("batched artifact must use batch_size greater than 1")

    fresh_questions = _question_map(fresh, "fresh")
    continuous_questions = _question_map([continuous], "continuous")
    batched_questions = _question_map([batched], "batched")
    prompt_hashes = set(fresh_questions)
    for route, questions in (
        ("continuous", continuous_questions),
        ("batched", batched_questions),
    ):
        if set(questions) != prompt_hashes:
            missing = sorted(prompt_hashes - set(questions))
            extra = sorted(set(questions) - prompt_hashes)
            raise ValueError(
                f"{route} prompt set mismatch: missing={missing}, extra={extra}"
            )

    comparisons = []
    for prompt_hash in sorted(prompt_hashes):
        reference = fresh_questions[prompt_hash]
        expected_answers = {
            reference["expected_answer"],
            continuous_questions[prompt_hash]["expected_answer"],
            batched_questions[prompt_hash]["expected_answer"],
        }
        if len(expected_answers) != 1:
            raise ValueError(f"expected_answer mismatch for prompt_hash {prompt_hash}")
        continuous_result = _route_comparison(
            reference, continuous_questions[prompt_hash]
        )
        batched_result = _route_comparison(reference, batched_questions[prompt_hash])
        fresh_zero_tail = reference["token_zero_tail_start"]
        finish_reasons = {
            reference["finish_reason"],
            continuous_result["finish_reason"],
            batched_result["finish_reason"],
        }
        passed = (
            continuous_result["exact"]
            and batched_result["exact"]
            and fresh_zero_tail is None
            and continuous_result["token_zero_tail_start"] is None
            and batched_result["token_zero_tail_start"] is None
            and len(finish_reasons) == 1
            and reference["finish_reason"] == "stop"
            and bool(reference["token_ids"])
        )
        comparisons.append(
            {
                "prompt_hash": prompt_hash,
                "question": reference["question"],
                "fresh": {
                    "token_count": len(reference["token_ids"]),
                    "token_zero_tail_start": fresh_zero_tail,
                    "finish_reason": reference["finish_reason"],
                    "predicted_answer": reference["predicted_answer"],
                    "expected_answer": reference["expected_answer"],
                    "correct": reference["correct"],
                },
                "continuous": continuous_result,
                "batched": batched_result,
                "passed": passed,
            }
        )

    statuses = [artifact.get("status") for _, artifact in all_artifacts]
    all_completed = all(status == "completed" for status in statuses)
    num_exact_continuous = sum(
        row["continuous"]["exact"] for row in comparisons
    )
    num_exact_batched = sum(row["batched"]["exact"] for row in comparisons)
    num_passed = sum(row["passed"] for row in comparisons)
    decision = "pass" if all_completed and num_passed == len(comparisons) else "fail"
    return {
        "schema_version": 1,
        "decision": decision,
        "manifest_compatibility": {
            "compatible": True,
            "compared_fields": list(_MANIFEST_FIELDS),
        },
        "sources": {
            "fresh": [_source_record(path, artifact) for path, artifact in fresh],
            "continuous": _source_record(*continuous),
            "batched": _source_record(*batched),
        },
        "summary": {
            "num_prompts": len(comparisons),
            "num_exact_continuous": num_exact_continuous,
            "num_exact_batched": num_exact_batched,
            "num_passed": num_passed,
        },
        "comparisons": comparisons,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-artifact", action="append", required=True)
    parser.add_argument("--continuous-artifact", required=True)
    parser.add_argument("--batched-artifact", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = compare_quality_consistency(
        fresh_artifacts=args.fresh_artifact,
        continuous_artifact=args.continuous_artifact,
        batched_artifact=args.batched_artifact,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return 0 if result["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
