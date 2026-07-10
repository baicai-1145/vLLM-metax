from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
TARGET_MODEL = Path("/root/models/Qwen3-8B")
DRAFT_MODEL = Path("/root/models/dspark_qwen3_8b_block7")
SMOKE = REPO_ROOT / "tools" / "dspark_qwen3_smoke.py"
EXPECTED_MATH_ANSWERS = (
    ("7", "apples"),
    ("24", "pencils"),
    ("7",),
    ("30", "miles"),
    ("72", "marbles"),
    ("21",),
)


def _require_local_models() -> None:
    if not TARGET_MODEL.exists() or not DRAFT_MODEL.exists():
        pytest.skip("Qwen3 target/draft DSpark models are not available locally")


def _run_smoke(args: list[str], timeout_s: int = 900) -> tuple[str, dict[str, object]]:
    _require_local_models()
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    command = f"""
set -euo pipefail
source {shlex.quote(str(REPO_ROOT / ".venv" / "bin" / "activate"))}
source {shlex.quote(str(REPO_ROOT / "env.sh"))}
TORCH_LIB="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")"
export LD_LIBRARY_PATH=${{TORCH_LIB}}:${{LD_LIBRARY_PATH:-}}
python {shlex.quote(str(SMOKE))} {quoted_args}
"""
    proc = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    assert proc.returncode == 0, proc.stdout[-8000:]
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return proc.stdout, json.loads(line.removeprefix("RESULT_JSON="))
    return proc.stdout, {}


def _print_spec_metrics(metrics: dict[str, object]) -> None:
    for name in (
        "acceptance_len",
        "overall_acceptance_rate",
        "per_pos_acceptance_rates",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
    ):
        print(f"{name}={metrics[name]}")


def test_dspark_qwen3_prompt5_regression_matches_baseline() -> None:
    output, result = _run_smoke(
        [
            "validate",
            "--profile",
            "math",
            "--prompt-index",
            "5",
            "--max-tokens",
            "4",
            "--num-speculative-tokens",
            "1",
        ]
    )

    assert "VALIDATION_RESULT=PASSED" in output
    assert result["texts"] == [" To solve this problem"]
    assert result["token_ids"] == [[2014, 11625, 419, 3491]]


def test_dspark_qwen3_mixed_prompt1_structural_punctuation_matches_baseline() -> None:
    output, result = _run_smoke(
        [
            "validate",
            "--profile",
            "mixed",
            "--prompt-index",
            "1",
            "--max-tokens",
            "47",
            "--num-speculative-tokens",
            "7",
            "--chat-template",
            "--disable-thinking",
        ]
    )
    assert "VALIDATION_RESULT=PASSED" in output
    assert result["texts"] == [
        "Let's solve the equation step by step:\n\n"
        "**Given:**\n$$\nx + 7 = 19\n$$\n\n"
        "**Step 1:** To isolate $ x $, subtract 7 from both sides "
        "of the equation.\n\n"
    ]
    assert result["token_ids"][0][-1] == 382


def test_dspark_qwen3_block7_chat_no_thinking_acceptance() -> None:
    output, result = _run_smoke(
        [
            "dspark",
            "--profile",
            "math",
            "--max-tokens",
            "64",
            "--num-speculative-tokens",
            "7",
            "--chat-template",
            "--disable-thinking",
            "--acceptance-gate",
            "math",
            "--json",
        ]
    )

    assert "ACCEPTANCE_RESULT=PASSED" in output
    metrics = result["spec_metrics"]
    _print_spec_metrics(metrics)
    assert metrics["acceptance_len"] >= 5.5
    assert metrics["overall_acceptance_rate"] >= 0.66
    assert len(metrics["per_pos_acceptance_rates"]) == 7


def test_dspark_qwen3_block7_chat_no_thinking_mixed_acceptance() -> None:
    output, result = _run_smoke(
        [
            "dspark",
            "--profile",
            "mixed",
            "--max-tokens",
            "64",
            "--num-speculative-tokens",
            "7",
            "--chat-template",
            "--disable-thinking",
            "--acceptance-gate",
            "mixed",
            "--json",
        ]
    )

    assert "ACCEPTANCE_RESULT=PASSED" in output
    metrics = result["spec_metrics"]
    _print_spec_metrics(metrics)
    assert metrics["acceptance_len"] >= 4.1
    assert metrics["overall_acceptance_rate"] >= 0.44
    assert len(metrics["per_pos_acceptance_rates"]) == 7


def _assert_math_answer_terms(texts: list[str]) -> None:
    assert len(texts) == len(EXPECTED_MATH_ANSWERS)
    for index, (text, expected_terms) in enumerate(
        zip(texts, EXPECTED_MATH_ANSWERS)
    ):
        normalized = text.lower()
        missing = [term for term in expected_terms if term not in normalized]
        assert not missing, (
            f"prompt {index} output is missing expected answer terms {missing}: "
            f"{text!r}"
        )


def test_dspark_qwen3_chat_no_thinking_math_answers() -> None:
    _, baseline = _run_smoke(
        [
            "baseline",
            "--profile",
            "math",
            "--max-tokens",
            "128",
            "--chat-template",
            "--disable-thinking",
            "--json",
        ]
    )
    _, dspark = _run_smoke(
        [
            "dspark",
            "--profile",
            "math",
            "--max-tokens",
            "128",
            "--num-speculative-tokens",
            "7",
            "--chat-template",
            "--disable-thinking",
            "--acceptance-gate",
            "math",
            "--json",
        ]
    )

    _print_spec_metrics(dspark["spec_metrics"])
    _assert_math_answer_terms(baseline["texts"])
    _assert_math_answer_terms(dspark["texts"])
