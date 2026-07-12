from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


TARGET = "/root/models/Qwen3-8B"
DRAFT = "/root/models/dspark_qwen3_8b_block7"
PROMPT = "Briefly explain speculative decoding."
ACCEPTANCE_THRESHOLDS = {
    "math": {"acceptance_len": 5.5, "overall_acceptance_rate": 0.66},
    "mixed": {"acceptance_len": 4.1, "overall_acceptance_rate": 0.44},
}
EXPECTED_MATH_ANSWER_TERMS = (
    ("7", "apples"),
    ("24", "pencils"),
    ("7",),
    ("30", "miles"),
    ("72", "marbles"),
    ("21",),
)


def get_prompts(profile: str) -> list[str]:
    if profile == "single":
        return [PROMPT]
    if profile == "mixed":
        return [
            "Briefly explain speculative decoding.",
            "Solve step by step: if x + 7 = 19, what is x?",
            "Write a Python function that returns the factorial of n.",
            "A train travels 180 miles in 3 hours. What is its average speed?",
            "Explain why caching can make repeated computations faster.",
            "Write a Python loop that sums the numbers from 1 to 10.",
        ]
    if profile == "math":
        return [
            "Solve step by step: Jamie has 12 apples and gives 5 away. How many apples remain?",
            "Solve step by step: A box has 4 rows of 6 pencils. How many pencils are there?",
            "Solve step by step: If 3x = 21, what is x?",
            "Solve step by step: A car travels 150 miles in 5 hours. What is the average speed?",
            "Solve step by step: There are 9 bags with 8 marbles each. How many marbles total?",
            "Solve step by step: If a notebook costs $3, how much do 7 notebooks cost?",
        ]
    raise ValueError(f"unknown prompt profile: {profile}")


def select_prompts(profile: str, prompt_index: int | None) -> list[str]:
    prompts = get_prompts(profile)
    if prompt_index is None:
        return prompts
    return [prompts[prompt_index]]


def select_math_answer_terms(prompt_index: int | None) -> list[tuple[str, ...]]:
    terms = list(EXPECTED_MATH_ANSWER_TERMS)
    if prompt_index is None:
        return terms
    return [terms[prompt_index]]


def format_prompts(
    prompts: list[str],
    chat_template: bool,
    enable_thinking: bool,
    assistant_prefix: str | None = None,
) -> list[str]:
    if not chat_template:
        if assistant_prefix is None:
            return prompts
        return [prompt + assistant_prefix for prompt in prompts]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        + (assistant_prefix or "")
        for prompt in prompts
    ]


def build_llm(
    mode: str,
    disable_async_scheduling: bool = False,
    num_speculative_tokens: int = 7,
    draft_attention_backend: str | None = None,
    attention_backend: str | None = None,
    rejection_sample_method: str | None = None,
) -> LLM:
    from vllm import LLM

    kwargs = {
        "model": TARGET,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 1024,
        "enforce_eager": True,
        "max_num_seqs": 1,
        "disable_log_stats": False,
    }
    if disable_async_scheduling:
        kwargs["async_scheduling"] = False
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
    if mode == "dspark":
        kwargs["speculative_config"] = {
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": num_speculative_tokens,
            "draft_sample_method": "greedy",
        }
        if draft_attention_backend is not None:
            kwargs["speculative_config"]["attention_backend"] = draft_attention_backend
        if rejection_sample_method is not None:
            kwargs["speculative_config"]["rejection_sample_method"] = (
                rejection_sample_method
            )
    return LLM(**kwargs)


def collect_spec_metrics(llm: LLM) -> dict[str, float | int]:
    counters = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    }
    values: dict[str, float | int] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    for metric in llm.get_metrics():
        key = counters.get(metric.name)
        if key is not None:
            values[key] = values[key] + metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values["num_accepted_tokens_per_pos"] = metric.values

    num_drafts = values["num_drafts"]
    num_draft_tokens = values["num_draft_tokens"]
    num_accepted_tokens = values["num_accepted_tokens"]
    per_pos_counts = values.get("num_accepted_tokens_per_pos", [])
    values["acceptance_len"] = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts else 0.0
    )
    values["overall_acceptance_rate"] = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
    )
    values["per_pos_acceptance_rates"] = (
        [count / num_drafts for count in per_pos_counts] if num_drafts else []
    )
    return values


def run_generation(
    mode: str,
    profile: str,
    max_tokens: int,
    prompt_index: int | None,
    chat_template: bool = False,
    enable_thinking: bool = True,
    assistant_prefix: str | None = None,
    disable_async_scheduling: bool = False,
    num_speculative_tokens: int = 7,
    draft_attention_backend: str | None = None,
    attention_backend: str | None = None,
    rejection_sample_method: str | None = None,
) -> dict[str, object]:
    if mode == "baseline-v2":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_METAX_FORCE_V2_MODEL_RUNNER"] = "1"

    from vllm import SamplingParams

    raw_prompts = select_prompts(profile, prompt_index)
    prompts = format_prompts(
        raw_prompts, chat_template, enable_thinking, assistant_prefix
    )
    print(f"constructing {mode} LLM")
    llm = build_llm(
        mode,
        disable_async_scheduling,
        num_speculative_tokens,
        draft_attention_backend,
        attention_backend,
        rejection_sample_method,
    )
    print(f"generating {mode}")
    generation_start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0),
    )
    generation_seconds = time.perf_counter() - generation_start
    texts = [output.outputs[0].text for output in outputs]
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    output_tokens = sum(len(ids) for ids in token_ids)
    for text in texts:
        assert text.strip()

    result: dict[str, object] = {
        "mode": mode,
        "profile": profile,
        "max_tokens": max_tokens,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "assistant_prefix": assistant_prefix,
        "raw_prompts": raw_prompts,
        "prompts": prompts,
        "texts": texts,
        "token_ids": token_ids,
        "generation_seconds": generation_seconds,
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds else 0.0
        ),
    }
    if mode == "dspark":
        result["spec_metrics"] = collect_spec_metrics(llm)
    return result


def print_text_result(result: dict[str, object]) -> None:
    mode = str(result["mode"])
    texts = result["texts"]
    assert isinstance(texts, list)
    for index, text in enumerate(texts):
        suffix = "" if len(texts) == 1 else f"_{index}"
        print(f"{mode.upper()}_TEXT{suffix}_START")
        print(text)
        print(f"{mode.upper()}_TEXT{suffix}_END")


def run_child(
    mode: str,
    profile: str,
    max_tokens: int,
    prompt_index: int | None,
    chat_template: bool = False,
    enable_thinking: bool = True,
    assistant_prefix: str | None = None,
    disable_async_scheduling: bool = False,
    num_speculative_tokens: int = 7,
    draft_attention_backend: str | None = None,
    attention_backend: str | None = None,
    rejection_sample_method: str | None = None,
) -> dict[str, object]:
    command = [
        sys.executable,
        __file__,
        mode,
        "--profile",
        profile,
        "--max-tokens",
        str(max_tokens),
        "--json",
    ]
    if prompt_index is not None:
        command.extend(["--prompt-index", str(prompt_index)])
    if chat_template:
        command.append("--chat-template")
    if not enable_thinking:
        command.append("--disable-thinking")
    if assistant_prefix is not None:
        command.extend(["--assistant-prefix", assistant_prefix])
    if disable_async_scheduling:
        command.append("--disable-async-scheduling")
    if attention_backend is not None:
        command.extend(["--attention-backend", attention_backend])
    if mode == "dspark":
        command.extend(["--num-speculative-tokens", str(num_speculative_tokens)])
        if draft_attention_backend is not None:
            command.extend(["--draft-attention-backend", draft_attention_backend])
        if rejection_sample_method is not None:
            command.extend(["--rejection-sample-method", rejection_sample_method])
    env = os.environ.copy()
    if mode == "baseline-v2":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        env["VLLM_METAX_FORCE_V2_MODEL_RUNNER"] = "1"
    proc = subprocess.run(
        command,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=proc.stdout
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise RuntimeError(f"{mode} child did not emit RESULT_JSON")


def check_acceptance(metrics: dict[str, object], gate: str | None) -> None:
    if gate is None:
        return
    thresholds = ACCEPTANCE_THRESHOLDS[gate]
    failures = []
    for name, threshold in thresholds.items():
        value = metrics.get(name, 0)
        if value < threshold:
            failures.append(f"{name}={value} < {threshold}")
    if failures:
        print("ACCEPTANCE_RESULT=FAILED")
        for failure in failures:
            print(f"ACCEPTANCE_FAILURE={failure}")
        raise SystemExit(1)
    print("ACCEPTANCE_RESULT=PASSED")


def print_token_diff(
    index: int,
    baseline_token_ids: list[int],
    dspark_token_ids: list[int],
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    max_len = max(len(baseline_token_ids), len(dspark_token_ids))
    for pos in range(max_len):
        baseline_id = baseline_token_ids[pos] if pos < len(baseline_token_ids) else None
        dspark_id = dspark_token_ids[pos] if pos < len(dspark_token_ids) else None
        if baseline_id == dspark_id:
            continue
        baseline_piece = (
            tokenizer.decode([baseline_id]) if baseline_id is not None else "<missing>"
        )
        dspark_piece = (
            tokenizer.decode([dspark_id]) if dspark_id is not None else "<missing>"
        )
        print(f"DIFF_PROMPT_INDEX={index}")
        print(f"FIRST_TOKEN_DIFF_POSITION={pos}")
        print(f"BASELINE_TOKEN_ID={baseline_id}")
        print(f"DSPARK_TOKEN_ID={dspark_id}")
        print(f"BASELINE_TOKEN={baseline_piece!r}")
        print(f"DSPARK_TOKEN={dspark_piece!r}")
        print(f"BASELINE_TOKEN_IDS={baseline_token_ids}")
        print(f"DSPARK_TOKEN_IDS={dspark_token_ids}")
        return


def check_math_answer_equivalence(
    baseline_texts: list[str],
    dspark_texts: list[str],
    prompt_index: int | None,
) -> list[str]:
    failures = []
    expected_terms = select_math_answer_terms(prompt_index)
    for index, (baseline_text, dspark_text, terms) in enumerate(
        zip(baseline_texts, dspark_texts, expected_terms)
    ):
        baseline_normalized = baseline_text.lower()
        dspark_normalized = dspark_text.lower()
        baseline_missing = [term for term in terms if term not in baseline_normalized]
        dspark_missing = [term for term in terms if term not in dspark_normalized]
        if baseline_missing:
            failures.append(f"prompt {index} baseline missing terms {baseline_missing}")
        if dspark_missing:
            failures.append(f"prompt {index} DSpark missing terms {dspark_missing}")
    return failures


def validate(
    profile: str,
    max_tokens: int,
    acceptance_gate: str | None,
    prompt_index: int | None,
    chat_template: bool,
    enable_thinking: bool,
    assistant_prefix: str | None,
    baseline_mode: str,
    disable_async_scheduling: bool,
    num_speculative_tokens: int,
    draft_attention_backend: str | None,
    attention_backend: str | None,
    rejection_sample_method: str | None,
    allow_math_answer_equivalence: bool,
) -> None:
    baseline = run_child(
        baseline_mode,
        profile,
        max_tokens,
        prompt_index,
        chat_template,
        enable_thinking,
        assistant_prefix,
        disable_async_scheduling,
        attention_backend=attention_backend,
    )
    dspark = run_child(
        "dspark",
        profile,
        max_tokens,
        prompt_index,
        chat_template,
        enable_thinking,
        assistant_prefix,
        disable_async_scheduling,
        num_speculative_tokens,
        draft_attention_backend,
        attention_backend,
        rejection_sample_method,
    )

    if baseline["token_ids"] != dspark["token_ids"]:
        mismatch_count = 0
        print("STRICT_VALIDATION_RESULT=FAILED")
        print("Strict token IDs differ between baseline and DSpark.")
        for index, (base_text, dspark_text, baseline_ids, dspark_ids) in enumerate(
            zip(
                baseline["texts"],
                dspark["texts"],
                baseline["token_ids"],
                dspark["token_ids"],
            )
        ):
            if baseline_ids != dspark_ids:
                mismatch_count += 1
                print_token_diff(index, baseline_ids, dspark_ids)
                print(f"BASELINE={base_text!r}")
                print(f"DSPARK={dspark_text!r}")
        if allow_math_answer_equivalence and profile == "math":
            baseline_texts = baseline["texts"]
            dspark_texts = dspark["texts"]
            assert isinstance(baseline_texts, list)
            assert isinstance(dspark_texts, list)
            failures = check_math_answer_equivalence(
                baseline_texts,
                dspark_texts,
                prompt_index,
            )
            if not failures:
                metrics = dspark.get("spec_metrics", {})
                print("VALIDATION_RESULT=PASSED_WITH_SEMANTIC_EQUIVALENCE")
                print("SEMANTIC_EQUIVALENCE=math_answer_terms")
                print(f"TOKEN_MISMATCHED_PROMPTS={mismatch_count}")
                print(f"PROMPTS={len(baseline['prompts'])}")
                print(f"MAX_TOKENS={max_tokens}")
                print(f"CHAT_TEMPLATE={chat_template}")
                print(f"ENABLE_THINKING={enable_thinking}")
                print(f"SPEC_METRICS={json.dumps(metrics, sort_keys=True)}")
                check_acceptance(metrics, acceptance_gate)
                return
            print("SEMANTIC_EQUIVALENCE_RESULT=FAILED")
            for failure in failures:
                print(f"SEMANTIC_EQUIVALENCE_FAILURE={failure}")
        raise SystemExit(1)

    metrics = dspark.get("spec_metrics", {})
    print("VALIDATION_RESULT=PASSED")
    print(f"PROMPTS={len(baseline['prompts'])}")
    print(f"MAX_TOKENS={max_tokens}")
    print(f"CHAT_TEMPLATE={chat_template}")
    print(f"ENABLE_THINKING={enable_thinking}")
    print(f"SPEC_METRICS={json.dumps(metrics, sort_keys=True)}")
    check_acceptance(metrics, acceptance_gate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["baseline", "baseline-v2", "dspark", "validate"]
    )
    parser.add_argument(
        "--profile", choices=["single", "mixed", "math"], default="single"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help="Run only one prompt from the selected profile.",
    )
    parser.add_argument(
        "--acceptance-gate",
        choices=sorted(ACCEPTANCE_THRESHOLDS),
        default=None,
        help="Assert documented acceptance thresholds for the selected gate.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply the target tokenizer chat template before generation.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="When --chat-template is set, render Qwen3 prompts with enable_thinking=False.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default=None,
        help="Append a fixed assistant prefix after optional chat template rendering.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["baseline", "baseline-v2"],
        default="baseline",
        help="Baseline runner to compare against in validate mode.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--disable-async-scheduling",
        action="store_true",
        help="Construct the LLM with async_scheduling=False.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument(
        "--draft-attention-backend",
        default=None,
        help="Optional speculative_config attention_backend for the draft model.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Optional LLM attention_backend for the target model.",
    )
    parser.add_argument(
        "--rejection-sample-method",
        choices=["standard", "block"],
        default=None,
        help="Optional speculative_config rejection_sample_method for DSpark.",
    )
    parser.add_argument(
        "--allow-math-answer-equivalence",
        action="store_true",
        help=(
            "In validate mode with --profile math, allow token drift only when "
            "baseline and DSpark both contain the expected answer terms."
        ),
    )
    args = parser.parse_args()

    if args.mode == "validate":
        validate(
            args.profile,
            args.max_tokens,
            args.acceptance_gate,
            args.prompt_index,
            args.chat_template,
            not args.disable_thinking,
            args.assistant_prefix,
            args.baseline_mode,
            args.disable_async_scheduling,
            args.num_speculative_tokens,
            args.draft_attention_backend,
            args.attention_backend,
            args.rejection_sample_method,
            args.allow_math_answer_equivalence,
        )
        return

    result = run_generation(
        args.mode,
        args.profile,
        args.max_tokens,
        args.prompt_index,
        args.chat_template,
        not args.disable_thinking,
        args.assistant_prefix,
        args.disable_async_scheduling,
        args.num_speculative_tokens,
        args.draft_attention_backend,
        args.attention_backend,
        args.rejection_sample_method,
    )
    print_text_result(result)
    if args.acceptance_gate is not None:
        if args.mode != "dspark":
            raise SystemExit("--acceptance-gate requires dspark or validate mode")
        metrics = result.get("spec_metrics", {})
        assert isinstance(metrics, dict)
        check_acceptance(metrics, args.acceptance_gate)
    if args.json:
        print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
