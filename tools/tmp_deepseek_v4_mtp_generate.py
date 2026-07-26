import hashlib
import json
import math
import os
import statistics
import time

os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

from vllm import LLM, SamplingParams, TokensPrompt


def _parse_expected_token_ids(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("EXPECTED_TOKEN_IDS JSON value must be a list")
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_prompt_texts_json(value: str, bench_runs: int) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(prompt, str) for prompt in parsed
    ):
        raise ValueError("PROMPT_TEXTS_JSON must be a JSON list of strings")
    if len(parsed) != bench_runs:
        raise ValueError(
            "PROMPT_TEXTS_JSON must contain exactly one prompt per replay: "
            f"BENCH_RUNS={bench_runs} prompts={len(parsed)}"
        )
    return parsed


def _parse_expected_run_token_ids_json(value: str) -> list[list[int]]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(token_ids, list) for token_ids in parsed
    ):
        raise ValueError(
            "EXPECTED_RUN_TOKEN_IDS must be a JSON list of token-ID lists"
        )
    return [[int(token_id) for token_id in token_ids] for token_ids in parsed]


def _parse_run_max_tokens_json(value: str, bench_runs: int) -> list[int]:
    parsed = json.loads(value)
    if (
        not isinstance(parsed, list)
        or len(parsed) != bench_runs
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in parsed
        )
    ):
        raise ValueError(
            "RUN_MAX_TOKENS_JSON must contain one positive integer per replay: "
            f"BENCH_RUNS={bench_runs}"
        )
    return parsed


def _validate_min_tokens(min_tokens: int, run_max_tokens: list[int]) -> None:
    if min_tokens < 1:
        raise ValueError("MIN_TOKENS must be at least 1")
    for run_index, max_tokens in enumerate(run_max_tokens):
        if min_tokens > max_tokens:
            raise ValueError(
                f"MIN_TOKENS={min_tokens} exceeds run={run_index} "
                f"max_tokens={max_tokens}"
            )


def _validate_expected_runs(
    run_token_ids: list[list[int]], expected: list[int]
) -> None:
    for run_index, actual in enumerate(run_token_ids):
        if actual != expected:
            raise SystemExit(
                "TOKEN_IDS_MISMATCH "
                f"run={run_index} expected={expected} actual={actual}"
            )


def _validate_expected_run_matrix(
    run_token_ids: list[list[int]], expected_runs: list[list[int]]
) -> None:
    if len(run_token_ids) != len(expected_runs):
        raise SystemExit(
            "TOKEN_IDS_RUN_COUNT_MISMATCH "
            f"expected={len(expected_runs)} actual={len(run_token_ids)}"
        )
    for run_index, (actual, expected) in enumerate(
        zip(run_token_ids, expected_runs, strict=True)
    ):
        if actual != expected:
            raise SystemExit(
                "TOKEN_IDS_MISMATCH "
                f"run={run_index} expected={expected} actual={actual}"
            )


def _token_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _serialize_run_token_ids(run_token_ids: list[list[int]]) -> str:
    return json.dumps(run_token_ids, separators=(",", ":"))


def _async_scheduling_override() -> dict[str, bool]:
    value = os.environ.get("ASYNC_SCHEDULING")
    if value is None:
        return {}
    if value not in {"0", "1"}:
        raise ValueError("ASYNC_SCHEDULING must be 0 or 1")
    return {"async_scheduling": value == "1"}


def _log_stats_override() -> dict[str, bool]:
    value = os.environ.get("DISABLE_LOG_STATS")
    if value is None:
        return {}
    if value not in {"0", "1"}:
        raise ValueError("DISABLE_LOG_STATS must be 0 or 1")
    return {"disable_log_stats": value == "1"}


def _build_speculative_config(num_speculative_tokens: int) -> dict | None:
    if num_speculative_tokens == 0:
        return None

    method = os.environ.get("SPECULATIVE_METHOD", "mtp")
    if method == "mtp":
        return {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
    if method != "dspark":
        raise ValueError("SPECULATIVE_METHOD must be mtp or dspark")

    draft_model = os.environ.get("SPECULATIVE_MODEL")
    if not draft_model:
        raise ValueError("SPECULATIVE_MODEL is required for dspark")
    return {
        "method": "dspark",
        "model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_sample_method": "greedy",
    }


def main() -> None:
    model = os.environ["MODEL"]
    tensor_parallel_size = int(os.environ.get("TP", "1"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEM", "0.7"))
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "512"))
    max_num_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    min_tokens = int(os.environ.get("MIN_TOKENS", "1"))
    input_tokens = int(os.environ.get("INPUT_TOKENS", "0"))
    warmup_requests = int(os.environ.get("WARMUP_REQUESTS", "0"))
    bench_runs = int(os.environ.get("BENCH_RUNS", "1"))
    if bench_runs < 1:
        raise ValueError("BENCH_RUNS must be at least 1")
    run_max_tokens_json = os.environ.get("RUN_MAX_TOKENS_JSON")
    run_max_tokens = (
        _parse_run_max_tokens_json(run_max_tokens_json, bench_runs)
        if run_max_tokens_json
        else [max_tokens] * bench_runs
    )
    _validate_min_tokens(min_tokens, run_max_tokens)
    enable_prefix_caching = os.environ.get("ENABLE_PREFIX_CACHING", "1") == "1"
    num_speculative_tokens = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "0"))
    profile_dir = os.environ.get("PROFILE_DIR")

    compilation_config = None
    if not enforce_eager:
        capture_sizes = [1]
        if num_speculative_tokens:
            capture_sizes.append(num_speculative_tokens + 1)
        compilation_config = {
            "cudagraph_mode": os.environ.get("CUDAGRAPH_MODE", "PIECEWISE"),
            "cudagraph_capture_sizes": capture_sizes,
        }
    speculative_config = _build_speculative_config(num_speculative_tokens)
    profiler_config = None
    if profile_dir:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_with_stack": True,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": True,
            "delay_iterations": int(os.environ.get("PROFILE_DELAY_ITERATIONS", "0")),
            "max_iterations": int(os.environ.get("PROFILE_MAX_ITERATIONS", "0")),
            "active_iterations": int(os.environ.get("PROFILE_ACTIVE_ITERATIONS", "5")),
            "ignore_frontend": os.environ.get("PROFILE_IGNORE_FRONTEND", "0") == "1",
        }

    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        speculative_config=speculative_config,
        profiler_config=profiler_config,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        **_async_scheduling_override(),
        **_log_stats_override(),
        **(
            {"max_num_batched_tokens": max_num_batched_tokens}
            if max_num_batched_tokens
            else {}
        ),
    )
    run_sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=value, min_tokens=min_tokens)
        for value in run_max_tokens
    ]
    sampling_params = run_sampling_params[-1]
    if input_tokens:
        if os.environ.get("PROMPT_TEXTS_JSON"):
            raise ValueError("PROMPT_TEXTS_JSON cannot be used with INPUT_TOKENS")
        tokenizer = llm.get_tokenizer()
        seed_ids = tokenizer.encode(
            "Deep learning inference performance depends on efficient kernels, "
            "memory access, communication, and scheduling. ",
            add_special_tokens=False,
        )
        if not seed_ids:
            raise RuntimeError("tokenizer produced an empty prefill seed")
        repeats = math.ceil(input_tokens / len(seed_ids))
        prompt_token_ids = (seed_ids * repeats)[:input_tokens]
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
        run_prompts = [prompts] * bench_runs
    else:
        prompt_texts_json = os.environ.get("PROMPT_TEXTS_JSON")
        if prompt_texts_json:
            prompt_texts = _parse_prompt_texts_json(prompt_texts_json, bench_runs)
        else:
            prompt_texts = [
                os.environ.get(
                    "PROMPT_TEXT",
                    "Complete this sentence in one short clause: "
                    "Speculative decoding is",
                )
            ] * bench_runs
        run_prompts = [[prompt_text] for prompt_text in prompt_texts]
        prompts = run_prompts[0]
        prompt_token_ids = []

    warmup_params = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1)
    for _ in range(warmup_requests):
        llm.generate(prompts, warmup_params, use_tqdm=False)
    if profile_dir:
        llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1),
            use_tqdm=False,
        )
        llm.start_profile()
    elapsed_runs = []
    run_token_ids = []
    run_finish_reasons = []
    out = None
    if input_tokens:
        print("PREFILL_BENCH_START", flush=True)
    else:
        print("DECODE_BENCH_START", flush=True)
    for run_prompt, run_params in zip(
        run_prompts, run_sampling_params, strict=True
    ):
        started = time.perf_counter()
        out = llm.generate(run_prompt, run_params, use_tqdm=False)
        elapsed_runs.append(time.perf_counter() - started)
        run_output = out[0].outputs[0]
        run_token_ids.append([int(token_id) for token_id in run_output.token_ids])
        run_finish_reasons.append(run_output.finish_reason)
    if input_tokens:
        print("PREFILL_BENCH_END", flush=True)
    else:
        print("DECODE_BENCH_END", flush=True)
    assert out is not None
    output = out[0].outputs[0]
    text = output.text
    if not text.strip():
        chat_out = llm.chat(
            [
                [
                    {"role": "system", "content": "Answer in one short sentence."},
                    {
                        "role": "user",
                        "content": "What is speculative decoding?",
                    },
                ]
            ],
            sampling_params,
        )
        output = chat_out[0].outputs[0]
        text = output.text
    elapsed = statistics.median(elapsed_runs)
    sorted_elapsed = sorted(elapsed_runs)
    p90_index = max(0, math.ceil(0.9 * len(sorted_elapsed)) - 1)
    elapsed_p90 = sorted_elapsed[p90_index]
    if profile_dir:
        llm.stop_profile()
    output_tps = len(output.token_ids) / elapsed
    print("GEN_OK")
    print(repr(text))
    print("FINISH_REASON", repr(output.finish_reason))
    print("TOKEN_IDS", output.token_ids)
    print("GENERATED_TOKENS", len(output.token_ids))
    print("GENERATE_SECONDS", f"{elapsed:.6f}")
    print("OUTPUT_TOKENS_PER_SECOND", f"{output_tps:.6f}")
    if input_tokens:
        print("PROMPT_TOKENS", len(prompt_token_ids))
        print("PREFILL_RUN_SECONDS", [round(value, 6) for value in elapsed_runs])
        print("PREFILL_MEDIAN_SECONDS", f"{elapsed:.6f}")
        print("PREFILL_P90_SECONDS", f"{elapsed_p90:.6f}")
        print("PREFILL_TOKENS_PER_SECOND", f"{input_tokens / elapsed:.6f}")
    else:
        print("DECODE_RUN_SECONDS", [round(value, 6) for value in elapsed_runs])
        print("DECODE_P90_SECONDS", f"{elapsed_p90:.6f}")
    print("RUN_TOKEN_HASHES", [_token_hash(tokens) for tokens in run_token_ids])
    print("RUN_TOKEN_IDS_JSON", _serialize_run_token_ids(run_token_ids))
    print("RUN_FINISH_REASONS", run_finish_reasons)
    print("RUN_MAX_TOKENS", run_max_tokens)

    expected_token_ids = os.environ.get("EXPECTED_TOKEN_IDS")
    expected_run_token_ids = os.environ.get("EXPECTED_RUN_TOKEN_IDS")
    if expected_token_ids and expected_run_token_ids:
        raise ValueError(
            "EXPECTED_TOKEN_IDS and EXPECTED_RUN_TOKEN_IDS are mutually exclusive"
        )
    if expected_token_ids:
        expected = _parse_expected_token_ids(expected_token_ids)
        _validate_expected_runs(run_token_ids, expected)
        print("TOKEN_IDS_MATCH_EXPECTED")
    if expected_run_token_ids:
        expected_runs = _parse_expected_run_token_ids_json(expected_run_token_ids)
        _validate_expected_run_matrix(run_token_ids, expected_runs)
        print("RUN_TOKEN_IDS_MATCH_EXPECTED")

    expected_finish_reasons = os.environ.get("EXPECTED_RUN_FINISH_REASONS")
    if expected_finish_reasons:
        expected_reasons = json.loads(expected_finish_reasons)
        if run_finish_reasons != expected_reasons:
            raise SystemExit(
                "FINISH_REASONS_MISMATCH "
                f"expected={expected_reasons} actual={run_finish_reasons}"
            )
        print("RUN_FINISH_REASONS_MATCH_EXPECTED")

    min_tps = os.environ.get("MIN_OUTPUT_TOKENS_PER_SECOND")
    if min_tps:
        threshold = float(min_tps)
        if output_tps < threshold:
            raise SystemExit(
                f"OUTPUT_TPS_BELOW_THRESHOLD threshold={threshold} actual={output_tps}"
            )
        print("OUTPUT_TPS_MEETS_THRESHOLD", f"{threshold:.6f}")


if __name__ == "__main__":
    main()
