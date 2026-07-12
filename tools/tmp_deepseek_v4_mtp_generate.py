import os
import time

os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

from vllm import LLM, SamplingParams


def _parse_expected_token_ids(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("["):
        import json

        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("EXPECTED_TOKEN_IDS JSON value must be a list")
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    model = os.environ["MODEL"]
    tensor_parallel_size = int(os.environ.get("TP", "1"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEM", "0.7"))
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    max_tokens = int(os.environ.get("MAX_TOKENS", "16"))
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
    speculative_config = None
    if num_speculative_tokens:
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
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
        max_model_len=512,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        speculative_config=speculative_config,
        profiler_config=profiler_config,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=1,
    )
    prompts = ["Complete this sentence in one short clause: Speculative decoding is"]
    if profile_dir:
        llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
        llm.start_profile()
    started = time.perf_counter()
    out = llm.generate(prompts, sampling_params)
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
    elapsed = time.perf_counter() - started
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

    expected_token_ids = os.environ.get("EXPECTED_TOKEN_IDS")
    if expected_token_ids:
        expected = _parse_expected_token_ids(expected_token_ids)
        if list(output.token_ids) != expected:
            raise SystemExit(
                f"TOKEN_IDS_MISMATCH expected={expected} actual={output.token_ids}"
            )
        print("TOKEN_IDS_MATCH_EXPECTED")

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
