import os

from vllm import LLM


def main() -> None:
    model = os.environ.get(
        "MODEL", "/home/waas/models/DeepSeek-V4-Flash-W4A16-FP8-MTP"
    )
    tensor_parallel_size = int(os.environ.get("TP", "1"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEM", "0.7"))
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=512,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print("LLM_LOAD_OK", llm)


if __name__ == "__main__":
    main()
