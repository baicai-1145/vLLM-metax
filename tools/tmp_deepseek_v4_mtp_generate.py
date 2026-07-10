import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ["MODEL"]
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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, min_tokens=1)
    out = llm.generate(
        ["Complete this sentence in one short clause: Speculative decoding is"],
        sampling_params,
    )
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
    print("GEN_OK")
    print(repr(text))
    print("FINISH_REASON", repr(output.finish_reason))
    print("TOKEN_IDS", output.token_ids)


if __name__ == "__main__":
    main()
