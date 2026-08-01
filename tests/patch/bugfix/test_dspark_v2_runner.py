from types import SimpleNamespace

import vllm_metax.patch.bugfix.dspark_v2_runner as patch


_DEEPSEEK_EXACT_ENVS = (
    "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM",
    "VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR",
    "VLLM_METAX_DSV4_TOKENWISE_INDEXER_DECODE",
    "VLLM_METAX_DSV4_TOKENWISE_O_PROJ",
    "VLLM_METAX_DSV4_TOKENWISE_Q_ONLY",
    "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B",
    "VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE",
    "VLLM_METAX_DSV4_TOKENWISE_FFN",
)


def _config(architecture: str):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(architectures=[architecture]),
        )
    )


def test_deepseek_v4_dspark_does_not_enable_qwen_tie_patch(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSPARK_GREEDY_TIE_PATCH", raising=False)
    monkeypatch.delenv("VLLM_METAX_USE_FP32_LOGITS", raising=False)
    for name in _DEEPSEEK_EXACT_ENVS:
        monkeypatch.delenv(name, raising=False)

    assert patch._use_v2_model_runner(_config("DSparkDraftModel")) is True

    assert patch.os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] == "0"
    assert patch.os.environ["VLLM_METAX_USE_FP32_LOGITS"] == "0"
    assert all(patch.os.environ[name] == "1" for name in _DEEPSEEK_EXACT_ENVS)


def test_deepseek_v4_dspark_preserves_explicit_exactness_override(monkeypatch):
    for name in _DEEPSEEK_EXACT_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR", "0")

    assert patch._use_v2_model_runner(_config("DSparkDraftModel")) is True

    assert patch.os.environ["VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR"] == "0"


def test_qwen3_dspark_retains_greedy_tie_patch(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSPARK_GREEDY_TIE_PATCH", raising=False)
    monkeypatch.delenv("VLLM_METAX_USE_FP32_LOGITS", raising=False)

    assert patch._use_v2_model_runner(_config("Qwen3DSparkModel")) is True

    assert patch.os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] == "1"
    assert patch.os.environ["VLLM_METAX_USE_FP32_LOGITS"] == "auto"


def test_qwen_tie_patch_does_not_leak_into_deepseek(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSPARK_GREEDY_TIE_PATCH", raising=False)

    assert patch._use_v2_model_runner(_config("Qwen3DSparkModel")) is True
    assert patch.os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] == "1"

    assert patch._use_v2_model_runner(_config("DSparkDraftModel")) is True
    assert patch.os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] == "0"
