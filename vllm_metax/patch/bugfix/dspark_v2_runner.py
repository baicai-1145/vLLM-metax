# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Upstream vLLM 0.25 DSpark is implemented only by the V2 GPU model
# runner. Keep MetaX's V1 default for other workloads, but let DSpark use the
# upstream-required V2 path even though the platform plugin disables V2 by
# default.
# -----------------------------------------------
import os

from vllm.config.vllm import VllmConfig


_original_use_v2_model_runner = VllmConfig.use_v2_model_runner.fget
_DEEPSEEK_EXACTNESS_ENVS = (
    "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM",
    "VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR",
    "VLLM_METAX_DSV4_TOKENWISE_INDEXER_DECODE",
    "VLLM_METAX_DSV4_TOKENWISE_O_PROJ",
    "VLLM_METAX_DSV4_TOKENWISE_Q_ONLY",
    "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B",
    "VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE",
    "VLLM_METAX_DSV4_TOKENWISE_FFN",
)


def _uses_qwen3_dspark_compat(speculative_config) -> bool:
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    architectures = getattr(draft_model_config, "architectures", ())
    return "Qwen3DSparkModel" in architectures


def _use_v2_model_runner(self) -> bool:
    if os.environ.get("VLLM_METAX_FORCE_V2_MODEL_RUNNER") == "1":
        return True
    speculative_config = self.speculative_config
    if speculative_config is not None and speculative_config.method == "dspark":
        if _uses_qwen3_dspark_compat(speculative_config):
            os.environ.setdefault("VLLM_METAX_USE_FP32_LOGITS", "auto")
            os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] = "1"
        else:
            os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] = "0"
            os.environ.setdefault("VLLM_METAX_USE_FP32_LOGITS", "0")
            for name in _DEEPSEEK_EXACTNESS_ENVS:
                os.environ.setdefault(name, "1")
        return True
    return _original_use_v2_model_runner(self)


VllmConfig.use_v2_model_runner = property(_use_v2_model_runner)
