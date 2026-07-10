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


def _use_v2_model_runner(self) -> bool:
    if os.environ.get("VLLM_METAX_FORCE_V2_MODEL_RUNNER") == "1":
        return True
    speculative_config = self.speculative_config
    if speculative_config is not None and speculative_config.method == "dspark":
        os.environ.setdefault("VLLM_METAX_USE_FP32_LOGITS", "auto")
        os.environ["VLLM_METAX_DSPARK_GREEDY_TIE_PATCH"] = "1"
        return True
    return _original_use_v2_model_runner(self)


VllmConfig.use_v2_model_runner = property(_use_v2_model_runner)
