# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from importlib.util import find_spec

if find_spec("vllm.model_executor.layers.quantization.awq") is not None:
    from . import awq  # noqa: F401

if find_spec("vllm.model_executor.layers.quantization.awq_marlin") is not None:
    from . import awq_marlin  # noqa: F401

from . import (
    compressed_tensors,  # noqa: F401
    auto_gptq,  # noqa: F401
    moe_wna16,  # noqa: F401
)
