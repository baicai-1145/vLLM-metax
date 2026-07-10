# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from . import mm_encoder_attention  # noqa: F401
from importlib import import_module

_fused_moe_layer = import_module("vllm.model_executor.layers.fused_moe.layer")
_unquantized_fused_moe_method = getattr(
    _fused_moe_layer, "UnquantizedFusedMoEMethod", None
)
if hasattr(_unquantized_fused_moe_method, "register_oot"):
    from . import unquantized_fused_moe_method  # noqa: F401
from . import sparse_attn_indexer  # noqa: F401
