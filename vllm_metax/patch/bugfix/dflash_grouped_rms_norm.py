# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Upstream DFlash/DSpark precomputes K/V for all draft layers together
# and normalizes K with a grouped [num_layers, head_dim] RMSNorm weight. The
# MetaX vLLM custom rms_norm op accepts a 1-D weight, so run the same operation
# per draft layer.
# -----------------------------------------------
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


_original_normalize_context_k = DFlashQwen3Model._normalize_context_k


def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
    weights = self._k_norm_weights
    if weights.ndim == 2 and all_k.ndim >= 2:
        all_k_normed = torch.empty_like(all_k)
        for layer_idx, weight in enumerate(weights):
            ops.rms_norm(
                all_k_normed[layer_idx],
                all_k[layer_idx],
                weight,
                self._rms_norm_eps,
            )
        return all_k_normed
    return _original_normalize_context_k(self, all_k)


DFlashQwen3Model._normalize_context_k = _normalize_context_k
