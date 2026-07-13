# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

from vllm_metax.utils.deep_gemm import bf16_einsum
from . import inv_rope


def _direct_bmm_enabled() -> bool:
    return os.getenv("VLLM_METAX_DSV4_O_PROJ_DIRECT_BMM", "0") == "1"


def deep_gemm_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """
    O projection: inverse RoPE + einsum + wo_b.

    """
    rope_kwargs = {
        "n_groups": n_groups,
        "heads_per_group": heads_per_group,
        "nope_dim": nope_dim,
        "rope_dim": rope_dim,
    }
    wo_a_bf16 = wo_a.weight.view(n_groups, o_lora_rank, -1)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    o_bf16 = inv_rope(o, positions, cos_sin_cache, **rope_kwargs)
    if _direct_bmm_enabled():
        torch.bmm(
            o_bf16.transpose(0, 1),
            wo_a_bf16.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    else:
        bf16_einsum(
            "bhr,hdr->bhd",
            o_bf16,
            wo_a_bf16,
            z,
        )
    output = wo_b(z.flatten(1))

    # The hook is inert unless explicitly enabled and runs only after the
    # production native path has completed.  It must never provide a fallback.
    if os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR"):
        from . import o_proj_debug

        o_proj_debug.maybe_capture_o_proj(
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            wo_a=wo_a,
            wo_b=wo_b,
            o_bf16=o_bf16,
            z=z,
            output=output,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
    return output
