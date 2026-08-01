# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

from vllm_metax.utils.deep_gemm import bf16_einsum
from . import inv_rope


def _deep_gemm_bf16_o_proj_stages(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    o_bf16 = inv_rope(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    wo_a_bf16 = wo_a.weight.view(n_groups, o_lora_rank, -1)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    bf16_einsum(
        "bhr,hdr->bhd",
        o_bf16,
        wo_a_bf16,
        z,
    )
    return o_bf16, z


def deep_gemm_bf16_o_proj_input(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    layer_idx: int | None = None,
    chunk_index: int | None = None,
) -> torch.Tensor:
    """Compute the native row-local input to the TP ``wo_b`` projection."""
    del layer_idx, chunk_index
    _, z = _deep_gemm_bf16_o_proj_stages(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
    )
    return z.flatten(1)


def deep_gemm_bf16_o_proj_row_inputs(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    layer_idx: int | None = None,
    chunk_index: int | None = None,
) -> list[torch.Tensor]:
    """Compute row-local ``wo_b`` inputs with shared inverse RoPE.

    C500 evidence shows batched inverse RoPE is row-exact, while batched
    DeepGEMM BF16 einsum is not. Keep the einsum rowwise and only share the
    inverse RoPE launch across verifier rows.
    """
    del layer_idx, chunk_index
    o_bf16 = inv_rope(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    wo_a_bf16 = wo_a.weight.view(n_groups, o_lora_rank, -1)
    row_inputs: list[torch.Tensor] = []
    for index in range(o.shape[0]):
        z = torch.empty(
            (1, n_groups, o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        bf16_einsum(
            "bhr,hdr->bhd",
            o_bf16[index : index + 1],
            wo_a_bf16,
            z,
        )
        row_inputs.append(z.flatten(1))
    return row_inputs


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
    layer_idx: int | None = None,
    chunk_index: int | None = None,
) -> torch.Tensor:
    """O projection: native inverse RoPE + einsum + TP ``wo_b``."""
    o_bf16, z = _deep_gemm_bf16_o_proj_stages(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
    )
    wo_b_input = z.flatten(1)
    wo_b_local = None
    if os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_WO_B_STAGES") == "1":
        from . import o_proj_debug

        wo_b_local, output = o_proj_debug.apply_wo_b_with_stages(
            wo_b, wo_b_input
        )
    else:
        output = wo_b(wo_b_input)

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
            layer_idx=layer_idx,
            chunk_index=chunk_index,
            wo_b_local=wo_b_local,
        )
    return output
