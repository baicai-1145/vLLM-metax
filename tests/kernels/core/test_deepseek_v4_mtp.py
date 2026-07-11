# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm_metax.patch  # noqa: F401
from vllm.models.deepseek_v4.common.ops import fused_mtp_input_rmsnorm


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


@torch.inference_mode()
def test_fused_mtp_input_rmsnorm() -> None:
    torch.manual_seed(0)
    num_tokens, hc_mult, hidden = 2, 4, 256
    inputs_embeds = torch.randn(
        num_tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    previous_hidden_states = torch.randn(
        num_tokens,
        hc_mult,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    enorm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    actual_enorm, actual_hnorm = fused_mtp_input_rmsnorm(
        inputs_embeds,
        positions,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        1e-6,
        hc_mult,
    )

    masked_inputs = torch.where(
        positions.unsqueeze(-1) == 0,
        torch.zeros_like(inputs_embeds),
        inputs_embeds,
    )
    expected_enorm = rmsnorm_reference(masked_inputs, enorm_weight, 1e-6)
    expected_hnorm = rmsnorm_reference(
        previous_hidden_states,
        hnorm_weight,
        1e-6,
    )
    torch.testing.assert_close(actual_enorm, expected_enorm, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_hnorm, expected_hnorm, atol=1e-2, rtol=1e-2)
