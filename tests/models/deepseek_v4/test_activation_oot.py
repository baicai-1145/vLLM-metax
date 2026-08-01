# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_metax.customized.ops.activation import MacaSiluAndMulWithClamp


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(6, 2 * 3584), (17, 2 * 3584)])
@torch.inference_mode()
def test_maca_silu_and_mul_with_clamp_default_params_match_native(dtype, shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=dtype)

    with set_current_vllm_config(VllmConfig()):
        layer = MacaSiluAndMulWithClamp(swiglu_limit=7.0)
        out = layer.forward_oot(x)
        ref = layer.forward_native(x)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_maca_silu_and_mul_with_clamp_non_default_params_fallback():
    x = torch.randn((3, 16), dtype=torch.float32)

    with set_current_vllm_config(VllmConfig()):
        layer = MacaSiluAndMulWithClamp(swiglu_limit=7.0, alpha=0.5, beta=1.0)
        out = layer.forward_oot(x)
        ref = layer.forward_native(x)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
