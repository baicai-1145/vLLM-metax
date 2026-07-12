# SPDX-License-Identifier: Apache-2.0
"""Tests for the caller-owned BF16 x BF16 -> FP32 GEMM extension."""

import pytest
import torch

import vllm_metax._metax_sparse_C  # noqa: F401


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible device"
)


def _gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops._metax_sparse_C.gemm_bf16_fp32_out(a, b, out)


def test_gemm_bf16_fp32_out_differential_and_repeated() -> None:
    a = torch.randn((7, 13), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((11, 13), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((7, 11), device="cuda", dtype=torch.float32)

    for _ in range(3):
        a.normal_()
        b.normal_()
        _gemm(a, b, out)
        torch.testing.assert_close(out, a.float() @ b.float().t(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("case", ["dtype", "shape", "noncontiguous", "alias"])
def test_gemm_bf16_fp32_out_rejects_invalid(case: str) -> None:
    a = torch.randn((4, 5), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((6, 5), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((4, 6), device="cuda", dtype=torch.float32)
    if case == "dtype":
        b = b.float()
    elif case == "shape":
        out = torch.empty((4, 5), device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        a = torch.randn((5, 4), device="cuda", dtype=torch.bfloat16).t()
    elif case == "alias":
        a = torch.randn((4, 6), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((3, 6), device="cuda", dtype=torch.bfloat16)
        out = a.view(torch.float32)

    with pytest.raises(RuntimeError, match="gemm_bf16_fp32_out"):
        _gemm(a, b, out)
