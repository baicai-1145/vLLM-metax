# SPDX-License-Identifier: Apache-2.0
"""Tests for the graph-safe in-place FP32 MoE GEMM."""

import pytest
import torch

import vllm_metax._metax_sparse_C  # noqa: F401


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires a CUDA-compatible device")


def _gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops._metax_sparse_C.gemm_fp32_out(a, b, out)


def test_gemm_fp32_out_non_aligned_differential() -> None:
    a = torch.randn((7, 13), device="cuda", dtype=torch.float32)
    b = torch.randn((11, 13), device="cuda", dtype=torch.float32)
    out = torch.empty((7, 11), device="cuda", dtype=torch.float32)

    _gemm(a, b, out)
    torch.testing.assert_close(out, a @ b.t(), atol=1e-5, rtol=1e-5)


def test_gemm_fp32_out_identity_and_guards() -> None:
    n = 9
    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.eye(n, device="cuda", dtype=torch.float32)
    storage = torch.full((n * n + 2, ), 12345.0, device="cuda")
    out = storage[1:-1].view(n, n)

    _gemm(a, b, out)
    # Compare against the framework GEMM: the device backend may use a
    # non-bitwise FP32 accumulation path even for an identity operand.
    torch.testing.assert_close(out, a @ b.t(), atol=1e-5, rtol=1e-5)
    assert storage[0].item() == 12345.0
    assert storage[-1].item() == 12345.0


@pytest.mark.parametrize("case",
                         ["dtype", "shape", "noncontiguous", "alias", "device"])
def test_gemm_fp32_out_rejects_invalid(case: str) -> None:
    a = torch.randn((4, 5), device="cuda", dtype=torch.float32)
    b = torch.randn((6, 5), device="cuda", dtype=torch.float32)
    out = torch.empty((4, 6), device="cuda", dtype=torch.float32)
    if case == "dtype":
        b = b.to(torch.float16)
    elif case == "shape":
        out = torch.empty((4, 5), device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        a = torch.randn((5, 4), device="cuda", dtype=torch.float32).t()
    elif case == "alias":
        a = torch.randn((4, 4), device="cuda", dtype=torch.float32)
        b = torch.randn((4, 4), device="cuda", dtype=torch.float32)
        out = b
    elif case == "device":
        b = torch.randn((6, 5), device="cpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="gemm_fp32_out"):
        _gemm(a, b, out)


def test_gemm_fp32_out_repeated() -> None:
    a = torch.empty((9, 13), device="cuda", dtype=torch.float32)
    b = torch.empty((11, 13), device="cuda", dtype=torch.float32)
    out = torch.empty((9, 11), device="cuda", dtype=torch.float32)
    for _ in range(8):
        a.normal_()
        b.normal_()
        _gemm(a, b, out)
        torch.testing.assert_close(out, a @ b.t(), atol=1e-5, rtol=1e-5)


def test_gemm_fp32_out_cuda_graph_replay() -> None:
    a = torch.randn((7, 13), device="cuda", dtype=torch.float32)
    b = torch.randn((11, 13), device="cuda", dtype=torch.float32)
    out = torch.empty((7, 11), device="cuda", dtype=torch.float32)
    _gemm(a, b, out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _gemm(a, b, out)
    torch.cuda.synchronize()

    for _ in range(3):
        a.normal_()
        b.normal_()
        expected = a @ b.t()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
