# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone sparse MLA FP32 GEMM extension."""

import sys

import pytest
import torch

import vllm_metax._metax_sparse_C  # noqa: F401
from vllm_metax.kernels.sparse_mla_decode import _gemm_fp32_out_op


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible device"
)


def test_metax_sparse_gemm_resolver_uses_unique_namespace() -> None:
    op = _gemm_fp32_out_op()
    assert op == torch.ops._metax_sparse_C.gemm_fp32_out
    assert "vllm_metax._moe_C" not in sys.modules


def test_metax_sparse_gemm_repeated_and_cuda_graph() -> None:
    a = torch.randn((7, 13), device="cuda", dtype=torch.float32)
    b = torch.randn((11, 13), device="cuda", dtype=torch.float32)
    out = torch.empty((7, 11), device="cuda", dtype=torch.float32)

    op = _gemm_fp32_out_op()
    op(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a @ b.t(), atol=1e-5, rtol=1e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(a, b, out)
    for _ in range(3):
        a.normal_()
        b.normal_()
        expected = a @ b.t()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(64, 256, 512), (64, 512, 256)],
)
@pytest.mark.parametrize("batch", [2, 5, 6])
def test_metax_sparse_grouped_gemm_matches_serial_tokens(
    batch: int, m: int, n: int, k: int
) -> None:
    torch.manual_seed(157 + k)
    a = torch.randn((batch, m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((batch, n, k), device="cuda", dtype=torch.float32)
    expected = torch.empty((batch, m, n), device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    serial_op = _gemm_fp32_out_op()

    for token in range(batch):
        serial_op(a[token], b[token], expected[token])
    torch.ops._metax_sparse_C.gemm_fp32_strided_batched_out(a, b, actual)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_metax_sparse_grouped_gemm_preserves_nonfinite_masks() -> None:
    torch.manual_seed(191)
    a = torch.randn((6, 64, 256), device="cuda", dtype=torch.float32)
    b = torch.randn((6, 512, 256), device="cuda", dtype=torch.float32)
    a[0, 0, 0] = float("nan")
    a[1, 0, 0] = float("inf")
    expected = torch.empty((6, 64, 512), device="cuda", dtype=torch.float32)
    actual = torch.full_like(expected, float("nan"))
    serial_op = _gemm_fp32_out_op()

    for token in range(6):
        serial_op(a[token], b[token], expected[token])
    torch.ops._metax_sparse_C.gemm_fp32_strided_batched_out(a, b, actual)
    torch.cuda.synchronize()

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite], expected[finite])
