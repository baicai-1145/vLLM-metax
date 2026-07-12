# SPDX-License-Identifier: Apache-2.0
"""Tests for the caller-owned FP32 row-softmax extension."""

import pytest
import torch

import vllm_metax._metax_sparse_C  # noqa: F401


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible device"
)


def _softmax(input: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops._metax_sparse_C.softmax_fp32_out(input, out)


def test_softmax_fp32_out_128_columns() -> None:
    input = torch.randn((9, 128), device="cuda", dtype=torch.float32)
    out = torch.empty_like(input)

    _softmax(input, out)

    torch.testing.assert_close(out, torch.softmax(input, dim=-1), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cols", [1, 127, 129, 1024])
def test_softmax_fp32_out_supported_column_boundaries(cols: int) -> None:
    input = torch.randn((3, cols), device="cuda", dtype=torch.float32)
    out = torch.empty_like(input)

    _softmax(input, out)

    torch.testing.assert_close(out, torch.softmax(input, dim=-1), atol=1e-6, rtol=1e-5)


def test_softmax_fp32_out_cuda_graph_replay() -> None:
    input = torch.randn((7, 128), device="cuda", dtype=torch.float32)
    out = torch.empty_like(input)
    _softmax(input, out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _softmax(input, out)
    torch.cuda.synchronize()

    for _ in range(3):
        input.normal_()
        expected = torch.softmax(input, dim=-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "case", ["dtype", "shape", "noncontiguous", "columns"]
)
def test_softmax_fp32_out_rejects_invalid(case: str) -> None:
    input = torch.randn((4, 128), device="cuda", dtype=torch.float32)
    out = torch.empty_like(input)
    if case == "dtype":
        out = out.to(torch.bfloat16)
    elif case == "shape":
        out = torch.empty((4, 127), device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        input = torch.randn((128, 4), device="cuda", dtype=torch.float32).t()
    elif case == "columns":
        input = torch.randn((1, 1025), device="cuda", dtype=torch.float32)
        out = torch.empty_like(input)
    with pytest.raises(RuntimeError, match="softmax_fp32_out"):
        _softmax(input, out)
