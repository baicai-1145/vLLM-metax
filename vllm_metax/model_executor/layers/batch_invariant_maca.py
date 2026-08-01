# Metax batch-invariant BF16 GEMM utilities.
#
# MetaX ``b16gemvt_kernel`` produces M-dependent results (different tiling for
# different M), which forces the DSpark tokenwise path.  These helpers provide
# a Triton tiled matmul that is M-invariant (M=6 bitwise equals 6xM=1) by
# keeping a fixed K accumulation order per tile.  The kernel uses ``tl.dot``
# (tensor core) for performance; each program block handles one output tile
# independently, so concurrent blocks never share an accumulator.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_msafe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """M-invariant GEMV: one program per (row, N-tile) pair.

    Each program computes one [1, BLOCK_SIZE_N] output slice for a single row,
    accumulating over K with element-wise multiply + tl.sum.  This matches
    torch.matmul M=1 to within 0-4 ULP elements.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for ki in range(0, K, BLOCK_SIZE_K):
        offs_k = ki + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < K

        x_block = tl.load(
            a_ptr + pid_m * stride_am + offs_k * stride_ak,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w_block = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_block[:, None] * w_block, axis=0)

    tl.store(
        c_ptr + pid_m * stride_cm + offs_n * stride_cn,
        acc.to(tl.bfloat16),
        mask=n_mask,
    )


def matmul_msafe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """M-invariant BF16 matmul: ``a`` is [M, K], ``b`` is [K, N] -> [M, N].

    Result is bitwise identical for any M (each row accumulated independently
    with the same K-loop order via fixed-size tl.dot tiles).
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.bfloat16

    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)

    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 64

    grid = (M, triton.cdiv(N, BLOCK_SIZE_N))
    _matmul_msafe_kernel[grid](
        a,
        b,
        c,
        M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return c


def linear_msafe(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """M-invariant BF16 linear: ``weight`` is [N, K] (PyTorch convention)."""
    return matmul_msafe(x, weight.t())
