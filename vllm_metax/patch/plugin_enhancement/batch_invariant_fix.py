# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
#
# Patch ``vllm.model_executor.layers.batch_invariant.matmul_persistent``
# with a MetaX-compatible implementation.
#
# The upstream persistent matmul kernel uses ``tl.range(flatten=True)``
# which requires Triton 3.1+.  MetaX ships Triton 3.0.0, so the upstream
# kernel fails to compile.  This patch provides a functionally equivalent
# implementation that:
#
#   * Uses the same tiled ``tl.dot`` algorithm (tensor-core matmul).
#   * Keeps a fixed K accumulation order per tile (M-invariant).
#   * Drops the ``flatten=True`` directive (plain Python ``for`` loop).
#
# When ``VLLM_BATCH_INVARIANT=1`` is set, ``UnquantizedLinearMethod.apply``
# dispatches to ``linear_batch_invariant`` → ``matmul_batch_invariant`` →
# ``matmul_persistent``, so patching this single function makes every
# unquantized BF16 linear layer (including the shared expert) M-invariant.

from __future__ import annotations

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_persistent_maca_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    tile_id = start_pid
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    while tile_id < num_tiles:
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K,
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(
                tl.float32
            )
            accumulator += bias
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

        tile_id += NUM_SMS


def matmul_persistent_maca(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties(a.device.index).multi_processor_count

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 4,
        },
    }

    cfg = configs.get(dtype)
    if cfg is None:
        # Fallback for non-bf16: use bf16 config as approximation
        cfg = configs[torch.bfloat16]

    grid = (min(NUM_SMS, triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"])),)

    _matmul_persistent_maca_kernel[grid](
        a,
        b,
        c,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return c


def linear_msafe_wrapper(input, weight, bias=None):
    """Drop-in replacement for upstream ``linear_batch_invariant``."""
    out = matmul_persistent_maca(input, weight.t())
    if bias is not None:
        out = out + bias
    return out


def apply_patch():
    """Monkey-patch upstream ``linear_batch_invariant`` on MetaX.

    We replace the whole ``linear_batch_invariant`` function (which in turn
    replaces ``matmul_persistent``) because the upstream persistent kernel
    uses ``tl.range(flatten=True)`` unavailable in Triton 3.0.
    """
    bi_name = "vllm.model_executor.layers.batch_invariant"
    bi_module = sys.modules.get(bi_name)
    if bi_module is None:
        return False  # not imported yet; import hook will retry
    if getattr(
        bi_module.linear_batch_invariant, "__name__", ""
    ) != "linear_msafe_wrapper":
        bi_module.linear_batch_invariant = linear_msafe_wrapper
        bi_module.matmul_persistent = matmul_persistent_maca
    return True


# Attempt import-time patch.  If ``batch_invariant`` is not yet imported,
# install an import hook that retries once it is imported.
if not apply_patch():
    import importlib.abc
    import importlib.machinery

    class _BatchInvariantLoader(importlib.abc.Loader):
        """Wrapper that applies the patch after batch_invariant is loaded."""

        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            return self._real.create_module(spec) if hasattr(self._real, "create_module") else None

        def exec_module(self, module):
            self._real.exec_module(module)
            apply_patch()

    _bi_name = "vllm.model_executor.layers.batch_invariant"
    _patched_finders = []

    class _BatchInvariantFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _bi_name:
                return None
            # Find the real spec via other finders
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path)
                if spec is not None:
                    spec.loader = _BatchInvariantLoader(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _BatchInvariantFinder())
