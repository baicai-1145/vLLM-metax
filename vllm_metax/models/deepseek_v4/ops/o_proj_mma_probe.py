# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Isolated TileLang MMA probe for the fixed TP=4 DeepSeek-V4 O projection.

This module is intentionally not imported by the production O projection.  It
exists to answer one narrow question: can the C500 matrix-core path reproduce
the BF16 inverse-RoPE plus grouped ``wo_a`` contraction at the production
one-token shape?  The contraction dimensions are fixed and it writes to
caller-owned storage, so it is suitable for graph-capture experiments.
"""

from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang


GROUPS = 2
HEADS_PER_GROUP = 8
HEADS = GROUPS * HEADS_PER_GROUP
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
RANK = 1024
D = HEADS_PER_GROUP * HEAD_DIM
M_TILE = 128
N_TILE = 128
K_TILE = 32


if current_platform.is_cuda_alike() and has_tilelang():
    import tilelang
    import tilelang.language as T
    from tilelang.language.math_intrinsics import ieee_add, ieee_mul
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    ieee_add = None  # type: ignore[assignment]
    ieee_mul = None  # type: ignore[assignment]


if tilelang is not None:

    @tilelang.jit(
        execution_backend="cython",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def _o_proj_mma_kernel() -> Any:
        """Compile the fixed [1,16,512] grouped O-projection MMA kernel."""

        max_position = T.dynamic("max_position")

        @T.prim_func
        def o_proj_mma(
            o: T.Tensor[(1, HEADS, HEAD_DIM), T.bfloat16],
            positions: T.Tensor[(1,), T.int64],
            cos_sin: T.Tensor[(max_position, ROPE_DIM), T.float32],
            wo_a: T.Tensor[(GROUPS, RANK, D), T.bfloat16],
            z: T.Tensor[(1, GROUPS, RANK), T.bfloat16],
        ) -> None:
            # Every CTA computes 128 rank rows for one output group.  N is
            # padded to 128 because the C500 MMA instruction requires
            # matrix-core-compatible M/N/K tiles; only column zero is used.
            with T.Kernel(RANK // M_TILE, GROUPS, threads=128) as (rank_tile, group):
                a_shared = T.alloc_shared((M_TILE, K_TILE), T.bfloat16)
                b_shared = T.alloc_shared((K_TILE, N_TILE), T.bfloat16)
                c_local = T.alloc_fragment((M_TILE, N_TILE), T.float32)
                T.clear(c_local)

                position = positions[0]
                for k_tile in T.serial(D // K_TILE):
                    for i, k in T.Parallel(M_TILE, K_TILE):
                        weight = T.cast(
                            wo_a[
                                group,
                                rank_tile * M_TILE + i,
                                k_tile * K_TILE + k,
                            ],
                            T.float32,
                        )
                        a_shared[i, k] = T.cast(weight, T.bfloat16)

                    for k, j in T.Parallel(K_TILE, N_TILE):
                        d_idx = k_tile * K_TILE + k
                        head_in_group = d_idx // HEAD_DIM
                        dim = d_idx - head_in_group * HEAD_DIM
                        head = group * HEADS_PER_GROUP + head_in_group
                        value = T.alloc_var(T.float32)
                        value = T.cast(o[0, head, dim], T.float32)
                        if dim >= NOPE_DIM:
                            rope_local = dim - NOPE_DIM
                            partner = T.cast(o[0, head, dim ^ 1], T.float32)
                            cache_col = rope_local >> 1
                            cos_value = cos_sin[position, cache_col]
                            sin_value = cos_sin[
                                position, ROPE_DIM // 2 + cache_col
                            ]
                            x_add = ieee_add(
                                ieee_mul(value, cos_value, "rn"),
                                ieee_mul(partner, sin_value, "rn"),
                                "rn",
                            )
                            # Inverse RoPE's odd lane is x*cos - partner*sin.
                            x_sub = ieee_add(
                                ieee_mul(value, cos_value, "rn"),
                                -ieee_mul(partner, sin_value, "rn"),
                                "rn",
                            )
                            value = T.if_then_else(
                                (rope_local & 1) == 0, x_add, x_sub
                            )
                        # inv_rope stores BF16 before the grouped GEMM.  Keep
                        # that rounding boundary explicit before MMA conversion.
                        value = T.cast(T.cast(value, T.bfloat16), T.float32)
                        b_shared[k, j] = T.cast(value, T.bfloat16)

                    T.sync_threads()
                    T.gemm(a_shared, b_shared, c_local, False, False)
                    T.sync_threads()

                for i, j in T.Parallel(M_TILE, N_TILE):
                    if j == 0:
                        z[0, group, rank_tile * M_TILE + i] = T.cast(
                            c_local[i, j], T.bfloat16
                        )

        return o_proj_mma

else:

    def _o_proj_mma_kernel() -> Any:
        raise RuntimeError("TileLang is unavailable on this platform")


def _validate_contract(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.Tensor,
    z: torch.Tensor,
) -> None:
    if tuple(o.shape) != (1, HEADS, HEAD_DIM) or o.dtype != torch.bfloat16:
        raise ValueError("o_proj MMA probe requires BF16 o[1,16,512]")
    if tuple(positions.shape) != (1,) or positions.dtype != torch.int64:
        raise ValueError("o_proj MMA probe requires int64 positions[1]")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] <= 0
        or cos_sin_cache.shape[1] != ROPE_DIM
    ):
        raise ValueError("o_proj MMA probe requires FP32 cos_sin[max_position,64]")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("o_proj MMA probe requires FP32 cos_sin")
    if tuple(wo_a.shape) != (GROUPS, RANK, D) or wo_a.dtype != torch.bfloat16:
        raise ValueError("o_proj MMA probe requires BF16 wo_a[2,1024,4096]")
    if tuple(z.shape) != (1, GROUPS, RANK) or z.dtype != torch.bfloat16:
        raise ValueError("o_proj MMA probe requires caller-owned BF16 z[1,2,1024]")
    tensors = (o, positions, cos_sin_cache, wo_a, z)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("o_proj MMA probe requires CUDA tensors")
    if any(t.device != o.device for t in tensors):
        raise ValueError("o_proj MMA probe tensors must share one CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("o_proj MMA probe tensors must be contiguous")


@torch.no_grad()
def run_o_proj_mma_probe(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.Tensor,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the isolated fixed-shape probe into caller-owned BF16 storage."""
    if z is None:
        z = torch.empty((1, GROUPS, RANK), device=o.device, dtype=torch.bfloat16)
    _validate_contract(o, positions, cos_sin_cache, wo_a, z)
    _o_proj_mma_kernel()(o, positions, cos_sin_cache, wo_a, z)
    return z


__all__ = ["run_o_proj_mma_probe"]
