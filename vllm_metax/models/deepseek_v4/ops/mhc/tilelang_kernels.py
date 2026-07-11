# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv

# TileLang is used for MHC on CUDA and ROCm. Keep non-GPU imports cheap so
# registering the Python wrapper modules does not require TileLang everywhere.
if TYPE_CHECKING or current_platform.is_cuda_alike():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
    from tilelang.language.math_intrinsics import ieee_add, ieee_mul
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    ieee_add = None  # type: ignore[assignment]
    ieee_mul = None  # type: ignore[assignment]


@cache
def compute_num_split(num_tokrns: int) -> int:
    return 16 if num_tokrns >= 512 else 64


def _tf32_round(value):
    bits = T.reinterpret(value, T.int32)
    return T.reinterpret(T.bitwise_and(bits + 0x1000, -0x2000), T.float32)


def _torch_like_scale_add(value, scale, base):
    return ieee_add(ieee_mul(value, scale, "rn"), base, "rn")


def _validate_exact_raw_contract(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> None:
    if tuple(residual_cur.shape) != (1, 4, 4096):
        raise ValueError(f"exact MHC raw residual shape mismatch: {residual_cur.shape}")
    if tuple(gemm_out_mul.shape) != (1, 1, 24):
        raise ValueError(f"exact MHC raw mul shape mismatch: {gemm_out_mul.shape}")
    if tuple(gemm_out_sqrsum.shape) != (1, 1):
        raise ValueError(
            f"exact MHC raw sqrsum shape mismatch: {gemm_out_sqrsum.shape}"
        )
    if tuple(hc_scale.shape) != (3,) or tuple(hc_base.shape) != (24,):
        raise ValueError("exact MHC raw scale/base shape mismatch")
    if (
        residual_cur.dtype != torch.bfloat16
        or gemm_out_mul.dtype != torch.float32
        or gemm_out_sqrsum.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
    ):
        raise ValueError("exact MHC raw dtype mismatch")
    if not (
        n_splits == 1
        and rms_eps == 1e-6
        and hc_pre_eps == 1e-6
        and hc_sinkhorn_eps == 1e-6
        and hc_post_mult_value == 2.0
        and sinkhorn_repeat == 20
    ):
        raise ValueError("exact MHC raw scalar contract mismatch")


def _mhc_pre_from_raw_exact_trace(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> dict[str, torch.Tensor]:
    _validate_exact_raw_contract(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    from .debug_diff import mhc_pre_from_raw_trace_torch

    return mhc_pre_from_raw_trace_torch(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_pre_from_raw_exact_fuse(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    *,
    post_mix_out: torch.Tensor | None = None,
    comb_mix_out: torch.Tensor | None = None,
    layer_input_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trace = _mhc_pre_from_raw_exact_trace(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    post_mix = trace["post_mix"].view(1, 4)
    comb_mix = trace["sinkhorn_col_19"].view(1, 16)
    layer_input = trace["layer_input_bf16"].view(1, 4096)
    if post_mix_out is not None:
        post_mix_out.copy_(post_mix)
        post_mix = post_mix_out
    if comb_mix_out is not None:
        comb_mix_out.copy_(comb_mix)
        comb_mix = comb_mix_out
    if layer_input_out is not None:
        layer_input_out.copy_(layer_input)
        layer_input = layer_input_out
    return post_mix, comb_mix, layer_input


@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=128) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 64:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            # The second warp consumes values published by the first warp.
            T.sync_threads()

            if T.get_thread_binding() < 64:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            _torch_like_scale_add(
                                mixes_shared[j + mhc_mult],
                                mhc_scale[1],
                                mhc_base[j + mhc_mult],
                            )
                        )
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = _torch_like_scale_add(
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2],
                        mhc_scale[2],
                        mhc_base[j * mhc_mult + k + mhc_mult * 2],
                    )

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre)
                pre_mix = T.alloc_fragment(mhc_mult, T.float32)
                for j in T.serial(mhc_mult):
                    pre_mix[j] = (
                        T.sigmoid(
                            _torch_like_scale_add(
                                mixes_shared[j],
                                mhc_scale[0],
                                mhc_base[j],
                            ),
                        )
                        + mhc_pre_eps
                    )
                ###################################################################
                # _mhc_pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    if mhc_mult == 4:
                        pre0 = T.alloc_fragment(1, T.float32)
                        pre1 = T.alloc_fragment(1, T.float32)
                        pre2 = T.alloc_fragment(1, T.float32)
                        pre3 = T.alloc_fragment(1, T.float32)
                        pre0[0] = pre_mix[0]
                        pre1[0] = pre_mix[1]
                        pre2[0] = pre_mix[2]
                        pre3[0] = pre_mix[3]
                        x0 = T.alloc_fragment(hidden_block, T.float32)
                        x1 = T.alloc_fragment(hidden_block, T.float32)
                        x2 = T.alloc_fragment(hidden_block, T.float32)
                        x3 = T.alloc_fragment(hidden_block, T.float32)
                        T.copy(xs[0, 0], x0, disable_tma=True)
                        T.copy(xs[1, 0], x1, disable_tma=True)
                        T.copy(xs[2, 0], x2, disable_tma=True)
                        T.copy(xs[3, 0], x3, disable_tma=True)
                        for i1_h in T.Parallel(hidden_block):
                            term0 = ieee_mul(pre0[0], x0[i1_h], "rn")
                            term1 = ieee_mul(pre1[0], x1[i1_h], "rn")
                            term2 = ieee_mul(pre2[0], x2[i1_h], "rn")
                            term3 = ieee_mul(pre3[0], x3[i1_h], "rn")
                            ol[i1_h] = ieee_add(
                                ieee_add(term0, term2, "rn"),
                                ieee_add(term1, term3, "rn"),
                                "rn",
                            )
                    else:
                        for i_mhc in T.serial(mhc_mult):
                            mhc_idx = mhc_mult - 1 - i_mhc
                            pre = pre_mix[mhc_idx]
                            for i1_h in T.Parallel(hidden_block):
                                ol[i1_h] = ieee_add(
                                    ol[i1_h],
                                    ieee_mul(pre, xl[mhc_idx, i1_h], "rn"),
                                    "rn",
                                )

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse


@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_pre_mix_debug(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    @T.prim_func
    def mhc_pre_mix_debug(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        normalized_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        pre_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=64) as pid:
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(mhc_mult3, T.float32)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, pid]
            rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(mhc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, pid, j]
                mixes[j] *= rms[0]
                normalized_mixes[pid, j] = mixes[j]
            for j in T.Parallel(mhc_mult):
                pre_mix[pid, j] = (
                    T.sigmoid(_torch_like_scale_add(mixes[j], mhc_scale[0], mhc_base[j]))
                    + mhc_pre_eps
                )

    return mhc_pre_mix_debug


@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_post_fwd(
    mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = 0.0
                    for i_mhci in T.serial(mhc):
                        mhc_idx = mhc - 1 - i_mhci
                        x_local[i_mhco, i1_h] = ieee_add(
                            x_local[i_mhco, i1_h],
                            ieee_mul(
                                _tf32_round(a_local[mhc_idx, i_mhco]),
                                b_local[mhc_idx, i1_h],
                                "rn",
                            ),
                            "rn",
                        )
                    x_local[i_mhco, i1_h] = ieee_add(
                        x_local[i_mhco, i1_h],
                        ieee_mul(c_local[i_mhco], d_local[i1_h], "rn"),
                        "rn",
                    )
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

    return _mhc_post_fwd_kernel


def _mhc_post_exact_tl(
    x_flat: torch.Tensor,
    residual_flat: torch.Tensor,
    post_layer_mix_flat: torch.Tensor,
    comb_res_mix_flat: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed decode-shape TileLang post candidate with a stable output buffer."""
    expected = (1, 4, 4096)
    if x_flat.dtype != torch.bfloat16 or tuple(x_flat.shape) != (1, 4096):
        raise ValueError("exact MHC post requires x BF16[1,4096]")
    if residual_flat.dtype != torch.bfloat16 or tuple(residual_flat.shape) != expected:
        raise ValueError("exact MHC post requires residual BF16[1,4,4096]")
    if post_layer_mix_flat.dtype != torch.float32 or tuple(post_layer_mix_flat.shape) != (1, 4):
        raise ValueError("exact MHC post requires post mix FP32[1,4]")
    if comb_res_mix_flat.dtype != torch.float32 or tuple(comb_res_mix_flat.shape) != (1, 4, 4):
        raise ValueError("exact MHC post requires comb mix FP32[1,4,4]")
    if out is None:
        out = torch.empty_like(residual_flat)
    if out.dtype != torch.bfloat16 or tuple(out.shape) != expected:
        raise ValueError("exact MHC post requires output BF16[1,4,4096]")
    _mhc_post_exact_mma()(
        comb_res_mix_flat, residual_flat, post_layer_mix_flat, x_flat, out
    )
    return out


@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_post_exact_mma() -> tilelang.JITKernel:
    """Exact post map using the MetaX TF32 MMA path.

    The MMA emitter requires M/K tiles compatible with the 16x16x8 TF32
    instruction. The logical four MHC streams are therefore embedded in a
    128x32 by 32x128 tile; padded rows and K values are zero and cannot affect
    the first four output rows.
    """

    @T.prim_func
    def mhc_post_exact_mma(
        comb_mix: T.Tensor[(1, 4, 4), T.float32],
        residual: T.Tensor[(1, 4, 4096), T.bfloat16],
        post_mix: T.Tensor[(1, 4), T.float32],
        x: T.Tensor[(1, 4096), T.bfloat16],
        out: T.Tensor[(1, 4, 4096), T.bfloat16],
    ) -> None:
        with T.Kernel(32, threads=128) as bx:
            a_shared = T.alloc_shared((128, 32), T.tfloat32)
            b_shared = T.alloc_shared((32, 128), T.tfloat32)
            c_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(c_local)
            for i, k in T.Parallel(128, 32):
                if i < 4 and k < 4:
                    a_shared[i, k] = T.cast(
                        _tf32_round(comb_mix[0, k, i]), T.tfloat32
                    )
                else:
                    a_shared[i, k] = T.cast(0.0, T.tfloat32)

            for k, j in T.Parallel(32, 128):
                if k < 4:
                    b_shared[k, j] = T.cast(
                        T.cast(residual[0, k, bx * 128 + j], T.float32),
                        T.tfloat32,
                    )
                else:
                    b_shared[k, j] = T.cast(0.0, T.tfloat32)

            T.gemm(a_shared, b_shared, c_local, False, False)

            for i, j in T.Parallel(4, 128):
                post_term = ieee_mul(
                    post_mix[0, i], T.cast(x[0, bx * 128 + j], T.float32), "rn"
                )
                value = ieee_add(c_local[i, j], post_term, "rn")
                out[0, i, bx * 128 + j] = T.cast(value, T.bfloat16)

    return mhc_post_exact_mma


@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
    round_weight: bool = True,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA"""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr
    num_warps = n_thr // 64

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = T.get_warp_idx()
        lane = T.get_lane_idx()

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)
        T.sync_threads()

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = _tf32_round(s_comb[k, j])

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = 0.0
                for k in T.unroll(hc):
                    mhc_idx = k
                    new_r[j] = ieee_add(
                        new_r[j],
                        ieee_mul(cm[mhc_idx, j], residual_in[i_n, mhc_idx, h_idx], "rn"),
                        "rn",
                    )
                new_r[j] = ieee_add(
                    new_r[j],
                    ieee_mul(pm[j], x_in[i_n, h_idx], "rn"),
                    "rn",
                )
                # mhc_post returns BF16. The following prenorm/GEMM must
                # consume that rounded value, not the pre-cast FP32 value.
                new_r[j] = T.cast(T.cast(new_r[j], T.bfloat16), T.float32)

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    weight_value = T.alloc_var(T.float32)
                    weight_value = weight_t[i_nt * tile_n + n, j, h_idx]
                    if round_weight:
                        weight_value = _tf32_round(weight_value)
                    acc[n] += (
                        weight_value * new_r[j]
                    )

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        # Cross-warp reduce via shared mem
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()

        # Warp 0 does the final cross-warp sum and writes outputs
        if warp_id == 0:
            if lane < tile_n:
                v = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v += s_warp[w, lane]
                yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_n] = v2

@tilelang.jit(
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def hc_head_fuse_tilelang(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Two-pass fused kernel for hc_head.

    Pass 1: accumulate per-token squared sum and hc_mult dot-products
            (projections onto fn rows) using cross-thread reducers.
    Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

    Avoids materialising mixes / rsqrt / pre tensors to global memory.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc_mult * hidden_size
    h_block = math.gcd(h_blk, hidden_size)
    n_h = hidden_size // h_block

    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]
    fn: T.Tensor[[hc_mult, hc_dim], T.float32]  # type: ignore[no-redef,valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef,valid-type]
    hc_base: T.Tensor[[hc_mult], T.float32]  # type: ignore[no-redef,valid-type]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        # ------------------------------------------------------------------
        # Pass 1 – for each residual channel m_c and h_block:
        #   • accumulate squared sum (for RMS norm denominator)
        #   • accumulate hc_mult dot-products with fn rows
        # ------------------------------------------------------------------
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for m_c in T.serial(hc_mult):
            for i_h in T.serial(n_h):
                x_local = T.alloc_fragment(h_block, T.float32)
                T.copy(residual[i, m_c, i_h * h_block], x_local)

                for k in T.Parallel(h_block):
                    sqrsum_r[0] += x_local[k] * x_local[k]

                for m_m in T.unroll(hc_mult):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # ------------------------------------------------------------------
        # Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
        # ------------------------------------------------------------------
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
        for m in T.Parallel(hc_mult):
            pre_mix_shared[m] = (
                T.sigmoid(
                    _torch_like_scale_add(
                        mixes_r[m] * rsqrt_val[0],
                        hc_scale[0],
                        hc_base[m],
                    )
                )
                + hc_eps
            )
        T.sync_threads()

        # ------------------------------------------------------------------
        # Pass 2 – apply_mix: pipelined weighted sum over residual channels
        # ------------------------------------------------------------------
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc_mult, h_block), T.float32)
            T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)
