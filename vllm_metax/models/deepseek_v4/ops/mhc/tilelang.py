# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from .tilelang_kernels import (
    compute_num_split,
    _mhc_pre_from_raw_exact_fuse,
    _mhc_pre_big_fuse,
    _mhc_pre_mix_debug,
    _mhc_post_fwd,
    mhc_fused_tilelang,
    hc_head_fuse_tilelang
)

logger = init_logger(__name__)
_MHC_DECODE_IMPL_LOGGED: set[str] = set()
_MHC_DECODE_DISPATCH_COUNTS: dict[str, int] = {}


def _require_exact_mhc_tilelang() -> bool:
    return os.getenv("VLLM_METAX_DSV4_MHC_REQUIRE_EXACT_TILELANG", "0") == "1"


def _is_exact_mhc_decode_contract(
    *,
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    n_splits: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> bool:
    return (
        num_tokens == 1
        and hc_mult == 4
        and hidden_size == 4096
        and n_splits == 1
        and rms_eps == 1e-6
        and hc_pre_eps == 1e-6
        and hc_sinkhorn_eps == 1e-6
        and hc_post_mult_value == 2.0
        and sinkhorn_repeat == 20
    )


def _log_mhc_decode_impl(
    name: str,
    *,
    fail_closed: bool,
    explicit: bool = False,
) -> None:
    _MHC_DECODE_DISPATCH_COUNTS[name] = _MHC_DECODE_DISPATCH_COUNTS.get(name, 0) + 1
    key = f"{name}:{fail_closed}:{explicit}"
    if key in _MHC_DECODE_IMPL_LOGGED:
        return
    _MHC_DECODE_IMPL_LOGGED.add(key)
    if name == "exact_tilelang":
        logger.warning(
            "DeepSeek V4 MHC decode implementation: exact_tilelang "
            "fail_closed=%s dispatch_count=%d",
            str(fail_closed).lower(),
            _MHC_DECODE_DISPATCH_COUNTS[name],
        )
    elif name == "torch_oracle":
        logger.warning(
            "DeepSeek V4 MHC decode implementation: torch_oracle "
            "explicit=%s dispatch_count=%d",
            str(explicit).lower(),
            _MHC_DECODE_DISPATCH_COUNTS[name],
        )


def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn  # noqa: F841

    n_splits = compute_num_split(num_tokens)

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    from vllm_metax.utils.deep_gemm import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, mhc_mult * hidden_size),
        fn_flat,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    # END of TileLang implementation of pre-norm-fn forward matmul

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_pre_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f"{x.dtype=}"
    assert residual.dtype == torch.bfloat16, f"{residual.dtype=}"
    assert post_layer_mix.dtype == torch.float32, f"{post_layer_mix.dtype=}"
    assert comb_res_mix.dtype == torch.float32, f"{comb_res_mix.dtype=}"
    assert x.shape == (num_tokens, hidden), f"{x.shape=}"
    assert post_layer_mix.shape == (num_tokens, mhc, 1), f"{post_layer_mix.shape=}"
    assert comb_res_mix.shape == (num_tokens, mhc, mhc), f"{comb_res_mix.shape=}"

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)

    kernel = _mhc_post_fwd(mhc, hidden)
    kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
    )
    return out


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)

    return mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out)


def _mhc_post_torch_bmm(
    x_flat: torch.Tensor,
    residual_flat: torch.Tensor,
    post_layer_mix_flat: torch.Tensor,
    comb_res_mix_flat: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.bmm(comb_res_mix_flat.transpose(1, 2), residual_flat.float())
    return (
        x_flat.float().unsqueeze(-2) * post_layer_mix_flat.unsqueeze(-1) + term2
    ).bfloat16()


def _mhc_apply_mix_torch_sum(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    n_splits: int,
) -> torch.Tensor:
    num_tokens = residual_cur.shape[0]
    hc_mult = residual_cur.shape[1]
    hidden_size = residual_cur.shape[2]
    normalized_mixes = torch.empty(
        num_tokens,
        hc_mult * (2 + hc_mult),
        dtype=torch.float32,
        device=residual_cur.device,
    )
    pre_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual_cur.device,
    )
    _mhc_pre_mix_debug(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        n_splits=n_splits,
        mhc_mult=hc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        normalized_mixes,
        pre_mix,
    )
    return (
        residual_cur.float() * pre_mix.view(num_tokens, hc_mult, 1)
    ).sum(dim=-2).bfloat16()


def _mhc_mixes_from_raw_torch(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .torch import mhc_pre_split_mixes_ref, sinkhorn_normalize_ref

    num_tokens = residual_cur.shape[0]
    hc_mult = residual_cur.shape[1]
    hidden_size = residual_cur.shape[2]
    rms_group_size = hc_mult * hidden_size
    rms = torch.rsqrt(gemm_out_sqrsum.sum(dim=0) / rms_group_size + rms_eps)
    mixes = (gemm_out_mul.sum(dim=0) * rms.unsqueeze(-1)).unsqueeze(0)
    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes_ref(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        hc_post_mult_value,
        hc_pre_eps,
    )
    comb_mix = sinkhorn_normalize_ref(
        comb_mix,
        repeat=sinkhorn_repeat,
        eps=hc_sinkhorn_eps,
    )
    pre_mix = pre_mix.view(num_tokens, hc_mult, 1)
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    return pre_mix, post_mix, comb_mix


def _mhc_exact_sqrsum_torch(residual_cur: torch.Tensor) -> torch.Tensor:
    residual_2d = residual_cur.view(residual_cur.shape[0], -1).float()
    return residual_2d.square().sum(-1).view(1, residual_cur.shape[0])


def _mhc_exact_raw_torch(
    residual_cur: torch.Tensor,
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_2d = residual_cur.view(residual_cur.shape[0], -1).float()
    gemm_out_mul = torch.nn.functional.linear(residual_2d, fn).view(
        1, residual_cur.shape[0], fn.shape[0]
    )
    gemm_out_sqrsum = residual_2d.square().sum(-1).view(1, residual_cur.shape[0])
    return gemm_out_mul, gemm_out_sqrsum


def _mhc_pre_from_raw_torch(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_mix, post_mix, comb_mix = _mhc_mixes_from_raw_torch(
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
    layer_input = (
        residual_cur.float() * pre_mix
    ).sum(dim=-2).bfloat16()
    return post_mix, comb_mix, layer_input


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def mhc_fused_post_pre_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    if num_tokens != 1:
        from .torch import mhc_fused_post_pre as mhc_fused_post_pre_torch

        return mhc_fused_post_pre_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )

    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    exact_raw_mode = os.getenv("VLLM_METAX_DSV4_MHC_EXACT_RAW", "0")
    use_torch_pre = (
        exact_raw_mode == "torch_pre"
        or (
            exact_raw_mode == "0"
            and os.getenv("VLLM_METAX_DSV4_MHC_UNSAFE_TILELANG_PRE", "0") != "1"
        )
    )
    if exact_raw_mode == "1" or use_torch_pre:
        residual_cur = _mhc_post_torch_bmm(
            x_flat,
            residual_flat,
            post_layer_mix_flat,
            comb_res_mix_flat,
        )
        from .debug_diff import maybe_capture_mhc_fused_post_prenorm

        maybe_capture_mhc_fused_post_prenorm(
            x_flat,
            residual_flat,
            post_layer_mix_flat,
            comb_res_mix_flat,
            fn,
            residual_cur,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            n_splits=1,
        )
        residual_cur_view = residual_cur.view(*outer_shape, hc_mult, hidden_size)
        if use_torch_pre:
            from .torch import mhc_pre as mhc_pre_torch

            post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_torch(
                residual_cur_view,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
            return residual_cur_view, post_mix_cur, comb_mix_cur, layer_input_cur
        post_mix_cur = torch.empty(
            num_tokens,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        )
        comb_mix_cur = torch.empty(
            num_tokens,
            hc_mult2,
            dtype=torch.float32,
            device=residual.device,
        )
        layer_input_cur = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        )
        residual_2d = residual_cur.view(num_tokens, hc_hidden_size).float()
        gemm_out_mul = torch.nn.functional.linear(residual_2d, fn).view(
            1, num_tokens, hc_mult3
        )
        gemm_out_sqrsum = residual_2d.square().sum(-1).view(1, num_tokens)
        from .debug_diff import maybe_capture_mhc_pre_raw

        maybe_capture_mhc_pre_raw(
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
            n_splits=1,
        )
        target_exact = _is_exact_mhc_decode_contract(
            num_tokens=num_tokens,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            n_splits=1,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        require_exact = _require_exact_mhc_tilelang()
        use_torch_split_from_raw = (
            os.getenv("VLLM_METAX_DSV4_MHC_TORCH_SPLIT_FROM_RAW", "0") == "1"
        )
        if target_exact and use_torch_split_from_raw and require_exact:
            raise RuntimeError(
                "VLLM_METAX_DSV4_MHC_REQUIRE_EXACT_TILELANG=1 conflicts with "
                "VLLM_METAX_DSV4_MHC_TORCH_SPLIT_FROM_RAW=1"
            )
        if target_exact and not use_torch_split_from_raw:
            post_mix_exact, comb_mix_exact, layer_input_exact = (
                _mhc_pre_from_raw_exact_fuse(
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
                    n_splits=1,
                    post_mix_out=post_mix_cur,
                    comb_mix_out=comb_mix_cur,
                    layer_input_out=layer_input_cur,
                )
            )
            _log_mhc_decode_impl("exact_tilelang", fail_closed=require_exact)
            return (
                residual_cur.view(*outer_shape, hc_mult, hidden_size),
                post_mix_exact.view(*outer_shape, hc_mult, 1),
                comb_mix_exact.view(*outer_shape, hc_mult, hc_mult),
                layer_input_exact.view(*outer_shape, hidden_size),
            )
        if use_torch_split_from_raw:
            if target_exact:
                _log_mhc_decode_impl(
                    "torch_oracle",
                    fail_closed=False,
                    explicit=True,
                )
            post_mix_torch, comb_mix_torch, layer_input_torch = _mhc_pre_from_raw_torch(
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
            return (
                residual_cur.view(*outer_shape, hc_mult, hidden_size),
                post_mix_torch.view(*outer_shape, hc_mult, 1),
                comb_mix_torch.view(*outer_shape, hc_mult, hc_mult),
                layer_input_torch.view(*outer_shape, hidden_size),
            )
        _mhc_pre_big_fuse(
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=1,
            mhc_mult=hc_mult,
        )(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
        )
        if os.getenv("VLLM_METAX_DSV4_MHC_TORCH_MIXES_BIG_LAYER", "0") == "1":
            _, post_mix_torch, comb_mix_torch = _mhc_mixes_from_raw_torch(
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
            return (
                residual_cur.view(*outer_shape, hc_mult, hidden_size),
                post_mix_torch.view(*outer_shape, hc_mult, 1),
                comb_mix_torch.view(*outer_shape, hc_mult, hc_mult),
                layer_input_cur.view(*outer_shape, hidden_size),
            )
        if os.getenv("VLLM_METAX_DSV4_MHC_USE_BIG_FUSE_LAYER", "0") != "1":
            layer_input_cur.copy_(
                _mhc_apply_mix_torch_sum(
                    residual_cur,
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    n_splits=1,
                )
            )
        return (
            residual_cur.view(*outer_shape, hc_mult, hidden_size),
            post_mix_cur.view(*outer_shape, hc_mult, 1),
            comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
            layer_input_cur.view(*outer_shape, hidden_size),
        )

    fma_token_threshold = 16
    if num_tokens <= fma_token_threshold:
        # TODO(gnovack): investigate autotuning these heuristics
        tile_n = 2 if num_tokens < 8 else 3
        # Decode runs one token at a time.  Splitting the reduction changes the
        # FP32 accumulation order enough to cross BF16 boundaries against the
        # Torch MHC contract on real activations, so keep the single-token path
        # unsplit until a split-k kernel is made numerically equivalent.
        n_splits = 1 if num_tokens == 1 else 4
    else:
        n_splits = compute_num_split(num_tokens)

    assert hidden_size % n_splits == 0
    if num_tokens <= fma_token_threshold:
        assert (hidden_size // n_splits) % 256 == 0
        assert hc_mult3 % tile_n == 0

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if num_tokens <= fma_token_threshold:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            split_k=n_splits,
            round_weight=os.getenv(
                "VLLM_METAX_DSV4_MHC_RAW_ROUND_WEIGHT", "1"
            ) != "0",
        )
        if (
            num_tokens == 1
            and n_splits == 1
            and os.getenv(
                "VLLM_METAX_DSV4_MHC_EXACT_RAW_FROM_TILE_POST", "0"
            ) == "1"
        ):
            exact_mul, exact_sqrsum = _mhc_exact_raw_torch(residual_cur, fn)
            gemm_out_mul.copy_(exact_mul)
            gemm_out_sqrsum.copy_(exact_sqrsum)
        if (
            num_tokens == 1
            and n_splits == 1
            and os.getenv("VLLM_METAX_DSV4_MHC_EXACT_SQRSUM", "0") == "1"
        ):
            gemm_out_sqrsum.copy_(_mhc_exact_sqrsum_torch(residual_cur))
    else:
        kernel = _mhc_post_fwd(residual.shape[-2], residual.shape[-1])
        kernel(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
        )

        from vllm_metax.utils.deep_gemm import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_cur.view(num_tokens, hc_mult * hidden_size),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )

    from .debug_diff import maybe_capture_mhc_pre_raw

    maybe_capture_mhc_pre_raw(
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
        n_splits=n_splits,
    )
    target_exact = _is_exact_mhc_decode_contract(
        num_tokens=num_tokens,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        n_splits=n_splits,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    require_exact = _require_exact_mhc_tilelang()

    use_torch_split_from_raw = (
        os.getenv("VLLM_METAX_DSV4_MHC_TORCH_SPLIT_FROM_RAW", "0") == "1"
    )
    if target_exact and use_torch_split_from_raw and require_exact:
        raise RuntimeError(
            "VLLM_METAX_DSV4_MHC_REQUIRE_EXACT_TILELANG=1 conflicts with "
            "VLLM_METAX_DSV4_MHC_TORCH_SPLIT_FROM_RAW=1"
        )
    if target_exact and not use_torch_split_from_raw:
        post_mix_exact, comb_mix_exact, layer_input_exact = _mhc_pre_from_raw_exact_fuse(
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
            n_splits=n_splits,
            post_mix_out=post_mix_cur,
            comb_mix_out=comb_mix_cur,
            layer_input_out=layer_input_cur,
        )
        _log_mhc_decode_impl("exact_tilelang", fail_closed=require_exact)
        return (
            residual_cur.view(*outer_shape, hc_mult, hidden_size),
            post_mix_exact.view(*outer_shape, hc_mult, 1),
            comb_mix_exact.view(*outer_shape, hc_mult, hc_mult),
            layer_input_exact.view(*outer_shape, hidden_size),
        )
    if use_torch_split_from_raw:
        if target_exact:
            _log_mhc_decode_impl("torch_oracle", fail_closed=False, explicit=True)
        post_mix_torch, comb_mix_torch, layer_input_torch = _mhc_pre_from_raw_torch(
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
        return (
            residual_cur.view(*outer_shape, hc_mult, hidden_size),
            post_mix_torch.view(*outer_shape, hc_mult, 1),
            comb_mix_torch.view(*outer_shape, hc_mult, hc_mult),
            layer_input_torch.view(*outer_shape, hidden_size),
        )

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=hc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    )
    if os.getenv("VLLM_METAX_DSV4_MHC_USE_BIG_FUSE_LAYER", "0") != "1":
        layer_input_cur.copy_(
            _mhc_apply_mix_torch_sum(
                residual_cur,
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                n_splits=n_splits,
            )
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and return the (T, H) bf16 result."""
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if num_tokens == 0:
        return out

    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )



direct_register_custom_op(
    op_name="mx_mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mx_mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)
direct_register_custom_op(
    op_name="mx_mhc_fused_post_pre",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mx_hc_head_fused_kernel",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
