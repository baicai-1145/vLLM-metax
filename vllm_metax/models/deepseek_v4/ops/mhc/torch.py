# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def expand_to_mhc_ref(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return (
        hidden.unsqueeze(-2)
        .expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
        .contiguous()
    )


def sinkhorn_normalize_ref(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def sinkhorn_normalize(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """Dispatch wrapper: use the fused triton kernel when the layout/dtype is
    supported, else fall back to the pure-PyTorch reference.

    The fused kernel replaces ~39 tiny 2-4us kernels per call (softmax + 39
    row/col normalizes for hc_sinkhorn_iters=20) with ~6 compiled triton
    launches. It requires a CUDA float32 tensor whose last two dims are
    (MHC_MULT, MHC_MULT)==(4, 4) contiguous with a dense leading layout.
    """
    if x.is_cuda and x.dtype == torch.float32:
        try:
            from .fused_sinkhorn import fused_sinkhorn as _fused_sinkhorn

            return _fused_sinkhorn(x, repeat, eps)
        except ValueError:
            # Unsupported layout/shape -> fall back to the reference path.
            pass
    return sinkhorn_normalize_ref(x, repeat, eps)


def mhc_head_compute_mix_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ],
    )
    # print(f">>>>>>>>>>>> mhc_pre_split_mixes_ref, input_mixes: {input_mixes.shape}, mhc_scale: {mhc_scale.shape}, mhc_base: {mhc_base.shape}")
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (
        input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value
    ).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return (x * mix).sum(-2).bfloat16()


def mhc_pre_norm_post(
    x_mixes: torch.Tensor,
    x_residual_flat: torch.Tensor,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """Dispatch wrapper: fuse the sqrsum + rsqrt-denom + mixes-scaling block
    that follows the einsum in `mhc_pre_norm_fn_ref` into one triton kernel.

    Args:
        x_mixes: [B, 1, N] fp32 einsum output.
        x_residual_flat: [B, rms_group_size] fp32 flattened residual.
        mhc_norm_eps: rms epsilon.
    Returns:
        [B, N] fp32 = (x_mixes * rsqrt(sqrsum / R + eps)).sum(-2), matching
        the reference block exactly.

    Falls back to the reference ops when the fused kernel cannot handle the
    layout (non-contiguous, non-CUDA, wrong dtype, or unsupported sizes).
    """
    if (
        x_mixes.is_cuda
        and x_residual_flat.is_cuda
        and x_mixes.dtype == torch.float32
        and x_residual_flat.dtype == torch.float32
    ):
        try:
            from .fused_mhc_pre_norm import fused_mhc_pre_norm_post as _fused

            return _fused(x_mixes, x_residual_flat, mhc_norm_eps)
        except ValueError:
            # Unsupported layout/shape -> fall back to the reference path.
            pass
    rms_group_size = x_residual_flat.shape[-1]
    sqrsum = x_residual_flat.square().sum(-1, keepdim=True)
    denom = (sqrsum / rms_group_size + mhc_norm_eps).rsqrt()
    return (x_mixes * denom.unsqueeze(-1)).sum(-2)


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.unsqueeze(0)
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        "mbk,nbk->mbn",
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    # Fused: sqrsum + rsqrt-denom + mixes-scaling in one triton kernel
    # (fallback to the reference ops below when unsupported).
    residual_flat = residual.view(-1, rms_group_size)
    mixes = mhc_pre_norm_post(mixes, residual_flat, mhc_norm_eps)
    return mixes.view(*residual.shape[:2], -1)


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    mixes = mhc_pre_norm_fn_ref(
        residual,
        fn,
        None,
        rms_eps,
        # fuse_grad_acc=False,
        # n_splits=n_splits,
    )

    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes_ref(
        mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )

    comb_mix = sinkhorn_normalize(
        comb_mix, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps
    )

    layer_input = mhc_pre_apply_mix_ref(residual, pre_mix)

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def mhc_pre(
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

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    post_mix, comb_mix, layer_input = big_fuse_reference(
        residual,
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
    return post_mix, comb_mix, layer_input


def _mhc_pre_fake(
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


##########################################################
def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.einsum("bmn,bmc->bnc", comb_res_mix, residual.float())
    # print(f">>>>>>>>>>>>>>>>..... mhc_post_ref, x: {x.shape}, comb_res_mix: {comb_res_mix.shape}, residual: {residual.shape}, post_layer_mix: {post_layer_mix.shape}")
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)

    out = mhc_post_ref(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        # out
    )
    return out


def mhc_fused_post_pre(
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
    residual_cur = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input = mhc_pre(
        residual_cur,
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
    return residual_cur, post_mix, comb_mix, layer_input


def hc_head_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    residual_flat = residual.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    return hc_head_ref(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps)


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def _mhc_fused_post_pre_fake(
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
    post_mix = torch.empty(*outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(*outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(*outer_shape, hidden_size, dtype=torch.bfloat16, device=residual.device)
    return residual_cur, post_mix, comb_mix, layer_input


def _hc_head_fused_kernel_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device)


direct_register_custom_op(
    op_name="mx_mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mx_mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
