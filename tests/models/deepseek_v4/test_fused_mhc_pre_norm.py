# SPDX-License-Identifier: Apache-2.0
# Differential test for the fused mhc_pre_norm post-processing kernel vs the
# pure-PyTorch reference.
#
# The reference post-einsum block in mhc_pre_norm_fn_ref was:
#     sqrsum = residual_flat.square().sum(-1, keepdim=True)
#     denom  = (sqrsum / rms_group_size + eps).rsqrt()
#     mixes  = (mixes * denom.unsqueeze(-1)).sum(-2)   # mixes is [B, 1, N]
# The fused triton kernel in ops/mhc/fused_mhc_pre_norm.py must match it to
# <=1e-5 abs: mixes feeds MHC routing (sinkhorn + downstream), and drift would
# change greedy decode tokens.

import pytest
import torch

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

_HAS_CUDA = torch.cuda.is_available()

from vllm_metax.models.deepseek_v4.ops.mhc.fused_mhc_pre_norm import (
    fused_mhc_pre_norm_post,
)
from vllm_metax.models.deepseek_v4.ops.mhc.torch import (
    mhc_pre_norm_fn_ref,
    mhc_pre_norm_post,
)

EPS = 1e-6
TOL = 1e-5
# Production: residual entry [outer..., 4, 4096] -> flatten -> [1, B, 16384];
# fn is [24, 16384] (hc_mult3=24 mix channels, hc_mult*hidden=16384).
# B = number of residual rows (1 for M=1 decode).
PROD_RESIDUAL_SHAPES = [(1, 4, 4096), (4, 4, 4096)]  # B = 1 and B = 4
PROD_RMS_SIZES = [16384, 8192, 4096]


def reference_post(mixes, residual_flat, mhc_norm_eps):
    """The exact pre-fusion reference block (independent oracle)."""
    rms_group_size = residual_flat.shape[-1]
    sqrsum = residual_flat.square().sum(-1, keepdim=True)
    denom = (sqrsum / rms_group_size + mhc_norm_eps).rsqrt()
    return (mixes * denom.unsqueeze(-1)).sum(-2)


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
@pytest.mark.parametrize("residual_shape", PROD_RESIDUAL_SHAPES)
@pytest.mark.parametrize("rms_size", PROD_RMS_SIZES)
def test_fused_mhc_pre_norm_post_vs_ref(residual_shape, rms_size):
    """Fused kernel vs the original reference block at production shapes."""
    torch.manual_seed(0)
    outer, mult, hidden = residual_shape
    # residual_flat is residual.flatten(2, 3): [B, mult*hidden]
    b = outer
    n = 24  # hc_mult3 mix channels
    residual = torch.randn(1, b, mult * hidden, dtype=torch.float32, device="cuda")
    if rms_size != mult * hidden:
        # independently-sized variant (must remain divisible by BLOCK_R=4096)
        residual = torch.randn(1, b, rms_size, dtype=torch.float32, device="cuda")
    residual_flat = residual.view(-1, residual.shape[-1])
    mixes = torch.randn(b, 1, n, dtype=torch.float32, device="cuda")

    ref = reference_post(mixes, residual_flat, EPS)
    got = fused_mhc_pre_norm_post(mixes, residual_flat, EPS)
    assert not torch.isnan(got).any().item(), "fused output has NaN"
    assert not torch.isinf(got).any().item(), "fused output has Inf"
    max_diff = (ref - got).abs().max().item()
    assert max_diff <= TOL, (
        f"residual_shape={residual_shape} rms={rms_size}: "
        f"max abs diff {max_diff:.3e} > {TOL}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
def test_fused_mhc_pre_norm_post_production_end_to_end():
    """Full mhc_pre_norm_fn_ref (production path) vs the original formula."""
    torch.manual_seed(1)
    # production: residual entry [1, 4, 4096] bf16, fn [24, 16384] fp32
    residual = torch.randn(1, 4, 4096, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 16384, dtype=torch.float32, device="cuda")
    out = mhc_pre_norm_fn_ref(residual, fn, None, EPS)
    assert out.shape == (1, 1, 24), out.shape

    # independent oracle replicating the original function body
    r = residual.unsqueeze(0).flatten(2, 3).float()
    mixes = torch.einsum(
        "mbk,nbk->mbn",
        r.view(-1, 1, 16384),
        fn.view(24, 1, 16384),
    )
    sqrsum = r.view(-1, 1, 16384).square().sum(-1)
    ref = (
        mixes * (sqrsum.unsqueeze(-1) / 16384 + EPS).rsqrt()
    ).sum(-2).view(*r.shape[:2], -1)

    max_diff = (ref - out).abs().max().item()
    assert not torch.isnan(out).any().item(), "output has NaN"
    assert not torch.isinf(out).any().item(), "output has Inf"
    # mixes after the einsum are O(100) (16384-product contraction), so an
    # absolute 1e-5 bound is below fp32 associativity noise (torch's own
    # two-half-order sum differs by 3.05e-05 from the full sum here). Use a
    # relative bound: 1e-5 of the O(100) magnitude is the meaningful check.
    assert torch.allclose(ref, out, rtol=1e-5, atol=1e-5), (
        f"e2e mhc_pre_norm_fn_ref: max abs diff {max_diff:.3e}"
        f" (not allclose rtol=1e-5 atol=1e-5)"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
def test_fused_mhc_pre_norm_post_non_contiguous_falls_back():
    """Non-contiguous residual_flat must fall back to the reference ops."""
    torch.manual_seed(2)
    b, n, rms = 2, 24, 16384
    mixes = torch.randn(b, 1, n, dtype=torch.float32, device="cuda")
    # non-contiguous residual: view of a wider buffer
    wide = torch.randn(b, rms * 2, dtype=torch.float32, device="cuda")
    residual_flat = wide[:, ::2]  # stride 2 -> not contiguous
    assert not residual_flat.is_contiguous()
    ref = reference_post(mixes, residual_flat, EPS)
    got = mhc_pre_norm_post(mixes, residual_flat, EPS)
    max_diff = (ref - got).abs().max().item()
    assert max_diff <= TOL, (
        f"fallback path: max abs diff {max_diff:.3e} > {TOL}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
def test_fused_mhc_pre_norm_post_mismatched_rows_raises():
    """Row-count mismatch between mixes and residual_flat must not be silent."""
    b, n, rms = 2, 24, 16384
    mixes = torch.randn(b, 1, n, dtype=torch.float32, device="cuda")
    residual_flat = torch.randn(b + 1, rms, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        fused_mhc_pre_norm_post(mixes, residual_flat, EPS)
