# SPDX-License-Identifier: Apache-2.0
# Differential test for the fused Sinkhorn kernel vs the PyTorch reference.
#
# sinkhorn_normalize_ref (ops/mhc/torch.py) is the pure-PyTorch oracle. The
# fused triton kernel in ops/mhc/fused_sinkhorn.py must match it to <=1e-5 abs
# because sinkhorn feeds MHC routing: numerical drift changes routing, which
# changes greedy decode tokens.

import pytest
import torch

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

_HAS_CUDA = torch.cuda.is_available()

from vllm_metax.models.deepseek_v4.ops.mhc.fused_sinkhorn import fused_sinkhorn
from vllm_metax.models.deepseek_v4.ops.mhc.torch import (
    sinkhorn_normalize,
    sinkhorn_normalize_ref,
)

# sinkhorn_normalize_ref is kept as the differential oracle.
SHAPES = [(1, 4, 4), (4, 4, 4), (64, 4, 4), (256, 4, 4)]
REPEATS = [1, 5, 20]  # 1 = loop body never runs; 20 = production hc_sinkhorn_iters
EPS = 1e-6
TOL = 1e-5
# Production comb_mix batch sizes (tokens). Must be tested because the real
# comb_mix is a STRIDED VIEW (block stride 24, not 16) sliced from a larger
# tensor via mhc_pre_split_mixes_ref; a previous kernel bug only surfaced at
# T>=2 on that layout (OOB read on contiguous ping-pong buffers).
PROD_TOKEN_COUNTS = [1, 2, 4, 8, 16, 64]


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("repeat", REPEATS)
def test_fused_sinkhorn_matches_ref_fp32(shape, repeat):
    """Fused triton kernel must match the PyTorch reference within TOL on fp32."""
    torch.manual_seed(hash((shape, repeat)) & 0xFFFF)
    x = torch.randn(*shape, dtype=torch.float32, device="cuda")
    ref = sinkhorn_normalize_ref(x.clone(), repeat=repeat, eps=EPS)
    got = fused_sinkhorn(x.clone(), repeat=repeat, eps=EPS)
    assert not torch.isnan(got).any().item(), "fused output has NaN"
    assert not torch.isinf(got).any().item(), "fused output has Inf"
    max_diff = (ref - got).abs().max().item()
    assert max_diff <= TOL, (
        f"shape={shape} repeat={repeat}: max abs diff {max_diff:.3e} > {TOL}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("repeat", REPEATS)
def test_dispatch_wrapper_uses_fused_on_supported_layout(shape, repeat):
    """The dispatch wrapper `sinkhorn_normalize` must match the fused kernel
    (i.e. actually dispatch to it) on the supported fp32 (...,4,4) layout."""
    torch.manual_seed(hash((shape, repeat, "wrap")) & 0xFFFF)
    x = torch.randn(*shape, dtype=torch.float32, device="cuda")
    via_wrapper = sinkhorn_normalize(x.clone(), repeat=repeat, eps=EPS)
    via_fused = fused_sinkhorn(x.clone(), repeat=repeat, eps=EPS)
    max_diff = (via_wrapper - via_fused).abs().max().item()
    assert max_diff <= TOL, (
        f"shape={shape} repeat={repeat}: wrapper vs fused diff {max_diff:.3e}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_dispatch_wrapper_falls_back_on_unsupported_layout():
    """The wrapper must fall back to the reference for an unsupported shape
    (e.g. wrong last-two dims) and still produce the correct result."""
    torch.manual_seed(7)
    # (8, 3, 3) is NOT the supported (..., 4, 4) layout -> fallback path.
    x = torch.randn(8, 3, 3, dtype=torch.float32, device="cuda")
    via_wrapper = sinkhorn_normalize(x.clone(), repeat=5, eps=EPS)
    ref = sinkhorn_normalize_ref(x.clone(), repeat=5, eps=EPS)
    max_diff = (via_wrapper - ref).abs().max().item()
    assert max_diff <= TOL, f"fallback path diff {max_diff:.3e} > {TOL}"


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
def test_fused_sinkhorn_repeat_zero_and_one():
    """Edge cases: repeat=0 and repeat=1 mean the iteration loop never runs;
    only the softmax+eps+col-normalize prologue executes."""
    torch.manual_seed(11)
    x = torch.randn(16, 4, 4, dtype=torch.float32, device="cuda")
    for repeat in (0, 1):
        ref = sinkhorn_normalize_ref(x.clone(), repeat=repeat, eps=EPS)
        got = fused_sinkhorn(x.clone(), repeat=repeat, eps=EPS)
        max_diff = (ref - got).abs().max().item()
        assert max_diff <= TOL, f"repeat={repeat}: diff {max_diff:.3e} > {TOL}"


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="requires triton")
@pytest.mark.parametrize("num_tokens", PROD_TOKEN_COUNTS)
def test_fused_sinkhorn_production_strided_layout(num_tokens):
    """Regression: production comb_mix is a STRIDED VIEW (block stride 24, not
    16) built by mhc_pre_split_mixes_ref. The fused kernel must handle this
    layout correctly: the prologue reads the strided input, and the iter
    kernel operates on the contiguous densified ping-pong buffers (stride 16).
    A previous bug used the input stride (24) on the contiguous buffers and
    went OOB at num_tokens>=2."""
    from vllm_metax.models.deepseek_v4.ops.mhc.torch import (
        mhc_pre_norm_fn_ref,
        mhc_pre_split_mixes_ref,
    )

    torch.manual_seed(0)
    hidden = 4096
    residual = torch.randn(
        num_tokens, 4, hidden, dtype=torch.bfloat16, device="cuda"
    )
    fn = torch.randn(24, 4 * hidden, dtype=torch.float32, device="cuda")
    scale = torch.randn(3, dtype=torch.float32, device="cuda")
    base = torch.randn(24, dtype=torch.float32, device="cuda")
    mixes = mhc_pre_norm_fn_ref(residual, fn, None, 1e-6)
    _, _, comb_mix = mhc_pre_split_mixes_ref(mixes, scale, base, 4, 1.0, 1e-6)
    # Sanity: confirm this really is the strided production layout.
    assert comb_mix.shape[-2:] == (4, 4)
    assert comb_mix.stride(-2) == 4 and comb_mix.stride(-1) == 1
    if num_tokens >= 2:
        assert comb_mix.stride(-3) == 24, (
            f"expected strided view block stride 24, got {comb_mix.stride(-3)}"
        )

    ref = sinkhorn_normalize_ref(comb_mix.clone(), repeat=20, eps=EPS)
    got = fused_sinkhorn(comb_mix.clone(), repeat=20, eps=EPS)
    assert not torch.isnan(got).any().item(), "fused output has NaN"
    assert not torch.isinf(got).any().item(), "fused output has Inf"
    max_diff = (ref - got).abs().max().item()
    assert max_diff <= TOL, (
        f"num_tokens={num_tokens} (strided layout): max abs diff "
        f"{max_diff:.3e} > {TOL}"
    )
