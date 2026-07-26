from types import SimpleNamespace

import pytest
import torch

from vllm_metax.models.deepseek_v4.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)


def _run_first_sparse_overlap_case(
    stale_overlap: bool, *, initial_overlap_boundary: int | None
) -> torch.Tensor:
    device = torch.device("cuda")
    head_dim = 512
    state_width = 2 * head_dim
    state_cache = torch.zeros(
        (1, 256, 2 * state_width), device=device, dtype=torch.float32
    )
    if stale_overlap:
        stale = torch.arange(4 * head_dim, device=device, dtype=torch.float32).reshape(
            4, head_dim
        )
        state_cache[0, 124:128, :head_dim] = torch.sin(stale * 0.013)
        state_cache[0, 124:128, state_width : state_width + head_dim] = (
            torch.cos(stale * 0.017)
        )
    current = torch.arange(
        4 * head_dim, device=device, dtype=torch.float32
    ).reshape(4, head_dim)
    state_cache[0, 128:132, head_dim:state_width] = torch.sin(current * 0.019)
    state_cache[
        0,
        128:132,
        state_width + head_dim : 2 * state_width,
    ] = torch.cos(current * 0.023)

    cos_sin_cache = torch.zeros((132, 64), device=device, dtype=torch.float32)
    cos_sin_cache[:, :32] = 1.0
    kv_cache = torch.zeros((1, 64, head_dim), device=device, dtype=torch.bfloat16)
    compress_norm_rope_store_triton(
        state_cache=state_cache,
        num_actual=1,
        token_to_req_indices=torch.zeros(1, device=device, dtype=torch.int32),
        positions=torch.tensor([131], device=device, dtype=torch.int32),
        slot_mapping=torch.tensor([131], device=device, dtype=torch.int32),
        block_table=torch.zeros((1, 1), device=device, dtype=torch.int32),
        block_size=256,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(
            slot_mapping=torch.zeros(1, device=device, dtype=torch.int32)
        ),
        pdl_kwargs={},
        head_dim=head_dim,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        initial_overlap_boundary=initial_overlap_boundary,
        use_fp4_cache=False,
        rms_norm_weight=torch.ones(head_dim, device=device),
        rms_norm_eps=1e-6,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
    )
    torch.cuda.synchronize()
    return kv_cache.cpu()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Compressor overlap test requires a MetaX CUDA-compatible device",
)
def test_first_sparse_compression_ignores_stale_initial_overlap() -> None:
    expected = _run_first_sparse_overlap_case(
        stale_overlap=False, initial_overlap_boundary=None
    )
    actual = _run_first_sparse_overlap_case(
        stale_overlap=True, initial_overlap_boundary=128
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="BF16 compressor cache test requires a MetaX CUDA-compatible device",
)
@pytest.mark.parametrize("kv_offset", [0, 1, 63])
def test_bf16_compressor_uses_cache_token_stride(kv_offset: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 512
    cache_block_size = 64

    # Use one state row with a finite, non-uniform value and score.  The
    # ratio=1/no-overlap case makes the independent oracle unambiguous.
    state_cache = torch.empty((1, 256, 2 * head_dim), device=device)
    kv_state = torch.linspace(
        -1.0, 1.0, head_dim, device=device, dtype=torch.float32
    )
    score_state = torch.zeros(head_dim, device=device, dtype=torch.float32)
    state_cache.zero_()
    state_cache[0, 0, :head_dim] = kv_state
    state_cache[0, 0, head_dim:] = score_state

    rms_weight = torch.linspace(
        0.5, 1.5, head_dim, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.empty((1, 64), device=device, dtype=torch.float32)
    cos_sin_cache[:, :32] = 0.8
    cos_sin_cache[:, 32:] = 0.6

    # Keep the target block and adjacent rows distinct so an incorrect packed
    # stride is observable as both a partial write and an overrun.
    sentinel = torch.tensor(17.0, device=device, dtype=dtype)
    kv_cache = torch.full(
        (2, cache_block_size, head_dim), sentinel, device=device, dtype=dtype
    )
    token_to_req_indices = torch.zeros(1, device=device, dtype=torch.int32)
    positions = torch.zeros(1, device=device, dtype=torch.int32)
    slot_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    block_table = torch.zeros((1, 256), device=device, dtype=torch.int32)
    kv_slot_mapping = torch.tensor(
        [cache_block_size + kv_offset], device=device, dtype=torch.int32
    )

    compress_norm_rope_store_triton(
        state_cache=state_cache,
        num_actual=1,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=256,
        state_width=head_dim,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=kv_slot_mapping),
        pdl_kwargs={},
        head_dim=head_dim,
        rope_head_dim=64,
        compress_ratio=1,
        overlap=False,
        use_fp4_cache=False,
        rms_norm_weight=rms_weight,
        rms_norm_eps=1e-6,
        quant_block=64,
        # This is the upstream packed-layout stride.  The launcher must use
        # kv_cache.stride(1) for the plain BF16 cache instead.
        token_stride=576,
        scale_dim=8,
    )
    torch.cuda.synchronize()

    variance = torch.mean(kv_state.square())
    expected = kv_state * torch.rsqrt(variance + 1e-6) * rms_weight
    # Independent FP32 GPT-J RoPE oracle for the final 64 dimensions.
    rope = expected[448:].reshape(32, 2)
    cos = cos_sin_cache[0, :32]
    sin = cos_sin_cache[0, 32:]
    expected = expected.clone()
    expected[448:] = torch.stack(
        (rope[:, 0] * cos - rope[:, 1] * sin,
         rope[:, 1] * cos + rope[:, 0] * sin),
        dim=1,
    ).reshape(-1)
    target = kv_cache[1, kv_offset].float()
    torch.testing.assert_close(target, expected, atol=0.02, rtol=0.02)

    # No neighboring row or sentinel may be touched by the token write.
    if kv_offset > 0:
        torch.testing.assert_close(
                kv_cache[1, kv_offset - 1],
            torch.full((head_dim,), sentinel, device=device, dtype=dtype),
            atol=0,
            rtol=0,
        )
    if kv_offset + 1 < cache_block_size:
        torch.testing.assert_close(
                kv_cache[1, kv_offset + 1],
            torch.full((head_dim,), sentinel, device=device, dtype=dtype),
            atol=0,
            rtol=0,
        )
    if kv_offset == cache_block_size - 1:
        torch.testing.assert_close(
            kv_cache[1],
            torch.cat(
                [
                    torch.full(
                        (cache_block_size - 1, head_dim),
                        sentinel,
                        device=device,
                        dtype=dtype,
                    ),
                    expected.to(dtype).unsqueeze(0),
                ]
            ),
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        kv_cache[0],
        torch.full_like(kv_cache[0], sentinel),
        atol=0,
        rtol=0,
    )
