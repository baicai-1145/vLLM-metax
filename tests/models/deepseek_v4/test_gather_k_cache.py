import pytest
import torch

from vllm_metax.models.deepseek_v4.ops.cache_utils import gather_k_cache


def _reference_gather(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> torch.Tensor:
    expected = out.clone()
    for batch in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[batch].item())
        gather_len = (
            seq_len if gather_lens is None else int(gather_lens[batch].item())
        )
        positions = torch.arange(
            seq_len - gather_len, seq_len, device=k_cache.device
        )
        blocks = block_table[batch, positions // block_size].long()
        rows = positions % block_size
        expected[batch, offset : offset + gather_len] = k_cache[blocks, rows]
    return expected


@pytest.mark.parametrize("head_size", [7, 512])
@pytest.mark.parametrize("use_gather_lens", [False, True])
@pytest.mark.parametrize("padded_cache_stride", [False, True])
def test_gather_k_cache_matches_block_table_reference(
    head_size: int, use_gather_lens: bool, padded_cache_stride: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-compatible MetaX device")

    device = torch.device("cuda:0")
    block_size = 4
    storage_head_size = head_size + 8 if padded_cache_stride else head_size
    k_cache_storage = torch.arange(
        4 * block_size * storage_head_size,
        device=device,
        dtype=torch.float32,
    ).reshape(4, block_size, storage_head_size).to(torch.bfloat16)
    k_cache = k_cache_storage[:, :, :head_size]
    seq_lens = torch.tensor([7, 5], device=device, dtype=torch.int32)
    gather_lens = (
        torch.tensor([5, 3], device=device, dtype=torch.int32)
        if use_gather_lens
        else None
    )
    block_table = torch.tensor(
        [[2, 0, 3], [1, 3, 0]], device=device, dtype=torch.int32
    )
    offset = 2
    out = torch.full(
        (2, 12, head_size),
        -7,
        device=device,
        dtype=torch.bfloat16,
    )
    expected = _reference_gather(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )

    gather_k_cache(
        out,
        k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=offset,
    )

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
