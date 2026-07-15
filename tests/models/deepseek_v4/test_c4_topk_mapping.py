import pytest
import torch

from vllm_metax.models.deepseek_v4.ops.cache_utils import (
    compute_global_topk_indices_and_lens_bounded,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-compatible MetaX device",
)
def test_c4_topk_mapping_bounds_indices_by_compressed_seq_len() -> None:
    device = torch.device("cuda")
    topk_indices = torch.tensor(
        [[3, 0, 63, 64, 511, -1]], device=device, dtype=torch.int32
    )
    token_to_req_indices = torch.tensor([0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([257], device=device, dtype=torch.int32)
    block_table = torch.full((1, 88), 7, device=device, dtype=torch.int32)
    block_table[0, 0] = 1
    is_valid_token = torch.tensor([True], device=device)

    global_indices, topk_lens = compute_global_topk_indices_and_lens_bounded(
        topk_indices,
        token_to_req_indices,
        seq_lens,
        block_table,
        block_size=64,
        compress_ratio=4,
        is_valid_token=is_valid_token,
    )

    torch.testing.assert_close(
        global_indices,
        torch.tensor(
            [[67, 64, 127, -1, -1, -1]], device=device, dtype=torch.int32
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        topk_lens,
        torch.tensor([3], device=device, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-compatible MetaX device",
)
def test_c4_topk_mapping_handles_boundaries_multiple_requests_and_padding() -> None:
    device = torch.device("cuda")
    topk_indices = torch.tensor(
        [
            [62, 63, 64, -1, 0, 511],
            [63, 64, 0, -1, 1, 511],
            [63, 64, 0, -1, 1, 511],
            [0, 1, 511, -1, 63, 64],
        ],
        device=device,
        dtype=torch.int32,
    )
    token_to_req_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([255, 256, 257, 257], device=device, dtype=torch.int32)
    block_table = torch.full((4, 88), 7, device=device, dtype=torch.int32)
    block_table[:, 0] = torch.tensor([1, 2, 3, 4], device=device)
    is_valid_token = torch.tensor([True, True, True, False], device=device)

    global_indices, topk_lens = compute_global_topk_indices_and_lens_bounded(
        topk_indices,
        token_to_req_indices,
        seq_lens,
        block_table,
        block_size=64,
        compress_ratio=4,
        is_valid_token=is_valid_token,
    )

    expected_indices = torch.tensor(
        [
            [126, -1, -1, -1, 64, -1],
            [191, -1, 128, -1, 129, -1],
            [255, -1, 192, -1, 193, -1],
            [256, 257, -1, -1, 319, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(global_indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(
        topk_lens,
        torch.tensor([2, 3, 3, 0], device=device, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
