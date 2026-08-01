import pytest
import torch

from vllm_metax.patch.bugfix.triton_support import rejection_sampler_v2


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _reference(target_argmax, draft_sampled, cu_num_logits, steps):
    target_argmax = target_argmax.cpu()
    draft_sampled = draft_sampled.cpu()
    cu_num_logits = cu_num_logits.cpu().tolist()
    sampled = torch.full(
        (len(cu_num_logits) - 1, steps + 1), -1, dtype=torch.int64
    )
    num_sampled = torch.empty(len(cu_num_logits) - 1, dtype=torch.int32)
    for req_idx, (start, end) in enumerate(
        zip(cu_num_logits, cu_num_logits[1:])
    ):
        accepted_len = 0
        for step in range(end - start - 1):
            draft_token = draft_sampled[start + step + 1]
            target_token = target_argmax[start + step]
            sampled[req_idx, step] = target_token
            if draft_token != target_token:
                break
            sampled[req_idx, step] = draft_token
            accepted_len += 1
        else:
            sampled[req_idx, accepted_len] = target_argmax[
                start + accepted_len
            ]
        num_sampled[req_idx] = accepted_len + 1
    return sampled, num_sampled


@pytest.mark.parametrize(
    ("target_argmax", "draft_sampled", "cu_num_logits", "steps"),
    [
        ([7, 8, 9, 10, 11, 12], [99, 7, 8, 9, 10, 11], [0, 6], 5),
        ([7, 8, 19, 10, 11, 12], [99, 7, 8, 9, 10, 11], [0, 6], 5),
        (
            [7, 8, 9, 10, 20, 21, 29, 23, 24],
            [99, 7, 8, 9, 88, 20, 21, 22, 23],
            [0, 4, 9],
            5,
        ),
    ],
)
def test_greedy_accept_kernel_matches_serial_reference(
    target_argmax, draft_sampled, cu_num_logits, steps
):
    device = torch.device("cuda")
    target_argmax = torch.tensor(target_argmax, device=device, dtype=torch.int64)
    draft_sampled = torch.tensor(draft_sampled, device=device, dtype=torch.int32)
    cu_num_logits = torch.tensor(cu_num_logits, device=device, dtype=torch.int32)
    expected = _reference(target_argmax, draft_sampled, cu_num_logits, steps)
    sampled = torch.full(
        (cu_num_logits.numel() - 1, steps + 1),
        -1,
        device=device,
        dtype=torch.int64,
    )
    num_sampled = torch.empty(
        cu_num_logits.numel() - 1, device=device, dtype=torch.int32
    )

    rejection_sampler_v2._greedy_accept_kernel[(sampled.shape[0],)](
        target_argmax,
        draft_sampled,
        cu_num_logits,
        sampled,
        sampled.stride(0),
        num_sampled,
        NUM_SPECULATIVE_STEPS=steps,
        num_warps=1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(sampled.cpu(), expected[0])
    torch.testing.assert_close(num_sampled.cpu(), expected[1])


def test_greedy_accept_kernel_replays_exactly():
    device = torch.device("cuda")
    target_argmax = torch.tensor([7, 8, 19, 10, 11, 12], device=device)
    draft_sampled = torch.tensor(
        [99, 7, 8, 9, 10, 11], device=device, dtype=torch.int32
    )
    cu_num_logits = torch.tensor([0, 6], device=device, dtype=torch.int32)
    sampled = torch.full((1, 6), -1, device=device, dtype=torch.int64)
    num_sampled = torch.empty(1, device=device, dtype=torch.int32)
    expected = _reference(target_argmax, draft_sampled, cu_num_logits, 5)
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        rejection_sampler_v2._greedy_accept_kernel[(1,)](
            target_argmax,
            draft_sampled,
            cu_num_logits,
            sampled,
            sampled.stride(0),
            num_sampled,
            NUM_SPECULATIVE_STEPS=5,
            num_warps=1,
        )

    pointers = (sampled.data_ptr(), num_sampled.data_ptr())
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert pointers == (sampled.data_ptr(), num_sampled.data_ptr())
        torch.testing.assert_close(sampled.cpu(), expected[0])
        torch.testing.assert_close(num_sampled.cpu(), expected[1])


def test_gpu_greedy_rejection_wrapper_matches_serial(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_GPU_GREEDY_ACCEPT", "1")
    device = torch.device("cuda")
    logits = torch.zeros(6, 32, device=device, dtype=torch.bfloat16)
    target_ids = torch.tensor([7, 8, 19, 10, 11, 12], device=device)
    logits[torch.arange(6, device=device), target_ids] = 1
    draft_sampled = torch.tensor(
        [99, 7, 8, 9, 10, 11], device=device, dtype=torch.int32
    )
    cu_num_logits = torch.tensor([0, 6], device=device, dtype=torch.int32)
    expected = _reference(target_ids, draft_sampled, cu_num_logits, 5)

    sampled, num_sampled = rejection_sampler_v2._greedy_rejection_sample(
        logits,
        draft_sampled,
        cu_num_logits,
        cu_num_logits.new_zeros(6),
        torch.zeros(1, device=device),
        torch.zeros(1, device=device, dtype=torch.int64),
        cu_num_logits.new_zeros(6),
        5,
        False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(sampled.cpu(), expected[0])
    torch.testing.assert_close(num_sampled.cpu(), expected[1])
