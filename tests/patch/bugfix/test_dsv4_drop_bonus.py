import json
from types import SimpleNamespace

import torch

import vllm_metax.patch.bugfix.dspark_greedy_punctuation_tie as patch


def test_deprecated_drop_bonus_flag_preserves_full_acceptance_contract(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_MTP_DROP_UNVERIFIED_BONUS", "1")

    for num_draft_tokens in (3, 4):
        sampled = torch.arange(
            11,
            11 + num_draft_tokens + 1,
            dtype=torch.int32,
        ).unsqueeze(0)
        monkeypatch.setattr(
            patch,
            "_ORIGINAL_V1_REJECTION_SAMPLE",
            lambda *args, sampled=sampled, **kwargs: sampled.clone(),
        )

        result = patch._v1_rejection_sample(
            draft_token_ids=sampled[0, :-1],
            num_draft_tokens=[num_draft_tokens],
            max_spec_len=num_draft_tokens,
            cu_num_draft_tokens=torch.tensor(
                [0, num_draft_tokens], dtype=torch.int32
            ),
            draft_probs=None,
            target_logits=torch.empty(num_draft_tokens, 32),
            bonus_token_ids=sampled[:, -1],
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )

        generated_token_ids = result[0][result[0].ge(0)]
        scheduler_accepted = max(len(generated_token_ids) - 1, 0)
        assert scheduler_accepted == num_draft_tokens
        torch.testing.assert_close(result, sampled)


def test_force_reject_drafts_uses_target_argmax_corrections(monkeypatch):
    sampled = torch.tensor(
        [
            [11, 12, 13],
            [21, -1, -1],
        ],
        dtype=torch.int32,
    )
    logits = torch.zeros(3, 32)
    logits[0, 14] = 1
    logits[1, 12] = 2
    logits[2, 22] = 3
    monkeypatch.setenv("VLLM_METAX_MTP_FORCE_REJECT_DRAFTS", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_V1_REJECTION_SAMPLE",
        lambda *args, **kwargs: sampled.clone(),
    )

    result = patch._v1_rejection_sample(
        draft_token_ids=torch.tensor([11, 12, 21], dtype=torch.int32),
        num_draft_tokens=[2, 1],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([0, 2, 3], dtype=torch.int32),
        draft_probs=None,
        target_logits=logits,
        bonus_token_ids=torch.tensor([13, 22], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [14, -1, -1],
                [22, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_force_reject_drafts_preserves_non_greedy_sampling(monkeypatch):
    logits = torch.zeros(2, 32)
    logits[0, 14] = 1
    logits[1, 22] = 1
    monkeypatch.setenv("VLLM_METAX_MTP_FORCE_REJECT_DRAFTS", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_V1_REJECTION_SAMPLE",
        lambda *args, **kwargs: torch.tensor([[11, 12, 13]], dtype=torch.int32),
    )

    result = patch._v1_rejection_sample(
        draft_token_ids=torch.tensor([11, 12], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([0, 2], dtype=torch.int32),
        draft_probs=None,
        target_logits=logits,
        bonus_token_ids=torch.tensor([13], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(all_greedy=False),
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[11, 12, 13]], dtype=torch.int32),
    )


def test_force_reject_drafts_noops_without_target_rows():
    result = torch.tensor([[11, 12]], dtype=torch.int32)
    forced = patch._force_reject_v1_greedy_result(
        result,
        target_logits=torch.empty(0, 32),
        cu_num_draft_tokens=torch.tensor([0, 1], dtype=torch.int32),
    )
    assert forced is result
def test_force_reject_drafts_accepts_per_request_counts():
    result = torch.tensor(
        [
            [11, 12, -1],
            [21, -1, -1],
        ],
        dtype=torch.int32,
    )
    logits = torch.zeros(3, 32)
    logits[0, 14] = 1
    logits[2, 22] = 1
    forced = patch._force_reject_v1_greedy_result(
        result,
        target_logits=logits,
        cu_num_draft_tokens=torch.tensor([2, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        forced,
        torch.tensor(
            [
                [14, -1, -1],
                [22, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_force_reject_v2_uses_each_requests_first_target_row():
    sampled = torch.tensor(
        [[11, 12, 13], [21, 22, -1]], dtype=torch.int32
    )
    num_sampled = torch.tensor([3, 2], dtype=torch.int32)
    logits = torch.zeros(7, 32)
    logits[0, 14] = 1
    logits[4, 24] = 1

    forced, forced_counts = patch._force_reject_v2_greedy_result(
        (sampled, num_sampled),
        target_logits=logits,
        cu_num_logits=torch.tensor([0, 4, 7], dtype=torch.int32),
    )

    torch.testing.assert_close(
        forced,
        torch.tensor([[14, -1, -1], [24, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(forced_counts, torch.ones_like(num_sampled))


def test_v2_force_reject_is_opt_in(monkeypatch):
    expected = (
        torch.tensor([[11, 12]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    logits = torch.zeros(2, 32)
    logits[0, 14] = 1
    monkeypatch.setattr(
        patch, "_ORIGINAL_V2_REJECTION_SAMPLE", lambda *args, **kwargs: expected
    )

    args = (
        logits,
        None,
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int64),
        1,
    )
    actual = patch._v2_rejection_sample(*args)
    assert actual is expected

    monkeypatch.setenv("VLLM_METAX_MTP_FORCE_REJECT_DRAFTS", "1")
    forced, counts = patch._v2_rejection_sample(*args)
    assert forced.tolist() == [[14, -1]]
    assert counts.tolist() == [1]

    non_greedy_args = (*args[:8], torch.ones(1), *args[9:])
    assert patch._v2_rejection_sample(*non_greedy_args) is expected


def test_v1_sampler_forward_capture_records_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    expected = object()
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_V1_REJECTION_SAMPLER_FORWARD",
        lambda *args, **kwargs: expected,
    )
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([11], dtype=torch.int32),
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2], dtype=torch.int32),
        target_logits_indices=torch.tensor([0], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([1], dtype=torch.int64),
        logits_indices=torch.tensor([30, 31], dtype=torch.int64),
    )
    logits = torch.eye(40)[[12, 22]]

    result = patch._v1_rejection_sampler_forward(
        object(),
        metadata,
        None,
        logits,
        SimpleNamespace(all_greedy=True),
    )

    assert result is expected
    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert records[0]["stage"] == "v1_sampler_metadata"
    assert records[0]["target_logits_indices"] == [0]
    assert records[0]["bonus_logits_indices"] == [1]
    assert records[0]["target_model_row_indices"] == [30]


def test_gpu_model_runner_sample_capture_records_positions(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    expected = object()
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_GPU_MODEL_RUNNER_SAMPLE",
        lambda *args, **kwargs: expected,
    )
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([11], dtype=torch.int32),
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2], dtype=torch.int32),
        target_logits_indices=torch.tensor([0], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([1], dtype=torch.int64),
        logits_indices=torch.tensor([30, 31], dtype=torch.int64),
    )
    runner = SimpleNamespace(
        input_ids=SimpleNamespace(gpu=torch.arange(64, dtype=torch.int32) + 1000),
        positions=torch.arange(64, dtype=torch.int64) + 600,
    )
    logits = torch.eye(40)[[12, 22]]

    result = patch._gpu_model_runner_sample(runner, logits, metadata)

    assert result is expected
    records = [
        json.loads(line)
        for line in (tmp_path / "rank0.jsonl").read_text().splitlines()
    ]
    assert records[0]["stage"] == "v1_model_runner_metadata"
    assert records[0]["target_input_ids"] == [1030]
    assert records[0]["target_positions"] == [630]
