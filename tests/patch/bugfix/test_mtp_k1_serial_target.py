import json
from types import SimpleNamespace

import pytest
import torch

from vllm_metax.patch.bugfix import mtp_k1_serial_target as patch


def test_serial_pair_capture_is_inert_without_capture_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", raising=False)
    scheduler = SimpleNamespace(requests={})

    patch._capture_pair(
        scheduler,
        request_id="req-0",
        draft_token_id=1,
        target_token_id=2,
        accepted=False,
        scheduled_token_count=1,
        req_row_index=0,
    )

    assert not hasattr(scheduler, patch._PAIR_CAPTURE_COUNTER_ATTR)
    assert list(tmp_path.iterdir()) == []


def test_serial_pair_capture_is_bounded_and_includes_request_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_PAIR_CAPTURE_STEPS", "2")
    request = SimpleNamespace(
        request_id="req-0", num_tokens=17, num_computed_tokens=11
    )
    scheduler = SimpleNamespace(requests={"req-0": request})

    for draft, target in ((10, 10), (20, 21), (30, 30)):
        patch._capture_pair(
            scheduler,
            request_id="req-0",
            draft_token_id=draft,
            target_token_id=target,
            accepted=draft == target,
            scheduled_token_count=3,
            req_row_index=4,
        )

    path = tmp_path / f"serial_pairs.rank{patch._rank_label(patch._capture_rank())}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert [record["comparison_index"] for record in records] == [0, 1]
    assert records[0] == {
        "accepted": True,
        "comparison_index": 0,
        "draft_token_id": 10,
        "draft_available": True,
        "num_computed_tokens": 11,
        "num_tokens": 17,
        "rank": records[0]["rank"],
        "req_row_index": 4,
        "request_id": "req-0",
        "scheduled_token_count": 3,
        "target_token_id": 10,
    }


def test_serial_pair_capture_write_failure_is_ignored(monkeypatch, tmp_path):
    capture_path = tmp_path / "not-a-directory"
    capture_path.write_text("occupied")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(capture_path))
    scheduler = SimpleNamespace(requests={})

    patch._capture_pair(
        scheduler,
        request_id="req-0",
        draft_token_id=1,
        target_token_id=1,
        accepted=True,
        scheduled_token_count=1,
        req_row_index=0,
    )

    assert getattr(scheduler, patch._PAIR_CAPTURE_COUNTER_ATTR) == 1


def test_serial_pair_capture_marks_negative_draft_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR", str(tmp_path))
    scheduler = SimpleNamespace(requests={})

    patch._capture_pair(
        scheduler,
        request_id="req-0",
        draft_token_id=-1,
        target_token_id=123,
        accepted=False,
        scheduled_token_count=1,
        req_row_index=0,
    )

    path = tmp_path / f"serial_pairs.rank{patch._rank_label(patch._capture_rank())}.jsonl"
    record = json.loads(path.read_text())
    assert record["accepted"] is False
    assert record["draft_available"] is False


def test_serial_target_removes_draft_slots_and_retains_draft(monkeypatch):
    seen = {}

    def original(scheduler, throttle_prefills=False):
        seen["drafts"] = {
            request.request_id: list(request.spec_token_ids)
            for request in scheduler.running
        }
        return "scheduled"

    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(patch, "_ORIGINAL_SCHEDULE", original)
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[123])
    scheduler = SimpleNamespace(running=[request])

    result = patch._schedule(scheduler)

    assert result == "scheduled"
    assert seen == {"drafts": {"req-0": []}}
    assert request.spec_token_ids == []
    assert scheduler._metax_k1_serial_pending_outputs[0][1] == {"req-0": [123]}


def test_serial_target_clears_pending_drafts_on_no_draft_cycle(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    outputs = iter((SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(patch, "_ORIGINAL_SCHEDULE", lambda *_: next(outputs))
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[123])
    scheduler = SimpleNamespace(running=[request])

    patch._schedule(scheduler)
    request.spec_token_ids = []
    patch._schedule(scheduler)

    assert scheduler._metax_k1_serial_pending_outputs[0][1] == {"req-0": [123]}
    assert scheduler._metax_k1_serial_pending_outputs[1][1] == {}


def test_serial_target_bounds_unmatched_pending_outputs(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_SCHEDULE",
        lambda *_: SimpleNamespace(num_scheduled_tokens={"req-0": 1}),
    )
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[])
    scheduler = SimpleNamespace(running=[request])
    total_cycles = patch._PENDING_OUTPUTS_MAX + 5

    last_output = None
    for token_id in range(total_cycles):
        request.spec_token_ids = [token_id]
        last_output = patch._schedule(scheduler)

    assert len(scheduler._metax_k1_serial_pending_outputs) == patch._PENDING_OUTPUTS_MAX
    assert scheduler._metax_k1_serial_dropped_pending_outputs == 5
    assert scheduler._metax_k1_serial_unavailable == 5
    assert scheduler._metax_k1_serial_pending_outputs[0][1] == {
        "req-0": [5]
    }
    assert patch._update_from_output(
        scheduler,
        last_output,
        SimpleNamespace(req_id_to_index={"req-0": 0}, sampled_token_ids=[[132]]),
    ) == "updated"
    assert scheduler._metax_k1_serial_accepted == 1
    assert scheduler._metax_k1_serial_rejected == 0
    assert "summary request=req-0 accepted=1 rejected=0 unavailable=5" in caplog.text


def test_serial_target_pairs_each_output_with_its_own_draft(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    outputs = [
        SimpleNamespace(num_scheduled_tokens={"req-0": 1}),
        SimpleNamespace(num_scheduled_tokens={"req-0": 1}),
    ]
    monkeypatch.setattr(patch, "_ORIGINAL_SCHEDULE", lambda *_: outputs.pop(0))
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[123])
    scheduler = SimpleNamespace(running=[request], requests={})

    first_output = patch._schedule(scheduler)
    request.spec_token_ids = [456]
    second_output = patch._schedule(scheduler)

    assert patch._update_from_output(
        scheduler,
        first_output,
        SimpleNamespace(req_id_to_index={"req-0": 0}, sampled_token_ids=[[123]]),
    ) == "updated"
    assert patch._update_from_output(
        scheduler,
        second_output,
        SimpleNamespace(req_id_to_index={"req-0": 0}, sampled_token_ids=[[999]]),
    ) == "updated"
    assert scheduler._metax_k1_serial_accepted == 1
    assert scheduler._metax_k1_serial_rejected == 1
    assert scheduler._metax_k1_serial_pending_outputs == []


def test_serial_target_is_inert_without_opt_in(monkeypatch):
    expected = object()
    seen = {}

    def original(scheduler, throttle_prefills=False):
        seen["drafts"] = list(scheduler.running[0].spec_token_ids)
        return expected

    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", raising=False)
    monkeypatch.setattr(patch, "_ORIGINAL_SCHEDULE", original)
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[123])
    scheduler = SimpleNamespace(running=[request])

    result = patch._schedule(scheduler)

    assert result is expected
    assert seen["drafts"] == [123]
    assert request.spec_token_ids == [123]


def test_serial_target_fails_closed_for_non_k1_drafts(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    request = SimpleNamespace(request_id="req-0", spec_token_ids=[123, 456])
    scheduler = SimpleNamespace(running=[request])

    try:
        patch._schedule(scheduler)
    except RuntimeError as exc:
        assert "exactly one draft token" in str(exc)
    else:
        raise AssertionError("k>1 must fail closed in k=1 serial target mode")


def test_serial_target_compares_retained_draft_with_target(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"req-0": [123]})],
        requests={},
    )
    model_output = SimpleNamespace(
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[[123]],
    )

    result = patch._update_from_output(scheduler, scheduler_output, model_output)

    assert result == "updated"
    assert scheduler._metax_k1_serial_pending_outputs == []
    assert scheduler._metax_k1_serial_accepted == 1
    assert scheduler._metax_k1_serial_rejected == 0
    assert "summary request=req-0 accepted=1 rejected=0 unavailable=0" in caplog.text


def test_serial_target_marks_negative_draft_unavailable(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"req-0": [-1]})],
        requests={},
    )
    model_output = SimpleNamespace(
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[[123]],
    )

    assert patch._update_from_output(scheduler, scheduler_output, model_output) == (
        "updated"
    )
    assert scheduler._metax_k1_serial_accepted == 0
    assert scheduler._metax_k1_serial_rejected == 0
    assert scheduler._metax_k1_serial_unavailable == 1
    assert "summary request=req-0 accepted=0 rejected=0 unavailable=1" in caplog.text


def test_serial_target_retires_missing_request_output(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"preempted": [123]})],
    )
    model_output = SimpleNamespace(req_id_to_index={}, sampled_token_ids=[])

    assert (
        patch._update_from_output(scheduler, scheduler_output, model_output)
        == "updated"
    )
    assert scheduler._metax_k1_serial_pending_outputs == []


def test_serial_target_retires_pending_when_sampled_tokens_unavailable(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"req-0": [123]})],
    )
    model_output = SimpleNamespace(req_id_to_index={"req-0": 0}, sampled_token_ids=None)

    assert (
        patch._update_from_output(scheduler, scheduler_output, model_output)
        == "updated"
    )
    assert scheduler._metax_k1_serial_pending_outputs == []


def test_serial_target_retires_invalid_sampled_row(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"req-0": [123]})],
    )
    model_output = SimpleNamespace(req_id_to_index={"req-0": 3}, sampled_token_ids=[])

    assert (
        patch._update_from_output(scheduler, scheduler_output, model_output)
        == "updated"
    )
    assert scheduler._metax_k1_serial_pending_outputs == []


def _run_tensor_sampled_row(monkeypatch, row):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_UPDATE_FROM_OUTPUT",
        lambda scheduler, scheduler_output, model_output: "updated",
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    scheduler = SimpleNamespace(
        _metax_k1_serial_pending_outputs=[(scheduler_output, {"req-0": [123]})],
    )
    model_output = SimpleNamespace(
        req_id_to_index={"req-0": 0}, sampled_token_ids=[row]
    )
    result = patch._update_from_output(scheduler, scheduler_output, model_output)
    return result, scheduler


def test_serial_target_accepts_one_token_tensor_row(monkeypatch):
    result, scheduler = _run_tensor_sampled_row(
        monkeypatch, torch.tensor([123], dtype=torch.int32)
    )

    assert result == "updated"
    assert scheduler._metax_k1_serial_pending_outputs == []
    assert scheduler._metax_k1_serial_accepted == 1


def test_serial_target_retires_empty_tensor_row(monkeypatch):
    result, scheduler = _run_tensor_sampled_row(
        monkeypatch, torch.tensor([], dtype=torch.int32)
    )

    assert result == "updated"
    assert scheduler._metax_k1_serial_pending_outputs == []
    assert scheduler._metax_k1_serial_accepted == 0
    assert scheduler._metax_k1_serial_rejected == 0


def test_serial_target_retires_multi_token_tensor_row(monkeypatch):
    result, scheduler = _run_tensor_sampled_row(
        monkeypatch, torch.tensor([123, 456], dtype=torch.int32)
    )

    assert result == "updated"
    assert scheduler._metax_k1_serial_pending_outputs == []
    assert scheduler._metax_k1_serial_accepted == 0
    assert scheduler._metax_k1_serial_rejected == 0


def _coordinator():
    return SimpleNamespace(
        attention_groups=[
            SimpleNamespace(use_eagle=True),
            SimpleNamespace(use_eagle=False),
            SimpleNamespace(use_eagle=True),
        ]
    )


def test_serial_target_cache_hit_disables_eagle_temporarily(monkeypatch):
    seen = []
    expected = object()

    def original(coordinator, block_hashes, max_cache_hit_length):
        seen.append([group.use_eagle for group in coordinator.attention_groups])
        return expected

    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(patch, "_ORIGINAL_FIND_LONGEST_CACHE_HIT", original)
    coordinator = _coordinator()

    result = patch._find_longest_cache_hit(coordinator, [], 512)

    assert result is expected
    assert seen == [[False, False, False]]
    assert [group.use_eagle for group in coordinator.attention_groups] == [
        True,
        False,
        True,
    ]


def test_serial_target_cache_hit_is_inert_without_opt_in(monkeypatch):
    seen = []
    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", raising=False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FIND_LONGEST_CACHE_HIT",
        lambda coordinator, block_hashes, max_cache_hit_length: seen.append(
            [group.use_eagle for group in coordinator.attention_groups]
        ),
    )
    coordinator = _coordinator()

    patch._find_longest_cache_hit(coordinator, [], 512)

    assert seen == [[True, False, True]]


def test_serial_target_cache_hit_restores_eagle_on_exception(monkeypatch):
    def original(*args, **kwargs):
        raise RuntimeError("cache lookup failed")

    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET", "1")
    monkeypatch.setattr(patch, "_ORIGINAL_FIND_LONGEST_CACHE_HIT", original)
    coordinator = _coordinator()

    with pytest.raises(RuntimeError, match="cache lookup failed"):
        patch._find_longest_cache_hit(coordinator, [], 512)

    assert [group.use_eagle for group in coordinator.attention_groups] == [
        True,
        False,
        True,
    ]
