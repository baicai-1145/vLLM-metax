from types import SimpleNamespace
import json
import os

from vllm_metax.patch.bugfix import spec_acceptance_capture as patch


def _scheduler():
    return SimpleNamespace(log_stats=True, num_spec_tokens=3)


def test_capture_is_inert_without_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    sentinel = object()
    monkeypatch.setattr(patch, "_ORIGINAL_MAKE_SPEC_DECODING_STATS", lambda *args, **kwargs: sentinel)

    scheduler = _scheduler()
    result = patch._make_spec_decoding_stats(
        scheduler,
        None,
        2,
        1,
        {"req-0": 1},
        "req-0",
    )

    assert result is sentinel
    assert not hasattr(scheduler, patch._SEQUENCE_ATTR)
    assert list(tmp_path.iterdir()) == []


def test_capture_records_order_counts_and_request_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    monkeypatch.setenv("RANK", "2")
    sentinel = object()
    monkeypatch.setattr(
        patch, "_ORIGINAL_MAKE_SPEC_DECODING_STATS", lambda *args, **kwargs: sentinel
    )
    scheduler = _scheduler()
    scheduler.requests = {
        "req-a": SimpleNamespace(
            num_prompt_tokens=7,
            num_tokens=10,
            num_computed_tokens=8,
            num_output_placeholders=2,
            max_tokens=100,
            status="RUNNING",
        ),
        "req-b": SimpleNamespace(num_prompt_tokens=3),
    }

    assert (
        patch._make_spec_decoding_stats(scheduler, None, 4, 2, {"req-a": 1}, "req-a")
        is sentinel
    )
    assert (
        patch._make_spec_decoding_stats(scheduler, None, 2, 0, {}, "req-b")
        is sentinel
    )

    paths = list(tmp_path.glob("spec_acceptance.pid*.rank2.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert [record["sequence"] for record in records] == [0, 1]
    assert records[0]["request_id"] == "req-a"
    assert records[0]["raw_draft_tokens"] == 4
    assert records[0]["effective_draft_tokens"] == 3
    assert records[0]["accepted_tokens"] == 2
    assert records[0]["invalid_tokens"] == 1
    assert records[0]["num_spec_tokens"] == 3
    assert records[0]["pid"] == os.getpid()
    assert records[0]["rank"] == 2
    assert records[0]["request_metadata"]["num_prompt_tokens"] == 7
    assert records[1]["request_id"] == "req-b"
    assert records[1]["effective_draft_tokens"] == 2


def test_capture_write_failure_preserves_original_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        patch, "_ORIGINAL_MAKE_SPEC_DECODING_STATS", lambda *args, **kwargs: "unchanged"
    )
    capture_path = tmp_path / "capture-file"
    capture_path.write_text("not a directory")
    monkeypatch.setenv(patch._CAPTURE_ENV, str(capture_path))
    warnings = []
    monkeypatch.setattr(patch.logger, "warning_once", lambda *args, **kwargs: warnings.append(args))

    result = patch._make_spec_decoding_stats(
        _scheduler(), None, 1, 1, None, "req-fail"
    )

    assert result == "unchanged"
    assert warnings


def test_update_capture_records_cpu_draft_and_sampled_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    monkeypatch.setenv("RANK", "1")
    sentinel = object()
    monkeypatch.setattr(patch, "_ORIGINAL_UPDATE_FROM_OUTPUT", lambda *a: sentinel)
    scheduler = _scheduler()
    scheduler.requests = {
        "req-a": SimpleNamespace(num_prompt_tokens=79, num_tokens=156),
    }
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"req-a": [11, 12, 13]},
    )
    model_output = SimpleNamespace(
        sampled_token_ids=[[11, 99]],
        req_id_to_index={"req-a": 0},
    )

    assert patch._update_from_output(scheduler, scheduler_output, model_output) is sentinel

    paths = list(tmp_path.glob("spec_tokens.pid*.rank1.jsonl"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["request_id"] == "req-a"
    assert record["scheduled_draft_token_ids"] == [11, 12, 13]
    assert record["sampled_token_ids"] == [11, 99]
    assert record["request_metadata"]["num_tokens"] == 156
