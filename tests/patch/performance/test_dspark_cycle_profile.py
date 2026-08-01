import json

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_metax.models.deepseek_v4.dspark import DSparkDeepseekV4Model
from vllm_metax.models.deepseek_v4.model import DeepseekV4ForCausalLM
from vllm_metax.patch.performance import dspark_cycle_profile


def test_wrap_method_emits_named_phase_and_preserves_result():
    class Runner:
        def sample(self, value):
            return value + 1

    dspark_cycle_profile._wrap_method(Runner, "sample", "target_accept")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        result = Runner().sample(4)

    assert result == 5
    assert "dspark_cycle: target_accept" in {
        event.key for event in profile.key_averages()
    }


def test_wrap_method_is_idempotent():
    class Proposer:
        def sample(self):
            return "ok"

    dspark_cycle_profile._wrap_method(Proposer, "sample", "draft_sample")
    first = Proposer.sample
    dspark_cycle_profile._wrap_method(Proposer, "sample", "draft_sample")

    assert Proposer.sample is first


def test_install_patch_targets_active_v2_dspark_path(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSPARK_PROFILE_PHASES", "1")

    dspark_cycle_profile._install_patch()

    assert (
        getattr(GPUModelRunner.sample, dspark_cycle_profile._PATCH_MARKER)
        == "target_accept"
    )
    assert (
        getattr(DeepseekV4ForCausalLM.forward, dspark_cycle_profile._PATCH_MARKER)
        == "target_forward"
    )
    assert (
        getattr(
            DSparkDeepseekV4Model.precompute_and_store_context_kv,
            dspark_cycle_profile._PATCH_MARKER,
        )
        == "draft_context_kv"
    )
    assert (
        getattr(DSparkDeepseekV4Model.forward, dspark_cycle_profile._PATCH_MARKER)
        == "draft_backbone"
    )
    assert (
        getattr(DSparkSpeculator._sample_sequential, dspark_cycle_profile._PATCH_MARKER)
        == "draft_sample"
    )


def test_wrap_method_records_normal_phase_timing(monkeypatch, tmp_path):
    class FakeEvent:
        next_timestamp = 0.0

        def __init__(self, enable_timing):
            assert enable_timing
            self.timestamp = None

        def record(self, stream):
            assert stream.cuda_stream == 17
            self.timestamp = FakeEvent.next_timestamp
            FakeEvent.next_timestamp += 2.5

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    class FakeStream:
        cuda_stream = 17

    class Runner:
        def sample(self, value):
            return value + 1

    output_dir = tmp_path / "phase-events"
    monkeypatch.delenv("VLLM_METAX_DSPARK_PROFILE_PHASES", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSPARK_PHASE_TIMING_DIR", str(output_dir))
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(dspark_cycle_profile.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda, "current_stream", lambda: FakeStream()
    )
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(dspark_cycle_profile.torch.cuda, "synchronize", lambda: None)
    dspark_cycle_profile._reset_event_timings_for_test()
    dspark_cycle_profile._wrap_method(Runner, "sample", "target_accept")

    assert Runner().sample(4) == 5
    path = dspark_cycle_profile._flush_event_timings()

    assert path is not None
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["phase"] == "target_accept"
    assert records[0]["rank"] == 2
    assert records[0]["cuda_ms"] == 2.5
    assert records[0]["stream"] == 17


def test_phase_timing_skips_cuda_graph_capture(monkeypatch, tmp_path):
    class Runner:
        def sample(self):
            return "captured"

    monkeypatch.delenv("VLLM_METAX_DSPARK_PROFILE_PHASES", raising=False)
    monkeypatch.setenv(
        "VLLM_METAX_DSPARK_PHASE_TIMING_DIR", str(tmp_path / "phase-events")
    )
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda,
        "Event",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("event allocated")),
    )
    dspark_cycle_profile._reset_event_timings_for_test()
    dspark_cycle_profile._wrap_method(Runner, "sample", "draft_sample")

    assert Runner().sample() == "captured"
    assert dspark_cycle_profile._flush_event_timings() is None


def test_phase_timing_disabled_does_not_touch_cuda(monkeypatch):
    class Runner:
        def sample(self):
            return "normal"

    monkeypatch.delenv("VLLM_METAX_DSPARK_PROFILE_PHASES", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSPARK_PHASE_TIMING_DIR", raising=False)
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda,
        "current_stream",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA timing touched")),
    )
    dspark_cycle_profile._reset_event_timings_for_test()
    dspark_cycle_profile._wrap_method(Runner, "sample", "target_forward")

    assert Runner().sample() == "normal"
    assert dspark_cycle_profile._flush_event_timings() is None


def test_phase_timing_flushes_once_at_record_limit(monkeypatch, tmp_path):
    class FakeEvent:
        timestamp = 0.0

        def __init__(self, enable_timing):
            assert enable_timing
            self.value = None

        def record(self, stream):
            self.value = FakeEvent.timestamp
            FakeEvent.timestamp += 1.0

        def elapsed_time(self, other):
            return other.value - self.value

    class FakeStream:
        cuda_stream = 19

    class Runner:
        def sample(self):
            return "ok"

    output_dir = tmp_path / "phase-events"
    monkeypatch.delenv("VLLM_METAX_DSPARK_PROFILE_PHASES", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSPARK_PHASE_TIMING_DIR", str(output_dir))
    monkeypatch.setenv("VLLM_METAX_DSPARK_PHASE_TIMING_MAX_RECORDS", "1")
    monkeypatch.setattr(dspark_cycle_profile.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda, "current_stream", lambda: FakeStream()
    )
    monkeypatch.setattr(
        dspark_cycle_profile.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(dspark_cycle_profile.torch.cuda, "synchronize", lambda: None)
    dspark_cycle_profile._reset_event_timings_for_test()
    dspark_cycle_profile._wrap_method(Runner, "sample", "target_forward")

    assert Runner().sample() == "ok"
    assert Runner().sample() == "ok"

    paths = list(output_dir.glob("*.jsonl"))
    assert len(paths) == 1
    assert len(paths[0].read_text().splitlines()) == 1
