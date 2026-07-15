import types

import pytest
from vllm.config import CUDAGraphMode

from vllm_metax.patch.performance import pre_outer_graph_device_sync as probe


ENV = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_DEVICE_SYNC"
BARRIER_ENV = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_TP_BARRIER"
ALIGNED_ENV = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_ALIGNED_REPLAY"


class _Capture:
    def __init__(self, order):
        self.order = order

    def replay(self):
        self.order.append("replay")


def _config(tp=4, dp=1, pp=1, spec_tokens=0):
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
        ),
        speculative_config=(
            None
            if spec_tokens == 0
            else types.SimpleNamespace(num_speculative_tokens=spec_tokens)
        ),
    )


_DEFAULT_CONFIG = object()


def _wrapper(vllm_config=_DEFAULT_CONFIG):
    wrapper = object.__new__(probe.BreakableCUDAGraphWrapper)
    wrapper.is_debugging_mode = False
    wrapper.vllm_config = (
        _config() if vllm_config is _DEFAULT_CONFIG else vllm_config
    )
    return wrapper


def _entry(order):
    output = object()
    return types.SimpleNamespace(
        capture=_Capture(order),
        output=output,
        input_addresses=None,
        batch_descriptor="decode",
    )


def test_default_replay_is_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    order = []
    offloader = types.SimpleNamespace(
        sync_prev_onload=lambda: order.append("offloader")
    )
    monkeypatch.setattr(probe, "get_offloader", lambda: offloader)
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: order.append("sync")),
    )
    entry = _entry(order)
    output = _wrapper()._replay(entry, (), {})
    assert output is entry.output
    assert order == ["offloader", "replay"]


def test_decode_piecewise_syncs_once_between_offloader_and_replay(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    order = []
    offloader = types.SimpleNamespace(
        sync_prev_onload=lambda: order.append("offloader")
    )
    monkeypatch.setattr(probe, "get_offloader", lambda: offloader)
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: order.append("sync")),
    )

    class Marker:
        def __enter__(self):
            order.append("marker-enter")

        def __exit__(self, exc_type, exc, tb):
            order.append("marker-exit")

    monkeypatch.setattr(
        probe,
        "record_function",
        lambda name: Marker() if name == "plan35.pre_outer_graph_device_sync" else None,
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)
    entry = _entry(order)
    _wrapper()._replay(entry, (), {})
    assert order == [
        "offloader",
        "marker-enter",
        "sync",
        "marker-exit",
        "replay",
    ]


def test_prefill_mixed_and_full_do_not_sync(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    cases = [
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=0, num_prefills=1)},
        ),
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=1)},
        ),
        (
            CUDAGraphMode.FULL,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    ]
    for mode, attn_metadata in cases:
        order = []
        monkeypatch.setattr(
            probe,
            "get_offloader",
            lambda order=order: types.SimpleNamespace(
                sync_prev_onload=lambda: order.append("offloader")
            ),
        )
        monkeypatch.setattr(
            probe,
            "get_forward_context",
            lambda mode=mode, attn_metadata=attn_metadata: types.SimpleNamespace(
                cudagraph_runtime_mode=mode,
                attn_metadata=attn_metadata,
            ),
        )
        monkeypatch.setattr(
            probe.torch.cuda,
            "current_stream",
            lambda order=order: types.SimpleNamespace(
                synchronize=lambda: order.append("unexpected-sync")
            ),
        )
        _wrapper()._replay(_entry(order), (), {})
        assert order == ["offloader", "replay"]


def test_outer_replay_syncs_once_when_capture_runs_inner_callback(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    calls = {"sync": 0, "inner": 0}
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(
            synchronize=lambda: calls.__setitem__("sync", calls["sync"] + 1)
        ),
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)

    class CaptureWithInner:
        def replay(self):
            calls["inner"] += 1

    entry = types.SimpleNamespace(
        capture=CaptureWithInner(),
        output=object(),
        input_addresses=None,
        batch_descriptor="decode",
    )
    _wrapper()._replay(entry, (), {})
    assert calls == {"sync": 1, "inner": 1}


def test_invalid_env_fails_before_offloader_or_replay(monkeypatch):
    monkeypatch.setenv(ENV, "true")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    with pytest.raises(ValueError, match=ENV):
        _wrapper()._replay(_entry(order), (), {})
    assert order == []


def test_diagnostic_warning_is_emitted_once(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(probe, "_warning_emitted", False)
    warnings = []
    monkeypatch.setattr(probe.logger, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    wrapper = _wrapper()
    wrapper._replay(_entry([]), (), {})
    wrapper._replay(_entry([]), (), {})
    assert len(warnings) == 1


def test_debug_address_assertion_and_output_identity_are_preserved(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    wrapper = _wrapper()
    wrapper.is_debugging_mode = True
    tensor = probe.torch.ones(1)
    entry = _entry(order)
    entry.input_addresses = [tensor.data_ptr()]
    output = wrapper._replay(entry, (tensor,), {})
    assert output is entry.output
    assert order == ["offloader", "replay"]

    mismatched = probe.torch.ones(1)
    with pytest.raises(AssertionError, match="Input tensor addresses changed"):
        wrapper._replay(entry, (mismatched,), {})
    assert order == ["offloader", "replay"]


def test_tp_gloo_barrier_runs_once_before_outer_replay(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    tp_group = types.SimpleNamespace(
        world_size=4,
        rank_in_group=2,
        cpu_group=cpu_group,
        barrier=lambda: order.append("barrier"),
    )
    monkeypatch.setattr(probe, "get_tp_group", lambda: tp_group, raising=False)
    monkeypatch.setattr(
        probe.torch.distributed,
        "get_backend",
        lambda group: "gloo" if group is cpu_group else pytest.fail("device group"),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: pytest.fail("CUDA sync path was called"),
    )

    class Marker:
        def __enter__(self):
            order.append("marker-enter")

        def __exit__(self, exc_type, exc, tb):
            order.append("marker-exit")

    monkeypatch.setattr(
        probe,
        "record_function",
        lambda name: Marker()
        if name == "plan35.pre_outer_graph_tp_gloo_barrier"
        else pytest.fail(name),
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)
    _wrapper()._replay(_entry(order), (), {})
    assert order == [
        "offloader",
        "marker-enter",
        "barrier",
        "marker-exit",
        "replay",
    ]


@pytest.mark.parametrize(
    ("sync_value", "barrier_value", "message"),
    [("0", "true", BARRIER_ENV), ("1", "1", "conflicts")],
)
def test_tp_barrier_invalid_and_mutual_env_fail_before_replay(
    monkeypatch, sync_value, barrier_value, message
):
    monkeypatch.setenv(ENV, sync_value)
    monkeypatch.setenv(BARRIER_ENV, barrier_value)
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        _wrapper()._replay(_entry(order), (), {})
    assert order == []


@pytest.mark.parametrize(
    ("world_size", "backend", "message"),
    [(2, "gloo", "parallel size 4"), (4, "nccl", "gloo CPU backend")],
)
def test_tp_barrier_rejects_non_tp4_and_non_gloo(
    monkeypatch, world_size, backend, message
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=world_size,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: order.append("unexpected-barrier"),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: backend
    )
    with pytest.raises(RuntimeError, match=message):
        _wrapper()._replay(_entry(order), (), {})
    assert order == ["offloader"]


def test_tp_barrier_exception_stops_outer_replay(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()

    def fail_barrier():
        raise RuntimeError("gloo barrier failed")

    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=fail_barrier,
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="gloo barrier failed"):
        _wrapper()._replay(_entry(order), (), {})
    assert order == ["offloader"]


def test_tp_barrier_warning_once_includes_diagnostic_fields(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    monkeypatch.setattr(probe, "_barrier_warning_emitted", False)
    warnings = []
    monkeypatch.setattr(
        probe.logger, "warning", lambda message, *args: warnings.append((message, args))
    )
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    tp_group = types.SimpleNamespace(
        world_size=4,
        rank_in_group=2,
        cpu_group=cpu_group,
        barrier=lambda: None,
    )
    monkeypatch.setattr(probe, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    wrapper = _wrapper()
    wrapper._replay(_entry([]), (), {})
    wrapper._replay(_entry([]), (), {})
    assert len(warnings) == 1
    message, args = warnings[0]
    rendered = message % args
    assert "mode=PIECEWISE" in rendered
    assert "backend=gloo" in rendered
    assert "world_size=4" in rendered
    assert "rank=2" in rendered
    assert f"pid={probe.os.getpid()}" in rendered


@pytest.mark.parametrize(
    ("mode", "attn_metadata"),
    [
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=0, num_prefills=1)},
        ),
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=1)},
        ),
        (
            CUDAGraphMode.FULL,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    ],
)
def test_tp_barrier_prefill_mixed_and_full_do_not_barrier(
    monkeypatch, mode, attn_metadata
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=mode,
            attn_metadata=attn_metadata,
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: pytest.fail("TP group requested outside pure decode"),
    )
    _wrapper()._replay(_entry(order), (), {})
    assert order == ["offloader", "replay"]


def test_tp_barrier_runs_once_when_outer_replay_invokes_inner_callback(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    calls = {"barrier": 0, "inner": 0}
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: calls.__setitem__(
                "barrier", calls["barrier"] + 1
            ),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)

    class CaptureWithInner:
        def replay(self):
            calls["inner"] += 1

    entry = types.SimpleNamespace(
        capture=CaptureWithInner(),
        output=object(),
        input_addresses=None,
        batch_descriptor="decode",
    )
    _wrapper()._replay(entry, (), {})
    assert calls == {"barrier": 1, "inner": 1}


@pytest.mark.parametrize(
    ("tp", "dp", "pp", "spec_tokens", "message"),
    [
        (2, 1, 1, 0, "TP4/DP1/PP1"),
        (4, 2, 1, 0, "TP4/DP1/PP1"),
        (4, 1, 2, 0, "TP4/DP1/PP1"),
        (4, 1, 1, 1, "MTP/speculation off"),
    ],
)
def test_tp_barrier_rejects_incompatible_vllm_config(
    monkeypatch, tp, dp, pp, spec_tokens, message
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: order.append("unexpected-barrier"),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    with pytest.raises(RuntimeError, match=message):
        _wrapper(_config(tp, dp, pp, spec_tokens))._replay(_entry(order), (), {})
    assert order == ["offloader"]


def test_aligned_replay_syncs_then_barriers_exactly_once(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=1,
            cpu_group=cpu_group,
            barrier=lambda: order.append("barrier"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: order.append("sync")),
    )

    class Marker:
        def __enter__(self):
            order.append("marker-enter")

        def __exit__(self, exc_type, exc, tb):
            order.append("marker-exit")

    monkeypatch.setattr(
        probe,
        "record_function",
        lambda name: Marker()
        if name == "plan35.pre_outer_graph_aligned_replay"
        else pytest.fail(name),
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)
    _wrapper()._replay(_entry(order), (), {})
    assert order == [
        "offloader",
        "marker-enter",
        "sync",
        "barrier",
        "marker-exit",
        "replay",
    ]


@pytest.mark.parametrize(
    ("sync_value", "barrier_value", "aligned_value", "error", "message"),
    [
        ("0", "0", "true", ValueError, ALIGNED_ENV),
        ("1", "0", "1", RuntimeError, "mutually exclusive"),
        ("0", "1", "1", RuntimeError, "mutually exclusive"),
    ],
)
def test_aligned_replay_invalid_and_mutual_env_fail_before_offloader(
    monkeypatch, sync_value, barrier_value, aligned_value, error, message
):
    monkeypatch.setenv(ENV, sync_value)
    monkeypatch.setenv(BARRIER_ENV, barrier_value)
    monkeypatch.setenv(ALIGNED_ENV, aligned_value)
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    with pytest.raises(error, match=message):
        _wrapper()._replay(_entry(order), (), {})
    assert order == []


@pytest.mark.parametrize(
    ("mode", "attn_metadata"),
    [
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=0, num_prefills=1)},
        ),
        (
            CUDAGraphMode.PIECEWISE,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=1)},
        ),
        (
            CUDAGraphMode.FULL,
            {"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    ],
)
def test_aligned_replay_prefill_mixed_and_full_do_not_align(
    monkeypatch, mode, attn_metadata
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=mode,
            attn_metadata=attn_metadata,
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: pytest.fail("TP group requested outside pure decode"),
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: pytest.fail("CUDA sync requested outside pure decode"),
    )
    _wrapper()._replay(_entry(order), (), {})
    assert order == ["offloader", "replay"]


@pytest.mark.parametrize(
    ("world_size", "backend", "message"),
    [(2, "gloo", "parallel size 4"), (4, "nccl", "gloo CPU backend")],
)
def test_aligned_replay_rejects_non_tp4_and_non_gloo(
    monkeypatch, world_size, backend, message
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=world_size,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: order.append("unexpected-barrier"),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: backend
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: pytest.fail("CUDA sync happened before guard validation"),
    )
    with pytest.raises(RuntimeError, match=message):
        _wrapper()._replay(_entry(order), (), {})
    assert order == ["offloader"]


@pytest.mark.parametrize("failure_stage", ["sync", "barrier"])
def test_aligned_replay_sync_and_barrier_errors_stop_outer_replay(
    monkeypatch, failure_stage
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()

    def synchronize():
        order.append("sync")
        if failure_stage == "sync":
            raise RuntimeError("aligned sync failed")

    def barrier():
        order.append("barrier")
        if failure_stage == "barrier":
            raise RuntimeError("aligned barrier failed")

    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=barrier,
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=synchronize),
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match=f"aligned {failure_stage} failed"):
        _wrapper()._replay(_entry(order), (), {})
    expected = ["offloader", "sync"]
    if failure_stage == "barrier":
        expected.append("barrier")
    assert order == expected


def test_aligned_replay_warning_once_includes_diagnostic_fields(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    monkeypatch.setattr(probe, "_aligned_warning_emitted", False)
    warnings = []
    monkeypatch.setattr(
        probe.logger, "warning", lambda message, *args: warnings.append((message, args))
    )
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    tp_group = types.SimpleNamespace(
        world_size=4,
        rank_in_group=3,
        cpu_group=cpu_group,
        barrier=lambda: None,
    )
    monkeypatch.setattr(probe, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    wrapper = _wrapper()
    wrapper._replay(_entry([]), (), {})
    wrapper._replay(_entry([]), (), {})
    assert len(warnings) == 1
    message, args = warnings[0]
    rendered = message % args
    assert "mode=PIECEWISE" in rendered
    assert "backend=gloo" in rendered
    assert "world_size=4" in rendered
    assert "rank=3" in rendered
    assert f"pid={probe.os.getpid()}" in rendered


def test_aligned_replay_runs_once_when_outer_replay_invokes_inner_callback(
    monkeypatch,
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(BARRIER_ENV, raising=False)
    monkeypatch.setenv(ALIGNED_ENV, "1")
    calls = {"sync": 0, "barrier": 0, "inner": 0}
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(sync_prev_onload=lambda: None),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: calls.__setitem__(
                "barrier", calls["barrier"] + 1
            ),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(
            synchronize=lambda: calls.__setitem__("sync", calls["sync"] + 1)
        ),
    )
    monkeypatch.setattr(probe.logger, "warning", lambda *args, **kwargs: None)

    class CaptureWithInner:
        def replay(self):
            calls["inner"] += 1

    entry = types.SimpleNamespace(
        capture=CaptureWithInner(),
        output=object(),
        input_addresses=None,
        batch_descriptor="decode",
    )
    _wrapper()._replay(entry, (), {})
    assert calls == {"sync": 1, "barrier": 1, "inner": 1}


@pytest.mark.parametrize(
    ("vllm_config", "message"),
    [(_config(tp=2), "TP4/DP1/PP1"), (None, "wrapper vllm_config")],
)
def test_wrapper_config_is_validated_when_global_config_is_unset(
    monkeypatch, vllm_config, message
):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(BARRIER_ENV, "1")
    monkeypatch.delenv(ALIGNED_ENV, raising=False)
    order = []
    monkeypatch.setattr(
        probe,
        "get_offloader",
        lambda: types.SimpleNamespace(
            sync_prev_onload=lambda: order.append("offloader")
        ),
    )
    monkeypatch.setattr(
        probe,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={"mla": types.SimpleNamespace(num_decodes=1, num_prefills=0)},
        ),
    )
    cpu_group = object()
    monkeypatch.setattr(
        probe,
        "get_tp_group",
        lambda: types.SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            cpu_group=cpu_group,
            barrier=lambda: order.append("unexpected-barrier"),
        ),
    )
    monkeypatch.setattr(
        probe.torch.distributed, "get_backend", lambda group: "gloo"
    )
    with pytest.raises(RuntimeError, match=message):
        _wrapper(vllm_config)._replay(_entry(order), (), {})
    assert order == ["offloader"]
