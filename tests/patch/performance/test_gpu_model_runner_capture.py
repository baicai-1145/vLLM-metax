import importlib
import types

import torch

from vllm_metax.patch.performance import gpu_model_runner_capture as capture


ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR"


def test_import_without_capture_env_leaves_execute_model_unpatched(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    original = capture._ORIGINAL_EXECUTE_MODEL
    capture.GPUModelRunner.execute_model = original

    importlib.reload(capture)

    assert capture.GPUModelRunner.execute_model is original


def _runner(
    monkeypatch,
    *,
    logits,
    sample_hidden_states,
    result,
    hidden_states=None,
    scheduler_output=None,
):
    capture._install_patch()
    runner = object.__new__(capture.GPUModelRunner)
    runner.execute_model_state = None

    def original(self, *args, **kwargs):
        self.execute_model_state = types.SimpleNamespace(
            logits=logits,
            sample_hidden_states=sample_hidden_states,
            hidden_states=hidden_states,
            scheduler_output=scheduler_output,
        )
        return result

    monkeypatch.setattr(capture, "_ORIGINAL_EXECUTE_MODEL", original)
    monkeypatch.setattr(capture, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        capture.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        capture.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    return runner


def test_enabled_capture_writes_post_execute_payload(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    sample_hidden_states = torch.tensor([[3.0, 5.0]])
    result = object()
    runner = _runner(
        monkeypatch,
        logits=logits,
        sample_hidden_states=sample_hidden_states,
        result=result,
    )

    returned = runner.execute_model(object())

    assert returned is result
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    payload = torch.load(files[0], weights_only=True)
    assert payload["schema"] == "vllm_metax.dsv4_graph_capture.v1"
    assert payload["rank"] == 2
    assert payload["call"] == 0
    assert torch.equal(payload["sample_hidden_states"], sample_hidden_states)
    assert torch.equal(payload["logits"], logits)
    assert payload["top2_indices"].tolist() == [[1, 2]]
    assert payload["argmax"].tolist() == [1]


def test_enabled_capture_includes_selected_pre_hc_hidden_states(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[7.0, 8.0]]),
        result="result",
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )
    pre_hc = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    runner.get_model = lambda: types.SimpleNamespace(
        get_mtp_target_hidden_states=lambda: pre_hc
    )

    runner.execute_model(object())

    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert torch.equal(payload["pre_hc_hidden_states"], pre_hc[:2])


def test_enabled_capture_includes_graph_layer_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[7.0, 8.0]]),
        result="result",
    )
    refs = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[[3.0, 4.0]]]),
        torch.tensor([[[5.0]]]),
        torch.tensor([[[6.0]]]),
    )
    layer = types.SimpleNamespace(
        layer_idx=30,
        _graph_capture_output_enabled=True,
        get_graph_capture_output_buffers=lambda: refs,
        get_graph_capture_stage_buffers=lambda: {"after_attention": refs},
        get_graph_capture_mhc_input=lambda: torch.tensor([[9.0, 10.0]]),
    )
    runner.get_model = lambda: types.SimpleNamespace(
        named_modules=lambda: iter((("", object()), ("model.layers.30", layer)))
    )

    runner.execute_model(object())

    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert payload["layer_outputs"]["30"]["module"] == "model.layers.30"
    assert payload["layer_outputs"]["30"]["before_mhc_hidden_states"].tolist() == [
        [9.0, 10.0]
    ]
    for name, expected in zip(
        ("hidden_states", "residual", "post_mix", "res_mix"), refs, strict=True
    ):
        assert torch.equal(payload["layer_outputs"]["30"][name], expected)
        assert torch.equal(
            payload["layer_outputs"]["30"]["stages"]["after_attention"][name],
            expected,
        )


def test_enabled_capture_includes_weak_layer_outputs_without_strong_capture_flag(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[7.0, 8.0]]),
        result="result",
    )
    refs = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[7.0, 8.0]]),
    )
    layer = types.SimpleNamespace(
        layer_idx=30,
        _graph_capture_output_enabled=False,
        get_graph_weak_stage_buffers=lambda: {"before_mhc": refs},
    )
    runner.get_model = lambda: types.SimpleNamespace(
        named_modules=lambda: iter((("model.layers.30", layer),))
    )

    runner.execute_model(object())

    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert payload["weak_layer_outputs"]["30"]["module"] == "model.layers.30"
    for name, expected in zip(
        ("hidden_states", "residual", "post_mix", "res_mix"), refs, strict=True
    ):
        assert torch.equal(
            payload["weak_layer_outputs"]["30"]["stages"]["before_mhc"][name],
            expected,
        )


def test_disabled_capture_is_strict_noop(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    original = capture._ORIGINAL_EXECUTE_MODEL
    capture.GPUModelRunner.execute_model = original
    capture._install_patch()

    assert capture.GPUModelRunner.execute_model is original
    assert not list(tmp_path.iterdir())


def test_rank_and_call_filters_select_only_matching_invocation(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    monkeypatch.setenv(capture._RANKS_ENV, "2")
    monkeypatch.setenv(capture._CALLS_ENV, "1")
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[3.0]]),
        result="result",
    )

    runner.execute_model(object())
    runner.execute_model(object())
    runner.execute_model(object())

    files = list(tmp_path.glob("*.pt"))
    assert [path.name for path in files] == ["rank2_call1.pt"]
    assert getattr(runner, capture._COUNTER_ATTR) == 3


def test_request_and_output_token_filters_ignore_global_call_index(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(ENV, str(tmp_path))
    monkeypatch.setenv(capture._REQUEST_IDS_ENV, "84")
    monkeypatch.setenv(capture._OUTPUT_TOKENS_ENV, "73")
    cached = types.SimpleNamespace(
        req_ids=["84-randomsuffix"],
        num_output_tokens=[73],
        num_computed_tokens=[999],
    )
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[3.0]]),
        result="result",
        scheduler_output=types.SimpleNamespace(scheduled_cached_reqs=cached),
    )

    runner.execute_model(object())

    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert payload["scheduled_requests"] == [
        {
            "request_id": "84-randomsuffix",
            "num_output_tokens": 73,
            "num_computed_tokens": 999,
        }
    ]


def test_capture_active_skips_capture_and_counter(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = _runner(
        monkeypatch,
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[3.0]]),
        result="result",
    )
    monkeypatch.setattr(capture.torch.cuda, "is_current_stream_capturing", lambda: True)

    assert runner.execute_model(object()) == "result"
    assert not list(tmp_path.iterdir())
    assert not hasattr(runner, capture._COUNTER_ATTR)


def test_capture_preserves_return_state_and_tensor_identity(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = object.__new__(capture.GPUModelRunner)
    runner.execute_model_state = None
    state = types.SimpleNamespace(
        logits=torch.tensor([[1.0, 2.0]]),
        sample_hidden_states=torch.tensor([[3.0]]),
    )
    state_logits_ptr = state.logits.data_ptr()
    state_hidden_ptr = state.sample_hidden_states.data_ptr()
    result = object()

    def original(self, *args, **kwargs):
        self.execute_model_state = state
        return result

    monkeypatch.setattr(capture, "_ORIGINAL_EXECUTE_MODEL", original)
    monkeypatch.setattr(capture, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(capture.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        capture.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )

    assert runner.execute_model(object()) is result
    assert runner.execute_model_state is state
    assert state.logits.data_ptr() == state_logits_ptr
    assert state.sample_hidden_states.data_ptr() == state_hidden_ptr


def test_missing_state_and_logits_are_serialized_as_none(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    runner = object.__new__(capture.GPUModelRunner)
    runner.execute_model_state = None
    states = [None, types.SimpleNamespace(logits=None, sample_hidden_states=None)]

    def original(self, *args, **kwargs):
        self.execute_model_state = states.pop(0)
        return None

    monkeypatch.setattr(capture, "_ORIGINAL_EXECUTE_MODEL", original)
    monkeypatch.setattr(capture, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(capture.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        capture.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )

    runner.execute_model(object())
    runner.execute_model(object())

    payloads = [torch.load(path, weights_only=True) for path in sorted(tmp_path.glob("*.pt"))]
    assert len(payloads) == 2
    for payload in payloads:
        assert payload["logits"] is None
        assert payload["sample_hidden_states"] is None
        assert payload["top2_values"] is None
        assert payload["top2_indices"] is None
        assert payload["argmax"] is None


def test_patch_is_idempotent(monkeypatch):
    wrapped = capture.GPUModelRunner.execute_model
    importlib.reload(capture)
    assert capture.GPUModelRunner.execute_model is wrapped
