from pathlib import Path

import pytest
import torch

from vllm_metax.models.deepseek_v4.ffn_debug import (
    active_ffn_capture,
    maybe_ffn_capture_context,
    maybe_prepare_ffn_capture,
    reset_ffn_capture_state,
)


@pytest.fixture(autouse=True)
def _reset_state():
    reset_ffn_capture_state()
    yield
    reset_ffn_capture_state()


def _enable(monkeypatch, tmp_path: Path, *, rank="0", layers=None, calls=None):
    monkeypatch.setenv("VLLM_METAX_DSV4_FFN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", rank)
    monkeypatch.delenv("VLLM_METAX_DSV4_FFN_CAPTURE_RANKS", raising=False)
    if layers is None:
        monkeypatch.delenv("VLLM_METAX_DSV4_FFN_CAPTURE_LAYERS", raising=False)
    else:
        monkeypatch.setenv("VLLM_METAX_DSV4_FFN_CAPTURE_LAYERS", layers)
    if calls is None:
        monkeypatch.delenv("VLLM_METAX_DSV4_FFN_CAPTURE_CALLS", raising=False)
    else:
        monkeypatch.setenv("VLLM_METAX_DSV4_FFN_CAPTURE_CALLS", calls)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_disabled_capture_is_true_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_FFN_CAPTURE_DIR", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("disabled capture inspected graph state"),
    )

    assert maybe_prepare_ffn_capture(0, torch.ones(2, 3)) is None
    with maybe_ffn_capture_context(0, torch.ones(2, 3)) as capture:
        assert capture is None
    assert active_ffn_capture(0) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name,value",
    [
        ("VLLM_METAX_DSV4_FFN_CAPTURE_RANKS", "0,bad"),
        ("VLLM_METAX_DSV4_FFN_CAPTURE_RANKS", "-1"),
        ("VLLM_METAX_DSV4_FFN_CAPTURE_LAYERS", "0,bad"),
        ("VLLM_METAX_DSV4_FFN_CAPTURE_CALLS", "-1"),
    ],
)
def test_invalid_filter_specs_raise(monkeypatch, tmp_path, name, value):
    _enable(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        maybe_prepare_ffn_capture(0, torch.ones(1, 2))


def test_rank_layer_and_call_filters(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, rank="2", layers="1,3", calls="1")
    monkeypatch.setenv("VLLM_METAX_DSV4_FFN_CAPTURE_RANKS", "1,2")

    assert maybe_prepare_ffn_capture(0, torch.ones(1, 2)) is None
    assert maybe_prepare_ffn_capture(1, torch.ones(1, 2)) is None
    selected = maybe_prepare_ffn_capture(1, torch.ones(1, 2))
    assert selected is not None and selected.call == 1
    with selected:
        selected.finish(torch.zeros(1, 2))
    assert (tmp_path / "rank2_layer1_call1_final.pt").exists()


def test_nested_contexts_align_shared_stages_by_layer(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, layers="0,1")
    outer_input = torch.tensor([[1.0, 2.0]])
    inner_input = torch.tensor([[3.0, 4.0]])
    outer = maybe_prepare_ffn_capture(0, outer_input)
    assert outer is not None
    with outer:
        inner = maybe_prepare_ffn_capture(1, inner_input)
        assert inner is not None
        with inner:
            assert active_ffn_capture(0) is outer
            assert active_ffn_capture(1) is inner
            active_ffn_capture(0).record_shared("shared_input", outer_input)
            active_ffn_capture(1).record_shared("shared_input", inner_input)
            inner.finish(torch.full_like(inner_input, 5))
        outer.record_shared("gate_up_proj_output", torch.tensor([[7.0, 8.0]]))
        outer.record_shared("activation_output", torch.tensor([[9.0, 10.0]]))
        outer.record_shared("shared_final_output", torch.tensor([[11.0, 12.0]]))
        outer.finish(torch.tensor([[13.0, 14.0]]))

    payload = torch.load(
        tmp_path / "rank0_layer0_call0_shared.pt", weights_only=False
    )
    assert payload["shared_input"].tolist() == [[1.0, 2.0]]
    assert payload["gate_up_proj_output"].tolist() == [[7.0, 8.0]]
    assert payload["activation_output"].tolist() == [[9.0, 10.0]]
    assert payload["shared_final_output"].tolist() == [[11.0, 12.0]]
    final = torch.load(tmp_path / "rank0_layer0_call0_final.pt", weights_only=False)
    assert final["ffn_input"].tolist() == [[1.0, 2.0]]
    assert final["ffn_output"].tolist() == [[13.0, 14.0]]


def test_shared_mlp_reports_raw_stages_to_active_context(monkeypatch, tmp_path):
    from vllm_metax.models.deepseek_v4.model import DeepseekV4MLP

    class GateUp(torch.nn.Module):
        def forward(self, value):
            return value + 1, None

    class Down(torch.nn.Module):
        def forward(self, value):
            return value * 3, None

    _enable(monkeypatch, tmp_path)
    mlp = DeepseekV4MLP.__new__(DeepseekV4MLP)
    torch.nn.Module.__init__(mlp)
    mlp.layer_idx = 0
    mlp._capture_shared_stages = True
    mlp.gate_up_proj = GateUp()
    mlp.act_fn = lambda value: value * 2
    mlp.down_proj = Down()

    value = torch.tensor([[2.0]])
    with maybe_ffn_capture_context(0, value) as capture:
        assert capture is not None
        output = mlp(value)
        capture.finish(output)

    payload = torch.load(
        tmp_path / "rank0_layer0_call0_shared.pt", weights_only=False
    )
    assert payload["shared_input"].item() == 2
    assert payload["gate_up_proj_output"].item() == 3
    assert payload["activation_output"].item() == 6
    assert payload["shared_final_output"].item() == 18


def test_graph_capture_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_FFN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert maybe_prepare_ffn_capture(0, torch.ones(1, 2)) is None
    assert not list(tmp_path.iterdir())


def test_reset_restarts_per_layer_call_indices(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    first = maybe_prepare_ffn_capture(0, torch.ones(1, 2))
    assert first is not None and first.call == 0
    reset_ffn_capture_state()
    second = maybe_prepare_ffn_capture(0, torch.ones(1, 2))
    assert second is not None and second.call == 0
