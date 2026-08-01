import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm import forward_context
import vllm_metax.models.deepseek_v4.layer_debug as layer_debug
from vllm_metax.models.deepseek_v4.layer_debug import (
    copy_graph_layer_decode_output_to_workspace,
    copy_graph_layer_mhc_input_to_workspace,
    graph_layer_workspace_outputs,
    graph_layer_workspace_mhc_input,
    graph_layer_capture_layer_enabled,
    graph_weak_capture_stages,
    layer_capture_layer_enabled,
    maybe_capture_attention_inputs,
    maybe_capture_attention_output,
    maybe_capture_qkv_prenorm_shadow_compare,
    maybe_capture_qkv_producer,
    maybe_layer_capture_context,
    maybe_prepare_q_stage_capture,
)
from vllm_metax.models.deepseek_v4.layer_debug import reset_layer_capture_state
from vllm_metax.models.deepseek_v4.model import DeepseekV4DecoderLayer


def test_layer_capture_defaults_to_layer_zero(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", raising=False)

    assert layer_capture_layer_enabled(0)
    assert not layer_capture_layer_enabled(1)

    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "  ")
    assert layer_capture_layer_enabled(0)
    assert not layer_capture_layer_enabled(1)


def test_layer_capture_selects_configured_layers(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", " 0, 7,60,7 ")

    assert layer_capture_layer_enabled(0)
    assert layer_capture_layer_enabled(7)
    assert layer_capture_layer_enabled(60)
    assert not layer_capture_layer_enabled(1)


def test_graph_layer_capture_requires_explicit_selected_layers(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR", raising=False)
    monkeypatch.delenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_LAYERS", raising=False)
    assert not graph_layer_capture_layer_enabled(0)

    monkeypatch.setenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR", "/tmp/unused")
    assert not graph_layer_capture_layer_enabled(0)

    monkeypatch.setenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_LAYERS", "0,30,60")
    assert graph_layer_capture_layer_enabled(0)
    assert graph_layer_capture_layer_enabled(30)
    assert graph_layer_capture_layer_enabled(60)
    assert not graph_layer_capture_layer_enabled(1)

    monkeypatch.setenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_LAYERS", "all")
    assert graph_layer_capture_layer_enabled(1)


def test_graph_weak_capture_requires_dir_layers_and_stages(monkeypatch):
    assert graph_weak_capture_stages(14) == set()

    monkeypatch.setenv("VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR", "/tmp/unused")
    monkeypatch.setenv("VLLM_METAX_DSV4_GRAPH_WEAK_CAPTURE_LAYERS", "13,14")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_GRAPH_WEAK_CAPTURE_STAGES",
        "before_mhc,after_attention",
    )

    assert graph_weak_capture_stages(14) == {"before_mhc", "after_attention"}
    assert graph_weak_capture_stages(15) == set()


def test_graph_weak_stage_buffers_do_not_retain_tensors():
    layer = object.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer._graph_weak_capture_stages = {"after_attention"}
    layer._graph_weak_refs = {}
    values = tuple(torch.tensor([float(index)]) for index in range(4))
    references = tuple(weakref.ref(value) for value in values)

    layer._store_graph_weak_refs("after_attention", *values)

    assert "after_attention" in layer.get_graph_weak_stage_buffers()
    del values
    gc.collect()
    assert all(reference() is None for reference in references)
    assert layer.get_graph_weak_stage_buffers() == {}


def test_graph_layer_workspace_capture_is_decode_only_and_allocation_free():
    norm_weight = torch.ones(2)
    buffers = {
        "normalized": torch.zeros(1, 2),
        "residual_cur": torch.ones(1, 1, 2),
        "post_mix": torch.ones(1, 1),
        "comb_mix": torch.ones(1, 1, 1),
    }
    workspace = {(norm_weight.data_ptr(), "cpu", 1): buffers}
    outputs = graph_layer_workspace_outputs(workspace, norm_weight)
    assert outputs is not None
    assert outputs[0] is buffers["normalized"]
    assert outputs[1] is buffers["residual_cur"]
    assert outputs[2].shape == (1, 1, 1)
    assert outputs[3] is buffers["comb_mix"]

    copy_graph_layer_decode_output_to_workspace(
        workspace, norm_weight, torch.full((1, 2), 3.0)
    )
    assert torch.equal(buffers["normalized"], torch.full((1, 2), 3.0))

    copy_graph_layer_decode_output_to_workspace(
        workspace, norm_weight, torch.full((2, 2), 7.0)
    )
    assert torch.equal(buffers["normalized"], torch.full((1, 2), 3.0))


def test_graph_layer_workspace_captures_mhc_input_without_allocation():
    norm_weight = torch.ones(2)
    residual_fp32 = torch.zeros(1, 8)
    workspace = {
        (norm_weight.data_ptr(), "cpu", 1): {"residual_fp32": residual_fp32}
    }
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    copy_graph_layer_mhc_input_to_workspace(
        workspace, norm_weight, hidden_states
    )

    captured = graph_layer_workspace_mhc_input(workspace, norm_weight, 2)
    assert captured is not None
    assert captured.data_ptr() == residual_fp32.data_ptr()
    assert torch.equal(captured, hidden_states.float())


@pytest.mark.parametrize("spec", ["bad", "-1", "0,bad", "0,-2"])
def test_layer_capture_rejects_invalid_rank_spec(monkeypatch, spec):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", spec)
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", "/tmp/unused")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(ValueError, match="VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS"):
        maybe_layer_capture_context(0)


def test_layer_capture_selects_rank_two_and_all(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "1,2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    context = maybe_layer_capture_context(0)
    assert context is not None
    assert context.rank == 2
    context.save_stage("before_attention", hidden_states=torch.ones(1, 2))
    assert next(tmp_path.glob("*.pt")).name.startswith("rank2_")
    output_path = maybe_capture_attention_output(
        0, torch.tensor([1]), torch.ones(1, 2)
    )
    assert output_path is not None and output_path.name.startswith("rank2_")

    for path in tmp_path.glob("*.pt"):
        path.unlink()
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "all")
    reset_layer_capture_state()
    assert maybe_capture_attention_inputs(
        0, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
    ).name.startswith("rank2_")


def test_q_stage_capture_preserves_raw_and_post_q(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    raw_q = torch.ones(1, 2, 3)
    capture = maybe_prepare_q_stage_capture(0, torch.tensor([543]), raw_q)
    assert capture is not None
    raw_q.mul_(2)
    path = capture.finish(raw_q)
    assert path is not None
    payload = torch.load(path, weights_only=False)
    assert payload["stage"] == "q_stages"
    assert payload["rank"] == 2
    assert payload["call"] == 0
    torch.testing.assert_close(payload["raw_q"], torch.ones(1, 2, 3))
    torch.testing.assert_close(payload["post_q"], torch.full((1, 2, 3), 2.0))
    assert payload["positions"].tolist() == [543]


def test_qkv_producer_capture_saves_pre_and_post_norm_rows(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "2")
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_POSITIONS", "658")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    qr_kv = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    qr = torch.tensor([[11.0, 21.0], [41.0, 51.0]])
    kv = torch.tensor([[31.0], [61.0]])
    path = maybe_capture_qkv_producer(
        0,
        torch.tensor([657, 658]),
        hidden,
        qr_kv,
        2,
        qr,
        kv,
    )
    assert path is not None
    payload = torch.load(path, weights_only=False)
    assert payload["stage"] == "qkv_producer"
    assert payload["rank"] == 2
    assert payload["call"] == 0
    assert payload["positions"].tolist() == [657, 658]
    tensors = payload["tensors"]
    torch.testing.assert_close(tensors["hidden_states"], hidden)
    torch.testing.assert_close(tensors["qr_kv"], qr_kv)
    torch.testing.assert_close(tensors["qr_pre_norm"], qr_kv[:, :2])
    torch.testing.assert_close(tensors["kv_pre_norm"], qr_kv[:, 2:])
    torch.testing.assert_close(tensors["qr"], qr)
    torch.testing.assert_close(tensors["kv"], kv)


def test_q_stage_capture_disabled_and_graph_safe(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", raising=False)
    assert maybe_prepare_q_stage_capture(
        0, torch.tensor([1]), torch.ones(1, 2, 3)
    ) is None

    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    reset_layer_capture_state()
    assert maybe_prepare_q_stage_capture(
        0, torch.tensor([1]), torch.ones(1, 2, 3)
    ) is None
    assert not list(tmp_path.iterdir())


def test_wq_b_shadow_compare_saves_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_WQ_B_SHADOW_COMPARE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    path = layer_debug.maybe_capture_wq_b_shadow_compare(
        0,
        torch.tensor([767, 768]),
        torch.zeros(2, 3),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[1.0, 2.5], [2.0, 4.0]]),
    )

    assert path is not None
    payload = torch.load(path, weights_only=False)
    assert payload["stage"] == "wq_b_shadow_compare"
    assert payload["rank"] == 2
    assert payload["positions"].tolist() == [767, 768]
    assert payload["num_diff"] == 2
    assert payload["max_abs"] == 1.0


def test_qkv_prenorm_shadow_compare_saves_qr_and_kv_summary(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_QKV_PRENORM_SHADOW_COMPARE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()
    path = maybe_capture_qkv_prenorm_shadow_compare(
        0,
        torch.tensor([673, 674]),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[1.0, 2.0, 3.5], [4.0, 5.0, 6.5]]),
        2,
    )
    assert path is not None
    payload = torch.load(path, weights_only=False)
    assert payload["stage"] == "qkv_prenorm_shadow_compare"
    assert payload["rank"] == 2
    assert payload["positions"].tolist() == [673, 674]
    assert payload["summary"]["qr"]["num_diff"] == 0
    assert payload["summary"]["kv"]["num_diff"] == 2
    assert payload["summary"]["kv"]["first_diff_index"] == [0, 0]
    assert payload["summary"]["kv"]["batched_value"] == 3.0
    assert payload["summary"]["kv"]["rowwise_value"] == 3.5


def test_q_stage_call_alignment_and_reset(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    input_path = maybe_capture_attention_inputs(
        0, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
    )
    capture = maybe_prepare_q_stage_capture(0, torch.tensor([1]), torch.ones(1, 1, 2))
    assert input_path is not None and capture is not None
    q_path = capture.finish(torch.zeros(1, 1, 2))
    assert q_path is not None
    assert "call0" in input_path.name and "call0" in q_path.name

    reset_layer_capture_state()
    capture = maybe_prepare_q_stage_capture(0, torch.tensor([1]), torch.ones(1, 1, 2))
    assert capture is not None and capture.call == 0


def test_layer_capture_rejects_invalid_layer_token(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0,bad")

    with pytest.raises(ValueError, match="VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS"):
        layer_capture_layer_enabled(0)


def test_layer_capture_rejects_negative_layer(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0,-1")

    with pytest.raises(ValueError, match="nonnegative"):
        layer_capture_layer_enabled(0)


def test_configured_layer_captures_decoder_and_attention_inputs(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "7")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    assert maybe_layer_capture_context(0) is None
    context = maybe_layer_capture_context(7, torch.tensor([1]), torch.tensor([2]))
    assert context is not None
    context.save_stage("before_attention", hidden_states=torch.ones(1, 2, 3))
    path = maybe_capture_attention_inputs(
        7, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
    )

    assert path is not None
    assert path.name == "rank0_layer7_call0_attention_inputs.pt"
    payload = torch.load(path, weights_only=False)
    assert payload["layer_idx"] == 7


def test_layer_capture_can_save_ffn_input_stage(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    context = maybe_layer_capture_context(
        0, torch.tensor([78], dtype=torch.int32), torch.tensor([588])
    )
    assert context is not None
    path = context.save_stage(
        "ffn_input",
        hidden_states=torch.ones(1, 4),
        pre_norm=torch.zeros(1, 4),
    )

    assert path is not None
    assert path.name == "rank0_layer0_call0_ffn_input.pt"
    payload = torch.load(path, weights_only=False)
    assert payload["stage"] == "ffn_input"
    assert payload["positions"].tolist() == [78]
    assert payload["input_ids"].tolist() == [588]
    assert payload["tensors"]["hidden_states"].tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert payload["tensors"]["pre_norm"].tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_layer_capture_call_indices_are_per_layer(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0,1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    decoder_calls = []
    attention_paths = []
    for layer_idx in (0, 1, 0, 1):
        context = maybe_layer_capture_context(layer_idx)
        assert context is not None
        decoder_calls.append((layer_idx, context.call))
        attention_paths.append(
            maybe_capture_attention_inputs(
                layer_idx,
                torch.tensor([1]),
                torch.ones(1, 2),
                torch.ones(1, 2),
            )
        )

    assert decoder_calls == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert [path.name for path in attention_paths if path is not None] == [
        "rank0_layer0_call0_attention_inputs.pt",
        "rank0_layer1_call0_attention_inputs.pt",
        "rank0_layer0_call1_attention_inputs.pt",
        "rank0_layer1_call1_attention_inputs.pt",
    ]

    reset_layer_capture_state()
    assert maybe_layer_capture_context(0).call == 0
    assert maybe_layer_capture_context(1).call == 0
    assert maybe_capture_attention_inputs(
        0, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
    ).name == "rank0_layer0_call0_attention_inputs.pt"
    assert maybe_capture_attention_inputs(
        1, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
    ).name == "rank0_layer1_call0_attention_inputs.pt"


def test_attention_output_capture_aligns_with_inputs_per_layer(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS", "0,1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    paths = []
    for layer_idx in (0, 1, 0, 1):
        positions = torch.tensor([layer_idx], dtype=torch.int32)
        maybe_capture_attention_inputs(
            layer_idx, positions, torch.ones(1, 2), torch.ones(1, 2)
        )
        paths.append(
            maybe_capture_attention_output(
                layer_idx,
                positions,
                torch.full((1, 3), float(layer_idx), dtype=torch.float32),
            )
        )

    assert [path.name for path in paths if path is not None] == [
        "rank0_layer0_call0_attention_output.pt",
        "rank0_layer1_call0_attention_output.pt",
        "rank0_layer0_call1_attention_output.pt",
        "rank0_layer1_call1_attention_output.pt",
    ]
    payload = torch.load(paths[1], weights_only=False)
    assert payload["stage"] == "attention_output"
    assert payload["layer_idx"] == 1
    assert payload["call"] == 0
    assert payload["positions"].device.type == "cpu"
    assert payload["output"].device.type == "cpu"
    assert payload["tensor_meta"]["output"] == {
        "shape": [1, 3],
        "dtype": "torch.float32",
        "stride": [3, 1],
    }

    reset_layer_capture_state()
    path = maybe_capture_attention_output(
        1, torch.tensor([1]), torch.ones(1, 3)
    )
    assert path is not None
    assert path.name == "rank0_layer1_call0_attention_output.pt"


def test_layer_capture_is_opt_in(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", raising=False)

    assert maybe_layer_capture_context(0, torch.tensor([1]), torch.tensor([2])) is None
    assert (
        maybe_capture_attention_inputs(
            0, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
        )
        is None
    )
    assert (
        maybe_capture_attention_output(0, torch.tensor([1]), torch.ones(1, 2))
        is None
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("full_tensors", [None, "0"])
def test_layer_capture_graph_filter_and_payload(
    monkeypatch, tmp_path: Path, full_tensors: str | None
):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    if full_tensors is None:
        monkeypatch.delenv(
            "VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS", raising=False
        )
    else:
        monkeypatch.setenv(
            "VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS", full_tensors
        )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_CALLS", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    assert maybe_layer_capture_context(1, torch.tensor([1]), torch.tensor([2])) is None
    selected = maybe_layer_capture_context(0, torch.tensor([3, 4]), torch.tensor([5, 6]))
    assert selected is not None
    selected.save_stage(
        "before_attention",
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        residual=torch.ones(2, 2, 3),
    )
    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=False)
    assert payload["rank"] == 0
    assert payload["layer_idx"] == 0
    assert payload["call"] == 0
    assert payload["stage"] == "before_attention"
    assert payload["positions"].device.type == "cpu"
    assert payload["tensors"]["hidden_states"].shape == (1, 2, 3)
    assert payload["tensor_meta"]["hidden_states"]["stride"] == [6, 3, 1]


def test_layer_capture_can_save_full_stage_tensors(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()
    context = maybe_layer_capture_context(0)
    assert context is not None
    tensors = {
        name: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + offset
        for offset, name in enumerate(
            ("hidden_states", "residual", "post_mix", "res_mix", "pre_norm")
        )
    }

    path = context.save_stage("before_attention", **tensors)

    assert path is not None
    payload = torch.load(path, weights_only=False)
    for name, tensor in tensors.items():
        torch.testing.assert_close(payload["tensors"][name], tensor)
        assert payload["tensor_meta"][name] == {
            "shape": [2, 2, 3],
            "dtype": "torch.float32",
            "stride": [6, 3, 1],
        }


def test_layer_capture_rejects_invalid_full_tensor_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS", "invalid")
    monkeypatch.delenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", raising=False)
    assert maybe_layer_capture_context(0) is None

    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    context = maybe_layer_capture_context(0)
    assert context is not None

    with pytest.raises(
        ValueError, match="VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS"
    ):
        context.save_stage("before_attention", hidden_states=torch.ones(2, 3))


def test_layer_capture_reads_nested_swa_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=2,
        seq_lens=torch.tensor([12]),
        prefill_seq_lens=torch.tensor([12]),
        prefill_gather_lens=torch.tensor([10]),
        query_start_loc_cpu=torch.tensor([0, 2]),
        token_to_req_indices=torch.tensor([0, 0]),
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"model.layers.0.attn.swa_cache": metadata}
        ),
    )

    context = maybe_layer_capture_context(
        0, torch.tensor([10, 11]), torch.tensor([20, 21])
    )
    assert context is not None
    context.save_stage("before_attention", hidden_states=torch.ones(2, 2, 3))

    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=False)
    attention = payload["attention_metadata"]
    assert attention["source_key"] == "model.layers.0.attn.swa_cache"
    assert attention["num_prefills"] == 1
    assert attention["num_prefill_tokens"] == 2
    assert attention["prefill_gather_lens"] == [10]


def test_layer_capture_fails_closed_during_graph_capture(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    reset_layer_capture_state()

    assert maybe_layer_capture_context(0, torch.tensor([1]), torch.tensor([2])) is None
    assert (
        maybe_capture_attention_inputs(
            0, torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2)
        )
        is None
    )
    assert (
        maybe_capture_attention_output(0, torch.tensor([1]), torch.ones(1, 2))
        is None
    )
    assert not list(tmp_path.iterdir())


def test_attention_input_capture_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_LAYER_CAPTURE_CALLS", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reset_layer_capture_state()

    positions = torch.tensor([3, 4], dtype=torch.int32)
    qr = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    kv = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    assert maybe_capture_attention_inputs(1, positions, qr, kv) is None
    monkeypatch.setenv("RANK", "1")
    assert maybe_capture_attention_inputs(0, positions, qr, kv) is None
    monkeypatch.setenv("RANK", "0")
    path = maybe_capture_attention_inputs(0, positions, qr, kv)

    assert path is not None
    assert path.name == "rank0_layer0_call0_attention_inputs.pt"
    payload = torch.load(path, weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["rank"] == 0
    assert payload["layer_idx"] == 0
    assert payload["call"] == 0
    assert payload["stage"] == "attention_inputs"
    assert payload["positions"].device.type == "cpu"
    assert payload["qr"].device.type == "cpu"
    assert payload["kv"].device.type == "cpu"
    assert payload["qr"].shape == (2, 6)
    assert payload["kv"].shape == (2, 4)
    assert payload["tensor_meta"]["qr"]["dtype"] == "torch.float32"
    assert payload["tensor_meta"]["kv"]["stride"] == [4, 1]
