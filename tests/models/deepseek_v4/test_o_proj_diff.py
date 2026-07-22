from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_metax.models.deepseek_v4.ops import o_proj
from vllm_metax.models.deepseek_v4.ops import o_proj_debug


def _inputs():
    torch.manual_seed(7)
    params = {
        "n_groups": 2,
        "heads_per_group": 2,
        "nope_dim": 2,
        "rope_dim": 2,
        "o_lora_rank": 3,
    }
    o = torch.randn(3, 4, 4, dtype=torch.bfloat16)
    positions = torch.tensor([4, 2, 4], dtype=torch.int64)
    cache = torch.randn(8, 2, dtype=torch.float32)
    wo_a = torch.randn(6, 8, dtype=torch.bfloat16)
    wo_b = torch.randn(5, 6, dtype=torch.bfloat16)
    return o, positions, cache, wo_a, wo_b, params


class _FakeQuantMethod:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def apply(self, layer, input_, bias):
        self.calls.append((layer, input_, bias))
        return self.value


class _FakeWoB:
    input_is_parallel = True
    reduce_results = True
    return_bias = False
    skip_bias_add = False
    tp_size = 4
    bias = None

    def __init__(self, value):
        self.value = value
        self.weight = torch.ones(2, 2, dtype=torch.bfloat16)
        self.quant_method = _FakeQuantMethod(value)
        self.calls = 0

    def __call__(self, input_):
        self.calls += 1
        return self.value


def test_wo_b_stage_capture_disabled_keeps_module_call(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_WO_B_STAGES", raising=False)
    wo_a = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    wo_b = _FakeWoB(torch.zeros(3, 5, dtype=torch.bfloat16))
    o = torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(o_proj, "inv_rope", lambda *args, **kwargs: o)
    monkeypatch.setattr(o_proj, "bf16_einsum", lambda *args: None)

    result = o_proj.deep_gemm_bf16_o_proj(
        o,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(8, 2),
        wo_a,
        wo_b,
        n_groups=2,
        heads_per_group=2,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=3,
    )
    assert wo_b.calls == 1
    assert result.shape == (3, 5)


def test_wo_b_stage_capture_enabled_calls_native_stages_and_captures_both(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_WO_B_STAGES", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()

    local = torch.full((3, 2), 2, dtype=torch.bfloat16)
    wo_b = _FakeWoB(local)
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed,
        "tensor_model_parallel_all_reduce",
        lambda value: value + 3,
    )
    input_ = torch.zeros(3, 2, dtype=torch.bfloat16)
    captured_local, reduced = o_proj_debug.apply_wo_b_with_stages(wo_b, input_)
    assert len(wo_b.quant_method.calls) == 1
    assert wo_b.quant_method.calls[0][1] is input_
    assert wo_b.quant_method.calls[0][2] is None
    torch.testing.assert_close(captured_local, local)
    torch.testing.assert_close(reduced, local + 3)

    wo_a = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    o_proj_debug.maybe_capture_o_proj(
        o=torch.zeros(3, 1, 2, dtype=torch.bfloat16),
        positions=torch.arange(3),
        cos_sin_cache=torch.zeros(3, 1),
        wo_a=wo_a,
        wo_b=wo_b,
        o_bf16=torch.zeros(3, 1, 2, dtype=torch.bfloat16),
        z=torch.zeros(3, 1, 2, dtype=torch.bfloat16),
        output=reduced,
        wo_b_local=captured_local,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=0,
        o_lora_rank=2,
    )
    payload = o_proj_debug.load_capture(next(tmp_path.glob("*.pt")))
    torch.testing.assert_close(payload["wo_b_local"], local)
    torch.testing.assert_close(payload["output"], local + 3)
    assert "wo_b_local" in payload["intermediate_meta"]


def test_wo_b_stage_capture_rejects_contract_mismatch():
    layer = _FakeWoB(torch.zeros(1))
    layer.input_is_parallel = False
    with pytest.raises(RuntimeError, match="RowParallelLinear contract"):
        o_proj_debug.apply_wo_b_with_stages(layer, torch.zeros(1))


def test_torch_replay_has_bitwise_stage_match(tmp_path, monkeypatch):
    o, positions, cache, wo_a_weight, wo_b_weight, params = _inputs()
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_RANKS", "0")
    monkeypatch.delenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_LAYERS", raising=False)
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_TOKEN_COUNTS", raising=False
    )
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES", raising=False
    )
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()
    wo_a = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    wo_b = nn.Linear(6, 5, bias=False, dtype=torch.bfloat16)
    wo_a.weight.data.copy_(wo_a_weight)
    wo_b.weight.data.copy_(wo_b_weight)
    trace = o_proj_debug.torch_o_proj_trace(o, positions, cache, wo_a_weight, wo_b_weight, **params)
    o_proj_debug.maybe_capture_o_proj(
        o=o, positions=positions, cos_sin_cache=cache, wo_a=wo_a, wo_b=wo_b,
        o_bf16=trace["o_bf16"], z=trace["z"], output=trace["output"],
        layer_idx=0, chunk_index=2, **params
    )
    path = next(tmp_path.glob("*.pt"))
    payload = o_proj_debug.load_capture(path)
    assert payload["layer_idx"] == 0
    assert payload["chunk_index"] == 2
    result = o_proj_debug.replay_capture(path)
    assert result["passed"]
    assert all(item["equal"] for item in result["stages"].values())


def test_capture_disabled_is_true_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", raising=False)
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: (_ for _ in ()).throw(AssertionError()))
    o_proj_debug.reset_o_proj_capture_state()
    sentinel = object()
    class Module:
        @property
        def weight(self):
            raise AssertionError("weight must not be inspected")
    o_proj_debug.maybe_capture_o_proj(
        o=torch.empty(1), positions=torch.empty(1), cos_sin_cache=torch.empty(1),
        wo_a=Module(), wo_b=Module(), o_bf16=torch.empty(1), z=torch.empty(1),
        output=sentinel, n_groups=1, heads_per_group=1, nope_dim=0, rope_dim=0,
        o_lora_rank=1,
    )
    assert not list(tmp_path.iterdir())


def test_capture_skips_cuda_graph_without_sync_or_files(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: True)
    monkeypatch.setattr(o_proj_debug, "_synchronize_capture_stream", lambda _: (_ for _ in ()).throw(AssertionError()))
    o_proj_debug.reset_o_proj_capture_state()
    o_proj_debug.maybe_capture_o_proj(
        o=torch.empty(1), positions=torch.empty(1), cos_sin_cache=torch.empty(1),
        wo_a=nn.Linear(1, 1, bias=False), wo_b=nn.Linear(1, 1, bias=False),
        o_bf16=torch.empty(1), z=torch.empty(1), output=torch.empty(1),
        n_groups=1, heads_per_group=1, nope_dim=0, rope_dim=0, o_lora_rank=1,
    )
    assert not list(tmp_path.iterdir())


def test_layer_filter_skips_before_inspection_and_call_accounting(
    monkeypatch, tmp_path
):
    o, positions, cache, wo_a_weight, wo_b_weight, params = _inputs()
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_LAYERS", "0")
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()
    trace = o_proj_debug.torch_o_proj_trace(
        o, positions, cache, wo_a_weight, wo_b_weight, **params
    )

    class UninspectableModule:
        @property
        def weight(self):
            raise AssertionError("filtered layers must not inspect weights")

    o_proj_debug.maybe_capture_o_proj(
        o=o,
        positions=positions,
        cos_sin_cache=cache,
        wo_a=UninspectableModule(),
        wo_b=UninspectableModule(),
        o_bf16=trace["o_bf16"],
        z=trace["z"],
        output=trace["output"],
        layer_idx=1,
        chunk_index=0,
        **params,
    )
    assert not list(tmp_path.iterdir())

    wo_a = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    wo_b = nn.Linear(6, 5, bias=False, dtype=torch.bfloat16)
    wo_a.weight.data.copy_(wo_a_weight)
    wo_b.weight.data.copy_(wo_b_weight)
    for chunk_index in (0, 1):
        o_proj_debug.maybe_capture_o_proj(
            o=o,
            positions=positions,
            cos_sin_cache=cache,
            wo_a=wo_a,
            wo_b=wo_b,
            o_bf16=trace["o_bf16"],
            z=trace["z"],
            output=trace["output"],
            layer_idx=0,
            chunk_index=chunk_index,
            **params,
        )

    paths = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in paths] == ["rank0_call0.pt", "rank0_call1.pt"]
    assert [o_proj_debug.load_capture(path)["chunk_index"] for path in paths] == [
        0,
        1,
    ]


def test_token_count_filter_matches_selected_lengths_before_accounting(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_TOKEN_COUNTS", "121,256"
    )
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES", "invalid"
    )
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()
    cache = torch.zeros(256, 1)

    class UninspectableModule:
        @property
        def weight(self):
            raise AssertionError("filtered token counts must not inspect weights")

    def capture(count, wo_a, wo_b):
        value = torch.zeros(count, 1, 1)
        o_proj_debug.maybe_capture_o_proj(
            o=value,
            positions=torch.arange(count),
            cos_sin_cache=cache,
            wo_a=wo_a,
            wo_b=wo_b,
            o_bf16=value,
            z=value,
            output=value,
            n_groups=1,
            heads_per_group=1,
            nope_dim=1,
            rope_dim=0,
            o_lora_rank=1,
            layer_idx=0,
            chunk_index=0,
        )

    capture(3, UninspectableModule(), UninspectableModule())
    assert not list(tmp_path.iterdir())
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES", raising=False
    )

    wo_a = nn.Linear(1, 1, bias=False)
    wo_b = nn.Linear(1, 1, bias=False)
    capture(121, wo_a, wo_b)
    capture(256, wo_a, wo_b)

    paths = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in paths] == ["rank0_call0.pt", "rank0_call1.pt"]
    assert [
        o_proj_debug.load_capture(path)["positions"].numel() for path in paths
    ] == [121, 256]


@pytest.mark.parametrize("spec", ["0", "-1", "121,bad", "121,"])
def test_invalid_token_count_filter_fails_closed(monkeypatch, tmp_path, spec):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_TOKEN_COUNTS", spec
    )
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)

    with pytest.raises(
        ValueError,
        match="VLLM_METAX_DSV4_O_PROJ_CAPTURE_TOKEN_COUNTS",
    ):
        o_proj_debug.maybe_capture_o_proj(
            o=torch.empty(1),
            positions=torch.tensor([0]),
            cos_sin_cache=torch.empty(1),
            wo_a=object(),
            wo_b=object(),
            o_bf16=torch.empty(1),
            z=torch.empty(1),
            output=torch.empty(1),
            n_groups=1,
            heads_per_group=1,
            nope_dim=0,
            rope_dim=0,
            o_lora_rank=1,
            layer_idx=0,
            chunk_index=0,
        )


def test_position_range_requires_exact_contiguous_sequence(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES", "512:632"
    )
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_TOKEN_COUNTS", "121"
    )
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()
    o = torch.zeros(121, 1, 2)
    positions = torch.arange(512, 633, dtype=torch.int64)
    cache = torch.zeros(633, 1)
    intermediate = torch.zeros(121, 1, 1)
    native_arange = torch.arange
    arange_calls = []

    def cpu_arange(*args, **kwargs):
        arange_calls.append(kwargs.copy())
        assert "device" not in kwargs
        return native_arange(*args, **kwargs)

    monkeypatch.setattr(o_proj_debug.torch, "arange", cpu_arange)

    class UninspectableModule:
        @property
        def weight(self):
            raise AssertionError("filtered positions must not inspect weights")

    noncontiguous = positions.clone()
    noncontiguous[60] += 1
    o_proj_debug.maybe_capture_o_proj(
        o=o,
        positions=noncontiguous,
        cos_sin_cache=cache,
        wo_a=UninspectableModule(),
        wo_b=UninspectableModule(),
        o_bf16=o,
        z=intermediate,
        output=o,
        n_groups=1,
        heads_per_group=1,
        nope_dim=1,
        rope_dim=1,
        o_lora_rank=1,
        layer_idx=0,
        chunk_index=2,
    )
    assert not list(tmp_path.iterdir())

    wo_a = nn.Linear(2, 1, bias=False)
    wo_b = nn.Linear(1, 2, bias=False)
    o_proj_debug.maybe_capture_o_proj(
        o=o,
        positions=positions,
        cos_sin_cache=cache,
        wo_a=wo_a,
        wo_b=wo_b,
        o_bf16=o,
        z=intermediate,
        output=o,
        n_groups=1,
        heads_per_group=1,
        nope_dim=1,
        rope_dim=1,
        o_lora_rank=1,
        layer_idx=0,
        chunk_index=2,
    )

    path = tmp_path / "rank0_call0.pt"
    assert path.exists()
    assert len(arange_calls) == 2
    torch.testing.assert_close(
        o_proj_debug.load_capture(path)["positions"], positions
    )


@pytest.mark.parametrize("spec", ["512", "632:512", "-1:2", "start:end"])
def test_invalid_position_range_fails_closed(monkeypatch, tmp_path, spec):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES", spec
    )
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)

    with pytest.raises(
        ValueError,
        match="VLLM_METAX_DSV4_O_PROJ_CAPTURE_POSITION_RANGES",
    ):
        o_proj_debug.maybe_capture_o_proj(
            o=torch.empty(1),
            positions=torch.tensor([0]),
            cos_sin_cache=torch.empty(1),
            wo_a=object(),
            wo_b=object(),
            o_bf16=torch.empty(1),
            z=torch.empty(1),
            output=torch.empty(1),
            n_groups=1,
            heads_per_group=1,
            nope_dim=0,
            rope_dim=0,
            o_lora_rank=1,
            layer_idx=0,
            chunk_index=0,
        )


def test_corrupt_capture_is_red(tmp_path, monkeypatch):
    o, positions, cache, wo_a_weight, wo_b_weight, params = _inputs()
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(o_proj_debug, "_rank", lambda: "0")
    monkeypatch.setattr(o_proj_debug, "_is_cuda_graph_capturing", lambda: False)
    o_proj_debug.reset_o_proj_capture_state()
    wo_a = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    wo_b = nn.Linear(6, 5, bias=False, dtype=torch.bfloat16)
    wo_a.weight.data.copy_(wo_a_weight)
    wo_b.weight.data.copy_(wo_b_weight)
    trace = o_proj_debug.torch_o_proj_trace(o, positions, cache, wo_a_weight, wo_b_weight, **params)
    o_proj_debug.maybe_capture_o_proj(
        o=o, positions=positions, cos_sin_cache=cache, wo_a=wo_a, wo_b=wo_b,
        o_bf16=trace["o_bf16"], z=trace["z"], output=trace["output"], **params
    )
    path = next(tmp_path.glob("*.pt"))
    payload = o_proj_debug.load_capture(path)
    payload["z"] = payload["z"].clone()
    payload["z"].flatten()[0] += 1
    corrupt = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt)
    result = o_proj_debug.replay_capture(corrupt)
    assert not result["passed"]
    assert not result["stages"]["z"]["equal"]


def test_cli_emits_json_summary():
    cli = runpy.run_path(str(Path("tools/debug/diff_deepseek_v4_o_proj.py")))
    assert "run_diff" in cli
