from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_metax.models.deepseek_v4.ops import o_proj
from vllm_metax.models.deepseek_v4.ops.o_proj_collective import (
    coalesce_wo_b_row_reductions,
)
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


class UnquantizedLinearMethod:
    def __init__(self):
        self.calls = 0

    def apply(self, _layer, input_, _bias):
        self.calls += 1
        return input_


class _NativeWoB(_FakeWoB):
    def __init__(self):
        super().__init__(torch.empty(0))
        self.weight = torch.ones(3, 2, dtype=torch.bfloat16)
        self.quant_method = UnquantizedLinearMethod()


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


def test_wo_b_row_local_outputs_use_one_collective_and_materialize_buffers():
    shared = torch.empty(1, 2, dtype=torch.bfloat16)
    layer = _FakeWoB(shared)

    def apply(_layer, input_, bias):
        assert bias is None
        shared.copy_(input_ + 3)
        return shared

    layer.quant_method.apply = apply
    inputs = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.bfloat16)
    reductions = []

    def all_reduce(value):
        reductions.append(value.clone())
        return value + 10

    actual = coalesce_wo_b_row_reductions(
        layer,
        [inputs[:1], inputs[1:]],
        all_reduce=all_reduce,
    )

    expected_local = inputs + 3
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0], expected_local)
    torch.testing.assert_close(actual, expected_local + 10)


def test_wo_b_row_local_outputs_support_grouped_reductions():
    shared = torch.empty(1, 2, dtype=torch.bfloat16)
    layer = _FakeWoB(shared)
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    reductions = []

    def apply(_layer, input_, bias):
        assert bias is None
        shared.copy_(input_ + 3)
        return shared

    layer.quant_method.apply = apply
    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        group_rows=3,
        all_reduce=lambda value: reductions.append(value.clone()) or value + 10,
    )

    assert [value.shape[0] for value in reductions] == [3, 3]
    torch.testing.assert_close(torch.cat(reductions), inputs + 3)
    torch.testing.assert_close(actual, inputs + 13)


def test_wo_b_native_serial_rows_reuse_output_workspace(monkeypatch):
    layer = _NativeWoB()
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    native_calls = []
    reductions = []

    def native_op(grouped_input, weight, output):
        native_calls.append((grouped_input.clone(), weight, output.data_ptr()))
        output.copy_(grouped_input[:, :1].expand_as(output))

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_serial_rows_out",
        native_op,
        raising=False,
    )

    first = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        all_reduce=lambda value: reductions.append(value.clone()) or value + 10,
    )
    second = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        all_reduce=lambda value: reductions.append(value.clone()) or value + 10,
    )

    assert layer.quant_method.calls == 0
    assert len(native_calls) == 2
    assert native_calls[0][2] == native_calls[1][2]
    torch.testing.assert_close(torch.cat(reductions), torch.cat([inputs[:, :1]] * 2).expand(-1, 3))
    torch.testing.assert_close(first, inputs[:, :1].expand(-1, 3) + 10)
    torch.testing.assert_close(second, first)


def test_wo_b_exact_grouped_rows_selects_grouped_op(monkeypatch):
    layer = _NativeWoB()
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    grouped_calls = []
    serial_calls = []
    reductions = []

    def grouped_op(grouped_input, weight, output):
        grouped_calls.append((grouped_input.clone(), weight, output.data_ptr()))
        output.copy_(grouped_input[:, :1].expand_as(output))

    def serial_op(_grouped_input, _weight, _output):
        serial_calls.append(True)

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_O_PROJ_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_oproj_grouped_rows_out",
        grouped_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_serial_rows_out",
        serial_op,
        raising=False,
    )

    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        group_rows=3,
        all_reduce=lambda value: reductions.append(value.clone()) or value + 10,
    )

    assert layer.quant_method.calls == 0
    assert serial_calls == []
    assert [call[0].shape[0] for call in grouped_calls] == [3, 3]
    assert grouped_calls[0][2] == grouped_calls[1][2]
    torch.testing.assert_close(torch.cat(reductions), inputs[:, :1].expand(-1, 3))
    torch.testing.assert_close(actual, inputs[:, :1].expand(-1, 3) + 10)


def test_wo_b_exact_row_list_skips_group_input_cat(monkeypatch):
    layer = _NativeWoB()
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    row_list_calls = []
    grouped_calls = []
    reductions = []

    def row_list_op(row_inputs, weight, output):
        row_list_calls.append(([row.clone() for row in row_inputs], weight))
        output.copy_(torch.cat(row_inputs, dim=0)[:, :1].expand_as(output))

    def grouped_op(_grouped_input, _weight, _output):
        grouped_calls.append(True)

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_O_PROJ_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_OPROJ_ROW_LIST", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_oproj_row_list_out",
        row_list_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_oproj_grouped_rows_out",
        grouped_op,
        raising=False,
    )

    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        group_rows=3,
        all_reduce=lambda value: reductions.append(value.clone()) or value + 10,
    )

    assert [len(call[0]) for call in row_list_calls] == [3, 3]
    assert grouped_calls == []
    assert [value.shape[0] for value in reductions] == [3, 3]
    torch.testing.assert_close(torch.cat(reductions), inputs[:, :1].expand(-1, 3))
    torch.testing.assert_close(actual, inputs[:, :1].expand(-1, 3) + 10)


def test_exact_oproj_grouped_kernel_keeps_row_serial_target_shape():
    source = (
        Path(__file__).parents[3] / "csrc" / "metax_sparse" / "gemm_fp32.cu"
    ).read_text(encoding="utf-8")
    grouped_start = source.index("void gemv_bf16_exact_grouped_rows_out")
    grouped_end = source.index(
        "void gemv_bf16_fp32_serial_rows_out", grouped_start
    )
    grouped_impl = source[grouped_start:grouped_end]

    assert "gemv_bf16_exact_grouped_rows_kernel<<<" in grouped_impl
    assert "if (o_proj_shape)" in grouped_impl
    assert "at::mm_out" in grouped_impl
    oproj_start = source.index("void gemv_bf16_exact_oproj_grouped_rows_out")
    oproj_end = source.index("void gemv_bf16_fp32_serial_rows_out", oproj_start)
    oproj_impl = source[oproj_start:oproj_end]
    assert "gemv_bf16_exact_grouped_rows_out(input, weight, out);" in oproj_impl


def test_wo_b_exact_grouped_rows_handles_singleton_tail(monkeypatch):
    layer = _NativeWoB()
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    grouped_shapes = []

    def grouped_op(grouped_input, _weight, output):
        grouped_shapes.append(tuple(grouped_input.shape))
        output.copy_(grouped_input[:, :1].expand_as(output))

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_O_PROJ_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_oproj_grouped_rows_out",
        grouped_op,
        raising=False,
    )

    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        group_rows=5,
        all_reduce=lambda value: value + 10,
    )

    assert grouped_shapes == [(5, 2), (1, 2)]
    torch.testing.assert_close(actual, inputs[:, :1].expand(-1, 3) + 10)


def test_wo_b_exact_grouped_rows_supports_single_row_groups(monkeypatch):
    layer = _NativeWoB()
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    grouped_shapes = []

    def grouped_op(grouped_input, _weight, output):
        grouped_shapes.append(tuple(grouped_input.shape))
        output.copy_(grouped_input[:, :1].expand_as(output))

    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_EXACT_GROUPED_O_PROJ_ROWS", "1")
    monkeypatch.setattr(
        torch.ops._metax_sparse_C,
        "gemv_bf16_exact_oproj_grouped_rows_out",
        grouped_op,
        raising=False,
    )

    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        group_rows=1,
        all_reduce=lambda value: value + 10,
    )

    assert grouped_shapes == [(1, 2)] * 6
    torch.testing.assert_close(actual, inputs[:, :1].expand(-1, 3) + 10)


def test_wo_b_native_serial_rows_reject_quantized_method(monkeypatch):
    layer = _FakeWoB(torch.zeros(1, 2, dtype=torch.bfloat16))
    layer.weight = torch.ones(2, 2, dtype=torch.bfloat16)
    monkeypatch.setenv("VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS", "1")

    with pytest.raises(RuntimeError, match="UnquantizedLinearMethod"):
        coalesce_wo_b_row_reductions(
            layer,
            [torch.zeros(1, 2, dtype=torch.bfloat16)] * 2,
            all_reduce=lambda value: value,
        )


def test_single_reduced_o_proj_group_can_be_returned_directly(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_RETURN_SINGLE_REDUCED_GROUP", "1")
    layer = _FakeWoB(torch.zeros(1, 2, dtype=torch.bfloat16))
    inputs = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    reduced = None

    def apply(_layer, input_, _bias):
        return input_ + 3

    def all_reduce(value):
        nonlocal reduced
        reduced = value + 10
        return reduced

    layer.quant_method.apply = apply
    actual = coalesce_wo_b_row_reductions(
        layer,
        list(inputs.split(1)),
        all_reduce=all_reduce,
    )

    assert actual is reduced
    torch.testing.assert_close(actual, inputs + 13)


@pytest.mark.parametrize("group_rows", [0, -1])
def test_wo_b_row_collective_rejects_invalid_group_rows(group_rows):
    with pytest.raises(ValueError, match="group_rows"):
        coalesce_wo_b_row_reductions(
            _FakeWoB(torch.zeros(1, 2)),
            [torch.zeros(1, 2)],
            group_rows=group_rows,
            all_reduce=lambda value: value,
        )


def test_wo_b_row_collective_rejects_contract_mismatch():
    layer = _FakeWoB(torch.zeros(1, 2))
    layer.reduce_results = False
    with pytest.raises(RuntimeError, match="RowParallelLinear contract"):
        coalesce_wo_b_row_reductions(
            layer,
            [torch.zeros(1, 2)],
            all_reduce=lambda value: value,
        )


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
