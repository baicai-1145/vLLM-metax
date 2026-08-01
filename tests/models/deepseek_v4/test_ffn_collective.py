from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_metax.models.deepseek_v4.ffn_collective import (
    coalesce_moe_row_reductions,
)


class _FakeQuantMethod:
    has_unpadded_output = False
    topk_indices_dtype = torch.int32


class _FakeRunner:
    def __init__(self, *, fused_output_is_reduced=False, sequence_parallel=False):
        self.moe_config = SimpleNamespace(
            is_sequence_parallel=sequence_parallel,
            skip_final_all_reduce=False,
            tp_size=4,
            ep_size=1,
            hidden_dim_unpadded=0,
        )
        self._fused_output_is_reduced_value = fused_output_is_reduced
        self._quant_method = _FakeQuantMethod()
        self.router = object()
        self.calls = []

    @property
    def _fused_output_is_reduced(self):
        return self._fused_output_is_reduced_value

    def apply_routed_input_transform(self, hidden_states):
        return hidden_states, hidden_states + 100

    def _maybe_pad_hidden_states(self, shared_input, hidden_states):
        return hidden_states, None, None

    def _encode_layer_name(self):
        return "model.layers.0.ffn.experts"

    def _forward_entry(
        self,
        hidden_states,
        router_logits,
        shared_input,
        input_ids,
        layer_name,
        hidden_dim_unpadded,
    ):
        self.calls.append(
            (
                hidden_states.clone(),
                router_logits.clone(),
                input_ids.clone(),
                layer_name,
                hidden_dim_unpadded,
            )
        )
        return shared_input + 1000, hidden_states + router_logits

    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        return shared_output, fused_output * 2

    def apply_routed_output_transform(self, fused_output):
        return fused_output + 10


class _GroupedRouter:
    def __init__(self):
        self.calls = []

    def select_experts(self, *, hidden_states, router_logits, **kwargs):
        self.calls.append((hidden_states.clone(), router_logits.clone(), kwargs))
        return router_logits[:, :1], torch.zeros(
            hidden_states.shape[0], 1, dtype=torch.int32
        )


class _GroupedRoutedExperts:
    def __init__(self):
        self.quant_method = SimpleNamespace(is_monolithic=False)
        self.calls = []

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_modular(self, *, x, topk_weights, topk_ids, **kwargs):
        self.calls.append((x.clone(), topk_weights.clone(), topk_ids.clone(), kwargs))
        return x + topk_weights


class _GroupedSharedExperts:
    def __init__(self):
        self.calls = []
        self._output = None

    def _determine_shared_experts_order(self, shared_input):
        del shared_input
        return 1

    def __call__(self, shared_input, order):
        self.calls.append((shared_input.clone(), order))
        self._output = shared_input * 3

    @property
    def output(self):
        output = self._output
        self._output = None
        return output


class _GroupedFakeRunner(_FakeRunner):
    def __init__(self, *, shared_experts=False):
        super().__init__()
        self.router = _GroupedRouter()
        self.routed_experts = _GroupedRoutedExperts()
        self._shared_experts = _GroupedSharedExperts() if shared_experts else None

    def _maybe_sync_shared_experts_stream(self, shared_input):
        del shared_input


class _CloneCounter(TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.clone.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


def test_coalesces_row_local_moe_outputs_before_one_reduce():
    runner = _FakeRunner()
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    input_ids = torch.tensor([7, 8])
    reduce_calls = []

    def router_logits(row, row_ids):
        return row + row_ids[:, None]

    def all_reduce(value):
        reduce_calls.append(value.clone())
        return value + 10_000

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        input_ids,
        router_logits,
        all_reduce=all_reduce,
    )

    expected_local = torch.cat(
        [
            (hidden_states[i : i + 1] + 1100)
            + 2
            * (
                hidden_states[i : i + 1]
                + hidden_states[i : i + 1]
                + input_ids[i : i + 1, None]
            )
            + 10
            for i in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(reduce_calls[0], expected_local)
    torch.testing.assert_close(actual, expected_local + 10_000)
    assert len(reduce_calls) == 1
    assert [call[2].tolist() for call in runner.calls] == [[7], [8]]
    assert runner.moe_config.skip_final_all_reduce is False


def test_groups_row_local_moe_outputs_without_batching_row_arithmetic():
    runner = _FakeRunner()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    input_ids = torch.arange(6)
    reductions = []

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        input_ids,
        lambda row, row_ids: row + row_ids[:, None],
        group_rows=3,
        all_reduce=lambda value: reductions.append(value.clone()) or value + 100,
    )

    assert [value.shape[0] for value in reductions] == [3, 3]
    assert [call[2].item() for call in runner.calls] == list(range(6))
    torch.testing.assert_close(actual, torch.cat(reductions) + 100)


def test_groups_routed_expert_kernel_while_preserving_row_local_routing():
    runner = _GroupedFakeRunner()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    input_ids = torch.arange(6)
    reductions = []

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        input_ids,
        lambda row, row_ids: row + row_ids[:, None],
        group_rows=6,
        group_routed_experts=True,
        all_reduce=lambda value: reductions.append(value.clone()) or value + 100,
    )

    assert len(runner.routed_experts.calls) == 1
    assert runner.routed_experts.calls[0][0].shape == (6, 2)
    assert len(runner.router.calls) == 6
    assert all(call[0].shape == (1, 2) for call in runner.router.calls)
    assert len(reductions) == 1
    torch.testing.assert_close(actual, reductions[0] + 100)


def test_groups_router_after_row_local_logits():
    runner = _GroupedFakeRunner()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    input_ids = torch.arange(6)

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        input_ids,
        lambda row, row_ids: row + row_ids[:, None],
        group_rows=6,
        group_routed_experts=True,
        group_router=True,
        all_reduce=lambda value: value,
    )

    assert len(runner.router.calls) == 1
    assert runner.router.calls[0][0].shape == (6, 2)
    assert len(runner.routed_experts.calls) == 1
    assert actual.shape == hidden_states.shape


def test_single_reduced_ffn_group_can_be_returned_directly(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_RETURN_SINGLE_REDUCED_GROUP", "1")
    runner = _GroupedFakeRunner()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    reduced = None

    def all_reduce(value):
        nonlocal reduced
        reduced = value + 100
        return reduced

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        torch.arange(6),
        lambda row, row_ids: row + row_ids[:, None],
        group_rows=6,
        group_routed_experts=True,
        all_reduce=all_reduce,
    )

    assert actual is reduced


def test_grouped_routed_experts_keep_shared_experts_row_local():
    runner = _GroupedFakeRunner(shared_experts=True)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    input_ids = torch.arange(6)

    actual = coalesce_moe_row_reductions(
        runner,
        hidden_states,
        input_ids,
        lambda row, row_ids: row + row_ids[:, None],
        group_rows=6,
        group_routed_experts=True,
        all_reduce=lambda value: value,
    )

    router_weight = hidden_states[:, :1] + input_ids[:, None]
    expected = (hidden_states + 100) * 3 + 2 * (hidden_states + router_weight) + 10
    torch.testing.assert_close(actual, expected)
    assert len(runner._shared_experts.calls) == 6
    assert all(call[0].shape == (1, 2) for call in runner._shared_experts.calls)


@pytest.mark.parametrize("rows", [2, 5, 6])
def test_grouped_routed_experts_can_retain_row_views_until_cat(monkeypatch, rows):
    hidden_states = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    input_ids = torch.arange(rows)

    def run(candidate: bool):
        runner = _GroupedFakeRunner(shared_experts=True)
        if candidate:
            monkeypatch.setenv("VLLM_METAX_DSV4_FFN_RETAIN_ROW_OUTPUT_VIEWS", "1")
        else:
            monkeypatch.delenv(
                "VLLM_METAX_DSV4_FFN_RETAIN_ROW_OUTPUT_VIEWS", raising=False
            )
        counter = _CloneCounter()
        with counter:
            output = coalesce_moe_row_reductions(
                runner,
                hidden_states,
                input_ids,
                lambda row, row_ids: row + row_ids[:, None],
                group_rows=rows,
                group_routed_experts=True,
                all_reduce=lambda value: value,
            )
        return output, counter.count

    control, control_clones = run(False)
    candidate, candidate_clones = run(True)

    torch.testing.assert_close(candidate, control)
    assert candidate_clones + hidden_states.shape[0] == control_clones


@pytest.mark.parametrize("rows", [2, 5, 6])
def test_grouped_routed_experts_can_retain_transferred_shared_outputs(
    monkeypatch, rows
):
    hidden_states = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    input_ids = torch.arange(rows)
    monkeypatch.setenv("VLLM_METAX_DSV4_FFN_RETAIN_ROW_OUTPUT_VIEWS", "1")

    def run(candidate: bool):
        runner = _GroupedFakeRunner(shared_experts=True)
        if candidate:
            monkeypatch.setenv("VLLM_METAX_DSV4_FFN_RETAIN_SHARED_OUTPUTS", "1")
        else:
            monkeypatch.delenv(
                "VLLM_METAX_DSV4_FFN_RETAIN_SHARED_OUTPUTS", raising=False
            )
        counter = _CloneCounter()
        with counter:
            output = coalesce_moe_row_reductions(
                runner,
                hidden_states,
                input_ids,
                lambda row, row_ids: row + row_ids[:, None],
                group_rows=rows,
                group_routed_experts=True,
                all_reduce=lambda value: value,
            )
        return output, counter.count

    control, control_clones = run(False)
    candidate, candidate_clones = run(True)

    torch.testing.assert_close(candidate, control)
    assert candidate_clones + hidden_states.shape[0] == control_clones


@pytest.mark.parametrize("group_rows", [0, -1])
def test_rejects_invalid_moe_reduction_group_rows(group_rows):
    with pytest.raises(ValueError, match="group_rows"):
        coalesce_moe_row_reductions(
            _FakeRunner(),
            torch.ones(2, 4),
            torch.tensor([1, 2]),
            lambda row, row_ids: row,
            group_rows=group_rows,
            all_reduce=lambda value: value,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fused_output_is_reduced": True}, "already reduces"),
        ({"sequence_parallel": True}, "sequence-parallel"),
    ],
)
def test_rejects_unsupported_reduction_ownership(kwargs, message):
    runner = _FakeRunner(**kwargs)
    with pytest.raises(RuntimeError, match=message):
        coalesce_moe_row_reductions(
            runner,
            torch.ones(2, 4),
            torch.tensor([1, 2]),
            lambda row, row_ids: row,
            all_reduce=lambda value: value,
        )


def test_rejects_runner_with_final_reduce_already_disabled():
    runner = _FakeRunner()
    runner.moe_config.skip_final_all_reduce = True
    with pytest.raises(RuntimeError, match="skip_final_all_reduce"):
        coalesce_moe_row_reductions(
            runner,
            torch.ones(2, 4),
            torch.tensor([1, 2]),
            lambda row, row_ids: row,
            all_reduce=lambda value: value,
        )


def test_rejects_zero_expert_post_reduce_semantics():
    runner = _FakeRunner()
    runner.router = type("ZeroExpertRouter", (), {})()
    with pytest.raises(RuntimeError, match="zero experts"):
        coalesce_moe_row_reductions(
            runner,
            torch.ones(2, 4),
            torch.tensor([1, 2]),
            lambda row, row_ids: row,
            all_reduce=lambda value: value,
        )
