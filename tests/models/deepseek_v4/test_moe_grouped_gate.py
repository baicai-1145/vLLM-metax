from types import SimpleNamespace

import pytest
import torch

from vllm_metax.models.deepseek_v4.moe_grouped_gate import (
    _stage_capture_setter,
    compare_grouped_routed_experts,
)


class _Router:
    def select_experts(self, *, hidden_states, router_logits, **kwargs):
        del hidden_states, kwargs
        return router_logits[:, :2], torch.tensor(
            [[0, 1]], dtype=torch.int32
        ).expand(router_logits.shape[0], -1)


class _RoutedExperts:
    def __init__(self, *, grouped_bias=0.0):
        self.grouped_bias = grouped_bias
        self.calls = []

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_modular(self, *, x, topk_weights, topk_ids, **kwargs):
        del topk_ids, kwargs
        self.calls.append(x.shape[0])
        bias = self.grouped_bias if x.shape[0] > 1 else 0.0
        return x + topk_weights.sum(dim=1, keepdim=True) + bias


class _Runner:
    def __init__(self, *, grouped_bias=0.0):
        self.router = _Router()
        self.routed_experts = _RoutedExperts(grouped_bias=grouped_bias)
        self._quant_method = SimpleNamespace(topk_indices_dtype=torch.int32)

    def apply_routed_input_transform(self, value):
        return value, value

    def _maybe_pad_hidden_states(self, shared_input, value):
        del shared_input
        return value, None, None


class _InternalGateRunner(_Runner):
    def __init__(self):
        super().__init__()
        self.gate = lambda row: (row[:, :2] + 3, None)


def _router_logits(row, row_ids):
    return torch.cat((row[:, :1] + row_ids[:, None], row[:, 1:2]), dim=1)


def test_grouped_routed_expert_gate_matches_independent_rows_bitwise():
    runner = _Runner()
    result = compare_grouped_routed_experts(
        runner,
        torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        torch.arange(6),
        _router_logits,
        repeats=5,
    )

    assert result["mismatch_count"] == 0
    assert result["repeat_mismatch_counts"] == [0, 0, 0, 0]
    assert result["router_id_mismatch_count"] == 0
    assert result["router_weight_mismatch_count"] == 0
    assert runner.routed_experts.calls == [1] * 6 + [6] * 5


def test_grouped_routed_expert_gate_is_red_capable():
    result = compare_grouped_routed_experts(
        _Runner(grouped_bias=1.0),
        torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        torch.arange(6),
        _router_logits,
    )

    assert result["mismatch_count"] == 24
    assert result["max_abs"] == 1.0


def test_grouped_routed_expert_gate_resolves_internal_router_logits():
    runner = _InternalGateRunner()

    result = compare_grouped_routed_experts(
        runner,
        torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        torch.arange(6),
        lambda row, row_ids: row,
    )

    assert result["mismatch_count"] == 0
    assert result["topk_weights"].shape == (6, 2)


@pytest.mark.parametrize("repeats", [0, -1])
def test_grouped_routed_expert_gate_rejects_invalid_repeats(repeats):
    with pytest.raises(ValueError, match="repeats"):
        compare_grouped_routed_experts(
            _Runner(),
            torch.ones(6, 4),
            torch.arange(6),
            _router_logits,
            repeats=repeats,
        )


_CAPTURE_CALLBACK = None


def set_dspark_moe_stage_capture(callback):
    global _CAPTURE_CALLBACK
    _CAPTURE_CALLBACK = callback


class _RuntimeExperts:
    pass


def test_stage_capture_setter_uses_runtime_expert_module():
    experts = _RuntimeExperts()
    runner = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=experts)
            )
        )
    )

    setter, module_name = _stage_capture_setter(runner)
    callback = lambda name, value: None
    setter(callback)

    assert module_name == __name__
    assert _CAPTURE_CALLBACK is callback
    setter(None)
