from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_metax.models.deepseek_v4.ops import o_proj, o_proj_debug
from vllm_metax.models.deepseek_v4.ops.fused_inv_rope_quant import inv_rope


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [1, 3, 17])
def test_direct_bmm_dispatch_matches_grouped_reference(monkeypatch, num_tokens):
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_DIRECT_BMM", "1")
    grouped = torch.randn(num_tokens, 2, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(o_proj, "inv_rope", lambda *args, **kwargs: grouped)
    wo_a = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    wo_b = nn.Linear(6, 5, bias=False, dtype=torch.bfloat16)
    result = o_proj.deep_gemm_bf16_o_proj(
        torch.empty(num_tokens, 2, 4, dtype=torch.bfloat16),
        torch.zeros(num_tokens, dtype=torch.int64),
        torch.empty(1, 2),
        wo_a,
        wo_b,
        n_groups=2,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=3,
    )
    expected_z = torch.bmm(
        grouped.transpose(0, 1),
        wo_a.weight.view(2, 3, 4).transpose(1, 2),
    ).transpose(0, 1)
    expected = wo_b(expected_z.flatten(1))
    assert torch.equal(result, expected)


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


def test_torch_replay_has_bitwise_stage_match(tmp_path, monkeypatch):
    o, positions, cache, wo_a_weight, wo_b_weight, params = _inputs()
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_RANKS", "0")
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


def test_inv_rope_rejects_nonmatching_caller_storage():
    o = torch.empty((1, 2, 4), dtype=torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.int64)
    cache = torch.zeros((1, 2), dtype=torch.float32)
    bad = torch.empty((1, 1, 7), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="caller-owned inv_rope output"):
        inv_rope(o, positions, cache, n_groups=1, heads_per_group=2,
                 nope_dim=2, rope_dim=2, quant_group_size=2, out=bad)
