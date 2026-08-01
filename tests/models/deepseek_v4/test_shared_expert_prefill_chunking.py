import pytest
import torch

from vllm_metax.models.deepseek_v4.model import DeepseekV4MLP


class _ShapeSensitiveGateUp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value):
        self.calls.append(value.shape[0])
        return value + value.new_tensor(float(value.shape[0])), None


class _ShapeSensitiveDown(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value):
        self.calls.append(value.shape[0])
        return value + value.new_tensor(float(value.shape[0])), None


def _mlp(*, enabled: bool, chunk_size: int = 16, shared: bool = True):
    mlp = DeepseekV4MLP.__new__(DeepseekV4MLP)
    torch.nn.Module.__init__(mlp)
    mlp.layer_idx = 0
    mlp._capture_shared_stages = shared
    mlp._prefill_gemm_chunking_enabled = enabled and shared
    mlp._prefill_gemm_chunk_size = chunk_size
    mlp.gate_up_proj = _ShapeSensitiveGateUp()
    mlp.act_fn = torch.nn.Identity()
    mlp.down_proj = _ShapeSensitiveDown()
    return mlp


def test_shared_prefill_chunking_aligns_prefix_overlap_shapes():
    hidden_states = torch.arange(633 * 2, dtype=torch.float32).reshape(633, 2)

    direct = _mlp(enabled=False)
    direct_full = direct(hidden_states)
    direct_overlap = direct(hidden_states[512:])
    assert direct.gate_up_proj.calls == [633, 121]
    assert direct.down_proj.calls == [633, 121]
    assert not torch.equal(direct_full[512:], direct_overlap)

    chunked = _mlp(enabled=True)
    chunked_full = chunked(hidden_states)
    chunked_overlap = chunked(hidden_states[512:])
    expected_full = [16] * 39 + [9]
    expected_overlap = [16] * 7 + [9]
    assert chunked.gate_up_proj.calls == expected_full + expected_overlap
    assert chunked.down_proj.calls == expected_full + expected_overlap
    torch.testing.assert_close(chunked_full[512:], chunked_overlap)


@pytest.mark.parametrize(
    ("enabled", "tokens"), [(False, 633), (True, 16), (True, 1)]
)
def test_shared_prefill_chunking_disabled_small_or_decode_calls_once(enabled, tokens):
    mlp = _mlp(enabled=enabled)
    mlp(torch.zeros(tokens, 2))
    assert mlp.gate_up_proj.calls == [tokens]
    assert mlp.down_proj.calls == [tokens]


def test_prefill_chunking_is_limited_to_shared_expert():
    mlp = _mlp(enabled=True, shared=False)
    mlp(torch.zeros(33, 2))
    assert mlp.gate_up_proj.calls == [33]
    assert mlp.down_proj.calls == [33]
