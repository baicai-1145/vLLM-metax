"""Small, deterministic differential harness for sparse MLA prefill.

The public wrapper is intentionally exercised rather than the private torch
reference.  Native prefill is only available on a MetaX CUDA-compatible
device, so native-path tests skip explicitly when that device is absent.
"""

import math

import pytest

torch = pytest.importorskip("torch")
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

pytest.importorskip("flash_mla")
from vllm_metax.v1.attention.ops.flashmla import (  # noqa: E402
    flash_mla_sparse_fwd_wrapper,
)
from vllm_metax.models.deepseek_v4.ops import sparse_mla_debug  # noqa: E402


LOG2E = math.log2(math.e)
LN2 = math.log(2.0)


def _oracle_device() -> torch.device:
    """Use the native device when available, otherwise keep the oracle runnable."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def metax_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("MetaX CUDA-compatible device unavailable")
    return torch.device("cuda")


def _case(
    *,
    device: torch.device,
    compression_ratio: int = 1,
    tokens: int,
    kv_tokens: int,
    heads: int,
    head_dim: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if compression_ratio not in (1, 4, 128):
        raise ValueError(f"unsupported synthetic compression ratio: {compression_ratio}")
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (tokens, heads, head_dim),
        device=device,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    kv = torch.randn(
        (kv_tokens, 1, head_dim),
        device=device,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    # Keep valid and invalid entries in every non-trivial row.  Padding after
    # each row's length is represented by -1, as in the production metadata.
    indices = (
        torch.arange(
            tokens * topk, dtype=torch.int32, device=device
        ).reshape(tokens, 1, topk)
        * 3
        + 1
    ).remainder(kv_tokens)
    # Model the compressed pool as the first ceil(kv_tokens / ratio) entries;
    # remaining entries represent the uncompressed/SWA pool.
    compressed_tokens = max(1, (kv_tokens + compression_ratio - 1) // compression_ratio)
    compressed_width = topk // 2
    if compressed_width:
        indices[:, 0, :compressed_width] %= compressed_tokens
    lengths = torch.tensor(
        [max(1, topk - (row % 3)) for row in range(tokens)],
        dtype=torch.int32,
        device=device,
    )
    for row, length in enumerate(lengths.tolist()):
        indices[row, 0, length:] = -1
    if topk > 1:
        indices[0, 0, -1] = -1
        indices[-1, 0, 0] = kv_tokens
    return q, kv, indices, lengths


def _oracle(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference using only dense torch primitives."""
    assert q.ndim == kv.ndim == 3
    tokens, heads, head_dim = q.shape
    kv_tokens = kv.shape[0]
    if attn_sink is not None:
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (heads,)
        assert attn_sink.device == q.device
    row_indices = indices[:, 0, :].to(torch.int64)
    valid = (row_indices >= 0) & (row_indices < kv_tokens)
    if topk_length is not None:
        valid &= (
            torch.arange(row_indices.shape[1], device=row_indices.device)
            < topk_length[:, None]
        )

    safe_indices = row_indices.masked_fill(~valid, 0)
    gathered = kv[:, 0, :].float().index_select(0, safe_indices.reshape(-1))
    gathered = gathered.reshape(tokens, safe_indices.shape[1], head_dim)
    scores = torch.matmul(q.float(), gathered.transpose(1, 2))
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    scores = scores * sm_scale * LOG2E
    kv_max = scores.max(dim=-1).values
    nonempty = valid.any(dim=-1)[:, None].expand(-1, heads)
    safe_max = torch.where(nonempty, kv_max, torch.zeros_like(kv_max))
    weights = torch.exp2(scores - safe_max.unsqueeze(-1)) * valid[:, None, :]
    kv_norm = weights.sum(dim=-1)
    norm = kv_norm
    value_acc = torch.matmul(weights, gathered[:, :, :512])
    if attn_sink is not None:
        sink_weight = torch.exp2(
            attn_sink.float()[None, :] * LOG2E - safe_max
        )
        sink_weight = torch.where(
            nonempty & (attn_sink[None, :] > float("-inf")),
            sink_weight,
            torch.zeros_like(sink_weight),
        )
        norm = norm + sink_weight
    safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    output = torch.where(
        (norm > 0.0).unsqueeze(-1), value_acc / safe_norm.unsqueeze(-1), 0.0
    ).to(torch.bfloat16)
    max_logits = torch.where(nonempty, kv_max, torch.full_like(kv_max, float("-inf")))
    # lse must retain the KV-only normalization, not the sink-augmented norm.
    lse = torch.where(
        nonempty,
        kv_max + torch.log2(kv_norm),
        torch.full_like(kv_max, float("-inf")),
    )
    return output, max_logits, lse


@pytest.mark.parametrize(
    ("compression_ratio", "tokens", "kv_tokens", "heads", "head_dim", "topk"),
    [
        (1, 1, 1, 1, 512, 1),  # compression ratio 1, shortest boundary case
        (4, 3, 7, 2, 513, 9),  # ratio 4, non-aligned dim and top-k width
        (128, 5, 17, 5, 576, 13),  # ratio 128, padded DeepSeek V4 layout
    ],
)
def test_sparse_prefill_matches_oracle_and_writes_output_buffer(
    metax_device: torch.device,
    compression_ratio: int,
    tokens: int,
    kv_tokens: int,
    heads: int,
    head_dim: int,
    topk: int,
) -> None:
    q, kv, indices, topk_length = _case(
        device=metax_device,
        compression_ratio=compression_ratio,
        tokens=tokens,
        kv_tokens=kv_tokens,
        heads=heads,
        head_dim=head_dim,
        topk=topk,
        seed=1000 + compression_ratio + tokens + kv_tokens + head_dim,
    )
    sm_scale = 0.125
    expected = _oracle(q, kv, indices, sm_scale, topk_length=topk_length)
    sentinel = 7.0
    storage = torch.full(
        (expected[0].numel() + 8,),
        sentinel,
        device=metax_device,
        dtype=torch.bfloat16,
    )
    output_buffer = storage[4:-4].view_as(expected[0])

    actual = flash_mla_sparse_fwd_wrapper(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
        topk_length=topk_length,
        out=output_buffer,
    )

    assert actual[0] is output_buffer
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual[0]).all()
    assert torch.isfinite(actual[1]).all()
    assert torch.isfinite(actual[2]).all()
    assert torch.equal(storage[:4], torch.full_like(storage[:4], sentinel))
    assert torch.equal(storage[-4:], torch.full_like(storage[-4:], sentinel))


def test_sparse_prefill_honors_topk_length(metax_device: torch.device) -> None:
    q, kv, indices, _ = _case(
        device=metax_device,
        tokens=3,
        kv_tokens=11,
        heads=2,
        head_dim=513,
        topk=8,
        seed=20260712,
    )
    topk_length = torch.tensor([1, 4, 7], dtype=torch.int32, device=metax_device)
    # All entries are valid here; only topk_length is expected to mask them.
    indices = indices.abs() % kv.shape[0]
    expected = _oracle(q, kv, indices, 0.2, topk_length=topk_length)
    actual = flash_mla_sparse_fwd_wrapper(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.2,
        topk_length=topk_length,
    )
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-3, rtol=2e-3)


def test_sparse_prefill_native_attention_sink_preserves_kv_metadata(
    metax_device: torch.device,
) -> None:
    q, kv, indices, topk_length = _case(
        device=metax_device,
        tokens=3,
        kv_tokens=9,
        heads=2,
        head_dim=513,
        topk=8,
        seed=2718,
    )
    sink = torch.tensor([0.75, -0.5], dtype=torch.float32, device=metax_device)
    baseline = flash_mla_sparse_fwd_wrapper(
        q=q, kv=kv, indices=indices, sm_scale=0.125, topk_length=topk_length
    )
    expected = _oracle(q, kv, indices, 0.125, topk_length, sink)
    storage = torch.full(
        (expected[0].numel() + 8,),
        3.0,
        dtype=torch.bfloat16,
        device=metax_device,
    )
    out = storage[4:-4].view_as(expected[0])
    actual = flash_mla_sparse_fwd_wrapper(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.125,
        attn_sink=sink,
        topk_length=topk_length,
        out=out,
    )
    assert actual[0] is out
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], baseline[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[2], baseline[2], atol=2e-3, rtol=2e-3)
    assert not torch.equal(actual[0], baseline[0])
    assert torch.equal(storage[:4], torch.full_like(storage[:4], 3.0))
    assert torch.equal(storage[-4:], torch.full_like(storage[-4:], 3.0))

    no_sink = torch.full_like(sink, float("-inf"))
    no_sink_result = flash_mla_sparse_fwd_wrapper(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.125,
        attn_sink=no_sink,
        topk_length=topk_length,
    )
    torch.testing.assert_close(no_sink_result[0], baseline[0], equal_nan=True)
    torch.testing.assert_close(no_sink_result[1], baseline[1], equal_nan=True)
    torch.testing.assert_close(no_sink_result[2], baseline[2], equal_nan=True)


def test_sparse_prefill_native_attention_sink_empty_rows_and_validation(
    metax_device: torch.device,
) -> None:
    q = torch.ones((2, 1, 4), dtype=torch.bfloat16, device=metax_device)
    kv = torch.ones((3, 1, 4), dtype=torch.bfloat16, device=metax_device)
    indices = torch.full((2, 1, 3), -1, dtype=torch.int32, device=metax_device)
    topk_length = torch.zeros((2,), dtype=torch.int32, device=metax_device)
    sink = torch.zeros((1,), dtype=torch.float32, device=metax_device)
    output, max_logits, lse = flash_mla_sparse_fwd_wrapper(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.125,
        d_v=4,
        attn_sink=sink,
        topk_length=topk_length,
    )
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isneginf(max_logits).all()
    assert torch.isneginf(lse).all()
    with pytest.raises(TypeError, match="float32"):
        flash_mla_sparse_fwd_wrapper(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=0.125,
            d_v=4,
            attn_sink=sink.to(torch.bfloat16),
            topk_length=topk_length,
        )
    with pytest.raises(ValueError, match="shape"):
        flash_mla_sparse_fwd_wrapper(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=0.125,
            d_v=4,
            attn_sink=torch.zeros((2,), dtype=torch.float32, device=metax_device),
            topk_length=topk_length,
        )


class _IndexSelectRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        self.output_numels: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.index_select.default and isinstance(
            result, torch.Tensor
        ):
            self.output_numels.append(result.numel())
        return result


def test_sparse_prefill_does_not_materialize_token_topk_dimension(
    metax_device: torch.device,
) -> None:
    q, kv, indices, topk_length = _case(
        device=metax_device,
        tokens=5,
        kv_tokens=257,
        heads=2,
        head_dim=576,
        topk=33,
        seed=4242,
    )
    recorder = _IndexSelectRecorder()
    with recorder:
        flash_mla_sparse_fwd_wrapper(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=0.125,
            topk_length=topk_length,
        )

    # A streaming kernel may use no index_select at all.  Any intermediate
    # larger than a small O(tokens * dim) budget exposes the old gather path.
    budget = 2 * q.shape[0] * q.shape[-1]
    observed = max(recorder.output_numels, default=0)
    assert observed <= budget, (
        "sparse prefill materialized an oversized gather: "
        f"observed={observed}, budget={budget}"
    )


def test_sparse_prefill_oracle_assertion_is_red_capable() -> None:
    q, kv, indices, _ = _case(
        device=_oracle_device(),
        tokens=2,
        kv_tokens=5,
        heads=2,
        head_dim=512,
        topk=4,
        seed=99,
    )
    expected = _oracle(q, kv, indices, 0.125)
    candidate = expected[0].clone()
    candidate[0, 0, 0] += torch.tensor(0.125, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(candidate, expected[0], atol=0, rtol=0)


def test_sparse_prefill_oracle_attention_sink_only_changes_output() -> None:
    q, kv, indices, topk_length = _case(
        device=_oracle_device(),
        tokens=2,
        kv_tokens=5,
        heads=2,
        head_dim=512,
        topk=4,
        seed=31415,
    )
    baseline = _oracle(q, kv, indices, 0.125, topk_length)
    sink = torch.tensor([0.5, -0.25], dtype=torch.float32, device=q.device)
    with_sink = _oracle(q, kv, indices, 0.125, topk_length, sink)
    assert not torch.equal(with_sink[0], baseline[0])
    torch.testing.assert_close(with_sink[1], baseline[1], equal_nan=True)
    torch.testing.assert_close(with_sink[2], baseline[2], equal_nan=True)

    no_sink = torch.full_like(sink, float("-inf"))
    no_sink_result = _oracle(q, kv, indices, 0.125, topk_length, no_sink)
    torch.testing.assert_close(no_sink_result[0], baseline[0], equal_nan=True)
    torch.testing.assert_close(no_sink_result[1], baseline[1], equal_nan=True)
    torch.testing.assert_close(no_sink_result[2], baseline[2], equal_nan=True)


def test_sparse_prefill_oracle_empty_invalid_rows_are_finite() -> None:
    device = _oracle_device()
    q = torch.ones((2, 1, 4), dtype=torch.bfloat16, device=device)
    kv = torch.ones((3, 1, 4), dtype=torch.bfloat16, device=device)
    indices = torch.full((2, 1, 3), -1, dtype=torch.int32, device=device)
    topk_length = torch.zeros((2,), dtype=torch.int32, device=device)
    output, max_logits, lse = _oracle(
        q,
        kv,
        indices,
        0.125,
        topk_length,
        torch.tensor([0.0], dtype=torch.float32, device=device),
    )
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isneginf(max_logits).all()
    assert torch.isneginf(lse).all()


def _capture_inputs() -> dict[str, object]:
    q = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    kv = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    indices = torch.tensor([[[0, 1]], [[2, -1]]], dtype=torch.int32)
    topk_length = torch.tensor([2, 1], dtype=torch.int32)
    output = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    max_logits = torch.tensor([[1.0], [2.0]])
    lse = torch.tensor([[3.0], [4.0]])
    return {
        "q": q,
        "kv": kv,
        "indices": indices,
        "sm_scale": 0.125,
        "d_v": 4,
        "attn_sink": None,
        "topk_length": topk_length,
        "out": output,
        "output": output,
        "max_logits": max_logits,
        "lse": lse,
    }


def _decode_capture_inputs() -> dict[str, object]:
    q_base = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    q = q_base.transpose(1, 2)
    swa_base = torch.arange(48, dtype=torch.float32).reshape(3, 4, 1, 4)
    swa_cache = swa_base.transpose(0, 1)
    compressed_base = torch.arange(64, dtype=torch.float32).reshape(4, 4, 1, 4)
    compressed_cache = compressed_base.transpose(0, 1)
    swa_indices = torch.tensor(
        [[[0, -1, 99]], [[1, 2, -1]]], dtype=torch.int32
    )
    topk_indices = torch.tensor(
        [[[3, -1]], [[0, 88]]], dtype=torch.int32
    )
    swa_lens = torch.tensor([1, 2], dtype=torch.int32)
    topk_lens = torch.tensor([1, 2], dtype=torch.int32)
    output_storage = torch.zeros((2, 3, 4), dtype=torch.float32)
    output = output_storage[..., ::2]
    attn_sink = torch.arange(3, dtype=torch.float32)
    return {
        "q": q,
        "swa_cache": swa_cache,
        "compressed_cache": compressed_cache,
        "swa_indices": swa_indices,
        "topk_indices": topk_indices,
        "swa_lens": swa_lens,
        "topk_lens": topk_lens,
        "sm_scale": 0.125,
        "d_v": 2,
        "attn_sink": attn_sink,
        "output": output,
    }


def test_sparse_mla_capture_disabled_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("disabled capture should not inspect or copy tensors")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(sparse_mla_debug.Path, "mkdir", fail)
    monkeypatch.setattr(
        sparse_mla_debug,
        "_synchronize_capture_stream",
        fail,
    )
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(
        **_capture_inputs()
    )
    assert not list(tmp_path.iterdir())


def test_sparse_mla_capture_during_graph_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(sparse_mla_debug, "_is_cuda_graph_capturing", lambda: True)

    def fail(*args, **kwargs):
        raise AssertionError("graph capture must not copy tensors or touch files")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(sparse_mla_debug.Path, "mkdir", fail)
    monkeypatch.setattr(
        sparse_mla_debug,
        "_synchronize_capture_stream",
        fail,
    )
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert not list(tmp_path.iterdir())


def test_sparse_mla_capture_synchronizes_before_cpu_clones(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(sparse_mla_debug, "_is_cuda_graph_capturing", lambda: False)
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()

    events: list[str] = []

    def sync(value):
        events.append("sync")

    def clone(value):
        events.append("clone")
        return value

    monkeypatch.setattr(sparse_mla_debug, "_synchronize_capture_stream", sync)
    monkeypatch.setattr(sparse_mla_debug, "_clone_to_cpu", clone)
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())

    assert events[0] == "sync"
    assert events.count("sync") == 1
    assert "clone" in events


def test_sparse_mla_decode_capture_disabled_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("disabled capture should not inspect or copy tensors")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(sparse_mla_debug.Path, "mkdir", fail)
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**_decode_capture_inputs())
    assert not list(tmp_path.iterdir())


def test_sparse_mla_capture_stage_filter_runs_before_budget_or_copy(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_STAGES", "decode")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "1")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")

    def fail(*args, **kwargs):
        raise AssertionError("filtered stage must not inspect or copy tensors")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert sparse_mla_debug._CAPTURE_CALL_COUNT == 0
    assert not list(tmp_path.iterdir())


def test_sparse_mla_capture_skip_runs_before_budget_or_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_SKIP_CALLS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "1")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    original_clone = sparse_mla_debug._clone_to_cpu

    def fail(*args, **kwargs):
        raise AssertionError("skipped calls must not copy tensors")

    monkeypatch.setattr(sparse_mla_debug, "_clone_to_cpu", fail)
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert sparse_mla_debug._CAPTURE_CALL_COUNT == 0
    assert sparse_mla_debug._CAPTURE_SKIP_COUNTS == {("0", "prefill"): 1}
    assert not list(tmp_path.iterdir())

    monkeypatch.setattr(sparse_mla_debug, "_clone_to_cpu", original_clone)
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert (tmp_path / "rank0_call0.pt").is_file()
    assert sparse_mla_debug._CAPTURE_CALL_COUNT == 1

    sparse_mla_debug.reset_sparse_mla_capture_state()
    assert sparse_mla_debug._CAPTURE_SKIP_COUNTS == {}


def test_sparse_mla_decode_mode_filter_and_classification(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DECODE_MODES", "topk")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _decode_capture_inputs()

    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=inputs["swa_cache"],
            compressed_cache=inputs["compressed_cache"],
            swa_indices=inputs["swa_indices"],
            topk_indices=inputs["topk_indices"],
            swa_lens=inputs["swa_lens"],
            topk_lens=inputs["topk_lens"],
        )
        == "dual"
    )
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**inputs)
    assert not list(tmp_path.iterdir())

    topk_inputs = dict(inputs)
    topk_inputs["swa_lens"] = torch.zeros_like(inputs["swa_lens"])
    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=topk_inputs["swa_cache"],
            compressed_cache=topk_inputs["compressed_cache"],
            swa_indices=topk_inputs["swa_indices"],
            topk_indices=topk_inputs["topk_indices"],
            swa_lens=topk_inputs["swa_lens"],
            topk_lens=topk_inputs["topk_lens"],
        )
        == "topk"
    )
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**topk_inputs)
    assert (tmp_path / "rank0_call0.pt").is_file()

    swa_inputs = dict(inputs)
    swa_inputs["topk_lens"] = torch.zeros_like(inputs["topk_lens"])
    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=swa_inputs["swa_cache"],
            compressed_cache=swa_inputs["compressed_cache"],
            swa_indices=swa_inputs["swa_indices"],
            topk_indices=swa_inputs["topk_indices"],
            swa_lens=swa_inputs["swa_lens"],
            topk_lens=swa_inputs["topk_lens"],
        )
        == "swa"
    )


def test_sparse_mla_decode_compat_mode_for_zero_or_absent_lens(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DECODE_MODES", "compat")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _decode_capture_inputs()
    compat_inputs = dict(inputs)
    compat_inputs["swa_lens"] = torch.zeros_like(inputs["swa_lens"])
    compat_inputs["topk_lens"] = torch.zeros_like(inputs["topk_lens"])

    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=compat_inputs["swa_cache"],
            compressed_cache=compat_inputs["compressed_cache"],
            swa_indices=compat_inputs["swa_indices"],
            topk_indices=compat_inputs["topk_indices"],
            swa_lens=compat_inputs["swa_lens"],
            topk_lens=compat_inputs["topk_lens"],
        )
        == "compat"
    )
    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=inputs["swa_cache"],
            compressed_cache=inputs["compressed_cache"],
            swa_indices=inputs["swa_indices"],
            topk_indices=inputs["topk_indices"],
            swa_lens=None,
            topk_lens=None,
        )
        == "compat"
    )
    assert (
        sparse_mla_debug._classify_decode_mode(
            swa_cache=inputs["swa_cache"],
            compressed_cache=inputs["compressed_cache"],
            swa_indices=None,
            topk_indices=None,
            swa_lens=None,
            topk_lens=None,
        )
        is None
    )

    sparse_mla_debug.maybe_capture_sparse_mla_decode(**compat_inputs)
    payload = torch.load(
        tmp_path / "rank0_call0.pt", map_location="cpu", weights_only=False
    )
    assert payload["decode_mode"] == "compat"


def test_sparse_mla_capture_skip_composes_with_filters_and_ratio_caps(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_STAGES", "decode")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DECODE_MODES", "topk")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_SKIP_CALLS", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "2")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_PER_RATIO_MAX_CALLS", "1"
    )
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _decode_capture_inputs()

    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**inputs, compress_ratio=1)
    assert sparse_mla_debug._CAPTURE_SKIP_COUNTS == {}

    topk_inputs = dict(inputs)
    topk_inputs["swa_lens"] = torch.zeros_like(inputs["swa_lens"])
    for ratio in (1, 1, 1, 4, 128):
        sparse_mla_debug.maybe_capture_sparse_mla_decode(
            **topk_inputs, compress_ratio=ratio
        )

    files = sorted(tmp_path.glob("rank0_call*.pt"))
    assert [path.name for path in files] == ["rank0_call0.pt", "rank0_call1.pt"]
    assert [
        torch.load(path, map_location="cpu", weights_only=False)["cache_meta"][
            "compress_ratio"
        ]
        for path in files
    ] == [1, 4]
    assert sparse_mla_debug._CAPTURE_SKIP_COUNTS == {("0", "decode"): 1}
    assert sparse_mla_debug._CAPTURE_CALL_COUNT == 2


def test_sparse_mla_capture_per_ratio_budget_and_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "0")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_PER_RATIO_MAX_CALLS", "1"
    )
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _decode_capture_inputs()
    for ratio in (1, 1, 4):
        sparse_mla_debug.maybe_capture_sparse_mla_decode(
            **inputs, compress_ratio=ratio
        )
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "rank0_call0.pt",
        "rank0_call1.pt",
    ]

    sparse_mla_debug.reset_sparse_mla_capture_state()
    assert sparse_mla_debug._CAPTURE_CALL_COUNT == 0
    assert sparse_mla_debug._CAPTURE_BUCKET_COUNTS == {}
    for path in tmp_path.glob("*.pt"):
        path.unlink()
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**inputs, compress_ratio=1)
    assert (tmp_path / "rank0_call0.pt").is_file()


def test_sparse_mla_prefill_per_ratio_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_PER_RATIO_MAX_CALLS", "1"
    )
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _capture_inputs()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**inputs, compress_ratio=1)
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**inputs, compress_ratio=1)
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**inputs, compress_ratio=4)
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "rank0_call0.pt",
        "rank0_call1.pt",
    ]


def test_sparse_mla_decode_capture_schema_preserves_strides_and_invalid_indices(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "1")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    inputs = _decode_capture_inputs()
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**inputs)
    sparse_mla_debug.maybe_capture_sparse_mla_decode(**inputs)

    files = sorted(tmp_path.glob("rank0_call*.pt"))
    assert [path.name for path in files] == ["rank0_call0.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["stage"] == "decode"
    assert payload["d_v"] == 2
    assert payload["head_dim"] == 4
    assert payload["q_heads"] == 3
    assert payload["input_meta"]["q"]["strides"] == list(inputs["q"].stride())
    assert payload["input_meta"]["swa_cache"]["strides"] == list(
        inputs["swa_cache"].stride()
    )
    assert payload["input_meta"]["compressed_cache"]["strides"] == list(
        inputs["compressed_cache"].stride()
    )
    assert torch.equal(payload["swa_indices"], inputs["swa_indices"])
    assert torch.equal(payload["topk_indices"], inputs["topk_indices"])
    assert torch.equal(payload["swa_lens"], inputs["swa_lens"])
    assert torch.equal(payload["topk_lens"], inputs["topk_lens"])
    assert payload["cache_meta"] == {
        "swa_block_size": inputs["swa_cache"].shape[1],
        "compressed_block_size": inputs["compressed_cache"].shape[1],
        "compress_ratio": None,
        "window_size": None,
    }


def test_sparse_mla_capture_schema_rank_and_call_filtering(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "2")
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "1")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())

    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    for _ in range(3):
        sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())

    files = sorted(tmp_path.glob("rank0_call*.pt"))
    assert [path.name for path in files] == ["rank0_call0.pt", "rank0_call1.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["stage"] == "prefill"
    assert payload["rank"] == 0
    assert payload["call"] == 0
    assert payload["sm_scale"] == 0.125
    assert payload["d_v"] == 4
    assert payload["caller_out"]["data_ptr"] > 0
    assert payload["caller_out"]["shape"] == [2, 1, 4]
    assert payload["input_meta"]["q"]["strides"] == [4, 4, 1]
    for name in ("q", "kv", "indices", "topk_length", "output", "max_logits", "lse"):
        assert isinstance(payload[name], torch.Tensor)
    assert payload["attn_sink"] is None


def test_sparse_mla_capture_filenames_use_numeric_call_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "all")
    monkeypatch.delenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", raising=False)
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    for _ in range(11):
        sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert (tmp_path / "rank0_call10.pt").is_file()


def test_sparse_mla_capture_does_not_overwrite_existing_call(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(sparse_mla_debug, "_rank", lambda: "0")
    existing = tmp_path / "rank0_call0.pt"
    existing.write_bytes(b"existing capture")
    sparse_mla_debug.reset_sparse_mla_capture_state()
    sparse_mla_debug.maybe_capture_sparse_mla_prefill(**_capture_inputs())
    assert existing.read_bytes() == b"existing capture"
    assert (tmp_path / "rank0_call1.pt").is_file()


def test_sparse_mla_topk_length_validation_is_device_assertion(monkeypatch):
    from vllm_metax.kernels import sparse_mla_prefill as kernel

    observed = []
    monkeypatch.setattr(torch, "_assert_async", lambda condition: observed.append(condition))
    values = torch.tensor([-1, 2, 3], dtype=torch.int32)
    kernel._assert_topk_length_device(values, topk=2)
    assert len(observed) == 1
    assert observed[0].ndim == 0
    assert not bool(observed[0])

    monkeypatch.delattr(torch, "_assert_async")
    with pytest.raises(RuntimeError, match="_assert_async"):
        kernel._assert_topk_length_device(values, topk=2)
