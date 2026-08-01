import os
import subprocess
import sys
import types

import pytest
import torch

import vllm_metax.models.deepseek_v4.attention as attention


ENV = "VLLM_METAX_DSV4_Q_INSERT_CUDAGRAPH_LAYER"


class _FakeLayer(attention.MacaDeepseekV4Attention):
    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def forward_mqa(self, q, kv, positions, output):
        raise NotImplementedError

    def _o_proj(self, o, positions):
        raise NotImplementedError


class _FakeGraph:
    created = 0

    def __init__(self):
        self.replays = 0
        type(self).created += 1

    def replay(self):
        self.replays += 1


class _FakeWqB:
    def __init__(self):
        self.weight = torch.ones((2, 2))

    def __call__(self, qr):
        return torch.ones((1, 2))


class _GraphContext:
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs

    def __enter__(self):
        return self.graph

    def __exit__(self, exc_type, exc, tb):
        return False


def _layer():
    layer = object.__new__(_FakeLayer)
    layer.layer_idx = 3
    layer.indexer = None
    layer.compressor = lambda *args: None
    layer.is_target_model = False
    layer._q_insert_cudagraphs = {}
    layer._q_insert_cudagraph_pool = None
    layer.aux_stream_list = None
    layer.ln_events = [object(), object(), object(), object()]
    layer.n_local_heads = 1
    layer.padded_heads = 2
    layer.head_dim = 2
    layer.eps = 1e-5
    layer.swa_cache_layer = types.SimpleNamespace(
        prefix="swa", kv_cache=torch.zeros((1, 2, 2))
    )
    layer.rotary_emb = types.SimpleNamespace(cos_sin_cache=torch.zeros((2, 2)))
    layer.q_lora_rank = 2
    layer.wq_b = _FakeWqB()
    layer._q_insert_cudagraph_native = lambda q, kv, positions, metadata: q
    return layer


def test_layer3_q_insert_capture_then_replay(monkeypatch):
    layer = _layer()
    compressor_calls = []
    layer.compressor = lambda *args: compressor_calls.append(args)
    monkeypatch.setenv(ENV, "3")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn, aux, *args, **kwargs: (fn(), aux()),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = types.SimpleNamespace(cuda_stream=17, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: _GraphContext(graph, **kwargs))
    pool_calls = []
    monkeypatch.setattr(
        attention,
        "_q_insert_cudagraph_pool_handle",
        lambda: pool_calls.append(1) or object(),
        raising=False,
    )

    forwards = []
    layer.forward_mqa = lambda q, kv, positions, out: forwards.append(q)
    hidden = torch.ones((1, 2))
    qr = torch.ones((1, 2))
    kv = torch.ones((1, 2))
    positions = torch.zeros((1,), dtype=torch.int64)
    out = torch.empty((1, 2, 2))

    layer.attention_impl(hidden, qr, kv, object(), object(), object(), positions, out)
    layer.attention_impl(hidden, qr, kv, object(), object(), object(), positions, out)

    assert _FakeGraph.created == 1
    assert len(layer._q_insert_cudagraphs) == 1
    assert layer._q_insert_cudagraphs[next(iter(layer._q_insert_cudagraphs))][0].replays == 2
    assert len(forwards) == 2
    assert forwards[0] is forwards[1]
    assert len(compressor_calls) == 2
    assert pool_calls == [1]


def test_attention_impl_preserves_eager_break_wrapper():
    code = (
        "from vllm_metax.models.deepseek_v4.attention import "
        "MacaDeepseekV4Attention; "
        "assert hasattr(MacaDeepseekV4Attention.attention_impl, '__wrapped__')"
    )
    env = os.environ.copy()
    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_q_insert_key_tracks_wq_b_weight_pointer(monkeypatch):
    layer = _layer()
    metadata = types.SimpleNamespace(
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    stream = types.SimpleNamespace(cuda_stream=17)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    qr = torch.ones((1, 2))
    kv = torch.ones((1, 2))
    positions = torch.zeros((1,), dtype=torch.int64)

    before = layer._q_insert_cudagraph_key(qr, kv, positions, metadata)
    layer.wq_b.weight = layer.wq_b.weight.clone()
    after = layer._q_insert_cudagraph_key(qr, kv, positions, metadata)

    assert before != after


@pytest.mark.parametrize("target", ["all", "8"])
def test_target_enables_another_eligible_layer(monkeypatch, target):
    layer = _layer()
    layer.layer_idx = 8
    monkeypatch.setenv(ENV, target)
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn, aux, *args, **kwargs: (fn(), aux()),
    )
    q = torch.ones((1, 2, 2))
    layer._q_insert_cudagraph_forward = lambda *args: q
    forwarded = []
    layer.forward_mqa = lambda *args: forwarded.append(args[0])

    layer.attention_impl(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        object(),
        object(),
        object(),
        torch.zeros((1,), dtype=torch.int64),
        torch.empty((1, 2, 2)),
    )

    assert forwarded == [q]


def test_all_target_keeps_c4a_indexer_and_compressor_eager(monkeypatch):
    layer = _layer()
    layer.layer_idx = 2
    layer.aux_stream_list = [object(), object(), object()]
    layer.indexer_rotary_emb = object()
    monkeypatch.setenv(ENV, "all")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attention_impl",
        lambda *args: pytest.fail("C4A candidate fell back to upstream"),
    )
    q = torch.ones((1, 2, 2))
    layer._q_insert_cudagraph_forward = lambda *args: q
    indexer_calls = []
    compressor_calls = []
    layer.indexer = lambda *args: indexer_calls.append(args)
    layer.compressor = lambda *args: compressor_calls.append(args)
    forwarded = []
    layer.forward_mqa = lambda *args: forwarded.append(args[0])

    def run_parallel(default_fn, aux_fns, *args, **kwargs):
        result = default_fn()
        for aux_fn in aux_fns:
            aux_fn()
        return result, None

    monkeypatch.setattr(attention, "execute_in_parallel", run_parallel, raising=False)
    layer.attention_impl(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.zeros((1,), dtype=torch.int64),
        torch.empty((1, 2, 2)),
    )

    assert forwarded == [q]
    assert len(indexer_calls) == 1
    assert len(compressor_calls) == 1


def test_all_target_enables_ratio0_attention_without_aux_work(monkeypatch):
    layer = _layer()
    layer.layer_idx = 0
    layer.indexer = None
    layer.compressor = None
    monkeypatch.setenv(ENV, "all")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attention_impl",
        lambda *args: pytest.fail("ratio-0 candidate fell back to upstream"),
    )
    q = torch.ones((1, 2, 2))
    layer._q_insert_cudagraph_forward = lambda *args: q
    forwarded = []
    layer.forward_mqa = lambda *args: forwarded.append(args[0])

    layer.attention_impl(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        object(),
        object(),
        object(),
        torch.zeros((1,), dtype=torch.int64),
        torch.empty((1, 2, 2)),
    )

    assert forwarded == [q]


@pytest.mark.parametrize(
    "env_value, layer_idx", [(None, 3), ("4", 3), ("3", 4)]
)
def test_default_off_and_non_layer_fallback(monkeypatch, env_value, layer_idx):
    layer = _layer()
    layer.layer_idx = layer_idx
    calls = []

    def upstream(self, *args):
        calls.append(args)

    monkeypatch.setattr(attention.DeepseekV4Attention, "attention_impl", upstream)
    if env_value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env_value)

    layer.attention_impl(
        *(torch.ones((1, 2)) for _ in range(3)),
        *(object() for _ in range(3)),
        torch.zeros(1),
        torch.empty((1, 2, 2)),
    )
    assert len(calls) == 1


def test_q_insert_key_change_captures_fresh_graph(monkeypatch):
    _FakeGraph.created = 0
    layer = _layer()
    monkeypatch.setenv(ENV, "3")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn, aux, *args, **kwargs: (fn(), aux()),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = types.SimpleNamespace(cuda_stream=17, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: _GraphContext(graph, **kwargs))
    pool = object()
    monkeypatch.setattr(attention, "_q_insert_cudagraph_pool_handle", lambda: pool)
    layer.forward_mqa = lambda *args: None
    args = (torch.ones((1, 2)), torch.ones((1, 2)), torch.ones((1, 2)))
    positions_a = torch.zeros((1,), dtype=torch.int64)
    positions_b = torch.ones((1,), dtype=torch.int64)
    common = (object(), object(), object())
    layer.attention_impl(*args, *common, positions_a, torch.empty((1, 2, 2)))
    layer.attention_impl(*args, *common, positions_b, torch.empty((1, 2, 2)))
    assert _FakeGraph.created == 2
    assert len(layer._q_insert_cudagraphs) == 2
    assert layer._q_insert_cudagraph_pool is pool


def test_active_outer_capture_fails_closed(monkeypatch):
    layer = _layer()
    monkeypatch.setenv(ENV, "3")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attention_impl",
        lambda *args: pytest.fail("nested capture fell back to upstream"),
    )
    with pytest.raises(RuntimeError, match="cannot nest"):
        layer.attention_impl(
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            object(),
            object(),
            object(),
            torch.zeros((1,), dtype=torch.int64),
            torch.empty((1, 2, 2)),
        )


def test_clear_synchronizes_and_uses_fresh_pool(monkeypatch):
    _FakeGraph.created = 0
    layer = _layer()
    monkeypatch.setenv(ENV, "3")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn, aux, *args, **kwargs: (fn(), aux()),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    syncs = []
    device_syncs = []
    stream = types.SimpleNamespace(cuda_stream=17, synchronize=lambda: syncs.append(1))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: device_syncs.append(1))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: _GraphContext(graph, **kwargs))
    pools = [object(), object()]
    pool_calls = []

    def pool_handle():
        pool_calls.append(1)
        return pools[len(pool_calls) - 1]

    monkeypatch.setattr(attention, "_q_insert_cudagraph_pool_handle", pool_handle)
    layer.forward_mqa = lambda *args: None
    args = (torch.ones((1, 2)), torch.ones((1, 2)), torch.ones((1, 2)))
    common = (object(), object(), object())
    positions = torch.zeros((1,), dtype=torch.int64)
    out = torch.empty((1, 2, 2))
    layer.attention_impl(*args, *common, positions, out)
    assert syncs == [1]
    layer.clear_q_insert_cudagraph_cache()
    assert not layer._q_insert_cudagraphs
    assert layer._q_insert_cudagraph_pool is None
    assert syncs == [1]
    assert device_syncs == [1]
    layer.attention_impl(*args, *common, positions, out)
    assert _FakeGraph.created == 2
    assert len(pool_calls) == 2


def test_invalid_layer_env_fails_closed(monkeypatch):
    monkeypatch.setenv(ENV, "layer3")
    with pytest.raises(ValueError, match=ENV):
        attention._q_insert_cudagraph_target_layer()


@pytest.mark.parametrize(
    "attn_metadata",
    [
        None,
        [],
        {"swa": types.SimpleNamespace(num_prefills=1, num_decode_tokens=0)},
    ],
)
def test_profile_prefill_and_non_dict_metadata_fallback(monkeypatch, attn_metadata):
    layer = _layer()
    monkeypatch.setenv(ENV, "3")
    calls = []
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "attention_impl",
        lambda self, *args: calls.append(args),
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=attn_metadata),
    )
    layer.attention_impl(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        object(),
        object(),
        object(),
        torch.zeros((1,), dtype=torch.int64),
        torch.empty((1, 2, 2)),
    )
    assert len(calls) == 1


def test_capture_exception_propagates_without_cache_entry(monkeypatch):
    layer = _layer()
    monkeypatch.setenv(ENV, "3")
    metadata = types.SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=1,
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        block_size=64,
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn, aux, *args, **kwargs: (fn(), aux()),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = types.SimpleNamespace(cuda_stream=17, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)

    class CaptureError:
        def __enter__(self):
            raise RuntimeError("capture failed")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: CaptureError())
    monkeypatch.setattr(attention, "_q_insert_cudagraph_pool_handle", lambda: object())
    with pytest.raises(RuntimeError, match="capture failed"):
        layer.attention_impl(
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            object(),
            object(),
            object(),
            torch.zeros((1,), dtype=torch.int64),
            torch.empty((1, 2, 2)),
        )
    assert not layer._q_insert_cudagraphs
