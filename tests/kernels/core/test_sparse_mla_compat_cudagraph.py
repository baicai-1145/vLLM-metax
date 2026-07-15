# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform

import vllm_metax.kernels.sparse_mla_decode as sparse


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False
        self.sync_calls = 0
        self.device_sync_calls = 0

        def synchronize():
            self.sync_calls += 1

        self.stream = SimpleNamespace(synchronize=synchronize)
        self.graphs: list[_FakeGraph] = []
        self.capture_calls = 0
        self.graph_streams: list[object | None] = []
        self.graph_pools: list[object] = []
        self.raise_capture = False

    def synchronize(self) -> None:
        self.device_sync_calls += 1

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def current_stream(self, *_args, **_kwargs):
        return self.stream

    def CUDAGraph(self):
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph

    def graph(self, _graph, *, pool, stream=None):
        self.capture_calls += 1
        self.graph_streams.append(stream)
        self.graph_pools.append(pool)
        if self.raise_capture:
            raise RuntimeError("capture failed")

        class _Capture:
            def __enter__(self):
                return _graph

            def __exit__(self, *_exc):
                return False

        return _Capture()


@pytest.fixture
def call_args():
    q = torch.empty((1, 1, 2), dtype=torch.bfloat16)
    cache = torch.empty((1, 2, 2), dtype=torch.bfloat16)
    swa_indices = torch.zeros((1, 1, 1), dtype=torch.int32)
    out = torch.empty((1, 1, 2), dtype=torch.bfloat16)
    return dict(
        q=q,
        swa_cache=cache,
        swa_indices=swa_indices,
        topk_indices=None,
        sm_scale=0.5,
        d_v=2,
        out=out,
        workspace=None,
    )


def _install_fakes(monkeypatch):
    cuda = _FakeCuda()
    sparse._COMPAT_CUDAGRAPHS.clear()
    sparse._COMPAT_CUDAGRAPH_POOL = None
    monkeypatch.setattr(sparse.torch, "cuda", cuda)
    pool_handles: list[object] = []

    def graph_pool_handle():
        handle = object()
        pool_handles.append(handle)
        return handle

    monkeypatch.setattr(current_platform, "graph_pool_handle", graph_pool_handle)
    calls: list[tuple] = []
    monkeypatch.setattr(
        sparse,
        "_sparse_mla_decode_compat_eager",
        lambda **kwargs: calls.append(tuple(kwargs.values())),
        raising=False,
    )
    return cuda, calls


def test_compat_cudagraph_is_disabled_by_default(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.delenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", raising=False)

    sparse._sparse_mla_decode_compat(**call_args)

    assert len(calls) == 1
    assert cuda.capture_calls == 0


def test_compat_cudagraph_warmup_capture_then_replay(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")

    sparse._sparse_mla_decode_compat(**call_args)
    sparse._sparse_mla_decode_compat(**call_args)

    assert len(calls) == 2
    assert cuda.capture_calls == 1
    assert cuda.sync_calls == 1
    assert cuda.graph_streams == [None]
    assert cuda.graphs[0].replays == 1


def test_compat_cudagraph_pointer_change_builds_new_entry(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")

    sparse._sparse_mla_decode_compat(**call_args)
    changed = dict(call_args, q=call_args["q"].clone())
    sparse._sparse_mla_decode_compat(**changed)

    assert len(calls) == 4
    assert cuda.capture_calls == 2


def test_compat_cudagraph_stream_change_builds_new_entry(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")
    cuda.stream.cuda_stream = 17

    sparse._sparse_mla_decode_compat(**call_args)
    cuda.stream.cuda_stream = 23
    sparse._sparse_mla_decode_compat(**call_args)

    assert len(calls) == 4
    assert cuda.capture_calls == 2


def test_compat_cudagraph_layout_change_builds_new_entry(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")

    sparse._sparse_mla_decode_compat(**call_args)
    cache_storage = torch.empty((1, 2, 4), dtype=torch.bfloat16)
    changed = dict(call_args, swa_cache=cache_storage[..., :2])
    sparse._sparse_mla_decode_compat(**changed)

    assert len(calls) == 4
    assert cuda.capture_calls == 2


def test_compat_cudagraph_active_capture_fails_closed(monkeypatch, call_args):
    cuda, calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")
    cuda.capturing = True

    with pytest.raises(RuntimeError, match="capture"):
        sparse._sparse_mla_decode_compat(**call_args)
    assert not calls


def test_compat_cudagraph_clear_rebuilds(monkeypatch, call_args):
    cuda, _calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")

    sparse._sparse_mla_decode_compat(**call_args)
    sparse.sparse_mla_decode_compat_cudagraph_clear_cache()
    sparse._sparse_mla_decode_compat(**call_args)

    assert cuda.capture_calls == 2
    assert cuda.graph_pools[0] is not cuda.graph_pools[1]
    assert cuda.sync_calls == 2
    assert cuda.device_sync_calls == 1


def test_compat_cudagraph_pool_is_shared_until_clear(monkeypatch, call_args):
    cuda, _calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")

    sparse._sparse_mla_decode_compat(**call_args)
    sparse._sparse_mla_decode_compat(
        **dict(call_args, q=call_args["q"].clone())
    )
    assert cuda.graph_pools[0] is cuda.graph_pools[1]

    sparse.sparse_mla_decode_compat_cudagraph_clear_cache()
    sparse._sparse_mla_decode_compat(
        **dict(call_args, q=call_args["q"].clone())
    )
    assert cuda.graph_pools[2] is not cuda.graph_pools[0]


def test_compat_cudagraph_capture_exception_propagates(monkeypatch, call_args):
    cuda, _calls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH", "1")
    cuda.raise_capture = True

    with pytest.raises(RuntimeError, match="capture failed"):
        sparse._sparse_mla_decode_compat(**call_args)
