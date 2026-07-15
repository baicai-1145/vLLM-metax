# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Diagnostic-only synchronization before breakable outer-graph replay."""

import os
import threading

import torch
from torch.profiler import record_function

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader

logger = init_logger(__name__)

_ENV_NAME = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_DEVICE_SYNC"
_BARRIER_ENV_NAME = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_TP_BARRIER"
_ALIGNED_ENV_NAME = "VLLM_METAX_DSV4_PRE_OUTER_GRAPH_ALIGNED_REPLAY"
_warning_emitted = False
_barrier_warning_emitted = False
_aligned_warning_emitted = False
_warning_lock = threading.Lock()


def _parse_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _enabled() -> bool:
    return _parse_flag(_ENV_NAME)


def _is_decode_only_piecewise() -> bool:
    forward_context = get_forward_context()
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
        return False
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        return False
    saw_decode = False
    for metadata in attn_metadata.values():
        if getattr(metadata, "num_prefills", 0) != 0:
            return False
        if getattr(metadata, "num_decodes", 0) > 0:
            saw_decode = True
    return saw_decode


def _warn_once() -> None:
    global _warning_emitted
    if _warning_emitted:
        return
    with _warning_lock:
        if _warning_emitted:
            return
        logger.warning(
            "DIAGNOSTIC_ONLY: synchronizing the current device stream before "
            "breakable outer-graph replay"
        )
        _warning_emitted = True


def _validated_tp_group(vllm_config):
    tp_group = get_tp_group()
    if getattr(tp_group, "world_size", None) != 4:
        raise RuntimeError("TP Gloo barrier probe requires tensor parallel size 4")
    cpu_group = getattr(tp_group, "cpu_group", None)
    if cpu_group is None:
        raise RuntimeError("TP Gloo barrier probe requires a CPU process group")
    backend = str(torch.distributed.get_backend(cpu_group)).lower().split(".")[-1]
    if backend != "gloo":
        raise RuntimeError(
            f"TP Gloo barrier probe requires gloo CPU backend, got {backend!r}"
        )

    if vllm_config is None:
        raise RuntimeError("TP Gloo barrier probe requires wrapper vllm_config")
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.tensor_parallel_size != 4
        or parallel_config.data_parallel_size != 1
        or parallel_config.pipeline_parallel_size != 1
    ):
        raise RuntimeError("TP Gloo barrier probe requires TP4/DP1/PP1")
    speculative_config = vllm_config.speculative_config
    speculative_tokens = (
        0
        if speculative_config is None
        else speculative_config.num_speculative_tokens or 0
    )
    if speculative_tokens != 0:
        raise RuntimeError("TP Gloo barrier probe requires MTP/speculation off")
    return tp_group, backend


def _warn_barrier_once(tp_group, backend: str) -> None:
    global _barrier_warning_emitted
    if _barrier_warning_emitted:
        return
    with _warning_lock:
        if _barrier_warning_emitted:
            return
        logger.warning(
            "DIAGNOSTIC_ONLY: pre-outer-graph TP barrier mode=%s backend=%s "
            "world_size=%s rank=%s pid=%s",
            CUDAGraphMode.PIECEWISE.name,
            backend,
            tp_group.world_size,
            getattr(tp_group, "rank_in_group", getattr(tp_group, "rank", None)),
            os.getpid(),
        )
        _barrier_warning_emitted = True


def _warn_aligned_once(tp_group, backend: str) -> None:
    global _aligned_warning_emitted
    if _aligned_warning_emitted:
        return
    with _warning_lock:
        if _aligned_warning_emitted:
            return
        logger.warning(
            "DIAGNOSTIC_ONLY: pre-outer-graph aligned replay mode=%s backend=%s "
            "world_size=%s rank=%s pid=%s",
            CUDAGraphMode.PIECEWISE.name,
            backend,
            tp_group.world_size,
            getattr(tp_group, "rank_in_group", getattr(tp_group, "rank", None)),
            os.getpid(),
        )
        _aligned_warning_emitted = True


def _replay(self, entry, args, kwargs):
    enabled = _enabled()
    barrier_enabled = _parse_flag(_BARRIER_ENV_NAME)
    aligned_enabled = _parse_flag(_ALIGNED_ENV_NAME)
    if sum((enabled, barrier_enabled, aligned_enabled)) > 1:
        raise RuntimeError(
            "enabled pre-outer-graph diagnostic mode conflicts; mutually exclusive: "
            f"{_ENV_NAME}, {_BARRIER_ENV_NAME}, {_ALIGNED_ENV_NAME}"
        )
    if self.is_debugging_mode and entry.input_addresses is not None:
        new_addresses = self._collect_tensor_addresses(args, kwargs)
        assert new_addresses == entry.input_addresses, (
            "Input tensor addresses changed between capture and replay "
            f"for {entry.batch_descriptor}. Expected "
            f"{entry.input_addresses}, got {new_addresses}."
        )
    get_offloader().sync_prev_onload()
    assert entry.capture is not None
    if enabled and _is_decode_only_piecewise():
        _warn_once()
        with record_function("plan35.pre_outer_graph_device_sync"):
            torch.cuda.current_stream().synchronize()
    elif barrier_enabled and _is_decode_only_piecewise():
        tp_group, backend = _validated_tp_group(self.vllm_config)
        _warn_barrier_once(tp_group, backend)
        with record_function("plan35.pre_outer_graph_tp_gloo_barrier"):
            tp_group.barrier()
    elif aligned_enabled and _is_decode_only_piecewise():
        tp_group, backend = _validated_tp_group(self.vllm_config)
        _warn_aligned_once(tp_group, backend)
        with record_function("plan35.pre_outer_graph_aligned_replay"):
            torch.cuda.current_stream().synchronize()
            tp_group.barrier()
    entry.capture.replay()
    return entry.output


_replay._vllm_metax_pre_outer_graph_device_sync = True
if not getattr(
    BreakableCUDAGraphWrapper._replay,
    "_vllm_metax_pre_outer_graph_device_sync",
    False,
):
    BreakableCUDAGraphWrapper._replay = _replay
