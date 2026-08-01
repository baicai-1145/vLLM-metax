# SPDX-License-Identifier: Apache-2.0
"""Opt-in coarse phase ranges for DSpark cycle profiling."""

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_metax.models.deepseek_v4.dspark import DSparkDeepseekV4Model
from vllm_metax.models.deepseek_v4.model import DeepseekV4ForCausalLM


_PROFILE_PHASES_ENV = "VLLM_METAX_DSPARK_PROFILE_PHASES"
_PHASE_TIMING_DIR_ENV = "VLLM_METAX_DSPARK_PHASE_TIMING_DIR"
_PHASE_TIMING_MAX_RECORDS_ENV = "VLLM_METAX_DSPARK_PHASE_TIMING_MAX_RECORDS"
_PATCH_MARKER = "__vllm_metax_dspark_cycle_phase__"
_EVENT_TIMINGS: list[dict[str, Any]] = []
_FLUSH_REGISTERED = False
_TIMING_COMPLETE = False


def _phase_timing_enabled() -> bool:
    return bool(os.getenv(_PHASE_TIMING_DIR_ENV))


def _runtime_rank() -> int:
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if rank is not None:
        return int(rank)
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        return int(distributed.get_rank())
    return 0


def _record_event_timing(phase: str, call: Any) -> Any:
    global _FLUSH_REGISTERED, _TIMING_COMPLETE

    if _TIMING_COMPLETE:
        return call()

    capture_probe = getattr(torch.cuda, "is_current_stream_capturing", None)
    if capture_probe is not None and capture_probe():
        return call()

    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start_ns = time.perf_counter_ns()
    start.record(stream)
    try:
        return call()
    finally:
        end.record(stream)
        host_end_ns = time.perf_counter_ns()
        _EVENT_TIMINGS.append(
            {
                "phase": phase,
                "rank": _runtime_rank(),
                "pid": os.getpid(),
                "stream": int(stream.cuda_stream),
                "host_start_ns": host_start_ns,
                "host_end_ns": host_end_ns,
                "start_event": start,
                "end_event": end,
            }
        )
        if not _FLUSH_REGISTERED:
            atexit.register(_flush_event_timings)
            _FLUSH_REGISTERED = True
        max_records = int(os.getenv(_PHASE_TIMING_MAX_RECORDS_ENV, "0"))
        if max_records > 0 and len(_EVENT_TIMINGS) >= max_records:
            _flush_event_timings()
            _TIMING_COMPLETE = True


def _flush_event_timings() -> Path | None:
    if not _EVENT_TIMINGS:
        return None
    output_dir = os.getenv(_PHASE_TIMING_DIR_ENV)
    if not output_dir:
        return None

    torch.cuda.synchronize()
    path = Path(output_dir) / f"rank{os.getenv('RANK', '0')}-pid{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for sequence, timing in enumerate(_EVENT_TIMINGS):
            record = {
                key: value
                for key, value in timing.items()
                if key not in {"start_event", "end_event"}
            }
            record["sequence"] = sequence
            record["host_ms"] = (
                record["host_end_ns"] - record["host_start_ns"]
            ) / 1_000_000
            record["cuda_ms"] = timing["start_event"].elapsed_time(
                timing["end_event"]
            )
            output.write(json.dumps(record, sort_keys=True) + "\n")
    _EVENT_TIMINGS.clear()
    return path


def _reset_event_timings_for_test() -> None:
    global _TIMING_COMPLETE
    _EVENT_TIMINGS.clear()
    _TIMING_COMPLETE = False


def _wrap_method(
    owner: type, method_name: str, phase: str, *, profile_range: bool = True
) -> None:
    original = getattr(owner, method_name)
    if getattr(original, _PATCH_MARKER, None) == phase:
        return

    @wraps(original)
    def profiled(*args: Any, **kwargs: Any) -> Any:
        profile_context = (
            torch.profiler.record_function(f"dspark_cycle: {phase}")
            if profile_range or os.getenv(_PROFILE_PHASES_ENV) == "1"
            else nullcontext()
        )

        def call() -> Any:
            with profile_context:
                return original(*args, **kwargs)

        if _phase_timing_enabled():
            return _record_event_timing(phase, call)
        return call()

    setattr(profiled, _PATCH_MARKER, phase)
    setattr(owner, method_name, profiled)


def _install_patch() -> None:
    profile_range = os.getenv(_PROFILE_PHASES_ENV) == "1"
    if not profile_range and not _phase_timing_enabled():
        return
    _wrap_method(
        DeepseekV4ForCausalLM,
        "forward",
        "target_forward",
        profile_range=profile_range,
    )
    _wrap_method(
        CudaGraphManager,
        "run_pw_graph",
        "target_pw_graph",
        profile_range=profile_range,
    )
    _wrap_method(
        GPUModelRunner, "sample", "target_accept", profile_range=profile_range
    )
    _wrap_method(
        DSparkDeepseekV4Model,
        "precompute_and_store_context_kv",
        "draft_context_kv",
        profile_range=profile_range,
    )
    _wrap_method(
        DSparkDeepseekV4Model,
        "forward",
        "draft_backbone",
        profile_range=profile_range,
    )
    _wrap_method(
        DSparkSpeculator,
        "_sample_sequential",
        "draft_sample",
        profile_range=profile_range,
    )


_install_patch()
