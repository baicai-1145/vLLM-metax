# SPDX-License-Identifier: Apache-2.0
"""Opt-in post-execute diagnostics for CUDA-graph replay investigation."""

import os
import tempfile
from pathlib import Path

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


_DIR_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR"
_RANKS_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_RANKS"
_CALLS_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_CALLS"
_REQUEST_IDS_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_REQUEST_IDS"
_OUTPUT_TOKENS_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_OUTPUT_TOKENS"
_COUNTER_ATTR = "_vllm_metax_graph_capture_call"
_SCHEMA = "vllm_metax.dsv4_graph_capture.v1"
_PATCH_MARKER = "__vllm_metax_graph_capture"
_LAYER_OUTPUT_NAMES = ("hidden_states", "residual", "post_mix", "res_mix")

_ORIGINAL_EXECUTE_MODEL = getattr(
    GPUModelRunner.execute_model,
    "__vllm_metax_graph_capture_original__",
    GPUModelRunner.execute_model,
)


def _parse_filter(name: str) -> set[int] | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers or all") from exc


def _parse_string_filter(name: str) -> set[str] | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _rank() -> int:
    try:
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(os.environ.get("RANK", "0"))


def _capture_active() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        # Unknown capture state must fail closed: never inject host work into
        # a graph capture on an incomplete CUDA-compatibility runtime.
        return True


def _to_cpu(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().to(device="cpu").clone()


def _logit_summary(logits):
    cpu_logits = _to_cpu(logits)
    if cpu_logits is None:
        return cpu_logits, None, None, None
    if cpu_logits.ndim == 0:
        return cpu_logits, cpu_logits.reshape(1), torch.zeros(1, dtype=torch.long), torch.zeros((), dtype=torch.long)
    width = cpu_logits.shape[-1]
    if width == 0:
        return cpu_logits, None, None, None
    values, indices = torch.topk(cpu_logits, k=min(2, width), dim=-1)
    return cpu_logits, values, indices, torch.argmax(cpu_logits, dim=-1)


def _pre_hc_hidden_states(self, row_source):
    if not isinstance(row_source, torch.Tensor):
        return None
    get_model = getattr(self, "get_model", None)
    if not callable(get_model):
        return None
    try:
        model = get_model()
        get_target_hidden_states = getattr(model, "get_mtp_target_hidden_states")
        buffer = get_target_hidden_states()
    except Exception:
        return None
    if not isinstance(buffer, torch.Tensor):
        return None
    rows = row_source.shape[0] if row_source.ndim >= 2 else 1
    selected = buffer if buffer.ndim == 0 else buffer[:rows]
    return _to_cpu(selected)


def _layer_outputs(self) -> dict[str, dict]:
    get_model = getattr(self, "get_model", None)
    if not callable(get_model):
        return {}
    try:
        modules = get_model().named_modules()
    except Exception:
        return {}
    outputs: dict[str, dict] = {}
    for module_name, module in modules:
        if not getattr(module, "_graph_capture_output_enabled", False):
            continue
        get_buffers = getattr(module, "get_graph_capture_output_buffers", None)
        refs = get_buffers() if callable(get_buffers) else None
        if not isinstance(refs, tuple) or len(refs) != len(_LAYER_OUTPUT_NAMES):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(layer_idx, int):
            continue
        get_stages = getattr(module, "get_graph_capture_stage_buffers", None)
        stage_refs = get_stages() if callable(get_stages) else {}
        get_mhc_input = getattr(module, "get_graph_capture_mhc_input", None)
        mhc_input = get_mhc_input() if callable(get_mhc_input) else None
        outputs[str(layer_idx)] = {
            "module": module_name,
            "before_mhc_hidden_states": _to_cpu(mhc_input),
            **{
                name: _to_cpu(value)
                for name, value in zip(_LAYER_OUTPUT_NAMES, refs, strict=True)
            },
            "stages": {
                stage: {
                    name: _to_cpu(value)
                    for name, value in zip(
                        _LAYER_OUTPUT_NAMES, values, strict=True
                    )
                }
                for stage, values in stage_refs.items()
            },
        }
    return outputs


def _weak_layer_outputs(self) -> dict[str, dict]:
    get_model = getattr(self, "get_model", None)
    if not callable(get_model):
        return {}
    try:
        modules = get_model().named_modules()
    except Exception:
        return {}
    outputs: dict[str, dict] = {}
    for module_name, module in modules:
        get_buffers = getattr(module, "get_graph_weak_stage_buffers", None)
        if not callable(get_buffers):
            continue
        try:
            stage_refs = get_buffers()
        except Exception:
            continue
        if not isinstance(stage_refs, dict):
            continue
        stages = {}
        for stage, values in stage_refs.items():
            if not isinstance(stage, str) or not isinstance(values, tuple):
                continue
            if len(values) != len(_LAYER_OUTPUT_NAMES) or not all(
                isinstance(value, torch.Tensor) for value in values
            ):
                continue
            stages[stage] = {
                name: _to_cpu(value)
                for name, value in zip(_LAYER_OUTPUT_NAMES, values, strict=True)
            }
        layer_idx = getattr(module, "layer_idx", None)
        if stages and isinstance(layer_idx, int):
            outputs[str(layer_idx)] = {
                "module": module_name,
                "stages": stages,
            }
    return outputs


def _scheduled_requests(state) -> list[dict]:
    scheduler_output = getattr(state, "scheduler_output", None)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = getattr(cached, "req_ids", ())
    output_tokens = getattr(cached, "num_output_tokens", ())
    computed_tokens = getattr(cached, "num_computed_tokens", ())
    return [
        {
            "request_id": str(request_id),
            "num_output_tokens": int(num_output),
            "num_computed_tokens": int(num_computed),
        }
        for request_id, num_output, num_computed in zip(
            req_ids, output_tokens, computed_tokens, strict=True
        )
    ]


def _request_selected(state) -> bool:
    request_ids = _parse_string_filter(_REQUEST_IDS_ENV)
    output_tokens = _parse_filter(_OUTPUT_TOKENS_ENV)
    if request_ids is None and output_tokens is None:
        return True

    def request_id_matches(actual: str) -> bool:
        return request_ids is None or any(
            actual == selected or actual.startswith(f"{selected}-")
            for selected in request_ids
        )

    return any(
        request_id_matches(item["request_id"])
        and (output_tokens is None or item["num_output_tokens"] in output_tokens)
        for item in _scheduled_requests(state)
    )


def _write_payload(directory: str, payload: dict) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    rank = payload["rank"]
    call = payload["call"]
    destination = path / f"rank{rank}_call{call}.pt"
    fd, temporary = tempfile.mkstemp(
        dir=path, prefix=f".rank{rank}_call{call}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _capture_after_execute(self, directory: str, call: int, rank: int) -> None:
    torch.cuda.current_stream().synchronize()
    state = getattr(self, "execute_model_state", None)
    logits = getattr(state, "logits", None) if state is not None else None
    sample_hidden_states = (
        getattr(state, "sample_hidden_states", None) if state is not None else None
    )
    hidden_states = (
        getattr(state, "hidden_states", None) if state is not None else None
    )
    row_source = (
        hidden_states if isinstance(hidden_states, torch.Tensor) else sample_hidden_states
    )
    pre_hc_hidden_states = _pre_hc_hidden_states(self, row_source)
    cpu_logits, top2_values, top2_indices, argmax = _logit_summary(logits)
    _write_payload(
        directory,
        {
            "schema": _SCHEMA,
            "rank": rank,
            "call": call,
            "sample_hidden_states": _to_cpu(sample_hidden_states),
            "pre_hc_hidden_states": pre_hc_hidden_states,
            "layer_outputs": _layer_outputs(self),
            "weak_layer_outputs": _weak_layer_outputs(self),
            "scheduled_requests": _scheduled_requests(state),
            "logits": cpu_logits,
            "top2_values": top2_values,
            "top2_indices": top2_indices,
            "argmax": argmax,
        },
    )


def _execute_model(self, *args, **kwargs):
    directory = os.environ.get(_DIR_ENV)
    if not directory:
        return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)
    if _capture_active():
        return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)

    result = _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)
    if _capture_active():
        return result

    call = int(getattr(self, _COUNTER_ATTR, 0))
    setattr(self, _COUNTER_ATTR, call + 1)
    rank = _rank()
    ranks = _parse_filter(_RANKS_ENV)
    calls = _parse_filter(_CALLS_ENV)
    state = getattr(self, "execute_model_state", None)
    if (
        (ranks is None or rank in ranks)
        and (calls is None or call in calls)
        and _request_selected(state)
    ):
        _capture_after_execute(self, directory, call, rank)
    return result


_execute_model.__vllm_metax_graph_capture = True


def _install_patch() -> None:
    """Install only when enabled before Python/plugin initialization."""
    global _ORIGINAL_EXECUTE_MODEL
    if not os.environ.get(_DIR_ENV):
        return
    if getattr(GPUModelRunner.execute_model, _PATCH_MARKER, False):
        return
    _ORIGINAL_EXECUTE_MODEL = GPUModelRunner.execute_model
    _execute_model.__vllm_metax_graph_capture_original__ = _ORIGINAL_EXECUTE_MODEL
    GPUModelRunner.execute_model = _execute_model


_install_patch()
