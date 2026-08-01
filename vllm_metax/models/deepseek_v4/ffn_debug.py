"""Opt-in eager-only DeepSeek V4 FFN/shared-expert stage capture.

The hook is inert unless ``VLLM_METAX_DSV4_FFN_CAPTURE_DIR`` is set.  A
context is installed around the existing MoE invocation so shared-expert
stages can be associated with the same layer/call as the routed output.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


SCHEMA_VERSION = 1
_DIR_ENV = "VLLM_METAX_DSV4_FFN_CAPTURE_DIR"
_RANKS_ENV = "VLLM_METAX_DSV4_FFN_CAPTURE_RANKS"
_LAYERS_ENV = "VLLM_METAX_DSV4_FFN_CAPTURE_LAYERS"
_CALLS_ENV = "VLLM_METAX_DSV4_FFN_CAPTURE_CALLS"
_CALLS: dict[int, int] = {}
_CALL_LOCK = threading.Lock()
_SAVE_LOCK = threading.Lock()
_LOCAL = threading.local()


def reset_ffn_capture_state() -> None:
    """Reset call counters and active contexts (primarily for tests)."""
    with _CALL_LOCK:
        _CALLS.clear()
    _LOCAL.stack = []


def _capture_dir() -> Path | None:
    value = os.getenv(_DIR_ENV)
    return Path(value) if value else None


def ffn_capture_enabled() -> bool:
    return _capture_dir() is not None


def _rank() -> int:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _is_cuda_graph_capturing() -> bool:
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _parse_nonnegative(name: str, value: str | None, default: set[int]) -> set[int]:
    if value is None or not value.strip():
        return default
    selected: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated set of nonnegative integers"
            ) from exc
        if index < 0:
            raise ValueError(
                f"{name} must be a comma-separated set of nonnegative integers"
            )
        selected.add(index)
    if not selected:
        raise ValueError(
            f"{name} must be a comma-separated set of nonnegative integers"
        )
    return selected


def _capture_ranks() -> set[int] | None:
    value = os.getenv(_RANKS_ENV)
    if value is None or not value.strip():
        return {0}
    if value.strip().lower() == "all":
        return None
    return _parse_nonnegative(_RANKS_ENV, value, {0})


def ffn_capture_layer_enabled(layer_idx: int) -> bool:
    return layer_idx in _parse_nonnegative(_LAYERS_ENV, os.getenv(_LAYERS_ENV), {0})


def _call_filter() -> set[int] | None:
    value = os.getenv(_CALLS_ENV)
    if value is None or not value.strip():
        return None
    return _parse_nonnegative(_CALLS_ENV, value, set())


def _next_call(layer_idx: int) -> int:
    with _CALL_LOCK:
        call = _CALLS.get(layer_idx, 0)
        _CALLS[layer_idx] = call + 1
    return call


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def _device_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().clone()


def _cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().contiguous().cpu().clone()


def _synchronize_capture_stream(value: torch.Tensor) -> None:
    if value.is_cuda:
        torch.cuda.current_stream(value.device).synchronize()


class FFNCaptureContext:
    def __init__(
        self,
        capture_dir: Path,
        rank: int,
        layer_idx: int,
        call: int,
        hidden_states: torch.Tensor | None,
    ) -> None:
        self.capture_dir = capture_dir
        self.rank = rank
        self.layer_idx = layer_idx
        self.call = call
        self._input = _device_clone(hidden_states)
        self._shared: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "FFNCaptureContext":
        stack = getattr(_LOCAL, "stack", None)
        if stack is None:
            stack = []
            _LOCAL.stack = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        stack = getattr(_LOCAL, "stack", [])
        if stack and stack[-1] is self:
            stack.pop()
        elif self in stack:
            stack.remove(self)

    def record_shared(self, stage: str, value: torch.Tensor) -> None:
        if stage in {
            "shared_input",
            "gate_up_proj_output",
            "activation_output",
            "shared_final_output",
        } and isinstance(value, torch.Tensor):
            self._shared[stage] = _device_clone(value)  # type: ignore[assignment]

    @torch.no_grad()
    def finish(self, output: torch.Tensor) -> tuple[Path, Path | None] | None:
        if _is_cuda_graph_capturing() or not isinstance(output, torch.Tensor):
            return None
        sync_value = self._shared.get("shared_final_output")
        if sync_value is None:
            sync_value = output
        _synchronize_capture_stream(sync_value)
        final_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "layer_idx": self.layer_idx,
            "call": self.call,
            "stage": "ffn_final",
            "ffn_input": _cpu_clone(self._input),
            "ffn_output": _cpu_clone(output),
            "tensor_meta": {
                "ffn_input": _tensor_meta(self._input) if self._input is not None else None,
                "ffn_output": _tensor_meta(output),
            },
        }
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.capture_dir / (
            f"rank{self.rank}_layer{self.layer_idx}_call{self.call}_final.pt"
        )
        shared_path: Path | None = None
        shared_payload: dict[str, Any] | None = None
        if self._shared:
            shared_payload = {
                "schema_version": SCHEMA_VERSION,
                "rank": self.rank,
                "layer_idx": self.layer_idx,
                "call": self.call,
                "stage": "shared_experts",
                **{name: _cpu_clone(value) for name, value in self._shared.items()},
                "tensor_meta": {name: _tensor_meta(value) for name, value in self._shared.items()},
            }
            shared_path = self.capture_dir / (
                f"rank{self.rank}_layer{self.layer_idx}_call{self.call}_shared.pt"
            )
        with _SAVE_LOCK:
            torch.save(final_payload, final_path)
            if shared_payload is not None and shared_path is not None:
                torch.save(shared_payload, shared_path)
        return final_path, shared_path


def active_ffn_capture(layer_idx: int | None) -> FFNCaptureContext | None:
    if layer_idx is None:
        return None
    for context in reversed(getattr(_LOCAL, "stack", [])):
        if context.layer_idx == layer_idx:
            return context
    return None


def maybe_prepare_ffn_capture(
    layer_idx: int | None, hidden_states: torch.Tensor | None = None
) -> FFNCaptureContext | None:
    """Reserve a layer/call context without synchronizing the host."""
    capture_dir = _capture_dir()
    if capture_dir is None or _is_cuda_graph_capturing() or layer_idx is None:
        return None
    if not ffn_capture_layer_enabled(layer_idx):
        return None
    rank = _rank()
    selected_ranks = _capture_ranks()
    if selected_ranks is not None and rank not in selected_ranks:
        return None
    call = _next_call(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None
    return FFNCaptureContext(capture_dir, rank, layer_idx, call, hidden_states)


@contextmanager
def maybe_ffn_capture_context(
    layer_idx: int | None, hidden_states: torch.Tensor | None = None
) -> Iterator[FFNCaptureContext | None]:
    context = maybe_prepare_ffn_capture(layer_idx, hidden_states)
    if context is None:
        yield None
        return
    with context:
        yield context
