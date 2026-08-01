# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-input capture for the DeepSeek-V4 INT8 indexer.

The capture path is deliberately inert unless ``VLLM_METAX_DSV4_INDEXER_CAPTURE_DIR``
is set.  It is intended for diagnosing native-indexer divergence, not for normal
serving or graph-captured execution.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import torch


CAPTURE_DIR_ENV = "VLLM_METAX_DSV4_INDEXER_CAPTURE_DIR"
CAPTURE_RANKS_ENV = "VLLM_METAX_DSV4_INDEXER_CAPTURE_RANKS"
CAPTURE_LAYERS_ENV = "VLLM_METAX_DSV4_INDEXER_CAPTURE_LAYERS"
CAPTURE_CALLS_ENV = "VLLM_METAX_DSV4_INDEXER_CAPTURE_CALLS"
CAPTURE_SEQ_LENS_ENV = "VLLM_METAX_DSV4_INDEXER_CAPTURE_SEQ_LENS"

_CALL_COUNTERS: dict[tuple[str, str], int] = {}
_CALL_COUNTERS_LOCK = threading.Lock()


def _capture_dir() -> str | None:
    value = os.getenv(CAPTURE_DIR_ENV, "").strip()
    if not value or value.lower() in {"0", "false", "off", "no"}:
        return None
    return value


def _is_cuda_graph_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (RuntimeError, AssertionError):
        # Capture must fail closed if the runtime cannot answer reliably.
        return True


def _rank() -> str:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is not None:
        return value
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return str(os.getpid())


def _csv_values(name: str) -> tuple[str, ...] | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def _matches_filters(rank: str, layer: str, call: int) -> bool:
    ranks = _csv_values(CAPTURE_RANKS_ENV)
    if ranks is not None and rank not in ranks:
        return False
    layers = _csv_values(CAPTURE_LAYERS_ENV)
    if layers is not None and not any(fragment in layer for fragment in layers):
        return False
    calls = _csv_values(CAPTURE_CALLS_ENV)
    if calls is not None:
        try:
            if call not in {int(item) for item in calls}:
                return False
        except ValueError:
            return False
    return True


def _decode_seq_lens_selected(seq_lens: torch.Tensor) -> bool:
    values = _csv_values(CAPTURE_SEQ_LENS_ENV)
    if values is None:
        return True
    try:
        selected = {int(value) for value in values}
    except ValueError as exc:
        raise ValueError(
            f"{CAPTURE_SEQ_LENS_ENV} must be a comma-separated set of integers"
        ) from exc
    return any(int(value) in selected for value in seq_lens.detach().reshape(-1))


def reset_call_counters() -> None:
    """Reset counters for tests or a new capture corpus."""
    with _CALL_COUNTERS_LOCK:
        _CALL_COUNTERS.clear()


def call_count(layer: str, rank: str | None = None) -> int:
    """Return the next call index for ``layer`` (or zero when unseen)."""
    key = (rank if rank is not None else _rank(), layer)
    with _CALL_COUNTERS_LOCK:
        return _CALL_COUNTERS.get(key, 0)


class CaptureContext:
    """Metadata for one selected indexer invocation."""

    def __init__(
        self,
        *,
        rank: str,
        layer: str,
        call: int,
        branch: str,
        num_tokens: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        topk_tokens: int,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.rank = int(rank) if rank.isdigit() else rank
        self.layer = layer
        self.call = call
        self.branch = branch
        self.num_tokens = num_tokens
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens
        self.hidden_states = hidden_states
        self.q_quant = q_quant
        self.weights = weights
        self.topk_tokens = topk_tokens
        self.slot_mapping = slot_mapping

    def _common(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "layer": self.layer,
            "call": self.call,
            "branch": self.branch,
            "num_tokens": self.num_tokens,
            "num_decode_tokens": self.num_decode_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "topk_tokens": self.topk_tokens,
            "hidden": _tensor_meta(self.hidden_states),
            "q_quant": _tensor_meta(self.q_quant),
            "weights": _tensor_meta(self.weights),
            "slot_mapping": _cpu_clone(self.slot_mapping),
            "last_token": {
                "hidden_states": _last_token(self.hidden_states),
                "q_quant": _last_token(self.q_quant),
                "weights": _last_token(self.weights),
            },
        }

    def save_prefill(
        self,
        *,
        path: Path,
        q_slice: torch.Tensor,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        weights_slice: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        block_table: torch.Tensor | None = None,
        native_logits: torch.Tensor,
        native_topk: torch.Tensor,
        chunk_index: int,
        token_start: int,
        token_end: int,
    ) -> None:
        if _is_cuda_graph_capturing():
            return
        payload = self._common()
        payload.update(
            {
                "chunk_index": chunk_index,
                "replay": {
                    "q_slice": _cpu_clone(q_slice),
                    "k_quant": _cpu_clone(k_quant),
                    "k_scale": _cpu_clone(k_scale),
                    "weights_slice": _cpu_clone(weights_slice),
                    "cu_seqlen_ks": _cpu_clone(cu_seqlen_ks),
                    "cu_seqlen_ke": _cpu_clone(cu_seqlen_ke),
                    "block_table": _cpu_clone(block_table)
                    if block_table is not None
                    else None,
                    "token_start": token_start,
                    "token_end": token_end,
                },
                "native_logits": _cpu_clone(native_logits),
                "native_topk": _cpu_clone(native_topk),
                "tensor_meta": {
                    "native_logits": _tensor_meta(native_logits),
                    "native_topk": _tensor_meta(native_topk),
                },
            }
        )
        save_cpu_payload(path, payload)

    def save_decode(
        self,
        *,
        path: Path,
        padded_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights_slice: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_metadata: torch.Tensor,
        decode_lens: torch.Tensor,
        native_logits: torch.Tensor,
        native_topk: torch.Tensor,
        requires_padding: bool,
        final_topk: torch.Tensor | None = None,
    ) -> None:
        if _is_cuda_graph_capturing():
            return
        if not _decode_seq_lens_selected(seq_lens):
            return
        payload = self._common()
        payload.update(
            {
                "replay": {
                    "padded_q": _cpu_clone(padded_q),
                    "kv_cache": _cpu_clone(kv_cache),
                    "weights_slice": _cpu_clone(weights_slice),
                    "seq_lens": _cpu_clone(seq_lens),
                    "block_table": _cpu_clone(block_table),
                    "schedule_metadata": _cpu_clone(schedule_metadata),
                    "decode_lens": _cpu_clone(decode_lens),
                    "requires_padding": requires_padding,
                },
                "native_logits": _cpu_clone(native_logits),
                "native_topk": _cpu_clone(native_topk),
                "final_topk": _cpu_clone(final_topk)
                if final_topk is not None
                else _cpu_clone(native_topk),
                "tensor_meta": {
                    "native_logits": _tensor_meta(native_logits),
                    "native_topk": _tensor_meta(native_topk),
                },
            }
        )
        save_cpu_payload(path, payload)


def begin_capture(
    *,
    layer: str,
    has_prefill: bool,
    has_decode: bool,
    num_tokens: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    slot_mapping: torch.Tensor,
) -> CaptureContext | None:
    """Select one invocation without copying tensors or touching the filesystem."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return None
    rank = _rank()
    key = (rank, layer)
    with _CALL_COUNTERS_LOCK:
        call = _CALL_COUNTERS.get(key, 0)
        _CALL_COUNTERS[key] = call + 1
    if not _matches_filters(rank, layer, call):
        return None
    if has_prefill and has_decode:
        branch = "mixed"
    elif has_prefill:
        branch = "prefill"
    else:
        branch = "decode"
    return CaptureContext(
        rank=rank,
        layer=layer,
        call=call,
        branch=branch,
        num_tokens=num_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        hidden_states=hidden_states,
        q_quant=q_quant,
        weights=weights,
        topk_tokens=topk_tokens,
        slot_mapping=slot_mapping,
    )


def capture_path(context: CaptureContext, suffix: str = "") -> Path:
    root = _capture_dir()
    if root is None:
        raise RuntimeError("capture_path called while capture is disabled")
    suffix_part = f"_{suffix}" if suffix else ""
    return Path(root) / (
        f"rank{context.rank}_layer{_safe_name(context.layer)}"
        f"_call{context.call}{suffix_part}.pt"
    )


def save_cpu_payload(path: Path, payload: Any) -> None:
    """Recursively snapshot tensors to CPU and serialize one payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpuize(payload), path)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _cpu_clone(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    return value.detach().to(device="cpu").clone()


def _cpuize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu_clone(value)
    if isinstance(value, dict):
        return {key: _cpuize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpuize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpuize(item) for item in value)
    return value


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def _last_token(value: torch.Tensor) -> dict[str, Any]:
    token = value.reshape(1) if value.ndim == 0 or value.shape[0] == 0 else value[-1:]
    token_cpu = _cpu_clone(token)
    raw = token_cpu.contiguous().view(torch.uint8).numpy().tobytes()
    return {"tensor": token_cpu, "sha256": hashlib.sha256(raw).hexdigest()}
