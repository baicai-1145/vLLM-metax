# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-input capture for DeepSeek V4 sparse MLA prefill and decode.

This module is intentionally inert unless ``VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR``
is set.  Captures are written after the native call has returned so enabling the
hook cannot change dispatch or provide a fallback for a failed native kernel.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch


_CAPTURE_CALL_COUNT = 0
_CAPTURE_BUCKET_COUNTS: dict[tuple[str, str, str, int | None], int] = {}
_CAPTURE_SKIP_COUNTS: dict[tuple[str, str], int] = {}
_CAPTURE_LOCK = threading.Lock()


def _rank() -> str:
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if rank is not None:
        return rank
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return str(os.getpid())


def _capture_dir() -> Path | None:
    path = os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR")
    return Path(path) if path else None


def _is_cuda_graph_capturing() -> bool:
    """Return whether the current stream is being CUDA-graph captured.

    The probe is intentionally best-effort: importing this debug hook must
    remain harmless on hosts without CUDA, while an unknown result fails
    closed so it cannot perform copies or filesystem work during capture.
    """
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _rank_enabled(rank: str) -> bool:
    ranks = os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_RANKS", "all")
    if not ranks or ranks.strip().lower() == "all":
        return True
    return rank in {item.strip() for item in ranks.split(",") if item.strip()}


def _capture_list(name: str, default: set[str]) -> set[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _stage_enabled(stage: str) -> bool:
    return stage in _capture_list(
        "VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_STAGES", {"prefill", "decode"}
    )


def _decode_mode_enabled(mode: str) -> bool:
    return mode in _capture_list(
        "VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DECODE_MODES",
        {"swa", "dual", "topk", "compat"},
    )


def _per_ratio_max_calls() -> int:
    value = os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_PER_RATIO_MAX_CALLS")
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _skip_calls() -> int:
    value = os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_SKIP_CALLS", "0")
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _max_calls() -> int:
    value = os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_MAX_CALLS", "0")
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def reset_sparse_mla_capture_state() -> None:
    """Reset the process-local call budget, primarily for tests."""
    global _CAPTURE_CALL_COUNT
    with _CAPTURE_LOCK:
        _CAPTURE_CALL_COUNT = 0
        _CAPTURE_BUCKET_COUNTS.clear()
        _CAPTURE_SKIP_COUNTS.clear()


def _tensor_meta(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.stride()),
    }


def _clone_to_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().contiguous().cpu()


def _synchronize_capture_stream(q: torch.Tensor) -> None:
    """Wait for the producing stream before copying captured tensors to CPU."""
    if not q.is_cuda:
        return
    torch.cuda.current_stream(q.device).synchronize()


def _caller_out_meta(value: torch.Tensor | None) -> dict[str, Any] | None:
    metadata = _tensor_meta(value)
    if metadata is None:
        return None
    metadata["data_ptr"] = int(value.data_ptr())
    return metadata


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reserve_capture_path(capture_dir: Path, rank: str, call: int) -> tuple[Path, int]:
    """Reserve a unique destination without overwriting an existing capture."""
    candidate = call
    while True:
        path = capture_dir / f"rank{rank}_call{candidate}.pt"
        try:
            # Reserve the final name atomically.  The placeholder is replaced
            # by _atomic_torch_save after the payload has been assembled.
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            return path, candidate
        except FileExistsError:
            candidate += 1


def _capture_preflight(stage: str) -> tuple[Path, str] | None:
    """Check cheap, side-effect-free capture filters before tensor inspection."""
    capture_dir = _capture_dir()
    if capture_dir is None or _is_cuda_graph_capturing():
        return None
    if not _stage_enabled(stage):
        return None
    rank = _rank()
    if not _rank_enabled(rank):
        return None
    return capture_dir, rank


def _prepare_capture(
    stage: str,
    mode: str,
    compress_ratio: int | None,
) -> tuple[Path, str, int] | None:
    """Run the shared opt-in checks and reserve one capture call/path."""
    # Keep this first check before metadata inspection or any tensor operation:
    # the normal serving path must not incur CPU copies when capture is off.
    preflight = _capture_preflight(stage)
    if preflight is None:
        return None
    capture_dir, rank = preflight

    global _CAPTURE_CALL_COUNT
    with _CAPTURE_LOCK:
        skip_calls = _skip_calls()
        skip_bucket = (rank, stage)
        skipped = _CAPTURE_SKIP_COUNTS.get(skip_bucket, 0)
        if skipped < skip_calls:
            _CAPTURE_SKIP_COUNTS[skip_bucket] = skipped + 1
            return None
        max_calls = _max_calls()
        if max_calls and _CAPTURE_CALL_COUNT >= max_calls:
            return None
        bucket = (rank, stage, mode, compress_ratio)
        per_ratio_max = _per_ratio_max_calls()
        if per_ratio_max and _CAPTURE_BUCKET_COUNTS.get(bucket, 0) >= per_ratio_max:
            return None
        call = _CAPTURE_CALL_COUNT
        _CAPTURE_CALL_COUNT += 1
        _CAPTURE_BUCKET_COUNTS[bucket] = _CAPTURE_BUCKET_COUNTS.get(bucket, 0) + 1

    capture_dir.mkdir(parents=True, exist_ok=True)
    path, call = _reserve_capture_path(capture_dir, rank, call)
    return path, rank, call


def _has_positive_lens(lens: torch.Tensor | None) -> bool:
    if lens is None or lens.numel() == 0:
        return False
    try:
        return bool(torch.any(lens > 0).item())
    except (RuntimeError, TypeError):
        return False


def _classify_decode_mode(
    *,
    swa_cache: torch.Tensor | None,
    compressed_cache: torch.Tensor | None,
    swa_indices: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
    swa_lens: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
) -> str | None:
    swa_active = (
        swa_cache is not None
        and swa_indices is not None
        and _has_positive_lens(swa_lens)
    )
    topk_active = (
        compressed_cache is not None
        and topk_indices is not None
        and _has_positive_lens(topk_lens)
    )
    if swa_active and topk_active:
        return "dual"
    if swa_active:
        return "swa"
    if topk_active:
        return "topk"
    if swa_indices is not None or topk_indices is not None:
        return "compat"
    return None


@torch.no_grad()
def maybe_capture_sparse_mla_prefill(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    out: torch.Tensor | None,
    output: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    compress_ratio: int | None = None,
) -> None:
    """Capture one completed native sparse MLA prefill call when enabled."""
    if _capture_preflight("prefill") is None:
        return
    prepared = _prepare_capture("prefill", "prefill", compress_ratio)
    if prepared is None:
        return
    path, rank, call = prepared
    _synchronize_capture_stream(q)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "prefill",
        "rank": int(rank) if rank.isdigit() else rank,
        "call": call,
        "q": _clone_to_cpu(q),
        "kv": _clone_to_cpu(kv),
        "indices": _clone_to_cpu(indices),
        "topk_length": _clone_to_cpu(topk_length),
        "attn_sink": _clone_to_cpu(attn_sink),
        "sm_scale": float(sm_scale),
        "d_v": int(d_v),
        "input_meta": {
            "q": _tensor_meta(q),
            "kv": _tensor_meta(kv),
            "indices": _tensor_meta(indices),
            "topk_length": _tensor_meta(topk_length),
            "attn_sink": _tensor_meta(attn_sink),
        },
        "caller_out": _caller_out_meta(out),
        "output": _clone_to_cpu(output),
        "max_logits": _clone_to_cpu(max_logits),
        "lse": _clone_to_cpu(lse),
    }
    # Keep a failed reservation in place rather than allowing a later capture
    # to overwrite a potentially useful artifact.
    _atomic_torch_save(payload, path)


@torch.no_grad()
def maybe_capture_sparse_mla_decode(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    swa_lens: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    swa_block_table: torch.Tensor | None = None,
    compressed_block_table: torch.Tensor | None = None,
    swa_block_size: int | None = None,
    compressed_block_size: int | None = None,
    compress_ratio: int | None = None,
    window_size: int | None = None,
    decode_backend: str = "native",
    native_decode_mode: str = "torch_compat",
) -> None:
    """Capture one completed Torch sparse MLA decode oracle call.

    Cache tensors are copied only when capture is explicitly enabled.  Their
    original shape/stride metadata is retained even though serialized tensor
    values are made contiguous for portable replay.
    """
    if _capture_preflight("decode") is None:
        return
    mode = _classify_decode_mode(
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        swa_lens=swa_lens,
        topk_lens=topk_lens,
    )
    if mode is None or not _decode_mode_enabled(mode):
        return
    prepared = _prepare_capture("decode", mode, compress_ratio)
    if prepared is None:
        return
    path, rank, call = prepared
    _synchronize_capture_stream(q)

    tensors = {
        "q": q,
        "swa_cache": swa_cache,
        "compressed_cache": compressed_cache,
        "swa_indices": swa_indices,
        "topk_indices": topk_indices,
        "swa_lens": swa_lens,
        "topk_lens": topk_lens,
        "attn_sink": attn_sink,
        "swa_block_table": swa_block_table,
        "compressed_block_table": compressed_block_table,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "decode",
        "decode_mode": mode,
        "decode_backend": decode_backend,
        "native_decode_mode": native_decode_mode,
        "rank": int(rank) if rank.isdigit() else rank,
        "call": call,
        "q": _clone_to_cpu(q),
        "swa_cache": _clone_to_cpu(swa_cache),
        "compressed_cache": _clone_to_cpu(compressed_cache),
        "swa_indices": _clone_to_cpu(swa_indices),
        "topk_indices": _clone_to_cpu(topk_indices),
        "swa_lens": _clone_to_cpu(swa_lens),
        "topk_lens": _clone_to_cpu(topk_lens),
        "swa_block_table": _clone_to_cpu(swa_block_table),
        "compressed_block_table": _clone_to_cpu(compressed_block_table),
        "sm_scale": float(sm_scale),
        "d_v": int(d_v),
        "head_dim": int(q.shape[-1]),
        "q_heads": int(q.shape[-2]),
        "attn_sink": _clone_to_cpu(attn_sink),
        "input_meta": {name: _tensor_meta(value) for name, value in tensors.items()},
        "cache_meta": {
            "swa_block_size": (
                int(swa_block_size)
                if swa_block_size is not None
                else (int(swa_cache.shape[1]) if swa_cache.ndim > 1 else None)
            ),
            "compressed_block_size": (
                int(compressed_block_size)
                if compressed_block_size is not None
                else (
                    int(compressed_cache.shape[1])
                    if compressed_cache is not None and compressed_cache.ndim > 1
                    else None
                )
            ),
            "compress_ratio": compress_ratio,
            "window_size": window_size,
        },
        "caller_out": _caller_out_meta(output),
        "output": _clone_to_cpu(output),
    }
    _atomic_torch_save(payload, path)


__all__ = [
    "maybe_capture_sparse_mla_prefill",
    "maybe_capture_sparse_mla_decode",
    "reset_sparse_mla_capture_state",
]
