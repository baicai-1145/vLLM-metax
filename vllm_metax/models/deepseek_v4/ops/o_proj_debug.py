# SPDX-License-Identifier: Apache-2.0
"""Opt-in DeepSeek V4 O-projection capture and offline replay helpers."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SCHEMA_VERSION = 1
_CAPTURE_CALL_COUNT = 0
_CAPTURE_SKIP_COUNTS: dict[str, int] = {}
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
    value = os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR")
    return Path(value) if value else None


def _is_cuda_graph_capturing() -> bool:
    # Unknown is treated as capturing so this hook cannot inject host work into
    # a graph capture on platforms with incomplete CUDA compatibility shims.
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _rank_enabled(rank: str) -> bool:
    value = os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_RANKS", "all")
    if not value or value.strip().lower() == "all":
        return True
    return rank in {item.strip() for item in value.split(",") if item.strip()}


def _max_calls() -> int:
    try:
        return max(0, int(os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_MAX_CALLS", "0")))
    except ValueError:
        return 0


def _skip_calls() -> int:
    try:
        return max(0, int(os.getenv("VLLM_METAX_DSV4_O_PROJ_CAPTURE_SKIP_CALLS", "0")))
    except ValueError:
        return 0


def reset_o_proj_capture_state() -> None:
    global _CAPTURE_CALL_COUNT
    with _CAPTURE_LOCK:
        _CAPTURE_CALL_COUNT = 0
        _CAPTURE_SKIP_COUNTS.clear()


def _tensor_meta(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.stride()),
    }


def _clone_to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def _synchronize_capture_stream(value: torch.Tensor) -> None:
    if value.is_cuda:
        torch.cuda.current_stream(value.device).synchronize()


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


def _reserve_path(directory: Path, rank: str, call: int) -> tuple[Path, int]:
    candidate = call
    while True:
        path = directory / f"rank{rank}_call{candidate}.pt"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            return path, candidate
        except FileExistsError:
            candidate += 1


def _compact_cache(
    cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only cache rows used by this call, preserving their positions."""
    pos = positions.detach().to(dtype=torch.long).reshape(-1)
    unique = torch.unique(pos, sorted=True)
    rows = cos_sin_cache.index_select(0, unique.to(cos_sin_cache.device))
    return rows, unique


def maybe_capture_o_proj(
    *,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    o_bf16: torch.Tensor,
    z: torch.Tensor,
    output: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> None:
    """Capture one completed native call when the explicit env hook is enabled."""
    # Keep this preflight before metadata/weight inspection or tensor copies.
    directory = _capture_dir()
    if directory is None or _is_cuda_graph_capturing():
        return
    rank = _rank()
    if not _rank_enabled(rank):
        return

    global _CAPTURE_CALL_COUNT
    with _CAPTURE_LOCK:
        skipped = _CAPTURE_SKIP_COUNTS.get(rank, 0)
        skip = _skip_calls()
        if skipped < skip:
            _CAPTURE_SKIP_COUNTS[rank] = skipped + 1
            return
        limit = _max_calls()
        if limit and _CAPTURE_CALL_COUNT >= limit:
            return
        call = _CAPTURE_CALL_COUNT
        _CAPTURE_CALL_COUNT += 1

    directory.mkdir(parents=True, exist_ok=True)
    path, call = _reserve_path(directory, rank, call)
    _synchronize_capture_stream(output)

    # Weight attributes are intentionally inspected only after capture is
    # enabled.  Linear modules are the production contract for these tensors.
    wo_a_weight = getattr(wo_a, "weight")
    wo_b_weight = getattr(wo_b, "weight")
    cache_rows, cache_positions = _compact_cache(cos_sin_cache, positions)
    tensors = {
        "o": _clone_to_cpu(o),
        "positions": _clone_to_cpu(positions),
        "cos_sin_cache": _clone_to_cpu(cache_rows),
        "cos_sin_positions": _clone_to_cpu(cache_positions),
        "wo_a": _clone_to_cpu(wo_a_weight),
        "wo_b": _clone_to_cpu(wo_b_weight),
        "o_bf16": _clone_to_cpu(o_bf16),
        "z": _clone_to_cpu(z),
        "output": _clone_to_cpu(output),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "op": "deepseek_v4_o_proj",
        "rank": int(rank) if rank.isdigit() else rank,
        "call": call,
        **tensors,
        "params": {
            "n_groups": int(n_groups),
            "heads_per_group": int(heads_per_group),
            "nope_dim": int(nope_dim),
            "rope_dim": int(rope_dim),
            "o_lora_rank": int(o_lora_rank),
        },
        "input_meta": {
            key: _tensor_meta(value)
            for key, value in {
                "o": o,
                "positions": positions,
                "cos_sin_cache": cos_sin_cache,
                "wo_a": wo_a_weight,
                "wo_b": wo_b_weight,
            }.items()
        },
        "intermediate_meta": {
            key: _tensor_meta(value)
            for key, value in {"o_bf16": o_bf16, "z": z, "output": output}.items()
        },
    }
    _atomic_torch_save(payload, path)


def load_capture(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{payload_path}: payload must be a mapping")
    required = {
        "schema_version", "op", "rank", "call", "o", "positions",
        "cos_sin_cache", "cos_sin_positions", "wo_a", "wo_b", "o_bf16",
        "z", "output", "params", "input_meta", "intermediate_meta",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{payload_path}: missing keys {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION or payload["op"] != "deepseek_v4_o_proj":
        raise ValueError(f"{payload_path}: unsupported O-proj schema")
    for key in required - {"schema_version", "op", "rank", "call", "params", "input_meta", "intermediate_meta"}:
        if not isinstance(payload[key], torch.Tensor) or payload[key].device.type != "cpu":
            raise ValueError(f"{payload_path}: {key} must be a CPU tensor")
    return payload


@torch.no_grad()
def _torch_inv_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    cos_sin_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens, heads, head_dim = o.shape
    if heads != n_groups * heads_per_group or head_dim != nope_dim + rope_dim:
        raise ValueError("O-proj input dimensions do not match params")
    if cos_sin_positions is not None:
        positions = positions.to(torch.long)
        lookup = {int(pos): index for index, pos in enumerate(cos_sin_positions.tolist())}
        row_ids = torch.tensor([lookup[int(pos)] for pos in positions.tolist()], device=o.device)
        cache = cos_sin_cache.to(o.device).index_select(0, row_ids)
    else:
        cache = cos_sin_cache.to(o.device).index_select(0, positions.to(torch.long))
    result = o.reshape(tokens, heads, head_dim).clone()
    rotated = result[..., nope_dim:]
    half = rope_dim // 2
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    cos = cache[..., :half].to(result.dtype).view(tokens, 1, half)
    sin = cache[..., half:].to(result.dtype).view(tokens, 1, half)
    rotated[..., 0::2] = even * cos + odd * sin
    rotated[..., 1::2] = odd * cos - even * sin
    return result.reshape(tokens, n_groups, heads_per_group * head_dim)


@torch.no_grad()
def torch_o_proj_trace(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    cos_sin_positions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    o_bf16 = _torch_inv_rope(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        cos_sin_positions=cos_sin_positions,
    )
    wo_a_bf16 = wo_a.to(o.device).view(n_groups, o_lora_rank, -1)
    z = torch.einsum("bhr,hdr->bhd", o_bf16, wo_a_bf16)
    output = torch.nn.functional.linear(z.flatten(1), wo_b.to(o.device))
    return {"o_bf16": o_bf16, "z": z, "output": output}


def replay_torch(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = payload["params"]
    return torch_o_proj_trace(
        payload["o"], payload["positions"], payload["cos_sin_cache"],
        payload["wo_a"], payload["wo_b"], cos_sin_positions=payload["cos_sin_positions"],
        **params,
    )


def replay_native(payload: dict[str, Any], device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    """Explicitly invoke the production native path; never used by serving."""
    from .o_proj import deep_gemm_bf16_o_proj
    from .fused_inv_rope_quant import inv_rope
    from vllm_metax.utils.deep_gemm import bf16_einsum

    params = payload["params"]
    target = torch.device(device) if device is not None else torch.device("cuda")
    o = payload["o"].to(target)
    positions = payload["positions"].to(target)
    # The compact cache is expanded into a lookup-sized tensor for native inv_rope.
    max_pos = int(positions.max().item()) + 1
    cache = torch.empty((max_pos, payload["cos_sin_cache"].shape[-1]), device=target,
                        dtype=payload["cos_sin_cache"].dtype)
    cache.zero_()
    cache.index_copy_(0, payload["cos_sin_positions"].to(target), payload["cos_sin_cache"].to(target))
    wo_a = nn.Linear(payload["wo_a"].shape[1], payload["wo_a"].shape[0], bias=False, device=target,
                     dtype=payload["wo_a"].dtype)
    wo_b = nn.Linear(payload["wo_b"].shape[1], payload["wo_b"].shape[0], bias=False, device=target,
                     dtype=payload["wo_b"].dtype)
    wo_a.weight.data.copy_(payload["wo_a"].to(target))
    wo_b.weight.data.copy_(payload["wo_b"].to(target))
    native_o_bf16 = inv_rope(o, positions, cache, **{
        key: params[key]
        for key in ("n_groups", "heads_per_group", "nope_dim", "rope_dim")
    })
    native_z = torch.empty(
        (o.shape[0], params["n_groups"], params["o_lora_rank"]),
        device=target,
        dtype=torch.bfloat16,
    )
    bf16_einsum(
        "bhr,hdr->bhd",
        native_o_bf16,
        wo_a.weight.view(params["n_groups"], params["o_lora_rank"], -1),
        native_z,
    )
    native_output = deep_gemm_bf16_o_proj(o, positions, cache, wo_a, wo_b, **params)
    return {"o_bf16": native_o_bf16, "z": native_z, "output": native_output}


def _raw_bits(value: torch.Tensor) -> torch.Tensor:
    if value.dtype in (torch.bfloat16, torch.float16):
        return value.view(torch.int16)
    if value.dtype == torch.float32:
        return value.view(torch.int32)
    return value


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return {"equal": False, "shape": [list(lhs.shape), list(rhs.shape)],
                "dtype": [str(lhs.dtype), str(rhs.dtype)]}
    left, right = lhs.detach().contiguous().cpu(), rhs.detach().contiguous().cpu()
    mask = _raw_bits(left) != _raw_bits(right)
    if not bool(mask.any()):
        return {"equal": True, "num_diff": 0}
    flat = int(mask.flatten().nonzero()[0].item())
    index = [int(v) for v in torch.unravel_index(torch.tensor(flat), mask.shape)]
    delta = (left.float() - right.float()).abs()
    lhs_bits = _raw_bits(left)[tuple(index)]
    rhs_bits = _raw_bits(right)[tuple(index)]
    return {
        "equal": False,
        "index": index,
        "lhs": left[tuple(index)].item(),
        "rhs": right[tuple(index)].item(),
        "lhs_bits": int(lhs_bits),
        "rhs_bits": int(rhs_bits),
        "num_diff": int(mask.sum()),
        "max_abs": float(delta.max()),
    }


def compare_trace(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    stages = {}
    for name in ("o_bf16", "z", "output"):
        if name in reference and name in candidate:
            stages[name] = tensor_diff(reference[name], candidate[name])
    return {"passed": all(item["equal"] for item in stages.values()), "stages": stages}


def replay_capture(path: str | Path, backend: str = "torch") -> dict[str, Any]:
    payload = load_capture(path)
    candidate = replay_torch(payload) if backend == "torch" else replay_native(payload)
    reference = {name: payload[name] for name in ("o_bf16", "z", "output") if name in payload}
    return {"path": str(path), "backend": backend, **compare_trace(reference, candidate)}


def _assert_trace(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> None:
    result = compare_trace(reference, candidate)
    if not result["passed"]:
        raise AssertionError(result)


def assert_bitwise_trace_equal(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> None:
    _assert_trace(reference, candidate)


__all__ = [
    "assert_bitwise_trace_equal",
    "compare_trace",
    "load_capture",
    "maybe_capture_o_proj",
    "replay_capture",
    "replay_native",
    "replay_torch",
    "reset_o_proj_capture_state",
    "tensor_diff",
    "torch_o_proj_trace",
]
