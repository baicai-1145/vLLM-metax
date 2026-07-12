#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect and replay captured DeepSeek V4 sparse MLA payloads.

The replay is deliberately an offline Torch oracle.  It does not load a model,
invoke the native kernel, or provide a serving fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import re
from pathlib import Path
from typing import Any

import torch


_CORPUS_RE = re.compile(r"rank(?P<rank>\d+)_call(?P<call>\d+)\.pt$")
_PREFILL_REQUIRED_KEYS = {
    "schema_version",
    "stage",
    "rank",
    "call",
    "q",
    "kv",
    "indices",
    "topk_length",
    "attn_sink",
    "sm_scale",
    "d_v",
    "input_meta",
    "caller_out",
    "output",
    "max_logits",
    "lse",
}
_INPUT_META_KEYS = {"q", "kv", "indices", "topk_length", "attn_sink"}
_DECODE_REQUIRED_KEYS = {
    "schema_version",
    "stage",
    "rank",
    "call",
    "q",
    "swa_cache",
    "compressed_cache",
    "swa_indices",
    "topk_indices",
    "swa_lens",
    "topk_lens",
    "swa_block_table",
    "compressed_block_table",
    "sm_scale",
    "d_v",
    "head_dim",
    "q_heads",
    "attn_sink",
    "input_meta",
    "cache_meta",
    "caller_out",
    "output",
}
_DECODE_INPUT_META_KEYS = {
    "q",
    "swa_cache",
    "compressed_cache",
    "swa_indices",
    "topk_indices",
    "swa_lens",
    "topk_lens",
    "attn_sink",
    "swa_block_table",
    "compressed_block_table",
}
_LOG2E = math.log2(math.e)


def _file_key(path: Path) -> tuple[int, int]:
    match = _CORPUS_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"malformed corpus filename: {path}")
    return int(match.group("rank")), int(match.group("call"))


def iter_corpus_files(corpus: str | Path) -> list[Path]:
    """Return all payloads in numeric ``(rank, call)`` order."""
    corpus_path = Path(corpus)
    if not corpus_path.is_dir():
        raise ValueError(f"corpus is not a directory: {corpus_path}")
    files = sorted(corpus_path.glob("*.pt"), key=_file_key)
    keys = [_file_key(path) for path in files]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate rank/call payload in {corpus_path}")
    return files


def _require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ValueError(f"{path}: {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_meta(meta: Any, path: Path, key: str, *, caller: bool = False) -> None:
    _require(isinstance(meta, dict), path, f"input_meta[{key}] must be a mapping")
    expected_keys = {"shape", "dtype", "strides"}
    if caller:
        expected_keys.add("data_ptr")
    _require(set(meta) == expected_keys, path, f"input_meta[{key}] keys")
    _require(isinstance(meta["shape"], list), path, f"input_meta[{key}].shape")
    _require(isinstance(meta["strides"], list), path, f"input_meta[{key}].strides")
    _require(isinstance(meta["dtype"], str), path, f"input_meta[{key}].dtype")
    if caller:
        _require(_is_int(meta["data_ptr"]) and meta["data_ptr"] > 0, path,
                 "caller_out.data_ptr")


def _validate_rank_call(payload: dict[str, Any], path: Path) -> None:
    _require(payload["schema_version"] == 1, path, "schema_version must be 1")
    file_rank, file_call = _file_key(path)
    _require(_is_int(payload["rank"]) and payload["rank"] == file_rank, path,
             "rank does not match filename")
    _require(_is_int(payload["call"]) and payload["call"] == file_call, path,
             "call does not match filename")


def _validate_payload(payload: Any, path: Path) -> None:
    _require(isinstance(payload, dict), path, "payload must be a mapping")
    stage = payload.get("stage")
    if stage is None:
        raise ValueError(f"{path}: keys={sorted(payload)}")
    if stage == "prefill":
        _validate_prefill_payload(payload, path)
    elif stage == "decode":
        _validate_decode_payload(payload, path)
    else:
        raise ValueError(f"{path}: stage must be prefill or decode")


def _validate_prefill_payload(payload: dict[str, Any], path: Path) -> None:
    _require(set(payload) == _PREFILL_REQUIRED_KEYS, path, f"keys={sorted(payload)}")
    _require(payload["stage"] == "prefill", path, "stage must be prefill")
    _validate_rank_call(payload, path)

    for key in ("q", "kv", "indices", "output", "max_logits", "lse"):
        _require(isinstance(payload[key], torch.Tensor), path, f"{key} is not tensor")
        _require(payload[key].device.type == "cpu", path, f"{key} must be CPU tensor")
    q, kv, indices = payload["q"], payload["kv"], payload["indices"]
    _require(q.ndim == 3 and q.shape[0] > 0 and q.shape[1] > 0 and q.shape[2] > 0,
             path, "q must have shape (tokens, heads, head_dim)")
    _require(kv.ndim == 3 and kv.shape[1] == 1 and kv.shape[0] > 0,
             path, "kv must have shape (kv_tokens, 1, head_dim)")
    _require(kv.shape[2] == q.shape[2], path, "q/kv head dimensions differ")
    _require(indices.ndim == 3 and indices.shape[0] == q.shape[0] and indices.shape[1] == 1
             and indices.shape[2] > 0,
             path, "indices must have shape (tokens, 1, topk)")
    _require(indices.dtype in (torch.int32, torch.int64), path, "indices dtype")
    topk_length = payload["topk_length"]
    if topk_length is not None:
        _require(isinstance(topk_length, torch.Tensor), path, "topk_length is not tensor")
        _require(topk_length.device.type == "cpu" and topk_length.ndim == 1 and
                 topk_length.shape[0] == q.shape[0], path, "topk_length shape/device")
        _require(topk_length.dtype in (torch.int32, torch.int64), path, "topk_length dtype")
        _require(bool(torch.all((topk_length >= 0) & (topk_length <= indices.shape[2]))),
                 path, "topk_length values must be in [0, topk]")
    attn_sink = payload["attn_sink"]
    if attn_sink is not None:
        _require(isinstance(attn_sink, torch.Tensor), path, "attn_sink is not tensor")
        _require(attn_sink.device.type == "cpu", path, "attn_sink must be CPU tensor")
        _require(attn_sink.dtype == torch.float32, path, "attn_sink dtype must be float32")
        _require(attn_sink.ndim == 1 and attn_sink.shape[0] == q.shape[1], path,
                 "attn_sink must have shape (heads,)")
    _require(isinstance(payload["sm_scale"], numbers.Real), path, "sm_scale must be numeric")
    _require(_is_int(payload["d_v"]) and 0 < payload["d_v"] <= q.shape[2], path, "d_v")

    output, max_logits, lse = payload["output"], payload["max_logits"], payload["lse"]
    _require(tuple(output.shape) == (q.shape[0], q.shape[1], payload["d_v"]), path,
             "output shape")
    _require(tuple(max_logits.shape) == (q.shape[0], q.shape[1]), path, "max_logits shape")
    _require(tuple(lse.shape) == (q.shape[0], q.shape[1]), path, "lse shape")
    _require(isinstance(payload["input_meta"], dict), path, "input_meta must be a mapping")
    _require(set(payload["input_meta"]) == _INPUT_META_KEYS, path, "input_meta keys")
    for key, value in payload["input_meta"].items():
        if key in {"q", "kv", "indices"}:
            _require(value is not None, path, f"input_meta[{key}] is required")
        if value is not None:
            _validate_meta(value, path, key)
    caller_out = payload["caller_out"]
    if caller_out is not None:
        _validate_meta(caller_out, path, "caller_out", caller=True)


def _validate_decode_tensor(value: Any, path: Path, key: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), path, f"{key} is not tensor")
    _require(value.device.type == "cpu", path, f"{key} must be CPU tensor")
    return value


def _validate_decode_cache(
    value: Any, path: Path, key: str, head_dim: int, *, required: bool
) -> torch.Tensor | None:
    if value is None:
        _require(not required, path, f"{key} is required")
        return None
    cache = _validate_decode_tensor(value, path, key)
    _require(cache.ndim in (3, 4), path, f"{key} must have shape (blocks, block, dim)")
    if cache.ndim == 4:
        _require(cache.shape[2] == 1, path, f"{key} singleton head dimension")
        cache_dim = cache.shape[3]
    else:
        cache_dim = cache.shape[2]
    _require(cache.shape[0] > 0 and cache.shape[1] > 0, path, f"{key} is empty")
    _require(cache_dim == head_dim, path, f"{key} head dimension")
    return cache


def _validate_decode_indices(
    value: Any, path: Path, key: str, batch: int
) -> torch.Tensor:
    indices = _validate_decode_tensor(value, path, key)
    _require(indices.ndim == 3 and indices.shape[0] == batch and indices.shape[1] == 1
             and indices.shape[2] > 0, path, f"{key} shape")
    _require(indices.dtype in (torch.int32, torch.int64), path, f"{key} dtype")
    return indices


def _validate_decode_lens(
    value: Any, path: Path, key: str, batch: int, width: int, *, required: bool
) -> torch.Tensor | None:
    if value is None:
        _require(not required, path, f"{key} is required")
        return None
    lens = _validate_decode_tensor(value, path, key)
    _require(lens.ndim == 1 and lens.shape[0] == batch, path, f"{key} shape")
    _require(lens.dtype in (torch.int32, torch.int64), path, f"{key} dtype")
    _require(bool(torch.all((lens >= 0) & (lens <= width))), path, f"{key} values")
    return lens


def _validate_decode_table(
    value: Any, path: Path, key: str, batch: int
) -> torch.Tensor | None:
    if value is None:
        return None
    table = _validate_decode_tensor(value, path, key)
    _require(table.ndim == 2 and table.shape[0] == batch and table.shape[1] > 0,
             path, f"{key} shape")
    _require(table.dtype in (torch.int32, torch.int64), path, f"{key} dtype")
    _require(bool(torch.all(table >= 0)), path, f"{key} values")
    return table


def _validate_decode_payload(payload: dict[str, Any], path: Path) -> None:
    _require(set(payload) == _DECODE_REQUIRED_KEYS, path, f"keys={sorted(payload)}")
    _require(payload["stage"] == "decode", path, "stage must be decode")
    _validate_rank_call(payload, path)
    q = _validate_decode_tensor(payload["q"], path, "q")
    _require(q.ndim == 3 and q.shape[0] > 0 and q.shape[1] > 0 and q.shape[2] > 0,
             path, "q must have shape (batch, heads, head_dim)")
    batch, q_heads, head_dim = q.shape
    _require(_is_int(payload["head_dim"]) and payload["head_dim"] == head_dim,
             path, "head_dim")
    _require(_is_int(payload["q_heads"]) and payload["q_heads"] == q_heads,
             path, "q_heads")
    swa_cache = _validate_decode_cache(payload["swa_cache"], path, "swa_cache", head_dim,
                                       required=True)
    compressed_cache = _validate_decode_cache(
        payload["compressed_cache"], path, "compressed_cache", head_dim, required=False
    )
    swa_indices = _validate_decode_indices(payload["swa_indices"], path, "swa_indices", batch)
    topk_indices = payload["topk_indices"]
    if topk_indices is not None:
        topk_indices = _validate_decode_indices(topk_indices, path, "topk_indices", batch)
    swa_lens = _validate_decode_lens(
        payload["swa_lens"], path, "swa_lens", batch, swa_indices.shape[2], required=False
    )
    topk_lens = _validate_decode_lens(
        payload["topk_lens"], path, "topk_lens", batch,
        topk_indices.shape[2] if topk_indices is not None else 0, required=topk_indices is not None
    )
    _require(topk_indices is not None or topk_lens is None, path,
             "topk_lens requires topk_indices")
    _require(topk_indices is None or compressed_cache is not None, path,
             "topk_indices require compressed_cache")
    _require(isinstance(payload["sm_scale"], numbers.Real), path, "sm_scale must be numeric")
    _require(_is_int(payload["d_v"]) and 0 < payload["d_v"] <= head_dim, path, "d_v")
    attn_sink = payload["attn_sink"]
    if attn_sink is not None:
        attn_sink = _validate_decode_tensor(attn_sink, path, "attn_sink")
        _require(attn_sink.dtype == torch.float32 and attn_sink.shape == (q_heads,),
                 path, "attn_sink must be float32 with shape (heads,)")
    swa_table = _validate_decode_table(payload["swa_block_table"], path, "swa_block_table", batch)
    compressed_table = _validate_decode_table(
        payload["compressed_block_table"], path, "compressed_block_table", batch
    )
    _require(isinstance(payload["cache_meta"], dict), path, "cache_meta must be a mapping")
    _require(set(payload["cache_meta"]) == {
        "swa_block_size", "compressed_block_size", "compress_ratio", "window_size"
    }, path, "cache_meta keys")
    cache_meta = payload["cache_meta"]
    for key in ("swa_block_size", "compressed_block_size", "compress_ratio", "window_size"):
        value = cache_meta[key]
        _require(value is None or _is_int(value), path, f"cache_meta[{key}]")
    _require(cache_meta["swa_block_size"] in (None, swa_cache.shape[1]), path,
             "cache_meta[swa_block_size]")
    if compressed_cache is not None:
        _require(cache_meta["compressed_block_size"] in (None, compressed_cache.shape[1]), path,
                 "cache_meta[compressed_block_size]")
    _require(isinstance(payload["input_meta"], dict), path, "input_meta must be a mapping")
    _require(set(payload["input_meta"]) == _DECODE_INPUT_META_KEYS, path, "input_meta keys")
    for key, value in payload["input_meta"].items():
        if key in {"q", "swa_cache", "swa_indices", "swa_lens"}:
            _require(value is not None, path, f"input_meta[{key}] is required")
        if key in {"topk_indices", "topk_lens"} and payload[key] is not None:
            _require(value is not None, path, f"input_meta[{key}] is required")
        if value is not None:
            _validate_meta(value, path, key)
    caller_out = payload["caller_out"]
    if caller_out is not None:
        _validate_meta(caller_out, path, "caller_out", caller=True)
    output = _validate_decode_tensor(payload["output"], path, "output")
    _require(tuple(output.shape) == (batch, q_heads, payload["d_v"]), path, "output shape")


def load_payload(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{payload_path}: unable to load corpus payload: {exc}") from exc
    _validate_payload(payload, payload_path)
    return payload


def torch_oracle(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute sparse attention using dense Torch operations on CPU."""
    tokens, _heads, head_dim = q.shape
    kv_tokens = kv.shape[0]
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32 or attn_sink.shape != (_heads,):
            raise ValueError("attn_sink must be float32 with shape (heads,)")
        if attn_sink.device != q.device:
            raise ValueError("attn_sink must be on q's device")
    row_indices = indices[:, 0, :].to(torch.int64)
    valid = (row_indices >= 0) & (row_indices < kv_tokens)
    if topk_length is not None:
        positions = torch.arange(row_indices.shape[1], device=row_indices.device)
        valid &= positions.unsqueeze(0) < topk_length.to(torch.int64)[:, None]
    safe_indices = row_indices.masked_fill(~valid, 0)
    gathered = kv[:, 0, :].float().index_select(0, safe_indices.reshape(-1))
    gathered = gathered.reshape(tokens, safe_indices.shape[1], head_dim)
    scores = torch.matmul(q.float(), gathered.transpose(1, 2))
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    scores = scores * float(sm_scale) * _LOG2E
    kv_max = scores.max(dim=-1).values
    nonempty = valid.any(dim=-1)[:, None].expand(-1, _heads)
    safe_max = torch.where(nonempty, kv_max, torch.zeros_like(kv_max))
    weights = torch.exp2(scores - safe_max.unsqueeze(-1)) * valid[:, None, :]
    kv_norm = weights.sum(dim=-1)
    norm = kv_norm
    if attn_sink is not None:
        sink_weight = torch.exp2(attn_sink.float()[None, :] * _LOG2E - safe_max)
        sink_weight = torch.where(
            nonempty & (attn_sink[None, :] > float("-inf")),
            sink_weight,
            torch.zeros_like(sink_weight),
        )
        norm = norm + sink_weight
    safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    value_acc = torch.matmul(weights, gathered[:, :, :d_v])
    output = torch.where(
        (norm > 0.0).unsqueeze(-1), value_acc / safe_norm.unsqueeze(-1), 0.0
    ).to(torch.bfloat16)
    max_logits = torch.where(nonempty, kv_max, torch.full_like(kv_max, float("-inf")))
    lse = torch.where(
        nonempty,
        kv_max + torch.log2(kv_norm),
        torch.full_like(kv_max, float("-inf")),
    )
    return output, max_logits, lse


def _cache_rows(cache: torch.Tensor) -> torch.Tensor:
    """View a paged cache as rows without converting the complete cache."""
    if cache.ndim == 4:
        cache = cache[:, :, 0, :]
    return cache.reshape(-1, cache.shape[-1])


def torch_decode_oracle(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    d_v: int,
    sm_scale: float,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reproduce the current captured ``_torch_sparse_decode`` exactly.

    This intentionally uses the SWA cache for both index sets and ignores
    lens, block-table, compressed-cache, and attention-sink metadata.  Those
    are reported as semantic gaps by :func:`run_diff` rather than silently
    being treated as the intended implementation.
    """
    q2 = q.float()
    combined = swa_indices[:, 0, :]
    if topk_indices is not None:
        combined = torch.cat([topk_indices[:, 0, :], combined], dim=-1)
    invalid = combined < 0
    gather_idx = combined.masked_fill(invalid, 0).to(torch.int64)
    if bool(torch.any((~invalid) & (gather_idx >= _cache_rows(swa_cache).shape[0]))):
        raise ValueError("decode indices contain out-of-range physical slot")
    gathered = _cache_rows(swa_cache).index_select(0, gather_idx.reshape(-1))
    gathered = gathered.reshape(q.shape[0], -1, q.shape[-1]).float()
    scores = torch.matmul(q2, gathered.transpose(1, 2))
    scores.masked_fill_(invalid[:, None, :], float("-inf"))
    probs = torch.softmax(scores * float(sm_scale), dim=-1)
    return torch.matmul(probs, gathered[:, :, :d_v]).to(output_dtype)


def _physical_rows(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor | None,
    block_table: torch.Tensor | None,
    block_size: int,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected physical slots and return rows plus a validity mask."""
    rows = indices[:, 0, :].to(torch.int64)
    selected = rows >= 0
    if lens is not None:
        selected &= torch.arange(rows.shape[1])[None, :] < lens.to(torch.int64)[:, None]
    num_blocks = cache.shape[0]
    physical_block = torch.div(rows.clamp_min(0), block_size, rounding_mode="floor")
    block_offset = torch.remainder(rows.clamp_min(0), block_size)
    valid = selected & (physical_block < num_blocks)
    if block_table is not None:
        # Captured indices are already global physical slots.  The table is
        # still checked so a corrupt capture cannot silently cross requests.
        in_table = (physical_block[:, :, None] == block_table[:, None, :]).any(dim=-1)
        valid &= in_table
    safe_block = physical_block.clamp(0, max(num_blocks - 1, 0))
    safe_rows = safe_block * block_size + block_offset
    flat = _cache_rows(cache).index_select(0, safe_rows.reshape(-1))
    flat = flat.reshape(rows.shape[0], rows.shape[1], cache.shape[-1]).float()
    if not bool(torch.all(~selected | valid)):
        bad = (selected & ~valid).nonzero(as_tuple=False)[0].tolist()
        raise ValueError(f"{name} contains out-of-range or unmapped physical index at {bad}")
    return flat, valid


def torch_decode_physical_oracle(payload: dict[str, Any]) -> torch.Tensor:
    """Replay separate paged caches, lengths, tables, and virtual sink zeros."""
    q = payload["q"].float()
    cache_meta = payload["cache_meta"]
    swa_cache = payload["swa_cache"]
    assert isinstance(swa_cache, torch.Tensor)
    swa_block_size = int(cache_meta["swa_block_size"] or swa_cache.shape[1])
    swa_rows, swa_valid = _physical_rows(
        swa_cache, payload["swa_indices"], payload["swa_lens"],
        payload["swa_block_table"], swa_block_size, name="swa_indices",
    )
    topk = payload["topk_indices"]
    compressed = payload["compressed_cache"]
    if topk is not None:
        if compressed is None:
            raise ValueError("topk_indices require compressed_cache")
        compressed_block_size = int(cache_meta["compressed_block_size"] or compressed.shape[1])
        top_rows, top_valid = _physical_rows(
            compressed, topk, payload["topk_lens"], payload["compressed_block_table"],
            compressed_block_size, name="topk_indices",
        )
    else:
        top_rows = swa_rows[:, :0]
        top_valid = swa_valid[:, :0]
    gathered = torch.cat([top_rows, swa_rows], dim=1)
    valid = torch.cat([top_valid, swa_valid], dim=1)
    scores = torch.matmul(q, gathered.transpose(1, 2)) * float(payload["sm_scale"])
    sink = payload["attn_sink"]
    if sink is not None:
        sink_scores = sink.float()[None, :]
        max_score = torch.maximum(
            torch.where(valid[:, None, :], scores, torch.full_like(scores, float("-inf"))).amax(-1),
            sink_scores,
        )
        weights = torch.where(valid[:, None, :], torch.exp(scores - max_score[..., None]), 0.0)
        sink_weight = torch.exp(sink_scores - max_score)
        norm = weights.sum(-1) + sink_weight
    else:
        max_score = torch.where(valid[:, None, :], scores, torch.full_like(scores, float("-inf"))).amax(-1)
        finite = torch.isfinite(max_score)
        safe_max = torch.where(finite, max_score, torch.zeros_like(max_score))
        weights = torch.where(valid[:, None, :], torch.exp(scores - safe_max[..., None]), 0.0)
        norm = weights.sum(-1)
    value_acc = torch.matmul(weights, gathered[:, :, : int(payload["d_v"])])
    return (value_acc / norm.clamp_min(1e-30).unsqueeze(-1)).to(torch.bfloat16)


def _semantic_gaps(payload: dict[str, Any]) -> list[str]:
    gaps = ["decode_reference_ignores_swa_lens", "decode_reference_ignores_block_tables"]
    if payload["compressed_cache"] is not None and payload["topk_indices"] is not None:
        gaps.append("decode_reference_uses_swa_cache_for_topk_instead_of_compressed_cache")
    if payload["topk_lens"] is not None:
        gaps.append("decode_reference_ignores_topk_lens")
    if payload["attn_sink"] is not None:
        gaps.append("decode_reference_ignores_attention_sink_virtual_zero")
    return gaps


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = (actual.float() - expected.float()).abs()
    finite = delta[torch.isfinite(delta)]
    return float(finite.max().item()) if finite.numel() else 0.0


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> float:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    except AssertionError as exc:
        raise ValueError(f"{name} mismatch (max_abs={_max_abs(actual, expected):.6g}): {exc}") from exc
    return _max_abs(actual, expected)


def run_diff(
    corpus: str | Path,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-3,
    max_files: int | None = None,
) -> dict[str, Any]:
    files = iter_corpus_files(corpus)
    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be non-negative")
        files = files[:max_files]
    details: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    passed = 0
    stage_counts: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    stage_rank_counts: dict[str, int] = {}
    semantic_gaps: dict[str, int] = {}
    for path in files:
        rank, call = _file_key(path)
        try:
            payload = load_payload(path)
            stage = payload["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            rank_key = str(rank)
            rank_counts[rank_key] = rank_counts.get(rank_key, 0) + 1
            stage_rank = f"{stage}/rank{rank}"
            stage_rank_counts[stage_rank] = stage_rank_counts.get(stage_rank, 0) + 1
            if stage == "prefill":
                expected = torch_oracle(
                    payload["q"], payload["kv"], payload["indices"], payload["sm_scale"],
                    payload["d_v"], payload["topk_length"], payload["attn_sink"],
                )
                comparisons = (
                    ("output", payload["output"], expected[0]),
                    ("max_logits", payload["max_logits"], expected[1]),
                    ("lse", payload["lse"], expected[2]),
                )
            else:
                expected_output = torch_decode_oracle(
                    payload["q"], payload["swa_cache"], payload["swa_indices"],
                    payload["topk_indices"], payload["d_v"], payload["sm_scale"],
                    payload["output"].dtype,
                )
                comparisons = (("output", payload["output"], expected_output),)
                for gap in _semantic_gaps(payload):
                    semantic_gaps[gap] = semantic_gaps.get(gap, 0) + 1
                # Execute the metadata-aware path as a corruption/shape check;
                # acceptance remains against the captured Torch reference.
                torch_decode_physical_oracle(payload)
            maxima = {f"{name}_max_abs": _max_abs(actual, oracle)
                      for name, actual, oracle in comparisons}
            details.append({"path": str(path), "stage": stage, "rank": rank,
                            "call": call, **maxima})
            comparison_errors: list[str] = []
            for name, actual, oracle in comparisons:
                try:
                    _compare(name, actual, oracle, atol, rtol)
                except ValueError as exc:
                    comparison_errors.append(str(exc))
            if comparison_errors:
                raise ValueError("; ".join(comparison_errors))
            passed += 1
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return {
        "files": len(files),
        "passed": passed,
        "failed": len(failures),
        "details": details,
        "failures": failures,
        "atol": atol,
        "rtol": rtol,
        "coverage": {
            "by_stage": stage_counts,
            "by_rank": rank_counts,
            "by_stage_rank": stage_rank_counts,
        },
        "groups": {
            stage: {
                rank_key: count
                for group_key, count in stage_rank_counts.items()
                if group_key.startswith(f"{stage}/")
                for rank_key in [group_key.split("/", 1)[1]]
            }
            for stage in stage_counts
        },
        "semantic_gaps": semantic_gaps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_diff(args.corpus, atol=args.atol, rtol=args.rtol, max_files=args.max_files)
    except (ValueError, RuntimeError, OSError) as exc:
        summary = {
            "files": 0,
            "passed": 0,
            "failed": 1,
            "details": [],
            "failures": [{"path": str(args.corpus), "error": str(exc)}],
            "atol": args.atol,
            "rtol": args.rtol,
            "coverage": {"by_stage": {}, "by_rank": {}, "by_stage_rank": {}},
            "groups": {},
            "semantic_gaps": {},
        }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
