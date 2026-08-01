# SPDX-License-Identifier: Apache-2.0
"""Opt-in target-sampling capture for DeepSeek V4 MTP diagnosis."""

from __future__ import annotations

import os
import json
import sys
import threading
from pathlib import Path
from typing import Any

from vllm.v1.worker import gpu_model_runner as _gpu_model_runner
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.core.block_pool import BlockPool

from vllm_metax.models.deepseek_v4.mtp_debug import (
    _is_cuda_graph_capturing,
    maybe_capture_mtp_stage,
)


_CAPTURE_ENV = "VLLM_METAX_DSV4_MTP_CAPTURE_DIR"
_PREFIX_CAPTURE_ENV = "VLLM_METAX_DSV4_PREFIX_CACHE_CAPTURE_DIR"
_START_STEP_ENV = "VLLM_METAX_DSV4_MTP_TARGET_CAPTURE_START_STEP"
_STEPS_ENV = "VLLM_METAX_DSV4_MTP_TARGET_CAPTURE_STEPS"
_COUNTER_ATTR = "_vllm_metax_target_runtime_step"
_DEFAULT_STEPS = 8
_PATCH_MARKER = "__vllm_metax_target_runtime_capture"
_KV_PATCH_MARKER = "__vllm_metax_prefix_hit_capture"
_CACHE_PATCH_MARKER = "__vllm_metax_cache_blocks_capture"
_PREFIX_CAPTURE_LOCK = threading.Lock()

_ORIGINAL_SAMPLE_TOKENS = getattr(
    _gpu_model_runner.GPUModelRunner.sample_tokens,
    "__vllm_metax_target_runtime_original__",
    _gpu_model_runner.GPUModelRunner.sample_tokens,
)
_ORIGINAL_GET_COMPUTED_BLOCKS = getattr(
    KVCacheManager.get_computed_blocks,
    "__vllm_metax_prefix_hit_capture_original__",
    KVCacheManager.get_computed_blocks,
)
_ORIGINAL_CACHE_BLOCKS = getattr(
    SingleTypeKVCacheManager.cache_blocks,
    "__vllm_metax_cache_blocks_original__",
    SingleTypeKVCacheManager.cache_blocks,
)
_ORIGINAL_MAYBE_EVICT = getattr(
    BlockPool._maybe_evict_cached_block,
    "__vllm_metax_eviction_capture_original__",
    BlockPool._maybe_evict_cached_block,
)
_ORIGINAL_REMOVE_CACHED_HASHES = getattr(
    BlockPool._remove_cached_block_hashes,
    "__vllm_metax_hash_removal_capture_original__",
    BlockPool._remove_cached_block_hashes,
)


def _capture_steps() -> int:
    try:
        return max(0, int(os.environ.get(_STEPS_ENV, _DEFAULT_STEPS)))
    except (TypeError, ValueError):
        return _DEFAULT_STEPS


def _capture_start_step() -> int:
    try:
        return max(0, int(os.environ.get(_START_STEP_ENV, 0)))
    except (TypeError, ValueError):
        return 0


def _capture_dir() -> Path | None:
    value = os.environ.get(_CAPTURE_ENV)
    return Path(value) if value else None


def _prefix_capture_dir() -> Path | None:
    value = os.environ.get(_PREFIX_CAPTURE_ENV) or os.environ.get(_CAPTURE_ENV)
    return Path(value) if value else None


def _capture_rank() -> int | str:
    value = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if value is None:
        return str(os.getpid())
    try:
        return int(value)
    except ValueError:
        return value


def _rank_label(rank: int | str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(rank))


def _group_diagnostics(coordinator: Any) -> dict[str, Any]:
    groups = getattr(getattr(coordinator, "kv_cache_config", None), "kv_cache_groups", ())
    specs = []
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        specs.append(
            {
                "spec_type": type(spec).__name__ if spec is not None else None,
                "block_size": getattr(spec, "block_size", None),
                "use_eagle": getattr(group, "is_eagle_group", None),
            }
        )
    return {
        "coordinator_type": type(coordinator).__name__,
        "group_specs": specs,
        "group_spec_types": [item["spec_type"] for item in specs],
        "group_block_sizes": [item["block_size"] for item in specs],
        "group_use_eagle": [item["use_eagle"] for item in specs],
        "use_eagle": getattr(coordinator, "use_eagle", None),
    }


def _write_jsonl(filename: str, record: dict[str, Any]) -> None:
    capture_dir = _prefix_capture_dir()
    if capture_dir is None:
        return
    capture_dir.mkdir(parents=True, exist_ok=True)
    rank = _capture_rank()
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jsonl"
    path = capture_dir / f"{stem}.rank{_rank_label(rank)}{suffix}"
    record = {"rank": rank, **record}
    with _PREFIX_CAPTURE_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_prefix_hit_record(record: dict[str, Any]) -> None:
    _write_jsonl("prefix_hits.jsonl", record)


def _get_computed_blocks(self: Any, request: Any) -> Any:
    diagnostic: dict[str, Any] | None = None
    if _prefix_capture_dir() is not None:
        try:
            coordinator = getattr(self, "coordinator")
            probe = getattr(coordinator, "find_longest_cache_hit_per_group", None)
            request_hashes = getattr(request, "block_hashes")
            num_tokens = int(getattr(request, "num_tokens"))
            if callable(probe):
                blocks_per_group, hit_lengths = probe(request_hashes, num_tokens - 1)
                diagnostic = {
                    "request_id": getattr(request, "request_id", None),
                    "num_tokens": num_tokens,
                    "max_cache_hit_length": num_tokens - 1,
                    "per_group_hit_lengths": [int(value) for value in hit_lengths],
                    "per_group_block_counts": [len(blocks) for blocks in blocks_per_group],
                    **_group_diagnostics(coordinator),
                }
        except Exception:
            diagnostic = None

    result = _ORIGINAL_GET_COMPUTED_BLOCKS(self, request)
    if diagnostic is not None:
        try:
            final_hit_length = int(result[1])
            diagnostic["final_hit_length"] = final_hit_length
            diagnostic["final_num_computed_tokens"] = final_hit_length
            _write_prefix_hit_record(diagnostic)
        except Exception:
            pass
    return result


def _snapshot_logical_blocks(manager: Any, request_id: Any) -> dict[str, dict[str, Any]]:
    try:
        blocks = getattr(manager, "req_to_blocks").get(request_id, ())
    except (AttributeError, TypeError):
        blocks = ()
    snapshot: dict[str, dict[str, Any]] = {}
    for logical_idx in range(1, 10):
        block = blocks[logical_idx] if logical_idx < len(blocks) else None
        if block is None:
            snapshot[str(logical_idx)] = {
                "block_id": None,
                "null": None,
                "is_null": None,
                "ref_cnt": None,
                "hash_present": False,
                "hash_num_tokens": None,
            }
            continue
        block_hash = getattr(block, "block_hash", None)
        snapshot[str(logical_idx)] = {
            "block_id": getattr(block, "block_id", None),
            "null": getattr(block, "is_null", None),
            "is_null": getattr(block, "is_null", None),
            "ref_cnt": getattr(block, "ref_cnt", None),
            "hash_present": block_hash is not None,
            "hash_num_tokens": getattr(block, "block_hash_num_tokens", None),
        }
    return snapshot


def _free_pool_count(manager: Any) -> int | None:
    try:
        pool = getattr(manager, "block_pool", manager)
        queue = getattr(pool, "free_block_queue")
        return int(getattr(queue, "num_free_blocks"))
    except (AttributeError, TypeError, ValueError):
        return None


def _reachable_indices(
    manager: Any,
    *,
    start_block: int,
    end_block: int,
    retention_interval: int | None,
    num_prompt_tokens: int | None,
) -> list[int] | None:
    try:
        mask = manager.reachable_block_mask(
            start_block=start_block,
            end_block=end_block,
            alignment_tokens=getattr(manager, "scheduler_block_size"),
            kv_cache_spec=getattr(manager, "kv_cache_spec"),
            use_eagle=bool(getattr(manager, "use_eagle", False)),
            retention_interval=retention_interval,
            num_prompt_tokens=num_prompt_tokens,
        )
        if mask is None:
            return list(range(start_block, end_block))
        return [start_block + idx for idx, reachable in enumerate(mask) if reachable]
    except Exception:
        return None


def _cache_blocks(self: Any, request: Any, num_tokens: int, retention_interval: int | None = None) -> None:
    diagnostic: dict[str, Any] | None = None
    if _prefix_capture_dir() is not None:
        try:
            block_size = int(getattr(getattr(self, "kv_cache_spec"), "block_size"))
            total_tokens = int(num_tokens)
            if block_size == 64 and total_tokens >= 512:
                request_id = getattr(request, "request_id", None)
                num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
                num_cached_before = int(
                    getattr(self, "num_cached_block").get(request_id, 0)
                )
                num_full_blocks = total_tokens // block_size
                diagnostic = {
                    "request_id": request_id,
                    "kv_cache_group_id": getattr(self, "kv_cache_group_id", None),
                    "spec_type": type(getattr(self, "kv_cache_spec")).__name__,
                    "block_size": block_size,
                    "use_eagle": bool(getattr(self, "use_eagle", False)),
                    "num_tokens": total_tokens,
                    "num_prompt_tokens": num_prompt_tokens,
                    "num_cached_blocks_before": num_cached_before,
                    "num_full_blocks": num_full_blocks,
                    "retention_interval": retention_interval,
                    "reachable_block_mask_indices": _reachable_indices(
                        self,
                        start_block=num_cached_before,
                        end_block=num_full_blocks,
                        retention_interval=retention_interval,
                        num_prompt_tokens=num_prompt_tokens,
                    ),
                    "logical_blocks_before": _snapshot_logical_blocks(self, request_id),
                    "free_pool_count_before": _free_pool_count(self),
                    "phase": (
                        "initial_prefill_commit"
                        if num_cached_before == 0
                        else "later_decode_or_replay"
                    ),
                }
        except Exception:
            diagnostic = None

    result = _ORIGINAL_CACHE_BLOCKS(
        self, request, num_tokens, retention_interval=retention_interval
    )
    if diagnostic is not None:
        try:
            request_id = diagnostic["request_id"]
            num_cached_after = int(getattr(self, "num_cached_block").get(request_id, 0))
            diagnostic["num_cached_blocks_after"] = num_cached_after
            diagnostic["logical_blocks_after"] = _snapshot_logical_blocks(self, request_id)
            diagnostic["free_pool_count_after"] = _free_pool_count(self)
            _write_jsonl("cache_blocks.jsonl", diagnostic)
        except Exception:
            pass
    return result


def _serialize_hash_key(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return {
            "hex": raw.hex(),
            "group_id": int.from_bytes(raw[-4:], "big") if len(raw) >= 4 else None,
        }
    return None


def _block_hash_keys(pool: Any, block: Any) -> list[dict[str, Any]]:
    values = []
    primary = _serialize_hash_key(getattr(block, "block_hash", None))
    if primary is not None:
        values.append(primary)
    try:
        aliases = getattr(pool, "cached_block_hashes_by_block").get(
            getattr(block, "block_id"), ()
        )
        for alias in aliases:
            encoded = _serialize_hash_key(alias)
            if encoded is not None:
                values.append(encoded)
    except (AttributeError, TypeError):
        pass
    return values


def _eviction_block_snapshot(pool: Any, block: Any) -> dict[str, Any]:
    return {
        "block_id": getattr(block, "block_id", None),
        "hash_num_tokens": getattr(block, "block_hash_num_tokens", None),
        "hash_present": getattr(block, "block_hash", None) is not None,
        "hash_keys": _block_hash_keys(pool, block),
        "ref_cnt": getattr(block, "ref_cnt", None),
        "free_pool_count": _free_pool_count(pool),
    }


def _maybe_evict_cached_block(self: Any, block: Any) -> bool:
    diagnostic: dict[str, Any] | None = None
    if _prefix_capture_dir() is not None:
        try:
            hash_tokens = getattr(block, "block_hash_num_tokens", None)
            hash_present = getattr(block, "block_hash", None) is not None
            if hash_present or (
                isinstance(hash_tokens, int) and 128 <= hash_tokens <= 640
            ):
                diagnostic = {
                    "before": _eviction_block_snapshot(self, block),
                }
        except Exception:
            diagnostic = None

    result = _ORIGINAL_MAYBE_EVICT(self, block)
    if diagnostic is not None:
        try:
            diagnostic["after"] = _eviction_block_snapshot(self, block)
            diagnostic["evicted"] = bool(result)
            _write_jsonl("evictions.jsonl", diagnostic)
        except Exception:
            pass
    return result


def _caller_tag() -> dict[str, str | None]:
    try:
        caller = sys._getframe(2)
        parent = caller.f_back
        return {
            "caller": caller.f_code.co_name,
            "parent_caller": parent.f_code.co_name if parent is not None else None,
        }
    except (AttributeError, RuntimeError):
        return {"caller": None, "parent_caller": None}


def _remove_cached_block_hashes(self: Any, block: Any) -> list[Any]:
    diagnostic: dict[str, Any] | None = None
    if _prefix_capture_dir() is not None:
        try:
            before_keys = _block_hash_keys(self, block)
            if before_keys:
                diagnostic = {
                    "before": {
                        "block_id": getattr(block, "block_id", None),
                        "hash_num_tokens": getattr(block, "block_hash_num_tokens", None),
                        "hash_present": getattr(block, "block_hash", None) is not None,
                        "hash_keys": before_keys,
                        "ref_cnt": getattr(block, "ref_cnt", None),
                    },
                    "caller": _caller_tag(),
                }
        except Exception:
            diagnostic = None

    removed = _ORIGINAL_REMOVE_CACHED_HASHES(self, block)
    if diagnostic is not None or removed:
        try:
            removed_list = list(removed)
            if diagnostic is None:
                diagnostic = {"caller": _caller_tag()}
            diagnostic["removed_hash_count"] = len(removed_list)
            diagnostic["removed_hash_keys"] = [
                encoded
                for item in removed_list
                if (encoded := _serialize_hash_key(item)) is not None
            ]
            diagnostic["after"] = {
                "block_id": getattr(block, "block_id", None),
                "hash_num_tokens": getattr(block, "block_hash_num_tokens", None),
                "hash_present": getattr(block, "block_hash", None) is not None,
                "hash_keys": _block_hash_keys(self, block),
                "ref_cnt": getattr(block, "ref_cnt", None),
            }
            _write_jsonl("hash_removals.jsonl", diagnostic)
        except Exception:
            pass
    return removed


def _device(value: Any) -> Any:
    return getattr(value, "gpu", value)


def _prefix(value: Any, count: int) -> Any:
    try:
        return value[:count]
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
        return None


def _capture_target_runtime(self: Any, step: int) -> None:
    # Avoid even indexing persistent GPU buffers while a graph is being
    # captured; the shared serializer applies the same guard for direct users.
    if _is_cuda_graph_capturing():
        return
    try:
        input_batch = getattr(self, "input_batch")
        num_reqs = int(getattr(input_batch, "num_reqs"))
        state = getattr(self, "execute_model_state")
        # The row derivation is valid only for serial target execution. Draft
        # slots make the last-token rows ambiguous, so skip them entirely.
        if getattr(state, "spec_decode_metadata", None) is not None:
            return
        query_start_loc = _device(getattr(self, "query_start_loc"))
        input_ids = _device(getattr(self, "input_ids"))
        positions = getattr(self, "positions")
        seq_lens = getattr(self, "seq_lens")
        row_indices = query_start_loc[1 : num_reqs + 1] - 1
        sampled_input_ids = input_ids[row_indices]
        sampled_positions = positions[row_indices]
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return

    num_tokens = getattr(input_batch, "num_tokens", None)
    if num_tokens is None:
        try:
            num_tokens = int(query_start_loc[num_reqs].item())
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            num_tokens = None
    else:
        try:
            num_tokens = int(num_tokens)
        except (TypeError, ValueError):
            num_tokens = None

    num_spec = getattr(self, "num_spec_tokens", None)
    prev_num_spec = getattr(self, "prev_num_spec_tokens", None)
    num_scheduled = _prefix(_device(getattr(self, "num_scheduled_tokens", None)), num_reqs)
    request_slots = _prefix(_device(getattr(self, "req_indices", None)), num_tokens or 0)
    controls = {
        "batch_size": num_reqs,
        "num_tokens": num_tokens,
        "num_input_tokens": num_scheduled,
        "num_scheduled_tokens": num_scheduled,
        "num_speculative_tokens": num_spec,
        "prev_num_speculative_tokens": prev_num_spec,
        "prev_num_spec_tokens": prev_num_spec,
        "request_slot_mapping": request_slots,
        "req_indices": request_slots,
        "row_indices": row_indices,
        "query_start_loc": _prefix(query_start_loc, num_reqs + 1),
        "seq_lens": _prefix(seq_lens, num_reqs),
        "spec_decode_metadata_present": False,
        # Existing serializer retains this control key.
        "needs_extra_input_slots": False,
    }
    fields = {
        **controls,
        "input_ids": sampled_input_ids,
        "positions": sampled_positions,
        "post_hidden_states": getattr(state, "sample_hidden_states", None),
        "logits": getattr(state, "logits", None),
    }
    try:
        maybe_capture_mtp_stage("v1_target_runtime", step, fields)
    except Exception:
        return

    groups = getattr(getattr(input_batch, "block_table", None), "block_tables", None)
    if groups is None:
        return
    for group_idx, group in enumerate(groups):
        try:
            block_buffer = getattr(group, "block_table")
            slot_buffer = getattr(group, "slot_mapping")
            group_fields = {
                **controls,
                "block_table": _device(block_buffer)[:num_reqs],
                "block_table_cpu": getattr(block_buffer, "cpu")[:num_reqs],
                "slot_mapping": _device(slot_buffer)[:num_tokens]
                if num_tokens is not None
                else None,
                "slot_mapping_cpu": getattr(slot_buffer, "cpu")[:num_tokens]
                if num_tokens is not None
                else None,
            }
            maybe_capture_mtp_stage(
                f"v1_target_runtime_kv_group_{group_idx}", step, group_fields
            )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        except Exception:
            continue


def _sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
    if os.environ.get(_CAPTURE_ENV) and getattr(self, "execute_model_state", None) is not None:
        try:
            step = int(getattr(self, _COUNTER_ATTR, 0))
        except (TypeError, ValueError):
            step = 0
        setattr(self, _COUNTER_ATTR, step + 1)
        start_step = _capture_start_step()
        if start_step <= step < start_step + _capture_steps():
            _capture_target_runtime(self, step)
    return _ORIGINAL_SAMPLE_TOKENS(self, *args, **kwargs)


setattr(_sample_tokens, _PATCH_MARKER, True)
setattr(_sample_tokens, "__vllm_metax_target_runtime_original__", _ORIGINAL_SAMPLE_TOKENS)
setattr(_get_computed_blocks, _KV_PATCH_MARKER, True)
setattr(
    _get_computed_blocks,
    "__vllm_metax_prefix_hit_capture_original__",
    _ORIGINAL_GET_COMPUTED_BLOCKS,
)
setattr(_cache_blocks, _CACHE_PATCH_MARKER, True)
setattr(_cache_blocks, "__vllm_metax_cache_blocks_original__", _ORIGINAL_CACHE_BLOCKS)
setattr(_maybe_evict_cached_block, "__vllm_metax_eviction_capture", True)
setattr(
    _maybe_evict_cached_block,
    "__vllm_metax_eviction_capture_original__",
    _ORIGINAL_MAYBE_EVICT,
)
setattr(_remove_cached_block_hashes, "__vllm_metax_hash_removal_capture", True)
setattr(
    _remove_cached_block_hashes,
    "__vllm_metax_hash_removal_capture_original__",
    _ORIGINAL_REMOVE_CACHED_HASHES,
)


def _install_patch() -> None:
    global _ORIGINAL_SAMPLE_TOKENS, _ORIGINAL_GET_COMPUTED_BLOCKS
    global _ORIGINAL_CACHE_BLOCKS, _ORIGINAL_MAYBE_EVICT
    global _ORIGINAL_REMOVE_CACHED_HASHES
    method = _gpu_model_runner.GPUModelRunner.sample_tokens
    if not getattr(method, _PATCH_MARKER, False):
        _ORIGINAL_SAMPLE_TOKENS = getattr(
            method, "__vllm_metax_target_runtime_original__", method
        )
        setattr(
            _sample_tokens,
            "__vllm_metax_target_runtime_original__",
            _ORIGINAL_SAMPLE_TOKENS,
        )
        _gpu_model_runner.GPUModelRunner.sample_tokens = _sample_tokens
    method = KVCacheManager.get_computed_blocks
    if not getattr(method, _KV_PATCH_MARKER, False):
        _ORIGINAL_GET_COMPUTED_BLOCKS = getattr(
            method, "__vllm_metax_prefix_hit_capture_original__", method
        )
        setattr(
            _get_computed_blocks,
            "__vllm_metax_prefix_hit_capture_original__",
            _ORIGINAL_GET_COMPUTED_BLOCKS,
        )
        KVCacheManager.get_computed_blocks = _get_computed_blocks
    method = SingleTypeKVCacheManager.cache_blocks
    if not getattr(method, _CACHE_PATCH_MARKER, False):
        _ORIGINAL_CACHE_BLOCKS = getattr(
            method, "__vllm_metax_cache_blocks_original__", method
        )
        setattr(_cache_blocks, "__vllm_metax_cache_blocks_original__", _ORIGINAL_CACHE_BLOCKS)
        SingleTypeKVCacheManager.cache_blocks = _cache_blocks
    method = BlockPool._maybe_evict_cached_block
    if not getattr(method, "__vllm_metax_eviction_capture", False):
        _ORIGINAL_MAYBE_EVICT = getattr(
            method, "__vllm_metax_eviction_capture_original__", method
        )
        setattr(
            _maybe_evict_cached_block,
            "__vllm_metax_eviction_capture_original__",
            _ORIGINAL_MAYBE_EVICT,
        )
        BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    method = BlockPool._remove_cached_block_hashes
    if not getattr(method, "__vllm_metax_hash_removal_capture", False):
        _ORIGINAL_REMOVE_CACHED_HASHES = getattr(
            method, "__vllm_metax_hash_removal_capture_original__", method
        )
        setattr(
            _remove_cached_block_hashes,
            "__vllm_metax_hash_removal_capture_original__",
            _ORIGINAL_REMOVE_CACHED_HASHES,
        )
        BlockPool._remove_cached_block_hashes = _remove_cached_block_hashes


_install_patch()
