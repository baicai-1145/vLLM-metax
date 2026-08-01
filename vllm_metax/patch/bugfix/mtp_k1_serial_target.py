# SPDX-License-Identifier: Apache-2.0
"""Default-off serial target execution for the MTP k=1 correctness gate."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler


logger = init_logger(__name__)
_ENV = "VLLM_METAX_DSV4_MTP_K1_SERIAL_TARGET"
_CAPTURE_ENV = "VLLM_METAX_DSV4_MTP_CAPTURE_DIR"
_PAIR_CAPTURE_STEPS_ENV = "VLLM_METAX_DSV4_MTP_K1_PAIR_CAPTURE_STEPS"
_PAIR_CAPTURE_DEFAULT_STEPS = 16
_PAIR_CAPTURE_COUNTER_ATTR = "_metax_k1_serial_pair_capture_count"
_PAIR_CAPTURE_INDEX_ATTR = "_metax_k1_serial_pair_comparison_index"
_PAIR_CAPTURE_LOCK = threading.Lock()
_SCHEDULE_PATCH_MARKER = "__vllm_metax_dsv4_mtp_k1_serial_target_schedule__"
_SCHEDULE_ORIGINAL_MARKER = (
    "__vllm_metax_dsv4_mtp_k1_serial_target_schedule_original__"
)
_UPDATE_PATCH_MARKER = "__vllm_metax_dsv4_mtp_k1_serial_target_update__"
_UPDATE_ORIGINAL_MARKER = (
    "__vllm_metax_dsv4_mtp_k1_serial_target_update_original__"
)
_HIT_PATCH_MARKER = "__vllm_metax_dsv4_mtp_k1_serial_target_hit__"
_HIT_ORIGINAL_MARKER = "__vllm_metax_dsv4_mtp_k1_serial_target_hit_original__"
_PENDING_OUTPUTS_ATTR = "_metax_k1_serial_pending_outputs"
# A conservative bound protects telemetry memory if an async output is lost.
_PENDING_OUTPUTS_MAX = 128
_ORIGINAL_SCHEDULE = getattr(
    Scheduler.schedule, _SCHEDULE_ORIGINAL_MARKER, Scheduler.schedule
)
_ORIGINAL_UPDATE_FROM_OUTPUT = getattr(
    Scheduler.update_from_output,
    _UPDATE_ORIGINAL_MARKER,
    Scheduler.update_from_output,
)
_ORIGINAL_FIND_LONGEST_CACHE_HIT = getattr(
    HybridKVCacheCoordinator.find_longest_cache_hit,
    _HIT_ORIGINAL_MARKER,
    HybridKVCacheCoordinator.find_longest_cache_hit,
)


def _enabled() -> bool:
    return os.getenv(_ENV, "0") == "1"


def _pair_capture_limit() -> int:
    if not os.getenv(_CAPTURE_ENV):
        return 0
    try:
        default = str(_PAIR_CAPTURE_DEFAULT_STEPS)
        return max(0, int(os.getenv(_PAIR_CAPTURE_STEPS_ENV, default)))
    except (TypeError, ValueError):
        return _PAIR_CAPTURE_DEFAULT_STEPS


def _capture_rank() -> int | str:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is None:
        return str(os.getpid())
    try:
        return int(value)
    except ValueError:
        return value


def _rank_label(rank: int | str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(rank))


def _request_for_id(scheduler: Any, request_id: Any) -> Any:
    requests = getattr(scheduler, "requests", None)
    if requests is not None:
        try:
            request = requests.get(request_id)
        except (AttributeError, TypeError):
            request = None
        if request is not None:
            return request
    for request in getattr(scheduler, "running", ()) or ():
        if getattr(request, "request_id", None) == request_id:
            return request
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _capture_pair(
    scheduler: Any,
    *,
    request_id: Any,
    draft_token_id: int,
    target_token_id: int | None,
    accepted: bool,
    scheduled_token_count: Any,
    req_row_index: Any,
    draft_available: bool | None = None,
) -> None:
    """Best-effort diagnostic capture; never affect serial target behavior."""
    try:
        limit = _pair_capture_limit()
        if limit <= 0:
            return
        count = getattr(scheduler, _PAIR_CAPTURE_COUNTER_ATTR, 0)
        if count >= limit:
            return
        comparison_index = getattr(scheduler, _PAIR_CAPTURE_INDEX_ATTR, 0)
        setattr(scheduler, _PAIR_CAPTURE_COUNTER_ATTR, count + 1)
        setattr(scheduler, _PAIR_CAPTURE_INDEX_ATTR, comparison_index + 1)
        request = _request_for_id(scheduler, request_id)
        rank = _capture_rank()
        record = {
            "comparison_index": comparison_index,
            "rank": rank,
            "request_id": request_id,
            "draft_token_id": draft_token_id,
            "target_token_id": target_token_id,
            "accepted": accepted,
            "draft_available": (
                draft_token_id >= 0
                if draft_available is None
                else draft_available
            ),
            "num_tokens": _int_or_none(getattr(request, "num_tokens", None)),
            "num_computed_tokens": _int_or_none(
                getattr(request, "num_computed_tokens", None)
            ),
            "scheduled_token_count": _int_or_none(scheduled_token_count),
            "req_row_index": _int_or_none(req_row_index),
        }
        capture_dir = Path(os.environ[_CAPTURE_ENV])
        path = capture_dir / f"serial_pairs.rank{_rank_label(rank)}.jsonl"
        capture_dir.mkdir(parents=True, exist_ok=True)
        with _PAIR_CAPTURE_LOCK, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        # Diagnostics are strictly best effort and must never alter execution.
        return


def _record_unavailable(scheduler: Any, request_id: Any) -> None:
    scheduler._metax_k1_serial_unavailable = (
        getattr(scheduler, "_metax_k1_serial_unavailable", 0) + 1
    )
    stats = getattr(scheduler, "_metax_k1_serial_stats", None)
    if stats is None:
        stats = scheduler._metax_k1_serial_stats = {}
    request_stats = stats.setdefault(request_id, [0, 0, 0])
    if len(request_stats) < 3:
        request_stats.extend([0] * (3 - len(request_stats)))
    request_stats[2] += 1


def _schedule(self, throttle_prefills=False):
    if not _enabled():
        return _ORIGINAL_SCHEDULE(self, throttle_prefills)
    pending = {}
    for request in self.running:
        draft_ids = list(getattr(request, "spec_token_ids", ()) or ())
        if len(draft_ids) > 1:
            raise RuntimeError(f"{_ENV}=1 requires exactly one draft token")
        if draft_ids:
            pending[request.request_id] = draft_ids
            request.spec_token_ids = []
    logger.warning_once(
        "DeepSeek V4 MTP k=1 correctness uses serial target verification; "
        "draft slots are not scheduled"
    )
    scheduler_output = _ORIGINAL_SCHEDULE(self, throttle_prefills)
    pending_outputs = getattr(self, _PENDING_OUTPUTS_ATTR, None)
    if pending_outputs is None:
        pending_outputs = []
        setattr(self, _PENDING_OUTPUTS_ATTR, pending_outputs)
    # Keep the output itself alongside the draft snapshot. Besides allowing
    # identity matching for unhashable outputs, the strong reference prevents
    # Python from reusing an object id while this output is in flight.
    overflow = len(pending_outputs) - _PENDING_OUTPUTS_MAX + 1
    if overflow > 0:
        for _, dropped_pending in pending_outputs[:overflow]:
            for request_id, draft_ids in dropped_pending.items():
                if draft_ids:
                    _record_unavailable(self, request_id)
        del pending_outputs[:overflow]
        dropped = getattr(self, "_metax_k1_serial_dropped_pending_outputs", 0)
        setattr(
            self,
            "_metax_k1_serial_dropped_pending_outputs",
            dropped + overflow,
        )
    pending_outputs.append((scheduler_output, pending))
    return scheduler_output


def _extract_target_token_id(target_row) -> int:
    if isinstance(target_row, torch.Tensor):
        if target_row.numel() != 1:
            raise ValueError("sampled target tensor must contain one token")
        return int(target_row.item())
    if not target_row:
        raise ValueError("empty sampled target row")
    return int(target_row[0])


def _update_from_output(self, scheduler_output, model_output):
    if _enabled():
        pending_outputs = getattr(self, _PENDING_OUTPUTS_ATTR, [])
        pending = {}
        for index, (scheduled_output, output_pending) in enumerate(pending_outputs):
            if scheduled_output is scheduler_output:
                pending = output_pending
                pending_outputs.pop(index)
                break
        sampled_token_ids = getattr(model_output, "sampled_token_ids", None)
        req_id_to_index = getattr(model_output, "req_id_to_index", {})
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        if not hasattr(self, "_metax_k1_serial_accepted"):
            self._metax_k1_serial_accepted = 0
        if not hasattr(self, "_metax_k1_serial_rejected"):
            self._metax_k1_serial_rejected = 0
        if not hasattr(self, "_metax_k1_serial_unavailable"):
            self._metax_k1_serial_unavailable = 0
        stats = getattr(self, "_metax_k1_serial_stats", None)
        if stats is None:
            stats = self._metax_k1_serial_stats = {}
        if sampled_token_ids is None:
            pending.clear()
            return _ORIGINAL_UPDATE_FROM_OUTPUT(
                self, scheduler_output, model_output
            )
        for request_id in list(pending):
            if request_id not in scheduled or request_id not in req_id_to_index:
                pending.pop(request_id, None)
                continue
            try:
                draft_token_id = int(pending[request_id][0])
            except (IndexError, KeyError, TypeError, ValueError):
                pending.pop(request_id, None)
                continue
            pending.pop(request_id, None)
            if draft_token_id < 0:
                try:
                    target_token_id = _extract_target_token_id(
                        sampled_token_ids[req_id_to_index[request_id]]
                    )
                except (IndexError, KeyError, TypeError, ValueError):
                    target_token_id = None
                _record_unavailable(self, request_id)
                _capture_pair(
                    self,
                    request_id=request_id,
                    draft_token_id=draft_token_id,
                    target_token_id=target_token_id,
                    accepted=False,
                    draft_available=False,
                    scheduled_token_count=scheduled[request_id],
                    req_row_index=req_id_to_index[request_id],
                )
                logger.info(
                    "[MTP_K1_SERIAL_TARGET] request=%s draft=%d target=%s "
                    "accepted=False draft_available=False",
                    request_id,
                    draft_token_id,
                    target_token_id,
                )
                continue
            try:
                target_token_id = _extract_target_token_id(
                    sampled_token_ids[req_id_to_index[request_id]]
                )
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            accepted = draft_token_id == target_token_id
            counter = (
                "_metax_k1_serial_accepted"
                if accepted
                else "_metax_k1_serial_rejected"
            )
            setattr(self, counter, getattr(self, counter, 0) + 1)
            request_stats = stats.setdefault(request_id, [0, 0, 0])
            if len(request_stats) < 3:
                request_stats.extend([0] * (3 - len(request_stats)))
            request_stats[0 if accepted else 1] += 1
            _capture_pair(
                self,
                request_id=request_id,
                draft_token_id=draft_token_id,
                target_token_id=target_token_id,
                accepted=accepted,
                draft_available=True,
                scheduled_token_count=scheduled[request_id],
                req_row_index=req_id_to_index[request_id],
            )
            logger.info(
                "[MTP_K1_SERIAL_TARGET] request=%s draft=%d target=%d "
                "accepted=%s",
                request_id,
                draft_token_id,
                target_token_id,
                accepted,
            )
    result = _ORIGINAL_UPDATE_FROM_OUTPUT(self, scheduler_output, model_output)
    if _enabled():
        requests = getattr(self, "requests", {})
        stats = getattr(self, "_metax_k1_serial_stats", {})
        for request_id in list(stats):
            if request_id in requests:
                continue
            accepted, rejected, unavailable = stats.pop(request_id)
            logger.warning(
                "[MTP_K1_SERIAL_TARGET] summary request=%s accepted=%d "
                "rejected=%d unavailable=%d",
                request_id,
                accepted,
                rejected,
                unavailable,
            )
    return result


def _find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    if not _enabled():
        return _ORIGINAL_FIND_LONGEST_CACHE_HIT(
            self, block_hashes, max_cache_hit_length
        )
    groups = getattr(self, "attention_groups", None)
    if groups is None:
        return _ORIGINAL_FIND_LONGEST_CACHE_HIT(
            self, block_hashes, max_cache_hit_length
        )

    original_groups = list(groups)
    mutable_flags = []
    try:
        for index, group in enumerate(original_groups):
            if not getattr(group, "use_eagle", False):
                continue
            try:
                group.use_eagle = False
                mutable_flags.append((group, True))
            except (AttributeError, TypeError):
                replacement = getattr(group, "_replace", None)
                if not callable(replacement):
                    return _ORIGINAL_FIND_LONGEST_CACHE_HIT(
                        self, block_hashes, max_cache_hit_length
                    )
                groups[index] = replacement(use_eagle=False)
        return _ORIGINAL_FIND_LONGEST_CACHE_HIT(
            self, block_hashes, max_cache_hit_length
        )
    finally:
        groups[:] = original_groups
        for group, use_eagle in mutable_flags:
            group.use_eagle = use_eagle


setattr(_schedule, _SCHEDULE_PATCH_MARKER, True)
setattr(_schedule, _SCHEDULE_ORIGINAL_MARKER, _ORIGINAL_SCHEDULE)
setattr(_update_from_output, _UPDATE_PATCH_MARKER, True)
setattr(_update_from_output, _UPDATE_ORIGINAL_MARKER, _ORIGINAL_UPDATE_FROM_OUTPUT)
setattr(_find_longest_cache_hit, _HIT_PATCH_MARKER, True)
setattr(_find_longest_cache_hit, _HIT_ORIGINAL_MARKER, _ORIGINAL_FIND_LONGEST_CACHE_HIT)
if not getattr(Scheduler.schedule, _SCHEDULE_PATCH_MARKER, False):
    Scheduler.schedule = _schedule
if not getattr(Scheduler.update_from_output, _UPDATE_PATCH_MARKER, False):
    Scheduler.update_from_output = _update_from_output
if not getattr(
    HybridKVCacheCoordinator.find_longest_cache_hit, _HIT_PATCH_MARKER, False
):
    HybridKVCacheCoordinator.find_longest_cache_hit = _find_longest_cache_hit
