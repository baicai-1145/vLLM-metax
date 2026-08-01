# SPDX-License-Identifier: Apache-2.0
"""Opt-in scheduler-side speculative acceptance diagnostics."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler


logger = init_logger(__name__)

_CAPTURE_ENV = "VLLM_METAX_SPEC_ACCEPTANCE_CAPTURE_DIR"
_SEQUENCE_ATTR = "_metax_spec_acceptance_capture_sequence"
_LOCK = threading.Lock()
_PATCH_MARKER = "__vllm_metax_spec_acceptance_capture__"
_ORIGINAL_MARKER = "__vllm_metax_spec_acceptance_capture_original__"
_UPDATE_PATCH_MARKER = "__vllm_metax_spec_token_capture__"
_UPDATE_ORIGINAL_MARKER = "__vllm_metax_spec_token_capture_original__"
_ORIGINAL_MAKE_SPEC_DECODING_STATS = getattr(
    Scheduler.make_spec_decoding_stats,
    _ORIGINAL_MARKER,
    Scheduler.make_spec_decoding_stats,
)
_ORIGINAL_UPDATE_FROM_OUTPUT = getattr(
    Scheduler.update_from_output,
    _UPDATE_ORIGINAL_MARKER,
    Scheduler.update_from_output,
)


def _capture_rank() -> int | str | None:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _rank_label(rank: int | str | None) -> str:
    value = "none" if rank is None else str(rank)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


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


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _request_metadata(scheduler: Any, request_id: Any) -> dict[str, Any]:
    request = _request_for_id(scheduler, request_id)
    if request is None:
        return {}
    metadata: dict[str, Any] = {}
    for name in (
        "num_prompt_tokens",
        "num_tokens",
        "num_computed_tokens",
        "num_output_placeholders",
        "max_tokens",
        "status",
    ):
        value = getattr(request, name, None)
        if value is not None:
            metadata[name] = _json_value(value)
    return metadata


def _capture(
    scheduler: Any,
    *,
    num_draft_tokens: Any,
    num_accepted_tokens: Any,
    num_invalid_spec_tokens: Any,
    request_id: Any,
) -> None:
    """Write one diagnostic record without affecting scheduler execution."""
    try:
        capture_dir_value = os.getenv(_CAPTURE_ENV)
        if not capture_dir_value:
            return
        raw_draft_tokens = int(num_draft_tokens)
        invalid_tokens = int((num_invalid_spec_tokens or {}).get(request_id, 0))
        effective_draft_tokens = raw_draft_tokens - invalid_tokens
        accepted_tokens = int(num_accepted_tokens)
        rank = _capture_rank()
        record = {
            "request_id": _json_value(request_id),
            "raw_draft_tokens": raw_draft_tokens,
            "effective_draft_tokens": effective_draft_tokens,
            "accepted_tokens": accepted_tokens,
            "invalid_tokens": invalid_tokens,
            "num_spec_tokens": int(getattr(scheduler, "num_spec_tokens", 0)),
            "pid": os.getpid(),
            "rank": rank,
            "request_metadata": _request_metadata(scheduler, request_id),
        }
        capture_dir = Path(capture_dir_value)
        path = capture_dir / (
            f"spec_acceptance.pid{os.getpid()}.rank{_rank_label(rank)}.jsonl"
        )
        with _LOCK:
            sequence = getattr(scheduler, _SEQUENCE_ATTR, 0)
            setattr(scheduler, _SEQUENCE_ATTR, sequence + 1)
            record["sequence"] = sequence
            capture_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as error:
        try:
            logger.warning_once(
                "Speculative acceptance capture failed; inference is unchanged: %s",
                error,
            )
        except Exception:
            pass


def _capture_token_pairs(
    scheduler: Any,
    scheduler_output: Any,
    model_runner_output: Any,
) -> None:
    try:
        capture_dir_value = os.getenv(_CAPTURE_ENV)
        sampled = getattr(model_runner_output, "sampled_token_ids", None)
        scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
        row_by_request = getattr(model_runner_output, "req_id_to_index", None)
        if (
            not capture_dir_value
            or not isinstance(sampled, (list, tuple))
            or not isinstance(scheduled, dict)
            or not isinstance(row_by_request, dict)
        ):
            return
        rank = _capture_rank()
        path = Path(capture_dir_value) / (
            f"spec_tokens.pid{os.getpid()}.rank{_rank_label(rank)}.jsonl"
        )
        records = []
        for request_id, draft_ids in scheduled.items():
            row_index = row_by_request.get(request_id)
            if not isinstance(draft_ids, (list, tuple)) or not isinstance(row_index, int):
                continue
            if row_index < 0 or row_index >= len(sampled):
                continue
            sampled_ids = sampled[row_index]
            if not isinstance(sampled_ids, (list, tuple)):
                continue
            records.append(
                {
                    "pid": os.getpid(),
                    "rank": rank,
                    "request_id": _json_value(request_id),
                    "scheduled_draft_token_ids": [int(token) for token in draft_ids],
                    "sampled_token_ids": [int(token) for token in sampled_ids],
                    "request_metadata": _request_metadata(scheduler, request_id),
                }
            )
        if not records:
            return
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as error:
        try:
            logger.warning_once(
                "Speculative token capture failed; inference is unchanged: %s",
                error,
            )
        except Exception:
            pass


def _update_from_output(self, scheduler_output, model_runner_output):
    _capture_token_pairs(self, scheduler_output, model_runner_output)
    return _ORIGINAL_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)


def _make_spec_decoding_stats(
    self,
    spec_decoding_stats,
    num_draft_tokens,
    num_accepted_tokens,
    num_invalid_spec_tokens,
    request_id,
):
    result = _ORIGINAL_MAKE_SPEC_DECODING_STATS(
        self,
        spec_decoding_stats,
        num_draft_tokens,
        num_accepted_tokens,
        num_invalid_spec_tokens,
        request_id,
    )
    if os.getenv(_CAPTURE_ENV) and num_draft_tokens:
        _capture(
            self,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            num_invalid_spec_tokens=num_invalid_spec_tokens,
            request_id=request_id,
        )
    return result


setattr(_make_spec_decoding_stats, _PATCH_MARKER, True)
setattr(_make_spec_decoding_stats, _ORIGINAL_MARKER, _ORIGINAL_MAKE_SPEC_DECODING_STATS)
setattr(_update_from_output, _UPDATE_PATCH_MARKER, True)
setattr(_update_from_output, _UPDATE_ORIGINAL_MARKER, _ORIGINAL_UPDATE_FROM_OUTPUT)
if not getattr(Scheduler.make_spec_decoding_stats, _PATCH_MARKER, False):
    Scheduler.make_spec_decoding_stats = _make_spec_decoding_stats
if not getattr(Scheduler.update_from_output, _UPDATE_PATCH_MARKER, False):
    Scheduler.update_from_output = _update_from_output
