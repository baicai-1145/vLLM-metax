"""Opt-in, lightweight DeepSeek V4 MTP stage capture and offline analysis.

The normal path does not inspect tensors or touch the filesystem.  Capture is
enabled only by ``VLLM_METAX_DSV4_MTP_CAPTURE_DIR`` and is deliberately disabled
while a CUDA graph is being captured.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
_CAPTURE_LOCK = threading.Lock()


def _rank() -> int | str:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        return str(os.getpid())
    try:
        return int(value)
    except ValueError:
        return value


def _capture_dir() -> Path | None:
    value = os.getenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR")
    return Path(value) if value else None


def _is_cuda_graph_capturing() -> bool:
    """Fail closed if the CUDA graph state cannot be determined."""
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    strides = list(value.stride())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": strides,
        "strides": strides,
    }


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = _cpu_tensor(value)
    try:
        raw = tensor.view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError, ValueError):
        raw = repr(tensor.tolist()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = _cpu_tensor(value)
        data = tensor.tolist()
        return data.item() if hasattr(data, "item") else data
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optional_field(fields: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        fields[name] = value


def _cad_fields(prefix: str, cad: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "_seq_lens_cpu",
        "_num_computed_tokens_cpu",
        "seq_lens_cpu_upper_bound",
        "slot_mapping",
        "block_table_tensor",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
        "num_reqs",
    ):
        if hasattr(cad, name):
            fields[f"{prefix}_{name.lstrip('_')}"] = getattr(cad, name)
    return fields


def _top_k(logits: Any, k: int = 5) -> Any:
    if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
        return _json_value(logits)
    values, indices = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)
    return [
        [[int(index), float(value)] for value, index in zip(row_values, row_indices)]
        for row_values, row_indices in zip(
            _cpu_tensor(values).tolist(), _cpu_tensor(indices).tolist()
        )
    ]


def _field(tensors: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in tensors:
            return _json_value(tensors[name])
    return None


@torch.no_grad()
def maybe_capture_v1_sampler_metadata(
    metadata: Any,
    logits: torch.Tensor,
    spec_step_idx: int = 0,
) -> dict[str, Any] | None:
    """Capture V1 sampler row-index metadata before target logits are sliced."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return None
    target_logits_indices = metadata.target_logits_indices
    bonus_logits_indices = metadata.bonus_logits_indices
    logits_indices = metadata.logits_indices
    target_model_row_indices = logits_indices[target_logits_indices]
    bonus_model_row_indices = logits_indices[bonus_logits_indices]
    fields = {
        "draft_ids": metadata.draft_token_ids,
        "cu_num_logits": metadata.cu_num_sampled_tokens,
        "cu_num_draft_tokens": metadata.cu_num_draft_tokens,
        "target_logits_indices": target_logits_indices,
        "bonus_logits_indices": bonus_logits_indices,
        "logits_indices": logits_indices,
        "target_model_row_indices": target_model_row_indices,
        "bonus_model_row_indices": bonus_model_row_indices,
        "logits": logits[target_logits_indices],
        "bonus_logits": logits[bonus_logits_indices],
    }
    return maybe_capture_mtp_stage("v1_sampler_metadata", spec_step_idx, fields)


@torch.no_grad()
def maybe_capture_v1_model_runner_metadata(
    metadata: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> dict[str, Any] | None:
    """Capture V1 spec-decode row metadata while global input rows are visible."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return None
    target_logits_indices = metadata.target_logits_indices
    bonus_logits_indices = metadata.bonus_logits_indices
    logits_indices = metadata.logits_indices
    target_model_row_indices = logits_indices[target_logits_indices]
    bonus_model_row_indices = logits_indices[bonus_logits_indices]
    row_input_ids = input_ids[logits_indices]
    row_positions = positions[logits_indices]
    fields = {
        "draft_ids": metadata.draft_token_ids,
        "cu_num_logits": metadata.cu_num_sampled_tokens,
        "cu_num_draft_tokens": metadata.cu_num_draft_tokens,
        "target_logits_indices": target_logits_indices,
        "bonus_logits_indices": bonus_logits_indices,
        "logits_indices": logits_indices,
        "target_model_row_indices": target_model_row_indices,
        "bonus_model_row_indices": bonus_model_row_indices,
        "input_ids": row_input_ids,
        "positions": row_positions,
        "target_input_ids": input_ids[target_model_row_indices],
        "target_positions": positions[target_model_row_indices],
        "bonus_input_ids": input_ids[bonus_model_row_indices],
        "bonus_positions": positions[bonus_model_row_indices],
    }
    if logits is not None:
        fields["logits"] = logits[target_logits_indices]
    return maybe_capture_mtp_stage("v1_model_runner_metadata", spec_step_idx, fields)
@torch.no_grad()
def maybe_capture_mtp_stage(
    stage: str, spec_step_idx: int, tensors: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Append one compact stage record when explicitly enabled.

    Tensor values are retained only for small control fields (IDs and counts);
    hidden states are represented by hashes and all tensors receive metadata.
    """
    capture_dir = _capture_dir()
    if capture_dir is None or _is_cuda_graph_capturing():
        return None

    rank = _rank()
    tensor_meta = {
        name: _tensor_meta(value)
        for name, value in tensors.items()
        if isinstance(value, torch.Tensor)
    }
    pre_hidden = _field(
        tensors, "pre_hidden_states", "previous_hidden_states", "hidden_states_pre"
    )
    post_hidden = _field(
        tensors, "post_hidden_states", "hidden_states", "hidden_states_post"
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "step": int(spec_step_idx),
        "spec_step_idx": int(spec_step_idx),
        "stage": stage,
        "tensor_meta": tensor_meta,
        "input_ids": _field(tensors, "input_ids"),
        "positions": _field(tensors, "positions"),
        "draft_ids": _field(tensors, "draft_ids", "draft_token_ids"),
        "target_ids": _field(tensors, "target_ids", "target_token_ids"),
        "committed_ids": _field(tensors, "committed_ids", "committed_token_ids"),
        "accepted_count": _field(tensors, "accepted_count", "num_accepted"),
        "pre_hidden_hash": (
            _tensor_hash(tensors["pre_hidden_states"])
            if isinstance(tensors.get("pre_hidden_states"), torch.Tensor)
            else None
        ),
        "post_hidden_hash": (
            _tensor_hash(tensors["post_hidden_states"])
            if isinstance(tensors.get("post_hidden_states"), torch.Tensor)
            else None
        ),
        "logits_top_k": _top_k(tensors.get("logits")),
    }
    for key in (
        "boundary",
        "request_idx",
        "method",
        "batch_size",
        "num_tokens",
        "num_input_tokens",
        "num_speculative_tokens",
        "needs_extra_input_slots",
        "extra_slots_per_request",
        "net_num_new_slots_per_request",
        "draft_ids",
        "target_ids",
        "committed_ids",
        "accepted_count",
        "next_token_ids",
        "row_indices",
        "positions",
        "target_positions",
        "input_ids_after",
        "positions_after",
        "target_input_ids",
        "target_positions",
        "bonus_input_ids",
        "bonus_positions",
        "expanded_local_pos",
        "expanded_idx_mapping",
        "cu_num_logits",
        "cu_num_draft_tokens",
        "valid_sampled_tokens_count",
        "num_rejected_tokens",
        "token_indices_to_sample",
        "is_rejected_token_mask",
        "is_masked_token_mask",
        "slot_mapping",
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "seq_lens_cpu",
        "num_computed_tokens_cpu",
        "seq_lens_cpu_upper_bound",
        "block_table",
        "before_query_start_loc",
        "before_query_start_loc_cpu",
        "before_seq_lens",
        "before_seq_lens_cpu",
        "before_num_computed_tokens_cpu",
        "before_seq_lens_cpu_upper_bound",
        "before_slot_mapping",
        "before_block_table_tensor",
        "before_num_actual_tokens",
        "before_max_query_len",
        "before_max_seq_len",
        "before_num_reqs",
        "after_query_start_loc",
        "after_query_start_loc_cpu",
        "after_seq_lens",
        "after_seq_lens_cpu",
        "after_num_computed_tokens_cpu",
        "after_seq_lens_cpu_upper_bound",
        "after_slot_mapping",
        "after_block_table_tensor",
        "after_num_actual_tokens",
        "after_max_query_len",
        "after_max_seq_len",
        "after_num_reqs",
        "target_logits_indices",
        "bonus_logits_indices",
        "logits_indices",
        "target_model_row_indices",
        "bonus_model_row_indices",
        "pre_hidden_hash",
        "post_hidden_hash",
        "logits_top_k",
    ):
        if key in tensors:
            record[key] = _json_value(tensors[key])
    if pre_hidden is not None and record["pre_hidden_hash"] is None:
        record["pre_hidden_hash"] = hashlib.sha256(
            json.dumps(pre_hidden, sort_keys=True).encode("utf-8")
        ).hexdigest()
    if post_hidden is not None and record["post_hidden_hash"] is None:
        record["post_hidden_hash"] = hashlib.sha256(
            json.dumps(post_hidden, sort_keys=True).encode("utf-8")
        ).hexdigest()

    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"rank{rank}.jsonl"
    with _CAPTURE_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


@torch.no_grad()
def maybe_capture_v1_proposer_prepare_inputs_padded(
    *,
    before_cad: Any,
    after_cad: Any,
    valid_sampled_tokens_count: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens: torch.Tensor,
    spec_step_idx: int = 0,
) -> dict[str, Any] | None:
    """Capture V1 padded proposer metadata before first-pass input expansion."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return None
    fields = {
        **_cad_fields("before", before_cad),
        **_cad_fields("after", after_cad),
        "valid_sampled_tokens_count": valid_sampled_tokens_count,
        "token_indices_to_sample": token_indices_to_sample,
        "num_rejected_tokens": num_rejected_tokens,
    }
    return maybe_capture_mtp_stage(
        "v1_proposer_prepare_inputs_padded", spec_step_idx, fields
    )


@torch.no_grad()
def maybe_capture_v1_proposer_first_pass(
    proposer: Any,
    *,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    before_cad: Any,
    after_cad: Any,
    num_rejected_tokens: torch.Tensor | None,
    num_tokens: int,
    spec_step_idx: int = 0,
) -> dict[str, Any] | None:
    """Capture compact V1 proposer state after first-pass buffers are built.

    The record intentionally excludes hidden states and full logits.  It keeps
    the control-plane tensors needed to audit rejected-token masking, positions,
    slot mappings, and SWA/KV cache metadata drift.
    """
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return None
    fields = {
        **_cad_fields("before", before_cad),
        **_cad_fields("after", after_cad),
        "method": getattr(proposer, "method", None),
        "num_tokens": int(num_tokens),
        "num_speculative_tokens": int(getattr(proposer, "num_speculative_tokens", 0)),
        "needs_extra_input_slots": bool(
            getattr(proposer, "needs_extra_input_slots", False)
        ),
        "extra_slots_per_request": int(
            getattr(proposer, "extra_slots_per_request", 0)
        ),
        "net_num_new_slots_per_request": int(
            getattr(proposer, "net_num_new_slots_per_request", 0)
        ),
        "target_token_ids": target_token_ids,
        "next_token_ids": next_token_ids,
        "target_positions": target_positions,
        "token_indices_to_sample": token_indices_to_sample,
        "num_rejected_tokens": num_rejected_tokens,
        "input_ids_after": proposer.input_ids[:num_tokens],
    }
    positions = getattr(proposer, "positions", None)
    if isinstance(positions, torch.Tensor):
        fields["positions_after"] = positions[:num_tokens]
    is_rejected = getattr(proposer, "is_rejected_token_mask", None)
    if isinstance(is_rejected, torch.Tensor):
        fields["is_rejected_token_mask"] = is_rejected[:num_tokens]
    is_masked = getattr(proposer, "is_masked_token_mask", None)
    if isinstance(is_masked, torch.Tensor):
        fields["is_masked_token_mask"] = is_masked[:num_tokens]
    return maybe_capture_mtp_stage("v1_proposer_first_pass", spec_step_idx, fields)


@torch.no_grad()
def maybe_capture_greedy_verifier(
    request_idx: int,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    accepted_count: int | torch.Tensor,
    committed_ids: torch.Tensor,
    target_logits: torch.Tensor | None = None,
    spec_step_idx: int = 0,
    row_indices: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    cu_num_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Capture the verify/commit boundary for one greedy request.

    ``accepted_count`` is the number of draft tokens accepted, excluding the
    target correction token included in ``committed_ids``.
    """
    fields = {
        "request_idx": request_idx,
        "draft_ids": draft_ids,
        "target_ids": target_ids,
        "accepted_count": accepted_count,
        "committed_ids": committed_ids,
        "logits": target_logits,
    }
    for name, value in (
        ("row_indices", row_indices),
        ("positions", positions),
        ("expanded_local_pos", expanded_local_pos),
        ("expanded_idx_mapping", expanded_idx_mapping),
        ("cu_num_logits", cu_num_logits),
    ):
        if value is not None:
            fields[name] = value
    verify = maybe_capture_mtp_stage("verify", spec_step_idx, fields)
    commit = maybe_capture_mtp_stage("commit", spec_step_idx, fields)
    return verify, commit


def _group_greedy_verifier_tensors(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    positions: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    target_argmax = torch.argmax(target_logits, dim=-1)
    cu = cu_num_logits.detach().cpu().tolist()
    groups: list[dict[str, Any]] = []
    for request_idx in range(len(cu) - 1):
        start = cu[request_idx]
        end = cu[request_idx + 1]
        draft_ids = draft_sampled[start + 1 : end]
        target_ids = target_argmax[start:end]
        accepted_count = 0
        for draft_token, target_token in zip(draft_ids, target_ids):
            if bool((draft_token == target_token).item()):
                accepted_count += 1
            else:
                break

        valid_count = min(int(num_sampled[request_idx].item()), sampled.shape[1])
        row = sampled[request_idx, :valid_count]
        invalid = torch.nonzero(row.eq(-1), as_tuple=False).flatten()
        committed_end = int(invalid[0].item()) if invalid.numel() else row.numel()
        groups.append(
            {
                "request_idx": request_idx,
                "draft_ids": draft_ids,
                "target_ids": target_ids,
                "accepted_count": accepted_count,
                "committed_ids": row[:committed_end],
                "target_logits": target_logits[start:end],
                "row_indices": torch.arange(
                    start, end, dtype=torch.int64, device=target_logits.device
                ),
                "positions": (
                    positions[start:end] if positions is not None else None
                ),
                "expanded_local_pos": (
                    expanded_local_pos[start:end]
                    if expanded_local_pos is not None
                    else None
                ),
                "expanded_idx_mapping": (
                    expanded_idx_mapping[start:end]
                    if expanded_idx_mapping is not None
                    else None
                ),
                "cu_num_logits": cu_num_logits,
            }
        )
    return groups


@torch.no_grad()
def maybe_capture_greedy_verifier_batch(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    spec_step_idx: int = 0,
    positions: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Capture verify/commit records from common rejection-sampler outputs."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return []
    records: list[dict[str, Any]] = []
    for fields in _group_greedy_verifier_tensors(
        target_logits,
        draft_sampled,
        cu_num_logits,
        sampled,
        num_sampled,
        positions=positions,
        expanded_local_pos=expanded_local_pos,
        expanded_idx_mapping=expanded_idx_mapping,
    ):
        verify, commit = maybe_capture_greedy_verifier(
            spec_step_idx=spec_step_idx, **fields
        )
        if verify is not None:
            records.append(verify)
        if commit is not None:
            records.append(commit)
    return records


@torch.no_grad()
def maybe_capture_v1_greedy_verifier_batch(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    sampled: torch.Tensor,
    spec_step_idx: int = 0,
) -> list[dict[str, Any]]:
    """Capture legacy V1 greedy verifier output grouped by request."""
    if _capture_dir() is None or _is_cuda_graph_capturing():
        return []
    target_argmax = torch.argmax(target_logits, dim=-1)
    records: list[dict[str, Any]] = []
    start = 0
    for request_idx, end in enumerate(cu_num_draft_tokens.detach().cpu().tolist()):
        draft_ids = draft_token_ids[start:end]
        target_ids = target_argmax[start:end]
        accepted_count = 0
        for draft_token, target_token in zip(draft_ids, target_ids):
            if bool((draft_token == target_token).item()):
                accepted_count += 1
            else:
                break
        row = sampled[request_idx]
        invalid = torch.nonzero(row.eq(-1), as_tuple=False).flatten()
        committed_end = int(invalid[0].item()) if invalid.numel() else row.numel()
        verify, commit = maybe_capture_greedy_verifier(
            request_idx=request_idx,
            draft_ids=draft_ids,
            target_ids=target_ids,
            accepted_count=accepted_count,
            committed_ids=row[:committed_end],
            target_logits=target_logits[start:end],
            spec_step_idx=spec_step_idx,
        )
        if verify is not None:
            records.append(verify)
        if commit is not None:
            records.append(commit)
        start = end
    return records


def _records(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    paths = sorted(root.glob("rank*.jsonl")) if root.is_dir() else [root]
    records: list[dict[str, Any]] = []
    for source in paths:
        with source.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def analyze_capture(path: str | Path) -> dict[str, Any]:
    """Summarize stage records and locate the first rejected-draft commit bug."""
    records = _records(path)
    last_verify: dict[str, Any] | None = None
    first_invalid: str | None = None
    for record in records:
        boundary = record.get("boundary", record.get("stage"))
        if boundary == "verify":
            last_verify = record
            continue
        if boundary != "commit" or first_invalid is not None:
            continue
        accepted = record.get("accepted_count")
        if accepted is None and last_verify is not None:
            accepted = last_verify.get("accepted_count")
        draft = record.get("draft_ids")
        if draft is None and last_verify is not None:
            draft = last_verify.get("draft_ids")
        target = record.get("target_ids")
        if target is None and last_verify is not None:
            target = last_verify.get("target_ids")
        committed = record.get("committed_ids")
        if accepted == 0 and committed and committed == draft and draft != target:
            first_invalid = "commit"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "first_invalid_boundary": first_invalid,
    }


def diff_captures(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Compare compact records and report the earliest stage-level difference."""
    reference_records = _records(reference)
    candidate_records = _records(candidate)
    first_difference: dict[str, Any] | None = None
    for index, (lhs, rhs) in enumerate(zip(reference_records, candidate_records)):
        for key in (
            "step",
            "stage",
            "input_ids",
            "positions",
            "draft_ids",
            "target_ids",
            "accepted_count",
            "pre_hidden_hash",
            "post_hidden_hash",
            "logits_top_k",
        ):
            if lhs.get(key) != rhs.get(key):
                first_difference = {"index": index, "field": key}
                break
        if first_difference is not None:
            break
    if first_difference is None and len(reference_records) != len(candidate_records):
        first_difference = {
            "index": min(len(reference_records), len(candidate_records)),
            "field": "record_count",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "first_difference": first_difference,
        "first_invalid_boundary": analyze_capture(candidate)[
            "first_invalid_boundary"
        ],
        "reference": analyze_capture(reference),
        "candidate": analyze_capture(candidate),
    }


__all__ = [
    "analyze_capture",
    "diff_captures",
    "maybe_capture_greedy_verifier",
    "maybe_capture_greedy_verifier_batch",
    "maybe_capture_v1_proposer_first_pass",
    "maybe_capture_v1_proposer_prepare_inputs_padded",
    "maybe_capture_v1_model_runner_metadata",
    "maybe_capture_v1_sampler_metadata",
    "maybe_capture_v1_greedy_verifier_batch",
    "maybe_capture_mtp_stage",
]
