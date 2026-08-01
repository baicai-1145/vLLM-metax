"""Opt-in routed-expert differential for DSpark grouped FFN diagnosis."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


_DONE: set[tuple[int, int]] = set()


def _stage_capture_setter(runner: Any) -> tuple[Callable[..., None], str]:
    quant_method = runner.routed_experts.quant_method
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    if moe_kernel is not None:
        module_name = type(moe_kernel.fused_experts).__module__
    else:
        from vllm_metax.utils.fused_moe import get_fused_experts_fn

        module_name = get_fused_experts_fn().__module__
    module = sys.modules.get(module_name)
    setter = getattr(module, "set_dspark_moe_stage_capture", None)
    if setter is None:
        raise RuntimeError(
            "runtime MoE expert module does not expose stage capture: "
            f"module={module_name}"
        )
    return setter, module_name


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK") or "0"
    try:
        return int(value)
    except ValueError:
        return 0


def _is_capturing() -> bool:
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _selected_layers() -> set[int]:
    value = os.getenv("VLLM_METAX_DSV4_MOE_GROUPED_GATE_LAYERS", "0")
    try:
        layers = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            "VLLM_METAX_DSV4_MOE_GROUPED_GATE_LAYERS must contain integers"
        ) from exc
    if not layers or min(layers) < 0:
        raise ValueError(
            "VLLM_METAX_DSV4_MOE_GROUPED_GATE_LAYERS must contain nonnegative integers"
        )
    return layers


def _local_rows_and_routes(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    weights = []
    ids = []
    logits = []
    for index in range(hidden_states.shape[0]):
        row_ids = None if input_ids is None else input_ids[index : index + 1]
        row, shared_input = runner.apply_routed_input_transform(
            hidden_states[index : index + 1]
        )
        row, _, _ = runner._maybe_pad_hidden_states(shared_input, row)
        internal_gate = getattr(runner, "gate", None)
        if internal_gate is None:
            router_logits = router_logits_fn(row, row_ids)
        else:
            router_logits, _ = internal_gate(row)
        row_weights, row_ids_out = runner.router.select_experts(
            hidden_states=row,
            router_logits=router_logits,
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
            input_ids=row_ids,
        )
        rows.append(row)
        weights.append(row_weights)
        ids.append(row_ids_out)
        logits.append(router_logits)
    return torch.cat(rows), torch.cat(weights), torch.cat(ids), torch.cat(logits)


@torch.no_grad()
def compare_grouped_routed_experts(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    *,
    repeats: int = 5,
) -> dict[str, Any]:
    """Compare one native M-row expert call with independent native M=1 calls."""
    if hidden_states.ndim != 2 or not 1 < hidden_states.shape[0] <= 6:
        raise ValueError("hidden_states must have shape [M, H] with 2 <= M <= 6")
    if input_ids is not None and input_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("input_ids rows must match hidden_states rows")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    runner.routed_experts._ensure_moe_quant_config_init()
    rows, topk_weights, topk_ids, router_logits = _local_rows_and_routes(
        runner, hidden_states, input_ids, router_logits_fn
    )
    grouped_topk_weights, grouped_topk_ids = runner.router.select_experts(
        hidden_states=rows,
        router_logits=router_logits,
        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
        input_ids=input_ids,
    )

    stage_calls: list[dict[str, torch.Tensor]] = []
    active_stage_call: dict[str, torch.Tensor] = {}

    def capture_stage(name: str, value: torch.Tensor) -> None:
        nonlocal active_stage_call
        if name == "stage1":
            active_stage_call = {}
        active_stage_call[name] = value.detach().clone()
        if name == "output":
            stage_calls.append(active_stage_call)

    capture_stages = (
        os.getenv("VLLM_METAX_DSV4_MOE_GROUPED_GATE_CAPTURE_STAGES") == "1"
    )
    stage_module = None
    if capture_stages:
        set_stage_capture, stage_module = _stage_capture_setter(runner)
        set_stage_capture(capture_stage)
    try:
        serial = []
        for index in range(rows.shape[0]):
            serial.append(
                runner.routed_experts.forward_modular(
                    x=rows[index : index + 1],
                    topk_weights=topk_weights[index : index + 1],
                    topk_ids=topk_ids[index : index + 1],
                    shared_experts=None,
                    shared_experts_input=None,
                ).clone()
            )
        oracle = torch.cat(serial)

        grouped_replays = []
        for _ in range(repeats):
            grouped_replays.append(
                runner.routed_experts.forward_modular(
                    x=rows,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                ).clone()
            )
    finally:
        if capture_stages:
            set_stage_capture(None)
    grouped = grouped_replays[0]
    mismatch = grouped != oracle
    repeat_mismatches = [
        int(torch.count_nonzero(value != grouped).item()) for value in grouped_replays[1:]
    ]
    abs_diff = (grouped.float() - oracle.float()).abs()
    result = {
        "row_count": int(rows.shape[0]),
        "input_shape": list(rows.shape),
        "input_dtype": str(rows.dtype),
        "input_stride": list(rows.stride()),
        "topk_shape": list(topk_ids.shape),
        "router_id_mismatch_count": int(
            torch.count_nonzero(grouped_topk_ids != topk_ids).item()
        ),
        "router_weight_mismatch_count": int(
            torch.count_nonzero(grouped_topk_weights != topk_weights).item()
        ),
        "router_weight_max_abs": float(
            (grouped_topk_weights.float() - topk_weights.float()).abs().max().item()
        ),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "max_abs": float(abs_diff.max().item()),
        "nan_count": int(torch.isnan(grouped).sum().item()),
        "inf_count": int(torch.isinf(grouped).sum().item()),
        "repeat_mismatch_counts": repeat_mismatches,
        "rows": rows.detach().cpu(),
        "topk_weights": topk_weights.detach().cpu(),
        "topk_ids": topk_ids.detach().cpu(),
        "oracle": oracle.detach().cpu(),
        "grouped": grouped.detach().cpu(),
    }
    if capture_stages:
        expected_stage_calls = int(rows.shape[0]) + repeats
        if len(stage_calls) != expected_stage_calls:
            raise RuntimeError(
                "incomplete routed-expert stage capture: "
                f"module={stage_module} expected={expected_stage_calls} "
                f"observed={len(stage_calls)}"
            )
        serial_stage_calls = stage_calls[: rows.shape[0]]
        grouped_stage_call = stage_calls[rows.shape[0]]
        stage_mismatch_counts = {}
        stage_max_abs = {}
        for stage in ("stage1", "activation", "stage2", "output"):
            stage_oracle = torch.cat([call[stage] for call in serial_stage_calls])
            stage_grouped = grouped_stage_call[stage]
            stage_diff = (stage_grouped.float() - stage_oracle.float()).abs()
            stage_mismatch_counts[stage] = int(
                torch.count_nonzero(stage_grouped != stage_oracle).item()
            )
            stage_max_abs[stage] = float(stage_diff.max().item())
        result["stage_mismatch_counts"] = stage_mismatch_counts
        result["stage_max_abs"] = stage_max_abs
    return result


def maybe_run_grouped_moe_gate(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    layer_idx: int,
) -> Path | None:
    output_dir = os.getenv("VLLM_METAX_DSV4_MOE_GROUPED_GATE_DIR")
    if (
        not output_dir
        or layer_idx not in _selected_layers()
        or hidden_states.shape[0] != 6
        or _is_capturing()
    ):
        return None
    rank = _rank()
    key = (rank, layer_idx)
    every_call = os.getenv("VLLM_METAX_DSV4_MOE_GROUPED_GATE_EVERY_CALL") == "1"
    if key in _DONE and not every_call:
        return None

    payload = compare_grouped_routed_experts(
        runner,
        hidden_states,
        input_ids,
        router_logits_fn,
        repeats=int(os.getenv("VLLM_METAX_DSV4_MOE_GROUPED_GATE_REPEATS", "5")),
    )
    payload.update({"rank": rank, "layer_idx": layer_idx})
    any_mismatch = (
        payload["mismatch_count"]
        or payload["router_id_mismatch_count"]
        or payload["router_weight_mismatch_count"]
    )
    path = Path(output_dir) / f"rank{rank}_layer{layer_idx}_routed_experts.pt"
    if any_mismatch or not every_call:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        _DONE.add(key)
    if any_mismatch and os.getenv(
        "VLLM_METAX_DSV4_MOE_GROUPED_GATE_FAIL_ON_MISMATCH", "1"
    ) == "1":
        raise RuntimeError(
            "grouped routed-expert differential mismatch: "
            f"rank={rank} layer={layer_idx} count={payload['mismatch_count']} "
            f"router_ids={payload['router_id_mismatch_count']} "
            f"router_weights={payload['router_weight_mismatch_count']} "
            f"artifact={path}"
        )
    return path


__all__ = ["compare_grouped_routed_experts", "maybe_run_grouped_moe_gate"]
