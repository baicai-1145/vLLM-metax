"""Row-exact TP reduction coalescing for DeepSeek-V4 O-projection."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from vllm.logger import init_logger


logger = init_logger(__name__)

_NATIVE_SERIAL_O_PROJ_ROWS_ENV = "VLLM_METAX_DSV4_NATIVE_SERIAL_O_PROJ_ROWS"
_EXACT_GROUPED_O_PROJ_ROWS_ENV = "VLLM_METAX_DSV4_EXACT_GROUPED_O_PROJ_ROWS"
_EXACT_OPROJ_ROW_LIST_ENV = "VLLM_METAX_DSV4_EXACT_OPROJ_ROW_LIST"


def _validate_wo_b(wo_b: nn.Module) -> Callable:
    expected = {
        "input_is_parallel": True,
        "reduce_results": True,
        "return_bias": False,
        "skip_bias_add": False,
        "tp_size": 4,
    }
    mismatches = [
        f"{name}={getattr(wo_b, name, None)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(wo_b, name, None) != value
    ]
    if getattr(wo_b, "bias", None) is not None:
        mismatches.append("bias must be None")
    apply = getattr(getattr(wo_b, "quant_method", None), "apply", None)
    if not callable(apply):
        mismatches.append("quant_method.apply must be callable")
    if mismatches:
        raise RuntimeError(
            "O-projection collective coalescing requires the wo_b "
            "RowParallelLinear contract; "
            + "; ".join(mismatches)
        )
    return apply


def coalesce_wo_b_row_reductions(
    wo_b: nn.Module,
    row_inputs: Sequence[torch.Tensor],
    *,
    group_rows: int | None = None,
    all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply native ``wo_b`` per row and reduce contiguous row groups."""
    apply = _validate_wo_b(wo_b)
    if not row_inputs:
        raise ValueError("row_inputs must not be empty")
    first = row_inputs[0]
    if first.ndim < 2 or first.shape[0] != 1:
        raise ValueError("each O-projection input must contain exactly one row")
    if group_rows is None:
        group_rows = len(row_inputs)
    if group_rows < 1:
        raise ValueError("group_rows must be positive")
    for row in row_inputs[1:]:
        if row.shape != first.shape or row.dtype != first.dtype:
            raise ValueError("O-projection row inputs must share shape and dtype")
    if all_reduce is None:
        from vllm.distributed import tensor_model_parallel_all_reduce

        all_reduce = tensor_model_parallel_all_reduce
    native_serial_rows = (
        os.getenv(_NATIVE_SERIAL_O_PROJ_ROWS_ENV, "0") == "1"
    )
    if native_serial_rows:
        quant_method = wo_b.quant_method
        weight = getattr(wo_b, "weight", None)
        if quant_method.__class__.__name__ != "UnquantizedLinearMethod":
            raise RuntimeError(
                "native serial O-projection rows require UnquantizedLinearMethod"
            )
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            raise RuntimeError(
                "native serial O-projection rows require contiguous BF16 weight"
            )
        if any(
            row.dtype != torch.bfloat16 or not row.is_contiguous()
            for row in row_inputs
        ):
            raise RuntimeError(
                "native serial O-projection rows require contiguous BF16 inputs"
            )
        exact_grouped = os.getenv(_EXACT_GROUPED_O_PROJ_ROWS_ENV, "0") == "1"
        op_name = (
            "gemv_bf16_exact_oproj_grouped_rows_out"
            if exact_grouped
            else "gemv_bf16_serial_rows_out"
        )
        try:
            op = getattr(torch.ops._metax_sparse_C, op_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"native O-projection rows operator {op_name} is unavailable"
            ) from exc

        workspaces = getattr(wo_b, "_native_serial_o_proj_workspaces", None)
        if workspaces is None:
            workspaces = {}
            wo_b._native_serial_o_proj_workspaces = workspaces
        reduced_groups = []
        for start in range(0, len(row_inputs), group_rows):
            group = row_inputs[start : start + group_rows]
            rows = len(group)
            key = (rows, weight.shape[0], first.dtype, first.device)
            local_output = workspaces.get(key)
            if local_output is None:
                local_output = torch.empty(
                    (rows, weight.shape[0]),
                    dtype=first.dtype,
                    device=first.device,
                )
                workspaces[key] = local_output
            if (
                exact_grouped
                and os.getenv(_EXACT_OPROJ_ROW_LIST_ENV, "0") == "1"
            ):
                try:
                    row_list_op = torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_row_list_out
                except AttributeError as exc:
                    raise RuntimeError(
                        "native O-projection row-list operator is unavailable"
                    ) from exc
                row_list_op(list(group), weight, local_output)
            else:
                grouped_input = torch.cat(group, dim=0)
                op(grouped_input, weight, local_output)
            reduced_groups.append(all_reduce(local_output))
        launches = len(row_inputs)
        logger.warning_once(
            "DeepSeek V4 O-projection uses native %s output workspace: "
            "rows=%d launches=%d",
            "exact-grouped-row dispatch" if exact_grouped else "serial-row",
            len(row_inputs),
            launches,
        )
        if (
            len(reduced_groups) == 1
            and os.getenv("VLLM_METAX_DSV4_RETURN_SINGLE_REDUCED_GROUP", "0")
            == "1"
        ):
            return reduced_groups[0]
        return torch.cat(reduced_groups, dim=0)

    reduced_groups = []
    for start in range(0, len(row_inputs), group_rows):
        local_rows = []
        for row in row_inputs[start : start + group_rows]:
            local = apply(wo_b, row, None)
            if not isinstance(local, torch.Tensor):
                raise RuntimeError("wo_b quant_method.apply must return a tensor")
            local_rows.append(local.clone())
        reduced_groups.append(all_reduce(torch.cat(local_rows, dim=0)))
    if (
        len(reduced_groups) == 1
        and os.getenv("VLLM_METAX_DSV4_RETURN_SINGLE_REDUCED_GROUP", "0") == "1"
    ):
        return reduced_groups[0]
    return torch.cat(reduced_groups, dim=0)


__all__ = ["coalesce_wo_b_row_reductions"]
