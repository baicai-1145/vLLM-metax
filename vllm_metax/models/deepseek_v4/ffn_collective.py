"""Row-exact TP reduction coalescing for DeepSeek-V4 MoE calls."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


_FFN_DIFF_DONE: set[tuple[int, str]] = set()


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.getenv("RANK") or os.getenv("LOCAL_RANK") or "0")


def _is_capturing() -> bool:
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _unpack_moe_result(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise RuntimeError("MoE local output must contain shared and fused tensors")
        return result
    return None, result


def _validate_runner(runner: Any) -> None:
    config = runner.moe_config
    if runner._fused_output_is_reduced:
        raise RuntimeError(
            "cannot coalesce a MoE backend that already reduces fused output"
        )
    if config.is_sequence_parallel:
        raise RuntimeError(
            "cannot coalesce MoE reductions owned by sequence-parallel execution"
        )
    if config.skip_final_all_reduce:
        raise RuntimeError(
            "cannot coalesce when skip_final_all_reduce is already enabled"
        )
    if config.ep_size != 1:
        raise RuntimeError("MoE row reduction coalescing currently requires EP=1")
    if config.tp_size <= 1:
        raise RuntimeError("MoE row reduction coalescing requires TP > 1")
    if runner.router.__class__.__name__ == "ZeroExpertRouter":
        raise RuntimeError("MoE row reduction coalescing does not support zero experts")


def _local_moe_row(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
) -> torch.Tensor:
    hidden_states, shared_input = runner.apply_routed_input_transform(hidden_states)
    hidden_states, original_pre, original_post = runner._maybe_pad_hidden_states(
        shared_input, hidden_states
    )
    router_logits = router_logits_fn(hidden_states, input_ids)
    result = runner._forward_entry(
        hidden_states,
        router_logits,
        shared_input,
        input_ids,
        runner._encode_layer_name(),
        runner.moe_config.hidden_dim_unpadded
        if runner._quant_method.has_unpadded_output
        else 0,
    )
    shared_output, fused_output = _unpack_moe_result(result)
    if original_pre is not None:
        fused_output = fused_output[..., :original_pre]
    shared_output, fused_output = runner._maybe_apply_routed_scale_to_output(
        shared_output, fused_output
    )
    fused_output = runner.apply_routed_output_transform(fused_output)
    combined = fused_output if shared_output is None else shared_output + fused_output
    if original_post is not None:
        combined = combined[..., :original_post]
    return combined


def _grouped_local_moe_rows(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    *,
    group_router: bool,
) -> torch.Tensor:
    if runner.routed_experts.quant_method.is_monolithic:
        raise RuntimeError("grouped routed experts require a modular MoE backend")
    runner.routed_experts._ensure_moe_quant_config_init()

    routed_inputs = []
    topk_weights = []
    topk_ids = []
    router_logits_rows = []
    row_metadata = []
    retain_shared_outputs = (
        os.getenv("VLLM_METAX_DSV4_FFN_RETAIN_SHARED_OUTPUTS", "0") == "1"
    )
    # When VLLM_METAX_DSV4_SHARED_EXPERT_MSAFE=1, batch the shared-expert
    # computation into a single M=6 call using an M-invariant Triton matmul.
    # This replaces the 6x M=1 loop while preserving tokenwise semantics.
    shared_experts = getattr(runner, "_shared_experts", None)
    use_msafe_shared = (
        shared_experts is not None
        and getattr(runner, "routed_input_transform", None) is None
        and os.getenv("VLLM_METAX_DSV4_SHARED_EXPERT_MSAFE", "0") == "1"
    )
    shared_output_all = None
    if use_msafe_shared:
        # Batch shared-expert computation into a single M=6 call.
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )

        runner._maybe_sync_shared_experts_stream(hidden_states)
        order_all = shared_experts._determine_shared_experts_order(hidden_states)
        shared_experts(hidden_states, order_all)
        shared_output_all = shared_experts.output
        if not retain_shared_outputs:
            shared_output_all = shared_output_all.clone()
    for index in range(hidden_states.shape[0]):
        row_input_ids = None if input_ids is None else input_ids[index : index + 1]
        row, shared_input = runner.apply_routed_input_transform(
            hidden_states[index : index + 1]
        )
        row, original_pre, original_post = runner._maybe_pad_hidden_states(
            shared_input, row
        )

        internal_gate = getattr(runner, "gate", None)
        if internal_gate is None:
            router_logits = router_logits_fn(row, row_input_ids)
        else:
            router_logits, _ = internal_gate(row)

        runner._maybe_sync_shared_experts_stream(shared_input)
        shared_output = None
        if use_msafe_shared:
            shared_output = shared_output_all[index : index + 1]
        elif shared_experts is not None:
            if shared_input is None:
                raise RuntimeError("shared experts require a transformed shared input")
            order = shared_experts._determine_shared_experts_order(shared_input)
            shared_experts(shared_input, order)
            shared_output = shared_experts.output
            if not retain_shared_outputs:
                shared_output = shared_output.clone()

        routed_inputs.append(row)
        router_logits_rows.append(router_logits)
        if not group_router:
            row_weights, row_ids = runner.router.select_experts(
                hidden_states=row,
                router_logits=router_logits,
                topk_indices_dtype=runner._quant_method.topk_indices_dtype,
                input_ids=row_input_ids,
            )
            topk_weights.append(row_weights)
            topk_ids.append(row_ids)
        row_metadata.append((shared_output, original_pre, original_post))

    grouped_inputs = torch.cat(routed_inputs)
    if group_router:
        grouped_weights, grouped_ids = runner.router.select_experts(
            hidden_states=grouped_inputs,
            router_logits=torch.cat(router_logits_rows),
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
    else:
        grouped_weights = torch.cat(topk_weights)
        grouped_ids = torch.cat(topk_ids)
    fused_output = runner.routed_experts.forward_modular(
        x=grouped_inputs,
        topk_weights=grouped_weights,
        topk_ids=grouped_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    retain_row_views = (
        os.getenv("VLLM_METAX_DSV4_FFN_RETAIN_ROW_OUTPUT_VIEWS", "0") == "1"
    )
    combined_rows = []
    for index, (shared_output, original_pre, original_post) in enumerate(row_metadata):
        row_output = fused_output[index : index + 1]
        if original_pre is not None:
            row_output = row_output[..., :original_pre]
        shared_output, row_output = runner._maybe_apply_routed_scale_to_output(
            shared_output, row_output
        )
        row_output = runner.apply_routed_output_transform(row_output)
        if shared_output is not None:
            row_output = shared_output + row_output
        if original_post is not None:
            row_output = row_output[..., :original_post]
        combined_rows.append(row_output if retain_row_views else row_output.clone())
    return torch.cat(combined_rows)


def coalesce_moe_row_reductions(
    runner: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    router_logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    *,
    group_rows: int | None = None,
    group_routed_experts: bool = False,
    group_router: bool = False,
    all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run native MoE rows independently and reduce contiguous row groups."""
    _validate_runner(runner)
    if hidden_states.ndim < 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must contain at least one row")
    if input_ids is not None and input_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("input_ids rows must match hidden_states rows")
    if group_rows is None:
        group_rows = hidden_states.shape[0]
    if group_rows < 1:
        raise ValueError("group_rows must be positive")
    if all_reduce is None:
        from vllm.distributed import tensor_model_parallel_all_reduce

        all_reduce = tensor_model_parallel_all_reduce

    reduced_groups = []
    for start in range(0, hidden_states.shape[0], group_rows):
        end = min(start + group_rows, hidden_states.shape[0])
        if group_routed_experts:
            diff_dir = os.getenv("VLLM_METAX_DSV4_GROUPED_FFN_DIFF_DIR")
            layer_name = runner._encode_layer_name()
            rank = _rank()
            diff_key = (rank, layer_name)
            diff_every_call = (
                os.getenv("VLLM_METAX_DSV4_GROUPED_FFN_DIFF_EVERY_CALL") == "1"
            )
            run_diff = (
                diff_dir is not None
                and (diff_every_call or diff_key not in _FFN_DIFF_DONE)
                and not _is_capturing()
            )
            if run_diff:
                serial_rows = [
                    _local_moe_row(
                        runner,
                        hidden_states[index : index + 1],
                        None if input_ids is None else input_ids[index : index + 1],
                        router_logits_fn,
                    ).clone()
                    for index in range(start, end)
                ]
                serial_output = torch.cat(serial_rows)
            local_output = _grouped_local_moe_rows(
                runner,
                hidden_states[start:end],
                None if input_ids is None else input_ids[start:end],
                router_logits_fn,
                group_router=group_router,
            )
            if run_diff:
                mismatch_count = int(
                    torch.count_nonzero(local_output != serial_output).item()
                )
                max_abs = float(
                    (local_output.float() - serial_output.float()).abs().max().item()
                )
                path = None
                if mismatch_count or not diff_every_call:
                    safe_layer_name = layer_name.replace(".", "_")
                    path = (
                        Path(diff_dir)
                        / f"rank{rank}_{safe_layer_name}_local_ffn_diff.pt"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "rank": rank,
                            "layer_name": layer_name,
                            "mismatch_count": mismatch_count,
                            "max_abs": max_abs,
                            "hidden_states": hidden_states[start:end].detach().cpu(),
                            "serial_output": serial_output.detach().cpu(),
                            "grouped_output": local_output.detach().cpu(),
                        },
                        path,
                    )
                    _FFN_DIFF_DONE.add(diff_key)
                if mismatch_count:
                    raise RuntimeError(
                        "grouped FFN local differential mismatch: "
                        f"rank={rank} layer={layer_name} count={mismatch_count} "
                        f"max_abs={max_abs} artifact={path}"
                    )
        else:
            rows = [
                _local_moe_row(
                    runner,
                    hidden_states[index : index + 1],
                    None if input_ids is None else input_ids[index : index + 1],
                    router_logits_fn,
                ).clone()
                for index in range(start, end)
            ]
            local_output = torch.cat(rows, dim=0)
        reduced_groups.append(all_reduce(local_output))
    if (
        len(reduced_groups) == 1
        and os.getenv("VLLM_METAX_DSV4_RETURN_SINGLE_REDUCED_GROUP", "0") == "1"
    ):
        return reduced_groups[0]
    return torch.cat(reduced_groups, dim=0)


__all__ = ["coalesce_moe_row_reductions"]
