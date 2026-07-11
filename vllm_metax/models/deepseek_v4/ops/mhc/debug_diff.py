# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Debug-only real activation differential checks for DeepSeek V4 MHC.

The hooks in this file are intentionally disabled by default.  They are used
to compare the exact runtime inputs of a model forward pass against the Torch
MHC reference and the TileLang MHC implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

_CALL_COUNT = 0
_MISMATCH_COUNT = 0
_RAW_CAPTURE_CALL_COUNT = 0


def enabled() -> bool:
    return os.getenv("VLLM_METAX_DSV4_MHC_DIFF", "0") == "1"


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


def _log_path() -> Path:
    path = os.getenv("VLLM_METAX_DSV4_MHC_DIFF_LOG")
    if path:
        return Path(path)
    return Path("/root/vLLM-metax/.logs") / f"dsv4_mhc_diff_rank{_rank()}.jsonl"


def _max_calls() -> int:
    return int(os.getenv("VLLM_METAX_DSV4_MHC_DIFF_MAX_CALLS", "0"))


def _token_filter() -> int:
    return int(os.getenv("VLLM_METAX_DSV4_MHC_DIFF_TOKENS", "0"))


def _dump_dir() -> Path | None:
    path = os.getenv("VLLM_METAX_DSV4_MHC_DIFF_DUMP_DIR")
    return Path(path) if path else None


def _clone_arg(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().clone()


def _as_raw_bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    if tensor.dtype == torch.float16:
        return tensor.view(torch.int16)
    return tensor


def _ordered_fp32_bits(bits: torch.Tensor) -> torch.Tensor:
    bits64 = bits.to(torch.int64)
    return torch.where(
        bits64 < 0,
        torch.tensor(0x80000000, dtype=torch.int64) - bits64,
        bits64,
    )


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return {
            "equal": False,
            "shape": [list(lhs.shape), list(rhs.shape)],
            "dtype": [str(lhs.dtype), str(rhs.dtype)],
        }
    lhs_cpu = lhs.detach().contiguous().cpu()
    rhs_cpu = rhs.detach().contiguous().cpu()
    lhs_bits = _as_raw_bits(lhs_cpu)
    rhs_bits = _as_raw_bits(rhs_cpu)
    mask = lhs_bits != rhs_bits
    if not bool(mask.any().item()):
        return {"equal": True, "num_diff": 0}

    flat_index = int(mask.flatten().nonzero()[0].item())
    index = list(torch.unravel_index(torch.tensor(flat_index), mask.shape))
    index = [int(item.item()) for item in index]
    lhs_value = lhs_cpu[tuple(index)].float().item()
    rhs_value = rhs_cpu[tuple(index)].float().item()
    result: dict[str, Any] = {
        "equal": False,
        "index": index,
        "lhs": lhs_value,
        "rhs": rhs_value,
        "lhs_bits": int(lhs_bits[tuple(index)].item()),
        "rhs_bits": int(rhs_bits[tuple(index)].item()),
        "abs": abs(lhs_value - rhs_value),
        "max_abs": (lhs_cpu.float() - rhs_cpu.float()).abs().max().item(),
        "num_diff": int(mask.sum().item()),
    }
    if lhs.dtype == torch.float32:
        ulps = _ordered_fp32_bits(lhs_bits) - _ordered_fp32_bits(rhs_bits)
        result["ulp"] = int(ulps[tuple(index)].item())
        result["max_ulp"] = int(ulps.abs().max().item())
    return result


def assert_bitwise_trace_equal(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> None:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise AssertionError(f"trace keys differ missing={missing} extra={extra}")
    for name in reference:
        diff = tensor_diff(reference[name], candidate[name])
        if not diff["equal"]:
            raise AssertionError(f"{name} differs: {diff}")


@torch.no_grad()
def mhc_pre_from_raw_trace_torch(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> dict[str, torch.Tensor]:
    num_tokens = residual_cur.shape[0]
    hc_mult = residual_cur.shape[1]
    hidden_size = residual_cur.shape[2]
    rms_group_size = hc_mult * hidden_size
    trace: dict[str, torch.Tensor] = {}

    sqrsum_reduced = gemm_out_sqrsum.sum(dim=0)
    trace["sqrsum_reduced"] = sqrsum_reduced
    rms = torch.rsqrt(sqrsum_reduced / rms_group_size + rms_eps)
    trace["rms"] = rms
    normalized_mixes = (gemm_out_mul.sum(dim=0) * rms.unsqueeze(-1)).unsqueeze(0)
    trace["normalized_mixes"] = normalized_mixes

    expanded_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    affine_logits = normalized_mixes * expanded_scale + hc_base
    trace["affine_logits"] = affine_logits
    pre_mix = affine_logits[:, :, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        affine_logits[:, :, hc_mult : 2 * hc_mult].sigmoid()
        * hc_post_mult_value
    ).unsqueeze(-1)
    comb_logits = affine_logits[:, :, 2 * hc_mult :].view(
        1, num_tokens, hc_mult, hc_mult
    )
    trace["pre_mix"] = pre_mix.view(num_tokens, hc_mult, 1)
    trace["post_mix"] = post_mix.view(num_tokens, hc_mult, 1)
    trace["comb_logits"] = comb_logits

    comb_mix = comb_logits.softmax(-1) + hc_sinkhorn_eps
    trace["sinkhorn_softmax_eps"] = comb_mix
    comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    trace["sinkhorn_col_0"] = comb_mix.view(num_tokens, hc_mult, hc_mult)
    for index in range(1, sinkhorn_repeat):
        comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        trace[f"sinkhorn_row_{index}"] = comb_mix.view(
            num_tokens, hc_mult, hc_mult
        )
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
        trace[f"sinkhorn_col_{index}"] = comb_mix.view(
            num_tokens, hc_mult, hc_mult
        )

    pre_mix_flat = pre_mix.view(num_tokens, hc_mult, 1)
    layer_products = residual_cur.float() * pre_mix_flat
    trace["layer_products"] = layer_products
    layer_input_fp32 = layer_products.sum(dim=-2)
    trace["layer_input_fp32"] = layer_input_fp32
    trace["layer_input_bf16"] = layer_input_fp32.bfloat16()
    return trace


def reset_mhc_pre_raw_capture_state() -> None:
    global _RAW_CAPTURE_CALL_COUNT
    _RAW_CAPTURE_CALL_COUNT = 0


def _raw_capture_dir() -> Path | None:
    path = os.getenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_DIR")
    return Path(path) if path else None


def _raw_capture_max_calls() -> int:
    return int(os.getenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_MAX_CALLS", "0"))


def _raw_capture_rank_enabled(rank: str) -> bool:
    ranks = os.getenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_RANKS", "all")
    if ranks == "all":
        return True
    return rank in {item.strip() for item in ranks.split(",") if item.strip()}


@torch.no_grad()
def maybe_capture_mhc_pre_raw(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> None:
    global _RAW_CAPTURE_CALL_COUNT
    capture_dir = _raw_capture_dir()
    if capture_dir is None:
        return
    rank = _rank()
    if not _raw_capture_rank_enabled(rank):
        return
    max_calls = _raw_capture_max_calls()
    if max_calls and _RAW_CAPTURE_CALL_COUNT >= max_calls:
        return
    call_index = _RAW_CAPTURE_CALL_COUNT
    _RAW_CAPTURE_CALL_COUNT += 1
    capture_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "rank": int(rank),
            "call": call_index,
            "residual_cur": residual_cur.detach().cpu(),
            "gemm_out_mul": gemm_out_mul.detach().cpu(),
            "gemm_out_sqrsum": gemm_out_sqrsum.detach().cpu(),
            "hc_scale": hc_scale.detach().cpu(),
            "hc_base": hc_base.detach().cpu(),
            "params": {
                "rms_eps": rms_eps,
                "hc_pre_eps": hc_pre_eps,
                "hc_sinkhorn_eps": hc_sinkhorn_eps,
                "hc_post_mult_value": hc_post_mult_value,
                "sinkhorn_repeat": sinkhorn_repeat,
                "n_splits": n_splits,
            },
        },
        capture_dir / f"rank{rank}_call{call_index}.pt",
    )


def _first_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        mask = lhs.view(torch.int16) != rhs.view(torch.int16)
    else:
        mask = lhs != rhs
    if not mask.any().item():
        return {"equal": True}

    flat_index = int(mask.flatten().nonzero()[0].item())
    index = list(torch.unravel_index(torch.tensor(flat_index, device=mask.device), mask.shape))
    index = [int(item.item()) for item in index]
    lhs_value = lhs[tuple(index)].float().item()
    rhs_value = rhs[tuple(index)].float().item()
    max_abs = (lhs.float() - rhs.float()).abs().max().item()
    return {
        "equal": False,
        "index": index,
        "torch": lhs_value,
        "tilelang": rhs_value,
        "abs": abs(lhs_value - rhs_value),
        "max_abs": max_abs,
        "num_diff": int(mask.sum().item()),
    }


def _first_bf16_mismatch(
    names: tuple[str, ...],
    diffs: dict[str, dict[str, Any]],
    outputs: tuple[torch.Tensor, ...],
) -> str | None:
    for name, output in zip(names, outputs):
        if output.dtype == torch.bfloat16 and not diffs[name]["equal"]:
            return name
    return None


def _write(record: dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@torch.no_grad()
def compare_fused_post_pre(
    *,
    layer_idx: int | None,
    stage: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> None:
    global _CALL_COUNT, _MISMATCH_COUNT

    if not enabled():
        return
    max_calls = _max_calls()
    if max_calls and _CALL_COUNT >= max_calls:
        return
    num_tokens = int(residual.reshape(-1, residual.shape[-2], residual.shape[-1]).shape[0])
    token_filter = _token_filter()
    if token_filter and num_tokens != token_filter:
        return

    call_index = _CALL_COUNT
    _CALL_COUNT += 1

    from .torch import mhc_fused_post_pre as torch_fused
    from .tilelang import mhc_fused_post_pre_tilelang as tilelang_fused

    args = (
        _clone_arg(x),
        _clone_arg(residual),
        _clone_arg(post_layer_mix),
        _clone_arg(comb_res_mix),
        _clone_arg(fn),
        _clone_arg(hc_scale),
        _clone_arg(hc_base),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    torch_out = torch_fused(*args)
    tile_args = tuple(_clone_arg(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
    tile_out = tilelang_fused(*tile_args)
    torch.cuda.synchronize()

    names = ("residual_cur", "post_mix", "comb_mix", "layer_input")
    diffs = {name: _first_diff(lhs, rhs) for name, lhs, rhs in zip(names, torch_out, tile_out)}
    first_stage = next((name for name in names if not diffs[name]["equal"]), None)
    first_bf16_stage = _first_bf16_mismatch(names, diffs, torch_out)
    record = {
        "call": call_index,
        "rank": _rank(),
        "layer": layer_idx,
        "stage": stage,
        "num_tokens": num_tokens,
        "hidden": int(residual.shape[-1]),
        "first_mismatch": first_stage,
        "first_any_mismatch": first_stage,
        "first_bf16_mismatch": first_bf16_stage,
        "diffs": diffs,
    }
    if first_stage is not None:
        _MISMATCH_COUNT += 1
        dump_dir = _dump_dir()
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / (
                f"rank{_rank()}_call{call_index}_{stage}_layer{layer_idx}.pt"
            )
            torch.save(
                {
                    "record": record,
                    "inputs": {
                        "x": args[0].cpu(),
                        "residual": args[1].cpu(),
                        "post_layer_mix": args[2].cpu(),
                        "comb_res_mix": args[3].cpu(),
                        "fn": args[4].cpu(),
                        "hc_scale": args[5].cpu(),
                        "hc_base": args[6].cpu(),
                    },
                    "params": {
                        "rms_eps": rms_eps,
                        "hc_pre_eps": hc_pre_eps,
                        "hc_sinkhorn_eps": hc_sinkhorn_eps,
                        "hc_post_mult_value": hc_post_mult_value,
                        "sinkhorn_repeat": sinkhorn_repeat,
                        "n_splits": n_splits,
                    },
                    "torch_out": tuple(out.cpu() for out in torch_out),
                    "tilelang_out": tuple(out.cpu() for out in tile_out),
                },
                dump_path,
            )
            record["dump_path"] = str(dump_path)
    _write(record)
