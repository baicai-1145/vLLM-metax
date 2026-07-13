#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay captured DeepSeek V4 MHC raw-boundary inputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from vllm_metax.models.deepseek_v4.ops.mhc.debug_diff import (
    assert_bitwise_trace_equal,
    mhc_pre_from_raw_trace_torch,
    tensor_diff,
)


_CORPUS_RE = re.compile(r"rank(?P<rank>\d+)_call(?P<call>\d+)\.pt$")
_REQUIRED_KEYS = {
    "schema_version",
    "rank",
    "call",
    "residual_cur",
    "gemm_out_mul",
    "gemm_out_sqrsum",
    "hc_scale",
    "hc_base",
    "params",
}
_REQUIRED_PARAMS = {
    "rms_eps",
    "hc_pre_eps",
    "hc_sinkhorn_eps",
    "hc_post_mult_value",
    "sinkhorn_repeat",
    "n_splits",
}
_RAW_NORM_REQUIRED_KEYS = _REQUIRED_KEYS | {
    "layer_idx",
    "stage",
    "fn",
    "pre_norm_output",
    "norm_weight",
    "normalized_output",
}
_FUSED_REQUIRED_KEYS = {
    "schema_version",
    "rank",
    "call",
    "x_flat",
    "residual_flat",
    "post_layer_mix_flat",
    "comb_res_mix_flat",
    "residual_cur_bf16",
    "params",
}


def _file_key(path: Path) -> tuple[int, int]:
    match = _CORPUS_RE.match(path.name)
    if match is None:
        raise ValueError(f"malformed corpus filename: {path}")
    return int(match.group("rank")), int(match.group("call"))


def iter_corpus_files(corpus: str | Path) -> list[Path]:
    corpus_path = Path(corpus)
    files = [path for path in corpus_path.glob("rank*_call*.pt") if path.is_file()]
    return sorted(files, key=_file_key)


def _require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ValueError(f"{path}: {message}")


def _validate_payload(payload: dict[str, Any], path: Path) -> None:
    schema_version = payload.get("schema_version")
    required_keys = _REQUIRED_KEYS if schema_version == 1 else _RAW_NORM_REQUIRED_KEYS
    _require(set(payload) == required_keys, path, f"keys={sorted(payload)}")
    _require(schema_version in (1, 2), path, "schema_version must be 1 or 2")
    file_rank, file_call = _file_key(path)
    _require(payload["rank"] == file_rank, path, "rank does not match filename")
    _require(payload["call"] == file_call, path, "call does not match filename")

    residual_cur = payload["residual_cur"]
    gemm_out_mul = payload["gemm_out_mul"]
    gemm_out_sqrsum = payload["gemm_out_sqrsum"]
    hc_scale = payload["hc_scale"]
    hc_base = payload["hc_base"]
    params = payload["params"]
    _require(set(params) == _REQUIRED_PARAMS, path, f"params={sorted(params)}")
    _require(params["sinkhorn_repeat"] == 20, path, "sinkhorn_repeat must be 20")
    _require(params["n_splits"] == 1, path, "n_splits must be 1")

    _require(isinstance(residual_cur, torch.Tensor), path, "residual_cur is not tensor")
    _require(residual_cur.dtype == torch.bfloat16, path, "residual_cur dtype")
    _require(residual_cur.ndim == 3, path, "residual_cur rank")
    _require(residual_cur.shape[0] == 1, path, "residual_cur token dim")
    _require(residual_cur.shape[1] == 4, path, "residual_cur hc_mult")
    hidden_size = int(residual_cur.shape[2])
    _require(hidden_size > 0, path, "residual_cur hidden dim")

    _require(isinstance(gemm_out_mul, torch.Tensor), path, "gemm_out_mul is not tensor")
    _require(gemm_out_mul.dtype == torch.float32, path, "gemm_out_mul dtype")
    _require(tuple(gemm_out_mul.shape) == (1, 1, 24), path, "gemm_out_mul shape")
    _require(
        isinstance(gemm_out_sqrsum, torch.Tensor),
        path,
        "gemm_out_sqrsum is not tensor",
    )
    _require(gemm_out_sqrsum.dtype == torch.float32, path, "gemm_out_sqrsum dtype")
    _require(
        tuple(gemm_out_sqrsum.shape) == (1, 1),
        path,
        "gemm_out_sqrsum shape",
    )
    _require(isinstance(hc_scale, torch.Tensor), path, "hc_scale is not tensor")
    _require(hc_scale.dtype == torch.float32, path, "hc_scale dtype")
    _require(tuple(hc_scale.shape) == (3,), path, "hc_scale shape")
    _require(isinstance(hc_base, torch.Tensor), path, "hc_base is not tensor")
    _require(hc_base.dtype == torch.float32, path, "hc_base dtype")
    _require(tuple(hc_base.shape) == (24,), path, "hc_base shape")
    if schema_version == 2:
        _require(payload["stage"] in ("attn", "ffn"), path, "invalid stage")
        _require(isinstance(payload["layer_idx"], int), path, "layer_idx")
        _require(tuple(payload["fn"].shape) == (24, 16384), path, "fn shape")
        _require(payload["fn"].dtype == torch.float32, path, "fn dtype")
        _require(
            tuple(payload["pre_norm_output"].shape) == (1, hidden_size),
            path,
            "pre_norm_output shape",
        )
        _require(
            tuple(payload["norm_weight"].shape) == (hidden_size,),
            path,
            "norm_weight shape",
        )
        _require(
            tuple(payload["normalized_output"].shape) == (1, hidden_size),
            path,
            "normalized_output shape",
        )


def load_payload(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    payload_path = Path(path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    _validate_payload(payload, payload_path)
    out = dict(payload)
    for key in (
        "residual_cur",
        "gemm_out_mul",
        "gemm_out_sqrsum",
        "hc_scale",
        "hc_base",
        "fn",
        "pre_norm_output",
        "norm_weight",
        "normalized_output",
    ):
        if key in out:
            out[key] = out[key].to(device)
    return out


def _rms_norm_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    value = payload["pre_norm_output"].float()
    weight = payload["norm_weight"].float()
    eps = payload["params"]["rms_eps"]
    return (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight
    ).to(payload["normalized_output"].dtype)


def _native_rms_norm_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    if payload["pre_norm_output"].device.type != "cuda":
        raise ValueError("native RMSNorm replay requires --device cuda")
    out = torch.empty_like(payload["normalized_output"])
    torch.ops._C.rms_norm(
        out,
        payload["pre_norm_output"],
        payload["norm_weight"],
        payload["params"]["rms_eps"],
    )
    return out


def _ir_rms_norm_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    from vllm import ir

    return ir.ops.rms_norm(
        payload["pre_norm_output"],
        payload["norm_weight"],
        payload["params"]["rms_eps"],
        None,
    )


def load_fused_post_payload(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    payload_path = Path(path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    _require(set(payload) >= _FUSED_REQUIRED_KEYS, payload_path, "fused keys")
    _require(payload["schema_version"] == 2, payload_path, "schema_version must be 2")
    file_rank, file_call = _file_key(payload_path)
    _require(payload["rank"] == file_rank, payload_path, "rank does not match filename")
    _require(payload["call"] == file_call, payload_path, "call does not match filename")
    _require(tuple(payload["x_flat"].shape) == (1, 4096), payload_path, "x shape")
    _require(
        tuple(payload["residual_flat"].shape) == (1, 4, 4096),
        payload_path,
        "residual shape",
    )
    _require(
        tuple(payload["post_layer_mix_flat"].shape) == (1, 4),
        payload_path,
        "post shape",
    )
    _require(
        tuple(payload["comb_res_mix_flat"].shape) == (1, 4, 4),
        payload_path,
        "comb shape",
    )
    _require(payload["x_flat"].dtype == torch.bfloat16, payload_path, "x dtype")
    _require(
        payload["residual_flat"].dtype == torch.bfloat16, payload_path, "residual dtype"
    )
    _require(
        payload["post_layer_mix_flat"].dtype == torch.float32,
        payload_path,
        "post dtype",
    )
    _require(
        payload["comb_res_mix_flat"].dtype == torch.float32, payload_path, "comb dtype"
    )
    _require(
        payload["residual_cur_bf16"].dtype == torch.bfloat16,
        payload_path,
        "output dtype",
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }


def run_exact_mhc_post_tilelang(
    payload: dict[str, Any], out: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_post_exact_tl,
    )

    return _mhc_post_exact_tl(
        payload["x_flat"],
        payload["residual_flat"],
        payload["post_layer_mix_flat"],
        payload["comb_res_mix_flat"],
        out=out,
    )


def run_post_diff(
    corpus: str | Path,
    *,
    device: str,
    require_bitwise: bool,
    check_graph_replay: bool,
    max_files: int | None,
) -> dict[str, Any]:
    files = iter_corpus_files(corpus)
    if max_files is not None:
        files = files[:max_files]
    result: dict[str, Any] = {
        "files": len(files),
        "passed": 0,
        "failed": 0,
        "first_failure": None,
        "bitwise": bool(require_bitwise),
    }
    for path in files:
        payload = load_fused_post_payload(path, device=device)
        got = run_exact_mhc_post_tilelang(payload)
        diff = tensor_diff(payload["residual_cur_bf16"], got)
        if diff["equal"]:
            result["passed"] += 1
        else:
            result["failed"] += 1
            if result["first_failure"] is None:
                result["first_failure"] = {"file": str(path), "diff": diff}
    if check_graph_replay:
        result["graph_replay"] = check_post_graph_replay(files, device=device)
    return result


def check_post_graph_replay(files: list[Path], *, device: str) -> dict[str, Any]:
    if device != "cuda":
        return {"skipped": True, "reason": "cuda graph replay requires cuda device"}
    if len(files) < 2:
        raise ValueError("graph replay requires at least two fused payloads")
    first, second = [load_fused_post_payload(path, device=device) for path in files[:2]]
    keys = ("x_flat", "residual_flat", "post_layer_mix_flat", "comb_res_mix_flat")
    inputs = {key: first[key].detach().clone() for key in keys}
    output = torch.empty_like(first["residual_cur_bf16"])
    for _ in range(3):
        run_exact_mhc_post_tilelang({**first, **inputs})
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    input_ptrs = {key: value.data_ptr() for key, value in inputs.items()}
    with torch.cuda.graph(graph):
        run_exact_mhc_post_tilelang({**first, **inputs}, out=output)
    output_ptr = output.data_ptr()
    for source in (first, second):
        for key in keys:
            inputs[key].copy_(source[key])
        graph.replay()
        torch.cuda.synchronize()
        if not tensor_diff(source["residual_cur_bf16"], output)["equal"]:
            raise AssertionError(f"post graph replay mismatch: {source}")
        if output.data_ptr() != output_ptr or any(
            inputs[key].data_ptr() != input_ptrs[key] for key in keys
        ):
            raise AssertionError("post graph replay pointer changed")
    return {"passed": True, "replays": 2, "pointers_stable": True}


def _torch_trace_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = payload["params"]
    return mhc_pre_from_raw_trace_torch(
        payload["residual_cur"],
        payload["gemm_out_mul"],
        payload["gemm_out_sqrsum"],
        payload["hc_scale"],
        payload["hc_base"],
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
    )


def run_exact_mhc_pre_from_raw_tilelang(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_pre_from_raw_exact_trace,
    )

    params = payload["params"]
    return _mhc_pre_from_raw_exact_trace(
        payload["residual_cur"],
        payload["gemm_out_mul"],
        payload["gemm_out_sqrsum"],
        payload["hc_scale"],
        payload["hc_base"],
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
        params["n_splits"],
    )


def run_mhc_pre_big_fuse_native(
    payload: dict[str, Any],
    *,
    debug_comb_stage: int = -1,
) -> dict[str, torch.Tensor]:
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_pre_big_fuse,
    )

    residual_cur = payload["residual_cur"]
    params = payload["params"]
    num_tokens, mhc_mult, hidden_size = residual_cur.shape
    post_mix = torch.empty(
        (num_tokens, mhc_mult), dtype=torch.float32, device=residual_cur.device
    )
    comb_mix = torch.empty(
        (num_tokens, mhc_mult * mhc_mult),
        dtype=torch.float32,
        device=residual_cur.device,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=residual_cur.device,
    )
    _mhc_pre_big_fuse(
        hidden_size,
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
        n_splits=params["n_splits"],
        mhc_mult=mhc_mult,
        debug_comb_stage=debug_comb_stage,
    )(
        payload["gemm_out_mul"],
        payload["gemm_out_sqrsum"],
        payload["hc_scale"],
        payload["hc_base"],
        residual_cur,
        post_mix,
        comb_mix,
        layer_input,
    )
    return {
        "post_mix": post_mix.view(num_tokens, mhc_mult, 1),
        "sinkhorn_col_19": comb_mix.view(num_tokens, mhc_mult, mhc_mult),
        "layer_input_bf16": layer_input,
    }


def run_mhc_big_fuse_comb_stage_native(
    payload: dict[str, Any], stage: int
) -> dict[str, torch.Tensor]:
    trace = run_mhc_pre_big_fuse_native(payload, debug_comb_stage=stage)
    name = ("comb_logits", "sinkhorn_softmax_eps", "sinkhorn_col_0")[stage]
    value = trace["sinkhorn_col_19"]
    if stage < 2:
        value = value.unsqueeze(0)
    return {name: value}


def run_mhc_pre_split_exact_native(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    from vllm_metax.kernels.sparse_mla_decode import _softmax_fp32_out_op

    trace = run_mhc_pre_big_fuse_native(payload, debug_comb_stage=0)
    comb_mix = trace["sinkhorn_col_19"]
    _softmax_fp32_out_op()(comb_mix.view(-1, 4), comb_mix.view(-1, 4))
    sinkhorn_op = getattr(
        getattr(torch.ops, "_metax_sparse_C", None),
        "mhc_sinkhorn_fp32_out",
        None,
    )
    if sinkhorn_op is None:
        raise RuntimeError("native MHC replay requires mhc_sinkhorn_fp32_out")
    sinkhorn_op(
        comb_mix,
        comb_mix,
        payload["params"]["hc_sinkhorn_eps"],
        payload["params"]["sinkhorn_repeat"],
    )
    return trace


def run_mhc_pre_rms_split_exact_native(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    from vllm_metax.kernels.sparse_mla_decode import _softmax_fp32_out_op

    trace = run_mhc_pre_big_fuse_native(payload, debug_comb_stage=0)
    comb_mix = trace["sinkhorn_col_19"]
    _softmax_fp32_out_op()(comb_mix.view(-1, 4), comb_mix.view(-1, 4))
    normalized = torch.empty_like(payload["normalized_output"])
    sinkhorn_rms_op = getattr(
        getattr(torch.ops, "_metax_sparse_C", None),
        "mhc_sinkhorn_rms_norm_out",
        None,
    )
    if sinkhorn_rms_op is None:
        raise RuntimeError("native MHC replay requires mhc_sinkhorn_rms_norm_out")
    sinkhorn_rms_op(
        comb_mix,
        trace["layer_input_bf16"],
        payload["norm_weight"],
        comb_mix,
        normalized,
        payload["params"]["hc_sinkhorn_eps"],
        payload["params"]["rms_eps"],
        payload["params"]["sinkhorn_repeat"],
    )
    trace["normalized_output"] = normalized
    return trace


def run_mhc_three_kernel_exact_native(
    payload: dict[str, Any],
    buffers: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    import vllm_metax._metax_sparse_C  # noqa: F401

    params = payload["params"]
    residual = payload["residual_cur"]
    if buffers is None:
        buffers = {
            "residual_fp32": torch.empty(
                (1, residual.numel()), dtype=torch.float32, device=residual.device
            ),
            "sqrsum": torch.empty_like(payload["gemm_out_sqrsum"]),
            "gemm_out": torch.empty_like(payload["gemm_out_mul"]),
            "post_mix": torch.empty(
                (1, 4), dtype=torch.float32, device=residual.device
            ),
            "comb_mix": torch.empty(
                (1, 4, 4), dtype=torch.float32, device=residual.device
            ),
            "pre_norm": torch.empty_like(payload["pre_norm_output"]),
            "normalized": torch.empty_like(payload["normalized_output"]),
        }
    residual_fp32 = buffers["residual_fp32"]
    sqrsum = buffers["sqrsum"]
    gemm_out = buffers["gemm_out"]
    post_mix = buffers["post_mix"]
    comb_mix = buffers["comb_mix"]
    pre_norm = buffers["pre_norm"]
    normalized = buffers["normalized"]
    ops = torch.ops._metax_sparse_C
    ops.mhc_cast_sqrsum_out(residual, residual_fp32, sqrsum)
    ops.mhc_gemv_fp32_out(residual_fp32, payload["fn"], gemm_out)
    ops.mhc_downstream_rms_out(
        residual,
        gemm_out,
        sqrsum,
        payload["hc_scale"],
        payload["hc_base"],
        payload["norm_weight"],
        post_mix,
        comb_mix,
        pre_norm,
        normalized,
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
    )
    return {
        "post_mix": post_mix.unsqueeze(-1),
        "sinkhorn_col_19": comb_mix,
        "layer_input_bf16": pre_norm,
        "normalized_output": normalized,
    }


def run_mhc_sinkhorn_trace_native(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Replay Sinkhorn stages with the TileLang debug kernel only."""
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_pre_sinkhorn_debug_trace,
    )


def run_mhc_softmax_trace_native(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_pre_softmax_debug_trace,
    )

    params = payload["params"]
    return _mhc_pre_softmax_debug_trace(
        payload["residual_cur"], payload["gemm_out_mul"],
        payload["gemm_out_sqrsum"], payload["hc_scale"], payload["hc_base"],
        params["rms_eps"], params["hc_pre_eps"], params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"], params["sinkhorn_repeat"],
        params["n_splits"],
    )

    params = payload["params"]
    return _mhc_pre_sinkhorn_debug_trace(
        payload["residual_cur"],
        payload["gemm_out_mul"],
        payload["gemm_out_sqrsum"],
        payload["hc_scale"],
        payload["hc_base"],
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
        params["n_splits"],
    )


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(payload)
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
    cloned["params"] = dict(payload["params"])
    return cloned


def _candidate_trace(
    payload: dict[str, Any],
    candidate: str,
) -> dict[str, torch.Tensor]:
    if candidate == "torch":
        return _torch_trace_from_payload(_clone_payload(payload))
    if candidate == "tilelang":
        return run_exact_mhc_pre_from_raw_tilelang(payload)
    if candidate == "native-big-fuse":
        return run_mhc_pre_big_fuse_native(payload)
    if candidate == "native-big-fuse-logits":
        return run_mhc_big_fuse_comb_stage_native(payload, 0)
    if candidate == "native-big-fuse-softmax":
        return run_mhc_big_fuse_comb_stage_native(payload, 1)
    if candidate == "native-big-fuse-col0":
        return run_mhc_big_fuse_comb_stage_native(payload, 2)
    if candidate == "native-split-exact":
        return run_mhc_pre_split_exact_native(payload)
    if candidate == "native-split-rms-exact":
        return run_mhc_pre_rms_split_exact_native(payload)
    if candidate == "native-three-kernel-exact":
        return run_mhc_three_kernel_exact_native(payload)
    if candidate == "native-sinkhorn-trace":
        return run_mhc_sinkhorn_trace_native(payload)
    if candidate == "native-softmax-trace":
        return run_mhc_softmax_trace_native(payload)
    raise ValueError(f"unknown candidate: {candidate}")


def _first_trace_failure(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[str | None, dict[str, Any] | None]:
    if not set(candidate).issubset(reference):
        return "trace_keys", {
            "extra": sorted(set(candidate) - set(reference)),
        }
    for name in candidate:
        diff = tensor_diff(reference[name], candidate[name])
        if not diff["equal"]:
            return name, diff
    return None, None


def run_diff(
    corpus: str | Path,
    *,
    candidate: str,
    device: str,
    require_bitwise: bool,
    check_graph_replay: bool = False,
    benchmark: bool = False,
    max_files: int | None = None,
    rms_candidate: str = "captured",
) -> dict[str, Any]:
    if candidate == "post-mma":
        return run_post_diff(
            corpus,
            device=device,
            require_bitwise=require_bitwise,
            check_graph_replay=check_graph_replay,
            max_files=max_files,
        )
    files = iter_corpus_files(corpus)
    if max_files is not None:
        files = files[:max_files]
    summary: dict[str, Any] = {
        "files": len(files),
        "passed": 0,
        "failed": 0,
        "first_failure": None,
        "stage_failures": {},
        "bitwise": bool(require_bitwise),
        "raw_norm_files": 0,
        "raw_norm_passed": 0,
    }
    stage_failures: defaultdict[str, int] = defaultdict(int)
    elapsed_s = 0.0

    for path in files:
        payload = load_payload(path, device=device)
        reference = _torch_trace_from_payload(_clone_payload(payload))
        if payload["schema_version"] == 2:
            summary["raw_norm_files"] += 1
            pre_norm_diff = tensor_diff(
                reference["layer_input_bf16"], payload["pre_norm_output"]
            )
            if rms_candidate == "captured":
                normalized_diff = {"equal": True, "num_diff": 0}
            elif rms_candidate == "torch":
                normalized_diff = tensor_diff(
                    _rms_norm_from_payload(payload), payload["normalized_output"]
                )
            elif rms_candidate == "native":
                normalized_diff = tensor_diff(
                    _native_rms_norm_from_payload(payload),
                    payload["normalized_output"],
                )
            elif rms_candidate == "ir":
                normalized_diff = tensor_diff(
                    _ir_rms_norm_from_payload(payload),
                    payload["normalized_output"],
                )
            else:
                raise ValueError(f"unknown RMSNorm candidate: {rms_candidate}")
            if not pre_norm_diff["equal"] or not normalized_diff["equal"]:
                summary["failed"] += 1
                stage_failures["raw_norm_replay"] += 1
                if summary["first_failure"] is None:
                    summary["first_failure"] = {
                        "file": str(path),
                        "stage": "raw_norm_replay",
                        "diff": {
                            "pre_norm": pre_norm_diff,
                            "normalized": normalized_diff,
                        },
                    }
                continue
            summary["raw_norm_passed"] += 1
        start = time.perf_counter()
        candidate_trace = _candidate_trace(_clone_payload(payload), candidate)
        if "normalized_output" in candidate_trace:
            reference["normalized_output"] = payload["normalized_output"]
        elapsed_s += time.perf_counter() - start
        if require_bitwise:
            try:
                assert_bitwise_trace_equal(reference, candidate_trace)
                first_stage, first_diff = None, None
            except AssertionError:
                first_stage, first_diff = _first_trace_failure(
                    reference, candidate_trace
                )
        else:
            first_stage, first_diff = _first_trace_failure(reference, candidate_trace)

        if first_stage is None:
            summary["passed"] += 1
            continue

        summary["failed"] += 1
        stage_failures[first_stage] += 1
        if summary["first_failure"] is None:
            summary["first_failure"] = {
                "file": str(path),
                "stage": first_stage,
                "diff": first_diff,
            }

    summary["stage_failures"] = dict(stage_failures)
    if benchmark:
        summary["benchmark"] = {
            "elapsed_s": elapsed_s,
            "files_per_s": (len(files) / elapsed_s) if elapsed_s else None,
        }
    if check_graph_replay:
        summary["graph_replay"] = check_direct_graph_replay(
            files,
            candidate=candidate,
            device=device,
        )
    return summary


def _final_outputs(trace: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        trace["post_mix"],
        trace["sinkhorn_col_19"],
        trace["layer_input_bf16"],
    )


def check_direct_graph_replay(
    files: list[Path],
    *,
    candidate: str,
    device: str,
) -> dict[str, Any]:
    if candidate == "native-three-kernel-exact":
        return check_three_kernel_graph_replay(files, device=device)
    if candidate != "tilelang":
        return {"skipped": True, "reason": "graph replay is only defined for tilelang"}
    if len(files) < 2:
        raise ValueError("graph replay requires at least two corpus files")
    first = load_payload(files[0], device=device)
    second = load_payload(files[1], device=device)

    inputs = {
        key: first[key].detach().clone()
        for key in (
            "residual_cur",
            "gemm_out_mul",
            "gemm_out_sqrsum",
            "hc_scale",
            "hc_base",
        )
    }
    payload = {**first, **inputs, "params": dict(first["params"])}
    expected_first = _final_outputs(_torch_trace_from_payload(_clone_payload(first)))
    expected_second = _final_outputs(_torch_trace_from_payload(_clone_payload(second)))

    # Warm outside capture.
    outputs = _final_outputs(run_exact_mhc_pre_from_raw_tilelang(payload))
    if device != "cuda":
        for actual, expected in zip(outputs, expected_first):
            diff = tensor_diff(actual, expected)
            if not diff["equal"]:
                raise AssertionError(f"graph warmup output mismatch: {diff}")
        return {"skipped": True, "reason": "cuda graph replay requires cuda device"}

    torch.cuda.synchronize()
    ptrs = {
        "inputs": {key: value.data_ptr() for key, value in inputs.items()},
        "outputs": [],
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _final_outputs(run_exact_mhc_pre_from_raw_tilelang(payload))
    torch.cuda.synchronize()
    ptrs["outputs"] = [value.data_ptr() for value in outputs]

    for source, expected in ((first, expected_first), (second, expected_second)):
        for key in inputs:
            inputs[key].copy_(source[key])
        graph.replay()
        torch.cuda.synchronize()
        for actual, ref in zip(outputs, expected):
            diff = tensor_diff(actual, ref)
            if not diff["equal"]:
                raise AssertionError(f"graph replay output mismatch: {diff}")
        if any(inputs[key].data_ptr() != ptrs["inputs"][key] for key in inputs):
            raise AssertionError("graph replay input pointer changed")
        if [value.data_ptr() for value in outputs] != ptrs["outputs"]:
            raise AssertionError("graph replay output pointer changed")
    return {"passed": True, "files": [str(files[0]), str(files[1])]}


def check_three_kernel_graph_replay(
    files: list[Path], *, device: str
) -> dict[str, Any]:
    if device != "cuda":
        return {"skipped": True, "reason": "cuda graph replay requires cuda device"}
    if len(files) < 2:
        raise ValueError("graph replay requires at least two corpus files")
    first = load_payload(files[0], device=device)
    second = load_payload(files[1], device=device)
    input_keys = ("residual_cur", "fn", "hc_scale", "hc_base", "norm_weight")
    inputs = {key: first[key].detach().clone() for key in input_keys}
    payload = {**first, **inputs, "params": dict(first["params"])}
    residual = inputs["residual_cur"]
    buffers = {
        "residual_fp32": torch.empty(
            (1, residual.numel()), dtype=torch.float32, device=residual.device
        ),
        "sqrsum": torch.empty_like(first["gemm_out_sqrsum"]),
        "gemm_out": torch.empty_like(first["gemm_out_mul"]),
        "post_mix": torch.empty((1, 4), dtype=torch.float32, device=device),
        "comb_mix": torch.empty((1, 4, 4), dtype=torch.float32, device=device),
        "pre_norm": torch.empty_like(first["pre_norm_output"]),
        "normalized": torch.empty_like(first["normalized_output"]),
    }
    for _ in range(3):
        run_mhc_three_kernel_exact_native(payload, buffers)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run_mhc_three_kernel_exact_native(payload, buffers)
    torch.cuda.synchronize()
    input_ptrs = {key: value.data_ptr() for key, value in inputs.items()}
    buffer_ptrs = {key: value.data_ptr() for key, value in buffers.items()}
    for source in (first, second):
        for key in input_keys:
            inputs[key].copy_(source[key])
        graph.replay()
        torch.cuda.synchronize()
        expected = _torch_trace_from_payload(_clone_payload(source))
        checks = (
            (outputs["post_mix"], expected["post_mix"]),
            (outputs["sinkhorn_col_19"], expected["sinkhorn_col_19"]),
            (outputs["layer_input_bf16"], source["pre_norm_output"]),
            (outputs["normalized_output"], source["normalized_output"]),
        )
        for actual, reference in checks:
            diff = tensor_diff(actual, reference)
            if not diff["equal"]:
                raise AssertionError(f"three-kernel graph replay mismatch: {diff}")
        if any(inputs[key].data_ptr() != input_ptrs[key] for key in input_keys):
            raise AssertionError("three-kernel graph input pointer changed")
        if any(buffers[key].data_ptr() != buffer_ptrs[key] for key in buffers):
            raise AssertionError("three-kernel graph buffer pointer changed")
    return {
        "passed": True,
        "files": [str(files[0]), str(files[1])],
        "pointers_stable": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--candidate",
        choices=(
            "torch",
            "tilelang",
            "native-big-fuse",
            "native-big-fuse-logits",
            "native-big-fuse-softmax",
            "native-big-fuse-col0",
            "native-split-exact",
            "native-split-rms-exact",
            "native-three-kernel-exact",
            "native-sinkhorn-trace",
            "native-softmax-trace",
            "post-mma",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--check-graph-replay", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--rms-candidate", choices=("captured", "torch", "native", "ir"),
        default="captured",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_diff(
        args.corpus,
        candidate=args.candidate,
        device=args.device,
        require_bitwise=args.require_bitwise,
        check_graph_replay=args.check_graph_replay,
        benchmark=args.benchmark,
        max_files=args.max_files,
        rms_candidate=args.rms_candidate,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
