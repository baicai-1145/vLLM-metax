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
    _require(set(payload) == _REQUIRED_KEYS, path, f"keys={sorted(payload)}")
    _require(payload["schema_version"] == 1, path, "schema_version must be 1")
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
    ):
        out[key] = out[key].to(device)
    return out


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
    raise ValueError(f"unknown candidate: {candidate}")


def _first_trace_failure(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[str | None, dict[str, Any] | None]:
    if set(reference) != set(candidate):
        return "trace_keys", {
            "missing": sorted(set(reference) - set(candidate)),
            "extra": sorted(set(candidate) - set(reference)),
        }
    for name in reference:
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
) -> dict[str, Any]:
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
    }
    stage_failures: defaultdict[str, int] = defaultdict(int)
    elapsed_s = 0.0

    for path in files:
        payload = load_payload(path, device=device)
        reference = _torch_trace_from_payload(_clone_payload(payload))
        start = time.perf_counter()
        candidate_trace = _candidate_trace(_clone_payload(payload), candidate)
        elapsed_s += time.perf_counter() - start
        if require_bitwise:
            try:
                assert_bitwise_trace_equal(reference, candidate_trace)
                first_stage, first_diff = None, None
            except AssertionError:
                first_stage, first_diff = _first_trace_failure(reference, candidate_trace)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--candidate", choices=("torch", "tilelang"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--check-graph-replay", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-files", type=int)
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
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
