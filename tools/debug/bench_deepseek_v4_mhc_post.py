#!/usr/bin/env python3
"""Benchmark the DeepSeek V4 MHC post implementations on one real payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.debug.diff_deepseek_v4_mhc_raw import load_fused_post_payload
from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import mhc_post_fwd
from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
    _mhc_post_exact_tl,
)


def _torch_post(x, residual, post_mix, comb_mix, out):
    term2 = torch.bmm(comb_mix.transpose(1, 2), residual.float())
    out.copy_((x.float().unsqueeze(-2) * post_mix.unsqueeze(-1) + term2).bfloat16())


def _scalar_tilelang_post(x, residual, post_mix, comb_mix, out):
    mhc_post_fwd(
        x,
        residual,
        post_mix.unsqueeze(-1),
        comb_mix,
        out,
    )


def _mma_post(x, residual, post_mix, comb_mix, out):
    _mhc_post_exact_tl(x, residual, post_mix, comb_mix, out=out)


def _elapsed_ms(fn, args, out, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(*args, out)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args, out)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _graph_elapsed_ms(fn, args, out, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(*args, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args, out)
    torch.cuda.synchronize()
    output_ptr = out.data_ptr()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0 / iterations
    if out.data_ptr() != output_ptr:
        raise RuntimeError("post benchmark graph changed output pointer")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("the post benchmark requires --device cuda")
    payload = load_fused_post_payload(args.payload, device=args.device)
    tensors = (
        payload["x_flat"],
        payload["residual_flat"],
        payload["post_layer_mix_flat"],
        payload["comb_res_mix_flat"],
    )
    reference = payload["residual_cur_bf16"]
    candidates = {
        "torch": _torch_post,
        "tilelang-scalar": _scalar_tilelang_post,
        "post-mma": _mma_post,
    }
    result = {
        "payload": str(args.payload),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "shape": [1, 4, 4096],
        "candidates": {},
    }
    for name, fn in candidates.items():
        eager_out = torch.empty_like(payload["residual_flat"])
        eager_ms = _elapsed_ms(fn, tensors, eager_out, args.warmup, args.iterations)
        eager_equal = torch.equal(eager_out.cpu(), reference.cpu())
        graph_out = torch.empty_like(payload["residual_flat"])
        graph_ms = _graph_elapsed_ms(
            fn, tensors, graph_out, args.warmup, args.iterations
        )
        graph_equal = torch.equal(graph_out.cpu(), reference.cpu())
        result["candidates"][name] = {
            "eager_ms": eager_ms,
            "graph_ms": graph_ms,
            "eager_bitwise_reference": eager_equal,
            "graph_bitwise_reference": graph_equal,
            "output_ptr_stable": True,
        }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
