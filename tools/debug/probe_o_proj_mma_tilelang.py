#!/usr/bin/env python3
"""Run the isolated fixed-shape O-projection TileLang MMA probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm_metax.models.deepseek_v4.ops.fused_inv_rope_quant import inv_rope
from vllm_metax.models.deepseek_v4.ops.o_proj_mma_probe import (
    GROUPS,
    HEADS_PER_GROUP,
    RANK,
    run_o_proj_mma_probe,
)


def _reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.Tensor,
) -> torch.Tensor:
    o_bf16 = inv_rope(
        o,
        positions,
        cos_sin_cache,
        n_groups=GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=448,
        rope_dim=64,
    )
    # Match the grouped BF16 contraction used by the production path.
    return torch.einsum("bhr,hdr->bhd", o_bf16, wo_a)


@torch.no_grad()
def run(seed_count: int, device: str) -> dict[str, object]:
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the MMA probe requires a CUDA device")

    # Keep the cache extent fixed so all seeds reuse one compiled kernel.
    cache_extent = 8
    bitwise_elements = 0
    total_elements = 0
    bitwise_seeds = 0
    max_abs = 0.0
    allclose = True
    compile_start = time.perf_counter()

    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for seed in range(seed_count):
        torch.manual_seed(seed)
        o = torch.randn((1, 16, 512), device=target, dtype=torch.bfloat16)
        positions = torch.tensor([seed % cache_extent], device=target, dtype=torch.int64)
        cache = torch.randn((cache_extent, 64), device=target, dtype=torch.float32)
        wo_a = torch.randn((GROUPS, RANK, 4096), device=target, dtype=torch.bfloat16)
        z = torch.empty((1, GROUPS, RANK), device=target, dtype=torch.bfloat16)
        samples.append((o, positions, cache, wo_a, z))

    # The first invocation includes TileLang lowering and device compilation.
    o, positions, cache, wo_a, z = samples[0]
    run_o_proj_mma_probe(o, positions, cache, wo_a, z)
    torch.cuda.synchronize(target)
    compile_seconds = time.perf_counter() - compile_start

    for o, positions, cache, wo_a, z in samples:
        run_o_proj_mma_probe(o, positions, cache, wo_a, z)
        expected = _reference(o, positions, cache, wo_a)
        torch.cuda.synchronize(target)
        equal = torch.equal(z.view(torch.int16), expected.view(torch.int16))
        delta = (z.float() - expected.float()).abs()
        bitwise_seeds += int(equal)
        bitwise_elements += int(
            (z.view(torch.int16) == expected.view(torch.int16)).sum().item()
        )
        total_elements += z.numel()
        max_abs = max(max_abs, float(delta.max().item()))
        allclose = allclose and bool(torch.allclose(z, expected, atol=0.125, rtol=0.01))

    # Measure the compiled candidate independently of reference work.
    bench_o, bench_pos, bench_cache, bench_wo, bench_z = samples[0]
    for _ in range(5):
        run_o_proj_mma_probe(bench_o, bench_pos, bench_cache, bench_wo, bench_z)
    torch.cuda.synchronize(target)
    timings_ms: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        run_o_proj_mma_probe(bench_o, bench_pos, bench_cache, bench_wo, bench_z)
        torch.cuda.synchronize(target)
        timings_ms.append((time.perf_counter() - start) * 1e3)

    return {
        "shape": {"o": [1, 16, 512], "wo_a": [2, 1024, 4096], "z": [1, 2, 1024]},
        "seed_count": seed_count,
        "compile_seconds": compile_seconds,
        "bitwise_seed_count": bitwise_seeds,
        "bitwise_element_count": bitwise_elements,
        "total_element_count": total_elements,
        "bitwise_element_fraction": bitwise_elements / total_elements,
        "max_abs": max_abs,
        "allclose": allclose,
        "latency_ms_median": statistics.median(timings_ms),
        "latency_ms_p90": statistics.quantiles(timings_ms, n=10)[8],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.seeds, args.device)
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        # This is an exactness probe: allclose is useful context, but a
        # non-bitwise seed must keep the gate red.
        result["passed"] = bool(
            result["bitwise_seed_count"] == result["seed_count"]
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
