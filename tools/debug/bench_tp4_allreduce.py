#!/usr/bin/env python3
"""Small TP all-reduce latency probe for MetaX/MCCL."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def sync() -> None:
    torch.accelerator.synchronize()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize_ms(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 90),
        "p99_ms": percentile(samples, 99),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def bench(fn, warmup: int, trials: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return summarize_ms(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    import vllm_metax.patch  # noqa: F401
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)

    cpu_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    pynccl = PyNcclCommunicator(group=cpu_group, device=device)

    dtype = torch.bfloat16
    results: dict[str, dict[str, dict[str, float]]] = {}
    for seq_len in args.seq_lens:
        tensor = torch.ones((seq_len, args.hidden_size), dtype=dtype, device=device)
        out = torch.empty_like(tensor)
        tensor_bytes = tensor.numel() * tensor.element_size()

        def run_pynccl() -> None:
            pynccl.all_reduce(tensor, out)

        def run_torch_nccl() -> None:
            tmp = tensor.clone()
            dist.all_reduce(tmp, group=nccl_group)

        pynccl_stats = bench(run_pynccl, args.warmup, args.trials)
        torch_stats = bench(run_torch_nccl, args.warmup, args.trials)
        result = {"pynccl_mccl": pynccl_stats, "torch_nccl": torch_stats}
        results[f"{seq_len}x{args.hidden_size}"] = result
        if rank == 0:
            mb = tensor_bytes / 1024 / 1024
            print(
                f"shape=({seq_len},{args.hidden_size}) bytes={tensor_bytes} ({mb:.3f} MiB) "
                f"pynccl_mean={pynccl_stats['mean_ms']:.3f}ms "
                f"pynccl_p99={pynccl_stats['p99_ms']:.3f}ms "
                f"torch_nccl_mean={torch_stats['mean_ms']:.3f}ms "
                f"torch_nccl_p99={torch_stats['p99_ms']:.3f}ms",
                flush=True,
            )

    if rank == 0 and args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "world_size": world_size,
                    "hidden_size": args.hidden_size,
                    "dtype": str(dtype),
                    "warmup": args.warmup,
                    "trials": args.trials,
                    "results": results,
                },
                f,
                indent=2,
            )

    pynccl.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
