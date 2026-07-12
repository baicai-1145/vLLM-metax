#!/usr/bin/env python3
"""Probe why vLLM custom all-reduce is disabled on MetaX."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    import vllm_metax.patch  # noqa: F401
    from vllm import _custom_ops as ops
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.device_communicators import custom_all_reduce as car_mod
    from vllm.distributed.device_communicators.all_reduce_utils import (
        gpu_p2p_access_check,
    )
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.platforms import current_platform

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    cpu_group = dist.new_group(backend="gloo")

    if rank == 0:
        print("platform", type(current_platform).__name__)
        print("platform.use_custom_allreduce", current_platform.use_custom_allreduce())
        print("custom_all_reduce.custom_ar", car_mod.custom_ar)
        try:
            print("ops.meta_size", ops.meta_size())
        except Exception as exc:
            print("ops.meta_size_error", repr(exc))

    try:
        pc = ParallelConfig(tensor_parallel_size=world_size)
        if rank == 0:
            print(
                "ParallelConfig.disable_custom_all_reduce", pc.disable_custom_all_reduce
            )
    except Exception as exc:
        if rank == 0:
            print("ParallelConfig_error", repr(exc))

    physical_device_id = current_platform.visible_device_id_to_physical_device_id(
        local_rank
    )
    tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
    gather_list = [
        torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
    ]
    dist.all_gather(gather_list, tensor, group=cpu_group)
    physical_ids = [t.item() for t in gather_list]
    fully_connected = current_platform.is_fully_connected(physical_ids)
    p2p_results = []
    for i in range(world_size):
        if i != rank:
            try:
                p2p_results.append((rank, i, bool(gpu_p2p_access_check(rank, i))))
            except Exception as exc:
                p2p_results.append((rank, i, repr(exc)))
    gathered_p2p = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_p2p, p2p_results, group=cpu_group)

    max_size = max(args.seq_lens) * args.hidden_size * torch.bfloat16.itemsize + 1
    init_error = None
    custom = None
    try:
        custom = CustomAllreduce(group=cpu_group, device=device, max_size=max_size)
    except Exception as exc:
        init_error = repr(exc)

    shape_results = {}
    if custom is not None:
        for seq_len in args.seq_lens:
            sample = torch.ones(
                (seq_len, args.hidden_size), dtype=torch.bfloat16, device=device
            )
            shape_results[f"{seq_len}x{args.hidden_size}"] = bool(
                custom.should_custom_ar(sample)
            )

    summary = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "physical_ids": physical_ids,
        "fully_connected": bool(fully_connected),
        "p2p_results_all_ranks": gathered_p2p,
        "platform_use_custom_allreduce": bool(current_platform.use_custom_allreduce()),
        "module_custom_ar": bool(car_mod.custom_ar),
        "custom_init_error": init_error,
        "custom_disabled": None if custom is None else bool(custom.disabled),
        "should_custom_ar": shape_results,
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, summary, group=cpu_group)

    if rank == 0:
        print(json.dumps(gathered, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(gathered, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
