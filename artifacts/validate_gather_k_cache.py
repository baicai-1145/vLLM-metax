import json
import math
import os
import statistics
import time

import torch

from vllm_metax.models.deepseek_v4.ops.cache_utils import gather_k_cache


def reference(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset):
    expected = out.clone()
    for b in range(seq_lens.numel()):
        seq_len = int(seq_lens[b].item())
        gl = seq_len if gather_lens is None else int(gather_lens[b].item())
        if gl == 0:
            continue
        pos = torch.arange(seq_len - gl, seq_len, device=k_cache.device)
        blocks = block_table[b, pos // block_size].long()
        rows = pos % block_size
        expected[b, offset : offset + gl] = k_cache[blocks, rows]
    return expected


def error_stats(out, expected):
    ob = out.view(torch.int16)
    eb = expected.view(torch.int16)
    bit_mismatch = int((ob != eb).sum().item())
    finite = torch.isfinite(out) & torch.isfinite(expected)
    if finite.any():
        diff = (out.float() - expected.float()).abs()[finite]
        denom = expected.float().abs()[finite].clamp_min(1e-30)
        max_abs = float(diff.max().item())
        max_rel = float((diff / denom).max().item())
    else:
        max_abs = max_rel = 0.0
    nonfinite_mask = (~torch.isfinite(out)) | (~torch.isfinite(expected))
    nonfinite_mismatch = int((nonfinite_mask & (ob != eb)).sum().item())
    return {
        "bit_mismatch_elements": bit_mismatch,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonfinite_mismatch_elements": nonfinite_mismatch,
        "out_nonfinite": int((~torch.isfinite(out)).sum().item()),
        "expected_nonfinite": int((~torch.isfinite(expected)).sum().item()),
    }


def make_cache(layout, nb, block, head, device):
    torch.manual_seed(20260715)
    if layout == "tight":
        return torch.randn((nb, block, head), device=device, dtype=torch.bfloat16)
    if layout == "padded":
        storage = torch.randn((nb, block, head + 8), device=device, dtype=torch.bfloat16)
        return storage[:, :, :head]
    if layout == "noncontiguous_stride2":
        storage = torch.randn((nb, block, head * 2), device=device, dtype=torch.bfloat16)
        return storage[:, :, ::2]
    raise ValueError(layout)


def run_case(layout, gather_mode, device):
    nb, block, head, batch, max_blocks = 8, 16, 512, 3, 4
    seq_lens = torch.tensor([31, 17, 8], device=device, dtype=torch.int32)
    gather_lens = None if gather_mode == "full" else torch.tensor([13, 5, 8], device=device, dtype=torch.int32)
    block_table = torch.tensor([[2, 0, 3, 1], [7, 4, 6, 5], [1, 3, 0, 2]], device=device, dtype=torch.int32)
    offset = 3
    k_cache = make_cache(layout, nb, block, head, device)
    # Exercise NaN/Inf propagation while retaining mostly random production-like data.
    k_cache[2, 0, 0] = float("nan")
    k_cache[2, 0, 1] = float("inf")
    out = torch.full((batch, 40, head), -7.0, device=device, dtype=torch.bfloat16)
    expected = reference(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
    # Warm up/compile outside any graph capture.
    for _ in range(2):
        out.fill_(-7.0)
        gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
    torch.cuda.synchronize(device)
    out.fill_(-7.0)
    gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
    torch.cuda.synchronize(device)
    stats = error_stats(out, expected)

    # Isolated launch timing after correctness; output is preallocated and all
    # inputs/pointers remain unchanged across samples.
    latencies_us = []
    try:
        for _ in range(10):
            gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(60):
            start.record()
            gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
            end.record()
            end.synchronize()
            latencies_us.append(float(start.elapsed_time(end)) * 1000.0)
        latency = {
            "method": "torch.cuda.Event",
            "samples": len(latencies_us),
            "median_us": statistics.median(latencies_us),
            "p90_us": sorted(latencies_us)[math.ceil(0.9 * len(latencies_us)) - 1],
        }
    except Exception as exc:
        latency = {"method": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    graph = {"attempted": True}
    ptrs_before = {"out": out.data_ptr(), "k_cache": k_cache.data_ptr(), "seq_lens": seq_lens.data_ptr(), "block_table": block_table.data_ptr()}
    try:
        g = torch.cuda.CUDAGraph()
        out.fill_(-7.0)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(g):
            gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block, offset)
        torch.cuda.synchronize(device)
        replay_stats = []
        for _ in range(3):
            out.fill_(-7.0)
            g.replay()
            torch.cuda.synchronize(device)
            replay_stats.append(error_stats(out, expected))
        ptrs_after = {"out": out.data_ptr(), "k_cache": k_cache.data_ptr(), "seq_lens": seq_lens.data_ptr(), "block_table": block_table.data_ptr()}
        graph.update({"success": True, "replays": replay_stats, "pointer_stable": ptrs_before == ptrs_after, "pointers_before": ptrs_before, "pointers_after": ptrs_after})
    except Exception as exc:
        graph.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "layout": layout,
        "gather": gather_mode,
        "shape": {"k_cache": list(k_cache.shape), "k_cache_stride": list(k_cache.stride()), "out": list(out.shape), "batch": batch, "head_size": head, "block_size": block},
        "dispatch": "native Triton _gather_k_cache_kernel (no fallback branch)",
        "stats": stats,
        "latency": latency,
        "graph": graph,
    }


def main():
    device = torch.device("cuda:0")
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(device), "cases": []}
    for layout in ("tight", "padded", "noncontiguous_stride2"):
        for mode in ("full", "partial"):
            result["cases"].append(run_case(layout, mode, device))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
