#!/usr/bin/env python3
"""Analyze vLLM all-reduce timing, rank skew, and fallback kernel fragmentation."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((p / 100) * (len(xs) - 1))))
    return xs[idx]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "sum_ms": sum(values) / 1000,
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p90_us": pct(values, 90),
        "p99_us": pct(values, 99),
        "min_us": min(values),
        "max_us": max(values),
    }


def rank_from_path(path: str) -> int:
    m = re.search(r"rank(\d+)", path)
    if not m:
        raise ValueError(f"cannot parse rank from {path}")
    return int(m.group(1))


def load_rank(path: str) -> dict:
    rank = rank_from_path(path)
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"]
    decode_ann = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in {"user_annotation", "gpu_user_annotation"}
        and str(e.get("name", "")).startswith("execute_context_0")
    ]
    if decode_ann:
        start_ts = min(e["ts"] for e in decode_ann)
        end_ts = max(e["ts"] + e.get("dur", 0) for e in decode_ann)
    else:
        xs = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
        start_ts = min(e["ts"] for e in xs)
        end_ts = max(e["ts"] + e.get("dur", 0) for e in xs)

    cpu_allreduce = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "cpu_op"
        and e.get("name") == "vllm::all_reduce"
    ]
    mccl_kernels = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "kernel"
        and "mcclKernel_AllReduce" in e.get("name", "")
    ]
    fallback_cpu_names = {
        "aten::sum",
        "aten::add",
        "aten::div",
        "aten::copy_",
        "aten::mul",
    }
    fallback_cpu = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "cpu_op"
        and e.get("name") in fallback_cpu_names
    ]
    fallback_kernel_substrings = {
        "aten_add_kernel": "add",
        "CUDAFunctorOnSelf_add": "add_self",
        "DivFunctor": "div",
        "sum_functor": "sum",
        "reduce_kernel_maca": "reduce",
        "copy_cast": "copy_cast",
        "direct_copy": "copy",
        "MulFunctor": "mul",
        "rsqrt_kernel": "rsqrt",
        "pow_tensor_scalar": "pow",
    }
    fallback_kernels = []
    fallback_kernel_counts = Counter()
    fallback_kernel_dur = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        for needle, label in fallback_kernel_substrings.items():
            if needle in name:
                fallback_kernels.append(e)
                fallback_kernel_counts[label] += 1
                fallback_kernel_dur[label] += e.get("dur", 0)
                break

    runtime = [
        e for e in events if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
    ]
    runtime_counts = Counter(e.get("name", "") for e in runtime)
    runtime_dur = defaultdict(float)
    for e in runtime:
        runtime_dur[e.get("name", "")] += e.get("dur", 0)

    all_kernel = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    all_kernel_sorted = sorted(all_kernel, key=lambda e: e["ts"])
    gaps = []
    prev_end = None
    for e in all_kernel_sorted:
        if prev_end is not None and e["ts"] > prev_end:
            gaps.append(e["ts"] - prev_end)
        prev_end = max(prev_end or e["ts"], e["ts"] + e.get("dur", 0))

    return {
        "rank": rank,
        "path": path,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "decode_window_ms": (end_ts - start_ts) / 1000,
        "cpu_allreduce": sorted(cpu_allreduce, key=lambda e: e["ts"]),
        "mccl_kernels": sorted(mccl_kernels, key=lambda e: e["ts"]),
        "fallback_cpu": fallback_cpu,
        "fallback_kernels": fallback_kernels,
        "fallback_kernel_counts": fallback_kernel_counts,
        "fallback_kernel_dur": dict(fallback_kernel_dur),
        "runtime_counts": runtime_counts,
        "runtime_dur": dict(runtime_dur),
        "kernel_gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_glob")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    paths = sorted(str(p) for p in Path("/").glob(args.trace_glob.lstrip("/")))
    if not paths:
        raise SystemExit(f"no traces match {args.trace_glob}")
    ranks = [load_rank(p) for p in paths]
    ranks.sort(key=lambda r: r["rank"])

    per_rank = {}
    for r in ranks:
        allr_durs = [e.get("dur", 0) for e in r["cpu_allreduce"]]
        mccl_durs = [e.get("dur", 0) for e in r["mccl_kernels"]]
        fallback_cpu_by_name = Counter(e.get("name", "") for e in r["fallback_cpu"])
        fallback_cpu_dur = defaultdict(float)
        for e in r["fallback_cpu"]:
            fallback_cpu_dur[e.get("name", "")] += e.get("dur", 0)
        per_rank[r["rank"]] = {
            "decode_window_ms": r["decode_window_ms"],
            "allreduce_cpu": stats(allr_durs),
            "mccl_kernel": stats(mccl_durs),
            "fallback_cpu_counts": dict(fallback_cpu_by_name),
            "fallback_cpu_total_ms": {k: v / 1000 for k, v in fallback_cpu_dur.items()},
            "fallback_kernel_counts": dict(r["fallback_kernel_counts"]),
            "fallback_kernel_total_ms": {
                k: v / 1000 for k, v in r["fallback_kernel_dur"].items()
            },
            "cuda_runtime_top": [
                {
                    "name": name,
                    "count": r["runtime_counts"][name],
                    "total_ms": r["runtime_dur"][name] / 1000,
                }
                for name, _ in r["runtime_counts"].most_common(12)
            ],
            "kernel_gap_us": stats(r["kernel_gaps"]),
        }

    min_mccl_count = min(len(r["mccl_kernels"]) for r in ranks)
    aligned = []
    for idx in range(min_mccl_count):
        starts = [r["mccl_kernels"][idx]["ts"] - r["start_ts"] for r in ranks]
        durs = [r["mccl_kernels"][idx].get("dur", 0) for r in ranks]
        aligned.append(
            {
                "idx": idx,
                "start_skew_us": max(starts) - min(starts),
                "duration_skew_us": max(durs) - min(durs),
                "max_duration_us": max(durs),
                "min_duration_us": min(durs),
                "mean_duration_us": statistics.fmean(durs),
            }
        )
    skew_values = [x["start_skew_us"] for x in aligned]
    dur_values = [x["mean_duration_us"] for x in aligned]
    worst_skew = sorted(aligned, key=lambda x: x["start_skew_us"], reverse=True)[:20]
    worst_duration = sorted(aligned, key=lambda x: x["max_duration_us"], reverse=True)[
        :20
    ]

    result = {
        "trace_paths": paths,
        "per_rank": per_rank,
        "aligned_mccl_count": min_mccl_count,
        "rank_start_skew_summary_us": stats(skew_values),
        "aligned_mccl_mean_duration_summary_us": stats(dur_values),
        "worst_start_skew": worst_skew,
        "worst_duration": worst_duration,
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
