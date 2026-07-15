#!/usr/bin/env python3
"""Analyze four-rank profiler traces for graph-relative collective arrivals."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import heapq
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_RANK_RE = re.compile(r"(?:rank|local_rank|tp)[_-]?(\d+)", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"(?:ordinal|collective)[_#=-]?(\d+)", re.IGNORECASE)


def map_dsv4_ordinal(ordinal: int) -> tuple[str, int | None]:
    if ordinal == 0:
        return "embedding", None
    if 1 <= ordinal <= 85 and ordinal % 2 == 1:
        return "attention", (ordinal - 1) // 2
    if 2 <= ordinal <= 86 and ordinal % 2 == 0:
        return "ffn", ordinal // 2 - 1
    return "unknown", None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p90": _percentile(values, 90),
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" or path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return {"traceEvents": data}
    return data


def _rank(path: str | Path, data: dict[str, Any]) -> int:
    match = _RANK_RE.search(str(path))
    if match:
        return int(match.group(1))
    top_level = data.get("distributedInfo")
    if isinstance(top_level, dict) and top_level.get("rank") is not None:
        return int(top_level["rank"])
    for event in data.get("traceEvents", []):
        args = event.get("args") or {}
        for key in ("rank", "local_rank", "distributedInfo.rank"):
            if key in args:
                return int(args[key])
        distributed = args.get("distributedInfo")
        if isinstance(distributed, dict) and distributed.get("rank") is not None:
            return int(distributed["rank"])
    raise ValueError(f"cannot determine rank from {path}")


def _args(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("args")
    return value if isinstance(value, dict) else {}


def _first(args: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in args and args[key] is not None:
            return args[key]
    return None


def _context(event: dict[str, Any]) -> str:
    args = _args(event)
    value = _first(args, "execute_context", "execute_context_id", "context")
    if value is not None:
        return str(value)
    match = re.search(r"execute_context[_ (=-]?(\d+)", str(event.get("name", "")), re.I)
    return match.group(1) if match else "default"


def _graph_id(event: dict[str, Any]) -> str:
    args = _args(event)
    value = _first(args, "External id", "external_id", "graph_parent", "graph_id", "capture_id")
    if value is not None:
        return str(value)
    return str(event.get("name", "graph"))


def _stream(event: dict[str, Any]) -> str:
    args = _args(event)
    value = _first(args, "stream", "stream_id", "device_stream")
    if value is None:
        value = event.get("tid", event.get("pid", "default"))
    return str(value)


def _is_graph(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).lower()
    if name.startswith("plan35."):
        return False
    args = _args(event)
    cat = str(event.get("cat", "")).lower()
    if cat == "cuda_runtime":
        return False
    return bool(
        _first(args, "graph_parent", "graph_id", "capture_id", "External id") is not None
        and ("execute_context" in name or "graph" in name or "user_annotation" in cat)
    ) or "graph_parent" in name or name.startswith("execute_context")


def _is_collective(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).lower()
    args = _args(event)
    if _is_graph(event):
        return False
    return bool(
        any(token in name for token in ("all_reduce", "allreduce"))
        or _first(args, "collective", "collective_name", "op") is not None
        and "allreduce" in str(_first(args, "collective", "collective_name", "op")).lower()
    )


def _ordinal(event: dict[str, Any], fallback: int) -> int:
    args = _args(event)
    value = _first(args, "ordinal", "collective_ordinal", "index")
    if value is not None:
        return int(value)
    match = _ORDINAL_RE.search(str(event.get("name", "")))
    return int(match.group(1)) if match else fallback


def _end(event: dict[str, Any]) -> float:
    return float(event.get("ts", 0)) + float(event.get("dur", 0))


def _device_event(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    return cat in {"kernel", "gpu", "gpu_user_annotation", "device"} or bool(_args(event).get("device"))


def _operator_category(name: str) -> str:
    lowered = name.lower()
    if "mhc" in lowered:
        return "MHC"
    if "moe" in lowered:
        return "MoE"
    if any(token in lowered for token in ("attention", "attn", "flash_attention", "scaled_dot_product_attention")):
        return "attention"
    if "memcpy" in lowered or "copy" in lowered:
        return "memcpy"
    if "graph" in lowered or "host" in lowered:
        return "graph/host gap"
    return "other"


def _event_summary(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if event is None:
        return None
    args = _args(event)
    return {
        "name": event.get("name", ""),
        "ts": float(event.get("ts", 0)),
        "dur": float(event.get("dur", 0)),
        "cat": event.get("cat", ""),
        "correlation": _first(args, "correlation", "correlation_id"),
        "external_id": _first(args, "External id", "external_id"),
        "record_function_id": _first(args, "Record function id", "record_function_id"),
    }


def _host_chain(item: dict[str, Any], device: dict[str, Any] | None) -> dict[str, Any] | None:
    if device is None:
        return None
    device_args = _args(device)
    correlation = _first(device_args, "correlation", "correlation_id")
    if correlation is None:
        return None
    runtimes = [
        runtime
        for runtime in item.get("runtime_by_correlation", {}).get(str(correlation), [])
        if _end(runtime) <= float(device["ts"])
    ]
    if not runtimes:
        return None
    latest_runtime_end = max(_end(runtime) for runtime in runtimes)
    nearest_runtimes = [runtime for runtime in runtimes if _end(runtime) == latest_runtime_end]
    if len(nearest_runtimes) != 1:
        return {
            "ambiguous": True,
            "device": _event_summary(device),
            "runtime": None,
            "host": None,
        }
    runtime = nearest_runtimes[0]
    device_start = float(device["ts"])
    device_end = _end(device)
    runtime_start = float(runtime["ts"])
    runtime_end = _end(runtime)
    runtime_args = _args(runtime)
    runtime_external = _first(runtime_args, "External id", "external_id")
    runtime_record = _first(runtime_args, "Record function id", "record_function_id")

    def linked_and_enclosing(host: dict[str, Any]) -> bool:
        host_start = float(host["ts"])
        host_end = _end(host)
        if host_start > min(runtime_start, device_start) or host_end < max(runtime_end, device_end):
            return False
        host_args = _args(host)
        host_external = _first(host_args, "External id", "external_id")
        host_record = _first(host_args, "Record function id", "record_function_id")
        id_link = (
            runtime_external is not None
            and host_external is not None
            and str(runtime_external) == str(host_external)
        ) or (
            runtime_record is not None
            and host_record is not None
            and str(runtime_record) == str(host_record)
        )
        thread_link = (
            runtime.get("pid") is not None
            and runtime.get("tid") is not None
            and host.get("pid") == runtime.get("pid")
            and host.get("tid") == runtime.get("tid")
        )
        return id_link or thread_link

    candidates = [
        host
        for host in item.get("host_active_by_device", {}).get(id(device), [])
        if linked_and_enclosing(host)
    ]
    if runtime_external is not None:
        candidates.extend(
            host
            for host in item.get("host_by_external", {}).get(str(runtime_external), [])
            if linked_and_enclosing(host)
        )
    unique = list({id(event): event for event in candidates}.values())
    families = {_operator_category(str(event.get("name", ""))) for event in unique}
    specific_families = families - {"other", "graph/host gap"}
    if len(specific_families) > 1:
        return {"ambiguous": True, "device": _event_summary(device), "runtime": _event_summary(runtime), "host": None}
    specific = [event for event in unique if _operator_category(str(event.get("name", ""))) in specific_families]
    ranked = specific or unique
    if ranked:
        smallest_span = min(_end(event) - float(event.get("ts", 0)) for event in ranked)
        best = [event for event in ranked if _end(event) - float(event.get("ts", 0)) == smallest_span]
        if len(best) == 1:
            unique = best
        else:
            unique = []
    if len(unique) != 1:
        return {
            "ambiguous": True,
            "device": _event_summary(device),
            "runtime": _event_summary(runtime),
            "host": None,
        }
    return {
        "ambiguous": False,
        "device": _event_summary(device),
        "runtime": _event_summary(runtime),
        "host": _event_summary(unique[0]),
    }


def _correlation_host_chain(item: dict[str, Any], device: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve a device kernel through mcGraphLaunch to one host execute context."""
    if device is None:
        return None
    correlation = _first(_args(device), "correlation", "correlation_id")
    if correlation is None:
        return None
    runtimes = [
        runtime
        for runtime in item.get("runtime_by_correlation", {}).get(str(correlation), [])
        if "mcgraphlaunch" in str(runtime.get("name", "")).lower()
        and _end(runtime) <= float(device["ts"])
    ]
    if not runtimes:
        return None
    latest_end = max(_end(runtime) for runtime in runtimes)
    nearest = [runtime for runtime in runtimes if _end(runtime) == latest_end]
    if len(nearest) != 1:
        return {"ambiguous": True, "device": _event_summary(device), "runtime": None, "host": None}
    runtime = nearest[0]
    runtime_args = _args(runtime)
    runtime_external = _first(runtime_args, "External id", "external_id")
    candidates = []
    for host in item.get("host_execute_contexts", []):
        host_start = float(host["ts"])
        host_end = _end(host)
        if host_start > float(runtime["ts"]) or host_end < _end(runtime):
            continue
        host_external = _first(_args(host), "External id", "external_id", "graph_parent", "graph_id")
        id_link = (
            runtime_external is not None
            and host_external is not None
            and str(runtime_external) == str(host_external)
        )
        thread_link = (
            runtime.get("pid") is not None
            and runtime.get("tid") is not None
            and host.get("pid") == runtime.get("pid")
            and host.get("tid") == runtime.get("tid")
        )
        if id_link or thread_link:
            candidates.append(host)
    unique = list({id(host): host for host in candidates}.values())
    if len(unique) != 1:
        return {
            "ambiguous": True,
            "device": _event_summary(device),
            "runtime": _event_summary(runtime),
            "host": None,
        }
    host = unique[0]
    host_external = _first(_args(host), "External id", "external_id", "graph_parent", "graph_id")
    return {
        "ambiguous": False,
        "device": _event_summary(device),
        "runtime": _event_summary(runtime),
        "host": _event_summary(host),
        "execute_context": _context(host),
        "graph_id": host_external if host_external is not None else runtime_external,
    }


def _stage_attribution(
    item: dict[str, Any], device: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None, str | None, bool]:
    if device is None:
        return None, None, None, False
    device_start = float(device.get("ts", 0))
    device_end = _end(device)
    covering = []
    partial = []
    for stage in item.get("stage_events", []):
        stage_start = float(stage.get("ts", 0))
        stage_end = _end(stage)
        if stage_start <= device_start and stage_end >= device_end:
            covering.append(stage)
        elif stage_start < device_end and stage_end > device_start:
            partial.append(stage)
    candidates = covering + partial
    if len(covering) != 1 or partial:
        if not candidates:
            return None, None, None, False
        return {
            "ambiguous": True,
            "candidates": [_event_summary(stage) for stage in candidates],
        }, None, None, True
    stage = covering[0]
    name = str(stage.get("name", ""))
    if name == "plan35.attention_o_proj":
        family, prefix = "attention_o_proj", None
    elif name.startswith("plan35.mlp_down_proj:"):
        family, prefix = "mlp_down_proj", name.split(":", 1)[1]
    else:
        family, prefix = None, None
    return _event_summary(stage), family, prefix, False


def _choose(events: list[dict[str, Any]]) -> dict[str, Any]:
    def score(event: dict[str, Any]) -> tuple[int, int, float]:
        args = _args(event)
        role = str(args.get("role", "")).lower()
        cat = str(event.get("cat", "")).lower()
        name = str(event.get("name", "")).lower()
        return (
            10 if role == "collective" else 0,
            (5 if cat == "kernel" else 0) + (4 if "mccl" in name else 0),
            -float(event.get("dur", 0)),
        )

    return max(events, key=score)


def _events_for(path: str | Path) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    data = _read_json(path)
    rank = _rank(path, data)
    events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X" and "ts" in e]
    for event in events:
        event.setdefault("dur", 0)
    return rank, events, data


def _calibration(data: dict[str, Any], events: list[dict[str, Any]]) -> tuple[Any, float | None]:
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    marker = _first(metadata, "calibration_marker", "clock_calibration_marker", "shared_calibration_marker")
    offset = _first(metadata, "clock_offset_us", "clock_offset", "timestamp_offset_us")
    for event in events:
        name = str(event.get("name", "")).lower()
        args = _args(event)
        if "calibration" in name or "clock_sync" in name or _first(args, "calibration_marker", "marker", "marker_id") is not None:
            marker = marker if marker is not None else _first(args, "marker", "marker_id", "id")
            offset = offset if offset is not None else _first(args, "offset_us", "clock_offset_us", "offset")
    try:
        return marker, float(offset) if offset is not None else None
    except (TypeError, ValueError):
        return marker, None


def analyze_traces(
    paths: Iterable[str | Path],
    *,
    expected_steps: int = 5,
    expected_collectives_per_step: int = 87,
) -> dict[str, Any]:
    """Return auditable collective records and summary diagnostics."""
    traces = [_events_for(path) for path in paths]
    traces.sort(key=lambda item: item[0])
    ranks = [item[0] for item in traces]
    per_rank: dict[int, dict[str, Any]] = {}
    calibration_by_rank: dict[int, tuple[Any, float | None]] = {}
    all_keys: set[tuple[str, int, str, int]] = set()
    for rank, events, _data in traces:
        graphs = [e for e in events if _is_graph(e)]
        graph_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for event in graphs:
            key = (_context(event), _graph_id(event))
            prior = graph_by_key.get(key)
            event_gpu = "gpu" in str(event.get("cat", "")).lower()
            prior_gpu = prior is not None and "gpu" in str(prior.get("cat", "")).lower()
            if prior is None or (event_gpu and not prior_gpu) or (
                event_gpu == prior_gpu and float(event.get("dur", 0)) > float(prior.get("dur", 0))
            ):
                graph_by_key[key] = event
        graph_by_step: dict[tuple[str, int], dict[str, Any]] = {}
        graph_key_to_step: dict[tuple[str, str], int] = {}
        context_graphs: defaultdict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for (graph_context, graph_id), parent in graph_by_key.items():
            context_graphs[graph_context].append((graph_id, parent))
        for graph_context, graph_items in context_graphs.items():
            for step_index, (graph_id, parent) in enumerate(sorted(graph_items, key=lambda item: float(item[1]["ts"]))):
                graph_key_to_step[(graph_context, graph_id)] = step_index
                graph_by_step[(graph_context, step_index)] = parent
        runtime_by_correlation: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in events:
            if str(candidate.get("cat", "")).lower() != "cuda_runtime":
                continue
            correlation = _first(_args(candidate), "correlation", "correlation_id")
            if correlation is not None:
                runtime_by_correlation[str(correlation)].append(candidate)
        host_events = [
            candidate
            for candidate in events
            if str(candidate.get("cat", "")).lower() in {"cpu_op", "python_function", "user_annotation", "host"}
        ]
        host_events.sort(key=lambda event: float(event["ts"]))
        host_execute_contexts = [
            candidate
            for candidate in host_events
            if "execute_context" in str(candidate.get("name", "")).lower()
            or _first(_args(candidate), "execute_context", "execute_context_id") is not None
        ]
        candidates: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
        counters: defaultdict[tuple[str, int, str], int] = defaultdict(int)
        parent_diagnostics: list[dict[str, Any]] = []
        for event in sorted((e for e in events if _is_collective(e)), key=lambda e: float(e["ts"])):
            context = _context(event)
            graph = _first(_args(event), "graph_parent", "graph_id", "capture_id")
            step_index = graph_key_to_step.get((context, str(graph))) if graph is not None else None
            if graph is not None and step_index is None:
                graph_matches = [
                    (key, candidate_step)
                    for key, candidate_step in graph_key_to_step.items()
                    if key[1] == str(graph)
                ]
                if len(graph_matches) == 1:
                    (context, _graph), step_index = graph_matches[0]
            correlation_chain = _correlation_host_chain(
                {
                    "runtime_by_correlation": runtime_by_correlation,
                    "host_execute_contexts": host_execute_contexts,
                },
                event,
            )
            chain_resolved = False
            if graph is None and correlation_chain and not correlation_chain.get("ambiguous"):
                chain_key = (
                    str(correlation_chain.get("execute_context", "default")),
                    str(correlation_chain.get("graph_id", "")),
                )
                step_index = graph_key_to_step.get(chain_key)
                if step_index is not None:
                    context = chain_key[0]
                    chain_resolved = True
                else:
                    graph_matches = [
                        (key, candidate_step)
                        for key, candidate_step in graph_key_to_step.items()
                        if key[1] == chain_key[1]
                    ]
                    if len(graph_matches) == 1:
                        (context, _graph), step_index = graph_matches[0]
                        chain_resolved = True
            if step_index is None or context == "default":
                containing = [
                    (key, parent)
                    for key, parent in graph_by_step.items()
                    if (context == "default" or key[0] == context)
                    and float(parent["ts"]) <= float(event["ts"]) <= _end(parent)
                ]
                if step_index is None and len(containing) > 1 and not chain_resolved and (
                    not correlation_chain or correlation_chain.get("ambiguous")
                ):
                    parent_diagnostics.append(
                        {
                            "reason": "ambiguous-parent",
                            "event": _event_summary(event),
                            "candidate_graphs": [
                                {"execute_context": key[0], "step_index": key[1], "graph_parent": _graph_id(parent)}
                                for key, parent in containing
                            ],
                        }
                    )
                    step_index = -1
                elif containing:
                    parent_key, _parent = min(containing, key=lambda item: _end(item[1]) - float(item[1]["ts"]))
                    context, step_index = parent_key
                elif step_index is None:
                    step_index = -1
            stream = _stream(event)
            base = (context, step_index, stream)
            ordinal = _ordinal(event, counters[base])
            counters[base] = max(counters[base], ordinal + 1)
            candidates[(context, step_index, stream, ordinal)].append(event)
        selected = {key: _choose(value) for key, value in candidates.items()}
        device_by_stream: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in events:
            if _device_event(candidate):
                device_by_stream[_stream(candidate)].append(candidate)
        for stream_events in device_by_stream.values():
            stream_events.sort(key=_end)
        device_end_times = {
            stream: [_end(candidate) for candidate in stream_events]
            for stream, stream_events in device_by_stream.items()
        }
        host_events = [
            candidate
            for candidate in events
            if str(candidate.get("cat", "")).lower() in {"cpu_op", "python_function", "user_annotation", "host"}
        ]
        host_events.sort(key=lambda event: float(event["ts"]))
        stage_events = [
            candidate
            for candidate in events
            if str(candidate.get("cat", "")).lower()
            in {"user_annotation", "gpu_user_annotation"}
            and str(candidate.get("name", "")).startswith("plan35.")
        ]
        host_by_external: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in host_events:
            external_id = _first(_args(candidate), "External id", "external_id")
            if external_id is not None:
                host_by_external[str(external_id)].append(candidate)
        host_active_by_device: dict[int, list[dict[str, Any]]] = {}
        active_hosts: dict[int, dict[str, Any]] = {}
        ending_hosts: list[tuple[float, int]] = []
        host_index = 0
        for device_event in sorted((candidate for candidate in events if _device_event(candidate)), key=lambda event: float(event["ts"])):
            device_ts = float(device_event["ts"])
            while host_index < len(host_events) and float(host_events[host_index]["ts"]) <= device_ts:
                host_event = host_events[host_index]
                host_id = id(host_event)
                active_hosts[host_id] = host_event
                heapq.heappush(ending_hosts, (_end(host_event), host_id))
                host_index += 1
            while ending_hosts and ending_hosts[0][0] < device_ts:
                _end_time, host_id = heapq.heappop(ending_hosts)
                active_hosts.pop(host_id, None)
            host_active_by_device[id(device_event)] = list(active_hosts.values())
        step_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
        for context, step_index, _stream_name, _ordinal_value in selected:
            step_counts[(context, step_index)] += 1
        per_rank[rank] = {
            "events": events,
            "graphs": graph_by_step,
            "graph_ids": {key: _graph_id(parent) for key, parent in graph_by_step.items()},
            "device_by_stream": device_by_stream,
            "device_end_times": device_end_times,
            "runtime_by_correlation": runtime_by_correlation,
            "host_events": host_events,
            "host_execute_contexts": host_execute_contexts,
            "stage_events": stage_events,
            "host_by_external": host_by_external,
            "host_active_by_device": host_active_by_device,
            "collectives": selected,
            "candidate_counts": {key: len(value) for key, value in candidates.items()},
            "step_counts": dict(step_counts),
            "parent_diagnostics": parent_diagnostics,
        }
        calibration_by_rank[rank] = _calibration(_data, events)
        all_keys.update(selected)

    markers = {marker for marker, _offset in calibration_by_rank.values()}
    offsets = [offset for marker, offset in calibration_by_rank.values() if marker is not None and offset is not None]
    if len(ranks) == 4 and len(markers) == 1 and None not in markers and len(offsets) == 4:
        residual = max(offsets) - min(offsets)
        absolute_alignment = "available" if residual <= 2 else "rejected-residual-over-2us"
    else:
        residual = None
        absolute_alignment = "unavailable"

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    duplicate_ranks = sorted(rank for rank, count in Counter(ranks).items() if count > 1)
    diagnostics.extend({"rank": rank, "reason": "duplicate-rank-trace"} for rank in duplicate_ranks)
    diagnostics.extend(
        {"rank": rank, **diagnostic}
        for rank, item in sorted(per_rank.items())
        for diagnostic in item.get("parent_diagnostics", [])
    )
    expected_ranks = set(range(4))
    for key in sorted(all_keys):
        context, step_index, stream, ordinal = key
        members = []
        missing = []
        for rank in range(4):
            item = per_rank.get(rank)
            event = item["collectives"].get(key) if item else None
            if event is None:
                missing.append(rank)
                continue
            anchor = item["graphs"].get((context, step_index))
            if anchor is None:
                missing.append(rank)
                continue
            start = float(event["ts"])
            end = _end(event)
            anchor_start = float(anchor["ts"])
            stream_events = item.get("device_by_stream", {}).get(stream, [])
            previous = None
            if stream_events:
                end_times = item["device_end_times"][stream]
                previous_index = bisect.bisect_right(end_times, start) - 1
                if previous_index >= 0:
                    previous = stream_events[previous_index]
                    if previous is event:
                        previous = stream_events[previous_index - 1] if previous_index else None
            parent = anchor
            host_chain = _host_chain(item, previous)
            stage_annotation, stage_family, stage_prefix, stage_ambiguous = _stage_attribution(item, previous)
            host_category = _operator_category(str(host_chain["host"].get("name", ""))) if host_chain and host_chain.get("host") else "other"
            if stage_ambiguous:
                preceding_category = "other"
            elif stage_family == "attention_o_proj":
                preceding_category = "attention"
            elif stage_family == "mlp_down_proj":
                preceding_category = "other"
            elif host_chain and not host_chain.get("ambiguous") and host_category not in {"other", "graph/host gap"}:
                preceding_category = host_category
            elif previous:
                preceding_category = _operator_category(previous.get("name", ""))
            else:
                preceding_category = _operator_category(previous.get("name", "")) if previous else "graph/host gap"
            if host_chain and host_chain.get("ambiguous"):
                preceding_category = "other"
            members.append(
                {
                    "rank": rank,
                    "event_name": event.get("name", ""),
                    "raw_start_us": start,
                    "raw_end_us": end,
                    "relative_start_us": start - anchor_start,
                    "relative_end_us": end - anchor_start,
                    "residency_us": end - start,
                    "pre_gap_us": start - _end(previous) if previous else None,
                    "preceding_event": previous.get("name") if previous else None,
                    "preceding_category": preceding_category,
                    "preceding_operator_category": preceding_category,
                    "preceding_host_chain": host_chain,
                    "preceding_stage_annotation": stage_annotation,
                    "preceding_stage_family": stage_family,
                    "preceding_stage_prefix": stage_prefix,
                    "enclosing_graph": parent.get("name", ""),
                    "graph_duration_us": float(anchor.get("dur", 0)),
                }
            )
        semantic_stage, layer = map_dsv4_ordinal(ordinal)
        record: dict[str, Any] = {
            "execute_context": context,
            "step_index": step_index,
            "graph_parent": per_rank.get(members[0]["rank"], {}).get("graph_ids", {}).get((context, step_index), "") if members else "",
            "stream": stream,
            "ordinal": ordinal,
            "semantic_stage": semantic_stage,
            "layer": layer,
            "matched_ranks": [member["rank"] for member in members],
            "missing_ranks": missing,
            "ranks": sorted(members, key=lambda member: member["rank"]),
            "nested_parent_child_collapsed": any(
                per_rank.get(member["rank"], {}).get("candidate_counts", {}).get(key, 0) > 1
                for member in members
            ),
        }
        if len(members) == 4:
            record["raw_start_us"] = [member["raw_start_us"] for member in members]
            record["raw_end_us"] = [member["raw_end_us"] for member in members]
        else:
            record["raw_start_us"] = None
            record["raw_end_us"] = None
        if len(members) == 4:
            starts = [member["relative_start_us"] for member in members]
            ends = [member["relative_end_us"] for member in members]
            completion_tail = max(max(0.0, end - max(starts)) for end in ends)
            completion_tail_interval = [max(starts), max(ends)]
            max_start = max(starts)
            late_candidates = [member["rank"] for member, start in zip(members, starts) if start == max_start]
            late_tie = len(late_candidates) > 1
            record.update(
                {
                    "relative_start_us": starts,
                    "relative_end_us": ends,
                    "arrival_skew_us": max(starts) - min(starts),
                    "completion_tail_us": completion_tail,
                    "completion_tail": completion_tail,
                    "completion_tail_interval_us": completion_tail_interval,
                    "late_rank": None if late_tie else late_candidates[0],
                    "late_rank_candidate": None if late_tie else late_candidates[0],
                    "late_rank_candidates": late_candidates,
                    "late_rank_tie": late_tie,
                }
            )
            if absolute_alignment == "available":
                absolute_offsets = [calibration_by_rank[member["rank"]][1] for member in members]
                absolute_starts = [member["raw_start_us"] - offset for member, offset in zip(members, absolute_offsets)]
                absolute_ends = [member["raw_end_us"] - offset for member, offset in zip(members, absolute_offsets)]
                record.update(
                    {
                        "absolute_start_us": absolute_starts,
                        "absolute_end_us": absolute_ends,
                        "absolute_arrival_skew_us": max(absolute_starts) - min(absolute_starts),
                        "absolute_completion_tail_us": max(
                            max(0.0, end - max(absolute_starts)) for end in absolute_ends
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "absolute_start_us": None,
                        "absolute_end_us": None,
                        "absolute_arrival_skew_us": None,
                        "absolute_completion_tail_us": None,
                    }
                )
        else:
            diagnostics.append({"key": list(key), "missing_ranks": missing, "reason": "incomplete-four-rank-match"})
            record.update(
                {
                    "relative_start_us": None,
                    "relative_end_us": None,
                    "arrival_skew_us": None,
                    "completion_tail_us": None,
                    "completion_tail": None,
                    "late_rank": None,
                    "late_rank_candidate": None,
                    "late_rank_candidates": [],
                    "late_rank_tie": False,
                    "absolute_start_us": None,
                    "absolute_end_us": None,
                    "absolute_arrival_skew_us": None,
                    "absolute_completion_tail_us": None,
                    "completion_tail_interval_us": None,
                }
            )
        records.append(record)

    previous_by_group: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        group = (record["execute_context"], record["step_index"], record["stream"])
        previous = previous_by_group.get(group)
        spans = None
        if record["relative_start_us"] is not None:
            if record["ordinal"] == 0:
                previous_ends = [0.0] * len(record["relative_start_us"])
            elif previous and previous["ordinal"] == record["ordinal"] - 1:
                previous_ends = previous["relative_end_us"]
            else:
                previous_ends = None
            if previous_ends is not None:
                spans = [
                    start - previous_end
                    for start, previous_end in zip(record["relative_start_us"], previous_ends)
                ]
        record["inter_collective_span_us"] = spans
        record["span_skew_us"] = max(spans) - min(spans) if spans is not None else None
        previous_by_group[group] = record

    diagnostics.extend(
        {"rank": rank, "reason": "missing-rank-trace"}
        for rank in sorted(expected_ranks - set(ranks))
    )
    graph_variants: defaultdict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for rank, item in per_rank.items():
        for (context, step_index), graph_id in item["graph_ids"].items():
            graph_variants[(context, step_index)][rank] = graph_id
    for (context, step_index), values in graph_variants.items():
        frequencies = Counter(values.values())
        if len(frequencies) > 1:
            diagnostics.append(
                {
                    "execute_context": context,
                    "step_index": step_index,
                    "graph_parents_by_rank": {str(rank): graph_id for rank, graph_id in sorted(values.items())},
                    "reason": "mismatched-graph-anchor",
                }
            )
    count_variants: defaultdict[tuple[str, int, str], dict[int, int]] = defaultdict(dict)
    for rank, item in per_rank.items():
        counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        for context, step_index, stream, ord_value in item["collectives"]:
            counts[(context, step_index, stream)] += 1
        for base in {(context, step_index, stream) for context, step_index, stream, ord_value in item["collectives"]} | set(counts):
            count_variants[base][rank] = counts.get(base, 0)
    diagnostics.extend(
        {
            "execute_context": context,
            "step_index": step_index,
            "stream": stream,
            "counts_by_rank": {str(rank): count for rank, count in sorted(values.items())},
            "reason": "mismatched-collective-count",
        }
        for (context, step_index, stream), values in count_variants.items()
        if len(set(values.values())) > 1
    )
    complete = len(ranks) == 4 and not diagnostics and bool(records)
    per_rank_collective_counts = {
        str(rank): len(item["collectives"]) for rank, item in sorted(per_rank.items())
    }
    per_step_collective_counts = {
        str(rank): [
            {
                "execute_context": context,
                "graph_parent": graph,
                "count": count,
            }
            for (context, graph), count in sorted(item["step_counts"].items())
        ]
        for rank, item in sorted(per_rank.items())
    }
    expected_total = expected_steps * expected_collectives_per_step
    count_gate = all(
        per_rank_collective_counts.get(str(rank)) == expected_total
        and len(per_step_collective_counts.get(str(rank), [])) == expected_steps
        and all(step["count"] == expected_collectives_per_step for step in per_step_collective_counts[str(rank)])
        for rank in range(4)
    )
    arrival_values = [record["arrival_skew_us"] for record in records if record["arrival_skew_us"] is not None]
    tail_values = [record["completion_tail_us"] for record in records if record["completion_tail_us"] is not None]
    pre_gap_values = [
        member["pre_gap_us"]
        for record in records
        for member in record["ranks"]
        if member["pre_gap_us"] is not None
    ]
    residency_by_rank: defaultdict[str, list[float]] = defaultdict(list)
    late_ranks: Counter[str] = Counter()
    late_rank_tie_count = sum(1 for record in records if record.get("late_rank_tie"))
    preceding_categories: Counter[str] = Counter()
    stage_annotations: Counter[str] = Counter()
    for record in records:
        if record["late_rank_candidate"] is not None:
            late_ranks[str(record["late_rank_candidate"])] += 1
        for member in record["ranks"]:
            residency_by_rank[str(member["rank"])].append(member["residency_us"])
            preceding_categories[member["preceding_operator_category"]] += 1
            stage_annotation = member.get("preceding_stage_annotation")
            if stage_annotation:
                stage_name = "ambiguous" if stage_annotation.get("ambiguous") else stage_annotation.get("name")
                if stage_name:
                    stage_annotations[str(stage_name)] += 1
    aggregates = {
        "arrival_skew_us": _metric_stats(arrival_values),
        "completion_tail_us": _metric_stats(tail_values),
        "pre_gap_us": _metric_stats(pre_gap_values),
        "residency_us_by_rank": {
            rank: _metric_stats(values) for rank, values in sorted(residency_by_rank.items())
        },
        "late_rank_distribution": dict(sorted(late_ranks.items())),
            "preceding_category_distribution": dict(sorted(preceding_categories.items())),
            "late_rank_tie_count": late_rank_tie_count,
            "stage_annotation_distribution": dict(sorted(stage_annotations.items())),
    }
    stage_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stage_records[record["semantic_stage"]].append(record)
    stage_aggregates: dict[str, Any] = {}
    for stage, values in sorted(stage_records.items()):
        ordinal_arrivals: defaultdict[int, list[float]] = defaultdict(list)
        ordinal_spans: defaultdict[int, list[float]] = defaultdict(list)
        for record in values:
            if record["arrival_skew_us"] is not None:
                ordinal_arrivals[record["ordinal"]].append(record["arrival_skew_us"])
            if record["span_skew_us"] is not None:
                ordinal_spans[record["ordinal"]].append(record["span_skew_us"])
        top_arrivals = [
            {"ordinal": ordinal, **_metric_stats(metric_values)}
            for ordinal, metric_values in ordinal_arrivals.items()
        ]
        top_spans = [
            {"ordinal": ordinal, **_metric_stats(metric_values)}
            for ordinal, metric_values in ordinal_spans.items()
        ]
        top_arrivals.sort(key=lambda item: (-(item["p90"] or 0), item["ordinal"]))
        top_spans.sort(key=lambda item: (-(item["p90"] or 0), item["ordinal"]))
        stage_aggregates[stage] = {
            "arrival_skew_us": _metric_stats(
                [record["arrival_skew_us"] for record in values if record["arrival_skew_us"] is not None]
            ),
            "span_skew_us": _metric_stats(
                [record["span_skew_us"] for record in values if record["span_skew_us"] is not None]
            ),
            "top_arrival_skew_ordinals": top_arrivals,
            "top_span_skew_ordinals": top_spans,
        }
    intervals_by_step: defaultdict[int, list[list[float]]] = defaultdict(list)
    active_windows_by_step: defaultdict[int, list[float]] = defaultdict(list)
    for record in records:
        interval = record.get("completion_tail_interval_us")
        if interval is not None:
            intervals_by_step[record["step_index"]].append(interval)
        active_windows_by_step[record["step_index"]].extend(
            member["graph_duration_us"] for member in record["ranks"]
        )
    completion_tail_union_us = 0.0
    for intervals in intervals_by_step.values():
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        completion_tail_union_us += sum(end - start for start, end in merged)
    active_device_window_us = sum(max(values) for values in active_windows_by_step.values() if values)
    completion_tail_union_fraction = (
        completion_tail_union_us / active_device_window_us if active_device_window_us else None
    )
    return {
        "ranks": ranks,
        "collectives": records,
        "summary": {
            "rank_count": len(ranks),
            "collective_count": len(records),
            "complete_four_rank_match": complete,
            "missing_ranks": sorted(expected_ranks - set(ranks)),
            "unmatched_diagnostics": diagnostics,
            "absolute_alignment": absolute_alignment,
            "calibration_residual_us": residual,
            "per_rank_collective_counts": per_rank_collective_counts,
            "active_step_count_by_rank": {
                rank: len(steps) for rank, steps in ((rank, item["step_counts"]) for rank, item in sorted(per_rank.items()))
            },
            "per_step_collective_counts": per_step_collective_counts,
            "expected": {
                "steps": expected_steps,
                "collectives_per_step": expected_collectives_per_step,
                "total_collectives": expected_total,
            },
            "routing_valid": complete and count_gate,
            "late_rank_tie_count": late_rank_tie_count,
            "aggregates": aggregates,
            "stage_aggregates": stage_aggregates,
            "completion_tail_union_us": completion_tail_union_us,
            "active_device_window_us": active_device_window_us,
            "completion_tail_union_fraction": completion_tail_union_fraction,
        },
    }


def write_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "collectives.jsonl").open("w", encoding="utf-8") as stream:
        for record in result["collectives"]:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = result["collectives"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["execute_context", "graph_parent", "stream", "ordinal", "arrival_skew_us", "completion_tail_us", "late_rank", "missing_ranks"])
        writer.writeheader()
        for record in rows:
            writer.writerow({field: record.get(field) for field in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="four plain or gzip profiler JSON traces")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--expected-steps", type=int, default=5)
    parser.add_argument("--expected-collectives-per-step", type=int, default=87)
    args = parser.parse_args()
    result = analyze_traces(
        args.traces,
        expected_steps=args.expected_steps,
        expected_collectives_per_step=args.expected_collectives_per_step,
    )
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
