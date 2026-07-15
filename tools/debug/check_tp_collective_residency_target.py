#!/usr/bin/env python3
"""Check the per-rank MCCL residency target from analyzer JSONL output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _collectives_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        analysis_path = candidate / "analysis" / "collectives.jsonl"
        candidate = analysis_path if analysis_path.is_file() else candidate / "collectives.jsonl"
    return candidate


def read_collectives(path: str | Path) -> list[dict[str, Any]]:
    """Read one analyzer ``collectives.jsonl`` file."""

    source = _collectives_path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: record is not an object")
            records.append(value)
    return records


def _run_name(path: Path) -> str:
    for parent in (path.parent, *path.parents):
        if parent.name.startswith("run") and parent.name[3:].isdigit():
            return parent.name
    return path.parent.name or path.name


def _rank_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("rank must be an integer")
    if isinstance(value, str):
        text = value.strip()
        if not text or (text[0] in "+-" and not text[1:].isdigit()) or not text.lstrip("+-").isdigit():
            raise ValueError("rank must be an integer")
        return int(text)
    rank = int(value)
    if rank != value:
        raise ValueError("rank must be an integer")
    return rank


def _residency_us(value: Any) -> float:
    residency = float(value)
    if not math.isfinite(residency) or residency < 0:
        raise ValueError("residency_us must be a finite non-negative number")
    return residency


def check_residency_target(
    paths: Iterable[str | Path] | str | Path,
    *,
    threshold_ms: float = 5.0,
    expected_rank_count: int = 4,
    expected_calls_per_rank: int = 435,
) -> dict[str, Any]:
    """Check every rank's summed residency in one or more analyzer outputs.

    The target is strict: a rank passes only when its total residency is less
    than ``threshold_ms``. Missing/extra ranks and unexpected call counts are
    structural failures even when the observed residency is below the target.
    """

    if threshold_ms < 0 or not math.isfinite(threshold_ms):
        raise ValueError("threshold_ms must be a finite non-negative number")
    if expected_rank_count <= 0 or expected_calls_per_rank <= 0:
        raise ValueError("expected rank and call counts must be positive")

    expected_ranks = set(range(expected_rank_count))
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    input_paths = [paths] if isinstance(paths, (str, Path)) else paths
    for input_path in input_paths:
        source = _collectives_path(input_path)
        run = {
            "run": _run_name(source),
            "path": str(source),
            "complete_four_ranks": False,
            "missing_ranks": [],
            "extra_ranks": [],
            "ranks": [],
            "diagnostics": [],
        }
        sums: defaultdict[int, float] = defaultdict(float)
        counts: defaultdict[int, int] = defaultdict(int)
        try:
            records = read_collectives(source)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            run["diagnostics"].append({"reason": "input-error", "detail": str(exc)})
            missing.append({"run": run["run"], "reason": "input-error", "detail": str(exc)})
            runs.append(run)
            continue

        for line_number, record in enumerate(records, start=1):
            members = record.get("ranks")
            if not isinstance(members, list):
                run["diagnostics"].append(
                    {"reason": "missing-ranks-field", "line": line_number}
                )
                continue
            for member in members:
                if not isinstance(member, dict):
                    run["diagnostics"].append(
                        {"reason": "invalid-rank-entry", "line": line_number}
                    )
                    continue
                try:
                    rank = _rank_number(member["rank"])
                    residency = _residency_us(member["residency_us"])
                except (KeyError, TypeError, ValueError) as exc:
                    run["diagnostics"].append(
                        {"reason": "invalid-rank-entry", "line": line_number, "detail": str(exc)}
                    )
                    continue
                counts[rank] += 1
                sums[rank] += residency

        observed_ranks = set(counts)
        missing_ranks = sorted(expected_ranks - observed_ranks)
        extra_ranks = sorted(observed_ranks - expected_ranks)
        run["missing_ranks"] = missing_ranks
        run["extra_ranks"] = extra_ranks
        run["complete_four_ranks"] = not missing_ranks and not extra_ranks
        if missing_ranks:
            for rank in missing_ranks:
                diagnostic = {"reason": "missing-rank", "rank": rank}
                run["diagnostics"].append(diagnostic)
                missing.append({"run": run["run"], **diagnostic})
        if extra_ranks:
            for rank in extra_ranks:
                diagnostic = {"reason": "unexpected-rank", "rank": rank}
                run["diagnostics"].append(diagnostic)
                missing.append({"run": run["run"], **diagnostic})

        for rank in sorted(observed_ranks | expected_ranks):
            calls = counts.get(rank, 0)
            total_us = sums.get(rank, 0.0)
            total_ms = total_us / 1000.0
            rank_result = {
                "rank": rank,
                "calls": calls,
                "total_residency_us": total_us,
                "total_residency_ms": total_ms,
                "under_target": calls == expected_calls_per_rank and total_ms < threshold_ms,
            }
            run["ranks"].append(rank_result)
            if calls != expected_calls_per_rank and rank not in missing_ranks:
                diagnostic = {
                    "reason": "unexpected-call-count",
                    "rank": rank,
                    "calls": calls,
                    "expected_calls": expected_calls_per_rank,
                }
                run["diagnostics"].append(diagnostic)
                missing.append({"run": run["run"], **diagnostic})
            if calls == expected_calls_per_rank and total_ms >= threshold_ms:
                failures.append(
                    {
                        "run": run["run"],
                        "rank": rank,
                        "total_residency_ms": total_ms,
                        "threshold_ms": threshold_ms,
                    }
                )
        runs.append(run)

    return {
        "ok": not failures and not missing and all(run["complete_four_ranks"] for run in runs),
        "threshold_ms": threshold_ms,
        "expected_rank_count": expected_rank_count,
        "expected_calls_per_rank": expected_calls_per_rank,
        "runs": runs,
        "failures": failures,
        "missing": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collectives", nargs="+", help="analyzer collectives.jsonl files or run directories")
    parser.add_argument("--threshold-ms", type=float, default=5.0)
    parser.add_argument("--expected-calls-per-rank", type=int, default=435)
    parser.add_argument("--output", "--output-json", dest="output", help="also write the JSON summary here")
    args = parser.parse_args(argv)
    try:
        summary = check_residency_target(
            args.collectives,
            threshold_ms=args.threshold_ms,
            expected_calls_per_rank=args.expected_calls_per_rank,
        )
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
