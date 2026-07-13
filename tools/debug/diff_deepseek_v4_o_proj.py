#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a captured DeepSeek V4 O-projection call and print JSON diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_metax.models.deepseek_v4.ops import o_proj_debug


def run_diff(path: str | Path, *, backend: str = "torch") -> dict[str, object]:
    path = Path(path)
    if path.is_dir():
        details = [
            run_diff(candidate, backend=backend)
            for candidate in sorted(path.glob("*.pt"))
        ]
        return {
            "path": str(path),
            "backend": backend,
            "files": len(details),
            "passed": all(bool(item.get("passed")) for item in details),
            "failed": sum(not bool(item.get("passed")) for item in details),
            "details": details,
        }
    try:
        return o_proj_debug.replay_capture(path, backend=backend)
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        return {
            "path": str(path),
            "backend": backend,
            "passed": False,
            "error": str(exc),
            "stages": {},
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--backend", choices=("torch", "native"), default="torch")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_diff(args.capture, backend=args.backend)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if summary.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
