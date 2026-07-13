#!/usr/bin/env python3
"""Compile and run the isolated fixed-shape Plan 03 O-projection probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
def _expand_cache(payload: dict[str, object], device: torch.device) -> torch.Tensor:
    compact = payload["cos_sin_cache"].to(device)
    compact_positions = payload["cos_sin_positions"].to(device=device, dtype=torch.long)
    max_position = int(compact_positions.max().item()) + 1
    cache = torch.zeros((max_position, 64), device=device, dtype=torch.float32)
    cache.index_copy_(0, compact_positions, compact)
    return cache


@torch.no_grad()
def run(capture: Path, build_dir: Path, device: str) -> dict[str, object]:
    payload = torch.load(capture, map_location="cpu", weights_only=True)
    del build_dir
    import vllm_metax._metax_sparse_C  # noqa: F401
    target = torch.device(device)
    o = payload["o"][:1].to(target)
    positions = payload["positions"][:1].to(target)
    cos_sin = _expand_cache(payload, target)
    wo_a = payload["wo_a"].view(2, 1024, 4096).to(target)
    z = torch.empty((1, 2, 1024), device=target, dtype=torch.bfloat16)
    torch.ops.metax_o_proj_probe.fused_bf16_out(o, positions, cos_sin, wo_a, z)
    torch.cuda.synchronize(target)

    expected = payload["z"][:1]
    delta = (z.float().cpu() - expected.float()).abs()
    bitwise = bool(torch.equal(z.cpu().view(torch.int16), expected.view(torch.int16)))
    return {
        "capture": str(capture),
        "shape": [1, 16, 512],
        "bitwise": bitwise,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "allclose": bool(torch.allclose(z.cpu(), expected, atol=0.125, rtol=0.01)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".logs/o_proj_probe_build")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.capture, args.build_dir, args.device)
    except Exception as exc:
        result = {"capture": str(args.capture), "passed": False, "error": str(exc)}
    else:
        result["passed"] = bool(result["allclose"])
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
