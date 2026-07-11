#!/usr/bin/env python3
"""Run the exact MetaX MMA MHC post candidate on one captured payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.debug.diff_deepseek_v4_mhc_raw import (
    load_fused_post_payload,
    run_exact_mhc_post_tilelang,
)
from vllm_metax.models.deepseek_v4.ops.mhc.debug_diff import tensor_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = load_fused_post_payload(args.payload, device=args.device)
    got = run_exact_mhc_post_tilelang(payload)
    if args.device == "cuda":
        torch.cuda.synchronize()
    diff = tensor_diff(payload["residual_cur_bf16"], got)
    print(json.dumps({"payload": str(args.payload), "post_mma": diff}, indent=2))
    if not diff["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
