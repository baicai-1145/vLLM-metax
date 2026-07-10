#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ATTN_RE = re.compile(
    r"layers\.\d+\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b|fused_wqa_wkv|q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)\.weight$"
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _is_attention_weight(name: str) -> bool:
    return bool(ATTN_RE.fullmatch(name))


def _expand_block_scales(scale: torch.Tensor, weight_shape: tuple[int, int], block_shape: tuple[int, int]) -> torch.Tensor:
    block_m, block_k = block_shape
    expanded = torch.repeat_interleave(scale.to(torch.float32), block_m, dim=0)
    expanded = torch.repeat_interleave(expanded, block_k, dim=1)
    return expanded[: weight_shape[0], : weight_shape[1]]


def _dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    expanded_scale = _expand_block_scales(scale, tuple(weight.shape), block_shape)
    return (weight.to(torch.float32) * expanded_scale).to(torch.bfloat16)


def _copy_aux_files(src: Path, dst: Path) -> None:
    for name in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def _rewrite_config(src_cfg: dict) -> dict:
    cfg = json.loads(json.dumps(src_cfg))
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        raise ValueError("config.json has no quantization_config")

    cfg.setdefault("_rewrite_notes", {})["attention_fp8_rewritten_to_bf16"] = True
    cfg.setdefault("_rewrite_notes", {})["source_quant_method"] = qcfg.get("quant_method")

    config_groups = qcfg.get("config_groups", {})
    group0 = config_groups.get("group_0")
    if group0 is not None:
        group0["targets"] = []

    ignore = qcfg.setdefault("ignore", [])
    attn_target = "re:.*attn\\.(wq_a|wq_b|wkv|wo_a|wo_b|fused_wqa_wkv|q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$"
    if attn_target not in ignore:
        ignore.append(attn_target)

    # Keep mixed-precision mode so expert W4A16 routing remains intact.
    # The attention linear layers are rewritten to plain BF16 tensors and excluded
    # from compressed-tensors matching via `ignore` / empty group_0 targets.
    return cfg


def _rewrite_index(src_index: dict) -> dict:
    new_index = json.loads(json.dumps(src_index))
    new_weight_map = {}
    for key, shard in src_index["weight_map"].items():
        if key.endswith(".weight_scale") and _is_attention_weight(key.replace(".weight_scale", ".weight")):
            continue
        new_weight_map[key] = shard
    new_index["weight_map"] = new_weight_map
    return new_index


def convert_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    index = _load_json(src_dir / "model.safetensors.index.json")
    cfg = _load_json(src_dir / "config.json")

    block_shape = tuple(cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["block_structure"])
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)

        out_path = dst_dir / shard_name
        if out_path.exists():
            print(f"[skip {shard_idx}/{len(shard_names)}] {shard_name} already exists", flush=True)
            continue

        print(f"[convert {shard_idx}/{len(shard_names)}] {shard_name}", flush=True)

        tensors_out: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            skip: set[str] = set()
            for key in keys:
                if key in skip:
                    continue

                if _is_attention_weight(key):
                    scale_key = key.replace(".weight", ".weight_scale")
                    if scale_key not in keys:
                        raise KeyError(f"Missing scale tensor for {key}")
                    w = f.get_tensor(key)
                    s = f.get_tensor(scale_key)
                    tensors_out[key] = _dequant_block_fp8(w, s, block_shape)
                    skip.add(scale_key)
                    continue

                if key.endswith(".weight_scale") and _is_attention_weight(key.replace(".weight_scale", ".weight")):
                    continue

                tensors_out[key] = f.get_tensor(key)

        save_file(tensors_out, str(out_path))
        print(f"[done {shard_idx}/{len(shard_names)}] {shard_name}", flush=True)

    new_cfg = _rewrite_config(cfg)
    new_index = _rewrite_index(index)
    _save_json(dst_dir / "config.json", new_cfg)
    _save_json(dst_dir / "model.safetensors.index.json", new_index)
    _copy_aux_files(src_dir, dst_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite DeepSeek-V4 mixed FP8/W4A16 MTP checkpoint into BF16-attn + W4A16 experts.")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.dst.exists() and any(args.dst.iterdir()) and not args.resume:
        raise SystemExit(f"Destination {args.dst} is not empty")

    convert_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
