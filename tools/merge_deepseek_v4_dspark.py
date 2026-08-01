#!/usr/bin/env python3
"""Merge official DeepSeek-V4 DSpark MTP shards into a local checkpoint.

The target checkpoint is hardlinked into a staging directory. Only DSpark MTP
weights are materialized; existing target shards are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
_STAGE_RE = re.compile(r"(?:^|\.)mtp\.([0-2])(?:\.|$)")
_LOGICAL_RE = re.compile(r"(?:^|\.)layers\.(4[345])(?:\.|$)")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_QUANTIZED_DTYPES = {
    "U8", "I8", "F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E4M3FNUZ",
}


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _stage_for_name(name: str) -> int | None:
    match = _STAGE_RE.search(name)
    if match:
        return int(match.group(1))
    match = _LOGICAL_RE.search(name)
    if match:
        return int(match.group(1)) - 43
    return None


def _normalize_name(name: str) -> tuple[str, int] | None:
    stage = _stage_for_name(name)
    if stage is None:
        return None
    name = re.sub(r"^model\.", "", name)
    name = re.sub(r"^layers\.4([345])\.",
                  lambda m: f"mtp.{int(m.group(1)) - 3}.", name)
    name = re.sub(r"^layers\.([0-2])\.", r"mtp.\1.", name)
    name = re.sub(r"^mtp\.([0-2])\.", r"mtp.\1.", name)
    name = name.replace(".weight_packed", ".weight")
    return name, stage


def _decode_e8m0(scale: Any) -> Any:
    import torch

    if scale.dtype == torch.uint8:
        # E8M0 stores a base-2 exponent with bias 127.
        return torch.ldexp(torch.ones_like(scale, dtype=torch.float32),
                           scale.to(torch.int16) - 127)
    return scale.to(torch.float32)


def _decode_fp8(weight: Any) -> Any:
    import torch

    if weight.dtype == torch.uint8:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("PyTorch lacks float8_e4m3fn for DSpark FP8")
        return weight.view(dtype).to(torch.float32)
    return weight.to(torch.float32)


def dequant_mxfp4(weight: Any, scale: Any, chunk_rows: int = 256) -> Any:
    """Decode low-nibble-first MXFP4 and return BF16."""
    import torch

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("MXFP4 weight and scale must be rank-2")
    rows, packed_cols = weight.shape
    cols = packed_cols * 2
    row_factor = 1 if scale.shape[0] == rows else 128
    col_factor = 32 if scale.shape[1] * 32 >= cols and scale.shape[1] != (cols + 127) // 128 else 128
    if scale.shape[0] * row_factor < rows or scale.shape[1] * col_factor < cols:
        raise ValueError(f"invalid MXFP4 scale shape {tuple(scale.shape)} for {tuple(weight.shape)}")
    lookup = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=weight.device)
    scales = _decode_e8m0(scale)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        packed = weight[start:stop].to(torch.uint8)
        nibbles = torch.empty((stop - start, cols), dtype=torch.uint8,
                              device=weight.device)
        nibbles[:, 0::2] = packed & 0x0F
        nibbles[:, 1::2] = packed >> 4
        values = lookup[nibbles.to(torch.long)]
        block_start, block_stop = start // row_factor, (stop + row_factor - 1) // row_factor
        expanded = torch.repeat_interleave(
            scales[block_start:block_stop], col_factor, dim=1
        )
        expanded = torch.repeat_interleave(expanded, row_factor, dim=0)
        expanded = expanded[start - block_start * row_factor:stop - block_start * row_factor]
        out[start:stop] = (values * expanded[:, :cols]).to(torch.bfloat16)
    return out


def dequant_fp8_block(weight: Any, scale: Any,
                      block_shape: tuple[int, int] = (128, 128),
                      chunk_rows: int = 256) -> Any:
    """Decode block-scaled FP8 E4M3 and return BF16."""
    import torch

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must be rank-2")
    rows, cols = weight.shape
    bm, bk = block_shape
    if scale.shape[0] * bm < rows or scale.shape[1] * bk < cols:
        raise ValueError(f"invalid FP8 scale shape {tuple(scale.shape)} for {tuple(weight.shape)}")
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=weight.device)
    scales = _decode_e8m0(scale)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        block_start, block_stop = start // bm, (stop + bm - 1) // bm
        row_scale = torch.repeat_interleave(
            scales[block_start:block_stop], bm, dim=0
        )[start - block_start * bm:stop - block_start * bm]
        row_scale = torch.repeat_interleave(row_scale, bk, dim=1)[:, :cols]
        values = _decode_fp8(weight[start:stop])
        out[start:stop] = (values * row_scale).to(torch.bfloat16)
    return out


def _paired_scale(keys: set[str], key: str) -> str | None:
    if key.endswith(".weight_packed"):
        base = key.removesuffix("_packed")
    elif key.endswith(".weight"):
        base = key
    else:
        return None
    for candidate in (base + "_scale", base.removesuffix(".weight") + ".scale",
                      key + "_scale"):
        if candidate in keys:
            return candidate
    return None


def _global_paired_scale(source_map: dict[str, str], key: str) -> str | None:
    """Find a pair using the complete source index, requiring one shard."""
    candidate = _paired_scale(set(source_map), key)
    if candidate is not None and source_map[candidate] != source_map[key]:
        raise ValueError(
            f"weight/scale pair is split across shards: {key} and {candidate}"
        )
    return candidate


def _is_scale_key(key: str) -> bool:
    return key.endswith((".weight_scale", ".scale"))


def _is_quantized_dtype(dtype: str) -> bool:
    return dtype.upper() in _QUANTIZED_DTYPES


def _validate_shard_name(root: Path, name: str) -> None:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe shard path {name!r}")
    try:
        (root / path).resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"shard path escapes checkpoint directory: {name!r}") from exc


def _filtered_shard_name(shard: str) -> str:
    path = Path(shard)
    return str(path.with_name(path.stem + "-base.safetensors"))


def _payload_nbytes(directory: Path, mapping: dict[str, str],
                    keys: set[str] | None = None) -> int:
    """Read safetensors metadata without materializing tensors."""
    from safetensors import safe_open

    dtype_bytes = {
        "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
        "U32": 4, "I32": 4, "U64": 8, "I64": 8, "F8_E4M3": 1,
        "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    }
    by_shard: dict[str, list[str]] = {}
    for key, shard in mapping.items():
        if keys is None or key in keys:
            by_shard.setdefault(shard, []).append(key)
    total = 0
    for shard, shard_keys in by_shard.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                view = handle.get_slice(key)
                try:
                    itemsize = dtype_bytes[view.get_dtype()]
                except KeyError as exc:
                    raise ValueError(f"unsupported safetensors dtype {view.get_dtype()}") from exc
                nitems = 1
                for dim in view.get_shape():
                    nitems *= dim
                total += nitems * itemsize
    return total


def _validate_configs(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ("model_type", "num_hidden_layers", "hidden_size", "vocab_size"):
        if field in target and field in source and target[field] != source[field]:
            raise ValueError(f"source/target config mismatch for {field}")
    required = ("dspark_block_size", "dspark_noise_token_id",
                "dspark_target_layer_ids", "dspark_markov_rank")
    if any(field not in source for field in required):
        raise ValueError("source config is not an official DSpark config")


def _rewrite_config(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(target))
    for field, value in source.items():
        if field.startswith("dspark_"):
            cfg[field] = json.loads(json.dumps(value))
    if "compress_ratios" in source:
        cfg["compress_ratios"] = source["compress_ratios"]
    cfg["compress_rope_theta"] = source.get("compress_rope_theta", cfg.get("compress_rope_theta"))
    cfg["num_nextn_predict_layers"] = target.get(
        "num_nextn_predict_layers", source.get("num_nextn_predict_layers", 1)
    )
    cfg["n_mtp_layers"] = 3
    qcfg = cfg.setdefault("quantization_config", {})
    ignore = qcfg.setdefault("ignore", [])
    for layer in (43, 44, 45):
        for prefix in ("re:^layers\\.", "re:^model\\.layers\\."):
            entry = f"{prefix}{layer}\\."
            if entry not in ignore:
                ignore.append(entry)
    return cfg


def merge_checkpoint(target_dir: Path, dspark_dir: Path, output_dir: Path,
                     *, resume: bool = False, chunk_rows: int = 256) -> None:
    target_dir, dspark_dir, output_dir = (p.resolve() for p in
                                          (target_dir, dspark_dir, output_dir))
    if output_dir in (target_dir, dspark_dir):
        raise ValueError("output directory must differ from target and DSpark directories")
    target_index = _json(target_dir / INDEX_NAME)
    source_index = _json(dspark_dir / INDEX_NAME)
    target_cfg = _json(target_dir / CONFIG_NAME)
    source_cfg = _json(dspark_dir / CONFIG_NAME)
    _validate_configs(target_cfg, source_cfg)
    target_map = target_index.get("weight_map")
    source_map = source_index.get("weight_map")
    if not isinstance(target_map, dict) or not isinstance(source_map, dict):
        raise ValueError("both indices must contain weight_map objects")
    for shard in set(target_map.values()):
        if not isinstance(shard, str):
            raise ValueError("target index shard names must be strings")
        _validate_shard_name(target_dir, shard)
    for shard in set(source_map.values()):
        if not isinstance(shard, str):
            raise ValueError("source index shard names must be strings")
        _validate_shard_name(dspark_dir, shard)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_items: dict[int, list[tuple[str, str]]] = {0: [], 1: [], 2: []}
    for key, shard in source_map.items():
        normalized = _normalize_name(key)
        if normalized is not None:
            source_items[normalized[1]].append((key, shard))
    if any(not values for values in source_items.values()):
        raise ValueError("source index must contain mtp stages 0, 1, and 2")
    for stage, values in source_items.items():
        shards = sorted({shard for _, shard in values})
        if any(not (dspark_dir / shard).is_file() for shard in shards):
            missing = [shard for shard in shards if not (dspark_dir / shard).is_file()]
            raise FileNotFoundError(f"missing DSpark shard(s) for stage {stage}: {missing}")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Validate all indexed weights and global weight/scale placement before
    # creating any output files.
    source_expected: dict[str, tuple[tuple[int, ...], str]] = {}
    for stage_items in source_items.values():
        by_shard: dict[str, list[str]] = {}
        for source_key, source_shard in stage_items:
            by_shard.setdefault(source_shard, []).append(source_key)
        for source_shard, shard_keys in by_shard.items():
            with safe_open(dspark_dir / source_shard, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                for source_key in shard_keys:
                    if source_key not in actual_keys:
                        raise KeyError(f"index key {source_key!r} absent from {source_shard}")
                    if _is_scale_key(source_key):
                        continue
                    pair = _global_paired_scale(source_map, source_key)
                    dtype = handle.get_slice(source_key).get_dtype()
                    if pair is None and (
                        source_key.endswith(".weight_packed")
                        or _is_quantized_dtype(dtype)
                    ):
                        raise ValueError(f"quantized weight {source_key} has no paired scale")
                    normalized = _normalize_name(source_key)
                    assert normalized is not None
                    shape = tuple(handle.get_slice(source_key).get_shape())
                    if pair is not None and ".experts." in source_key:
                        shape = (shape[0], shape[1] * 2)
                    source_expected[normalized[0]] = (
                        shape,
                        "BF16" if pair is not None else dtype,
                    )

    target_mtp_keys = {key for key in target_map if _stage_for_name(key) is not None}
    target_keys_by_shard: dict[str, list[str]] = {}
    for key, shard in target_map.items():
        target_keys_by_shard.setdefault(shard, []).append(key)
    new_map: dict[str, str] = {}
    for shard, keys in target_keys_by_shard.items():
        base_keys = [key for key in keys if _stage_for_name(key) is None]
        if not base_keys:
            continue
        src = target_dir / shard
        if not src.is_file():
            raise FileNotFoundError(src)
        mixed = len(base_keys) != len(keys)
        output_shard = _filtered_shard_name(shard) if mixed else shard
        dst = output_dir / output_shard
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mixed:
            if dst.exists() and resume:
                with safe_open(dst, framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != set(base_keys):
                        raise ValueError(f"resume filtered shard has stale keys: {output_shard}")
            elif not dst.exists():
                tensors = {}
                with safe_open(src, framework="pt", device="cpu") as handle:
                    for key in base_keys:
                        tensors[key] = handle.get_tensor(key)
                tmp = dst.with_name(dst.name + ".tmp")
                save_file(tensors, str(tmp))
                os.replace(tmp, dst)
        elif not dst.exists():
            try:
                os.link(src, dst)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot hardlink base shard {src} into staging output; "
                    "use an output directory on the same filesystem"
                ) from exc
        for key in base_keys:
            new_map[key] = output_shard

    for aux in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "tokenizer.model"):
        src = target_dir / aux
        if src.is_file() and not (output_dir / aux).exists():
            shutil.copy2(src, output_dir / aux)

    for stage in range(3):
        out_shard = f"model-mtp-{stage + 1:05d}-of-00003.safetensors"
        out_path = output_dir / out_shard
        stage_items = source_items[stage]
        stage_expected: dict[str, tuple[tuple[int, ...], str]] = {}
        for source_key, _ in stage_items:
            if _is_scale_key(source_key):
                continue
            normalized = _normalize_name(source_key)
            assert normalized is not None
            stage_expected[normalized[0]] = source_expected[normalized[0]]
        if out_path.exists() and resume:
            print(f"[resume] validating {out_shard}", flush=True)
            with safe_open(out_path, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(stage_expected):
                    raise ValueError(f"resume shard keys mismatch: {out_shard}")
                for key, (shape, dtype) in stage_expected.items():
                    view = handle.get_slice(key)
                    if tuple(view.get_shape()) != shape or view.get_dtype() != dtype:
                        raise ValueError(f"resume shard metadata mismatch for {key}")
                    new_map[key] = out_shard
            continue
        print(f"[convert] stage={stage} tensors={len(stage_items)} output={out_shard}", flush=True)
        tensors: dict[str, torch.Tensor] = {}
        by_shard: dict[str, list[str]] = {}
        for source_key, source_shard in stage_items:
            by_shard.setdefault(source_shard, []).append(source_key)
        for source_shard, shard_keys in by_shard.items():
            with safe_open(dspark_dir / source_shard, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                for source_key in shard_keys:
                    if _is_scale_key(source_key):
                        continue
                    normalized = _normalize_name(source_key)
                    assert normalized is not None
                    output_key = normalized[0]
                    pair = _global_paired_scale(source_map, source_key)
                    if source_key not in actual_keys:
                        raise KeyError(f"index key {source_key!r} absent from {source_shard}")
                    tensor = handle.get_tensor(source_key)
                    if pair is not None:
                        if pair not in actual_keys:
                            raise KeyError(f"paired scale {pair!r} absent from {source_shard}")
                        scale = handle.get_tensor(pair)
                        if ".experts." in source_key:
                            tensor = dequant_mxfp4(tensor, scale, chunk_rows)
                        else:
                            tensor = dequant_fp8_block(tensor, scale, (128, 128), chunk_rows)
                    tensors[output_key] = tensor
                    new_map[output_key] = out_shard
        tmp = out_path.with_name(out_path.name + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, out_path)
        print(f"[done] stage={stage} bytes={out_path.stat().st_size}", flush=True)

    stale_target_mtp = {
        key for key in target_mtp_keys
        if new_map.get(key) == target_map.get(key)
    }
    if stale_target_mtp:
        raise ValueError(
            f"stale target mtp mappings remain in output index: "
            f"{sorted(stale_target_mtp)[:3]}"
        )
    output_index = json.loads(json.dumps(target_index))
    output_index["weight_map"] = new_map
    metadata = output_index.setdefault("metadata", {})
    target_total = metadata.get("total_size")
    if not isinstance(target_total, int):
        target_total = _payload_nbytes(target_dir, target_map)
    old_mtp_bytes = _payload_nbytes(target_dir, target_map, target_mtp_keys)
    new_mtp_keys = {key for key in new_map if _stage_for_name(key) is not None}
    new_mtp_bytes = _payload_nbytes(output_dir, new_map, new_mtp_keys)
    metadata["total_size"] = target_total - old_mtp_bytes + new_mtp_bytes
    _write_json_atomic(output_dir / CONFIG_NAME, _rewrite_config(target_cfg, source_cfg))
    _write_json_atomic(output_dir / INDEX_NAME, output_index)
    print(
        f"[complete] output={output_dir} tensors={len(new_map)} "
        f"payload_bytes={metadata['total_size']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dspark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="resume an output directory with completed MTP shards")
    args = parser.parse_args()
    merge_checkpoint(args.target_dir, args.dspark_dir, args.output_dir,
                     resume=args.resume)


if __name__ == "__main__":
    main()
