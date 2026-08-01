from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[2] / "tools" / "merge_deepseek_v4_dspark.py"
SPEC = importlib.util.spec_from_file_location("merge_dspark", SCRIPT)
assert SPEC and SPEC.loader
merge_dspark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_dspark
SPEC.loader.exec_module(merge_dspark)


def _config(*, dspark: bool) -> dict:
    value = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 8,
        "vocab_size": 32,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0, 0, 4],
        "quantization_config": {"ignore": ["re:^layers\\.43\\."]},
    }
    if dspark:
        value.update({
            "dspark_block_size": 5,
            "dspark_noise_token_id": 7,
            "dspark_target_layer_ids": [40, 41, 42],
            "dspark_markov_rank": 256,
            "expert_dtype": "fp4",
            "compress_rope_theta": 160000,
        })
    return value


def _write_checkpoint(path: Path, cfg: dict, tensors: dict[str, torch.Tensor],
                      shard: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(tensors, str(path / shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1},
                    "weight_map": {key: shard for key in tensors}}),
        encoding="utf-8")


def _make_valid_inputs(tmp_path: Path, *, quantized: bool = False,
                       split_scale: bool = False) -> tuple[Path, Path]:
    target = tmp_path / "target"
    dspark = tmp_path / "dspark"
    _write_checkpoint(
        target,
        _config(dspark=False),
        {"embed.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
        "target.safetensors",
    )
    dspark.mkdir()
    source_map: dict[str, str] = {}
    for stage in range(3):
        shard = f"model-{46 + stage:05d}-of-00048.safetensors"
        key = f"mtp.{stage}.e_proj.weight"
        tensors = {key: torch.ones((2, 2), dtype=torch.uint8 if quantized else torch.float32)}
        if quantized and stage == 0 and not split_scale:
            tensors[key + "_scale"] = torch.tensor([[128]], dtype=torch.uint8)
        if quantized and stage == 1 and split_scale:
            tensors["mtp.0.e_proj.weight_scale"] = torch.tensor([[128]], dtype=torch.uint8)
        save_file(tensors, str(dspark / shard))
        source_map.update({name: shard for name in tensors})
    (dspark / "config.json").write_text(json.dumps(_config(dspark=True)), encoding="utf-8")
    (dspark / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": source_map}), encoding="utf-8")
    return target, dspark


def test_mxfp4_and_fp8_dequant_numeric() -> None:
    packed = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    decoded = merge_dspark.dequant_mxfp4(packed, scale)
    assert decoded.tolist()[0] == [0.0, 0.5, 1.0, 1.5]

    block_packed = torch.tensor([[0x54]], dtype=torch.uint8).repeat(384, 1)
    block_scale = torch.tensor([[127], [127], [127]], dtype=torch.uint8)
    block_decoded = merge_dspark.dequant_mxfp4(block_packed, block_scale)
    assert block_decoded.shape == (384, 2)
    assert block_decoded[256].tolist() == [2.0, 3.0]

    weight = torch.tensor([[1.0, 2.0, -1.0, 4.0]], dtype=torch.float32)
    block_scale = torch.tensor([[128, 128]], dtype=torch.uint8)
    fp8 = merge_dspark.dequant_fp8_block(weight, block_scale,
                                         block_shape=(2, 2))
    assert torch.allclose(fp8.float(), weight * 2, atol=0.01)


def test_merge_rewrites_config_index_and_preserves_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    dspark = tmp_path / "dspark"
    output = tmp_path / "output"
    target_tensors = {
        "embed.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "mtp.0.old.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "mtp.0.enorm.weight": torch.zeros((1,), dtype=torch.float32),
    }
    _write_checkpoint(target, _config(dspark=False), target_tensors,
                      "target.safetensors")
    stage_tensors = {
        "mtp.0.ffn.experts.0.w1.weight": torch.tensor([[0x10, 0x32]], dtype=torch.uint8),
        "mtp.0.ffn.experts.0.w1.weight_scale": torch.tensor([[127]], dtype=torch.uint8),
        "mtp.0.e_proj.weight": torch.ones((2, 2), dtype=torch.float32),
        "mtp.0.e_proj.weight_scale": torch.tensor([[128]], dtype=torch.uint8),
        "mtp.0.enorm.weight": torch.tensor([3.0], dtype=torch.float32),
    }
    source_map = {}
    for stage in range(3):
        shard = f"model-{46 + stage:05d}-of-00048.safetensors"
        tensors = {key.replace("mtp.0.", f"mtp.{stage}."): value
                   for key, value in stage_tensors.items()}
        save_file(tensors, str(dspark / shard)) if dspark.exists() else None
        dspark.mkdir(exist_ok=True)
        if not (dspark / shard).exists():
            save_file(tensors, str(dspark / shard))
        source_map.update({key: shard for key in tensors})
    (dspark / "config.json").write_text(json.dumps(_config(dspark=True)), encoding="utf-8")
    (dspark / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": source_map}), encoding="utf-8")

    merge_dspark.merge_checkpoint(target, dspark, output)
    result = json.loads((output / "model.safetensors.index.json").read_text())
    assert "mtp.0.old.weight" not in result["weight_map"]
    assert result["weight_map"]["mtp.0.enorm.weight"].startswith("model-mtp-")
    assert result["weight_map"]["embed.weight"] == "target-base.safetensors"
    assert result["weight_map"]["mtp.2.e_proj.weight"].startswith("model-mtp-")
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["num_nextn_predict_layers"] == 1
    assert cfg["dspark_target_layer_ids"] == [40, 41, 42]
    assert cfg["n_mtp_layers"] == 3
    assert "re:^layers\\.45\\." in cfg["quantization_config"]["ignore"]
    with safe_open(output / "target-base.safetensors", framework="pt", device="cpu") as handle:
        keys = handle.keys()
        assert "mtp.0.old.weight" not in keys
    merge_dspark.merge_checkpoint(target, dspark, output, resume=True)


def test_merge_rejects_missing_stage_and_wrong_source(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_checkpoint(target, _config(dspark=False), {"x": torch.ones(1)}, "t.safetensors")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps(_config(dspark=False)), encoding="utf-8")
    (bad / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="official DSpark"):
        merge_dspark.merge_checkpoint(target, bad, tmp_path / "out")


def test_rejects_cross_shard_and_absent_quant_scale(tmp_path: Path) -> None:
    target, dspark = _make_valid_inputs(tmp_path / "cross", quantized=True,
                                        split_scale=True)
    with pytest.raises(ValueError, match="split across shards"):
        merge_dspark.merge_checkpoint(target, dspark, tmp_path / "cross-out")

    target, dspark = _make_valid_inputs(tmp_path / "missing", quantized=True)
    index_path = dspark / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("mtp.0.e_proj.weight_scale")
    index_path.write_text(json.dumps(index))
    # Remove the scale from the shard as well; the index must not permit a
    # quantized weight to silently pass through as uint8.
    shard = dspark / "model-00046-of-00048.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {
            key: handle.get_tensor(key)
            for key in keys
            if not key.endswith("_scale")
        }
    save_file(tensors, str(shard))
    with pytest.raises(ValueError, match="has no paired scale"):
        merge_dspark.merge_checkpoint(target, dspark, tmp_path / "missing-out")


def test_rejects_output_alias_and_unsafe_shard_path(tmp_path: Path) -> None:
    target, dspark = _make_valid_inputs(tmp_path / "alias")
    with pytest.raises(ValueError, match="must differ"):
        merge_dspark.merge_checkpoint(target, dspark, target)
    with pytest.raises(ValueError, match="must differ"):
        merge_dspark.merge_checkpoint(target, dspark, dspark)

    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["embed.weight"] = "../escape.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unsafe shard path"):
        merge_dspark.merge_checkpoint(target, dspark, tmp_path / "unsafe-out")


def test_resume_rejects_stale_shape_or_dtype(tmp_path: Path) -> None:
    target, dspark = _make_valid_inputs(tmp_path / "resume")
    output = tmp_path / "resume-out"
    merge_dspark.merge_checkpoint(target, dspark, output)
    shard = output / "model-mtp-00001-of-00003.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
    tensors["mtp.0.e_proj.weight"] = torch.zeros((3, 3), dtype=torch.float32)
    save_file(tensors, str(shard))
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_dspark.merge_checkpoint(target, dspark, output, resume=True)


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--target-dir" in result.stdout
