"""Offline sparse MLA corpus inspector tests (no MetaX device required)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_CLI_PATH = Path(__file__).parents[3] / "tools/debug/diff_deepseek_v4_sparse_mla.py"
_SPEC = importlib.util.spec_from_file_location("sparse_mla_corpus_cli", _CLI_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def _payload(*, rank: int = 0, call: int = 0) -> dict[str, object]:
    q = torch.tensor(
        [[[1.0, 0.5, -1.0, 0.25]], [[-0.5, 1.5, 0.25, 2.0]]], dtype=torch.bfloat16
    )
    kv = torch.tensor(
        [[[1.0, 0.0, 0.5, 2.0]], [[-1.0, 2.0, 1.0, 0.0]], [[0.5, 1.0, -0.5, 1.5]]],
        dtype=torch.bfloat16,
    )
    indices = torch.tensor([[[0, 2, -1]], [[1, 99, 0]]], dtype=torch.int32)
    topk_length = torch.tensor([2, 1], dtype=torch.int32)
    output, max_logits, lse = cli.torch_oracle(q, kv, indices, 0.125, 3, topk_length)

    def meta(
        tensor: torch.Tensor | None, *, caller: bool = False
    ) -> dict[str, object] | None:
        if tensor is None:
            return None
        result = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "strides": list(tensor.stride()),
        }
        if caller:
            result["data_ptr"] = int(tensor.data_ptr())
        return result

    return {
        "schema_version": 1,
        "stage": "prefill",
        "rank": rank,
        "call": call,
        "q": q,
        "kv": kv,
        "indices": indices,
        "topk_length": topk_length,
        "attn_sink": None,
        "sm_scale": 0.125,
        "d_v": 3,
        "input_meta": {
            "q": meta(q),
            "kv": meta(kv),
            "indices": meta(indices),
            "topk_length": meta(topk_length),
            "attn_sink": None,
        },
        "caller_out": meta(output, caller=True),
        "output": output,
        "max_logits": max_logits,
        "lse": lse,
    }


def _write(tmp_path: Path, payload: dict[str, object], *, rank: int = 0, call: int = 0) -> Path:
    path = tmp_path / f"rank{rank}_call{call}.pt"
    torch.save(payload, path)
    return path


def _decode_payload(*, rank: int = 0, call: int = 0, swa_only: bool = False) -> dict[str, object]:
    q = torch.tensor(
        [[[1.0, 0.0, 0.5, -1.0], [0.5, 1.0, -0.5, 0.25]],
         [[-0.5, 1.5, 0.25, 2.0], [1.0, -1.0, 0.0, 0.5]]],
        dtype=torch.bfloat16,
    )
    # Slice a padded allocation to preserve a non-contiguous physical cache.
    swa_storage = torch.arange(4 * 2 * 1 * 6, dtype=torch.float32).reshape(4, 2, 1, 6)
    swa_cache = swa_storage[..., :4].to(torch.bfloat16)
    swa_indices = torch.tensor([[[3, 0, -1]], [[4, 2, -1]]], dtype=torch.int32)
    swa_lens = torch.tensor([2, 2], dtype=torch.int32)
    swa_table = torch.tensor([[1, 0], [2, 1]], dtype=torch.int32)
    compressed_cache = None
    topk_indices = None
    topk_lens = None
    compressed_table = None
    compressed_block_size = None
    if not swa_only:
        compressed_cache = torch.arange(3 * 2 * 1 * 4, dtype=torch.float32).reshape(
            3, 2, 1, 4
        ).to(torch.bfloat16)
        topk_indices = torch.tensor([[[1, -1]], [[2, 0]]], dtype=torch.int32)
        topk_lens = torch.tensor([1, 2], dtype=torch.int32)
        compressed_table = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
        compressed_block_size = 2
    attn_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
    output = cli.torch_decode_oracle(
        q, swa_cache, swa_indices, topk_indices, 3, 0.125
    )

    def meta(tensor: torch.Tensor | None, *, caller: bool = False):
        if tensor is None:
            return None
        result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                  "strides": list(tensor.stride())}
        if caller:
            result["data_ptr"] = int(tensor.data_ptr())
        return result

    tensors = {
        "q": q, "swa_cache": swa_cache, "compressed_cache": compressed_cache,
        "swa_indices": swa_indices, "topk_indices": topk_indices,
        "swa_lens": swa_lens, "topk_lens": topk_lens, "attn_sink": attn_sink,
        "swa_block_table": swa_table, "compressed_block_table": compressed_table,
    }
    return {
        "schema_version": 1, "stage": "decode", "rank": rank, "call": call,
        **tensors, "sm_scale": 0.125, "d_v": 3, "head_dim": 4, "q_heads": 2,
        "input_meta": {name: meta(value) for name, value in tensors.items()},
        "cache_meta": {"swa_block_size": 2, "compressed_block_size": compressed_block_size,
                       "compress_ratio": None if swa_only else 4, "window_size": 3},
        "caller_out": meta(output, caller=True), "output": output,
    }


def test_numeric_rank_call_order_and_cpu_replay(tmp_path: Path) -> None:
    for call in (10, 2, 1):
        _write(tmp_path, _payload(call=call), call=call)

    files = cli.iter_corpus_files(tmp_path)
    assert [path.name for path in files] == [
        "rank0_call1.pt",
        "rank0_call2.pt",
        "rank0_call10.pt",
    ]
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 3
    assert summary["failed"] == 0
    assert summary["details"][0]["output_max_abs"] == 0.0


def test_corrupt_schema_fails_nonzero_and_reports_json(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("stage")
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="keys="):
        cli.load_payload(tmp_path / "rank0_call0.pt")
    json_out = tmp_path / "summary.json"
    assert cli.main([str(tmp_path), "--json-out", str(json_out)]) == 1
    assert json.loads(json_out.read_text())["failed"] == 1


def test_truncated_payload_fails_nonzero_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "rank0_call0.pt").write_bytes(b"truncated torch archive")
    json_out = tmp_path / "summary.json"
    assert cli.main([str(tmp_path), "--json-out", str(json_out)]) == 1
    summary = json.loads(json_out.read_text())
    assert summary["failed"] == 1
    assert "unable to load corpus payload" in summary["failures"][0]["error"]
    assert "Traceback" not in capsys.readouterr().out


def test_topk_length_range_is_validated(tmp_path: Path) -> None:
    payload = _payload()
    payload["topk_length"] = torch.tensor([4, 1], dtype=torch.int32)
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="topk_length values"):
        cli.load_payload(tmp_path / "rank0_call0.pt")


def test_corrupt_native_output_is_red_capable(tmp_path: Path) -> None:
    payload = _payload()
    output = payload["output"]
    assert isinstance(output, torch.Tensor)
    payload["output"] = output.clone()
    payload["output"][0, 0, 0] += torch.tensor(1.0, dtype=output.dtype)
    _write(tmp_path, payload)
    summary = cli.run_diff(tmp_path, atol=0.0, rtol=0.0)
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert "output mismatch" in summary["failures"][0]["error"]


def test_decode_noncontiguous_physical_cache_replays_and_reports_dual_cache_gap(
    tmp_path: Path,
) -> None:
    payload = _decode_payload()
    _write(tmp_path, payload)
    loaded = cli.load_payload(tmp_path / "rank0_call0.pt")
    assert loaded["swa_cache"].stride()[-1] == 1
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["coverage"]["by_stage"] == {"decode": 1}
    assert "decode_reference_uses_swa_cache_for_topk_instead_of_compressed_cache" in summary[
        "semantic_gaps"
    ]


def test_decode_swa_only_has_no_topk_cache_gap(tmp_path: Path) -> None:
    _write(tmp_path, _decode_payload(swa_only=True))
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 1
    assert "decode_reference_uses_swa_cache_for_topk_instead_of_compressed_cache" not in summary[
        "semantic_gaps"
    ]


def test_decode_invalid_physical_index_is_red(tmp_path: Path) -> None:
    payload = _decode_payload(swa_only=True)
    indices = payload["swa_indices"]
    assert isinstance(indices, torch.Tensor)
    payload["swa_indices"] = indices.clone()
    payload["swa_indices"][0, 0, 0] = 99
    _write(tmp_path, payload)
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert "out-of-range" in summary["failures"][0]["error"]


def test_decode_numeric_order_and_stage_rank_coverage(tmp_path: Path) -> None:
    _write(tmp_path, _decode_payload(rank=2, call=10), rank=2, call=10)
    _write(tmp_path, _decode_payload(rank=1, call=2), rank=1, call=2)
    _write(tmp_path, _decode_payload(rank=1, call=1), rank=1, call=1)
    files = cli.iter_corpus_files(tmp_path)
    assert [path.name for path in files] == [
        "rank1_call1.pt", "rank1_call2.pt", "rank2_call10.pt"
    ]
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 3
    assert summary["coverage"]["by_stage_rank"] == {"decode/rank1": 2, "decode/rank2": 1}


def test_decode_corrupt_output_is_red_capable(tmp_path: Path) -> None:
    payload = _decode_payload()
    output = payload["output"]
    assert isinstance(output, torch.Tensor)
    payload["output"] = output.clone()
    payload["output"][0, 0, 0] += torch.tensor(1.0, dtype=output.dtype)
    _write(tmp_path, payload)
    summary = cli.run_diff(tmp_path, atol=0.0, rtol=0.0)
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert "output mismatch" in summary["failures"][0]["error"]


def test_attention_sink_is_replayed_and_validated(tmp_path: Path) -> None:
    payload = _payload()
    sink = torch.tensor([0.5], dtype=torch.float32)
    payload["attn_sink"] = sink
    payload["input_meta"]["attn_sink"] = {
        "shape": list(sink.shape),
        "dtype": str(sink.dtype),
        "strides": list(sink.stride()),
    }
    payload["output"], payload["max_logits"], payload["lse"] = cli.torch_oracle(
        payload["q"], payload["kv"], payload["indices"], payload["sm_scale"],
        payload["d_v"], payload["topk_length"], sink,
    )
    _write(tmp_path, payload)
    loaded = cli.load_payload(tmp_path / "rank0_call0.pt")
    assert torch.equal(loaded["attn_sink"], sink)
    summary = cli.run_diff(tmp_path)
    assert summary["passed"] == 1
    assert summary["failed"] == 0


def test_attention_sink_validation_rejects_wrong_dtype_or_shape(tmp_path: Path) -> None:
    for sink in (torch.tensor([0.0], dtype=torch.bfloat16), torch.tensor([0.0, 1.0])):
        payload = _payload()
        payload["attn_sink"] = sink
        _write(tmp_path, payload)
        with pytest.raises(ValueError, match="attn_sink"):
            cli.load_payload(tmp_path / "rank0_call0.pt")
        (tmp_path / "rank0_call0.pt").unlink()
