from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.debug.diff_deepseek_v4_sparse_mla import (  # noqa: E402
    diff_decode_pair,
    load_payload,
)


def _meta(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.stride()),
    }


def _decode_payload(*, output_delta: float = 0.0) -> dict[str, object]:
    q = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)
    swa_cache = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 1, 4)
    swa_cache[0, 0, 0, 0] = float("nan")
    compressed_cache = (torch.arange(16, dtype=torch.bfloat16) + 10).reshape(1, 4, 1, 4)
    swa_indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    topk_indices = torch.tensor([[[2, 3]]], dtype=torch.int32)
    swa_lens = torch.tensor([2], dtype=torch.int32)
    topk_lens = torch.tensor([2], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    attn_sink = torch.zeros(2, dtype=torch.float32)
    output = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    if output_delta:
        output[0, 0, 0] = output_delta
    positions = torch.tensor([767], dtype=torch.int64)
    token_indices = torch.tensor([0], dtype=torch.int64)
    token_to_req = torch.tensor([0], dtype=torch.int32)
    return {
        "schema_version": 1,
        "stage": "decode",
        "decode_mode": "dual",
        "decode_backend": "native",
        "native_decode_mode": "topk",
        "rank": 0,
        "layer_idx": 0,
        "call": 0,
        "observed_call": 17,
        "positions": positions,
        "token_indices": token_indices,
        "token_to_req": token_to_req,
        "q": q,
        "swa_cache": swa_cache,
        "compressed_cache": compressed_cache,
        "swa_indices": swa_indices,
        "topk_indices": topk_indices,
        "swa_lens": swa_lens,
        "topk_lens": topk_lens,
        "swa_block_table": block_table,
        "compressed_block_table": block_table,
        "sm_scale": 0.125,
        "d_v": 4,
        "head_dim": 4,
        "q_heads": 2,
        "attn_sink": attn_sink,
        "input_meta": {
            "q": _meta(q),
            "swa_cache": _meta(swa_cache),
            "compressed_cache": _meta(compressed_cache),
            "swa_indices": _meta(swa_indices),
            "topk_indices": _meta(topk_indices),
            "swa_lens": _meta(swa_lens),
            "topk_lens": _meta(topk_lens),
            "attn_sink": _meta(attn_sink),
            "positions": _meta(positions),
            "token_to_req": _meta(token_to_req),
            "swa_block_table": _meta(block_table),
            "compressed_block_table": _meta(block_table),
        },
        "cache_meta": {
            "swa_block_size": 4,
            "compressed_block_size": 4,
            "compress_ratio": 4,
            "window_size": 128,
        },
        "caller_out": {
            **_meta(output),
            "data_ptr": int(output.data_ptr()),
        },
        "output": output,
    }


def _remapped_payload() -> dict[str, object]:
    payload = _decode_payload()
    swa_cache = torch.empty((2, 4, 1, 4), dtype=torch.bfloat16)
    swa_cache[0] = payload["swa_cache"]
    swa_cache[1] = payload["swa_cache"]
    payload["swa_cache"] = swa_cache
    payload["swa_indices"] = torch.tensor([[[4, 5]]], dtype=torch.int32)
    payload["swa_block_table"] = torch.tensor([[1]], dtype=torch.int32)
    payload["input_meta"]["swa_cache"] = _meta(swa_cache)
    payload["input_meta"]["swa_indices"] = _meta(payload["swa_indices"])
    payload["input_meta"]["swa_block_table"] = _meta(payload["swa_block_table"])
    payload["cache_meta"]["swa_block_size"] = 4
    return payload


def _save(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def test_load_payload_accepts_current_decode_capture_schema(tmp_path):
    path = _save(tmp_path / "rank0_call0.pt", _decode_payload())

    payload = load_payload(path)

    assert payload["stage"] == "decode"
    assert payload["positions"].tolist() == [767]
    assert payload["input_meta"]["positions"]["shape"] == [1]


def test_diff_decode_pair_reports_exact_match(tmp_path):
    base = _save(tmp_path / "base" / "rank0_call0.pt", _decode_payload())
    candidate = _save(tmp_path / "candidate" / "rank0_call0.pt", _decode_payload())

    result = diff_decode_pair(base, candidate)

    assert result["decision"] == "pass"
    assert result["first_difference"] is None
    assert result["first_tensor_difference"] is None
    assert result["first_attention_input_difference"] is None


def test_diff_decode_pair_reports_first_tensor_difference(tmp_path):
    base = _save(tmp_path / "base" / "rank0_call0.pt", _decode_payload())
    candidate = _save(
        tmp_path / "candidate" / "rank0_call0.pt",
        _decode_payload(output_delta=1.0),
    )

    result = diff_decode_pair(base, candidate)

    assert result["decision"] == "fail"
    assert result["first_difference"] == "output"
    assert result["first_tensor_difference"] == "output"
    assert result["first_attention_input_difference"] == "output"
    output_diff = next(item for item in result["comparisons"] if item["key"] == "output")
    assert output_diff["num_diff"] == 1
    assert output_diff["max_abs"] == 1.0
    assert output_diff["first_diff_index"] == [0, 0, 0]
    assert output_diff["base_value"] == 0.0
    assert output_diff["candidate_value"] == 1.0


def test_diff_decode_pair_separates_physical_indices_from_gathered_rows(tmp_path):
    base = _save(tmp_path / "base" / "rank0_call0.pt", _decode_payload())
    candidate = _save(tmp_path / "candidate" / "rank0_call0.pt", _remapped_payload())

    result = diff_decode_pair(base, candidate)

    assert result["decision"] == "fail"
    assert result["first_attention_input_difference"] == "swa_indices"
    gathered = next(
        item for item in result["comparisons"] if item["key"] == "gathered_swa_rows"
    )
    assert gathered["exact"]
    assert gathered["num_diff"] == 0
