import json
import runpy
from pathlib import Path

import pytest
import torch

from vllm_metax.models.deepseek_v4.ops.mhc import debug_diff


EXPECTED_TRACE_KEYS = {
    "sqrsum_reduced",
    "rms",
    "normalized_mixes",
    "affine_logits",
    "pre_mix",
    "post_mix",
    "comb_logits",
    "sinkhorn_softmax_eps",
    *(f"sinkhorn_row_{index}" for index in range(1, 20)),
    *(f"sinkhorn_col_{index}" for index in range(20)),
    "layer_products",
    "layer_input_fp32",
    "layer_input_bf16",
}


def _raw_inputs(hidden_size: int = 4096):
    torch.manual_seed(0)
    residual_cur = torch.randn(1, 4, hidden_size, dtype=torch.float32).bfloat16()
    gemm_out_mul = torch.randn(1, 1, 24, dtype=torch.float32)
    gemm_out_sqrsum = torch.rand(1, 1, dtype=torch.float32) + 1.0
    hc_scale = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32)
    hc_base = torch.linspace(-0.75, 0.75, 24, dtype=torch.float32)
    return residual_cur, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base


def _trace_kwargs():
    return {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
    }


def _make_payload(call: int = 0, hidden_size: int = 4096):
    residual_cur, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base = _raw_inputs(
        hidden_size
    )
    return {
        "schema_version": 1,
        "rank": 0,
        "call": call,
        "residual_cur": residual_cur,
        "gemm_out_mul": gemm_out_mul,
        "gemm_out_sqrsum": gemm_out_sqrsum,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "params": {**_trace_kwargs(), "n_splits": 1},
    }


def _load_cli():
    path = Path("tools/debug/diff_deepseek_v4_mhc_raw.py")
    return runpy.run_path(str(path))


def test_torch_trace_keys_and_final_output_equivalence():
    residual_cur, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base = _raw_inputs()

    trace = debug_diff.mhc_pre_from_raw_trace_torch(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        **_trace_kwargs(),
    )

    assert set(trace) == EXPECTED_TRACE_KEYS

    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import (
        _mhc_pre_from_raw_torch,
    )

    post_mix, comb_mix, layer_input = _mhc_pre_from_raw_torch(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        **_trace_kwargs(),
    )
    assert torch.equal(trace["post_mix"], post_mix)
    assert torch.equal(trace["sinkhorn_col_19"], comb_mix)
    assert torch.equal(trace["layer_input_bf16"], layer_input)


def test_tensor_diff_reports_fp32_and_bf16_raw_mismatch():
    lhs = torch.tensor([1.0, 2.0], dtype=torch.float32)
    rhs = torch.tensor([1.0, 2.000000238418579], dtype=torch.float32)
    diff = debug_diff.tensor_diff(lhs, rhs)
    assert not diff["equal"]
    assert diff["index"] == [1]
    assert diff["num_diff"] == 1
    assert diff["ulp"] != 0
    assert diff["max_ulp"] >= abs(diff["ulp"])

    bf16_lhs = torch.tensor([1.0], dtype=torch.bfloat16)
    bf16_rhs = torch.tensor([1.0078125], dtype=torch.bfloat16)
    bf16_diff = debug_diff.tensor_diff(bf16_lhs, bf16_rhs)
    assert not bf16_diff["equal"]
    assert bf16_diff["lhs_bits"] != bf16_diff["rhs_bits"]


def test_exact_post_contract_rejects_non_decode_shape():
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
        _mhc_post_exact_tl,
    )

    x = torch.zeros((1, 8), dtype=torch.bfloat16)
    residual = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    post_mix = torch.zeros((1, 4), dtype=torch.float32)
    comb_mix = torch.zeros((1, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"x BF16\[1,4096\]"):
        _mhc_post_exact_tl(x, residual, post_mix, comb_mix)


def test_assert_bitwise_trace_equal_rejects_stage_mismatch():
    reference = {"stage": torch.tensor([1.0], dtype=torch.float32)}
    candidate = {"stage": torch.tensor([2.0], dtype=torch.float32)}
    with pytest.raises(AssertionError, match="stage"):
        debug_diff.assert_bitwise_trace_equal(reference, candidate)


def test_disabled_capture_is_true_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_DIR", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("capture path should be disabled")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(torch, "save", fail)
    monkeypatch.setattr(Path, "mkdir", fail)

    debug_diff.maybe_capture_mhc_pre_raw(
        *_raw_inputs(hidden_size=8),
        **_trace_kwargs(),
        n_splits=1,
    )


def test_enabled_capture_schema_and_call_filtering(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_RANKS", "0")
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_MAX_CALLS", "2")
    monkeypatch.setattr(debug_diff, "_rank", lambda: "0")
    debug_diff.reset_mhc_pre_raw_capture_state()

    for _ in range(3):
        debug_diff.maybe_capture_mhc_pre_raw(
            *_raw_inputs(hidden_size=8),
            **_trace_kwargs(),
            n_splits=1,
        )

    files = sorted(tmp_path.glob("rank0_call*.pt"))
    assert [path.name for path in files] == ["rank0_call0.pt", "rank0_call1.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["rank"] == 0
    assert payload["call"] == 0
    assert set(payload) == {
        "schema_version",
        "rank",
        "call",
        "residual_cur",
        "gemm_out_mul",
        "gemm_out_sqrsum",
        "hc_scale",
        "hc_base",
        "params",
    }
    assert payload["params"] == {**_trace_kwargs(), "n_splits": 1}


def test_corpus_ordering_is_numeric(tmp_path):
    cli = _load_cli()
    for call in (10, 2, 1):
        torch.save(_make_payload(call=call, hidden_size=8), tmp_path / f"rank0_call{call}.pt")

    files = cli["iter_corpus_files"](tmp_path)
    assert [path.name for path in files] == [
        "rank0_call1.pt",
        "rank0_call2.pt",
        "rank0_call10.pt",
    ]


def test_malformed_schema_rejection_includes_path(tmp_path):
    cli = _load_cli()
    path = tmp_path / "rank0_call0.pt"
    payload = _make_payload(hidden_size=8)
    payload["params"]["sinkhorn_repeat"] = 19
    torch.save(payload, path)

    with pytest.raises(ValueError, match=str(path)):
        cli["load_payload"](path, device="cpu")


def test_torch_selfcheck_summary(tmp_path):
    cli = _load_cli()
    for rank in (0, 1):
        for call in (0, 1):
            payload = _make_payload(call=call, hidden_size=8)
            payload["rank"] = rank
            torch.save(payload, tmp_path / f"rank{rank}_call{call}.pt")

    summary = cli["run_diff"](
        tmp_path,
        candidate="torch",
        device="cpu",
        require_bitwise=True,
    )

    assert summary == {
        "files": 4,
        "passed": 4,
        "failed": 0,
        "first_failure": None,
        "stage_failures": {},
        "bitwise": True,
    }


def test_cli_writes_json_summary(tmp_path):
    cli = _load_cli()
    payload_path = tmp_path / "rank0_call0.pt"
    json_path = tmp_path / "summary.json"
    torch.save(_make_payload(hidden_size=8), payload_path)

    exit_code = cli["main"](
        [
            str(tmp_path),
            "--candidate",
            "torch",
            "--device",
            "cpu",
            "--require-bitwise",
            "--json-out",
            str(json_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(json_path.read_text())["passed"] == 1


def test_exact_decode_contract_selector():
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import (
        _is_exact_mhc_decode_contract,
    )

    assert _is_exact_mhc_decode_contract(
        num_tokens=1,
        hc_mult=4,
        hidden_size=4096,
        n_splits=1,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    assert not _is_exact_mhc_decode_contract(
        num_tokens=2,
        hc_mult=4,
        hidden_size=4096,
        n_splits=1,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    assert not _is_exact_mhc_decode_contract(
        num_tokens=1,
        hc_mult=4,
        hidden_size=4096,
        n_splits=1,
        rms_eps=1e-5,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
