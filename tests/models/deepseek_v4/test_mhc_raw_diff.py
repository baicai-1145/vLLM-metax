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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible device"
)
def test_mhc_downstream_rms_uses_round_to_nearest_sigmoid_division():
    import vllm_metax._metax_sparse_C  # noqa: F401

    pre_logits = torch.tensor(
        [-13.204452514648438, -13.204453468322754, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32,
    )
    post_logits = torch.tensor(
        [
            -13.204547882080078,
            13.822408676147461,
            -4.442898273468018,
            16.341108322143555,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    op = torch.ops._metax_sparse_C

    pre_probe = torch.empty(16, device="cuda", dtype=torch.float32)
    post_probe = torch.empty_like(pre_probe)
    op.mhc_sigmoid_probe_out(pre_logits, pre_probe)
    op.mhc_sigmoid_probe_out(post_logits, post_probe)

    residual = torch.zeros((1, 4, 4096), device="cuda", dtype=torch.bfloat16)
    residual[:, 0].fill_(1)
    residual[:, 1].fill_(-1)
    gemm_out = torch.zeros(24, device="cuda", dtype=torch.float32)
    gemm_out[:4].copy_(pre_logits)
    gemm_out[4:8].copy_(post_logits)
    sqrsum = torch.zeros(1, device="cuda", dtype=torch.float32)
    scale = torch.tensor([1.0, 1.0, 0.0], device="cuda", dtype=torch.float32)
    base = torch.zeros(24, device="cuda", dtype=torch.float32)
    norm_weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    post_out = torch.empty(4, device="cuda", dtype=torch.float32)
    comb_out = torch.empty(16, device="cuda", dtype=torch.float32)
    pre_norm_out = torch.empty(4096, device="cuda", dtype=torch.bfloat16)
    norm_out = torch.empty_like(pre_norm_out)
    op.mhc_downstream_rms_out(
        residual,
        gemm_out,
        sqrsum,
        scale,
        base,
        norm_weight,
        post_out,
        comb_out,
        pre_norm_out,
        norm_out,
        1.0,
        0.0,
        1e-6,
        1.0,
        20,
    )
    torch.cuda.synchronize()

    expected_pre = (
        torch.sigmoid(pre_logits[0]) - torch.sigmoid(pre_logits[1])
    ).bfloat16()
    rcpf_pre = (pre_probe[8] - pre_probe[9]).bfloat16()
    expected_post = torch.sigmoid(post_logits)
    assert rcpf_pre != expected_pre
    assert torch.any(post_probe[8:12] != expected_post)
    assert torch.equal(pre_norm_out, expected_pre.expand_as(pre_norm_out))
    assert torch.equal(post_out, expected_post)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible device"
)
def test_mhc_downstream_rms_matches_torch_reference_non_symmetric_sinkhorn():
    import vllm_metax._metax_sparse_C  # noqa: F401

    from vllm_metax.models.deepseek_v4.ops.mhc import debug_diff

    torch.manual_seed(123)
    residual = torch.randn(1, 4, 4096, device="cuda", dtype=torch.float32).bfloat16()
    gemm_out = torch.randn(24, device="cuda", dtype=torch.float32)
    sqrsum = residual.float().square().sum().reshape(1)
    scale = torch.tensor([0.25, -0.5, 0.375], device="cuda", dtype=torch.float32)
    base = torch.linspace(-0.75, 0.75, 24, device="cuda", dtype=torch.float32)
    base[8:24] = torch.tensor(
        [
            1.7,
            -0.4,
            0.2,
            -1.1,
            -0.3,
            1.2,
            -0.8,
            0.6,
            0.9,
            -1.5,
            0.4,
            0.1,
            -0.7,
            0.8,
            1.4,
            -0.2,
        ],
        device="cuda",
    )
    norm_weight = torch.randn(4096, device="cuda", dtype=torch.float32).bfloat16()
    post_out = torch.empty(4, device="cuda", dtype=torch.float32)
    comb_out = torch.empty(16, device="cuda", dtype=torch.float32)
    pre_norm_out = torch.empty(4096, device="cuda", dtype=torch.bfloat16)
    norm_out = torch.empty_like(pre_norm_out)
    params = _trace_kwargs()

    op = torch.ops._metax_sparse_C
    op.mhc_downstream_rms_out(
        residual,
        gemm_out,
        sqrsum,
        scale,
        base,
        norm_weight,
        post_out,
        comb_out,
        pre_norm_out,
        norm_out,
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
    )
    torch.cuda.synchronize()

    trace = debug_diff.mhc_pre_from_raw_trace_torch(
        residual,
        gemm_out.view(1, 1, 24),
        sqrsum.view(1, 1),
        scale,
        base,
        **params,
    )
    reference_comb = trace["sinkhorn_col_19"].reshape(-1)
    reference_pre_norm = trace["layer_input_bf16"].reshape(-1)
    reference_inverse_rms = torch.rsqrt(
        reference_pre_norm.float().square().sum() / 4096 + params["rms_eps"]
    )
    reference_norm = (
        (reference_pre_norm.float() * reference_inverse_rms).bfloat16().float()
        * norm_weight.float()
    ).bfloat16()

    assert torch.equal(comb_out, reference_comb)
    assert torch.equal(pre_norm_out, reference_pre_norm)
    assert torch.equal(norm_out, reference_norm)


def test_first_trace_failure_accepts_native_output_subset():
    cli = _load_cli()
    reference = {
        "intermediate": torch.tensor([1.0]),
        "post_mix": torch.tensor([2.0]),
    }

    stage, diff = cli["_first_trace_failure"](
        reference, {"post_mix": torch.tensor([2.0])}
    )

    assert stage is None
    assert diff is None


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


def test_exact_post_mma_opt_in_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_MHC_EXACT_POST_MMA", raising=False)
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import (
        _exact_post_mma_enabled,
    )

    assert not _exact_post_mma_enabled()


def test_exact_post_mma_keeps_non_decode_shape_explicitly_out_of_scope(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_EXACT_POST_MMA", "1")
    from vllm_metax.models.deepseek_v4.ops.mhc import tilelang

    x = torch.zeros((2, 4096), dtype=torch.bfloat16)
    residual = torch.zeros((2, 4, 4096), dtype=torch.bfloat16)
    post_mix = torch.zeros((2, 4, 1), dtype=torch.float32)
    comb_mix = torch.zeros((2, 4, 4), dtype=torch.float32)
    sentinel = object()
    monkeypatch.setattr(tilelang, "mhc_post_fwd", lambda *args, **kwargs: sentinel)
    assert tilelang.mhc_post_tilelang(x, residual, post_mix, comb_mix) is sentinel


def test_exact_post_pre_rms_fake_shapes():
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import (
        _mhc_exact_post_pre_rms_fake,
    )

    residual = torch.empty((1, 4, 4096), dtype=torch.bfloat16)
    outputs = _mhc_exact_post_pre_rms_fake(
        torch.empty((1, 4096), dtype=torch.bfloat16),
        residual,
        torch.empty((1, 4, 1), dtype=torch.float32),
        torch.empty((1, 4, 4), dtype=torch.float32),
        torch.empty((24, 16384), dtype=torch.float32),
        torch.empty(3, dtype=torch.float32),
        torch.empty(24, dtype=torch.float32),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        torch.empty(4096, dtype=torch.bfloat16),
    )

    assert [tuple(output.shape) for output in outputs] == [
        (1, 4, 4096),
        (1, 4, 1),
        (1, 4, 4),
        (1, 4096),
        (1, 4096),
    ]
    assert [output.dtype for output in outputs] == [
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
    ]


def test_exact_post_pre_rms_rejects_unsupported_contract():
    from vllm_metax.models.deepseek_v4.ops.mhc.tilelang import (
        mhc_exact_post_pre_rms,
    )

    with pytest.raises(RuntimeError, match="exact decode contract"):
        mhc_exact_post_pre_rms(
            torch.empty((2, 4096), dtype=torch.bfloat16),
            torch.empty((2, 4, 4096), dtype=torch.bfloat16),
            torch.empty((2, 4, 1), dtype=torch.float32),
            torch.empty((2, 4, 4), dtype=torch.float32),
            torch.empty((24, 16384), dtype=torch.float32),
            torch.empty(3, dtype=torch.float32),
            torch.empty(24, dtype=torch.float32),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            torch.empty(4096, dtype=torch.bfloat16),
        )


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


def test_disabled_raw_norm_capture_is_true_noop(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_MHC_RAW_NORM_CAPTURE_DIR", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("raw norm capture should be disabled")

    monkeypatch.setattr(torch.Tensor, "float", fail)
    debug_diff.maybe_capture_mhc_raw_norm(
        layer_idx=0,
        stage="attn",
        residual_cur=torch.empty(1, 4, 8, dtype=torch.bfloat16),
        fn=torch.empty(24, 32),
        hc_scale=torch.empty(3),
        hc_base=torch.empty(24),
        pre_norm_output=torch.empty(1, 8, dtype=torch.bfloat16),
        norm_weight=torch.empty(8, dtype=torch.bfloat16),
        normalized_output=torch.empty(1, 8, dtype=torch.bfloat16),
        **_trace_kwargs(),
        n_splits=1,
    )


def test_raw_norm_capture_schema_and_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_RAW_NORM_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_RAW_CAPTURE_RANKS", "0")
    monkeypatch.setattr(debug_diff, "_rank", lambda: "0")
    debug_diff.reset_mhc_raw_norm_capture_state()
    residual_cur = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
    fn = torch.randn(24, 16384, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)
    pre_norm = torch.randn(1, 4096, dtype=torch.bfloat16)
    norm_weight = torch.randn(4096, dtype=torch.bfloat16)
    eps = 1e-6
    normalized = (
        pre_norm.float()
        * torch.rsqrt(pre_norm.float().square().mean(-1, keepdim=True) + eps)
        * norm_weight.float()
    ).bfloat16()

    debug_diff.maybe_capture_mhc_raw_norm(
        layer_idx=7,
        stage="ffn",
        residual_cur=residual_cur,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        pre_norm_output=pre_norm,
        norm_weight=norm_weight,
        normalized_output=normalized,
        **_trace_kwargs(),
        n_splits=1,
    )

    payload = torch.load(
        tmp_path / "rank0_call0.pt", map_location="cpu", weights_only=True
    )
    assert payload["schema_version"] == 2
    assert payload["layer_idx"] == 7
    assert payload["stage"] == "ffn"
    torch.testing.assert_close(payload["normalized_output"], normalized)
    assert "trace" not in payload


def test_corpus_ordering_is_numeric(tmp_path):
    cli = _load_cli()
    for call in (10, 2, 1):
        torch.save(
            _make_payload(call=call, hidden_size=8), tmp_path / f"rank0_call{call}.pt"
        )

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
        "raw_norm_files": 0,
        "raw_norm_passed": 0,
    }


def test_schema_v2_raw_norm_selfcheck_and_corruption(tmp_path):
    cli = _load_cli()
    residual_cur = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
    fn = torch.randn(24, 16384, dtype=torch.float32)
    residual_2d = residual_cur.view(1, -1).float()
    gemm_out_mul = torch.nn.functional.linear(residual_2d, fn).view(1, 1, 24)
    gemm_out_sqrsum = residual_2d.square().sum(-1).view(1, 1)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)
    trace = debug_diff.mhc_pre_from_raw_trace_torch(
        residual_cur,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        **_trace_kwargs(),
    )
    pre_norm = trace["layer_input_bf16"]
    norm_weight = torch.randn(4096, dtype=torch.bfloat16)
    normalized = (
        pre_norm.float()
        * torch.rsqrt(pre_norm.float().square().mean(-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    ).bfloat16()
    payload = {
        "schema_version": 2,
        "rank": 0,
        "call": 0,
        "layer_idx": 1,
        "stage": "attn",
        "residual_cur": residual_cur,
        "fn": fn,
        "gemm_out_mul": gemm_out_mul,
        "gemm_out_sqrsum": gemm_out_sqrsum,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "pre_norm_output": pre_norm,
        "norm_weight": norm_weight,
        "normalized_output": normalized,
        "params": {**_trace_kwargs(), "n_splits": 1},
    }
    path = tmp_path / "rank0_call0.pt"
    torch.save(payload, path)
    summary = cli["run_diff"](
        tmp_path,
        candidate="torch",
        device="cpu",
        require_bitwise=True,
        rms_candidate="torch",
    )
    assert summary["raw_norm_passed"] == 1
    assert summary["failed"] == 0

    payload["normalized_output"] = normalized.clone()
    payload["normalized_output"].view(torch.int16)[0] += 1
    torch.save(payload, path)
    summary = cli["run_diff"](
        tmp_path,
        candidate="torch",
        device="cpu",
        require_bitwise=True,
        rms_candidate="torch",
    )
    assert summary["failed"] == 1
    assert summary["first_failure"]["stage"] == "raw_norm_replay"


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
