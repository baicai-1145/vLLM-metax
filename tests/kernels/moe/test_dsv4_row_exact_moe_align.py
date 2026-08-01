import pytest
import torch
import vllm_metax.model_executor.layers.fused_moe.fused_moe as fused_moe_module

from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    _apply_moe_sum,
    _row_exact_moe_config_tokens,
    _use_row_exact_moe_align,
)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
def test_row_exact_moe_align_selects_only_dspark_verifier_rows(monkeypatch, rows):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")

    assert _use_row_exact_moe_align(rows, 6)


@pytest.mark.parametrize("rows", [1, 7, 16])
def test_row_exact_moe_align_rejects_other_row_counts(monkeypatch, rows):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")

    assert not _use_row_exact_moe_align(rows, 6)


def test_row_exact_moe_align_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", raising=False)

    assert not _use_row_exact_moe_align(6, 6)


def test_row_exact_moe_uses_m1_tuning_config(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_MOE_ROW_EXACT_CONFIG_TOKENS", raising=False
    )

    assert _row_exact_moe_config_tokens(6, 6, True) == 1
    assert _row_exact_moe_config_tokens(6, 6, False) == 6


def test_row_exact_moe_uses_m1_tuning_for_every_split_chunk(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")
    monkeypatch.delenv(
        "VLLM_METAX_DSV4_MOE_ROW_EXACT_CONFIG_TOKENS", raising=False
    )

    assert [
        _row_exact_moe_config_tokens(rows, 6, True) for rows in (4, 2)
    ] == [1, 1]


@pytest.mark.parametrize("config_tokens", [1, 2, 4, 8])
def test_row_exact_moe_allows_diagnostic_tuning_config(
    monkeypatch, config_tokens
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_MOE_ROW_EXACT_CONFIG_TOKENS", str(config_tokens)
    )

    assert _row_exact_moe_config_tokens(6, 6, True) == config_tokens


@pytest.mark.parametrize("config_tokens", ["0", "3", "invalid"])
def test_row_exact_moe_rejects_unknown_diagnostic_tuning_config(
    monkeypatch, config_tokens
):
    monkeypatch.setenv("VLLM_METAX_DSV4_MOE_ROW_EXACT_ALIGN", "1")
    monkeypatch.setenv(
        "VLLM_METAX_DSV4_MOE_ROW_EXACT_CONFIG_TOKENS", config_tokens
    )

    with pytest.raises(ValueError, match="must be one of 1, 2, 4, or 8"):
        _row_exact_moe_config_tokens(6, 6, True)


def test_row_exact_moe_sum_preserves_m1_native_launches(monkeypatch):
    calls = []

    def moe_sum(value, output):
        calls.append(value.shape)
        output.copy_(value.sum(dim=1))

    monkeypatch.setattr(fused_moe_module.ops, "moe_sum", moe_sum)
    value = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    output = torch.empty(6, 2)

    _apply_moe_sum(value, output, row_exact=True)

    assert calls == [torch.Size([1, 2, 2])] * 6
    torch.testing.assert_close(output, value.sum(dim=1))
