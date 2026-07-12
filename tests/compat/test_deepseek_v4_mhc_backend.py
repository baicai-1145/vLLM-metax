import importlib
import importlib.util
from types import SimpleNamespace
import pytest


def _reload_backend(monkeypatch, value):
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_BACKEND', value)
    mod = importlib.import_module('vllm_metax.models.deepseek_v4.ops.mhc.backend')
    return importlib.reload(mod)


def test_default_backend_is_torch(monkeypatch):
    mod = _reload_backend(monkeypatch, 'torch')
    assert mod.get_mhc_backend_name() == 'torch'


def test_tilelang_backend_resolution_matches_runtime(monkeypatch):
    monkeypatch.delenv('VLLM_METAX_DSV4_MHC_TILELANG_OPS', raising=False)
    mod = _reload_backend(monkeypatch, 'tilelang')
    has_tilelang = importlib.util.find_spec('tilelang') is not None
    try:
        deep_gemm = importlib.import_module('deep_gemm')
        has_deep_gemm = getattr(deep_gemm, 'tf32_hc_prenorm_gemm', None) is not None
    except Exception:
        has_deep_gemm = False
    expected = 'tilelang' if has_tilelang and has_deep_gemm else 'torch'
    assert mod.get_mhc_backend_name() == expected
    if expected == 'tilelang':
        torch_mod = importlib.import_module(
            'vllm_metax.models.deepseek_v4.ops.mhc.torch'
        )
        tilelang_mod = importlib.import_module(
            'vllm_metax.models.deepseek_v4.ops.mhc.tilelang'
        )
        assert mod.mhc_pre is torch_mod.mhc_pre
        assert mod.mhc_post is torch_mod.mhc_post
        assert mod.hc_head_fused_kernel is torch_mod.hc_head_fused_kernel
        assert mod.mhc_fused_post_pre is tilelang_mod.mhc_fused_post_pre_tilelang


def test_tilelang_backend_falls_back_when_deep_gemm_symbol_is_missing(
    monkeypatch,
):
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == 'deep_gemm':
            return SimpleNamespace()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', import_module)
    mod = _reload_backend(monkeypatch, 'tilelang')
    torch_mod = original_import_module(
        'vllm_metax.models.deepseek_v4.ops.mhc.torch'
    )
    assert mod.get_mhc_backend_name() == 'torch'
    assert mod.mhc_pre is torch_mod.mhc_pre
    assert mod.mhc_post is torch_mod.mhc_post
    assert mod.mhc_fused_post_pre is torch_mod.mhc_fused_post_pre
    assert mod.hc_head_fused_kernel is torch_mod.hc_head_fused_kernel


def test_tilelang_backend_raises_when_required_and_deep_gemm_symbol_is_missing(
    monkeypatch,
):
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == 'deep_gemm':
            return SimpleNamespace()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', import_module)
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_REQUIRE_EXACT_TILELANG', '1')
    with pytest.raises(RuntimeError, match='REQUIRE_EXACT_TILELANG'):
        _reload_backend(monkeypatch, 'tilelang')


def test_tilelang_backend_can_keep_final_ops_on_torch(monkeypatch):
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_TILELANG_OPS', 'pre,fused')
    mod = _reload_backend(monkeypatch, 'tilelang')
    if mod.get_mhc_backend_name() != 'tilelang':
        return
    torch_mod = importlib.import_module('vllm_metax.models.deepseek_v4.ops.mhc.torch')
    tilelang_mod = importlib.import_module(
        'vllm_metax.models.deepseek_v4.ops.mhc.tilelang'
    )
    assert mod.mhc_pre is tilelang_mod.mhc_pre_tilelang
    assert mod.mhc_fused_post_pre is tilelang_mod.mhc_fused_post_pre_tilelang
    assert mod.mhc_post is torch_mod.mhc_post
    assert mod.hc_head_fused_kernel is torch_mod.hc_head_fused_kernel


def test_exact_post_mma_requires_tilelang_backend(monkeypatch):
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_EXACT_POST_MMA', '1')
    with pytest.raises(RuntimeError, match='EXACT_POST_MMA'):
        _reload_backend(monkeypatch, 'torch')


def test_exact_post_mma_selects_tilelang_post_when_available(monkeypatch):
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_EXACT_POST_MMA', '1')
    monkeypatch.setenv('VLLM_METAX_DSV4_MHC_TILELANG_OPS', 'fused')
    mod = _reload_backend(monkeypatch, 'tilelang')
    if mod.get_mhc_backend_name() != 'tilelang':
        pytest.skip('TileLang/DeepGEMM runtime is unavailable')
    tilelang_mod = importlib.import_module(
        'vllm_metax.models.deepseek_v4.ops.mhc.tilelang'
    )
    assert mod.mhc_post is tilelang_mod.mhc_post_tilelang
