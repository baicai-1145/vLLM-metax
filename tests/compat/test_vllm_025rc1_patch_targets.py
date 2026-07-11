import importlib


def test_patch_package_imports_against_vllm_025rc1():
    importlib.import_module("vllm_metax.patch")


def test_customized_and_model_registration_against_vllm_025rc1():
    import vllm_metax

    vllm_metax.register_customized()
    vllm_metax.register_model()


def test_deepseek_v4_mtp_uses_vllm_025rc1_fused_moe_mapping():
    from vllm_metax.models.deepseek_v4 import mtp

    assert callable(mtp.fused_moe_make_expert_params_mapping)
