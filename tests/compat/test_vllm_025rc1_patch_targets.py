import importlib


def test_patch_package_imports_against_vllm_025rc1():
    importlib.import_module("vllm_metax.patch")


def test_customized_and_model_registration_against_vllm_025rc1():
    import vllm_metax

    vllm_metax.register_customized()
    vllm_metax.register_model()
