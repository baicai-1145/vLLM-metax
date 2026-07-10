import inspect

import vllm
import vllm_metax


def test_vllm_is_025rc1_checkout():
    assert getattr(vllm, "__version__", None) == "0.25.0rc1"
    assert "/root/vllm-0.25rc1/" in inspect.getfile(vllm)


def test_vllm_metax_register_functions_import():
    assert vllm_metax.register() == "vllm_metax.platform.MacaPlatform"
    assert callable(vllm_metax.register_customized)
    assert callable(vllm_metax.register_model)
