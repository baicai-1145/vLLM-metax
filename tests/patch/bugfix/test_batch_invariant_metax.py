import inspect
from types import SimpleNamespace

from vllm.model_executor.layers import batch_invariant
from vllm_metax.patch.bugfix import batch_invariant_metax


def test_set_fp32_precision_skips_missing_backend_component():
    backend = SimpleNamespace()

    batch_invariant_metax._set_fp32_precision_if_available(backend, "conv")


def test_set_fp32_precision_updates_available_backend_component():
    component = SimpleNamespace(fp32_precision="tf32")
    backend = SimpleNamespace(conv=component)

    batch_invariant_metax._set_fp32_precision_if_available(backend, "conv")

    assert component.fp32_precision == "ieee"


def test_metax_matmul_kernel_override_is_installed_without_flatten():
    kernel = batch_invariant.matmul_kernel_persistent

    assert kernel is batch_invariant_metax.matmul_kernel_persistent_metax
    assert "flatten=" not in inspect.getsource(kernel.fn)
