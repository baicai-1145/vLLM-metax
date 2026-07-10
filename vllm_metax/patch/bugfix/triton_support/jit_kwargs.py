# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: MetaX Triton does not support newer upstream Triton JIT keyword args.
# -----------------------------------------------
import inspect
from functools import wraps

from vllm.triton_utils import triton


def _patch_triton_jit_kwargs() -> None:
    jit = getattr(triton, "jit", None)
    if jit is None or getattr(jit, "_metax_kwargs_compat", False):
        return
    try:
        parameters = inspect.signature(jit).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "do_not_specialize_on_alignment" in parameters:
        return

    @wraps(jit)
    def jit_kwargs_compat(*args, **kwargs):
        kwargs.pop("do_not_specialize_on_alignment", None)
        return jit(*args, **kwargs)

    jit_kwargs_compat._metax_kwargs_compat = True
    triton.jit = jit_kwargs_compat


_patch_triton_jit_kwargs()
