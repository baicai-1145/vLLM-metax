# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Disable Triton JIT monitoring because MACA does not support it.
#
# Affected versions: v0.21.0
# -----------------------------------------------
from importlib import import_module

_jit_monitor = import_module("vllm.utils.jit_monitor")
logger = _jit_monitor.logger

# ------------------------------------------------------------------
# JIT monitor patch, this feature is not supported in
# maca, Just disable the JIT monitor and print a warning if activate() is called.
# ------------------------------------------------------------------


def activate(*args, **kwargs) -> None:
    """Enable JIT compilation monitoring after warmup.

    Call once per worker process at the end of
    :func:`compile_or_warm_up_model`.  After activation every Triton
    kernel compilation or autotuning benchmark that happens during
    inference will be logged as a warning.

    Safe to call multiple times — subsequent calls are no-ops.

    If the user has explicitly set ``TRITON_PRINT_AUTOTUNING=0`` in
    their environment, autotuning printing is left disabled; the JIT
    compilation hook is still registered regardless.
    """

    logger.info(
        "Kernel JIT monitor activated is not supported on triton==3.0.0+maca. Currently"
        "disabled"
    )

    return


_jit_monitor.activate = activate
