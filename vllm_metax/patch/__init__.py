# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Aggregate and activate all MetaX patch categories.
#
# Affected versions: v0.21.0
# -----------------------------------------------
from . import bugfix  # noqa: F401
from . import plugin_enhancement  # noqa: F401
from . import performance  # noqa: F401

# DSpark PIECEWISE cudagraph for draft model (enabled by default)
try:
    from .performance.dspark_piecewise_cg import _apply_dflash_piecewise_cg_patch
    _apply_dflash_piecewise_cg_patch()
except Exception:
    pass

# Debug timing patch (enabled via VLLM_METAX_TIMING_PATCH=1)
try:
    from .debug.timing_patch import apply_timing_patch
    apply_timing_patch()
except ImportError:
    pass

# Debug V1 cycle host timing (enabled via VLLM_METAX_DSV4_V1_CYCLE_HOST=1)
try:
    from .debug.v1_cycle_host_patch import apply_v1_cycle_host_patch
    apply_v1_cycle_host_patch()
except ImportError:
    pass

# Debug accept count (enabled via VLLM_METAX_DSV4_ACCEPT_COUNT=1)
try:
    from .debug.accept_count_patch import apply_accept_count_patch
    apply_accept_count_patch()
except ImportError:
    pass

# Debug per-position accept rate (enabled via VLLM_METAX_DSV4_PERPOS_ACCEPT=1)
try:
    from .debug.perpos_accept_patch import apply_perpos_accept_patch
    apply_perpos_accept_patch()
except ImportError:
    pass

# Debug linear M-safe probe (enabled via VLLM_METAX_LINEAR_MSAFE_PROBE=1)
try:
    from .debug.linear_msafe_probe import apply_linear_msafe_probe
    apply_linear_msafe_probe()
except ImportError:
    pass

# M-safe linear patch (enabled via VLLM_METAX_MSAFE_LINEAR=1)
# NOTE: disabled — Triton accumulation order differs from native M=1.
# Using native batched path with quality-equivalence validation instead.
# try:
#     from .debug.msafe_linear_patch import apply_msafe_linear_patch
#     apply_msafe_linear_patch()
# except ImportError:
#     pass
