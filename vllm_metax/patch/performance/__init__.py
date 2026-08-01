# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Load performance-oriented MetaX patch modules.
#
# Affected versions: v0.21.0
# -----------------------------------------------
import os

from . import grouped_topk_router  # noqa: F401
from . import gpu_model_runner_capture  # noqa: F401
from . import pre_outer_graph_device_sync  # noqa: F401
from . import speculative_decode_perf  # noqa: F401

if (
    os.getenv("VLLM_METAX_DSPARK_PROFILE_PHASES") == "1"
    or os.getenv("VLLM_METAX_DSPARK_PHASE_TIMING_DIR")
):
    from . import dspark_cycle_profile  # noqa: F401
