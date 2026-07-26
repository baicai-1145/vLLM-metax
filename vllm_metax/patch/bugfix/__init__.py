# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Aggregate bugfix patches for MetaX compatibility.
#
# Affected versions: v0.21.0
# -----------------------------------------------
# from . import dp_fix  # noqa: F401
from . import triton_support  # noqa: F401
from . import deepseek_v4  # noqa: F401
from . import dspark_v2_runner  # noqa: F401
from . import dflash_grouped_rms_norm  # noqa: F401
from . import parallel_drafting  # noqa: F401
from . import request_state_reset  # noqa: F401
from . import transformers_utils
from . import fp32_logits  # noqa: F401
from . import dspark_greedy_punctuation_tie  # noqa: F401
from . import plan08_mtp_debug_loop  # noqa: F401
from . import mtp_k1_serial_target  # noqa: F401
from . import plan08_stop_aware_output  # noqa: F401
from . import batch_invariant_metax  # noqa: F401
from . import input_batch_condense  # noqa: F401
from . import mtp_target_runtime_capture  # noqa: F401
from . import spec_acceptance_capture  # noqa: F401
