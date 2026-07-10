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
