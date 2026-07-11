# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Keep branch-local values the same type for Triton 3.0 on MetaX.
#
# Affected versions: triton==3.0.0+maca
# Remove at: after the MetaX Triton frontend accepts mixed branch input types.
# -----------------------------------------------
from importlib import import_module

from vllm.triton_utils import tl, triton

_mtp_rmsnorm = import_module(
    "vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm"
)
_rmsnorm_row = _mtp_rmsnorm._rmsnorm_row


@triton.jit
def _fused_mtp_input_rmsnorm_kernel(
    inputs_embeds_ptr,
    positions_ptr,
    prev_hidden_ptr,
    enorm_weight_ptr,
    hnorm_weight_ptr,
    enorm_out_ptr,
    hnorm_out_ptr,
    eps,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HIDDEN

    if pid_task == 0:
        pos = tl.load(positions_ptr + token_idx)
        x = tl.load(
            inputs_embeds_ptr + token_idx * HIDDEN + block,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.where(pos != 0, x, 0.0)
        _rmsnorm_row(
            x,
            enorm_weight_ptr,
            enorm_out_ptr + token_idx * HIDDEN,
            block,
            mask,
            eps,
            HIDDEN,
        )
    else:
        slot = pid_task - 1
        row_offset = (token_idx * HC_MULT + slot) * HIDDEN
        x = tl.load(
            prev_hidden_ptr + row_offset + block,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        _rmsnorm_row(
            x,
            hnorm_weight_ptr,
            hnorm_out_ptr + row_offset,
            block,
            mask,
            eps,
            HIDDEN,
        )


_mtp_rmsnorm._fused_mtp_input_rmsnorm_kernel = _fused_mtp_input_rmsnorm_kernel
