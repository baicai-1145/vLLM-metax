# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Opt-in Plan08 MTP proposer state capture for the TP=4 k=1 correctness
# feedback loop.  This patch is inert unless VLLM_METAX_DSV4_MTP_CAPTURE_DIR is
# set.
# -----------------------------------------------
from __future__ import annotations

import os

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


_ORIGINAL_PREPARE_INPUTS_PADDED = SpecDecodeBaseProposer.prepare_inputs_padded
_ORIGINAL_SET_INPUTS_FIRST_PASS = SpecDecodeBaseProposer.set_inputs_first_pass


def _capture_enabled() -> bool:
    return bool(os.getenv("VLLM_METAX_DSV4_MTP_CAPTURE_DIR"))


def _prepare_inputs_padded(self, common_attn_metadata, spec_decode_metadata,
                           valid_sampled_tokens_count):
    result = _ORIGINAL_PREPARE_INPUTS_PADDED(
        self,
        common_attn_metadata,
        spec_decode_metadata,
        valid_sampled_tokens_count,
    )
    if _capture_enabled():
        from vllm_metax.models.deepseek_v4.mtp_debug import (
            maybe_capture_v1_proposer_prepare_inputs_padded,
        )

        spec_common_attn_metadata, token_indices_to_sample, num_rejected_tokens = (
            result
        )
        maybe_capture_v1_proposer_prepare_inputs_padded(
            before_cad=common_attn_metadata,
            after_cad=spec_common_attn_metadata,
            valid_sampled_tokens_count=valid_sampled_tokens_count,
            token_indices_to_sample=token_indices_to_sample,
            num_rejected_tokens=num_rejected_tokens,
        )
    return result


def _set_inputs_first_pass(
    self,
    target_token_ids,
    next_token_ids,
    target_positions,
    target_hidden_states,
    token_indices_to_sample,
    cad,
    num_rejected_tokens_gpu,
):
    before_cad = cad
    result = _ORIGINAL_SET_INPUTS_FIRST_PASS(
        self,
        target_token_ids,
        next_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        cad,
        num_rejected_tokens_gpu,
    )
    if _capture_enabled():
        from vllm_metax.models.deepseek_v4.mtp_debug import (
            maybe_capture_v1_proposer_first_pass,
        )

        num_tokens, new_token_indices_to_sample, after_cad = result
        maybe_capture_v1_proposer_first_pass(
            self,
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            token_indices_to_sample=new_token_indices_to_sample,
            before_cad=before_cad,
            after_cad=after_cad,
            num_rejected_tokens=num_rejected_tokens_gpu,
            num_tokens=num_tokens,
        )
    return result


SpecDecodeBaseProposer.prepare_inputs_padded = _prepare_inputs_padded
SpecDecodeBaseProposer.set_inputs_first_pass = _set_inputs_first_pass
