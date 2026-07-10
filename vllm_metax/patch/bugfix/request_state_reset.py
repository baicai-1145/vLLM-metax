# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Clear reusable speculative request-state slots on add.
# -----------------------------------------------

from vllm.v1.worker.gpu.states import RequestState

_original_add_request = RequestState.add_request


def _metax_add_request(
    self,
    req_id: str,
    prompt_len: int,
    all_token_ids: list[int],
    num_computed_tokens: int,
    max_tokens: int,
) -> None:
    _original_add_request(
        self,
        req_id,
        prompt_len,
        all_token_ids,
        num_computed_tokens,
        max_tokens,
    )
    req_idx = self.req_id_to_index[req_id]
    if num_computed_tokens == 0:
        self.last_sampled_tokens[req_idx : req_idx + 1].zero_()
    self.next_prefill_tokens[req_idx : req_idx + 1].zero_()


RequestState.add_request = _metax_add_request
