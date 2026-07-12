# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Qwen3 DSpark draft configs store mask_token_id at top level.
# -----------------------------------------------
import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

_original_model_returns_tuple = SpecDecodeBaseProposer.model_returns_tuple


def _init_parallel_drafting_params(self):
    model_hf_config = self.draft_model_config.hf_config
    self.parallel_drafting_token_id = get_parallel_drafting_token_id(model_hf_config)
    if self.speculative_config.method == "dspark":
        self.pass_hidden_states_to_model = False
        return
    if self.pass_hidden_states_to_model:
        self.parallel_drafting_hidden_state_tensor = torch.empty(
            self.hidden_size, dtype=self.dtype, device=self.device
        )


SpecDecodeBaseProposer._init_parallel_drafting_params = _init_parallel_drafting_params


def _model_returns_tuple(self) -> bool:
    if self.speculative_config.method == "dspark":
        return False
    return _original_model_returns_tuple(self)


SpecDecodeBaseProposer.model_returns_tuple = _model_returns_tuple
