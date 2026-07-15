# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# -----------------------------------------------
# Note: Preserve per-row logits token-id state while condensing InputBatch.
# -----------------------------------------------

from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.sample.logits_processor import MoveDirectionality


_original_condense = InputBatch.condense


def _metax_condense(self: InputBatch) -> None:
    row_state = getattr(self, "logits_processing_needs_token_ids", None)
    if row_state is None:
        _original_condense(self)
        return

    previous_state = row_state.copy()
    moved_start = len(self.batch_update_builder.moved)
    _original_condense(self)
    for move in self.batch_update_builder.moved[moved_start:]:
        source, destination, direction = move
        row_state[destination] = previous_state[source]
        if direction == MoveDirectionality.SWAP:
            row_state[source] = previous_state[destination]


InputBatch.condense = _metax_condense
