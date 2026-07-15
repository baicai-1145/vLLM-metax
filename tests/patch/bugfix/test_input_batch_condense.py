from unittest.mock import Mock

import torch

import vllm_metax.patch.bugfix.input_batch_condense  # noqa: F401
from vllm.v1.sample.logits_processor import MoveDirectionality
from vllm.v1.worker.gpu_input_batch import InputBatch


def test_condense_moves_logits_token_id_row_state() -> None:
    batch = InputBatch(
        max_num_reqs=3,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    batch._req_ids[:] = ["req0", None, "req2"]
    batch.req_id_to_index.update({"req0": 0, "req2": 2})
    batch.req_output_token_ids[:] = [[], [], []]
    batch.spec_token_ids[:] = [[], [], []]
    batch.num_tokens_no_spec[:] = 1
    batch.num_prompt_tokens[:] = 1
    batch.num_computed_tokens_cpu[:] = 1
    batch.logits_processing_needs_token_ids[:] = [False, False, True]
    batch.block_table = Mock()
    batch.batch_update_builder.removed_append(1)

    batch.condense()

    assert batch.req_ids[1] == "req2"
    assert batch.logits_processing_needs_token_ids[1]


def test_condense_ignores_preexisting_swap_record() -> None:
    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    batch._req_ids[:] = ["req0", "req1"]
    batch.req_id_to_index.update({"req0": 0, "req1": 1})
    batch.req_output_token_ids[:] = [[], []]
    batch.spec_token_ids[:] = [[], []]
    batch.logits_processing_needs_token_ids[:] = [False, True]
    batch.batch_update_builder.moved.append((0, 1, MoveDirectionality.SWAP))

    batch.condense()

    assert batch.logits_processing_needs_token_ids.tolist() == [False, True]
