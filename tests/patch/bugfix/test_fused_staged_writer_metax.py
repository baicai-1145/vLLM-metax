import torch
import vllm._custom_ops  # noqa: F401
import vllm_metax.patch.bugfix.triton_support.mrv2  # noqa: F401

from vllm.v1.worker.gpu.block_table import BlockTables


def test_metax_fused_staged_writer_supports_multiple_kv_groups():
    device = torch.device("cuda")
    tables = BlockTables(
        block_sizes=[64, 256, 256],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4, 4, 4],
        device=device,
        kernel_block_sizes=[64, 256, 256],
    )
    tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10], [20, 21]),
        overwrite=True,
    )

    tables.apply_staged_writes()
    torch.cuda.synchronize()

    assert tables.block_tables[0].gpu[0, :2].tolist() == [1, 2]
    assert tables.block_tables[1].gpu[0, :1].tolist() == [10]
    assert tables.block_tables[2].gpu[0, :2].tolist() == [20, 21]
