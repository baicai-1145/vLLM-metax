from types import SimpleNamespace

import pytest
import torch

import vllm_metax.v1.attention.backends.mla.flashmla_sparse as sparse


@pytest.mark.parametrize(
    "builder_method",
    ["_build_fp8_separate_prefill_decode", "_build_bf16_separate_prefill_decode"],
)
def test_decode_metadata_uses_actual_sequence_lengths(monkeypatch, builder_method):
    monkeypatch.setattr(
        sparse,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (2, 0, 4, 0),
    )
    monkeypatch.setattr(sparse, "get_mla_metadata", lambda _: (object(), None))
    builder = object.__new__(sparse.FlashMLASparseMetadataBuilder)
    builder.reorder_batch_threshold = None
    builder.dummy_block_table = torch.zeros(2, 1, dtype=torch.int32)
    builder.max_model_len_tensor = torch.full((2,), 4096, dtype=torch.int32)
    seq_lens = torch.tensor([101, 202], dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=seq_lens,
    )

    metadata = getattr(builder, builder_method)(common)

    assert metadata.decode is not None
    cache_lens = metadata.decode.kernel_metadata.cache_lens
    assert cache_lens.data_ptr() == seq_lens.data_ptr()
    torch.testing.assert_close(cache_lens, seq_lens)


@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_sparse_decode_kernel_enables_causal_mask(monkeypatch, dtype):
    captured = {}

    def fake_flash_mla_with_kvcache(**kwargs):
        captured.update(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(
        sparse, "flash_mla_with_kvcache", fake_flash_mla_with_kvcache
    )
    impl = object.__new__(sparse.FlashMLASparseImpl)
    impl.softmax_scale = 0.5
    q = torch.zeros(1, 2, 2, 8)
    cache = torch.zeros(4, 8)
    indices = torch.zeros(1, 2, 1, dtype=torch.int32)
    metadata_type = (
        sparse.FlashMLASparseMetadata.FP8KernelMetadata
        if dtype == "fp8"
        else sparse.FlashMLASparseMetadata.BF16KernelMetadata
    )
    metadata = metadata_type(
        scheduler_metadata=object(),
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.tensor([2], dtype=torch.int32),
    )

    if dtype == "fp8":
        impl.fp8_decode_padded_heads = 2
        impl._fp8_flash_mla_kernel(q, cache, indices, metadata)
    else:
        impl.bf16_decode_padded_heads = 2
        impl._bf16_flash_mla_kernel_with_kvcache(q, cache, indices, metadata)

    assert captured["causal"] is True


def test_sparse_metadata_build_uses_device_request_id_builder(monkeypatch):
    captured = {}

    def fake_build_token_to_req_indices_out(
        query_start_loc, num_reqs, num_tokens, out, *, max_query_len
    ):
        captured.update(
            {
                "query_start_loc": query_start_loc,
                "num_reqs": num_reqs,
                "num_tokens": num_tokens,
                "out_ptr": out.data_ptr(),
                "max_query_len": max_query_len,
            }
        )
        out[:num_tokens].copy_(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
        return out[:num_tokens]

    monkeypatch.setattr(
        sparse, "build_token_to_req_indices_out", fake_build_token_to_req_indices_out
    )

    builder = object.__new__(sparse.FlashMLASparseMetadataBuilder)
    builder.req_id_per_token_buffer = torch.full((8,), -1, dtype=torch.int32)
    builder.compress_ratio = 1
    builder.use_fp8_kv_cache = False
    builder.use_bf16_kv_cache = False
    builder.is_deepseek_v4 = True
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.topk_tokens = 0
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=4,
        num_reqs=2,
        max_query_len=2,
        max_seq_len=16,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
    )

    metadata = builder.build(0, common)

    assert captured["query_start_loc"] is query_start_loc
    assert captured["num_reqs"] == 2
    assert captured["num_tokens"] == 4
    assert captured["out_ptr"] == builder.req_id_per_token_buffer.data_ptr()
    assert captured["max_query_len"] == 2
    torch.testing.assert_close(
        metadata.req_id_per_token,
        torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
