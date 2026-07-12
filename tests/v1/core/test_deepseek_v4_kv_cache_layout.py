from dataclasses import dataclass
from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import _get_kv_cache_config_packed
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm_metax.patch.bugfix.deepseek_v4.kv_cache_layout import (
    _ORIGINAL_GET_KV_CACHE_CONFIG_PACKED,
    _get_kv_cache_config_packed_deepseek_v4,
)


@dataclass(frozen=True)
class StubSpec(KVCacheSpec):
    page_size: int

    @property
    def page_size_bytes(self) -> int:
        return self.page_size


def _config(architecture: str = "DeepseekV4ForCausalLM", override=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
    )


def _groups(indexer_page=20, sibling_page=20, nested_page=65):
    indexer = "layers.0.self_attn.indexer.k_cache"
    sibling_compressor = "layers.0.self_attn.compressor.state_cache"
    nested_compressor = "layers.0.self_attn.indexer.compressor.state_cache"
    other0 = "layers.0.self_attn"
    other1 = "layers.1.self_attn"
    other2 = "layers.2.self_attn"
    return [
        KVCacheGroupSpec(
            [other0, indexer, sibling_compressor],
            UniformTypeKVCacheSpecs(
                block_size=1,
                kv_cache_specs={
                    other0: StubSpec(block_size=1, page_size=10),
                    indexer: StubSpec(block_size=1, page_size=indexer_page),
                    sibling_compressor: StubSpec(block_size=1, page_size=sibling_page),
                },
            ),
        ),
        KVCacheGroupSpec(
            [nested_compressor, other1, other2],
            UniformTypeKVCacheSpecs(
                block_size=1,
                kv_cache_specs={
                    nested_compressor: StubSpec(block_size=1, page_size=nested_page),
                    other1: StubSpec(block_size=1, page_size=30),
                    other2: StubSpec(block_size=1, page_size=40),
                },
            ),
        ),
    ]


def test_indexer_cache_is_standalone_and_compressor_slots_stay_packed():
    num_blocks, tensors = _get_kv_cache_config_packed_deepseek_v4(
        _config(), _groups(), available_memory=1850
    )

    assert num_blocks == 10
    assert [tensor.shared_by for tensor in tensors] == [
        ["layers.0.self_attn"],
        ["layers.0.self_attn.compressor.state_cache"],
        ["layers.0.self_attn.indexer.compressor.state_cache"],
        ["layers.1.self_attn"],
        ["layers.2.self_attn"],
        ["layers.0.self_attn.indexer.k_cache"],
    ]
    assert [
        (tensor.size, tensor.offset, tensor.block_stride) for tensor in tensors
    ] == [
        (1650, 0, 165),
        (1650, 10, 165),
        (1650, 30, 165),
        (1650, 95, 165),
        (1650, 125, 165),
        (200, 0, 0),
    ]


def test_num_gpu_blocks_override_recomputes_sizes():
    _, tensors = _get_kv_cache_config_packed_deepseek_v4(
        _config(override=3), _groups(), available_memory=1850
    )
    assert [tensor.size for tensor in tensors] == [495, 495, 495, 495, 495, 60]


def test_non_deepseek_wrapper_delegates_unchanged():
    groups = _groups()
    config = _config(architecture="OtherModel")
    assert _get_kv_cache_config_packed(config, groups, 1000) == (
        _ORIGINAL_GET_KV_CACHE_CONFIG_PACKED(config, groups, 1000)
    )
