# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Keep DeepSeek V4 indexer/compressor cache aliases contiguous.

The upstream packed layout puts all same-sized cache slots in one backing
allocation.  The indexer cache is written by kernels that require its page to
be contiguous, so it is carved out as an independent allocation while the
remaining slots stay packed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)


def _is_deepseek_v4_config(vllm_config: VllmConfig) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False

    values: list[Any] = list(getattr(model_config, "architectures", ()) or ())
    for config_name in ("hf_config", "hf_text_config"):
        config = getattr(model_config, config_name, None)
        if config is not None:
            values.append(getattr(config, "model_type", None))

    return any("deepseekv4" in str(value).lower().replace("_", "") for value in values)


def _page_sizes_by_layer(
    kv_cache_groups: Iterable[KVCacheGroupSpec],
) -> dict[str, int]:
    page_sizes: dict[str, int] = {}
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if isinstance(spec, UniformTypeKVCacheSpecs):
                try:
                    page_size = spec.kv_cache_specs[layer_name].page_size_bytes
                except KeyError as exc:
                    raise ValueError(
                        f"KV cache group is missing spec for layer {layer_name!r}"
                    ) from exc
            else:
                page_size = spec.page_size_bytes
            if layer_name in page_sizes and page_sizes[layer_name] != page_size:
                raise ValueError(f"Layer {layer_name!r} has conflicting page sizes")
            page_sizes[layer_name] = page_size
    return page_sizes


def _indexer_cache_names(layer_names: Iterable[str]) -> list[str]:
    return [
        name for name in layer_names if name.split(".")[-2:] == ["indexer", "k_cache"]
    ]


def _get_kv_cache_config_packed_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    page_sizes = _page_sizes_by_layer(kv_cache_groups)
    all_names = [name for group in kv_cache_groups for name in group.layer_names]
    indexer_names = _indexer_cache_names(all_names)
    if not indexer_names:
        return _ORIGINAL_GET_KV_CACHE_CONFIG_PACKED(
            vllm_config, kv_cache_groups, available_memory
        )

    indexer_names_set = set(indexer_names)

    buckets = kv_cache_utils._bucket_layers_by_page_size(kv_cache_groups)
    packed_slots: list[tuple[int, list[str]]] = []
    for page_size, slots in buckets.items():
        for slot in slots:
            remaining = [name for name in slot if name not in indexer_names_set]
            if remaining:
                packed_slots.append((page_size, remaining))

    packed_bytes_per_block = sum(page_size for page_size, _ in packed_slots)
    standalone_bytes_per_block = sum(page_sizes[name] for name in indexer_names)
    total_bytes_per_block = packed_bytes_per_block + standalone_bytes_per_block
    if total_bytes_per_block <= 0:
        raise ValueError("DeepSeek V4 KV cache layout has no positive page size")

    num_blocks = available_memory // total_bytes_per_block
    num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)

    tensors: list[KVCacheTensor] = []
    if packed_slots:
        packed_size = packed_bytes_per_block * num_blocks
        byte_offset = 0
        for page_size, shared_by in packed_slots:
            tensors.append(
                KVCacheTensor(
                    size=packed_size,
                    shared_by=shared_by,
                    offset=byte_offset,
                    block_stride=packed_bytes_per_block,
                )
            )
            byte_offset += page_size

    for indexer_name in indexer_names:
        page_size = page_sizes[indexer_name]
        tensors.append(
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=[indexer_name],
                block_stride=0,
            )
        )

    return num_blocks, tensors


_ORIGINAL_GET_KV_CACHE_CONFIG_PACKED = getattr(
    kv_cache_utils,
    "_metax_original_get_kv_cache_config_packed",
    kv_cache_utils._get_kv_cache_config_packed,
)
if not hasattr(kv_cache_utils, "_metax_original_get_kv_cache_config_packed"):
    kv_cache_utils._metax_original_get_kv_cache_config_packed = (
        _ORIGINAL_GET_KV_CACHE_CONFIG_PACKED
    )


def _get_kv_cache_config_packed(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if not _is_deepseek_v4_config(vllm_config):
        return _ORIGINAL_GET_KV_CACHE_CONFIG_PACKED(
            vllm_config, kv_cache_groups, available_memory
        )
    return _get_kv_cache_config_packed_deepseek_v4(
        vllm_config, kv_cache_groups, available_memory
    )


kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_packed
kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_packed
