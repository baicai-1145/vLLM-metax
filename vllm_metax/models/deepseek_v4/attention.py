# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
import time
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import torch

# Diagnostic: accumulate attention_impl eager-break wall time
_ATTN_TIMING = {"total": 0.0, "count": 0, "enabled": False}

import atexit as _atexit

def _print_attn_timing():
    if _ATTN_TIMING["count"] > 0:
        import os as _os
        _rank = int(_os.getenv("RANK", _os.getenv("LOCAL_RANK", "0")))
        _path = f"/tmp/attn_timing_rank{_rank}.txt"
        with open(_path, "w") as _f:
            _f.write(f"rank={_rank}\n")
            _f.write(f"total={_ATTN_TIMING['total']:.6f}\n")
            _f.write(f"count={_ATTN_TIMING['count']}\n")
            _f.write(f"avg_ms={_ATTN_TIMING['total']/max(1,_ATTN_TIMING['count'])*1000:.4f}\n")

_atexit.register(_print_attn_timing)

import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm_metax.customized.layers.sparse_attn_indexer.sparse_attn_indexer import (
    MacaSparseAttnIndexer,
)
from .ops import (
    fused_indexer_q_rope_int8_quant,
)
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm


from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from .compressor import MacaDeepseekCompressor
from .row_indices import get_cached_row_indices
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm_metax.v1.attention.backends.mla.indexer import (
    MacaDeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm_metax.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
import vllm_metax.envs as mx_envs

from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
    _resolve_dsv4_kv_cache_dtype,
)
from .layer_debug import (
    layer_capture_enabled,
    layer_capture_layer_enabled,
    maybe_capture_attention_inputs,
    maybe_capture_attention_output,
    maybe_capture_qkv_prenorm_shadow_compare,
    maybe_capture_qkv_producer,
    maybe_capture_wq_b_shadow_compare,
    maybe_prepare_qkv_insert_capture,
    maybe_prepare_q_stage_capture,
    qkv_prenorm_shadow_compare_enabled,
    qkv_prenorm_shadow_compare_selected,
    wq_b_shadow_compare_selected,
)
from .mtp_candidate import (
    env_or_k1_candidate_enabled,
    k1_correctness_candidate_enabled,
    k1_native_kv_prenorm_candidate_enabled,
    k1_native_wq_b_candidate_enabled,
    k1_native_wq_b_candidate_layer_enabled,
    k1_native_wq_b_candidate_layer_scoped,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

logger = init_logger(__name__)

_Q_INSERT_CUDAGRAPH_LAYER_ENV = "VLLM_METAX_DSV4_Q_INSERT_CUDAGRAPH_LAYER"
_TOKENWISE_WQ_B_ENV = "VLLM_METAX_DSV4_TOKENWISE_WQ_B"
_TOKENWISE_TARGET_WQ_B_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B"
_NATIVE_SERIAL_WQ_B_ROWS_ENV = (
    "VLLM_METAX_DSV4_NATIVE_SERIAL_WQ_B_ROWS"
)
_EXACT_GROUPED_WQ_B_ROWS_ENV = (
    "VLLM_METAX_DSV4_EXACT_GROUPED_WQ_B_ROWS"
)
_EXACT_GROUPED_INDEXER_WQ_B_ROWS_ENV = (
    "VLLM_METAX_DSV4_EXACT_GROUPED_INDEXER_WQ_B_ROWS"
)
_TOKENWISE_INDEXER_WQ_B_ENV = "VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B"
_TOKENWISE_INDEXER_WQ_B_LAYERS_ENV = (
    "VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B_LAYERS"
)
_TOKENWISE_INDEXER_WEIGHT_ROWS_ENV = (
    "VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS"
)
_TOKENWISE_ATTN_GEMM_ENV = "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM"
_TOKENWISE_ATTN_GEMM_AUX_ENV = "VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM_AUX"
_NATIVE_SERIAL_ATTN_GEMM_ROWS_ENV = (
    "VLLM_METAX_DSV4_NATIVE_SERIAL_ATTN_GEMM_ROWS"
)
_TOKENWISE_TARGET_WQ_B_LAYERS_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS"
_TOKENWISE_TARGET_WQ_B_POSITIONS_ENV = (
    "VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS"
)
_TOKENWISE_TARGET_QKV_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV"
_TOKENWISE_TARGET_QKV_PRENORM_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_PRENORM"
_TOKENWISE_TARGET_KV_PRENORM_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM"
_TOKENWISE_TARGET_QKV_LAYERS_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS"
_TOKENWISE_TARGET_QKV_POSITIONS_ENV = (
    "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS"
)
_TOKENWISE_TARGET_QKV_CALLS_ENV = "VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_CALLS"
_K1_RESPECT_TARGET_QKV_SCOPE_ENV = (
    "VLLM_METAX_DSV4_MTP_K1_RESPECT_TARGET_QKV_SCOPE"
)


def resolve_layer_compress_ratio(config, layer_id: int) -> tuple[int, bool]:
    """Resolve operational compress ratio and draft-layer RoPE mode.

    Some DeepSeek V4 checkpoints include the MTP draft layer in
    ``compress_ratios`` with a raw value of 0. KV cache users still need an
    operational ratio of 1, but that raw 0 selects plain, unscaled RoPE for the
    draft layer.
    """
    compress_ratios = getattr(config, "compress_ratios", None)
    if not compress_ratios:
        return 1, False
    if layer_id < config.num_hidden_layers:
        return max(1, compress_ratios[layer_id]), False
    if layer_id < len(compress_ratios):
        raw_compress_ratio = compress_ratios[layer_id]
        return max(1, raw_compress_ratio), raw_compress_ratio == 0
    return 1, False


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
    use_unscaled_rope: bool = False,
):
    rope_parameters = config.rope_parameters
    if "rope_type" not in rope_parameters:
        rope_parameters = (
            rope_parameters["main"]
            if use_unscaled_rope or compress_ratio <= 1
            else rope_parameters["compress"]
        )
    rope_parameters = dict(rope_parameters)
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if use_unscaled_rope:
        rope_parameters["rope_type"] = "default"
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    rope_parameters["mscale"] = 0
    rope_parameters["mscale_all_dim"] = 0
    rope_parameters["is_deepseek_v4"] = True
    rope_parameters["rope_dim"] = rope_head_dim
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
        dtype=torch.float32,
    )


def _target_tokenwise_wq_b_enabled() -> bool:
    if (
        os.getenv(_TOKENWISE_WQ_B_ENV) == "1"
        or os.getenv(_TOKENWISE_TARGET_WQ_B_ENV) == "1"
    ):
        return True
    return (
        k1_correctness_candidate_enabled()
        and (
            not k1_native_wq_b_candidate_enabled()
            or k1_native_wq_b_candidate_layer_scoped()
        )
    )


def _native_serial_wq_b_rows_enabled() -> bool:
    return os.getenv(_NATIVE_SERIAL_WQ_B_ROWS_ENV, "0") == "1"


def _exact_grouped_wq_b_rows_enabled() -> bool:
    return os.getenv(_EXACT_GROUPED_WQ_B_ROWS_ENV, "0") == "1"


def _exact_grouped_indexer_wq_b_rows_enabled() -> bool:
    return os.getenv(_EXACT_GROUPED_INDEXER_WQ_B_ROWS_ENV, "0") == "1"


def _native_serial_attn_gemm_rows_enabled() -> bool:
    return os.getenv(_NATIVE_SERIAL_ATTN_GEMM_ROWS_ENV, "0") == "1"


def _target_tokenwise_wq_b_layer_enabled(layer_idx: int) -> bool:
    if (
        os.getenv(_TOKENWISE_WQ_B_ENV) == "1"
        or os.getenv(_TOKENWISE_TARGET_WQ_B_ENV) == "1"
    ):
        return True
    if (
        k1_correctness_candidate_enabled()
        and k1_native_wq_b_candidate_layer_enabled(layer_idx)
    ):
        return False
    if k1_correctness_candidate_enabled():
        return True
    value = os.getenv(_TOKENWISE_TARGET_WQ_B_LAYERS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_WQ_B_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_TOKENWISE_TARGET_WQ_B_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        )
    return layer_idx in selected


def _target_tokenwise_wq_b_position_enabled(positions: torch.Tensor) -> bool:
    if k1_correctness_candidate_enabled():
        return True
    value = os.getenv(_TOKENWISE_TARGET_WQ_B_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_WQ_B_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return True
    position_values = positions.detach().reshape(-1)
    return any(bool(torch.any(position_values == position).item()) for position in selected)
def _target_tokenwise_wq_b_selected_indices(positions: torch.Tensor) -> torch.Tensor:
    flat_positions = positions.detach().reshape(-1)
    if k1_correctness_candidate_enabled():
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    value = os.getenv(_TOKENWISE_TARGET_WQ_B_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_WQ_B_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    mask = torch.zeros_like(flat_positions, dtype=torch.bool)
    for position in selected:
        mask |= flat_positions == position
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _target_tokenwise_qkv_enabled() -> bool:
    return os.getenv(_TOKENWISE_TARGET_QKV_ENV) == "1"


def _k1_target_qkv_scope_unrestricted() -> bool:
    return (
        k1_correctness_candidate_enabled()
        and os.getenv(_K1_RESPECT_TARGET_QKV_SCOPE_ENV, "0") != "1"
    )


def _target_tokenwise_qkv_prenorm_enabled() -> bool:
    return os.getenv(_TOKENWISE_TARGET_QKV_PRENORM_ENV) == "1"
def _target_tokenwise_kv_prenorm_enabled() -> bool:
    if os.getenv(_TOKENWISE_TARGET_KV_PRENORM_ENV) == "1":
        return True
    return (
        k1_correctness_candidate_enabled()
        and not k1_native_kv_prenorm_candidate_enabled()
    )
def _target_tokenwise_qkv_layer_enabled(layer_idx: int) -> bool:
    if _k1_target_qkv_scope_unrestricted():
        return True
    value = os.getenv(_TOKENWISE_TARGET_QKV_LAYERS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        )
    return layer_idx in selected
def _target_tokenwise_qkv_position_enabled(positions: torch.Tensor) -> bool:
    if _k1_target_qkv_scope_unrestricted():
        return True
    value = os.getenv(_TOKENWISE_TARGET_QKV_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return True
    position_values = positions.detach().reshape(-1)
    return any(bool(torch.any(position_values == position).item()) for position in selected)


def _target_tokenwise_qkv_call_enabled(call_index: int) -> bool:
    if _k1_target_qkv_scope_unrestricted():
        return True
    value = os.getenv(_TOKENWISE_TARGET_QKV_CALLS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_CALLS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer call indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_CALLS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer call indices"
        )
    if not selected:
        return True
    return call_index in selected


def _next_target_tokenwise_qkv_call_index(layer: object) -> int:
    value = getattr(layer, "_target_tokenwise_qkv_call_index", 0)
    setattr(layer, "_target_tokenwise_qkv_call_index", value + 1)
    return value


def _target_tokenwise_qkv_selected_indices(positions: torch.Tensor) -> torch.Tensor:
    flat_positions = positions.detach().reshape(-1)
    if _k1_target_qkv_scope_unrestricted():
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    value = os.getenv(_TOKENWISE_TARGET_QKV_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return get_cached_row_indices(flat_positions.numel(), positions.device)
    mask = torch.zeros_like(flat_positions, dtype=torch.bool)
    for position in selected:
        mask |= flat_positions == position
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _target_tokenwise_qkv_position_mask(positions: torch.Tensor) -> torch.Tensor:
    flat_positions = positions.reshape(-1)
    value = os.getenv(_TOKENWISE_TARGET_QKV_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return torch.ones_like(flat_positions, dtype=torch.bool)
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_TARGET_QKV_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return torch.ones_like(flat_positions, dtype=torch.bool)
    mask = torch.zeros_like(flat_positions, dtype=torch.bool)
    for position in selected:
        mask |= flat_positions == position
    return mask


def _indexer_tokenwise_wq_b_enabled(layer_idx: int) -> bool:
    enabled = (
        os.getenv(_TOKENWISE_WQ_B_ENV) == "1"
        or os.getenv(_TOKENWISE_INDEXER_WQ_B_ENV) == "1"
    )
    if not enabled:
        return False
    value = os.getenv(_TOKENWISE_INDEXER_WQ_B_LAYERS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_INDEXER_WQ_B_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_TOKENWISE_INDEXER_WQ_B_LAYERS_ENV} must be 'all' or a "
            "comma-separated set of nonnegative integer layer indices"
        )
    return layer_idx in selected


def _indexer_tokenwise_weight_rows_enabled(
    layer_idx: int, num_hidden_layers: int
) -> bool:
    value = os.getenv(_TOKENWISE_INDEXER_WEIGHT_ROWS_ENV, "").strip().lower()
    if not value or value in {"0", "false", "off", "no"}:
        return False
    if value == "target":
        return layer_idx < num_hidden_layers
    raise ValueError(
        f"{_TOKENWISE_INDEXER_WEIGHT_ROWS_ENV} must be off or target"
    )


def _tokenwise_attn_gemm_aux_components() -> tuple[str, ...]:
    value = os.getenv(_TOKENWISE_ATTN_GEMM_AUX_ENV, "").strip()
    if not value or value.lower() in {"0", "false", "off", "no"}:
        return ()
    valid = {
        "kv_score",
        "indexer_kv_score",
        "indexer_weights",
    }
    components = tuple(
        component.strip()
        for component in value.split(",")
        if component.strip()
    )
    invalid = [component for component in components if component not in valid]
    if invalid:
        raise ValueError(
            f"{_TOKENWISE_ATTN_GEMM_AUX_ENV} must be a comma-separated subset "
            "of kv_score,indexer_kv_score,indexer_weights"
        )
    return components


def _qnorm_rope_kv_insert_native(
    q,
    kv,
    cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    block_size,
):
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_insert(
        q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )


def _run_qnorm_rope_kv_insert(
    q,
    kv,
    cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    block_size,
) -> None:
    if (
        os.getenv("VLLM_METAX_DSV4_TOKENWISE_QKV_INSERT") != "1"
        or not 1 < q.shape[0] <= 6
    ):
        _qnorm_rope_kv_insert_native(
            q,
            kv,
            cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps,
            block_size,
        )
        return
    logger.warning_once(
        "DeepSeek V4 speculative attention uses tokenwise Q/KV cache insert"
    )
    for index in range(q.shape[0]):
        _qnorm_rope_kv_insert_native(
            q[index : index + 1],
            kv[index : index + 1],
            cache,
            slot_mapping[index : index + 1],
            positions[index : index + 1],
            cos_sin_cache,
            eps,
            block_size,
        )


def _q_insert_cudagraph_target_layer() -> int | Literal["all"] | None:
    value = os.environ.get(_Q_INSERT_CUDAGRAPH_LAYER_ENV)
    if value is None:
        return None
    if value == "all":
        return "all"
    try:
        layer_idx = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_Q_INSERT_CUDAGRAPH_LAYER_ENV} must be an integer layer id, "
            f"got {value!r}"
        ) from exc
    if layer_idx < 0:
        raise ValueError(
            f"{_Q_INSERT_CUDAGRAPH_LAYER_ENV} must be a non-negative layer id, "
            f"got {value!r}"
        )
    return layer_idx


def _q_insert_tensor_key(tensor: torch.Tensor) -> tuple:
    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.data_ptr(),
        tensor.storage_offset(),
    )


def _q_insert_cudagraph_pool_handle():
    from vllm.platforms import current_platform

    return current_platform.graph_pool_handle()


class MacaDeepseekV4Attention(DeepseekV4Attention):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` / ``DeepseekV4FlashInferMLAAttention``
    (CUDA) or ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the
    platform-specific deepseek_v4 model module. The base is never instantiated
    directly.
    """
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = FlashMLA / ROCm fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8);
    # False = FlashInfer plain bf16 / per-tensor fp8 KV row.
    # ------------------------------------------------------------
    # Note(Metax): use bf16
    use_flashmla_fp8_layout: ClassVar[bool] = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        is_target_model: bool = False,
    ) -> None:
        super(DeepseekV4Attention, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)
        self.layer_idx = layer_id
        self.is_target_model = is_target_model
        self._attention_input_capture_enabled = (
            layer_capture_enabled()
            and layer_capture_layer_enabled(self.layer_idx)
        )

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.compress_ratio, use_unscaled_rope = resolve_layer_compress_ratio(
            config, layer_id
        )
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
            use_unscaled_rope=use_unscaled_rope,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = MacaDeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self._prefill_gemm_chunking_enabled = (
            mx_envs.VLLM_METAX_DSV4_PREFILL_GEMM_CHUNKING
        )
        self._prefill_gemm_chunk_size = cache_config.block_size
        if self._prefill_gemm_chunk_size <= 0:
            raise ValueError(
                "DeepSeek V4 prefill GEMM chunk size must be positive, got "
                f"{self._prefill_gemm_chunk_size}"
            )

        # Cache per-call env-flag decisions (constant per session/layer)
        # to reduce host-side eager dispatch overhead in attention_impl.
        self._cached_tw_qkv_enabled = _target_tokenwise_qkv_enabled()
        self._cached_tw_qkv_layer = _target_tokenwise_qkv_layer_enabled(
            self.layer_idx
        )
        self._cached_tw_wq_b_enabled = _target_tokenwise_wq_b_enabled()
        self._cached_tw_wq_b_layer = _target_tokenwise_wq_b_layer_enabled(
            self.layer_idx
        )
        self._cached_is_target_tw_wq_b = (
            self.is_target_model
            and self._cached_tw_wq_b_enabled
            and self._cached_tw_wq_b_layer
        )
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format (a
        # ClassVar set by the subclass): fp8_ds_mla (UE8M0 block-scaled fp8 as
        # uint8) for FlashMLA / ROCm, vs a plain bf16 / per-tensor fp8 row for
        # FlashInfer. The same resolution drives the SWA cache tensor dtype
        # below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self.use_flashmla_fp8_layout, cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = MacaDeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

        self._q_insert_cudagraphs: dict[tuple, tuple[object, torch.Tensor]] = {}
        self._q_insert_cudagraph_pool = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        single_hidden_states = hidden_states
        if hidden_states.dim() == 3:
            single_hidden_states = hidden_states[:, 0, :]
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(single_hidden_states)
        )
        if (
            qkv_prenorm_shadow_compare_enabled()
            and 1 < single_hidden_states.shape[0] <= 6
        ):
            self._capture_qkv_prenorm_shadow_compare(
                positions, single_hidden_states, qr_kv
            )
        qr_kv = self._replace_target_fused_qkv_prenorm(
            single_hidden_states, positions, qr_kv
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        if layer_capture_enabled():
            self._capture_qkv_producer(
                positions,
                single_hidden_states,
                qr_kv,
                qr,
                kv,
            )
        self.attention_impl(
            single_hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def attn_gemm_parallel_execute(self, hidden_states):
        if (
            os.getenv(_TOKENWISE_ATTN_GEMM_ENV) == "1"
            and 1 < hidden_states.shape[0] <= 6
        ):
            logger.warning_once(
                "DeepSeek V4 speculative attention uses tokenwise projection GEMMs"
            )
            results = [
                super(MacaDeepseekV4Attention, self).attn_gemm_parallel_execute(
                    hidden_states[index : index + 1]
                )
                for index in range(hidden_states.shape[0])
            ]
            none_mask = tuple(value is None for value in results[0])
            if any(
                tuple(value is None for value in result) != none_mask
                for result in results[1:]
            ):
                raise RuntimeError(
                    "DeepSeek V4 tokenwise projection GEMMs returned inconsistent "
                    "None/non-None components"
                )
            return tuple(
                None
                if is_none
                else torch.cat([result[index] for result in results], dim=0)
                for index, is_none in enumerate(none_mask)
            )
        if self._qkv_kv_score_with_batched_indexer_enabled(hidden_states):
            return self._attn_gemm_qkv_kv_score_with_batched_indexer(
                hidden_states
            )
        if (
            not self._prefill_gemm_chunking_enabled
            or hidden_states.shape[0] <= self._prefill_gemm_chunk_size
        ):
            result = super().attn_gemm_parallel_execute(hidden_states)
            result = self._replace_fused_q_tokenwise(hidden_states, result)
            return self._replace_attn_gemm_aux_tokenwise(hidden_states, result)

        chunk_results = []
        chunk_size = self._prefill_gemm_chunk_size
        for start in range(0, hidden_states.shape[0], chunk_size):
            chunk_results.append(
                super().attn_gemm_parallel_execute(
                    hidden_states[start : start + chunk_size]
                )
            )

        first_result = chunk_results[0]
        expected_none = tuple(component is None for component in first_result)
        for result in chunk_results[1:]:
            if len(result) != len(first_result) or tuple(
                component is None for component in result
            ) != expected_none:
                raise RuntimeError(
                    "DeepSeek V4 prefill GEMM chunks returned inconsistent "
                    "None/non-None components"
                )

        result = tuple(
            None
            if is_none
            else torch.cat([result[index] for result in chunk_results], dim=0)
            for index, is_none in enumerate(expected_none)
        )
        result = self._replace_fused_q_tokenwise(hidden_states, result)
        return self._replace_attn_gemm_aux_tokenwise(hidden_states, result)

    def _replace_attn_gemm_aux_tokenwise(self, hidden_states, result):
        components = _tokenwise_attn_gemm_aux_components()
        if not components or not 1 < hidden_states.shape[0] <= 6:
            return result

        component_indices = {
            "kv_score": 1,
            "indexer_kv_score": 2,
            "indexer_weights": 3,
        }
        selected_indices = [
            component_indices[component] for component in components
        ]
        selected_indices = [
            index for index in selected_indices if result[index] is not None
        ]
        if not selected_indices:
            return result

        logger.warning_once(
            "DeepSeek V4 speculative attention uses tokenwise auxiliary "
            "projection GEMMs: %s",
            ",".join(components),
        )
        row_results = [
            super(MacaDeepseekV4Attention, self).attn_gemm_parallel_execute(
                hidden_states[index : index + 1]
            )
            for index in range(hidden_states.shape[0])
        ]
        expected_none = tuple(value is None for value in row_results[0])
        if any(
            tuple(value is None for value in row_result) != expected_none
            for row_result in row_results[1:]
        ):
            raise RuntimeError(
                "DeepSeek V4 tokenwise auxiliary projection GEMMs returned "
                "inconsistent None/non-None components"
            )

        output = list(result)
        for index in selected_indices:
            if expected_none[index]:
                output[index] = None
            else:
                output[index] = torch.cat(
                    [row_result[index] for row_result in row_results], dim=0
                )
        return tuple(output)

    def _qkv_kv_score_with_batched_indexer_enabled(self, hidden_states) -> bool:
        components = _tokenwise_attn_gemm_aux_components()
        return (
            os.getenv("VLLM_METAX_DSV4_TOKENWISE_QKV") == "1"
            and len(components) == 1
            and components[0] == "kv_score"
            and getattr(self, "compressor", None) is not None
            and 1 < hidden_states.shape[0] <= 6
        )

    def _attn_gemm_qkv_kv_score_with_batched_indexer(self, hidden_states):
        logger.warning_once(
            "DeepSeek V4 speculative attention uses tokenwise QKV and "
            "kv_score projection GEMMs with batched indexer auxiliary GEMMs"
        )
        compressor = self.compressor
        assert compressor is not None
        indexer = getattr(self, "indexer", None)

        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        def fused_wqa_wkv_tokenwise() -> torch.Tensor:
            if _native_serial_attn_gemm_rows_enabled():
                return self._project_fused_qkv_native_serial_rows(hidden_states)
            return torch.cat(
                [
                    self.fused_wqa_wkv(hidden_states[index : index + 1])[
                        0
                    ].clone()
                    for index in range(hidden_states.shape[0])
                ],
                dim=0,
            )

        def compressor_kv_score_tokenwise() -> torch.Tensor:
            if _native_serial_attn_gemm_rows_enabled():
                return self._project_kv_score_fp32_serial_rows(hidden_states)
            return torch.cat(
                [
                    torch.mm(
                        hidden_states[index : index + 1],
                        compressor.fused_wkv_wgate.weight.T,
                        out_dtype=torch.float32,
                    ).clone()
                    for index in range(hidden_states.shape[0])
                ],
                dim=0,
            )

        aux_fns = [compressor_kv_score_tokenwise, None, None]
        if indexer is not None:

            def indexer_weights_proj() -> torch.Tensor:
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv_tokenwise,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=aux_streams is not None,
        )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def _project_fused_qkv_native_serial_rows(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        weight = getattr(self.fused_wqa_wkv, "weight", None)
        expected_width = self.q_lora_rank + self.head_dim
        if not 2 <= rows <= 6:
            raise RuntimeError("native serial attention QKV requires 2 <= rows <= 6")
        if (
            hidden_states.ndim != 2
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or tuple(weight.shape) != (expected_width, hidden_states.shape[1])
            or hidden_states.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or hidden_states.device != weight.device
            or not hidden_states.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise RuntimeError(
                "native serial attention QKV requires contiguous BF16 input "
                f"[B,K] and weight [{expected_width},K] on the same device"
            )
        try:
            op = torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out
        except AttributeError as exc:
            raise RuntimeError(
                "native attention QKV operator gemv_bf16_serial_rows_out "
                "is unavailable"
            ) from exc

        key = (rows, expected_width, hidden_states.dtype, hidden_states.device)
        workspaces = getattr(self, "_native_serial_attn_qkv_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._native_serial_attn_qkv_workspaces = workspaces
        output = workspaces.get(key)
        if output is None:
            output = torch.empty(
                (rows, expected_width),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            workspaces[key] = output
        op(hidden_states, weight, output)
        logger.warning_once(
            "DeepSeek V4 speculative attention QKV uses native serial-row "
            "workspace: rows=%d launches=%d",
            rows,
            rows,
        )
        return output

    def _project_kv_score_fp32_serial_rows(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        compressor = self.compressor
        weight = getattr(compressor.fused_wkv_wgate, "weight", None)
        if not 2 <= rows <= 6:
            raise RuntimeError("serial attention kv_score requires 2 <= rows <= 6")
        if (
            hidden_states.ndim != 2
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.shape[1] != hidden_states.shape[1]
            or hidden_states.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or hidden_states.device != weight.device
            or not hidden_states.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise RuntimeError(
                "serial attention kv_score requires contiguous BF16 input "
                "[B,K] and BF16 weight [N,K] on the same device"
            )
        try:
            op = torch.ops._metax_sparse_C.gemv_bf16_fp32_serial_rows_out
        except AttributeError as exc:
            raise RuntimeError(
                "native FP32 serial attention kv_score operator is unavailable"
            ) from exc

        key = (rows, weight.shape[0], torch.float32, hidden_states.device)
        workspaces = getattr(self, "_serial_attn_kv_score_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._serial_attn_kv_score_workspaces = workspaces
        output = workspaces.get(key)
        if output is None:
            output = torch.empty(
                (rows, weight.shape[0]),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            workspaces[key] = output
        op(hidden_states, weight, output)
        logger.warning_once(
            "DeepSeek V4 speculative attention kv_score uses native FP32 "
            "serial-row workspace: rows=%d launches=%d",
            rows,
            rows,
        )
        return output

    def _replace_fused_q_tokenwise(self, hidden_states, result):
        tokenwise_qkv = os.getenv("VLLM_METAX_DSV4_TOKENWISE_QKV") == "1"
        tokenwise_q_only = os.getenv("VLLM_METAX_DSV4_TOKENWISE_Q_ONLY") == "1"
        if (
            not (tokenwise_qkv or tokenwise_q_only)
            or not 1 < hidden_states.shape[0] <= 6
        ):
            return result
        logger.warning_once(
            "DeepSeek V4 speculative attention uses tokenwise %s projection",
            "QKV" if tokenwise_qkv else "Q-only",
        )
        tokenwise_qr_kv = torch.cat(
            [
                self.fused_wqa_wkv(hidden_states[index : index + 1])[0].clone()
                for index in range(hidden_states.shape[0])
            ],
            dim=0,
        )
        if tokenwise_qkv:
            return (tokenwise_qr_kv, *result[1:])
        qr_kv = result[0]
        return (
            torch.cat(
                [
                    tokenwise_qr_kv[:, : self.q_lora_rank],
                    qr_kv[:, self.q_lora_rank :],
                ],
                dim=-1,
            ),
            *result[1:],
        )

    def _replace_target_fused_qkv_prenorm(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        qr_kv: torch.Tensor,
    ) -> torch.Tensor:
        tokenwise_qkv = _target_tokenwise_qkv_prenorm_enabled()
        tokenwise_kv = _target_tokenwise_kv_prenorm_enabled()
        if (
            not (tokenwise_qkv or tokenwise_kv)
            or not _target_tokenwise_qkv_layer_enabled(self.layer_idx)
            or not 1 < hidden_states.shape[0] <= 6
        ):
            return qr_kv
        call_index = _next_target_tokenwise_qkv_call_index(self)
        if not _target_tokenwise_qkv_call_enabled(call_index):
            return qr_kv
        logger.warning_once(
            "DeepSeek V4 speculative target attention uses prenorm tokenwise %s "
            "projection",
            "QKV" if tokenwise_qkv else "KV",
        )
        if not _k1_target_qkv_scope_unrestricted() and k1_correctness_candidate_enabled():
            selected_mask = _target_tokenwise_qkv_position_mask(positions)
            tokenwise_qr_kv = torch.cat(
                [
                    self.fused_wqa_wkv(hidden_states[index : index + 1])[0].clone()
                    for index in range(hidden_states.shape[0])
                ],
                dim=0,
            )
            selected_qr_kv = torch.where(
                selected_mask[:, None], tokenwise_qr_kv, qr_kv
            )
            if tokenwise_qkv:
                return selected_qr_kv.contiguous()
            return torch.cat(
                [
                    qr_kv[:, : self.q_lora_rank],
                    selected_qr_kv[:, self.q_lora_rank :],
                ],
                dim=-1,
            ).contiguous()
        selected_indices = _target_tokenwise_qkv_selected_indices(positions)
        if selected_indices.numel() == 0:
            return qr_kv
        if selected_indices.numel() == hidden_states.shape[0]:
            selected_range = range(hidden_states.shape[0])
        else:
            selected_range = selected_indices.tolist()
        tokenwise_qr_kv = torch.cat(
            [
                self.fused_wqa_wkv(hidden_states[index : index + 1])[0].clone()
                for index in selected_range
            ],
            dim=0,
        )
        if selected_indices.numel() == hidden_states.shape[0]:
            selected_qr_kv = tokenwise_qr_kv
        else:
            selected_qr_kv = qr_kv.clone()
            selected_qr_kv[selected_indices] = tokenwise_qr_kv
        if tokenwise_qkv:
            return selected_qr_kv.contiguous()
        return torch.cat(
            [
                qr_kv[:, : self.q_lora_rank],
                selected_qr_kv[:, self.q_lora_rank :],
            ],
            dim=-1,
        ).contiguous()

    @eager_break_during_capture
    def _capture_qkv_prenorm_shadow_compare(
        self,
        positions: torch.Tensor,
        single_hidden_states: torch.Tensor,
        qr_kv: torch.Tensor,
    ) -> None:
        if not qkv_prenorm_shadow_compare_selected(self.layer_idx, positions):
            return
        rowwise_qr_kv = torch.cat(
            [
                self.fused_wqa_wkv(single_hidden_states[index : index + 1])[
                    0
                ].clone()
                for index in range(single_hidden_states.shape[0])
            ],
            dim=0,
        )
        maybe_capture_qkv_prenorm_shadow_compare(
            self.layer_idx,
            positions,
            qr_kv,
            rowwise_qr_kv,
            self.q_lora_rank,
        )

    @eager_break_during_capture
    def _capture_qkv_producer(
        self,
        positions: torch.Tensor,
        single_hidden_states: torch.Tensor,
        qr_kv: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        maybe_capture_qkv_producer(
            self.layer_idx,
            positions,
            single_hidden_states,
            qr_kv,
            self.q_lora_rank,
            qr,
            kv,
        )

    def _project_target_qkv_tokenwise(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = torch.cat(
            [
                self.fused_wqa_wkv(hidden_states[index : index + 1])[0].clone()
                for index in range(hidden_states.shape[0])
            ],
            dim=0,
        )
        qr, kv = qkv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return (
            self.q_norm(qr).contiguous(),
            self.kv_norm(kv).contiguous(),
        )

    def _project_wq_b_prefill(self, qr: torch.Tensor) -> torch.Tensor:
        """Project Q in cache-block-sized GEMMs for long SWA-only prefills."""
        if (
            not self._prefill_gemm_chunking_enabled
            or qr.shape[0] <= self._prefill_gemm_chunk_size
        ):
            return self.wq_b(qr)

        chunk_size = self._prefill_gemm_chunk_size
        return torch.cat(
            [
                self.wq_b(qr[start : start + chunk_size])
                for start in range(0, qr.shape[0], chunk_size)
            ],
            dim=0,
        )

    def _project_wq_b_tokenwise(self, qr: torch.Tensor) -> torch.Tensor:
        if _native_serial_wq_b_rows_enabled():
            rows = qr.shape[0]
            weight = getattr(self.wq_b, "weight", None)
            if not 2 <= rows <= 6:
                raise RuntimeError(
                    "native serial WQ-B rows require 2 <= rows <= 6"
                )
            if (
                not isinstance(weight, torch.Tensor)
                or qr.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16
                or not qr.is_contiguous()
                or not weight.is_contiguous()
            ):
                raise RuntimeError(
                    "native serial WQ-B rows require contiguous BF16 input "
                    "and weight"
                )
            exact_grouped = _exact_grouped_wq_b_rows_enabled()
            op_name = (
                "gemv_bf16_exact_grouped_rows_out"
                if exact_grouped
                else "gemv_bf16_serial_rows_out"
            )
            try:
                op = getattr(torch.ops._metax_sparse_C, op_name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"native WQ-B rows operator {op_name} is unavailable"
                ) from exc

            key = (rows, weight.shape[0], qr.dtype, qr.device)
            workspaces = getattr(self, "_native_serial_wq_b_workspaces", None)
            if workspaces is None:
                workspaces = {}
                self._native_serial_wq_b_workspaces = workspaces
            output = workspaces.get(key)
            if output is None:
                output = torch.empty(
                    (rows, weight.shape[0]), dtype=qr.dtype, device=qr.device
                )
                workspaces[key] = output
            op(qr, weight, output)
            launches = 1 if exact_grouped else rows
            logger.warning_once(
                "DeepSeek V4 target WQ-B uses native %s output workspace: "
                "rows=%d launches=%d",
                "exact-grouped-row" if exact_grouped else "serial-row",
                rows,
                launches,
            )
            return output
        return torch.cat(
            [
                self.wq_b(qr[index : index + 1]).clone()
                for index in range(qr.shape[0])
            ],
            dim=0,
        )

    def _project_wq_b_selected_rows(
        self,
        qr: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        if selected_indices.numel() == qr.shape[0]:
            return self._project_wq_b_tokenwise(qr)
        q = self.wq_b(qr)
        for index in selected_indices.tolist():
            q[index : index + 1] = self.wq_b(qr[index : index + 1]).clone()
        return q.contiguous()

    def _attention_impl_tokenwise_wq_b(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        selected_indices: torch.Tensor | None = None,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata

        def wq_b_kv_insert() -> torch.Tensor:
            if selected_indices is None:
                q = self._project_wq_b_tokenwise(qr)
            else:
                q = self._project_wq_b_selected_rows(qr, selected_indices)
            q = q.view(
                -1, self.n_local_heads, self.head_dim
            )
            return self._fused_qnorm_rope_kv_insert(
                q, kv, positions, attn_metadata
            )

        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor
            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = wq_b_kv_insert()
        self.forward_mqa(q, kv, positions, out)

    def _attention_impl_wq_b_shadow_compare(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata

        def wq_b_kv_insert() -> torch.Tensor:
            rowwise_q = self._project_wq_b_tokenwise(qr)
            batched_q = self.wq_b(qr)
            maybe_capture_wq_b_shadow_compare(
                self.layer_idx, positions, qr, batched_q, rowwise_q
            )
            q = batched_q.view(-1, self.n_local_heads, self.head_dim)
            return self._fused_qnorm_rope_kv_insert(
                q, kv, positions, attn_metadata
            )

        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor
            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = wq_b_kv_insert()
        self.forward_mqa(q, kv, positions, out)

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        _ATTN_TIMING["enabled"] = os.getenv("VLLM_METAX_DSV4_ATTENTION_TIMING") == "1"
        _do_sync = os.getenv("VLLM_METAX_DSV4_ATTENTION_SYNC") == "1"
        _t0 = time.perf_counter() if _ATTN_TIMING["enabled"] else 0.0
        if _ATTN_TIMING["enabled"]:
            # Track gap between calls (framework overhead between forwards)
            _last_t = _ATTN_TIMING.get("last_call_t", 0.0)
            _gap = _t0 - _last_t if _last_t > 0 else 0.0
            _ATTN_TIMING["last_call_t"] = _t0
            _ATTN_TIMING.setdefault("gaps", [])
            if _gap > 5.0:  # only track large gaps (>5ms = between forwards)
                _ATTN_TIMING["gaps"].append(_gap)
        if _ATTN_TIMING["enabled"] and _do_sync:
            torch.cuda.synchronize()
            _t_sync = time.perf_counter()
        else:
            _t_sync = _t0
        try:
            return self._attention_impl_inner(
                hidden_states, qr, kv, kv_score, indexer_kv_score,
                indexer_weights, positions, out,
            )
        finally:
            if _ATTN_TIMING["enabled"]:
                _t_end = time.perf_counter()
                _ATTN_TIMING["total"] += _t_end - _t0
                _ATTN_TIMING["count"] += 1
                if _do_sync:
                    _ATTN_TIMING.setdefault("sync_total", 0.0)
                    _ATTN_TIMING.setdefault("sync_count", 0)
                    _ATTN_TIMING["sync_total"] += _t_end - _t_sync
                    _ATTN_TIMING["sync_count"] += 1
                if _ATTN_TIMING["count"] % 100 == 0:
                    _rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
                    _gaps = _ATTN_TIMING.get("gaps", [])
                    _gap_total = sum(_gaps)
                    _gap_count = len(_gaps)
                    with open(f"/tmp/attn_timing_rank{_rank}.txt", "w") as _f:
                        _f.write(f"rank={_rank}\n")
                        _f.write(f"total={_ATTN_TIMING['total']:.6f}\n")
                        _f.write(f"count={_ATTN_TIMING['count']}\n")
                        _f.write(f"avg_ms={_ATTN_TIMING['total']/max(1,_ATTN_TIMING['count'])*1000:.4f}\n")
                        _f.write(f"gap_count={_gap_count}\n")
                        _f.write(f"gap_total={_gap_total:.6f}\n")
                        _f.write(f"gap_avg_ms={_gap_total/max(1,_gap_count)*1000:.4f}\n")
                        if _gaps:
                            _f.write(f"gap_first_10={[round(g,4) for g in _gaps[:10]]}\n")

    def _attention_impl_inner(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if (
            self._cached_tw_qkv_enabled
            and self._cached_tw_qkv_layer
            and _target_tokenwise_qkv_position_enabled(positions)
            and 1 < hidden_states.shape[0] <= 6
        ):
            logger.warning_once(
                "DeepSeek V4 speculative target attention uses tokenwise QKV "
                "projection"
            )
            qr, kv = self._project_target_qkv_tokenwise(hidden_states)
        if self._attention_input_capture_enabled:
            maybe_capture_attention_inputs(self.layer_idx, positions, qr, kv)
        if (
            self._cached_is_target_tw_wq_b
            and 1 < hidden_states.shape[0] <= 6
        ):
            target_wq_b_indices = _target_tokenwise_wq_b_selected_indices(
                positions
            )
            if target_wq_b_indices.numel() > 0:
                logger.warning_once(
                    "DeepSeek V4 speculative attention uses tokenwise wq_b projection"
                )
                self._attention_impl_tokenwise_wq_b(
                    hidden_states,
                    qr,
                    kv,
                    kv_score,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    out,
                    target_wq_b_indices,
                )
                if self._attention_input_capture_enabled:
                    maybe_capture_attention_output(self.layer_idx, positions, out)
                return
        if (
            1 < hidden_states.shape[0] <= 6
            and wq_b_shadow_compare_selected(self.layer_idx, positions)
        ):
            logger.warning_once(
                "DeepSeek V4 speculative attention compares batched and "
                "tokenwise wq_b projection"
            )
            self._attention_impl_wq_b_shadow_compare(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                out,
            )
            if self._attention_input_capture_enabled:
                maybe_capture_attention_output(self.layer_idx, positions, out)
            return
        if (
            getattr(self, "_prefill_gemm_chunking_enabled", False)
            and hidden_states.shape[0]
            > getattr(self, "_prefill_gemm_chunk_size", 0)
            and self.indexer is None
            and self.compressor is None
        ):
            # Keep SWA-only prefill wq_b shapes aligned with cache-block
            # boundaries; this preserves native GEMM numerics for prefix hits.
            q = self._project_wq_b_prefill(qr).view(
                -1, self.n_local_heads, self.head_dim
            )
            q = self._fused_qnorm_rope_kv_insert(
                q, kv, positions, get_forward_context().attn_metadata
            )
            self.forward_mqa(q, kv, positions, out)
            if self._attention_input_capture_enabled:
                maybe_capture_attention_output(self.layer_idx, positions, out)
            return
        target_layer = _q_insert_cudagraph_target_layer()
        target_enabled = target_layer == "all" or target_layer == self.layer_idx
        if (
            not target_enabled
            or hidden_states.shape[0] != 1
        ):
            super().attention_impl(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                out,
            )
            if self._attention_input_capture_enabled:
                maybe_capture_attention_output(self.layer_idx, positions, out)
            return

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            super().attention_impl(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                out,
            )
            if self._attention_input_capture_enabled:
                maybe_capture_attention_output(self.layer_idx, positions, out)
            return
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        if (
            swa_metadata is None
            or getattr(swa_metadata, "num_prefills", None) != 0
            or getattr(swa_metadata, "num_decode_tokens", None) != 1
        ):
            super().attention_impl(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                out,
            )
            if self._attention_input_capture_enabled:
                maybe_capture_attention_output(self.layer_idx, positions, out)
            return

        if self._q_insert_cudagraph_capture_active():
            raise RuntimeError(
                "DeepSeek V4 Q/KV-insert CUDA graph cannot nest outer capture"
            )

        compressor = self.compressor

        def wq_b_kv_insert() -> torch.Tensor:
            return self._q_insert_cudagraph_forward(qr, kv, positions, swa_metadata)

        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert compressor is not None
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif compressor is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = wq_b_kv_insert()
        self.forward_mqa(q, kv, positions, out)
        if self._attention_input_capture_enabled:
            maybe_capture_attention_output(self.layer_idx, positions, out)


    @staticmethod
    def _q_insert_cudagraph_capture_active() -> bool:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        if probe is None:
            raise RuntimeError(
                "DeepSeek V4 Q/KV-insert CUDA graph capture status is unavailable"
            )
        try:
            return bool(probe())
        except Exception as exc:
            raise RuntimeError(
                "DeepSeek V4 Q/KV-insert CUDA graph capture status is unavailable"
            ) from exc

    def _q_insert_cudagraph_native(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata,
    ) -> torch.Tensor:
        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        qkv_insert_capture = maybe_prepare_qkv_insert_capture(
            self.layer_idx,
            positions,
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            swa_metadata.block_size,
        )
        _run_qnorm_rope_kv_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        if qkv_insert_capture is not None:
            qkv_insert_capture.finish(swa_kv_cache_2d)
        if self.n_local_heads < self.padded_heads:
            return F.pad(
                q,
                (0, 0, 0, self.padded_heads - self.n_local_heads),
                value=0.0,
            )
        return q

    def _q_insert_cudagraph_compute(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata,
    ) -> torch.Tensor:
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q_capture = (
            maybe_prepare_q_stage_capture(self.layer_idx, positions, q)
            if self._attention_input_capture_enabled
            else None
        )
        output = self._q_insert_cudagraph_native(q, kv, positions, swa_metadata)
        if q_capture is not None:
            q_capture.finish(q)
        return output

    def _q_insert_cudagraph_key(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata,
    ) -> tuple:
        stream = torch.cuda.current_stream(qr.device)
        stream_key = getattr(stream, "cuda_stream", id(stream))
        return (
            _q_insert_tensor_key(qr),
            _q_insert_tensor_key(kv),
            _q_insert_tensor_key(self.wq_b.weight),
            _q_insert_tensor_key(positions),
            _q_insert_tensor_key(self.swa_cache_layer.kv_cache),
            _q_insert_tensor_key(swa_metadata.slot_mapping),
            _q_insert_tensor_key(self.rotary_emb.cos_sin_cache),
            float(self.eps),
            int(swa_metadata.block_size),
            stream_key,
        )

    def _q_insert_cudagraph_forward(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata,
    ) -> torch.Tensor:
        if self._q_insert_cudagraph_capture_active():
            raise RuntimeError(
                "DeepSeek V4 Q/KV-insert CUDA graph cannot nest outer capture"
            )
        key = self._q_insert_cudagraph_key(qr, kv, positions, swa_metadata)
        graphs = self._q_insert_cudagraphs
        cached = graphs.get(key)
        if cached is not None:
            cached[0].replay()
            return cached[1]

        stream = torch.cuda.current_stream(qr.device)
        self._q_insert_cudagraph_compute(qr, kv, positions, swa_metadata)
        stream.synchronize()

        pool = self._q_insert_cudagraph_pool
        if pool is None:
            pool = _q_insert_cudagraph_pool_handle()
            self._q_insert_cudagraph_pool = pool
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            graph_out = self._q_insert_cudagraph_compute(
                qr, kv, positions, swa_metadata
            )
        graph.replay()
        graphs[key] = (graph, graph_out)
        return graph_out

    def clear_q_insert_cudagraph_cache(self) -> None:
        if self._q_insert_cudagraphs:
            torch.cuda.synchronize()
        self._q_insert_cudagraphs.clear()
        self._q_insert_cudagraph_pool = None

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        q_capture = (
            maybe_prepare_q_stage_capture(self.layer_idx, positions, q)
            if self._attention_input_capture_enabled
            else None
        )

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        qkv_insert_capture = maybe_prepare_qkv_insert_capture(
            self.layer_idx,
            positions,
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            swa_metadata.block_size,
        )
        _run_qnorm_rope_kv_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        if qkv_insert_capture is not None:
            qkv_insert_capture.finish(swa_kv_cache_2d)
        if q_capture is not None:
            q_capture.finish(q)
        if self.n_local_heads < self.padded_heads:
            return F.pad(
                q,
                (0, 0, 0, self.padded_heads - self.n_local_heads),
                value=0.0,
            )
        return q

class MacaDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # --------------------------------------
            # Note(Metax): no alignment for indexer kvcache continuity
            # alignment=576,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MacaDeepseekV4IndexerBackend


class MacaDeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "INT8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "int8"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        # ----------------------------------------------
        # Note(Metax): int8 indxer cache use the same layout as FP8:
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        
        
        self.k_cache = MacaDeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )

        self.compressor = MacaDeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )
        self._short_context_pending = False
        self.compressor._clear_initial_overlap = False
        self.compressor._initial_overlap_start = (
            self.config.sliding_window - self.compress_ratio
        )

        self.indexer_op = MacaSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # -----------------------------------------------------
        # None(Metax): maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def _project_wq_b_exact_grouped_rows(self, qr: torch.Tensor) -> torch.Tensor:
        rows = qr.shape[0]
        weight = getattr(self.wq_b, "weight", None)
        expected_weight_shape = (self.n_head * self.head_dim, self.q_lora_rank)
        if not 2 <= rows <= 6:
            raise RuntimeError(
                "exact-grouped indexer WQ-B requires 2 <= rows <= 6"
            )
        if (
            qr.ndim != 2
            or not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != expected_weight_shape
            or qr.shape[1] != self.q_lora_rank
            or qr.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or qr.device != weight.device
            or not qr.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise RuntimeError(
                "exact-grouped indexer WQ-B requires contiguous BF16 "
                f"input [B,{self.q_lora_rank}] and weight "
                f"[{expected_weight_shape[0]},{expected_weight_shape[1]}] "
                "on the same device"
            )
        try:
            op = torch.ops._metax_sparse_C.gemv_bf16_exact_grouped_rows_out
        except AttributeError as exc:
            raise RuntimeError(
                "native exact-grouped indexer WQ-B operator is unavailable"
            ) from exc

        key = (rows, qr.dtype, qr.device)
        workspaces = getattr(self, "_exact_grouped_indexer_wq_b_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._exact_grouped_indexer_wq_b_workspaces = workspaces
        output = workspaces.get(key)
        if output is None:
            output = torch.empty(
                (rows, expected_weight_shape[0]),
                dtype=qr.dtype,
                device=qr.device,
            )
            workspaces[key] = output
        op(qr, weight, output)
        logger.warning_once(
            "DeepSeek V4 target indexer WQ-B uses native exact-grouped-row "
            "workspace: rows=%d launches=1",
            rows,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        swa_metadata = None
        if isinstance(attn_metadata, dict):
            swa_metadata = attn_metadata.get(self.prefix.replace('.indexer', '.swa_cache'))
            if (
                swa_metadata is not None
                and swa_metadata.is_short_context(self.config.sliding_window)
            ):
                self._short_context_pending = True
                assert self.topk_indices_buffer is not None
                self.topk_indices_buffer[: hidden_states.shape[0]].fill_(-1)
                return self.topk_indices_buffer

        compressor = self.compressor
        compressor._tokenwise_min_position = (
            self.config.sliding_window
            if swa_metadata is not None
            and getattr(swa_metadata, "num_prefills", 0) == 0
            else None
        )
        compressor._initial_overlap_boundary = (
            self.config.sliding_window
            if compressor.overlap
            and swa_metadata is not None
            and getattr(swa_metadata, "num_prefills", 0) == 0
            else None
        )
        compressor._clear_initial_overlap = getattr(
            self, "_short_context_pending", False
        )
        self._short_context_pending = False

        def wq_b_and_q_quant():
            def project_quantize(
                qr_chunk: torch.Tensor,
                positions_chunk: torch.Tensor,
                indexer_weights_chunk: torch.Tensor,
                chunk_index: int | None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if chunk_index is not None:
                    logger.warning_once(
                        "DeepSeek V4 speculative indexer uses tokenwise Q "
                        "projection and quantization"
                    )
                # ReplicatedLinear returns (output, bias); bias is None.
                q, _ = self.wq_b(qr_chunk)
                q = q.view(-1, self.n_head, self.head_dim)
                # ----------------------------------------
                # Note: Metax use int8 quant in indexer
                return fused_indexer_q_rope_int8_quant(
                    positions_chunk,
                    q,
                    rotary_emb.cos_sin_cache,
                    indexer_weights_chunk,
                    self.softmax_scale,
                    self.n_head**-0.5,
                )

            if (
                _indexer_tokenwise_wq_b_enabled(self.layer_idx)
                and 1 < qr.shape[0] <= 6
            ):
                split_weight_rows = _indexer_tokenwise_weight_rows_enabled(
                    self.layer_idx, self.config.num_hidden_layers
                )
                if (
                    _exact_grouped_indexer_wq_b_rows_enabled()
                    and self.layer_idx < self.config.num_hidden_layers
                ):
                    if not split_weight_rows:
                        raise RuntimeError(
                            "exact-grouped target indexer WQ-B requires "
                            "token-aligned indexer weight rows"
                        )
                    rows = qr.shape[0]
                    if (
                        positions.ndim != 1
                        or positions.shape[0] != rows
                        or positions.dtype != torch.int64
                        or positions.device != qr.device
                        or indexer_weights.ndim != 2
                        or tuple(indexer_weights.shape) != (rows, self.n_head)
                        or indexer_weights.dtype
                        not in (torch.bfloat16, torch.float32)
                        or indexer_weights.device != qr.device
                    ):
                        logger.warning_once(
                            "DeepSeek V4 target indexer WQ-B exact-grouped "
                            "path skipped: requires int64 positions [B] and "
                            "BF16/FP32 indexer weights [B,%d] aligned with "
                            "input rows on the same device; got "
                            "positions=%s/%s weights=%s/%s",
                            self.n_head,
                            tuple(positions.shape),
                            positions.dtype,
                            tuple(indexer_weights.shape),
                            indexer_weights.dtype,
                        )
                    else:
                        q = self._project_wq_b_exact_grouped_rows(qr).view(
                            -1, self.n_head, self.head_dim
                        )
                        return fused_indexer_q_rope_int8_quant(
                            positions.contiguous(),
                            q,
                            rotary_emb.cos_sin_cache,
                            indexer_weights.contiguous(),
                            self.softmax_scale,
                            self.n_head**-0.5,
                        )
                q_parts = []
                weight_parts = []
                split_current_weight_rows = (
                    split_weight_rows
                    and indexer_weights.ndim >= 1
                    and indexer_weights.shape[0] == qr.shape[0]
                )
                for index in range(qr.shape[0]):
                    q_quant, weights = project_quantize(
                        qr[index : index + 1],
                        positions[index : index + 1],
                        (
                            indexer_weights[index : index + 1]
                            if split_current_weight_rows
                            else indexer_weights
                        ),
                        index,
                    )
                    q_parts.append(q_quant.clone())
                    weight_parts.append(weights.clone())
                return torch.cat(q_parts, dim=0), torch.cat(weight_parts, dim=0)

            return project_quantize(qr, positions, indexer_weights, None)

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
