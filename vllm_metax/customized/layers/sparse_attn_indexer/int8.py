# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Custom Sparse Attention Indexer layers."""

import json
import os
import threading
from pathlib import Path

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_metax.utils.deep_gemm import (
    int8_mqa_logits,
    int8_paged_mqa_logits,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm_metax.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view


from vllm_metax import _custom_ops as mx_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_INDEXER_CACHE_LAYOUT_LOG_ENV = "VLLM_METAX_DSV4_INDEXER_CACHE_LAYOUT_LOG"
_INDEXER_CACHE_LAYOUT_LOGGED = False
_INDEXER_CACHE_LAYOUT_LOG_LOCK = threading.Lock()


def _indexer_cache_layout_is_standard(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int,
    is_contiguous: bool,
) -> bool:
    """Return whether metadata describes a zero-offset contiguous tensor."""
    if storage_offset != 0 or not is_contiguous:
        return False
    expected_stride: list[int] = []
    running = 1
    for size in reversed(shape):
        expected_stride.append(running)
        running *= size
    return stride == tuple(reversed(expected_stride))


def _indexer_cache_layout_log_path(setting: str, rank: str) -> Path:
    if setting.strip().lower() in {"1", "true", "yes", "on"}:
        return Path("/root/vLLM-metax/.logs") / (
            f"dsv4_indexer_cache_layout_rank{rank}.jsonl"
        )
    return Path(setting)


def _indexer_cache_layout_rank() -> str:
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if rank is not None:
        return rank
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return str(os.getpid())


def _maybe_log_indexer_cache_layout(kv_cache: torch.Tensor) -> None:
    """Write one cache-layout metadata record when explicitly enabled."""
    global _INDEXER_CACHE_LAYOUT_LOGGED
    setting = os.getenv(_INDEXER_CACHE_LAYOUT_LOG_ENV)
    if not setting or setting.strip().lower() in {"0", "false", "off", "no"}:
        return
    with _INDEXER_CACHE_LAYOUT_LOG_LOCK:
        if _INDEXER_CACHE_LAYOUT_LOGGED:
            return
        _INDEXER_CACHE_LAYOUT_LOGGED = True

    rank = _indexer_cache_layout_rank()
    shape = tuple(int(value) for value in kv_cache.shape)
    stride = tuple(int(value) for value in kv_cache.stride())
    metadata = {
        "schema_version": 1,
        "rank": int(rank) if rank.isdigit() else rank,
        "shape": list(shape),
        "stride": list(stride),
        "dtype": str(kv_cache.dtype),
        "storage_offset": int(kv_cache.storage_offset()),
        "is_contiguous": bool(kv_cache.is_contiguous()),
    }
    metadata["is_standard_layout"] = _indexer_cache_layout_is_standard(
        shape,
        stride,
        metadata["storage_offset"],
        metadata["is_contiguous"],
    )
    path = _indexer_cache_layout_log_path(setting, rank)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(metadata, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    except OSError:
        logger.exception("Failed to write indexer cache layout log: %s", path)


def _reset_indexer_cache_layout_log_state() -> None:
    """Reset the one-record budget for unit tests."""
    global _INDEXER_CACHE_LAYOUT_LOGGED
    with _INDEXER_CACHE_LAYOUT_LOG_LOCK:
        _INDEXER_CACHE_LAYOUT_LOGGED = False


def _fill_topk_indices_torch(logits: torch.Tensor, topk_indices: torch.Tensor) -> None:
    k = min(topk_indices.shape[-1], logits.shape[-1])
    topk_indices.fill_(-1)
    if k == 0:
        return
    topk = torch.topk(logits, k=k, dim=-1).indices.to(torch.int32)
    topk_indices_view = topk_indices.reshape(-1, topk_indices.shape[-1])
    topk_view = topk.reshape(-1, k)
    topk_indices_view[:, :k].copy_(topk_view)


def _gather_workspace_shapes_int8(
    total_seq_lens: int,
    head_dim: int,
    int8_dtype: torch.dtype,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace."""
    return (
        ((total_seq_lens, head_dim), int8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


@eager_break_during_capture
def sparse_attn_indexer_int8(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata

    # ----------------------------------------------
    # Metax Note: we use int8 here
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes_int8(
            total_seq_lens, head_dim, torch.int8
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_int8_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    assert use_fp4_cache is False, "not supported"
    assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None, "must set skip_k_cache_insert=True for k is None"

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes_int8(
            total_seq_lens, head_dim, torch.int8
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                mx_ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_slice_cast = q_slice
            k_quant_cast = k_quant
            k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = int8_mqa_logits(
                q_slice_cast,
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            _fill_topk_indices_torch(logits, topk_indices)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, False)
        _maybe_log_indexer_cache_layout(kv_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            padded_q_quant_decode_tokens = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens
            )
            padded_q_scale = None  # noqa: F841
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            padded_q_scale = None  # noqa: F841
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = padded_q_quant_decode_tokens
        logits = int8_paged_mqa_logits(
            padded_q_quant_cast,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        _fill_topk_indices_torch(logits, topk_indices)

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_int8_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="mx_sparse_attn_indexer_int8",
    op_func=sparse_attn_indexer_int8,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_int8_fake,
    dispatch_key=current_platform.dispatch_key,
)
