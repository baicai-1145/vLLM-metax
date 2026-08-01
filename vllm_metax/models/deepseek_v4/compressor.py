# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from vllm.triton_utils import tl, triton

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform

from vllm.models.deepseek_v4.compressor import (
    CompressorMetadata,
    DeepseekCompressor
)

from .ops.fused_compress_quant_cache import (
   compress_norm_rope_store_triton
)

_COMPRESSOR_CAPTURE_CALL_COUNT = 0
_COMPRESSOR_CAPTURE_LOCK = threading.Lock()
_COMPRESSOR_CAPTURE_STATE_ONLY_ENV = (
    "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_STATE_ONLY"
)
_FUSED_SAVE_PARTIAL_STATES_ENV = (
    "VLLM_METAX_DSV4_COMPRESSOR_FUSED_SAVE_PARTIAL_STATES"
)
logger = init_logger(__name__)


@triton.jit
def _zero_initial_overlap_state_kernel(
    state_cache,
    state_stride0,
    state_stride1,
    block_table,
    block_table_stride,
    block_size,
    start_position,
    STATE_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    request_idx = pid // 4
    offset = pid % 4
    logical_position = start_position + offset
    block_index = logical_position // block_size
    block_offset = logical_position % block_size
    physical_block = tl.load(
        block_table + request_idx * block_table_stride + block_index
    )
    state_ptr = (
        state_cache
        + physical_block * state_stride0
        + block_offset * state_stride1
    )
    indices = tl.arange(0, STATE_WIDTH)
    tl.store(state_ptr + indices, 0.0, mask=indices < STATE_WIDTH)


def _zero_initial_overlap_state(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    start_position: int,
) -> None:
    if block_table.ndim != 2 or block_table.shape[0] == 0:
        return
    _zero_initial_overlap_state_kernel[(block_table.shape[0] * 4,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        block_table,
        block_table.stride(0),
        block_size,
        start_position,
        STATE_WIDTH=triton.next_power_of_2(state_cache.shape[-1]),
    )


def _rank() -> str:
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


def _is_cuda_graph_capturing() -> bool:
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _parse_int_set(value: str | None, env_name: str) -> set[int] | None:
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None
    selected: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        try:
            selected.add(int(token))
        except ValueError as exc:
            raise ValueError(
                f"{env_name} must be a comma-separated set of integers"
            ) from exc
    return selected


def _layer_from_prefix(prefix: Any) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(prefix))
    return int(match.group(1)) if match else None


def _rank_enabled(rank: str) -> bool:
    ranks = os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_RANKS", "all")
    if not ranks or ranks.strip().lower() == "all":
        return True
    return rank in {item.strip() for item in ranks.split(",") if item.strip()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _CompressorCapture:
    def __init__(
        self,
        path: Path,
        payload: dict[str, Any],
        kv_cache: torch.Tensor,
        cache_block_indices: torch.Tensor,
        cache_slot_offsets: torch.Tensor,
        head_dim: int,
    ) -> None:
        self.path = path
        self.payload = payload
        self.kv_cache = kv_cache
        self.cache_block_indices = cache_block_indices
        self.cache_slot_offsets = cache_slot_offsets
        self.head_dim = head_dim

    def finish(self) -> Path:
        self.payload["kv_cache_rows_after"] = _capture_kv_cache_rows(
            self.kv_cache,
            self.cache_block_indices,
            self.cache_slot_offsets,
            self.head_dim,
        )
        _atomic_torch_save(self.payload, self.path)
        return self.path


def _capture_kv_cache_rows(
    kv_cache: torch.Tensor,
    cache_block_indices: torch.Tensor,
    cache_slot_offsets: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Copy selected cache rows, accounting for MetaX's planar INT8 layout."""
    if cache_block_indices.numel() == 0:
        return torch.empty(
            (0, kv_cache.shape[-1]), dtype=kv_cache.dtype, device="cpu"
        )
    if kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == head_dim + 4:
        block_size = kv_cache.shape[1]
        flat = kv_cache.reshape(kv_cache.shape[0], -1)
        value_offsets = cache_slot_offsets[:, None] * head_dim + torch.arange(
            head_dim, device=kv_cache.device
        )[None, :]
        scale_offsets = (
            block_size * head_dim
            + cache_slot_offsets[:, None] * 4
            + torch.arange(4, device=kv_cache.device)[None, :]
        )
        values = flat[cache_block_indices[:, None], value_offsets]
        scales = flat[cache_block_indices[:, None], scale_offsets]
        return torch.cat((values, scales), dim=1).detach().cpu().clone()
    return (
        kv_cache[cache_block_indices, cache_slot_offsets]
        .detach()
        .cpu()
        .clone()
    )


def _maybe_prepare_compressor_capture(
    *,
    layer_prefix: Any,
    positions: torch.Tensor,
    kv_score: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    token_to_req_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> _CompressorCapture | None:
    capture_dir_value = os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_DIR")
    if not capture_dir_value or _is_cuda_graph_capturing():
        return None
    rank = _rank()
    if not _rank_enabled(rank):
        return None
    layer_idx = _layer_from_prefix(layer_prefix)
    selected_layers = _parse_int_set(
        os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_LAYERS"),
        "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_LAYERS",
    )
    if selected_layers is not None and (
        layer_idx is None or layer_idx not in selected_layers
    ):
        return None
    selected_head_dims = _parse_int_set(
        os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_HEAD_DIMS"),
        "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_HEAD_DIMS",
    )
    if selected_head_dims is not None and head_dim not in selected_head_dims:
        return None

    selected_slots = _parse_int_set(
        os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_SLOTS"),
        "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_SLOTS",
    )
    selected_positions = _parse_int_set(
        os.getenv("VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_POSITIONS"),
        "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_POSITIONS",
    )
    state_only = os.getenv(_COMPRESSOR_CAPTURE_STATE_ONLY_ENV) == "1"
    if state_only and selected_positions is None:
        raise ValueError(
            f"{_COMPRESSOR_CAPTURE_STATE_ONLY_ENV}=1 requires "
            "VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_POSITIONS"
        )
    positions_cpu = positions.detach().cpu()
    state_slots_cpu = slot_mapping.detach().cpu()
    kv_slots_cpu = kv_slot_mapping.detach().cpu()
    if state_only:
        write_mask = state_slots_cpu >= 0
    else:
        write_mask = (
            ((positions_cpu + 1) % compress_ratio == 0)
            & (kv_slots_cpu >= 0)
        )
    if selected_slots is not None:
        write_mask &= torch.tensor(
            [int(slot) in selected_slots for slot in kv_slots_cpu],
            dtype=torch.bool,
        )
    if selected_positions is not None:
        write_mask &= torch.tensor(
            [int(position) in selected_positions for position in positions_cpu],
            dtype=torch.bool,
        )
    token_indices = torch.nonzero(write_mask, as_tuple=False).flatten()
    if token_indices.numel() == 0:
        return None

    capture_dir = Path(capture_dir_value)
    capture_dir.mkdir(parents=True, exist_ok=True)
    global _COMPRESSOR_CAPTURE_CALL_COUNT
    with _COMPRESSOR_CAPTURE_LOCK:
        call = _COMPRESSOR_CAPTURE_CALL_COUNT
        _COMPRESSOR_CAPTURE_CALL_COUNT += 1
    path = capture_dir / f"rank{rank}_layer{layer_idx}_call{call}_compressor.pt"

    selected_positions_cpu = positions_cpu[token_indices]
    selected_kv_slots_cpu = kv_slots_cpu[token_indices].to(torch.long)
    valid_kv_slots = selected_kv_slots_cpu >= 0
    valid_selected_kv_slots = selected_kv_slots_cpu[valid_kv_slots]
    cache_block_indices = valid_selected_kv_slots // kv_cache.shape[1]
    cache_slot_offsets = valid_selected_kv_slots % kv_cache.shape[1]
    cache_block_indices_device = cache_block_indices.to(kv_cache.device)
    cache_slot_offsets_device = cache_slot_offsets.to(kv_cache.device)

    state_positions: list[list[int]] = []
    state_block_indices: list[list[int]] = []
    state_slot_offsets: list[list[int]] = []
    state_rows: list[torch.Tensor] = []
    span = (1 + int(overlap)) * compress_ratio
    token_to_req_cpu = token_to_req_indices.detach().cpu()
    for token_index, position in zip(
        token_indices.tolist(), selected_positions_cpu.tolist()
    ):
        req_idx = int(token_to_req_cpu[token_index])
        start = int(position) - span + 1
        row_positions = [start + offset for offset in range(span)]
        row_block_indices: list[int] = []
        row_slot_offsets: list[int] = []
        rows: list[torch.Tensor] = []
        for row_position in row_positions:
            if row_position < 0:
                row_block_indices.append(-1)
                row_slot_offsets.append(-1)
                rows.append(torch.zeros(state_cache.shape[-1], dtype=state_cache.dtype))
                continue
            block_index = row_position // block_size
            slot_offset = row_position % block_size
            block_number = int(block_table[req_idx, block_index].detach().cpu())
            row_block_indices.append(block_number)
            row_slot_offsets.append(slot_offset)
            rows.append(state_cache[block_number, slot_offset].detach().cpu().clone())
        state_positions.append(row_positions)
        state_block_indices.append(row_block_indices)
        state_slot_offsets.append(row_slot_offsets)
        state_rows.append(torch.stack(rows))

    payload = {
        "layer_idx": layer_idx,
        "layer_prefix": str(layer_prefix),
        "rank": rank,
        "call": call,
        "compress_ratio": compress_ratio,
        "overlap": bool(overlap),
        "head_dim": head_dim,
        "block_size": block_size,
        "state_width": state_width,
        "token_indices": token_indices.cpu().clone(),
        "positions": selected_positions_cpu.clone(),
        "slot_mapping": slot_mapping.detach().cpu()[token_indices].clone(),
        "kv_slot_mapping": selected_kv_slots_cpu.clone(),
        "token_to_req_indices": token_to_req_cpu[token_indices].clone(),
        "kv_score": kv_score.detach().cpu()[token_indices].clone(),
        "kv": kv.detach().cpu()[token_indices].clone(),
        "score": score.detach().cpu()[token_indices].clone(),
        "state_positions": torch.tensor(state_positions, dtype=torch.long),
        "state_block_indices": torch.tensor(state_block_indices, dtype=torch.long),
        "state_slot_offsets": torch.tensor(state_slot_offsets, dtype=torch.long),
        "state_cache_rows": torch.stack(state_rows),
        "cache_block_indices": cache_block_indices.clone(),
        "cache_slot_offsets": cache_slot_offsets.clone(),
        "kv_cache_token_indices": token_indices[valid_kv_slots].cpu().clone(),
        "kv_cache_rows_before": _capture_kv_cache_rows(
            kv_cache,
            cache_block_indices_device,
            cache_slot_offsets_device,
            head_dim,
        ),
    }
    return _CompressorCapture(
        path,
        payload,
        kv_cache,
        cache_block_indices_device,
        cache_slot_offsets_device,
        head_dim,
    )


class MacaDeepseekCompressor(DeepseekCompressor):

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache
        if getattr(self, "_clear_initial_overlap", False):
            _zero_initial_overlap_state(
                state_cache,
                state_metadata.block_table,
                block_size,
                int(getattr(self, "_initial_overlap_start", 0)),
            )
            self._clear_initial_overlap = False
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        # ---------------------------------------------
        # Note: Metax not support pdl
        pdl_kwargs = (
            {}
            if current_platform.is_out_of_tree()
            else {"launch_pdl": False}
        )

        cos_sin_cache = rotary_emb.cos_sin_cache
        # -----------------------------------------------
        # Note: Metax use full attn bf16 + indexer int8
        compress_norm_rope_store_fn = compress_norm_rope_store_triton

        def save_rows(token_slice: slice) -> None:
            save_partial_states(
                kv=kv[token_slice],
                score=score[token_slice],
                ape=self.ape,
                positions=positions[token_slice],
                state_cache=state_cache,
                slot_mapping=slot_mapping[token_slice],
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )

        def compress_rows(
            token_slice: slice, *, fuse_save_partial_states: bool = False
        ) -> None:
            sliced_k_cache_metadata = SimpleNamespace(
                slot_mapping=k_cache_metadata.slot_mapping[token_slice],
            )
            capture = _maybe_prepare_compressor_capture(
                layer_prefix=self.k_cache_prefix,
                positions=positions[token_slice],
                kv_score=kv_score[token_slice],
                kv=kv[token_slice],
                score=score[token_slice],
                state_cache=state_cache,
                slot_mapping=slot_mapping[token_slice],
                block_table=block_table,
                block_size=block_size,
                state_width=state_width,
                token_to_req_indices=token_to_req_indices[token_slice],
                kv_cache=kv_cache,
                kv_slot_mapping=sliced_k_cache_metadata.slot_mapping,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
            )
            compress_norm_rope_store_fn(
                state_cache=state_cache,
                num_actual=positions[token_slice].shape[0],
                token_to_req_indices=token_to_req_indices[token_slice],
                positions=positions[token_slice],
                slot_mapping=slot_mapping[token_slice],
                block_table=block_table,
                block_size=block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=sliced_k_cache_metadata,
                pdl_kwargs=pdl_kwargs,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                use_fp4_cache=self.use_fp4_cache,
                rms_norm_weight=self.norm.weight,
                rms_norm_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
                token_stride=self._token_stride,
                scale_dim=self._scale_dim,
                initial_overlap_boundary=getattr(
                    self, "_initial_overlap_boundary", None
                ),
                kv=kv[token_slice] if fuse_save_partial_states else None,
                score=score[token_slice] if fuse_save_partial_states else None,
                ape=self.ape if fuse_save_partial_states else None,
                fuse_save_partial_states=fuse_save_partial_states,
            )
            if capture is not None:
                capture.finish()

        if (
            os.getenv("VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR") == "1"
            and 1 < num_actual <= 6
        ):
            fuse_save_partial_states = (
                os.getenv(_FUSED_SAVE_PARTIAL_STATES_ENV) == "1"
            )
            if fuse_save_partial_states:
                logger.warning_once(
                    "DeepSeek V4 tokenwise compressor fuses partial-state save "
                    "into the native compress kernel"
                )
            min_position = getattr(self, "_tokenwise_min_position", None)
            for index in range(num_actual):
                # The compressor is called only from attention_impl's
                # eager-break segment, so this data-dependent branch is not
                # recorded in the surrounding CUDA graph.
                if (
                    min_position is not None
                    and positions[index].item() < min_position
                ):
                    continue
                token_slice = slice(index, index + 1)
                if not fuse_save_partial_states:
                    save_rows(token_slice)
                compress_rows(
                    token_slice,
                    fuse_save_partial_states=fuse_save_partial_states,
                )
        else:
            save_rows(slice(0, num_actual))
            compress_rows(slice(0, num_actual))
