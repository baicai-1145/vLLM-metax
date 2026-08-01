"""Opt-in, eager-only DeepSeek V4 decoder-layer stage capture.

The helper is deliberately inert unless ``VLLM_METAX_DSV4_LAYER_CAPTURE_DIR``
is set.  It captures selected layers on the configured ranks and never
performs tensor copies or filesystem work while a CUDA graph is being captured.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
_CAPTURE_LAYERS_ENV = "VLLM_METAX_DSV4_LAYER_CAPTURE_LAYERS"
_CAPTURE_POSITIONS_ENV = "VLLM_METAX_DSV4_LAYER_CAPTURE_POSITIONS"
_CAPTURE_FULL_TENSORS_ENV = "VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS"
_CAPTURE_RANKS_ENV = "VLLM_METAX_DSV4_LAYER_CAPTURE_RANKS"
_GRAPH_CAPTURE_DIR_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_DIR"
_GRAPH_CAPTURE_LAYERS_ENV = "VLLM_METAX_DSV4_GRAPH_CAPTURE_LAYERS"
_GRAPH_WEAK_LAYERS_ENV = "VLLM_METAX_DSV4_GRAPH_WEAK_CAPTURE_LAYERS"
_GRAPH_WEAK_STAGES_ENV = "VLLM_METAX_DSV4_GRAPH_WEAK_CAPTURE_STAGES"
_QKV_INSERT_CAPTURE_DIR_ENV = "VLLM_METAX_DSV4_QKV_INSERT_CAPTURE_DIR"
_QKV_INSERT_CAPTURE_POSITIONS_ENV = "VLLM_METAX_DSV4_QKV_INSERT_CAPTURE_POSITIONS"
_WQ_B_SHADOW_COMPARE_DIR_ENV = "VLLM_METAX_DSV4_WQ_B_SHADOW_COMPARE_DIR"
_QKV_PRENORM_SHADOW_COMPARE_DIR_ENV = (
    "VLLM_METAX_DSV4_QKV_PRENORM_SHADOW_COMPARE_DIR"
)
_MHC_PRE_SHADOW_COMPARE_DIR_ENV = "VLLM_METAX_DSV4_MHC_PRE_SHADOW_COMPARE_DIR"
_CALL_INDICES: dict[int, int] = {}
_ATTENTION_CALL_INDICES: dict[int, int] = {}
_ATTENTION_OUTPUT_CALL_INDICES: dict[int, int] = {}
_Q_STAGE_CALL_INDICES: dict[int, int] = {}
_QKV_PRODUCER_CALL_INDICES: dict[int, int] = {}
_MHC_PRE_SHADOW_CALL_INDICES: dict[int, int] = {}
_ATTENTION_LAST_CALL: dict[int, int] = {}
_Q_STAGE_CAPTURED_CALLS: set[tuple[int, int]] = set()
_CALL_LOCK = threading.Lock()
_SAVE_LOCK = threading.Lock()


def reset_layer_capture_state() -> None:
    """Reset the process-local call counter (used by tests)."""
    with _CALL_LOCK:
        _CALL_INDICES.clear()
        _ATTENTION_CALL_INDICES.clear()
        _ATTENTION_OUTPUT_CALL_INDICES.clear()
        _Q_STAGE_CALL_INDICES.clear()
        _QKV_PRODUCER_CALL_INDICES.clear()
        _MHC_PRE_SHADOW_CALL_INDICES.clear()
        _ATTENTION_LAST_CALL.clear()
        _Q_STAGE_CAPTURED_CALLS.clear()


def _capture_dir() -> Path | None:
    value = os.getenv("VLLM_METAX_DSV4_LAYER_CAPTURE_DIR")
    return Path(value) if value else None


def _wq_b_shadow_compare_dir() -> Path | None:
    value = os.getenv(_WQ_B_SHADOW_COMPARE_DIR_ENV)
    return Path(value) if value else None


def _qkv_prenorm_shadow_compare_dir() -> Path | None:
    value = os.getenv(_QKV_PRENORM_SHADOW_COMPARE_DIR_ENV)
    return Path(value) if value else None


def _mhc_pre_shadow_compare_dir() -> Path | None:
    value = os.getenv(_MHC_PRE_SHADOW_COMPARE_DIR_ENV)
    return Path(value) if value else None


def layer_capture_enabled() -> bool:
    """Return whether a model constructed now should enable layer capture."""
    return _capture_dir() is not None


def wq_b_shadow_compare_enabled() -> bool:
    """Return whether target wq_b batched-vs-rowwise shadow comparison is enabled."""
    return _wq_b_shadow_compare_dir() is not None


def qkv_prenorm_shadow_compare_enabled() -> bool:
    """Return whether target pre-norm QKV batched-vs-rowwise shadow compare is enabled."""
    return _qkv_prenorm_shadow_compare_dir() is not None


def mhc_pre_shadow_compare_enabled() -> bool:
    """Return whether same-input batched-vs-rowwise MHC pre capture is enabled."""
    return _mhc_pre_shadow_compare_dir() is not None


def qkv_prenorm_shadow_compare_selected(
    layer_idx: int,
    positions: torch.Tensor,
) -> bool:
    """Return whether pre-norm QKV rowwise shadow work should run."""
    capture_dir = _qkv_prenorm_shadow_compare_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    return (
        capture_dir is not None
        and not _is_cuda_graph_capturing()
        and layer_capture_layer_enabled(layer_idx)
        and _rank_capture_enabled(rank, capture_ranks)
        and _position_capture_enabled(positions)
    )


def wq_b_shadow_compare_selected(
    layer_idx: int,
    positions: torch.Tensor,
) -> bool:
    """Return whether rowwise shadow work should run for this layer and position."""
    capture_dir = _wq_b_shadow_compare_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    return (
        capture_dir is not None
        and not _is_cuda_graph_capturing()
        and layer_capture_layer_enabled(layer_idx)
        and _rank_capture_enabled(rank, capture_ranks)
        and _position_capture_enabled(positions)
    )


def mhc_pre_shadow_compare_selected(
    layer_idx: int,
    positions: torch.Tensor,
) -> bool:
    """Return whether MHC pre rowwise shadow work should run."""
    capture_dir = _mhc_pre_shadow_compare_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    return (
        capture_dir is not None
        and not _is_cuda_graph_capturing()
        and layer_capture_layer_enabled(layer_idx)
        and _rank_capture_enabled(rank, capture_ranks)
        and _position_capture_enabled(positions)
    )


def layer_capture_layer_enabled(layer_idx: int) -> bool:
    """Return whether capture is selected for ``layer_idx``."""
    value = os.getenv(_CAPTURE_LAYERS_ENV)
    tokens = [] if value is None else [item.strip() for item in value.split(",")]
    try:
        selected = {int(item) for item in tokens if item} or {0}
    except ValueError as exc:
        raise ValueError(
            f"{_CAPTURE_LAYERS_ENV} must be a comma-separated set of "
            "nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_CAPTURE_LAYERS_ENV} must be a comma-separated set of "
            "nonnegative integer layer indices"
        )
    return layer_idx in selected


def graph_layer_capture_layer_enabled(layer_idx: int) -> bool:
    """Return whether post-replay capture should retain this layer's outputs."""
    if not os.getenv(_GRAPH_CAPTURE_DIR_ENV):
        return False
    value = os.getenv(_GRAPH_CAPTURE_LAYERS_ENV)
    if value is None or not value.strip():
        return False
    if value.strip().lower() == "all":
        return True
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_GRAPH_CAPTURE_LAYERS_ENV} must be 'all' or a comma-separated "
            "set of nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{_GRAPH_CAPTURE_LAYERS_ENV} must be 'all' or a comma-separated "
            "set of nonnegative integer layer indices"
        )
    return layer_idx in selected


def graph_weak_capture_stages(layer_idx: int) -> set[str]:
    """Return no-retention graph stages selected for ``layer_idx``."""
    if not os.getenv(_GRAPH_CAPTURE_DIR_ENV):
        return set()
    value = os.getenv(_GRAPH_WEAK_LAYERS_ENV, "").strip()
    if not value:
        return set()
    if value.lower() == "all":
        selected = None
    else:
        try:
            selected = {
                int(item.strip()) for item in value.split(",") if item.strip()
            }
        except ValueError as exc:
            raise ValueError(
                f"{_GRAPH_WEAK_LAYERS_ENV} must be 'all' or comma-separated "
                "nonnegative integers"
            ) from exc
        if any(index < 0 for index in selected):
            raise ValueError(
                f"{_GRAPH_WEAK_LAYERS_ENV} must be 'all' or comma-separated "
                "nonnegative integers"
            )
    if selected is not None and layer_idx not in selected:
        return set()
    stages = {
        item.strip()
        for item in os.getenv(_GRAPH_WEAK_STAGES_ENV, "").split(",")
        if item.strip()
    }
    allowed = {"before_mhc", "after_attention", "after_ffn"}
    if not stages or stages - allowed:
        raise ValueError(
            f"{_GRAPH_WEAK_STAGES_ENV} must select one or more of "
            "before_mhc,after_attention,after_ffn"
        )
    return stages


def graph_layer_workspace_outputs(
    workspace: Mapping[tuple[int, str, int], Mapping[str, torch.Tensor]],
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return the one-token FFN exact-MHC workspace capture buffers."""
    for (weight_ptr, _, num_tokens), buffers in workspace.items():
        if weight_ptr == norm_weight.data_ptr() and num_tokens == 1:
            return (
                buffers["normalized"],
                buffers["residual_cur"],
                buffers["post_mix"].unsqueeze(-1),
                buffers["comb_mix"],
            )
    return None


def copy_graph_layer_decode_output_to_workspace(
    workspace: Mapping[tuple[int, str, int], Mapping[str, torch.Tensor]],
    norm_weight: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Stash the final FFN output in an otherwise-dead exact-MHC buffer."""
    if hidden_states.ndim < 1 or hidden_states.shape[0] != 1:
        return
    outputs = graph_layer_workspace_outputs(workspace, norm_weight)
    if outputs is not None:
        outputs[0].copy_(hidden_states)


def graph_layer_workspace_mhc_input(
    workspace: Mapping[tuple[int, str, int], Mapping[str, torch.Tensor]],
    norm_weight: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor | None:
    """Return the FP32 slot used to retain a one-token MHC input."""
    for (weight_ptr, _, num_tokens), buffers in workspace.items():
        if weight_ptr == norm_weight.data_ptr() and num_tokens == 1:
            return buffers["residual_fp32"][:, :hidden_size]
    return None


def copy_graph_layer_mhc_input_to_workspace(
    workspace: Mapping[tuple[int, str, int], Mapping[str, torch.Tensor]],
    norm_weight: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Stash the original MHC hidden input after the native op has finished."""
    if hidden_states.ndim < 1 or hidden_states.shape[0] != 1:
        return
    output = graph_layer_workspace_mhc_input(
        workspace, norm_weight, hidden_states.shape[-1]
    )
    if output is not None:
        output.copy_(hidden_states)


def _capture_ranks() -> set[int] | None:
    """Return selected ranks, or ``None`` for the explicit ``all`` value."""
    value = os.getenv(_CAPTURE_RANKS_ENV)
    if value is None or not value.strip():
        return {0}
    if value.strip().lower() == "all":
        return None
    ranks: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        try:
            rank = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{_CAPTURE_RANKS_ENV} must be 'all' or a comma-separated "
                "set of nonnegative integer ranks"
            ) from exc
        if rank < 0:
            raise ValueError(
                f"{_CAPTURE_RANKS_ENV} must be 'all' or a comma-separated "
                "set of nonnegative integer ranks"
            )
        ranks.add(rank)
    if not ranks:
        raise ValueError(
            f"{_CAPTURE_RANKS_ENV} must be 'all' or a comma-separated "
            "set of nonnegative integer ranks"
        )
    return ranks


def _rank_capture_enabled(
    rank: int | None, selected: set[int] | None = None
) -> bool:
    if selected is None:
        selected = _capture_ranks()
    return rank is not None and (selected is None or rank in selected)


def _rank() -> int | None:
    value = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        return None
    return 0


def _is_cuda_graph_capturing() -> bool:
    try:
        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else True
    except Exception:
        return True


def _call_filter() -> set[int] | None:
    value = os.getenv("VLLM_METAX_DSV4_LAYER_CAPTURE_CALLS")
    if value is None or not value.strip():
        return None
    calls: set[int] = set()
    for item in value.split(","):
        try:
            calls.add(int(item.strip()))
        except ValueError:
            continue
    return calls


def _capture_positions() -> set[int] | None:
    value = os.getenv(_CAPTURE_POSITIONS_ENV)
    if value is None or not value.strip():
        return None
    positions: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            positions.add(int(token))
        except ValueError as exc:
            raise ValueError(
                f"{_CAPTURE_POSITIONS_ENV} must be a comma-separated set of "
                "integer positions"
            ) from exc
    return positions or None


def _position_capture_enabled(positions: torch.Tensor | None) -> bool:
    selected_positions = _capture_positions()
    if selected_positions is None:
        return True
    if not isinstance(positions, torch.Tensor):
        return False
    position_values = positions.detach().reshape(-1)
    for position in selected_positions:
        if bool(torch.any(position_values == position).item()):
            return True
    return False


def _next_call_index(layer_idx: int) -> int:
    with _CALL_LOCK:
        value = _CALL_INDICES.get(layer_idx, 0)
        _CALL_INDICES[layer_idx] = value + 1
    return value


def _next_attention_call_index(layer_idx: int) -> int:
    with _CALL_LOCK:
        value = _ATTENTION_CALL_INDICES.get(layer_idx, 0)
        _ATTENTION_CALL_INDICES[layer_idx] = value + 1
    return value


def _next_attention_output_call_index(layer_idx: int) -> int:
    with _CALL_LOCK:
        value = _ATTENTION_OUTPUT_CALL_INDICES.get(layer_idx, 0)
        _ATTENTION_OUTPUT_CALL_INDICES[layer_idx] = value + 1
    return value


def _next_q_stage_call_index(layer_idx: int) -> int:
    with _CALL_LOCK:
        value = _Q_STAGE_CALL_INDICES.get(layer_idx, 0)
        _Q_STAGE_CALL_INDICES[layer_idx] = value + 1
    return value
def _next_qkv_producer_call_index(layer_idx: int) -> int:
    with _CALL_LOCK:
        value = _QKV_PRODUCER_CALL_INDICES.get(layer_idx, 0)
        _QKV_PRODUCER_CALL_INDICES[layer_idx] = value + 1
    return value


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def _cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().contiguous().cpu().clone()


def _last_token(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.ndim == 0:
        return _cpu_clone(value)
    return _cpu_clone(value[-1:])


def _capture_full_tensors_enabled() -> bool:
    value = os.getenv(_CAPTURE_FULL_TENSORS_ENV, "0")
    try:
        return bool(int(value))
    except ValueError as exc:
        raise ValueError(f"{_CAPTURE_FULL_TENSORS_ENV} must be an integer") from exc


def _qkv_insert_capture_dir() -> Path | None:
    value = os.getenv(_QKV_INSERT_CAPTURE_DIR_ENV)
    return Path(value) if value else None


def _qkv_insert_capture_positions() -> set[int] | None:
    value = os.getenv(_QKV_INSERT_CAPTURE_POSITIONS_ENV)
    if value is None or not value.strip():
        return None
    positions: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            position = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{_QKV_INSERT_CAPTURE_POSITIONS_ENV} must be a comma-separated "
                "set of integer positions"
            ) from exc
        positions.add(position)
    return positions or None


def _qkv_insert_selected_token_indices(positions: torch.Tensor) -> torch.Tensor | None:
    selected_positions = _qkv_insert_capture_positions()
    if selected_positions is None:
        return torch.arange(positions.reshape(-1).shape[0], device=positions.device)
    position_values = positions.reshape(-1)
    mask = torch.zeros_like(position_values, dtype=torch.bool)
    for position in selected_positions:
        mask |= position_values == position
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    return indices if indices.numel() else None


def _compact(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        if tensor.numel() > 4096:
            return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        data = tensor.tolist()
        return data.item() if hasattr(data, "item") else data
    if isinstance(value, Mapping):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value[:4096]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return sorted(str(key) for key in value)
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return sorted(str(key) for key in values)
    return []


def _attention_metadata() -> dict[str, Any]:
    try:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
    except Exception:
        metadata = None
    if not isinstance(metadata, Mapping):
        return {"mode": "unknown", "metadata_keys": []}

    source_key = None
    source = None
    for key, value in metadata.items():
        if hasattr(value, "prefill_gather_lens") and hasattr(
            value, "num_prefills"
        ):
            source_key = str(key)
            source = value
            break
    if source is None:
        return {"mode": "unknown", "metadata_keys": _metadata_keys(metadata)}

    names = (
        "num_prefills",
        "num_decodes",
        "num_decode_tokens",
        "num_prefill_tokens",
        "seq_lens",
        "prefill_seq_lens",
        "prefill_gather_lens",
        "query_start_loc_cpu",
        "token_to_req_indices",
    )
    values: dict[str, Any] = {}
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            values[name] = _compact(value)

    num_prefills = values.get("num_prefills", 0) or 0
    num_decodes = values.get("num_decodes", 0) or 0
    if num_decodes:
        mode = "decode"
    elif num_prefills:
        mode = "prefill"
    else:
        mode = "unknown"
    return {
        "mode": mode,
        "source_key": source_key,
        "metadata_keys": _metadata_keys(metadata),
        **values,
    }


class LayerCaptureContext:
    def __init__(
        self,
        capture_dir: Path,
        rank: int,
        layer_idx: int,
        call: int,
        positions: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> None:
        self.capture_dir = capture_dir
        self.rank = rank
        self.layer_idx = layer_idx
        self.call = call
        self.positions = _cpu_clone(positions)
        self.input_ids = _cpu_clone(input_ids)
        self.attention_metadata = _attention_metadata()

    @torch.no_grad()
    def save_stage(
        self,
        stage: str,
        *,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        pre_norm: torch.Tensor | None = None,
    ) -> Path | None:
        if _is_cuda_graph_capturing() or stage not in {
            "before_attention",
            "after_attention",
            "ffn_input",
            "after_ffn",
        }:
            return None
        tensors = {
            name: value
            for name, value in {
                "hidden_states": hidden_states,
                "residual": residual,
                "post_mix": post_mix,
                "res_mix": res_mix,
                "pre_norm": pre_norm,
            }.items()
            if isinstance(value, torch.Tensor)
        }
        clone_tensor = _cpu_clone if _capture_full_tensors_enabled() else _last_token
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "layer_idx": self.layer_idx,
            "call": self.call,
            "stage": stage,
            "positions": self.positions,
            "input_ids": self.input_ids,
            "attention_metadata": self.attention_metadata,
            "tensor_meta": {name: _tensor_meta(value) for name, value in tensors.items()},
            "tensors": {name: clone_tensor(value) for name, value in tensors.items()},
        }
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / f"rank{self.rank}_layer{self.layer_idx}_call{self.call}_{stage}.pt"
        with _SAVE_LOCK:
            torch.save(payload, path)
        return path


class QStageCapture:
    def __init__(
        self,
        capture_dir: Path,
        rank: int,
        layer_idx: int,
        call: int,
        positions: torch.Tensor,
        raw_q: torch.Tensor,
    ) -> None:
        self.capture_dir = capture_dir
        self.rank = rank
        self.layer_idx = layer_idx
        self.call = call
        self.positions_meta = _tensor_meta(positions)
        self.raw_q_meta = _tensor_meta(raw_q)
        self.positions = positions.detach().clone()
        # Keep this clone on-device until the fused op has completed.
        self.raw_q = raw_q.detach().clone()

    @torch.no_grad()
    def finish(self, post_q: torch.Tensor) -> Path | None:
        if _is_cuda_graph_capturing():
            return None
        if post_q.is_cuda:
            torch.cuda.current_stream(post_q.device).synchronize()
        raw_q = self.raw_q.detach().contiguous().cpu().clone()
        post_q_cpu = post_q.detach().contiguous().cpu().clone()
        positions = self.positions.detach().contiguous().cpu().clone()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "layer_idx": self.layer_idx,
            "call": self.call,
            "stage": "q_stages",
            "positions": positions,
            "raw_q": raw_q,
            "post_q": post_q_cpu,
            "tensor_meta": {
                "positions": self.positions_meta,
                "raw_q": self.raw_q_meta,
                "post_q": _tensor_meta(post_q),
            },
        }
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / (
            f"rank{self.rank}_layer{self.layer_idx}_call{self.call}_q_stages.pt"
        )
        with _SAVE_LOCK:
            torch.save(payload, path)
        return path


class QKVInsertCapture:
    def __init__(
        self,
        capture_dir: Path,
        rank: int,
        layer_idx: int,
        call: int,
        token_indices: torch.Tensor,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        self.capture_dir = capture_dir
        self.rank = rank
        self.layer_idx = layer_idx
        self.call = call
        self.block_size = int(block_size)
        self.token_indices = token_indices.detach().to(torch.int64).cpu().clone()
        self.positions_meta = _tensor_meta(positions)
        self.q_meta = _tensor_meta(q)
        self.kv_meta = _tensor_meta(kv)
        self.cache_meta = _tensor_meta(cache)
        self.slot_mapping_meta = _tensor_meta(slot_mapping)
        self.positions = (
            positions.detach().reshape(-1).index_select(0, token_indices).cpu().clone()
        )
        self.q = q.detach().index_select(0, token_indices).cpu().clone()
        self.kv = kv.detach().index_select(0, token_indices).cpu().clone()
        self.slot_mapping = (
            slot_mapping.detach()
            .reshape(-1)
            .index_select(0, token_indices)
            .to(torch.int64)
        )
        valid_slots = self.slot_mapping >= 0
        self.cache_block_indices = torch.div(
            self.slot_mapping[valid_slots],
            self.block_size,
            rounding_mode="floor",
        )
        self.cache_slot_offsets = self.slot_mapping[valid_slots] % self.block_size
        self.cache_before = self._clone_cache_blocks(cache)

    def _clone_cache_blocks(self, cache: torch.Tensor) -> torch.Tensor | None:
        if self.cache_block_indices.numel() == 0:
            return None
        block_indices = self.cache_block_indices.to(device=cache.device)
        return cache.detach().index_select(0, block_indices).cpu().clone()

    @torch.no_grad()
    def finish(self, cache: torch.Tensor) -> Path | None:
        if _is_cuda_graph_capturing():
            return None
        if cache.is_cuda:
            torch.cuda.current_stream(cache.device).synchronize()
        cache_after = self._clone_cache_blocks(cache)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "layer_idx": self.layer_idx,
            "call": self.call,
            "stage": "qkv_insert",
            "token_indices": self.token_indices,
            "positions": self.positions,
            "q": self.q,
            "kv": self.kv,
            "slot_mapping": self.slot_mapping.cpu().clone(),
            "block_size": self.block_size,
            "cache_block_indices": self.cache_block_indices.cpu().clone(),
            "cache_slot_offsets": self.cache_slot_offsets.cpu().clone(),
            "cache_before": self.cache_before,
            "cache_after": cache_after,
            "tensor_meta": {
                "positions": self.positions_meta,
                "q": self.q_meta,
                "kv": self.kv_meta,
                "cache": self.cache_meta,
                "slot_mapping": self.slot_mapping_meta,
            },
        }
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / (
            f"rank{self.rank}_layer{self.layer_idx}_call{self.call}_qkv_insert.pt"
        )
        with _SAVE_LOCK:
            torch.save(payload, path)
        return path


def maybe_layer_capture_context(
    layer_idx: int,
    positions: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> LayerCaptureContext | None:
    capture_dir = _capture_dir()
    if capture_dir is None:
        return None
    capture_ranks = _capture_ranks()
    if (
        _is_cuda_graph_capturing()
        or not layer_capture_layer_enabled(layer_idx)
        or not _rank_capture_enabled(_rank(), capture_ranks)
        or not _position_capture_enabled(positions)
    ):
        return None
    call = _next_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None
    rank = _rank()
    assert rank is not None
    return LayerCaptureContext(capture_dir, rank, layer_idx, call, positions, input_ids)


@torch.no_grad()
def maybe_capture_attention_inputs(
    layer_idx: int,
    positions: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
) -> Path | None:
    """Capture post-norm attention inputs for an eligible layer call."""
    capture_dir = _capture_dir()
    if capture_dir is None:
        return None
    capture_ranks = _capture_ranks()
    if (
        _is_cuda_graph_capturing()
        or not layer_capture_layer_enabled(layer_idx)
        or not _rank_capture_enabled(_rank(), capture_ranks)
        or not _position_capture_enabled(positions)
    ):
        return None
    if not all(isinstance(value, torch.Tensor) for value in (positions, qr, kv)):
        return None

    call = _next_attention_call_index(layer_idx)
    _ATTENTION_LAST_CALL[layer_idx] = call
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None

    tensors = {
        "positions": positions,
        "qr": qr,
        "kv": kv,
    }
    rank = _rank()
    assert rank is not None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "attention_inputs",
        "positions": _cpu_clone(positions),
        "qr": _cpu_clone(qr),
        "kv": _cpu_clone(kv),
        "tensor_meta": {
            name: _tensor_meta(value) for name, value in tensors.items()
        },
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"rank{rank}_layer{layer_idx}_call{call}_attention_inputs.pt"
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


@torch.no_grad()
def maybe_capture_attention_output(
    layer_idx: int,
    positions: torch.Tensor,
    output: torch.Tensor,
) -> Path | None:
    """Capture the completed raw attention output for an eligible layer call."""
    capture_dir = _capture_dir()
    if capture_dir is None:
        return None
    capture_ranks = _capture_ranks()
    if (
        _is_cuda_graph_capturing()
        or not layer_capture_layer_enabled(layer_idx)
        or not _rank_capture_enabled(_rank(), capture_ranks)
        or not _position_capture_enabled(positions)
    ):
        return None
    if not all(isinstance(value, torch.Tensor) for value in (positions, output)):
        return None

    call = _next_attention_output_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None

    rank = _rank()
    assert rank is not None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "attention_output",
        "positions": _cpu_clone(positions),
        "output": _cpu_clone(output),
        "tensor_meta": {
            "positions": _tensor_meta(positions),
            "output": _tensor_meta(output),
        },
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"rank{rank}_layer{layer_idx}_call{call}_attention_output.pt"
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


@torch.no_grad()
def maybe_capture_qkv_producer(
    layer_idx: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    qr_kv: torch.Tensor,
    q_lora_rank: int,
    qr: torch.Tensor,
    kv: torch.Tensor,
) -> Path | None:
    """Capture QKV producer tensors for an eligible layer call."""
    capture_dir = _capture_dir()
    if capture_dir is None:
        return None
    capture_ranks = _capture_ranks()
    if (
        _is_cuda_graph_capturing()
        or not layer_capture_layer_enabled(layer_idx)
        or not _rank_capture_enabled(_rank(), capture_ranks)
        or not _position_capture_enabled(positions)
    ):
        return None
    if not all(
        isinstance(value, torch.Tensor)
        for value in (positions, hidden_states, qr_kv, qr, kv)
    ):
        return None
    call = _next_qkv_producer_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None

    qr_pre_norm, kv_pre_norm = qr_kv.split([q_lora_rank, kv.shape[-1]], dim=-1)
    tensors = {
        "hidden_states": hidden_states,
        "qr_kv": qr_kv,
        "qr_pre_norm": qr_pre_norm,
        "kv_pre_norm": kv_pre_norm,
        "qr": qr,
        "kv": kv,
    }
    rank = _rank()
    assert rank is not None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "qkv_producer",
        "positions": _cpu_clone(positions),
        "q_lora_rank": int(q_lora_rank),
        "tensors": {name: _cpu_clone(value) for name, value in tensors.items()},
        "tensor_meta": {
            name: _tensor_meta(value) for name, value in tensors.items()
        },
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"rank{rank}_layer{layer_idx}_call{call}_qkv_producer.pt"
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


@torch.no_grad()
def maybe_prepare_q_stage_capture(
    layer_idx: int,
    positions: torch.Tensor,
    raw_q: torch.Tensor,
) -> QStageCapture | None:
    """Prepare an opt-in raw/post q snapshot without synchronizing before q fusion."""
    capture_dir = _capture_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    if capture_dir is None or _is_cuda_graph_capturing():
        return None
    if not layer_capture_layer_enabled(layer_idx) or not _rank_capture_enabled(
        rank, capture_ranks
    ):
        return None
    if not _position_capture_enabled(positions):
        return None
    if not isinstance(positions, torch.Tensor) or not isinstance(raw_q, torch.Tensor):
        return None
    call = _ATTENTION_LAST_CALL.get(layer_idx)
    if call is None:
        call = _next_q_stage_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None
    if (layer_idx, call) in _Q_STAGE_CAPTURED_CALLS:
        return None
    _Q_STAGE_CAPTURED_CALLS.add((layer_idx, call))
    assert rank is not None
    return QStageCapture(capture_dir, rank, layer_idx, call, positions, raw_q)


@torch.no_grad()
def maybe_capture_wq_b_shadow_compare(
    layer_idx: int,
    positions: torch.Tensor,
    qr: torch.Tensor,
    batched_q: torch.Tensor,
    rowwise_q: torch.Tensor,
) -> Path | None:
    """Save an opt-in same-input batched-vs-rowwise target wq_b summary."""
    capture_dir = _wq_b_shadow_compare_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    if capture_dir is None or _is_cuda_graph_capturing():
        return None
    if not layer_capture_layer_enabled(layer_idx) or not _rank_capture_enabled(
        rank, capture_ranks
    ):
        return None
    if not _position_capture_enabled(positions):
        return None
    if (
        not isinstance(positions, torch.Tensor)
        or not isinstance(qr, torch.Tensor)
        or not isinstance(batched_q, torch.Tensor)
        or not isinstance(rowwise_q, torch.Tensor)
    ):
        return None
    call = _ATTENTION_LAST_CALL.get(layer_idx)
    if call is None:
        call = _next_q_stage_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None
    if batched_q.is_cuda:
        torch.cuda.current_stream(batched_q.device).synchronize()
    batched_cpu = batched_q.detach().contiguous().cpu()
    rowwise_cpu = rowwise_q.detach().contiguous().cpu()
    diff = (batched_cpu.float() - rowwise_cpu.float()).abs()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "wq_b_shadow_compare",
        "positions": positions.detach().contiguous().cpu().clone(),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "num_diff": int(torch.count_nonzero(batched_cpu != rowwise_cpu).item()),
        "tensor_meta": {
            "positions": _tensor_meta(positions),
            "qr": _tensor_meta(qr),
            "batched_q": _tensor_meta(batched_q),
            "rowwise_q": _tensor_meta(rowwise_q),
        },
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / (
        f"rank{rank}_layer{layer_idx}_call{call}_wq_b_shadow_compare.pt"
    )
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


def _shadow_tensor_summary(batched: torch.Tensor, rowwise: torch.Tensor) -> dict[str, Any]:
    batched_cpu = batched.detach().contiguous().cpu()
    rowwise_cpu = rowwise.detach().contiguous().cpu()
    diff = (batched_cpu.float() - rowwise_cpu.float()).abs()
    equal = (batched_cpu == rowwise_cpu) | (
        torch.isnan(batched_cpu) & torch.isnan(rowwise_cpu)
    )
    num_diff = int(torch.count_nonzero(~equal).item())
    result: dict[str, Any] = {
        "shape": list(batched_cpu.shape),
        "dtype": str(batched_cpu.dtype),
        "num_diff": num_diff,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
    }
    if num_diff:
        index = (~equal).nonzero(as_tuple=False)[0].tolist()
        result.update(
            {
                "first_diff_index": index,
                "batched_value": batched_cpu[tuple(index)].item(),
                "rowwise_value": rowwise_cpu[tuple(index)].item(),
            }
        )
    return result


@torch.no_grad()
def maybe_capture_mhc_pre_shadow_compare(
    *,
    layer_idx: int,
    positions: torch.Tensor,
    residual_cur: torch.Tensor,
    actual_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rowwise_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Path | None:
    """Save same-input batched-vs-rowwise MHC pre outputs."""
    if not mhc_pre_shadow_compare_selected(layer_idx, positions):
        return None
    if residual_cur.is_cuda:
        torch.cuda.current_stream(residual_cur.device).synchronize()
    names = ("post_mix", "res_mix", "pre_norm")
    rank = _rank()
    assert rank is not None
    with _CALL_LOCK:
        call = _MHC_PRE_SHADOW_CALL_INDICES.get(layer_idx, 0)
        _MHC_PRE_SHADOW_CALL_INDICES[layer_idx] = call + 1
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "mhc_pre_shadow_compare",
        "positions": _cpu_clone(positions),
        "residual_cur": _cpu_clone(residual_cur),
        "summary": {
            name: _shadow_tensor_summary(actual, rowwise)
            for name, actual, rowwise in zip(
                names, actual_outputs, rowwise_outputs, strict=True
            )
        },
        "actual_outputs": {
            name: _cpu_clone(value)
            for name, value in zip(names, actual_outputs, strict=True)
        },
        "rowwise_outputs": {
            name: _cpu_clone(value)
            for name, value in zip(names, rowwise_outputs, strict=True)
        },
    }
    capture_dir = _mhc_pre_shadow_compare_dir()
    assert capture_dir is not None
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / (
        f"rank{rank}_layer{layer_idx}_call{call}_mhc_pre_shadow_compare.pt"
    )
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


@torch.no_grad()
def maybe_capture_qkv_prenorm_shadow_compare(
    layer_idx: int,
    positions: torch.Tensor,
    batched_qr_kv: torch.Tensor,
    rowwise_qr_kv: torch.Tensor,
    q_lora_rank: int,
) -> Path | None:
    """Save an opt-in batched-vs-rowwise target pre-norm QKV summary."""
    capture_dir = _qkv_prenorm_shadow_compare_dir()
    rank = _rank()
    capture_ranks = _capture_ranks() if capture_dir is not None else None
    if capture_dir is None or _is_cuda_graph_capturing():
        return None
    if not layer_capture_layer_enabled(layer_idx) or not _rank_capture_enabled(
        rank, capture_ranks
    ):
        return None
    if not _position_capture_enabled(positions):
        return None
    if not all(
        isinstance(value, torch.Tensor)
        for value in (positions, batched_qr_kv, rowwise_qr_kv)
    ):
        return None
    calls = _call_filter()
    call = _QKV_PRODUCER_CALL_INDICES.get(layer_idx, 0)
    if calls is not None and call not in calls:
        return None
    if batched_qr_kv.is_cuda:
        torch.cuda.current_stream(batched_qr_kv.device).synchronize()
    batched_qr, batched_kv = batched_qr_kv.split(
        [q_lora_rank, batched_qr_kv.shape[-1] - q_lora_rank], dim=-1
    )
    rowwise_qr, rowwise_kv = rowwise_qr_kv.split(
        [q_lora_rank, rowwise_qr_kv.shape[-1] - q_lora_rank], dim=-1
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "layer_idx": layer_idx,
        "call": call,
        "stage": "qkv_prenorm_shadow_compare",
        "positions": positions.detach().contiguous().cpu().clone(),
        "q_lora_rank": int(q_lora_rank),
        "summary": {
            "qr_kv": _shadow_tensor_summary(batched_qr_kv, rowwise_qr_kv),
            "qr": _shadow_tensor_summary(batched_qr, rowwise_qr),
            "kv": _shadow_tensor_summary(batched_kv, rowwise_kv),
        },
        "tensor_meta": {
            "positions": _tensor_meta(positions),
            "batched_qr_kv": _tensor_meta(batched_qr_kv),
            "rowwise_qr_kv": _tensor_meta(rowwise_qr_kv),
        },
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / (
        f"rank{rank}_layer{layer_idx}_call{call}_qkv_prenorm_shadow_compare.pt"
    )
    with _SAVE_LOCK:
        torch.save(payload, path)
    return path


@torch.no_grad()
def maybe_prepare_qkv_insert_capture(
    layer_idx: int,
    positions: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> QKVInsertCapture | None:
    capture_dir = _qkv_insert_capture_dir()
    if capture_dir is None or _is_cuda_graph_capturing():
        return None
    rank = _rank()
    if (
        not layer_capture_layer_enabled(layer_idx)
        or not _rank_capture_enabled(rank)
        or not all(
            isinstance(value, torch.Tensor)
            for value in (positions, q, kv, cache, slot_mapping)
        )
        or positions.ndim == 0
        or slot_mapping.ndim == 0
    ):
        return None
    token_indices = _qkv_insert_selected_token_indices(positions)
    if token_indices is None:
        return None
    call = _ATTENTION_LAST_CALL.get(layer_idx)
    if call is None:
        call = _next_q_stage_call_index(layer_idx)
    calls = _call_filter()
    if calls is not None and call not in calls:
        return None
    assert rank is not None
    return QKVInsertCapture(
        capture_dir,
        rank,
        layer_idx,
        call,
        token_indices,
        positions,
        q,
        kv,
        cache,
        slot_mapping,
        block_size,
    )
