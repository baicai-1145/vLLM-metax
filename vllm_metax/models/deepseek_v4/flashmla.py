# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from .attention import MacaDeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
)
from .ops.o_proj import (
    deep_gemm_bf16_o_proj,
)
from .ops import (
    compute_global_topk_indices_and_lens_bounded,
    gather_k_cache,
)
from .sparse_mla import (
    MacaDeepseekV4FlashMLABackend,
)
from .mtp_candidate import (
    env_or_k1_candidate_enabled,
    k1_correctness_candidate_enabled,
    k1_native_o_proj_candidate_enabled,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
)
from vllm_metax.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm_metax.kernels.sparse_mla_decode import (
    SPARSE_MLA_DECODE_MODE,
    sparse_mla_decode,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)
_TOKENWISE_O_PROJ_POSITIONS_ENV = "VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS"


def _tokenwise_o_proj_selected_indices(positions: torch.Tensor) -> torch.Tensor:
    flat_positions = positions.detach().reshape(-1)
    if k1_correctness_candidate_enabled():
        return torch.arange(flat_positions.numel(), device=positions.device)
    value = os.getenv(_TOKENWISE_O_PROJ_POSITIONS_ENV)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return torch.arange(flat_positions.numel(), device=positions.device)
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{_TOKENWISE_O_PROJ_POSITIONS_ENV} must be 'all' or a "
            "comma-separated set of integer positions"
        ) from exc
    if not selected:
        return torch.arange(flat_positions.numel(), device=positions.device)
    mask = torch.zeros_like(flat_positions, dtype=torch.bool)
    for position in selected:
        mask |= flat_positions == position
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _tokenwise_o_proj_enabled() -> bool:
    if os.getenv("VLLM_METAX_DSV4_TOKENWISE_O_PROJ") == "1":
        return True
    if (
        k1_correctness_candidate_enabled()
        and not k1_native_o_proj_candidate_enabled()
    ):
        return True
    return False

_SPARSE_MLA_DECODE_BACKEND_ENV = "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND"
_SPARSE_MLA_DECODE_BACKENDS = {"native", "torch_reference"}
_SPARSE_MLA_DECODE_SYNC_ENV = "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_SYNC"
_SPARSE_MLA_DECODE_DIFF_ENV = "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_DIFF"
_SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV = (
    "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_DIFF_MAX_CALLS"
)
_SPARSE_MLA_DECODE_DIFF_DUMP_DIR_ENV = (
    "VLLM_METAX_DSV4_SPARSE_MLA_DECODE_DIFF_DUMP_DIR"
)
_torch_reference_warning_emitted = False
_sparse_mla_decode_sync_warning_emitted = False
_sparse_mla_decode_diff_call_count = 0
_sparse_mla_decode_diff_comparison_count = 0
_sparse_mla_decode_diff_logged_count = 0
_sparse_mla_decode_diff_first_mismatch_call: int | None = None


def _get_sparse_mla_decode_backend() -> str:
    backend = os.getenv(_SPARSE_MLA_DECODE_BACKEND_ENV, "native").strip().lower()
    if backend not in _SPARSE_MLA_DECODE_BACKENDS:
        allowed = ", ".join(sorted(_SPARSE_MLA_DECODE_BACKENDS))
        raise ValueError(
            f"Invalid {_SPARSE_MLA_DECODE_BACKEND_ENV}={backend!r}; "
            f"expected one of: {allowed}"
        )
    return backend


def _get_sparse_mla_decode_sync() -> bool:
    value = os.getenv(_SPARSE_MLA_DECODE_SYNC_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid {_SPARSE_MLA_DECODE_SYNC_ENV}={value!r}; "
            "expected one of: 0, 1"
        )
    return value == "1"


def _get_sparse_mla_decode_diff() -> bool:
    value = os.getenv(_SPARSE_MLA_DECODE_DIFF_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid {_SPARSE_MLA_DECODE_DIFF_ENV}={value!r}; "
            "expected one of: 0, 1"
        )
    return value == "1"


def _get_sparse_mla_decode_diff_max_calls() -> int:
    value = os.getenv(_SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV, "256").strip()
    try:
        calls = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV}={value!r}; "
            "expected a non-negative integer"
        ) from exc
    if calls < 0:
        raise ValueError(
            f"Invalid {_SPARSE_MLA_DECODE_DIFF_MAX_CALLS_ENV}={value!r}; "
            "expected a non-negative integer"
        )
    return calls


def _sparse_mla_decode_diff_rank() -> str:
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


def _sparse_mla_decode_diff_reset_state() -> None:
    global _sparse_mla_decode_diff_call_count
    global _sparse_mla_decode_diff_comparison_count
    global _sparse_mla_decode_diff_logged_count
    global _sparse_mla_decode_diff_first_mismatch_call
    _sparse_mla_decode_diff_call_count = 0
    _sparse_mla_decode_diff_comparison_count = 0
    _sparse_mla_decode_diff_logged_count = 0
    _sparse_mla_decode_diff_first_mismatch_call = None


def _sparse_mla_decode_diff_stats(
    native: torch.Tensor, reference: torch.Tensor
) -> tuple[int, float]:
    if native.shape != reference.shape or native.dtype != reference.dtype:
        return max(native.numel(), reference.numel()), float("inf")
    lhs = native.detach().contiguous()
    rhs = reference.detach().contiguous()
    # Compare complete element representations, including NaN payloads, rather
    # than values (which would miss bitwise differences and signed zero).
    lhs_bytes = lhs.view(torch.uint8).reshape(lhs.numel(), -1)
    rhs_bytes = rhs.view(torch.uint8).reshape(rhs.numel(), -1)
    mismatch = (lhs_bytes != rhs_bytes).any(dim=1)
    mismatch_count = int(mismatch.sum().item())
    if not mismatch_count:
        return 0, 0.0
    max_abs = float((lhs.float() - rhs.float()).abs().max().item())
    return mismatch_count, max_abs


def _get_sparse_mla_decode_diff_dump_dir() -> Path | None:
    value = os.getenv(_SPARSE_MLA_DECODE_DIFF_DUMP_DIR_ENV)
    return Path(value) if value and value.strip() else None


def _sparse_mla_decode_diff_cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().contiguous().cpu().clone()


def _sparse_mla_decode_diff_atomic_save(
    payload: dict[str, object], path: Path
) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sparse_mla_decode_diff_dump(
    *,
    attention: "MacaDeepseekV4FlashMLAAttention",
    call: int,
    q_raw: torch.Tensor,
    swa_cache_physical: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    native_output: torch.Tensor,
    reference_output: torch.Tensor,
    reference_probs: torch.Tensor,
    reference_fp32_output: torch.Tensor,
    compress_ratio: int,
    mode: str,
    backend: str,
) -> None:
    if not _get_sparse_mla_decode_diff():
        return
    dump_dir = _get_sparse_mla_decode_diff_dump_dir()
    if dump_dir is None:
        return
    try:
        # The caller has synchronized the producing stream before entering this
        # function, so all CPU clones below represent one coherent invocation.
        dump_dir.mkdir(parents=True, exist_ok=True)
        rank = _sparse_mla_decode_diff_rank()
        rank_value: int | str = int(rank) if rank.isdigit() else rank
        tensors = {
            "q_raw": q_raw,
            "swa_cache_physical": swa_cache_physical,
            "compressed_cache": compressed_cache,
            "swa_indices": swa_indices,
            "topk_indices": topk_indices,
            "native_output": native_output,
            "reference_output": reference_output,
            "reference_probs": reference_probs,
            "reference_fp32_output": reference_fp32_output,
        }
        from vllm_metax.kernels.sparse_mla_decode import (
            sparse_mla_decode_compat_workspace,
        )
        workspace = sparse_mla_decode_compat_workspace()
        if workspace is not None:
            tensors.update({
                "native_probs": workspace[0],
                "native_values": workspace[1],
                "native_transposed": workspace[2],
                "native_fp32_output": workspace[3],
            })
        payload: dict[str, object] = {
            "schema": "dsv4_sparse_mla_decode_diff",
            "schema_version": 1,
            "rank": rank_value,
            "call": call,
            "layer_prefix": getattr(attention, "prefix", None),
            "backend": backend,
            "mode": mode,
            "scale": float(attention.scale),
            "compress_ratio": int(compress_ratio),
            **{name: _sparse_mla_decode_diff_cpu_clone(value)
               for name, value in tensors.items()},
            "shapes": {
                name: (list(value.shape) if value is not None else None)
                for name, value in tensors.items()
            },
            "strides": {
                name: (list(value.stride()) if value is not None else None)
                for name, value in tensors.items()
            },
            "dtypes": {
                name: (str(value.dtype) if value is not None else None)
                for name, value in tensors.items()
            },
        }
        path = dump_dir / f"rank{rank}_call{call}.pt"
        _sparse_mla_decode_diff_atomic_save(payload, path)
        logger.warning("DIAGNOSTIC_ONLY sparse MLA decode diff dump: %s", path)
    except Exception as exc:
        # A debug artifact must never prevent restoration of the native output.
        logger.warning("DIAGNOSTIC_ONLY sparse MLA decode diff dump failed: %s", exc)


@torch.no_grad()
def _maybe_diff_sparse_mla_decode(
    *,
    attention: "MacaDeepseekV4FlashMLAAttention",
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    topk_indices: torch.Tensor | None,
    output: torch.Tensor,
    compress_ratio: int,
    mode: str,
    q_raw: torch.Tensor | None = None,
    swa_cache_physical: torch.Tensor | None = None,
    backend: str = "native",
    compressed_cache: torch.Tensor | None = None,
) -> None:
    """Compare native output with Torch without changing the native result."""
    global _sparse_mla_decode_diff_call_count
    global _sparse_mla_decode_diff_comparison_count
    global _sparse_mla_decode_diff_logged_count
    global _sparse_mla_decode_diff_first_mismatch_call

    if _sparse_mla_decode_diff_first_mismatch_call is not None:
        return
    call = _sparse_mla_decode_diff_call_count
    _sparse_mla_decode_diff_call_count += 1
    native_output = output.detach().clone()
    reference_output = torch.empty_like(output)
    reference_probs, reference_fp32_output = attention._torch_sparse_decode(
        q=q,
        swa_cache=swa_cache,
        compressed_cache=compressed_cache,
        swa_indices=swa_indices,
        topk_indices=topk_indices,
        output=reference_output,
        scale=attention.scale,
    )
    if q.is_cuda:
        torch.cuda.current_stream(q.device).synchronize()
    mismatch_count, max_abs = _sparse_mla_decode_diff_stats(
        native_output, reference_output
    )
    _sparse_mla_decode_diff_comparison_count += 1

    max_calls = _get_sparse_mla_decode_diff_max_calls()
    should_log = max_calls == 0 or _sparse_mla_decode_diff_logged_count < max_calls
    if mismatch_count:
        _sparse_mla_decode_diff_first_mismatch_call = call
        should_log = True
        _sparse_mla_decode_diff_dump(
            attention=attention,
            call=call,
            q_raw=q_raw if q_raw is not None else q,
            swa_cache_physical=(
                swa_cache_physical if swa_cache_physical is not None else swa_cache
            ),
            compressed_cache=compressed_cache,
            swa_indices=swa_indices,
            topk_indices=topk_indices,
            native_output=native_output,
            reference_output=reference_output,
            reference_probs=reference_probs,
            reference_fp32_output=reference_fp32_output,
            compress_ratio=compress_ratio,
            mode=mode,
            backend=backend,
        )
    if should_log:
        record = {
            "event": "first_mismatch" if mismatch_count else "comparison",
            "diagnostic": "sparse_mla_decode_diff",
            "rank": _sparse_mla_decode_diff_rank(),
            "call": call,
            "layer_prefix": getattr(attention, "prefix", None),
            "native_shape": list(native_output.shape),
            "reference_shape": list(reference_output.shape),
            "q_shape": list(q.shape),
            "swa_cache_shape": list(swa_cache.shape),
            "swa_indices_shape": list(swa_indices.shape),
            "topk_indices_shape": (
                list(topk_indices.shape) if topk_indices is not None else None
            ),
            "compress_ratio": compress_ratio,
            "mode": mode,
            "mismatch_count": mismatch_count,
            "max_abs": max_abs,
        }
        logger.warning("DIAGNOSTIC_ONLY %s", json.dumps(record, sort_keys=True))
        _sparse_mla_decode_diff_logged_count += 1
    if mismatch_count:
        summary = {
            "event": "summary",
            "diagnostic": "sparse_mla_decode_diff",
            "rank": _sparse_mla_decode_diff_rank(),
            "comparisons": _sparse_mla_decode_diff_comparison_count,
            "first_mismatch_call": call,
            "logged_records": _sparse_mla_decode_diff_logged_count,
        }
        logger.warning("DIAGNOSTIC_ONLY %s", json.dumps(summary, sort_keys=True))

    # The reference is diagnostic-only; return the native bytes to the caller.
    output.copy_(native_output)


def _synchronize_sparse_mla_decode(q: torch.Tensor) -> None:
    global _sparse_mla_decode_sync_warning_emitted
    if not _sparse_mla_decode_sync_warning_emitted:
        logger.warning(
            "DIAGNOSTIC_ONLY: sparse MLA native decode stream synchronization "
            "is enabled; this adds diagnostic synchronization overhead"
        )
        _sparse_mla_decode_sync_warning_emitted = True
    torch.cuda.current_stream(q.device).synchronize()


def _maybe_sync_sparse_mla_decode(q: torch.Tensor) -> None:
    if _get_sparse_mla_decode_sync():
        _synchronize_sparse_mla_decode(q)


def _run_sparse_mla_decode(**kwargs) -> None:
    q = kwargs["q"]
    if (
        os.getenv("VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE") != "1"
        or not 1 < q.shape[0] <= 5
    ):
        sparse_mla_decode(**kwargs)
        return
    logger.warning_once(
        "DeepSeek V4 speculative attention uses tokenwise sparse MLA decode"
    )
    token_aligned = (
        "q",
        "swa_indices",
        "topk_indices",
        "swa_lens",
        "topk_lens",
        "token_to_req",
        "out",
    )
    for index in range(q.shape[0]):
        row_kwargs = kwargs.copy()
        for name in token_aligned:
            value = row_kwargs[name]
            if value is not None:
                row_kwargs[name] = value[index : index + 1]
        sparse_mla_decode(**row_kwargs)


def _warn_torch_reference_decode() -> None:
    global _torch_reference_warning_emitted
    if _torch_reference_warning_emitted:
        return
    logger.warning(
        "DIAGNOSTIC_ONLY: sparse MLA decode backend=%s; this Torch reference "
        "path is not a native production backend",
        "torch_reference",
    )
    _torch_reference_warning_emitted = True


class MacaDeepseekV4FlashMLAAttention(MacaDeepseekV4Attention):
    backend_cls = MacaDeepseekV4FlashMLABackend

    @staticmethod
    def _torch_sparse_decode(
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        topk_indices: torch.Tensor | None,
        output: torch.Tensor,
        scale: float,
        compressed_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q2 = q.squeeze(1).float()
        batch = q2.shape[0]
        q_heads = q2.shape[1]
        head_dim = q2.shape[2]
        value_dim = output.shape[-1]

        if topk_indices is not None and compressed_cache is None:
            raise ValueError("topk_indices require compressed_cache")

        def gather(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            invalid = indices < 0
            gather_idx = indices.masked_fill(invalid, 0)
            flat_idx = gather_idx.reshape(-1)
            cache_block_size = cache.shape[1]
            # The paged cache has padded block strides. Gather selected rows
            # before converting to FP32 instead of materializing the KV pool.
            block_idx = torch.div(flat_idx, cache_block_size, rounding_mode="floor")
            block_offset = torch.remainder(flat_idx, cache_block_size)
            return cache[block_idx, block_offset, 0].view(
                batch, -1, head_dim
            ).float()

        swa_selected = swa_indices[:, 0, :]
        swa_invalid = swa_selected < 0
        swa_gathered = gather(swa_cache, swa_selected)
        if topk_indices is not None:
            topk_selected = topk_indices[:, 0, :]
            topk_invalid = topk_selected < 0
            topk_gathered = gather(compressed_cache, topk_selected)
            gathered = torch.cat([topk_gathered, swa_gathered], dim=1)
            invalid = torch.cat([topk_invalid, swa_invalid], dim=1)
        else:
            gathered = swa_gathered
            invalid = swa_invalid

        attn = torch.matmul(q2, gathered.transpose(1, 2))
        attn.masked_fill_(invalid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(attn * scale, dim=-1)
        out = torch.matmul(probs, gathered[:, :, :value_dim])
        output.copy_(out.to(output.dtype))
        return probs, out

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        def project(
            o_chunk: torch.Tensor,
            positions_chunk: torch.Tensor,
            chunk_index: int,
        ):
            return deep_gemm_bf16_o_proj(
                o_chunk,
                positions_chunk,
                self.rotary_emb.cos_sin_cache,
                self.wo_a,
                self.wo_b,
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                o_lora_rank=self.o_lora_rank,
                layer_idx=self.layer_idx,
                chunk_index=chunk_index,
            )

        if (
            _tokenwise_o_proj_enabled()
            and 1 < o.shape[0] <= 5
        ):
            selected_indices = _tokenwise_o_proj_selected_indices(positions)
            if selected_indices.numel() == 0:
                return project(o, positions, 0)
            logger.warning_once(
                "DeepSeek V4 speculative attention uses tokenwise output "
                "projection"
            )
            if selected_indices.numel() == o.shape[0]:
                return torch.cat(
                    [
                        project(
                            o[index : index + 1],
                            positions[index : index + 1],
                            index,
                        ).clone()
                        for index in range(o.shape[0])
                    ],
                    dim=0,
                )
            output = project(o, positions, 0)
            for index in selected_indices.tolist():
                output[index : index + 1] = project(
                    o[index : index + 1],
                    positions[index : index + 1],
                    index,
                ).clone()
            return output.contiguous()

        if (
            not self._prefill_gemm_chunking_enabled
            or o.shape[0] <= self._prefill_gemm_chunk_size
        ):
            return project(o, positions, 0)

        outputs = []
        chunk_size = self._prefill_gemm_chunk_size
        for chunk_index, start in enumerate(range(0, o.shape[0], chunk_size)):
            end = start + chunk_size
            outputs.append(
                project(o[start:end], positions[start:end], chunk_index)
            )
        return torch.cat(outputs, dim=0)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        short_context_only = (
            swa_metadata.is_short_context(self.window_size)
        )
        effective_swa_only = swa_only or short_context_only
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not effective_swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=None if effective_swa_only else flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                positions=positions[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=effective_swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                assert swa_metadata.seq_lens is not None
                assert swa_metadata.token_to_req_indices is not None
                global_indices, topk_lens = (
                    compute_global_topk_indices_and_lens_bounded(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        swa_metadata.seq_lens,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        self.compress_ratio,
                        is_valid,
                    )
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Keep the original decode inputs for the opt-in debug capture.  The
        # Torch oracle below receives unsqueezed cache/query views, but the
        # compressed cache and pre-unsqueeze query are useful for replay.
        capture_q = q
        capture_compressed_cache = kv_cache
        capture_swa_cache = self.swa_cache_layer.kv_cache

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        decode_backend = _get_sparse_mla_decode_backend()
        decode_sync = (
            _get_sparse_mla_decode_sync() if decode_backend == "native" else False
        )
        decode_diff = (
            _get_sparse_mla_decode_diff() if decode_backend == "native" else False
        )
        if decode_diff:
            # Validate the bound before launching the native kernel.  The
            # diagnostic remains strictly opt-in and native dispatch is never
            # replaced by the reference result.
            _get_sparse_mla_decode_diff_max_calls()
        if decode_backend == "torch_reference":
            _warn_torch_reference_decode()
            self._torch_sparse_decode(
                q=q,
                swa_cache=swa_cache,
                compressed_cache=kv_cache,
                swa_indices=swa_indices,
                topk_indices=topk_indices,
                output=output,
                scale=self.scale,
            )
        else:
            _run_sparse_mla_decode(
                q=capture_q,
                swa_cache=capture_swa_cache,
                compressed_cache=capture_compressed_cache,
                swa_indices=swa_indices,
                topk_indices=topk_indices,
                swa_lens=swa_lens,
                topk_lens=topk_lens,
                swa_block_table=swa_metadata.block_table,
                compressed_block_table=(
                    attn_metadata.block_table
                    if attn_metadata is not None and not swa_only
                    else None
                ),
                swa_block_size=swa_metadata.block_size,
                compressed_block_size=(
                    attn_metadata.block_size // self.compress_ratio
                    if attn_metadata is not None and not swa_only
                    else None
                ),
                sm_scale=self.scale,
                d_v=output.shape[-1],
                attn_sink=self.attn_sink,
                out=output,
                token_to_req=swa_metadata.token_to_req_indices[:num_decode_tokens],
                swa_indices_are_global=True,
                compressed_indices_are_global=True,
                compatibility_mode=True,
            )
            if decode_sync:
                _synchronize_sparse_mla_decode(capture_q)
            if decode_diff:
                _maybe_diff_sparse_mla_decode(
                    attention=self,
                    q=q,
                    swa_cache=swa_cache,
                    compressed_cache=kv_cache,
                    swa_indices=swa_indices,
                    topk_indices=topk_indices,
                    output=output,
                    compress_ratio=self.compress_ratio,
                    mode=SPARSE_MLA_DECODE_MODE,
                    q_raw=capture_q,
                    swa_cache_physical=capture_swa_cache,
                    backend=decode_backend,
                )

        # Capture only after the reference output is complete.  Avoid even
        # constructing metadata views when the opt-in hook is disabled.
        if os.getenv("VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR"):
            from vllm_metax.models.deepseek_v4.ops.sparse_mla_debug import (
                maybe_capture_sparse_mla_decode,
            )

            maybe_capture_sparse_mla_decode(
                q=capture_q,
                swa_cache=capture_swa_cache,
                compressed_cache=capture_compressed_cache,
                swa_indices=swa_indices,
                topk_indices=topk_indices,
                swa_lens=swa_lens,
                topk_lens=topk_lens,
                sm_scale=self.scale,
                d_v=output.shape[-1],
                attn_sink=self.attn_sink,
                output=output,
                positions=positions,
                token_to_req=(
                    swa_metadata.token_to_req_indices[:num_decode_tokens]
                    if swa_metadata.token_to_req_indices is not None
                    else None
                ),
                swa_block_table=swa_metadata.block_table,
                compressed_block_table=(
                    attn_metadata.block_table
                    if attn_metadata is not None and not swa_only
                    else None
                ),
                swa_block_size=swa_metadata.block_size,
                compressed_block_size=(
                    attn_metadata.block_size // self.compress_ratio
                    if attn_metadata is not None and not swa_only
                    else None
                ),
                compress_ratio=self.compress_ratio,
                window_size=self.window_size,
                decode_backend=decode_backend,
                native_decode_mode=SPARSE_MLA_DECODE_MODE,
                layer_idx=self.layer_idx,
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
                compress_ratio=self.compress_ratio,
                layer_idx=self.layer_idx,
            )
