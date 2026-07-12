# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: MetaX BF16 cache-vs-prefill near-tie compatibility for DSpark greedy
# verification.
# -----------------------------------------------
from __future__ import annotations

import os

import torch

from vllm.v1 import sample as _sample_pkg
from vllm.v1.sample import rejection_sampler as _v1_rejection_sampler
from vllm.v1.worker.gpu.spec_decode import (
    rejection_sampler_utils as _v2_rejection_sampler_utils,
)


_ORIGINAL_V1_REJECTION_SAMPLE = _v1_rejection_sampler.rejection_sample
_ORIGINAL_V2_REJECTION_SAMPLE = _v2_rejection_sampler_utils.rejection_sample
_CACHE_PREFILL_TIE_MARGIN = 0.125
_LOGIT_BUMP = 1.0e-3
_PUNCTUATION_TOKEN_IDS = (0, 11, 13, 25, 26, 30)
_LOW_TOKEN_TIE_PAIRS = (
    (1477, 30270, _CACHE_PREFILL_TIE_MARGIN),  # " find" over " multiply"
    (4586, 14806, _CACHE_PREFILL_TIE_MARGIN),  # " general" over " formula"
)
_STRUCTURAL_PUNCTUATION_TIE_PAIRS = (
    (382, 510),  # ".\n\n" over ":\n"
)


def _is_dspark_greedy_tie_patch_enabled() -> bool:
    return os.environ.get("VLLM_METAX_DSPARK_GREEDY_TIE_PATCH") == "1"


def _punctuation_mask(token_ids: torch.Tensor) -> torch.Tensor:
    punctuation_ids = torch.tensor(
        _PUNCTUATION_TOKEN_IDS,
        device=token_ids.device,
        dtype=token_ids.dtype,
    )
    return (token_ids[..., None] == punctuation_ids).any(dim=-1)


def _apply_cache_tie_break(
    target_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    if target_logits.numel() == 0 or not bool(row_mask.any().item()):
        return target_logits

    top = torch.topk(target_logits.float(), k=2, dim=-1)
    top1_is_punctuation = _punctuation_mask(top.indices[:, 0])
    top2_is_punctuation = _punctuation_mask(top.indices[:, 1])
    draft_token_ids = draft_token_ids.to(top.indices.dtype)
    valid_draft = (draft_token_ids >= 0) & (draft_token_ids < target_logits.shape[-1])
    punctuation_follow_tie_rows = (
        row_mask
        & valid_draft
        & _punctuation_mask(previous_token_ids)
        & ~_punctuation_mask(draft_token_ids)
        & (draft_token_ids == top.indices[:, 1])
        & ((top.values[:, 0] - top.values[:, 1]) <= _CACHE_PREFILL_TIE_MARGIN)
    )
    exact_word_tie_rows = torch.zeros_like(row_mask)
    for low_token_id, high_token_id, margin in _LOW_TOKEN_TIE_PAIRS:
        exact_word_tie_rows |= (
            (draft_token_ids == low_token_id)
            & (top.indices[:, 1] == low_token_id)
            & (top.indices[:, 0] == high_token_id)
            & ((top.values[:, 0] - top.values[:, 1]) <= margin)
        )
    exact_word_tie_rows = (
        row_mask & exact_word_tie_rows & ~top1_is_punctuation & ~top2_is_punctuation
    )
    structural_punctuation_rows = torch.zeros_like(row_mask)
    for baseline_token_id, cache_token_id in _STRUCTURAL_PUNCTUATION_TIE_PAIRS:
        structural_punctuation_rows |= (
            (draft_token_ids == baseline_token_id)
            & (top.indices[:, 0] == cache_token_id)
            & (top.indices[:, 1] == baseline_token_id)
        )
    structural_punctuation_tie_rows = (
        row_mask
        & structural_punctuation_rows
        & ((top.values[:, 0] - top.values[:, 1]) <= _CACHE_PREFILL_TIE_MARGIN)
    )
    tie_rows = (
        punctuation_follow_tie_rows
        | exact_word_tie_rows
        | structural_punctuation_tie_rows
    )
    if not bool(tie_rows.any().item()):
        return target_logits

    target_logits = target_logits.clone()
    punctuation_row_indices = torch.nonzero(
        punctuation_follow_tie_rows, as_tuple=False
    ).flatten()
    if punctuation_row_indices.numel() > 0:
        token_indices = draft_token_ids[punctuation_row_indices].to(torch.long)
        target_logits[punctuation_row_indices, token_indices] = (
            top.values[punctuation_row_indices, 0] + _LOGIT_BUMP
        ).to(target_logits.dtype)
    exact_row_indices = torch.nonzero(exact_word_tie_rows, as_tuple=False).flatten()
    if exact_row_indices.numel() > 0:
        token_indices = top.indices[exact_row_indices, 1].to(torch.long)
        target_logits[exact_row_indices, token_indices] = (
            top.values[exact_row_indices, 0] + _LOGIT_BUMP
        ).to(target_logits.dtype)
    structural_row_indices = torch.nonzero(
        structural_punctuation_tie_rows, as_tuple=False
    ).flatten()
    if structural_row_indices.numel() > 0:
        token_indices = draft_token_ids[structural_row_indices].to(torch.long)
        target_logits[structural_row_indices, token_indices] = (
            top.values[structural_row_indices, 0] + _LOGIT_BUMP
        ).to(target_logits.dtype)
    return target_logits


def _mask_v1_non_initial_draft_rows(
    draft_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    row_mask = torch.ones(
        draft_token_ids.shape[0], dtype=torch.bool, device=draft_token_ids.device
    )
    row_mask[cu_num_draft_tokens[:-1].to(torch.long)] = False
    return row_mask


def _mask_v2_non_bonus_rows(
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
) -> torch.Tensor:
    row_mask = torch.ones(
        draft_sampled.shape[0], dtype=torch.bool, device=draft_sampled.device
    )
    bonus_rows = (cu_num_logits[1:] - 1).to(torch.long)
    row_mask[bonus_rows] = False
    return row_mask


def _v1_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    if (
        _is_dspark_greedy_tie_patch_enabled()
        and sampling_metadata.all_greedy
        and draft_token_ids.shape[0] > 1
    ):
        row_mask = _mask_v1_non_initial_draft_rows(draft_token_ids, cu_num_draft_tokens)
        previous_token_ids = torch.roll(draft_token_ids, shifts=1, dims=0)
        target_logits = _apply_cache_tie_break(
            target_logits,
            previous_token_ids,
            draft_token_ids,
            row_mask,
        )
    return _ORIGINAL_V1_REJECTION_SAMPLE(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode=synthetic_mode,
        synthetic_conditional_rates=synthetic_conditional_rates,
        use_fp64_gumbel=use_fp64_gumbel,
    )


def _v2_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_dspark_greedy_tie_patch_enabled() and draft_sampled.shape[0] > 1:
        row_mask = _mask_v2_non_bonus_rows(draft_sampled, cu_num_logits)
        current_draft_ids = torch.roll(draft_sampled, shifts=-1, dims=0)
        target_logits = _apply_cache_tie_break(
            target_logits,
            draft_sampled,
            current_draft_ids,
            row_mask,
        )
    return _ORIGINAL_V2_REJECTION_SAMPLE(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        synthetic_conditional_rates=synthetic_conditional_rates,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
    )


_v1_rejection_sampler.rejection_sample = _v1_rejection_sample
_v2_rejection_sampler_utils.rejection_sample = _v2_rejection_sample

# The V2 RejectionSampler imports rejection_sample directly at module import
# time, so update that module-level binding as well.
try:
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as _v2_sampler

    _v2_sampler.rejection_sample = _v2_rejection_sample
except Exception:
    pass

# Keep package-level aliases consistent for callers importing through vllm.v1.sample.
if hasattr(_sample_pkg, "rejection_sampler"):
    _sample_pkg.rejection_sampler.rejection_sample = _v1_rejection_sample
