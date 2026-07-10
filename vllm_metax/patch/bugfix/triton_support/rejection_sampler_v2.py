# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: MetaX Triton may compile pointer expressions guarded by
# `USE_BLOCK_VERIFICATION and not is_greedy`, even when the constexpr is false.
# Split the V2 rejection sampler into constexpr-only block-verification and
# normal paths so None block-verification buffers are not lowered.
# -----------------------------------------------
import os

import torch
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rsu

tl = rsu.tl
_original_rejection_sample = rsu.rejection_sample


@triton.jit
def _compute_global_target_argmax_scalar(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
    return tl.load(
        target_local_argmax_ptr
        + logit_idx * target_local_argmax_stride
        + max_block_idx
    ).to(tl.int64)


@triton.jit
def _rejection_kernel(
    sampled_ptr,
    sampled_stride,
    rejected_steps_ptr,
    target_rejected_logsumexp_ptr,
    draft_rejected_logsumexp_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    synthetic_conditional_rates_ptr,
    cumulative_log_p_ptr,
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True

    if USE_BLOCK_VERIFICATION:
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            pos = tl.load(pos_ptr + logit_idx)
            u = rsu.tl_rand32(seed, pos, includes_zero=False)
            if not is_greedy:
                prefix_joint_ratio = tl.exp(
                    tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
                )
                if i < num_draft_tokens - 1:
                    residual_mass = rsu._compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_sampled_ptr,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            elif accepted:
                target_argmax = _compute_global_target_argmax_scalar(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= (u < rate) & (draft_sampled >= 0)
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_sampled, target_argmax),
                )
                accepted_length += accepted
    else:
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            pos = tl.load(pos_ptr + logit_idx)
            u = rsu.tl_rand32(seed, pos, includes_zero=False)
            if accepted:
                target_argmax = _compute_global_target_argmax_scalar(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= (u < rate) & (draft_sampled >= 0)
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_sampled, target_argmax),
                )
                accepted_length += accepted

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION:
        if not is_greedy and accepted_length < num_draft_tokens:
            rejected_idx = start_idx + accepted_length
            target_lse = rsu._compute_global_logsumexp(
                target_local_max_ptr,
                target_local_max_stride,
                target_local_sumexp_ptr,
                target_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            if HAS_DRAFT_LOGITS:
                draft_lse = rsu._compute_global_logsumexp(
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    rejected_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


rsu._rejection_kernel = _rejection_kernel


def _greedy_rejection_sample(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    num_speculative_steps: int,
    use_fp64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    target_argmax = torch.argmax(target_logits, dim=-1)
    sampled = draft_sampled.new_full(
        (num_reqs, num_speculative_steps + 1), -1, dtype=torch.int64
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=draft_sampled.device)

    cu = cu_num_logits.detach().cpu().tolist()
    for req_idx in range(num_reqs):
        start = cu[req_idx]
        end = cu[req_idx + 1]
        num_draft_tokens = end - start - 1
        accepted_len = 0
        for i in range(num_draft_tokens):
            logit_idx = start + i
            draft_token = draft_sampled[logit_idx + 1]
            target_token = target_argmax[logit_idx]
            if bool((draft_token == target_token).item()):
                sampled[req_idx, i] = draft_token
                accepted_len += 1
            else:
                sampled[req_idx, i] = target_token
                break
        else:
            sampled[req_idx, accepted_len] = target_argmax[start + accepted_len]

        num_sampled[req_idx] = accepted_len + 1
    return sampled, num_sampled


def _rejection_sample(
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
    if (
        os.environ.get("VLLM_METAX_DISABLE_GREEDY_REJECTION_FASTPATH") != "1"
        and synthetic_conditional_rates is None
        and not use_block_verification
        and bool(torch.all(temperature[idx_mapping] == 0).item())
    ):
        return _greedy_rejection_sample(
            target_logits,
            draft_sampled,
            cu_num_logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            num_speculative_steps,
            use_fp64,
        )
    return _original_rejection_sample(
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
        synthetic_conditional_rates,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
    )


rsu.rejection_sample = _rejection_sample

try:
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

    rs.rejection_sample = _rejection_sample
except ImportError:
    pass
