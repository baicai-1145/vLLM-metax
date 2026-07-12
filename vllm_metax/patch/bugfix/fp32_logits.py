# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Debug/compatibility path for greedy speculative verification ties.
# -----------------------------------------------
import os

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


_original_get_logits = LogitsProcessor._get_logits
_STANDALONE_PUNCTUATION_TOKEN_IDS = (0, 11, 13, 25, 26, 30)
_MARKDOWN_STRUCTURAL_TOKEN_IDS = (
    14374,  # "###"
    44364,  # "---\n\n"
)
_PUNCTUATION_TIE_MARGIN = 0.25


def _fp32_logits_mode() -> str:
    override = os.environ.get("VLLM_METAX_USE_FP32_LOGITS")
    if override is not None:
        if override == "0":
            return "off"
        if override == "1":
            return "force"
        if override.lower() == "auto":
            return "auto"
        raise ValueError("VLLM_METAX_USE_FP32_LOGITS must be one of 0, 1, or auto")

    vllm_config = get_current_vllm_config_or_none()
    speculative_config = (
        vllm_config.speculative_config if vllm_config is not None else None
    )
    if speculative_config is not None and speculative_config.method == "dspark":
        return "auto"
    return "off"


def _can_compute_fp32_logits(
    lm_head: torch.nn.Module,
    embedding_bias: torch.Tensor | None,
) -> bool:
    return (
        embedding_bias is None
        and hasattr(lm_head, "weight")
        and isinstance(
            getattr(lm_head, "quant_method", None), UnquantizedEmbeddingMethod
        )
    )


def _compute_fp32_logits(
    processor: LogitsProcessor,
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
) -> torch.Tensor | None:
    logits = F.linear(hidden_states.float(), lm_head.weight.float())
    logits = processor._gather_logits(logits)
    if logits is not None:
        logits = logits[..., : processor.org_vocab_size]
    return logits


def _get_logits(
    self: LogitsProcessor,
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    mode = _fp32_logits_mode()
    if mode == "off" or not _can_compute_fp32_logits(lm_head, embedding_bias):
        return _original_get_logits(self, hidden_states, lm_head, embedding_bias)

    if mode == "force":
        return _compute_fp32_logits(self, hidden_states, lm_head)

    logits = _original_get_logits(self, hidden_states, lm_head, embedding_bias)
    if logits is None:
        return logits

    # Keep baseline bf16 ordering for ordinary word-vs-word ties. The observed
    # DSpark verifier drift is a punctuation-led near tie ("," vs a word) that
    # flips greedy output, so only resolve that narrow class in fp32.
    top = torch.topk(logits.float(), k=2, dim=-1)
    punctuation_ids = torch.tensor(
        _STANDALONE_PUNCTUATION_TOKEN_IDS,
        device=logits.device,
        dtype=top.indices.dtype,
    )
    markdown_structural_ids = torch.tensor(
        _MARKDOWN_STRUCTURAL_TOKEN_IDS,
        device=logits.device,
        dtype=top.indices.dtype,
    )
    argmax_indices = logits.argmax(dim=-1)
    argmax_is_punctuation = (argmax_indices[..., None] == punctuation_ids).any(dim=-1)
    argmax_is_markdown_structural = (
        argmax_indices[..., None] == markdown_structural_ids
    ).any(dim=-1)
    top1_is_punctuation = (top.indices[..., 0, None] == punctuation_ids).any(dim=-1)
    top2_is_punctuation = (top.indices[..., 1, None] == punctuation_ids).any(dim=-1)
    top1_is_markdown_structural = (
        top.indices[..., 0, None] == markdown_structural_ids
    ).any(dim=-1)
    top2_is_markdown_structural = (
        top.indices[..., 1, None] == markdown_structural_ids
    ).any(dim=-1)
    near_tie_rows = (top.values[..., 0] - top.values[..., 1]) <= (
        _PUNCTUATION_TIE_MARGIN
    )
    punctuation_tie_rows = (
        near_tie_rows
        & argmax_is_punctuation
        & ~(top1_is_punctuation & top2_is_punctuation)
    )
    markdown_structural_tie_rows = (
        near_tie_rows
        & argmax_is_markdown_structural
        & top1_is_markdown_structural
        & top2_is_markdown_structural
    )
    tied_rows = punctuation_tie_rows | markdown_structural_tie_rows
    if bool(tied_rows.any().item()):
        fp32_logits = _compute_fp32_logits(self, hidden_states, lm_head)
        if fp32_logits is not None:
            logits = logits.float()
            logits[tied_rows] = fp32_logits[tied_rows]
    return logits


LogitsProcessor._get_logits = _get_logits
