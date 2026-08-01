# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX DSpark adapter for the DeepSeek-V4 BF16 KV-cache path."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as _UpstreamDSparkForCausalLM,
    DSparkDeepseekV4Model as _UpstreamDSparkModel,
)

from .model import DeepseekV4DecoderLayer
from .ops.mhc.backend import hc_head_fused_kernel, mhc_post

_BF16_INSERT_OP = "fused_deepseek_v4_qnorm_rope_kv_rope_insert"
_REPLICATE_MARKOV_HEAD_ENV = "VLLM_METAX_DSV4_REPLICATE_DSPARK_MARKOV_HEAD"
logger = init_logger(__name__)


class _ReplicatedMarkovEmbedding(nn.Module):
    def __init__(self, vocab_size: int, rank: int, params_dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    @staticmethod
    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                "replicated DSpark Markov embedding weight shape mismatch: "
                f"expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.long(), self.weight)


class _ReplicatedMarkovLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, params_dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    @staticmethod
    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                "replicated DSpark Markov linear weight shape mismatch: "
                f"expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)


class _ReplicatedMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        rank: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.markov_w1 = _ReplicatedMarkovEmbedding(vocab_size, rank, params_dtype)
        del prefix
        self.markov_w2 = _ReplicatedMarkovLinear(rank, draft_vocab_size, params_dtype)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        del logits_processor
        return self.markov_w2(markov_embed)


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert context KV through the MetaX BF16 native path only."""
    swa_cache_layer = attn.swa_cache_layer
    swa_cache = swa_cache_layer.kv_cache
    if swa_cache.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise ValueError(
            "MetaX DSpark context KV insertion requires bfloat16 KV and cache, "
            f"got kv={kv.dtype}, cache={swa_cache.dtype}"
        )

    n_ctx = kv.shape[0] if kv.ndim == 2 else -1
    if kv.ndim != 2 or kv.shape[1] != attn.head_dim:
        raise ValueError(
            "MetaX DSpark context KV must have shape "
            f"[num_tokens, {attn.head_dim}], got {tuple(kv.shape)}"
        )
    if not kv.is_contiguous():
        raise ValueError("MetaX DSpark context KV must be contiguous")
    for name, indices in (("positions", positions), ("slot_mapping", slot_mapping)):
        if indices.ndim != 1 or indices.shape[0] != n_ctx:
            raise ValueError(
                f"MetaX DSpark {name} length must equal KV rows ({n_ctx}), "
                f"got shape {tuple(indices.shape)}"
            )
        if indices.dtype is not torch.int64:
            raise ValueError(f"MetaX DSpark {name} must use int64 indices")
        if not indices.is_contiguous():
            raise ValueError(f"MetaX DSpark {name} must be contiguous")

    cos_sin = attn.rotary_emb.cos_sin_cache
    tensors = (kv, swa_cache, positions, slot_mapping, cos_sin)
    if any(t.device != kv.device for t in tensors[1:]):
        raise ValueError("MetaX DSpark context KV inputs must be on the same device")
    if swa_cache.ndim < 2 or swa_cache.shape[0] == 0:
        raise ValueError(
            f"MetaX DSpark SWA cache has invalid shape {tuple(swa_cache.shape)}"
        )
    expected_block_elements = swa_cache_layer.block_size * attn.head_dim
    block_elements = swa_cache.numel() // swa_cache.shape[0]
    if block_elements != expected_block_elements:
        raise ValueError(
            "MetaX DSpark SWA cache block layout mismatch: "
            f"expected {expected_block_elements} elements, got {block_elements}"
        )
    try:
        cache_2d = swa_cache.view(swa_cache.shape[0], -1)
    except RuntimeError as exc:
        raise ValueError(
            "MetaX DSpark SWA cache must be viewable as blocks by elements"
        ) from exc
    if cache_2d.stride(1) != 1 or cache_2d.stride(0) < expected_block_elements:
        raise ValueError(
            "MetaX DSpark SWA cache requires contiguous elements within each "
            f"block, got stride {cache_2d.stride()}"
        )
    if cos_sin.dtype is not torch.float32 or cos_sin.ndim != 2:
        raise ValueError("MetaX DSpark rotary cos/sin cache must be 2D float32")
    if not cos_sin.is_contiguous():
        raise ValueError("MetaX DSpark rotary cos/sin cache must be contiguous")

    try:
        native_op = getattr(torch.ops._C, _BF16_INSERT_OP)
    except (AttributeError, RuntimeError):
        native_op = None
    if native_op is None:
        raise RuntimeError(
            f"native BF16 DSpark context KV op is unavailable: {_BF16_INSERT_OP}"
        )

    dummy_q = torch.zeros(
        (n_ctx, attn.n_local_heads, attn.head_dim),
        dtype=torch.bfloat16,
        device=kv.device,
    )
    native_op(
        dummy_q,
        kv,
        cache_2d,
        slot_mapping,
        positions,
        cos_sin,
        attn.eps,
        swa_cache_layer.block_size,
    )


class DSparkDeepseekV4Model(_UpstreamDSparkModel):
    """Upstream DSpark model behavior with MetaX decoder and MHC paths."""

    decoder_layer_cls = DeepseekV4DecoderLayer

    @classmethod
    def build_decoder_layers(
        cls,
        *,
        decoder_layer_cls: type[nn.Module] | None = None,
        vllm_config: VllmConfig,
        prefix: str,
        num_hidden_layers: int,
        num_dspark_layers: int,
    ) -> nn.ModuleList:
        layer_cls = decoder_layer_cls or cls.decoder_layer_cls
        return nn.ModuleList(
            [
                layer_cls(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{num_hidden_layers + i}"),
                )
                for i in range(num_dspark_layers)
            ]
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = self.build_decoder_layers(
            vllm_config=get_current_vllm_config(),
            prefix=prefix,
            num_hidden_layers=self.num_hidden_layers,
            num_dspark_layers=self.num_dspark_layers,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_prefix = maybe_prefix(prefix, "markov_head")
        if os.getenv(_REPLICATE_MARKOV_HEAD_ENV) == "1":
            self.markov_head = _ReplicatedMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                config.dspark_markov_rank,
                self.embed_tokens.params_dtype,
                markov_prefix,
            )
            logger.warning_once(
                "DeepSeek V4 DSpark uses replicated Markov embedding and "
                "vocab projection: vocab=%d rank=%d",
                config.vocab_size,
                config.dspark_markov_rank,
            )
        else:
            from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead

            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                config.dspark_markov_rank,
                prefix=markov_prefix,
            )

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_slot_mappings is not None and len(context_slot_mappings) != len(
            self.layers
        ):
            raise ValueError(
                "DSpark context_slot_mappings must contain exactly one entry per layer"
            )
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            qr_kv, _ = layer.attn.fused_wqa_wkv(main_x)
            kv = layer.attn.kv_norm(qr_kv[..., layer.attn.q_lora_rank :])
            if slot_mapping is not None:
                _insert_context_kv(layer.attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        return hc_head_fused_kernel(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class DSparkDeepseekV4ForCausalLM(_UpstreamDSparkForCausalLM):
    """DSpark wrapper selecting the MetaX BF16 draft model implementation."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DSparkDeepseekV4Model",
]
