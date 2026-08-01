# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Patch: Enable PIECEWISE cudagraph for DSpark/DFlash speculator.
#
# Upstream DFlashSpeculator has two issues preventing PIECEWISE cudagraph
# for the draft model:
#
# 1. init_cudagraph_manager forces cudagraph_mode=NONE when target uses
#    PIECEWISE (not FULL). This means the draft's CudaGraphManager never
#    captures PIECEWISE graphs.
#
# 2. _run_model always calls self.model(...) eagerly, even when
#    cudagraph_runtime_mode=PIECEWISE. It never routes through the
#    captured PIECEWISE graph.
#
# Together these cause the entire draft forward to run in eager mode,
# where every TP AllReduce pays full synchronization latency (~665μs
# each vs ~8μs inside a graph).
#
# This patch fixes both issues: allows PIECEWISE mode for the draft's
# CudaGraphManager and routes _run_model through run_pw_graph when
# PIECEWISE is active.
#
# Enabled by default. Set VLLM_METAX_DSV4_DFLASH_PIECEWISE_CG=0 to disable.
# -----------------------------------------------
import os
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _apply_dflash_piecewise_cg_patch():
    global _PATCHED
    if _PATCHED:
        return
    if os.getenv("VLLM_METAX_DSV4_DFLASH_PIECEWISE_CG", "1") == "0":
        return

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
        DFlashCudaGraphManager,
    )

    # --- Fix 1: init_cudagraph_manager ---
    _orig_init = DFlashSpeculator.init_cudagraph_manager

    def _patched_init(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            effective_mode = CUDAGraphMode.FULL_DECODE_ONLY
        elif (
            cudagraph_mode.decode_mode() == CUDAGraphMode.PIECEWISE
            or cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE
        ):
            effective_mode = cudagraph_mode
        else:
            effective_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            effective_mode,
            decode_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )

    DFlashSpeculator.init_cudagraph_manager = _patched_init

    # --- Fix 2: _run_model routes PIECEWISE through run_pw_graph ---
    def _patched_run_model(
        self,
        num_tokens: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ):
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
            if (
                cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                and self.query_cudagraph_manager is not None
            ):
                mgr = self.query_cudagraph_manager
                if mgr.use_breakable_cg and mgr.breakable_cg_runner is None:
                    mgr.init_breakable_cg_runner(self.model)
                if mgr.breakable_cg_runner is not None:
                    last_hidden_states = mgr.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    last_hidden_states = self.model(**model_inputs)
            else:
                last_hidden_states = self.model(**model_inputs)
        return last_hidden_states

    DFlashSpeculator._run_model = _patched_run_model

    _PATCHED = True
    logger.info("DFlash/DSpark PIECEWISE cudagraph patch applied")


_apply_dflash_piecewise_cg_patch()
