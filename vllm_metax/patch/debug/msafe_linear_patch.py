"""MetaX plugin patch: M-invariant BF16 linear for DeepSeek-V4.

Unlike VLLM_BATCH_INVARIANT=1 (which changes attention backends, cascade
attention, symm_mem, MLA, etc.), this patch ONLY replaces
UnquantizedLinearMethod.apply with the fixed matmul_persistent_maca kernel.

This makes BF16 GEMMs M-invariant (batched == cat(per-token)) without
changing any other vLLM behavior.

Enabled via VLLM_METAX_MSAFE_LINEAR=1.
"""
import os
import torch

_PATCHED = False


def apply_msafe_linear_patch():
    """Replace UnquantizedLinearMethod.apply with M-invariant kernel."""
    global _PATCHED
    if os.getenv("VLLM_METAX_MSAFE_LINEAR", "0") != "1":
        return
    if _PATCHED:
        return

    try:
        from vllm_metax.patch.plugin_enhancement.batch_invariant_fix import (
            matmul_persistent_maca,
        )
        import vllm.model_executor.layers.linear as linear_mod
        from vllm.platforms import current_platform

        orig_apply = linear_mod.UnquantizedLinearMethod.apply

        def _msafe_apply(self, layer, x, bias=None):
            # Only use M-invariant kernel for MetaX BF16 with M in 2..6
            # (speculative decode range). For M=1, prefill (large M), or
            # non-BF16, use the original path.
            if (
                x.dtype == torch.bfloat16
                and x.dim() >= 2
                and 1 < x.shape[0] <= 6
                and current_platform.is_cuda_alike()
            ):
                weight = layer.weight
                out = matmul_persistent_maca(x, weight.t())
                if bias is not None:
                    out = out + bias
                return out
            return orig_apply(self, layer, x, bias)

        linear_mod.UnquantizedLinearMethod.apply = _msafe_apply
        _PATCHED = True
        print("[MSAFE_LINEAR] Patched UnquantizedLinearMethod.apply", flush=True)

    except Exception as e:
        print(f"[MSAFE_LINEAR] Failed to apply: {e}", flush=True)
