"""MetaX plugin patch: instrument BF16 Linear M-safety in actual model context.

Records max diff between batched M=6 and cat(6xM=1) for every UnquantizedLinearMethod.apply call.
"""
import os
import torch
from collections import defaultdict

_DIFF_THRESHOLD = 0.0  # record ALL diffs
_max_diffs = defaultdict(float)
_call_counts = defaultdict(int)


def apply_linear_msafe_probe():
    """Patch UnquantizedLinearMethod.apply to measure M=6 vs 6xM=1 diff."""
    if os.getenv("VLLM_METAX_LINEAR_MSAFE_PROBE", "0") != "1":
        return

    try:
        import vllm.model_executor.layers.linear as linear_mod

        orig_apply = linear_mod.UnquantizedLinearMethod.apply
        _hook_count = [0]

        def _instrumented_apply(self, layer, x, bias=None):
            result = orig_apply(self, layer, x, bias)
            _hook_count[0] += 1
            if (
                x.dim() >= 2
                and x.shape[0] > 1
                and x.shape[0] <= 6
                and x.dtype == torch.bfloat16
                and not x.requires_grad
            ):
                with torch.no_grad():
                    per_token = [
                        orig_apply(self, layer, x[i : i + 1], bias)
                        for i in range(x.shape[0])
                    ]
                    per_token_cat = torch.cat(per_token)
                    diff = (
                        (result.float() - per_token_cat.float()).abs().max().item()
                    )
                    # Try to get a meaningful name from layer or its weight
                    name = ""
                    for attr in ("prefix", "weight_name", "layer_name", "linear_name"):
                        val = getattr(layer, attr, "")
                        if val:
                            name = str(val)
                            break
                    if not name:
                        w = getattr(layer, "weight", None)
                        if w is not None:
                            name = f"W{list(w.shape)}"
                    if not name:
                        name = f"x{x.shape}"
                    _call_counts[name] += 1
                    if diff > _max_diffs[name]:
                        _max_diffs[name] = diff
            return result

        linear_mod.UnquantizedLinearMethod.apply = _instrumented_apply
        print("[LINEAR_MSAFE_PROBE] Instrumented", flush=True)

        # Also patch torch.nn.functional.linear to catch direct calls
        _orig_fn_linear = torch.nn.functional.linear
        _fn_call_count = [0]
        def _hooked_fn_linear(input, weight, bias=None):
            r = _orig_fn_linear(input, weight, bias)
            _fn_call_count[0] += 1
            if (
                input.dim() >= 2
                and input.shape[0] > 1
                and input.shape[0] <= 6
                and input.dtype == torch.bfloat16
                and weight.dtype == torch.bfloat16
                and not input.requires_grad
            ):
                with torch.no_grad():
                    pt = [_orig_fn_linear(input[i:i+1], weight, bias) for i in range(input.shape[0])]
                    pt_cat = torch.cat(pt)
                    d = (r.float() - pt_cat.float()).abs().max().item()
                    key = f"fn_linear_W{list(weight.shape)}"
                    _call_counts[key] += 1
                    if d > _max_diffs[key]:
                        _max_diffs[key] = d
            return r
        # Only hook if explicitly requested to catch all
        if os.getenv("VLLM_METAX_LINEAR_MSAFE_PROBE_FN", "0") == "1":
            torch.nn.functional.linear = _hooked_fn_linear
            print("[LINEAR_MSAFE_PROBE] Also hooked torch.nn.functional.linear", flush=True)

        import atexit

        @atexit.register
        def _report():
            print(f"[LINEAR_MSAFE_PROBE] total apply calls={_hook_count[0]}",
                  flush=True)
            if os.getenv("VLLM_METAX_LINEAR_MSAFE_PROBE_FN", "0") == "1":
                print(f"[LINEAR_MSAFE_PROBE] total fn.linear calls={_fn_call_count[0]}",
                      flush=True)
            if not _max_diffs:
                print(
                    "\n[LINEAR_MSAFE_PROBE] No M-dependent Linear layers "
                    "(all diff < threshold)",
                    flush=True,
                )
                return
            print("\n[LINEAR_MSAFE_PROBE] === M=6 vs 6xM=1 diff per layer ===", flush=True)
            # Only show top 20 by diff
            sorted_diffs = sorted(_max_diffs.items(), key=lambda kv: -kv[1])
            for name, diff in sorted_diffs[:30]:
                count = _call_counts[name]
                safe = "OK" if diff < 1e-3 else "!!"
                print(
                    f"  [{safe}] diff={diff:.6e} calls={count:4d}  {name}",
                    flush=True,
                )
            print(
                f"  (total unique layers: {len(_max_diffs)})", flush=True
            )

    except Exception as e:
        print(f"[LINEAR_MSAFE_PROBE] Failed: {e}", flush=True)
