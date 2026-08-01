"""MetaX debug patch: per-position draft/target accept rate for DSpark.

DSpark forces V2 Model Runner (vllm/v1/worker/gpu/model_runner.py), which uses
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (NOT vllm/v1/sample/...).

This patch hooks the V2 RejectionSampler.__call__ to capture per-cycle
accepted-length distribution (0..k tokens accepted per request per cycle).

Loaded via VLLM_METAX_DSV4_PERPOS_ACCEPT=1 (must be set before python start).
"""
import os
import json
import atexit
from collections import defaultdict

_ENABLED = os.getenv("VLLM_METAX_DSV4_PERPOS_ACCEPT", "0") == "1"

# Accept-length histogram: key = number of draft tokens accepted (0..k)
# value = number of cycles with that acceptance count
_accept_hist = defaultdict(int)
_n_calls = 0
_pid = os.getpid()
_out_path = f"/tmp/perpos_accept_pid{_pid}.json"


def apply_perpos_accept_patch():
    if not _ENABLED:
        return
    try:
        from vllm.v1.worker.gpu.spec_decode import rejection_sampler as v2rs_mod

        _orig_call = v2rs_mod.RejectionSampler.__call__

        def _instrumented_call(self, logits, input_batch, draft_logits=None):
            global _n_calls
            _n_calls += 1

            result = _orig_call(self, logits, input_batch, draft_logits)

            try:
                # result.num_sampled: per-request number of tokens produced
                #   = 1 + num_accepted_draft_tokens (0..k for k=5)
                # result.num_rejected: per-request number rejected
                num_sampled = result.num_sampled
                import torch
                if isinstance(num_sampled, torch.Tensor):
                    ns_cpu = num_sampled.cpu().tolist()
                elif isinstance(num_sampled, list):
                    ns_cpu = num_sampled
                else:
                    ns_cpu = []

                for cnt in ns_cpu:
                    if cnt >= 1:  # valid entry (1 = base only = 0 accepted)
                        accepted = cnt - 1
                        _accept_hist[accepted] += 1

                if _n_calls % 20 == 0:
                    _flush()
            except Exception as e:
                if _n_calls <= 3:
                    with open(f"/tmp/perpos_err_{_pid}.txt", "a") as f:
                        f.write(f"call #{_n_calls} error: {e}\n")

            return result

        v2rs_mod.RejectionSampler.__call__ = _instrumented_call

        # Verify
        _actual = v2rs_mod.RejectionSampler.__call__
        with open(f"/tmp/perpos_verify_{_pid}.txt", "w") as f:
            f.write(f"__call__ qualname: {_actual.__qualname__}\n")
            f.write(f"is instrumented: "
                    f"{_actual.__qualname__ == 'apply_perpos_accept_patch.<locals>._instrumented_call'}\n")
        print(f"[PERPOS] Patched V2 RejectionSampler.__call__ pid={_pid} "
              f"qualname={_actual.__qualname__}", flush=True)

        def _flush():
            if not _accept_hist:
                return
            total = sum(_accept_hist.values())
            out = {
                "pid": _pid,
                "n_calls": _n_calls,
                "n_cycles_with_spec": total,
                "accept_length_histogram": {str(k): v for k, v in sorted(_accept_hist.items())},
            }
            if total > 0:
                # Compute average accepted tokens per cycle
                avg_acc = sum(int(k) * v for k, v in _accept_hist.items()) / total
                out["avg_accepted_per_cycle"] = round(avg_acc, 3)
                # Overall accept rate = avg_accepted / k
                # Need to know k (num_speculative_tokens)
                # The max key+1 tells us k (since accepted ranges 0..k)
                max_k = max(int(k) for k in _accept_hist.keys())
                out["k"] = max_k
                out["overall_accept_rate"] = round(avg_acc / max_k, 3)
                # Per-position conditional accept rate:
                # P(accept >= pos+1) = P(accepted >= pos+1) = sum(hist[j] for j >= pos+1) / total
                # P(accept == pos) = hist[pos] / total
                # Conditional: P(accept >= p | accept >= p-1) = P(accept >= p) / P(accept >= p-1)
                cumul = {}
                for p in range(max_k + 2):
                    # P(accept >= p) = P(accepted >= p tokens)
                    # = sum(hist[j] for j >= p) / total
                    cumul[p] = sum(_accept_hist.get(str(j), 0) for j in range(p, max_k + 1)) / total
                # Conditional: P(accept >= p+1 | accept >= p) = P(accept >= p+1) / P(accept >= p)
                cond = {}
                for p in range(max_k):
                    if cumul.get(p, 0) > 0:
                        cond[p] = round(cumul[p + 1] / cumul[p], 3)
                    else:
                        cond[p] = 0
                out["conditional_accept_rate_per_position"] = cond
            with open(_out_path, "w") as f:
                json.dump(out, f, indent=2)

        atexit.register(_flush)

    except Exception as e:
        import traceback
        print(f"[PERPOS] Failed: {e}\n{traceback.format_exc()}", flush=True)
