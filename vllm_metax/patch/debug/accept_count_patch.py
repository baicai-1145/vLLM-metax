"""MetaX debug patch: count DSpark acceptance statistics.

Hooks RejectionSampler.forward to count accepted vs proposed draft tokens.
Loaded via VLLM_METAX_DSV4_ACCEPT_COUNT=1.
"""
import os
import json
import atexit

_ENABLED = os.getenv("VLLM_METAX_DSV4_ACCEPT_COUNT", "0") == "1"

_stats = {"total_proposed": 0, "total_accepted": 0, "n_calls": 0, "n_cycles": 0}
_pid = os.getpid()
_out_path = f"/tmp/dspark_accept_stats_pid{_pid}.json"


def apply_accept_count_patch():
    if not _ENABLED:
        return
    try:
        from vllm.v1.sample.rejection_sampler import RejectionSampler

        _orig_forward = RejectionSampler.forward

        def _counted_forward(self, metadata, draft_probs, logits, sampling_metadata):
            if _stats["n_cycles"] < 2:
                print(f"[ACCEPT_COUNT] forward CALLED pid={_pid} n_cycles={_stats['n_cycles']}", flush=True)
            result = _orig_forward(self, metadata, draft_probs, logits, sampling_metadata)
            try:
                # num_sampled includes bonus token; num_rejected = proposed - accepted
                # Each spec decode step: proposed = num_spec_tokens, accepted = num_sampled - 1
                if hasattr(result, "num_sampled") and result.num_sampled is not None:
                    # num_sampled is per-request tensor
                    ns = result.num_sampled
                    nr = result.num_rejected if result.num_rejected is not None else None
                    import torch
                    if isinstance(ns, torch.Tensor):
                        batch = ns.numel()
                        sampled_total = ns.sum().item()
                        # accepted = sampled - 1 per request (bonus token)
                        accepted = sampled_total - batch
                        proposed = batch * metadata.max_spec_len
                        _stats["total_proposed"] += proposed
                        _stats["total_accepted"] += accepted
                        _stats["n_calls"] += batch
                        _stats["n_cycles"] += 1
                        # Flush every 50 cycles
                        if _stats["n_cycles"] % 50 == 0:
                            _flush_stats()
            except Exception:
                pass
            return result

        RejectionSampler.forward = _counted_forward
        is_patched = getattr(RejectionSampler.forward, "__wrapped__", None) is not None
        print(f"[ACCEPT_COUNT] Patched RejectionSampler.forward pid={_pid} "
              f"method={RejectionSampler.forward.__qualname__}", flush=True)

        def _flush_stats():
            if _stats["n_cycles"] == 0:
                return
            rate = _stats["total_accepted"] / max(_stats["total_proposed"], 1)
            out = {**_stats, "accept_rate": rate}
            with open(_out_path, "w") as f:
                json.dump(out, f, indent=2)

        @atexit.register
        def _dump():
            _flush_stats()
            if _stats["n_cycles"] > 0:
                rate = _stats["total_accepted"] / max(_stats["total_proposed"], 1)
                print(f"[ACCEPT_COUNT] pid={_pid} accept_rate={rate*100:.1f}% "
                      f"proposed={_stats['total_proposed']} accepted={_stats['total_accepted']} "
                      f"cycles={_stats['n_cycles']}", flush=True)
    except Exception as e:
        print(f"[ACCEPT_COUNT] Failed: {e}", flush=True)
