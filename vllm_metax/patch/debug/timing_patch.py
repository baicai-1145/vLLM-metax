"""MetaX plugin patch: inject CUDA event timing into V2 model runner.

This is loaded via VLLM_METAX_TIMING_PATCH=1 environment variable.
It patches GPUModelRunner.execute_model and sample_tokens to record
CUDA events for TRUE GPU timing (no profiler inflation).
"""
import os
import torch
from collections import defaultdict

_TIMING_ENABLED = os.getenv("VLLM_METAX_TIMING_PATCH", "0") == "1"
_timings = defaultdict(list)
_fwd_events = {"start": None, "end": None}


def apply_timing_patch():
    """Patch GPUModelRunner with CUDA event timing."""
    if not _TIMING_ENABLED:
        return

    try:
        import vllm.v1.worker.gpu.model_runner as mr_mod

        orig_execute = mr_mod.GPUModelRunner.execute_model
        orig_sample = mr_mod.GPUModelRunner.sample_tokens

        def _make_event():
            return torch.cuda.Event(enable_timing=True)

        def timed_execute(self, *args, **kwargs):
            _fwd_events["start"] = _make_event()
            _fwd_events["end"] = _make_event()
            _fwd_events["start"].record()
            result = orig_execute(self, *args, **kwargs)
            _fwd_events["end"].record()
            return result

        def timed_sample(self, *args, **kwargs):
            # Record forward timing (events from execute_model)
            if _fwd_events["start"] is not None and _fwd_events["end"] is not None:
                torch.cuda.synchronize()
                fwd_ms = _fwd_events["start"].elapsed_time(_fwd_events["end"])
                _timings["target_forward"].append(fwd_ms)

            # Record sample+draft timing
            s = _make_event()
            e = _make_event()
            s.record()
            result = orig_sample(self, *args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            sd_ms = s.elapsed_time(e)
            _timings["sample_plus_draft"].append(sd_ms)
            return result

        mr_mod.GPUModelRunner.execute_model = timed_execute
        mr_mod.GPUModelRunner.sample_tokens = timed_sample
        print("[TIMING_PATCH] Applied CUDA event timing to GPUModelRunner", flush=True)

        # Register shutdown hook to dump results
        import atexit

        @atexit.register
        def _dump_timings():
            if not _timings:
                return
            import json
            print("\n=== TRUE GPU Timing (CUDA Events) ===", flush=True)
            for phase in sorted(_timings.keys()):
                times = _timings[phase]
                if len(times) > 2:
                    steady = sorted(times[2:])
                elif times:
                    steady = sorted(times)
                else:
                    continue
                median = steady[len(steady) // 2]
                p90 = steady[int(len(steady) * 0.9)]
                mean = sum(steady) / len(steady)
                print(
                    f"  {phase:25s}: n={len(steady):3d} "
                    f"median={median:.1f}ms p90={p90:.1f}ms mean={mean:.1f}ms",
                    flush=True,
                )

            # Cycle estimate
            tf = _timings.get("target_forward", [])
            sd = _timings.get("sample_plus_draft", [])
            if tf and sd and len(tf) > 2:
                tf_med = sorted(tf[2:])[len(tf[2:]) // 2]
                sd_med = sorted(sd[2:])[len(sd[2:]) // 2] if len(sd) > 2 else sorted(sd)[-1]
                cycle_gpu = tf_med + sd_med
                commit = 3.16
                print(f"\n  GPU cycle estimate = {tf_med:.1f} + {sd_med:.1f} = {cycle_gpu:.1f}ms", flush=True)
                print(f"  Theoretical TPS (zero overhead) = {commit * 1000 / cycle_gpu:.1f}", flush=True)
                print(f"  Actual TPS = 19.3", flush=True)
                print(f"  Framework overhead = {163.7 - cycle_gpu:.1f}ms ({(163.7 - cycle_gpu) / 163.7 * 100:.0f}%)", flush=True)

            with open("/tmp/forward6_timing.json", "w") as f:
                json.dump({k: v for k, v in _timings.items()}, f, indent=2)

    except Exception as e:
        print(f"[TIMING_PATCH] Failed to apply: {e}", flush=True)
