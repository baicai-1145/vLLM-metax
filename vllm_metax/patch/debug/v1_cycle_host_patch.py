"""MetaX debug patch: V1 GPUModelRunner host-side cycle timing for DSpark.

Loaded via VLLM_METAX_DSV4_V1_CYCLE_HOST=1.
"""
import os
import time
from collections import defaultdict

_ENABLED = os.getenv("VLLM_METAX_DSV4_V1_CYCLE_HOST", "0") == "1"
_stamps: list[dict] = []
_NL = "\n"


def apply_v1_cycle_host_patch():
    if not _ENABLED:
        return
    try:
        import vllm.v1.worker.gpu_model_runner as mr_mod
        import vllm.v1.worker.gpu_worker as gw_mod
        GPUModelRunner = mr_mod.GPUModelRunner
        Worker = gw_mod.Worker

        _pid = os.getpid()
        out_path = f"/tmp/v1_cycle_host_pid{_pid}.txt"

        # Hook Worker.execute_model (the actual entry point via collective_rpc)
        _orig_worker_exec = Worker.execute_model
        _orig_sample_tokens = Worker.sample_tokens
        _wstate = {"n": 0}
        _wstamps = []

        def _worker_execute(self, scheduler_output, *args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_worker_exec(self, scheduler_output, *args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            _wstamps.append(elapsed)
            _wstate["n"] += 1
            if _wstate["n"] % 10 == 0:
                try:
                    steady = sorted(_wstamps[2:])
                    med = steady[len(steady)//2] if steady else 0
                    p90 = steady[int(len(steady)*0.9)] if steady else 0
                    with open(out_path, "w") as f:
                        f.write(f"Worker.execute_model n={len(_wstamps)} "
                                f"median={med:.1f}ms p90={p90:.1f}ms" + _NL)
                except Exception:
                    pass
            return result

        def _worker_sample(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_sample_tokens(self, *args, **kwargs)
            _sstamps.append((time.perf_counter() - t0) * 1000)
            return result

        Worker.execute_model = _worker_execute
        _sstamps = []
        Worker.sample_tokens = _worker_sample

        print(f"[V1_CYCLE_HOST] Patched Worker.execute_model pid={_pid}", flush=True)
    except Exception as e:
        print(f"[V1_CYCLE_HOST] Failed: {e}", flush=True)
