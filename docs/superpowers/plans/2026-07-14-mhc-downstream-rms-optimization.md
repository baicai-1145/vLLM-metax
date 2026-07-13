# MHC Downstream RMS Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `executing-plans` to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Reduce TP=4 Plan 02-on decode latency by optimizing the production
`mhc_downstream_rms_kernel` without changing token IDs, arithmetic precision,
graph behavior, dispatch, or workload.

**Architecture:** Keep the existing three-kernel exact MHC contract and output
buffers. First parallelize only the independent rows and columns of the 4x4
Sinkhorn loop while preserving the operation order within each four-element
sum and retaining `expf` plus `__fdiv_rn`; accept the candidate only after real
payload differential, graph replay, isolated latency, TP=4 greedy, and normal
throughput gates.

**Tech Stack:** MetaX C500, MACA CUDA bridge, C++/CUDA,
`vllm_metax._metax_sparse_C`, PyTorch, vLLM TP=4, CUDA Graph PIECEWISE.

## Global Constraints

- Acceptance remains TP=4 with the frozen DeepSeek-V4 workload.
- Keep MTP off, `MAX_TOKENS=100`, PIECEWISE graph, prefix caching on, and
  `GPU_MEM=0.9` for end-to-end acceptance.
- No Torch arithmetic, eager, backend, shape, workload, or fallback substitution.
- Preserve `expf`, `__fdiv_rn`, 20 Sinkhorn repetitions, FP32 intermediates,
  BF16 rounding points, and the existing output-buffer contract.
- Do not change runtime defaults or start Plan 04.
- Do not commit from the dirty main worktree; record checkpoints with artifacts
  and review instead.

---

### Task 1: Freeze the Plan 02-on Baseline

**Files:**

- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/02-mhc-rmsnorm.md`
- Create: `.logs/mhc_downstream_opt_baseline_20260714/`
- Create: `.logs/plan02_on_decode_profile_20260714/`

**Interfaces:**

- Consumes: corrected Plan 02-on `_metax_sparse_C` binary and frozen TP=4
  workload.
- Produces: immutable correctness, graph, latency, and end-to-end comparison
  points for later tasks.

- [x] **Step 1: Record the normal serving baseline**

  Record exact-off `15.965887 TPS` and exact-on `25.875913 TPS`, with identical
  100-token IDs and no fallback.

- [x] **Step 2: Record the latest decode profile**

  Record exact MHC at `47.9%` of rank0 self CUDA time and
  `mhc_downstream_rms_kernel` at `33.59%` (`139.048 us/call` in trace).

- [x] **Step 3: Record the isolated production-shape baseline**

  Record eager median/P90 `151.81/153.09 us` and graph median/P90
  `166.14/167.68 us` over real late payloads. Preserve
  `.logs/mhc_downstream_opt_baseline_20260714/{summary.json,manifest.json}`.

### Task 2: Parallelize the 4x4 Sinkhorn Rows and Columns

**Files:**

- Modify: `csrc/metax_sparse/gemm_fp32.cu:552-610`
- Test: `tests/models/deepseek_v4/test_mhc_raw_diff.py`

**Interfaces:**

- Consumes: `comb_mix[16]` in shared FP32 storage and fixed `repeat=20`.
- Produces: bitwise-identical `comb_mix_out`, `pre_norm_out`, and `norm_out`
  through the unchanged `mhc_downstream_rms_out` schema.

- [x] **Step 1: Extend the production-op regression test**

  Add a real `mhc_downstream_rms_out` case whose 4x4 logits are non-symmetric.
  Compare `comb_out`, `pre_norm_out`, and `norm_out` to the Torch reference so
  a row/column indexing or synchronization error fails.

- [x] **Step 2: Run the regression before the optimization**

  Run:

  ```bash
  source .venv/bin/activate
  source ./env.sh
  python -m pytest -q \
    tests/models/deepseek_v4/test_mhc_raw_diff.py \
    -k 'downstream_rms'
  ```

  Expected: PASS on the current serial implementation, establishing the oracle.

- [x] **Step 3: Implement four-way row/column parallelism**

  In `mhc_downstream_rms_kernel`, assign threads `0..3` one row each and then
  one column each. Within each row or column, keep the current four additions
  and four `__fdiv_rn` operations in the same order. Synchronize between row
  and column phases and before the final `comb_mix_out` write. Do not alter the
  later BF16 pre-norm or RMS reduction.

- [x] **Step 4: Rebuild the native extension**

  Run the repository MACA build path and record the loaded `.so` path, mtime,
  size, and SHA256. A build that leaves the prior `.so` loaded is a failure.
  The final main-tree record is
  `.logs/mhc_downstream_sinkhorn_parallel_main_final_20260714/build_manifest.txt`.

- [x] **Step 5: Run focused project tests**

  Run the full `test_mhc_raw_diff.py`, Ruff checks, and `git diff --check`.
  Expected: all tests pass and no formatting errors.

### Task 3: Validate Correctness, Graph Safety, and Isolated Latency

**Files:**

- Create: `.logs/mhc_downstream_sinkhorn_parallel_20260714/`

**Interfaces:**

- Consumes: the rebuilt candidate native extension and the 1,892 real late
  payload corpus.
- Produces: the go/no-go result for end-to-end validation.

- [x] **Step 1: Run real-payload differential validation**

  Compare every checked stage on calls `0..1891`, including former boundary
  calls `1728`, `1841`, and `1861`. Require bitwise equality, finite outputs,
  intended native dispatch, and no hidden fallback.

- [x] **Step 2: Run graph capture and repeated replay**

  Capture production shapes and replay at least nine times. Require stable
  pointers, unchanged allocated/reserved memory, and bitwise outputs.

- [x] **Step 3: Run the isolated microbenchmark**

  Use at least 20 warmups and 200 synchronized repetitions. Report eager and
  graph median/P90. The performance gate is eager median `<=145 us` and graph
  median `<=158 us`, at least about 5% below the frozen baseline.

- [x] **Step 4: Apply the stop/go rule**

  Reject or revise the candidate on any correctness failure, graph failure,
  fallback, or failure to meet the isolated latency threshold. Do not run an
  end-to-end benchmark for a rejected kernel.

### Task 4: Run TP=4 End-to-End Acceptance

**Files:**

- Create: `.logs/mhc_downstream_sinkhorn_parallel_e2e_20260714/`
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/02-mhc-rmsnorm.md`

**Interfaces:**

- Consumes: a Task 3 candidate that passed every kernel gate.
- Produces: final accepted or rejected optimization status.

- [x] **Step 1: Run the TP=4 23-token fast greedy gate**

  Require exact IDs and index 22 token `372` under PIECEWISE graph mode.

- [x] **Step 2: Run the TP=4 100-token greedy gate**

  Compare exact-off oracle and exact-on candidate byte-for-byte. Require no
  mismatch and no fallback.

- [x] **Step 3: Run the normal five-run throughput comparison**

  Preserve the frozen manifest. Report median/P90 TPS and latency, per-GPU
  utilization, and raw artifacts. Keep profiler-instrumented results separate.

- [x] **Step 4: Review and document the result**

  Run a focused code review, Markdown lint, and `git diff --check`. Record both
  accepted and rejected candidates so the baseline is not silently replaced.

## Final result: accepted (2026-07-14)

The 4x4 Sinkhorn optimization is accepted for the explicit Plan 02-on path.
Threads `tid<4` parallelize independent rows and columns while preserving the
existing operation order, `__fdiv_rn`, and 20 repetitions. The main-tree
validation SHA is `28ea10c` (no commit was created or fabricated by this
record).

- Focused project validation: 21 tests passed; native dispatch remained active
  with no fallback.
- Real late-payload differential: `1892/1892` bitwise, all finite; graph replay
  `9/9`, pointers and allocations stable.
- Isolated production-shape latency: eager median/P90 `81.92/83.456 us` and
  graph median/P90 `97.024/98.816 us`, from eager baseline
  `151.81/153.09 us` and graph baseline `166.14/167.68 us`.
- TP=4 PIECEWISE greedy gates: 23-token and 100-token IDs passed byte-for-byte;
  no fallback.
- Normal five-run serving comparison: baseline `25.875913 TPS` /
  `3.864598 s` median latency versus final `30.640709 TPS` / `3.263632 s`;
  `+18.414%` TPS and `-15.551%` latency. Average GPU utilization was
  `[29.7333, 29.5333, 29.5333, 29.6000]%`.

Artifacts:

- `.logs/mhc_downstream_sinkhorn_parallel_main_final_20260714/`
- `.logs/mhc_downstream_sinkhorn_parallel_main_e2e_20260714/`
- `.logs/mhc_downstream_sinkhorn_parallel_after_profile_20260714/`

The after-profile trace reports downstream RMS `59.095 -> 29.413 ms`
(`-50.23%`, `139.048 -> 69.208 us/call`) and exact MHC grouped time
`84.266 -> 54.550 ms` (`-35.26%`). MCCL residency is rank-asymmetric
(ranks 0-2 about 48 ms, rank 3 `3.289 ms`); these residency values are
synchronization/wait evidence and must not be summed as pure communication
link time. The next diagnosis is rank arrival/collective wait. Plan 04 remains
not started.
