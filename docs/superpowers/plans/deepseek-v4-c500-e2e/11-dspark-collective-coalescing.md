# DeepSeek-V4 DSpark Collective Coalescing Implementation Plan

> **Status:** historical implementation plan. Its target-only indexer and
> collective results remain evidence, but the accepted checkpoint is now v104
> at `18.208126 TPS`. Continue from
> [`12-dspark-cycle-reconstruction.md`](12-dspark-cycle-reconstruction.md).

**For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to
implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
tracking.

**Goal:** Restore DSpark decode performance on four MetaX C500 GPUs by reducing
verification-row TP collective multiplication while preserving the accepted
rowwise BF16 arithmetic and exact greedy token sequence.

**Architecture:** Keep each verification row's native GEMM/MoE invocation and
rank-local output unchanged. Where evidence permits, concatenate rank-local
row outputs and perform one TP all-reduce for the whole row set instead of one
all-reduce per row. Every candidate remains opt-in until row differential,
graph replay, TP=4 greedy, and frozen-workload performance gates pass.

**Tech Stack:** Python 3.12, PyTorch BF16, vLLM 0.25rc1, MetaX native kernels,
MCCL TP collectives, PIECEWISE CUDA graph, pytest, torch.profiler traces.

## Global Constraints

- Acceptance hardware is `4x MetaX C500` with `TP=4`.
- Draft weights remain BF16 and speculative length remains `k=5`.
- Greedy decoding, prefix cache ON, async scheduling ON, PIECEWISE graph, and
  breakable graph remain enabled.
- Corpus is
  `tools/debug/corpora/deepseek_v4_mtp_new_heldout_3_20260724.jsonl`, SHA256
  `232696e55f01f85e3bb480e2550b277d9456c638964adaf22eadac722106be50`.
- Performance uses three prompts, 100 output tokens, warmup 1, runs 3,
  `MAX_MODEL_LEN=512`, `MAX_NUM_BATCHED_TOKENS=8192`, and `GPU_MEM=0.9`.
- Do not use Torch arithmetic fallback, eager acceptance, another backend or
  dtype, a different workload, lower TP, or reduced workload difficulty.
- Preserve the dirty worktree and all existing DSpark, MTP, capture, test, and
  artifact changes.
- A candidate stops at the first failed native differential, graph replay,
  TP=4 greedy exactness, or unchanged-workload performance gate.

## Historical frozen evidence

> **Superseded for current exact-candidate attribution.** The following
> `517`/`541` census records the 2026-07-26 pre-coalescing trace. The current
> exact five-cycle trace reconciles `111` all-reduces per cycle as `87` target,
> `19` draft backbone, and `5` draft sampling. There is no remaining 24-event
> attribution gap.

- Baseline commit: `c399373a033b7779e6094a984ded262eae37d7b7` plus dirty
  worktree.
- Target normal throughput: `26.9972 TPS`.
- DSpark normal throughput: `5.5939 TPS`; all three 100-token samples match the
  frozen greedy oracle exactly.
- Target TP all-reduces reconcile at `1305 / 15 = 87` per steady step.
- DSpark's named step scopes contain `7755 / 15 = 517` all-reduces, while the
  full trace contains `8115 / 15 = 541`. The missing `24` events per step must
  be attributed before optimization; do not mix the two counting scopes.
- DSpark fused-MoE kernels are `7740 / 15 = 516` per profile window step, so
  tokenwise FFN is the first implementation target and explains about 95% of
  the full-trace collective count.
- DSpark rank-0 all-reduce latency: mean `793.6592 us`, P90 `1521.92 us`, P99
  `6239.488 us`; all-reduce is `70.68%-75.44%` of self CUDA time.
- Primary artifacts:
  `.logs/deepseek_v4_dspark_profile_20260726/profile_summary.json` and
  `.logs/deepseek_v4_dspark_profile_20260726/{profile_target_short,profile_dspark_short}`.

Current exact artifacts:

- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_phase_5active/phase_allreduce_summary.tsv`
- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_phase_5active/allreduce_totals.tsv`
- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_shapes_5active/rank0_phase_mm_shape_census.tsv`

## CUDA reference and optimization map

The CUDA source explains the intended execution structure but does not, by
itself, prove a particular latency. In
`/root/vllm-0.25rc1/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, a
`k=5` DSpark request constructs five query rows and runs the three-layer draft
backbone once over the complete row set. Only the rank-256 Markov head samples
sequentially. The CUDA decoder does not split FFN or O-projection by row, so a
larger row dimension does not multiply TP collective calls.

The CUDA reference also uses native fused MHC kernels, sparse MLA, FP8
O-projection, FP4 experts where configured, stable buffers, and a dedicated
FULL/FULL_DECODE_ONLY graph for the draft step. Upstream explicitly disables
the DFlash/DSpark draft graph when the configured decode graph is PIECEWISE.
The frozen MetaX workload must remain PIECEWISE, so FULL graph is reference
context, not an allowed workload change.

### Attribution to establish

The following target-verifier attribution is a strong hypothesis because it
reconstructs the scoped profiler count exactly, but it remains subject to the
Task 1 census:

```text
CUDA-style batched verifier:
43 target layers * (1 O-proj reduction + 1 FFN reduction) + 1 final reduction
= 87 collectives

Current six-row MetaX verifier:
43 target layers * 6 rows * (1 O-proj reduction + 1 FFN reduction)
  + 1 final reduction
= 517 collectives
```

The complete DSpark trace contains `541` rather than `517` collectives per
profile step. The remaining `24` events must stay unattributed until records
distinguish target verification, the three-layer draft backbone, context-KV
precompute, auxiliary streams, and step-boundary events.

### Ordered optimization backlog

| Priority | Surface | Required change | Success evidence |
| --- | --- | --- | --- |
| P0 complete | Collective census | Record step, target/draft phase, layer, site, row count, rank, dtype/bytes, and correlation ID | Fresh exact phase census: `111 = 87 + 19 + 5` per cycle |
| P1 | Target FFN | Keep router, routed/shared expert, and native GEMM execution row-exact; obtain rank-local output per row, concatenate in row order, then issue one final TP reduction per layer | FFN reduction count changes from six per layer to one with row-exact native differential |
| P1 | Target O-projection | Keep inverse RoPE, BF16 projection arithmetic, and local `wo_b` result row-exact; concatenate local rows before one TP reduction | O-proj reduction count changes from six per layer to one without changing captured rows |
| P2 | Target MHC/RMS | Remove repeated per-row MHC/RMS and workspace copies only where a native batched or stacked implementation reproduces every row | Native differential, stable pointers, and no new allocations during graph replay |
| P2 | WQ-B and other projections | Attribute launch multiplication separately from collectives; preserve row-local GEMM shape when ordinary batching changes bytes | Reduced GEMM/D2D launches with exact captured stage outputs |
| P2 | Draft backbone | Confirm all five query rows enter one three-layer forward; prevent target-only tokenwise compatibility controls from splitting draft work | Separate draft-phase census and unchanged draft logits/tokens |
| P3 | Context KV/cache writes | Keep variable-shape precompute outside fixed query execution while batching native inserts where layouts permit | BF16 native dispatch, correct slot writes, and graph-stable query replay |
| P3 | PIECEWISE launch path | Retain the frozen graph mode while removing Python-side row loops and maintaining stable workspaces | Repeated PIECEWISE capture/replay with no eager or FULL-graph substitution |
| P4 | Draft quantization | Evaluate W8A8 or W4A16 only after the BF16 path passes every gate | Exact greedy tokens plus accepted-length and cycle-latency comparison |

### Implementation constraints by surface

For target FFN, the upstream MoE runner's late-reduction design is the semantic
reference: routed and shared rank-local outputs are combined before one final
`tensor_model_parallel_all_reduce`. A MetaX implementation must not toggle a
shared `moe_config.skip_final_all_reduce` during concurrent execution or graph
replay. If the current runner cannot expose local output immutably, add a
MetaX-owned adapter with an explicit local-output interface.

For target O-projection, do not replace rowwise arithmetic with an ordinary
batched GEMM. Invoke the existing native local projection once per row, clone
or write each result into stable row storage, concatenate those local outputs,
and reduce the complete tensor once. This preserves each row's GEMM shape and
the TP rank reduction order for every element.

WQ-B is primarily a GEMM/launch issue rather than an established TP-reduction
source. MHC/RMS and D2D copies are secondary multipliers. Neither should delay
the P1 FFN/O-projection collective work, and sparse MLA remains out of the
initial optimization path because its measured CUDA share is near 0.1%.

### Per-candidate gate order

Every candidate must pass these gates in order; a failure stops that candidate:

1. Red-capable captured-row differential against the current accepted path.
2. Intended native dispatch with no Torch arithmetic or backend fallback.
3. Production and non-aligned shapes, NaN/Inf checks, output-buffer checks,
   repeated stability, and stable allocation/pointer behavior.
4. PIECEWISE graph capture and repeated replay under TP=4.
5. TP=4 16-token fast greedy exactness against the frozen oracle.
6. TP=4 three-prompt by 100-token greedy exactness with the frozen manifest.
7. Unchanged-workload normal median/P90 TPS and per-GPU utilization.
8. A new trace proving the expected collective and launch-count reduction.

---

### Task 1: Per-Layer/Projection Census and Red-Capable Row Gate

**Files:**

- Create: `vllm_metax/models/deepseek_v4/collective_census.py`
- Create: `tests/models/deepseek_v4/test_collective_census.py`
- Create: `tools/debug/summarize_deepseek_v4_collective_census.py`
- Create: `tests/tools/test_summarize_deepseek_v4_collective_census.py`
- Modify only after unit interfaces pass:
  `vllm_metax/models/deepseek_v4/model.py`,
  `vllm_metax/models/deepseek_v4/attention.py`, and
  `vllm_metax/models/deepseek_v4/ops/o_proj.py`.

**Interfaces:**

- `collective_census(projection, layer_idx, rows, expects_reduce)` records one
  module-local invocation only inside an explicit diagnostic context.
- `collective_census_context()` returns deterministic in-memory records and is
  disabled by default with no filesystem or synchronization side effects.
- `assert_row_exact(reference_rows, candidate_rows, atol, rtol)` reports the
  first mismatching row and maximum absolute/relative error.
- The JSON summarizer groups records by layer and projection and reports
  invocation count, row count, and expected collective count.

- [x] **Step 1: Write a red-capable row differential test**

  Use a batch-sensitive fake projection whose batched result differs from
  independently evaluated rows. Verify `assert_row_exact` rejects the ordinary
  batched candidate and accepts concatenated rowwise outputs. This proves the
  gate catches the known failure mode rather than merely checking shapes.

- [x] **Step 2: Run the focused test and confirm RED**

  Run:

  ```bash
  source .venv/bin/activate
  source ./env.sh
  pytest -q tests/models/deepseek_v4/test_collective_census.py
  ```

  Expected before implementation: import or missing-symbol failure.

- [x] **Step 3: Implement the inert recorder and row assertion**

  Use a `ContextVar`-backed context so nested tests and concurrent request
  workers do not share records. Validate non-negative layer/row values and
  tensor shape/dtype equality before numerical comparison. Do not call CUDA,
  distributed, synchronization, or filesystem APIs in this module.

- [x] **Step 4: Confirm GREEN and prove red capability explicitly**

  Run the focused pytest command. The test must include both the expected
  mismatch exception and the accepted rowwise candidate.

- [x] **Step 5: Add the JSON summarizer test-first**

  Feed unsorted synthetic records spanning two layers and three projections.
  Assert stable layer/projection ordering and exact aggregate counts. Reject
  malformed schema rather than guessing missing values.

- [ ] **Step 6: Add module-local census call sites**

  Record WQ-B, O-projection `wo_b`, and fused-MoE output-reduction invocations
  at their existing wrappers. Do not globally monkeypatch distributed APIs.
  Census labels are evidence only and must reflect the module's actual
  `reduce_results`/`skip_final_all_reduce` contract.

- [x] **Step 7: Capture the TP=4 `k=5` census**

  Run the frozen graph workload with census explicitly enabled and store the
  summary under `.logs/deepseek_v4_dspark_collective_census_<date>/`. Record the
  command, repository state, per-layer/projection counts, and whether capture
  and replay calls are distinguishable. If Python instrumentation cannot count
  graph replay, use profiler event attribution and document that limitation;
  do not report capture-time counts as per-step counts.

**Task 1 evidence (2026-07-26):**

- RED command failed because the two focused test modules did not exist before
  implementation; the implemented differential test explicitly rejects a
  batch-sensitive projection and accepts independent row evaluation.
- GREEN command:
  `pytest -q tests/models/deepseek_v4/test_collective_census.py tests/tools/test_summarize_deepseek_v4_collective_census.py`
  returned `14 passed` after review fixes.
- `ruff check` on the four Python files, `markdownlint-cli2` on this plan, and
  scoped `git diff --check` all passed after review fixes.
- FFN and O-projection now have module-local census call sites and tests. WQ-B
  remains to be instrumented, so Step 6 stays open.
- Python census executes at graph capture rather than replay. The accepted
  candidate was therefore reconciled against the unmerged profiler table:
  `1,665 / 15 = 111` all-reduces per step, exactly
  `541 - 43 * 5 * 2`. Artifacts are under
  `.logs/deepseek_v4_dspark_collective_profile_20260726/candidate_short/`.
- This arithmetic is retained as historical coalescing evidence. The current
  exact phase census directly attributes `111` as target `87`, draft backbone
  `19`, and draft sampling `5`; the earlier residual 24 is no longer open.

### Task 2: Target-Verifier Fused-MoE/FFN Local Rows With One Collective

**Files:**

- Modify: `vllm_metax/models/deepseek_v4/model.py`
- Modify: `tests/models/deepseek_v4/test_mhc_tokenwise.py`
- Add a MetaX-owned adapter only if upstream `MoERunner` cannot expose local
  output without mutating shared configuration.

**Interfaces:**

- Preserve each row's router, routed-expert, shared-expert, and native GEMM
  invocation.
- Suppress only the final per-row TP all-reduce, concatenate complete rank-local
  row outputs, then perform one all-reduce with unchanged rank reduction order.
- Never mutate a shared `moe_config.skip_final_all_reduce` during concurrent
  execution or graph replay.
- Apply this candidate to target verification first. Draft-backbone behavior
  remains separately observable and must not change implicitly.

- [x] **Step 1:** Audit the concrete MetaX `MoERunner` and prove how
  `skip_final_all_reduce` is owned and captured.
- [x] **Step 2:** Add a red fake-MoE test for N row-local calls plus one final
  all-reduce, including shared-output addition order.
- [x] **Step 3:** Implement the smallest graph-stable local-output interface;
  reject the candidate if it requires process-global or mutable shared state.
- [x] **Step 4:** Run focused tests and Task 1 row differential on captured
  production BF16 rows.
- [x] **Step 5:** Delegate graph replay and native differential to
  `kernel-validator`, then run TP=4 16-token and frozen 3x100 greedy gates.
- [x] **Step 6:** Run unchanged-workload normal benchmark through
  `benchmark-runner`; retain only evidence-backed improvement.

### Task 3: Target-Verifier O-Projection Local Rows With One Collective

**Files:**

- Modify: `vllm_metax/models/deepseek_v4/ops/o_proj.py`
- Modify: `vllm_metax/models/deepseek_v4/ops/o_proj_debug.py`
- Modify: `vllm_metax/models/deepseek_v4/flashmla.py`
- Modify: `tests/models/deepseek_v4/test_o_proj_diff.py`
- Modify: `tests/models/deepseek_v4/test_prefill_gemm_chunking.py`

**Interfaces:**

- A native helper evaluates `wo_b.quant_method.apply` once per row, clones or
  writes each local result to stable storage, concatenates in original row
  order, and invokes `tensor_model_parallel_all_reduce` exactly once.
- The scoped `517` reconstruction predicts that target O-projection is a P1
  collective source. The residual 24 full-trace events remain a separate
  attribution problem and must not be assigned to this task without evidence.

- [x] **Step 1:** Add a fake native projection test proving N local row calls,
  one all-reduce, stable row order, and materialization of reused buffers.
- [x] **Step 2:** Add row-exact and intentional-batched-mismatch tests using the
  Task 1 assertion.
- [x] **Step 3:** Implement the minimal opt-in coalesced-collective helper while
  retaining `RowParallelLinear` contract validation and native quant apply.
- [x] **Step 4:** Run focused pytest and `git diff --check`.
- [ ] **Step 5:** Delegate production-shape BF16 differential, non-aligned rows,
  repeated stability, and PIECEWISE capture/replay to `kernel-validator`.
- [x] **Step 6:** Run TP=4 16-token greedy exactness, then frozen 3x100 greedy
  exactness. Stop and revert only this candidate if either token gate fails.
- [x] **Step 7:** Delegate unchanged-workload normal performance to
  `benchmark-runner`; retain only if median and P90 improve without fallback.

### Task 4: Remaining Tokenwise Projection and Copy Multipliers

**Files:**

- Modify only the census-proven seams in
  `vllm_metax/models/deepseek_v4/attention.py`,
  `vllm_metax/models/deepseek_v4/compressor.py`,
  `vllm_metax/models/deepseek_v4/flashmla.py`, and
  `vllm_metax/models/deepseek_v4/model.py`.
- Extend their existing focused tests under `tests/models/deepseek_v4/`.

- [x] **Step 1:** Reprofile accepted Tasks 2-3 and rank remaining WQ-B, MHC/RMS,
  D2D, and cache-write multipliers.
- [x] **Step 2:** Optimize only the highest measured remaining multiplier with
  one red-capable row/stage gate.
- [x] **Step 3:** Repeat native differential, graph replay, TP=4 16-token,
  frozen 3x100 exactness, and unchanged-workload performance gates.
- [x] **Step 4:** Stop when the next candidate does not improve median/P90 or
  when collectives reach the target-like floor without a safe semantic merge.

### Task 5: Final Evidence and Handoff Update

**Files:**

- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/10-dspark-performance-profile-handoff.md`
- Modify: this plan's checkbox/evidence sections.

- [x] **Step 1:** Run focused/project tests through `test-runner` and review all
  accepted changes through `reviewer`.
- [x] **Step 2:** Reprofile through `profiler`; report per-step collective count,
  all-reduce CUDA share, GEMM/D2D counts, and artifact paths.
- [ ] **Step 3:** Run the final normal benchmark through `benchmark-runner` with
  the exact frozen manifest and report median/P90 TPS and per-GPU utilization.
- [ ] **Step 4:** Recheck the 100-prompt x 32-token DSpark quality corpus only
  after frozen exactness and performance pass.
- [x] **Step 5:** Update authoritative evidence with repository state, commands,
  raw artifacts, native dispatch/fallback status, accepted/rejected candidates,
  and residual numerical or graph risks.

## Self-Review

- Spec coverage: census, row-exact red gate, native differential, graph replay,
  TP=4 greedy exactness, frozen performance, profiler attribution, and dirty
  worktree preservation each have an explicit task and stop condition.
- Placeholder scan: no task relies on a Torch fallback, workload change, or
  unspecified acceptance configuration.
- Type consistency: Task 1's census and row assertion are the shared evidence
  interfaces used by Tasks 2-4; no production optimization depends on the JSON
  reporting layer.

## Phase 1-5 Results (2026-07-26)

- FFN production preconditions passed on every TP rank:
  `CompressedTensorsWNA16MoEMethod`, `_fused_output_is_reduced=false`,
  sequence parallel off, `skip_final_all_reduce=false`, TP=4, EP=1, and native
  PYNCCL. Rows 2 and 6 were bitwise exact with one reduction instead of 2/6;
  fixed-six-row graph replay was pointer-stable and repeat-bitwise. Evidence:
  `.logs/dspark_collective_candidate_validation_20260726/`.
- A control run proved the complete frozen dispatch environment before judging
  candidates. With both coalescing switches off, all three 16-token samples
  matched the target oracle under PIECEWISE graph. Evidence:
  `.logs/deepseek_v4_dspark_collective_control_20260726/fast16_frozen_env/`.
- FFN-only passed 3 x 16 and 3 x 100 greedy exactness. Its normal 3 x 100
  median was `5.796072 TPS`; artifact:
  `.logs/deepseek_v4_dspark_ffn_collective_20260726/full3x100_frozen_env/`.
- FFN plus O-projection passed 3 x 16 and 3 x 100 greedy exactness. Its normal
  3 x 100 median was `6.033146 TPS`, up 7.85% from `5.593861 TPS`; P90 latency
  was `20.005463 s` versus baseline `21.890262 s`. Artifact:
  `.logs/deepseek_v4_dspark_o_proj_collective_20260726/full3x100_frozen_env/`.
- Reprofiling reduced all-reduce calls from `8,115` to `1,665`, or from 541 to
  111 per step. Rank-0 all-reduce CUDA time changed only from `6.440545 s` to
  `6.362464 s`; mean rose from `793.659 us` to `3.821300 ms`, P90 to
  `7.341056 ms`, and P99 to `7.568384 ms`. Larger coalesced MCCL payloads are
  now the primary limiter. The residual compute multipliers are unchanged:
  `aten::mm=39,960`, MHC downstream RMS `8,115`, fused MoE kernels `7,740`,
  and D2D copies `12,105`. Artifact:
  `.logs/deepseek_v4_dspark_collective_profile_20260726/candidate_short/`.
- The preceding timing and multiplier totals are historical. Current exact
  per-cycle shape counts are target `aten::mm=2,562`, MHC GEMV/RMS `516` each,
  fused MoE `516`, `aten::cat=341`, and D2D `861`; see the fresh exact shape
  artifact listed above.
- O-projection production-shape kernel-validator dispatch failed twice at the
  agent transport layer before starting. A local fail-closed TP=4 rerun then
  passed on all ranks: BF16 rows two and six were bitwise exact and finite, and
  fixed-six-row graph replay was repeat-bitwise with a stable output pointer.
  Evidence: `.logs/dspark_collective_candidate_validation_20260726/o_proj_differential_probe_serial_retry.log`.
- The fail-closed MCCL row curve measured aligned one-to-six-row BF16 reductions
  at only `42--105 us` median with `MCCLLibrary`/PYNCCL on every rank. This
  falsifies raw payload bandwidth as the source of the `3.821 ms` production
  mean; repeated synchronization and rank-arrival skew are dominant. Evidence:
  `.logs/deepseek_v4_dspark_mccl_curve_20260726/mccl_native.log`.
- The exact MHC hybrid candidate passed its native and graph gates and the
  frozen 3 x 16 token gate, but produced `4.741845 TPS` versus `4.746371 TPS`
  for the matched full-coalescing control. It was rejected before 3 x 100.
  Evidence: `.logs/deepseek_v4_dspark_mhc_hybrid_20260726/fast16/`.
- WQ-B remains the largest producer multiplier (`39,960` `aten::mm` calls,
  `553.190 ms` CUDA), but ordinary batching is byte-inexact and can flip greedy
  output. No grouped native single-row GEMM API exists locally, and global
  batch-invariant mode would replace multiple frozen backends, so no compliant
  WQ-B candidate was implemented.
- Final focused regression passed: `123 passed`; Ruff and `git diff --check`
  passed.

### Superseding exactness evidence

Fresh final acceptance is red independently of coalescing. Candidate and
control hashes match under each request history (`8fec197...` isolated and
`3df594...` ordered), proving coalescing does not introduce the divergence.
Low-intrusion scheduler capture localizes the common first error to verifier
cycle 29: both histories accept one draft and commit `[554, 1749]`, but the
MTP=0 oracle commits `366` after `554`. The histories diverge only on the next
cycle, so request history affects recovery rather than causing the first
error. Prefix capture reports zero hits for all five KV groups. The current
blocker is six-row target-verifier bonus-row numerical drift, not prefix-cache
reuse. Candidate throughput from the red rerun (`5.986386 TPS`) is invalid and
must not be reported as an accepted speedup.

Evidence:

- `.logs/deepseek_v4_dspark_final_20260726/full3x100/`
- `.logs/deepseek_v4_dspark_final_20260726/replay_sample1/`
- `.logs/deepseek_v4_dspark_final_20260726/control_sample1/`
- `.logs/deepseek_v4_dspark_final_20260726/control_first2/`
- `.logs/deepseek_v4_dspark_history_20260726/ordered_token_pairs/`
- `.logs/deepseek_v4_dspark_history_20260726/isolated_token_pairs/`

### Exactness blocker resolved (2026-07-27)

The cycle-29 drift came from the tokenwise INT8 sparse-indexer Q path in the
target verifier. Each single-token native quant invocation received the full
six-row `indexer_weights` tensor. The native kernel sized its output like that
tensor but launched for one token, so concatenating the six results did not
produce one valid folded-weight row per verifier token. A red integration gate
reproduced the resulting `[4, 1]` output from two input rows.

Applying the obvious row slice to every DeepSeek-V4 layer was rejected: it also
changed the DSpark draft layers and failed all three 100-token oracle hashes.
The accepted opt-in mode,
`VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS=target`, slices weights only
when `layer_idx < num_hidden_layers`; DSpark's layers at
`num_hidden_layers + i` retain their frozen behavior. Invalid values fail
closed. Production `(T=6, H=64, D=128)` native differential and graph replay
are row-exact and repeat-exact; the target/draft selection tests pass.

With FFN and O-projection collective coalescing also enabled, fast 3 x 16 and
two independent normal 3 x 100 runs matched all frozen hashes. The normal runs
measured `5.965803` and `5.977872 TPS`, respectively, or `+6.65%` and `+6.87%`
over the `5.593861 TPS` baseline. The second run's decode-window GPU
utilization averages were `16.959%`, `17.265%`, `15.714%`, and `17.061%` from
49 one-second samples per GPU. The accepted result remains far below the
`26.997175 TPS` MTP=0 target, so communication/launch optimization continues.

Evidence:

- Native and graph gate: `artifacts/kernel_validation/indexer_weight_slice_gate.log`
- Rejected all-layer candidate:
  `.logs/deepseek_v4_dspark_indexer_weight_fix_20260727/full3x100_candidate/`
- Reverted control retaining the cycle-29 failure:
  `.logs/deepseek_v4_dspark_indexer_weight_fix_20260727/full3x100_reverted_control/`
- Accepted fast gate:
  `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/fast16/`
- Accepted normal runs:
  `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/full3x100/`
  and
  `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/full3x100_repeat_util/`
